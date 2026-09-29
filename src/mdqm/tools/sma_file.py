"""``mdqm-sma-file``: the SMA DQM over one MIDAS file, with no MIDAS running.

The manual path beside the ``sma_analyzer`` daemon. When the daemon or the DAQ
is down, a shifter runs the identical plugin (`mdqm.plugins.sma.SmaPlugin`)
over one subrun file and gets the same numbers the pages would have shown::

    mdqm-sma-file /data/run01008_00001.mid.lz4
    mdqm-sma-file run00682_00005.mid.lz4 --shift 3 --out /tmp/682-5
    mdqm-sma-file FILE --settings '{"Cuts": {"coinc window ns": 30}}' --no-png
    mdqm-sma-file FILE --settings my-sma-settings.json --frames 50 --skip 10

What it does
------------
The plugin is built with the daemon's defaults (``SETTINGS_DEFAULTS``, what a
fresh ``/DQM/SMA`` holds), overridden by ``--settings`` (inline JSON or a JSON
file, same tree as ``/DQM/SMA``; unknown keys are an error) and then
``--shift``, and applied with ``apply_settings(settings, rebuild=True)`` as the
analyzer does on its first ODB read. Every readout event (id 301, bank H000)
is fed to ``process(event, run_number)``, as the daemon does with
``Sampling/process all`` on (the default); the file's own run number (the
begin-of-run serial, else the file name) stands in for ``/Runinfo``.

Differences from the daemon, all deliberate:

* the plugin's clock is the event time stamp, not the wall clock, so the 1 s
  trend rows and the summary window (60 s, longer than a subrun) are data
  seconds and a rerun gives the same output;
* there is no run: the summary is asked for with ``run_active=false``, so the
  "no SMA frames for > 5 s while a run is active" flag cannot fire;
* ``Sampling`` is ignored: every frame is processed.

Outputs (``--out``, default ``./sma-file-<run>_<subrun>/``)
------------------------------------------------------------
``hists.npz``      every histogram: ``<name>`` = counts with under/overflow
                   (index 0 and -1 on each axis; 2D is [y, x]), ``<name>.entries``,
                   ``<name>.dropped``, ``<name>.axes`` = [[bins, lo, hi], ...]
                   (x first); ``__meta__`` = JSON (file, run, histogram titles and
                   axis titles, settings used).
``summary.json``   exactly what ``sma::summary`` returns (the JSON inside the
                   envelope).
``trend.json``     exactly what ``sma::trend`` returns (1 s rows).
``summary.png``    one page (unless ``--no-png``; needs matplotlib).

A short text summary goes to stdout.

Exit status: 0 on success; 3 when the summary carries at least one
error-severity flag (shift mismatch, suspect time base, all frames stale,
fine/coarse mismatch above the error fraction on a flagged channel); 2 for a
bad argument or an unreadable file.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mdqm.dqm import framing
from mdqm.dqm.hist import HistStore
from mdqm.plugins import sma as P
from mdqm.plugins import sma_words as W
from mdqm.tools import midasfile as MF

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_ERROR_FLAG = 3


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------

def merge_strict(defaults: dict, given: dict, where: str = "") -> dict:
    """`given` over `defaults`; a key the defaults do not have is an error.

    The daemon ignores stray ODB keys (an operator's typo must not stop it);
    here a typo would silently run the defaults, so it is refused instead.
    """
    out = copy.deepcopy(defaults)
    for k, v in given.items():
        path = f"{where}/{k}" if where else k
        if k not in out:
            raise KeyError(f"unknown setting {path!r}; known here: {', '.join(sorted(out))}")
        if isinstance(out[k], dict):
            if not isinstance(v, dict):
                raise KeyError(f"setting {path!r} is a directory, got {v!r}")
            out[k] = merge_strict(out[k], v, path)
        else:
            out[k] = v
    return out


def load_settings_arg(arg: str | None) -> dict:
    """``--settings``: inline JSON (starts with ``{``) or a path to a JSON file."""
    if not arg:
        return {}
    text = arg if arg.lstrip().startswith("{") else Path(arg).read_text()
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("--settings must be a JSON object (the /DQM/SMA tree)")
    return obj


def build_settings(overrides: dict | None = None, shift: int | None = None) -> dict:
    """The full ``/DQM/SMA`` tree the daemon would read from a fresh ODB, overridden."""
    s = merge_strict(P.SETTINGS_DEFAULTS, overrides or {})
    if shift is not None:
        s["Coarse shift"] = int(shift)
    return s


# ---------------------------------------------------------------------------
# running the plugin
# ---------------------------------------------------------------------------

class DataClock:
    """The plugin's clock, set to each frame's event time stamp."""

    def __init__(self, t: float = 0.0):
        self.t = float(t)

    def __call__(self) -> float:
        return self.t


def make_plugin(settings: dict, clock) -> P.SmaPlugin:
    """Build the plugin the way the analyzer does: defaults, then the first apply."""
    plugin = P.SmaPlugin(HistStore(), clock=clock)
    plugin.apply_settings(settings, rebuild=True)
    return plugin


@dataclass
class FeedStats:
    frames_seen: int = 0          # id-301 events read (before --skip/--frames)
    frames_fed: int = 0           # handed to process()
    frames_skipped: int = 0
    frames_rejected: int = 0      # process() returned False (no H000)
    bank_errors: list = field(default_factory=list)
    t_first: int | None = None
    t_last: int | None = None


def feed(events, settings: dict, run_number, *, frames: int | None = None, skip: int = 0,
         clock: DataClock | None = None) -> tuple[P.SmaPlugin, FeedStats]:
    """Feed every readout event of `events` to a fresh plugin.

    `events` is any iterable of event objects: this module's `MidasFile`, or
    ``midas.file_reader.MidasFile(path, use_numpy=True)`` (the compare script
    uses both). The plugin is built at the first readout frame, with the
    clock already at that frame's time, so the trend does not start with ten
    minutes of empty seconds before the data.
    """
    clock = clock or DataClock(time.time())
    stats = FeedStats()
    plugin = None
    for ev in events:
        h = ev.header
        if h.event_id not in P.SmaPlugin.event_ids:
            continue
        stats.frames_seen += 1
        if stats.frames_seen <= skip:
            stats.frames_skipped += 1
            continue
        if frames is not None and stats.frames_fed >= frames:
            break
        clock.t = float(h.timestamp)
        if plugin is None:
            plugin = make_plugin(settings, clock)
        err = getattr(ev, "bank_error", None)
        if err:
            stats.bank_errors.append(f"serial {h.serial_number}: {err}")
        if stats.t_first is None:
            stats.t_first = h.timestamp
        stats.t_last = h.timestamp
        stats.frames_fed += 1
        if not plugin.process(ev, run_number=run_number):
            stats.frames_rejected += 1
    if plugin is None:
        plugin = make_plugin(settings, clock)
    return plugin, stats


def command_json(plugin: P.SmaPlugin, cmd: str, args: dict | None = None) -> dict:
    """Call a ``sma::`` command as the page would and decode its JSON envelope."""
    blob = plugin.commands()[cmd](json.dumps(args or {}))
    _size, tag, payload = framing.parse_envelope(blob)
    if tag != framing.TAG_JSON:
        raise RuntimeError(f"{cmd} returned tag {tag!r}, not json")
    return json.loads(payload)


def offline_summary(plugin: P.SmaPlugin) -> dict:
    """``sma::summary`` with ``run_active=false``: offline there is no run."""
    return command_json(plugin, "sma::summary", {"run_active": False})


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

def _axes_of(h) -> list:
    axes = [h.x] + ([h.y] if hasattr(h, "y") else [])
    return [[float(a.n), float(a.lo), float(a.hi)] for a in axes]


def hist_arrays(store: HistStore) -> dict[str, np.ndarray]:
    """Every histogram as npz entries (see the module docstring)."""
    out: dict[str, np.ndarray] = {}
    for name in store.names():
        h = store.get(name)
        out[name] = np.array(h.counts, copy=True)
        out[f"{name}.entries"] = np.int64(h.entries)
        out[f"{name}.dropped"] = np.int64(h.dropped)
        out[f"{name}.axes"] = np.array(_axes_of(h), dtype=np.float64)
    return out


def save_npz(store: HistStore, path: Path, meta: dict) -> None:
    arrays = hist_arrays(store)
    meta = dict(meta)
    meta["histograms"] = [store.get(n).metadata() for n in store.names()]
    arrays["__meta__"] = np.array(json.dumps(meta, default=str))
    np.savez_compressed(path, **arrays)


def _fmt_rate(x) -> str:
    if x is None:
        return "-"
    if x >= 1e6:
        return f"{x / 1e6:.3g}M"
    if x >= 1e3:
        return f"{x / 1e3:.3g}k"
    return f"{x:.3g}"


def _frac(x, digits=1) -> str:
    return "-" if x is None else f"{100 * x:.{digits}f}%"


def _good(f: dict) -> int:
    return f["processed"] - f["stale"] - f["empty"] - f.get("suspect", 0)


def _eff(e: dict, digits=1) -> str:
    """A timed efficiency, or "n/a (reason)" when the plugin withheld it."""
    if e["eff"] is None:
        return f"n/a ({e['reason']})" if e.get("reason") else "n/a"
    return _frac(e["eff"], digits)


def counter_roles(summary: dict) -> list[tuple[str, str, int]]:
    """``(name, label, channel)`` of S1..Sn: S1 from the channel roles, the rest
    from the efficiency list (which leaves S1 out)."""
    s1 = [c for c in summary["channels"] if c["role"] == "s1"]
    out = [("S1", c["label"], c["ch"]) for c in s1[:1]]
    return out + [(e["counter"], e["label"], e["ch"]) for e in summary["efficiency"]]


def text_summary(summary: dict, stats: FeedStats, file_label: str, elapsed_s: float) -> str:
    f = summary["frames"]
    sh = summary["shift"]
    span = (None if stats.t_first is None else stats.t_last - stats.t_first)
    lines = [
        f"{file_label}: run {summary['run']}, {stats.frames_fed} SMA frames processed "
        f"({stats.frames_skipped} skipped) in {elapsed_s:.1f} s"
        + (f", {span} s of data" if span is not None else ""),
        f"frames: good {_good(f)}, stale {f['stale']}, suspect {f.get('suspect', 0)}, "
        f"empty {f['empty']}, "
        f"rejected {f['rejected']}, missed by serial {f['missed_by_serial']}, "
        f"gap resets {f['gap_resets']}",
        f"live fraction {_frac(summary['live_fraction'])}, kept-hit span "
        f"{summary['span_s'] or 0:.3g} s, covered {summary.get('covered_s') or 0:.3g} s",
    ]
    fr = dict(zip(sh["scan"], sh["fractions"], strict=True))
    best = sh["best"]
    lines.append(
        f"coarse shift {sh['configured']}: check {sh['verdict']}"
        + (f", best {best} ({_frac(fr.get(best))})" if best is not None else "")
        + f", configured {_frac(fr.get(sh['configured']))} of {sh['words']} S1 words")
    ch = [c for c in summary["channels"] if c["hits"]]
    lines.append("rates (Hz, hits / covered time): "
                 + ("  ".join(f"{c['label']} {_fmt_rate(c['rate_hz'])}" for c in ch)
                    or "no kept hits in good frames"))
    mm = [c for c in ch if c["mismatch_frac"]]
    if mm:
        lines.append("fine/coarse mismatch: "
                     + "  ".join(f"{c['label']} {_frac(c['mismatch_frac'])}" for c in mm))
    lines.append("timed efficiency given S1: "
                 + "  ".join(f"{e['counter']} {_eff(e)}" for e in summary["efficiency"]))
    rf = summary["rf"]
    lines.append(f"RF: valid gate {_frac(rf['valid_frac'])}, vetoed {_frac(rf['vetoed_frac'])} "
                 f"of {rf['n_s1']} S1 hits")
    flags = summary["flags"]
    if flags:
        lines.append(f"flags ({len(flags)}):")
        for fl in flags:
            lines.append(f"  {fl['severity'].upper():5s} {fl['code']}: {fl['text']}")
    else:
        lines.append("flags: none")
    for e in stats.bank_errors[:5]:
        lines.append(f"bank error: {e}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the one-page figure
# ---------------------------------------------------------------------------

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def _edges(h, axis="x"):
    a = getattr(h, axis)
    return np.linspace(a.lo, a.hi, a.n + 1)


def _step(ax, h, counts=None, **kw):
    c = h.counts[1:-1] if counts is None else counts
    ax.stairs(c.astype(np.float64), _edges(h), **kw)


TEXT_PT = 8.5
MAX_TEXT_LINES = 30


def figure_text(summary: dict) -> list[str]:
    """The text block of the summary page, wrapped to the page width."""
    f = summary["frames"]
    sh = summary["shift"]
    fr = dict(zip(sh["scan"], sh["fractions"], strict=True))
    lines = [
        f"frames processed {f['processed']}: good {_good(f)}, stale {f['stale']}, "
        f"suspect {f.get('suspect', 0)}, empty {f['empty']}, rejected {f['rejected']}; "
        "missed by serial "
        f"{f['missed_by_serial']}; gap resets {f['gap_resets']}",
        f"live fraction {_frac(summary['live_fraction'])};  RF valid gate "
        f"{_frac(summary['rf']['valid_frac'])} of {summary['rf']['n_s1']} S1 hits;  "
        "timed efficiency given S1: " + ", ".join(
            f"{e['counter']} {_eff(e, 0)}" for e in summary["efficiency"]),
        f"coarse shift {sh['configured']}: {sh['verdict']}"
        + (f" (best {sh['best']}, {_frac(fr.get(sh['best']))})" if sh["best"] is not None
           else "")
        + "; consistent fraction per shift: "
        + ", ".join(f"{s}: {_frac(x, 0)}" for s, x in fr.items()),
        "",
    ]
    flags = summary["flags"]
    if not flags:
        lines.append("No flags.")
    lines += [f"{fl['severity'].upper()} {fl['code']}: {fl['text']}" for fl in flags]
    wrapped_lines = []
    for ln in lines:
        wrapped_lines.extend(textwrap.wrap(ln, width=125, subsequent_indent="    ") or [""])
    if len(wrapped_lines) > MAX_TEXT_LINES:
        wrapped_lines = wrapped_lines[:MAX_TEXT_LINES - 1] + ["... (see summary.json)"]
    return wrapped_lines


def summary_figure(summary: dict, store: HistStore, title: str):
    """One page: rates, mismatch, ToT, dt, RF phase, pattern, flags. Returns the Figure.

    Built without pyplot (a bare ``Figure`` on an Agg canvas), so it never
    touches the caller's backend.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    # The text block sets the page height: every line gets its room below the
    # plots, so a long flag list never runs off the page or into a panel.
    text = figure_text(summary)
    line_in = TEXT_PT * 1.2 / 72                  # matplotlib's default line spacing
    plots_in, gap_in, top_in, bottom_in = 10.0, 0.8, 0.75, 0.3
    text_in = (len(text) + 1.5) * line_in
    height = top_in + plots_in + gap_in + text_in + bottom_in
    fig = Figure(figsize=(11, height), dpi=100)
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(3, 2, hspace=0.5, wspace=0.25, left=0.08, right=0.97,
                          top=1 - top_in / height,
                          bottom=(bottom_in + text_in + gap_in) / height)
    fig.suptitle(title, fontsize=13)

    def get(name):
        return store.get(f"sma/{name}")

    labels = [c["label"] for c in summary["channels"]]
    xs = np.arange(len(labels))

    # Rates.
    ax = fig.add_subplot(gs[0, 0])
    rates = np.array([c["rate_hz"] or 0.0 for c in summary["channels"]])
    ax.bar(xs, np.where(rates > 0, rates, np.nan), color="#4c72b0")
    if (rates > 0).any():
        ax.set_yscale("log")
    ax.set_xticks(xs, labels, rotation=90, fontsize=8)
    ax.set_ylabel("rate (Hz)")
    ax.set_title("Rate per channel (kept hits / covered time)", fontsize=10)

    # Fine vs coarse mismatch.
    ax = fig.add_subplot(gs[0, 1])
    fvc = get("fine_vs_coarse")
    cons = fvc.counts[1, 1:-1].astype(np.float64)
    mism = fvc.counts[2, 1:-1].astype(np.float64)
    tot = cons + mism
    frac = np.divide(mism, tot, out=np.full_like(tot, np.nan), where=tot > 0)
    ax.bar(xs, frac, color="#c44e52")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(xs, labels, rotation=90, fontsize=8)
    ax.set_ylabel("mismatch fraction")
    ax.set_title("Fine vs coarse inconsistent, per channel", fontsize=10)

    roles = counter_roles(summary)

    # ToT per counter.
    ax = fig.add_subplot(gs[1, 0])
    drawn = False
    for k, (_name, lab, ch) in enumerate(roles):
        pair = [get(f"tot_ch{ch:02d}_lsb{b}") for b in (0, 1)]
        if pair[0] is None:
            continue
        c = pair[0].counts[1:-1] + pair[1].counts[1:-1]
        if c.sum():
            _step(ax, pair[0], c, label=lab, color=COLORS[k % len(COLORS)])
            drawn = True
    if drawn:
        ax.set_yscale("log")
        ax.legend(fontsize=8, ncol=1, loc="upper right")
    ax.set_xlabel("ToT code")
    ax.set_ylabel("hits")
    ax.set_title("ToT per counter (good frames)", fontsize=10)

    # dt to S1.
    ax = fig.add_subplot(gs[1, 1])
    drawn = False
    for k, (name, lab, _ch) in enumerate(roles):
        h = get(f"dt_{name}_S1")
        if h is None or not h.counts.sum():
            continue
        _step(ax, h, label=f"{lab} - S1", color=COLORS[k % len(COLORS)])
        drawn = True
    if drawn:
        ax.set_yscale("log")
        ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("t - t(S1) (ns)")
    ax.set_ylabel("pairs")
    ax.set_title("Counter time minus S1", fontsize=10)

    # RF phase.
    ax = fig.add_subplot(gs[2, 0])
    h = get("rf_phase_s1")
    _step(ax, h, color="#55a868", fill=True, alpha=0.6)
    ax.set_xlabel("last RF pulse - S1 (ns)")
    ax.set_ylabel("S1 hits")
    ax.set_title(f"RF phase of S1 ({h.entries} valid gates)", fontsize=10)

    # Pattern.
    ax = fig.add_subplot(gs[2, 1])
    h = get("pattern")
    ax.bar(np.arange(h.x.n), h.counts[1:-1].astype(np.float64), width=0.9, color="#8172b2")
    if h.counts.sum():
        ax.set_yscale("log")
    nbits = len(roles)
    ax.set_xlabel(f"pattern: bit k = counter S(k+1) within the window ({nbits} counters)")
    ax.set_ylabel("S1 hits")
    ax.set_title("Coincidence pattern per S1", fontsize=10)
    step = max(1, h.x.n // 16)
    ax.set_xticks(np.arange(0, h.x.n, step))

    # Text: frames, shift, flags.
    y0 = (bottom_in + text_in) / height
    fig.text(0.08, y0, "Frames, shift check and flags", va="top", ha="left", fontsize=10,
             weight="bold")
    fig.text(0.08, y0 - 1.5 * line_in / height, "\n".join(text), va="top", ha="left",
             fontsize=TEXT_PT, family="monospace")
    return fig


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="mdqm-sma-file",
        description="Run the SMA DQM plugin over one MIDAS file (.mid, .mid.lz4, .mid.gz) "
                    "with no MIDAS running; writes hists.npz, summary.json, trend.json "
                    "and summary.png.",
        epilog="Exit status: 0 ok, 3 if the summary has an error-severity flag, "
               "2 for a bad argument or file.")
    ap.add_argument("file", nargs="*", help="MIDAS file (with --serial: one or more, in order)")
    ap.add_argument("--shift", type=int, default=None,
                    help="coarse shift (overrides /DQM/SMA/Coarse shift; default "
                         f"{P.SETTINGS_DEFAULTS['Coarse shift']})")
    ap.add_argument("--frames", type=int, default=None,
                    help="process at most this many SMA frames (after --skip)")
    ap.add_argument("--skip", type=int, default=0, help="skip the first N SMA frames")
    ap.add_argument("--out", default=None,
                    help="output directory; default ./sma-file-<run>_<subrun>/")
    ap.add_argument("--settings", default=None,
                    help="settings over the defaults: inline JSON or a JSON file, same "
                         "tree as /DQM/SMA, e.g. '{\"Cuts\": {\"coinc window ns\": 30}}'")
    ap.add_argument("--no-png", action="store_true", help="skip summary.png")
    ap.add_argument("--quiet", action="store_true", help="no stdout summary")
    g = ap.add_argument_group(
        "find one event (the tag copied from SMAEvents: run + serial)",
        "With --serial, nothing is histogrammed: the files (or --run/--dir) are "
        "scanned in order for the readout event (id 301) with that serial, its "
        "words are printed and it is written as a one-event .mid file.")
    g.add_argument("--serial", type=int, default=None, help="the event's serial number")
    g.add_argument("--run", type=int, default=None,
                   help="scan the subruns run<RUN>_*.mid* of --dir in order")
    g.add_argument("--dir", default=".", help="directory for --run (default .)")
    g.add_argument("--words", default=None,
                   help="A:B, only bank words A..B (inclusive; word = 64-bit index in "
                        "H000, as the page shows); all word types are listed then")
    g.add_argument("--max-lines", type=int, default=200,
                   help="at most this many table lines without --words (default 200)")
    g.add_argument("--event-out", default=None,
                   help="where to write the event; default sma_run<run>_serial<serial>.mid "
                        "in --out (or .)")
    return ap


def _default_out(path: Path, run, subrun) -> Path:
    if run is None:
        return Path(f"sma-file-{path.name.split('.')[0]}")
    return Path(f"sma-file-{run}" + (f"_{subrun}" if subrun is not None else ""))


def _subrun_files(run: int, directory: str) -> list[Path]:
    """run<RUN>_<SUB>.mid[.lz4|.gz] in `directory`, in subrun order."""
    out = []
    for f in Path(directory).glob(f"run{run:05d}_*.mid*"):
        if not re.search(r"\.mid(\.lz4|\.gz)?$", f.name):
            continue
        r, sub = MF.run_subrun_from_name(f)
        if r == run:
            out.append((sub if sub is not None else -1, f))
    return [f for _s, f in sorted(out)]


def find_serial(args) -> int:
    """`--serial`: find, print and write one readout event. 0 found, 1 not found."""
    files = [Path(f) for f in args.file]
    if args.run is not None:
        files += _subrun_files(args.run, args.dir)
    if not files:
        print("mdqm-sma-file: --serial needs files or --run/--dir", file=sys.stderr)
        return EXIT_USAGE
    word_range = None
    if args.words:
        try:
            a, b = (int(x) for x in args.words.split(":"))
            word_range = (min(a, b), max(a, b))
        except ValueError:
            print("mdqm-sma-file: --words must be A:B", file=sys.stderr)
            return EXIT_USAGE
    target = int(args.serial)
    for path in files:
        if not path.is_file():
            print(f"mdqm-sma-file: no such file: {path}", file=sys.stderr)
            return EXIT_USAGE
        reader = MF.MidasFile(path, event_ids=P.SmaPlugin.event_ids, keep_raw=True)
        first = None
        for ev in reader:
            s = ev.header.serial_number
            if first is None:
                first = s
                if s > target:
                    # Serials only grow through a run: an earlier file had it.
                    print(f"serial {target} not found: {path.name} starts at serial {s}",
                          file=sys.stderr)
                    return 1
            if s == target:
                run = reader.bor_run_number
                if run is None:
                    run = MF.run_subrun_from_name(path)[0]
                return _show_event(ev, path, run, word_range, args)
        if not args.quiet:
            print(f"  {path.name}: serials {first}..{s if first is not None else '-'}, "
                  "not here", file=sys.stderr)
    print(f"serial {target} not found in {len(files)} file(s)", file=sys.stderr)
    return 1


def _show_event(ev, path: Path, run, word_range, args) -> int:
    from mdqm.plugins.sma import frame_tag

    h = ev.header
    bank = ev.get_bank(W.BANK)
    words = W.words_from_bank(bank.data) if bank is not None else np.zeros(0, dtype="<u8")
    d = W.decode(words, args.shift if args.shift is not None else W.DEFAULT_SHIFT)
    shift = args.shift if args.shift is not None else W.DEFAULT_SHIFT
    diff = W.fine_coarse_diff_ns(d["coarse"], d["fine"], shift)
    cons = W.fine_coarse_consistent(diff, shift)
    _r, subrun = MF.run_subrun_from_name(path)
    print(frame_tag(run, h.event_id, h.serial_number, h.timestamp))
    print(f"file {path}  subrun {subrun}  event position {ev.position} "
          "(0-based among all events of the file = nearline rec-ntuple entry)")
    print(f"{words.size} words: {d['n_trigger']} trigger, {d['n_filler']} filler, "
          f"{d['n_pixel']} pixel; coarse shift {shift}")
    widx = d["word_index"]
    if word_range is not None:
        a, b = word_range
        b = min(b, words.size - 1)
        pos = {int(w): k for k, w in enumerate(widx)}
        print(f"{'word':>7} {'raw':>18} {'type':>7} {'ch':>3} {'ToT':>4} {'fine':>7} "
              f"{'coarse':>9} {'time_ns':>15} fine/coarse")
        for w in range(a, b + 1):
            raw = int(words[w])
            k = pos.get(w)
            if k is None:
                kind = "filler" if raw == W.FILLER else "pixel"
                print(f"{w:7d} 0x{raw:016x} {kind:>7}")
                continue
            print(f"{w:7d} 0x{raw:016x} {'trigger':>7} {int(d['ch'][k]):3d} "
                  f"{int(d['tot'][k]):4d} 0x{int(d['fine'][k]):05x} 0x{int(d['coarse'][k]):07x} "
                  f"{int(d['time'][k]):15d} {'ok' if cons[k] else 'MISMATCH'}")
    else:
        n = d["n_trigger"]
        m = min(n, max(0, args.max_lines))
        print(f"{'word':>7} {'raw':>18} {'ch':>3} {'ToT':>4} {'fine':>7} {'coarse':>9} "
              f"{'time_ns':>15} fine/coarse")
        for k in range(m):
            print(f"{int(widx[k]):7d} 0x{int(d['raw'][k]):016x} {int(d['ch'][k]):3d} "
                  f"{int(d['tot'][k]):4d} 0x{int(d['fine'][k]):05x} 0x{int(d['coarse'][k]):07x} "
                  f"{int(d['time'][k]):15d} {'ok' if cons[k] else 'MISMATCH'}")
        if m < n:
            print(f"... {n - m} more trigger words (--max-lines, or --words A:B)")
    out = Path(args.event_out) if args.event_out else (
        Path(args.out or ".") / f"sma_run{run}_serial{h.serial_number}.mid")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(ev.raw)
    print(f"wrote {out} ({len(ev.raw)} bytes, one MIDAS event: mdump -x, "
          "midas.file_reader and mdqm-sma-file read it)")
    return 0


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.serial is not None:
        return find_serial(args)
    if len(args.file) != 1:
        print("mdqm-sma-file: give one file (or --serial N with files or --run/--dir)",
              file=sys.stderr)
        return EXIT_USAGE
    path = Path(args.file[0])
    if not path.is_file():
        print(f"mdqm-sma-file: no such file: {path}", file=sys.stderr)
        return EXIT_USAGE
    if args.skip < 0 or (args.frames is not None and args.frames < 0):
        print("mdqm-sma-file: --skip and --frames must be >= 0", file=sys.stderr)
        return EXIT_USAGE
    if args.shift is not None and not 0 <= args.shift <= W.MAX_COARSE_SHIFT:
        print(f"mdqm-sma-file: --shift must be 0..{W.MAX_COARSE_SHIFT}", file=sys.stderr)
        return EXIT_USAGE
    try:
        settings = build_settings(load_settings_arg(args.settings), args.shift)
    except (OSError, ValueError, KeyError) as exc:
        print(f"mdqm-sma-file: bad --settings: {exc}", file=sys.stderr)
        return EXIT_USAGE

    run_name, subrun = MF.run_subrun_from_name(path)
    reader = MF.MidasFile(path, event_ids=P.SmaPlugin.event_ids)
    t0 = time.monotonic()
    events = iter(reader)
    try:
        # The begin-of-run serial is the run number; it is read with the first
        # event, before any readout frame reaches the plugin.
        first = next(events, None)
        run = reader.bor_run_number if reader.bor_run_number is not None else run_name

        def chained():
            if first is not None:
                yield first
            yield from events

        plugin, stats = feed(chained(), settings, run, frames=args.frames, skip=args.skip)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"mdqm-sma-file: cannot read {path}: {exc}", file=sys.stderr)
        return EXIT_USAGE
    finally:
        events.close()
    elapsed = time.monotonic() - t0

    summary = offline_summary(plugin)
    trend = command_json(plugin, "sma::trend")
    out = Path(args.out) if args.out else _default_out(path, run, subrun)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"file": str(path.resolve()), "run": run, "subrun": subrun,
            "frames_seen": stats.frames_seen, "frames_fed": stats.frames_fed,
            "frames_skipped": stats.frames_skipped, "t_first": stats.t_first,
            "t_last": stats.t_last, "truncated": reader.truncated,
            "bank_errors": stats.bank_errors, "settings": settings,
            "clock": "event time stamp", "run_active": False}
    save_npz(plugin.store, out / "hists.npz", meta)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out / "trend.json").write_text(json.dumps(trend) + "\n")

    png = None
    if not args.no_png:
        try:
            title = (f"SMA DQM  {path.name}  (run {run}"
                     + (f", subrun {subrun}" if subrun is not None else "")
                     + f", {stats.frames_fed} frames)")
            fig = summary_figure(summary, plugin.store, title)
            png = out / "summary.png"
            fig.savefig(png)
        except ImportError:
            print("mdqm-sma-file: matplotlib not installed, no summary.png "
                  "(pip install 'mdqm[offline]', or pass --no-png)", file=sys.stderr)

    if not args.quiet:
        print(text_summary(summary, stats, path.name, elapsed))
        if reader.truncated:
            print("warning: the file ends inside an event (truncated)")
        if stats.frames_fed == 0:
            print("warning: no SMA readout frames (event 301) were processed")
        print(f"wrote {out}/: hists.npz, summary.json, trend.json"
              + (", summary.png" if png else ""))
    errors = [f for f in summary["flags"] if f["severity"] == "error"]
    return EXIT_ERROR_FLAG if errors else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
