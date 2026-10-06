"""``mdqm-sma-file``: the SMA DQM over one MIDAS file, with no MIDAS running.

The manual path beside the ``sma_analyzer`` daemon. When the daemon or the DAQ
is down, a shifter runs the identical plugin (`mdqm.plugins.sma.SmaPlugin`)
over one subrun file and gets the same numbers the pages would have shown::

    mdqm-sma-file /data/run01008_00001.mid.lz4
    mdqm-sma-file run00682_00005.mid.lz4 --shift 3 --out /tmp/682-5
    mdqm-sma-file FILE --settings '{"Cuts": {"coinc window ns": 30}}' --no-png
    mdqm-sma-file FILE --settings my-sma-settings.json --frames 50 --skip 10
    mdqm-sma-file FILE --merge         # counters = merged TOT + NIM hits (NIM/merge = y)
    mdqm-sma-file FILE --no-merge      # TOT words only, even if --settings merges
    mdqm-sma-file FILE --stage 5 -2.5  # MuPix x/y: the XY table at x = 5, y = -2.5 mm
                                       # (default: the file's begin-of-run ODB)

What it does
------------
The plugin is built with the daemon's defaults (``SETTINGS_DEFAULTS``, what a
fresh ``/DQM/SMA`` holds), overridden by ``--settings`` (inline JSON or a JSON
file, same tree as ``/DQM/SMA``; unknown keys are an error) and then
``--shift`` and ``--merge`` / ``--no-merge``, and applied with ``apply_settings(settings,
rebuild=True)`` as the
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
* ``Sampling`` is ignored: every frame is processed;
* MuPix x/y and the MuPix pairs take the XY table's position
  (``/Equipment/XYTable/Variables/Measured``) from the file's begin-of-run ODB
  dump, as the nearline does, not from a live ODB; ``--stage X Y`` (mm)
  overrides it. The summary says which (``xy.stage.source`` "file" or
  "manual"; "missing", with (0, 0) and a note, when the dump has no such key).

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
``nim.png``        the TOT + NIM page, when a counter has a NIM copy (same condition).
``mupix_xy.png``   the MuPix x/y page (hit maps, tracks all / light / heavy, ToT map),
                   when MuPix/XY is on (same condition).
``mupix_pairs.png`` the unseeded MuPix L1-L2 pairs page (L2 - L1 dt, partners per L1
                   pixel, pairs all / light / heavy), when MuPix/Pairs is on (same
                   condition).

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


def build_settings(overrides: dict | None = None, shift: int | None = None,
                   merge: bool | None = None) -> dict:
    """The full ``/DQM/SMA`` tree the daemon would read from a fresh ODB, overridden;
    ``merge`` sets ``NIM/merge`` (``--merge`` / ``--no-merge``)."""
    s = merge_strict(P.SETTINGS_DEFAULTS, overrides or {})
    if shift is not None:
        s["Coarse shift"] = int(shift)
    if merge is not None:
        s["NIM"]["merge"] = bool(merge)
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


def make_plugin(settings: dict, clock, stage=None) -> P.SmaPlugin:
    """Build the plugin the way the analyzer does: defaults, then the first apply.

    ``stage``: the XY table's position for MuPix x/y, in place of the daemon's
    ODB read (`SmaPlugin.poll_odb`): ``(x, y)`` in mm (source "manual") or
    ``(x, y, source, note)`` (`stage_of`); None leaves it at (0, 0).
    """
    plugin = P.SmaPlugin(HistStore(), clock=clock)
    plugin.apply_settings(settings, rebuild=True)
    if stage is not None:
        x, y, source, note = (*stage, "manual", None) if len(stage) == 2 else stage
        plugin.set_stage(x, y, source=source, note=note)
    return plugin


def _odb_value(payload: bytes, path: str):
    """The value at ``path`` in a begin-of-run ODB dump (MIDAS JSON or XML); None if absent."""
    text = payload.rstrip(b"\0").decode("utf-8", errors="replace").lstrip()
    parts = [p for p in path.split("/") if p]
    if text.startswith("{"):
        node = json.loads(text, strict=False)
        for part in parts:
            if not isinstance(node, dict):
                return None
            hit = [k for k in node if k.lower() == part.lower()]
            if not hit:
                return None
            node = node[hit[0]]
        return node
    if text.startswith("<"):
        import xml.etree.ElementTree as ET

        node = ET.fromstring(text)
        for part in parts:
            nxt = [c for c in node if (c.get("name") or "").lower() == part.lower()]
            if not nxt:
                return None
            node = nxt[0]
        vals = [v.text for v in node.iter("value")] if node.tag == "keyarray" else [node.text]
        return vals
    return None


def stage_of(payload: bytes | None) -> tuple:
    """``(x, y, source, note)``: the XY table's position from a begin-of-run ODB dump."""
    from mdqm.plugins import sma_mupix_xy as X

    if payload:
        try:
            v = _odb_value(payload, X.STAGE_PATH)
            if v is not None:
                v = v if isinstance(v, list) else [v]
                x, y = float(v[0]), float(v[1])
                if np.isfinite(x) and np.isfinite(y):
                    return x, y, "file", None
        except (ValueError, TypeError, IndexError, SyntaxError):
            pass
    where = "the file's begin-of-run ODB" if payload else "the file (no begin-of-run ODB)"
    return (0.0, 0.0, "missing",
            f"{X.STAGE_PATH} not in {where}: (0, 0) mm used; --stage X Y sets it")


@dataclass
class FeedStats:
    frames_seen: int = 0          # id-301 events read (before --skip/--frames)
    frames_fed: int = 0           # handed to process()
    frames_skipped: int = 0
    frames_rejected: int = 0      # process() returned False (no H000, or oversize)
    frames_zero: int = 0          # nothing but zero words (Cuts/drop zero words), not rejected
    bank_errors: list = field(default_factory=list)
    t_first: int | None = None
    t_last: int | None = None


def feed(events, settings: dict, run_number, *, frames: int | None = None, skip: int = 0,
         clock: DataClock | None = None, stage=None) -> tuple[P.SmaPlugin, FeedStats]:
    """Feed every readout event of `events` to a fresh plugin.

    `events` is any iterable of event objects: this module's `MidasFile`, or
    ``midas.file_reader.MidasFile(path, use_numpy=True)`` (the compare script
    uses both). The plugin is built at the first readout frame, with the
    clock already at that frame's time, so the trend does not start with ten
    minutes of empty seconds before the data. ``stage``: see `make_plugin`; None
    with a `MidasFile` takes it from the file's begin-of-run ODB (`stage_of`).
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
            if stage is None and hasattr(events, "bor_odb"):
                # A MidasFile: its begin-of-run ODB has been read by now, as the CLI does.
                stage = stage_of(events.bor_odb)
            plugin = make_plugin(settings, clock, stage)
        err = getattr(ev, "bank_error", None)
        if err:
            stats.bank_errors.append(f"serial {h.serial_number}: {err}")
        if stats.t_first is None:
            stats.t_first = h.timestamp
        stats.t_last = h.timestamp
        stats.frames_fed += 1
        n_zero = plugin.frames_zero
        if not plugin.process(ev, run_number=run_number):
            if plugin.frames_zero > n_zero:
                stats.frames_zero += 1
            else:
                stats.frames_rejected += 1
    if plugin is None:
        plugin = make_plugin(settings, clock, stage)
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


def _light(c: dict) -> str:
    """The light cut of a summary's cuts block; older summaries have no light min."""
    lo = c.get("tot_light_min")
    return f"<= {c['tot_light_max']}" if lo is None else f"in [{lo}, {c['tot_light_max']}]"


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


def mupix_line(mp: dict) -> str:
    """One line on the MuPix pixel words: the in-time shares of S1 hits and the sync verdict."""
    if not mp.get("enabled", True):
        return "MuPix: analysis off (MuPix/max pixel hits per frame = 0)"
    f = mp["fractions"]
    w, sb = mp["window_ns"], mp["sideband_ns"]
    parts = [f"{k} {_frac(f[key]['in'])} (sideband {_frac(f[key]['side'])}, corrected "
             f"{_frac(f[key]['corr'])})" for k, key in (("L1", "L1"), ("L2", "L2"),
                                                          ("L1+L2", "both"))]
    ppf = mp.get("pixels_per_frame")
    return (f"MuPix: S1 hits with a pixel hit in [{w[0]}, {w[1]}) ns (sideband [{sb[0]}, {sb[1]}) "
            f"ns), of {mp['n_s1']}: " + ", ".join(parts)
            + f"; {ppf if ppf is not None else 0:.0f} pixel words per frame"
            + f"; time sync {mp.get('sync', {}).get('state', '?')}")


def nim_lines(summary: dict) -> list[str]:
    """The TOT + NIM lines: merge state, then per counter pair efficiency, purity,
    NIM-only share, median NIM - TOT, the lag votes and the lag state now."""
    nim = summary.get("nim") or {}
    rows = nim.get("counters") or []
    if not rows:
        return []
    held = summary.get("nim_lag_held") or []
    out = [f"TOT + NIM: merge {'on' if summary.get('nim_merge') else 'off'}"
           + (f", NIM-only hits held back (lag) on ch {', '.join(map(str, held))}" if held
              else "")]
    for r in rows:
        lg = r["lag"]
        med = r["median_dt_ns"]
        name = r["label"] if r["label"] == r["counter"] else f"{r['counter']} {r['label']}"
        out.append(f"  {name} + {r['nim_label']} (ch {r['ch']}+{r['nim_ch']}): "
                   f"pair eff {_frac(r['pair_eff'])}, purity {_frac(r['purity'])}, NIM-only "
                   f"{_frac(r['nim_only_frac'])} of hits, median NIM - TOT "
                   f"{'-' if med is None else f'{med:g} ns'}; lag ok {lg['ok']}, faulted "
                   f"{lg['faulted']}, ambiguous {lg['ambiguous']}, none {lg['none']}, not "
                   f"voted {lg.get('skipped', 0)}, state {lg.get('state') or '-'}")
    return out


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
        + (f"zero {f['zero']} ({f.get('zero_words', 0)} zero words dropped), "
           if f.get("zero") or f.get("zero_words") else "")
        + f"rejected {f['rejected']}, missed by serial {f['missed_by_serial']}, "
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
    mp = summary.get("mupix")
    if mp:
        lines.append(mupix_line(mp))
    if summary.get("xy"):
        lines.append(mupix_xy_line(summary["xy"]))
    if summary.get("pairs"):
        lines.append(mupix_pairs_line(summary["pairs"]))
    lines += nim_lines(summary)
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
    ]
    if summary.get("mupix"):
        lines.append(mupix_line(summary["mupix"]))
    lines.append("")
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


def nim_figure(summary: dict, store: HistStore, title: str):
    """The TOT + NIM page: NIM - TOT (near and wide), hit classes and the lag
    votes' input, per counter with a NIM copy; None when there is none."""
    rows = (summary.get("nim") or {}).get("counters") or []
    if not rows:
        return None
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    text = [ln for line in nim_lines(summary)
            for ln in (textwrap.wrap(line, width=125, subsequent_indent="    ") or [""])]
    line_in = TEXT_PT * 1.2 / 72
    plots_in, gap_in, top_in, bottom_in = 7.0, 0.8, 0.75, 0.3
    text_in = (len(text) + 1.5) * line_in
    height = top_in + plots_in + gap_in + text_in + bottom_in
    fig = Figure(figsize=(11, height), dpi=100)
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.25, left=0.08, right=0.97,
                          top=1 - top_in / height,
                          bottom=(bottom_in + text_in + gap_in) / height)
    fig.suptitle(title + "  -  TOT + NIM", fontsize=13)

    def get(name):
        return store.get(f"sma/{name}")

    def overlay(ax, stem, xlabel, ylabel, ttl, xlim=None):
        drawn = False
        for k, r in enumerate(rows):
            h = get(f"{stem}_{r['counter']}")
            if h is None or not h.counts[1:-1].sum():
                continue
            _step(ax, h, label=f"{r['nim_label']} - {r['label']}", color=COLORS[k % len(COLORS)])
            drawn = True
        if drawn:
            ax.set_yscale("log")
            ax.legend(fontsize=8, loc="upper right")
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(ttl, fontsize=10)

    overlay(fig.add_subplot(gs[0, 0]), "nim_dt", "t'_NIM - t'_TOT (ns)", "NIM words",
            "NIM minus the nearest TOT word (after the offsets)", (-60, 60))
    overlay(fig.add_subplot(gs[1, 0]), "nim_dt_wide", "t_NIM - t_TOT (ns)", "pairs",
            "NIM minus TOT within +-2^19 ns (a lag fault shows off 0)")
    overlay(fig.add_subplot(gs[1, 1]), "nim_lag", "(fine - fine(S1)) mod 2^20 (ns)", "NIM words",
            "Lag vote input: NIM fine minus S1 fine")

    # Hit classes, as shares of each counter's hits.
    ax = fig.add_subplot(gs[0, 1])
    names = ("paired", "tot_only", "nim_only", "echo")
    xs = np.arange(len(rows))
    width = 0.8 / len(names)
    for j, name in enumerate(names):
        tot = np.array([max(1, sum(r[x] for x in names)) for r in rows], dtype=np.float64)
        ax.bar(xs + (j - 1.5) * width, [r[name] for r in rows] / tot, width,
               label=name.replace("_", "-"), color=COLORS[j])
    ax.set_xticks(xs, [r["counter"] for r in rows])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share of hits")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.set_title("Hits by class (merged counter hits)", fontsize=10)

    y0 = (bottom_in + text_in) / height
    fig.text(0.08, y0, "Pairing per counter", va="top", ha="left", fontsize=10, weight="bold")
    fig.text(0.08, y0 - 1.5 * line_in / height, "\n".join(text), va="top", ha="left",
             fontsize=TEXT_PT, family="monospace")
    return fig


def mupix_xy_line(xy: dict) -> str:
    """One line on MuPix x/y: the track states, the ToT classes, the stage, the cuts."""
    if not xy.get("enabled"):
        return "MuPix x/y: off (MuPix/XY/enable = n, or the MuPix analysis is off)"
    f, c, st = xy["fractions"], xy["cuts"], xy["stage"]
    return (f"MuPix x/y ({xy['geometry']}): {xy['n_s1']} S1 hits judged: track "
            f"{_frac(f['track'])}, ambiguous {_frac(f['ambiguous'])}, no L1 "
            f"{_frac(f['no_l1'])}, no L2 {_frac(f['no_l2'])}; of the {xy['tracks']} tracks light "
            f"(ToT {_light(c)}) {_frac(xy['light_frac'])}, heavy (ToT >= "
            f"{c['tot_heavy_min']}) {_frac(xy['heavy_frac'])}; cluster square "
            f"{c['cluster_box_px']} px; stage x {st['x_mm']}, y {st['y_mm']} mm ({st['source']}"
            + (f", shift {st['shift_mm'][0]:+g}, {st['shift_mm'][1]:+g} mm" if st["applied"]
               else ", not applied") + ")")


def _map(fig, ax, h, title, xlabel, ylabel):
    """A 2D histogram as an image, empty bins blank, with its colour bar."""
    c = h.counts[1:-1, 1:-1].astype(np.float64)
    img = np.where(c > 0, c, np.nan)
    m = ax.pcolormesh(_edges(h, "x"), _edges(h, "y"), img, cmap="viridis", shading="flat")
    if np.isfinite(img).any():
        cb = fig.colorbar(m, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=7)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(f"{title} ({h.entries})", fontsize=9)


def mupix_xy_figure(summary: dict, store: HistStore, title: str):
    """The MuPix x/y page: hit maps, ToT map, tracks all / light / heavy; None when off."""
    xy = summary.get("xy") or {}
    if not xy.get("enabled") or store.get("sma/mupix_track_xy") is None:
        return None
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    text = textwrap.wrap(mupix_xy_line(xy), width=125, subsequent_indent="    ")
    line_in = TEXT_PT * 1.2 / 72
    plots_in, gap_in, top_in, bottom_in = 13.0, 0.8, 0.75, 0.3
    text_in = (len(text) + 1.5) * line_in
    height = top_in + plots_in + gap_in + text_in + bottom_in
    fig = Figure(figsize=(11, height), dpi=100)
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(4, 3, hspace=0.55, wspace=0.5, left=0.07, right=0.93,
                          top=1 - top_in / height,
                          bottom=(bottom_in + text_in + gap_in) / height)
    fig.suptitle(title + "  -  MuPix x/y (+x beam-left)", fontsize=13)

    def get(name):
        return store.get(f"sma/{name}")

    c = xy["cuts"]
    _map(fig, fig.add_subplot(gs[0, 0]), get("mupix_hits_xy_L1"), "L1 pixel hits in time",
         "x (mm)", "y (mm)")
    _map(fig, fig.add_subplot(gs[0, 1]), get("mupix_hits_xy_L2"), "L2 pixel hits in time",
         "x (mm)", "y (mm)")
    _map(fig, fig.add_subplot(gs[0, 2]), get("mupix_track_tot"), "Tracks: max ToT",
         "L1 max ToT", "L2 max ToT")
    rows = (("", "all tracks"), ("_light", f"light, ToT {_light(c)}"),
            ("_heavy", f"heavy, ToT >= {c['tot_heavy_min']}"))
    for r, (suf, what) in enumerate(rows, start=1):
        _map(fig, fig.add_subplot(gs[r, 0]), get(f"mupix_track_xy{suf}"), f"L1 y / x, {what}",
             "x (mm)", "y (mm)")
        _map(fig, fig.add_subplot(gs[r, 1]), get(f"mupix_track_xxp{suf}"), f"x' / x, {what}",
             "x (mm)", "x' (mrad)")
        _map(fig, fig.add_subplot(gs[r, 2]), get(f"mupix_track_yyp{suf}"), f"y' / y, {what}",
             "y (mm)", "y' (mrad)")
    y0 = (bottom_in + text_in) / height
    fig.text(0.07, y0, "S1-seeded tracks", va="top", ha="left", fontsize=10, weight="bold")
    fig.text(0.07, y0 - 1.5 * line_in / height, "\n".join(text), va="top", ha="left",
             fontsize=TEXT_PT, family="monospace")
    return fig


def mupix_pairs_line(pr: dict) -> str:
    """One line on the unseeded MuPix pairs: paired fraction, partners, classes, window, stage."""
    if not pr.get("enabled"):
        return f"MuPix pairs: off ({pr.get('off_reason') or 'MuPix/Pairs/enable = n'})"
    c, st = pr["cuts"], pr["stage"]
    return (f"MuPix pairs (unseeded, nearest L2 pixel within +-{pr['window_ns']} ns, raw times): "
            f"{pr['n_l1']} L1 pixels sampled (at most {pr['max_l1']} a frame), paired "
            f"{_frac(pr['paired_frac'])}, "
            + (f"{pr['mean_partners']}" if pr["mean_partners"] is not None else "-")
            + " L2 candidates per pair; of the "
            f"{pr['n_pairs']} pairs light (both ToT {_light(c)}) "
            f"{_frac(pr['light_frac'])}, heavy (both ToT >= {c['tot_heavy_min']}) "
            f"{_frac(pr['heavy_frac'])}; stage x {st['x_mm']}, y {st['y_mm']} mm ({st['source']}"
            + (f", shift {st['shift_mm'][0]:+g}, {st['shift_mm'][1]:+g} mm" if st["applied"]
               else ", not applied") + "). Entries are pixel pairs, not particles.")


def mupix_pairs_figure(summary: dict, store: HistStore, title: str):
    """The unseeded MuPix pairs page: dt, partners, pairs all / light / heavy; None when off."""
    pr = summary.get("pairs") or {}
    if not pr.get("enabled") or store.get("sma/mupix_pair_xy") is None:
        return None
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    text = textwrap.wrap(mupix_pairs_line(pr), width=125, subsequent_indent="    ")
    line_in = TEXT_PT * 1.2 / 72
    plots_in, gap_in, top_in, bottom_in = 13.0, 0.8, 0.75, 0.3
    text_in = (len(text) + 1.5) * line_in
    height = top_in + plots_in + gap_in + text_in + bottom_in
    fig = Figure(figsize=(11, height), dpi=100)
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(4, 3, hspace=0.55, wspace=0.5, left=0.07, right=0.93,
                          top=1 - top_in / height,
                          bottom=(bottom_in + text_in + gap_in) / height)
    fig.suptitle(title + "  -  MuPix L1-L2 pairs, unseeded (+x beam-left)", fontsize=13)

    def get(name):
        return store.get(f"sma/{name}")

    w = pr["window_ns"]
    hd = get("mupix_pair_dt")
    ax = fig.add_subplot(gs[0, 0:2])
    _step(ax, hd, color=COLORS[0])
    for x in (-w, w):
        ax.axvline(x, color=COLORS[3], ls="--", lw=1)
    ax.set_xlabel("t(L2) - t(L1) (ns)", fontsize=8)
    ax.set_ylabel("L2 candidates", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(f"L2 - L1 time, every L2 pixel within +-100 ns; window +-{w} ns dashed "
                 f"({hd.entries})", fontsize=9)
    hn = get("mupix_pair_partners")
    ax = fig.add_subplot(gs[0, 2])
    _step(ax, hn, color=COLORS[0])
    ax.set_xlabel("L2 pixels in the window", fontsize=8)
    ax.set_ylabel("L1 pixels", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(f"Partners per L1 pixel ({hn.entries})", fontsize=9)
    c = pr["cuts"]
    rows = (("", "all pairs"), ("_light", f"light, ToT {_light(c)}"),
            ("_heavy", f"heavy, ToT >= {c['tot_heavy_min']}"))
    for r, (suf, what) in enumerate(rows, start=1):
        _map(fig, fig.add_subplot(gs[r, 0]), get(f"mupix_pair_xy{suf}"), f"L1 y / x, {what}",
             "x (mm)", "y (mm)")
        _map(fig, fig.add_subplot(gs[r, 1]), get(f"mupix_pair_xxp{suf}"), f"x' / x, {what}",
             "x (mm)", "x' (mrad)")
        _map(fig, fig.add_subplot(gs[r, 2]), get(f"mupix_pair_yyp{suf}"), f"y' / y, {what}",
             "y (mm)", "y' (mrad)")
    y0 = (bottom_in + text_in) / height
    fig.text(0.07, y0, "Unseeded L1-L2 pixel pairs (the nearline monitor's rule)", va="top",
             ha="left", fontsize=10, weight="bold")
    fig.text(0.07, y0 - 1.5 * line_in / height, "\n".join(text), va="top", ha="left",
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
    ap.add_argument("--no-png", action="store_true",
                    help="skip summary.png, nim.png, mupix_xy.png and mupix_pairs.png")
    ap.add_argument("--stage", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help="the XY table's position in mm (/Equipment/XYTable/Variables/Measured) "
                         "for MuPix x/y; default: from the file's begin-of-run ODB, else 0 0. "
                         "Applied as (-X, +Y) unless "
                         "MuPix/XY/apply stage shift = n")
    mg = ap.add_mutually_exclusive_group()
    mg.add_argument("--merge", action="store_true",
                    help="counters are the merged TOT + NIM hits (NIM/merge = y; the default "
                         "is n until the NIM offsets are measured)")
    mg.add_argument("--no-merge", action="store_true",
                    help="counters are the TOT words alone (NIM/merge = n, the default); the "
                         "NIM pairing is still measured")
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
        print(f"{'':>7} {'':>18} {'pixel':>7} chip, col, row, TS2, ToT (x 256 ns), time_ns "
              "(pixel words)")
        for w in range(a, b + 1):
            raw = int(words[w])
            k = pos.get(w)
            if k is None and raw != W.FILLER:
                px = W.pixel_decode(np.array([raw], dtype=np.uint64))
                print(f"{w:7d} 0x{raw:016x} {'pixel':>7} chip {int(px['chip'][0])}, col "
                      f"{int(px['col'][0])}, row {int(px['row'][0])}, TS2 {int(px['ts2'][0])}, "
                      f"ToT {int(px['tot'][0])}, {int(px['time'][0])} ns")
                continue
            if k is None:
                print(f"{w:7d} 0x{raw:016x} {'filler':>7}")
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
        settings = build_settings(load_settings_arg(args.settings), args.shift,
                                  True if args.merge else False if args.no_merge else None)
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

        # The BOR event precedes the first readout frame: its ODB is read by now.
        stage = tuple(args.stage) if args.stage is not None else stage_of(reader.bor_odb)
        plugin, stats = feed(chained(), settings, run, frames=args.frames, skip=args.skip,
                             stage=stage)
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
            "frames_skipped": stats.frames_skipped, "frames_rejected": stats.frames_rejected,
            "frames_zero": stats.frames_zero, "t_first": stats.t_first,
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
            nfig = nim_figure(summary, plugin.store, title)
            if nfig is not None:
                nfig.savefig(out / "nim.png")
            xfig = mupix_xy_figure(summary, plugin.store, title)
            if xfig is not None:
                xfig.savefig(out / "mupix_xy.png")
            pfig = mupix_pairs_figure(summary, plugin.store, title)
            if pfig is not None:
                pfig.savefig(out / "mupix_pairs.png")
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
              + (", summary.png" if png else "")
              + (", nim.png" if png and (out / "nim.png").exists() else "")
              + (", mupix_xy.png" if png and (out / "mupix_xy.png").exists() else "")
              + (", mupix_pairs.png" if png and (out / "mupix_pairs.png").exists() else ""))
    errors = [f for f in summary["flags"] if f["severity"] == "error"]
    return EXIT_ERROR_FLAG if errors else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
