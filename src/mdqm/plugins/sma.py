"""SMA (MuSiP trigger/ToT board): per-frame DQM on every readout frame.

Runs as its own analyzer client (``mdqm-analyzer --plugin sma``), requesting
only the musip readout event (id 301, bank ``H000``). The per-frame arithmetic
is ``sma_words`` (pure numpy, shared with the offline CLI); this module decides
what a frame is worth, fills the histograms, keeps the 1 s trend and the shift
self-check, and serves the ``sma::`` commands.

Frame classes
-------------
Every frame is exactly one of

* **empty**: no trigger words at all;
* **stale**: the words are not this run's data (see `classify_frame`);
* **suspect**: this run's data, but the time base is wrong -- in practice the
  configured coarse shift (see `classify_frame`);
* **good**: everything else.

Stale and suspect frames count in the health histograms (``word_types``,
``words_per_frame``, ``trigger_words_per_frame``, ``stale_per_ch``,
``frame_class``, ``s1_best_frac``) and in the summary, stay available to the
raster view with their flag, and are kept out of every physics histogram, the
gap chain and the trend rates. Suspect frames (only) feed the shift check.
Within a good frame the stale-*hit* cluster rule of ``sma_words.prepare_frame``
still applies.

Histograms (all ``sma/...``, uint64 counts; "per S1" = one entry per kept S1 hit)
--------------------------------------------------------------------------------
Health (all frames unless marked good):

``word_types``               6 bins: filler, pixel, trigger kept, trigger stale
                             (cluster rule), trigger in a stale frame, in a suspect frame
``frame_class``              4 bins: good, stale, empty, suspect
``words_per_frame``          [0, Binning/words per frame max), 100 bins
``trigger_words_per_frame``  same axis
``s1_best_frac``             kept S1 words consistent at the best scanned shift,
                             per frame with enough S1; 51 bins on [0, 1.02)
``stale_per_ch``             channel; cluster-stale words + all words of stale frames
``frame_span_log10_ms``      good; log10(span / ms) on [-2, 4), 120 bins
``frame_gap_ms``             good, -max overlap <= gap <= max gap; [gap min, gap max) ms
``live_fraction``            good, same gap cut; min(span, dlast)/dlast, 51 bins on [0, 1.02)
``rate_vs_ch``               good; channel x log10(hits/span in Hz), channels with hits
``words_per_ch``             good; kept hits per channel

ToT / corruption (good frames, kept hits):

``tot_vs_ch_lsb0``/``_lsb1``   channel x ToT code (256), split by fine bit 0
``tot_chNN_lsb0``/``_lsb1``    ToT of each counter and the RF channel
``tot_ge250_per_ch``           channel; ToT >= Cuts/tot corrupt
``fine_coarse_diff``           channel x (coarse - fine) in coarse ticks (2^shift ns)
``fine_vs_coarse``             channel x class: y=0 consistent, 1 mismatch,
                               2+b mismatched word whose fine bit b disagrees
``fine_bit_occupancy``         channel x fine bit (0..19) set

Timing (good frames; S1 = the "s1" role):

``dt_S{k}_S1``      counter k (S2..S5) minus S1, every pair in +-Cuts/dt window, 1 ns bins
``pattern``         per S1: bit k = counter k within +-coinc window (32 bins)
``s1_coinc``        counter index; per S1 with that counter in the window (entries = S1 hits)
``s1_partner_hits`` counter index x partners in the window per S1
``s1_spacing_us``   consecutive kept S1 hits

RF / delayed (good frames):

``rf_npulses``          per S1: RF pulses in (t, t + gate]
``rf_phase_s1``         valid gates: last pulse - S1 (ns), 1 ns bins
``rf_period``           valid gates: last - second-to-last pulse (ns)
``rf_phase_vs_s1_tot``  phase x S1 ToT
``delayed_dt_chNN``     delayed channel minus S1, all pairs in [lo, hi], in us

Shift (good frames):

``shift_check``     good and suspect frames: consistent S1 words (every S1 word of
                    the frame, kept or not) per scanned shift (3, 12-16 and the
                    configured one); entries = S1 words, so a bin over the entries
                    is the consistent fraction

Commands: ``sma::summary``, ``sma::trend``, ``sma::frame``, ``sma::raw`` (see `commands`).
"""

from __future__ import annotations

import copy
import json
import math
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field

import numpy as np

from mdqm.dqm import framing
from mdqm.dqm.hist import Axis, Hist1D, Hist2D
from mdqm.plugins import sma_words as W

NCH = W.N_CHANNELS
CH_AXIS = (NCH, -0.5, NCH - 0.5)

#: Everything under ``/DQM/SMA``. A dict is an ODB directory, anything else a key.
SETTINGS_DEFAULTS: dict[str, object] = {
    #: The board's coarse = time >> shift. A per-run board setting; the self
    #: check says when another shift fits the data better.
    "Coarse shift": W.DEFAULT_SHIFT,
    "Channel roles": {
        "s1": 1,
        #: S1..S5 in order; the first must be the s1 channel for the pattern
        #: bit 0 to mean "S1".
        "counters": [1, 2, 3, 4, 5],
        "rf": 6,
        "current": 7,
        "delayed": [8, 9, 10],
        #: Per-channel display names, 16 entries. Empty means the role name
        #: (S1..S5, RF, current) or "chNN". Changing them never resets a plot.
        "labels": [""] * NCH,
    },
    "Cuts": {
        "coinc window ns": W.COINC_NS,
        "dt window ns": W.DT_WINDOW_NS,
        "seed pre ns": W.SEED_PRE_NS,
        "seed post ns": W.SEED_POST_NS,
        "seeds": W.N_SEEDS,
        "rf gate ns": W.RF_GATE_NS,
        "rf min pulses": W.RF_MIN_PULSES,
        "rf max pulses": W.RF_MAX_PULSES,
        "latch margin ns": W.LATCH_MARGIN_NS,
        #: See sma_words.STALE_GAP_NS: genuine hits of a frame are never more
        #: than ~4 ms apart, words left over from an earlier run 180 ms or more.
        "stale gap ms": W.STALE_GAP_NS / 1e6,
        #: A far cluster with this many words, this fraction consistent at the
        #: best scanned shift, is genuine (beam trip, slow buffer) and kept.
        "rescue min words": W.RESCUE_MIN_WORDS,
        "rescue min fraction": W.RESCUE_MIN_FRACTION,
        #: Counters/RF with fewer hits in a frame do not limit where seeds may sit.
        "seed min hits": W.SEED_MIN_HITS,
        "tot corrupt": W.TOT_CORRUPT_MIN,
        "delayed lo ns": W.DELAYED_LO_NS,
        "delayed hi ns": W.DELAYED_HI_NS,
        #: Frames further apart than this are a loop, subrun or run boundary:
        #: no gap or live-fraction entry, and the gap chain restarts.
        "max gap s": 10.0,
        #: Real frames overlap by at most ~1 ms; a frame starting further back
        #: than this is a replay loop or a clock restart, and resets the chain
        #: like a long gap does.
        "max overlap ms": 10.0,
        #: Stale-frame rule, see classify_frame.
        "stale frame min S1": 10,
        "stale frame S1 fraction": 0.9,
        "stale frame junk words": 100,
        "stale frame junk fraction": 0.9,
        #: A frame keeping less than this fraction of its trigger words in its
        #: time clusters has no usable time base: almost always a wrong coarse
        #: shift (the times scatter over the whole 2^40 ns range).
        "suspect kept fraction": 0.5,
        #: At most this many S1 hits per frame get the S1-seeded analyses
        #: (pattern, efficiencies, S2..S5 - S1, RF phase, delayed pairs); a
        #: denser frame uses an evenly spread sample of its S1 hits. Word
        #: counts, rates, ToT, fine/coarse and stale use every hit. 0 = no cap.
        "max S1 per frame": 2000,
        #: Frames with more words than this (1 Mi words = 8 MiB) are counted as
        #: "oversize" and not decoded: one would take ~0.6 s of CPU and
        #: hundreds of MB. 0 = no limit.
        "max words per frame": 1 << 20,
    },
    "Binning": {
        "words per frame max": 50000,
        #: Log axis: spans run from ~10 ms to seconds (beam trips, slow runs).
        "span log10 ms min": -2.0,
        "span log10 ms max": 4.0,
        "span bins": 120,
        "gap min ms": -5.0,
        "gap max ms": 45.0,
        "gap bins": 100,
        "rate log10 Hz min": 0.0,
        "rate log10 Hz max": 7.0,
        "rate bins": 140,
        "s1 spacing max us": 500.0,
        "s1 spacing bins": 250,
        "rf period max ns": 60,
        "delayed bins": 220,
        "partner max": 9,
    },
    "Self check": {
        #: Shift verdict over this many seconds of good frames.
        "shift window s": 30.0,
        #: Another shift must beat the configured one by this much...
        "shift margin": 0.2,
        #: ...and reach this consistent fraction...
        "shift min fraction": 0.9,
        #: ...on at least this many S1 words.
        "shift min words": 1000,
        #: sma::summary averages over this many seconds.
        "summary window s": 60.0,
        "mismatch warn fraction": 0.05,
        "mismatch error fraction": 0.5,
        "tot corrupt warn fraction": 0.05,
        #: Channels with fewer hits in the window are not judged.
        "min hits": 200,
        "no frames s": 5.0,
        #: Efficiency drop: the last "efficiency window s" against the whole
        #: trend (10 min), per counter.
        "efficiency window s": 30.0,
        "efficiency drop": 0.1,
        #: Only these channels raise mismatch / ToT flags (S1, the counters
        #: and the RF by default); the others are still in the summary table.
        #: Here and not under Cuts, so editing it never resets a plot.
        "mismatch flag channels": [1, 2, 3, 4, 5, 6],
    },
    "Sampling": {
        #: The analyzer's CPU, in % of one core, everything included (reading
        #: the buffer, decoding, filling, answering the pages). It analyses as
        #: many frames per second as fit and skips the rest without reading
        #: them. At most 50 (the analyzer clamps it); no budget only with
        #: mdqm-analyzer --no-cpu-budget (development). 0: analyse nothing.
        "CPU budget %": 20.0,
        #: A hard cap on analysed frames per second, on top of the budget.
        "max events per s": 1000.0,
        #: The raw bytes of the last analysed frames, for "Download raw event"
        #: (sma::raw), at most this many MB in all. 0: none kept.
        "raw ring MB": 16.0,
    },
}

TREND_S = 600
#: Frame classes, in the order of the ``frame_class`` bins.
CLASSES = ("good", "stale", "empty", "suspect")


def _num(x, digits=6):
    """A JSON-safe float: None for NaN/inf (``JSON.parse`` rejects NaN)."""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    return float(f"{x:.{digits}g}")


def _ratio(a, b, digits=6):
    return _num(a / b, digits) if b else None


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """The parsed ``/DQM/SMA`` tree; every field has a safe value."""

    shift: int = W.DEFAULT_SHIFT
    roles: W.Roles = field(default_factory=W.Roles)
    cuts: W.Cuts = field(default_factory=W.Cuts)
    labels: list = field(default_factory=lambda: [""] * NCH)
    stale_min_s1: int = 10
    stale_s1_frac: float = 0.9
    stale_junk_words: int = 100
    stale_junk_frac: float = 0.9
    suspect_kept_frac: float = 0.5
    flag_channels: tuple = (1, 2, 3, 4, 5, 6)
    max_words: int | None = 1 << 20
    max_gap_ns: int = 10 * 10**9
    max_overlap_ns: int = 10 * 10**6
    raw_ring_bytes: int = 16 << 20
    binning: dict = field(default_factory=dict)
    check: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)

    @property
    def scan(self) -> tuple:
        return tuple(self.cuts.shift_scan)

    def role_channels(self) -> set:
        r = self.roles
        return {r.s1, r.rf, r.current, *r.counters, *r.delayed}


def _merge(defaults: dict, given: dict | None) -> dict:
    out = copy.deepcopy(defaults)
    for k, v in (given or {}).items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = _merge(out[k], v)
        elif k in out:
            out[k] = v
    return out


def parse_settings(settings: dict | None) -> Config:
    """Settings dict -> Config. Never raises: a bad value falls back to its default.

    An operator typing into the ODB must not be able to stop the analyzer; the
    fallback is reported in `Config.errors` (and in dqm::status).
    """
    s = _merge(SETTINGS_DEFAULTS, settings)
    d = SETTINGS_DEFAULTS
    errors: list[str] = []

    def get(section, key, conv, check=None):
        raw = s[section][key] if section else s[key]
        dflt = d[section][key] if section else d[key]
        try:
            v = conv(raw)
            if check is not None and not check(v):
                raise ValueError("out of range")
            return v
        except (TypeError, ValueError) as exc:
            errors.append(f"{section + '/' if section else ''}{key}={raw!r}: {exc}; "
                          f"using {dflt!r}")
            return conv(dflt)

    def chan(v):
        v = int(v)
        if not 0 <= v < NCH:
            raise ValueError("not a channel 0-15")
        return v

    def chans(v):
        v = [chan(x) for x in (v if isinstance(v, list | tuple) else [v])]
        return tuple(v)

    shift = get(None, "Coarse shift", int, lambda v: 0 <= v <= W.MAX_COARSE_SHIFT)
    R = "Channel roles"
    roles = W.Roles(s1=get(R, "s1", chan), counters=get(R, "counters", chans, len),
                    rf=get(R, "rf", chan), current=get(R, "current", chan),
                    delayed=get(R, "delayed", chans))
    raw_labels = s[R]["labels"]
    raw_labels = raw_labels if isinstance(raw_labels, list | tuple) else [raw_labels]
    labels = [str(x) for x in raw_labels][:NCH]
    labels += [""] * (NCH - len(labels))

    C = "Cuts"
    pos = lambda v: v > 0  # noqa: E731
    cuts = W.Cuts(
        latch_margin_ns=get(C, "latch margin ns", int, lambda v: v >= 0),
        stale_gap_ns=int(get(C, "stale gap ms", float, pos) * 1e6),
        tot_corrupt_min=get(C, "tot corrupt", int),
        rf_gate_ns=get(C, "rf gate ns", float, pos),
        rf_min_pulses=get(C, "rf min pulses", int),
        rf_max_pulses=get(C, "rf max pulses", int),
        coinc_ns=get(C, "coinc window ns", int, lambda v: v >= 0),
        dt_window_ns=get(C, "dt window ns", int, pos),
        delayed_lo_ns=get(C, "delayed lo ns", int),
        delayed_hi_ns=get(C, "delayed hi ns", int),
        seed_pre_ns=get(C, "seed pre ns", int, lambda v: v >= 0),
        seed_post_ns=get(C, "seed post ns", int, lambda v: v >= 0),
        n_seeds=get(C, "seeds", int, lambda v: v >= 0),
        # The scan always covers 12..16 and the configured shift, so the
        # verdict can compare the two whatever is configured.
        shift_scan=tuple(sorted(set(W.SHIFT_SCAN) | {shift})),
        seed_min_hits=get(C, "seed min hits", int, lambda v: v >= 1),
        rescue_min_words=get(C, "rescue min words", int, lambda v: v >= 1),
        rescue_min_fraction=get(C, "rescue min fraction", float),
        max_s1=get(C, "max S1 per frame", int, lambda v: v >= 0) or None,
    )
    if cuts.delayed_hi_ns <= cuts.delayed_lo_ns:
        errors.append("Cuts/delayed hi ns <= delayed lo ns; using the defaults")
        cuts.delayed_lo_ns, cuts.delayed_hi_ns = W.DELAYED_LO_NS, W.DELAYED_HI_NS

    B = "Binning"
    binning = {
        "words max": get(B, "words per frame max", int, pos),
        "span min": get(B, "span log10 ms min", float),
        "span max": get(B, "span log10 ms max", float),
        "span bins": get(B, "span bins", int, pos),
        "gap min": get(B, "gap min ms", float),
        "gap max": get(B, "gap max ms", float),
        "gap bins": get(B, "gap bins", int, pos),
        "rate min": get(B, "rate log10 Hz min", float),
        "rate max": get(B, "rate log10 Hz max", float),
        "rate bins": get(B, "rate bins", int, pos),
        "spacing max": get(B, "s1 spacing max us", float, pos),
        "spacing bins": get(B, "s1 spacing bins", int, pos),
        "period max": get(B, "rf period max ns", int, pos),
        "delayed bins": get(B, "delayed bins", int, pos),
        "partner max": get(B, "partner max", int, pos),
    }
    if binning["gap max"] <= binning["gap min"]:
        errors.append("Binning/gap max ms <= gap min ms; using the defaults")
        binning["gap min"], binning["gap max"] = -5.0, 45.0
    if binning["span max"] <= binning["span min"]:
        errors.append("Binning/span log10 ms max <= min; using the defaults")
        binning["span min"], binning["span max"] = -2.0, 4.0
    if binning["rate max"] <= binning["rate min"]:
        errors.append("Binning/rate log10 Hz max <= min; using the defaults")
        binning["rate min"], binning["rate max"] = 0.0, 7.0

    S = "Self check"
    check = {k: get(S, k, float) for k in d[S] if k != "mismatch flag channels"}
    flag_channels = get(S, "mismatch flag channels", chans)

    return Config(
        shift=shift, roles=roles, cuts=cuts, labels=labels,
        stale_min_s1=get(C, "stale frame min S1", int, lambda v: v >= 1),
        stale_s1_frac=get(C, "stale frame S1 fraction", float),
        stale_junk_words=get(C, "stale frame junk words", int, lambda v: v >= 1),
        stale_junk_frac=get(C, "stale frame junk fraction", float),
        suspect_kept_frac=get(C, "suspect kept fraction", float),
        flag_channels=flag_channels,
        max_words=get(C, "max words per frame", int, lambda v: v >= 0) or None,
        raw_ring_bytes=int(get("Sampling", "raw ring MB", float, lambda v: v >= 0) * (1 << 20)),
        max_gap_ns=int(get(C, "max gap s", float, pos) * 1e9),
        max_overlap_ns=int(get(C, "max overlap ms", float, lambda v: v >= 0) * 1e6),
        binning=binning, check=check, errors=errors)


def shape_fingerprint(settings: dict) -> str:
    """What changes the histograms: shift, roles (but not labels), cuts, binning.

    Labels are left out on purpose -- renaming a channel must never reset an
    afternoon of plots. The self-check and sampling settings change no plot.
    """
    s = _merge(SETTINGS_DEFAULTS, settings)
    roles = {k: v for k, v in s["Channel roles"].items() if k != "labels"}
    return json.dumps({"shift": s["Coarse shift"], "roles": roles, "Cuts": s["Cuts"],
                       "Binning": s["Binning"]}, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# the stale-frame rule
# ---------------------------------------------------------------------------

def kept_s1_scan(fr: W.Frame, cfg: Config) -> tuple[int, np.ndarray]:
    """``(n, counts)``: kept S1 words and how many are consistent at each scanned shift."""
    idx = fr.order[fr.chan[cfg.roles.s1]]
    return int(idx.size), W.shift_scan(fr.coarse[idx], fr.fine[idx], cfg.scan,
                                       cfg.cuts.latch_margin_ns)


def classify_frame(fr: W.Frame, cfg: Config, n_s1: int, scan_counts, n_s1_all: int = 0,
                   scan_all=None) -> tuple[str, str, float]:
    """``(class, reason, best S1 fraction)``: "good", "stale", "suspect" or "empty".

    Why a frame-level rule at all: the hit-level cluster rule keeps the time
    cluster holding the median, which is right when a few old words sit in a
    genuine frame -- and wrong when the frame is *mostly* old. At a run start
    the FEB can send its whole previous buffer: the kept cluster is then the
    previous run's data (only ~1-2 % of its S1 words fine/coarse consistent),
    or a cluster of channel 0/15 garbage with no S1 at all.

    The rule, on the kept S1 words (the S1 role):

    1. At least ``stale frame min S1`` of them: stale when the best consistent
       fraction over the scanned shifts (12..16 and the configured one) is
       below ``stale frame S1 fraction`` (0.9). Genuine frames sit at 0.98-1.0
       at their shift; chance is 3/2^(20-s), at most 0.19 at shift 16; the
       replayed buffers measured 0.00-0.55.
    2. Fewer: stale when there are at least ``stale frame junk words`` kept
       words and at least ``stale frame junk fraction`` of them are on channels
       with no role.

    Neither step asks whether the *configured* shift is right. The best
    fraction is over all scanned shifts, and with a 50 ms stale gap the kept
    cluster survives a slightly wrong shift (13 or 15 for 14 misplaces times
    by ~1 ms). The one failure this cannot tell from a replay is S1's own
    fine/coarse breaking at every shift: every frame then turns stale, which
    the summary reports as an error of its own.

    3. Not stale, but the time base is wrong: **suspect**. Either every S1
       word of the frame (``n_s1_all``/``scan_all``, at least ``stale frame
       min S1`` of them) fits another scanned shift (>= ``stale frame S1
       fraction``) while fewer than half fit the configured one, or fewer than
       ``suspect kept fraction`` of the trigger words were kept. A far-wrong
       shift (14 on shift-3 data) stretches or scatters the times; sometimes
       they still form one cluster, which is why the per-frame shift test is
       needed besides the kept fraction. Nothing computed from those times
       means anything, so suspect frames stay out of the physics; they still
       feed the shift check, which reads the fine and coarse fields and needs
       no times -- that is how the page learns which shift to set.
    """
    if fr.n_trigger == 0:
        return "empty", "no trigger words", float("nan")
    best = float(np.max(scan_counts)) / n_s1 if n_s1 else float("nan")
    if n_s1 >= cfg.stale_min_s1 and best < cfg.stale_s1_frac:
        s = cfg.scan[int(np.argmax(scan_counts))]
        return ("stale", f"S1 fine/coarse agree for {best:.0%} of {n_s1} words at best "
                f"(shift {s})", best)
    n_kept = int(fr.s_ch.size)
    if n_s1 < cfg.stale_min_s1 and n_kept >= cfg.stale_junk_words:
        junk_ch = np.ones(NCH, dtype=bool)
        junk_ch[list(cfg.role_channels())] = False
        n_junk = int(np.count_nonzero(junk_ch[fr.s_ch]))
        if n_junk >= cfg.stale_junk_frac * n_kept:
            return ("stale", f"{n_s1} S1 words; {n_junk} of {n_kept} words on channels "
                    "with no role", best)
    if scan_all is not None and n_s1_all >= cfg.stale_min_s1:
        fr_all = np.asarray(scan_all) / n_s1_all
        b = int(np.argmax(fr_all))
        conf = fr_all[cfg.scan.index(cfg.shift)]
        if cfg.scan[b] != cfg.shift and fr_all[b] >= cfg.stale_s1_frac and conf < 0.5:
            return ("suspect", f"S1 fits coarse shift {cfg.scan[b]} ({fr_all[b]:.0%}), not the "
                    f"configured {cfg.shift} ({conf:.0%})", best)
    if n_kept < cfg.suspect_kept_frac * fr.n_trigger:
        return ("suspect", f"only {n_kept} of {fr.n_trigger} words in the frame's time "
                "clusters: the coarse shift is probably wrong", best)
    return "good", "", best


# ---------------------------------------------------------------------------
# per-second accumulators (trend, summary, shift ring)
# ---------------------------------------------------------------------------

class _Second:
    """Everything the trend, summary and shift check need, summed over one second."""

    __slots__ = ("t", "epoch", "frames", "offered", "stale", "empty", "suspect", "span_ns",
                 "cover_ns", "delta_ns", "live_ns", "hits", "rate_hits", "oversize",
                 "mismatch", "tot_bad",
                 "stale_words", "n_s1", "n_s1_kept", "eff", "rf_valid", "rf_vetoed", "scan",
                 "shift_counts", "shift_n")

    def __init__(self, t: int, epoch: int, n_counters: int, scan: tuple):
        self.t = t
        self.epoch = epoch
        self.frames = self.stale = self.empty = self.suspect = 0
        #: Frames sent, from the serial numbers (frames counts the analysed).
        self.offered = 0
        #: span_ns sums the frame spans; cover_ns is their union (overlaps
        #: counted once), the denominator of every rate.
        self.span_ns = self.cover_ns = self.delta_ns = self.live_ns = 0
        self.hits = np.zeros(NCH, dtype=np.int64)
        #: Hits of the frames the rates use (see SmaPlugin._fill_good).
        self.rate_hits = np.zeros(NCH, dtype=np.int64)
        self.oversize = 0
        self.mismatch = np.zeros(NCH, dtype=np.int64)
        self.tot_bad = np.zeros(NCH, dtype=np.int64)
        self.stale_words = np.zeros(NCH, dtype=np.int64)
        #: n_s1: S1 hits analysed (the sample); n_s1_kept: every kept one.
        self.n_s1 = self.n_s1_kept = 0
        self.eff = np.zeros(n_counters, dtype=np.int64)
        self.rf_valid = self.rf_vetoed = 0
        self.scan = scan
        self.shift_counts = np.zeros(len(scan), dtype=np.int64)
        self.shift_n = 0

    def row(self, counters=(), mismatch_max: float | None = None) -> dict:
        """One trend row. With `counters` and `mismatch_max`, a counter whose
        fine/coarse mismatch fraction this second exceeds `mismatch_max` gets
        a null efficiency -- the rule `summary` applies to its window: with a
        timestamp fault the coincidence misses and the efficiency would read
        as a dead counter rather than as the fault it is."""
        cover_s = self.cover_ns * 1e-9
        eff = None
        if self.n_s1:
            eff = [_ratio(e, self.n_s1, 4) for e in self.eff[1:]]
            if mismatch_max is not None:
                for k, c in enumerate(list(counters)[1:len(eff) + 1]):
                    h = int(self.hits[c])
                    if h and self.mismatch[c] / h > mismatch_max:
                        eff[k] = None
        return {
            "t": self.t,
            "frames": self.frames,
            "offered": self.offered,
            "stale": self.stale,
            "suspect": self.suspect,
            "rate_hz": [_ratio(h, cover_s, 4) for h in self.rate_hits] if cover_s > 0 else None,
            "live": _ratio(self.live_ns, self.delta_ns, 4),
            "n_s1": self.n_s1,
            # Counters after S1 only: S1's own entry is 1 by definition.
            "eff": eff,
            "rf_valid": _ratio(self.rf_valid, self.n_s1, 4),
        }


@dataclass
class _RawEntry:
    """One analysed frame's MIDAS event, as bytes, for sma::raw."""

    seq: int
    run: int
    serial: int
    timestamp: int
    event_id: int
    data: bytes


def event_bytes(event, data=None) -> bytes:
    """The MIDAS event (16-byte header + bank header + banks) of a received event.

    Preferably the bytes as received: ``event.mdqm_raw`` (set by the analyzer
    from the client's receive buffer) or ``event.raw`` (mdqm's file reader with
    ``keep_raw``). Else a ``midas.event.Event`` packs itself (every bank, its own
    bank format; bank32a's reserved header bytes become 0). Otherwise (``mdqm.tools.midasfile`` events, tests) the banks are encoded
    again as bank32a, which is what the SMA readout writes. Either way the
    bytes on their own are a valid one-event .mid file: ``mdqm.tools.midasfile``,
    ``midas.file_reader`` and ``mdump`` all read it without a BOR record.
    """
    for attr in ("mdqm_raw", "raw"):
        got = getattr(event, attr, None)
        if isinstance(got, bytes | bytearray) and len(got) >= 16:
            return bytes(got)
    pack = getattr(event, "pack", None)
    if pack is not None:
        try:
            buf = pack()
            return bytes(getattr(buf, "raw", buf))
        except Exception:                               # noqa: BLE001
            pass
    from mdqm.tools import midasfile as MF

    h = event.header
    banks = []
    for name, b in (getattr(event, "banks", None) or {}).items():
        d = getattr(b, "data", None)
        if d is None:
            continue
        tid = int(getattr(b, "type", 0) or 0)
        if not tid:
            tid, d = 6, np.asarray(d, dtype="<u4")          # TID_DWORD, as H000
        banks.append((name, tid, d if isinstance(d, np.ndarray | bytes) else bytes(d)))
    if not banks and data is not None:
        banks = [(W.BANK, 6, np.asarray(data, dtype="<u4"))]
    return MF.encode_event(int(h.event_id), int(getattr(h, "serial_number", 0) or 0),
                           int(getattr(h, "timestamp", 0) or 0), banks,
                           trigger_mask=int(getattr(h, "trigger_mask", 0) or 0))


def frame_tag(run, event_id, serial, timestamp, seq=None) -> str:
    """The text a shifter pastes into the elog to find a frame again.

    ``SMA run 1008 · event 301 serial 4757 · 2026-09-28 07:31:02 UTC · frame seq 4750``.
    Run + serial find the event in the files (`mdqm-sma-file --serial`); the
    timestamp is the event header's (1 s); the frame seq is this analyzer's
    own counter, useful only while it runs.
    """
    import datetime as _dt

    ts = (_dt.datetime.fromtimestamp(int(timestamp), _dt.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
          if timestamp else "time unknown")
    out = f"SMA run {run if run else '?'} · event {event_id} serial {serial} · {ts}"
    return out + (f" · frame seq {seq}" if seq is not None else "")


@dataclass
class _Snapshot:
    """The last frame, as the event display needs it."""

    seq: int
    run: int
    serial: int
    fr: W.Frame
    an: W.FrameAnalysis | None
    cls: str
    reason: str
    gap_ns: int | None
    shift: int
    tot_min: int
    pre_ns: int
    post_ns: int
    #: The MIDAS event header of the frame: what finds it again in a file.
    event_id: int = W.EVID_READOUT
    timestamp: int = 0
    trigger_mask: int = 0


# ---------------------------------------------------------------------------
# the plugin
# ---------------------------------------------------------------------------

class SmaPlugin:
    """Classify, analyse and histogram every SMA readout frame."""

    name = "sma"
    client_name = "sma_analyzer"
    event_ids = frozenset({W.EVID_READOUT})
    #: The musip DAQ-loss counter is not known yet (an open item at PSI), so
    #: the throttle is off rather than watching the WaveDREAM one.
    dropped_path = ""
    #: Sampled by CPU budget (``Sampling/CPU budget %``), until the ODB says
    #: otherwise: a live peek for shifters, not a lossless record.
    cpu_budget_pct = 20.0
    #: Consecutive frames per analysis slot when sampling, so that the gap and
    #: live fraction (which need a frame and the one before it) are measured.
    sample_burst = 2
    #: Ask the analyzer for each analysed event's bytes as received (raw ring).
    wants_raw = True
    settings_root = "/DQM/SMA"
    settings_defaults = SETTINGS_DEFAULTS

    def __init__(self, store, settings: dict | None = None, clock=time.time):
        self.store = store
        self._clock = clock
        self.cfg = parse_settings(settings)
        self.run_number = None
        #: Set by the analyzer from /Runinfo/State; None when nobody has.
        self.run_active = None
        self.epoch = 0
        self.frames = 0
        self.frames_stale = 0
        self.frames_empty = 0
        self.frames_suspect = 0
        self.frames_rejected = 0
        #: Frames above Cuts/max words per frame: counted, not decoded.
        self.frames_oversize = 0
        self.gap_resets = 0
        self.missed_by_serial = 0
        #: Frames the DAQ sent, from the serial numbers of the analysed ones.
        self.offered_by_serial = 0
        #: Gap chains broken because the next analysed frame was not the next
        #: one sent (sampling, or loss).
        self.serial_breaks = 0
        #: Set by the analyzer (CPU budget, measured CPU, rate limit); None
        #: when nobody has (tests, the offline CLI).
        self.sampling_state = None
        self.rebuilds = 0
        self.last_frame_at = None
        self._last_serial = None
        self._prev_extent = None
        self._prev_timed = None
        self._last: _Snapshot | None = None
        self._last_good: _Snapshot | None = None
        self._last_seeded: _Snapshot | None = None
        self._frame_cache: dict = {}
        #: seq -> _RawEntry of the last analysed frames, bounded by
        #: Sampling/raw ring MB (see _keep_raw).
        self._raw: OrderedDict[int, _RawEntry] = OrderedDict()
        self._raw_bytes = 0
        self.raw_not_kept = 0
        self._seconds: deque[_Second] = deque(maxlen=TREND_S + 1)
        self._cur: _Second | None = None
        self._build()

    # -- settings --------------------------------------------------------------

    def shape_fingerprint(self, settings: dict) -> str:
        return shape_fingerprint(settings)

    def apply_settings(self, settings: dict, rebuild: bool) -> None:
        """Adopt new settings; rebuild (and so reset) the histograms only if asked.

        A rebuild also starts a new summary epoch: a mismatch fraction or an
        efficiency measured under the old shift or cuts is not comparable with
        one under the new, so the summary does not average across the change.
        The trend keeps its rows (rates do not depend on the cuts).
        """
        self.cfg = parse_settings(settings)
        if rebuild:
            for name in self.store.names():
                if name.startswith("sma/"):
                    self.store.remove(name)
            self._build()
            self.rebuilds += 1
            self._new_epoch()

    def _new_epoch(self) -> None:
        self.epoch += 1
        self._roll(force=True)

    # -- histograms ------------------------------------------------------------

    def _h1(self, name, axis, title):
        return self.store.add(Hist1D(f"sma/{name}", axis, title=title, dtype=np.uint64))

    def _h2(self, name, x, y, title):
        return self.store.add(Hist2D(f"sma/{name}", x, y, title=title, dtype=np.uint64))

    def _build(self) -> None:
        cfg, b, c = self.cfg, self.cfg.binning, self.cfg.cuts
        cha = Axis(*CH_AXIS, "SMA channel")
        tot = Axis(256, 0, 256, "ToT code")
        self.h = h = {}

        h["word_types"] = self._h1("word_types", Axis(6, -0.5, 5.5, "0 filler, 1 pixel, "
                                   "2 trigger kept, 3 trigger stale hit, 4 in stale frame, "
                                   "5 in suspect frame"),
                                   "Readout words by type")
        h["frame_class"] = self._h1("frame_class", Axis(4, -0.5, 3.5,
                                                        "0 good, 1 stale, 2 empty, 3 suspect"),
                                    "Frames by class")
        wax = Axis(100, 0, b["words max"], "words")
        h["words_per_frame"] = self._h1("words_per_frame", wax, "Words per frame (H000)")
        h["trigger_words_per_frame"] = self._h1("trigger_words_per_frame", wax,
                                                "SMA trigger words per frame")
        h["s1_best_frac"] = self._h1("s1_best_frac", Axis(51, 0, 1.02, "fraction"),
                                     "Kept S1 words consistent at the best shift, per frame")
        h["stale_per_ch"] = self._h1("stale_per_ch", cha, "Stale words per channel")
        h["frame_span_log10_ms"] = self._h1(
            "frame_span_log10_ms", Axis(b["span bins"], b["span min"], b["span max"],
                                        "log10(span / ms)"), "Frame span (kept hits)")
        h["frame_gap_ms"] = self._h1("frame_gap_ms", Axis(b["gap bins"], b["gap min"],
                                                          b["gap max"], "ms"),
                                     "Gap to the previous frame (negative: overlap)")
        h["live_fraction"] = self._h1("live_fraction", Axis(51, 0, 1.02, "fraction"),
                                      "Live fraction per frame")
        h["rate_vs_ch"] = self._h2("rate_vs_ch", cha, Axis(b["rate bins"], b["rate min"],
                                                           b["rate max"], "log10(rate / Hz)"),
                                   "Per-frame rate by channel")
        h["words_per_ch"] = self._h1("words_per_ch", cha, "Kept hits per channel")

        h["tot_vs_ch"] = [self._h2(f"tot_vs_ch_lsb{k}", cha, tot,
                                   f"ToT by channel, fine bit 0 = {k}") for k in (0, 1)]
        self._tot_chans = sorted(set(cfg.roles.counters) | {cfg.roles.rf})
        h["tot_ch"] = {ch: [self._h1(f"tot_ch{ch:02d}_lsb{k}", tot,
                                     f"ToT, channel {ch}, fine bit 0 = {k}") for k in (0, 1)]
                       for ch in self._tot_chans}
        h["tot_ge250_per_ch"] = self._h1("tot_ge250_per_ch", cha,
                                         f"Hits with ToT >= {c.tot_corrupt_min}")
        half = 1 << max(0, 19 - cfg.shift)
        self._fc_half = half
        nd = 2 * half if cfg.shift < W.FINE_BITS else 1
        h["fine_coarse_diff"] = self._h2(
            "fine_coarse_diff", cha,
            Axis(nd, -half - 0.5, nd - half - 0.5, f"coarse - fine (ticks of 2^{cfg.shift} ns)"),
            "Coarse minus fine, by channel")
        h["fine_vs_coarse"] = self._h2(
            "fine_vs_coarse", cha,
            Axis(W.FINE_BITS + 2, -0.5, W.FINE_BITS + 1.5,
                 "0 consistent, 1 mismatch, 2+b fine bit b differs"),
            "Fine vs coarse, by channel")
        h["fine_bit_occupancy"] = self._h2("fine_bit_occupancy", cha,
                                           Axis(W.FINE_BITS, -0.5, W.FINE_BITS - 0.5, "fine bit"),
                                           "Fine bits set, by channel")

        wdt = c.dt_window_ns
        h["dt"] = {}
        for k, ch in enumerate(cfg.roles.counters):
            if ch == cfg.roles.s1:
                continue
            h["dt"][ch] = self._h1(f"dt_S{k + 1}_S1", Axis(2 * wdt + 1, -wdt - 0.5, wdt + 0.5,
                                                            "t - t(S1) (ns)"),
                                   f"S{k + 1} (ch {ch}) minus S1")
        nc = len(cfg.roles.counters)
        h["pattern"] = self._h1("pattern", Axis(1 << nc, -0.5, (1 << nc) - 0.5,
                                                "bit k = counter k"),
                                f"Coincidence pattern per S1 (+-{c.coinc_ns} ns)")
        cax = Axis(nc, -0.5, nc - 0.5, "counter (0 = S1)")
        h["s1_coinc"] = self._h1("s1_coinc", cax,
                                 "S1 hits with the counter in the window (entries = S1 hits)")
        pm = b["partner max"]
        h["s1_partner_hits"] = self._h2("s1_partner_hits", cax,
                                        Axis(pm + 1, -0.5, pm + 0.5, "hits in window"),
                                        "Counter hits within the window of each S1")
        h["s1_spacing_us"] = self._h1("s1_spacing_us", Axis(b["spacing bins"], 0,
                                                            b["spacing max"], "us"),
                                      "Time between consecutive S1 hits")

        h["rf_npulses"] = self._h1("rf_npulses", Axis(10, -0.5, 9.5, "RF pulses"),
                                   f"RF pulses in the {c.rf_gate_ns:g} ns gate, per S1")
        g = int(math.ceil(c.rf_gate_ns))
        phax = Axis(g + 1, -0.5, g + 0.5, "last RF pulse - S1 (ns)")
        h["rf_phase_s1"] = self._h1("rf_phase_s1", phax, "RF phase of S1 (valid gates)")
        pmx = b["period max"]
        h["rf_period"] = self._h1("rf_period", Axis(pmx + 1, -0.5, pmx + 0.5, "ns"),
                                  "RF period (last two pulses of a valid gate)")
        h["rf_phase_vs_s1_tot"] = self._h2("rf_phase_vs_s1_tot", phax,
                                           Axis(128, 0, 256, "S1 ToT code"),
                                           "S1 ToT vs RF phase")
        dax = Axis(b["delayed bins"], c.delayed_lo_ns / 1e3, c.delayed_hi_ns / 1e3,
                   "t - t(S1) (us)")
        h["delayed"] = {ch: self._h1(f"delayed_dt_ch{ch:02d}", dax,
                                     f"Channel {ch} minus S1, all pairs")
                        for ch in cfg.roles.delayed}
        scan = cfg.scan
        h["shift_check"] = self._h1("shift_check",
                                    Axis(scan[-1] - scan[0] + 1, scan[0] - 0.5, scan[-1] + 0.5,
                                         "coarse shift"),
                                    "Consistent S1 words per trial shift (entries = S1 words)")

    # -- per event -------------------------------------------------------------

    def accepts(self, event) -> bool:
        return event.header.event_id in self.event_ids

    @staticmethod
    def _bank_data(event):
        getter = getattr(event, "get_bank", None)
        bank = getter(W.BANK) if getter is not None else None
        if bank is None:
            bank = getattr(event, "banks", {}).get(W.BANK)
        if bank is None or bank.data is None:
            return None
        data = bank.data
        if isinstance(data, list | tuple):
            # The tuple path (use_numpy=False): TID_DWORD values.
            data = np.asarray(data, dtype=np.uint32)
        return data

    def process(self, event, run_number=None) -> bool:
        """Classify and fill one readout frame. False if it is not ours."""
        if not self.accepts(event):
            return False
        data = self._bank_data(event)
        if data is None:
            self.frames_rejected += 1
            return False

        now = self._clock()
        if run_number is not None and run_number != self.run_number:
            # A new run: the gap chain, the serial baseline and the summary
            # window start again. The analyzer clears the histograms.
            if self.run_number is not None:
                self._new_epoch()
            self.run_number = run_number
            self._prev_extent = None
            self._prev_timed = None
            self._last_serial = None

        serial = int(getattr(event.header, "serial_number", 0) or 0)
        offered = 1
        if self._last_serial is not None and serial > self._last_serial + 1:
            # Skipped on purpose (sampling) or lost: sent either way.
            self.missed_by_serial += serial - self._last_serial - 1
            offered = serial - self._last_serial
        # A serial going backwards is a replay loop or a restart, not a loss.
        # Only the next serial continues the gap chain: the gap and the live
        # fraction are between a frame and the one sent just before it, and a
        # sampled stream has others in between. (Two frames without serials,
        # both 0, are taken as consecutive, as a file reader or a test hands
        # them over.)
        consecutive = self._last_serial is not None and (
            serial == self._last_serial + 1 or serial == self._last_serial == 0)
        if self._last_serial is not None and not consecutive and self._prev_extent is not None:
            self.serial_breaks += 1
        if not consecutive:
            self._prev_extent = None
            self._prev_timed = None
        self._last_serial = serial
        self.offered_by_serial += offered

        cfg = self.cfg
        c = cfg.cuts
        n_words = int(getattr(data, "nbytes", len(data))) // 8
        if cfg.max_words is not None and n_words > cfg.max_words:
            # Not decoded: bounded cost per frame. Its neighbour has no
            # predecessor for the gap chain.
            self.frames_oversize += 1
            self.last_frame_at = now
            sec = self._roll(now)
            sec.offered += offered
            sec.oversize += 1
            self._prev_extent = None
            self._prev_timed = None
            return False
        fr = W.prepare_frame(data, cfg.shift, c.stale_gap_ns, c.latch_margin_ns,
                             rescue_shifts=cfg.scan, rescue_min_words=c.rescue_min_words,
                             rescue_min_fraction=c.rescue_min_fraction)
        n_s1, scan_counts = kept_s1_scan(fr, cfg)
        s1_all = fr.ch == cfg.roles.s1
        n_s1_all = int(np.count_nonzero(s1_all))
        scan_all = W.shift_scan(fr.coarse[s1_all], fr.fine[s1_all], cfg.scan, c.latch_margin_ns)
        cls, reason, best = classify_frame(fr, cfg, n_s1, scan_counts, n_s1_all, scan_all)

        self.frames += 1
        self.last_frame_at = now
        sec = self._roll(now)
        sec.frames += 1
        sec.offered += offered
        h = self.h

        # -- health: every frame --
        n_stale_hits = int(fr.n_trigger - fr.s_ch.size)
        if cls == "stale":
            types = [fr.n_filler, fr.n_pixel, 0, 0, fr.n_trigger, 0]
            stale_ch = np.bincount(fr.ch.astype(np.intp), minlength=NCH)[:NCH]
        elif cls == "suspect":
            types = [fr.n_filler, fr.n_pixel, 0, 0, 0, fr.n_trigger]
            stale_ch = np.zeros(NCH, dtype=np.int64)
        else:
            types = [fr.n_filler, fr.n_pixel, int(fr.s_ch.size), n_stale_hits, 0, 0]
            stale_ch = fr.stale_per_ch
        h["word_types"].add_counts(np.array([0, *types, 0], dtype=np.int64),
                                   entries=fr.n_words)
        cls_i = CLASSES.index(cls)
        _fill_index(h["frame_class"], cls_i + 1)
        _fill_value(h["words_per_frame"], fr.n_words)
        _fill_value(h["trigger_words_per_frame"], fr.n_trigger)
        if n_s1 >= cfg.stale_min_s1:
            _fill_value(h["s1_best_frac"], best)
        h["stale_per_ch"].add_counts(_with_flow(stale_ch))
        sec.stale_words += stale_ch

        an = None
        gap = None
        if cls in ("good", "suspect"):
            # The shift check reads every S1 word's fine and coarse fields and
            # needs no times, so a suspect frame -- the wrong-shift case --
            # feeds it too. Stale frames (old data) do not.
            self._fill_shift(scan_all, n_s1_all, sec)
        if cls == "stale":
            self.frames_stale += 1
            sec.stale += 1
        elif cls == "empty":
            self.frames_empty += 1
            sec.empty += 1
        elif cls == "suspect":
            self.frames_suspect += 1
            sec.suspect += 1
        else:
            gap = self._fill_good(fr, sec, now)
            # The all-S1 shift scan was done above to classify the frame.
            an = W.analyse_frame(fr, cfg.roles, cfg.cuts, shift_counts=scan_all)
            self._fill_analysis(fr, an, sec)

        hdr = event.header
        snap = _Snapshot(seq=self.frames, run=int(self.run_number or 0), serial=serial,
                         fr=fr, an=an, cls=cls, reason=reason, gap_ns=gap, shift=cfg.shift,
                         tot_min=cfg.cuts.tot_corrupt_min, pre_ns=cfg.cuts.seed_pre_ns,
                         post_ns=cfg.cuts.seed_post_ns,
                         event_id=int(getattr(hdr, "event_id", W.EVID_READOUT)),
                         timestamp=int(getattr(hdr, "timestamp", 0) or 0),
                         trigger_mask=int(getattr(hdr, "trigger_mask", 0) or 0))
        self._keep_raw(snap, event, data)
        self._last = snap
        if cls == "good":
            self._last_good = snap
            if an.seeds.size:
                self._last_seeded = snap
        live = {self._last.seq, self._seeded_snap().seq if self._seeded_snap() else None}
        for key in [k for k in self._frame_cache if k[0] not in live]:
            del self._frame_cache[key]
        return True

    def reset_serial_baseline(self) -> None:
        """After a MIDAS reconnect: the serial gap is outage, not frames offered."""
        self._last_serial = None
        self._prev_extent = None
        self._prev_timed = None

    # -- raw events ------------------------------------------------------------

    def _keep_raw(self, snap: _Snapshot, event, data) -> None:
        """Keep this analysed frame's event bytes, dropping the oldest to stay in bounds.

        Only analysed frames (oversize ones never get here). A frame larger
        than the whole ring is not kept, and counted.
        """
        limit = self.cfg.raw_ring_bytes
        if limit <= 0:
            if self._raw:
                self._raw.clear()
                self._raw_bytes = 0
            return
        try:
            raw = event_bytes(event, data)
        except Exception:                               # noqa: BLE001
            self.raw_not_kept += 1
            return
        if len(raw) > limit:
            self.raw_not_kept += 1
            return
        self._raw[snap.seq] = _RawEntry(snap.seq, snap.run, snap.serial, snap.timestamp,
                                        snap.event_id, raw)
        self._raw_bytes += len(raw)
        while self._raw_bytes > limit and self._raw:
            _k, old = self._raw.popitem(last=False)
            self._raw_bytes -= len(old.data)

    def raw_event(self, seq: int) -> _RawEntry | None:
        return self._raw.get(int(seq))

    def _snapshot_for(self, seq: int) -> _Snapshot | None:
        """A held frame by its seq: a live snapshot, or rebuilt from the raw ring.

        Rebuilt with the current settings, without touching any histogram;
        its gap to the previous frame is unknown (None).
        """
        for snap in (self._last, self._last_good, self._last_seeded):
            if snap is not None and snap.seq == seq:
                return snap
        entry = self._raw.get(int(seq))
        if entry is None:
            return None
        from mdqm.tools import midasfile as MF

        eid, tmask, serial, ts, dsz = MF.EVENT_HEADER.unpack_from(entry.data, 0)
        ev = MF.Event(MF.EventHeader(eid, tmask, serial, ts, dsz))
        MF.parse_banks(ev, entry.data[MF.EVENT_HEADER.size:MF.EVENT_HEADER.size + dsz])
        bank = ev.get_bank(W.BANK)
        if bank is None:
            return None
        cfg, c = self.cfg, self.cfg.cuts
        fr = W.prepare_frame(bank.data, cfg.shift, c.stale_gap_ns, c.latch_margin_ns,
                             rescue_shifts=cfg.scan, rescue_min_words=c.rescue_min_words,
                             rescue_min_fraction=c.rescue_min_fraction)
        n_s1, scan_counts = kept_s1_scan(fr, cfg)
        s1_all = fr.ch == cfg.roles.s1
        scan_all = W.shift_scan(fr.coarse[s1_all], fr.fine[s1_all], cfg.scan, c.latch_margin_ns)
        cls, reason, _best = classify_frame(fr, cfg, n_s1, scan_counts,
                                            int(np.count_nonzero(s1_all)), scan_all)
        an = (W.analyse_frame(fr, cfg.roles, cfg.cuts, shift_counts=scan_all)
              if cls == "good" else None)
        return _Snapshot(seq=entry.seq, run=entry.run, serial=entry.serial, fr=fr, an=an,
                         cls=cls, reason=reason, gap_ns=None, shift=cfg.shift,
                         tot_min=c.tot_corrupt_min, pre_ns=c.seed_pre_ns, post_ns=c.seed_post_ns,
                         event_id=eid, timestamp=ts, trigger_mask=tmask)

    def _seeded_snap(self):
        """The last frame that had seeds, else the last good one.

        A frame without seeds (one sparse counter used to be enough) would
        blank the seeded display; the last one worth showing stays up instead,
        and its sequence number tells the page how old it is.
        """
        return self._last_seeded or self._last_good

    def _fill_shift(self, counts, n_s1: int, sec: _Second) -> None:
        cfg = self.cfg
        sec.shift_n += n_s1
        if len(sec.scan) == len(counts):
            sec.shift_counts += counts
        hs = self.h["shift_check"]
        sc = np.zeros(hs.counts.shape, dtype=np.int64)
        sc[np.asarray(cfg.scan) - cfg.scan[0] + 1] = counts
        hs.add_counts(sc, entries=n_s1)

    def _fill_good(self, fr: W.Frame, sec: _Second, now: float) -> int | None:
        """Health histograms of a good frame; returns its gap to the previous one."""
        h, cfg = self.h, self.cfg
        s_ch = fr.s_ch.astype(np.intp)
        hits = np.bincount(s_ch, minlength=NCH)[:NCH]
        h["words_per_ch"].add_counts(_with_flow(hits))
        span = fr.span_ns
        if span > 0:
            _fill_value(h["frame_span_log10_ms"], math.log10(span * 1e-6))
        else:
            _fill_index(h["frame_span_log10_ms"], 0)
        sec.span_ns += span
        sec.hits += hits
        # The rates divide by the time the frame covers, taken from its
        # fine/coarse-consistent hits (Frame.timed_extent): a few faulty words
        # sit up to ~1 ms outside the frame. With every frame analysed that
        # hardly matters, since consecutive frames' overlap is taken out; a
        # sampled frame has no neighbour to take it out, and its all-hit span
        # made the rates 4 % low at the 1008 rate and 25 % low at ten times it.
        tspan = fr.timed_span_ns
        if tspan > 0:
            nz = np.flatnonzero(hits)
            rate = np.log10(hits[nz] / (tspan * 1e-9))
            _fill_2d(h["rate_vs_ch"], nz.astype(np.float64), rate)

        gap = None
        ext = fr.extent
        timed = fr.timed_extent
        cover = tspan
        # Rates use only frames whose predecessor (the previous serial) was
        # analysed too: the second frame of each sampled pair, and every frame
        # but the first of a chain when all are analysed. After a skip the
        # first frame read is the one being written at that moment, which
        # favours long frames (length-biased sampling: long frames span more
        # time, so a skip lands in them more often); around beam trips that
        # reads the rate low by ~1/(1 + CV^2) of the frame length. The frame
        # after it was chosen by nothing but being next. Below shift 12 there
        # is no chain (see below) and every frame is used.
        chained = fr.time_bits < W.TIME_BITS
        if fr.time_bits < W.TIME_BITS:
            # Below shift 12 the time wraps every 2^(28 + shift) ns (2.1 s at
            # shift 3), shorter than the spacing of slow frames: the gap to the
            # previous frame is only known modulo that, so no gap, no live.
            self._prev_extent = None
            self._prev_timed = None
        elif self._prev_extent is not None:
            # Gap and live fraction: the kept extents (PISMAWord / PITMidasMusip).
            gap = W.gap_ns(self._prev_extent, ext, fr.time_bits)
            if -cfg.max_overlap_ns <= gap <= cfg.max_gap_ns:
                _fill_value(h["frame_gap_ms"], gap * 1e-6)
                delta = W.gap_ns(self._prev_extent, W.Extent(ext.last, ext.last), fr.time_bits)
                if delta > 0:
                    live = min(span, delta)
                    _fill_value(h["live_fraction"], live / delta)
                    sec.delta_ns += delta
                    sec.live_ns += live
                # Consecutive frames can still overlap in their timed extents
                # too; that time is counted once, or the rates come out low.
                tgap = W.gap_ns(self._prev_timed, timed, fr.time_bits)
                cover = tspan - min(tspan, max(0, -tgap))
                chained = True
            else:
                self.gap_resets += 1
        self._prev_extent = ext
        self._prev_timed = timed
        if chained:
            sec.cover_ns += cover
            sec.rate_hits += hits

        # ToT, by channel and fine LSB, in one bincount.
        o = fr.order
        fine = fr.fine[o]
        lsb = (fine & 1).astype(np.intp)
        tot = fr.s_tot.astype(np.intp)
        ny, nx = 258, NCH + 2
        flat = lsb * (ny * nx) + (tot + 1) * nx + (s_ch + 1)
        both = np.bincount(flat, minlength=2 * ny * nx).reshape(2, ny, nx)
        for k in (0, 1):
            h["tot_vs_ch"][k].add_counts(both[k])
            for ch, pair in h["tot_ch"].items():
                pair[k].add_counts(both[k][:, ch + 1])
        bad_tot = np.bincount(s_ch[tot >= cfg.cuts.tot_corrupt_min], minlength=NCH)[:NCH]
        h["tot_ge250_per_ch"].add_counts(_with_flow(bad_tot))
        sec.tot_bad += bad_tot

        # Fine vs coarse.
        diff = fr.diff_ns[o] >> fr.shift          # exact: a multiple of 2^shift
        hd = h["fine_coarse_diff"]
        # Integer bins: the y axis has one bin per tick centred on the integers
        # from -half, so bin = diff + half (+1 for the underflow), the same
        # counts as the float path of _fill_2d without its round trip.
        _fill_2d_index(hd, s_ch + 1, diff + (self._fc_half + 1))
        cons = fr.consistent[o]
        bad = ~cons
        n_bad = np.bincount(s_ch[bad], minlength=NCH)[:NCH]
        sec.mismatch += n_bad
        fvc = np.zeros(h["fine_vs_coarse"].counts.shape, dtype=np.int64)
        fvc[1, 1:NCH + 1] = hits - n_bad
        fvc[2, 1:NCH + 1] = n_bad
        if n_bad.any():
            xor = W.fine_coarse_xor(fr.coarse[o][bad], fine[bad], fr.shift)
            nb = W.shared_bits(fr.shift)
            bits = W.per_channel_bit_counts(s_ch[bad], xor, nb)
            # xor bit i is fine bit shift + i -> row 2 + shift + i (+1 for underflow).
            fvc[3 + fr.shift: 3 + fr.shift + nb, 1:NCH + 1] = bits.T
        h["fine_vs_coarse"].add_counts(fvc, entries=int(hits.sum()))
        occ = np.zeros(h["fine_bit_occupancy"].counts.shape, dtype=np.int64)
        occ[1:W.FINE_BITS + 1, 1:NCH + 1] = W.per_channel_bit_counts(s_ch, fine).T
        h["fine_bit_occupancy"].add_counts(occ, entries=int(hits.sum()))
        return gap

    def _fill_analysis(self, fr: W.Frame, an: W.FrameAnalysis, sec: _Second) -> None:
        h = self.h
        # The S1 spacing needs every kept S1 hit; the rest is filled from the
        # S1 sample when the frame has more than Cuts/max S1 per frame (the
        # fractions it gives are unbiased, see sma_words.analyse_frame).
        t_all = an.t_s1 if an.t_s1_all is None else an.t_s1_all
        sec.n_s1_kept += int(t_all.size)
        an = an.sampled()
        n1 = int(an.t_s1.size)
        sec.n_s1 += n1
        if n1 == 0:
            return

        for ch, (_i, dt) in an.dt.items():
            hh = h["dt"].get(ch)
            if hh is not None and dt.size:
                _fill_values(hh, dt)
        _fill_values(h["pattern"], an.pattern)
        nc = an.partner_counts.shape[1]
        coinc = np.count_nonzero(an.partner_counts > 0, axis=0)
        sec.eff += coinc
        cc = np.zeros(h["s1_coinc"].counts.shape, dtype=np.int64)
        cc[1:nc + 1] = coinc
        h["s1_coinc"].add_counts(cc, entries=n1)
        # Integer bins on both axes (counter index, partner count).
        k = np.broadcast_to(np.arange(1, nc + 1, dtype=np.intp), an.partner_counts.shape)
        _fill_2d_index(h["s1_partner_hits"], k.ravel(),
                       an.partner_counts.ravel().astype(np.intp) + 1)
        if t_all.size > 1:
            _fill_values(h["s1_spacing_us"], np.diff(t_all) * 1e-3)

        _fill_values(h["rf_npulses"], an.rf_n)
        v = an.rf_valid
        nv = int(np.count_nonzero(v))
        sec.rf_valid += nv
        sec.rf_vetoed += int(np.count_nonzero(an.rf_vetoed))
        if nv:
            ph = an.rf_phase[v]
            _fill_values(h["rf_phase_s1"], ph)
            _fill_values(h["rf_period"], an.rf_period[v])
            _fill_2d(h["rf_phase_vs_s1_tot"], ph, an.s1_tot[v].astype(np.float64))
        for ch, (_i, dt) in an.delayed_dt.items():
            hh = h["delayed"].get(ch)
            if hh is not None and dt.size:
                _fill_values(hh, dt * 1e-3)

    # -- seconds ---------------------------------------------------------------

    def _roll(self, now: float | None = None, force: bool = False) -> _Second:
        """The bucket for `now`; closes finished seconds (and empty ones in between)."""
        now = self._clock() if now is None else now
        t = int(now)
        cur = self._cur
        nc = len(self.cfg.roles.counters)
        if cur is not None and cur.t == t and not force and cur.epoch == self.epoch:
            return cur
        if cur is not None:
            if force and cur.t == t:
                # A new epoch inside a second: close this one early and start a
                # fresh one for the same second, so no bucket mixes epochs.
                self._seconds.append(cur)
            else:
                self._seconds.append(cur)
                # Seconds with no frames are rows too: a trend that skips them
                # would draw straight across an outage.
                for s in range(max(cur.t + 1, t - TREND_S), t):
                    self._seconds.append(_Second(s, self.epoch, nc, self.cfg.scan))
        self._cur = _Second(t, self.epoch, nc, self.cfg.scan)
        return self._cur

    def _window(self, seconds: float, now: float, epoch_only=True) -> list[_Second]:
        lo = now - seconds
        out = [s for s in self._seconds if s.t >= lo and (not epoch_only or s.epoch == self.epoch)]
        cur = self._cur
        if cur is not None and cur.t >= lo and (not epoch_only or cur.epoch == self.epoch):
            out.append(cur)
        return out

    # -- self check -------------------------------------------------------------

    def shift_verdict(self, now: float | None = None) -> dict:
        """Which scanned shift makes the most S1 words consistent over the window."""
        now = self._clock() if now is None else now
        chk, cfg = self.cfg.check, self.cfg
        scan = cfg.scan
        # The ring spans rebuilds (a shift edit must be judged at once, not 30 s
        # later) but only while the scanned shifts are the same.
        secs = [s for s in self._window(chk["shift window s"], now, epoch_only=False)
                if s.scan == scan]
        counts = sum((s.shift_counts for s in secs), np.zeros(len(scan), dtype=np.int64))
        n = int(sum(s.shift_n for s in secs))
        fr = counts / n if n else np.full(len(scan), np.nan)
        conf_i = scan.index(cfg.shift)
        out = {"configured": cfg.shift, "scan": list(scan),
               "fractions": [_num(x, 4) for x in fr], "words": n,
               "window_s": chk["shift window s"], "best": None, "verdict": "insufficient"}
        if n < chk["shift min words"]:
            return out
        best_i = int(np.argmax(fr))
        out["best"] = scan[best_i]
        if (best_i != conf_i and fr[best_i] - fr[conf_i] > chk["shift margin"]
                and fr[best_i] >= chk["shift min fraction"]):
            out["verdict"] = "mismatch"
        elif fr[conf_i] >= chk["shift min fraction"]:
            out["verdict"] = "ok"
        else:
            out["verdict"] = "no fit"
        return out

    # -- labels ------------------------------------------------------------------

    def labels(self) -> list[str]:
        r = self.cfg.roles
        names = {}
        for c in r.delayed:
            names[c] = f"ch{c:02d}"
        names[r.current] = "current"
        names[r.rf] = "RF"
        for k, c in enumerate(r.counters):
            names[c] = f"S{k + 1}"
        return [self.cfg.labels[c] or names.get(c, f"ch{c:02d}") for c in range(NCH)]

    def _role_of(self, c: int) -> str:
        r = self.cfg.roles
        if c in r.counters:
            return "s1" if c == r.s1 else "counter"
        if c == r.rf:
            return "rf"
        if c == r.current:
            return "current"
        if c in r.delayed:
            return "delayed"
        return ""

    # -- summary -----------------------------------------------------------------

    def summary(self, run_active: bool | None = None) -> dict:
        now = self._clock()
        self._roll(now)
        chk, cfg = self.cfg.check, self.cfg
        # The analyzer's polled run state wins; the page's value is the fallback
        # for a plugin run without one (tests, the offline CLI).
        run_active = self.run_active if self.run_active is not None else run_active
        secs = self._window(chk["summary window s"], now)

        def total(attr):
            return sum(getattr(s, attr) for s in secs)

        def vec(attr, n=NCH):
            return sum((getattr(s, attr) for s in secs if len(getattr(s, attr)) == n),
                       np.zeros(n, dtype=np.int64))

        frames, stale, empty = total("frames"), total("stale"), total("empty")
        offered = total("offered")
        suspect = total("suspect")
        good = frames - stale - empty - suspect
        span_s = total("span_ns") * 1e-9
        cover_s = total("cover_ns") * 1e-9
        hits, mism, totb, stw = vec("hits"), vec("mismatch"), vec("tot_bad"), vec("stale_words")
        rate_hits = vec("rate_hits")
        n_s1 = total("n_s1")
        nc = len(cfg.roles.counters)
        eff = vec("eff", nc)
        labels = self.labels()

        channels = [{
            "ch": c, "label": labels[c], "role": self._role_of(c),
            "hits": int(hits[c]),
            "rate_hz": _ratio(rate_hits[c], cover_s, 5),
            "hits_per_frame": _ratio(hits[c], good, 5),
            "tot_ge250_frac": _ratio(totb[c], hits[c], 4),
            "mismatch_frac": _ratio(mism[c], hits[c], 4),
            "stale": int(stw[c]),
            "flagged": c in cfg.flag_channels,
        } for c in range(NCH)]
        efficiency = []
        for k, c in enumerate(cfg.roles.counters):
            if c == cfg.roles.s1:
                continue
            m = channels[c]["mismatch_frac"]
            reason = None
            e = _ratio(eff[k], n_s1, 4)
            # A timed efficiency needs the counter's timestamps: with a fine
            # fault the coincidence misses by microseconds and the number
            # would read as a dead counter. Null, with the reason, instead.
            if m is not None and m > chk["mismatch warn fraction"]:
                e, reason = None, f"timestamp fault: {m:.1%} mismatch"
            efficiency.append({"counter": f"S{k + 1}", "ch": c, "label": labels[c],
                               "eff": e, "reason": reason})
        shift = self.shift_verdict(now)
        age = None if self.last_frame_at is None else now - self.last_frame_at
        n_s1_kept = total("n_s1_kept")
        st = self.sampling_state or {}
        sampling = {
            # Frames: analysed / sent (from the serial numbers), in the window.
            "analysed_frac": _ratio(frames, offered, 4),
            "offered_per_s": _ratio(offered, max(1.0, len(secs)), 4),
            # S1 hits given the S1-seeded analyses / kept S1 hits (Cuts/max S1
            # per frame), over the analysed frames.
            "s1_analysed_frac": _ratio(n_s1, n_s1_kept, 4),
            "max_s1_per_frame": cfg.cuts.max_s1,
            "mode": st.get("mode"),
            "cpu_budget_pct": _num(st.get("cpu_budget_pct"), 4),
            "cpu_pct": _num(st.get("cpu_pct"), 4),
            "rate_limit": _num(st.get("rate_limit"), 4),
        }

        out = {
            "t": now, "run": self.run_number, "run_active": run_active,
            "window_s": chk["summary window s"], "epoch": self.epoch,
            "frames": {
                "processed": self.frames, "stale": self.frames_stale,
                "empty": self.frames_empty, "suspect": self.frames_suspect,
                "rejected": self.frames_rejected,
                "oversize": self.frames_oversize,
                "missed_by_serial": self.missed_by_serial,
                "seen_by_serial": self.frames + self.missed_by_serial,
                #: Frames the DAQ sent (serial numbers), and the analysed share.
                "offered": self.offered_by_serial,
                "analysed_frac": _ratio(self.frames, self.offered_by_serial, 4),
                "gap_resets": self.gap_resets,
                "serial_breaks": self.serial_breaks,
                "last_age_s": _num(age, 4),
                "window": {"frames": frames, "good": good, "stale": stale, "empty": empty,
                           "suspect": suspect, "offered": offered,
                           "oversize": total("oversize"),
                           "analysed_frac": sampling["analysed_frac"],
                           "per_s": _ratio(frames, max(1.0, len(secs)), 4)},
            },
            "sampling": sampling,
            "live_fraction": _ratio(total("live_ns"), total("delta_ns"), 4),
            "span_s": _num(span_s, 5),
            "covered_s": _num(cover_s, 5),
            "channels": channels,
            #: S1-conditional, by timestamps: a counter hit within the
            #: coincidence window of an S1 hit.
            "efficiency_kind": "timed",
            "efficiency": efficiency,
            "rf": {"n_s1": n_s1, "valid_frac": _ratio(total("rf_valid"), n_s1, 4),
                   "vetoed_frac": _ratio(total("rf_vetoed"), n_s1, 4)},
            "shift": shift,
            "settings_errors": list(cfg.errors),
        }
        out["flags"] = self._flags(out, now, run_active)
        return out

    def _flags(self, s: dict, now: float, run_active) -> list[dict]:
        chk = self.cfg.check
        flags = []

        def add(sev, code, text):
            flags.append({"severity": sev, "code": code, "text": text})

        f = s["frames"]
        w = f["window"]
        sh = s["shift"]
        # Everything per channel below reads fine against coarse at the
        # configured shift. Until the shift check says that shift is right,
        # those numbers describe the setting, not the board: one flag about
        # the time base instead of a mismatch flag on every channel.
        time_base_ok = sh["verdict"] == "ok"
        if sh["verdict"] == "mismatch":
            fr = dict(zip(sh["scan"], sh["fractions"], strict=True))
            extra = (f"; {w['suspect']} frame(s) had no usable time base" if w["suspect"]
                     else "")
            add("error", "shift_mismatch",
                f"Coarse shift {sh['configured']} is configured but {sh['best']} fits better "
                f"({fr[sh['best']]:.0%} vs {fr[sh['configured']] or 0:.0%} of {sh['words']} "
                f"S1 words): set /DQM/SMA/Coarse shift = {sh['best']}{extra}. Per-channel "
                "fine/coarse flags are suppressed until then")
        elif w["suspect"]:
            add("error", "time_base",
                f"{w['suspect']} of {w['frames']} frames kept less than "
                f"{self.cfg.suspect_kept_frac:.0%} of their hits in one time cluster: the time "
                "base is suspect, most likely the coarse shift (shift check: "
                f"{sh['verdict']})")
        elif sh["verdict"] == "no fit":
            add("warn", "shift_no_fit",
                f"No scanned shift makes S1 fine/coarse consistent "
                f"(best {max(x or 0 for x in sh['fractions']):.0%}); per-channel fine/coarse "
                "flags are suppressed")
        elif sh["verdict"] == "insufficient" and w["frames"]:
            add("info", "shift_unchecked",
                f"Shift check has {sh['words']} S1 words (needs "
                f"{chk['shift min words']:.0f}); per-channel fine/coarse flags wait for it")

        age = f["last_age_s"]
        if age is not None and age > chk["no frames s"] and run_active:
            add("error", "no_frames", f"No SMA frames for {age:.0f} s while a run is active")
        elif age is None and run_active:
            add("warn", "no_frames", "No SMA frame received yet while a run is active")
        if w.get("oversize"):
            add("warn", "oversize",
                f"{w['oversize']} frame(s) in the last {s['window_s']:.0f} s had more than "
                f"{self.cfg.max_words} words (Cuts/max words per frame) and were not decoded")
        if w["frames"] and w["stale"] == w["frames"]:
            add("error", "all_stale",
                f"All {w['frames']} frames in the last {s['window_s']:.0f} s are stale: the "
                "board is sending old data, or S1 fine/coarse is broken at every shift "
                "(see s1_best_frac and the raster)")
        elif w["stale"]:
            add("warn", "stale_frames",
                f"{w['stale']} of {w['frames']} frames in the last {s['window_s']:.0f} s were "
                "stale (not this run's data) and left out")

        min_hits = chk["min hits"]
        for c in s["channels"]:
            if c["hits"] < min_hits or not c["flagged"]:
                continue
            m = c["mismatch_frac"] or 0.0
            if not time_base_ok:
                pass
            elif m > chk["mismatch error fraction"]:
                add("error", "mismatch",
                    f"{c['label']} (ch {c['ch']}): {m:.0%} of hits fine/coarse inconsistent "
                    "(a fine-bit fault such as fine = t/2)")
            elif m > chk["mismatch warn fraction"]:
                add("warn", "mismatch",
                    f"{c['label']} (ch {c['ch']}): {m:.1%} of hits fine/coarse inconsistent")
            t = c["tot_ge250_frac"] or 0.0
            if t > chk["tot corrupt warn fraction"]:
                add("warn", "tot_corrupt",
                    f"{c['label']} (ch {c['ch']}): {t:.1%} of hits with ToT >= "
                    f"{self.cfg.cuts.tot_corrupt_min}")

        if time_base_ok:
            faulty = {e["ch"] for e in s["efficiency"] if e["eff"] is None}
            flags.extend(self._efficiency_flags(now, faulty))
        smp = s["sampling"]
        af = smp["analysed_frac"]
        if af is not None and af < 0.999:
            # Information, not a warning: sampling is how the analyzer keeps
            # within its CPU budget, and it biases none of the fractions.
            why = (f" to stay within its CPU budget of {smp['cpu_budget_pct']:g} % of a core"
                   if smp.get("cpu_budget_pct") is not None and smp.get("mode") == "cpu budget"
                   else "")
            add("info", "sampling",
                f"Analysing {af:.0%} of the frames{why}. Histogram counts are from the "
                "analysed sample; rates, fractions and efficiencies are not affected")
        if s["settings_errors"]:
            add("warn", "settings", "; ".join(s["settings_errors"]))
        return flags

    def _efficiency_flags(self, now: float, skip=frozenset()) -> list[dict]:
        chk, cfg = self.cfg.check, self.cfg
        nc = len(cfg.roles.counters)
        recent = self._window(chk["efficiency window s"], now)
        base = self._window(TREND_S, now)

        def eff(secs):
            n = sum(s.n_s1 for s in secs)
            e = sum((s.eff for s in secs if s.eff.size == nc), np.zeros(nc, dtype=np.int64))
            return n, e

        n_r, e_r = eff(recent)
        n_b, e_b = eff(base)
        if n_r < chk["min hits"] or n_b - n_r < chk["min hits"]:
            return []
        # The baseline without the recent part, so a drop is not diluted by itself.
        n_o, e_o = n_b - n_r, e_b - e_r
        out = []
        labels = self.labels()
        for k, c in enumerate(cfg.roles.counters):
            if c == cfg.roles.s1 or c in skip:
                continue
            r, b = e_r[k] / n_r, e_o[k] / n_o
            if b - r > chk["efficiency drop"]:
                out.append({"severity": "warn", "code": "efficiency_drop",
                            "text": f"{labels[c]} (ch {c}) timed efficiency given S1 fell "
                                    f"from {b:.0%} to {r:.0%} in the last "
                                    f"{chk['efficiency window s']:.0f} s"})
        return out

    # -- trend -------------------------------------------------------------------

    def trend(self, since: float | None = None) -> dict:
        """Completed 1 s rows of the last 10 min (only those after `since`)."""
        now = self._clock()
        self._roll(now)
        # Two buckets share a second when an epoch starts inside it; a row is
        # a second, so they are merged.
        by_t: dict[int, list] = {}
        for s in self._seconds:
            if s.t >= now - TREND_S and (since is None or s.t > since):
                by_t.setdefault(s.t, []).append(s)
        warn = self.cfg.check["mismatch warn fraction"]
        counters = self.cfg.roles.counters
        rows = [(v[0] if len(v) == 1 else _merge_seconds(v)).row(counters, warn)
                for _t, v in sorted(by_t.items())]
        return {"t": now, "labels": self.labels(),
                # The columns of each row's "eff": the counters after S1.
                "counters": [f"S{k + 1}" for k in range(1, len(self.cfg.roles.counters))],
                "rows": rows}

    # -- event display -------------------------------------------------------------

    def frame_blob(self, view: str = "seeded", drop=(), max_hits: int | None = None,
                   words: bool | None = None, seq: int | None = None) -> bytes | None:
        """A frame as an ``smaf`` payload, encoded once per (frame, view, drop, words).

        The seeded view shows the last frame that had seeds (`_seeded_snap`); the
        raster shows the last frame of any class, flagged if stale or suspect.
        With `seq`, that frame instead, if still held (a live snapshot or the
        raw ring), else None. `words`: ship each hit's raw word and bank index
        (smaf v2); default yes for seeded, no for the raster (see framing).
        """
        if view not in ("seeded", "raster"):
            raise ValueError(f"view must be 'seeded' or 'raster', got {view!r}")
        if seq is not None:
            snap = self._snapshot_for(int(seq))
        else:
            snap = self._seeded_snap() if view == "seeded" else self._last
        if snap is None:
            return None
        words = (view == "seeded") if words is None else bool(words)
        drop = tuple(sorted({int(c) for c in drop}))
        max_hits = None if max_hits is None else max(0, int(max_hits))
        key = (snap.seq, view, drop, max_hits, words)
        blob = self._frame_cache.get(key)
        if blob is None:
            blob = self._encode(snap, view, drop, max_hits, words)
            self._frame_cache[key] = blob
        return blob

    def _encode(self, snap: _Snapshot, view, drop, max_hits, words=False) -> bytes:
        fr, an = snap.fr, snap.an
        o = fr.order
        n = int(o.size)
        s_t = fr.s_t
        in_seed = np.zeros(n, dtype=bool)
        windows = []
        seeds = an.seeds if an is not None else np.zeros(0, dtype=np.intp)
        for i in seeds:
            ts = int(an.t_s1[i])
            a = int(np.searchsorted(s_t, ts - snap.pre_ns, side="left"))
            b = int(np.searchsorted(s_t, ts + snap.post_ns, side="right"))
            in_seed[a:b] = True
            windows.append((a, b))

        sel = in_seed.copy() if view == "seeded" else np.ones(n, dtype=bool)
        if drop:
            sel &= ~np.isin(fr.s_ch, drop)
        idx = np.flatnonzero(sel)
        truncated = max_hits is not None and idx.size > max_hits
        n_total = int(idx.size)
        if truncated:
            # The latest hits, like the seeds: the end of the frame is what
            # the seeded view shows next to it.
            idx = idx[idx.size - max_hits:]

        # Times relative to the first *shipped* hit, in units of 2^time_shift ns.
        # A frame is normally tens of ms (time_shift 0), but a frame with a beam
        # trip spans seconds, beyond a u32 of ns; then the unit grows instead
        # of the times clipping.
        t0 = int(s_t[idx[0]]) if idx.size else int(fr.first)
        span = int(s_t[idx[-1]]) - t0 if idx.size else 0
        k = 0
        while (span >> k) > 0xFFFFFFFF:
            k += 1
        t_rel = ((s_t[idx] - t0) >> k).astype("<u4")
        ch = fr.s_ch[idx].astype(np.uint8)
        tot = fr.s_tot[idx].astype(np.uint8)
        fine = fr.fine[o[idx]]
        flags = ((~fr.consistent[o[idx]]).astype(np.uint8) * framing.HIT_MISMATCH
                 | (fr.s_tot[idx] >= snap.tot_min).astype(np.uint8) * framing.HIT_TOT_CORRUPT
                 | (fine & 1).astype(np.uint8) * framing.HIT_FINE_LSB
                 | in_seed[idx].astype(np.uint8) * framing.HIT_IN_SEED)
        if snap.cls == "stale":
            flags |= framing.HIT_STALE

        seed_meta = []
        widx = fr.word_index[o[idx]] if fr.word_index is not None else None
        for i, (a, b) in zip(seeds, windows, strict=True):
            seed_meta.append({
                "t_rel": int(an.t_s1[i]) - t0,          # ns, never shifted
                "s1_tot": int(an.s1_tot[i]),
                "rf_phase": _num(an.rf_phase[i]), "rf_period": _num(an.rf_period[i]),
                "rf_n": int(an.rf_n[i]), "rf_valid": bool(an.rf_valid[i]),
                "rf_vetoed": bool(an.rf_vetoed[i]), "pattern": int(an.pattern[i]),
                "hits": [int(np.searchsorted(idx, a)), int(np.searchsorted(idx, b))],
                # Bank word indices of every hit in the window (shipped or
                # not): the range, to find the seed's words in the file.
                "word_range": ([int(fr.word_index[o[a:b]].min()), int(fr.word_index[o[a:b]].max())]
                               if fr.word_index is not None and b > a else None),
                "s1_word": (int(fr.word_index[o[fr.chan[self.cfg.roles.s1][
                    int(i if an.s1_rows is None else an.s1_rows[i])]]])
                            if fr.word_index is not None else None),
            })
        per_ch = np.bincount(fr.s_ch.astype(np.intp), minlength=NCH)[:NCH]
        meta = {
            "view": view, "seq": snap.seq, "run": snap.run, "serial": snap.serial,
            "class": snap.cls, "stale": snap.cls == "stale", "suspect": snap.cls == "suspect",
            "stale_reason": snap.reason,
            "n_words": fr.n_words, "n_filler": fr.n_filler, "n_pixel": fr.n_pixel,
            "n_trigger": fr.n_trigger, "n_kept": n, "n_rescued": int(fr.n_rescued),
            "span_ns": int(fr.span_ns), "gap_ns": snap.gap_ns,
            "t0_ns": t0, "time_shift": k, "frame_first_ns": int(fr.first), "shift": snap.shift,
            "per_channel": [int(x) for x in per_ch],
            "stale_per_channel": [int(x) for x in fr.stale_per_ch],
            "labels": self.labels(),
            "window": {"pre_ns": snap.pre_ns, "post_ns": snap.post_ns},
            "seeds": seed_meta,
            "dropped": list(drop), "n_selected": n_total, "truncated": bool(truncated),
            "tot_corrupt": snap.tot_min,
            "event": {"id": snap.event_id, "serial": snap.serial, "timestamp": snap.timestamp,
                      "trigger_mask": snap.trigger_mask},
            "tag": frame_tag(snap.run, snap.event_id, snap.serial, snap.timestamp, snap.seq),
            # Whether sma::raw can still hand this frame's bytes over.
            "raw_held": snap.seq in self._raw,
            "words": bool(words and widx is not None),
        }
        return framing.encode_sma_frame(
            meta, t_rel, ch, tot, flags, frame_seq=snap.seq, run_number=snap.run,
            seeded=view == "seeded", stale=snap.cls == "stale", truncated=truncated,
            suspect=snap.cls == "suspect", time_shift=k,
            raw_words=fr.raw[o[idx]] if words and fr.raw is not None else None,
            word_index=widx if words and fr.raw is not None else None)

    # -- commands ----------------------------------------------------------------

    def commands(self) -> dict:
        """``sma::summary`` / ``sma::trend`` / ``sma::frame``, framed for brpc.

        Args are JSON (empty means defaults):

        * summary: ``{"run_active": bool}`` -- the page knows the run state;
        * trend: ``{"since": unix_s}`` -- only rows after it, for incremental polls;
        * frame: ``{"view": "seeded"|"raster", "drop": [ch...], "max_hits": N}``;
          no frame yet gives ``json {"no_frame": true}``.
        """
        return {"sma::summary": self._cmd_summary, "sma::trend": self._cmd_trend,
                "sma::frame": self._cmd_frame, "sma::raw": self._cmd_raw}

    @staticmethod
    def _args(args: str) -> dict:
        args = (args or "").strip()
        if not args:
            return {}
        out = json.loads(args)
        if not isinstance(out, dict):
            raise ValueError("args must be a JSON object")
        return out

    @staticmethod
    def _json(obj) -> bytes:
        return framing.envelope(framing.TAG_JSON,
                                json.dumps(obj, allow_nan=False, default=_json_default).encode())

    def _cmd_summary(self, args: str) -> bytes:
        a = self._args(args)
        ra = a.get("run_active")
        return self._json(self.summary(None if ra is None else bool(ra)))

    def _cmd_trend(self, args: str) -> bytes:
        a = self._args(args)
        since = a.get("since")
        return self._json(self.trend(None if since is None else float(since)))

    def _cmd_frame(self, args: str) -> bytes:
        a = self._args(args)
        seq = a.get("seq")
        blob = self.frame_blob(str(a.get("view", "seeded")), a.get("drop") or (),
                               a.get("max_hits"), a.get("words"),
                               None if seq is None else int(seq))
        if blob is None:
            if seq is not None:
                return framing.envelope(framing.TAG_ERROR, (f"frame {int(seq)} no longer held — use the tag").encode())
            return framing.envelope(framing.TAG_JSON, b'{"no_frame": true}')
        return framing.envelope(framing.TAG_SMAF, blob)

    def _cmd_raw(self, args: str) -> bytes:
        """``{"seq": N}``: the raw MIDAS event of frame N (tag ``mevt``), a one-event .mid."""
        a = self._args(args)
        if "seq" not in a:
            return framing.envelope(framing.TAG_ERROR, ('sma::raw needs {"seq": N}').encode())
        entry = self.raw_event(int(a["seq"]))
        if entry is None:
            return framing.envelope(framing.TAG_ERROR, (f"frame {int(a['seq'])} no longer held — use the tag").encode())
        return framing.envelope(framing.TAG_MEVT, entry.data)

    # -- reporting ---------------------------------------------------------------

    def status(self) -> dict:
        last = self._last
        return {
            "plugin": self.name,
            "decoded": self.frames,
            "frames_stale": self.frames_stale,
            "frames_empty": self.frames_empty,
            "frames_suspect": self.frames_suspect,
            "frames_rejected": self.frames_rejected,
            "frames_oversize": self.frames_oversize,
            "missed_by_serial": self.missed_by_serial,
            "offered_by_serial": self.offered_by_serial,
            "analysed_frac": _ratio(self.frames, self.offered_by_serial, 4),
            "gap_resets": self.gap_resets,
            "serial_breaks": self.serial_breaks,
            "max_s1_per_frame": self.cfg.cuts.max_s1,
            "raw_ring": {"frames": len(self._raw), "bytes": self._raw_bytes,
                         "limit_bytes": self.cfg.raw_ring_bytes,
                         "oldest_seq": next(iter(self._raw), None), "not_kept": self.raw_not_kept},
            "coarse_shift": self.cfg.shift,
            "rebuilds": self.rebuilds,
            "settings_errors": list(self.cfg.errors),
            "last_frame": None if last is None else {
                "seq": last.seq, "serial": last.serial, "class": last.cls,
                "reason": last.reason, "n_words": last.fr.n_words,
                "n_trigger": last.fr.n_trigger, "span_ns": int(last.fr.span_ns),
            },
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return _num(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _merge_seconds(secs: list[_Second]) -> _Second:
    out = _Second(secs[0].t, secs[-1].epoch, secs[-1].eff.size, secs[-1].scan)
    for s in secs:
        for a in ("frames", "offered", "stale", "empty", "suspect", "span_ns", "cover_ns",
                  "delta_ns", "live_ns", "n_s1", "n_s1_kept", "rf_valid", "rf_vetoed",
                  "oversize"):
            setattr(out, a, getattr(out, a) + getattr(s, a))
        out.hits += s.hits
        out.rate_hits += s.rate_hits
        out.mismatch += s.mismatch
        if s.eff.size == out.eff.size:
            out.eff += s.eff
    return out


def _with_flow(counts) -> np.ndarray:
    """Per-channel counts with an empty under/overflow pair around them."""
    out = np.zeros(len(counts) + 2, dtype=np.int64)
    out[1:-1] = counts
    return out


def _index(values, ax: Axis) -> np.ndarray:
    """Full bin index (0 underflow .. n+1 overflow) of finite `values`."""
    v = np.asarray(values, dtype=np.float64).ravel()
    idx = np.floor((v - ax.lo) * (ax.n / (ax.hi - ax.lo))).astype(np.intp) + 1
    np.clip(idx, 0, ax.n + 1, out=idx)
    return idx


def _finite(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    return v if np.isfinite(v).all() else v[np.isfinite(v)]


def _fill_values(h: Hist1D, values) -> None:
    v = _finite(values)
    if v.size:
        h.add_counts(np.bincount(_index(v, h.x), minlength=h.x.n + 2), entries=v.size)


def _fill_value(h: Hist1D, value) -> None:
    if math.isfinite(value):
        _fill_index(h, int(_index([value], h.x)[0]))


def _fill_index(h: Hist1D, i: int) -> None:
    h.counts[i] += np.uint64(1)
    h.entries += 1


def _fill_2d_index(h: Hist2D, ix, iy) -> None:
    """Fill from full bin indices (0 underflow .. n+1 overflow), clipped like `_index`."""
    ix = np.clip(np.asarray(ix, dtype=np.intp).ravel(), 0, h.x.n + 1)
    iy = np.clip(np.asarray(iy, dtype=np.intp).ravel(), 0, h.y.n + 1)
    if ix.size == 0:
        return
    flat = iy * (h.x.n + 2) + ix
    h.add_counts(np.bincount(flat, minlength=h.counts.size).reshape(h.counts.shape),
                 entries=ix.size)


def _fill_2d(h: Hist2D, xs, ys) -> None:
    x = np.asarray(xs, dtype=np.float64).ravel()
    y = np.asarray(ys, dtype=np.float64).ravel()
    good = np.isfinite(x) & np.isfinite(y)
    if not good.all():
        x, y = x[good], y[good]
    if x.size == 0:
        return
    flat = _index(y, h.y) * (h.x.n + 2) + _index(x, h.x)
    h.add_counts(np.bincount(flat, minlength=h.counts.size).reshape(h.counts.shape),
                 entries=x.size)
