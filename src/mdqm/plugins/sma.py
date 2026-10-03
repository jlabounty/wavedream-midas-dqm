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
``pattern``         per S1: bit k = counter k within +-coinc window (2^n bins, n counters)
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

TOT + NIM (good frames; one set per counter k with a NIM copy, ``/DQM/SMA/NIM``):

``nim_dt_Sk``          t'_NIM - t'_TOT of the nearest TOT word, every NIM word, +-200 ns
``nim_dt_wide_Sk``     raw t_NIM - t_TOT, sampled pairs within +-2^19 ns (a lag fault)
``nim_walk_Sk``        pairs: t'_NIM - t'_TOT x the TOT word's ToT
``nim_classes_Sk``     0 paired (pairs), 1 TOT-only, 2 NIM-only, 3 echo, then sub-counts
                       of those: 4 lag-held (NIM-only), 5 in TOT shadow, 6 multi-candidate
``nim_width_Sk``       NIM word ToT field
``nim_candidates_Sk``  NIM words in the pair window per (non-echo) TOT word
``nim_lag_Sk``         (fine - fine of the reference S1 word) mod 2^20, the lag vote's input;
                       filled in the voted frames only (``NIM/lag vote every``)
``s1_coinc_tot``       ``s1_coinc`` on the TOT words alone (no merge), beside the merged one

MuPix x/y (good frames, ``/DQM/SMA/MuPix/XY``; booked only with ``enable``; mm in the reco
frame bt2026-v4, +x beam-left, plus the XY table's shift; see ``sma_mupix_xy``):

``mupix_hits_xy_L1``/``_L2``    pixels in time with a sampled S1 hit, each once, 130 x 130
``mupix_track_xy[_light|_heavy]``   L1 position of the S1-seeded tracks (all, light, heavy)
``mupix_track_xxp[...]``, ``mupix_track_yyp[...]``  x vs x', y vs y' (mrad), 130 x 77
``mupix_track_tot``     tracks: max pixel ToT of the L1 cluster x of the L2 cluster, 32 x 32
``mupix_track_state``   per judged S1 hit: 0 no L1, 1 no L2, 2 ambiguous, 3 track

With ``NIM/merge`` on (off by default), the counters (S1 included) are the
merged hits wherever counter times are used: ``dt_S*``, ``pattern``,
``s1_coinc``, ``s1_partner_hits``, ``s1_spacing_us``, the RF and delayed
histograms, the seeds, the efficiencies and the MuPix in-time matching (its
S1 sample). ``rf_phase_vs_s1_tot`` leaves the NIM-only S1 hits out (no ToT of
their own). The rest reads the words (see `pair_frame`).

Commands: ``sma::summary``, ``sma::trend``, ``sma::frame``, ``sma::raw`` (see `commands`).
"""

from __future__ import annotations

import copy
import json
import math
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from mdqm.dqm import framing
from mdqm.dqm.hist import Axis, Hist1D, Hist2D
from mdqm.plugins import sma_mupix_xy as X
from mdqm.plugins import sma_nim as N
from mdqm.plugins import sma_words as W

NCH = W.N_CHANNELS
CH_AXIS = (NCH, -0.5, NCH - 0.5)

#: Everything under ``/DQM/SMA``. A dict is an ODB directory, anything else a key.
SETTINGS_DEFAULTS: dict[str, object] = {
    #: The board's coarse = time >> shift. A per-run board setting; the self
    #: check says when another shift fits the data better.
    "Coarse shift": W.DEFAULT_SHIFT,
    #: The defaults are the cabling since run 1015: 0 clock, 1 S1, 2 S2,
    #: 3 S1L, 4 S4, 5 S5, 6 RF, 7 S3 (TOT), 8 WD trigger copy, 9-12 S2L-S5L.
    #: The proton current is no longer on the SMA.
    "Channel roles": {
        "s1": 1,
        #: S1..S5 in order, at most 8 (the pattern is one byte); the first
        #: must be the s1 channel for the pattern bit 0 to mean "S1".
        "counters": [1, 2, 7, 4, 5],
        "rf": 6,
        #: -1 = no current channel.
        "current": -1,
        #: Channels histogrammed against S1 over microseconds; -1 entries
        #: are ignored, so [-1] means none (an ODB array cannot be empty).
        "delayed": [-1],
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
        #: t(pixel) - t(S1), all pairs: the range reaches back to the default
        #: sideband so that it is seen to be flat.
        "mupix dt min ns": -2560,
        "mupix dt max ns": 2000,
        "mupix dt bin ns": 8,
        "mupix hits per chip max": 4000,
    },
    #: The MuPix pixel words of the same bank (bit 63 clear). Editing any of
    #: these rebuilds the histograms, as the Cuts do.
    "MuPix": {
        #: Chip ids (the pixel word's 5-bit field, a global ASIC id set by
        #: /Equipment/Quads/Settings/DAQ/Links/Mapping) of each plane. The
        #: identity Mapping of runs >= 200 gives L1 = 0-3, L2 = 4-7; runs up to
        #: 186 had L1 = 1-4, L2 = 5, 6, 7, 0. A chip in neither list is shown
        #: as "chip N (no plane)".
        "L1 chips": list(W.L1_CHIPS),
        "L2 chips": list(W.L2_CHIPS),
        #: log2(ckdivend2 + 1): ToT = (TS2 - ((time >> shift) & 31)) & 31, in
        #: 2^shift ticks of 8 ns (256 ns at 5).
        "ts2 shift": W.TS2_SHIFT,
        #: t(pixel) - t(S1) called in time, half open: the time walk spreads
        #: real hits over about -150..+450 ns. No time-walk correction.
        "window lo ns": W.MUPIX_WINDOW_NS[0],
        "window hi ns": W.MUPIX_WINDOW_NS[1],
        #: The accidental rate, from a window before the prompt one (it must
        #: end at or before "window lo ns": after it are decays and delayed
        #: hits). Of the same width by default; another width is scaled.
        "sideband lo ns": W.MUPIX_SIDEBAND_NS[0],
        "sideband hi ns": W.MUPIX_SIDEBAND_NS[1],
        #: Pixel hits examined per frame (the latest); more are counted and
        #: skipped. Bounds the cost of a dense frame. 0 = MuPix analysis off
        #: (pixel words are still counted).
        "max pixel hits per frame": W.MAX_PIXELS,
        #: S1 hits per frame matched against the pixels (in-time fractions,
        #: t(pixel) - t(S1)), an evenly spread subset of those the S1-seeded
        #: analyses use. 0 = all of them.
        "max S1 per frame": W.MUPIX_MAX_S1,
        #: MuPix positions in mm (reco geometry bt2026-v4, +x beam-left) and
        #: S1-seeded tracks: one accepted cluster in L1 and one in L2 in the
        #: in-time window of a sampled S1 hit (sma_mupix_xy). Editing these
        #: rebuilds, except "max S1 per frame" (a CPU knob).
        "XY": {
            #: n: no x/y histograms are booked or filled.
            "enable": True,
            #: A plane is accepted when all its in-window pixels lie on one chip
            #: within a box x box pixel square (1-64); position = their mean.
            "cluster box px": X.CLUSTER_BOX_PX,
            #: Track ToT classes, from the largest pixel ToT of the L1 and the L2
            #: cluster (ToT counts, 0-31): light = both <= this, heavy = both >=
            #: "tot heavy min". PROVISIONAL: set them from mupix_track_tot.
            "tot light max": X.TOT_LIGHT_MAX,
            "tot heavy min": X.TOT_HEAVY_MIN,
            #: Add the XY table's position (-x, +y of
            #: /Equipment/XYTable/Variables/Measured, re-read every 2 s) to every
            #: position, as the nearline does (COND:isel). Without the key: 0.
            "apply stage shift": True,
            #: S1 hits per frame given to the track finder, an evenly spread
            #: subset of the MuPix sample above (0 = all of it). The cost is
            #: ~0.1 ms + ~1 us per row on a dense frame. A CPU knob: no plot resets.
            "max S1 per frame": X.XY_MAX_S1,
        },
    },
    #: The NIM copies of the counters (since run 1015) and how they are paired
    #: with the TOT words (sma_nim.pair_counter, reco's rule). The per-counter
    #: lists follow Channel roles/counters, one entry each. Editing any of
    #: these rebuilds the histograms, except the CPU knobs (NIM_CPU_KEYS).
    "NIM": {
        #: The NIM channel of each counter, -1 = none ([-1] alone: no NIM at all).
        "channels": [3, 9, 10, 11, 12],
        #: Per counter: t'_NIM = t - offset, in whole ns. The DQM has no fine
        #: offset or lag corrections, so these are its own constants: set them
        #: from the nim_dt peak.
        "offset ns": [0, 0, 0, 0, 0],
        #: Per counter: the NIM copy's expected (fine - fine of S1) [ns], the
        #: cable delay and flight time; the lag vote judges "faulted" from it.
        "lag nominal ns": [0, 0, 0, 0, 0],
        #: NIM-only hits join the counters (pattern, efficiencies, seeds). Off
        #: until the offsets above are measured on a clean run >= 1015: with an
        #: unmeasured offset a NIM copy more than "pair window ns" off its TOT
        #: word pairs with nothing, and merging would count every particle
        #: twice. The pairing plots, summary and flags run either way.
        "merge": False,
        #: Also merge a NIM channel's NIM-only hits in a frame whose lag vote
        #: says "faulted" (off: those hits are held back and counted).
        "merge when lagged": False,
        "pair window ns": N.NimConfig.pair_window_ns,
        #: "tot" or "nim": which time a paired hit takes.
        "time source": N.NimConfig.time_source,
        #: The ToT code a NIM-only hit is given.
        "nim only tot": N.NimConfig.nim_only_tot,
        #: Counter numbers (1 = S1) whose TOT words get the echo rule; -1 = none.
        "echo counters": [3],
        "echo late tot": N.NimConfig.echo_late_tot,
        "echo edge tol ns": N.NimConfig.echo_edge_tol_ns,
        "lag tolerance ns": N.NimConfig.lag_tol_ns,
        "lag min pairs": N.NimConfig.lag_min_pairs,
        "lag dominance": N.NimConfig.lag_dominance,
        #: Vote a NIM channel's lag in every this many of its frames (1 = every
        #: frame; always while no vote of this epoch has decided). The frames
        #: between take the last decisive vote. A CPU knob: no plot resets.
        "lag vote every": 4,
        #: nim_dt_wide pairs per counter and frame, at most (about; 0 = none).
        #: A CPU knob: no plot resets.
        "wide pairs per frame": 1024,
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
        #: Good frames arrive but none has had an S1 seed for this long: the
        #: S1-seeded view is showing an old frame (flag ``no_seeds``).
        "no seeds s": 10.0,
        #: The SMA <-> MuPix time-sync monitor (flag ``mupix_sync``): the share
        #: of S1 hits with an L1 AND an L2 pixel hit in time, accidentals taken
        #: out, over the last "mupix sync window s", below "mupix sync min
        #: fraction" for more than "mupix sync hold s" while S1 fires (at least
        #: "mupix sync min S1" analysed S1 hits in the window). It clears above
        #: min fraction + "mupix sync clear margin". Run 1008 sits at 0.88 and
        #: run 682 at 0.84; a lost time sync gives ~0.
        "mupix sync min fraction": 0.3,
        "mupix sync hold s": 30.0,
        "mupix sync window s": 10.0,
        "mupix sync min S1": 200,
        "mupix sync clear margin": 0.05,
        #: TOT + NIM (flags nim_pairing, nim_offset, nim_lag): pair efficiency
        #: paired / (paired + TOT-only) below these; the median nim_dt further
        #: than this from 0; the lag state "faulted" in more than this share of
        #: the frames with a lag state, after at least "nim lag min votes"
        #: frames voted "faulted" since the last rebuild or run start. Each
        #: needs "min hits" in the window.
        "nim pairing warn fraction": 0.8,
        "nim pairing error fraction": 0.5,
        "nim offset max ns": 5.0,
        "nim lag max fraction": 0.5,
        "nim lag min votes": 3,
        #: Only these channels raise mismatch / ToT flags (S1, the counters
        #: and the RF by default); the others are still in the summary table.
        #: Here and not under Cuts, so editing it never resets a plot.
        "mismatch flag channels": [1, 2, 4, 5, 6, 7],
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
        #: The last good frames kept for the event display's other seed
        #: choices and filters (sma::frame "seed"/"filters"), which search
        #: back through them when the newest frame has no matching seed. At
        #: most this many frames and this many MB (estimated array bytes); the
        #: newest good frame is always kept.
        "seed ring frames": 8,
        "seed ring MB": 24.0,
    },
}

TREND_S = 600
#: The largest `Sampling/seed ring frames` accepted: bounds a no-match search.
SEED_RING_MAX = 64
#: Cached frame payloads and seed selections, at most (LRU on top of the
#: per-frame purge): bounds what many viewers' seed/filter choices can hold.
BLOB_CACHE_MAX = 64
SEL_CACHE_MAX = 256
#: CPU (process time) one sma::frame request may spend on new seed selections
#: while searching back; the rest of the ring is searched by the next polls
#: (selections already made are cached and cost nothing). One selection of a
#: 40000-word frame is ~2-6 ms.
SEARCH_BUDGET_S = 0.02
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


def _pct(frac) -> str:
    """A fraction as the page says it: "93 %", "6.2 %" below 10 %."""
    x = 100.0 * float(frac)
    return f"{x:.1f} %" if x < 10 else f"{x:.0f} %"


def _ratio(a, b, digits=6):
    return _num(a / b, digits) if b else None


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------

@dataclass
class NimSettings:
    """The parsed ``/DQM/SMA/NIM``: per counter (``Roles.counters`` order) and shared."""

    #: NIM channel per counter, -1 = none.
    channels: tuple = ()
    #: Per counter, whole ns.
    offsets: tuple = ()
    nominal: tuple = ()
    #: Per counter: the echo rule runs on its TOT words.
    echo: tuple = ()
    merge: bool = False
    merge_when_lagged: bool = False
    #: Vote each NIM channel's lag every this many of its frames.
    lag_every: int = 4
    wide_budget: int = 1024
    cfg: N.NimConfig = field(default_factory=N.NimConfig)

    def pairs(self, counters) -> list[tuple[int, int, int]]:
        """``(k, TOT channel, NIM channel)`` of every counter with a NIM copy."""
        return [(k, c, n) for k, (c, n) in enumerate(zip(counters, self.channels, strict=False))
                if n >= 0]

    @property
    def active(self) -> bool:
        return any(n >= 0 for n in self.channels)


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
    flag_channels: tuple = (1, 2, 4, 5, 6, 7)
    max_words: int | None = 1 << 20
    max_gap_ns: int = 10 * 10**9
    max_overlap_ns: int = 10 * 10**6
    raw_ring_bytes: int = 16 << 20
    seed_ring_frames: int = 8
    seed_ring_bytes: int = 24 << 20
    mupix: W.MuPixCuts = field(default_factory=W.MuPixCuts)
    xy: X.XYSettings = field(default_factory=X.XYSettings)
    nim: NimSettings = field(default_factory=NimSettings)
    binning: dict = field(default_factory=dict)
    check: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)

    @property
    def scan(self) -> tuple:
        return tuple(self.cuts.shift_scan)

    def role_channels(self) -> set:
        """Every channel with a role, the NIM copies included; an unset
        optional role (-1) is none."""
        r = self.roles
        return {c for c in (r.s1, r.rf, r.current, *r.counters, *r.delayed, *self.nim.channels)
                if c >= 0}

    @property
    def merging(self) -> bool:
        """Counters are the merged TOT + NIM hits (NIM/merge on and a NIM copy cabled)."""
        return self.nim.merge and self.nim.active


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

    def at(tree, section):
        for part in section.split("/"):     # "MuPix/XY": a directory in a directory
            tree = tree[part]
        return tree

    def get(section, key, conv, check=None):
        raw = at(s, section)[key] if section else s[key]
        dflt = at(d, section)[key] if section else d[key]
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

    def opt_chan(v):
        """A channel, or -1 for "none" (an optional role)."""
        v = int(v)
        return -1 if v == -1 else chan(v)

    def opt_chans(v):
        """Channels; -1 entries are dropped, so [] and [-1] are both "none"."""
        return tuple(c for c in (opt_chan(x) for x in (v if isinstance(v, list | tuple)
                                                         else [v])) if c >= 0)

    shift = get(None, "Coarse shift", int, lambda v: 0 <= v <= W.MAX_COARSE_SHIFT)
    R = "Channel roles"
    roles = W.Roles(s1=get(R, "s1", chan),
                    counters=get(R, "counters", chans, lambda v: 1 <= len(v) <= W.MAX_COUNTERS),
                    rf=get(R, "rf", chan), current=get(R, "current", opt_chan),
                    delayed=get(R, "delayed", opt_chans))
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
        "mupix dt min": get(B, "mupix dt min ns", int),
        "mupix dt max": get(B, "mupix dt max ns", int),
        "mupix dt bin": get(B, "mupix dt bin ns", int, pos),
        "mupix chip max": get(B, "mupix hits per chip max", int, pos),
    }
    if binning["mupix dt max"] <= binning["mupix dt min"]:
        errors.append("Binning/mupix dt max ns <= mupix dt min ns; using the defaults")
        binning["mupix dt min"], binning["mupix dt max"] = -2560, 2000
    if binning["gap max"] <= binning["gap min"]:
        errors.append("Binning/gap max ms <= gap min ms; using the defaults")
        binning["gap min"], binning["gap max"] = -5.0, 45.0
    if binning["span max"] <= binning["span min"]:
        errors.append("Binning/span log10 ms max <= min; using the defaults")
        binning["span min"], binning["span max"] = -2.0, 4.0
    if binning["rate max"] <= binning["rate min"]:
        errors.append("Binning/rate log10 Hz max <= min; using the defaults")
        binning["rate min"], binning["rate max"] = 0.0, 7.0

    M = "MuPix"

    def chip_ids(v):
        # -1 is an empty quadrant slot (MuPix x/y): no chip, left out of the planes.
        v = [int(x) for x in (v if isinstance(v, list | tuple) else [v])]
        if any(not (0 <= x < W.N_CHIP_IDS or x == X.EMPTY_SLOT) for x in v):
            raise ValueError("not a chip id 0-31 (or -1, an empty quadrant)")
        return tuple(v)

    # The lists as written (quadrant order, -1 slots: x/y) and the plane members.
    slots1, slots2 = get(M, "L1 chips", chip_ids), get(M, "L2 chips", chip_ids)
    l1 = tuple(c for c in slots1 if c != X.EMPTY_SLOT)
    l2 = tuple(c for c in slots2 if c != X.EMPTY_SLOT)
    if set(l1) & set(l2):
        errors.append(f"MuPix/L1 chips and L2 chips share chip(s) {sorted(set(l1) & set(l2))}; "
                      "using the defaults")
        l1, l2 = slots1, slots2 = W.L1_CHIPS, W.L2_CHIPS
    window = (get(M, "window lo ns", int), get(M, "window hi ns", int))
    sideband = (get(M, "sideband lo ns", int), get(M, "sideband hi ns", int))
    if window[1] <= window[0]:
        errors.append("MuPix/window hi ns <= window lo ns; using the defaults")
        window = W.MUPIX_WINDOW_NS
    if sideband[1] <= sideband[0] or sideband[1] > window[0]:
        errors.append("MuPix/sideband must be a window before the in-time window (sideband lo "
                      "< sideband hi <= window lo ns); using the defaults")
        sideband = W.MUPIX_SIDEBAND_NS
        if sideband[1] > window[0]:
            window = W.MUPIX_WINDOW_NS
    mupix = W.MuPixCuts(
        l1=l1, l2=l2,
        ts2_shift=get(M, "ts2 shift", int, lambda v: 0 <= v <= W.MAX_TS2_SHIFT),
        window_ns=window, sideband_ns=sideband,
        max_pixels=get(M, "max pixel hits per frame", int, lambda v: v >= 0),
        max_s1=get(M, "max S1 per frame", int, lambda v: v >= 0) or None)

    XY = "MuPix/XY"
    tot_code = lambda v: 0 <= v <= W.TS2_MASK  # noqa: E731
    light, heavy = get(XY, "tot light max", int, tot_code), get(XY, "tot heavy min", int, tot_code)
    if light >= heavy:
        errors.append(f"MuPix/XY/tot light max ({light}) must be below tot heavy min ({heavy}); "
                      "using the defaults")
        light, heavy = X.TOT_LIGHT_MAX, X.TOT_HEAVY_MIN
    xy_max_s1 = get(XY, "max S1 per frame", int)
    if not 1 <= xy_max_s1 <= X.XY_MAX_S1_LIMIT:
        clamped = min(max(xy_max_s1, 1), X.XY_MAX_S1_LIMIT)
        errors.append(f"MuPix/XY/max S1 per frame={xy_max_s1}: a cap within 1-"
                      f"{X.XY_MAX_S1_LIMIT} (0 is not 'all'); using {clamped}")
        xy_max_s1 = clamped
    xy_enable = get(XY, "enable", _as_bool)
    # Placed only from lists that say where each chip is (one per quadrant);
    # anything else turns x/y off with one settings error, rather than putting
    # chips silently in the wrong quadrant.
    bad_lists = X.check_chip_lists(slots1, slots2)
    off_reason = None
    if not mupix.enabled:
        off_reason = "the MuPix analysis is off (MuPix/max pixel hits per frame = 0)"
    elif bad_lists:
        off_reason = bad_lists
        if xy_enable:
            errors.append(f"MuPix/XY: {bad_lists}; MuPix x/y is off")
    xy = X.XYSettings(
        enable=xy_enable,
        box=get(XY, "cluster box px", int, lambda v: 1 <= v <= X.MAX_BOX_PX),
        tot_light_max=light, tot_heavy_min=heavy,
        apply_stage=get(XY, "apply stage shift", _as_bool),
        max_s1=xy_max_s1,
        placement=X.placement(slots1, slots2) if not bad_lists else X.placement((), ()),
        off_reason=off_reason)

    nim = _parse_nim(s, roles, get, opt_chan, errors)

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
        seed_ring_frames=get("Sampling", "seed ring frames", int,
                             lambda v: 0 <= v <= SEED_RING_MAX),
        seed_ring_bytes=int(get("Sampling", "seed ring MB", float,
                                lambda v: 0 <= v <= 1024) * (1 << 20)),
        max_gap_ns=int(get(C, "max gap s", float, pos) * 1e9),
        max_overlap_ns=int(get(C, "max overlap ms", float, lambda v: v >= 0) * 1e6),
        mupix=mupix, xy=xy, nim=nim, binning=binning, check=check, errors=errors)


def _as_bool(v) -> bool:
    """An ODB BOOL, or y/n/true/false/1/0 typed as text or a number."""
    if isinstance(v, bool | np.bool_):
        return bool(v)
    if isinstance(v, int | np.integer) and int(v) in (0, 1):
        return bool(v)
    t = str(v).strip().lower()
    if t in ("y", "yes", "true", "1"):
        return True
    if t in ("n", "no", "false", "0"):
        return False
    raise ValueError("not a yes/no")


def _whole_ns(v) -> int:
    """A time in ns rounded to a whole ns (the pairing keeps int64 times exact)."""
    x = float(v)
    if not math.isfinite(x):
        raise ValueError("not finite")
    return int(round(x))


def _parse_nim(s: dict, roles: W.Roles, get, opt_chan, errors: list) -> NimSettings:
    """``/DQM/SMA/NIM`` -> NimSettings, falling back per key (see parse_settings).

    The per-counter lists must have one entry per counter (``[-1]`` alone is
    accepted for "no NIM copies", and without any NIM channel the offsets and
    nominals are not judged). A NIM channel that is also S1, a counter, the
    RF, ``current`` or a ``delayed`` channel means the channel roles and the
    NIM map disagree -- typically a pre-1015 ODB that got the seeded NIM
    defaults -- so NIM is switched off altogether (no pairing, no merge) with
    one settings error naming the collisions. A NIM channel repeating another
    counter's is dropped (-1) alone. Offsets and nominals are whole ns; a
    fractional one is rounded with a settings note.
    """
    K = "NIM"
    d = SETTINGS_DEFAULTS[K]
    nc = len(roles.counters)

    def as_list(v):
        return list(v) if isinstance(v, list | tuple) else [v]

    def per_counter(key, conv, fill, allow_none=False):
        raw = s[K][key]
        dflt = [conv(x) for x in d[key]]
        fallback = tuple(dflt) if len(dflt) == nc else (fill,) * nc
        try:
            vals = [conv(x) for x in as_list(raw)]
        except (TypeError, ValueError) as exc:
            errors.append(f"NIM/{key}={raw!r}: {exc}; using {list(fallback)!r}")
            return fallback
        if allow_none and vals == [-1]:
            return (-1,) * nc
        if len(vals) != nc:
            errors.append(f"NIM/{key}={raw!r}: {len(vals)} entries for {nc} counters; "
                          f"using {list(fallback)!r}")
            return fallback
        return tuple(vals)

    chans = list(per_counter("channels", opt_chan, -1, allow_none=True))
    # What each role channel is, for the message (S1 first: it is a counter too).
    role_of: dict[int, str] = {}
    for c, what in ((roles.s1, "S1"),
                    *((c, f"counter S{k + 1}") for k, c in enumerate(roles.counters)),
                    (roles.rf, "the RF"), (roles.current, "current"),
                    *((c, "delayed") for c in roles.delayed)):
        if c >= 0:
            role_of.setdefault(c, what)
    clash = [(k, n) for k, n in enumerate(chans) if n >= 0 and n in role_of]
    if clash:
        errors.append(
            "Channel roles look pre-1015: NIM/channels "
            + ", ".join(f"{n} (S{k + 1}'s NIM copy) is {role_of[n]}" for k, n in clash)
            + "; NIM is off (no pairing, no merge). Set the run-1015 roles with the odbedit "
            "lines in docs/SMA-DQM.md ('Settings'), or NIM/channels = -1 for the old cabling")
        chans = [-1] * nc
    seen = set()
    for k, n in enumerate(chans):
        if n < 0:
            continue
        if n in seen:
            errors.append(f"NIM/channels: S{k + 1}'s NIM channel {n} is also another counter's "
                          f"NIM channel; S{k + 1} gets no NIM copy")
            chans[k] = -1
            continue
        seen.add(n)

    def nominal(v):
        v = _whole_ns(v)
        if not -N.HALF_FINE_WRAP_NS < v < N.HALF_FINE_WRAP_NS:
            raise ValueError("not in (-2^19, 2^19) ns")
        return v

    def note_rounding(key, vals):
        raw = as_list(s[K][key])
        try:
            frac = [float(x) for x in raw]
        except (TypeError, ValueError):
            return
        if len(frac) == len(vals) and any(f != v for f, v in zip(frac, vals, strict=True)):
            errors.append(f"NIM/{key}={raw!r}: whole ns only; using {list(vals)!r}")

    if any(n >= 0 for n in chans):
        offsets = per_counter("offset ns", _whole_ns, 0)
        nominals = per_counter("lag nominal ns", nominal, 0)
        note_rounding("offset ns", offsets)
        note_rounding("lag nominal ns", nominals)
    else:                                   # nothing to align: not judged
        offsets = nominals = (0,) * nc

    def counter_numbers(v):
        out = {int(x) for x in as_list(v)} - {-1}
        if any(not 1 <= x <= nc for x in out):
            raise ValueError(f"not a counter number 1-{nc} (or -1)")
        return frozenset(out)

    raw = s[K]["echo counters"]
    try:
        echo = counter_numbers(raw)
    except (TypeError, ValueError) as exc:
        echo = frozenset(x for x in d["echo counters"] if 1 <= x <= nc)
        errors.append(f"NIM/echo counters={raw!r}: {exc}; using {sorted(echo) or [-1]!r}")
    echo = tuple(k + 1 in echo for k in range(nc))
    try:
        cfg = N.NimConfig(
            pair_window_ns=get(K, "pair window ns", float, lambda v: v > 0),
            time_source=get(K, "time source", lambda v: str(v).strip().lower(),
                            lambda v: v in N.TIME_SOURCES),
            nim_only_tot=get(K, "nim only tot", int, lambda v: 0 <= v <= 255),
            echo_late_tot=get(K, "echo late tot", int),
            echo_edge_tol_ns=get(K, "echo edge tol ns", float),
            lag_tol_ns=get(K, "lag tolerance ns", float, lambda v: 0 < v < 1024),
            lag_min_pairs=get(K, "lag min pairs", int, lambda v: v >= 1),
            lag_dominance=get(K, "lag dominance", float, lambda v: v >= 1))
    except ValueError as exc:                     # a combination NimConfig refuses
        errors.append(f"NIM: {exc}; using the pairing defaults")
        cfg = N.NimConfig()
    return NimSettings(
        channels=tuple(chans), offsets=offsets, nominal=nominals, echo=echo,
        merge=get(K, "merge", _as_bool), merge_when_lagged=get(K, "merge when lagged", _as_bool),
        lag_every=get(K, "lag vote every", int, lambda v: v >= 1),
        wide_budget=get(K, "wide pairs per frame", int, lambda v: v >= 0), cfg=cfg)


#: ``/DQM/SMA/NIM`` keys that only bound the CPU: editing them rebuilds nothing.
NIM_CPU_KEYS = ("wide pairs per frame", "lag vote every")


def shape_fingerprint(settings: dict) -> str:
    """What changes the histograms: shift, roles (but not labels), cuts, binning, MuPix, NIM.

    Labels are left out on purpose -- renaming a channel must never reset an
    afternoon of plots. The self-check and sampling settings change no plot,
    nor do NIM's CPU knobs (`NIM_CPU_KEYS`). ``MuPix/XY`` is left out too: its
    keys reset only the x/y maps (`SmaPlugin.apply_settings`), and the XY
    table's position is no setting at all (`SmaPlugin.poll_odb`).
    """
    s = _merge(SETTINGS_DEFAULTS, settings)
    roles = {k: v for k, v in s["Channel roles"].items() if k != "labels"}
    # NIM's CPU knobs change how much is filled, not what a plot means.
    nim = {k: v for k, v in s["NIM"].items() if k not in NIM_CPU_KEYS}
    mupix = {k: v for k, v in s["MuPix"].items() if k != "XY"}
    return json.dumps({"shift": s["Coarse shift"], "roles": roles, "Cuts": s["Cuts"],
                       "Binning": s["Binning"], "MuPix": mupix, "NIM": nim},
                      sort_keys=True,
                      default=str)


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
# TOT + NIM pairing of a good frame
# ---------------------------------------------------------------------------

#: `NimWords.cls` of a kept hit that is neither a paired counter's TOT word
#: nor its NIM word (the RF, S1 without a NIM copy, ...).
NIM_NO_CLASS = 255
#: `NimWords.flags` bit (above sma_nim's flag bits): a NIM-only word held back
#: from the merge because its channel's lag vote said "faulted" in this frame.
NIM_LAG_HELD = 1 << 15
#: Columns of `_Second.nim` (one row per counter). The first nine are counts
#: of words / hits; "frames" counts the frames paired; then the frames by
#: lag-vote state (sma_nim.LAG_STATES), the frames not voted (NIM/lag vote
#: every), and the frames by lag state (`NimFrame.state`: the frame's own
#: decisive vote, else the epoch's last).
NIM_COLS = ("tot_words", "nim_words", "paired", "tot_only", "nim_only", "echo", "lag_held",
            "shadow", "multi", "frames", *(f"lag_{x}" for x in N.LAG_STATES), "lag_skipped",
            "state_ok", "state_faulted")
_NC = {name: k for k, name in enumerate(NIM_COLS)}


class NimWords(NamedTuple):
    """A frame's pairing per kept hit (``Frame.s_*`` index), for the event display.

    ``cls``: sma_nim.PAIRED / TOT_ONLY / ECHO_WORD for a paired counter's TOT
    word, PAIRED / NIM_ONLY for its NIM word, NIM_NO_CLASS otherwise.
    ``partner``: the ``s_*`` index of the word it is paired with, -1 if none.
    ``flags``: sma_nim's flag bits (a paired NIM word carries its pair's), plus
    NIM_LAG_HELD.
    """

    cls: np.ndarray        # uint8
    partner: np.ndarray    # int32
    flags: np.ndarray      # uint16


@dataclass
class LagMemory:
    """One NIM channel's lag state over an epoch (`pair_frame`'s ``memory``).

    ``state`` is the last decisive vote ("ok" or "faulted") since the last
    rebuild or run start, None before the first. A frame whose own vote does
    not decide ("none": too few NIM words with an S1 reference; "ambiguous"),
    or that is not voted at all (``NIM/lag vote every``), takes it: the lag
    fault is a whole-file state, so a quiet frame after a faulted one is
    still faulted. A later "ok" vote clears it.
    """

    state: str | None = None
    #: Frames to pass before the next vote.
    skip: int = 0
    #: Decisive votes this epoch, and how many of them said "faulted".
    votes: int = 0
    faulted_votes: int = 0


@dataclass
class NimFrame:
    """`pair_frame` on one frame; the dicts are keyed by counter index k (0 = S1)."""

    results: dict          # k -> sma_nim.PairResult
    votes: dict            # k -> sma_nim.LagVote, or None: not voted in this frame
    #: k -> the channel's lag state in this frame: its own decisive vote,
    #: else the epoch's last (`LagMemory`); None: none known.
    state: dict
    held: dict             # k -> bool: its NIM-only hits were held back from the merge
    #: TOT channel -> W.CounterHits, the merged hits (empty when not merging).
    hits: dict
    words: NimWords


def pair_frame(fr: W.Frame, cfg: Config, memory: dict | None = None,
               update: bool = True) -> NimFrame | None:
    """Pair every counter that has a NIM copy, on all its kept words, and vote the lags.

    Per counter k with TOT channel c and NIM channel n (``NIM/channels``):

    * `sma_nim.pair_counter` on c's and n's kept words: n aligned by its
      ``NIM/offset ns``, the echo rule if k is in ``NIM/echo counters``, the
      frame's first/last kept hit and the ``Frame.chan`` positions for the edge
      flag and the ties, as nearline does.
    * `sma_nim.lag_vote` of n against the S1 role channel on the raw fine and
      coarse fields of every trigger word, with k's ``NIM/lag nominal ns``.
      Measured only: nothing is corrected. With ``memory`` (k -> `LagMemory`,
      the plugin's, per epoch) the vote runs in every ``NIM/lag vote every``-th
      frame of the channel once a vote has decided, and a frame without a
      decisive vote of its own takes the memory's state; ``update`` False
      reads the memory without changing it (a frame rebuilt for a page).
      Without ``memory`` every frame votes and stands alone.
    * With merging on (`Config.merging`), c's counter hits become the merged
      hits (`W.CounterHits`, set on the frame by the caller). In a frame whose
      lag state is "faulted", n's NIM-only hits are held back unless
      ``NIM/merge when lagged``: a lagged NIM word sits ~150 us or ~0.9 us off
      and would only add accidentals. Nothing is held with merging off.

    None when no counter has a NIM copy. With merging off the counters stay
    the TOT words (``hits`` is empty), and ``NIM/time source`` changes nothing.
    """
    nim = cfg.nim
    if not nim.active:
        return None
    n_kept = int(fr.s_t.size)
    cls = np.full(n_kept, NIM_NO_CLASS, dtype=np.uint8)
    partner = np.full(n_kept, -1, dtype=np.int32)
    flags = np.zeros(n_kept, dtype=np.uint16)
    out = NimFrame(results={}, votes={}, state={}, held={}, hits={},
                   words=NimWords(cls, partner, flags))
    s1 = cfg.roles.s1
    pairs = nim.pairs(cfg.roles.counters)
    mems = {k: (memory.setdefault(k, LagMemory()) if memory is not None else None)
            for k, _c, _n in pairs}
    # Which channels vote in this frame, then the stream positions of their
    # words and of S1's in one pass over the bank (the vote of one NIM channel
    # reads only its words and S1's).
    voting = {k for k, _c, _n in pairs
              if mems[k] is None or not update or mems[k].state is None or mems[k].skip <= 0}
    pos = {}
    if voting:
        want = [s1] + [n for k, _c, n in pairs if k in voting]
        sel = np.flatnonzero(np.isin(fr.ch, want))
        chs = fr.ch[sel]
        pos = {c: sel[chs == c] for c in want}
    for k, c, n in pairs:
        it = np.asarray(fr.chan[c], dtype=np.intp)
        inn = np.asarray(fr.chan[n], dtype=np.intp)
        pr = N.pair_counter(fr.s_t[it], fr.s_tot[it], fr.s_t[inn], fr.s_tot[inn], nim.cfg,
                            frame_lo=fr.first, frame_hi=fr.last, idx_tot=it, idx_nim=inn,
                            nim_offset_ns=nim.offsets[k], echo=nim.echo[k])
        mem = mems[k]
        vote = None
        if k in voting:
            vote = N.lag_vote(fr.ch, fr.coarse, fr.fine, n, s1, nim.cfg,
                              nominal_ns=nim.nominal[k], ref_idx=pos[s1], nim_idx=pos[n])
        decided = vote is not None and vote.state in ("ok", "faulted")
        state = vote.state if decided else (mem.state if mem is not None else None)
        if mem is not None and update:
            if decided:
                mem.state = vote.state
                mem.votes += 1
                mem.faulted_votes += vote.faulted
            # The next vote: in lag_every frames once a state is known, else next frame.
            mem.skip = (nim.lag_every - 1 if vote is not None else mem.skip - 1) \
                if mem.state is not None else 0
        held = state == "faulted" and nim.merge and not nim.merge_when_lagged
        out.results[k], out.votes[k], out.state[k], out.held[k] = pr, vote, state, held
        if nim.merge:
            m = pr.merged_hits(nim_only=not held)
            from_nim = m.cls == N.NIM_ONLY
            carrier = np.empty(m.n, dtype=np.intp)
            carrier[from_nim] = inn[m.src[from_nim]]
            carrier[~from_nim] = it[m.src[~from_nim]]
            out.hits[c] = W.CounterHits(np.asarray(m.t, dtype=np.int64), m.tot, carrier)
        cls[it] = pr.cls_tot
        cls[inn] = pr.cls_nim
        flags[it] = pr.flags_tot
        f_nim = pr.flags_nim
        if held:
            f_nim = f_nim | np.where(pr.cls_nim == N.NIM_ONLY, NIM_LAG_HELD, 0).astype(np.uint16)
        flags[inn] = f_nim
        pa = pr.partner_tot >= 0
        partner[it[pa]] = inn[pr.partner_tot[pa]]
        pb = pr.partner_nim >= 0
        partner[inn[pb]] = it[pr.partner_nim[pb]]
    return out


# ---------------------------------------------------------------------------
# per-second accumulators (trend, summary, shift ring)
# ---------------------------------------------------------------------------

class _Second:
    """Everything the trend, summary and shift check need, summed over one second."""

    __slots__ = ("t", "epoch", "frames", "offered", "stale", "empty", "suspect", "span_ns",
                 "cover_ns", "delta_ns", "live_ns", "hits", "rate_hits", "oversize",
                 "mismatch", "tot_bad",
                 "stale_words", "n_s1", "n_s1_kept", "eff", "rf_valid", "rf_vetoed", "scan",
                 "shift_counts", "shift_n", "mp_frames", "mp_pix", "mp_examined", "mp_skipped",
                 "mp_n", "mp_in", "mp_side", "mp_chip", "mp_rows", "mp_unsorted", "nim",
                 "xy_state", "xy_light", "xy_heavy", "xy_ctrk")

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
        #: MuPix, good frames: frames analysed with MuPix, their pixel words,
        #: those examined and skipped (the cap), S1 hits judged, of them with
        #: L1 / L2 / both in time and in the sideband, examined hits per chip,
        #: rows >= 250, frames whose pixel stream was out of time order.
        self.mp_frames = self.mp_pix = self.mp_examined = self.mp_skipped = 0
        self.mp_n = self.mp_rows = self.mp_unsorted = 0
        self.mp_in = np.zeros(3, dtype=np.int64)
        self.mp_side = np.zeros(3, dtype=np.int64)
        self.mp_chip = np.zeros(W.N_CHIP_IDS, dtype=np.int64)
        #: TOT + NIM, per counter (rows) and NIM_COLS (columns); all zero for
        #: a counter without a NIM copy.
        self.nim = np.zeros((n_counters, len(NIM_COLS)), dtype=np.int64)
        #: MuPix x/y: judged S1 rows per track state (sma_mupix_xy.STATE_NAMES),
        #: the tracks of the light and heavy ToT classes, and the tracks classed
        #: (xy_ctrk, their denominator: restarts with a ToT-cut edit).
        self.xy_state = np.zeros(len(X.STATE_NAMES), dtype=np.int64)
        self.xy_light = self.xy_heavy = self.xy_ctrk = 0

    def nim_eff(self) -> list | None:
        """Pair efficiency paired / (paired + TOT-only) per counter (S1 first),
        None for a counter without such hits; None when no counter has any."""
        p, t = self.nim[:, _NC["paired"]], self.nim[:, _NC["tot_only"]]
        if not (p + t).any():
            return None
        return [_ratio(a, a + b, 4) for a, b in zip(p, t, strict=True)]

    def mupix_row(self, widths=(1.0, 1.0)) -> dict | None:
        """The trend's MuPix entry: in-time, sideband and accidental-corrected
        fractions of the judged S1 hits, [L1, L2, L1+L2]; None without any."""
        if not self.mp_n:
            return None
        fin = [_ratio(x, self.mp_n, 4) for x in self.mp_in]
        fside = [_ratio(x, self.mp_n, 4) for x in self.mp_side]
        return {"n_s1": self.mp_n, "in": fin, "side": fside,
                "corr": [_num(W.accidental_corrected(a, b, *widths), 4)
                         for a, b in zip(fin, fside, strict=True)],
                "pix_per_frame": _ratio(self.mp_pix, self.mp_frames, 5)}

    def row(self, counters=(), mismatch_max: float | None = None, widths=(1.0, 1.0)) -> dict:
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
            "mupix": self.mupix_row(widths),
            # TOT + NIM pair efficiency per counter, S1 first (trend's nim_counters).
            "nim_eff": self.nim_eff(),
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
    #: When it was analysed (the plugin's clock) and how many good frames had
    #: been analysed by then (itself included): the seeded view's staleness.
    at: float = 0.0
    good_n: int = 0
    #: The NIM view of the settings the frame was analysed (paired) under:
    #: ``{"roles": ..., "nim_merge": bool}``, None without NIM copies. The
    #: event display encodes from it, never from the settings of now: a frame
    #: held across a settings edit keeps the map and merge state of its pairing.
    nim: dict | None = None


def snapshot_bytes(snap: _Snapshot) -> int:
    """Estimated bytes a held snapshot keeps alive: its frame's and analysis's arrays."""
    n = 0
    for obj in (snap.fr, snap.an):
        if obj is None:
            continue
        for v in vars(obj).values():
            if isinstance(v, W.Pixels):
                n += v.nbytes()
            elif isinstance(v, np.ndarray):
                n += v.nbytes
            elif isinstance(v, list | tuple):
                n += sum(x.nbytes for x in v if isinstance(x, np.ndarray))
            elif isinstance(v, dict):
                for x in v.values():
                    if isinstance(x, tuple):
                        n += sum(y.nbytes for y in x if isinstance(y, np.ndarray))
    return n


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
        self._frame_cache: OrderedDict = OrderedDict()
        self.frames_good = 0
        #: The last good frames, oldest first, for the seed choices and filters
        #: (Sampling/seed ring frames and MB); the newest is always there.
        self._ring: deque[_Snapshot] = deque()
        self._ring_bytes = 0
        self._ring_sizes: dict[int, int] = {}
        #: (seq, seed mode, filters) -> sma_words.SeedSelection, see _selection.
        self._sel_cache: OrderedDict = OrderedDict()
        #: Since when good frames have had no S1 seed (None: the last had one),
        #: why the newest one had none, and when the last good frame came.
        self._noseed_since: float | None = None
        self._noseed_reason = ""
        self._last_good_at: float | None = None
        #: seq -> _RawEntry of the last analysed frames, bounded by
        #: Sampling/raw ring MB (see _keep_raw).
        self._raw: OrderedDict[int, _RawEntry] = OrderedDict()
        self._raw_bytes = 0
        self.raw_not_kept = 0
        self._seconds: deque[_Second] = deque(maxlen=TREND_S + 1)
        self._cur: _Second | None = None
        #: The MuPix time-sync monitor (see _mupix_sync): since when the
        #: L1+L2 in-time fraction has been low, whether it is flagged, the last
        #: value judged and when it was last evaluated.
        self._mp_low_since: float | None = None
        self._mp_flagged = False
        self._mp_last: dict = {"state": "insufficient"}
        self._mp_eval_t: int | None = None
        #: Per counter index k: the last frame's lag vote with a lag,
        #: ``(when, lag ns, state)``; cleared with each epoch.
        self._nim_last: dict[int, tuple] = {}
        #: Per counter index k: its NIM channel's lag state this epoch
        #: (`LagMemory`: the sticky state, the vote cadence); cleared with each epoch.
        self._nim_mem: dict[int, LagMemory] = {}
        #: The XY table's position for MuPix x/y: (xpos, ypos) in mm as the
        #: table reads it, where it came from, and a note when it is not the
        #: ODB's (see poll_odb, set_stage). One tuple, swapped whole.
        self._stage: tuple = (0.0, 0.0, "none", "no stage reading yet: (0, 0) mm used")
        #: x/y map resets by MuPix/XY edits (no rebuild; see apply_settings).
        self.xy_resets = 0
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

        ``MuPix/XY`` edits never rebuild: enable, the cluster square and
        ``apply stage shift`` (and a chip-list change that only moves x/y)
        re-book the x/y maps alone; the ToT cuts re-book only the light and
        heavy maps (``mupix_track_tot``, which the cuts are read from, keeps
        filling). The x/y summary counters restart with them; nothing else
        does, and the epoch stays.
        """
        old = self.cfg.xy
        self.cfg = parse_settings(settings)
        if rebuild:
            for name in self.store.names():
                if name.startswith("sma/"):
                    self.store.remove(name)
            self._build()
            self.rebuilds += 1
            self._new_epoch()
            return
        new = self.cfg.xy
        if new.reset_key() != old.reset_key():
            self._build_xy()
            self._reset_xy_counters(classes_only=False)
            self.xy_resets += 1
        elif (new.tot_light_max, new.tot_heavy_min) != (old.tot_light_max, old.tot_heavy_min):
            self._build_xy(classes_only=True)
            self._reset_xy_counters(classes_only=True)
            self.xy_resets += 1

    def _reset_xy_counters(self, classes_only: bool) -> None:
        for sec in {id(x): x for x in (*self._seconds, self._cur) if x is not None}.values():
            sec.xy_light = sec.xy_heavy = sec.xy_ctrk = 0
            if not classes_only:
                sec.xy_state[:] = 0

    # -- the XY table ---------------------------------------------------------------

    def poll_odb(self, odb_get) -> None:
        """Read the ODB values that change no histogram: the XY table's position.

        Called by the analyzer on every settings poll (every 2 s) with
        ``client.odb_get``. Not a setting (`shape_fingerprint` never sees it):
        a moving stage resets no plot, the frames after the poll take the new
        shift. Only read while MuPix x/y is on with ``apply stage shift``; a
        missing or unreadable key (a standalone rig, an old ODB) gives (0, 0)
        with a note in the summary, never an error. The table's own readback
        period comes on top of the 2 s: frames up to a few seconds after a
        move take the old shift, so a stage step blurs that long.

        Any other exception (a bug here) keeps the last position, marks the
        source "error" with the exception in the note, and is raised for the
        analyzer to log once.
        """
        try:
            xy = self.cfg.xy
            if not (xy.active and xy.apply_stage):
                return
            try:
                v = odb_get(X.STAGE_PATH)
                v = list(v) if isinstance(v, list | tuple | np.ndarray) else [v]
                x, y = float(v[0]), float(v[1])
                if not (math.isfinite(x) and math.isfinite(y)):
                    raise ValueError("not finite")
            except Exception as exc:                # noqa: BLE001
                self._stage = (0.0, 0.0, "missing",
                               f"{X.STAGE_PATH} not readable ({type(exc).__name__}): "
                               "(0, 0) mm used")
                return
            self._stage = (x, y, "odb", None)
        except Exception as exc:
            x, y = self._stage[:2]
            self._stage = (x, y, "error", f"reading the XY table failed ({type(exc).__name__}: "
                           f"{exc}); the last position ({x:g}, {y:g}) mm is kept")
            raise

    def set_stage(self, xpos: float, ypos: float, source: str = "manual",
                  note: str | None = None) -> None:
        """Set the XY table's position without the ODB (the offline CLI: the file's
        begin-of-run ODB or ``--stage``; tests)."""
        self._stage = (float(xpos), float(ypos), source, note)

    def _xy_shift(self) -> tuple[float, float]:
        """The (dx, dy) in mm added to every MuPix position."""
        if not self.cfg.xy.apply_stage:
            return 0.0, 0.0
        return X.stage_shift(self._stage[0], self._stage[1])

    def _new_epoch(self) -> None:
        self.epoch += 1
        self._mp_low_since = None
        self._mp_flagged = False
        self._nim_last = {}
        self._nim_mem = {}
        self._roll(force=True)

    # -- histograms ------------------------------------------------------------

    def _h1(self, name, axis, title, dtype=np.uint64):
        return self.store.add(Hist1D(f"sma/{name}", axis, title=title, dtype=dtype))

    def _h2(self, name, x, y, title, dtype=np.uint64):
        return self.store.add(Hist2D(f"sma/{name}", x, y, title=title, dtype=dtype))

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
        self._build_mupix()
        self._build_nim()

    def _build_nim(self) -> None:
        """The TOT + NIM histograms, per counter with a NIM copy (see _fill_nim)."""
        cfg, h = self.cfg, self.h
        nim = cfg.nim
        h["nim"] = {}
        if not nim.active:
            return
        labels = self.labels()
        w = int(math.ceil(nim.cfg.pair_window_ns))
        H = N.HALF_FINE_WRAP_NS
        for k, c, n in nim.pairs(cfg.roles.counters):
            name = f"S{k + 1}"
            what = f"{name} (TOT ch {c}, NIM ch {n})"
            h["nim"][k] = {
                "dt": self._h1(f"nim_dt_{name}", Axis(401, -200.5, 200.5, "t'_NIM - t'_TOT (ns)"),
                               f"{what}: NIM minus the nearest TOT word, every NIM word "
                               f"(offset {nim.offsets[k]} ns)"),
                "wide": self._h1(f"nim_dt_wide_{name}", Axis(4096, -H, H, "t_NIM - t_TOT (ns)"),
                                 f"{what}: NIM minus TOT within +-2^19 ns, raw times, sampled"),
                "walk": self._h2(f"nim_walk_{name}", Axis(2 * w + 1, -w - 0.5, w + 0.5,
                                                          "t'_NIM - t'_TOT (ns)"),
                                 Axis(256, 0, 256, "TOT ToT code"),
                                 f"{what}: pairs, NIM minus TOT vs the TOT word's ToT"),
                "classes": self._h1(f"nim_classes_{name}",
                                    Axis(7, -0.5, 6.5,
                                         "0 pair 1 TOT 2 NIM 3 echo 4 held 5 shadow 6 multi"),
                                    f"{what}: hits by class (entries = hits; 4-6 are "
                                    "sub-counts: lag-held, in TOT shadow, multi-candidate)"),
                "width": self._h1(f"nim_width_{name}", Axis(256, 0, 256, "NIM ToT code"),
                                  f"{labels[n]} (ch {n}): NIM word width"),
                "candidates": self._h1(f"nim_candidates_{name}",
                                       Axis(11, -0.5, 10.5, "NIM words in the pair window"),
                                       f"{what}: NIM candidates per TOT word "
                                       f"(+-{nim.cfg.pair_window_ns} ns)"),
                "lag": self._h1(f"nim_lag_{name}",
                                Axis(1024, 0, W.FINE_WRAP_NS, "(fine - fine(S1)) mod 2^20 (ns)"),
                                f"{labels[n]} (ch {n}) fine minus the reference S1 word's "
                                f"(lag vote; nominal {nim.nominal[k]} ns)"),
            }
        cax = self.h["s1_coinc"].x
        h["s1_coinc_tot"] = self._h1("s1_coinc_tot", Axis(cax.n, cax.lo, cax.hi, cax.title),
                                     "S1 hits with the counter in the window, TOT words only "
                                     "(entries = S1 hits)")

    def _build_mupix(self) -> None:
        """The MuPix histograms (good frames; see _fill_mupix)."""
        cfg, b, h = self.cfg, self.cfg.binning, self.h
        m = cfg.mupix
        lo, hi, w = b["mupix dt min"], b["mupix dt max"], b["mupix dt bin"]
        nb = max(1, -(-(hi - lo) // w))
        dax = Axis(nb, lo, lo + nb * w, "t(pixel) - t(S1) (ns)")
        (wlo, whi), (slo, shi) = m.window_ns, m.sideband_ns
        chips = {W.PLANE_L1: m.l1, W.PLANE_L2: m.l2}
        h["mupix_dt"] = {
            p: self._h1(f"mupix_dt_{W.PLANE_NAMES[p]}", dax,
                        f"MuPix {W.PLANE_NAMES[p]} (chips {','.join(map(str, chips[p]))}) minus "
                        f"S1, all pairs; in time [{wlo}, {whi}) ns, sideband [{slo}, {shi}) ns")
            for p in (W.PLANE_L1, W.PLANE_L2)}
        h["mupix_s1_match"] = self._h1(
            "mupix_s1_match", Axis(7, -0.5, 6.5, "0 S1 judged, 1 L1 in time, 2 L2, 3 L1+L2, "
                                   "4 L1 sideband, 5 L2 sideband, 6 L1+L2 sideband"),
            "S1 hits with a MuPix hit in time and in the sideband (entries = S1 hits judged)")
        tax = Axis(32, -0.5, 31.5, f"pixel ToT ({m.tot_ns} ns counts)")
        h["mupix_tot"] = {p: self._h1(f"mupix_tot_{W.PLANE_NAMES[p]}", tax,
                                      f"Pixel ToT, {W.PLANE_NAMES[p]}, all pixel hits")
                          for p in (W.PLANE_L1, W.PLANE_L2)}
        cax = Axis(W.N_CHIP_IDS, -0.5, W.N_CHIP_IDS - 0.5, "chip id")
        h["mupix_hits_chip"] = self._h2(
            "mupix_hits_chip", cax, Axis(100, 0, b["mupix chip max"], "pixel hits per frame"),
            "Pixel hits per frame, per chip")
        h["mupix_col_chip"] = self._h2("mupix_col_chip", cax, Axis(256, -0.5, 255.5, "column"),
                                       "Column occupancy per chip, all pixel hits")
        h["mupix_row_chip"] = self._h2("mupix_row_chip", cax, Axis(256, -0.5, 255.5, "row"),
                                       f"Row occupancy per chip (rows >= {W.PIXEL_ROWS} are not on "
                                       "the sensor)")
        self._build_xy()

    #: The x/y maps of each class (``_build_xy``); the ToT cuts re-book only these.
    _XY_CLASS_KEYS = tuple(f"{k}{c}" for c in ("_light", "_heavy") for k in ("xy", "xxp", "yyp"))

    def _build_xy(self, classes_only: bool = False) -> None:
        """MuPix x/y (MuPix/XY; see _fill_xy): booked only while x/y is active.

        Removes and re-books the x/y maps (only the light and heavy ones with
        ``classes_only``): a re-book is how an XY edit resets them without
        touching any other plot (apply_settings).

        Fixed axes, the nearline's: x, y in 130 bins of 0.64 mm (8 pixels) over
        +-41.6 mm, shifted by a quarter pixel so that no pixel centre sits on an
        edge (track_xy_expanded at half its 260 bins); slopes one bin per 2.667
        mrad step (one pixel over 30 mm) on +-102.67 mrad (xxp_central), with
        its half-step edge bias (sma_mupix_xy.SLOPE_BINS). 150850 bins in all,
        booked uint32 so that a page refresh carries 4 bytes a bin (600 kB, not
        1.2 MB); a map is widened to uint64 before any bin could pass 2^32 - 1
        (`_widen`), so a count never wraps.
        """
        cfg, h = self.cfg, self.h
        xy, m = cfg.xy, cfg.mupix
        old = h.get("xy") or {}
        drop = self._XY_CLASS_KEYS if classes_only else tuple(old)
        for k in drop:
            if k in old:
                self.store.remove(old[k].name)
        if not xy.active:
            h["xy"] = None
            return
        out = {k: v for k, v in old.items() if k not in drop}
        g = f"{X.GEOMETRY_TAG}, +x beam-left"
        pos = lambda u: Axis(X.POS_BINS, X.POS_LO_MM, X.POS_HI_MM, f"{u} (mm)")  # noqa: E731
        slope = lambda u: Axis(X.SLOPE_BINS, -X.SLOPE_HALF_MRAD, X.SLOPE_HALF_MRAD,  # noqa: E731
                               f"{u} (mrad)")
        h2 = lambda *a: self._h2(*a, dtype=np.uint32)  # noqa: E731
        win = f"[{m.window_ns[0]}, {m.window_ns[1]}) ns"
        cls = {"": "all tracks",
               "_light": f"light: max pixel ToT <= {xy.tot_light_max} in L1 and L2",
               "_heavy": f"heavy: max pixel ToT >= {xy.tot_heavy_min} in L1 and L2"}
        if not classes_only:
            for p in (W.PLANE_L1, W.PLANE_L2):
                n = W.PLANE_NAMES[p]
                out[f"hits_{n}"] = h2(
                    f"mupix_hits_xy_{n}", pos("x"), pos("y"),
                    f"MuPix {n} pixel hits in time {win} with a sampled S1 hit ({g})")
        for suf, what in cls.items():
            if classes_only and not suf:
                continue
            out[f"xy{suf}"] = h2(f"mupix_track_xy{suf}", pos("x"), pos("y"),
                                 f"S1-seeded MuPix tracks at L1, {what} ({g})")
            out[f"xxp{suf}"] = h2(f"mupix_track_xxp{suf}", pos("x"), slope("x'"),
                                  f"S1-seeded MuPix tracks x / x' at L1, {what} ({g})")
            out[f"yyp{suf}"] = h2(f"mupix_track_yyp{suf}", pos("y"), slope("y'"),
                                  f"S1-seeded MuPix tracks y / y' at L1, {what} ({g})")
        if not classes_only:
            tax = lambda n: Axis(32, -0.5, 31.5,  # noqa: E731
                                 f"max pixel ToT {n} ({m.tot_ns} ns counts)")
            out["tot"] = h2("mupix_track_tot", tax("L1"), tax("L2"),
                            "S1-seeded MuPix tracks: max pixel ToT, L1 vs L2")
            out["state"] = self._h1(
                "mupix_track_state", Axis(4, -0.5, 3.5, "0 no L1, 1 no L2, 2 ambiguous, 3 track"),
                f"Sampled S1 hits by MuPix track state ({xy.box} x {xy.box} pixel cluster "
                "square; entries = S1 hits judged)", dtype=np.uint32)
        h["xy"] = out

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
                             rescue_min_fraction=c.rescue_min_fraction, mupix=cfg.mupix)
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
            # TOT + NIM: pair every word, then (merging) the counters are the
            # merged hits from here on -- before the S1 sample is drawn.
            nf = self._pair(fr, update=True)
            if nf is not None:
                self._fill_nim(fr, nf, sec, now)
            # The all-S1 shift scan was done above to classify the frame.
            an = W.analyse_frame(fr, cfg.roles, cfg.cuts, shift_counts=scan_all)
            sampled = self._fill_analysis(fr, an, sec)
            self._fill_mupix(fr, sampled.t_s1, sec)

        hdr = event.header
        snap = _Snapshot(seq=self.frames, run=int(self.run_number or 0), serial=serial,
                         fr=fr, an=an, cls=cls, reason=reason, gap_ns=gap, shift=cfg.shift,
                         tot_min=cfg.cuts.tot_corrupt_min, pre_ns=cfg.cuts.seed_pre_ns,
                         post_ns=cfg.cuts.seed_post_ns,
                         event_id=int(getattr(hdr, "event_id", W.EVID_READOUT)),
                         timestamp=int(getattr(hdr, "timestamp", 0) or 0),
                         trigger_mask=int(getattr(hdr, "trigger_mask", 0) or 0),
                         nim=self._nim_view())
        snap.at = now
        self._keep_raw(snap, event, data)
        self._last = snap
        if cls == "good":
            self.frames_good += 1
            snap.good_n = self.frames_good
            self._last_good = snap
            self._last_good_at = now
            if an.seeds.size:
                self._last_seeded = snap
                self._noseed_since = None
            else:
                if self._noseed_since is None:
                    self._noseed_since = now
                n1 = an.n_s1_kept
                self._noseed_reason = (f"no {self.labels()[cfg.roles.s1]} hits" if n1 == 0 else
                                       f"none of {n1} {self.labels()[cfg.roles.s1]} hits has a "
                                       "complete window")
            self._push_ring(snap)
        if int(now) != self._mp_eval_t:
            self._mupix_sync(now)
        live = self._held_seqs()
        for key in [k for k in self._frame_cache if k[0] not in live]:
            del self._frame_cache[key]
        for key in [k for k in self._sel_cache if k[0] not in live]:
            del self._sel_cache[key]
        return True

    def _held_seqs(self) -> set:
        seeded = self._seeded_snap()
        return ({self._last.seq if self._last else None, seeded.seq if seeded else None}
                | set(self._ring_sizes))

    def _push_ring(self, snap: _Snapshot) -> None:
        """Keep a good frame for the seed choices, within Sampling/seed ring frames and MB."""
        size = snapshot_bytes(snap)
        self._ring.append(snap)
        self._ring_sizes[snap.seq] = size
        self._ring_bytes += size
        cfg = self.cfg
        while len(self._ring) > 1 and (len(self._ring) > cfg.seed_ring_frames
                                       or self._ring_bytes > cfg.seed_ring_bytes):
            old = self._ring.popleft()
            self._ring_bytes -= self._ring_sizes.pop(old.seq, 0)

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
        for snap in (self._last, self._last_good, self._last_seeded, *self._ring):
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
                             rescue_min_fraction=c.rescue_min_fraction, mupix=cfg.mupix)
        n_s1, scan_counts = kept_s1_scan(fr, cfg)
        s1_all = fr.ch == cfg.roles.s1
        scan_all = W.shift_scan(fr.coarse[s1_all], fr.fine[s1_all], cfg.scan, c.latch_margin_ns)
        cls, reason, _best = classify_frame(fr, cfg, n_s1, scan_counts,
                                            int(np.count_nonzero(s1_all)), scan_all)
        an = None
        if cls == "good":
            self._pair(fr)
            an = W.analyse_frame(fr, cfg.roles, cfg.cuts, shift_counts=scan_all)
        return _Snapshot(seq=entry.seq, run=entry.run, serial=entry.serial, fr=fr, an=an,
                         cls=cls, reason=reason, gap_ns=None, shift=cfg.shift,
                         tot_min=c.tot_corrupt_min, pre_ns=c.seed_pre_ns, post_ns=c.seed_post_ns,
                         event_id=eid, timestamp=ts, trigger_mask=tmask, nim=self._nim_view())

    def _nim_view(self) -> dict | None:
        """`_Snapshot.nim` under the current settings (a frame is paired with them now)."""
        cfg = self.cfg
        if not cfg.nim.active:
            return None
        r = cfg.roles
        return {"roles": {"s1": r.s1, "counters": list(r.counters), "rf": r.rf,
                          "current": r.current, "delayed": list(r.delayed),
                          "nim": list(cfg.nim.channels)},
                "nim_merge": bool(cfg.merging)}

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

    def _pair(self, fr: W.Frame, update: bool = False) -> NimFrame | None:
        """`pair_frame` with the epoch's lag memory (advanced only with
        ``update``: a frame analysed as it arrives, not one rebuilt for a page),
        and its result put on the frame: the merged counter hits
        (``Frame.counter_hits``) and the per-hit pairing (``Frame.pairing``)."""
        nf = pair_frame(fr, self.cfg, self._nim_mem, update=update)
        if nf is not None:
            fr.counter_hits = nf.hits or None
            fr.pairing = nf.words
        return nf

    def _fill_nim(self, fr: W.Frame, nf: NimFrame, sec: _Second, now: float) -> None:
        """The TOT + NIM histograms and counts of a good frame (every word)."""
        hs = self.h["nim"]
        budget = self.cfg.nim.wide_budget
        H = N.HALF_FINE_WRAP_NS
        for k, pr in nf.results.items():
            hh = hs.get(k)
            if hh is None:
                continue
            vote, held, state = nf.votes[k], nf.held[k], nf.state[k]
            cnt = pr.counts()
            n_held = cnt["nim_only"] if held else 0
            row = sec.nim[k]
            for name in ("tot_words", "nim_words", "paired", "tot_only", "nim_only", "echo",
                         "shadow", "multi"):
                row[_NC[name]] += cnt[name]
            row[_NC["lag_held"]] += n_held
            row[_NC["frames"]] += 1
            if vote is None:
                row[_NC["lag_skipped"]] += 1
            else:
                row[_NC[f"lag_{vote.state}"]] += 1
                if vote.lag_ns is not None:
                    self._nim_last[k] = (now, vote.lag_ns, vote.state)
            if state is not None:
                row[_NC[f"state_{state}"]] += 1

            classes = np.array([0, cnt["paired"], cnt["tot_only"], cnt["nim_only"], cnt["echo"],
                                n_held, cnt["shadow"], cnt["multi"], 0], dtype=np.int64)
            hh["classes"].add_counts(classes, entries=int(classes[1:5].sum()))
            # Whole-ns times (the offsets are rounded): integer bins, the
            # values' own (bin centres on the integers).
            if pr.nim_dt.size and pr.t_tot.size:
                _fill_1d_index(hh["dt"], pr.nim_dt.astype(np.intp) + (hh["dt"].x.n // 2 + 1))
            if pr.tot_nim.size:
                wc = np.bincount(pr.tot_nim.astype(np.intp), minlength=256)[:256]
                hh["width"].add_counts(_with_flow(wc), entries=int(pr.tot_nim.size))
            live = pr.n_cand_tot[~pr.echo]
            if live.size:
                _fill_1d_index(hh["candidates"], live.astype(np.intp) + 1)
            dtp = pr.pair_dt
            if dtp.size:
                hw = hh["walk"]
                _fill_2d_index(hw, dtp.astype(np.intp) + (hw.x.n // 2 + 1),
                               pr.pair_tot.astype(np.intp) + 1)
            # The lag vote's input, from the voted frames only (NIM/lag vote
            # every): the shape is the same, the entries fewer.
            if vote is not None and vote.d.size:
                _fill_1d_index(hh["lag"], (vote.d >> 10).astype(np.intp) + 1)
            if budget > 0:
                c, n = self.cfg.roles.counters[k], self.cfg.nim.channels[k]
                wd = N.wide_dt(fr.times(c), fr.times(n), fr.first, fr.last, budget)
                if wd.size:
                    _fill_1d_index(hh["wide"], ((wd + H) >> 8) + 1)

    def _fill_analysis(self, fr: W.Frame, an: W.FrameAnalysis,
                       sec: _Second) -> W.FrameAnalysis:
        """The S1-seeded fills; returns the rows filled (the S1 sample)."""
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
            return an

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
        if "s1_coinc_tot" in h:
            self._fill_coinc_tot(fr, cc, n1)
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
            # A NIM-only S1 hit (merged from S1L) has no ToT of its own: it is
            # left out of the ToT-binned plot rather than drawing a stripe at
            # NIM/nim only tot.
            tv = v
            s1c = self.cfg.roles.s1
            if fr.counter_hits and s1c in fr.counter_hits:
                rows = (np.arange(n1) if an.s1_rows is None else an.s1_rows)
                carrier = fr.counter_idx(s1c)[rows]
                tv = v & (fr.s_ch[carrier] == s1c)
            _fill_2d(h["rf_phase_vs_s1_tot"], an.rf_phase[tv], an.s1_tot[tv].astype(np.float64))
        for ch, (_i, dt) in an.delayed_dt.items():
            hh = h["delayed"].get(ch)
            if hh is not None and dt.size:
                _fill_values(hh, dt * 1e-3)
        return an

    def _fill_coinc_tot(self, fr: W.Frame, merged_cc, n_merged: int) -> None:
        """``s1_coinc_tot``: ``s1_coinc`` on the TOT words alone, S1 sampled the
        same way (the merged counts as they are when nothing is merged)."""
        h = self.h["s1_coinc_tot"]
        if not fr.counter_hits:
            h.add_counts(merged_cc, entries=n_merged)
            return
        cfg = self.cfg
        t1 = fr.times(cfg.roles.s1)
        m = cfg.cuts.max_s1
        if m is not None and t1.size > m:
            t1 = t1[W.even_sample(t1.size, int(m))]
        if not t1.size:
            return
        _pat, counts = W.coincidence(t1, [fr.times(c) for c in cfg.roles.counters],
                                     cfg.cuts.coinc_ns)
        nc = counts.shape[1]
        cc = np.zeros(h.counts.shape, dtype=np.int64)
        cc[1:nc + 1] = np.count_nonzero(counts > 0, axis=0)
        h.add_counts(cc, entries=int(t1.size))

    def _fill_mupix(self, fr: W.Frame, t_s1, sec: _Second) -> None:
        """MuPix of a good frame: occupancy, ToT, and the S1 matching.

        Occupancy and ToT use every examined pixel hit (noise included). The
        matching uses the S1 rows the S1-seeded analyses use (the sample under
        Cuts/max S1 per frame), at most MuPix/max S1 per frame of them, evenly
        spread (sma_words.even_sample): which have an L1 / L2 hit in the in-time window
        and in the sideband (only S1 hits whose windows lie in the pixel data
        are judged, sma_words.mupix_match), and every t(pixel) - t(S1) pair in
        the histogram range (bounded by MuPixCuts.max_pairs).
        """
        px = fr.px
        if px is None:
            return
        h, m = self.h, self.cfg.mupix
        sec.mp_frames += 1
        sec.mp_pix += px.n_words
        sec.mp_examined += px.n
        sec.mp_skipped += px.n_skipped
        sec.mp_unsorted += 0 if px.was_sorted else 1
        chip = px.chip.astype(np.intp)
        per_chip = np.bincount(chip, minlength=W.N_CHIP_IDS)[:W.N_CHIP_IDS]
        sec.mp_chip += per_chip
        sec.mp_rows += int(np.count_nonzero(px.row >= W.PIXEL_ROWS))
        # Hits per frame of every chip that has hits or a plane (a dead chip
        # of the map fills its 0 bin).
        show = np.flatnonzero((per_chip > 0) | (m.planes > 0))
        hc = h["mupix_hits_chip"]
        _fill_2d_index(hc, show + 1, (per_chip[show] * hc.y.n) // int(hc.y.hi) + 1)
        if px.n:
            _fill_2d_index(h["mupix_col_chip"], chip + 1, px.col.astype(np.intp) + 1)
            _fill_2d_index(h["mupix_row_chip"], chip + 1, px.row.astype(np.intp) + 1)
            tot = np.bincount(px.plane.astype(np.intp) * 32 + px.tot.astype(np.intp),
                              minlength=96)
            for p in (W.PLANE_L1, W.PLANE_L2):
                c = np.zeros(34, dtype=np.int64)
                c[1:33] = tot[32 * p: 32 * p + 32]
                h["mupix_tot"][p].add_counts(c, entries=int(c.sum()))
        t_s1 = np.asarray(t_s1, dtype=np.int64)
        if not t_s1.size or not px.n:
            return
        if m.max_s1 is not None and t_s1.size > m.max_s1:
            t_s1 = t_s1[W.even_sample(t_s1.size, m.max_s1)]
        self._fill_xy(t_s1, px, sec)
        mt = W.mupix_match(t_s1, px, m.window_ns, m.sideband_ns)
        n, fin, fside = mt.counts()
        sec.mp_n += n
        sec.mp_in += fin
        sec.mp_side += fside
        c = np.zeros(9, dtype=np.int64)
        c[1], c[2:5], c[5:8] = n, fin, fside
        h["mupix_s1_match"].add_counts(c, entries=n)
        b = self.cfg.binning
        for p in (W.PLANE_L1, W.PLANE_L2):
            dt, _used = W.mupix_pairs(t_s1, px.times(p), b["mupix dt min"], b["mupix dt max"],
                                      m.max_pairs)
            if dt.size:
                hh = h["mupix_dt"][p]
                _fill_1d_index(hh, (dt - int(hh.x.lo)) // b["mupix dt bin"] + 1)

    def _fill_xy(self, t_s1, px: W.Pixels, sec: _Second) -> None:
        """MuPix x/y of a good frame, on the MuPix S1 sample.

        Of that sample, the S1 hits whose window lies in the frame's pixel data
        (the rest cannot be judged: on run 1008 the pixels start ~0.5 ms after
        the SMA hits), at most MuPix/XY/max S1 per frame of them, evenly spread.
        sma_mupix_xy.s1_tracks: per S1 hit the track state, and for the tracks
        the L1 position, the slopes and the max pixel ToT of each plane. The
        hit maps take every pixel in at least one window, once. The shift of
        the XY table (as of the last poll) is added to every position.
        """
        hx = self.h.get("xy")
        if hx is None:
            return
        xy, m = self.cfg.xy, self.cfg.mupix
        t_s1 = t_s1[W.mupix_coverage(t_s1, px, *m.window_ns)]
        if t_s1.size > xy.max_s1:
            t_s1 = t_s1[W.even_sample(t_s1.size, xy.max_s1)]
        tr = X.s1_tracks(t_s1, px, m.window_ns, xy.box, xy.placement, self._xy_shift())
        sc = tr.state_counts()
        sec.xy_state += sc
        n = int(sc.sum())
        if n:
            _widen(hx["state"], n)
            hx["state"].add_counts(np.array([0, *sc, 0], dtype=np.int64), entries=n)
        # Few entries into large maps: in-place adds of flat bin indices
        # (_add_flat), each index computed once and reused by the light and
        # heavy maps, which take a subset of the tracks.
        pax = hx["xy"].x
        nxp = pax.n + 2
        for p in (W.PLANE_L1, W.PLANE_L2):
            u, v = tr.hits[p]
            if u.size:
                _add_flat(hx[f"hits_{W.PLANE_NAMES[p]}"], _bin_of(v, pax) * nxp + _bin_of(u, pax))
        trk = tr.track
        if not trk.any():
            return
        sax = hx["xxp"].y
        ix, iy = _bin_of(tr.x1[trk], pax), _bin_of(tr.y1[trk], pax)
        flat = {"xy": iy * nxp + ix, "xxp": _bin_of(tr.xp[trk], sax) * nxp + ix,
                "yyp": _bin_of(tr.yp[trk], sax) * nxp + iy}
        t1, t2 = tr.tot1[trk], tr.tot2[trk]
        light = (t1 <= xy.tot_light_max) & (t2 <= xy.tot_light_max)
        heavy = (t1 >= xy.tot_heavy_min) & (t2 >= xy.tot_heavy_min)
        nl, nh = int(np.count_nonzero(light)), int(np.count_nonzero(heavy))
        sec.xy_light += nl
        sec.xy_heavy += nh
        sec.xy_ctrk += int(t1.size)
        for k, f in flat.items():
            _add_flat(hx[k], f)
            if nl:
                _add_flat(hx[f"{k}_light"], f[light])
            if nh:
                _add_flat(hx[f"{k}_heavy"], f[heavy])
        # ToT 0..31 on a 32-bin axis from -0.5: bin index = ToT + 1, never out of range.
        _add_flat(hx["tot"], (t2.astype(np.intp) + 1) * (hx["tot"].x.n + 2) + t1 + 1)

    def _mupix_widths(self) -> tuple[float, float]:
        m = self.cfg.mupix
        return (float(m.window_ns[1] - m.window_ns[0]), float(m.sideband_ns[1] - m.sideband_ns[0]))

    def _mupix_sync(self, now: float) -> dict:
        """The SMA <-> MuPix time-sync monitor, evaluated once per second.

        Over the last ``mupix sync window s``: with S1 firing (at least ``mupix
        sync min S1`` analysed S1 hits) the share of judged S1 hits with an L1
        and an L2 hit in time, accidentals taken out, is compared with ``mupix
        sync min fraction``. Below it for more than ``mupix sync hold s``: the
        flag ``mupix_sync`` is raised, and it clears once the share is above
        min fraction + ``mupix sync clear margin`` (hysteresis). No pixel words
        at all while S1 fires counts as 0. Too few S1 hits (beam off) or too
        few judged ones: no verdict, and the flag and its timer are cleared.
        """
        chk = self.cfg.check
        self._mp_eval_t = int(now)
        secs = self._window(chk["mupix sync window s"], now)
        n_s1 = sum(x.n_s1 for x in secs)
        n = sum(x.mp_n for x in secs)
        pix = sum(x.mp_pix for x in secs)
        both = sum(int(x.mp_in[2]) for x in secs)
        side = sum(int(x.mp_side[2]) for x in secs)
        thr = chk["mupix sync min fraction"]
        out = {"state": "insufficient", "value": None, "n_s1": n, "pixels": pix,
               "threshold": _num(thr, 4), "window_s": chk["mupix sync window s"],
               "hold_s": chk["mupix sync hold s"]}
        if not self.cfg.mupix.enabled:
            out["state"] = "off"
            f = None
        elif n_s1 < chk["mupix sync min S1"]:
            f = None
        elif pix == 0:
            f = 0.0
            out["absent"] = True
        elif n < chk["mupix sync min S1"]:
            f = None
        else:
            f = W.accidental_corrected(both / n, side / n, *self._mupix_widths())
        if f is None:
            self._mp_low_since = None
            self._mp_flagged = False
        else:
            out["value"] = _num(f, 4)
            if f < thr:
                if self._mp_low_since is None:
                    self._mp_low_since = now
                if now - self._mp_low_since > chk["mupix sync hold s"]:
                    self._mp_flagged = True
            elif f >= thr + chk["mupix sync clear margin"]:
                self._mp_low_since = None
                self._mp_flagged = False
            elif not self._mp_flagged:
                self._mp_low_since = None
            out["state"] = ("flagged" if self._mp_flagged else
                            "low" if self._mp_low_since is not None else "ok")
        out["low_for_s"] = (None if self._mp_low_since is None
                            else _num(now - self._mp_low_since, 4))
        self._mp_last = out
        return out

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
        if r.current >= 0:
            names[r.current] = "current"
        names[r.rf] = "RF"
        for k, _c, n in self.cfg.nim.pairs(r.counters):
            names[n] = f"S{k + 1}L"
        for k, c in enumerate(r.counters):
            names[c] = f"S{k + 1}"
        return [self.cfg.labels[c] or names.get(c, f"ch{c:02d}") for c in range(NCH)]

    def _pair_of(self, c: int) -> int | None:
        """A NIM channel's counter channel, a counter's NIM channel; None otherwise."""
        for _k, ct, n in self.cfg.nim.pairs(self.cfg.roles.counters):
            if c == n:
                return ct
            if c == ct:
                return n
        return None

    def _role_of(self, c: int) -> str:
        r = self.cfg.roles
        if c in r.counters:
            return "s1" if c == r.s1 else "counter"
        if c == r.rf:
            return "rf"
        if c >= 0 and c in self.cfg.nim.channels:
            return "nim"
        if c == r.current and c >= 0:
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
            #: A NIM row's counter channel, a counter's NIM channel, else None.
            "pair_of": self._pair_of(c),
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
            #: Counters with a known fine/coarse fault (the ``mismatch`` flags'
            #: rule); SMAEvents' "incomplete pattern" leaves them out.
            "timestamp_faults": self._timestamp_faults(hits, mism, shift["verdict"]),
            "mupix": self._mupix_summary(secs, now),
            "xy": self._xy_summary(secs),
            #: The channel map, so the pages need not guess roles from labels;
            #: "nim" follows "counters" (-1 = no NIM copy).
            "roles": {"s1": cfg.roles.s1, "counters": list(cfg.roles.counters),
                      "rf": cfg.roles.rf, "current": cfg.roles.current,
                      "delayed": list(cfg.roles.delayed),
                      "nim": list(cfg.nim.channels) or [-1] * nc},
            "settings_errors": list(cfg.errors),
        }
        nim = self._nim_summary(secs, now, hits)
        out["nim"] = nim
        #: Whether the counters are the merged TOT + NIM hits (pattern,
        #: efficiencies, seeds), and the NIM channels whose NIM-only hits a
        #: lag fault held back from the merge in the window.
        out["nim_merge"] = cfg.merging
        out["nim_lag_held"] = [r["nim_ch"] for r in nim["counters"] if r["lag_held"]]
        out["flags"] = self._flags(out, now, run_active)
        return out

    def _nim_summary(self, secs: list, now: float, hits) -> dict:
        """The TOT + NIM part of sma::summary, per counter with a NIM copy, over
        the summary window -- except the median dt and its entries, which are
        read from ``nim_dt_Sk`` and so run since the last rebuild or run start.

        Per counter: word and hit counts (`NIM_COLS`; ``nim_only`` counts every
        unpaired NIM word, ``lag_held`` those of them held back from the
        merge), ``pair_eff`` = paired / (paired + TOT-only; echo words are
        left out of the denominator, as reco's pair fraction does and smanim's
        ``tot_with_nim`` = pairs / all TOT words does not), ``purity`` =
        paired / NIM words, ``nim_only_frac`` = NIM-only / merged hits (paired
        + TOT-only + echo + NIM-only: what the merge adds), ``median_dt_ns``
        and ``dt_entries``, ``lag``: frames per vote state (``skipped``: not
        voted, NIM/lag vote every), the faulted share of the frames with a
        decisive vote, frames per lag state (``state_*``: the frame's own
        decisive vote, else the epoch's last) and its faulted share, the
        epoch's lag state now and its decisive votes, and the last voted lag.
        """
        cfg = self.cfg
        nim = cfg.nim
        nc = len(cfg.roles.counters)
        tab = sum((x.nim for x in secs if x.nim.shape == (nc, len(NIM_COLS))),
                  np.zeros((nc, len(NIM_COLS)), dtype=np.int64))
        labels = self.labels()
        rows = []
        for k, c, n in nim.pairs(cfg.roles.counters):
            a = {name: int(tab[k, j]) for j, name in enumerate(NIM_COLS)}
            merged = a["paired"] + a["tot_only"] + a["echo"] + a["nim_only"]
            hd = self.h["nim"].get(k, {}).get("dt")
            med, n_dt = _hist_median(hd) if hd is not None else (None, 0)
            voted = a["lag_ok"] + a["lag_faulted"]
            stated = a["state_ok"] + a["state_faulted"]
            last = self._nim_last.get(k)
            mem = self._nim_mem.get(k) or LagMemory()
            rows.append({
                "counter": f"S{k + 1}", "k": k + 1, "ch": c, "nim_ch": n,
                "label": labels[c], "nim_label": labels[n],
                "tot_hits": int(hits[c]), "nim_hits": int(hits[n]),
                **{name: a[name] for name in NIM_COLS[:10]},
                "pair_eff": _ratio(a["paired"], a["paired"] + a["tot_only"], 4),
                "purity": _ratio(a["paired"], a["nim_words"], 4),
                "nim_only_frac": _ratio(a["nim_only"], merged, 4),
                "median_dt_ns": _num(med, 4), "dt_entries": n_dt,
                "offset_ns": nim.offsets[k], "echo_rule": bool(nim.echo[k]),
                "lag": {**{st: a[f"lag_{st}"] for st in N.LAG_STATES},
                        "skipped": a["lag_skipped"],
                        "voted": voted, "faulted_frac": _ratio(a["lag_faulted"], voted, 4),
                        "state_ok": a["state_ok"], "state_faulted": a["state_faulted"],
                        "state_faulted_frac": _ratio(a["state_faulted"], stated, 4),
                        "state": mem.state, "epoch_votes": mem.votes,
                        "epoch_faulted_votes": mem.faulted_votes,
                        "nominal_ns": nim.nominal[k],
                        "last_ns": None if last is None else int(last[1]),
                        "last_state": None if last is None else last[2],
                        "last_age_s": None if last is None else _num(now - last[0], 4)},
            })
        return {"active": nim.active, "merge": nim.merge, "merging": cfg.merging,
                "merge_when_lagged": nim.merge_when_lagged, "lag_vote_every": nim.lag_every,
                "pair_window_ns": nim.cfg.pair_window_ns, "time_source": nim.cfg.time_source,
                "counters": rows}

    def _timestamp_faults(self, hits, mism, verdict: str) -> dict:
        """The counters (S1..S5) with a known timestamp fault, by the rule the
        summary's per-channel ``mismatch`` flags use (`_flags`): a flagged
        channel with at least ``min hits`` hits whose fine/coarse mismatch
        fraction over the summary window is above ``mismatch warn fraction``,
        and only while the shift verdict is ok (otherwise every channel's
        mismatch describes the setting, not the board: nothing is judged).

        ``{"verdict", "judged", "counters": [{"counter": k (1 = S1), "ch",
        "label", "mismatch_frac"}]}``.
        """
        chk, cfg = self.cfg.check, self.cfg
        labels = self.labels()
        out = {"verdict": verdict, "judged": verdict == "ok", "counters": []}
        if verdict != "ok":
            return out
        for k, c in enumerate(cfg.roles.counters):
            m = _ratio(mism[c], hits[c], 4)
            if (c in cfg.flag_channels and hits[c] >= chk["min hits"] and m is not None
                    and m > chk["mismatch warn fraction"]):
                out["counters"].append({"counter": k + 1, "ch": int(c), "label": labels[c],
                                        "mismatch_frac": m})
        return out

    def timestamp_faults(self, now: float | None = None) -> dict:
        """`_timestamp_faults` now, over the summary window (what ``sma::summary``
        says as ``timestamp_faults``), without building the rest of the summary."""
        now = self._clock() if now is None else now
        secs = self._window(self.cfg.check["summary window s"], now)

        def vec(attr):
            return sum((getattr(s, attr) for s in secs if len(getattr(s, attr)) == NCH),
                       np.zeros(NCH, dtype=np.int64))

        return self._timestamp_faults(vec("hits"), vec("mismatch"),
                                      self.shift_verdict(now)["verdict"])

    def incomplete_rule(self, faults: dict) -> dict:
        """How the ``incomplete`` oddity judges seeds, given `timestamp_faults`:
        which counters it ignores (0-based ``ignore``), whether it can judge at
        all, and the words the page shows for it."""
        nc = len(self.cfg.roles.counters)
        ign = [f for f in faults["counters"] if 1 <= f["counter"] <= nc]
        judged = nc - len(ign) >= 2
        base = self.filter_label("incomplete")
        if not faults["judged"]:
            note = f"no counter ignored: shift check {faults['verdict']}"
        elif not ign:
            note = ""
        elif not judged:
            names = ", ".join(f["label"] for f in ign)
            note = (f"cannot judge: {names} {'has a' if len(ign) == 1 else 'have'} "
                    "timestamp fault" + ("" if len(ign) == 1 else "s"))
        else:
            note = "ignoring " + ", ".join(
                f"{f['label']}: timestamp fault {_pct(f['mismatch_frac'])}" for f in ign)
        return {"ignore": tuple(f["counter"] - 1 for f in ign) if faults["judged"] else (),
                "judged": judged or not faults["judged"],
                "out": {"judged": judged or not faults["judged"],
                        "verdict": faults["verdict"],
                        "ignored": [dict(f) for f in ign] if faults["judged"] else [],
                        "note": note,
                        "label": f"{base} ({note})" if note else base}}

    def _mupix_summary(self, secs: list, now: float) -> dict:
        """The MuPix part of sma::summary, over the summary window."""
        m = self.cfg.mupix
        if int(now) != self._mp_eval_t:
            self._mupix_sync(now)
        tot = lambda a: sum(getattr(x, a) for x in secs)  # noqa: E731
        n, frames = tot("mp_n"), tot("mp_frames")
        fin = sum((x.mp_in for x in secs), np.zeros(3, dtype=np.int64))
        fside = sum((x.mp_side for x in secs), np.zeros(3, dtype=np.int64))
        per_chip = sum((x.mp_chip for x in secs), np.zeros(W.N_CHIP_IDS, dtype=np.int64))
        widths = self._mupix_widths()
        fractions = {}
        for k, name in enumerate(("L1", "L2", "both")):
            a, b = _ratio(fin[k], n, 4), _ratio(fside[k], n, 4)
            fractions[name] = {"in": a, "side": b,
                               "corr": _num(W.accidental_corrected(a, b, *widths), 4)}
        planes = m.planes
        chips = [{"chip": int(c), "plane": W.PLANE_NAMES[int(planes[c])] if planes[c] else None,
                  "hits": int(per_chip[c]), "per_frame": _ratio(per_chip[c], frames, 5)}
                 for c in range(W.N_CHIP_IDS) if per_chip[c] or planes[c]]
        pix = tot("mp_pix")
        return {
            "enabled": bool(m.enabled),
            "planes": {"L1": list(m.l1), "L2": list(m.l2)},
            "window_ns": list(m.window_ns), "sideband_ns": list(m.sideband_ns),
            "ts2_shift": m.ts2_shift, "tot_ns": m.tot_ns,
            "frames": frames, "n_s1": n, "pixels": pix,
            "pixels_per_frame": _ratio(pix, frames, 5),
            "skipped_frac": _ratio(tot("mp_skipped"), pix, 4),
            "max_pixels": m.max_pixels,
            "unmapped": int(per_chip[planes == 0].sum()),
            "rows_off_sensor": tot("mp_rows"),
            "unsorted_frames": tot("mp_unsorted"),
            "chips": chips,
            "fractions": fractions,
            "sync": dict(self._mp_last),
        }

    def _xy_summary(self, secs: list) -> dict:
        """The MuPix x/y part of sma::summary, over the summary window.

        Fractions of the judged S1 hits per track state (``n_s1``: the S1 hits
        x/y judged -- window inside the pixel data, at most XY/max S1 per frame
        -- not the same count as ``mupix.n_s1``, whose S1 hits also need the
        sideband inside); ``light_frac`` and ``heavy_frac`` are of the tracks
        classed since the last ToT-cut edit (None right after one).
        ``enabled``: x/y is booked and filled; ``off_reason`` why not with
        XY/enable = y. ``quadrants``: where each placed chip is. ``stage``: the
        XY table as read (``x_mm``, ``y_mm``), where from (``source``: "odb";
        "file" -- the begin-of-run ODB of the offline CLI's file --; "manual"
        -- set by hand, ``--stage`` --; "missing" -- the key was not readable,
        0 used --; "error" -- the read failed, see the note --; "none" -- no
        reading yet), whether it is applied, the shift added to every position
        (``shift_mm``, (-x, +y)) and a ``note`` when it is not a plain reading.
        ``resets``: x/y map resets by XY edits since the start.
        """
        xy, m = self.cfg.xy, self.cfg.mupix
        state = sum((x.xy_state for x in secs), np.zeros(len(X.STATE_NAMES), dtype=np.int64))
        n, trk = int(state.sum()), int(state[X.TRACK])
        light = sum(x.xy_light for x in secs)
        heavy = sum(x.xy_heavy for x in secs)
        ctrk = sum(x.xy_ctrk for x in secs)
        sx, sy, src, note = self._stage
        dx, dy = self._xy_shift()
        return {
            "enabled": bool(xy.active),
            "off_reason": (None if xy.active else xy.off_reason if xy.enable
                           else "MuPix/XY/enable = n"),
            "geometry": X.GEOMETRY_TAG,
            "quadrants": xy.placement.quadrant_map() if xy.active else [],
            "resets": self.xy_resets,
            "n_s1": n, "tracks": trk,
            "fractions": {"track": _ratio(trk, n, 4),
                          "ambiguous": _ratio(state[X.AMBIGUOUS], n, 4),
                          "no_l1": _ratio(state[X.NO_L1], n, 4),
                          "no_l2": _ratio(state[X.NO_L2], n, 4)},
            "light_frac": _ratio(light, ctrk, 4),
            "heavy_frac": _ratio(heavy, ctrk, 4),
            "stage": {"x_mm": _num(sx, 6), "y_mm": _num(sy, 6), "source": src,
                      "applied": bool(xy.apply_stage),
                      "shift_mm": [_num(dx, 6) or 0.0, _num(dy, 6) or 0.0],
                      "note": note if xy.apply_stage else None},
            "cuts": {"cluster_box_px": xy.box, "tot_light_max": xy.tot_light_max,
                     "tot_heavy_min": xy.tot_heavy_min, "tot_ns": m.tot_ns,
                     "window_ns": list(m.window_ns), "max_s1": xy.max_s1},
            "unplaced_chips": list(xy.placement.unplaced),
        }

    def _flags(self, s: dict, now: float, run_active) -> list[dict]:
        chk = self.cfg.check
        flags = []

        def add(sev, code, text, **fields):
            flags.append({"severity": sev, "code": code, "text": text, **fields})

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
        if self._noseed_since is not None and self._last_good_at is not None:
            # Good frames keep coming but none has an S1 seed: S1 is missing
            # from the data (absent, dead, mis-cabled, or the s1 role points at
            # the wrong channel). The S1-seeded view then shows an old frame.
            quiet = now - self._noseed_since
            if quiet > chk["no seeds s"] and now - self._last_good_at <= chk["no frames s"]:
                c1 = self.cfg.roles.s1
                lab = self.labels()[c1]
                add("warn", "no_seeds",
                    f"No {lab} seed for {quiet:.0f} s although good frames arrive "
                    f"({self._noseed_reason}): {lab} (ch {c1}) is absent, dead or mis-cabled, "
                    "or /DQM/SMA/Channel roles/s1 is wrong. SMAEvents shows an older frame; "
                    "its seed choice 'any counter' shows events without it")
        self._mupix_flags(s, add)
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

        self._nim_flags(s, add, time_base_ok)
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

    def _nim_flags(self, s: dict, add, time_base_ok: bool = True) -> None:
        """nim_missing, nim_pairing, nim_offset, nim_lag, per counter with a NIM
        copy; each needs ``min hits`` (TOT hits; TOT words paired or not; dt
        entries; NIM words) in the summary window.

        Only ``nim_missing`` (it counts hits) is judged whatever the time base.
        The others compare times: like the efficiency flags they wait for the
        shift check to say ok (``time_base_ok``), and ``nim_pairing`` /
        ``nim_offset`` skip a counter whose TOT channel has a known timestamp
        fault (``timestamp_faults``). The NIM channel's own fine/coarse
        mismatch does not gate them: the lag fault is such a mismatch.
        ``nim_lag`` also needs ``nim lag min votes`` frames voted "faulted"
        since the last rebuild or run start, and judges the lag state of the
        window's frames (a quiet or unvoted frame takes the last decisive
        vote)."""
        chk = self.cfg.check
        min_hits = chk["min hits"]
        win = f"{s['window_s']:.0f} s"
        bad_tot = {f["ch"] for f in (s.get("timestamp_faults") or {}).get("counters", [])}
        flag = add
        for r in s["nim"]["counters"]:
            tot, nim = f"{r['label']} (ch {r['ch']})", f"{r['nim_label']} (ch {r['nim_ch']})"

            # The channels as fields too, so the pages need not parse the text.
            def add(sev, code, text, _chans={"ch": r["ch"], "nim_ch": r["nim_ch"]}):  # noqa: B006
                flag(sev, code, text, **_chans)

            if r["tot_hits"] >= min_hits and r["nim_hits"] == 0:
                add("warn", "nim_missing",
                    f"{nim}: no hits in the last {win} while {tot} has {r['tot_hits']}: the NIM "
                    "copy is not cabled, its discriminator is off, or /DQM/SMA/NIM/channels "
                    "is wrong")
                continue
            if not time_base_ok:
                continue
            e = r["pair_eff"]
            if r["ch"] in bad_tot:
                e = None
            if e is not None and r["paired"] + r["tot_only"] >= min_hits:
                sev = ("error" if e < chk["nim pairing error fraction"] else
                       "warn" if e < chk["nim pairing warn fraction"] else None)
                if sev:
                    add(sev, "nim_pairing",
                        f"{tot}: only {_pct(e)} of its TOT words have a NIM word ({nim}) within "
                        f"±{s['nim']['pair_window_ns']} ns (median NIM - TOT "
                        f"{r['median_dt_ns']} ns): NIM threshold, timing (NIM/offset ns) "
                        "or a lag fault")
            m = r["median_dt_ns"]
            if (m is not None and r["ch"] not in bad_tot and r["dt_entries"] >= min_hits
                    and abs(m) > chk["nim offset max ns"]):
                add("warn", "nim_offset",
                    f"{nim}: median NIM - TOT (since the run start or the last settings change) "
                    f"is {m:g} ns after the {r['offset_ns']} ns offset: "
                    f"set /DQM/SMA/NIM/offset ns[{r['k'] - 1}] = {r['offset_ns'] + round(m)}")
            lg = r["lag"]
            ff = lg["state_faulted_frac"]
            if (ff is not None and r["nim_words"] >= min_hits and ff > chk["nim lag max fraction"]
                    and lg["epoch_faulted_votes"] >= chk["nim lag min votes"]):
                held = (f"; its NIM-only hits ({r['lag_held']}) were held back from the merge"
                        if r["lag_held"] else "")
                add("warn", "nim_lag",
                    f"{nim}: fine-time lag fault in {lg['state_faulted']} of "
                    f"{lg['state_ok'] + lg['state_faulted']} frames ({lg['faulted']} of "
                    f"{lg['voted']} voted; last lag {lg['last_ns']} ns, nominal "
                    f"{lg['nominal_ns']} ns){held}. Measured only; the DQM corrects nothing")

    def _mupix_flags(self, s: dict, add) -> None:
        mp = s.get("mupix") or {}
        sy = mp.get("sync") or {}
        chk = self.cfg.check
        lab = self.labels()[self.cfg.roles.s1]
        if sy.get("state") == "flagged":
            low = sy.get("low_for_s") or 0
            if sy.get("absent"):
                add("warn", "mupix_sync",
                    f"No MuPix pixel words in the SMA frames for {low:.0f} s while {lab} fires: "
                    "the MuPix readout, its link or its chips are off (or the H000 bank no "
                    "longer carries them)")
            else:
                add("warn", "mupix_sync",
                    f"MuPix L1+L2 in time with {lab} for only {sy['value']:.0%} of {lab} hits "
                    f"(accidentals taken out) for {low:.0f} s, below "
                    f"{chk['mupix sync min fraction']:.0%}: the SMA and MuPix time bases may "
                    "have lost sync (wrong coarse shift, a time-base fault on either side), a "
                    "plane is dead, or the beam does not reach MuPix (degrader, stopping run). "
                    "The MuPix tab's t(pixel) - t(S1) peak sits near 0 when they agree")
        if mp.get("unmapped"):
            bad = [c["chip"] for c in mp.get("chips", []) if c["plane"] is None and c["hits"]]
            add("warn", "mupix_unmapped",
                f"{mp['unmapped']} pixel hits in the last {s['window_s']:.0f} s on chip(s) "
                f"{', '.join(map(str, bad))}, in neither /DQM/SMA/MuPix/L1 chips "
                f"({','.join(map(str, mp['planes']['L1']))}) nor L2 chips "
                f"({','.join(map(str, mp['planes']['L2']))}): the plane map does not match the "
                "FEB Mapping")
        sk = mp.get("skipped_frac")
        if sk:
            add("info", "mupix_skipped",
                f"MuPix: {sk:.0%} of the pixel hits were skipped (at most "
                f"{mp['max_pixels']} examined per frame, MuPix/max pixel hits per frame); "
                "fractions use only S1 hits inside the examined part")

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
        widths = self._mupix_widths()
        rows = [(v[0] if len(v) == 1 else _merge_seconds(v)).row(counters, warn, widths)
                for _t, v in sorted(by_t.items())]
        return {"t": now, "labels": self.labels(),
                # The columns of each row's "eff": the counters after S1.
                "counters": [f"S{k + 1}" for k in range(1, len(self.cfg.roles.counters))],
                # The columns of each row's "nim_eff" (TOT + NIM pair efficiency):
                # every counter, S1 first; a column is null without a NIM copy.
                "nim_counters": [f"S{k + 1}" for k in range(len(self.cfg.roles.counters))],
                "rows": rows}

    # -- event display -------------------------------------------------------------

    def frame_blob(self, view: str = "seeded", drop=(), max_hits: int | None = None,
                   words: bool | None = None, seq: int | None = None, seed=None,
                   filters=None, mupix=None, pixels: bool | None = None,
                   pattern=None) -> bytes | None:
        """A frame as an ``smaf`` payload, encoded once per (frame, view, drop, words).

        The seeded view shows the last frame that had seeds (`_seeded_snap`); the
        raster shows the last frame of any class, flagged if stale or suspect.
        With `seq`, that frame instead, if still held (a live snapshot, the seed
        ring or the raw ring), else None. `words`: ship each hit's raw word and
        bank index (smaf v2); default yes for seeded, no for the raster (see
        framing).

        Seeded view only: `seed` (``"s1"``, ``"any"`` or ``"ch<N>"``) and
        `filters` (names from ``sma_words.FILTERS``, OR-ed) choose the seeds at
        request time (`_chosen`). ``"s1"`` without filters is the default and
        sends exactly the payload the analysis made, plus -- only when the frame
        shown is not the newest good one -- a ``stale_view`` entry saying why.
        `mupix` (``"any"``, ``"both"``, ``"either"``, ``"none"``): the MuPix
        selector, AND-ed with the filters. `pattern` (``{"1": "present", "3":
        "absent"}``, counters S1..S5 by number, "any" = no condition): the
        per-counter pattern selector, AND-ed too (`sma_words.parse_pattern`).
        The ``incomplete`` oddity leaves out counters with a known timestamp
        fault (`incomplete_rule`); the pattern selector never does.

        `pixels` (default yes): ship the frame's MuPix pixel hits too, as the
        smaf pixel block (framing, "pixel block"): the seed windows' in the
        seeded view, every examined one (at most `max_hits`, the latest) in
        the raster. No changes nothing else in the payload.
        """
        if view not in ("seeded", "raster"):
            raise ValueError(f"view must be 'seeded' or 'raster', got {view!r}")
        mode = W.parse_seed(seed)
        filt = W.parse_filters(filters)
        mp = W.parse_mupix(mupix)
        req = W.parse_pattern(pattern, len(self.cfg.roles.counters))
        words = (view == "seeded") if words is None else bool(words)
        pixels = True if pixels is None else bool(pixels)
        # Channels to leave out; anything not a channel (-1: "no current") is ignored.
        drop = tuple(sorted({int(c) for c in drop} & set(range(NCH))))
        max_hits = None if max_hits is None else max(0, int(max_hits))
        if view == "seeded" and (mode != W.SEED_S1 or filt or mp != "any" or req):
            return self._chosen(mode, filt, mp, drop, max_hits, words, seq, pixels, req)
        if seq is not None:
            snap = self._snapshot_for(int(seq))
        else:
            snap = self._seeded_snap() if view == "seeded" else self._last
        if snap is None:
            return None
        key = (snap.seq, view, drop, max_hits, words, pixels)
        blob = self._cached(key, lambda: self._encode(snap, view, drop, max_hits, words,
                                                      pixels=pixels))
        if view == "seeded" and seq is None:
            newest = self._last_good
            if newest is not None and newest.seq != snap.seq:
                blob = framing.smaf_update_meta(blob, {"stale_view": self._stale_view(
                    snap, newest, self._noseed_reason)})
        return blob

    def _cached(self, key, make) -> bytes:
        blob = self._frame_cache.get(key)
        if blob is None:
            blob = make()
            self._frame_cache[key] = blob
            while len(self._frame_cache) > BLOB_CACHE_MAX:
                self._frame_cache.popitem(last=False)
        else:
            self._frame_cache.move_to_end(key)
        return blob

    def _stale_view(self, snap: _Snapshot, newest: _Snapshot, reason: str) -> dict:
        """Why the seeded view shows `snap` and not the newest good frame."""
        return {"shown_seq": snap.seq, "newest_seq": newest.seq,
                "age_s": _num(max(0.0, self._clock() - snap.at), 4),
                "good_since": max(0, self.frames_good - snap.good_n), "reason": reason}

    def _selection(self, snap: _Snapshot, mode: str, filt: tuple,
                   mp: str = "any", req: tuple = (), ignore: tuple = ()) -> W.SeedSelection:
        """The seeds of one held frame for a choice, computed once (`sma_words.select_seeds_by`).

        Only good frames have seeds, as in the default view.
        """
        key = (snap.seq, mode, filt, mp, req, ignore)
        sel = self._sel_cache.get(key)
        if sel is not None:
            self._sel_cache.move_to_end(key)
            return sel
        cfg = self.cfg
        fr = snap.fr if snap.cls == "good" else _EMPTY_FRAME
        sel = W.select_seeds_by(fr, cfg.roles, cfg.cuts, mode, filt, mupix=mp, mcuts=cfg.mupix,
                                require=req, ignore=ignore)
        self._sel_cache[key] = sel
        while len(self._sel_cache) > SEL_CACHE_MAX:
            self._sel_cache.popitem(last=False)
        return sel

    def seed_label(self, mode: str) -> str:
        if mode == W.SEED_ANY:
            return "any counter"
        c = self.cfg.roles.s1 if mode == W.SEED_S1 else int(mode[2:])
        return f"{self.labels()[c]} (ch {c})"

    def filter_label(self, name: str) -> str:
        return {"incomplete": "incomplete pattern", "mismatch": "fine/coarse mismatch",
                "tot": f"ToT \u2265 {self.cfg.cuts.tot_corrupt_min}",
                "rf": "RF not valid or vetoed"}[name]

    def pattern_label(self, req: tuple) -> str:
        """``((1, "present"), (3, "absent"))`` -> "S1 present, S3 absent"."""
        labels, counters = self.labels(), self.cfg.roles.counters
        return ", ".join(f"{labels[counters[k - 1]]} {st}" for k, st in req)

    @staticmethod
    def mupix_label(mp: str) -> str:
        return {"any": "any MuPix", "both": "L1+L2 in time", "either": "L1 or L2 in time",
                "none": "no MuPix hit in time"}[mp]

    def _chosen(self, mode, filt, mp, drop, max_hits, words, seq, pixels=True,
                req=()) -> bytes | None:
        """The seeded view for a seed choice and filters.

        Newest good frame first, then back through the seed ring
        (`Sampling/seed ring frames`) to the first frame with a matching seed;
        with `seq`, that frame only. Each frame's selection is computed once and
        cached, so a poll costs one new frame's selection. When no frame
        matches, the newest searched frame is sent without seeds and with
        ``search.no_match`` -- an explicit state, not an empty display.

        The ``incomplete`` oddity's rule (`incomplete_rule`: the counters it
        ignores for a timestamp fault, or that it cannot judge) goes with every
        reply as meta ``incomplete``, outside the cached payload: its numbers
        move while the selections stay valid.
        """
        now = self._clock()
        rule = self.incomplete_rule(self.timestamp_faults(now))
        ign = rule["ignore"]
        if seq is not None:
            snap = self._snapshot_for(int(seq))
            frames = [snap] if snap is not None else []
        else:
            frames = list(reversed(self._ring))
        if not frames:
            return None
        match = None
        searched = made = 0
        t_cpu = time.process_time()
        for k, snap in enumerate(frames):
            if (snap.seq, mode, filt, mp, req, ign) not in self._sel_cache:
                # At least one new selection per request, so the search always
                # moves on; more only while the request is within its budget.
                if made and time.process_time() - t_cpu > SEARCH_BUDGET_S:
                    break                           # out of time: the next poll goes on
                made += 1
            found = self._selection(snap, mode, filt, mp, req, ign)
            searched = k + 1
            if found.idx.size:
                match, pick, sel = k, snap, found
                break
        if match is None:
            pick = frames[0]
            sel = self._selection(pick, mode, filt, mp, req, ign)
        select = {"seed": mode, "seed_label": self.seed_label(mode), "filters": list(filt),
                  "filter_labels": [self.filter_label(f) for f in filt],
                  "candidates": sel.n_candidates, "examined": sel.n_examined,
                  "matching": sel.n_matching, "capped": bool(sel.capped)}
        if mp != "any":
            select["mupix"] = mp
            select["mupix_label"] = self.mupix_label(mp)
        if req:
            select["pattern"] = {str(k): st for k, st in req}
            select["pattern_label"] = self.pattern_label(req)
        key = (pick.seq, "seeded", drop, max_hits, words, mode, filt, mp, pixels, req, ign)
        blob = self._cached(key, lambda: self._encode(pick, "seeded", drop, max_hits, words,
                                                      chosen=sel, extra={"select": select},
                                                      pixels=pixels))
        inc = rule["out"]
        flabel = {f: (inc["label"] if f == "incomplete" else self.filter_label(f)) for f in filt}
        what = ("seeds on any counter" if mode == W.SEED_ANY
                else f"{self.seed_label(mode)} seeds") + (
            f" with {' or '.join(flabel[f] for f in filt)}" if filt else "") + (
            f" and {self.pattern_label(req)}" if req else "") + (
            f" and {self.mupix_label(mp)}" if mp != "any" else "")
        search = {"frames_searched": searched, "newest_seq": frames[0].seq,
                  "ring_frames": len(self._ring)}
        extra = {"search": search, "incomplete": inc}
        if match is None:
            oldest = frames[searched - 1]
            partial = searched < len(frames)
            search["no_match"] = {
                "frames": searched, "oldest_seq": oldest.seq, "partial": partial,
                "span_s": _num(max(0.0, now - oldest.at), 4) if oldest.at else None,
                "text": (f"no {what} in frame seq {oldest.seq}" if seq is not None else
                         f"no {what} in the last {searched} good frame"
                         f"{'' if searched == 1 else 's'}"
                         + (f" (of {len(frames)} held; searching on)" if partial else ""))}
        elif match > 0:
            extra["stale_view"] = self._stale_view(pick, frames[0], f"no {what}")
        return framing.smaf_update_meta(blob, extra)

    def _encode(self, snap: _Snapshot, view, drop, max_hits, words=False,
                chosen: W.SeedSelection | None = None, extra: dict | None = None,
                pixels: bool = True) -> bytes:
        """`chosen`: request-time seeds (`_chosen`) instead of the analysis's S1 seeds;
        `extra`: more meta keys, appended after the default ones; `pixels`: add
        the MuPix pixel block and its ``mupix`` meta (after every other key, so
        that dropping both gives the payload without MuPix byte for byte)."""
        fr, an = snap.fr, snap.an
        o = fr.order
        n = int(o.size)
        s_t = fr.s_t
        in_seed = np.zeros(n, dtype=bool)
        windows = []
        if chosen is not None:
            seeds = np.arange(chosen.idx.size)
            seed_t = chosen.t
        else:
            seeds = an.seeds if an is not None else np.zeros(0, dtype=np.intp)
            seed_t = an.t_s1 if an is not None else None
        for i in seeds:
            ts = int(seed_t[i])
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
        if chosen is not None:
            seed_meta = self._chosen_meta(fr, chosen, windows, idx, t0)
        for i, (a, b) in zip(seeds if chosen is None else (), windows, strict=False):
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
                # The S1 hit's carrier word (a NIM-only S1 hit: its S1L word).
                "s1_word": (int(fr.word_index[o[fr.counter_idx(self.cfg.roles.s1)[
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
        if snap.nim is not None:
            # With NIM copies configured when the frame was analysed: its
            # channel map (as sma::summary's "roles"; -1 = none, "nim" follows
            # "counters"), so the event page never guesses which lane is whose
            # copy, and whether its counters are the merged TOT + NIM hits.
            # Without NIM copies the payload stays byte for byte what it was
            # (the page then reads the summary's roles block).
            meta["roles"] = snap.nim["roles"]
            meta["nim_merge"] = snap.nim["nim_merge"]
        if extra:
            meta.update(extra)
        block = None
        if pixels and fr.px is not None:
            seed_times = [int(seed_t[i]) for i in seeds]
            meta["mupix"], block = self._pixel_block(snap, view, seed_times, max_hits, words)
        # The TOT + NIM pairing per shipped hit (smaf v3), when the frame was
        # paired; the partner indices only with the words (framing, "Version 3").
        pair = pcls = None
        if getattr(fr, "pairing", None) is not None and snap.nim is not None:
            pair, pcls = self._smaf_pairing(fr, idx, snap.nim["roles"]["nim"])
            if not (words and fr.raw is not None):
                pair = None
        return framing.encode_sma_frame(
            meta, t_rel, ch, tot, flags, frame_seq=snap.seq, run_number=snap.run,
            seeded=view == "seeded", stale=snap.cls == "stale", truncated=truncated,
            suspect=snap.cls == "suspect", time_shift=k,
            raw_words=fr.raw[o[idx]] if words and fr.raw is not None else None,
            word_index=widx if words and fr.raw is not None else None, pixels=block,
            pair=pair, cls=pcls)

    @staticmethod
    def _smaf_pairing(fr: W.Frame, idx: np.ndarray, nim_map) -> tuple[np.ndarray, np.ndarray]:
        """``(pair, cls)`` of the shipped hits `idx` (``Frame.s_*`` indices), smaf v3.

        ``pair``: the partner's position among the shipped hits, -1 when there
        is none or it was not shipped. ``cls``: `NimWords.cls` in bits 2:0
        (``framing.PAIR_NONE`` for NIM_NO_CLASS), the sub-flags the page shows
        and whether the word is a NIM copy (framing, "Version 3").
        """
        pw = fr.pairing
        pos = np.full(pw.cls.size, -1, dtype=np.int32)
        pos[idx] = np.arange(idx.size, dtype=np.int32)
        partner = pw.partner[idx]
        pair = np.where(partner >= 0, pos[np.maximum(partner, 0)], -1).astype("<i4")
        c = pw.cls[idx]
        f = pw.flags[idx]
        nim_chans = [n for n in nim_map if n >= 0]

        def b(m, bit):
            return (m != 0).astype(np.uint8) * np.uint8(bit)

        code = (np.where(c == NIM_NO_CLASS, framing.PAIR_NONE,
                         c & framing.PAIR_CLASS_MASK).astype(np.uint8)
                | b(f & N.MULTI_CANDIDATE, framing.PAIR_MULTI)
                | b(f & N.IN_TOT_SHADOW, framing.PAIR_SHADOW)
                | b(f & N.NEAR_FRAME_EDGE, framing.PAIR_EDGE)
                | b(f & NIM_LAG_HELD, framing.PAIR_LAG_HELD)
                | b(np.isin(fr.s_ch[idx], nim_chans), framing.PAIR_NIM_SIDE))
        return pair, code.astype(np.uint8)

    def _pixel_block(self, snap: _Snapshot, view: str, seed_times: list, max_hits, words):
        """``(meta, block)``: the ``mupix`` meta of a frame and its smaf pixel block.

        Seeded view: the pixel hits inside the seed windows, and per seed its
        range in the shipped pixels (``pix``), its in-time L1 / L2 hits and
        whether its in-time window lies wholly in the pixel data (``covered``:
        without it a count of 0 says nothing). Raster: every examined pixel hit, the
        latest `max_hits` at most. Pixel times are ns from ``t0_ns`` (the first
        shipped pixel, absolute on the frame's time basis) in 2^time_shift ns.
        """
        px, m = snap.fr.px, self.cfg.mupix
        wlo, whi = m.window_ns
        t = px.t
        in_seed = np.zeros(px.n, dtype=bool)
        wins = []
        for ts in seed_times:
            a = int(np.searchsorted(t, ts - snap.pre_ns, side="left"))
            b = int(np.searchsorted(t, ts + snap.post_ns, side="right"))
            in_seed[a:b] = True
            wins.append((a, b))
        sel = in_seed if view == "seeded" else np.ones(px.n, dtype=bool)
        pidx = np.flatnonzero(sel)
        n_sel = int(pidx.size)
        truncated = max_hits is not None and pidx.size > max_hits
        if truncated:
            pidx = pidx[pidx.size - max_hits:]
        t0 = int(t[pidx[0]]) if pidx.size else int(px.first)
        span = int(t[pidx[-1]]) - t0 if pidx.size else 0
        k = 0
        while (span >> k) > 0xFFFFFFFF:
            k += 1
        pflags = (px.plane[pidx] & np.uint8(framing.PIX_PLANE_MASK)
                  | (px.row[pidx] >= W.PIXEL_ROWS).astype(np.uint8) * framing.PIX_OFF_SENSOR
                  | in_seed[pidx].astype(np.uint8) * framing.PIX_IN_SEED)
        block = {"t_rel": ((t[pidx] - t0) >> k).astype("<u4"), "time_shift": k,
                 "chip": px.chip[pidx], "col": px.col[pidx], "row": px.row[pidx],
                 "tot": px.tot[pidx], "flags": pflags}
        if words:
            block["raw_words"] = px.raw[pidx]
            block["word_index"] = px.word_index[pidx]
        meta = {"t0_ns": t0, "time_shift": k, "n_words": px.n_words, "n_examined": px.n,
                "n_skipped": px.n_skipped, "n_selected": n_sel, "truncated": bool(truncated),
                "per_plane": [int(x.size) for x in px.planes],
                "planes": {"L1": list(m.l1), "L2": list(m.l2)},
                "window_ns": [wlo, whi], "sideband_ns": list(m.sideband_ns),
                "ts2_shift": m.ts2_shift, "tot_ns": m.tot_ns, "rows": W.PIXEL_ROWS}
        if seed_times:
            ts = np.asarray(seed_times, dtype=np.int64)
            cov = W.mupix_coverage(ts, px, wlo, whi)
            n1, n2 = W.mupix_in_window(ts, px, wlo, whi)
            tsh = t[pidx]
            meta["seeds"] = [{
                "pix": [int(np.searchsorted(tsh, x - snap.pre_ns, side="left")),
                        int(np.searchsorted(tsh, x + snap.post_ns, side="right"))],
                "l1": int(n1[j]), "l2": int(n2[j]), "covered": bool(cov[j])}
                for j, x in enumerate(seed_times)]
        return meta, block

    def _chosen_meta(self, fr: W.Frame, sel: W.SeedSelection, windows, idx, t0) -> list:
        """Per-seed meta of request-time seeds: the default keys (``s1_tot``, RF,
        ``pattern``, ``hits``, ``word_range``, ``s1_word``) plus the seed's own
        channel, ToT and word, ``s1_dt`` (the S1 hit the RF is borrowed from,
        minus the seed), ``rf_na`` (no S1 near the seed) and ``odd`` (every
        oddity the seed has, asked for or not)."""
        o = fr.order
        wi = fr.word_index
        out = []
        for j, (a, b) in enumerate(windows):
            has = bool(sel.has_s1[j])
            s1 = int(sel.s1_idx[j])
            out.append({
                "t_rel": int(sel.t[j]) - t0,
                "s1_tot": (int(fr.s_tot[s1] if sel.s1_tot is None else sel.s1_tot[j])
                           if has else None),
                "rf_phase": _num(sel.rf_phase[j]), "rf_period": _num(sel.rf_period[j]),
                "rf_n": int(sel.rf_n[j]) if has else None, "rf_valid": bool(sel.rf_valid[j]),
                "rf_vetoed": bool(sel.rf_vetoed[j]), "pattern": int(sel.pattern[j]),
                "hits": [int(np.searchsorted(idx, a)), int(np.searchsorted(idx, b))],
                "word_range": ([int(wi[o[a:b]].min()), int(wi[o[a:b]].max())]
                               if wi is not None and b > a else None),
                "s1_word": int(wi[o[s1]]) if wi is not None and has else None,
                "seed_ch": int(sel.ch[j]), "seed_tot": int(sel.tot[j]),
                "seed_word": int(wi[o[int(sel.idx[j])]]) if wi is not None else None,
                "s1_dt": int(sel.s1_dt[j]) if has else None, "rf_na": not has,
                "odd": [f for f in W.FILTERS if int(sel.odd[j]) & W.FILTER_BITS[f]],
            })
        return out

    # -- commands ----------------------------------------------------------------

    def commands(self) -> dict:
        """``sma::summary`` / ``sma::trend`` / ``sma::frame``, framed for brpc.

        Args are JSON (empty means defaults):

        * summary: ``{"run_active": bool}`` -- the page knows the run state;
        * trend: ``{"since": unix_s}`` -- only rows after it, for incremental polls;
        * frame: ``{"view": "seeded"|"raster", "drop": [ch...], "max_hits": N,
          "words": bool, "seq": N, "pixels": bool}``, and for the seeded view
          ``"seed": "s1" | "ch<N>" | "any"``, ``"filters": ["incomplete",
          "mismatch", "tot", "rf"]``, ``"mupix": "any" | "both" | "either" |
          "none"`` and ``"pattern": {"1": "present", "3": "absent"}`` (see
          `frame_blob`); no frame yet gives ``json {"no_frame": true}``.
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
                               None if seq is None else int(seq),
                               seed=a.get("seed"), filters=a.get("filters"),
                               mupix=a.get("mupix"), pixels=a.get("pixels"),
                               pattern=a.get("pattern"))
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
            "seed_ring": {"frames": len(self._ring), "bytes": self._ring_bytes,
                          "limit_frames": self.cfg.seed_ring_frames,
                          "limit_bytes": self.cfg.seed_ring_bytes,
                          "oldest_seq": self._ring[0].seq if self._ring else None,
                          "selections_cached": len(self._sel_cache),
                          "frames_cached": len(self._frame_cache)},
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

#: No hits at all: what a non-good frame is searched as (it has no seeds).
_EMPTY_FRAME = W.prepare_frame(np.zeros(0, dtype="<u8"))


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
                  "oversize", "mp_frames", "mp_pix", "mp_examined", "mp_skipped", "mp_n",
                  "mp_rows", "mp_unsorted", "xy_light", "xy_heavy", "xy_ctrk"):
            setattr(out, a, getattr(out, a) + getattr(s, a))
        out.xy_state += s.xy_state
        out.hits += s.hits
        out.rate_hits += s.rate_hits
        out.mismatch += s.mismatch
        out.mp_in += s.mp_in
        out.mp_side += s.mp_side
        out.mp_chip += s.mp_chip
        if s.eff.size == out.eff.size:
            out.eff += s.eff
        if s.nim.shape == out.nim.shape:
            out.nim += s.nim
    return out


def _hist_median(h: Hist1D) -> tuple[float | None, int]:
    """``(median bin centre, in-range entries)`` of a 1D histogram; None when empty."""
    c = np.asarray(h.counts[1:-1], dtype=np.int64)
    n = int(c.sum())
    if n == 0:
        return None, 0
    i = int(np.searchsorted(np.cumsum(c), (n + 1) / 2))
    w = (h.x.hi - h.x.lo) / h.x.n
    return h.x.lo + (i + 0.5) * w, n


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


def _fill_1d_index(h: Hist1D, i) -> None:
    """Fill from full bin indices (0 underflow .. n+1 overflow), clipped."""
    i = np.clip(np.asarray(i, dtype=np.intp).ravel(), 0, h.x.n + 1)
    if i.size:
        h.add_counts(np.bincount(i, minlength=h.x.n + 2), entries=i.size)


def _fill_2d_index(h: Hist2D, ix, iy) -> None:
    """Fill from full bin indices (0 underflow .. n+1 overflow), clipped like `_index`."""
    ix = np.clip(np.asarray(ix, dtype=np.intp).ravel(), 0, h.x.n + 1)
    iy = np.clip(np.asarray(iy, dtype=np.intp).ravel(), 0, h.y.n + 1)
    if ix.size == 0:
        return
    flat = iy * (h.x.n + 2) + ix
    h.add_counts(np.bincount(flat, minlength=h.counts.size).reshape(h.counts.shape),
                 entries=ix.size)


def _add_flat(h: Hist2D, flat) -> None:
    """Add one count per flat index (``iy * (nx + 2) + ix``) in place.

    For fills of a few hundred entries into a large 2D histogram: no
    histogram-sized temporary, unlike the bincount of `_fill_2d_index` (a
    130 x 130 map costs ~30 us that way, whatever the entries).
    """
    flat = np.asarray(flat, dtype=np.intp).ravel()
    if not flat.size:
        return
    _widen(h, flat.size)
    if h.counts.flags.c_contiguous:             # always (np.zeros): reshape is a view
        np.add.at(h.counts.reshape(-1), flat, h.counts.dtype.type(1))
        h.entries += int(flat.size)
    else:
        h.add_counts(np.bincount(flat, minlength=h.counts.size).reshape(h.counts.shape))


def _widen(h, n: int) -> None:
    """Make a uint32 histogram uint64 before ``n`` more counts could wrap a bin.

    No bin holds more than ``entries`` (every fill adds as many entries as
    counts, Clear zeroes both), so below 2^32 - 1 entries nothing can wrap.
    Past it the map travels as f64 (8 bytes a bin) from then on: at the
    sampled rates that takes weeks of one run.
    """
    if h.counts.dtype == np.uint32 and h.entries + int(n) >= np.iinfo(np.uint32).max:
        h.counts = h.counts.astype(np.uint64)
        h.dtype = np.uint64


def _bin_of(values, ax: Axis) -> np.ndarray:
    """`_index` for finite values, with fewer numpy calls (no clip wrapper)."""
    i = np.floor((np.asarray(values, dtype=np.float64) - ax.lo) * (ax.n / (ax.hi - ax.lo)))
    np.maximum(i, -1.0, out=i)
    np.minimum(i, float(ax.n), out=i)
    return i.astype(np.intp) + 1


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
