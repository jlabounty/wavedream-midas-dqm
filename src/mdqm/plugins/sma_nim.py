"""TOT + NIM pairing of the SMA counters and the NIM fine-time lag vote, free of MIDAS.

Pure numpy, no plugin state. Since run 1015 every scintillator counter reaches
the SMA board twice: its TOT-box output (the TOT word: time and ToT) and a
low-threshold NIM discriminator copy (the NIM word, "S*k*L", on its own
channel). This module merges the two into one hit per particle and counter, as
reco does, and measures the whole-frame fine-time lag a NIM channel can carry.

Provenance
----------
Ported, not imported (the C++ is a reco header, this is the live DQM):

* :func:`pair_counter` and :func:`wide_dt` are ``PIPSMSMANimPairing::Pair``
  and ``detail::WidePairs`` of
  ``reco_testbeam/psm/exp/alg/include/PIPSMSMANimPairing.hh`` (branch
  ``feature/sma-nim-pairing``, HEAD ``f25bf7c``) for one counter whose NIM is
  expected (cabled and with an offset). The flag bits, the classes, the tie
  rule and the merged order are that header's. ``tests/test_sma_nim.py`` runs
  the real header on the hit lists of ``tests/data/nim_pairing_golden.json``
  (``tests/cpp/``) and requires identical classes, flags, partners and merged
  hits.
* :func:`lag_pairs`, :func:`lag_modes` and :func:`decide_lag` are the lag role
  of ``pi_midas/include/PISMAFineOffset.hh`` (``Scan``, ``DecideLagChannel``)
  and ``PISMAWord.hh`` (``LagDiff``, ``SignedLag``, ``FindLagModes``,
  ``DecideLag``, ``LagShiftNs``) on the same branch. The golden file holds
  ``Scan`` results on synthetic banks too. Measure only: the DQM never removes
  a lag, so ``Corrector`` and the fallback lag of the job are not ported.

Conventions
-----------
* Times are those of :class:`mdqm.plugins.sma_words.Frame` (``s_t``): int64 ns,
  one channel ascending. Aligned times are ``t' = t - offset``. Pass integer
  offsets and windows to keep the int64 arithmetic exact; a float offset makes
  the aligned times float64 (exact below 2^53 ns, and the same IEEE operations
  as the C++, which works in double throughout).
* dt is always ``t'_NIM - t'_TOT``.
* "Index" of a word, for the ties: its position in the frame's sorted kept
  hits (``Frame.chan[c]``), which is the raw index order of the C++ (time,
  then stream order). The per-side arrays must be ascending in (time, index).
  :func:`pair_counter` requires the indices and the frame's first/last raw
  hit time, which the C++ takes from the whole frame.
* Offsets: pass whole ns (the plugin rounds them); a whole-number float is
  taken as an int, a fractional one makes the aligned times float64.
* Echo rule, matching and classes: see :func:`pair_counter`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .sma_words import FINE_WRAP_NS, window_pairs

# --- flag bits (PIPSMSMANimPairing::Flag, the sma_hits flags column) ----------

HAS_TOT = 1 << 0
HAS_NIM = 1 << 1
NIM_EXPECTED = 1 << 2
INCOMPLETE = 1 << 3
TIME_FROM_NIM = 1 << 4
TOT_SUBSTITUTED = 1 << 5
MULTI_CANDIDATE = 1 << 6
IN_TOT_SHADOW = 1 << 7
NEAR_FRAME_EDGE = 1 << 8
ECHO = 1 << 9

FLAG_NAMES = ("has_tot", "has_nim", "nim_expected", "incomplete", "time_from_nim",
              "tot_substituted", "multi_candidate", "in_tot_shadow", "near_frame_edge",
              "echo")

# --- classes of a merged hit (one per TOT word, one per unpaired NIM word) ----

PAIRED = 0
TOT_ONLY = 1
ECHO_WORD = 2
NIM_ONLY = 3
CLASS_NAMES = ("paired", "tot_only", "echo", "nim_only")

TIME_SOURCES = ("tot", "nim")

#: The 8-bit ToT field's largest value: the shadow look-back is 255 ToT units
#: (``Setup::maxTotNs``).
MAX_TOT = 255

#: Greedy rounds done vectorised before the rest is taken in a plain loop.
#: Real frames finish in 2-3 rounds; only a long chain of equal |dt| (a pulser
#: at ~1/window) needs more, and the loop then handles its remainder exactly.
GREEDY_ROUNDS = 16

#: :func:`decide_lag` states. "none": fewer than ``lag_min_pairs`` pairs in the
#: fullest window (no pairs at all included); "ambiguous": enough, but not
#: ``lag_dominance`` times the runner-up; "ok": a vote within ``lag_tol_ns`` of
#: the nominal delay; "faulted": a vote further away (the lag fault).
LAG_STATES = ("none", "ambiguous", "ok", "faulted")

_FINE_MASK = FINE_WRAP_NS - 1
_HALF_FINE = FINE_WRAP_NS >> 1


@dataclass(frozen=True)
class NimConfig:
    """The pairing and lag-vote settings (``/DQM/SMA/NIM``) that are not per counter.

    Per counter (offsets, whether the echo rule runs) are arguments of
    :func:`pair_counter`; the channel map is the plugin's.
    """

    #: Pair when ``|t'_NIM - t'_TOT| <= pair_window_ns`` (inclusive).
    pair_window_ns: int = 20
    #: "tot": a paired hit keeps the TOT time; "nim": it takes ``t'_NIM``.
    time_source: str = "tot"
    #: The ToT a NIM-only hit is given (``Config::nimOnlyTot``).
    nim_only_tot: int = 1
    #: ns per ToT unit, for a word's trailing edge ``t + ToT * unit``.
    tot_unit_ns: int = 1
    #: A TOT word with ToT >= this is an echo (late) word; <= 0 turns it off.
    echo_late_tot: int = 128
    #: A TOT word starting within this of the previous word's trailing edge
    #: is an echo word; < 0 turns it off.
    echo_edge_tol_ns: int = 3
    #: Lag vote: the mode window is ``2 * lag_tol_ns`` wide, and a lag within
    #: ``lag_tol_ns`` of ``lag_nominal_ns`` is a real delay ("ok").
    lag_tol_ns: float = 50.0
    #: Lag vote: the fullest window needs at least this many pairs ...
    lag_min_pairs: int = 50
    #: ... and at least this many times the runner-up window's.
    lag_dominance: float = 2.0
    #: Lag vote: the NIM copy's expected delay from S1 [ns], when the call
    #: gives none; the plugin passes each channel's own (``nominal_ns``).
    lag_nominal_ns: int = 0
    #: Lag vote: a NIM word whose preceding S1 word has another coarse field
    #: may pair with the next S1 word (``Config::useNextReference``).
    lag_next_reference: bool = True

    def __post_init__(self):
        # ODB values arrive as Python ints or floats (DOUBLE keys). A
        # whole-number float becomes an int, so 20.0 keeps int64 times exact;
        # an int-only field refuses a fraction instead of truncating it.
        for name in ("lag_min_pairs", "lag_nominal_ns", "echo_late_tot"):
            object.__setattr__(self, name, _as_int(getattr(self, name), name))
        for name in ("pair_window_ns", "tot_unit_ns", "echo_edge_tol_ns", "nim_only_tot"):
            object.__setattr__(self, name, _as_number(getattr(self, name), name))
        # The checks of PIPSMSMANimPairing::Prepare and PISMAFineOffset::CheckConfig.
        if not self.pair_window_ns > 0:
            raise ValueError(f"the pair window must be positive, got {self.pair_window_ns}")
        if not self.tot_unit_ns > 0:
            raise ValueError(f"the ToT unit must be positive, got {self.tot_unit_ns}")
        if self.time_source not in TIME_SOURCES:
            raise ValueError(f"time source must be one of {TIME_SOURCES}, got {self.time_source!r}")
        if not (math.isfinite(self.lag_tol_ns) and 0 < self.lag_tol_ns < 1024):
            raise ValueError(f"the lag tolerance must be in (0, 1024) ns, got {self.lag_tol_ns}")
        if self.lag_min_pairs < 1:
            raise ValueError(f"lag min pairs must be >= 1, got {self.lag_min_pairs}")
        if not (math.isfinite(self.lag_dominance) and self.lag_dominance >= 1):
            raise ValueError(f"the lag dominance must be finite and >= 1, got {self.lag_dominance}")
        _check_nominal(self.lag_nominal_ns)


def _as_int(v, name) -> int:
    """``v`` as an int: an int, or a float with no fraction; anything else raises."""
    if isinstance(v, (bool, np.bool_)):
        raise ValueError(f"{name} must be a whole number, got {v!r}")
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return int(v)
    raise ValueError(f"{name} must be a whole number, got {v!r}")


def _as_number(v, name):
    """``v`` as an int when it is a whole number, else as a finite float."""
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a number, got {v!r}")
    if not math.isfinite(float(v)):
        raise ValueError(f"{name} must be finite, got {v!r}")
    return int(v) if float(v).is_integer() else float(v)


def _check_nominal(nominal_ns) -> int:
    n = _as_int(nominal_ns, "the lag nominal")
    if not -_HALF_FINE < n < _HALF_FINE:
        raise ValueError(f"the lag nominal must be in (-2^19, 2^19) ns, got {nominal_ns}")
    return n


# ==============================================================================
# Pairing (PIPSMSMANimPairing.hh)
# ==============================================================================

@dataclass
class Merged:
    """One counter's merged hits, ascending in time (ties: the carrier's index).

    ``src`` indexes the carrier word in its side's arrays: the TOT word for
    paired, TOT-only and echo hits, the NIM word for a NIM-only hit (``cls``
    tells which side).
    """

    t: np.ndarray       # the hit time (aligned): t'_TOT, or t'_NIM (NIM-only, time source nim)
    tot: np.ndarray     # the TOT word's ToT, or NimConfig.nim_only_tot
    cls: np.ndarray     # uint8, PAIRED / TOT_ONLY / ECHO_WORD / NIM_ONLY
    src: np.ndarray     # intp
    flags: np.ndarray   # uint16

    @property
    def n(self) -> int:
        return int(self.t.size)


@dataclass
class PairResult:
    """:func:`pair_counter` on one counter and frame.

    Per TOT word (``*_tot``, the input order) and per NIM word (``*_nim``).
    A paired NIM word carries its pair's flags; ``n_cand_tot`` is 0 for an
    echo word, which takes no part in the matching (the C++ ``TotDiag`` is
    written for the other TOT words only).
    """

    t_tot: np.ndarray            # aligned TOT times
    t_nim: np.ndarray            # aligned NIM times
    tot_tot: np.ndarray          # ToT field of the TOT words
    tot_nim: np.ndarray          # ToT field (logic width) of the NIM words
    echo: np.ndarray             # bool[n_tot]
    partner_tot: np.ndarray      # intp[n_tot]: the NIM word paired with it, -1
    partner_nim: np.ndarray      # intp[n_nim]: the TOT word paired with it, -1
    n_cand_tot: np.ndarray       # int32[n_tot]: NIM words within the window
    n_cand_nim: np.ndarray       # int32[n_nim]: non-echo TOT words within the window
    cls_tot: np.ndarray          # uint8[n_tot]: PAIRED / TOT_ONLY / ECHO_WORD
    cls_nim: np.ndarray          # uint8[n_nim]: PAIRED / NIM_ONLY
    flags_tot: np.ndarray        # uint16[n_tot]
    flags_nim: np.ndarray        # uint16[n_nim]
    #: Per NIM word: t'_NIM - t'_TOT of the nearest TOT word (echoes included,
    #: a tie to the earlier one), float64 (exact), NaN without TOT words. The
    #: C++ ``NimDiag::dtAligned``: the ``nim_dt`` histogram.
    nim_dt: np.ndarray
    #: ToT of that nearest TOT word, -1 without one (``NimDiag::nearestTot``).
    nim_nearest_tot: np.ndarray
    merged: Merged

    @property
    def paired(self) -> np.ndarray:
        """Indices of the paired TOT words (ascending)."""
        return np.flatnonzero(self.partner_tot >= 0)

    @property
    def pair_dt(self) -> np.ndarray:
        """t'_NIM - t'_TOT of every pair, in TOT order (``PairDiag::dt``, the walk)."""
        a = self.paired
        return self.t_nim[self.partner_tot[a]] - self.t_tot[a]

    @property
    def pair_tot(self) -> np.ndarray:
        """The TOT word's ToT of every pair, in TOT order (``PairDiag::tot``)."""
        return self.tot_tot[self.paired]

    def counts(self) -> dict:
        """The C++ ``CounterCounts`` of an expected counter (``totPassed`` and
        ``nimDropped`` are 0 there): words, classes and flagged hits."""
        hit_flags = self.merged.flags
        return {
            "tot_words": int(self.t_tot.size),
            "nim_words": int(self.t_nim.size),
            "paired": int(np.count_nonzero(self.cls_tot == PAIRED)),
            "tot_only": int(np.count_nonzero(self.cls_tot == TOT_ONLY)),
            "nim_only": int(np.count_nonzero(self.cls_nim == NIM_ONLY)),
            "echo": int(np.count_nonzero(self.cls_tot == ECHO_WORD)),
            "shadow": int(np.count_nonzero(hit_flags & IN_TOT_SHADOW)),
            "multi": int(np.count_nonzero(hit_flags & MULTI_CANDIDATE)),
            "near_edge": int(np.count_nonzero(hit_flags & NEAR_FRAME_EDGE)),
        }

    def merged_hits(self, nim_only: bool = True) -> Merged:
        """:attr:`merged`, or without its NIM-only hits (merge off, lag held back)."""
        if nim_only:
            return self.merged
        m = self.merged
        k = m.cls != NIM_ONLY
        return Merged(m.t[k], m.tot[k], m.cls[k], m.src[k], m.flags[k])


def _offset(offset):
    """An offset as an int when it is a whole number (an ODB 0.0 or 3.0), so
    int64 times stay int64; a fractional one stays a float (float64 times)."""
    return _as_number(offset, "offset")


def _align(t, offset):
    t = np.asarray(t)
    if t.dtype.kind in "ui" or t.size == 0:      # an empty list arrives as float64
        t = t.astype(np.int64, copy=False)
    # A zero offset leaves the times as they are (and int64).
    return t if offset == 0 else t - offset


def _first_of_group(keys, n_keys) -> np.ndarray:
    """Mask of the first occurrence of each value of ``keys`` (ints < ``n_keys``)."""
    first = np.full(n_keys, keys.size, dtype=np.intp)
    # a minimum per key, O(len): explicit, where a reversed fancy assignment
    # would lean on numpy's handling of repeated indices
    np.minimum.at(first, keys, np.arange(keys.size, dtype=np.intp))
    m = np.zeros(keys.size, dtype=bool)
    m[first[first < keys.size]] = True
    return m


def greedy_match(a, b, rank_key, n_tot, n_nim, max_rounds=GREEDY_ROUNDS):
    """One-to-one matching of candidate pairs ``(a[k], b[k])``, taken in order.

    ``rank_key`` sorts the candidates (a ``lexsort`` key tuple, most
    significant *last*, and a strict order: no two candidates equal on all
    keys); :func:`greedy_match_by_dt` builds the pairing's. Equivalent to
    walking the sorted candidates once and keeping a pair when both words are
    still free (the C++ loop):

    * a candidate whose two words have no other candidate is taken whatever
      the order, so those are taken first and only the rest is sorted
      (``Pair()`` does the same; most of a frame is such isolated pairs);
    * the rest goes in rounds: a candidate that comes first among the
      remaining candidates of both its words is taken by the walk whatever
      happens elsewhere (nothing earlier can take its words), so every such
      candidate of a round is kept at once, the candidates of the words taken
      are dropped, and the next round starts. After ``max_rounds`` the
      remainder is walked in a loop.

    Returns ``(partner_tot intp[n_tot], partner_nim intp[n_nim])``, -1 = free.
    """
    p_tot = np.full(n_tot, -1, dtype=np.intp)
    p_nim = np.full(n_nim, -1, dtype=np.intp)
    if len(a) == 0:
        return p_tot, p_nim
    a = np.asarray(a, dtype=np.intp)
    b = np.asarray(b, dtype=np.intp)
    alone = (np.bincount(a, minlength=n_tot)[a] == 1) & (np.bincount(b, minlength=n_nim)[b] == 1)
    p_tot[a[alone]] = b[alone]
    p_nim[b[alone]] = a[alone]
    rest = np.flatnonzero(~alone)
    if rest.size == 0:
        return p_tot, p_nim
    order = rest[np.lexsort(tuple(np.asarray(k)[rest] for k in rank_key))]
    A, B = a[order], b[order]
    for _ in range(max_rounds):
        if A.size == 0:
            return p_tot, p_nim
        take = _first_of_group(A, n_tot) & _first_of_group(B, n_nim)
        p_tot[A[take]] = B[take]
        p_nim[B[take]] = A[take]
        free = (p_tot[A] < 0) & (p_nim[B] < 0)
        A, B = A[free], B[free]
    if A.size == 0:
        return p_tot, p_nim
    pt, pn = p_tot.tolist(), p_nim.tolist()        # list indexing: ~10x numpy scalars
    for i, j in zip(A.tolist(), B.tolist()):
        if pt[i] < 0 and pn[j] < 0:
            pt[i] = j
            pn[j] = i
    return np.asarray(pt, dtype=np.intp), np.asarray(pn, dtype=np.intp)


def greedy_match_by_dt(a, b, dt, n_tot, n_nim, max_rounds=GREEDY_ROUNDS):
    """:func:`greedy_match` in the pairing's order: increasing ``|dt|``, a tie
    to the earlier TOT word ``a``, then the earlier NIM word ``b``
    (``Pair()``'s sort of the contested candidates)."""
    return greedy_match(a, b, (b, a, np.abs(dt)), n_tot, n_nim, max_rounds)


def echo_mask(t_tot, tot_tot, cfg: NimConfig) -> np.ndarray:
    """Echo words of one counter's TOT words (aligned times, ascending).

    A word with ToT >= ``echo_late_tot``, or one starting within
    ``+-echo_edge_tol_ns`` of the previous word's trailing edge ``t_prev +
    ToT_prev * tot_unit_ns``, each word against the one before it, echoes
    included (so a chain of echoes is all echoes).
    """
    t = np.asarray(t_tot)
    tot = np.asarray(tot_tot).astype(np.int64)
    e = np.zeros(t.size, dtype=bool)
    if cfg.echo_late_tot > 0:
        e |= tot >= cfg.echo_late_tot
    if cfg.echo_edge_tol_ns >= 0 and t.size > 1:
        edge = t[:-1] + tot[:-1] * cfg.tot_unit_ns
        e[1:] |= np.abs(t[1:] - edge) <= cfg.echo_edge_tol_ns
    return e


def nearest_dt(t_tot, t_nim):
    """Per NIM word, ``t_nim - t_tot`` of the nearest TOT word (a tie to the
    earlier one) and that word's index; ``(NaN, -1)`` without TOT words."""
    tt = np.asarray(t_tot)
    tn = np.asarray(t_nim)
    if tt.size == 0:
        return np.full(tn.size, np.nan), np.full(tn.size, -1, dtype=np.intp)
    k = np.searchsorted(tt, tn, side="left")
    lo = np.clip(k - 1, 0, tt.size - 1)
    hi = np.clip(k, 0, tt.size - 1)
    near = np.where(tn - tt[lo] <= tt[hi] - tn, lo, hi)
    near = np.where(k == 0, 0, np.where(k == tt.size, tt.size - 1, near))
    return (tn - tt[near]).astype(np.float64), near


#: Half the 2^20 ns fine span: the wide dt reaches this far (``kHalfFineWrapNs``).
HALF_FINE_WRAP_NS = _HALF_FINE


def wide_dt(t_tot, t_nim, frame_lo, frame_hi, budget) -> np.ndarray:
    """The wide dt sample of one counter and frame (``detail::WidePairs``): raw
    ``t_NIM - t_TOT`` of NIM words against every TOT word within
    ``[t_NIM - 2^19, t_NIM + 2^19)`` ns, the ``nim_dt_wide`` histogram (a lag
    fault shows up there; the raw times are used, no offsets).

    Raw times, each side ascending; ``frame_lo``/``frame_hi`` the frame's first
    and last raw SMA hit time. Never all pairs (a busy counter has millions):
    NIM words are sampled evenly over the frame's interior (at least 2^19 ns
    from both ends, so the whole window around each exists in the frame;
    the whole frame when it has no interior), each paired with every TOT word
    in its window, until ``budget`` pairs are reached (the last NIM word's
    pairs are all kept, so the sample can exceed ``budget`` by one window).
    The number of NIM words aimed for is ``budget`` over the expected TOT
    words per window. Returns the dt values (dtype of the times), grouped by
    NIM word, TOT ascending within one; empty for ``budget <= 0`` or an empty
    side.
    """
    tt = np.asarray(t_tot)
    tn = np.asarray(t_nim)
    budget = int(budget)
    H = HALF_FINE_WRAP_NS
    if budget <= 0 or tt.size == 0 or tn.size == 0:
        return np.zeros(0, dtype=np.result_type(tt, tn))
    lo = int(np.searchsorted(tn, frame_lo + H, side="left"))
    hi = int(np.searchsorted(tn, frame_hi - H, side="right"))
    first, last = (lo, hi) if lo < hi else (0, tn.size)
    # the C++ arithmetic, in double, step by step
    span = max(float(frame_hi - frame_lo), 1.0)
    per_nim = max(1.0, float(tt.size) * min(1.0, 2.0 * H / span))
    want = max(1, int(float(budget) / per_nim))
    n = last - first
    m = min(want, n)
    step = float(n) / float(m)
    b = first + ((np.arange(m + 1, dtype=np.float64) + 0.5) * step).astype(np.int64)
    # the first index b >= last ends the walk: keep the prefix below it
    stop = np.flatnonzero(b >= last)
    if stop.size:
        b = b[: stop[0]]
    k0 = np.searchsorted(tt, tn[b] - H, side="left")
    k1 = np.searchsorted(tt, tn[b] + H, side="left")
    cnt = k1 - k0
    before = np.cumsum(cnt) - cnt             # pairs taken before each NIM word
    keep = before < budget
    b, k0, cnt = b[keep], k0[keep], cnt[keep]
    total = int(cnt.sum())
    if total == 0:
        return np.zeros(0, dtype=np.result_type(tt, tn))
    rep = np.repeat(np.arange(b.size), cnt)
    start = np.cumsum(cnt) - cnt
    j = np.arange(total) - start[rep] + k0[rep]
    return tn[b[rep]] - tt[j]


def pair_counter(t_tot, tot_tot, t_nim, tot_nim, cfg: NimConfig | None = None, *,
                 frame_lo, frame_hi, idx_tot, idx_nim,
                 tot_offset_ns=0, nim_offset_ns=0, echo: bool = False) -> PairResult:
    """Pair one counter's TOT and NIM words of one frame (``Pair()`` of an expected counter).

    Inputs are raw times (ascending, ties in index order) and the ToT fields
    of each side; ``tot_offset_ns``/``nim_offset_ns`` align them (``t' = t -
    offset``). Whole-number offsets (int, or a float like 3.0) keep int64
    times int64; a fractional one makes the aligned times float64, so the
    plugin rounds its offsets to whole ns. ``echo``: run the echo rule on this
    counter's TOT words.

    Required, because the C++ takes them from the whole frame and a default
    would silently differ from nearline:

    * ``frame_lo``/``frame_hi``: the frame's first and last raw SMA hit time,
      every channel (``Frame.first``/``Frame.last``). ``None`` for both turns
      the nearFrameEdge flag off.
    * ``idx_tot``/``idx_nim``: each word's index for the merged ties, the
      positions in the frame's sorted kept hits (``Frame.chan[c]`` of the TOT
      and of the NIM channel). The Frame sort is stable, so these order exact
      time ties by stream order, as the C++ raw index does.

    * **Echo words** (:func:`echo_mask`) take no part in the matching and
      become hits of class ``ECHO_WORD`` (flags hasTot, nimExpected, echo).
    * **Matching**: one to one within ``|dt| <= pair_window_ns`` (inclusive),
      candidates taken by increasing ``|dt|``, a pair only when both words are
      free; a tie goes to the earlier TOT word, then the earlier NIM word
      (:func:`greedy_match_by_dt`).
    * **Classes and flags**: paired (hasTot, hasNim; timeFromNim with time
      source "nim"), TOT-only (hasTot, incomplete), NIM-only (hasNim,
      incomplete, timeFromNim, totSubstituted), all with nimExpected.
      multiCandidate: the hit's TOT or NIM word had a candidate other than its
      partner. inTotShadow (NIM-only): inside ``[t'_TOT, t'_TOT + ToT * unit]``
      of a TOT word of the counter (echoes included). nearFrameEdge: the
      partner window around the hit's aligned time, taken back to the partner
      channel's raw time, reaches ``frame_lo``/``frame_hi``. A paired NIM
      word carries its pair's flags (``flags_nim == flags_tot`` of its TOT).
    * **Merged hits**: one per TOT word and per NIM-only word, ascending in
      time, ties by the carrier's index.
    """
    cfg = cfg or NimConfig()
    if (frame_lo is None) != (frame_hi is None):
        raise ValueError("frame_lo and frame_hi go together (both None: no edge flag)")
    t_tot_raw = _align(t_tot, 0)
    t_nim_raw = _align(t_nim, 0)
    tot_tot = np.asarray(tot_tot) if len(tot_tot) else np.zeros(0, dtype=np.int16)
    tot_nim = np.asarray(tot_nim) if len(tot_nim) else np.zeros(0, dtype=np.int16)
    n_t, n_n = t_tot_raw.size, t_nim_raw.size
    idx_tot = np.asarray(idx_tot, dtype=np.int64).reshape(-1)
    idx_nim = np.asarray(idx_nim, dtype=np.int64).reshape(-1)
    if idx_tot.size != n_t or idx_nim.size != n_n:
        raise ValueError(f"idx_tot/idx_nim must have one entry per word: {idx_tot.size} vs "
                         f"{n_t} TOT, {idx_nim.size} vs {n_n} NIM")
    tot_offset_ns = _offset(tot_offset_ns)
    nim_offset_ns = _offset(nim_offset_ns)
    tt = _align(t_tot_raw, tot_offset_ns)
    tn = _align(t_nim_raw, nim_offset_ns)
    W = cfg.pair_window_ns
    unit = cfg.tot_unit_ns

    e = echo_mask(tt, tot_tot, cfg) if echo else np.zeros(n_t, dtype=bool)

    # candidates: non-echo TOT word i, NIM word j, |tn - tt| <= W
    live = np.flatnonzero(~e)
    i, j, dt = window_pairs(tt[live], tn, -W, W)
    a = live[i]
    n_cand_tot = np.bincount(a, minlength=n_t).astype(np.int32)
    n_cand_nim = np.bincount(j, minlength=n_n).astype(np.int32)
    p_tot, p_nim = greedy_match_by_dt(a, j, dt, n_t, n_n)

    # flags per TOT word
    paired = p_tot >= 0
    pb = p_tot[paired]
    flags_tot = np.full(n_t, HAS_TOT | NIM_EXPECTED, dtype=np.uint16)
    cls_tot = np.full(n_t, TOT_ONLY, dtype=np.uint8)
    flags_tot[e] |= ECHO
    cls_tot[e] = ECHO_WORD
    flags_tot[paired] |= HAS_NIM | (TIME_FROM_NIM if cfg.time_source == "nim" else 0)
    cls_tot[paired] = PAIRED
    multi = np.zeros(n_t, dtype=bool)
    multi[paired] = (n_cand_tot[paired] > 1) | (n_cand_nim[pb] > 1)
    tot_only = ~e & ~paired
    flags_tot[tot_only] |= INCOMPLETE
    multi[tot_only] = n_cand_tot[tot_only] > 0
    flags_tot[multi] |= MULTI_CANDIDATE

    # flags of the NIM-only words
    nim_only = p_nim < 0
    cls_nim = np.where(nim_only, NIM_ONLY, PAIRED).astype(np.uint8)
    flags_nim = np.zeros(n_n, dtype=np.uint16)
    flags_nim[nim_only] = HAS_NIM | NIM_EXPECTED | INCOMPLETE | TIME_FROM_NIM | TOT_SUBSTITUTED
    flags_nim[nim_only & (n_cand_nim > 0)] |= MULTI_CANDIDATE
    q = np.flatnonzero(nim_only)
    if q.size and n_t:
        # TOT words starting in [t'_NIM - 255 units, t'_NIM] whose pulse reaches it
        ii, kk, _ = window_pairs(tn[q], tt, -MAX_TOT * unit, 0)
        inside = tn[q][ii] <= tt[kk] + tot_tot[kk].astype(np.int64) * unit
        shadow = np.zeros(q.size, dtype=bool)
        shadow[ii[inside]] = True
        flags_nim[q[shadow]] |= IN_TOT_SHADOW

    # near the frame edge: the partner window in the partner's raw time
    if frame_lo is not None:
        tp = tt + nim_offset_ns
        flags_tot[(tp - W <= frame_lo) | (tp + W >= frame_hi)] |= NEAR_FRAME_EDGE
        tq = tn[q] + tot_offset_ns
        flags_nim[q[(tq - W <= frame_lo) | (tq + W >= frame_hi)]] |= NEAR_FRAME_EDGE
    # a paired NIM word carries its pair's flags, edge flag included
    flags_nim[~nim_only] = flags_tot[p_nim[~nim_only]]

    nim_dt, near = nearest_dt(tt, tn)
    nearest_tot = (tot_tot[near].astype(np.int64) if n_t
                   else np.full(n_n, -1, dtype=np.int64))

    # merged hits
    t_hit = tt
    if cfg.time_source == "nim" and pb.size:
        t_hit = tt.astype(np.result_type(tt, tn))     # a copy, in the common dtype
        t_hit[paired] = tn[pb]
    tot_dtype = np.promote_types(tot_tot.dtype, np.asarray(cfg.nim_only_tot).dtype)
    t_all = np.concatenate([t_hit, tn[q]])
    carrier = np.concatenate([idx_tot, idx_nim[q]])
    s = np.lexsort((carrier, t_all))
    merged = Merged(
        t=t_all[s],
        tot=np.concatenate([tot_tot.astype(tot_dtype),
                            np.full(q.size, cfg.nim_only_tot, dtype=tot_dtype)])[s],
        cls=np.concatenate([cls_tot, cls_nim[q]])[s],
        src=np.concatenate([np.arange(n_t, dtype=np.intp), q])[s],
        flags=np.concatenate([flags_tot, flags_nim[q]])[s])

    return PairResult(t_tot=tt, t_nim=tn, tot_tot=tot_tot, tot_nim=tot_nim, echo=e,
                      partner_tot=p_tot, partner_nim=p_nim,
                      n_cand_tot=n_cand_tot, n_cand_nim=n_cand_nim,
                      cls_tot=cls_tot, cls_nim=cls_nim,
                      flags_tot=flags_tot, flags_nim=flags_nim,
                      nim_dt=nim_dt, nim_nearest_tot=nearest_tot, merged=merged)


def merge_counter(t_tot, tot_tot, t_nim, tot_nim, cfg: NimConfig | None = None, *,
                  nim_only: bool = True, **kw) -> Merged:
    """The merged hits of :func:`pair_counter` (same arguments, the same
    required keywords): ascending times, ToT, class and carrier;
    ``nim_only=False`` leaves the NIM-only hits out."""
    return pair_counter(t_tot, tot_tot, t_nim, tot_nim, cfg, **kw).merged_hits(nim_only)


# ==============================================================================
# The NIM fine-time lag (PISMAFineOffset.hh / PISMAWord.hh, measure only)
# ==============================================================================

@dataclass
class LagVote:
    """One NIM channel's lag in one frame (:func:`lag_vote`).

    Lags are signed in [-2^19, 2^19) ns: the median of a window's d values
    (``SignedLag``). ``shift_ns`` is ``LagShiftNs``: ``lag - nominal`` folded
    onto the 2^20 circle when that lies beyond ``lag_tol_ns``, else 0; it is
    what reco would remove, and this DQM never does.
    """

    state: str                       # LAG_STATES
    lag_ns: int | None               # the voted lag; None without a vote
    shift_ns: int                    # 0 unless "faulted"
    n_pairs: int                     # NIM words with a same-coarse S1 word
    n_best: int
    n_runner_up: int
    best_lag_ns: int | None          # the fullest window's lag, voted or not
    runner_up_lag_ns: int | None
    d: np.ndarray                    # uint32: (fine - fine_S1) mod 2^20, stream order

    @property
    def faulted(self) -> bool:
        return self.state == "faulted"


def signed_lag(d):
    """``d`` mod 2^20 as a signed lag in [-2^19, 2^19) (``SignedLag``); int64."""
    v = np.asarray(d, dtype=np.int64) & _FINE_MASK
    return np.where(v >= _HALF_FINE, v - FINE_WRAP_NS, v)


def lag_shift_ns(lag_ns, nominal_ns, tol_ns) -> int:
    """``LagShiftNs``: the circular ``lag - nominal`` when it exceeds ``tol_ns``, else 0."""
    if lag_ns is None:
        return 0
    wrapped = int(signed_lag(int(lag_ns) - int(nominal_ns)))
    return wrapped if abs(wrapped) > tol_ns else 0


def lag_pairs(ch, coarse, fine, nim_ch, s1_ch=1, next_reference=True, *,
              ref_idx=None, nim_idx=None) -> np.ndarray:
    """d = (fine - fine_S1) mod 2^20 of channel ``nim_ch``'s words (uint32, stream order).

    Stream-order arrays of every trigger word of the bank (``Frame.ch``,
    ``.coarse``, ``.fine``). A NIM word pairs with the last S1 word before it
    in stream order when that word's raw coarse field equals its own; else,
    with ``next_reference``, with the first S1 word after it when *its* coarse
    field is equal (``Scan``'s ``haveRef``/``nextRef``); else it has no pair.
    It is not "the nearest S1 word with an equal coarse field": an earlier S1
    word of the same coarse field behind a later different one is not used.

    ``ref_idx``/``nim_idx``: the stream positions of the S1 and of the NIM
    words (ascending), when the caller has them (``ch`` is not read then):
    the same result without scanning the whole bank once per channel.
    """
    coarse = np.asarray(coarse)
    fine = np.asarray(fine)
    ref = np.flatnonzero(np.asarray(ch) == s1_ch) if ref_idx is None else np.asarray(ref_idx)
    nim = np.flatnonzero(np.asarray(ch) == nim_ch) if nim_idx is None else np.asarray(nim_idx)
    if ref.size == 0 or nim.size == 0:
        return np.zeros(0, dtype=np.uint32)
    k = np.searchsorted(ref, nim)        # S1 words before each NIM word
    c_nim = coarse[nim]
    prev = ref[np.maximum(k - 1, 0)]
    use = np.where((k > 0) & (coarse[prev] == c_nim), prev, -1)
    if next_reference:
        nxt = ref[np.minimum(k, ref.size - 1)]
        ok = (use < 0) & (k < ref.size) & (coarse[nxt] == c_nim)
        use = np.where(ok, nxt, use)
    m = use >= 0
    d = fine[nim[m]].astype(np.int64) - fine[use[m]].astype(np.int64)
    return (d & _FINE_MASK).astype(np.uint32)


def lag_modes(d, width_ns):
    """``FindLagModes``: the fullest window of width ``width_ns`` over the d
    values (across the 2^20 wrap) and the fullest one sharing no value with it.

    A window starts at a value v and holds the values u with ``0 <= (u - v)
    mod 2^20 <= width_ns``. Returns ``(n_best, best_lag, n_runner_up,
    runner_up_lag)``: counts and the median (lower middle) of each window as a
    signed lag, None for an empty one. A tie goes to the window starting at
    the lowest value. The runner-up starts more than ``width_ns`` from the
    best window's start on either side.
    """
    s = np.sort(np.asarray(d, dtype=np.int64) & _FINE_MASK)
    n = s.size
    if n == 0:
        return 0, None, 0, None
    width = int(width_ns)
    ext = np.concatenate([s, s + FINE_WRAP_NS])
    idx = np.arange(n)
    j = np.minimum(np.searchsorted(ext, s + width, side="right"), idx + n)
    count = j - idx

    def median(first, c):
        return int(signed_lag(ext[first + (c - 1) // 2]))

    b = int(np.argmax(count))           # the first of the fullest
    n_best = int(count[b])
    rel = (s - s[b]) & _FINE_MASK
    allowed = (rel > width) & (rel < FINE_WRAP_NS - width)
    if not allowed.any():
        return n_best, median(b, n_best), 0, None
    c2 = np.where(allowed, count, 0)
    r = int(np.argmax(c2))
    return n_best, median(b, n_best), int(c2[r]), median(r, int(c2[r]))


def decide_lag(d, cfg: NimConfig | None = None, nominal_ns=None) -> LagVote:
    """The frame's lag from its d values (``DecideLagChannel`` without the
    fallback): a vote when ``n_best >= lag_min_pairs`` and ``n_best >=
    lag_dominance * n_runner_up``; "faulted" when the voted lag lies more than
    ``lag_tol_ns`` from the channel's nominal delay on the 2^20 circle.

    ``nominal_ns``: this NIM channel's nominal delay from S1 (the C++
    ``lagNominalNs[c]``, one per lag channel: d holds the S_k-to-S1 cable delay
    and time of flight); None takes ``cfg.lag_nominal_ns``.
    """
    cfg = cfg or NimConfig()
    nominal = _check_nominal(cfg.lag_nominal_ns if nominal_ns is None else nominal_ns)
    d = np.asarray(d, dtype=np.uint32)
    width = int(2.0 * cfg.lag_tol_ns)       # static_cast<uint32_t>: truncates
    n_best, best, n_run, run = lag_modes(d, width)
    vote = n_best >= int(cfg.lag_min_pairs) and n_best >= cfg.lag_dominance * n_run
    lag = best if vote else None
    shift = lag_shift_ns(lag, nominal, cfg.lag_tol_ns)
    if vote:
        state = "faulted" if shift != 0 else "ok"
    else:
        state = "ambiguous" if n_best >= int(cfg.lag_min_pairs) else "none"
    return LagVote(state=state, lag_ns=lag, shift_ns=shift, n_pairs=int(d.size),
                   n_best=n_best, n_runner_up=n_run, best_lag_ns=best,
                   runner_up_lag_ns=run, d=d)


def lag_vote(ch, coarse, fine, nim_ch, s1_ch=1, cfg: NimConfig | None = None,
             nominal_ns=None, *, ref_idx=None, nim_idx=None) -> LagVote:
    """:func:`lag_pairs` then :func:`decide_lag`: one NIM channel's lag in one
    frame, against that channel's ``nominal_ns`` (None: ``cfg.lag_nominal_ns``);
    ``ref_idx``/``nim_idx`` as for :func:`lag_pairs`."""
    cfg = cfg or NimConfig()
    return decide_lag(lag_pairs(ch, coarse, fine, nim_ch, s1_ch, cfg.lag_next_reference,
                                ref_idx=ref_idx, nim_idx=nim_idx), cfg, nominal_ns)


__all__ = [
    "HAS_TOT", "HAS_NIM", "NIM_EXPECTED", "INCOMPLETE", "TIME_FROM_NIM", "TOT_SUBSTITUTED",
    "MULTI_CANDIDATE", "IN_TOT_SHADOW", "NEAR_FRAME_EDGE", "ECHO", "FLAG_NAMES",
    "PAIRED", "TOT_ONLY", "ECHO_WORD", "NIM_ONLY", "CLASS_NAMES", "TIME_SOURCES",
    "LAG_STATES", "NimConfig", "Merged", "PairResult", "LagVote",
    "HALF_FINE_WRAP_NS", "greedy_match", "greedy_match_by_dt", "echo_mask", "nearest_dt",
    "wide_dt", "pair_counter", "merge_counter",
    "signed_lag", "lag_shift_ns", "lag_pairs", "lag_modes", "decide_lag", "lag_vote",
]
