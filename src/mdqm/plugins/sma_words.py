"""SMA readout words -> per-frame arrays: everything the SMA DQM computes, free of MIDAS.

Pure numpy. The SMA plugin (``sma.py``), the offline CLI and the tests all call
these functions, so the daemon and the manual path cannot drift apart.

Provenance
----------
Vendored, not imported: psm-analysis is a personal analysis checkout, not an
installable package. ``tests/test_sma_golden.py`` pins the vendored functions
against the originals on stored real frames; fix a bug in both places.

* From ``psm-analysis-josh-2026/sma-tot-vs-wd/sma_reader.py``: :func:`decode`
  (``decode``), :func:`time_of` (``time_of``), :func:`encode` (``encode``),
  :func:`gate_veto` (``gate_veto``), :func:`burst_phase_many`
  (``burst_phase_many``, with ``_valid_n``), :func:`match_to_gates`
  (``match_to_gates``) and :func:`pulse_offset` (``pulse_offset``). The RF
  rule and its constants are ``sma_wd_constants.py`` and
  ``scratch/sma-tot-vs-wd/BURST_RULE.md``.
* From ``main/reco_testbeam/pi_midas/include/PISMAWord.hh``:
  :func:`shared_bits` (``SharedBits``), :func:`wrap_signed` (``WrapSigned``),
  :func:`fine_coarse_xor` (``FineCoarseXor``), :func:`fine_coarse_diff_ns`
  (``FineCoarseDiffNs``), :func:`extent` (``Extent``) and :func:`gap_ns`
  (``GapNs``). ``tests/test_sma_words.py`` repeats the vectors of
  ``reco_testbeam/tests/test_psm_sma.cpp`` (``test_fine_coarse_diff``).

New here (no reference): the stale-hit cluster rule (with the rescue of far
consistent clusters), the shift scan, the coincidence windows and the seed
selection.

Conventions
-----------
* A word is ``uint64``. Bit 63 set: an SMA trigger word; clear: a MuPix pixel
  word; all ones (:data:`FILLER`): padding. Trigger word fields: channel 59:56,
  ToT 55:48, coarse 47:20, fine 19:0.
* Times are integer ns in the MuPix epoch: ``fine + 2^20 * round(((coarse <<
  shift) - fine) / 2^20)`` masked to 2^40 ns (:func:`time_of`). Within one
  frame they are then *unwrapped* around the frame's median time
  (:func:`unwrap_near`) modulo 2^:func:`time_bits` (40, or 28 + shift below
  shift 12), so a frame that straddles the wrap still sorts and subtracts
  correctly; away from the wrap this changes nothing.
* "Stream order" is the order of the trigger words in the bank; "sorted" is
  the kept (non-stale) hits in ascending time, ties in stream order.
* dt is always ``t_other - t_reference``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

# --- the word ------------------------------------------------------------------

#: The FEB pads its buffer with this. Bit 63 is set, so it would otherwise
#: decode as a trigger word on channel 15.
FILLER = 0xFFFFFFFFFFFFFFFF
#: MIDAS event id of the musip readout frames, and their bank.
EVID_READOUT = 301
BANK = "H000"
N_CHANNELS = 16
N_TOT = 256
FINE_BITS = 20
FINE_WRAP_NS = 1 << FINE_BITS
COARSE_BITS = 28
#: The MuPix time is 37 bits of 8 ns: the SMA time is masked to the same 2^40 ns.
TIME_BITS = 40
TIME_WRAP_NS = 1 << TIME_BITS
#: The coarse field is the time in ns shifted right by this (a truncation). It
#: is a per-run setting of the board, not a constant: 14 on every run since the
#: 125 ns RF gate firmware, but 3 and 15 have been seen.
DEFAULT_SHIFT = 14
#: The rounding in :func:`time_of` needs the truncation below 2^19 ns.
MAX_COARSE_SHIFT = 18

# --- defaults of the DQM cuts (ODB /DQM/SMA/Cuts overrides them) ---------------

#: A word is fine/coarse consistent when ``|diff| <= 2^shift + margin``: the
#: coarse field may be latched one coarse tick away from the fine one. At shift
#: 14 this is 20000 ns, the reco default ``smaDiagLatchToleranceNs``.
LATCH_MARGIN_NS = 3616
#: ...but never less than this: the reco default (``smaDiagLatchToleranceNs``).
#: At shift 3 the coarse field is latched up to ~20 us after the fine one, so
#: 2^3 + margin rejected three quarters of the genuine words of a shift-3 run.
MIN_LATCH_TOLERANCE_NS = 20000
#: Stale-hit rule: a gap larger than this splits the frame into clusters.
#: 50 ms: genuine hits of one frame were never measured more than ~4 ms apart
#: (low-rate runs behind a thick degrader), while words left over from an
#: earlier run sit 180 ms to minutes away. At 1 ms a low-rate frame was mostly
#: thrown away.
STALE_GAP_NS = 50_000_000
#: A cluster other than the median one is kept after all when it has at least
#: this many words and at least this fraction of them is fine/coarse consistent
#: at the best scanned shift: beam trips and slow buffers put genuine hits
#: 0.05-8 s away (75-85 % consistent), a replayed old buffer is <= 3 %.
RESCUE_MIN_WORDS = 20
RESCUE_MIN_FRACTION = 0.5
#: ToT codes from here up are not a plausible counter width (254/255 markers).
TOT_CORRUPT_MIN = 250
#: RF burst rule (BURST_RULE.md, "last" rule with the 125 ns gate and S1 veto).
RF_GATE_NS = 125.0
RF_MIN_PULSES = 2
RF_MAX_PULSES = 4
RF_BURST_SIZE = 4
RF_PULSE_INDEX = 2
RF_RULE = "last"
RF_RULES = ("last", "dqm")
#: A partner borrows the phase of the nearest valid S1 gate within this.
RF_COINC_NS = 50.0
#: Coincidence pattern window and the dt histogram window, around each S1 hit.
COINC_NS = 50
DT_WINDOW_NS = 200
#: Delayed channels: all pairs with an S1 hit in [lo, hi].
DELAYED_LO_NS = -1_000
DELAYED_HI_NS = 10_000
#: Event display seeds: the latest N S1 hits whose [t - pre, t + post] is complete.
SEED_PRE_NS = 200
SEED_POST_NS = 3_000
N_SEEDS = 4
#: Shifts the self-check tries.
#: 3 is the shift of the runs before the RF-gate firmware; 12-16 bracket 14.
SHIFT_SCAN = (3, 12, 13, 14, 15, 16)
#: Seeds: channels with fewer hits than this in a frame do not limit the
#: complete range (a sparse counter would otherwise leave no seed at all).
SEED_MIN_HITS = 5

_U = np.uint64


# ==============================================================================
# Words
# ==============================================================================

def words_from_bank(data) -> np.ndarray:
    """The H000 bank as a little-endian ``uint64`` array (a view when possible).

    H000 is a ``TID_DWORD`` bank (bank32a), so ``midas`` hands it over as a
    ``uint32`` array: each word is two consecutive dwords, low half first.
    Accepts that, a ``uint64`` array, or ``bytes``/``bytearray``/``memoryview``.
    A trailing odd dword (a broken bank) is dropped, as the reference reader
    drops a trailing partial word.
    """
    if isinstance(data, np.ndarray):
        a = np.ascontiguousarray(data)
        if a.dtype.itemsize == 8 and a.dtype.kind in "ui":
            return a.view("<u8")
        b = a.view(np.uint8).reshape(-1)
    else:
        b = np.frombuffer(data, dtype=np.uint8)
    n = b.size // 8
    return b[: n * 8].view("<u8")


def word_counts(words) -> tuple[int, int, int]:
    """``(n_filler, n_pixel, n_trigger)`` of one bank; they add up to its length."""
    w = np.asarray(words, dtype=_U)
    n_filler = int(np.count_nonzero(w == _U(FILLER)))
    n_high = int(np.count_nonzero(w.view(np.int64) < 0))     # bit 63 set
    return n_filler, int(w.size) - n_high, n_high - n_filler


def time_of(coarse, fine, shift=DEFAULT_SHIFT) -> np.ndarray:
    """The SMA time in ns, in the MuPix epoch (``sma_reader.time_of``).

    ``fine + 2^20 * round(((coarse << shift) - fine) / 2^20)`` (round half up,
    in integers), masked to 2^40 ns. Note: ``PISMAWord::TimeNs`` masks to
    2^min(28 + shift, 40) instead; the two agree for shift >= 12, which is every
    run this DQM reads. Below that the C++ time wraps with the coarse field.
    """
    c = np.asarray(coarse, dtype=np.int64) << int(shift)
    f = np.asarray(fine, dtype=np.int64)
    k = (c - f + (FINE_WRAP_NS >> 1)) >> FINE_BITS
    return (f + (k << FINE_BITS)) & np.int64(TIME_WRAP_NS - 1)


def decode(words, shift=DEFAULT_SHIFT) -> dict:
    """The trigger words of one frame as arrays, in stream order (``sma_reader.decode``).

    Returns ``{ch int16, tot int16, fine uint32, coarse uint32, time int64,
    n_filler, n_pixel, n_trigger}``; the arrays have one entry per trigger
    word (filler and pixel words dropped).
    """
    w = np.asarray(words, dtype=_U)
    filler = w == _U(FILLER)
    trig = (w.view(np.int64) < 0) & ~filler
    t = w[trig]
    ch = ((t >> _U(56)) & _U(0xF)).astype(np.int16)
    tot = ((t >> _U(48)) & _U(0xFF)).astype(np.int16)
    coarse = ((t >> _U(20)) & _U(0xFFFFFFF)).astype(np.uint32)
    fine = (t & _U(0xFFFFF)).astype(np.uint32)
    n_filler = int(np.count_nonzero(filler))
    return {"ch": ch, "tot": tot, "fine": fine, "coarse": coarse,
            "time": time_of(coarse, fine, shift),
            # Where each trigger word sits in the bank (64-bit words, filler
            # and pixel words counted), and the word itself: to find it again.
            "word_index": np.flatnonzero(trig).astype(np.uint32), "raw": t,
            "n_filler": n_filler, "n_pixel": int(w.size) - n_filler - int(t.size),
            "n_trigger": int(t.size)}


def encode(ch, tot, coarse, fine):
    """Inverse of :func:`decode`: one word from its fields (scalars or arrays).

    Scalars give a Python ``int`` (as ``sma_reader.encode``), arrays a ``uint64``
    array.
    """
    if all(np.isscalar(x) for x in (ch, tot, coarse, fine)):
        return ((1 << 63) | ((int(ch) & 0xF) << 56) | ((int(tot) & 0xFF) << 48)
                | ((int(coarse) & 0xFFFFFFF) << 20) | (int(fine) & 0xFFFFF))
    ch, tot, coarse, fine = (np.asarray(x).astype(_U) for x in (ch, tot, coarse, fine))
    return (_U(1 << 63) | ((ch & _U(0xF)) << _U(56)) | ((tot & _U(0xFF)) << _U(48))
            | ((coarse & _U(0xFFFFFFF)) << _U(20)) | (fine & _U(0xFFFFF)))


def fields_of(t_ns, shift=DEFAULT_SHIFT):
    """``(coarse, fine)`` a healthy board writes for true time ``t_ns`` (test helper)."""
    t = np.asarray(t_ns, dtype=np.int64)
    return ((t >> int(shift)) & 0xFFFFFFF).astype(np.uint32), (t & 0xFFFFF).astype(np.uint32)


# ==============================================================================
# Fine vs coarse (PISMAWord.hh)
# ==============================================================================

def shared_bits(shift) -> int:
    """How many fine bits the coarse field repeats: bits shift..19 (``SharedBits``)."""
    shift = int(shift)
    return 0 if shift >= FINE_BITS else FINE_BITS - shift


def wrap_signed(d, bits):
    """``d`` modulo 2^bits into [-2^(bits-1), 2^(bits-1)) (``WrapSigned``); int64."""
    d = np.asarray(d, dtype=np.int64)
    span = np.int64(1) << np.int64(bits)
    half = span >> np.int64(1)
    return ((d + half) & (span - np.int64(1))) - half


def fine_coarse_xor(coarse, fine, shift=DEFAULT_SHIFT) -> np.ndarray:
    """Fine bits shift..19 that disagree with the coarse field (``FineCoarseXor``).

    Bit i of the result stands for fine bit ``shift + i``. ``uint32``.
    """
    s = int(shift)
    mask = np.uint32((1 << shared_bits(s)) - 1)
    c = np.asarray(coarse, dtype=np.uint32)
    f = np.asarray(fine, dtype=np.uint32)
    return ((f >> np.uint32(s)) ^ c) & mask


def fine_coarse_diff_ns(coarse, fine, shift=DEFAULT_SHIFT) -> np.ndarray:
    """Coarse minus fine over the bits they share, in ns (``FineCoarseDiffNs``).

    A multiple of 2^shift in [-2^19, 2^19): 0 on a word latched in one instant,
    +-2^b for a flipped fine bit b, +2^shift for a coarse field one tick ahead.
    ``int64``.
    """
    s = int(shift)
    n = shared_bits(s)
    mask = np.int64((1 << n) - 1)
    c = np.asarray(coarse, dtype=np.int64)
    f = np.asarray(fine, dtype=np.int64)
    return wrap_signed((c & mask) - ((f >> s) & mask), n) << np.int64(s)


def latch_tolerance_ns(shift=DEFAULT_SHIFT, margin_ns=LATCH_MARGIN_NS) -> int:
    """The largest |diff| still called consistent: one coarse tick plus the
    margin, and never less than :data:`MIN_LATCH_TOLERANCE_NS` (20000 ns at
    shift 14 either way)."""
    return max((1 << int(shift)) + int(margin_ns), MIN_LATCH_TOLERANCE_NS)


def fine_coarse_consistent(diff_ns, shift=DEFAULT_SHIFT, margin_ns=LATCH_MARGIN_NS):
    """``|diff| <= latch_tolerance_ns`` (``FineCoarseConsistent`` with that tolerance)."""
    return np.abs(np.asarray(diff_ns, dtype=np.int64)) <= latch_tolerance_ns(shift, margin_ns)


def shift_scan(coarse, fine, shifts=SHIFT_SCAN, margin_ns=LATCH_MARGIN_NS) -> np.ndarray:
    """Words consistent at each trial shift: ``int64[len(shifts)]``.

    At the wrong shift the coarse bits and the fine bits they are compared
    with are different bits of the time, so only a random few agree.
    """
    c = np.asarray(coarse, dtype=np.uint32)
    f = np.asarray(fine, dtype=np.uint32)
    return np.array([int(np.count_nonzero(
        fine_coarse_consistent(fine_coarse_diff_ns(c, f, s), s, margin_ns)))
        for s in shifts], dtype=np.int64)


def per_channel_bit_counts(ch, values, nbits=FINE_BITS, n_channels=N_CHANNELS) -> np.ndarray:
    """``int64[n_channels, nbits]``: per channel, how many ``values`` have bit b set.

    Used for the fine-bit occupancy (``values`` = fine) and, on the mismatched
    words, for which fine bits differ (``values`` = xor; bit i = fine bit shift + i).
    """
    # One bincount per byte of the value, keyed by (channel, byte value), then
    # the 256 byte values times their 8 bits: three passes over the words
    # instead of one per bit.
    ch = np.asarray(ch).astype(np.intp)
    v = np.asarray(values, dtype=np.uint32)
    nbytes = (nbits + 7) // 8
    out = np.zeros((n_channels, 8 * nbytes), dtype=np.int64)
    if v.size == 0:
        return out[:, :nbits]
    key0 = ch << 8
    for k in range(nbytes):
        byte = ((v >> np.uint32(8 * k)) & np.uint32(0xFF)).astype(np.intp)
        h = np.bincount(key0 + byte, minlength=n_channels * 256)[: n_channels * 256]
        out[:, 8 * k: 8 * k + 8] = h.reshape(n_channels, 256) @ _BYTE_BITS
    return out[:, :nbits]


#: Bit b of every byte value: ``_BYTE_BITS[value, b]``.
_BYTE_BITS = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1,
                           bitorder="little").astype(np.int64)


# ==============================================================================
# Stale hits, extent and gap
# ==============================================================================

def time_bits(shift=DEFAULT_SHIFT) -> int:
    """Bits of time the words carry: the coarse field's 28 bits above ``shift``,
    at most :data:`TIME_BITS` (``PISMAWord::TimeBits``). Below shift 12 the time
    wraps with the coarse field -- every 2^31 ns = 2.1 s at shift 3 -- and a
    frame can straddle that wrap."""
    return min(COARSE_BITS + int(shift), TIME_BITS)


def unwrap_near(times, ref, bits=TIME_BITS) -> np.ndarray:
    """``times`` moved by whole 2^bits ns to within half a wrap of ``ref`` (int64).

    With ``bits`` below 40 this also folds :func:`time_of`'s 2^40 mask, which
    is a multiple of 2^bits."""
    t = np.asarray(times, dtype=np.int64)
    return np.int64(ref) + wrap_signed(t - np.int64(ref), bits)


def _clusters(times, gap_ns, bits=TIME_BITS):
    """``(order, bounds, k_med, u)``: argsort of the unwrapped times ``u``, the
    cluster boundaries in sorted order (cluster k is ``[bounds[k], bounds[k+1])``)
    and the index of the cluster holding the median."""
    t = np.asarray(times, dtype=np.int64)
    n = t.size
    if n == 0:
        return np.zeros(0, dtype=np.intp), np.zeros(2, dtype=np.intp), 0, t
    ref = np.partition(t, (n - 1) // 2)[(n - 1) // 2]
    u = unwrap_near(t, ref, bits)
    order = np.argsort(u, kind="stable")
    us = u[order]
    split = np.flatnonzero(np.diff(us) > gap_ns) + 1         # first index of each new cluster
    bounds = np.r_[0, split, n].astype(np.intp)
    k = int(np.searchsorted(split, (n - 1) // 2, side="right"))
    return order, bounds, k, u


def _stale_sorted(times, gap_ns):
    """``(order, n_keep_lo, n_keep_hi, u)``: argsort of the unwrapped times ``u``
    and the slice [lo, hi) of it that is the cluster holding the median."""
    order, bounds, k, u = _clusters(times, gap_ns)
    return order, int(bounds[k]), int(bounds[k + 1]), u


def stale_mask(times, gap_ns=STALE_GAP_NS) -> np.ndarray:
    """Keep-mask (stream order) of the hits in the frame's main time cluster.

    Times only: :func:`prepare_frame`, which has the fine and coarse fields,
    additionally keeps other clusters that are fine/coarse consistent.

    Sort the times, split wherever consecutive hits are more than ``gap_ns``
    apart, and keep the cluster holding the median time (the lower middle hit
    for an even count). The rest are stale: words left in the buffer from
    before a run start or a clock restart, seconds to minutes away.
    """
    order, lo, hi, _u = _stale_sorted(times, gap_ns)
    keep = np.zeros(order.size, dtype=bool)
    keep[order[lo:hi]] = True
    return keep


@dataclass(frozen=True)
class Extent:
    """First and last time of a frame in ns (``PISMAWord::FrameExtent``)."""

    first: int = 0
    last: int = 0

    @property
    def span_ns(self) -> int:
        return self.last - self.first


def extent(times, bits=TIME_BITS) -> Extent:
    """Earliest and latest of ``times``, safe across the wrap (``PISMAWord::Extent``).

    Offsets are taken with :func:`wrap_signed` from the first time, so a frame
    straddling the wrap gets a ``last`` beyond 2^bits instead of a span of
    nearly the whole range. Empty ``times`` give ``Extent(0, 0)``.
    """
    t = np.asarray(times, dtype=np.int64)
    if t.size == 0:
        return Extent()
    t0 = int(t[0])
    d = wrap_signed(t - t0, bits)
    return Extent(t0 + min(0, int(d.min())), t0 + max(0, int(d.max())))


def gap_ns(prev: Extent, nxt: Extent, bits=TIME_BITS) -> int:
    """First time of ``nxt`` minus last of ``prev``, wrap-safe; negative on overlap (``GapNs``)."""
    return int(wrap_signed(nxt.first - prev.last, bits))


# ==============================================================================
# The RF burst rule (sma_reader.py, BURST_RULE.md)
# ==============================================================================

def _valid_n(n, rule, min_pulses=RF_MIN_PULSES, max_pulses=RF_MAX_PULSES,
             burst_size=RF_BURST_SIZE):
    if rule == "last":
        return (n >= min_pulses) & (n <= max_pulses)
    if rule == "dqm":
        return n == burst_size
    raise ValueError(f"RF rule {rule!r}; 'last' or 'dqm'")


def gate_veto(t_gates, gate_ns=RF_GATE_NS):
    """``(vetoed, gap_ns)`` per gate (``sma_reader.gate_veto``).

    ``t_gates`` are EVERY gate (S1) hit of the frame, any order. ``gap`` is
    the time to the next strictly later hit (inf when none); a gate is vetoed
    when that hit is inside it, 0 < gap <= gate_ns. Equal times do not veto.
    """
    g = np.asarray(t_gates, dtype=np.float64)
    gs = np.sort(g)
    j = np.searchsorted(gs, g, side="right")
    nxt = np.full(g.shape, np.inf)
    later = j < gs.size
    nxt[later] = gs[j[later]]
    gap = nxt - g
    return gap <= gate_ns, gap


def burst_phase_many(rf_sorted, t_gates, rule=RF_RULE, gate_ns=RF_GATE_NS,
                     min_pulses=RF_MIN_PULSES, max_pulses=RF_MAX_PULSES,
                     burst_size=RF_BURST_SIZE, pulse_index=RF_PULSE_INDEX, vetoed=None):
    """The RF phase of every gate (S1) hit (``sma_reader.burst_phase_many``).

    Pulses of a gate at t: RF times in (t, t + gate_ns]. ``rule="last"``: valid
    with ``min_pulses``..``max_pulses`` pulses and not vetoed (:func:`gate_veto`);
    phase = last pulse - t, period = last - second-to-last pulse. ``"dqm"``:
    exactly ``burst_size`` pulses, phase from pulse ``pulse_index``.

    ``rf_sorted`` ascending; ``t_gates`` must be every S1 hit of the frame --
    unless ``vetoed`` (one bool per gate) is given: the veto needs every S1 hit,
    the rest does not, so a caller analysing a subsample of the S1 hits computes
    :func:`gate_veto` on all of them and passes the subsample's entries.
    Returns ``(n int32, valid bool, phase f8, period f8, lo intp, pulse_t f8)``,
    one entry per gate; ``lo`` indexes each gate's first pulse in ``rf_sorted``,
    ``pulse_t`` is the phase pulse's time; phase/period/pulse_t NaN if invalid.
    """
    rf = np.asarray(rf_sorted, dtype=np.float64)
    g = np.asarray(t_gates, dtype=np.float64)
    lo = np.searchsorted(rf, g, side="right")
    hi = np.searchsorted(rf, g + gate_ns, side="right")
    n = (hi - lo).astype(np.int32)
    if vetoed is None:
        vetoed, _gap = gate_veto(g, gate_ns)
    valid = _valid_n(n, rule, min_pulses, max_pulses, burst_size) & ~vetoed
    phase = np.full(g.shape, np.nan)
    period = np.full(g.shape, np.nan)
    pulse_t = np.full(g.shape, np.nan)
    if valid.any():
        li, hv = lo[valid], hi[valid]
        k = (hv - 1) if rule == "last" else (li + pulse_index)
        pulse_t[valid] = rf[k]
        phase[valid] = pulse_t[valid] - g[valid]
        period[valid] = rf[hv - 1] - rf[hv - 2]
    return n, valid, phase, period, lo, pulse_t


def pulse_offset(rf_sorted, t_gates, n, lo, j):
    """Offset of pulse ``j`` of each gate, NaN where it has <= j pulses.

    ``sma_reader.pulse_offset``.
    """
    out = np.full(np.shape(t_gates), np.nan)
    sel = np.asarray(n) > j
    if sel.any():
        out[sel] = (np.asarray(rf_sorted, float)[np.asarray(lo)[sel] + j]
                    - np.asarray(t_gates, float)[sel])
    return out


def match_to_gates(t_hits, gate_t, gate_pulse_t, coinc_ns=RF_COINC_NS):
    """Nearest VALID gate within ``coinc_ns`` of each hit, ties to the earlier
    (``sma_reader.match_to_gates``).

    ``gate_t`` sorted valid gate times, ``gate_pulse_t`` their phase-pulse
    times. Returns ``(idx int64, dt f8, phase f8)``; idx = -1 and NaN when
    unmatched; dt = t_hit - t_gate, phase = pulse - t_hit.
    """
    t = np.asarray(t_hits, dtype=np.float64)
    gt = np.asarray(gate_t, dtype=np.float64)
    idx = np.full(t.shape, -1, dtype=np.int64)
    dt = np.full(t.shape, np.nan)
    phase = np.full(t.shape, np.nan)
    if gt.size == 0 or t.size == 0:
        return idx, dt, phase
    i = np.searchsorted(gt, t, side="left")
    a = np.clip(i - 1, 0, gt.size - 1)
    b = np.clip(i, 0, gt.size - 1)
    da = np.where(i > 0, np.abs(t - gt[a]), np.inf)
    db = np.where(i < gt.size, np.abs(t - gt[b]), np.inf)
    pick = np.where(da <= db, a, b)
    ok = np.minimum(da, db) <= coinc_ns
    idx[ok] = pick[ok]
    dt[ok] = t[ok] - gt[pick[ok]]
    phase[ok] = np.asarray(gate_pulse_t, dtype=np.float64)[pick[ok]] - t[ok]
    return idx, dt, phase


# ==============================================================================
# Coincidences and seeds (searchsorted on per-channel sorted times)
# ==============================================================================

def _bounds(t_ref, t_other, lo_ns, hi_ns):
    t_ref = np.asarray(t_ref)
    t_other = np.asarray(t_other)
    a = np.searchsorted(t_other, t_ref + lo_ns, side="left")
    b = np.searchsorted(t_other, t_ref + hi_ns, side="right")
    return a, b


def window_counts(t_ref, t_other, lo_ns, hi_ns) -> np.ndarray:
    """Per reference hit, how many ``t_other`` have ``lo <= t_o - t_ref <= hi`` (intp).

    Both inputs ascending. Pass integer windows to keep int64 times exact.
    """
    a, b = _bounds(t_ref, t_other, lo_ns, hi_ns)
    return b - a


def window_pairs(t_ref, t_other, lo_ns, hi_ns):
    """Every pair with ``lo <= t_o - t_ref <= hi``: ``(i_ref, j_other, dt)``.

    Both inputs ascending; pairs come grouped by reference hit, ``t_o``
    ascending within a group. ``dt = t_other[j] - t_ref[i]``.
    """
    t_ref = np.asarray(t_ref)
    t_other = np.asarray(t_other)
    a, b = _bounds(t_ref, t_other, lo_ns, hi_ns)
    n = b - a
    total = int(n.sum())
    if total == 0:
        e = np.zeros(0, dtype=np.intp)
        return e, e.copy(), np.zeros(0, dtype=np.result_type(t_ref, t_other))
    i = np.repeat(np.arange(t_ref.size), n)
    start = np.cumsum(n) - n
    j = np.arange(total) - np.repeat(start - a, n)
    return i, j, t_other[j] - t_ref[i]


def coincidence(t_seed, counters: Sequence, window_ns=COINC_NS):
    """Pattern and partner counts of each seed hit.

    ``counters[k]`` are the ascending times of counter k (S1..S5 by default,
    so S1 itself is counter 0 and its bit is always set for an S1 seed).
    Returns ``(pattern uint8, counts int32[n_seed, n_counters])``; bit k of
    ``pattern`` is set when counter k has a hit within ``|dt| <= window_ns``,
    ``counts`` how many (the seed itself included when it is in ``counters``).
    """
    t_seed = np.asarray(t_seed)
    counts = np.zeros((t_seed.size, len(counters)), dtype=np.int32)
    pattern = np.zeros(t_seed.size, dtype=np.uint8)
    for k, tk in enumerate(counters):
        c = window_counts(t_seed, tk, -window_ns, window_ns)
        counts[:, k] = c
        pattern |= (c > 0).astype(np.uint8) << np.uint8(k)
    return pattern, counts


def complete_range(channel_times: Sequence):
    """``(start, end)`` covered by every non-empty channel: the latest first hit
    and the earliest last hit. ``None`` when all are empty."""
    firsts = [int(t[0]) for t in channel_times if len(t)]
    if not firsts:
        return None
    lasts = [int(t[-1]) for t in channel_times if len(t)]
    return max(firsts), min(lasts)


def select_seeds(t_s1, start, end, n_seeds=N_SEEDS, pre_ns=SEED_PRE_NS,
                 post_ns=SEED_POST_NS) -> np.ndarray:
    """Indices (ascending) into ascending ``t_s1`` of the latest ``n_seeds`` hits
    whose window [t - pre, t + post] lies inside [start, end] (:func:`complete_range`
    of the counter and RF channels), i.e. no channel's data ends inside it."""
    t = np.asarray(t_s1)
    hi = np.searchsorted(t, end - post_ns, side="right")    # t + post <= end
    lo = np.searchsorted(t, start + pre_ns, side="left")    # t - pre >= start
    return np.arange(max(lo, hi - int(n_seeds)), hi, dtype=np.intp) if hi > lo \
        else np.zeros(0, dtype=np.intp)


# ==============================================================================
# One frame
# ==============================================================================

@dataclass
class Roles:
    """Which SMA channel is what (ODB ``/DQM/SMA/Channel roles``)."""

    s1: int = 1
    counters: tuple = (1, 2, 3, 4, 5)
    rf: int = 6
    current: int = 7
    delayed: tuple = (8, 9, 10)


@dataclass
class Cuts:
    """The per-frame cuts (ODB ``/DQM/SMA/Cuts``), defaults as above."""

    latch_margin_ns: int = LATCH_MARGIN_NS
    stale_gap_ns: int = STALE_GAP_NS
    tot_corrupt_min: int = TOT_CORRUPT_MIN
    rf_rule: str = RF_RULE
    rf_gate_ns: float = RF_GATE_NS
    rf_min_pulses: int = RF_MIN_PULSES
    rf_max_pulses: int = RF_MAX_PULSES
    coinc_ns: int = COINC_NS
    dt_window_ns: int = DT_WINDOW_NS
    delayed_lo_ns: int = DELAYED_LO_NS
    delayed_hi_ns: int = DELAYED_HI_NS
    seed_pre_ns: int = SEED_PRE_NS
    seed_post_ns: int = SEED_POST_NS
    n_seeds: int = N_SEEDS
    seed_min_hits: int = SEED_MIN_HITS
    shift_scan: tuple = SHIFT_SCAN
    rescue_min_words: int = RESCUE_MIN_WORDS
    rescue_min_fraction: float = RESCUE_MIN_FRACTION
    #: At most this many S1 hits per frame get the S1-seeded analyses (RF
    #: phase, coincidences, time differences, delayed pairs); None = all. See
    #: :func:`analyse_frame`.
    max_s1: int | None = None


@dataclass
class Frame:
    """Steps 1-4 of the plugin on one readout frame (:func:`prepare_frame`).

    Stream-order arrays hold every trigger word (stale ones included); the
    ``s_*`` arrays hold the kept words sorted by time; ``chan[c]`` indexes
    channel c's hits in the ``s_*`` arrays, ascending in time.
    """

    shift: int
    n_words: int
    n_filler: int
    n_pixel: int
    n_trigger: int
    # stream order, every trigger word
    ch: np.ndarray
    tot: np.ndarray
    fine: np.ndarray
    coarse: np.ndarray
    time: np.ndarray            # int64, masked to 2^40 (time_of)
    diff_ns: np.ndarray         # fine_coarse_diff_ns at ``shift``
    consistent: np.ndarray      # bool
    keep: np.ndarray            # bool, not stale
    stale_per_ch: np.ndarray    # int64[16]
    # kept, sorted by time
    order: np.ndarray           # indices into the stream-order arrays
    s_ch: np.ndarray
    s_tot: np.ndarray
    s_t: np.ndarray             # int64, unwrapped around the median
    chan: list = field(default_factory=list)
    first: int = 0
    last: int = 0
    #: Kept words outside the median cluster (consistent far clusters).
    n_rescued: int = 0
    #: time_bits(shift): the modulus of `time`, for gaps between frames.
    time_bits: int = TIME_BITS
    #: First and last kept hit whose fine and coarse agree (`first`/`last`
    #: when fewer than two do). A word with a fine fault has a time that can
    #: be wrong by up to ~1 ms -- at a coarse-minus-fine of half a wrap
    #: (-32 ticks at shift 14) time_of lands one 2^20 ns wrap early -- so the
    #: kept extent of a real 1008 frame starts ~1 ms early on a few faulty
    #: S5/RF words. The rates divide by this extent instead (see
    #: SmaPlugin._fill_good).
    t_first: int | None = None
    t_last: int | None = None
    #: Stream order, every trigger word: its index in the H000 bank (64-bit
    #: words, filler and pixel included) and the raw word.
    word_index: np.ndarray | None = None
    raw: np.ndarray | None = None

    @property
    def span_ns(self) -> int:
        return self.last - self.first

    @property
    def timed_extent(self) -> Extent:
        return Extent(self.first if self.t_first is None else self.t_first,
                      self.last if self.t_last is None else self.t_last)

    @property
    def timed_span_ns(self) -> int:
        e = self.timed_extent
        return e.last - e.first

    @property
    def extent(self) -> Extent:
        return Extent(self.first, self.last)

    def times(self, c) -> np.ndarray:
        """Ascending kept times of channel ``c``."""
        return self.s_t[self.chan[c]]


def split_by_channel(ch, n_channels=N_CHANNELS) -> list:
    """Per channel, the indices of its entries in ``ch`` (order preserved)."""
    c = np.asarray(ch)
    if c.dtype.itemsize > 2:        # a stable sort of 8/16-bit ints is a radix sort
        c = c.astype(np.int16)
    order = np.argsort(c, kind="stable")
    counts = np.bincount(c, minlength=n_channels)[:n_channels]
    return np.split(order, np.cumsum(counts)[:-1])


def prepare_frame(words, shift=DEFAULT_SHIFT, stale_gap_ns=STALE_GAP_NS,
                  latch_margin_ns=LATCH_MARGIN_NS, rescue_shifts=SHIFT_SCAN,
                  rescue_min_words=RESCUE_MIN_WORDS,
                  rescue_min_fraction=RESCUE_MIN_FRACTION) -> Frame:
    """Word counts, decode, fine vs coarse, stale filter, sort, channel split.

    Stale filter: the cluster holding the median time (:func:`stale_mask`)
    is kept, and so is any other cluster of at least ``rescue_min_words``
    words of which at least ``rescue_min_fraction`` are fine/coarse consistent
    at the best of ``rescue_shifts``. The best over several shifts, not the
    configured one, so a wrong configured shift cannot turn genuine clusters
    stale.
    """
    w = words_from_bank(words)
    d = decode(w, shift)
    ch, t = d["ch"], d["time"]
    diff = fine_coarse_diff_ns(d["coarse"], d["fine"], shift)
    bits = time_bits(shift)
    order_all, bounds, k_med, u = _clusters(t, stale_gap_ns, bits)
    keep_sorted = np.zeros(t.size, dtype=bool)
    keep_sorted[bounds[k_med]:bounds[k_med + 1]] = True
    n_rescued = 0
    for k in range(bounds.size - 1):
        a, b = int(bounds[k]), int(bounds[k + 1])
        if k == k_med or b - a < rescue_min_words:
            continue
        idx = order_all[a:b]
        best = shift_scan(d["coarse"][idx], d["fine"][idx], rescue_shifts, latch_margin_ns).max()
        if best >= rescue_min_fraction * (b - a):
            keep_sorted[a:b] = True
            n_rescued += b - a
    order = order_all[keep_sorted]
    keep = np.zeros(t.size, dtype=bool)
    keep[order] = True
    s_ch = ch[order]
    s_t = u[order]
    consistent = fine_coarse_consistent(diff, shift, latch_margin_ns)
    ct = s_t[consistent[order]]
    t_first, t_last = (int(ct[0]), int(ct[-1])) if ct.size >= 2 else (None, None)
    return Frame(
        shift=int(shift), n_words=int(w.size), n_filler=d["n_filler"],
        n_pixel=d["n_pixel"], n_trigger=d["n_trigger"],
        ch=ch, tot=d["tot"], fine=d["fine"], coarse=d["coarse"], time=t,
        diff_ns=diff, consistent=consistent,
        keep=keep,
        stale_per_ch=np.bincount(ch[~keep].astype(np.intp), minlength=N_CHANNELS)[:N_CHANNELS]
        .astype(np.int64),
        order=order, s_ch=s_ch, s_tot=d["tot"][order], s_t=s_t,
        chan=split_by_channel(s_ch),
        first=int(s_t[0]) if s_t.size else 0, last=int(s_t[-1]) if s_t.size else 0,
        n_rescued=n_rescued, time_bits=bits, t_first=t_first, t_last=t_last,
        word_index=d["word_index"], raw=d["raw"])


@dataclass
class FrameAnalysis:
    """Steps 5-8 of the plugin (:func:`analyse_frame`). S1 arrays are ascending in time.

    One row per analysed S1 hit. Without an S1 cap that is every kept S1 hit
    and ``sample`` is None. With one (``Cuts.max_s1``) the rows are the evenly
    spread sample plus the seeds, and ``sample`` marks the sample rows: fills
    use :meth:`sampled`, the event display the seed rows.
    """

    t_s1: np.ndarray
    s1_tot: np.ndarray
    shift_counts: np.ndarray        # consistent S1 words (all, not only kept) per Cuts.shift_scan
    # RF, one per S1 hit
    rf_n: np.ndarray
    rf_valid: np.ndarray
    rf_vetoed: np.ndarray
    rf_gap: np.ndarray              # to the next later S1 hit (inf: none)
    rf_phase: np.ndarray
    rf_period: np.ndarray
    # coincidences, one row per S1 hit, one column per Roles.counters
    pattern: np.ndarray
    partner_counts: np.ndarray
    dt: dict                        # counter ch -> (i_s1, dt) all pairs in +-dt_window
    delayed_dt: dict                # delayed ch -> (i_s1, dt) all pairs in [lo, hi]
    seeds: np.ndarray               # indices into t_s1
    #: Rows in the evenly spread S1 sample (bool per row); None: every row.
    sample: np.ndarray | None = None
    #: Every kept S1 time of the frame (the S1 spacing needs all of them);
    #: None: the same as t_s1.
    t_s1_all: np.ndarray | None = None
    #: Per row, its index among the frame's kept S1 hits (``frame.chan[s1]``);
    #: None: row i is kept S1 hit i.
    s1_rows: np.ndarray | None = None

    @property
    def n_s1_kept(self) -> int:
        """Kept S1 hits of the frame, analysed or not."""
        return int((self.t_s1 if self.t_s1_all is None else self.t_s1_all).size)

    def sampled(self) -> FrameAnalysis:
        """Only the sample rows (itself when there is no cap); pair indices renumbered."""
        m = self.sample
        if m is None:
            return self
        new_i = np.cumsum(m) - 1

        def pairs(d):
            out = {}
            for c, (i, dt) in d.items():
                keep = m[i]
                out[c] = (new_i[i[keep]], dt[keep])
            return out

        return FrameAnalysis(
            t_s1=self.t_s1[m], s1_tot=self.s1_tot[m], shift_counts=self.shift_counts,
            rf_n=self.rf_n[m], rf_valid=self.rf_valid[m], rf_vetoed=self.rf_vetoed[m],
            rf_gap=self.rf_gap[m], rf_phase=self.rf_phase[m], rf_period=self.rf_period[m],
            pattern=self.pattern[m], partner_counts=self.partner_counts[m],
            dt=pairs(self.dt), delayed_dt=pairs(self.delayed_dt),
            seeds=np.zeros(0, dtype=np.intp), sample=None, t_s1_all=self.t_s1_all,
            s1_rows=None if self.s1_rows is None else self.s1_rows[m])


def even_sample(n: int, m: int) -> np.ndarray:
    """``m`` indices of ``range(n)``, evenly spread and deterministic (all if m >= n).

    Index k is the middle of the k-th of m equal slices, so the sample covers
    the whole frame in time order and depends on nothing but n and m.
    """
    if m >= n:
        return np.arange(n, dtype=np.intp)
    k = np.arange(m, dtype=np.int64)
    return ((2 * k + 1) * n // (2 * m)).astype(np.intp)


def analyse_frame(frame: Frame, roles: Roles | None = None, cuts: Cuts | None = None,
                  shift_counts=None) -> FrameAnalysis:
    """Shift scan, RF phase, coincidences and seeds on a prepared frame.

    ``shift_counts``: the all-S1 shift scan when the caller has it already
    (the plugin computes it to classify the frame); computed here otherwise.

    **S1 cap.** With ``cuts.max_s1`` set and more kept S1 hits than that, the
    S1-seeded analyses run on an evenly spread sample of ``max_s1`` S1 hits
    (:func:`even_sample`), plus the seeds for the event display. That bounds
    the cost of a dense frame, where every S1 has many partners. The sample is
    chosen by position in time order alone, never by what a hit's partners
    are, so a fraction computed on it (pattern, efficiency given S1, RF valid)
    is an unbiased estimate of the same fraction over every S1 hit; only the
    counts shrink. The RF gate veto still looks at every S1 hit (it asks
    whether another S1 is inside the gate), and so does the S1 spacing.
    """
    roles = roles or Roles()
    cuts = cuts or Cuts()
    i1 = frame.chan[roles.s1]
    t_s1 = frame.s_t[i1]
    if shift_counts is None:
        # Every S1 word of the frame, not only the kept ones: the scan needs no
        # times, and at a far-wrong shift the kept cluster is a handful of words.
        s1_all = frame.ch == roles.s1
        shift_counts = shift_scan(frame.coarse[s1_all], frame.fine[s1_all], cuts.shift_scan,
                                  cuts.latch_margin_ns)
    t_rf = frame.times(roles.rf)
    counters = [frame.times(c) for c in roles.counters]
    rng = complete_range([x for x in counters + [t_rf] if len(x) >= cuts.seed_min_hits])
    seeds = (select_seeds(t_s1, rng[0], rng[1], cuts.n_seeds, cuts.seed_pre_ns,
                          cuts.seed_post_ns) if rng else np.zeros(0, dtype=np.intp))
    vetoed, gap = gate_veto(t_s1, cuts.rf_gate_ns)
    s1_tot = frame.s_tot[i1]

    sample = None
    t_all = None
    rows = None
    if cuts.max_s1 is not None and t_s1.size > cuts.max_s1:
        pick = even_sample(t_s1.size, int(cuts.max_s1))
        rows = np.union1d(pick, seeds).astype(np.intp)
        sample = np.zeros(rows.size, dtype=bool)
        sample[np.searchsorted(rows, pick)] = True
        seeds = np.searchsorted(rows, seeds).astype(np.intp)
        t_all = t_s1
        t_s1, s1_tot, vetoed, gap = t_s1[rows], s1_tot[rows], vetoed[rows], gap[rows]

    n, valid, phase, period, _lo, _pt = burst_phase_many(
        t_rf, t_s1, rule=cuts.rf_rule, gate_ns=cuts.rf_gate_ns,
        min_pulses=cuts.rf_min_pulses, max_pulses=cuts.rf_max_pulses, vetoed=vetoed)
    pattern, counts = coincidence(t_s1, counters, cuts.coinc_ns)
    dts = {}
    for c, tc in zip(roles.counters, counters, strict=True):
        if c == roles.s1:
            continue
        i, _j, dt = window_pairs(t_s1, tc, -cuts.dt_window_ns, cuts.dt_window_ns)
        dts[c] = (i, dt)
    delayed = {}
    for c in roles.delayed:
        i, _j, dt = window_pairs(t_s1, frame.times(c), cuts.delayed_lo_ns, cuts.delayed_hi_ns)
        delayed[c] = (i, dt)
    return FrameAnalysis(
        t_s1=t_s1, s1_tot=s1_tot, shift_counts=shift_counts,
        rf_n=n, rf_valid=valid, rf_vetoed=vetoed, rf_gap=gap, rf_phase=phase,
        rf_period=period, pattern=pattern, partner_counts=counts, dt=dts,
        delayed_dt=delayed, seeds=seeds, sample=sample, t_s1_all=t_all, s1_rows=rows)
