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

MuPix pixel words (the same bank, bit 63 clear): :func:`pixel_decode` follows
``main/reco_testbeam/pi_midas/include/PIMuPixWord.hh`` (``ChipId``, ``Col``,
``Row``, ``Ts2``, ``Time``, ``Tot``), which is the ``pixelhit`` of
``PITMidasMusip.cpp``; ``psm-analysis-josh-2026/sma-tot-vs-wd/mupix_phase.py``
(``pixel_words``) and ``mupix-timewalk/timewalk_lib.py`` (``pixel_tot``) are the
Python references. ``tests/test_mupix_golden.py`` pins all three on stored
real frames. The S1 matching (:func:`mupix_match`), the plane map and the
per-frame cap are new here.

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
* A pixel word: chip 62:58 (a global ASIC id, see :class:`MuPixCuts`), column
  57:50, row 49:42, TS2 41:37, time stamp 36:0 in 8 ns ticks. Its time in ns
  (tick x 8, 2^40 ns) is the SMA time's epoch; within a frame it is unwrapped
  around the same reference as the SMA hits, so the two subtract directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

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

# --- MuPix pixel words (PIMuPixWord.hh) ------------------------------------------

PIXEL_TICK_NS = 8
PIXEL_TIME_BITS = 37
PIXEL_TIME_MASK = (1 << PIXEL_TIME_BITS) - 1
TS2_MASK = 0x1F
#: log2(ckdivend2 + 1): ckdivend2 = 0x1f on every bt2026 run, so one ToT count
#: is 2^5 ticks = 256 ns (PIMuPixWord::kDefaultTs2Shift).
TS2_SHIFT = 5
#: The largest TS2 shift that leaves TS2's 5 bits inside the 37-bit time stamp.
MAX_TS2_SHIFT = PIXEL_TIME_BITS - 5
#: The chip field has 5 bits.
N_CHIP_IDS = 32
#: Rows of the sensor. Row codes 250-255 occur in the data; they are not
#: dropped here (the reco drops them, PITMidasMusip.cpp), only flagged.
PIXEL_ROWS = 250
PLANE_NONE, PLANE_L1, PLANE_L2 = 0, 1, 2
PLANE_NAMES = ("no plane", "L1", "L2")
#: The chip field is the GLOBAL ASIC id the switching board assigns from
#: /Equipment/Quads/Settings/DAQ/Links/Mapping. From run 200 on that array is
#: the identity, and chip 0 pairs with 4, 1 with 5, 2 with 6, 3 with 7 (the same
#: particle in both planes, measured on runs 682 and 1008): L1 = 0-3, L2 = 4-7,
#: as the readout map bt2026-febmap says for runs >= 200. (Runs up to 186 had
#: Mapping [1..7, 0]: L1 = 1-4, L2 = 5, 6, 7, 0.)
L1_CHIPS = (0, 1, 2, 3)
L2_CHIPS = (4, 5, 6, 7)
#: t(pixel) - t(S1) of a pixel "in time" with S1, half open [lo, hi): the MuPix
#: time walk puts real hits from about -150 ns (high ToT) to +450 ns (low ToT)
#: around S1. No time-walk correction.
MUPIX_WINDOW_NS = (-150, 450)
#: The accidental rate is measured in an off-time sideband of the same width
#: before the prompt window (never after it: decays and delayed hits live there).
MUPIX_SIDEBAND_NS = (-2400, -1800)
#: Pixel hits examined per frame, the latest in the stream; more are counted
#: and skipped (a 40000-word run-1008 frame has ~7300).
MAX_PIXELS = 20_000
#: S1 hits per frame matched against the pixels, at most (an evenly spread
#: subset of the S1 rows the S1-seeded analyses use): 500 a frame at 35
#: frames/s is ~17k S1 hits a second, plenty for a fraction, at a quarter of
#: the cost of 2000.
MUPIX_MAX_S1 = 500
#: All-pairs t(pixel) - t(S1) entries per frame and plane, at most; a denser
#: frame uses an evenly spread subset of its S1 rows.
MAX_MUPIX_PAIRS = 200_000
#: The MuPix selector of the event display (sma::frame "mupix").
MUPIX_MODES = ("any", "both", "either", "none")
#: The per-counter pattern selector of the event display (sma::frame "pattern"):
#: a hit on that counter within +-coinc of the seed, or none; "any" is no condition.
PATTERN_STATES = ("any", "present", "absent")
#: Counters (S1..Sn) at most: the coincidence pattern is one byte per seed.
MAX_COUNTERS = 8

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
# MuPix pixel words (PIMuPixWord.hh)
# ==============================================================================

def pixel_tot(ts2, tick, ts2_shift=TS2_SHIFT) -> np.ndarray:
    """The time over threshold, 0..31 in 2^ts2_shift ticks (``PIMuPixWord::Tot``).

    Bits 41:37 are TS2, a counter latched at the falling edge, not a ToT; the
    time stamp was latched at the rising edge, so ToT = (TS2 - ((tick >>
    ts2_shift) & 0x1F)) & 0x1F. uint8.
    """
    ts2 = np.asarray(ts2).astype(np.int64)
    tick = np.asarray(tick, dtype=np.int64)
    return ((ts2 - ((tick >> int(ts2_shift)) & TS2_MASK)) & TS2_MASK).astype(np.uint8)


def pixel_decode(words, ts2_shift=TS2_SHIFT, last: int | None = None) -> dict:
    """The pixel words (bit 63 clear) of one bank, in stream order, as arrays.

    ``{chip, col, row, ts2, tot (uint8), tick (int64, 8 ns), time (int64 ns,
    < 2^40), word_index (uint32), raw (uint64), n_words}``. With ``last``, only
    the last ``last`` pixel words of the stream are decoded; ``n_words`` still
    counts all of them. The filler word has bit 63 set and is never a pixel.
    """
    w = np.asarray(words, dtype=_U)
    idx = np.flatnonzero(w.view(np.int64) >= 0)
    n_words = int(idx.size)
    if last is not None and idx.size > last:
        idx = idx[idx.size - max(0, int(last)):]
    p = w[idx]
    tick = (p & _U(PIXEL_TIME_MASK)).astype(np.int64)
    ts2 = ((p >> _U(37)) & _U(TS2_MASK)).astype(np.uint8)
    return {"chip": ((p >> _U(58)) & _U(0x1F)).astype(np.uint8),
            "col": ((p >> _U(50)) & _U(0xFF)).astype(np.uint8),
            "row": ((p >> _U(42)) & _U(0xFF)).astype(np.uint8),
            "ts2": ts2, "tot": pixel_tot(ts2, tick, ts2_shift), "tick": tick,
            "time": tick * PIXEL_TICK_NS, "word_index": idx.astype(np.uint32), "raw": p,
            "n_words": n_words}


def encode_pixel(chip, col, row, ts2, tick):
    """Inverse of :func:`pixel_decode`: one pixel word from its fields (test helper).

    Scalars give an ``int``, arrays a ``uint64`` array.
    """
    if all(np.isscalar(x) for x in (chip, col, row, ts2, tick)):
        return (((int(chip) & 0x1F) << 58) | ((int(col) & 0xFF) << 50) | ((int(row) & 0xFF) << 42)
                | ((int(ts2) & 0x1F) << 37) | (int(tick) & PIXEL_TIME_MASK))
    chip, col, row, ts2, tick = (np.asarray(x).astype(_U) for x in (chip, col, row, ts2, tick))
    return (((chip & _U(0x1F)) << _U(58)) | ((col & _U(0xFF)) << _U(50))
            | ((row & _U(0xFF)) << _U(42)) | ((ts2 & _U(0x1F)) << _U(37))
            | (tick & _U(PIXEL_TIME_MASK)))


def ts2_of(tick, tot, ts2_shift=TS2_SHIFT):
    """The TS2 a pixel of time stamp ``tick`` and ToT ``tot`` carries (test helper)."""
    tick = np.asarray(tick, dtype=np.int64)
    return (((tick >> int(ts2_shift)) + np.asarray(tot, dtype=np.int64)) & TS2_MASK)


def plane_lookup(l1=L1_CHIPS, l2=L2_CHIPS) -> np.ndarray:
    """``uint8[32]``: the plane (:data:`PLANE_L1`, :data:`PLANE_L2`, else
    :data:`PLANE_NONE`) of every chip id."""
    out = np.zeros(N_CHIP_IDS, dtype=np.uint8)
    out[np.asarray(list(l1), dtype=np.intp)] = PLANE_L1
    out[np.asarray(list(l2), dtype=np.intp)] = PLANE_L2
    return out


@dataclass
class MuPixCuts:
    """The MuPix settings (ODB ``/DQM/SMA/MuPix``), defaults as above.

    ``l1``/``l2``: chip ids (the global ASIC ids of the pixel words) of each
    plane; a chip in neither is shown as "chip N (no plane)". ``window_ns``
    and ``sideband_ns`` are half open, relative to S1 (or to a display seed).
    ``max_pixels`` 0 turns the MuPix analysis off (pixel words are still
    counted); None, no cap.
    """

    l1: tuple = L1_CHIPS
    l2: tuple = L2_CHIPS
    ts2_shift: int = TS2_SHIFT
    window_ns: tuple = MUPIX_WINDOW_NS
    sideband_ns: tuple = MUPIX_SIDEBAND_NS
    max_pixels: int | None = MAX_PIXELS
    max_pairs: int = MAX_MUPIX_PAIRS
    #: S1 rows per frame matched (MUPIX_MAX_S1); None: all.
    max_s1: int | None = MUPIX_MAX_S1

    @property
    def planes(self) -> np.ndarray:
        cached = self.__dict__.get("_planes")
        if cached is None or cached[0] != (tuple(self.l1), tuple(self.l2)):
            cached = ((tuple(self.l1), tuple(self.l2)), plane_lookup(self.l1, self.l2))
            self.__dict__["_planes"] = cached
        return cached[1]

    @property
    def enabled(self) -> bool:
        return self.max_pixels is None or self.max_pixels > 0

    @property
    def tot_ns(self) -> int:
        """One ToT count in ns."""
        return PIXEL_TICK_NS << int(self.ts2_shift)


@dataclass
class Pixels:
    """The examined pixel hits of one frame, sorted by time (:func:`prepare_pixels`).

    ``t`` is in ns on the frame's SMA time basis (unwrapped around the same
    reference as ``Frame.s_t``). ``planes[p]`` indexes plane p's hits (0 = no
    plane, 1 = L1, 2 = L2), ascending in time.
    """

    n_words: int                    # pixel words in the bank
    n_skipped: int                  # not examined (MuPixCuts.max_pixels)
    was_sorted: bool                # the examined words were already in time order
    t: np.ndarray                   # int64
    chip: np.ndarray                # uint8
    col: np.ndarray
    row: np.ndarray
    ts2: np.ndarray
    tot: np.ndarray
    plane: np.ndarray               # uint8
    word_index: np.ndarray          # uint32
    raw: np.ndarray                 # uint64
    planes: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(self.t.size)

    @property
    def first(self) -> int:
        return int(self.t[0]) if self.t.size else 0

    @property
    def last(self) -> int:
        return int(self.t[-1]) if self.t.size else 0

    def times(self, plane) -> np.ndarray:
        return self.t[self.planes[plane]]

    def nbytes(self) -> int:
        return sum(v.nbytes for v in vars(self).values() if isinstance(v, np.ndarray)) + sum(
            x.nbytes for x in self.planes)


def prepare_pixels(words, cuts: MuPixCuts | None = None, ref_ns: int | None = None,
                   bits: int = TIME_BITS) -> Pixels:
    """Decode, cap, unwrap, sort and split the pixel words of one bank.

    At most ``cuts.max_pixels`` pixel words are examined: the latest in the
    stream (which is close to time order; the rest are counted as skipped).
    Times are unwrapped modulo 2^``bits`` around ``ref_ns`` (the frame's SMA
    reference; the pixels' own median without one), so a frame straddling
    the 2^40 ns wrap still sorts and subtracts correctly, and below coarse
    shift 12 they land in the SMA time's shorter span. The MuPix stream is not
    promised to be time-sorted within a frame: it is sorted here (stable), at
    O(n) cost when it already is.
    """
    cuts = cuts or MuPixCuts()
    last = cuts.max_pixels
    d = pixel_decode(words, cuts.ts2_shift, last=last)
    t = d["time"]
    n = t.size
    if n:
        ref = ref_ns if ref_ns is not None else int(np.partition(t, (n - 1) // 2)[(n - 1) // 2])
        t = unwrap_near(t, ref, bits)
    was_sorted = bool(n < 2 or np.all(t[1:] >= t[:-1]))
    keys = ("chip", "col", "row", "ts2", "tot", "word_index", "raw")
    if not was_sorted:
        o = np.argsort(t, kind="stable")
        t = t[o]
        d = {k: d[k][o] for k in keys} | {"n_words": d["n_words"]}
    plane = cuts.planes[d["chip"]] if n else np.zeros(0, dtype=np.uint8)
    return Pixels(n_words=d["n_words"], n_skipped=d["n_words"] - n, was_sorted=was_sorted,
                  t=np.asarray(t, dtype=np.int64), chip=d["chip"], col=d["col"], row=d["row"],
                  ts2=d["ts2"], tot=d["tot"], plane=plane, word_index=d["word_index"],
                  raw=d["raw"], planes=split_by_channel(plane, 3))


def mupix_coverage(t_ref, px: Pixels, lo_ns, hi_ns) -> np.ndarray:
    """Per reference time, whether [t + lo, t + hi) lies inside the examined pixel data.

    The MuPix and SMA parts of a frame do not cover the same time (run 1008:
    the pixels start ~0.5 ms after the SMA hits), and the per-frame cap
    examines only the latest pixels: an S1 hit outside says nothing about
    MuPix and is not judged.
    """
    t = np.asarray(t_ref, dtype=np.int64)
    if px is None or px.n == 0:
        return np.zeros(t.shape, dtype=bool)
    return (t + int(lo_ns) >= px.first) & (t + int(hi_ns) <= px.last + 1)


def mupix_in_window(t_ref, px: Pixels, lo_ns, hi_ns):
    """``(n_l1, n_l2)``: pixel hits of each plane in [t + lo, t + hi) per reference (intp)."""
    t = np.asarray(t_ref, dtype=np.int64)
    lo, hi = int(lo_ns), int(hi_ns) - 1         # integer ns: [lo, hi) == [lo, hi - 1]
    return (window_counts(t, px.times(PLANE_L1), lo, hi),
            window_counts(t, px.times(PLANE_L2), lo, hi))


@dataclass
class MuPixMatch:
    """In-time and sideband MuPix hits of S1 hits (:func:`mupix_match`).

    Per S1 row: ``judged`` (its window and sideband lie in the pixel data) and,
    for the judged rows, whether L1 / L2 had a hit in the window (``in_l1``,
    ``in_l2``) and in the sideband (``side_l1``, ``side_l2``); False elsewhere.
    """

    judged: np.ndarray
    in_l1: np.ndarray
    in_l2: np.ndarray
    side_l1: np.ndarray
    side_l2: np.ndarray

    @property
    def n(self) -> int:
        return int(np.count_nonzero(self.judged))

    def counts(self) -> tuple[int, np.ndarray, np.ndarray]:
        """``(n_judged, in [L1, L2, both], side [L1, L2, both])``, int64 counts."""
        i1, i2, s1, s2 = self.in_l1, self.in_l2, self.side_l1, self.side_l2
        c = lambda a, b: np.array([np.count_nonzero(a), np.count_nonzero(b),  # noqa: E731
                                   np.count_nonzero(a & b)], dtype=np.int64)
        return self.n, c(i1, i2), c(s1, s2)


def mupix_match(t_s1, px: Pixels | None, window_ns=MUPIX_WINDOW_NS,
                sideband_ns=MUPIX_SIDEBAND_NS) -> MuPixMatch:
    """Which S1 hits have an L1 / L2 pixel hit in the in-time window, and in the sideband.

    ``t_s1`` ascending. Only S1 hits whose window and sideband both lie inside
    the examined pixel data are judged (:func:`mupix_coverage`). The sideband
    has the same width as the window by default and measures the accidentals:
    the fraction of S1 hits with a sideband hit is the chance of a hit in a
    window of that width with nothing to do with S1.
    """
    t = np.asarray(t_s1, dtype=np.int64)
    z = np.zeros(t.shape, dtype=bool)
    if px is None or px.n == 0 or t.size == 0:
        return MuPixMatch(z, z.copy(), z.copy(), z.copy(), z.copy())
    (wlo, whi), (slo, shi) = window_ns, sideband_ns
    judged = mupix_coverage(t, px, min(wlo, slo), max(whi, shi))
    n1, n2 = mupix_in_window(t, px, wlo, whi)
    m1, m2 = mupix_in_window(t, px, slo, shi)
    return MuPixMatch(judged, judged & (n1 > 0), judged & (n2 > 0),
                      judged & (m1 > 0), judged & (m2 > 0))


def accidental_corrected(f_in, f_side, width_in=1.0, width_side=1.0):
    """The in-time fraction with the accidentals taken out.

    With a chance ``a`` of an accidental hit in the window, P(in) = 1 - (1 -
    p)(1 - a), so p = (P(in) - a) / (1 - a). ``a`` comes from the sideband,
    scaled to the window's width (Poisson: 1 - (1 - f_side)^(w_in / w_side)).
    None when either is None or a = 1.
    """
    if f_in is None or f_side is None:
        return None
    a = 1.0 - (1.0 - float(f_side)) ** (float(width_in) / float(width_side))
    if a >= 1.0:
        return None
    return (float(f_in) - a) / (1.0 - a)


def mupix_pairs(t_s1, t_pix, lo_ns, hi_ns, max_pairs=MAX_MUPIX_PAIRS):
    """``(dt, n_s1_used)``: every t(pixel) - t(S1) in [lo, hi), both inputs ascending.

    At most about ``max_pairs`` pairs: when the frame would give more, an
    evenly spread subset of the S1 rows is used (every k-th), and
    ``n_s1_used`` says how many.
    """
    t1 = np.asarray(t_s1, dtype=np.int64)
    tp = np.asarray(t_pix, dtype=np.int64)
    lo, hi = int(lo_ns), int(hi_ns) - 1
    if t1.size == 0 or tp.size == 0:
        return np.zeros(0, dtype=np.int64), int(t1.size)
    a, b = _bounds(t1, tp, lo, hi)
    total = int((b - a).sum())
    if total > max_pairs > 0:
        k = -(-total // int(max_pairs))
        t1 = t1[::k]
    _i, _j, dt = window_pairs(t1, tp, lo, hi)
    return dt, int(t1.size)


# ==============================================================================
# One frame
# ==============================================================================

@dataclass
class Roles:
    """Which SMA channel is what (ODB ``/DQM/SMA/Channel roles``)."""

    s1: int = 1
    #: S1..S5 in order, the first the s1 channel; at most MAX_COUNTERS.
    counters: tuple = (1, 2, 7, 4, 5)
    rf: int = 6
    #: -1 = none (no proton current on the SMA since run 1015).
    current: int = -1
    delayed: tuple = ()


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


class CounterHits(NamedTuple):
    """One counter's hits in one frame, ascending in time.

    By default a counter's hits are its channel's kept words
    (:meth:`Frame.counter_hits_of`). The plugin can hand over others per
    counter channel (:attr:`Frame.counter_hits`): the TOT + NIM merged hits,
    where a hit carried by a NIM word takes that word's aligned time and a
    substituted ToT. ``idx`` always points at the carrier word in the frame's
    kept hits (``Frame.s_*``), so a seed on such a hit has a word to show.
    """

    t: np.ndarray       # int64 ns, the frame's time basis
    tot: np.ndarray     # ToT code
    idx: np.ndarray     # intp, into Frame.s_*


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
    #: The MuPix pixel hits (:func:`prepare_pixels`), on the same time basis as
    #: ``s_t``; None when the frame was prepared without MuPix.
    px: Pixels | None = None
    #: Counter hits that replace a counter channel's kept words in every
    #: counter use (:func:`analyse_frame`, :func:`select_seeds_by`): channel ->
    #: :class:`CounterHits`. Set by the plugin (the TOT + NIM merge); None, or a
    #: channel not in it, means the channel's own kept words. Everything that
    #: reads the words themselves (rates, ToT, fine/coarse, stale, the shift
    #: scan, the raster) ignores it.
    counter_hits: dict | None = None
    #: The plugin's per-hit TOT/NIM pairing of this frame, indexed like
    #: ``s_*`` (``sma.NimWords``); None when nothing was paired. Carried for
    #: the event display only; nothing here reads it.
    pairing: tuple | None = None

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

    def counter_hits_of(self, c) -> CounterHits:
        """Channel ``c``'s hits as a counter: :attr:`counter_hits` when it has
        them, else its kept words."""
        h = self.counter_hits.get(c) if self.counter_hits else None
        if h is not None:
            return h
        i = self.chan[c]
        return CounterHits(self.s_t[i], self.s_tot[i], i)

    def counter_times(self, c) -> np.ndarray:
        """Ascending times of channel ``c`` as a counter (:meth:`counter_hits_of`)."""
        h = self.counter_hits.get(c) if self.counter_hits else None
        return self.s_t[self.chan[c]] if h is None else h.t

    def counter_idx(self, c) -> np.ndarray:
        """The carrier word (into ``s_*``) of each of channel ``c``'s counter hits."""
        h = self.counter_hits.get(c) if self.counter_hits else None
        return self.chan[c] if h is None else h.idx


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
                  rescue_min_fraction=RESCUE_MIN_FRACTION,
                  mupix: MuPixCuts | None = None) -> Frame:
    """Word counts, decode, fine vs coarse, stale filter, sort, channel split.

    With ``mupix`` (and its analysis enabled) the pixel words too
    (:func:`prepare_pixels`), unwrapped around the kept SMA hits' median.

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
    px = None
    if mupix is not None and mupix.enabled:
        px = prepare_pixels(w, mupix, int(s_t[(s_t.size - 1) // 2]) if s_t.size else None, bits)
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
        word_index=d["word_index"], raw=d["raw"], px=px)


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
    #: Per row, its index among the frame's S1 hits as a counter
    #: (``frame.counter_idx(s1)``: the kept S1 words, or the merged S1 hits);
    #: None: row i is S1 hit i.
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
    # S1 and the counters as counters (Frame.counter_hits: the merged TOT +
    # NIM hits when the plugin set them); the shift scan reads the raw words.
    h1 = frame.counter_hits_of(roles.s1)
    t_s1 = h1.t
    if shift_counts is None:
        # Every S1 word of the frame, not only the kept ones: the scan needs no
        # times, and at a far-wrong shift the kept cluster is a handful of words.
        s1_all = frame.ch == roles.s1
        shift_counts = shift_scan(frame.coarse[s1_all], frame.fine[s1_all], cuts.shift_scan,
                                  cuts.latch_margin_ns)
    t_rf = frame.times(roles.rf)
    counters = [frame.counter_times(c) for c in roles.counters]
    rng = complete_range([x for x in counters + [t_rf] if len(x) >= cuts.seed_min_hits])
    seeds = (select_seeds(t_s1, rng[0], rng[1], cuts.n_seeds, cuts.seed_pre_ns,
                          cuts.seed_post_ns) if rng else np.zeros(0, dtype=np.intp))
    vetoed, gap = gate_veto(t_s1, cuts.rf_gate_ns)
    s1_tot = h1.tot

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


# ==============================================================================
# Seed choice and odd-event filters (the event display, at request time)
# ==============================================================================
#
# analyse_frame picks the display's seeds while filling the histograms: the
# latest Cuts.n_seeds S1 hits with a complete window. The event display can also
# ask for other seeds, per viewer and per request: another channel, "any
# counter", and only seeds with an oddity in their window. Those are computed
# here from a prepared Frame alone -- vectorised, bounded, never while filling --
# and with seed "s1" and no filter they give exactly analyse_frame's seeds.

SEED_S1 = "s1"
SEED_ANY = "any"
#: The oddities a seed can be asked for; a seed matches when it has ANY of the
#: ones asked for (OR). Bit k of SeedSelection.odd is FILTERS[k].
FILTERS = ("incomplete", "mismatch", "tot", "rf")
FILTER_BITS = {name: 1 << k for k, name in enumerate(FILTERS)}
#: Candidates examined per frame, latest first: bounds a request on a dense frame.
MAX_CANDIDATES = 20_000


def parse_seed(seed) -> str:
    """The canonical seed mode: ``"s1"``, ``"any"`` or ``"ch<N>"`` (N 0..15).

    Accepts those strings (any case) and a bare channel number. ValueError otherwise.
    """
    if seed is None:
        return SEED_S1
    if isinstance(seed, bool):
        raise ValueError(f"seed {seed!r}")
    if isinstance(seed, int | np.integer):
        c = int(seed)
    else:
        s = str(seed).strip().lower()
        if s in (SEED_S1, SEED_ANY):
            return s
        if not s.startswith("ch") or not s[2:].isdigit():
            raise ValueError(f"seed must be 's1', 'any' or 'ch<N>', got {seed!r}")
        c = int(s[2:])
    if not 0 <= c < N_CHANNELS:
        raise ValueError(f"seed channel {c} is not 0-15")
    return f"ch{c}"


def parse_mupix(mode) -> str:
    """The canonical MuPix selector: one of :data:`MUPIX_MODES` ("any" for None)."""
    if mode is None:
        return "any"
    m = str(mode).strip().lower()
    if m not in MUPIX_MODES:
        raise ValueError(f"mupix must be one of {list(MUPIX_MODES)}, got {mode!r}")
    return m


def mupix_selects(mode: str, covered, n_l1, n_l2) -> np.ndarray:
    """Per seed, whether it passes the MuPix selector (an AND on top of the filters).

    ``both``: an L1 and an L2 hit in the in-time window; ``either``: one of
    them; ``none``: neither. A hit found is a hit, wherever the pixel data
    ends; but the absence of one means something only when the whole window
    lies inside the pixel data (``covered``), so ``none`` needs that.
    """
    covered = np.asarray(covered, dtype=bool)
    if mode == "any":
        return np.ones(covered.shape, dtype=bool)
    a, b = np.asarray(n_l1) > 0, np.asarray(n_l2) > 0
    if mode == "both":
        return a & b
    if mode == "either":
        return a | b
    if mode == "none":
        return covered & ~a & ~b
    raise ValueError(f"mupix mode {mode!r}")


def parse_pattern(pattern, n_counters: int = 5) -> tuple:
    """The canonical per-counter pattern selector: ``((k, state), ...)``, k the
    counter number (1 = S1 .. n_counters), state "present" or "absent", sorted
    by k; "any" entries are dropped, so no condition at all is ``()``.

    Accepts a dict ``{"1": "present", "3": "absent"}`` (keys as strings or
    ints) or None. ValueError on an unknown counter or state.
    """
    if pattern is None:
        return ()
    if not isinstance(pattern, dict):
        raise ValueError(f"pattern must be an object {{counter: state}}, got {pattern!r}")
    out = {}
    for key, state in pattern.items():
        if isinstance(key, bool):
            raise ValueError(f"pattern counter {key!r}")
        try:
            k = int(str(key).strip().upper().removeprefix("S"))
        except ValueError:
            raise ValueError(f"pattern counter must be 1-{n_counters}, got {key!r}") from None
        if not 1 <= k <= n_counters:
            raise ValueError(f"pattern counter must be 1-{n_counters}, got {key!r}")
        st = str(state).strip().lower() if state is not None else "any"
        if st not in PATTERN_STATES:
            raise ValueError(f"pattern state must be one of {list(PATTERN_STATES)}, got {state!r}")
        if st != "any":
            out[k] = st
    return tuple(sorted(out.items()))


def pattern_selects(require: tuple, pattern) -> np.ndarray:
    """Per seed, whether its coincidence pattern (bit k-1 = counter k within
    +-coinc) passes the pattern selector ``require`` (:func:`parse_pattern`)."""
    pattern = np.asarray(pattern)
    ok = np.ones(pattern.shape, dtype=bool)
    for k, st in require:
        lit = ((pattern >> (k - 1)) & 1).astype(bool)
        ok &= lit if st == "present" else ~lit
    return ok


def parse_filters(filters) -> tuple:
    """The canonical filter tuple (in FILTERS order, no repeats). ValueError on an unknown name."""
    if filters is None:
        return ()
    if isinstance(filters, str):
        filters = [filters]
    names = {str(f).strip().lower() for f in filters}
    bad = sorted(names - set(FILTERS))
    if bad:
        raise ValueError(f"unknown filter(s) {bad}; known: {list(FILTERS)}")
    return tuple(f for f in FILTERS if f in names)


@dataclass
class SeedSelection:
    """The seeds of one frame for one (seed mode, filters) choice.

    Per-seed arrays hold the chosen seeds only, ascending in time; ``idx``
    indexes the frame's kept sorted hits (``Frame.s_*``). The RF fields come from
    the S1 hit the seed borrows (itself for an S1 seed, else the nearest S1 hit
    within the coincidence window); ``has_s1`` False means there was none, and
    ``rf_n`` is -1 and the phase NaN.
    """

    mode: str
    filters: tuple
    idx: np.ndarray
    t: np.ndarray
    ch: np.ndarray
    tot: np.ndarray
    pattern: np.ndarray
    odd: np.ndarray                 # uint8, FILTER_BITS of every oddity (asked for or not)
    has_s1: np.ndarray
    s1_idx: np.ndarray              # into Frame.s_*, -1 without S1
    s1_dt: np.ndarray               # t(S1) - t(seed), 0 without S1
    rf_n: np.ndarray
    rf_valid: np.ndarray
    rf_vetoed: np.ndarray
    rf_phase: np.ndarray
    rf_period: np.ndarray
    #: Seed candidates of the frame whose window is complete, how many of them
    #: were examined (the latest MAX_CANDIDATES), and how many of those match.
    n_candidates: int = 0
    n_examined: int = 0
    n_matching: int = 0
    #: The MuPix selector, and per chosen seed its in-time L1 / L2 hits and
    #: whether its in-time window lies wholly in the pixel data.
    mupix: str = "any"
    mp_l1: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.intp))
    mp_l2: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.intp))
    mp_covered: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    #: The pattern selector (:func:`parse_pattern`); the counters (0-based
    #: indices into Roles.counters) the ``incomplete`` oddity ignored, and
    #: whether it could judge at all (False: every seed's ``incomplete`` bit is 0).
    require: tuple = ()
    ignore: tuple = ()
    incomplete_judged: bool = True
    #: Per chosen seed, the ToT of the S1 hit it borrows (the merged hit's:
    #: a NIM-only S1 hit has the substituted ToT); -1 without S1. None: read
    #: ``Frame.s_tot[s1_idx]``.
    s1_tot: np.ndarray | None = None

    @property
    def capped(self) -> bool:
        return self.n_examined < self.n_candidates


def _nearest_within(t_ref, t_sorted, window_ns) -> np.ndarray:
    """Index into ascending ``t_sorted`` of the hit nearest each ``t_ref`` within
    ``|dt| <= window_ns``, ties to the earlier; -1 where none. int64 times, exact."""
    t_ref = np.asarray(t_ref, dtype=np.int64)
    ts = np.asarray(t_sorted, dtype=np.int64)
    out = np.full(t_ref.shape, -1, dtype=np.intp)
    if ts.size == 0 or t_ref.size == 0:
        return out
    i = np.searchsorted(ts, t_ref, side="left")
    a = np.clip(i - 1, 0, ts.size - 1)
    b = np.clip(i, 0, ts.size - 1)
    big = np.iinfo(np.int64).max
    da = np.where(i > 0, np.abs(t_ref - ts[a]), big)
    db = np.where(i < ts.size, np.abs(ts[b] - t_ref), big)
    pick = np.where(da <= db, a, b)
    ok = np.minimum(da, db) <= window_ns
    out[ok] = pick[ok]
    return out


def seed_candidates(frame: Frame, roles: Roles, cuts: Cuts, mode: str) -> np.ndarray:
    """Indices into ``frame.s_*`` (ascending in time) of every possible seed.

    ``s1``: the S1 hits as a counter (:meth:`Frame.counter_hits_of`: its kept
    words, or the merged hits, each by its carrier word). ``ch<N>``: that
    channel's kept words, merged or not. ``any``: the first hit of each time
    cluster of the counter hits (S1..S5), a cluster being hits chained within
    ``cuts.coinc_ns`` of the one before.
    """
    return _candidates(frame, roles, cuts, mode).idx


def _candidates(frame: Frame, roles: Roles, cuts: Cuts, mode: str) -> CounterHits:
    """:func:`seed_candidates` with each candidate's time and ToT."""
    if mode == SEED_ANY:
        if frame.counter_hits:
            hs = [frame.counter_hits_of(c) for c in roles.counters]
            idx = np.concatenate([np.asarray(h.idx, dtype=np.intp) for h in hs])
            t = np.concatenate([h.t for h in hs])
            tot = np.concatenate([np.asarray(h.tot, dtype=np.int64) for h in hs])
            o = np.lexsort((idx, t))
            idx, t, tot = idx[o], t[o], tot[o]
        else:
            counters = np.asarray(roles.counters, dtype=frame.s_ch.dtype)
            idx = np.flatnonzero(np.isin(frame.s_ch, counters))
            t, tot = frame.s_t[idx], frame.s_tot[idx]
        if idx.size == 0:
            return CounterHits(t, tot, idx.astype(np.intp))
        first = np.empty(idx.size, dtype=bool)
        first[0] = True
        first[1:] = np.diff(t) > cuts.coinc_ns
        return CounterHits(t[first], tot[first], idx[first].astype(np.intp))
    if mode == SEED_S1:
        h = frame.counter_hits_of(roles.s1)
        return CounterHits(h.t, h.tot, np.asarray(h.idx, dtype=np.intp))
    i = np.asarray(frame.chan[int(mode[2:])], dtype=np.intp)
    return CounterHits(frame.s_t[i], frame.s_tot[i], i)


def select_seeds_by(frame: Frame, roles: Roles | None = None, cuts: Cuts | None = None,
                    mode: str = SEED_S1, filters=(), max_candidates: int = MAX_CANDIDATES,
                    mupix: str = "any", mcuts: MuPixCuts | None = None,
                    require=(), ignore=()) -> SeedSelection:
    """The display's seeds of one frame for a seed mode and a set of filters.

    The window, its completeness rule and ``n_seeds`` are :func:`analyse_frame`'s:
    a candidate's [t - pre, t + post] must lie inside :func:`complete_range` of
    the counters and the RF with at least ``seed_min_hits`` hits. Of those, the
    latest ``max_candidates`` are examined, and the latest ``n_seeds`` that have
    any of ``filters`` (all of them without filters) are chosen.

    Oddities, each over the seed's window or its coincidence window:

    * ``incomplete``: not every counter (S1..S5) has a hit within +-coinc of it.
      Counters in ``ignore`` (0-based indices into ``roles.counters``: those
      with a known timestamp fault, which are almost never in time) are left
      out of that; with fewer than two counters left there is nothing to judge
      and no seed has it (``SeedSelection.incomplete_judged`` False);
    * ``mismatch``: a kept hit in the window has fine and coarse disagreeing;
    * ``tot``: a kept hit in the window has ToT >= ``tot_corrupt_min``;
    * ``rf``: the S1 hit it borrows the RF from has no valid gate, or a vetoed
      one. A seed with no S1 hit near it has no RF measurement and does not
      match (``incomplete`` finds those).

    ``mupix`` (:data:`MUPIX_MODES`) is a selection, not an oddity: it is AND-ed
    with the filters (:func:`mupix_selects`), on L1 / L2 pixel hits in
    ``mcuts.window_ns`` around the seed. ``require`` (:func:`parse_pattern`) is
    another AND: per counter, a hit within +-coinc of the seed present or
    absent (:func:`pattern_selects`); explicit, so nothing is ignored there.
    """
    roles = roles or Roles()
    cuts = cuts or Cuts()
    mcuts = mcuts or MuPixCuts()
    mode = parse_seed(mode)
    filters = parse_filters(filters)
    mupix = parse_mupix(mupix)
    nc = len(roles.counters)
    require = parse_pattern(dict(require), nc)      # a dict or parse_pattern's pairs
    ignore = tuple(sorted({int(k) for k in ignore if 0 <= int(k) < nc}))
    s_t = frame.s_t
    cand_t, cand_tot, cand = _candidates(frame, roles, cuts, mode)
    t_c = cand_t
    counters = [frame.counter_times(c) for c in roles.counters]
    t_rf = frame.times(roles.rf)
    rng = complete_range([x for x in counters + [t_rf] if len(x) >= cuts.seed_min_hits])
    if rng is None:
        lo = hi = 0
    else:
        hi = int(np.searchsorted(t_c, rng[1] - cuts.seed_post_ns, side="right"))
        lo = int(np.searchsorted(t_c, rng[0] + cuts.seed_pre_ns, side="left"))
        hi = max(lo, hi)
    n_cand = hi - lo
    e0 = max(lo, hi - int(max_candidates))
    ex = cand[e0:hi]
    te = t_c[e0:hi]

    pattern, _counts = coincidence(te, counters, cuts.coinc_ns)
    full = (1 << len(counters)) - 1
    for k in ignore:
        full &= ~(1 << k)
    judged = nc - len(ignore) >= 2
    a = np.searchsorted(s_t, te - cuts.seed_pre_ns, side="left")
    b = np.searchsorted(s_t, te + cuts.seed_post_ns, side="right")
    if judged:
        odd = np.where((pattern & full) != full, FILTER_BITS["incomplete"], 0).astype(np.uint8)
    else:
        odd = np.zeros(ex.size, dtype=np.uint8)
    if ex.size:
        bad = np.zeros(s_t.size + 1, dtype=np.int64)
        np.cumsum(~frame.consistent[frame.order], out=bad[1:])
        odd |= np.where(bad[b] > bad[a], FILTER_BITS["mismatch"], 0).astype(np.uint8)
        hot = np.zeros(s_t.size + 1, dtype=np.int64)
        np.cumsum(frame.s_tot >= cuts.tot_corrupt_min, out=hot[1:])
        odd |= np.where(hot[b] > hot[a], FILTER_BITS["tot"], 0).astype(np.uint8)

    # RF: from S1 -- the seed itself, or the nearest S1 hit within the window
    # (S1 as a counter: merged when the plugin merged it).
    h1 = frame.counter_hits_of(roles.s1)
    i1 = np.asarray(h1.idx, dtype=np.intp)
    t1 = h1.t
    if mode == SEED_S1:
        j = np.arange(e0, hi, dtype=np.intp)          # candidates are the S1 hits
    else:
        j = _nearest_within(te, t1, cuts.coinc_ns)
    has = j >= 0
    rf_n = np.full(ex.size, -1, dtype=np.int32)
    rf_valid = np.zeros(ex.size, dtype=bool)
    rf_vetoed = np.zeros(ex.size, dtype=bool)
    rf_phase = np.full(ex.size, np.nan)
    rf_period = np.full(ex.size, np.nan)
    if has.any():
        vetoed_all, _gap = gate_veto(t1, cuts.rf_gate_ns)
        jj = j[has]
        n, valid, phase, period, _lo, _pt = burst_phase_many(
            t_rf, t1[jj], rule=cuts.rf_rule, gate_ns=cuts.rf_gate_ns,
            min_pulses=cuts.rf_min_pulses, max_pulses=cuts.rf_max_pulses,
            vetoed=vetoed_all[jj])
        rf_n[has], rf_valid[has], rf_vetoed[has] = n, valid, vetoed_all[jj]
        rf_phase[has], rf_period[has] = phase, period
        odd |= np.where(has & ~(rf_valid & ~rf_vetoed), FILTER_BITS["rf"], 0).astype(np.uint8)

    if filters:
        mask = np.uint8(sum(FILTER_BITS[f] for f in filters))
        match = (odd & mask) != 0
    else:
        match = np.ones(ex.size, dtype=bool)
    px = frame.px
    if px is not None and px.n and ex.size:
        wlo, whi = mcuts.window_ns
        mp_cov = mupix_coverage(te, px, wlo, whi)
        mp1, mp2 = mupix_in_window(te, px, wlo, whi)
    else:
        mp_cov = np.zeros(ex.size, dtype=bool)
        mp1 = mp2 = np.zeros(ex.size, dtype=np.intp)
    if mupix != "any":
        match &= mupix_selects(mupix, mp_cov, mp1, mp2)
    if require:
        match &= pattern_selects(require, pattern)
    mi = np.flatnonzero(match)
    pick = mi[max(0, mi.size - int(cuts.n_seeds)):] if cuts.n_seeds > 0 else mi[:0]
    s1_idx = np.full(pick.size, -1, dtype=np.intp)
    hp = has[pick]
    jp = j[pick][hp]
    s1_idx[hp] = i1[jp]
    s1_dt = np.zeros(pick.size, dtype=np.int64)
    s1_dt[hp] = t1[jp] - te[pick][hp]
    s1_tot = np.full(pick.size, -1, dtype=np.int64)
    s1_tot[hp] = h1.tot[jp]
    return SeedSelection(
        mode=mode, filters=filters, idx=ex[pick], t=te[pick].astype(np.int64),
        ch=frame.s_ch[ex[pick]], tot=cand_tot[e0:hi][pick], pattern=pattern[pick],
        odd=odd[pick], has_s1=hp, s1_idx=s1_idx, s1_dt=s1_dt, rf_n=rf_n[pick],
        rf_valid=rf_valid[pick],
        rf_vetoed=rf_vetoed[pick], rf_phase=rf_phase[pick], rf_period=rf_period[pick],
        n_candidates=n_cand, n_examined=int(ex.size), n_matching=int(mi.size),
        mupix=mupix, mp_l1=mp1[pick], mp_l2=mp2[pick], mp_covered=mp_cov[pick],
        require=require, ignore=ignore, incomplete_judged=judged, s1_tot=s1_tot)
