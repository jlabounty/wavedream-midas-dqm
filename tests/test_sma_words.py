"""``sma_words`` on synthetic words, where the right answer is known by construction.

The C++ parity block repeats the vectors of
``main/reco_testbeam/tests/test_psm_sma.cpp`` (``test_fine_coarse_diff``,
``test_frame_extent``); the RF block repeats the test vectors of
``scratch/sma-tot-vs-wd/BURST_RULE.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

from mdqm.plugins import sma_words as W

RNG = np.random.default_rng(20260928)


def words_at(ch, t, shift=14, tot=10, fine=None):
    """Trigger words for true times ``t`` (arrays), optionally with a forged fine field."""
    t = np.asarray(t, dtype=np.int64)
    coarse, f = W.fields_of(t, shift)
    if fine is not None:
        f = np.asarray(fine, dtype=np.uint32)
    ch = np.broadcast_to(np.asarray(ch), t.shape)
    tot = np.broadcast_to(np.asarray(tot), t.shape)
    return W.encode(ch, tot, coarse, f)


# ==============================================================================
# The word
# ==============================================================================

def test_encode_decode_roundtrip():
    w = np.array([W.encode(3, 254, 0xABCDEF1, 0x12345), W.FILLER, 0x0123456789ABCDEF],
                 dtype=np.uint64)
    d = W.decode(w, 14)
    assert d["n_filler"] == 1 and d["n_pixel"] == 1 and d["n_trigger"] == 1
    assert (d["ch"][0], d["tot"][0], d["coarse"][0], d["fine"][0]) == (3, 254, 0xABCDEF1, 0x12345)
    assert W.word_counts(w) == (1, 1, 1)
    # vectorised encode gives the same words as the scalar one
    arr = W.encode(np.array([3]), np.array([254]), np.array([0xABCDEF1]), np.array([0x12345]))
    assert int(arr[0]) == int(w[0])


def test_words_from_bank_accepts_dwords_bytes_and_u8():
    w = words_at(1, np.arange(5) * 1000 + 123456789)
    dw = w.view("<u4")
    assert dw.dtype == np.uint32 and dw.size == 10
    for src in (dw, dw.tobytes(), bytearray(dw.tobytes()), memoryview(dw.tobytes()), w):
        assert np.array_equal(W.words_from_bank(src), w)
    # an odd trailing dword is dropped
    assert np.array_equal(W.words_from_bank(np.r_[dw, np.uint32(7)]), w)


@pytest.mark.parametrize("shift", [12, 13, 14, 15, 16])
def test_time_of_recovers_the_true_time(shift):
    t = RNG.integers(0, W.TIME_WRAP_NS >> 1, 5000, dtype=np.int64)
    coarse, fine = W.fields_of(t, shift)
    assert np.array_equal(W.time_of(coarse, fine, shift), t & (W.TIME_WRAP_NS - 1))


def test_time_of_masks_to_the_mupix_wrap():
    t = np.array([W.TIME_WRAP_NS - 5, W.TIME_WRAP_NS + 7], dtype=np.int64)
    coarse, fine = W.fields_of(t, 14)
    assert list(W.time_of(coarse, fine, 14)) == [W.TIME_WRAP_NS - 5, 7]


def test_tot_254_is_flagged():
    w = words_at(1, [1000, 2000, 3000], tot=np.array([12, 254, 250]))
    d = W.decode(w, 14)
    assert list(d["tot"] >= W.TOT_CORRUPT_MIN) == [False, True, True]


# ==============================================================================
# Fine vs coarse: C++ parity (test_psm_sma.cpp, test_fine_coarse_diff)
# ==============================================================================

def cpp_fields(t, shift):
    """``FieldsOf`` of the C++ test."""
    return (t >> shift) & 0xFFFFFFF, t & 0xFFFFF


def test_cpp_shared_bits():
    assert W.shared_bits(14) == 6 and W.shared_bits(0) == 20 and W.shared_bits(18) == 2


@pytest.mark.parametrize("shift", [0, 3, 14, 15, 18])
@pytest.mark.parametrize("t", [0, 12345, 0x3FFFF, 1234567890123])
def test_cpp_one_instant_agrees(shift, t):
    c, f = cpp_fields(t, shift)
    assert W.fine_coarse_xor(c, f, shift) == 0
    assert W.fine_coarse_diff_ns(c, f, shift) == 0


def test_cpp_s4_bit16_flip():
    shift, t = 14, 987654321
    c, f = cpp_fields(t, shift)
    bad = f ^ (1 << 16)
    d = int(W.fine_coarse_diff_ns(c, bad, shift))
    assert d == (65536 if (f >> 16) & 1 else -65536)
    # the C++ test's tolerance is 20000 ns = 2^14 + the default margin
    assert W.latch_tolerance_ns(14) == 20000
    assert not W.fine_coarse_consistent(d, shift)
    assert int(W.fine_coarse_xor(c, bad, shift)) == 1 << 2        # bit 2 = fine bit 16
    d2 = int(W.fine_coarse_diff_ns(c, f ^ (3 << 16), shift))
    assert d2 % 65536 == 0 and abs(d2) >= 65536 and not W.fine_coarse_consistent(d2, shift)


def test_cpp_coarse_one_tick_ahead():
    shift = 14
    tc = (0x5A << 20) | (0x7 << 14) | 0x3FFF
    c, f = cpp_fields(tc, shift)
    assert W.fine_coarse_diff_ns(c + 1, f, shift) == 16384
    assert W.fine_coarse_consistent(16384, shift)
    assert W.fine_coarse_xor(c + 1, f, shift) == 0xF


def test_cpp_shift3_latch_lag():
    t3 = 0x1234567
    c, _f = cpp_fields(t3 - 10000, 3)
    fine3 = t3 & 0xFFFFF
    d3 = int(W.fine_coarse_diff_ns(c, fine3, 3))
    assert -10008 <= d3 <= -9992 and d3 % 8 == 0
    # the C++ uses a fixed 20000 ns; ours never goes below it either
    assert W.fine_coarse_consistent(d3, 3)
    assert W.fine_coarse_consistent(d3, 3, margin_ns=20000 - 8)
    assert not W.fine_coarse_consistent(W.fine_coarse_diff_ns(c, fine3 ^ (1 << 16), 3), 3,
                                        margin_ns=20000 - 8)


def test_cpp_wrap_and_shared_bits_only():
    c, f = cpp_fields(0xFFFFF, 0)
    assert W.fine_coarse_diff_ns(c + 1, f, 0) == 1
    assert W.fine_coarse_diff_ns(0, 1 << 19, 0) == -(1 << 19)
    c, f = cpp_fields(987654321, 14)
    assert W.fine_coarse_diff_ns(c ^ (1 << 6), f, 14) == 0
    assert W.fine_coarse_diff_ns(c, f ^ 0x3FFF, 14) == 0


def test_cpp_frame_extent():
    assert W.wrap_signed(5, 10) == 5 and W.wrap_signed(-5, 10) == -5
    assert W.wrap_signed(1024 - 3, 10) == -3 and W.wrap_signed(-(1024 - 3), 10) == 3
    assert W.wrap_signed(512, 10) == -512
    bits = 31
    span = 1 << bits
    e = W.extent([1000, 500, 3000, 2000], bits)
    assert (e.first, e.last, e.span_ns) == (500, 3000, 2500)
    w = W.extent([span - 100, span - 50, 150, 200], bits)
    assert w.span_ns == 300 and w.first == span - 100
    nxt = W.extent([1200, 1500], bits)
    assert W.gap_ns(w, nxt, bits) == 1000
    assert W.gap_ns(nxt, w, bits) < 0
    assert W.gap_ns(e, W.extent([3000 + 88000000], bits), bits) == 88000000


# ==============================================================================
# Fine vs coarse on synthetic frames: the real fault signatures
# ==============================================================================

def test_s4_fine_bit16_flip_is_found_and_named():
    t = np.sort(RNG.integers(10**11, 10**11 + 20_000_000, 400))
    good = words_at(4, t[:200])
    c, f = W.fields_of(t[200:], 14)
    bad = W.encode(np.full(200, 4), np.full(200, 9), c, f ^ np.uint32(1 << 16))
    d = W.decode(np.r_[good, bad], 14)
    diff = W.fine_coarse_diff_ns(d["coarse"], d["fine"], 14)
    ok = W.fine_coarse_consistent(diff, 14)
    assert ok[:200].all() and not ok[200:].any()
    assert set(np.abs(diff[200:])) == {65536}
    xor = W.fine_coarse_xor(d["coarse"], d["fine"], 14)
    per_bit = W.per_channel_bit_counts(d["ch"][~ok], xor[~ok], W.shared_bits(14))
    assert per_bit[4, 16 - 14] == 200 and per_bit.sum() == 200


def test_s5_fine_is_half_the_time():
    """S5 fault: fine bits 18:0 hold time bits 19:1; the coarse field is right."""
    t = np.sort(RNG.integers(10**11, 10**11 + 20_000_000, 2000))
    c, _f = W.fields_of(t, 14)
    fine_half = ((t >> 1) & 0xFFFFF).astype(np.uint32)
    diff = W.fine_coarse_diff_ns(c, fine_half, 14)
    assert W.fine_coarse_consistent(diff, 14).mean() < 0.2


def test_per_channel_bit_counts_matches_a_loop():
    ch = RNG.integers(0, 16, 3000)
    v = RNG.integers(0, 1 << 20, 3000).astype(np.uint32)
    got = W.per_channel_bit_counts(ch, v, 20)
    want = np.array([[np.count_nonzero((v[ch == c] >> b) & 1) for b in range(20)]
                     for c in range(16)])
    assert np.array_equal(got, want)


# ==============================================================================
# Shift scan
# ==============================================================================

@pytest.mark.parametrize("true_shift", [12, 13, 14, 15, 16])
def test_shift_scan_prefers_the_true_shift(true_shift):
    t = np.sort(RNG.integers(10**11, 10**11 + 30_000_000, 3000))
    c, f = W.fields_of(t, true_shift)
    counts = W.shift_scan(c, f)
    assert W.SHIFT_SCAN[int(np.argmax(counts))] == true_shift
    assert counts[W.SHIFT_SCAN.index(true_shift)] == t.size
    others = np.delete(counts, W.SHIFT_SCAN.index(true_shift))
    assert others.max() < 0.3 * t.size


def test_shift_scan_survives_a_coarse_tick_ahead():
    t = np.sort(RNG.integers(10**11, 10**11 + 30_000_000, 3000))
    c, f = W.fields_of(t, 14)
    counts = W.shift_scan(c + np.uint32(1), f)
    assert counts[W.SHIFT_SCAN.index(14)] == t.size


# ==============================================================================
# Stale hits
# ==============================================================================

def test_stale_cluster_is_dropped():
    main = 5 * 10**11 + np.sort(RNG.integers(0, 26_000_000, 1000))     # 26 ms frame
    old = 5 * 10**11 - 700 * 10**9 + RNG.integers(0, 400_000, 20)      # 700 s earlier
    later = 5 * 10**11 + 2 * 10**9 + np.arange(5)                      # 2 s later
    t = np.r_[old[:10], main[:500], later, main[500:], old[10:]]
    keep = W.stale_mask(t)
    assert keep.sum() == 1000
    assert np.array_equal(np.sort(t[keep]), main)


def test_stale_keeps_a_frame_with_short_gaps_whole():
    t = np.cumsum(RNG.integers(1, 900_000, 200))            # gaps up to 0.9 ms
    assert W.stale_mask(t).all()
    assert W.stale_mask(t, gap_ns=100_000).sum() < t.size


def test_stale_and_sort_across_the_2_40_wrap():
    t = np.r_[np.arange(W.TIME_WRAP_NS - 5000, W.TIME_WRAP_NS, 100),
              np.arange(0, 5000, 100)].astype(np.int64)
    words = words_at(1, RNG.permutation(t))
    fr = W.prepare_frame(words, 14)
    assert fr.keep.all() and fr.span_ns == 9900
    assert np.all(np.diff(fr.s_t) == 100)


def test_prepare_frame_counts_stale_per_channel():
    main = 10**11 + np.arange(0, 1_000_000, 1000)
    words = np.r_[words_at(1, main), words_at(7, [10**11 - 5 * 10**9] * 3),
                  np.array([W.FILLER] * 4, dtype=np.uint64), np.arange(6, dtype=np.uint64)]
    fr = W.prepare_frame(words, 14)
    assert (fr.n_words, fr.n_filler, fr.n_pixel, fr.n_trigger) == (1013, 4, 6, 1003)
    assert fr.stale_per_ch[7] == 3 and fr.stale_per_ch.sum() == 3
    assert fr.first == main[0] and fr.last == main[-1]


# ==============================================================================
# RF: BURST_RULE.md test vectors
# ==============================================================================

def bp(gates, rf, rule="last", gate_ns=W.RF_GATE_NS):
    n, valid, phase, period, _lo, _pt = W.burst_phase_many(np.array(rf, float),
                                                           np.array(gates, float), rule, gate_ns)
    return n, valid, phase, period


@pytest.mark.parametrize("rf, n, phase, period", [
    ([1047, 1079, 1099], 3, 99, 20),                     # L1
    ([1047, 1066, 1086, 1106], 4, 106, 20),              # L2
    ([1047, 1066, 1086, 1106, 1125], 5, None, None),     # L3
    ([1047], 1, None, None),                             # L4
    ([], 0, None, None),                                 # L5
    ([990, 1047, 1079, 1099, 1250], 3, 99, 20),          # L7
])
def test_last_rule_vectors(rf, n, phase, period):
    got_n, valid, ph, pe = bp([1000], rf)
    assert got_n[0] == n
    if phase is None:
        assert not valid[0] and np.isnan(ph[0]) and np.isnan(pe[0])
    else:
        assert valid[0] and ph[0] == phase and pe[0] == period


@pytest.mark.parametrize("rf, n, phase, period", [
    ([1047, 1066, 1086, 1106], 4, 86, 20),                # 1
    ([1047, 1066, 1086], 3, None, None),                  # 2
    ([990, 1047, 1066, 1086, 1106, 1250], 4, 86, 20),     # 3
    ([1000, 1050, 1070, 1090, 1125], 4, 90, 35),          # 4
    ([1047, 1066, 1086, 1106, 1125], 5, None, None),      # 6
    ([], 0, None, None),                                  # 7
])
def test_dqm_rule_vectors(rf, n, phase, period):
    got_n, valid, ph, pe = bp([1000], rf, "dqm")
    assert got_n[0] == n
    if phase is None:
        assert not valid[0]
    else:
        assert valid[0] and ph[0] == phase and pe[0] == period


def test_gate_veto_vectors():
    # P1: 1000 vetoed (n=3 kept, gap 120), 1120 valid phase 99 period 20
    n, valid, ph, pe = bp([1000, 1120], [1047, 1079, 1099, 1167, 1199, 1219])
    assert list(n) == [3, 3] and list(valid) == [False, True] and ph[1] == 99 and pe[1] == 20
    v, gap = W.gate_veto([1000, 1120])
    assert list(v) == [True, False] and gap[0] == 120 and np.isinf(gap[1])
    # P2: 1000 vetoed; 1070 has the first burst's tail: n=5 invalid
    n, valid, _ph, _pe = bp([1000, 1070], [1047, 1079, 1099, 1117, 1149, 1169])
    assert list(valid) == [False, False] and n[1] == 5
    # P2': gate 200
    n, valid, ph, _pe = bp([1000, 1140], [1047, 1079, 1099, 1187, 1219, 1239], gate_ns=200)
    assert list(valid) == [False, True] and ph[1] == 99
    # P3: 1126 is outside the gate; 1125 is inside (edge inclusive)
    _n, valid, ph, _pe = bp([1000, 1126], [1047, 1079, 1099, 1173, 1205, 1225])
    assert list(valid) == [True, True] and list(ph) == [99, 99]
    assert list(W.gate_veto([1000, 1125])[0]) == [True, False]
    # P4: equal times do not veto each other; a third S1 inside vetoes both
    _n, valid, ph, _pe = bp([1000, 1000], [1047, 1079, 1099])
    assert list(valid) == [True, True] and list(ph) == [99, 99]
    assert list(W.gate_veto([1000, 1000, 1060])[0]) == [True, True, False]


def test_partner_pairing_vectors():
    # 8 (dqm rule) and L6: partner phase = phase pulse - t_partner, within 50 ns
    rf = np.array([1047., 1066, 1086, 1106])
    n, valid, _ph, _pe, _lo, pt = W.burst_phase_many(rf, np.array([1000.]), "dqm")
    idx, dt, ph = W.match_to_gates([1010., 1100.], [1000.], pt[valid])
    assert list(idx) == [0, -1] and ph[0] == 76 and dt[0] == 10 and np.isnan(ph[1])
    # P5: S3@1010 unmatched (1000 vetoed); S3@1130 pairs with 1120, phase 89
    g = np.array([1000., 1120.])
    _n, valid, _ph, _pe, _lo, pt = W.burst_phase_many(
        np.array([1047., 1079, 1099, 1167, 1199, 1219]), g)
    idx, _dt, ph = W.match_to_gates([1010., 1130.], g[valid], pt[valid])
    assert idx[0] == -1 and ph[1] == 89
    # 9 (each gate alone): the tie at |dt| = 20 goes to the earlier gate, phase 66
    idx, _dt, ph = W.match_to_gates([1020.], [1000., 1040.], [1086., 1086.])
    assert idx[0] == 0 and ph[0] == 66
    # 9 under the veto: 1000 is vetoed, S2 pairs with 1040, phase 1086 - 1020 = 66
    g = np.array([1000., 1040.])
    _n, valid, _ph, _pe, _lo, pt = W.burst_phase_many(rf, g, "dqm")
    assert list(valid) == [False, True]
    idx, _dt, ph = W.match_to_gates([1020.], g[valid], pt[valid])
    assert idx[0] == 0 and ph[0] == 66


# ==============================================================================
# Coincidences and seeds
# ==============================================================================

def test_window_pairs_matches_brute_force():
    a = np.sort(RNG.integers(0, 100_000, 300))
    b = np.sort(RNG.integers(0, 100_000, 500))
    i, j, dt = W.window_pairs(a, b, -200, 300)
    want = sorted((x, y) for x in range(a.size) for y in range(b.size)
                  if -200 <= b[y] - a[x] <= 300)
    assert sorted(zip(i.tolist(), j.tolist(), strict=True)) == want
    assert np.array_equal(dt, b[j] - a[i])
    assert np.array_equal(W.window_counts(a, b, -200, 300), np.bincount(i, minlength=a.size))
    e = W.window_pairs(a, np.zeros(0, dtype=np.int64), -1, 1)
    assert all(x.size == 0 for x in e)


def test_coincidence_pattern():
    s1 = np.array([1000, 5000, 9000])
    s2 = np.array([1010, 9060])
    s3 = np.array([990, 1049, 5051])
    s4 = np.array([], dtype=np.int64)
    s5 = np.array([8950])
    pattern, counts = W.coincidence(s1, [s1, s2, s3, s4, s5], 50)
    assert list(pattern) == [0b00111, 0b00001, 0b10001]
    assert counts[0].tolist() == [1, 1, 2, 0, 0]


def test_seed_completeness():
    t_s1 = np.arange(0, 100_000, 1000)                       # S1 every us up to 99 us
    counters = [t_s1, np.array([100, 50_000, 96_500]), np.array([300, 99_900])]
    start, end = W.complete_range(counters + [np.zeros(0, dtype=np.int64)])
    assert (start, end) == (300, 96_500)
    seeds = W.select_seeds(t_s1, start, end, n_seeds=4, pre_ns=200, post_ns=3000)
    # t + 3000 <= 96500 -> t <= 93500 -> latest four S1: 90..93 us
    assert t_s1[seeds].tolist() == [90_000, 91_000, 92_000, 93_000]
    # t - 200 >= 300: the first S1 at 0 is never a seed
    assert W.select_seeds(t_s1, start, end, n_seeds=1000).min() >= 1
    assert W.select_seeds(t_s1, 0, 1000).size == 0
    assert W.complete_range([np.zeros(0)]) is None


def test_analyse_frame_end_to_end():
    """A clean synthetic frame: 4-counter coincidences with a 3-pulse RF burst each."""
    t1 = 10**11 + np.arange(0, 2_000_000, 10_000)            # 200 S1 hits, 10 us apart
    parts = [words_at(1, t1)]
    for c, d in ((2, 3), (3, 5), (4, 7), (5, 9)):
        parts.append(words_at(c, t1 + d))
    parts.append(words_at(6, np.concatenate([t1 + 47, t1 + 79, t1 + 99])))
    parts.append(words_at(9, t1[::10] + 2200))               # delayed hit 2.2 us after
    parts.append(words_at(7, 10**11 + np.arange(0, 2_000_000, 777)))
    words = np.concatenate(parts)
    fr = W.prepare_frame(RNG.permutation(words), 14)
    assert fr.consistent.all() and fr.keep.all()
    a = W.analyse_frame(fr)
    assert a.rf_valid.all() and np.all(a.rf_phase == 99) and np.all(a.rf_period == 20)
    assert np.all(a.pattern == 0b11111)
    assert np.all(a.partner_counts == 1)
    assert set(a.dt[4][1].tolist()) == {7}
    i9, dt9 = a.delayed_dt[9]
    assert i9.size == 20 and set(dt9.tolist()) == {2200}
    assert a.shift_counts[W.SHIFT_SCAN.index(14)] == 200
    assert a.seeds.size == 4 and a.t_s1[a.seeds[-1]] + 3000 <= fr.times(6)[-1]


# ==============================================================================
# Review fixes: latch floor, shift-3 time wrap, far genuine clusters, seeds
# ==============================================================================

def test_the_latch_tolerance_never_drops_below_the_reco_20_us():
    assert W.latch_tolerance_ns(3) == 20000
    assert W.latch_tolerance_ns(14) == 20000
    assert W.latch_tolerance_ns(16) == 65536 + W.LATCH_MARGIN_NS
    # a shift-3 word whose coarse field was latched 15 us late is genuine
    _c, f = W.fields_of(10**9, 3)
    late, _f = W.fields_of(10**9 + 15000, 3)
    assert W.fine_coarse_consistent(W.fine_coarse_diff_ns(late, f, 3), 3)


def test_a_shift_3_frame_across_the_2_31_wrap_stays_one_cluster():
    assert W.time_bits(3) == 31 and W.time_bits(14) == 40
    t = (1 << 31) - 1_000_000 + np.arange(0, 2_000_000, 1000)       # straddles 2^31 ns
    c, f = W.fields_of(t, 3)
    fr = W.prepare_frame(W.encode(np.full(t.size, 1), np.full(t.size, 9), c, f), 3)
    assert fr.keep.all() and fr.time_bits == 31
    assert fr.span_ns == 1_999_000
    assert np.all(np.diff(fr.s_t) == 1000)


def _cluster_words(t, consistent=True):
    c, f = W.fields_of(t, 14)
    if not consistent:
        c = RNG.integers(0, 1 << 28, t.size).astype(np.uint32)
    return W.encode(np.full(t.size, 2), np.full(t.size, 9), c, f)


def test_a_far_consistent_cluster_is_kept_a_replayed_one_is_not():
    main = 10**12 + np.arange(0, 20_000_000, 20_000)                   # 1000 hits, 20 ms
    trip = 10**12 + 3 * 10**9 + np.arange(0, 30_000, 1000)             # 30 hits 3 s later
    old = 10**12 - 400 * 10**9 + np.arange(0, 30_000, 1000)            # 30 old, inconsistent
    few = 10**12 + 6 * 10**9 + np.arange(0, 5000, 1000)                # 5 consistent: too few
    words = np.r_[_cluster_words(main), _cluster_words(trip), _cluster_words(old, False),
                  _cluster_words(few)]
    fr = W.prepare_frame(words, 14)
    assert fr.n_rescued == 30
    assert int(fr.keep.sum()) == 1030 and int(fr.stale_per_ch.sum()) == 35
    assert fr.first == main[0] and fr.last == trip[-1]
    # the time-only rule still keeps the median cluster alone
    assert W.stale_mask(W.decode(words)["time"]).sum() == 1000
    # the rescue is what keeps it...
    assert W.prepare_frame(words, 14, rescue_min_fraction=1.01).n_rescued == 0
    # ...and it does not depend on the configured shift being right
    assert W.prepare_frame(words, 13).n_rescued == 30


def test_the_shift_scan_counts_every_s1_word_not_only_the_kept_ones():
    t1 = 10**11 + np.arange(0, 2_000_000, 10_000)
    far = 10**11 - 300 * 10**9 + np.arange(5)                        # 5 stale S1 words
    fr = W.prepare_frame(np.r_[words_at(1, t1), words_at(1, far)], 14)
    assert fr.chan[1].size == 200
    a = W.analyse_frame(fr)
    assert a.shift_counts[W.SHIFT_SCAN.index(14)] == 205


def test_seeds_ignore_a_counter_with_too_few_hits():
    t1 = 10**11 + np.arange(0, 2_000_000, 10_000)
    parts = [words_at(1, t1), words_at(2, t1 + 3), words_at(3, t1 + 5), words_at(4, t1 + 7),
             words_at(5, t1[:2] + 9),                                # S5: two hits only
             words_at(6, np.concatenate([t1 + 47, t1 + 79]))]
    fr = W.prepare_frame(np.concatenate(parts), 14)
    assert W.analyse_frame(fr).seeds.size == 4
    strict = W.Cuts(seed_min_hits=1)
    assert W.analyse_frame(fr, cuts=strict).seeds.size == 0
