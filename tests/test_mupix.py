"""MuPix pixel words in the SMA DQM: decode, planes, S1 matching, sync flag, smaf pixel block.

Synthetic frames are built event by event (``test_sma_seed_choice.build``) and
get pixel words with known chip, time and ToT (``sma_words.encode_pixel``), so
every count is known. The golden comparison against the reference decoders is
``test_mupix_golden.py``; the physics on real frames is in the docs.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from test_sma_plugin import R682, R1008, _Clock, _dispatch, _feed, _plugin
from test_sma_seed_choice import SPACING, T0, build, ev

from mdqm.dqm import framing
from mdqm.plugins import sma as P
from mdqm.plugins import sma_words as W


def pixels(hits, ts2_shift=5):
    """Pixel words of ``[(t_ns, chip, tot), ...]`` or ``[(t_ns, chip, tot, col, row)]``."""
    out = []
    for h in hits:
        t, chip, tot = h[:3]
        col, row = (h[3], h[4]) if len(h) > 3 else (10, 20)
        tick = int(t) // 8
        out.append(W.encode_pixel(chip, col, row, int(W.ts2_of(tick, tot, ts2_shift)), tick))
    return np.array(out, dtype=np.uint64)


def frame_words(n=40, l1=(-40,), l2=(-32,), noise=(), events=None):
    """``n`` healthy events (S1..S5, RF), and per event a pixel hit on chip 1 (L1) at
    each ``l1`` offset from S1 and on chip 5 (L2) at each ``l2``; ``noise`` are extra
    ``(t_ns, chip, tot)`` hits. The pixel words follow the trigger words, as in
    the real bank."""
    sma = build(events if events is not None else [ev() for _ in range(n)])
    n = n if events is None else len(events)
    hits = []
    for k in range(n):
        t1 = T0 + k * SPACING
        hits += [(t1 + d, 1, 12) for d in l1] + [(t1 + d, 5, 9) for d in l2]
    hits += list(noise)
    hits.sort()
    return np.concatenate([sma, pixels(hits)]) if hits else sma


# --- decode, planes, sort, wrap, cap ----------------------------------------------------------

def test_the_plane_map_and_its_defaults():
    lut = W.plane_lookup()
    assert list(np.flatnonzero(lut == W.PLANE_L1)) == [0, 1, 2, 3]
    assert list(np.flatnonzero(lut == W.PLANE_L2)) == [4, 5, 6, 7]
    assert (lut[8:] == W.PLANE_NONE).all()
    old = W.plane_lookup((1, 2, 3, 4), (5, 6, 7, 0))       # runs <= 186
    assert old[0] == W.PLANE_L2 and old[4] == W.PLANE_L1
    cfg = P.parse_settings({"MuPix": {"L1 chips": [1, 2, 3, 4], "L2 chips": [5, 6, 7, 0]}})
    assert cfg.mupix.l1 == (1, 2, 3, 4) and cfg.mupix.l2 == (5, 6, 7, 0) and not cfg.errors
    assert cfg.mupix.planes[0] == W.PLANE_L2


@pytest.mark.parametrize("tree, err", [
    ({"L1 chips": [0, 1], "L2 chips": [1, 2]}, "share chip"),
    ({"L1 chips": [0, 40]}, "L1 chips"),
    ({"window lo ns": 100, "window hi ns": 50}, "window hi"),
    ({"sideband lo ns": -400, "sideband hi ns": -100}, "sideband"),       # overlaps the window
    ({"sideband lo ns": 500, "sideband hi ns": 1100}, "sideband"),        # the delayed region
    ({"ts2 shift": 40}, "ts2 shift"),
    ({"max pixel hits per frame": -1}, "max pixel"),
])
def test_bad_mupix_settings_fall_back_and_are_reported(tree, err):
    cfg = P.parse_settings({"MuPix": tree})
    assert any(err in e for e in cfg.errors), cfg.errors
    d = W.MuPixCuts()
    assert cfg.mupix.window_ns == d.window_ns and cfg.mupix.sideband_ns == d.sideband_ns
    assert not set(cfg.mupix.l1) & set(cfg.mupix.l2)


def test_a_mupix_edit_rebuilds_but_the_sync_threshold_does_not():
    base = P.shape_fingerprint({})
    assert P.shape_fingerprint({"MuPix": {"window hi ns": 400}}) != base
    assert P.shape_fingerprint({"Self check": {"mupix sync min fraction": 0.5}}) == base


def test_tot_is_ts2_minus_the_time_stamp_on_ts2s_scale():
    tick = np.array([0, 31, 32, 1000, 2**37 - 1], dtype=np.int64)
    for tot in (0, 1, 7, 31):
        ts2 = W.ts2_of(tick, tot)
        assert (W.pixel_tot(ts2, tick, 5) == tot).all()
    # Another divider: TS2 counts every 2^4 ticks.
    assert (W.pixel_tot(W.ts2_of(tick, 9, 4), tick, 4) == 9).all()
    cuts = W.MuPixCuts(ts2_shift=4)
    assert cuts.tot_ns == 128 and W.MuPixCuts().tot_ns == 256


def test_an_unsorted_stream_is_sorted_and_split_by_plane():
    hits = [(T0 + 800, 1, 3), (T0 + 80, 5, 4), (T0 + 400, 9, 5), (T0 + 8, 2, 6)]
    px = W.prepare_pixels(pixels(hits))
    assert not px.was_sorted
    assert list(px.t) == sorted(h[0] for h in hits)
    assert list(px.chip) == [2, 5, 9, 1]
    assert list(px.plane) == [W.PLANE_L1, W.PLANE_L2, W.PLANE_NONE, W.PLANE_L1]
    assert list(px.times(W.PLANE_L1)) == [T0 + 8, T0 + 800]
    assert list(px.times(W.PLANE_NONE)) == [T0 + 400]
    assert list(px.tot) == [6, 4, 5, 3]
    # word_index and raw travel with their hit.
    assert np.array_equal(pixels(hits)[px.word_index], px.raw)
    assert W.prepare_pixels(pixels(sorted(hits))).was_sorted


def test_the_2_to_the_40_wrap_is_unwrapped_like_the_sma_times():
    wrap = W.TIME_WRAP_NS
    hits = [(wrap - 800, 1, 3), (wrap - 16, 5, 3), (8, 1, 3), (400, 5, 3)]
    words = pixels(hits)
    # Pixel times are the 37-bit stamp x 8 ns: the same 2^40 ns epoch as time_of.
    assert list(W.pixel_decode(words)["time"]) == [wrap - 800, wrap - 16, 8, 400]
    px = W.prepare_pixels(words, ref_ns=wrap - 100)
    assert list(px.t) == [wrap - 800, wrap - 16, wrap + 8, wrap + 400]
    assert px.was_sorted
    # An S1 hit just before the wrap sees the pixels just after it.
    t1 = np.array([wrap - 20])
    n1, n2 = W.mupix_in_window(t1, px, -150, 450)
    assert (int(n1[0]), int(n2[0])) == (1, 2)


def test_a_frame_straddling_the_wrap_matches_across_it():
    """The whole chain: SMA and pixel times on both sides of 2^40 ns."""
    wrap = W.TIME_WRAP_NS
    t = np.array([wrap - 5000, wrap - 5000 + 3, wrap + 5000, wrap + 5000 + 3], dtype=np.int64)
    ch = np.array([1, 2, 1, 2])
    coarse, fine = W.fields_of(t & (wrap - 1))
    sma = W.encode(ch, np.full(4, 20), coarse, fine)
    pw = pixels([(wrap - 9000, 7, 1), (wrap - 5040, 1, 5), (wrap + 4960, 5, 5),
                 (wrap + 9000, 7, 1)])
    fr = W.prepare_frame(np.concatenate([sma, pw]), mupix=W.MuPixCuts())
    assert fr.span_ns == 10003
    assert list(fr.px.t[1:3] - fr.times(1)) == [-40, -40]
    m = W.mupix_match(fr.times(1), fr.px, (-150, 450), (-50, -45))
    assert list(m.in_l1) == [True, False] and list(m.in_l2) == [False, True]


def test_the_cap_examines_the_latest_pixels_and_counts_the_rest():
    hits = [(T0 + 8 * k, 1 + k % 2, 3) for k in range(100)]
    px = W.prepare_pixels(pixels(hits), W.MuPixCuts(max_pixels=30))
    assert (px.n_words, px.n, px.n_skipped) == (100, 30, 70)
    assert px.first == T0 + 8 * 70 and px.last == T0 + 8 * 99
    off = W.prepare_frame(frame_words(n=5), mupix=W.MuPixCuts(max_pixels=0))
    assert off.px is None and off.n_pixel == 10, "MuPix off: counted, not analysed"
    assert W.prepare_frame(frame_words(n=5)).px is None, "no MuPix unless asked"


# --- the S1 matching ----------------------------------------------------------------------------

def test_in_time_and_sideband_counts_and_the_coverage():
    t1 = T0 + SPACING * np.arange(10)
    hits = []
    for k, t in enumerate(t1):
        if k % 2 == 0:
            hits.append((t - 40, 1, 12))              # L1 in time
        if k % 3 == 0:
            hits.append((t + 448, 5, 1))              # L2, the last 8 ns tick in [-150, 450)
        if k == 4:
            hits.append((t + 456, 6, 1))              # L2 just outside it
        if k in (3, 6):
            hits.append((t - 2000, 2, 3))             # L1 in the sideband
    hits += [(T0 - 3000, 7, 1), (t1[-1] + 3000, 7, 1)]    # the pixel data covers every S1 hit
    px = W.prepare_pixels(pixels(sorted(hits)))
    m = W.mupix_match(t1, px, (-150, 450), (-2400, -1800))
    assert m.n == 10
    n, fin, fside = m.counts()
    assert list(fin) == [5, 4, 2] and list(fside) == [2, 0, 0]
    # Without pixels before the first S1 hit's sideband, it is not judged.
    px2 = W.prepare_pixels(pixels(sorted(hits[:-2])))
    m2 = W.mupix_match(t1, px2, (-150, 450), (-2400, -1800))
    assert not m2.judged[0] and not m2.in_l1[0]
    assert W.mupix_match(t1, None).n == 0


def test_the_accidental_correction():
    assert W.accidental_corrected(0.9, 0.0) == pytest.approx(0.9)
    assert W.accidental_corrected(0.1, 0.1) == pytest.approx(0.0)
    # P(in) = 1 - (1 - p)(1 - a): p = 0.8, a = 0.25 gives 0.85.
    assert W.accidental_corrected(0.85, 0.25) == pytest.approx(0.8)
    # A sideband twice as wide: a = 1 - (1 - 0.4375)^(1/2) = 0.25.
    assert W.accidental_corrected(0.85, 0.4375, 600, 1200) == pytest.approx(0.8)
    assert W.accidental_corrected(None, 0.1) is None
    assert W.accidental_corrected(0.5, 1.0) is None


def test_all_pairs_are_bounded():
    t1 = T0 + 1000 * np.arange(100)
    tp = np.sort(np.concatenate([t1 - 40, t1 + 100]))
    dt, used = W.mupix_pairs(t1, tp, -2560, 2000)
    assert used == 100 and dt.min() >= -2560 and dt.max() < 2000
    assert np.count_nonzero(dt == -40) == 100 and np.count_nonzero(dt == 100) == 100
    # ~900 pairs; at most ~100 asked for: every 9th S1 row.
    dt2, used2 = W.mupix_pairs(t1, tp, -2560, 2000, max_pairs=100)
    assert used2 == 12 and dt2.size < 130
    assert np.count_nonzero(dt2 == -40) == 12


# --- the event display's MuPix selector -------------------------------------------------------

def _mixed_frame():
    """12 events: pixel pattern per event k: 0-2 both, 3-5 L1 only, 6-8 none, 9-11 both;
    event 9 and 10 have an incomplete pattern (no S3)."""
    events = [ev() for _ in range(9)] + [ev(drop=[3]), ev(drop=[3]), ev()]
    hits = []
    for k in range(12):
        t = T0 + k * SPACING
        if k < 6 or k >= 9:
            hits.append((t - 40, 1, 12))
        if k < 3 or k >= 9:
            hits.append((t + 24, 5, 9))
    hits += [(T0 - 4000, 7, 1), (T0 + 12 * SPACING + 4000, 7, 1)]
    return np.concatenate([build(events), pixels(sorted(hits))])


@pytest.mark.parametrize("mode, filters, want", [
    # Event 0 and 11 have no complete window; the latest 4 matching seeds are shown.
    ("any", (), [7, 8, 9, 10]),
    ("both", (), [1, 2, 9, 10]),
    ("either", (), [4, 5, 9, 10]),
    ("none", (), [6, 7, 8]),
    ("both", ("incomplete",), [9, 10]),       # AND with an oddity filter
    ("none", ("incomplete",), []),
    ("either", ("incomplete", "tot"), [9, 10]),
])
def test_the_mupix_selector_is_an_and_on_the_filters(mode, filters, want):
    fr = W.prepare_frame(_mixed_frame(), mupix=W.MuPixCuts())
    cuts = W.Cuts(n_seeds=4)
    sel = W.select_seeds_by(fr, W.Roles(), cuts, "s1", filters, mupix=mode, mcuts=W.MuPixCuts())
    got = [int(round((t - T0) / SPACING)) for t in sel.t]
    assert got == want
    for k, n1, n2 in zip(got, sel.mp_l1, sel.mp_l2, strict=True):
        assert (int(n1) > 0) == (k < 6 or k >= 9)
        assert (int(n2) > 0) == (k < 3 or k >= 9)


@pytest.mark.parametrize("mode, filters, pattern, want", [
    # Events 9 and 10 have no S3; pixels: 0-2 both, 3-5 L1 only, 6-8 none, 9-11 both.
    ("both", (), {"3": "absent"}, [9, 10]),
    ("both", (), {"3": "present"}, [1, 2]),
    ("either", (), {"3": "present"}, [2, 3, 4, 5]),
    ("none", (), {"3": "absent"}, []),
    ("any", (), {"1": "present", "3": "absent"}, [9, 10]),
    ("both", ("incomplete",), {"3": "absent"}, [9, 10]),
    ("both", ("incomplete",), {"2": "absent"}, []),
    ("both", ("tot",), {"3": "absent"}, []),
    ("either", ("incomplete", "tot"), {"1": "present"}, [9, 10]),
])
def test_the_pattern_selector_is_an_and_with_mupix_and_the_filters(mode, filters, pattern, want):
    fr = W.prepare_frame(_mixed_frame(), mupix=W.MuPixCuts())
    sel = W.select_seeds_by(fr, W.Roles(), W.Cuts(n_seeds=4), "s1", filters, mupix=mode,
                            mcuts=W.MuPixCuts(), require=W.parse_pattern(pattern))
    assert [int(round((t - T0) / SPACING)) for t in sel.t] == want
    # Through the plugin: the reply names the pattern and MuPix both.
    p = _plugin()
    _feed(p, _mixed_frame(), run=1)
    d = framing.decode_sma_frame(p.frame_blob("seeded", filters=list(filters), mupix=mode,
                                              pattern=pattern))
    sel = d["meta"]["select"]
    assert sel["pattern"] == {k: v for k, v in pattern.items() if v != "any"}
    assert sel.get("mupix", "any") == mode


def test_a_hit_counts_past_the_end_of_the_pixel_data_an_absence_does_not():
    """The pixel data ends 32 ns after event 18's S1: event 18's window runs past
    it but holds hits (a hit is a hit); events 19-22 have none, and that says
    nothing (MuPix n/a), so "none" does not pick them."""
    hits = ([(T0 - 4000, 7, 1)] + [(T0 + k * SPACING - 40, 1, 12) for k in range(19)]
            + [(T0 + k * SPACING - 32, 5, 9) for k in range(19)])
    words = np.concatenate([build([ev() for _ in range(24)]), pixels(sorted(hits))])
    fr = W.prepare_frame(words, mupix=W.MuPixCuts())
    sel = W.select_seeds_by(fr, mupix="both")
    assert [int(round((t - T0) / SPACING)) for t in sel.t] == [15, 16, 17, 18]
    assert list(sel.mp_covered) == [True, True, True, False]
    assert list(sel.mp_l1) == [1, 1, 1, 1] and list(sel.mp_l2) == [1, 1, 1, 1]
    assert W.select_seeds_by(fr, mupix="none").idx.size == 0
    p = _plugin()
    _feed(p, words, run=1)
    ms = framing.decode_sma_frame(p.frame_blob("seeded"))["meta"]["mupix"]["seeds"]
    assert [(m["covered"], m["l1"], m["l2"]) for m in ms] == [(False, 0, 0)] * 4, "events 19-22"


def test_without_pixels_only_any_selects():
    fr = W.prepare_frame(build([ev() for _ in range(8)]), mupix=W.MuPixCuts())
    for mode, n in (("any", 4), ("both", 0), ("either", 0), ("none", 0)):
        assert W.select_seeds_by(fr, W.Roles(), W.Cuts(), "s1", mupix=mode).idx.size == n, mode
    with pytest.raises(ValueError):
        W.parse_mupix("L1")
    assert W.parse_mupix(None) == "any" and W.parse_mupix(" BOTH ") == "both"


# --- the plugin: fills, summary, trend ------------------------------------------------------------

def test_a_good_frame_fills_the_mupix_histograms():
    p = _plugin()
    words = frame_words(n=40, noise=[(T0 + 5 * SPACING + 3000, 9, 2, 3, 252)])
    _feed(p, words, run=1)
    h = lambda n: p.store.get(f"sma/{n}")  # noqa: E731
    # 40 S1, the last ~one without a complete window does not matter here: all judged
    # but those whose sideband/window leave the pixel data (the first and the last).
    c = h("mupix_s1_match").counts[1:-1]
    n = int(c[0])
    assert 35 <= n <= 40 and list(c[1:4]) == [n, n, n] and list(c[4:]) == [0, 0, 0]
    d1 = h("mupix_dt_L1")
    assert d1.x.lo == -2560 and d1.x.n == 570
    i40 = int((-40 - d1.x.lo) // 8) + 1
    assert d1.counts[i40] >= 38
    assert int(h("mupix_tot_L1").counts[1 + 12]) == 40
    assert int(h("mupix_tot_L2").counts[1 + 9]) == 40
    col = h("mupix_col_chip").counts
    assert int(col[1 + 10, 1 + 1]) == 40 and int(col[1 + 3, 1 + 9]) == 1
    row = h("mupix_row_chip").counts
    assert int(row[1 + 252, 1 + 9]) == 1, "a row off the sensor is kept, not dropped"
    hc = h("mupix_hits_chip")
    assert int(hc.counts[:, 1 + 1].sum()) == 1, "chip 1: one entry per frame"
    assert int(hc.counts[:, 1 + 0].sum()) == 1, "a mapped chip without hits fills 0"
    s = p.summary()
    mp = s["mupix"]
    assert mp["fractions"]["both"]["in"] == 1.0 and mp["fractions"]["both"]["side"] == 0.0
    assert mp["fractions"]["both"]["corr"] == 1.0
    assert mp["unmapped"] == 1 and mp["rows_off_sensor"] == 1
    assert {c["chip"]: c["plane"] for c in mp["chips"]}[9] is None
    assert [f for f in s["flags"] if f["code"] == "mupix_unmapped"], s["flags"]
    assert mp["planes"] == {"L1": [0, 1, 2, 3], "L2": [4, 5, 6, 7]}
    json.dumps(s, allow_nan=False)


def test_the_trend_carries_the_mupix_fractions():
    clk = _Clock()
    p = _plugin(clock=clk)
    _feed(p, frame_words(n=40, l2=()), run=1)
    clk.t += 1.5
    rows = p.trend()["rows"]
    mp = rows[0]["mupix"]
    assert mp["in"] == [1.0, 0.0, 0.0] and mp["corr"][0] == 1.0 and mp["n_s1"] > 30
    assert mp["pix_per_frame"] == 40.0
    clk.t += 1
    assert p.trend()["rows"][-1]["mupix"] is None, "a second without frames"


def _run_seconds(p, clk, words, seconds, per_s=2):
    for _ in range(seconds):
        for _ in range(per_s):
            clk.t += 1.0 / per_s
            _feed(p, words, run=1)


def test_the_sync_flag_needs_its_hold_time_and_clears_with_hysteresis():
    clk = _Clock()
    p = _plugin({"Self check": {"mupix sync min S1": 50, "mupix sync hold s": 10,
                                "mupix sync window s": 3}}, clock=clk)
    good = frame_words(n=60)
    lost = frame_words(n=60, l1=(-40 + 5000,), l2=(-32 + 5000,))   # the pixels 5 us late
    half = frame_words(n=60, l1=(-40,), l2=(), noise=[])            # L1 only: both = 0
    _run_seconds(p, clk, good, 5)
    s = p.summary()
    assert s["mupix"]["sync"]["state"] == "ok" and s["mupix"]["sync"]["value"] == 1.0
    _run_seconds(p, clk, lost, 8)
    sy = p.summary()["mupix"]["sync"]
    assert sy["state"] == "low" and sy["value"] < 0.05
    assert not [f for f in p.summary()["flags"] if f["code"] == "mupix_sync"]
    _run_seconds(p, clk, lost, 8)
    flags = [f for f in p.summary()["flags"] if f["code"] == "mupix_sync"]
    assert len(flags) == 1 and flags[0]["severity"] == "warn"
    assert "lost sync" in flags[0]["text"] and "coarse shift" in flags[0]["text"]
    # Back to good: cleared once the window is above min fraction + margin.
    _run_seconds(p, clk, good, 5)
    assert p.summary()["mupix"]["sync"]["state"] == "ok"
    assert not [f for f in p.summary()["flags"] if f["code"] == "mupix_sync"]
    # Inside the band (min fraction <= f < min + margin) a raised flag stays.
    p2 = _plugin({"Self check": {"mupix sync min S1": 50, "mupix sync hold s": 2,
                                 "mupix sync window s": 2, "mupix sync min fraction": 0.5,
                                 "mupix sync clear margin": 0.3}}, clock=clk)
    _run_seconds(p2, clk, half, 5)
    assert p2.summary()["mupix"]["sync"]["state"] == "flagged"
    # 3 of 5 events with both planes: 0.6, above 0.5 but below 0.8.
    mid = np.concatenate([build([ev() for _ in range(60)]), pixels(sorted(
        [(T0 + k * SPACING - 40, 1, 12) for k in range(60)]
        + [(T0 + k * SPACING - 32, 5, 9) for k in range(60) if k % 5 < 3]))])
    _run_seconds(p2, clk, mid, 5)
    sy = p2.summary()["mupix"]["sync"]
    assert 0.5 < sy["value"] < 0.8 and sy["state"] == "flagged"


def test_no_pixel_words_while_s1_fires_is_flagged_as_absent():
    clk = _Clock()
    p = _plugin({"Self check": {"mupix sync min S1": 50, "mupix sync hold s": 5,
                                "mupix sync window s": 3}}, clock=clk)
    _run_seconds(p, clk, build([ev() for _ in range(60)]), 10)
    flags = [f for f in p.summary()["flags"] if f["code"] == "mupix_sync"]
    assert len(flags) == 1 and "No MuPix pixel words" in flags[0]["text"]


def test_no_s1_is_no_verdict():
    clk = _Clock()
    p = _plugin({"Self check": {"mupix sync min S1": 50, "mupix sync hold s": 2,
                                "mupix sync window s": 2}}, clock=clk)
    no_s1 = frame_words(events=[ev(drop=[1]) for _ in range(60)], l1=(), l2=())
    _run_seconds(p, clk, no_s1, 8)
    s = p.summary()
    assert s["mupix"]["sync"]["state"] == "insufficient"
    assert not [f for f in s["flags"] if f["code"] == "mupix_sync"]


def test_mupix_off_counts_pixels_and_books_no_verdict():
    clk = _Clock()
    p = _plugin({"MuPix": {"max pixel hits per frame": 0}}, clock=clk)
    _run_seconds(p, clk, frame_words(n=40), 3)
    s = p.summary()
    assert s["mupix"]["enabled"] is False and s["mupix"]["sync"]["state"] == "off"
    assert int(p.store.get("sma/word_types").counts[2]) == 3 * 2 * 80
    d = framing.decode_sma_frame(p.frame_blob("seeded"))
    assert d["pixels"] is None and "mupix" not in d["meta"]


def test_real_frames_find_the_prompt_peak():
    """Run 1008: every analysed S1 hit sees L1 and L2 within ~100 ns; accidentals ~6 %."""
    p = _plugin()
    _feed(p, R1008, run=1008)
    mp = p.summary()["mupix"]
    assert mp["fractions"]["both"]["corr"] > 0.8
    assert 0.02 < mp["fractions"]["both"]["side"] < 0.15
    d = p.store.get("sma/mupix_dt_L1")
    c = d.counts[1:-1]
    x = d.x.lo + 8 * (np.arange(c.size) + 0.5)
    assert -80 < x[int(np.argmax(c))] < 0
    for w in R682[2:]:
        _feed(p, w, run=682)
    assert p.summary()["mupix"]["unmapped"] == 0


# --- the smaf pixel block ------------------------------------------------------------------------

def test_the_pixel_block_round_trips_and_old_decoders_see_the_same_hits():
    px = {"t_rel": [0, 8, 4_000_000_000], "time_shift": 3, "chip": [1, 5, 31],
          "col": [0, 255, 7], "row": [249, 250, 0], "tot": [0, 31, 5],
          "flags": [1, 2 | framing.PIX_OFF_SENSOR, framing.PIX_IN_SEED]}
    for words in (False, True):
        kw = {"raw_words": [1 << 62, 5, 7], "word_index": [3, 9, 4294967295]} if words else {}
        extra = dict(raw_words=[0x8100000000000001], word_index=[0]) if words else {}
        b = framing.encode_sma_frame({"view": "raster"}, [5], [1], [20], [0],
                                     pixels=dict(px, **kw), **extra)
        d = framing.decode_sma_frame(b)
        assert d["flags"] & framing.SMAF_PIXELS
        q = d["pixels"]
        assert q["n"] == 3 and q["time_shift"] == 3 and q["words"] is words
        for k in ("t_rel", "chip", "col", "row", "tot", "flags"):
            assert list(q[k]) == list(px[k]), k
        if words:
            assert list(q["raw_words"]) == [1 << 62, 5, 7]
            assert list(q["word_index"]) == [3, 9, 4294967295]
        assert q["offset"] % 8 == 0
        bare = framing.encode_sma_frame({"view": "raster"}, [5], [1], [20], [0], **extra)
        assert framing.smaf_drop_pixels(b) == bare
        # The trigger hits read the same with or without the block.
        e = framing.decode_sma_frame(bare)
        assert list(e["t_rel_ns"]) == list(d["t_rel_ns"]) and e["pixels"] is None
        # A metadata rewrite keeps the block intact and aligned.
        u = framing.decode_sma_frame(framing.smaf_update_meta(b, {"x": "y" * 5}))
        assert list(u["pixels"]["t_rel"]) == px["t_rel"] and u["pixels"]["offset"] % 8 == 0
    with pytest.raises(ValueError):
        framing.encode_sma_frame({}, [], [], [], [], pixels=dict(px, chip=[1]))


def test_the_seeded_view_ships_the_seed_windows_pixels_with_per_seed_counts():
    p = _plugin()
    _feed(p, frame_words(n=40, noise=[(T0 + 38 * SPACING + 1000, 2, 4)]), run=1)
    d = framing.decode_sma_frame(p.frame_blob("seeded"))
    q, mp = d["pixels"], d["meta"]["mupix"]
    seeds = d["meta"]["seeds"]
    assert len(mp["seeds"]) == len(seeds) == 4
    t_seed = [d["meta"]["t0_ns"] + s["t_rel"] for s in seeds]
    t_pix = mp["t0_ns"] + q["t_rel"].astype(np.int64) * 2 ** q["time_shift"]
    for ts, ms in zip(t_seed, mp["seeds"], strict=True):
        a, b = ms["pix"]
        rel = t_pix[a:b] - ts
        assert ((rel >= -200) & (rel <= 3000)).all() and b > a
        assert ms["covered"] and ms["l1"] >= 1 and ms["l2"] >= 1
    assert (q["flags"] & framing.PIX_IN_SEED).all()
    planes = q["flags"] & framing.PIX_PLANE_MASK
    assert set(planes.tolist()) == {1, 2}
    assert q["words"]
    # Every shipped pixel's word index points at its raw word in the bank.
    bank = frame_words(n=40, noise=[(T0 + 38 * SPACING + 1000, 2, 4)])
    assert np.array_equal(bank[q["word_index"]], q["raw_words"])
    assert mp["window_ns"] == [-150, 450] and mp["tot_ns"] == 256


def test_the_raster_ships_every_pixel_hidden_on_request_and_capped():
    p = _plugin()
    _feed(p, frame_words(n=40), run=1)
    full = framing.decode_sma_frame(p.frame_blob("raster", drop=[7]))
    assert full["pixels"]["n"] == 80 and not full["pixels"]["words"]
    hidden = framing.decode_sma_frame(p.frame_blob("raster", drop=[7], pixels=False))
    assert hidden["pixels"] is None and "mupix" not in hidden["meta"]
    assert np.array_equal(hidden["t_rel_ns"], full["t_rel_ns"])
    capped = framing.decode_sma_frame(p.frame_blob("raster", max_hits=50))
    assert capped["pixels"]["n"] == 50 and capped["meta"]["mupix"]["truncated"]
    assert capped["meta"]["mupix"]["n_selected"] == 80
    t = capped["meta"]["mupix"]["t0_ns"] + capped["pixels"]["t_rel"].astype(np.int64)
    assert t[0] == T0 + 15 * SPACING - 40, "the latest 50"


def test_the_command_takes_mupix_and_pixels():
    p = _plugin()
    _feed(p, _mixed_frame(), run=1)
    tag, body = _dispatch(p, "sma::frame", json.dumps({"view": "seeded", "mupix": "none"}))
    d = framing.decode_sma_frame(body)
    assert d["meta"]["select"]["mupix"] == "none"
    assert d["meta"]["select"]["mupix_label"] == "no MuPix hit in time"
    assert all(s["l1"] == 0 and s["l2"] == 0 for s in d["meta"]["mupix"]["seeds"])
    tag, body = _dispatch(p, "sma::frame", json.dumps({"view": "raster", "pixels": False}))
    assert framing.decode_sma_frame(body)["pixels"] is None
    tag, body = _dispatch(p, "sma::frame", json.dumps({"mupix": "L1"}))
    assert tag == framing.TAG_ERROR


def test_no_match_with_the_mupix_selector_names_it():
    clk = _Clock()
    p = _plugin(clock=clk)
    _feed(p, frame_words(n=20), run=1)
    d = framing.decode_sma_frame(p.frame_blob("seeded", mupix="none"))
    assert "no MuPix hit in time" in d["meta"]["search"]["no_match"]["text"]


def test_snapshot_bytes_count_the_pixels():
    p = _plugin()
    _feed(p, frame_words(n=40), run=1)
    snap = p._last
    assert P.snapshot_bytes(snap) >= snap.fr.px.nbytes() > 0


# --- the manual path: mdqm-sma-file ---------------------------------------------------------------

def test_the_cli_summarises_mupix_and_lists_pixel_words(tmp_path, capsys):
    from test_sma_file import write_file

    from mdqm.tools import sma_file as SF

    frames = [frame_words(n=40) for _ in range(4)]
    path = write_file(tmp_path / "run00682_00005.mid", frames)
    rc = SF.main([str(path), "--out", str(tmp_path / "out"), "--no-png"])
    assert rc == SF.EXIT_OK
    text = capsys.readouterr().out
    assert "MuPix: S1 hits with a pixel hit in [-150, 450) ns (sideband [-2400, -1800) ns)" in text
    assert "L1+L2 100.0% (sideband 0.0%, corrected 100.0%)" in text, text
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["mupix"]["fractions"]["both"]["in"] == 1.0
    with np.load(tmp_path / "out" / "hists.npz") as z:
        assert int(z["sma/mupix_s1_match"][1]) > 0 and "sma/mupix_col_chip" in z.files

    # --words lists a pixel word's fields.
    w = frames[1]
    k = int(np.flatnonzero(w.view(np.int64) >= 0)[0])
    rc = SF.main(["--serial", "101", str(path), "--out", str(tmp_path / "o"),
                  "--words", f"{k}:{k}"])
    assert rc == 0
    out = capsys.readouterr().out
    d = W.pixel_decode(w[k:k + 1])
    assert (f"chip {int(d['chip'][0])}, col {int(d['col'][0])}, row {int(d['row'][0])}, "
            f"TS2 {int(d['ts2'][0])}, ToT {int(d['tot'][0])}, {int(d['time'][0])} ns") in out, out
