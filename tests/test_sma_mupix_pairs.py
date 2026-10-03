"""Unseeded MuPix L1-L2 pairs on the SMA DQM: the MakePairs port, its parity, the plugin's fills.

The vectorised pairing (``sma_mupix_pairs.l1l2_pairs``) is compared with a
literal per-hit port of ``PIPSMMuPixCore::MakePairs`` (reco_testbeam
``psm/exp/alg/include/PIPSMMuPixCore.hh``: ``PlaneIndices``, ``WindowRange``,
``MakePairs``) on random frames and on hand-made edge cases. Frames are built
from pixel words (``W.encode_pixel``), so times are whole 8 ns ticks as in the
data.
"""

from __future__ import annotations

import bisect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sma_layouts import old_layout
from test_mupix import frame_words, pixels
from test_sma_plugin import R1008, _Clock, _feed, _plugin
from test_sma_seed_choice import T0

from mdqm.plugins import sma as P
from mdqm.plugins import sma_mupix_pairs as PR
from mdqm.plugins import sma_mupix_xy as X
from mdqm.plugins import sma_words as W

PL = X.placement()
WS = Path(__file__).resolve().parents[3]          # the testbeam-env workspace


def _px(hits):
    """Pixels of ``[(t, chip, tot, col, row), ...]`` in this STREAM order (not sorted)."""
    return W.prepare_pixels(pixels(hits))


# --- the literal port of MakePairs ------------------------------------------------------------

def make_pairs_ref(px, pl, window, max_l1=None, t_min=None):
    """``(n_l1, [(i1, i2, t(L2) - t(L1), nPartners), ...])`` the way the C++ does it.

    The hit collection is the decoder's: the pixel words in stream order, rows >=
    250 dropped (PITMidasMusip). PlaneIndices: the hits of a plane, stable-sorted
    by time. MakePairs: per L1 hit, WindowRange (lower_bound of t - w, upper_bound
    of t + w), then the first hit of the window, replaced only on a strictly
    smaller |dt|. Unpaired L1 hits (which the C++ skips) are kept as (i1, -1, 0, 0)
    to compare with the DQM's arrays. ``max_l1`` / ``t_min``: the DQM's sample of
    the L1 list, applied to the C++ list.
    """
    stream = sorted(range(px.n), key=lambda i: int(px.word_index[i]))
    hits = [i for i in stream if int(px.row[i]) < W.PIXEL_ROWS]

    def plane_indices(plane):
        idx = [i for i in hits if int(pl.plane[int(px.chip[i])]) == plane]
        return sorted(idx, key=lambda i: int(px.t[i]))         # Python's sort is stable

    l1, l2 = plane_indices(W.PLANE_L1), plane_indices(W.PLANE_L2)
    if t_min is not None:
        l1 = [a for a in l1 if int(px.t[a]) >= t_min]
    n_l1 = len(l1)
    if max_l1 is not None and n_l1 > max_l1:
        l1 = [l1[j] for j in W.even_sample(n_l1, max_l1)]
    t2 = [int(px.t[i]) for i in l2]
    out = []
    for a in l1:
        t1 = int(px.t[a])
        first = bisect.bisect_left(t2, t1 - window)
        last = bisect.bisect_right(t2, t1 + window, lo=first)
        n = last - first
        if n == 0:
            out.append((a, -1, 0, 0))
            continue
        best, best_dt = first, t2[first] - t1
        for j in range(first + 1, last):
            dt = t2[j] - t1
            if abs(dt) < abs(best_dt):
                best, best_dt = j, dt
        out.append((a, l2[best], best_dt, n))        # Pair::dt = t(L2) - t(L1)
    return n_l1, out


def _check_parity(px, window=40, max_l1=10**9, t_min=None, shift=(0.0, 0.0)):
    got = PR.l1l2_pairs(px, window, max_l1, PL, shift, t_min)
    n_l1, ref = make_pairs_ref(px, PL, window, max_l1, t_min)
    assert got.n_l1 == n_l1
    want = np.array(ref, dtype=np.int64).reshape(-1, 4)
    assert got.n == len(ref)
    np.testing.assert_array_equal(got.i1, want[:, 0])
    np.testing.assert_array_equal(got.i2, want[:, 1])
    np.testing.assert_array_equal(got.dt, want[:, 2])
    np.testing.assert_array_equal(got.n_partners, want[:, 3])
    # Positions and slopes from the chosen pixels, as PIPSMMuPixMonitor fills them.
    if got.n:
        x1, y1 = PL.xy(px.chip[got.i1], px.col[got.i1], px.row[got.i1])
        np.testing.assert_allclose(got.x1, x1 + shift[0], atol=1e-12)
        np.testing.assert_allclose(got.y1, y1 + shift[1], atol=1e-12)
        p = got.paired
        j = got.i2[p]
        x2, y2 = PL.xy(px.chip[j], px.col[j], px.row[j])
        np.testing.assert_allclose(got.x2[p], x2 + shift[0], atol=1e-12)
        np.testing.assert_allclose(got.xp[p], 1000 * (x2 - x1[p]) / 30.0, atol=1e-9)
        np.testing.assert_allclose(got.yp[p], 1000 * (y2 - y1[p]) / 30.0, atol=1e-9)
        assert np.isnan(got.x2[~p]).all() and np.isnan(got.xp[~p]).all()
        assert (got.tot1 == px.tot[got.i1]).all()
        assert (got.tot2[p] == px.tot[j]).all() and (got.tot2[~p] == -1).all()
    return got


def _random_hits(rng, n, span_ns, chips=12, bad_rows=True):
    """Random pixels on chips 0..chips-1 (8+ have no plane), rows up to 255 (>= 250
    are off the sensor), times in whole ticks, in a random stream order."""
    t = T0 + 8 * rng.integers(0, max(1, span_ns // 8), n)
    hits = list(zip(t.tolist(), rng.integers(0, chips, n).tolist(),
                    rng.integers(0, 32, n).tolist(), rng.integers(0, 256, n).tolist(),
                    rng.integers(0, 256 if bad_rows else 250, n).tolist(), strict=True))
    rng.shuffle(hits)
    return hits


@pytest.mark.parametrize("seed", range(12))
def test_random_frames_match_the_literal_port(seed):
    rng = np.random.default_rng(seed)
    # From sparse (~0.1 partners) to dense (~10 partners in +-40 ns), with many
    # equal time stamps, out-of-order streams, rows >= 250 and plane-less chips.
    n = int(rng.integers(50, 3000))
    span = int(rng.choice([2_000, 20_000, 200_000, 2_000_000]))
    px = _px(_random_hits(rng, n, span))
    for window in (0, 8, 40, 100):
        _check_parity(px, window)
    _check_parity(px, 40, max_l1=int(rng.integers(1, 200)))
    _check_parity(px, 40, t_min=int(px.first) + 8 * int(rng.integers(0, max(1, span // 16))))
    _check_parity(px, 40, shift=(-2.5, 1.25))


def test_a_dense_frame_matches_the_literal_port():
    rng = np.random.default_rng(99)
    px = _px(_random_hits(rng, 6000, 4_000, chips=8, bad_rows=False))
    got = _check_parity(px, 40)
    assert got.n_partners.mean() > 20 and got.paired.all()
    _check_parity(px, 40, max_l1=500)


def _one(hits, window=40, **kw):
    px = _px(hits)
    return px, _check_parity(px, window, **kw)


def test_an_exact_tie_goes_to_the_earlier_pixel():
    # L1 at T0; L2 at T0 - 16 and T0 + 16: |dt| equal, the earlier wins.
    px, got = _one([(T0 + 16, 5, 3, 1, 1), (T0, 1, 7, 10, 20), (T0 - 16, 4, 3, 2, 2)])
    assert got.n == 1 and got.dt[0] == -16 and got.n_partners[0] == 2
    assert px.t[got.i2[0]] == T0 - 16


def test_equal_time_stamps_go_to_the_first_in_the_stream():
    # Two L2 pixels at the same time: the first word of the stream wins, whatever its
    # column (the C++ PlaneIndices is a stable sort of the readout order).
    px, got = _one([(T0 + 8, 5, 3, 200, 7), (T0, 1, 7, 10, 20), (T0 + 8, 5, 3, 3, 9)])
    assert got.n_partners[0] == 2 and px.col[got.i2[0]] == 200
    px, got = _one([(T0 + 8, 5, 3, 3, 9), (T0, 1, 7, 10, 20), (T0 + 8, 5, 3, 200, 7)])
    assert px.col[got.i2[0]] == 3
    # Equal times before the L1 pixel: the first of them too, and it beats a later one
    # equally far on the other side.
    px, got = _one([(T0 - 24, 4, 1, 50, 1), (T0 - 24, 4, 1, 60, 1), (T0 + 24, 4, 1, 70, 1),
                    (T0, 0, 7, 10, 20)])
    assert px.col[got.i2[0]] == 50 and got.dt[0] == -24 and got.n_partners[0] == 3


def test_the_window_edges_are_inclusive():
    for d, paired in ((40, True), (-40, True), (48, False), (-48, False), (0, True)):
        px, got = _one([(T0, 1, 7, 10, 20), (T0 + d, 5, 3, 1, 1)])
        assert bool(got.paired[0]) is paired, d
        if paired:
            assert got.dt[0] == d and got.n_partners[0] == 1
    # A nearer pixel inside beats one on the edge.
    px, got = _one([(T0, 1, 7, 10, 20), (T0 + 40, 5, 3, 1, 1), (T0 - 8, 5, 3, 2, 2)])
    assert got.dt[0] == -8 and got.n_partners[0] == 2


def test_an_empty_plane_pairs_nothing():
    px, got = _one([(T0, 1, 7, 10, 20), (T0 + 8, 2, 7, 10, 20)])        # no L2
    assert got.n == 2 and not got.paired.any() and (got.n_partners == 0).all()
    assert np.isnan(got.xp).all() and (got.tot2 == -1).all()
    assert got.candidate_dt().size == 0
    px, got = _one([(T0, 5, 7, 10, 20), (T0 + 8, 6, 7, 10, 20)])        # no L1
    assert got.n == 0 and got.n_l1 == 0
    assert PR.l1l2_pairs(None).n == 0


def test_one_l2_pixel_serves_several_l1_pixels():
    hits = [(T0 + 8 * k, 1, 7, 10 + k, 20) for k in range(5)] + [(T0 + 16, 5, 3, 1, 1)]
    px, got = _one(hits)
    assert got.paired.all() and len(set(got.i2.tolist())) == 1
    assert list(got.dt) == [16, 8, 0, -8, -16]


def test_rows_off_the_sensor_and_plane_less_chips_are_no_candidates():
    # An L2 pixel at row 250 and one on chip 9 (no plane) are nearer than the real one.
    px, got = _one([(T0, 1, 7, 10, 20), (T0, 5, 3, 1, 250), (T0, 9, 3, 1, 1),
                    (T0 + 32, 5, 3, 1, 1), (T0 + 8, 1, 7, 10, 255)])
    assert got.n_l1 == 1 and got.n_partners[0] == 1 and got.dt[0] == 32
    # A chip in the plane lists without a place (a fifth entry) is left out too.
    pl = X.placement((0, 1, 2, 3, 9), (4, 5, 6, 7))
    res = PR.l1l2_pairs(px, 40, 100, pl)
    assert res.n_l1 == 1 and res.n_partners[0] == 1


def test_the_sample_is_even_over_the_candidates():
    rng = np.random.default_rng(5)
    px = _px(_random_hits(rng, 4000, 400_000, chips=8, bad_rows=False))
    cand = PR.candidates(px, PL, W.PLANE_L1)
    got = PR.l1l2_pairs(px, 40, 100, PL)
    assert got.n == 100 and got.n_l1 == cand.size
    np.testing.assert_array_equal(got.i1, cand[W.even_sample(cand.size, 100)])
    assert (np.diff(got.t1) >= 0).all()


def test_candidate_dt_is_every_l2_pixel_within_reach():
    rng = np.random.default_rng(7)
    px = _px(_random_hits(rng, 3000, 300_000, chips=8, bad_rows=False))
    got = PR.l1l2_pairs(px, 40, 300, PL)
    dt = got.candidate_dt(100)
    want = (got.t2[None, :] - got.t1[:, None]).ravel()
    want = want[np.abs(want) <= 100]
    np.testing.assert_array_equal(np.sort(dt), np.sort(want))
    assert (dt % 8 == 0).all()
    # The plugin's path: the same entries from the pairing's own search.
    fused = PR.l1l2_pairs(px, 40, 300, PL, dt_half=100)
    np.testing.assert_array_equal(fused.cand_dt, dt)
    assert PR.l1l2_pairs(px, 40, 300, PL).cand_dt is None
    # The cap takes an even subset of the sampled L1 pixels.
    assert 0 < got.candidate_dt(100, max_entries=50).size <= 50


def test_a_burst_caps_the_dt_entries_at_a_few_per_l1_pixel():
    # 4000 pixels in 2 us: hundreds of L2 pixels within +-100 ns of every L1 pixel.
    rng = np.random.default_rng(11)
    px = _px(sorted(_random_hits(rng, 4000, 2_000, chips=8, bad_rows=False)))
    got = PR.l1l2_pairs(px, 64, 200, PL, dt_half=100)
    full = (got.t2[None, :] - got.t1[:, None]).ravel()
    assert (np.abs(full) <= 100).sum() > 40 * PR.DT_PER_L1 * got.n
    assert 0 < got.cand_dt.size <= PR.DT_PER_L1 * got.n
    assert (np.abs(got.cand_dt) <= 100).all()
    # A cap below one row's entries keeps that row's first entries, never more.
    assert PR.l1l2_pairs(px, 64, 200, PL, dt_half=100, max_dt=3).cand_dt.size == 3
    assert PR.l1l2_pairs(px, 64, 200, PL, dt_half=100, max_dt=0).cand_dt.size == 0


def test_a_dense_frame_is_fast_enough():
    """2000 L1 pixels on a 20000-pixel frame (~0.25 ms hot; asserted loosely here)."""
    import time

    rng = np.random.default_rng(1)
    px = _px(sorted(_random_hits(rng, 20000, 30_000_000, chips=8, bad_rows=False)))
    PR.l1l2_pairs(px, 40, 2000, PL)
    best = 1.0
    for _ in range(20):
        t0 = time.perf_counter()
        PR.l1l2_pairs(px, 40, 2000, PL, dt_half=100)
        best = min(best, time.perf_counter() - t0)
    assert best < 5e-3, best


# --- the plugin -----------------------------------------------------------------------------------

PAIR_MAPS = tuple(f"mupix_pair_{k}{c}" for k in ("xy", "xxp", "yyp")
                  for c in ("", "_light", "_heavy"))
PLANE_MAPS = ("mupix_pair_hits_xy_L1", "mupix_pair_hits_xy_L2")
PAIR_HISTS = (*PAIR_MAPS, *PLANE_MAPS, "mupix_pair_dt", "mupix_pair_partners")
SUMMARY_KEYS = {"enabled", "off_reason", "resets", "mupix_only_frames", "n_l1", "n_pairs",
                "paired_frac",
                "mean_partners", "light_frac", "heavy_frac", "window_ns", "max_l1", "cuts",
                "stage", "hits", "max_hits"}


def _pair_names(p):
    return sorted(n[4:] for n in p.store.names() if n.startswith("sma/mupix_pair_"))


def _bin(h, x, y):
    ix = int(np.floor((x - h.x.lo) / ((h.x.hi - h.x.lo) / h.x.n))) + 1
    iy = int(np.floor((y - h.y.lo) / ((h.y.hi - h.y.lo) / h.y.n))) + 1
    return int(h.counts[iy, ix])


def test_the_plugin_books_the_pair_histograms_on_the_xy_axes():
    p = _plugin()
    assert _pair_names(p) == sorted(PAIR_HISTS)
    h = lambda n: p.store.get(f"sma/{n}")  # noqa: E731
    for k in ("xy", "xxp", "yyp"):
        a, b = h(f"mupix_pair_{k}"), h(f"mupix_track_{k}")
        assert (a.x, a.y) == (b.x, b.y), k
    for pl in ("L1", "L2"):
        a, b = h(f"mupix_pair_hits_xy_{pl}"), h(f"mupix_hits_xy_{pl}")
        assert (a.x, a.y) == (b.x, b.y), pl
        assert a.x.n == a.y.n == 130 and abs((a.x.hi - a.x.lo) / a.x.n - 0.64) < 1e-9
        assert f"{pl} alone" in a.title and f"at most {PR.MAX_HITS} a frame" in a.title
    for n in PAIR_HISTS:
        assert h(n).counts.dtype == np.uint32, n
    dt = h("mupix_pair_dt").x
    assert (dt.n, dt.lo, dt.hi) == (25, -100.0, 100.0)
    centres = dt.lo + 8 * (np.arange(dt.n) + 0.5)
    assert (centres % 8 == 0).all() and 40 in centres and -40 in centres
    pa = h("mupix_pair_partners").x
    assert (pa.n, pa.lo, pa.hi) == (11, -0.5, 10.5)
    # Eleven pair plots ~0.46 MB, the two single-plane maps 2 x 70 kB.
    assert sum(h(n).counts.nbytes for n in PAIR_HISTS) < 0.65e6


def test_a_frame_with_known_pairs_fills_every_pair_plot():
    # frame_words: per event an L1 pixel on chip 1 (10, 20) at S1 - 40, ToT 12, and an
    # L2 pixel on chip 5 (10, 20) at S1 - 32, ToT 9: one pair each, dt L2 - L1 = +8.
    p = _plugin({"MuPix": {"XY": {"tot light max": 5, "tot heavy min": 9}}})
    _feed(p, frame_words(n=40), run=1)
    h = lambda n: p.store.get(f"sma/{n}")  # noqa: E731
    n = h("mupix_pair_xy").entries
    assert n == 40
    x, y = PL.xy(1, 10, 20)
    x, y = float(x), float(y)
    assert _bin(h("mupix_pair_xy"), x, y) == n
    assert _bin(h("mupix_pair_xxp"), x, 0.0) == n and _bin(h("mupix_pair_yyp"), y, 0.0) == n
    assert h("mupix_pair_xy_heavy").entries == n and h("mupix_pair_xy_light").entries == 0
    dt = h("mupix_pair_dt")
    assert dt.entries == n and int(dt.counts[1 + (8 + 100) // 8]) == n
    pa = h("mupix_pair_partners")
    assert pa.entries == n and int(pa.counts[1 + 1]) == n
    s = p.summary()["pairs"]
    assert (s["n_l1"], s["n_pairs"], s["paired_frac"], s["mean_partners"]) == (n, n, 1.0, 1.0)
    assert (s["heavy_frac"], s["light_frac"]) == (1.0, 0.0)


def test_pairs_need_no_s1():
    # The same pixels with no S1 seed in the frame (S1 on ch 1 dropped from the roles:
    # s1 on an empty channel) still pair.
    p = _plugin({"Channel roles": {"s1": 13, "counters": [13, 2, 3, 4, 5]}})
    _feed(p, frame_words(n=40), run=1)
    assert p.summary()["mupix"]["n_s1"] == 0
    assert p.store.get("sma/mupix_pair_xy").entries == 40


@pytest.mark.parametrize("light, heavy, cls", [
    (12, 13, "light"),        # ToT 12 and 9 both <= 12
    (11, 12, None),           # 12 > 11: not light; 9 < 12: not heavy
    (8, 9, "heavy"),          # both >= 9
    (8, 10, None),
])
def test_the_tot_classes_at_the_edges(light, heavy, cls):
    p = _plugin({"MuPix": {"XY": {"tot light max": light, "tot heavy min": heavy}}})
    _feed(p, frame_words(n=40), run=1)
    nl = p.store.get("sma/mupix_pair_xy_light").entries
    nh = p.store.get("sma/mupix_pair_xy_heavy").entries
    assert (nl, nh) == ((40, 0) if cls == "light" else (0, 40) if cls == "heavy" else (0, 0))
    s = p.summary()["pairs"]
    assert s["light_frac"] == nl / 40 and s["heavy_frac"] == nh / 40
    assert s["cuts"] == {"tot_light_max": light, "tot_heavy_min": heavy, "tot_ns": 256}


def test_the_window_key_moves_the_edge():
    # dt L2 - L1 = +8: inside +-8, outside +-0.
    p = _plugin({"MuPix": {"Pairs": {"window ns": 0}}})
    _feed(p, frame_words(n=40), run=1)
    assert p.store.get("sma/mupix_pair_xy").entries == 0
    assert p.store.get("sma/mupix_pair_dt").entries == 40          # dt keeps its +-100
    assert p.summary()["pairs"]["paired_frac"] == 0.0
    p = _plugin({"MuPix": {"Pairs": {"window ns": 8}}})
    _feed(p, frame_words(n=40), run=1)
    assert p.store.get("sma/mupix_pair_xy").entries == 40


@pytest.mark.parametrize("settings, reason", [
    ({"MuPix": {"Pairs": {"enable": False}}}, "MuPix/Pairs/enable = n"),
    ({"MuPix": {"max pixel hits per frame": 0}}, "MuPix analysis is off"),
    ({"MuPix": {"L1 chips": [0, 2, 3]}}, "3 entries"),
])
def test_pairs_off_books_and_fills_nothing(settings, reason):
    p = _plugin(settings)
    assert _pair_names(p) == [] and p.h["pairs"] is None
    _feed(p, frame_words(n=40), run=1)
    s = p.summary()["pairs"]
    assert s["enabled"] is False and reason in s["off_reason"]
    assert s["n_l1"] == 0 and s["paired_frac"] is None and s["mean_partners"] is None
    if "entries" in reason:
        assert any(e.startswith("MuPix/Pairs: ") and "pairs are off" in e for e in p.cfg.errors)


def test_xy_off_with_pairs_on_books_only_the_pairs_and_reads_the_stage():
    p = _plugin({"MuPix": {"XY": {"enable": False}}})
    assert p.h["xy"] is None and _pair_names(p) == sorted(PAIR_HISTS)
    p.poll_odb(lambda path: [3.0, -1.0])
    _feed(p, frame_words(n=40), run=1)
    x, y = PL.xy(1, 10, 20)
    assert _bin(p.store.get("sma/mupix_pair_xy"), float(x) - 3.0, float(y) - 1.0) == 40
    st = p.summary()["pairs"]["stage"]
    assert st == {"x_mm": 3.0, "y_mm": -1.0, "source": "odb", "applied": True,
                  "shift_mm": [-3.0, -1.0], "note": None}
    assert st == p.summary()["xy"]["stage"]


@pytest.mark.parametrize("tree, err", [
    ({"enable": "maybe"}, "enable"),
    ({"window ns": -1}, "window ns"),
    ({"window ns": 1001}, "window ns"),
    ({"window ns": "wide"}, "window ns"),
    ({"max L1 per frame": 0}, "0 is not 'all'"),
    ({"max L1 per frame": -5}, "using 1"),
    ({"max L1 per frame": 50000}, "using 20000"),
    ({"max L1 per frame": "lots"}, "max L1 per frame"),
    ({"max hits per frame": 0}, "max hits per frame=0"),
    ({"max hits per frame": -5}, "max hits per frame=-5"),
    ({"max hits per frame": 50000}, "max hits per frame=50000"),
    ({"max hits per frame": "lots"}, "max hits per frame"),
])
def test_bad_pair_settings_fall_back_and_are_reported(tree, err):
    cfg = P.parse_settings({"MuPix": {"Pairs": tree}})
    assert any(e.startswith("MuPix/Pairs/") and err in e for e in cfg.errors), cfg.errors
    assert 0 <= cfg.pairs.window_ns <= PR.WINDOW_LIMIT_NS
    assert 1 <= cfg.pairs.max_l1 <= PR.MAX_L1_LIMIT
    assert 1 <= cfg.pairs.max_hits <= PR.MAX_HITS_LIMIT
    if "window" in err:
        assert cfg.pairs.window_ns == PR.WINDOW_NS


def test_good_pair_settings_parse():
    cfg = P.parse_settings({"MuPix": {"Pairs": {"enable": "n", "window ns": 64,
                                                "max L1 per frame": 20000,
                                                "max hits per frame": 20000},
                                      "XY": {"tot light max": 3, "tot heavy min": 10,
                                             "apply stage shift": False}}})
    assert not cfg.errors
    pr = cfg.pairs
    assert (pr.enable, pr.window_ns, pr.max_l1, pr.max_hits, pr.tot_light_max, pr.tot_heavy_min,
            pr.apply_stage) == (False, 64, 20000, 20000, 3, 10, False)
    d = P.parse_settings({}).pairs
    assert (d.enable, d.window_ns, d.max_l1, d.max_hits, d.active) == (
        True, PR.WINDOW_NS, PR.MAX_L1, PR.MAX_HITS, True)
    assert P.parse_settings({"MuPix": {"Pairs": {"max hits per frame": 1}}}).pairs.max_hits == 1


def test_no_pair_key_is_in_the_shape_fingerprint():
    base = P.shape_fingerprint({})
    for k, v in (("enable", False), ("window ns", 64), ("max L1 per frame", 9),
                 ("max hits per frame", 9)):
        assert P.shape_fingerprint({"MuPix": {"Pairs": {k: v}}}) == base, k


def _non_pair_counts(p):
    return {n: (p.store.get(n).entries, p.store.get(n).counts.copy()) for n in p.store.names()
            if not n.startswith("sma/mupix_pair_")}


def _pair_entries(p):
    return {n[4:]: p.store.get(n).entries for n in p.store.names()
            if n.startswith("sma/mupix_pair_")}


@pytest.mark.parametrize("edit, reset", [
    ({"Pairs": {"window ns": 32}}, "all"),
    ({"Pairs": {"max L1 per frame": 100}}, "none"),        # a CPU knob, as XY's
    ({"Pairs": {"max hits per frame": 100}}, "none"),      # likewise
    ({"Pairs": {"enable": True, "window ns": 64}}, "none"),  # the defaults: no change
    ({"XY": {"apply stage shift": False}}, "all"),
    ({"XY": {"tot light max": 5, "tot heavy min": 9}}, "classes"),
    ({"XY": {"cluster box px": 4, "max S1 per frame": 100}}, "none"),
])
def test_a_pair_edit_resets_only_the_pair_maps(edit, reset):
    base = {"MuPix": {"XY": {"tot light max": 8, "tot heavy min": 9}}}
    p = _plugin(base)
    _feed(p, frame_words(n=40), run=1)
    before, pr0 = _non_pair_counts(p), _pair_entries(p)
    xy_only = {n for n in before if n.startswith(("sma/mupix_track_", "sma/mupix_hits_xy_"))}
    s0 = p.summary()
    epoch, rebuilds = p.epoch, p.rebuilds
    assert pr0["mupix_pair_xy"] == 40 and pr0["mupix_pair_xy_heavy"] == 40
    assert pr0["mupix_pair_hits_xy_L1"] == 40 == pr0["mupix_pair_hits_xy_L2"]
    new = {"XY": {**base["MuPix"]["XY"], **edit.get("XY", {})}, "Pairs": edit.get("Pairs", {})}
    settings = old_layout({"MuPix": new})
    p.apply_settings(settings, rebuild=p.shape_fingerprint(settings) != p.shape_fingerprint(
        old_layout(base)))
    assert (p.epoch, p.rebuilds) == (epoch, rebuilds), "no rebuild, no new epoch"
    after = _non_pair_counts(p)
    assert after.keys() == before.keys()
    for n, (e, c) in before.items():
        if n in xy_only:                      # x/y has its own reset rule (test_sma_mupix_xy)
            continue
        assert after[n][0] == e and np.array_equal(after[n][1], c), n
    pr1 = _pair_entries(p)
    assert set(pr1) == set(pr0)
    for k, e in pr1.items():
        zeroed = reset == "all" or (reset == "classes" and k.endswith(("_light", "_heavy")))
        assert e == (0 if zeroed else pr0[k]), k
    s1 = p.summary()["pairs"]
    assert (s1["n_l1"] == 0) == (reset == "all")
    # The single-plane maps and their counters reset with the full set only.
    assert s1["hits"] == ({"n_l1": 0, "n_l2": 0} if reset == "all" else s0["pairs"]["hits"])
    assert s1["resets"] == (0 if reset == "none" else 1)
    if reset == "classes":
        assert "ToT >= 9" in p.store.get("sma/mupix_pair_xy_heavy").title
        assert s1["heavy_frac"] is None and s1["n_pairs"] == s0["pairs"]["n_pairs"]


def test_switching_pairs_off_and_on_books_only_the_pair_maps():
    p = _plugin()
    _feed(p, frame_words(n=40), run=1)
    before = _non_pair_counts(p)
    p.apply_settings(old_layout({"MuPix": {"Pairs": {"enable": False}}}), rebuild=False)
    assert _pair_names(p) == [] and p.h["pairs"] is None
    assert _non_pair_counts(p).keys() == before.keys()
    _feed(p, frame_words(n=40), run=1)
    p.apply_settings(old_layout({}), rebuild=False)
    assert _pair_names(p) == sorted(PAIR_HISTS)
    assert all(e == 0 for e in _pair_entries(p).values())
    assert p.epoch == 0 and p.rebuilds == 0


def test_the_pair_histograms_widen_before_a_bin_could_wrap():
    p = _plugin()
    h = p.store.get("sma/mupix_pair_dt")
    h.counts[5] = np.iinfo(np.uint32).max - 1
    h.entries = int(np.iinfo(np.uint32).max) - 1
    _feed(p, frame_words(n=40), run=1)
    assert h.counts.dtype == np.uint64 and int(h.counts[5]) == 2**32 - 2
    m = p.store.get("sma/mupix_pair_xy")
    m.entries = int(np.iinfo(np.uint32).max)
    P._widen(m, 1)
    assert m.counts.dtype == np.uint64
    hm = p.store.get("sma/mupix_pair_hits_xy_L1")
    assert hm.counts.dtype == np.uint32
    hm.entries = int(np.iinfo(np.uint32).max) - 1
    _feed(p, frame_words(n=40), run=1)
    assert hm.counts.dtype == np.uint64 and hm.entries == int(np.iinfo(np.uint32).max) + 39


def test_the_cap_and_the_skipped_pixels():
    p = _plugin({"MuPix": {"Pairs": {"max L1 per frame": 10}}})
    _feed(p, frame_words(n=40), run=1)
    assert p.summary()["pairs"]["n_l1"] == 10 == p.store.get("sma/mupix_pair_partners").entries
    # The pixel cap skips the earliest words: L1 pixels within reach of the first
    # examined pixel are left out (their partners may be gone).
    p = _plugin({"MuPix": {"max pixel hits per frame": 41}})
    _feed(p, frame_words(n=40), run=1)
    s = p.summary()["pairs"]
    assert s["paired_frac"] == 1.0 and s["n_l1"] == 20


def test_the_summary_block_keys():
    p = _plugin(clock=_Clock())
    _feed(p, R1008, run=1008)
    s = p.summary()["pairs"]
    assert set(s) == SUMMARY_KEYS
    assert set(s["cuts"]) == {"tot_light_max", "tot_heavy_min", "tot_ns"}
    assert set(s["hits"]) == {"n_l1", "n_l2"} and s["max_hits"] == PR.MAX_HITS
    assert s["hits"]["n_l1"] == p.store.get("sma/mupix_pair_hits_xy_L1").entries > 0
    assert s["hits"]["n_l2"] == p.store.get("sma/mupix_pair_hits_xy_L2").entries > 0
    assert set(s["stage"]) == {"x_mm", "y_mm", "source", "applied", "shift_mm", "note"}
    assert (s["enabled"], s["off_reason"], s["resets"]) == (True, None, 0)
    assert (s["window_ns"], s["max_l1"]) == (PR.WINDOW_NS, PR.MAX_L1)
    json.dumps(s, allow_nan=False)
    # Run 1008: most L1 pixels have an L2 pixel in the window; one entry per L1 pixel.
    assert s["n_l1"] == p.store.get("sma/mupix_pair_partners").entries <= PR.MAX_L1
    assert s["n_pairs"] == p.store.get("sma/mupix_pair_xy").entries
    assert 0.5 < s["paired_frac"] <= 1.0 and s["mean_partners"] >= 1.0


def test_real_frames_put_the_pairs_where_the_tracks_are():
    """Run 1008: the pair beam spot sits on the S1-seeded one (same frame, same shift)."""
    p = _plugin()
    _feed(p, R1008, run=1008)
    a = p.store.get("sma/mupix_pair_xy").counts[1:-1, 1:-1].astype(float)
    b = p.store.get("sma/mupix_track_xy").counts[1:-1, 1:-1].astype(float)
    xc = X.POS_LO_MM + 0.64 * (np.arange(130) + 0.5)
    mean = lambda c, axis: (c.sum(axis=axis) * xc).sum() / c.sum()  # noqa: E731
    assert abs(mean(a, 0) - mean(b, 0)) < 2.0 and abs(mean(a, 1) - mean(b, 1)) < 2.0
    assert a.sum() > 300                     # one frame, at most MAX_L1 L1 pixels sampled


def test_mupix_only_frames_fill_the_pairs_and_stay_empty_frames():
    # A frame of pixel words only (no SMA trigger words): class "empty", but its
    # MuPix part is analysed -- occupancy, ToT and the pairs; nothing S1-seeded.
    hits = []
    for k in range(30):
        t = T0 + 10_000 * k
        hits += [(t, 1, 12, 10, 20), (t + 8, 5, 9, 10, 20)]
    p = _plugin()
    _feed(p, pixels(hits), run=1)
    s = p.summary()
    assert s["frames"]["window"]["empty"] == 1 and s["frames"]["window"]["good"] == 0
    pr = s["pairs"]
    assert pr["mupix_only_frames"] == 1 and pr["n_l1"] == 30 and pr["paired_frac"] == 1.0
    assert p.store.get("sma/mupix_pair_xy").entries == 30
    assert p.store.get("sma/mupix_pair_dt").entries == 30
    assert p.store.get("sma/mupix_tot_L1").entries == 30
    assert s["mupix"]["frames"] == 1 and s["mupix"]["n_s1"] == 0
    assert p.store.get("sma/mupix_track_state").entries == 0
    assert s["xy"]["n_s1"] == 0
    # A frame with neither trigger nor pixel words analyses nothing.
    p = _plugin()
    _feed(p, np.array([W.FILLER] * 4, dtype=np.uint64), run=1)
    assert p.summary()["pairs"]["mupix_only_frames"] == 0


def _plane_frame():
    """Pixel words only: 30 L1 pixels on chip 1 (10, 20), 20 L2 pixels on chip 5
    (30, 40) far from them in time (no pairs), 5 rows >= 250 on chip 1 and 5 hits on
    a chip without a plane (9): none of the last ten is a candidate."""
    hits = [(T0 + 10_000 * k, 1, 12, 10, 20) for k in range(30)]
    hits += [(T0 + 10_000 * k + 5_000, 5, 9, 30, 40) for k in range(20)]
    hits += [(T0 + 10_000 * k + 2_000, 1, 7, 10, W.PIXEL_ROWS + k) for k in range(5)]
    hits += [(T0 + 10_000 * k + 3_000, 9, 7, 10, 20) for k in range(5)]
    return pixels(sorted(hits))


def test_the_single_plane_maps_take_every_candidate_of_each_plane():
    # A MuPix-only frame (no trigger words), and no pairs at all: the plane maps fill anyway.
    p = _plugin()
    p.poll_odb(lambda path: [3.0, -1.0])                # stage shift (-3, -1) mm
    _feed(p, _plane_frame(), run=1)
    h1, h2 = (p.store.get(f"sma/{n}") for n in PLANE_MAPS)
    assert (h1.entries, h2.entries) == (30, 20)
    x1, y1 = (float(v) for v in PL.xy(1, 10, 20))
    x2, y2 = (float(v) for v in PL.xy(5, 30, 40))
    assert _bin(h1, x1 - 3.0, y1 - 1.0) == 30 and _bin(h2, x2 - 3.0, y2 - 1.0) == 20
    s = p.summary()["pairs"]
    assert s["hits"] == {"n_l1": 30, "n_l2": 20} and s["mupix_only_frames"] == 1
    assert s["n_l1"] == 30 and s["n_pairs"] == 0
    assert p.store.get("sma/mupix_pair_xy").entries == 0
    # The in-time hit maps of the phase-space tab are S1-gated: nothing here.
    assert p.store.get("sma/mupix_hits_xy_L1").entries == 0


def test_the_single_plane_cap_is_an_even_sample_per_plane():
    p = _plugin({"MuPix": {"Pairs": {"max hits per frame": 10}}})
    _feed(p, _plane_frame(), run=1)
    assert [p.store.get(f"sma/{n}").entries for n in PLANE_MAPS] == [10, 10]
    assert p.summary()["pairs"]["hits"] == {"n_l1": 10, "n_l2": 10}
    assert p.summary()["pairs"]["max_hits"] == 10
    # plane_sample: W.even_sample over each plane's candidates, in time order.
    hits = [(T0 + 1000 * k, 1, 5, k, 20) for k in range(100)]
    hits += [(T0 + 1000 * k + 500, 6, 5, k, 30) for k in range(3)]
    px = W.prepare_pixels(pixels(hits))
    idx, n = PR.plane_sample(PR.plane_candidates(px, PL), 7)
    assert list(n) == [7, 3]
    assert list(px.col[idx[:7]]) == list(W.even_sample(100, 7))
    assert list(px.chip[idx[7:]]) == [6, 6, 6]


def test_the_bin_tables_bin_every_pixel_as_its_position_does():
    """The (chip, col) / (chip, row) tables give the bin of PL.xy + shift, bit for bit."""
    rng = np.random.default_rng(5)
    hits = [(T0 + 100 * k, int(rng.integers(0, 8)), 5, int(rng.integers(0, 256)),
             int(rng.integers(0, 250))) for k in range(3000)]
    px = W.prepare_pixels(pixels(sorted(hits)))
    p = _plugin()
    ax = p.store.get("sma/mupix_pair_hits_xy_L1").x
    for shift in ((0.0, 0.0), (-3.25, 1.7), (-40.0, 40.0)):
        bx, by = p._plane_bins(PL, shift, ax)
        x, y = PL.xy(px.chip, px.col, px.row)
        assert np.array_equal(bx[px.chip, px.col], P._bin_of(x + shift[0], ax))
        assert np.array_equal(by[px.chip, px.row], P._bin_of(y + shift[1], ax))
    # Cached: the same tables while nothing changes, new ones when the shift moves.
    a = p._plane_bins(PL, (1.0, 2.0), ax)
    assert p._plane_bins(PL, (1.0, 2.0), ax)[0] is a[0]
    assert p._plane_bins(PL, (1.5, 2.0), ax)[0] is not a[0]


def test_the_single_plane_maps_cost_little():
    """A 20000-pixel frame at the default cap (asserted loosely; measured ~0.03 ms)."""
    import time

    rng = np.random.default_rng(3)
    px = _px(sorted(_random_hits(rng, 20000, 30_000_000, chips=8, bad_rows=True)))
    p = _plugin()
    hp = p.h["pairs"]
    cands = PR.plane_candidates(px, PL)
    best = 1.0
    for _ in range(20):
        t0 = time.perf_counter()
        idx, nh = PR.plane_sample(cands, PR.MAX_HITS)
        bx, by = p._plane_bins(PL, (0.0, 0.0), hp["hits_L1"].x)
        chip = px.chip[idx]
        flat = by[chip, px.row[idx]] * 132 + bx[chip, px.col[idx]]
        P._add_flat(hp["hits_L1"], flat[:nh[0]])
        P._add_flat(hp["hits_L2"], flat[nh[0]:])
        best = min(best, time.perf_counter() - t0)
    assert list(nh) == [PR.MAX_HITS, PR.MAX_HITS]
    assert best < 2e-3, best


def test_a_window_wider_than_the_dt_axis_is_noted():
    cfg = P.parse_settings({"MuPix": {"Pairs": {"window ns": 150}}})
    assert cfg.pairs.window_ns == 150
    assert any(e.startswith("MuPix/Pairs/window ns=150") and "+-100" in e for e in cfg.errors)
    assert not P.parse_settings({"MuPix": {"Pairs": {"window ns": 100}}}).errors


def test_resets_count_only_while_the_pairs_are_on():
    off = {"MuPix": {"Pairs": {"enable": False}}}
    p = _plugin(off)
    for tot in ((5, 9), (6, 10)):
        settings = old_layout({"MuPix": {"Pairs": {"enable": False},
                                         "XY": {"tot light max": tot[0],
                                                "tot heavy min": tot[1]}}})
        p.apply_settings(settings, rebuild=False)
    assert p.pair_resets == 0
    p.apply_settings(old_layout({}), rebuild=False)          # on again: books the maps
    assert p.pair_resets == 1 and p.summary()["pairs"]["resets"] == 1


def test_the_offline_line_without_pairs_prints_no_none():
    from mdqm.tools import sma_file as S

    p = _plugin()
    line = S.mupix_pairs_line(p.summary()["pairs"])
    assert "None" not in line and "- L2 candidates per pair" in line


# --- the offline tool -----------------------------------------------------------------------------

def _offline(tmp_path, *args):
    from sma_layouts import OLD_LAYOUT
    from test_sma_file import write_file

    from mdqm.tools import sma_file as S

    tmp_path.mkdir(parents=True, exist_ok=True)
    path = write_file(tmp_path / "run01008_00001.mid", [R1008], run=1008)
    out = tmp_path / "out"
    rc = S.main([str(path), "--out", str(out), "--quiet", "--settings", json.dumps(OLD_LAYOUT),
                 *args])
    return rc, out


def test_the_offline_tool_draws_the_pairs_page(tmp_path):
    pytest.importorskip("matplotlib")
    rc, out = _offline(tmp_path)
    assert rc in (0, 3)
    assert (out / "mupix_pairs.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    s = json.loads((out / "summary.json").read_text())["pairs"]
    assert s["enabled"] and s["n_pairs"] > 300
    z = np.load(out / "hists.npz")
    assert int(z["sma/mupix_pair_xy.entries"]) == s["n_pairs"]


def test_the_offline_tool_without_pairs_draws_no_pairs_page(tmp_path):
    pytest.importorskip("matplotlib")
    rc, out = _offline(tmp_path, "--settings",
                       json.dumps(old_layout({"MuPix": {"Pairs": {"enable": False}}})))
    assert rc in (0, 3)
    assert not (out / "mupix_pairs.png").exists()
    assert json.loads((out / "summary.json").read_text())["pairs"]["enabled"] is False


def test_the_pairs_page_passes_the_figure_checks():
    pytest.importorskip("matplotlib")
    root = WS / "psm-analysis-josh-2026/artifact-checks"
    if not (root / "artifact_checks.py").exists():
        pytest.skip("artifact-checks not here")
    sys.path.insert(0, str(root))
    try:
        from artifact_checks import check_figure
    finally:
        sys.path.remove(str(root))
    from mdqm.tools import sma_file as S

    p = _plugin()
    _feed(p, R1008, run=1008)
    fig = S.mupix_pairs_figure(S.offline_summary(p), p.store,
                               "SMA DQM  run01008_00001  (run 1008)")
    assert check_figure(fig, "mupix_pairs") == []
    assert "pairs" in S.mupix_pairs_line(S.offline_summary(p))
