"""MuPix x/y on the SMA DQM: the placement, the S1-seeded track finder and the plugin's fills.

The geometry is pinned to the reco chain's worked examples and, when the
psm-analysis ``pane_weights`` package is importable, to its geometry on every
corner of every chip. Tracks are built from pixel words (``W.encode_pixel``)
at known pixel centres and must come back exactly.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from sma_layouts import old_layout
from test_mupix import frame_words, pixels
from test_sma_plugin import R1008, _Clock, _feed, _plugin
from test_sma_seed_choice import SPACING, T0

from mdqm.plugins import sma as P
from mdqm.plugins import sma_mupix_xy as X
from mdqm.plugins import sma_words as W

PL = X.placement()
WS = Path(__file__).resolve().parents[3]          # the testbeam-env workspace


def centre(chip, col, row):
    x, y = PL.xy(chip, col, row)
    return float(x), float(y)


# --- geometry -------------------------------------------------------------------------------------

@pytest.mark.parametrize("chip, col, row, want", [
    (0, 0, 0, (0.20, -20.12)),          # L1 q0, beam-left bottom, its inner corner pixel
    (4, 255, 249, (20.60, -0.20)),      # L2 q0, the far corner
    (2, 0, 0, (20.60, 20.12)),          # L1 q2, turned by pi: col 0 is its outer edge
])
def test_the_worked_examples_of_the_reco_chain(chip, col, row, want):
    assert centre(chip, col, row) == pytest.approx(want, abs=1e-9)


def test_quadrants_follow_the_list_position_and_the_rotation():
    for chip in range(8):
        q = chip % 4
        gx, gy, s = X.QUADRANTS[q]
        assert PL.plane[chip] == (W.PLANE_L1 if chip < 4 else W.PLANE_L2)
        xs, ys = PL.xy([chip] * 4, [0, 255, 0, 255], [0, 0, 249, 249])
        assert (np.sign(xs) == gx).all() and (np.sign(ys) == gy).all(), chip
        # +col runs to +x on the lower chips and to -x on the turned upper ones.
        assert np.sign(xs[1] - xs[0]) == s and np.sign(ys[2] - ys[0]) == s
    assert (PL.plane[8:] == 0).all()
    # Runs <= 186: L1 = 1-4, L2 = 5, 6, 7, 0. Chip 0 is then L2 q3.
    old = X.placement((1, 2, 3, 4), (5, 6, 7, 0))
    assert old.plane[0] == W.PLANE_L2 and old.quadrant[0] == 3 and old.quadrant[1] == 0
    x, y = old.xy(0, 0, 0)
    assert (float(x), float(y)) == pytest.approx((-0.20, 20.12), abs=1e-9)
    # A fifth chip in a list has no place.
    extra = X.placement((0, 1, 2, 3, 9), (4, 5, 6, 7))
    assert extra.unplaced == (9,) and extra.plane[9] == 0


def _pane_weights_geometry():
    root = Path(os.environ.get("MDQM_PANE_WEIGHTS",
                               WS / "psm-analysis-josh-2026/mupix-pane-weights"))
    paths = [WS / "main/reco_testbeam/conditions/bt2026_psm_geometry.json",
             WS / "scratch/worktrees/reco_testbeam-release-1003/conditions"
                  / "bt2026_psm_geometry.json"]
    if not root.is_dir():
        pytest.skip("psm-analysis pane_weights not here")
    sys.path.insert(0, str(root))
    try:
        from pane_weights.geometry import load_geometry
    except Exception as exc:                      # noqa: BLE001
        pytest.skip(f"pane_weights not importable: {exc}")
    finally:
        sys.path.remove(str(root))
    for path in paths:
        try:
            return load_geometry(X.GEOMETRY_TAG, path)
        except Exception:                          # noqa: BLE001
            continue
    pytest.skip(f"no geometry JSON with tag {X.GEOMETRY_TAG}")


def test_every_corner_of_every_chip_agrees_with_pane_weights():
    g = _pane_weights_geometry()
    cols, rows = np.array([0, 255, 0, 255]), np.array([0, 0, 249, 249])
    for chip in range(8):
        plane, q = (1001, chip) if chip < 4 else (1002, chip - 4)
        c = g.chips[plane * 10 + q + 1]
        x, y = PL.xy(np.full(4, chip), cols, rows)
        np.testing.assert_allclose(x, c.col_x(cols), atol=1e-6)
        np.testing.assert_allclose(y, c.row_y(rows), atol=1e-6)
    assert g.distance_l12 == X.LEVER_MM


def test_the_stage_shift_sign():
    # The stage's +x is beam-right: the frame moves the other way in x.
    assert X.stage_shift(2.5, -1.0) == (-2.5, -1.0)


def test_the_axes_match_the_nearline():
    assert X.SLOPE_STEP_MRAD == pytest.approx(8 / 3)
    assert X.SLOPE_HALF_MRAD == pytest.approx(102.6667, abs=1e-3)
    assert (X.POS_HI_MM - X.POS_LO_MM) / X.POS_BINS == pytest.approx(0.64)
    # No pixel centre and no two-pixel mean on a position edge.
    edges = X.POS_LO_MM + 0.64 * np.arange(X.POS_BINS + 1)
    xs = np.concatenate([PL.xy(np.full(256, c), np.arange(256), np.zeros(256))[0]
                         for c in range(4)])
    half = np.sort(xs)[:-1] + 0.04
    for v in (xs, half):
        d = np.abs(v[:, None] - edges[None, :]).min()
        assert d > 0.01, d


# --- the track finder -----------------------------------------------------------------------------

def _px(hits, pad=True):
    """Pixels of ``[(t, chip, tot, col, row), ...]``; with ``pad``, two plane-less
    hits far out so that every S1 window lies in the pixel data."""
    if pad:
        hits = list(hits) + [(T0 - 100_000, 9, 1, 0, 0), (T0 + 10**8, 9, 1, 0, 0)]
    return W.prepare_pixels(pixels(sorted(hits)))


def _cluster(t, chip, cells, tot=7):
    return [(t, chip, tot, c, r) for c, r in cells]


def test_synthetic_tracks_come_back_exactly():
    rng = np.random.default_rng(3)
    shapes = ([(0, 0)], [(0, 0), (1, 0)], [(0, 0), (0, 1)], [(0, 0), (1, 0), (0, 1), (1, 1)],
              [(0, 0), (2, 2)], [(0, 0), (1, 1), (2, 0)])
    t_s1, hits, want = [], [], []
    for k in range(200):
        t = T0 + k * SPACING
        t_s1.append(t)
        row = []
        for p, chips in ((1, range(4)), (2, range(4, 8))):
            chip = int(rng.choice(list(chips)))
            col0, row0 = int(rng.integers(0, 250)), int(rng.integers(0, 245))
            cells = [(col0 + a, row0 + b) for a, b in shapes[int(rng.integers(len(shapes)))]]
            dt = 8 * int(rng.integers(-18, 56))      # whole ticks in [-144, 440]
            hits += _cluster(t + dt, chip, cells)
            xs, ys = PL.xy([chip] * len(cells), [c for c, _ in cells], [r for _, r in cells])
            row += [xs.mean(), ys.mean()]
        want.append(row)
    want = np.array(want)
    tr = X.s1_tracks(np.array(t_s1), _px(hits), pl=PL)
    assert (tr.state == X.TRACK).all()
    np.testing.assert_allclose(np.c_[tr.x1, tr.y1, tr.x2, tr.y2], want, atol=1e-9)
    np.testing.assert_allclose(tr.xp, 1000 * (want[:, 2] - want[:, 0]) / 30, atol=1e-9)
    np.testing.assert_allclose(tr.yp, 1000 * (want[:, 3] - want[:, 1]) / 30, atol=1e-9)
    assert list(tr.state_counts()) == [0, 0, 0, 200]


def test_overlapping_windows_share_their_pixels_and_each_pixel_is_mapped_once():
    t = T0 + np.array([0, 100, 5000])
    hits = _cluster(T0 + 150, 1, [(10, 20)]) + _cluster(T0 + 160, 5, [(10, 20)])
    tr = X.s1_tracks(t, _px(hits), pl=PL)
    assert list(tr.state) == [X.TRACK, X.TRACK, X.NO_L1]
    assert tr.hits[W.PLANE_L1][0].size == 1 and tr.hits[W.PLANE_L2][0].size == 1


def test_the_window_is_half_open_on_raw_times():
    assert T0 % 8 == 0                  # pixel times are whole 8 ns ticks
    t = T0 + np.array([0, 10_000, 20_000, 30_000])
    l2 = [h for k in t for h in _cluster(k, 5, [(10, 20)])]
    for dt, inside in ((-144, True), (-152, False), (440, True), (448, False)):
        hits = l2 + [h for k in t for h in _cluster(k + dt, 1, [(10, 20)])]
        tr = X.s1_tracks(t, _px(hits), (-144, 448), pl=PL)
        assert (tr.state == (X.TRACK if inside else X.NO_L1)).all(), dt


@pytest.mark.parametrize("l1, why", [
    (_cluster(T0, 1, [(10, 20), (40, 20)]), "two particles on one chip"),
    (_cluster(T0, 1, [(10, 20)]) + _cluster(T0, 2, [(10, 20)]), "two chips"),
    (_cluster(T0, 1, [(10 + a, 20 + b) for a in range(4) for b in range(3)]), "12-pixel burst"),
    (_cluster(T0, 1, [(10, 20), (13, 20)]), "4 columns: one past the 3 x 3 square"),
])
def test_what_fails_the_square_is_ambiguous(l1, why):
    hits = l1 + _cluster(T0, 5, [(10, 20)])
    tr = X.s1_tracks(np.array([T0]), _px(hits), pl=PL)
    assert tr.state[0] == X.AMBIGUOUS, why
    assert np.isnan(tr.x1[0]) and np.isfinite(tr.x2[0]) and np.isnan(tr.xp[0])


def test_a_wider_box_accepts_the_four_column_cluster():
    hits = _cluster(T0, 1, [(10, 20), (13, 20)]) + _cluster(T0, 5, [(10, 20)])
    tr = X.s1_tracks(np.array([T0]), _px(hits), box=4, pl=PL)
    assert tr.state[0] == X.TRACK and tr.x1[0] == pytest.approx(centre(1, 11.5, 20)[0])


def test_rows_off_the_sensor_and_unplaced_chips_are_dropped():
    t = T0 + np.array([0, 10_000, 20_000])
    hits = (_cluster(t[0], 1, [(10, 252)]) + _cluster(t[0], 5, [(10, 20)])          # only row 252
            + _cluster(t[1], 1, [(10, 20), (90, 252)]) + _cluster(t[1], 5, [(10, 20)])
            + _cluster(t[2], 1, [(10, 20)]) + _cluster(t[2], 5, [(10, 20)])
            + _cluster(t[2], 9, [(100, 100)]))                                     # no plane
    tr = X.s1_tracks(t, _px(hits), pl=PL)
    assert list(tr.state) == [X.NO_L1, X.TRACK, X.TRACK]
    assert tr.x1[1] == pytest.approx(centre(1, 10, 20)[0])
    # A chip past its plane's fourth: no place, its pixels are not seen.
    pl = X.placement((0, 1, 2, 3, 9), (4, 5, 6, 7))
    cuts = W.MuPixCuts(l1=(0, 1, 2, 3, 9))
    hits = _cluster(T0, 9, [(10, 20)]) + _cluster(T0, 5, [(10, 20)])
    px = W.prepare_pixels(pixels(sorted(hits + [(T0 - 100_000, 9, 1, 0, 0),
                                                (T0 + 10**8, 9, 1, 0, 0)])), cuts)
    assert X.s1_tracks(np.array([T0]), px, pl=pl).state[0] == X.NO_L1


def test_the_state_order_and_the_unjudged_rows():
    t = T0 + np.array([0, 10_000, 20_000, 30_000])
    hits = (_cluster(t[0], 5, [(10, 20)])                                           # no L1
            + _cluster(t[1], 1, [(10, 20)])                                         # no L2
            + _cluster(t[2], 1, [(10, 20)]) + _cluster(t[2], 5, [(10, 20), (80, 80)])
            + _cluster(t[3], 5, [(10, 20), (80, 80)]))       # no L1 wins over ambiguous L2
    tr = X.s1_tracks(t, _px(hits), pl=PL)
    assert list(tr.state) == [X.NO_L1, X.NO_L2, X.AMBIGUOUS, X.NO_L1]
    assert list(tr.state_counts()) == [2, 1, 1, 0]
    # Without the padding, the first and last windows stick out of the pixel data.
    tr = X.s1_tracks(t, _px(hits, pad=False), pl=PL)
    assert tr.state[0] == X.NOT_JUDGED and tr.state[-1] == X.NOT_JUDGED
    assert tr.state_counts().sum() == 2
    empty = X.s1_tracks(t, None)
    assert (empty.state == X.NOT_JUDGED).all() and empty.state_counts().sum() == 0


@pytest.mark.parametrize("tot1, tot2, light, heavy", [
    (5, 5, True, False), (0, 5, True, False), (6, 5, False, False), (5, 6, False, False),
    (9, 9, False, True), (31, 9, False, True), (8, 9, False, False), (9, 8, False, False),
    (5, 9, False, False),
])
def test_the_tot_classes_at_the_threshold_edges(tot1, tot2, light, heavy):
    # The plane ToT is the largest of its cluster's pixels.
    hits = (_cluster(T0, 1, [(10, 20)], tot=tot1) + _cluster(T0, 1, [(11, 20)], tot=0)
            + _cluster(T0, 5, [(10, 20)], tot=tot2))
    tr = X.s1_tracks(np.array([T0]), _px(hits), pl=PL)
    assert (tr.tot1[0], tr.tot2[0]) == (tot1, tot2)
    lt, hv = tr.classes(5, 9)
    assert (bool(lt[0]), bool(hv[0])) == (light, heavy)


def test_the_shift_moves_positions_not_slopes():
    hits = _cluster(T0, 1, [(10, 20)]) + _cluster(T0, 6, [(30, 40)])
    a = X.s1_tracks(np.array([T0]), _px(hits), pl=PL)
    b = X.s1_tracks(np.array([T0]), _px(hits), pl=PL, shift=X.stage_shift(2.0, 3.0))
    assert b.x1[0] == pytest.approx(a.x1[0] - 2.0) and b.y1[0] == pytest.approx(a.y1[0] + 3.0)
    assert b.x2[0] == pytest.approx(a.x2[0] - 2.0) and b.y2[0] == pytest.approx(a.y2[0] + 3.0)
    assert b.xp[0] == pytest.approx(a.xp[0]) and b.yp[0] == pytest.approx(a.yp[0])
    hx, hy = b.hits[W.PLANE_L1]
    assert hx[0] == pytest.approx(a.hits[W.PLANE_L1][0][0] - 2.0)


def _reference(t_s1, px, window, box, pl, judged):
    """s1_tracks the slow way: one S1 row at a time, plain Python."""
    lo, hi = window
    out = []
    for i, t in enumerate(t_s1):
        if not judged[i]:
            out.append((X.NOT_JUDGED, None, None, -1, -1))
            continue
        planes = []
        for p in (W.PLANE_L1, W.PLANE_L2):
            sel = [j for j in px.planes[p] if t + lo <= px.t[j] < t + hi
                   and px.row[j] < W.PIXEL_ROWS and pl.plane[px.chip[j]] == p]
            if not sel:
                planes.append((False, False, None, -1))
                continue
            chips = {int(px.chip[j]) for j in sel}
            cols = [int(px.col[j]) for j in sel]
            rows = [int(px.row[j]) for j in sel]
            ok = (len(chips) == 1 and max(cols) - min(cols) < box
                  and max(rows) - min(rows) < box)
            xy = None
            if ok:
                xs, ys = pl.xy([chips.pop()] * len(sel), cols, rows)
                xy = (float(xs.mean()), float(ys.mean()))
            planes.append((True, ok, xy, max(int(px.tot[j]) for j in sel)))
        (h1, ok1, xy1, t1), (h2, ok2, xy2, t2) = planes
        st = (X.NO_L1 if not h1 else X.NO_L2 if not h2 else
              X.TRACK if ok1 and ok2 else X.AMBIGUOUS)
        out.append((st, xy1, xy2, t1, t2))
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_dense_random_frame_matches_the_slow_reference(seed):
    """Overlapping windows, unjudged rows, rows >= 250, unplaced chips, bursts."""
    rng = np.random.default_rng(seed)
    n = 3000
    t = np.sort(rng.integers(T0, T0 + 2_000_000, n))
    hits = list(zip(t.tolist(), rng.integers(0, 10, n).tolist(), rng.integers(0, 32, n).tolist(),
                    rng.integers(0, 256, n).tolist(), rng.integers(0, 256, n).tolist(),
                    strict=True))
    # Small clusters next to some hits, so that part of the windows is accepted.
    for k in rng.choice(n, 600, replace=False):
        tk, c, tot, col, row = hits[k]
        hits.append((tk + 8, c, int(rng.integers(0, 32)), min(col + 1, 255), row))
    pl = X.placement((0, 1, 2, 3), (4, 5, 6, 7, 8))          # chip 8: no place; 9: no plane
    cuts = W.MuPixCuts(l1=(0, 1, 2, 3), l2=(4, 5, 6, 7, 8))
    px = W.prepare_pixels(pixels(sorted(hits)), cuts)
    t_s1 = np.sort(rng.integers(T0 - 5000, T0 + 2_005_000, 400))  # the ends stick out
    win = (-150, 450)
    tr = X.s1_tracks(t_s1, px, win, 3, pl)
    judged = W.mupix_coverage(t_s1, px, *win)
    assert 0 < judged.sum() < t_s1.size
    ref = _reference(t_s1, px, win, 3, pl, judged)
    states = [r[0] for r in ref]
    assert list(tr.state) == states
    assert set(states) >= {X.NO_L1, X.NO_L2, X.AMBIGUOUS, X.TRACK}
    for i, (st, xy1, xy2, t1, t2) in enumerate(ref):
        assert (tr.tot1[i], tr.tot2[i]) == (t1, t2), i
        for got, want in (((tr.x1[i], tr.y1[i]), xy1), ((tr.x2[i], tr.y2[i]), xy2)):
            if want is None:
                assert np.isnan(got[0]) and np.isnan(got[1]), i
            else:
                assert got == pytest.approx(want, abs=1e-9), i
    # The hit maps: every kept pixel in a judged window, once.
    for p in (W.PLANE_L1, W.PLANE_L2):
        want = set()
        for i, tt in enumerate(t_s1):
            if judged[i]:
                want |= {j for j in px.planes[p] if tt - 150 <= px.t[j] < tt + 450
                         and px.row[j] < W.PIXEL_ROWS and pl.plane[px.chip[j]] == p}
        assert tr.hits[p][0].size == len(want)
        wx, wy = pl.xy(px.chip[sorted(want)], px.col[sorted(want)], px.row[sorted(want)])
        np.testing.assert_allclose(np.sort(tr.hits[p][0]), np.sort(wx), atol=1e-9)


def test_a_dense_frame_is_fast_enough():
    """500 S1 rows on a 20000-pixel frame: the cost target is 0.3 ms (asserted loosely)."""
    import time

    rng = np.random.default_rng(1)
    n = 20000
    t = np.sort(rng.integers(T0, T0 + 30_000_000, n))
    hits = list(zip(t.tolist(), rng.integers(0, 8, n).tolist(), rng.integers(0, 32, n).tolist(),
                    rng.integers(0, 256, n).tolist(), rng.integers(0, 250, n).tolist(),
                    strict=True))
    px = _px(hits)
    t_s1 = np.sort(rng.integers(T0 + 1000, T0 + 29_000_000, 500))
    X.s1_tracks(t_s1, px, pl=PL)
    best = min(_timed(lambda: X.s1_tracks(t_s1, px, pl=PL), time) for _ in range(20))
    assert best < 5e-3, best


def _timed(fn, time):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


# --- the plugin -----------------------------------------------------------------------------------

XY_HISTS = ("mupix_hits_xy_L1", "mupix_hits_xy_L2", "mupix_track_xy", "mupix_track_xy_light",
            "mupix_track_xy_heavy", "mupix_track_xxp", "mupix_track_xxp_light",
            "mupix_track_xxp_heavy", "mupix_track_yyp", "mupix_track_yyp_light",
            "mupix_track_yyp_heavy", "mupix_track_tot", "mupix_track_state")


def _bin(h, x, y):
    ix = int(np.floor((x - h.x.lo) / ((h.x.hi - h.x.lo) / h.x.n))) + 1
    iy = int(np.floor((y - h.y.lo) / ((h.y.hi - h.y.lo) / h.y.n))) + 1
    return int(h.counts[iy, ix])


def test_the_plugin_books_the_table_and_fills_a_track_frame():
    # Explicit cuts: the defaults are provisional and will be retuned.
    p = _plugin({"MuPix": {"XY": {"tot light max": 5, "tot heavy min": 9}}})
    names = [n[4:] for n in p.store.names() if n.startswith("sma/mupix_track")
             or n.startswith("sma/mupix_hits_xy")]
    assert sorted(names) == sorted(XY_HISTS)
    h = lambda n: p.store.get(f"sma/{n}")  # noqa: E731
    assert (h("mupix_track_xy").x.n, h("mupix_track_xy").y.n) == (130, 130)
    assert (h("mupix_track_xxp").x.n, h("mupix_track_xxp").y.n) == (130, 77)
    assert (h("mupix_track_tot").x.n, h("mupix_track_tot").y.n) == (32, 32)
    assert sum(p.store.get(f"sma/{n}").counts.size for n in XY_HISTS) * 8 < 1.25e6
    # frame_words: per S1, an L1 pixel on chip 1 (10, 20), ToT 12, an L2 one on chip 5, ToT 9.
    _feed(p, frame_words(n=40), run=1)
    st = h("mupix_track_state").counts[1:-1]
    n = int(st.sum())
    assert 35 <= n <= 40 and list(st) == [0, 0, 0, n]
    x, y = centre(1, 10, 20)
    assert _bin(h("mupix_track_xy"), x, y) == n
    assert _bin(h("mupix_track_xxp"), x, 0.0) == n and _bin(h("mupix_track_yyp"), y, 0.0) == n
    assert h("mupix_track_xy_heavy").entries == n and h("mupix_track_xy_light").entries == 0
    assert int(h("mupix_track_tot").counts[1 + 9, 1 + 12]) == n
    assert _bin(h("mupix_hits_xy_L1"), x, y) == n
    xy = p.summary()["xy"]
    assert xy["fractions"]["track"] == 1.0 and xy["heavy_frac"] == 1.0 and xy["light_frac"] == 0.0
    assert xy["tracks"] == n == xy["n_s1"]


def test_the_stage_comes_from_the_odb_poll_and_moves_no_plot():
    p = _plugin()
    calls = []

    def odb_get(path):
        calls.append(path)
        return [4.0, -1.5]

    p.poll_odb(odb_get)
    assert calls == [X.STAGE_PATH]
    _feed(p, frame_words(n=40), run=1)
    x, y = centre(1, 10, 20)
    h = p.store.get("sma/mupix_track_xy")
    n = h.entries
    assert n and _bin(h, x - 4.0, y - 1.5) == n
    st = p.summary()["xy"]["stage"]
    assert st == {"x_mm": 4.0, "y_mm": -1.5, "source": "odb", "applied": True,
                  "shift_mm": [-4.0, -1.5], "note": None}
    assert P.shape_fingerprint({}) == p.shape_fingerprint({}), "the stage is no setting"


def test_a_missing_stage_key_is_zero_with_a_note():
    p = _plugin()

    def odb_get(path):
        raise KeyError(path)

    p.poll_odb(odb_get)
    st = p.summary()["xy"]["stage"]
    assert st["source"] == "missing" and st["shift_mm"] == [0.0, 0.0]
    assert X.STAGE_PATH in st["note"]
    p.poll_odb(lambda path: [1.0])                      # one entry: unreadable too
    assert p.summary()["xy"]["stage"]["source"] == "missing"


def test_apply_stage_shift_off_reads_nothing_and_shifts_nothing():
    p = _plugin({"MuPix": {"XY": {"apply stage shift": False}}})
    p.poll_odb(lambda path: pytest.fail("read the stage"))
    p.set_stage(4.0, 2.0)
    _feed(p, frame_words(n=40), run=1)
    x, y = centre(1, 10, 20)
    h = p.store.get("sma/mupix_track_xy")
    assert _bin(h, x, y) == h.entries > 0
    st = p.summary()["xy"]["stage"]
    assert st["applied"] is False and st["shift_mm"] == [0.0, 0.0]


def test_the_analyzer_hands_the_plugin_its_odb_reader_every_settings_poll():
    from test_sma_plugin import _analyzer_with_odb

    a, c = _analyzer_with_odb({X.STAGE_PATH: [1.25, 2.5]})
    st = a.plugin.summary()["xy"]["stage"]
    assert (st["source"], st["x_mm"], st["y_mm"]) == ("odb", 1.25, 2.5)
    rebuilds = a.plugin.rebuilds
    c.tree[X.STAGE_PATH] = [3.0, 2.5]
    a._settings_checked = 0.0
    a.apply_settings(c)
    assert a.plugin.summary()["xy"]["stage"]["x_mm"] == 3.0
    assert a.plugin.rebuilds == rebuilds, "a moving stage resets no plot"
    del c.tree[X.STAGE_PATH]
    a._settings_checked = 0.0
    a.apply_settings(c)
    assert a.plugin.summary()["xy"]["stage"]["source"] == "missing"


@pytest.mark.parametrize("settings", [
    {"MuPix": {"XY": {"enable": False}}},
    {"MuPix": {"max pixel hits per frame": 0}},           # the MuPix analysis off altogether
])
def test_xy_off_books_and_fills_nothing(settings):
    p = _plugin(settings)
    assert not [n for n in p.store.names() if n[4:] in XY_HISTS]
    p.poll_odb(lambda path: pytest.fail("read the stage"))
    _feed(p, frame_words(n=40), run=1)
    xy = p.summary()["xy"]
    assert xy["enabled"] is False and xy["n_s1"] == 0 and xy["fractions"]["track"] is None


@pytest.mark.parametrize("tree, err", [
    ({"cluster box px": 0}, "cluster box px"),
    ({"cluster box px": 65}, "cluster box px"),
    ({"tot light max": 9, "tot heavy min": 9}, "must be below"),
    ({"tot heavy min": 32}, "tot heavy min"),
    ({"tot light max": -1}, "tot light max"),
    ({"enable": "maybe"}, "enable"),
    ({"apply stage shift": 2}, "apply stage shift"),
    ({"max S1 per frame": -5}, "max S1 per frame"),
    ({"max S1 per frame": 0}, "0 is not 'all'"),
    ({"max S1 per frame": 5000}, "using 2000"),
])
def test_bad_xy_settings_fall_back_and_are_reported(tree, err):
    cfg = P.parse_settings({"MuPix": {"XY": tree}})
    assert any(e.startswith("MuPix/XY/") and err in e for e in cfg.errors), cfg.errors
    d = X.XYSettings()
    assert cfg.xy.tot_light_max < cfg.xy.tot_heavy_min
    assert 1 <= cfg.xy.box <= X.MAX_BOX_PX
    if "tot" in err or "below" in err:
        assert (cfg.xy.tot_light_max, cfg.xy.tot_heavy_min) == (d.tot_light_max, d.tot_heavy_min)


def test_good_xy_settings_parse():
    cfg = P.parse_settings({"MuPix": {"XY": {"enable": "n", "cluster box px": 5,
                                             "tot light max": 3, "tot heavy min": 10,
                                             "apply stage shift": 0, "max S1 per frame": 7}}})
    assert not cfg.errors
    assert (cfg.xy.enable, cfg.xy.box, cfg.xy.tot_light_max, cfg.xy.tot_heavy_min,
            cfg.xy.apply_stage, cfg.xy.max_s1) == (False, 5, 3, 10, False, 7)
    # The placement follows the MuPix chip lists.
    cfg = P.parse_settings({"MuPix": {"L1 chips": [1, 2, 3, 4], "L2 chips": [5, 6, 7, 0]}})
    assert cfg.xy.placement.quadrant[0] == 3 and cfg.xy.placement.plane[0] == W.PLANE_L2


def test_no_xy_key_is_in_the_shape_fingerprint():
    base = P.shape_fingerprint({})
    for k, v in (("enable", False), ("cluster box px", 4), ("tot light max", 4),
                 ("tot heavy min", 10), ("apply stage shift", False), ("max S1 per frame", 9)):
        assert P.shape_fingerprint({"MuPix": {"XY": {k: v}}}) == base, k
    assert P.shape_fingerprint({"MuPix": {"L1 chips": [1, 0, 2, 3]}}) != base


def _sma_counts(p):
    return {n: (p.store.get(n).entries, p.store.get(n).counts.copy()) for n in p.store.names()
            if not n.startswith(("sma/mupix_track_", "sma/mupix_hits_xy_"))}


def _xy_entries(p):
    return {n[4:]: p.store.get(n).entries for n in p.store.names()
            if n.startswith(("sma/mupix_track_", "sma/mupix_hits_xy_"))}


@pytest.mark.parametrize("edit, reset", [
    ({"cluster box px": 4}, "all"),
    ({"apply stage shift": False}, "all"),
    ({"tot light max": 5, "tot heavy min": 9}, "classes"),
    ({"max S1 per frame": 100}, "none"),
])
def test_an_xy_edit_resets_only_the_xy_maps(edit, reset):
    base = {"MuPix": {"XY": {"tot light max": 8, "tot heavy min": 9}}}
    p = _plugin(base)
    _feed(p, frame_words(n=40), run=1)
    before, xy0 = _sma_counts(p), _xy_entries(p)
    s0 = p.summary()
    epoch, rebuilds = p.epoch, p.rebuilds
    assert xy0["mupix_track_xy"] > 0 and xy0["mupix_track_xy_heavy"] > 0   # ToT 12 and 9
    settings = old_layout({"MuPix": {"XY": {**base["MuPix"]["XY"], **edit}}})
    p.apply_settings(settings, rebuild=p.shape_fingerprint(settings) != p.shape_fingerprint(
        old_layout(base)))
    assert (p.epoch, p.rebuilds) == (epoch, rebuilds), "no rebuild, no new epoch"
    after = _sma_counts(p)
    assert after.keys() == before.keys()
    for n, (e, c) in before.items():
        assert after[n][0] == e and np.array_equal(after[n][1], c), n
    xy1 = _xy_entries(p)
    assert set(xy1) == set(xy0)
    classes = {k for k in xy0 if k.endswith(("_light", "_heavy"))}
    for k, e in xy1.items():
        zeroed = reset == "all" or (reset == "classes" and k in classes)
        assert e == (0 if zeroed else xy0[k]), k
    s1 = p.summary()
    assert s1["epoch"] == s0["epoch"] and s1["frames"] == s0["frames"]
    assert (s1["xy"]["n_s1"] == 0) == (reset == "all")
    assert s1["xy"]["resets"] == (0 if reset == "none" else 1)
    if reset == "classes":
        assert p.store.get("sma/mupix_track_xy_heavy").title.count("ToT >= 9")
        assert s1["xy"]["heavy_frac"] is None and s1["xy"]["tracks"] == s0["xy"]["tracks"]


def test_switching_xy_off_and_on_books_only_the_xy_maps():
    p = _plugin()
    _feed(p, frame_words(n=40), run=1)
    before = _sma_counts(p)
    p.apply_settings(old_layout({"MuPix": {"XY": {"enable": False}}}), rebuild=False)
    assert not _xy_entries(p) and p.h["xy"] is None
    assert _sma_counts(p).keys() == before.keys()
    _feed(p, frame_words(n=40), run=1)                    # fills nothing x/y
    p.apply_settings(old_layout({}), rebuild=False)
    assert sorted(_xy_entries(p)) == sorted(XY_HISTS)
    assert all(e == 0 for e in _xy_entries(p).values())
    assert p.epoch == 0 and p.rebuilds == 0


def test_the_maps_are_uint32_and_widen_before_a_bin_could_wrap():
    p = _plugin()
    h = p.store.get("sma/mupix_track_xy")
    assert h.counts.dtype == np.uint32
    h.counts[5, 5] = np.iinfo(np.uint32).max - 1
    h.entries = int(np.iinfo(np.uint32).max) - 1
    P._add_flat(h, np.array([5 * 132 + 5, 5 * 132 + 5]))
    assert h.counts.dtype == np.uint64 and int(h.counts[5, 5]) == 2**32
    st = p.store.get("sma/mupix_track_state")
    st.entries = int(np.iinfo(np.uint32).max)
    P._widen(st, 1)
    assert st.counts.dtype == np.uint64


def test_the_xy_cap_takes_an_even_subset_of_the_mupix_sample():
    p = _plugin({"MuPix": {"XY": {"max S1 per frame": 10}}})
    _feed(p, frame_words(n=40), run=1)
    assert p.store.get("sma/mupix_track_state").entries <= 10
    assert p.summary()["mupix"]["n_s1"] > 30, "the MuPix matching keeps its own sample"


def test_the_summary_block_keys():
    clk = _Clock()
    p = _plugin(clock=clk)
    _feed(p, R1008, run=1008)
    xy = p.summary()["xy"]
    assert set(xy) == {"enabled", "off_reason", "geometry", "quadrants", "resets", "n_s1",
                       "tracks", "fractions", "light_frac", "heavy_frac", "stage", "cuts",
                       "unplaced_chips"}
    assert xy["off_reason"] is None and xy["resets"] == 0
    assert xy["quadrants"][0] == {"chip": 0, "plane": "L1", "quadrant": 0,
                                  "where": "beam-left bottom"}
    assert [q["chip"] for q in xy["quadrants"]] == list(range(8))
    assert set(xy["fractions"]) == {"track", "ambiguous", "no_l1", "no_l2"}
    assert set(xy["stage"]) == {"x_mm", "y_mm", "source", "applied", "shift_mm", "note"}
    assert set(xy["cuts"]) == {"cluster_box_px", "tot_light_max", "tot_heavy_min", "tot_ns",
                               "window_ns", "max_s1"}
    assert xy["geometry"] == "bt2026-v4" and xy["stage"]["source"] == "none"
    json.dumps(xy, allow_nan=False)
    # Run 1008: most analysed S1 hits make a track (an L1 and an L2 cluster in time).
    f = xy["fractions"]
    assert f["track"] > 0.7 and sum(f.values()) == pytest.approx(1.0, abs=1e-3)
    assert xy["tracks"] == p.store.get("sma/mupix_track_xy").entries


def test_real_frames_put_the_beam_where_the_hit_maps_are():
    """Run 1008: the track positions are the hit-map positions (same frame, same shift)."""
    p = _plugin()
    _feed(p, R1008, run=1008)
    h = p.store.get("sma/mupix_track_xy").counts[1:-1, 1:-1].astype(float)
    m = p.store.get("sma/mupix_hits_xy_L1").counts[1:-1, 1:-1].astype(float)
    xc = X.POS_LO_MM + 0.64 * (np.arange(130) + 0.5)
    mean = lambda c, axis: (c.sum(axis=axis) * xc).sum() / c.sum()  # noqa: E731
    # ~250 tracks of a ~9 mm RMS beam: the means agree within their ~0.6 mm errors.
    assert abs(mean(h, 0) - mean(m, 0)) < 2.0 and abs(mean(h, 1) - mean(m, 1)) < 2.0
    assert h.sum() > 150


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


def test_the_offline_tool_takes_the_stage_and_draws_the_xy_page(tmp_path):
    pytest.importorskip("matplotlib")
    rc0, out0 = _offline(tmp_path / "a", "--no-png")
    rc, out = _offline(tmp_path / "b", "--stage", "2.0", "-1.0")
    assert rc0 == rc and rc in (0, 3)          # one frame: a flag may be up
    s0 = json.loads((out0 / "summary.json").read_text())["xy"]
    s = json.loads((out / "summary.json").read_text())["xy"]
    # The test file's begin-of-run ODB has no XY table: (0, 0) with a note.
    assert s0["stage"]["source"] == "missing" and s0["stage"]["shift_mm"] == [0.0, 0.0]
    assert "--stage" in s0["stage"]["note"]
    assert s["stage"] == {"x_mm": 2.0, "y_mm": -1.0, "source": "manual", "applied": True,
                          "shift_mm": [-2.0, -1.0], "note": None}
    assert s["tracks"] == s0["tracks"] > 150
    assert (out / "mupix_xy.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert not (out0 / "mupix_xy.png").exists()
    # The track map moved by (-2, -1) mm: same counts, shifted mean.
    h0 = np.load(out0 / "hists.npz")["sma/mupix_track_xy"][1:-1, 1:-1].astype(float)
    h = np.load(out / "hists.npz")["sma/mupix_track_xy"][1:-1, 1:-1].astype(float)
    xc = X.POS_LO_MM + 0.64 * (np.arange(130) + 0.5)
    mx = lambda c: (c.sum(axis=0) * xc).sum() / c.sum()  # noqa: E731
    my = lambda c: (c.sum(axis=1) * xc).sum() / c.sum()  # noqa: E731
    assert mx(h) - mx(h0) == pytest.approx(-2.0, abs=0.35)
    assert my(h) - my(h0) == pytest.approx(-1.0, abs=0.35)


def test_the_offline_tool_without_xy_draws_no_xy_page(tmp_path):
    pytest.importorskip("matplotlib")
    rc, out = _offline(tmp_path, "--settings", '{"MuPix": {"XY": {"enable": false}}}')
    # The second --settings wins (argparse): the old layout is not applied, the frame
    # is still read.
    assert rc in (0, 3)
    assert not (out / "mupix_xy.png").exists()
    assert json.loads((out / "summary.json").read_text())["xy"]["enabled"] is False


def test_the_xy_page_passes_the_figure_checks():
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
    fig = S.mupix_xy_figure(S.offline_summary(p), p.store, "SMA DQM  run01008_00001  (run 1008)")
    assert check_figure(fig, "mupix_xy") == []


# --- chip lists (S1), the analyzer's error path, the file's stage -------------------------------

@pytest.mark.parametrize("l1, l2, why", [
    ([0, 2, 3], [4, 5, 6, 7], "3 entries"),                 # a dropped dead chip
    ([0, 1, 2, 3, 8], [4, 5, 6, 7], "5 entries"),
    ([0, 0, 2, 3], [4, 5, 6, 7], "more than once"),          # a duplicate
])
def test_chip_lists_that_cannot_place_the_chips_turn_xy_off(l1, l2, why):
    p = _plugin({"MuPix": {"L1 chips": l1, "L2 chips": l2}})
    errs = [e for e in p.cfg.errors if e.startswith("MuPix/XY: ")]
    assert len(errs) == 1 and why in errs[0] and "x/y is off" in errs[0], p.cfg.errors
    assert not _xy_entries(p) and p.h["xy"] is None
    _feed(p, frame_words(n=40), run=1)
    xy = p.summary()["xy"]
    assert xy["enabled"] is False and why in xy["off_reason"] and xy["quadrants"] == []
    # The rest of the MuPix analysis keeps the plane membership it had.
    assert p.summary()["mupix"]["n_s1"] > 30
    assert (p.cfg.mupix.planes[2] == W.PLANE_L1) and (p.cfg.mupix.planes[4] == W.PLANE_L2)


def test_a_reordered_list_moves_the_chips_and_the_summary_says_where():
    p = _plugin({"MuPix": {"L1 chips": [1, 0, 2, 3]}})
    assert not p.cfg.errors
    q = {d["chip"]: (d["plane"], d["quadrant"], d["where"]) for d in p.summary()["xy"]["quadrants"]}
    assert q[1] == ("L1", 0, "beam-left bottom") and q[0] == ("L1", 1, "beam-right bottom")
    _feed(p, frame_words(n=40), run=1)                 # L1 hits on chip 1, now at q0
    x, y = X.placement((1, 0, 2, 3)).xy(1, 10, 20)
    assert float(x) > 0                                # beam-left, mirrored from q1
    assert _bin(p.store.get("sma/mupix_track_xy"), float(x), float(y)) > 30


def test_a_minus_one_slot_is_an_empty_quadrant():
    p = _plugin({"MuPix": {"L1 chips": [0, -1, 2, 3]}})
    assert not p.cfg.errors and p.cfg.xy.active
    assert p.cfg.mupix.l1 == (0, 2, 3) and p.cfg.mupix.planes[1] == W.PLANE_NONE
    q = p.summary()["xy"]["quadrants"]
    assert [d["chip"] for d in q if d["plane"] == "L1"] == [0, 2, 3]
    assert [d["quadrant"] for d in q if d["plane"] == "L1"] == [0, 2, 3]
    # frame_words puts L1 on chip 1: now no plane, so every judged S1 hit is "no L1".
    _feed(p, frame_words(n=40), run=1)
    st = p.store.get("sma/mupix_track_state").counts[1:-1]
    assert st[0] == st.sum() > 30
    bad = P.parse_settings({"MuPix": {"L1 chips": [0, -2, 2, 3]}})
    assert any("L1 chips" in e for e in bad.errors)


def test_the_analyzer_logs_a_failing_odb_poll_once_and_goes_on():
    from test_sma_plugin import _analyzer_with_odb

    a, c = _analyzer_with_odb({X.STAGE_PATH: [1.0, 2.0]})
    # A failure outside the plugin's own key read (here: its cfg raising).
    real_cfg = a.plugin.cfg

    class Boom:
        def __getattr__(self, name):
            raise RuntimeError("cfg replaced mid-rebuild")

    a.plugin.cfg = Boom()
    with pytest.raises(RuntimeError, match="mid-rebuild"):
        a.plugin.poll_odb(c.odb_get)
    a.plugin.cfg = real_cfg
    st = a.plugin.summary()["xy"]["stage"]
    assert st["source"] == "error" and "cfg replaced mid-rebuild" in st["note"]
    assert (st["x_mm"], st["y_mm"]) == (1.0, 2.0), "the last position is kept"
    # Through the analyzer: logged once per distinct error, the poll goes on.
    a.plugin.poll_odb = lambda odb_get: (_ for _ in ()).throw(ValueError("bad stage"))
    n = len(c.messages)
    for _ in range(3):
        a._settings_checked = 0.0
        a.apply_settings(c)
    msgs = [m for m in c.messages[n:] if "bad stage" in m]
    assert len(msgs) == 1, c.messages[n:]
    assert a._poll_error == "ValueError: bad stage"


def test_the_offline_tool_reads_the_stage_from_the_begin_of_run_odb(tmp_path):
    from sma_layouts import OLD_LAYOUT
    from test_sma_file import h000

    from mdqm.tools import midasfile as MF
    from mdqm.tools import sma_file as S

    odb = {"Equipment": {"XYTable": {"Variables": {"Measured": [3.5, -1.25],
                                                   "Demand": [3.5, -1.25]}}},
           "Runinfo": {"Run number": 1008}}
    blob = (MF.encode_internal(MF.EVID_BOR, 1008, 1_790_000_000, json.dumps(odb).encode() + b"\0")
            + MF.encode_event(301, 100, 1_790_000_000, [h000(R1008)])
            + MF.encode_internal(MF.EVID_EOR, 1008, 1_790_000_000, b"{}\0"))
    path = tmp_path / "run01008_00001.mid"
    path.write_bytes(blob)
    for args, want in (((), (3.5, -1.25, "file", [-3.5, -1.25])),
                       (("--stage", "1", "2"), (1.0, 2.0, "manual", [-1.0, 2.0]))):
        out = tmp_path / f"out{len(args)}"
        rc = S.main([str(path), "--out", str(out), "--quiet", "--no-png", "--settings",
                     json.dumps(OLD_LAYOUT), *args])
        assert rc in (0, 3)
        st = json.loads((out / "summary.json").read_text())["xy"]["stage"]
        assert (st["x_mm"], st["y_mm"], st["source"], st["shift_mm"]) == want
    assert S.stage_of(b'<odb><dir name="Equipment"><dir name="XYTable"><dir name="Variables">'
                      b'<keyarray name="Measured" type="FLOAT" num_values="2"><value>1.5</value>'
                      b'<value>-2</value></keyarray></dir></dir></dir></odb>\0')[:3] == (
        1.5, -2.0, "file")
    assert S.stage_of(b"{not json")[2] == "missing"
