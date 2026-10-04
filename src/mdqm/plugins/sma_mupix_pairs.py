"""Unseeded MuPix L1-L2 pixel pairs for the SMA DQM: the nearline monitor's ``MakePairs``.

Pure functions over `sma_words.Pixels`, with no plugin state (the plugin,
``sma.py``, books and fills; the offline CLI runs the same code). This module
imports `sma_words` and `sma_mupix_xy` (the placement, the lever arm); neither
imports it.

The rule
--------
``PIPSMMuPixCore::MakePairs`` of reco_testbeam (``PIPSMMuPixCore.hh``, the
window from ``WindowRange``), with no S1 and no clustering: each L1 pixel takes
the L2 pixel nearest in time within +-``window`` ns, both edges inclusive. Ties
go to the earlier L2 pixel: the C++ walks the window in time order (its plane
lists are stable-sorted by time, so equal times keep the readout order) and
replaces its choice only on a strictly smaller |dt|, so of two pixels equally
far away the one before the L1 pixel wins, and of pixels with the same time
stamp the first in the stream. Nothing is consumed: one L2 pixel may partner
several L1 pixels. ``n_partners`` is the number of L2 pixels in the window
(the C++ ``nPartners``), 0 for an unpaired L1 pixel (which the C++ drops).

The candidate pixels are those the reco decoder keeps: rows >= 250 are not on
the sensor and ``PITMidasMusip`` drops them before anything else sees them;
pixels on a chip without a place in the plane lists (plane 0) belong to
neither plane. Both are dropped here before the pairing, so the candidate sets
are the same. Unlike the reco, nothing is masked (no hot-pixel mask) and the
times are the raw pixel time stamps (8 ns ticks, no time-walk correction).

Cost
----
At most ``max_l1`` L1 pixels per frame are paired, an even sample
(`sma_words.even_sample`) of the frame's candidates in time order. The search
is vectorised and exact on int64 ns: the window edges and the first L2 pixel
at or after each L1 pixel come from one ``searchsorted`` of the sampled L1
times into the sorted L2 times (integer times: ``t + w`` inclusive is ``t + w
+ 1`` exclusive), a second one finds the first pixel of the nearest earlier
time stamp. The two neighbours of each L1 time are compared; the rest is a
handful of gathers. About O(m log n) for m sampled L1 and n L2 pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mdqm.plugins import sma_mupix_xy as X
from mdqm.plugins import sma_words as W

#: Defaults of ``/DQM/SMA/MuPix/Pairs``. The nearline's window
#: (``PSM_MUPIX_WINDOW_NS``, PIPSMMuPixMonitor's ``Window``) is 40 ns on
#: time-walk corrected times; on the DQM's raw times 64 ns gives the
#: nearline's pair count (run 1008: 646k vs 655k; 40 ns pairs 73 % of the
#: L1 pixels against the nearline's 87 %).
WINDOW_NS = 64
#: The largest window accepted (whole ns): far beyond any real L1-L2 spread.
WINDOW_LIMIT_NS = 1000
#: L1 pixels paired per frame, at most (``Pairs/max L1 per frame``), within
#: 1..MAX_L1_LIMIT: the frame's examined pixels are capped at MAX_PIXELS anyway.
#: 1000 costs ~0.39 ms a dense frame in the analyzer's loop (~0.21 ms hot);
#: 500 ~0.29 ms, 2000 ~0.55 ms. At ~35 frames/s 1000 is ~3x10^4 pairs a
#: second, plenty for the maps in the 60 s summary window.
MAX_L1 = 1000
MAX_L1_LIMIT = W.MAX_PIXELS
#: ``mupix_pair_dt``: t(L2) - t(L1) (the nearline's sign, PIPSMMuPixMonitor
#: ``dt``) of every L2 candidate within +-DT_HALF_NS of a sampled L1 pixel,
#: in bins of one MuPix tick (the raw times are whole ticks, so finer bins
#: would be a comb), each centred on a multiple of 8 ns.
DT_HALF_NS = 100
DT_BIN_NS = W.PIXEL_TICK_NS
#: Entries of ``mupix_pair_dt`` per frame, at most DT_PER_L1 per sampled L1
#: pixel: a denser frame (a noise burst, a spill) uses an evenly spread subset
#: of the sampled L1 pixels for it, so its cost is bounded by the L1 cap like
#: the rest of the fill (~1 at run-1008 density, 7 at 20000 pixels in 300 us).
DT_PER_L1 = 4
#: ``mupix_pair_partners``: 0..PARTNER_MAX, more in the overflow bin.
PARTNER_MAX = 10
#: ``mupix_pair_hits_xy_L1`` / ``_L2`` (each plane alone, unseeded): pixels per
#: plane and frame, at most (``Pairs/max hits per frame``), an even sample of
#: the plane's candidates, within 1..MAX_HITS_LIMIT. A run-1008 frame has
#: ~3700 candidates a plane (dense synthetic ~6400), so 2000 is about every
#: second pixel there. The fill costs ~0.06 ms a frame at 2000 on one core
#: (~0.08 ms at 20000 on run 1008, ~0.13 ms on dense frames).
MAX_HITS = 2000
MAX_HITS_LIMIT = W.MAX_PIXELS


@dataclass
class PairSettings:
    """The parsed ``/DQM/SMA/MuPix/Pairs``, with the ToT cuts and the stage switch of
    ``MuPix/XY`` and the placement of the MuPix chip lists."""

    enable: bool = True
    window_ns: int = WINDOW_NS
    max_l1: int = MAX_L1
    #: Pixels per plane and frame on the single-plane maps (a CPU knob, no reset).
    max_hits: int = MAX_HITS
    tot_light_min: int = X.TOT_LIGHT_MIN
    tot_light_max: int = X.TOT_LIGHT_MAX
    tot_heavy_min: int = X.TOT_HEAVY_MIN
    apply_stage: bool = True
    placement: X.Placement = field(default_factory=X.placement)
    #: Why the pairs are off with enable = y (bad chip lists, the MuPix analysis off).
    off_reason: str | None = None

    @property
    def active(self) -> bool:
        return self.enable and self.off_reason is None

    def reset_key(self) -> tuple:
        """What, when it changes, resets every pair map (the ToT cuts reset only the classes)."""
        p = self.placement
        # max_l1 and max_hits are CPU knobs, as XY's "max S1 per frame": no reset.
        return (self.active, self.window_ns, self.apply_stage,
                p.quadrant.tobytes(), p.plane.tobytes())


@dataclass
class L1L2Pairs:
    """The sampled L1 pixels of one frame and their L2 partners (`l1l2_pairs`).

    Per sampled L1 pixel, in time order: ``i1`` its index into the `Pixels`,
    ``i2`` the partner's (-1 if none), ``dt`` = t(L2) - t(L1) in ns (the
    nearline's ``Pair::dt``; 0 if none), ``n_partners`` the L2 pixels in the
    window, ``x1, y1`` (mm, the shift added), ``x2, y2`` and the slopes ``xp, yp`` = 1000 (x2 - x1) / 30
    mm in mrad (NaN if none), ``tot1, tot2`` the pixel ToTs (tot2 -1 if none).
    ``n_l1``: the L1 candidates before the sample. ``t1``, ``t2``: the sampled
    L1 times and every L2 candidate's, ascending (for `candidate_dt`).
    """

    n_l1: int
    i1: np.ndarray
    i2: np.ndarray
    dt: np.ndarray
    n_partners: np.ndarray
    x1: np.ndarray
    y1: np.ndarray
    x2: np.ndarray
    y2: np.ndarray
    xp: np.ndarray
    yp: np.ndarray
    tot1: np.ndarray
    tot2: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    #: With ``l1l2_pairs(..., dt_half=h)``: what ``candidate_dt(h)`` returns, from the
    #: same search; None otherwise.
    cand_dt: np.ndarray | None = None

    @property
    def n(self) -> int:
        """L1 pixels sampled."""
        return int(self.i1.size)

    @property
    def paired(self) -> np.ndarray:
        return self.i2 >= 0

    def classes(self, light_min: int, light_max: int,
                heavy_min: int) -> tuple[np.ndarray, np.ndarray]:
        """``(light, heavy)`` masks over the sampled L1 pixels: pairs whose two pixel
        ToTs are both within [light_min, light_max], or both >= heavy_min."""
        p = self.paired
        return (p & X.light_class(self.tot1, self.tot2, light_min, light_max),
                p & (self.tot1 >= heavy_min) & (self.tot2 >= heavy_min))

    def candidate_dt(self, half_ns=DT_HALF_NS, max_entries=None) -> np.ndarray:
        """t(L2) - t(L1) of every L2 candidate within +-half_ns (inclusive) of a sampled
        L1 pixel, int64. At most ``max_entries`` (default DT_PER_L1 per sampled L1
        pixel): beyond that an evenly spread subset of the sampled L1 pixels is
        used (`_window_dt`)."""
        h = int(half_ns)
        lo = np.searchsorted(self.t2, self.t1 - h, side="left")
        hi = np.searchsorted(self.t2, self.t1 + h, side="right")
        cap = DT_PER_L1 * self.n if max_entries is None else int(max_entries)
        return _window_dt(self.t1, self.t2, lo, hi, cap)


def _empty(n_l1: int = 0) -> L1L2Pairs:
    zi = np.zeros(0, dtype=np.intp)
    zf = np.zeros(0)
    z16 = np.zeros(0, dtype=np.int16)
    z64 = np.zeros(0, dtype=np.int64)
    return L1L2Pairs(n_l1, zi, zi.copy(), z64, zi.copy(), zf, zf.copy(), zf.copy(), zf.copy(),
                     zf.copy(), zf.copy(), z16, z16.copy(), z64.copy(), z64.copy())


def candidates(px: W.Pixels, pl: X.Placement, plane: int) -> np.ndarray:
    """Indices into ``px`` of the pixels of ``plane`` the pairing may use, ascending in time.

    Those on the sensor (row < 250) of a chip placed on that plane; the reco
    decoder drops the other rows before any pairing.
    """
    idx = px.planes[plane]
    if not idx.size:
        return idx
    good = (px.row[idx] < W.PIXEL_ROWS) & (pl.plane[px.chip[idx]] == plane)
    return idx if good.all() else idx[good]


def plane_candidates(px: W.Pixels, pl: X.Placement) -> tuple[np.ndarray, np.ndarray]:
    """`candidates` of L1 and of L2, computed once for the pairing and the plane maps."""
    return candidates(px, pl, W.PLANE_L1), candidates(px, pl, W.PLANE_L2)


def plane_sample(cands, max_hits=MAX_HITS) -> tuple[np.ndarray, np.ndarray]:
    """Each plane alone, unseeded: an even sample of at most ``max_hits`` candidates a plane.

    ``cands``: `plane_candidates`. Returns ``(idx, n)``: indices into the
    `Pixels`, the sampled L1 pixels first (in time order, `W.even_sample`),
    then the L2 ones; ``n`` = (L1, L2) sampled, so ``idx[:n[0]]`` is L1.
    """
    out = []
    for c in cands:
        if c.size > max_hits:
            c = c[W.even_sample(c.size, int(max_hits))]
        out.append(c)
    return np.concatenate(out), np.array([out[0].size, out[1].size], dtype=np.intp)


def position_luts(pl: X.Placement, shift=(0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """``(xs, ys)``: x in mm of every (chip, column) and y of every (chip, row), shift added.

    Shape (N_CHIP_IDS, 256) each, exactly ``Placement.xy`` plus the shift (the
    same float operations in the same order), so binning a table entry gives
    the bin of the pixel itself; columns and rows are 8-bit. A chip without a
    place has 0 + 0 * col (its pixels are never candidates).
    """
    c = np.arange(W.N_CHIP_IDS)[:, None]
    v = np.arange(256, dtype=np.float64)[None, :]
    xs = pl.x0[c] + pl.sx[c] * v
    ys = pl.y0[c] + pl.sy[c] * v
    xs += float(shift[0])
    ys += float(shift[1])
    return xs, ys


def l1l2_pairs(px: W.Pixels | None, window_ns=WINDOW_NS, max_l1=MAX_L1,
               pl: X.Placement | None = None, shift=(0.0, 0.0),
               t_min: int | None = None, dt_half: int | None = None,
               max_dt: int | None = None, cands=None) -> L1L2Pairs:
    """Pair each sampled L1 pixel with its nearest-in-time L2 pixel (`L1L2Pairs`).

    ``window_ns``: the half window, whole ns, both edges inclusive;
    ``max_l1``: L1 pixels sampled at most (even, in time order); ``pl``: the
    `sma_mupix_xy.Placement` (the default chip lists when None); ``shift``:
    (dx, dy) in mm added to every position (`sma_mupix_xy.stage_shift`; it
    cancels in the slopes); ``t_min``: L1 pixels before it are not sampled
    (the plugin passes the first examined pixel time plus the reach when the
    per-frame pixel cap skipped the earlier words, whose L2 partners are gone).
    ``dt_half``: also collect `L1L2Pairs.cand_dt`, every t(L2) - t(L1) within
    +-dt_half (inclusive) in the same search, at most ``max_dt`` of them
    (default DT_PER_L1 per sampled L1 pixel; an evenly spread subset of the
    sampled L1 pixels beyond that, `_window_dt`). ``cands``: `plane_candidates`
    of ``px`` and ``pl`` when the caller has them already.
    """
    if px is None or px.n == 0:
        return _empty()
    pl = pl or X.placement()
    i1, i2c = cands if cands is not None else plane_candidates(px, pl)
    if t_min is not None and i1.size:
        i1 = i1[int(np.searchsorted(px.t[i1], int(t_min), side="left")):]
    n_l1 = int(i1.size)
    if n_l1 == 0:
        return _empty()
    if n_l1 > max_l1:
        i1 = i1[W.even_sample(n_l1, int(max_l1))]
    m = int(i1.size)
    t1 = px.t[i1]
    t2 = px.t[i2c]
    n2 = int(t2.size)
    dx, dy = float(shift[0]), float(shift[1])
    if n2 == 0:
        x1, y1 = pl.xy(px.chip[i1], px.col[i1], px.row[i1])
        nan = np.full(m, np.nan)
        out = _empty(n_l1)
        out.i1, out.i2, out.dt = i1, np.full(m, -1, dtype=np.intp), np.zeros(m, dtype=np.int64)
        out.n_partners = np.zeros(m, dtype=np.intp)
        out.x1, out.y1 = x1 + dx, y1 + dy
        out.x2, out.y2, out.xp, out.yp = nan, nan.copy(), nan.copy(), nan.copy()
        out.tot1, out.tot2 = px.tot[i1].astype(np.int16), np.full(m, -1, dtype=np.int16)
        out.t1, out.t2 = t1, t2
        if dt_half is not None:
            out.cand_dt = np.zeros(0, dtype=np.int64)
        return out
    w = int(window_ns)
    # One search for every edge (integer ns: t + w inclusive = t + w + 1 exclusive):
    # a = first L2 >= t1 - w, k = first L2 >= t1, b = first L2 > t1 + w, and with
    # dt_half the same for +-dt_half.
    nq = 3 if dt_half is None else 5
    q = np.empty((nq, m), dtype=np.int64)
    np.subtract(t1, w, out=q[0])
    q[1] = t1
    np.add(t1, w + 1, out=q[2])
    if dt_half is not None:
        np.subtract(t1, int(dt_half), out=q[3])
        np.add(t1, int(dt_half) + 1, out=q[4])
    s = np.searchsorted(t2, q.reshape(-1), side="left").reshape(nq, m)
    a, k, b = s[0], s[1], s[2]
    npart = b - a
    # The two neighbours: after = k (the first of its time stamp already);
    # before = the FIRST pixel of the time stamp just before t1 (the C++ keeps
    # the earliest of equal |dt|, so of equal times the first in its order).
    has_a = k < b
    has_b = k > a
    ka = np.minimum(k, n2 - 1)
    tb = t2[np.maximum(k - 1, 0)]
    pb = np.searchsorted(t2, tb, side="left")
    take_b = (t1 - tb) <= (t2[ka] - t1)
    take_b &= has_b
    take_b |= has_b & ~has_a
    paired = npart > 0
    best = np.where(take_b, pb, ka)
    j2 = i2c[best]
    dt = t2[best] - t1
    # Both pixels of every row in one pass: [L1 | partner].
    both = np.concatenate([i1, j2])
    x, y = pl.xy(px.chip[both], px.col[both], px.row[both])
    tot = px.tot[both].astype(np.int16)
    x1, x2, y1, y2 = x[:m], x[m:], y[:m], y[m:]
    k_mrad = 1000.0 / X.LEVER_MM
    xp = x2 - x1
    xp *= k_mrad
    yp = y2 - y1
    yp *= k_mrad
    x += dx
    y += dy
    tot2 = tot[m:]
    i2 = j2
    if not paired.all():
        un = ~paired
        i2 = np.where(paired, j2, -1)
        dt[un] = 0
        x2[un] = np.nan
        y2[un] = np.nan
        xp[un] = np.nan
        yp[un] = np.nan
        tot2[un] = -1
    out = L1L2Pairs(n_l1, i1, i2, dt, npart, x1, y1, x2, y2, xp, yp, tot[:m], tot2, t1, t2)
    if dt_half is not None:
        cap = DT_PER_L1 * m if max_dt is None else int(max_dt)
        out.cand_dt = _window_dt(t1, t2, s[3], s[4], cap)
    return out


def _window_dt(t1, t2, lo, hi, max_dt) -> np.ndarray:
    """t2[j] - t1[i] for every j in [lo[i], hi[i]), at most ``max_dt`` entries.

    Over the cap, every k-th row (k doubled until the rows fit); a single row
    still over it keeps its first ``max_dt`` entries (only a pathological
    burst with a tiny cap). Nothing larger than ``max_dt`` is allocated.
    """
    n = hi - lo
    total = int(n.sum())
    max_dt = max(0, int(max_dt))
    if total > max_dt:
        step = -(-total // max(1, max_dt))
        while True:
            sub = n[::step]
            total = int(sub.sum())
            if total <= max_dt or sub.size <= 1:
                break
            step *= 2
        t1, lo, n = t1[::step], lo[::step], sub
        if total > max_dt:
            n = np.minimum(n, max_dt)
            total = int(n.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    start = np.cumsum(n)
    start -= n
    j = np.repeat(lo - start, n)
    j += np.arange(total)
    out = t2[j]
    out -= np.repeat(t1, n)
    return out
