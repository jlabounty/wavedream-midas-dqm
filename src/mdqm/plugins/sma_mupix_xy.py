"""MuPix positions and S1-seeded tracks for the SMA DQM: pixel -> mm, one cluster per plane.

Pure functions over `sma_words.Pixels`, with no plugin state (the plugin,
``sma.py``, books and fills; the offline CLI runs the same code). This module
imports `sma_words`; `sma_words` never imports it.

Frame
-----
The reco geometry, tag ``bt2026-v4`` of ``bt2026_psm_geometry.json``: +z
downstream, +y up, **+x beam-left** (a map drawn with x to the right is the view
looking upstream). Each plane is a 2 x 2 quad of 256 x 250-pixel chips at
0.080 mm pitch, chip centres at x = +-10.40, y = +-10.16 mm (edge to edge plus
the PROVISIONAL 0.32 mm quad gap, two constants below), half lengths 10.24 x
10.0 mm; L1 at z = 0, L2 at z = 30 mm. Quadrants q0 beam-left bottom, q1
beam-right bottom, q2 beam-left top, q3 beam-right top; q2 and q3 are mounted
turned by pi, so their columns run towards -x and their rows towards -y. A
pixel (col, row) of a chip sits at

    x = cx + s ((col + 0.5) p - hx),    y = cy + s ((row + 0.5) p - hy)

with s = +1 (rot 0) or -1 (rot pi), as ``PITMidasMusip`` places it and as
``psm-analysis-josh-2026/mupix-pane-weights/pane_weights/geometry.py``
computes it (which uses hx as float32, 2.3e-7 mm off; nothing here is that
close to a bin edge). A chip's quadrant is its position in the plane's chip
list (``/DQM/SMA/MuPix/L1 chips``, ``L2 chips``): the default 0-3 / 4-7 is the
identity map of runs >= 200, chip 0 = L1 q0 ... chip 7 = L2 q3. For x/y each
list must have exactly four entries, q0..q3, distinct chip ids or -1 for an
empty quadrant (`check_chip_lists`); a shorter or reordered list would put
chips silently in the wrong place, so the plugin turns x/y off instead.

The XY table moves the whole telescope; ``stage_shift`` turns its reading into
the shift added to every position, (-xpos, +ypos) (the stage's +x is
beam-right; ``PIPSMIselParams`` of reco_testbeam).

Tracks
------
Per S1 row (`s1_tracks`): the pixels of each plane in [t + lo, t + hi) (raw
times, no time-walk correction; rows >= 250 and chips without a place
dropped). A plane is **accepted** when all of them lie on one chip within a
``box`` x ``box`` pixel square; its position is the mean of their pixel
centres, its ToT the largest pixel ToT. A track is an S1 row with both planes
accepted; the slopes are 1000 (x2 - x1) / 30 mm in mrad. Two particles, a
burst or a noisy neighbour fail the square and the row is "ambiguous". The
square is not reco's 0.12 mm single linkage: they agree on touching clusters
inside 3 x 3, but the square also accepts non-touching pixels inside it ((0, 0)
+ (2, 2)) and rejects a 4-in-a-row cluster that the linkage keeps. A pixel that
fires twice in one window counts twice in the mean (rare; not removed, it would
need a per-window de-duplication).

Everything is vectorised, with no Python loop over hits or rows, and both
planes go through one pass: the windows are merged into runs, only the pixels
of the runs are gathered (O(pixels in windows), not O(pixels in the frame)),
and the per-window min / max / sum come from ``ufunc.reduceat`` over segments.
The cost is mostly per numpy call: ~0.25 ms for 250 S1 rows on a dense frame,
~0.4 ms with the fills (``sma.SmaPlugin._fill_xy``). Times stay int64 (exact);
a window is half open on whole ns, so [lo, hi) is [lo, hi - 1] for the
searchsorted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mdqm.plugins import sma_words as W

GEOMETRY_TAG = "bt2026-v4"
PITCH_MM = 0.080
#: Half lengths of a chip's active area (256 x 0.08, 250 x 0.08 over 2).
CHIP_HX_MM = 10.24
CHIP_HY_MM = 10.0
#: |x|, |y| of the chip centres: the half length plus half the quad gap. The
#: gap (0.32 mm, both axes) is PROVISIONAL; a re-measurement changes these two.
QUAD_CX_MM = 10.40
QUAD_CY_MM = 10.16
PLANE_Z_MM = {W.PLANE_L1: 0.0, W.PLANE_L2: 30.0}
LEVER_MM = PLANE_Z_MM[W.PLANE_L2] - PLANE_Z_MM[W.PLANE_L1]
#: Per quadrant q0..q3: (sign of cx, sign of cy, s). q2, q3 are turned by pi.
QUADRANTS = ((+1, -1, +1), (-1, -1, +1), (+1, +1, -1), (-1, +1, -1))
QUADRANT_NAMES = ("beam-left bottom", "beam-right bottom", "beam-left top", "beam-right top")

#: Track states, in the order of the ``mupix_track_state`` bins; NOT_JUDGED
#: rows (window outside the pixel data) are in no bin.
NOT_JUDGED, NO_L1, NO_L2, AMBIGUOUS, TRACK = -1, 0, 1, 2, 3
STATE_NAMES = ("no L1", "no L2", "ambiguous", "track")

#: Defaults of ``/DQM/SMA/MuPix/XY``.
CLUSTER_BOX_PX = 3
TOT_LIGHT_MAX = 9
TOT_HEAVY_MIN = 13
#: S1 rows per frame given to the track finder (``XY/max S1 per frame``):
#: half the MuPix sample, which saves ~0.1 ms of a dense frame's ~0.5 ms.
#: Always a cap, within 1..XY_MAX_S1_LIMIT (~1 us per row).
XY_MAX_S1 = 250
XY_MAX_S1_LIMIT = 2000
#: Chip slots per plane list, one per quadrant; -1 = an empty quadrant.
N_QUADRANTS = 4
EMPTY_SLOT = -1
#: The largest square accepted: the chip test needs it well below the
#: 1024 per chip of the packed keys below.
MAX_BOX_PX = 64
#: ``chip * KEY + col`` keeps one min / max per plane for "one chip and a
#: narrow column range" together: two chips are >= KEY - 255 apart.
_KEY = 1024

#: The histogram axes, the nearline's (PIPSMMuPixMonitor): positions in 130
#: bins of 0.64 mm (8 pixels) over +-41.6 mm plus a quarter pixel, so that no
#: pixel centre (nor a two-pixel mean) sits on an edge (track_xy_expanded at
#: half its 260 bins); slopes one bin per step of one pixel over the lever
#: arm, 1000 * 0.08 / 30 = 2.667 mrad, 77 bins centred on 0 (xxp_central).
#: Kept identical to the nearline on purpose, with its bias: a 2-pixel cluster
#: in one plane and 1 pixel in the other gives a half-step slope, which sits
#: exactly on a bin edge, and float rounding picks the bin (mostly the lower:
#: ~3-5 % of the tracks, about -0.05 mrad on the mean, a faint comb).
POS_BINS = 130
POS_LO_MM = -41.6 + 0.25 * PITCH_MM
POS_HI_MM = 41.6 + 0.25 * PITCH_MM
SLOPE_STEP_MRAD = 1000.0 * PITCH_MM / LEVER_MM
SLOPE_BINS = 77
SLOPE_HALF_MRAD = 0.5 * SLOPE_BINS * SLOPE_STEP_MRAD

#: The ODB key the stage is read from, [x, y] in mm.
STAGE_PATH = "/Equipment/XYTable/Variables/Measured"


@dataclass(frozen=True)
class Placement:
    """Where each chip id's pixels are: built once per settings change (`placement`).

    Per chip id (32 entries): ``plane`` (0 = no place), ``quadrant`` (-1),
    and the affine map x = x0 + sx col, y = y0 + sy row (mm; 0 where no place).
    """

    plane: np.ndarray       # uint8[32]
    quadrant: np.ndarray    # int8[32]
    x0: np.ndarray          # float64[32]
    sx: np.ndarray
    y0: np.ndarray
    sy: np.ndarray
    #: Chips of the lists past their plane's fourth: no place, left out
    #: (never with lists that pass `check_chip_lists`).
    unplaced: tuple = ()

    def xy(self, chip, col, row) -> tuple[np.ndarray, np.ndarray]:
        """Pixel-centre (x, y) in mm of pixels on placed chips (no stage shift)."""
        c = np.asarray(chip, dtype=np.intp)
        return (self.x0[c] + self.sx[c] * np.asarray(col, dtype=np.float64),
                self.y0[c] + self.sy[c] * np.asarray(row, dtype=np.float64))

    def quadrant_map(self) -> list[dict]:
        """Every placed chip: ``{"chip", "plane", "quadrant", "where"}`` (plane, then quadrant)."""
        out = [{"chip": int(c), "plane": W.PLANE_NAMES[int(self.plane[c])],
                "quadrant": int(self.quadrant[c]), "where": QUADRANT_NAMES[int(self.quadrant[c])]}
               for c in np.flatnonzero(self.plane > 0)]
        return sorted(out, key=lambda d: (d["plane"], d["quadrant"]))


def placement(l1=W.L1_CHIPS, l2=W.L2_CHIPS) -> Placement:
    """The `Placement` of the plane lists: list position = quadrant, -1 = an empty one."""
    n = W.N_CHIP_IDS
    plane = np.zeros(n, dtype=np.uint8)
    quad = np.full(n, -1, dtype=np.int8)
    x0, sx, y0, sy = (np.zeros(n) for _ in range(4))
    unplaced = []
    for p, chips in ((W.PLANE_L1, l1), (W.PLANE_L2, l2)):
        for q, c in enumerate(chips):
            c = int(c)
            if c == EMPTY_SLOT:
                continue
            if q >= len(QUADRANTS):
                unplaced.append(c)
                continue
            gx, gy, s = QUADRANTS[q]
            plane[c], quad[c] = p, q
            # x = cx + s ((col + 0.5) p - hx) = [cx + s (0.5 p - hx)] + [s p] col
            x0[c] = gx * QUAD_CX_MM + s * (0.5 * PITCH_MM - CHIP_HX_MM)
            y0[c] = gy * QUAD_CY_MM + s * (0.5 * PITCH_MM - CHIP_HY_MM)
            sx[c] = sy[c] = s * PITCH_MM
    return Placement(plane=plane, quadrant=quad, x0=x0, sx=sx, y0=y0, sy=sy,
                     unplaced=tuple(unplaced))


def check_chip_lists(l1, l2) -> str | None:
    """Why the chip lists cannot place the chips for x/y, or None when they can.

    Each list: exactly four entries (q0..q3), chip ids 0-31 or -1 for an empty
    quadrant, no id twice (in either list).
    """
    for name, chips in (("L1 chips", l1), ("L2 chips", l2)):
        chips = list(chips)
        if len(chips) != N_QUADRANTS:
            return (f"MuPix/{name} has {len(chips)} entries; x/y needs exactly "
                    f"{N_QUADRANTS} in quadrant order (beam-left bottom, beam-right bottom, "
                    "beam-left top, beam-right top), -1 for an empty quadrant")
        bad = [c for c in chips if not (c == EMPTY_SLOT or 0 <= int(c) < W.N_CHIP_IDS)]
        if bad:
            return f"MuPix/{name} has {bad}: not a chip id 0-31 or -1"
    ids = [int(c) for c in (*l1, *l2) if int(c) != EMPTY_SLOT]
    dup = sorted({c for c in ids if ids.count(c) > 1})
    if dup:
        return f"MuPix/L1 chips and L2 chips list chip(s) {dup} more than once"
    return None


def stage_shift(xpos, ypos) -> tuple[float, float]:
    """The shift (dx, dy) in mm added to every position for XY-table reading (xpos, ypos).

    The stage's +x is beam-right, the frame's +x beam-left, so x enters negated
    (``PIPSMIselParams::Transform`` of reco_testbeam).
    """
    return -float(xpos), float(ypos)


@dataclass
class XYSettings:
    """The parsed ``/DQM/SMA/MuPix/XY``; ``placement`` follows the MuPix chip lists."""

    enable: bool = True
    box: int = CLUSTER_BOX_PX
    tot_light_max: int = TOT_LIGHT_MAX
    tot_heavy_min: int = TOT_HEAVY_MIN
    apply_stage: bool = True
    #: S1 rows per frame given to the track finder, from the MuPix sample (1..2000).
    max_s1: int = XY_MAX_S1
    placement: Placement = field(default_factory=placement)
    #: Why x/y is off with enable = y (bad chip lists, the MuPix analysis off); None when on.
    off_reason: str | None = None

    @property
    def active(self) -> bool:
        return self.enable and self.off_reason is None

    def reset_key(self) -> tuple:
        """What, when it changes, resets every x/y map (the ToT cuts reset only the classes)."""
        p = self.placement
        return (self.active, self.box, self.apply_stage, p.quadrant.tobytes(), p.plane.tobytes())


@dataclass
class S1Tracks:
    """Per S1 row of `s1_tracks`: the state and, where accepted, the plane positions.

    ``x1, y1`` (L1), ``x2, y2`` (L2) in mm, NaN where that plane was not
    accepted; ``xp, yp`` in mrad, NaN unless a track. ``tot1, tot2`` the
    largest pixel ToT of each plane's window, -1 without pixels. ``hits[p]``
    (p = 1, 2): (x, y) of every pixel of plane p in at least one judged window,
    each pixel once.
    """

    state: np.ndarray       # int8
    x1: np.ndarray
    y1: np.ndarray
    x2: np.ndarray
    y2: np.ndarray
    xp: np.ndarray
    yp: np.ndarray
    tot1: np.ndarray        # int16
    tot2: np.ndarray
    hits: dict = field(default_factory=dict)

    @property
    def judged(self) -> np.ndarray:
        return self.state >= 0

    @property
    def track(self) -> np.ndarray:
        return self.state == TRACK

    def state_counts(self) -> np.ndarray:
        """int64[4]: rows per state (NO_L1, NO_L2, AMBIGUOUS, TRACK), judged rows only."""
        s = self.state[self.state >= 0].astype(np.intp)
        return np.bincount(s, minlength=len(STATE_NAMES)).astype(np.int64)

    def classes(self, light_max: int, heavy_min: int) -> tuple[np.ndarray, np.ndarray]:
        """``(light, heavy)`` masks: tracks with both planes' max ToT <= light_max,
        or both >= heavy_min."""
        t = self.track
        return (t & (self.tot1 <= light_max) & (self.tot2 <= light_max),
                t & (self.tot1 >= heavy_min) & (self.tot2 >= heavy_min))


def _empty_tracks(m: int) -> S1Tracks:
    nan = np.full(m, np.nan)
    e = (np.zeros(0), np.zeros(0))
    return S1Tracks(np.full(m, NOT_JUDGED, np.int8), nan, nan.copy(), nan.copy(), nan.copy(),
                    nan.copy(), nan.copy(), np.full(m, -1, np.int16), np.full(m, -1, np.int16),
                    {W.PLANE_L1: e, W.PLANE_L2: e})


def s1_tracks(t_s1, px: W.Pixels | None, window_ns=W.MUPIX_WINDOW_NS, box=CLUSTER_BOX_PX,
              pl: Placement | None = None, shift=(0.0, 0.0)) -> S1Tracks:
    """Per S1 row: the state, L1 / L2 positions, slopes and max ToT (`S1Tracks`).

    ``t_s1`` ascending int64 (the plugin's MuPix S1 sample); ``window_ns`` (lo,
    hi) half open, relative to S1, on raw times; ``box`` the square in pixels;
    ``pl`` the `Placement` (default chip lists when None); ``shift`` (dx, dy) in
    mm added to every position (`stage_shift`), which cancels in the slopes.
    Only rows whose window lies inside the examined pixel data are judged
    (`sma_words.mupix_coverage`); the rest are NOT_JUDGED.

    States: NO_L1 (no L1 pixel), else NO_L2, else AMBIGUOUS (a plane fails
    the one-chip square), else TRACK.

    Both planes go through one pass (the cost is mostly per numpy call, not
    per element): plane 2's pixels follow plane 1's in one index, and the 2m
    windows (m rows of L1, then m of L2) are segments of it.
    """
    t = np.asarray(t_s1, dtype=np.int64)
    m = t.size
    if px is None or px.n == 0 or m == 0:
        return _empty_tracks(m)
    pl = pl or placement()
    lo, hi = int(window_ns[0]), int(window_ns[1])
    judged = W.mupix_coverage(t, px, lo, hi)
    i1, i2 = px.planes[W.PLANE_L1], px.planes[W.PLANE_L2]
    n1 = i1.size
    if n1 + i2.size == 0 or not judged.any():
        out = _empty_tracks(m)
        out.state[judged] = NO_L1
        return out
    # Per row and plane, the window [a, b) in the joint index of the plane
    # pixels, L1's first (integer ns: [lo, hi) is [lo, hi - 1]); rows not
    # judged get an empty window. Both a and b ascend: t_s1 does.
    a = np.empty(2 * m, dtype=np.intp)
    b = np.empty(2 * m, dtype=np.intp)
    a[:m], b[:m] = W._bounds(t, px.t[i1], lo, hi - 1)
    a[m:], b[m:] = W._bounds(t, px.t[i2], lo, hi - 1)
    a[m:] += n1
    b[m:] += n1
    if not judged.all():
        jj = np.concatenate([judged, judged])
        a = np.where(jj, a, b)
    # The pixels in at least one window, without touching the others: the
    # non-empty windows, both ends ascending, merge into runs where one
    # starts before the previous ends; the runs' pixels are gathered (O(window
    # pixels), not O(frame pixels)) and each window re-indexed into them.
    ne = np.flatnonzero(b > a)
    if ne.size == 0:
        out = _empty_tracks(m)
        out.state[judged] = NO_L1
        return out
    ae, be = a[ne], b[ne]
    start = np.ones(ne.size, dtype=bool)
    np.greater_equal(ae[1:], be[:-1], out=start[1:])
    last = np.ones(ne.size, dtype=bool)
    last[:-1] = start[1:]
    gs, ln = ae[start], be[last] - ae[start]
    off = np.cumsum(ln) - ln
    pos = np.repeat(gs - off, ln) + np.arange(int(ln.sum()))
    gid = np.cumsum(start) - 1
    shift_ = off[gid] - gs[gid]
    a = np.zeros(2 * m, dtype=np.intp)
    b = np.zeros(2 * m, dtype=np.intp)
    a[ne], b[ne] = ae + shift_, be + shift_
    k1 = int(np.searchsorted(pos, n1))
    sel = np.concatenate([i1[pos[:k1]], i2[pos[k1:] - n1]])
    # Of those, the pixels on the sensor (row < 250) of a placed chip of the
    # right plane: rows >= 250 are not pixels, a chip past its list's fourth
    # has no place. Dropping any re-indexes the windows once more.
    chip = px.chip[sel]
    row = px.row[sel]
    want = np.full(sel.size, W.PLANE_L1, dtype=np.uint8)
    want[k1:] = W.PLANE_L2
    good = (row < W.PIXEL_ROWS) & (pl.plane[chip] == want)
    if not good.all():
        cum = np.zeros(sel.size + 1, dtype=np.intp)
        np.cumsum(good, out=cum[1:])
        a, b, k1 = cum[a], cum[b], int(cum[k1])
        sel, chip, row = sel[good], chip[good], row[good]
    if sel.size == 0:
        out = _empty_tracks(m)
        out.state[judged] = NO_L1
        return out
    n = b - a
    has = n > 0

    chip = chip.astype(np.int32)
    col = px.col[sel].astype(np.int32)
    row = row.astype(np.int32)
    # chip * KEY + col (row): one chip within the box <=> a narrow range of
    # the key, as two chips' keys are >= KEY - 255 apart. One sentinel element
    # so that b == len is a valid reduceat index; the segments (a_i, b_i) are
    # at even positions, the odd ones are discarded, and an empty segment
    # gives some element, masked by `has`.
    ns = sel.size
    u = np.empty(ns + 1, dtype=np.int64)
    v = np.empty(ns + 1, dtype=np.int64)
    np.multiply(chip, _KEY, out=u[:-1])
    u[:-1] += col
    np.multiply(chip, _KEY, out=v[:-1])
    v[:-1] += row
    u[-1] = v[-1] = 0
    tot = np.zeros(ns + 1, dtype=np.int16)
    tot[:-1] = px.tot[sel]
    seg = np.empty(4 * m, dtype=np.intp)
    seg[0::2], seg[1::2] = a, b
    umin = np.minimum.reduceat(u, seg)[0::2]
    umax = np.maximum.reduceat(u, seg)[0::2]
    vmin = np.minimum.reduceat(v, seg)[0::2]
    vmax = np.maximum.reduceat(v, seg)[0::2]
    tmax = np.maximum.reduceat(tot, seg)[0::2]
    if ns < 1 << 16:
        # Both sums in one: u, v < 2^15 and fewer than 2^16 pixels, so the sum
        # of v stays below 2^31 and the sum of u * 2^32 below 2^63, with no
        # carry between them -- exact in int64.
        uv = np.add.reduceat((u << 32) | v, seg)[0::2]
        usum, vsum = uv >> 32, uv & 0xFFFFFFFF
    else:                                   # only with a raised max pixel hits per frame
        usum, vsum = np.add.reduceat(u, seg)[0::2], np.add.reduceat(v, seg)[0::2]

    ok = has & (umax - umin < box) & (vmax - vmin < box)
    x = np.full(2 * m, np.nan)
    y = np.full(2 * m, np.nan)
    if ok.any():
        c = umin[ok] // _KEY
        nn = n[ok].astype(np.float64)
        x[ok] = pl.x0[c] + pl.sx[c] * (usum[ok] / nn - c * _KEY)
        y[ok] = pl.y0[c] + pl.sy[c] * (vsum[ok] / nn - c * _KEY)
    tmax = np.where(has, tmax, np.int16(-1)).astype(np.int16)

    has1, has2, ok1, ok2 = has[:m], has[m:], ok[:m], ok[m:]
    state = np.full(m, NOT_JUDGED, dtype=np.int8)
    state[judged] = TRACK
    state[judged & ~(ok1 & ok2)] = AMBIGUOUS
    state[judged & ~has2] = NO_L2
    state[judged & ~has1] = NO_L1
    trk = state == TRACK
    dx, dy = float(shift[0]), float(shift[1])
    k = 1000.0 / LEVER_MM
    xp = np.where(trk, (x[m:] - x[:m]) * k, np.nan)
    yp = np.where(trk, (y[m:] - y[:m]) * k, np.nan)
    x += dx
    y += dy
    hx, hy = pl.xy(chip, col, row)
    hx += dx
    hy += dy
    return S1Tracks(state, x[:m], y[:m], x[m:], y[m:], xp, yp, tmax[:m], tmax[m:],
                    {W.PLANE_L1: (hx[:k1], hy[:k1]), W.PLANE_L2: (hx[k1:], hy[k1:])})
