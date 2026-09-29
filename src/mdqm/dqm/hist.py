"""Histogram accumulators.

Plain numpy, no ROOT. Counting histograms with under- and overflow bins, filled
vectorised, and encodable straight to the wire format the browser reads.

Binning follows the usual convention and the one musip's format assumes: index 0
is underflow, 1..n are the real bins, n+1 is overflow. The encoder ships all of
them; the page strips the two ends.

Counts are ``uint32`` unless a histogram asks for ``uint64``. A u32 bin fed
every hit of a busy channel wraps within hours, and wraps silently; u64 bins
go over the wire as f64, exact to 2**53.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mdqm.dqm import framing


def _indices(values: np.ndarray, n: int, lo: float, hi: float) -> tuple[np.ndarray, int]:
    """Bin indices in [0, n+1] for `values`, plus how many were dropped as NaN.

    NaN is dropped rather than binned. Putting it in overflow would be the
    tempting one-liner and it would quietly turn "this quantity could not be
    computed" into "this quantity was very large", which is a lie a monitoring
    plot should not tell. The count is reported instead.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    good = np.isfinite(v)
    dropped = int(v.size - np.count_nonzero(good))
    v = v[good]
    if v.size == 0:
        return np.empty(0, dtype=np.intp), dropped

    width = (hi - lo) / n
    idx = np.floor((v - lo) / width).astype(np.intp) + 1
    np.clip(idx, 0, n + 1, out=idx)
    return idx, dropped


_COUNT_DTYPES = (np.dtype(np.uint32), np.dtype(np.uint64))


def _count_dtype(dtype) -> np.dtype:
    dt = np.dtype(dtype)
    if dt not in _COUNT_DTYPES:
        raise ValueError(f"histogram counts must be uint32 or uint64, got {dt}")
    return dt


def _add_counts(hist, counts, entries) -> None:
    """Add a pre-binned array to `hist`, for plugins that bincount themselves.

    `counts` has the histogram's full shape, under- and overflow included, so a
    plugin that already has bin indices skips the float round trip of `fill`.
    `entries` defaults to the sum of `counts`.
    """
    arr = np.asarray(counts)
    if arr.shape != hist.counts.shape:
        raise ValueError(f"{hist.name}: counts shape {arr.shape} does not match "
                         f"{hist.counts.shape} (under/overflow included)")
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"{hist.name}: add_counts takes integer counts, got {arr.dtype}")
    if np.issubdtype(arr.dtype, np.signedinteger) and arr.size and arr.min() < 0:
        # Cast to unsigned, a negative count becomes an enormous one.
        raise ValueError(f"{hist.name}: negative counts")
    limit = np.iinfo(hist.counts.dtype).max
    if arr.size and int(arr.max()) > limit:
        # The cast would wrap, and a wrapped count is indistinguishable from a
        # real one on the page.
        raise ValueError(f"{hist.name}: a count of {int(arr.max())} does not fit "
                         f"{hist.counts.dtype}; use dtype=np.uint64")
    hist.counts += arr.astype(hist.counts.dtype, copy=False)
    hist.entries += int(arr.sum()) if entries is None else int(entries)


@dataclass
class Axis:
    n: int
    lo: float
    hi: float
    title: str = ""

    def __post_init__(self):
        if self.n < 1:
            raise ValueError(f"axis needs at least one bin, got {self.n}")
        if not self.hi > self.lo:
            raise ValueError(f"axis range must be increasing, got [{self.lo}, {self.hi}]")


@dataclass
class Hist1D:
    """A 1D counting histogram."""

    name: str
    x: Axis
    title: str = ""
    #: Set by the definition; the analyzer clears these at run start.
    clear_on_run_start: bool = True
    #: np.uint32 or np.uint64; see the module docstring.
    dtype: object = np.uint32
    counts: np.ndarray = field(init=False)
    entries: int = field(default=0, init=False)
    dropped: int = field(default=0, init=False)

    def __post_init__(self):
        self.counts = np.zeros(self.x.n + 2, dtype=_count_dtype(self.dtype))

    def fill(self, values) -> None:
        idx, dropped = _indices(values, self.x.n, self.x.lo, self.x.hi)
        self.dropped += dropped
        if idx.size:
            # bincount rather than np.add.at: same result, several times faster,
            # and this is the per-sample path for persistence-style fills.
            self.counts += np.bincount(idx, minlength=self.x.n + 2).astype(self.counts.dtype)
            self.entries += int(idx.size)

    def add_counts(self, counts, entries: int | None = None) -> None:
        _add_counts(self, counts, entries)

    def clear(self) -> None:
        self.counts.fill(0)
        self.entries = 0
        self.dropped = 0

    def encode(self) -> bytes:
        return framing.encode_histogram(self.counts, [(self.x.lo, self.x.hi)], self.entries)

    def metadata(self) -> dict:
        return {
            "name": self.name, "title": self.title, "dimensions": 1,
            "entries": self.entries, "dropped": self.dropped,
            "axes": [{"bins": self.x.n, "lo": self.x.lo, "hi": self.x.hi,
                      "title": self.x.title}],
        }


@dataclass
class Hist2D:
    """A 2D counting histogram. `counts` is (ny+2, nx+2), x fastest."""

    name: str
    x: Axis
    y: Axis
    title: str = ""
    clear_on_run_start: bool = True
    dtype: object = np.uint32
    counts: np.ndarray = field(init=False)
    entries: int = field(default=0, init=False)
    dropped: int = field(default=0, init=False)

    def __post_init__(self):
        self.counts = np.zeros((self.y.n + 2, self.x.n + 2), dtype=_count_dtype(self.dtype))

    def fill(self, xs, ys) -> None:
        xv = np.asarray(xs, dtype=np.float64).ravel()
        yv = np.asarray(ys, dtype=np.float64).ravel()
        if xv.size != yv.size:
            raise ValueError(f"x and y must be the same length, got {xv.size} and {yv.size}")

        good = np.isfinite(xv) & np.isfinite(yv)
        self.dropped += int(xv.size - np.count_nonzero(good))
        xv, yv = xv[good], yv[good]
        if xv.size == 0:
            return

        ix, _ = _indices(xv, self.x.n, self.x.lo, self.x.hi)
        iy, _ = _indices(yv, self.y.n, self.y.lo, self.y.hi)

        # One bincount over the flattened index: the 2D fill is the hot path
        # (persistence puts 1024 samples per channel per event through here), and
        # np.add.at on a 2D array is markedly slower for the same answer.
        nx = self.x.n + 2
        flat = iy * nx + ix
        self.counts += np.bincount(
            flat, minlength=self.counts.size
        ).astype(self.counts.dtype).reshape(self.counts.shape)
        self.entries += int(xv.size)

    def add_counts(self, counts, entries: int | None = None) -> None:
        """`counts` is (ny+2, nx+2), x fastest, like `self.counts`."""
        _add_counts(self, counts, entries)

    def clear(self) -> None:
        self.counts.fill(0)
        self.entries = 0
        self.dropped = 0

    def encode(self) -> bytes:
        return framing.encode_histogram(
            self.counts, [(self.x.lo, self.x.hi), (self.y.lo, self.y.hi)], self.entries)

    def metadata(self) -> dict:
        return {
            "name": self.name, "title": self.title, "dimensions": 2,
            "entries": self.entries, "dropped": self.dropped,
            "axes": [{"bins": self.x.n, "lo": self.x.lo, "hi": self.x.hi, "title": self.x.title},
                     {"bins": self.y.n, "lo": self.y.lo, "hi": self.y.hi, "title": self.y.title}],
        }


class HistStore:
    """The analyzer's histograms, by name.

    Deliberately a plain container that lives *outside* the MIDAS connection, so
    a reconnect after a MIDAS restart does not lose what has been accumulated --
    which is what makes a bounce invisible to whoever is watching the page.
    """

    def __init__(self):
        self._by_name: dict[str, Hist1D | Hist2D] = {}

    def add(self, hist):
        self._by_name[hist.name] = hist
        return hist

    def get(self, name):
        return self._by_name.get(name)

    def names(self) -> list[str]:
        return sorted(self._by_name)

    def remove(self, name) -> None:
        self._by_name.pop(name, None)

    def clear(self, selector: str = "") -> int:
        """Clear everything, one collection, or one histogram. Returns how many.

        `selector` matches musip's semantics: empty clears all, `coll` clears
        every name under that prefix, `coll/name` clears exactly one.
        """
        cleared = 0
        for name, hist in self._by_name.items():
            if selector and name != selector and not name.startswith(selector + "/"):
                continue
            hist.clear()
            cleared += 1
        return cleared

    def clear_for_new_run(self) -> int:
        """Clear only the histograms that asked to be cleared at run start."""
        cleared = 0
        for hist in self._by_name.values():
            if hist.clear_on_run_start:
                hist.clear()
                cleared += 1
        return cleared

    def __len__(self):
        return len(self._by_name)

    def __contains__(self, name):
        return name in self._by_name
