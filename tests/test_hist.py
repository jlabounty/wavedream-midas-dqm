"""Histogram storage: the u32 default and the u64 counters a busy channel needs.

A u32 bin fed every hit of a high-rate channel wraps within hours and says
nothing when it does, so the u64 path has to be exact on the wire as well as in
memory -- which means past 2**32, not just up to it.
"""

from __future__ import annotations

import numpy as np
import pytest

from mdqm.dqm import framing
from mdqm.dqm.hist import Axis, Hist1D, Hist2D


def test_the_default_stays_u32_on_the_wire():
    """The WaveDREAM histograms must encode exactly as before."""
    h = Hist1D("h", Axis(4, 0.0, 4.0))
    h.fill([0.5, 1.5, 1.5])
    assert h.counts.dtype == np.uint32
    got = framing.decode_histogram(h.encode())
    assert got["type"] == framing.TYPE_1D_U32
    assert got["counts"].dtype == np.uint32

    h2 = Hist2D("h2", Axis(3, 0, 3), Axis(2, 0, 2))
    h2.fill([0.5], [1.5])
    assert framing.decode_histogram(h2.encode())["type"] == framing.TYPE_2D_U32


def test_u64_counts_round_trip_as_f64_past_2_to_the_32():
    h = Hist1D("big", Axis(4, 0.0, 4.0), dtype=np.uint64)
    h.fill([0.5, 2.5])
    h.counts[2] += np.uint64(2**40)            # a count u32 could not hold

    got = framing.decode_histogram(h.encode())
    assert got["type"] == framing.TYPE_1D_F64
    assert got["counts"].dtype == np.float64
    assert int(got["counts"][2]) == 2**40
    assert got["counts"].tolist() == h.counts.astype(np.float64).tolist()
    assert got["entries"] == 2


def test_u64_2d_round_trip_keeps_x_fastest():
    h = Hist2D("big2", Axis(3, 0.0, 3.0), Axis(2, 0.0, 2.0), dtype=np.uint64)
    h.fill([0.5, 2.5], [0.5, 1.5])
    got = framing.decode_histogram(h.encode())
    assert got["type"] == framing.TYPE_2D_F64
    assert got["counts"].shape == (4, 5)
    np.testing.assert_array_equal(got["counts"], h.counts.astype(np.float64))
    assert got["counts"][1, 1] == 1 and got["counts"][2, 3] == 1


def test_the_encoder_never_narrows_a_wide_integer_silently():
    """Handing encode_histogram a u64 array directly must not wrap it to u32."""
    counts = np.array([0, 2**32 + 5, 1], dtype=np.uint64)
    got = framing.decode_histogram(framing.encode_histogram(counts, [(0, 1)], 3))
    assert got["counts"][1] == 2**32 + 5


def test_add_counts_adds_a_pre_binned_array():
    h = Hist1D("pre", Axis(4, 0.0, 4.0), dtype=np.uint64)
    idx = np.array([0, 1, 1, 5, 3])            # under/overflow included
    h.add_counts(np.bincount(idx, minlength=6), entries=5)
    h.add_counts(np.bincount(idx, minlength=6))       # entries from the sum
    assert h.counts.tolist() == [2, 4, 0, 2, 0, 2]
    assert h.entries == 10

    # The same answer as fill() on the equivalent values.
    ref = Hist1D("ref", Axis(4, 0.0, 4.0), dtype=np.uint64)
    ref.fill([-1.0, 0.5, 0.5, 9.0, 2.5] * 2)
    np.testing.assert_array_equal(h.counts, ref.counts)


def test_add_counts_on_2d_and_u32():
    h = Hist2D("pre2", Axis(2, 0, 2), Axis(1, 0, 1))
    add = np.zeros((3, 4), dtype=np.int64)
    add[1, 2] = 7
    h.add_counts(add, entries=7)
    assert h.counts.dtype == np.uint32
    assert h.counts[1, 2] == 7 and h.entries == 7


def test_add_counts_refuses_what_it_cannot_add_honestly():
    h = Hist1D("strict", Axis(4, 0.0, 4.0), dtype=np.uint64)
    with pytest.raises(ValueError, match="shape"):
        h.add_counts(np.zeros(4, dtype=np.int64))          # no under/overflow
    with pytest.raises(TypeError):
        h.add_counts(np.zeros(6, dtype=np.float64))
    with pytest.raises(ValueError, match="negative"):
        h.add_counts(np.array([0, -1, 0, 0, 0, 0]))
    assert h.entries == 0 and not h.counts.any()


def test_only_unsigned_count_types_are_accepted():
    with pytest.raises(ValueError):
        Hist1D("f", Axis(2, 0, 1), dtype=np.float64)


def test_clear_keeps_the_dtype():
    h = Hist1D("c", Axis(2, 0, 1), dtype=np.uint64)
    h.fill([0.2])
    h.clear()
    assert h.counts.dtype == np.uint64 and h.entries == 0


def test_add_counts_refuses_a_count_the_storage_cannot_hold():
    h = Hist1D("small", Axis(1, 0, 1))                    # u32
    with pytest.raises(ValueError, match="uint64"):
        h.add_counts(np.array([0, 2**32, 0], dtype=np.uint64))
    assert h.entries == 0 and not h.counts.any()
    h.add_counts(np.array([0, 2**32 - 1, 0], dtype=np.uint64))
    assert h.counts[1] == 2**32 - 1
