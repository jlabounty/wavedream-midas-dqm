"""``sma_words`` against its references on stored real frames.

The files in ``tests/data`` hold raw H000 words of real readout frames and
what the reference implementations make of them (``generate_sma_golden.py``):
``sma_reader.py`` of psm-analysis for the decode, the time, the RF burst rule
and the S2-S5 pairing; ``PISMAWord.hh`` of reco_testbeam, compiled, for the
fine-vs-coarse comparison, ``TimeNs`` and ``Extent``. Needs numpy and pytest
only.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from mdqm.plugins import sma_words as W

DATA = Path(__file__).resolve().parent / "data"
FILES = ("sma_run00682_frames.npz", "sma_run01008_frame.npz")
S1, RF = 1, 6
OTHERS = {"S2": 2, "S3": 3, "S4": 4, "S5": 5}


def digest(a) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(f"{a.dtype.str}{a.shape}".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _frames():
    out = []
    for name in FILES:
        path = DATA / name
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as z:
            g = {k: z[k] for k in z.files}
        for i, label in enumerate(g["labels"]):
            p = f"f{i}_"
            f = {k[len(p):]: v for k, v in g.items() if k.startswith(p)}
            f["shift"] = int(g["shift"])
            f["label"] = str(label)
            out.append(pytest.param(f, id=f"{name.split('.')[0]}-{i}"))
    return out


FRAMES = _frames()


def test_golden_files_present():
    assert len(FRAMES) == 7, [p.id for p in FRAMES]


def eq_nan(a, b):
    """Exact equality with NaN == NaN."""
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


@pytest.fixture(params=FRAMES)
def frame(request):
    f = request.param
    d = W.decode(f["words"], f["shift"])
    order = np.argsort(d["time"], kind="stable")
    f["_d"] = d
    f["_order"] = order
    f["_ch"] = d["ch"][order]
    f["_t"] = d["time"][order]
    return f


def test_words_from_bank_uint32_view(frame):
    w = frame["words"]
    as_dword = w.view("<u4")            # what midas hands over for a TID_DWORD bank
    assert np.array_equal(W.words_from_bank(as_dword), w)
    assert np.array_equal(W.words_from_bank(as_dword.tobytes()), w)


def test_decode_matches_reference(frame):
    d = frame["_d"]
    for k in ("ch", "tot", "fine", "coarse"):
        assert digest(d[k]) == str(frame[f"{k}_sha256"]), k
    assert np.array_equal(d["time"], frame["time"])
    assert d["time"].dtype == np.int64
    assert d["n_filler"] == int(frame["n_filler"])
    assert d["n_pixel"] == int(frame["n_pixel"])
    nf, npx, ntr = W.word_counts(frame["words"])
    assert (nf, npx, ntr) == (d["n_filler"], d["n_pixel"], d["time"].size)
    assert digest(frame["_order"]) == str(frame["order_sha256"])


def test_time_matches_cpp(frame):
    # PISMAWord::TimeNs masks to 2^min(28 + shift, 40); equal to time_of at shift 14.
    assert digest(frame["_d"]["time"].astype(np.uint64)) == str(frame["cpp_time_sha256"])


def test_fine_coarse_matches_cpp(frame):
    d = frame["_d"]
    diff = W.fine_coarse_diff_ns(d["coarse"], d["fine"], frame["shift"])
    xor = W.fine_coarse_xor(d["coarse"], d["fine"], frame["shift"])
    assert np.array_equal(diff, frame["cpp_diff"])
    assert np.array_equal(xor, frame["cpp_xor"])


def test_extent_matches_cpp(frame):
    e = W.extent(frame["_d"]["time"])
    assert (e.first, e.last) == tuple(int(x) for x in frame["cpp_extent"])


@pytest.mark.parametrize("rule", ["last", "dqm"])
def test_burst_phase_matches_reference(frame, rule):
    ch, t = frame["_ch"], frame["_t"]
    n, valid, phase, period, lo, pulse_t = W.burst_phase_many(t[ch == RF], t[ch == S1], rule=rule)
    p = f"bpm_{rule}_"
    assert np.array_equal(n, frame[p + "n"])
    assert np.array_equal(valid, frame[p + "valid"])
    assert np.array_equal(lo, frame[p + "lo"])
    for k, v in (("phase", phase), ("period", period), ("pulse_t", pulse_t)):
        assert eq_nan(v, frame[p + k]), k


def test_gate_veto_matches_reference(frame):
    ch, t = frame["_ch"], frame["_t"]
    vetoed, gap = W.gate_veto(t[ch == S1])
    assert np.array_equal(vetoed, frame["veto"])
    assert eq_nan(gap, frame["veto_gap"])


@pytest.mark.parametrize("rule", ["last", "dqm"])
def test_pairing_matches_reference(frame, rule):
    """S2-S5 borrow the phase of the nearest valid S1 gate (``pair_frame``)."""
    ch, t = frame["_ch"], frame["_t"]
    rf, g = t[ch == RF], t[ch == S1]
    n, valid, _ph, _pe, lo, pulse_t = W.burst_phase_many(rf, g, rule=rule)
    assert eq_nan(W.pulse_offset(rf, g, n, lo, 0), frame["off0"])
    for name, c in OTHERS.items():
        _idx, dt, phase = W.match_to_gates(t[ch == c], g[valid], pulse_t[valid])
        assert eq_nan(dt, frame[f"pair_{rule}_{name}_dt"]), name
        assert eq_nan(phase, frame[f"pair_{rule}_{name}_phase"]), name


def test_prepare_and_analyse_run(frame):
    """The composite path runs and agrees with the primitives on the kept hits."""
    fr = W.prepare_frame(frame["words"], frame["shift"])
    assert fr.n_trigger == frame["_d"]["time"].size
    assert fr.keep.sum() + fr.stale_per_ch.sum() == fr.n_trigger
    assert np.all(np.diff(fr.s_t) >= 0)
    for c in range(W.N_CHANNELS):
        assert np.all(fr.s_ch[fr.chan[c]] == c)
    a = W.analyse_frame(fr)
    if fr.keep.all():
        # nothing stale: the RF phase is the reference's on the whole frame
        assert eq_nan(a.rf_phase, frame["bpm_last_phase"])
        assert np.array_equal(a.rf_valid, frame["bpm_last_valid"])


def _by_frame(fname, i):
    return next(p.values[0] for p in FRAMES
                if p.values[0]["label"].startswith(f"{fname}#{i} "))


@pytest.mark.parametrize("i, n_stale, n_kept, span_ms", [
    # The first frame of a subrun replays the previous run's buffer: 185 words
    # in three far clusters (324 s, 523 s, 1072 s) around a 0.5 ms main cluster.
    # That cluster is kept, but it is stale too: at shift 14 only 1 % of its
    # words are fine/coarse consistent, so the plugin must also judge a frame
    # by its consistency, not by the cluster rule alone.
    (0, 185, 3815, 0.518),
    # 29 early hits of the run start, 2.2 s spread, then the 12 ms frame.
    (2, 29, 1715, 11.954),
    # An ordinary 26 ms frame: nothing stale.
    (3, 0, 3696, 26.463),
])
def test_stale_real_frames(i, n_stale, n_kept, span_ms):
    f = _by_frame("run00682_00000.mid.lz4", i)
    fr = W.prepare_frame(f["words"], f["shift"])
    assert int(fr.stale_per_ch.sum()) == n_stale
    assert int(fr.keep.sum()) == n_kept
    assert fr.span_ns / 1e6 == pytest.approx(span_ms, abs=1e-3)
