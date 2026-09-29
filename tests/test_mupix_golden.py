"""``sma_words.pixel_decode`` against its references on stored real pixel words.

``tests/data/mupix_golden.npz`` (``generate_mupix_golden.py``) holds, for the
real frames of ``sma_run00682_frames.npz`` and ``sma_run01008_frame.npz``, what
``PIMuPixWord.hh`` (compiled) and the psm-analysis Python (``mupix_phase.
pixel_words``, ``timewalk_lib.pixel_tot``) make of every pixel word: chip,
column, row, TS2, the time stamp and the ToT at TS2 shifts 5 and 4. Needs
numpy and pytest only.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from mdqm.plugins import sma_words as W

DATA = Path(__file__).resolve().parent / "data"
GOLDEN = DATA / "mupix_golden.npz"


def digest(a) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(f"{a.dtype.str}{a.shape}".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _frames():
    with np.load(GOLDEN, allow_pickle=False) as z:
        g = {k: z[k] for k in z.files}
    words = {}
    out = []
    for i, label in enumerate(g["labels"]):
        p = f"f{i}_"
        f = {k[len(p):]: v for k, v in g.items() if k.startswith(p)}
        src = str(f["source"])
        if src not in words:
            with np.load(DATA / src) as z:
                words[src] = [z[f"f{j}_words"] for j in range(len(z["labels"]))]
        f["words"] = words[src][int(f["index"])]
        f["label"] = str(label)
        f["shifts"] = tuple(int(x) for x in g["ts2_shifts"])
        out.append(pytest.param(f, id=f"{src.split('.')[0]}-{int(f['index'])}"))
    return out


FRAMES = _frames()


def test_every_golden_frame_is_there_with_pixels():
    assert len(FRAMES) == 7
    assert sum(int(p.values[0]["n_pixel"]) for p in FRAMES) > 9000


@pytest.fixture(params=FRAMES)
def frame(request):
    return request.param


def _check(name, got, f):
    assert digest(got) == str(f[f"{name}_sha256"]), (name, got[:10], f[f"{name}_head"][:10])
    head = f[f"{name}_head"]
    assert np.array_equal(np.asarray(got)[:head.size], head), name


def test_pixel_fields_match_the_references(frame):
    d = W.pixel_decode(frame["words"], ts2_shift=5)
    assert d["n_words"] == int(frame["n_pixel"]) == d["chip"].size
    _check("word_index", d["word_index"], frame)
    for k in ("chip", "col", "row", "ts2"):
        assert d[k].dtype == np.uint8
        _check(k, d[k], frame)
    _check("tick", d["tick"], frame)
    _check("time", d["time"], frame)
    assert np.all(d["time"] < W.TIME_WRAP_NS), "the same 2^40 ns epoch as the SMA time"


@pytest.mark.parametrize("shift_i", [0, 1])
def test_tot_from_ts2_matches_the_references(frame, shift_i):
    s = frame["shifts"][shift_i]
    d = W.pixel_decode(frame["words"], ts2_shift=s)
    _check(f"tot{s}", d["tot"], frame)
    assert np.array_equal(W.pixel_tot(d["ts2"], d["tick"], s), d["tot"])


def test_the_raw_word_is_given_back(frame):
    d = W.pixel_decode(frame["words"])
    assert np.array_equal(d["raw"], frame["words"][d["word_index"]])
    again = W.encode_pixel(d["chip"], d["col"], d["row"], d["ts2"], d["tick"])
    assert np.array_equal(again, d["raw"]), "every pixel bit is a field: re-encoding is exact"


def test_the_word_counts_agree_with_the_trigger_decode(frame):
    d = W.decode(frame["words"])
    assert W.pixel_decode(frame["words"])["n_words"] == d["n_pixel"]


def test_hand_checked_words():
    # 0x0c3ad2c0a1b2c3d4: chip 3, col 14, row 180, TS2 22, time stamp 0x00a1b2c3d4.
    w = 0x0C3AD2C0A1B2C3D4
    d = W.pixel_decode(np.array([w, W.FILLER, 1 << 63], dtype=np.uint64))
    assert d["n_words"] == 1, "the filler and a trigger word are not pixels"
    fields = (int(d["chip"][0]), int(d["col"][0]), int(d["row"][0]), int(d["ts2"][0]))
    assert fields == (3, 14, 180, 22)
    assert int(d["tick"][0]) == 0x00A1B2C3D4 and int(d["time"][0]) == 0x00A1B2C3D4 * 8
    # ToT = (22 - ((0xa1b2c3d4 >> 5) & 31)) & 31
    assert int(d["tot"][0]) == (22 - ((0xA1B2C3D4 >> 5) & 31)) & 31
