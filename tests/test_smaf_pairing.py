"""smaf v3: the per-hit TOT + NIM pairing on the event-display frame.

The layout (framing.py, "Version 3"), its round trip with and without words
and pixels, and what the plugin puts in it for a run-1015-cabled frame: the
partner index among the shipped hits, the packed class byte, and the roles
block in the meta. Frames without NIM copies must stay v1/v2 byte for byte
(tests/test_sma_seed_choice.py holds the hashes).
"""

from __future__ import annotations

import numpy as np
import pytest
from sma_layouts import NIM_1015, dense_1015, old_layout

from mdqm.dqm import framing
from mdqm.dqm.hist import HistStore
from mdqm.plugins import sma as P
from mdqm.plugins import sma_nim as N


class _Event:
    def __init__(self, words, serial=0):
        self.header = type("H", (), {"event_id": 301, "serial_number": serial,
                                     "timestamp": 0, "trigger_mask": 0})()
        self.banks = {"H000": type("B", (), {"data": np.asarray(words, "<u8").view("<u4")})()}


T = [0, 2, 400, 900, 902]
CH = [7, 10, 7, 11, 4]
TOT = [30, 10, 200, 10, 40]
FL = [framing.HIT_IN_SEED] * 5
PAIR = [1, 0, -1, -1, -1]
CLS = [N.PAIRED, N.PAIRED | framing.PAIR_NIM_SIDE, N.ECHO_WORD,
       N.NIM_ONLY | framing.PAIR_LAG_HELD | framing.PAIR_NIM_SIDE, N.TOT_ONLY]
PIX = {"t_rel": [0, 8], "time_shift": 0, "chip": [0, 5], "col": [1, 2], "row": [3, 4],
       "tot": [5, 6], "flags": [1, 2]}


@pytest.mark.parametrize("words", [False, True])
@pytest.mark.parametrize("pixels", [False, True])
def test_v3_round_trips_with_and_without_words_and_pixels(words, pixels):
    kw = {"raw_words": [0x87 << 56 | k for k in range(5)], "word_index": [10, 11, 12, 13, 14]} \
        if words else {}
    # The partners travel with the words only (the live raster needs the classes alone).
    b = framing.encode_sma_frame({"view": "seeded"}, T, CH, TOT, FL, frame_seq=5, run_number=1015,
                                 pair=PAIR if words else None, cls=CLS,
                                 pixels=PIX if pixels else None, **kw)
    d = framing.decode_sma_frame(b)
    assert d["version"] == framing.SMAF_VERSION_PAIRING and d["pairing"]
    assert bool(d["flags"] & framing.SMAF_PAIRING) and d["words"] == words
    assert d["cls"].tolist() == CLS
    assert (d["pair"].tolist() if words else d["pair"]) == (PAIR if words else None)
    assert d["t_rel_ns"].tolist() == T and d["ch"].tolist() == CH and d["tot"].tolist() == TOT
    assert framing.smaf_bytes_per_hit(d["flags"]) == (24 if words else 8)
    if pixels:
        assert d["pixels"]["n"] == 2 and d["pixels"]["chip"].tolist() == [0, 5]
        # Dropping the block and rewriting the meta keep the pairing arrays.
        bare = framing.decode_sma_frame(framing.smaf_drop_pixels(b))
        assert bare["pixels"] is None and bare["cls"].tolist() == CLS
    u = framing.decode_sma_frame(framing.smaf_update_meta(b, {"x": "y" * 3}))
    assert u["cls"].tolist() == CLS and u["meta"]["x"] == "yyy"


def test_without_pairing_v1_and_v2_are_unchanged():
    v1 = framing.encode_sma_frame({}, T, CH, TOT, FL)
    assert v1[0] == 1 and framing.decode_sma_frame(v1)["pair"] is None
    v2 = framing.encode_sma_frame({}, T, CH, TOT, FL, raw_words=[1] * 5, word_index=[0] * 5)
    assert v2[0] == 2 and not framing.decode_sma_frame(v2)["pairing"]


def test_bad_pairing_arrays_are_refused():
    w = {"raw_words": [1] * 5, "word_index": [0] * 5}
    with pytest.raises(ValueError, match="pair needs cls"):
        framing.encode_sma_frame({}, T, CH, TOT, FL, pair=PAIR, **w)
    with pytest.raises(ValueError, match="pair goes with the words"):
        framing.encode_sma_frame({}, T, CH, TOT, FL, pair=PAIR, cls=CLS)
    with pytest.raises(ValueError, match="pair goes with the words"):
        framing.encode_sma_frame({}, T, CH, TOT, FL, cls=CLS, **w)
    with pytest.raises(ValueError, match="differ in length"):
        framing.encode_sma_frame({}, T, CH, TOT, FL, pair=PAIR[:4], cls=CLS, **w)
    with pytest.raises(ValueError, match="index into the hit list"):
        framing.encode_sma_frame({}, T, CH, TOT, FL, pair=[5, 0, -1, -1, -1], cls=CLS, **w)
    b = bytearray(framing.encode_sma_frame({}, T, CH, TOT, FL, cls=CLS))
    b[1] &= ~framing.SMAF_PAIRING & 0xFF                     # v3 without its flag
    with pytest.raises(ValueError, match="version 3 with flags"):
        framing.decode_sma_frame(bytes(b))


def _plugin_1015(settings=None, **frame_kw):
    p = P.SmaPlugin(HistStore(), clock=lambda: 1000.0, settings=settings)
    assert p.process(_Event(dense_1015(**frame_kw)), run_number=1015)
    return p


def test_the_plugin_ships_the_pairing_of_every_shipped_hit():
    p = _plugin_1015({"NIM": {"merge": True}}, seed=1, fine_lag={11: 155_000})
    fr = p._last.fr
    for view, kw in (("seeded", {}), ("raster", {"max_hits": 2000}),
                     ("raster", {"drop": [6], "words": True})):
        d = framing.decode_sma_frame(p.frame_blob(view, **kw))
        assert d["version"] == 3, (view, kw)
        assert d["meta"]["roles"] == {"s1": 1, "counters": [1, 2, 7, 4, 5], "rf": 6, "current": -1,
                                      "delayed": [], "nim": list(NIM_1015)}
        assert d["meta"]["nim_merge"] is True
        pr, cl = d["pair"], d["cls"]
        assert (pr is None) == (not d["words"]), "partners only with the words"
        if pr is not None:
            i = np.flatnonzero(pr >= 0)
            assert i.size and np.array_equal(pr[pr[i]], i), "partners point back"
            assert ((cl[i] & framing.PAIR_CLASS_MASK) == N.PAIRED).all()
        nim_side = np.isin(d["ch"], NIM_1015)
        assert np.array_equal((cl & framing.PAIR_NIM_SIDE) != 0, nim_side)
        assert np.all((cl[nim_side] & framing.PAIR_CLASS_MASK) != framing.PAIR_NONE)
        rf = d["ch"] == 6
        assert np.all((cl[rf] & framing.PAIR_CLASS_MASK) == framing.PAIR_NONE)
        held = (cl & framing.PAIR_LAG_HELD) != 0
        assert held.any() and np.all(d["ch"][held] == 11), "only the lagged S4L is held back"
        assert np.all((cl[held] & framing.PAIR_CLASS_MASK) == N.NIM_ONLY)
    # The classes are the frame's pairing, hit for hit (the full raster, every hit shipped).
    d = framing.decode_sma_frame(p.frame_blob("raster", max_hits=None, words=True))
    assert d["n_hits"] == fr.s_t.size
    want = np.where(fr.pairing.cls == P.NIM_NO_CLASS, framing.PAIR_NONE, fr.pairing.cls)
    assert np.array_equal(d["cls"] & framing.PAIR_CLASS_MASK, want)
    assert np.array_equal(d["pair"], fr.pairing.partner)


def test_a_partner_not_shipped_is_minus_one_but_still_paired():
    p = _plugin_1015({"NIM": {"merge": False}}, seed=2)
    d = framing.decode_sma_frame(p.frame_blob("raster", max_hits=301, words=True))
    cl = d["cls"] & framing.PAIR_CLASS_MASK
    lost = (cl == N.PAIRED) & (d["pair"] < 0)
    # The cut keeps the latest hits: only pairs straddling it lose a word, so
    # every paired hit without a partner sits at the start of what was shipped.
    assert lost.any() and np.all(d["t_rel_ns"][lost] <= 40), d["t_rel_ns"][lost]
    full = framing.decode_sma_frame(p.frame_blob("raster", drop=[3, 9, 10, 11, 12], max_hits=None,
                                                 words=True))
    cf = full["cls"] & framing.PAIR_CLASS_MASK
    assert np.all(full["pair"] == -1), "every NIM copy dropped: no partner shipped"
    assert (cf == N.PAIRED).any(), "the TOT words still say paired"


def test_old_layout_frames_stay_v1_and_v2_without_roles():
    from test_sma_plugin import synth_frame
    p = P.SmaPlugin(HistStore(), clock=lambda: 1000.0, settings=old_layout())
    p.process(_Event(synth_frame(10**12, seed=1)), run_number=682)
    s = framing.decode_sma_frame(p.frame_blob("seeded"))
    r = framing.decode_sma_frame(p.frame_blob("raster"))
    assert (s["version"], r["version"]) == (2, 1)
    assert "roles" not in s["meta"] and "nim_merge" not in s["meta"]


def test_merge_off_shows_the_classes_and_holds_nothing_back():
    """The default: NIM-only words are classed and shown, never held back (nothing
    is merged), while the lag fault itself is still measured."""
    p = _plugin_1015({"NIM": {"merge": False}}, seed=1, fine_lag={11: 155_000})
    for view, kw in (("seeded", {}), ("raster", {"max_hits": None})):
        d = framing.decode_sma_frame(p.frame_blob(view, **kw))
        assert d["version"] == 3 and d["meta"]["nim_merge"] is False
        cl = d["cls"]
        assert not ((cl & framing.PAIR_LAG_HELD) != 0).any(), view
        assert ((cl & framing.PAIR_CLASS_MASK) == N.NIM_ONLY).any(), view
    s = p.summary(run_active=True)
    r4 = next(r for r in s["nim"]["counters"] if r["counter"] == "S4")
    assert r4["lag"]["last_state"] == "faulted" and s["nim_lag_held"] == []


def test_a_held_frame_keeps_the_nim_view_it_was_paired_with():
    """A settings edit (merge, the NIM map) after a frame was analysed: the frame
    is still encoded with its own merge state and map, also for a request that
    was never cached (a seed choice)."""
    p = _plugin_1015({"NIM": {"merge": True}}, seed=1, fine_lag={11: 155_000})
    p.apply_settings({"NIM": {"merge": False, "channels": [3, 9, 10, 11, -1]}}, rebuild=True)
    assert not p.cfg.merging and p.cfg.nim.channels[4] == -1
    for kw in ({}, {"seed": "any"}, {"pattern": {"2": "present"}}):
        d = framing.decode_sma_frame(p.frame_blob("seeded", **kw))
        assert d["meta"]["nim_merge"] is True, kw
        assert d["meta"]["roles"]["nim"] == list(NIM_1015), kw
        s5l = d["ch"] == 12
        assert s5l.any() and np.all((d["cls"][s5l] & framing.PAIR_NIM_SIDE) != 0), kw
    # A frame analysed after the edit carries the new view.
    p.process(_Event(dense_1015(seed=3, t0=2 * 10**12)), run_number=1015)
    d = framing.decode_sma_frame(p.frame_blob("seeded"))
    assert d["meta"]["nim_merge"] is False and d["meta"]["roles"]["nim"][4] == -1
    s5l = d["ch"] == 12
    assert np.all((d["cls"][s5l] & framing.PAIR_NIM_SIDE) == 0)
