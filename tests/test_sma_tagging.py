"""Finding an odd SMA event again: tags, word indices, the raw ring, sma::raw,
mdqm-sma-file --serial, and replaying with the file's own headers."""

from __future__ import annotations

import ctypes
import json
import struct

import numpy as np
import pytest

from mdqm.dqm import analyzer as A
from mdqm.dqm import framing
from mdqm.dqm.hist import HistStore
from mdqm.plugins import sma as P
from mdqm.plugins import sma_words as W
from mdqm.tools import midasfile as MF
from mdqm.tools import sma_file as SF


def synth_words(t0=10**12, n=300, spacing=3000, seed=1, pixels=5):
    """A healthy frame with some pixel and filler words mixed in."""
    rng = np.random.default_rng(seed)
    t1 = t0 + np.arange(n, dtype=np.int64) * spacing + rng.integers(0, 500, n)
    ts = [t1] + [t1 + c for c in (2, 3, 4, 5)] + [t1 + 40, t1 + 60]
    chs = [np.full(n, 1)] + [np.full(n, c) for c in (2, 3, 4, 5)] + [np.full(n, 6)] * 2
    t, ch = np.concatenate(ts), np.concatenate(chs)
    o = np.argsort(t, kind="stable")
    coarse, fine = W.fields_of(t[o], 14)
    trig = W.encode(ch[o], np.full(t.size, 20), coarse, fine)
    pix = np.full(pixels, 0x0123456789ABCDEF, dtype=np.uint64)          # bit 63 clear
    fill = np.full(2, W.FILLER, dtype=np.uint64)
    return np.concatenate([pix, trig[:100], fill, trig[100:]])


class _Ev:
    def __init__(self, words, serial, ts=1790000000, event_id=301):
        self.header = MF.EventHeader(event_id, 0, serial, ts, 0)
        self.banks = {"H000": MF.Bank("H000", 6, words.nbytes, words.view("<u4"))}

    def get_bank(self, name):
        return self.banks.get(name)


def _plugin(**sampling):
    return P.SmaPlugin(HistStore(), settings={"Sampling": sampling} if sampling else None)


# --- smaf v2 --------------------------------------------------------------------------------

def test_smaf_v1_is_unchanged_and_v2_round_trips():
    meta = {"x": 1}
    t = np.array([0, 5, 9], dtype="<u4")
    ch, tot, fl = (np.array([1, 2, 3], dtype=np.uint8),) * 3
    v1 = framing.encode_sma_frame(meta, t, ch, tot, fl, frame_seq=7, run_number=3)
    assert v1[0] == 1 and len(v1) == 32 + 8 + 7 * 3
    d = framing.decode_sma_frame(v1)
    assert d["version"] == 1 and d["words"] is False and d["raw_words"] is None
    raw = np.array([0x8100000000000001, 0x8200000000000002, 0xFFFFFFFFFFFFFFFF], dtype="<u8")
    wi = np.array([10, 11, 40000], dtype="<u4")
    v2 = framing.encode_sma_frame(meta, t, ch, tot, fl, frame_seq=7, run_number=3,
                                  raw_words=raw, word_index=wi)
    assert v2[0] == 2 and v2[1] & framing.SMAF_WORDS and len(v2) == 32 + 8 + 19 * 3
    d = framing.decode_sma_frame(v2)
    assert d["words"] and np.array_equal(d["raw_words"], raw) and np.array_equal(d["word_index"], wi)
    assert np.array_equal(d["t_rel_ns"], t) and np.array_equal(d["ch"], ch)
    with pytest.raises(ValueError):
        framing.encode_sma_frame(meta, t, ch, tot, fl, raw_words=raw)


def test_frames_carry_the_tag_and_the_word_of_every_hit():
    words = synth_words()
    p = _plugin()
    p.process(_Ev(words, serial=4757), run_number=1008)
    d = framing.decode_sma_frame(p.frame_blob("seeded"))
    m = d["meta"]
    # Default (run-1015) roles: ch 3 is S1L, so the frame is paired (v3); words either way.
    assert d["words"] and d["version"] == 3, "the seeded view always ships words"
    assert m["tag"] == ("SMA run 1008 · event 301 serial 4757 · 2026-09-21 14:13:20 UTC"
                        " · frame seq 1")
    assert m["event"] == {"id": 301, "serial": 4757, "timestamp": 1790000000, "trigger_mask": 0}
    assert m["raw_held"] is True
    # Every shipped hit's word index points at its own raw word in the bank.
    assert np.array_equal(words[d["word_index"].astype(np.intp)], d["raw_words"])
    ch_from_raw = ((d["raw_words"] >> np.uint64(56)) & np.uint64(0xF)).astype(np.uint8)
    assert np.array_equal(ch_from_raw, d["ch"])
    s0 = m["seeds"][0]
    assert s0["word_range"][0] <= s0["s1_word"] <= s0["word_range"][1]
    assert ((words[s0["s1_word"]] >> np.uint64(56)) & np.uint64(0xF)) == 1, "an S1 word"

    r = framing.decode_sma_frame(p.frame_blob("raster"))
    assert not r["words"], "the live raster stays lean"
    r2 = framing.decode_sma_frame(p.frame_blob("raster", words=True))
    assert r2["words"] and r2["n_hits"] == r["n_hits"]
    assert np.array_equal(words[r2["word_index"].astype(np.intp)], r2["raw_words"])


def test_a_frame_by_seq_is_rebuilt_from_the_raw_ring():
    p = _plugin()
    frames = [synth_words(t0=10**12 + k * 10**7, seed=k) for k in range(3)]
    for k, w in enumerate(frames):
        p.process(_Ev(w, serial=100 + k), run_number=1)
    live = framing.decode_sma_frame(p.frame_blob("raster", words=True, seq=3))
    old = framing.decode_sma_frame(p.frame_blob("raster", words=True, seq=1))
    assert live["meta"]["event"]["serial"] == 102 and old["meta"]["event"]["serial"] == 100
    assert np.array_equal(frames[0][old["word_index"].astype(np.intp)], old["raw_words"])
    assert p.frame_blob("raster", seq=99) is None
    tag, body = _dispatch(p, "sma::frame", '{"view": "raster", "seq": 99}')
    assert tag == framing.TAG_ERROR and b"no longer held" in body


# --- the raw ring and sma::raw ----------------------------------------------------------------

def _dispatch(p, cmd, args=""):
    _size, tag, body = framing.parse_envelope(p.commands()[cmd](args))
    return tag, body


def test_sma_raw_returns_a_one_event_file_that_the_readers_accept(tmp_path):
    words = synth_words()
    p = _plugin()
    p.process(_Ev(words, serial=42), run_number=7)
    tag, body = _dispatch(p, "sma::raw", '{"seq": 1}')
    assert tag == framing.TAG_MEVT
    f = tmp_path / "sma_run7_serial42.mid"
    f.write_bytes(body)
    evs = list(MF.MidasFile(f))
    assert len(evs) == 1 and evs[0].header.serial_number == 42 and evs[0].position == 0
    got = W.words_from_bank(evs[0].get_bank("H000").data)
    assert np.array_equal(got, words)
    tag, body = _dispatch(p, "sma::raw", '{"seq": 5}')
    assert tag == framing.TAG_ERROR and body.decode() == "frame 5 no longer held — use the tag"
    tag, _b = _dispatch(p, "sma::raw", "")
    assert tag == framing.TAG_ERROR


def test_the_raw_ring_is_bounded_by_bytes():
    one = len(P.event_bytes(_Ev(synth_words(), 0)))
    p = _plugin(**{"raw ring MB": 3.5 * one / (1 << 20)})
    for k in range(10):
        p.process(_Ev(synth_words(t0=10**12 + k * 10**7, seed=k), serial=k), run_number=1)
    st = p.status()["raw_ring"]
    assert st["frames"] == 3 and st["bytes"] <= st["limit_bytes"]
    assert st["oldest_seq"] == 8, "the latest three are held"
    assert p.raw_event(10) is not None and p.raw_event(7) is None

    q = _plugin(**{"raw ring MB": 0.5 * one / (1 << 20)})
    q.process(_Ev(synth_words(), 0), run_number=1)
    assert q.status()["raw_ring"]["frames"] == 0 and q.raw_not_kept == 1, "larger than the ring"
    z = _plugin(**{"raw ring MB": 0})
    z.process(_Ev(synth_words(), 0), run_number=1)
    assert z.status()["raw_ring"]["frames"] == 0


def test_oversize_frames_are_never_stored():
    p = P.SmaPlugin(HistStore(), settings={"Cuts": {"max words per frame": 100}})
    p.process(_Ev(synth_words(), 0), run_number=1)
    assert p.frames_oversize == 1 and p.status()["raw_ring"]["frames"] == 0


def test_the_bytes_as_received_are_kept_when_the_analyzer_has_them():
    words = synth_words()
    ev = _Ev(words, 3)
    exact = MF.encode_event(301, 3, 1790000000, [("H000", 6, words.view("<u4"))])
    exact = exact[:36] + b"\x12\x34\x56\x78" + exact[40:]   # the frontend's pad bytes
    ev.raw = exact
    assert P.event_bytes(ev) == exact

    class _Client:
        def __init__(self):
            self.event_buffers = {0: ctypes.create_string_buffer(exact + b"\0" * 64)}

    class _RawEv:
        def __init__(self):
            self.header = type("H", (), {"event_data_size_bytes": len(exact) - 16})()

    e = _RawEv()
    A.Analyzer._attach_raw(_Client(), e)
    assert e.mdqm_raw == exact


# --- mdqm-sma-file --serial -------------------------------------------------------------------

def _write_run(tmp_path, run=9999, per_file=3, files=2):
    frames = {}
    serial = 0
    for sub in range(files):
        with open(tmp_path / f"run{run:05d}_{sub:05d}.mid", "wb") as fh:
            fh.write(MF.encode_internal(MF.EVID_BOR, run, 1790000000, b""))
            fh.write(MF.encode_event(401, 0, 1790000000, [("W000", 6, np.zeros(4, "<u4"))]))
            for _k in range(per_file):
                w = synth_words(t0=10**12 + serial * 10**7, seed=serial)
                frames[serial] = (sub, w)
                fh.write(MF.encode_event(301, serial, 1790000000 + serial,
                                         [("H000", 6, w.view("<u4"))]))
                serial += 1
    return frames


def test_cli_finds_a_serial_across_subruns(tmp_path, capsys):
    frames = _write_run(tmp_path)
    rc = SF.main(["--serial", "4", "--run", "9999", "--dir", str(tmp_path),
                  "--out", str(tmp_path / "o"), "--words", "3:8"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "SMA run 9999 · event 301 serial 4" in out
    # Subrun 1: BOR (0), the WD event (1), serial 3 (2), serial 4 (3).
    assert "subrun 1  event position 3" in out
    sub, w = frames[4]
    assert f"0x{int(w[5]):016x}" in out, "the trigger word at index 5, as raw hex"
    assert "pixel" in out and "filler" in out
    written = tmp_path / "o" / "sma_run9999_serial4.mid"
    ev = next(iter(MF.MidasFile(written)))
    assert np.array_equal(W.words_from_bank(ev.get_bank("H000").data), w)
    assert SF.main(["--serial", "99", "--run", "9999", "--dir", str(tmp_path), "--quiet"]) == 1
    assert SF.main(["--serial", "1", "--words", "x", str(tmp_path / "run09999_00000.mid")]) == 2


def test_cli_one_file_mode_still_needs_exactly_one_file(tmp_path):
    _write_run(tmp_path, files=1)
    f = tmp_path / "run09999_00000.mid"
    assert SF.main([str(f), str(f), "--no-png", "--quiet"]) == 2
    assert SF.main([str(f), "--no-png", "--quiet", "--out", str(tmp_path / "x")]) in (0, 3)


# --- replaying with the file's own headers ----------------------------------------------------

def test_a_replay_loop_with_kept_serials_is_a_restart_not_a_gap():
    p = _plugin()
    t = 10**12
    for s in (205, 206, 207, 0, 1):          # 196 frames, the loop starts again at 0
        p.process(_Ev(synth_words(t0=t, seed=s), serial=s), run_number=1008)
        t += 10**7
    assert p.offered_by_serial == 5, "a backwards serial counts one frame, not a gap"
    assert p.missed_by_serial == 0

    tm = type("T", (), {"t": 1000.0})()
    a = A.Analyzer(lambda s: P.SmaPlugin(s), clock=lambda: tm.t)
    for s in (205, 206, 207, 0, 1):
        a._note_serial(_Ev(synth_words(), serial=s))
    assert a.offered == 5


def test_replay_run_has_keep_header():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "replay_run", Path(__file__).parents[1] / "scripts" / "replay-run.py")
    src = spec.loader.get_source("replay_run")
    assert "--keep-header" in src and "if not keep_header:" in src
