"""Emit ``smaf`` (SMA frame) cases for the JavaScript decoder to read.

The scope-frame arrangement again: the Python suite writes the cases, so the
encoder and the page's decoder cannot drift apart unnoticed. Payloads are hex
(the envelope already sliced off, as the page hands it to its decoder), and
each case carries the header fields, the JSON metadata and the four hit arrays
as plain lists.

Most cases come from the real plugin on the stored real frames, so the page is
tested against what the analyzer actually sends; the synthetic ones pin the
edges (no hits, a JSON block that is already a multiple of 8, the flags). The
real frames carry the MuPix pixel block (``SMAF_PIXELS``) as the plugin sends
it by default; ``pixels=False`` cases are the same frames without it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mdqm.dqm import framing
from mdqm.dqm.hist import HistStore

OUT = Path(__file__).resolve().parent / "js" / "smaframe-cases.json"
DATA = Path(__file__).resolve().parent / "data"


class _Bank:
    def __init__(self, data):
        self.data = data


class _Event:
    def __init__(self, words, serial=0, event_id=301):
        self.header = type("H", (), {"event_id": event_id, "serial_number": serial,
                                     "timestamp": 1_790_000_000, "trigger_mask": 0})()
        self.banks = {"H000": _Bank(np.asarray(words, dtype="<u8").view("<u4"))}


def _plugin_frames():
    """(name, payload) from the plugin on real frames."""
    from mdqm.plugins.sma import SmaPlugin

    out = []
    with np.load(DATA / "sma_run00682_frames.npz") as z:
        words = [z[f"f{i}_words"] for i in range(len(z["labels"]))]
    with np.load(DATA / "sma_run01008_frame.npz") as z:
        big = z["f0_words"]

    p = SmaPlugin(HistStore(), clock=lambda: 1000.0)
    p.process(_Event(words[4], serial=3553), run_number=682)
    out.append(("real 4k-word frame, seeded (v2, words)", p.frame_blob("seeded")))
    out.append(("real 4k-word frame, seeded (v2, words), no pixel block",
                p.frame_blob("seeded", pixels=False)))
    out.append(("real 4k-word frame, raster without ch 7, no pixel block (v1 as before)",
                p.frame_blob("raster", drop=[7], pixels=False)))
    out.append(("real 4k-word frame, raster without ch 7",
                p.frame_blob("raster", drop=[7])))
    out.append(("real 4k-word frame, raster without ch 7, with words (v2)",
                p.frame_blob("raster", drop=[7], words=True)))
    out.append(("real 4k-word frame, seeded without words (v1)",
                p.frame_blob("seeded", words=False)))
    out.append(("real frame, raster capped at 50 hits (truncated)",
                p.frame_blob("raster", max_hits=50)))

    p.process(_Event(words[1], serial=1), run_number=682)
    out.append(("real stale frame (channel 0/15 garbage), raster",
                p.frame_blob("raster", pixels=False)))

    p = SmaPlugin(HistStore(), clock=lambda: 1000.0)
    p.process(_Event(big, serial=206), run_number=1008)
    out.append(("real 40k-word-firmware frame, seeded (v2)", p.frame_blob("seeded")))
    out.append(("real 40k-word-firmware frame, raster with pixels, latest 600 hits of each kind",
                p.frame_blob("raster", drop=[7], max_hits=600)))
    out.append(("real 40k-word-firmware frame, seeded, MuPix selector both",
                p.frame_blob("seeded", mupix="both")))

    # A shift-3 run read at 14: no usable time base, the kept words spread over
    # hundreds of seconds, so time_shift is large.
    with np.load(DATA / "sma_run00342_frames.npz") as z:
        w342 = z["f0_words"]
    p = SmaPlugin(HistStore(), clock=lambda: 1000.0)
    p.process(_Event(w342, serial=1), run_number=342)
    out.append(("real frame with a wrong shift (suspect), raster, time_shift > 0",
                p.frame_blob("raster", drop=[6])))

    # A genuine frame longer than 4.29 s (a far consistent cluster): time_shift 1.
    from test_sma_plugin import synth_frame
    p = SmaPlugin(HistStore(), clock=lambda: 1000.0)
    p.process(_Event(synth_frame(10**12, n=40, far=5_500_000_000)), run_number=1)
    out.append(("5.5 s frame, raster, time_shift 1", p.frame_blob("raster")))
    return out


def _synthetic_frames():
    out = []
    empty = framing.encode_sma_frame({"view": "raster", "seeds": []}, [], [], [], [],
                                     frame_seq=1, run_number=7)
    out.append(("no hits", empty))

    # Pad the metadata until the JSON block ends exactly on 8 bytes, so the
    # arrays follow with no padding at all -- the case an off-by-one in the
    # decoder's alignment would get wrong.
    meta = {"view": "seeded", "pad": ""}
    while len(json.dumps(meta, separators=(",", ":"))) % 8:
        meta["pad"] += "x"
    t = np.array([0, 1, 250, 4_000_000_000], dtype=np.uint32)
    blob = framing.encode_sma_frame(
        meta, t, [1, 2, 6, 15], [0, 254, 255, 17],
        [0, framing.HIT_MISMATCH | framing.HIT_TOT_CORRUPT,
         framing.HIT_FINE_LSB | framing.HIT_IN_SEED, 0x1F],
        frame_seq=2**40 + 5, run_number=4294967295, seeded=True, stale=True,
        truncated=True, suspect=True, time_shift=32)
    out.append(("JSON already 8-aligned; every flag; extreme header values", blob))

    out.append(("no hits, with words (v2)",
                framing.encode_sma_frame({"view": "raster", "seeds": []}, [], [], [], [],
                                         frame_seq=3, run_number=7, raw_words=[],
                                         word_index=[])))

    # v2 with the JSON block ending off 8 (so padding precedes the u64 array),
    # words with the top bit set (past 2^63: a signed read would go negative)
    # and past 2^53 (a Number would round), and the largest word index.
    meta = {"view": "raster", "pad": "x"}
    while len(json.dumps(meta, separators=(",", ":"))) % 8 != 3:
        meta["pad"] += "x"
    words = np.array([0x8512_3456_789A_BCDE, 0xFFFF_FFFF_FFFF_FFFF, 1, 0x0020_0000_0000_0001],
                     dtype=np.uint64)
    blob = framing.encode_sma_frame(
        meta, [0, 7, 9, 4_000_000_000], [5, 1, 15, 6], [37, 0, 255, 12],
        [framing.HIT_MISMATCH, 0, framing.HIT_TOT_CORRUPT, framing.HIT_IN_SEED],
        frame_seq=4750, run_number=1008, time_shift=0,
        raw_words=words, word_index=np.array([12345, 0, 4294967295, 7], dtype=np.uint32))
    out.append(("v2 synthetic: padded JSON, words past 2^63 and 2^53, max word index", blob))

    # Pixel blocks: after a v1 hit array that ends off 8 (3 hits: 21 bytes, so
    # 3 bytes of padding precede the block), and after v2; pixel words past 2^53;
    # a pixel time shift; every pixel flag; an empty block.
    pix = {"t_rel": [0, 8, 4_000_000_000], "time_shift": 2, "chip": [0, 5, 31],
           "col": [255, 0, 17], "row": [249, 250, 255], "tot": [31, 0, 7],
           "flags": [framing.PIX_IN_SEED | 1, 2 | framing.PIX_OFF_SENSOR, 0]}
    out.append(("v1 with a pixel block (no words), block after 3 bytes of padding",
                framing.encode_sma_frame({"view": "raster", "mupix": {"t0_ns": 5}},
                                         [0, 3, 9], [1, 2, 6], [1, 2, 3], [0, 0, 0],
                                         frame_seq=9, run_number=1008, pixels=pix)))
    pw = dict(pix, raw_words=np.array([0x7C3A_D2C0_A1B2_C3D4, 0x0020_0000_0000_0001, 0],
                                      dtype=np.uint64),
              word_index=np.array([4294967295, 0, 12], dtype=np.uint32))
    out.append(("v2 with a pixel block with words (past 2^53), a pixel time shift",
                framing.encode_sma_frame({"view": "seeded"}, [0, 1], [1, 2], [5, 6], [0, 0],
                                         frame_seq=10, run_number=1008, seeded=True,
                                         raw_words=[0x8100000000000001, 0x8200000000000002],
                                         word_index=[1, 2], pixels=pw)))
    empty_px = {k: [] for k in ("t_rel", "chip", "col", "row", "tot", "flags")}
    out.append(("an empty pixel block after no hits",
                framing.encode_sma_frame({"view": "raster"}, [], [], [], [], frame_seq=11,
                                         pixels=dict(empty_px, time_shift=0))))
    return out


def build_cases() -> list[dict]:
    cases = []
    for name, blob in _plugin_frames() + _synthetic_frames():
        d = framing.decode_sma_frame(blob)
        cases.append({
            "name": name,
            "payload_hex": blob.hex(),
            "expect": {
                "version": d["version"], "flags": d["flags"], "n_hits": d["n_hits"],
                "frame_seq": d["frame_seq"], "run_number": d["run_number"],
                "json_len": d["json_len"], "arrays_offset": d["arrays_offset"],
                "seeded": d["seeded"], "stale": d["stale"], "truncated": d["truncated"],
                "suspect": d["suspect"], "time_shift": d["time_shift"],
                "meta": d["meta"],
                "t_rel_ns": [int(x) for x in d["t_rel_ns"]],
                "ch": [int(x) for x in d["ch"]],
                "tot": [int(x) for x in d["tot"]],
                "hit_flags": [int(x) for x in d["hit_flags"]],
                "words": d["words"],
                # u64 as hex strings: JSON numbers stop being exact at 2^53.
                "raw_words": (None if d["raw_words"] is None
                              else [f"{int(x):016x}" for x in d["raw_words"]]),
                "word_index": (None if d["word_index"] is None
                               else [int(x) for x in d["word_index"]]),
                "pixels": _pixels_expect(d["pixels"]),
            },
        })
    return cases


def _pixels_expect(q):
    if q is None:
        return None
    return {"n": q["n"], "time_shift": q["time_shift"], "words": q["words"],
            "offset": q["offset"], "t_rel": [int(x) for x in q["t_rel"]],
            **{k: [int(x) for x in q[k]] for k in ("chip", "col", "row", "tot", "flags")},
            "raw_words": (None if q["raw_words"] is None
                          else [f"{int(x):016x}" for x in q["raw_words"]]),
            "word_index": (None if q["word_index"] is None
                           else [int(x) for x in q["word_index"]])}


def test_generate_smaframe_cases():
    cases = build_cases()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "_doc": "Generated by tests/generate_smaframe_cases.py (smaf v1 and v2 payloads, "
                "envelope removed, hex). Layout: src/mdqm/dqm/framing.py, 'SMA frames'. A "
                "hit's time is meta.t0_ns + t_rel_ns * 2**time_shift (header u16 at offset "
                "2). frame_seq is 2^40+5 in the synthetic case. raw_words are 16-digit hex "
                "strings (u64). pixels: the MuPix pixel block (header flag SMAF_PIXELS, "
                "'pixel block' in framing.py), null when there is none; a pixel's time is "
                "meta.mupix.t0_ns + t_rel * 2**time_shift.",
        "cases": cases,
    }, indent=1))
    assert len(cases) >= 20
    assert {c["expect"]["version"] for c in cases} == {1, 2}
    px = [c for c in cases if c["expect"]["pixels"] is not None]
    assert {c["expect"]["version"] for c in px} == {1, 2}, "a pixel block after v1 and after v2"
    assert {c["expect"]["pixels"]["words"] for c in px} == {False, True}
    assert any(c["expect"]["pixels"]["n"] >= 600 for c in px), "a real raster's pixels"
    assert {c["expect"]["time_shift"] for c in cases} >= {0, 1, 32}
    assert any(c["expect"]["time_shift"] > 1 and c["expect"]["suspect"] for c in cases)


def test_python_decodes_its_own_cases():
    for case in build_cases():
        d = framing.decode_sma_frame(bytes.fromhex(case["payload_hex"]))
        e = case["expect"]
        assert d["n_hits"] == len(e["t_rel_ns"]) == len(e["ch"]), case["name"]
        assert d["arrays_offset"] % 8 == 0, case["name"]
        per_hit = 19 if d["words"] else 7
        end = d["arrays_offset"] + per_hit * d["n_hits"]
        size = len(bytes.fromhex(case["payload_hex"]))
        if d["pixels"] is None:
            assert size == end
        else:
            q = d["pixels"]
            assert q["offset"] % 8 == 0 and 0 <= q["offset"] - end < 8, case["name"]
            assert size == q["offset"] + 8 + (21 if q["words"] else 9) * q["n"], case["name"]
