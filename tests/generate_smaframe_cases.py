"""Emit ``smaf`` (SMA frame) cases for the JavaScript decoder to read.

The scope-frame arrangement again: the Python suite writes the cases, so the
encoder and the page's decoder cannot drift apart unnoticed. Payloads are hex
(the envelope already sliced off, as the page hands it to its decoder), and
each case carries the header fields, the JSON metadata and the four hit arrays
as plain lists.

Most cases come from the real plugin on the stored real frames, so the page is
tested against what the analyzer actually sends; the synthetic ones pin the
edges (no hits, a JSON block that is already a multiple of 8, the flags).
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
                p.frame_blob("raster")))

    p = SmaPlugin(HistStore(), clock=lambda: 1000.0)
    p.process(_Event(big, serial=206), run_number=1008)
    out.append(("real 40k-word-firmware frame, seeded (v2)", p.frame_blob("seeded")))

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
            },
        })
    return cases


def test_generate_smaframe_cases():
    cases = build_cases()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "_doc": "Generated by tests/generate_smaframe_cases.py (smaf v1 and v2 payloads, "
                "envelope removed, hex). Layout: src/mdqm/dqm/framing.py, 'SMA frames'. A "
                "hit's time is meta.t0_ns + t_rel_ns * 2**time_shift (header u16 at offset "
                "2). frame_seq is 2^40+5 in the synthetic case. raw_words are 16-digit hex "
                "strings (u64).",
        "cases": cases,
    }, indent=1))
    assert len(cases) >= 13
    assert {c["expect"]["version"] for c in cases} == {1, 2}
    assert {c["expect"]["time_shift"] for c in cases} >= {0, 1, 32}
    assert any(c["expect"]["time_shift"] > 1 and c["expect"]["suspect"] for c in cases)


def test_python_decodes_its_own_cases():
    for case in build_cases():
        d = framing.decode_sma_frame(bytes.fromhex(case["payload_hex"]))
        e = case["expect"]
        assert d["n_hits"] == len(e["t_rel_ns"]) == len(e["ch"]), case["name"]
        assert d["arrays_offset"] % 8 == 0, case["name"]
        per_hit = 19 if d["words"] else 7
        assert len(bytes.fromhex(case["payload_hex"])) == d["arrays_offset"] + per_hit * d["n_hits"]
