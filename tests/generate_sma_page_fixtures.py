"""Emit the fixtures the SMA page tests (tests/js/sma*.test.js) run against.

The pages are tested against what the real plugin sends, not against JSON
typed in by hand: a field renamed in ``sma.py`` then fails a page test instead
of blanking a table cell at the beamtime. Everything comes from `SmaPlugin`
driven over the stored real frames in ``tests/data`` (the same frames the
golden and plugin tests use), with a fake clock so the output is reproducible.

Written to ``tests/js/sma-summary-fixture.json``:

``summary``            sma::summary after a few seconds of real 682 frames
``summary_shift13``    the same data with /DQM/SMA/Coarse shift = 13 configured
                       (the real 40k-word 1008 frame has thousands of S1 words,
                       enough for the self-check to call the mismatch)
``summary_timebase``   run 342's frames at Coarse shift 14: suspect frames and a
                       ``time_base`` error; ``raster_suspect`` one of those frames
``summary_budget``     good 682 frames under the CPU budget: 9 of 20 analysed
                       (45 %), max S1 per frame = 100; ``status_budget`` the
                       matching dqm::status
``summary_lossy``      "process all" with 5 frames lost by serial (real loss)
``summary_nolive``     CPU budget, no two consecutive frames: live fraction null
                       (each with its ``status_*``)
``trend``              sma::trend over those seconds
``status``             dqm::status-shaped wrapper around the plugin status
``hist_names``         dqm::list, and ``hist_dims`` the dimension of each
``hists``              two encoded histograms (hex): one 1D, one 2D
``seeded`` / ``raster``  smaf payloads (hex, envelope removed); the raster is
                       requested with drop=[current channel], as the page does.
                       The seeded view is smaf v2 (words), the raster v1
``raster_words``       the same raster frame again by its seq with words (v2), as
                       the page asks for it on Freeze
``raw_seq`` / ``raw_event``  the seeded frame's seq and its sma::raw payload (hex):
                       one raw MIDAS event, a one-event .mid file

Run by hand after a change to the plugin's JSON or the smaf layout::

    PYTHONPATH=src python tests/generate_sma_page_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mdqm.dqm import framing
from mdqm.dqm.hist import HistStore
from mdqm.plugins.sma import SmaPlugin

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "js" / "sma-summary-fixture.json"

T0 = 1_790_000_000.0


class _Bank:
    def __init__(self, data):
        self.data = data


#: The event header time of every fixture event (2026-09-21 14:13:20 UTC), so the tags carry a date.
EVENT_TS = 1_790_000_000


class _Event:
    def __init__(self, words, serial=0, event_id=301):
        self.header = type("H", (), {"event_id": event_id, "serial_number": serial,
                                     "timestamp": EVENT_TS, "trigger_mask": 0})()
        self.banks = {"H000": _Bank(np.asarray(words, dtype="<u8").view("<u4"))}


class _Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def _load(name):
    with np.load(DATA / name) as z:
        return [z[f"f{i}_words"] for i in range(len(z["labels"]))]


def build() -> dict:
    r682 = _load("sma_run00682_frames.npz")
    r1008 = _load("sma_run01008_frame.npz")[0]

    # -- the normal case: the good 682 frames, a few per second for 6 s --
    clock = _Clock(T0)
    p = SmaPlugin(HistStore(), clock=clock)
    serial = 0
    for sec in range(6):
        clock.t = T0 + sec + 0.1
        for w in r682[2:]:                  # frames 0 and 1 are the stale ones
            p.process(_Event(w, serial=serial), run_number=682)
            serial += 1
    # One stale frame last, so the summary carries a stale-frame flag and the
    # raster below is of a good frame (the seeded view skips stale frames).
    clock.t = T0 + 6.2
    p.process(_Event(r682[0], serial=serial), run_number=682)
    serial += 1
    p.process(_Event(r682[4], serial=serial + 3), run_number=682)   # 3 missed by serial
    clock.t = T0 + 7.5

    summary = p.summary(run_active=True)
    trend = p.trend()
    current = p.cfg.roles.current
    seeded = p.frame_blob("seeded")
    raster = p.frame_blob("raster", drop=[current])
    # What Freeze asks for: the raster on screen again, with its words.
    raster_seq = framing.decode_sma_frame(raster)["frame_seq"]
    raster_words = p.frame_blob("raster", drop=[current], max_hits=60000, words=True,
                                seq=raster_seq)
    seeded_seq = framing.decode_sma_frame(seeded)["frame_seq"]
    raw = p.raw_event(seeded_seq)
    assert raw is not None, "the seeded frame's raw event is held"
    names = p.store.names()
    dims = {n: (2 if hasattr(p.store.get(n), "y") else 1) for n in names}
    h1 = "sma/words_per_ch"
    h2 = "sma/tot_vs_ch_lsb0"
    status = {
        "client": "sma_analyzer", "run_number": 682, "run_active": True,
        "events_seen": p.frames, "events_processed": p.frames, "processed_per_s": 4.2,
        "throttled": False, "rate_limit": 1000.0, "configured_rate": 1000.0,
        "reconnects": 0, "plugin": p.status(),
    }

    # -- the shift-mismatch case --
    clock2 = _Clock(T0)
    q = SmaPlugin(HistStore(), settings={"Coarse shift": 13}, clock=clock2)
    for k in range(3):
        clock2.t = T0 + k
        q.process(_Event(r1008, serial=k), run_number=1008)
    clock2.t = T0 + 3.5
    shift13 = q.summary(run_active=True)
    assert shift13["shift"]["verdict"] == "mismatch", shift13["shift"]
    assert any(f["code"] == "shift_mismatch" for f in shift13["flags"])

    # -- a broken time base: run 342's frames at the default shift 14 --
    # (the board ran shift 3 then; most hits fall outside any cluster and the
    # frames come out "suspect"). With enough S1 words the shift check wins
    # and says shift_mismatch instead; time_base is what a shifter sees before
    # it has enough, so the check's word minimum is raised out of reach here.
    r342 = _load("sma_run00342_frames.npz")
    clock3 = _Clock(T0)
    t = SmaPlugin(HistStore(), clock=clock3, settings={
        "Coarse shift": 14, "Self check": {"shift min words": 10**9}})
    for k in range(4):
        clock3.t = T0 + k
        for j, w in enumerate(r342):
            t.process(_Event(w, serial=2 * k + j), run_number=342)
    clock3.t = T0 + 4.5
    timebase = t.summary(run_active=True)
    assert any(f["code"] == "time_base" for f in timebase["flags"]), timebase["flags"]
    suspect_raster = t.frame_blob("raster", drop=[t.cfg.roles.current])

    good682 = r682[2:]

    def sampled(serial_steps, mode, settings=None, cpu=None):
        """The good 682 frames at the given serial steps, as the analyzer would
        hand them over under ``mode`` (its sampling_state set by hand, as
        DqmAnalyzer._publish_sampling_state does)."""
        ck = _Clock(T0)
        q = SmaPlugin(HistStore(), clock=ck, settings=settings)
        serial = 100
        for k in range(len(serial_steps) + 1):
            ck.t = T0 + 0.5 * k + 0.1
            if k:
                serial += serial_steps[k - 1]
            q.process(_Event(good682[k % len(good682)], serial=serial), run_number=682)
        q.sampling_state = {
            "mode": mode, "cpu_budget_pct": 20.0 if mode == "cpu budget" else None,
            "cpu_pct": cpu, "rate_limit": 2.3 if mode == "cpu budget" else None}
        ck.t = T0 + 0.5 * len(serial_steps) + 1.0
        s = q.summary(run_active=True)
        st = {
            "client": "sma_analyzer", "run_number": 682, "run_active": True,
            "events_seen": q.offered_by_serial, "events_processed": q.frames,
            "processed_per_s": 1.8, "throttled": False, "rate_limit": 1000.0,
            "configured_rate": 1000.0, "reconnects": 0,
            "sampling_mode": mode,
            "cpu_budget_pct": 20.0 if mode == "cpu budget" else None,
            "cpu_pct": cpu, "cpu_pct_p95": None if cpu is None else round(cpu * 1.2, 2),
            "analysis_rate_limit": 2.3 if mode == "cpu budget" else None,
            "skips": q.missed_by_serial, "skip_method": "skip_event",
            "events_truncated": 0, "plugin": q.status(),
        }
        return s, st

    # -- the CPU budget at work: 9 of 20 frames analysed, in pairs (so the gap
    # chain, and the live fraction, survive), and at most 100 S1 hits per frame
    # given the S1-seeded analyses --
    budget, status_budget = sampled([1, 4, 1, 4, 1, 4, 1, 3], "cpu budget",
                                    settings={"Cuts": {"max S1 per frame": 100}}, cpu=19.4)
    assert budget["sampling"]["analysed_frac"] == 0.45, budget["sampling"]
    assert budget["sampling"]["s1_analysed_frac"] < 1, budget["sampling"]
    assert budget["live_fraction"] is not None
    assert any(f["code"] == "sampling" for f in budget["flags"])

    # -- every frame asked for, and some lost anyway: real loss --
    lossy, status_lossy = sampled([1, 1, 3, 1, 1, 1, 4, 1, 1, 1, 1], "process all")
    assert lossy["frames"]["missed_by_serial"] == 5, lossy["frames"]

    # -- no two consecutive frames analysed: no live fraction to quote --
    nolive, status_nolive = sampled([2] * 9, "cpu budget", cpu=26.0)
    assert nolive["live_fraction"] is None, nolive["live_fraction"]

    return {
        "_doc": "Generated by tests/generate_sma_page_fixtures.py from SmaPlugin on the real "
                "frames in tests/data. Do not edit by hand.",
        "summary": json.loads(json.dumps(summary)),
        "summary_shift13": json.loads(json.dumps(shift13)),
        "summary_timebase": json.loads(json.dumps(timebase)),
        "raster_suspect": suspect_raster.hex(),
        "summary_budget": json.loads(json.dumps(budget)),
        "status_budget": json.loads(json.dumps(status_budget, default=str)),
        "summary_lossy": json.loads(json.dumps(lossy)),
        "status_lossy": json.loads(json.dumps(status_lossy, default=str)),
        "summary_nolive": json.loads(json.dumps(nolive)),
        "status_nolive": json.loads(json.dumps(status_nolive, default=str)),
        "trend": json.loads(json.dumps(trend)),
        "status": json.loads(json.dumps(status, default=str)),
        "hist_names": names,
        "hist_dims": dims,
        "hists": {h1: p.store.get(h1).encode().hex(), h2: p.store.get(h2).encode().hex()},
        "current_channel": current,
        "seeded": seeded.hex(),
        "raster": raster.hex(),
        "raster_words": raster_words.hex(),
        "raw_seq": int(seeded_seq),
        "raw_event": bytes(raw.data).hex(),
    }


def test_generate_sma_page_fixtures():
    fx = build()
    OUT.write_text(json.dumps(fx, indent=1))
    assert fx["summary"]["channels"] and len(fx["summary"]["channels"]) == 16
    assert json.loads(json.dumps(fx["seeded"])), "seeded frame present"


if __name__ == "__main__":
    test_generate_sma_page_fixtures()
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
