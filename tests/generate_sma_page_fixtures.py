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
``seeded_any``         the seeded view with seed "any" on the same frame (``select``)
``seeded_filtered``    seed S1 with filters ["tot", "rf"] (fewer seeds, each ``odd``)
``seeded_stale``       the default seeded view after frames without S1 (682 frames
                       with the S1 words taken out): the old frame, with ``stale_view``
``seeded_any_nos1``    seed "any" on such a frame: seeds without S1 (``rf_na``)
``seeded_nomatch``     seed "ch8" with filter "tot": ``search.no_match``
``summary_noseeds``    sma::summary after 12 s of frames without S1: ``no_seeds``
``seeded_mupix_both``  seed S1 with the MuPix selector "both" (``select.mupix``)
``seeded_mupix_none``  seed S1 with the MuPix selector "none"
``summary_mupix_sync`` the 682 frames with the MuPix in-time window moved 1 us late
                       (nothing in it), after the sync hold: a ``mupix_sync`` warning
``trend_mupix_sync``   its sma::trend (MuPix fractions near 0)
``summary_1008``       sma::summary of the real 1008 frame at shift 14: S5's fine =
                       t/2 fault in ``timestamp_faults``
``seeded_pattern``     that frame, filter "incomplete" (S5 ignored: ``meta.incomplete``)
                       and pattern {"1": "present", "3": "absent"} (``select.pattern``)
``hists`` also carries four MuPix histograms of the normal case (dt L1/L2, the
S1 match, the column occupancy).

The 682 frames carry MuPix pixel words, so ``seeded`` / ``raster`` /
``raster_words`` have the smaf pixel block (``meta.mupix``) as the analyzer
sends it by default.

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
    # The MuPix histograms the page tests look at (the others fall back to the
    # generic 1D/2D ones: a 2D histogram is 140 kB of hex here).
    mupix_hists = {n: p.store.get(n).encode().hex() for n in
                   ("sma/mupix_dt_L1", "sma/mupix_dt_L2", "sma/mupix_s1_match", "sma/mupix_col_chip")}
    status = {
        "client": "sma_analyzer", "run_number": 682, "run_active": True,
        "events_seen": p.frames, "events_processed": p.frames, "processed_per_s": 4.2,
        "throttled": False, "rate_limit": 1000.0, "configured_rate": 1000.0,
        "reconnects": 0, "plugin": p.status(),
    }

    seeded_any = p.frame_blob("seeded", seed="any")
    seeded_filtered = p.frame_blob("seeded", filters=["tot", "rf"])
    assert framing.decode_sma_frame(seeded_filtered)["meta"]["seeds"], "a filtered seed"
    seeded_mupix_both = p.frame_blob("seeded", mupix="both")
    d = framing.decode_sma_frame(seeded_mupix_both)
    assert d["meta"]["select"]["mupix"] == "both" and d["meta"]["mupix"]["seeds"], "L1+L2 seeds"
    assert d["pixels"] is not None and d["pixels"]["n"] > 0
    seeded_mupix_none = p.frame_blob("seeded", mupix="none")
    assert framing.decode_sma_frame(raster)["pixels"]["n"] > 100, "the raster's pixel hits"

    # -- MuPix out of time: the in-time window moved 1 us late (nothing real
    # in it), the sync flag after its hold time --
    clock5 = _Clock(T0)
    v = SmaPlugin(HistStore(), clock=clock5, settings={
        "MuPix": {"window lo ns": 1000, "window hi ns": 1600},
        "Self check": {"mupix sync hold s": 3.0, "mupix sync window s": 3.0,
                       "mupix sync min S1": 100}})
    serial5 = 0
    for sec in range(8):
        for w in r682[2:]:
            clock5.t = T0 + sec + 0.1 + 0.2 * (serial5 % 4)
            v.process(_Event(w, serial=serial5), run_number=682)
            serial5 += 1
    clock5.t = T0 + 8.5
    mupix_sync = v.summary(run_active=True)
    assert any(f["code"] == "mupix_sync" for f in mupix_sync["flags"]), mupix_sync["flags"]
    trend_sync = v.trend()

    # -- S1 goes missing: 682 frames without their S1 words, for 12 s --
    clock4 = _Clock(T0)
    u = SmaPlugin(HistStore(), clock=clock4)
    u.process(_Event(r682[4], serial=0), run_number=682)
    s1 = u.cfg.roles.s1

    def no_s1(w):
        w = np.asarray(w, dtype="<u8")
        trig = (w >> np.uint64(63)) == 1
        ch = (w >> np.uint64(56)) & np.uint64(0xF)
        return w[~(trig & (ch == s1) & (w != np.uint64(0xFFFFFFFFFFFFFFFF)))]

    for k in range(12):
        clock4.t = T0 + 1 + k
        u.process(_Event(no_s1(r682[2 + k % 4]), serial=1 + k), run_number=682)
    clock4.t = T0 + 13.2
    seeded_stale = u.frame_blob("seeded")
    assert framing.decode_sma_frame(seeded_stale)["meta"]["stale_view"]["good_since"] == 12
    seeded_any_nos1 = u.frame_blob("seeded", seed="any")
    seeds = framing.decode_sma_frame(seeded_any_nos1)["meta"]["seeds"]
    assert seeds and all(x["rf_na"] for x in seeds), seeds
    seeded_nomatch = u.frame_blob("seeded", seed="ch8", filters=["tot"])
    assert framing.decode_sma_frame(seeded_nomatch)["meta"]["search"]["no_match"]
    noseeds = u.summary(run_active=True)
    assert any(f["code"] == "no_seeds" for f in noseeds["flags"]), noseeds["flags"]

    # -- the shift-mismatch case --
    clock2 = _Clock(T0)
    q = SmaPlugin(HistStore(), settings={"Coarse shift": 13}, clock=clock2)
    for k in range(3):
        clock2.t = T0 + k
        q.process(_Event(r1008, serial=k), run_number=1008)
    clock2.t = T0 + 3.5
    shift13 = q.summary(run_active=True)
    assert shift13["shift"]["verdict"] == "mismatch", shift13["shift"]

    # -- run 1008 at its own shift: S5's fine = t/2 fault is a known timestamp fault --
    clock5 = _Clock(T0)
    f = SmaPlugin(HistStore(), clock=clock5)
    for k in range(3):
        clock5.t = T0 + k
        f.process(_Event(r1008, serial=k), run_number=1008)
    clock5.t = T0 + 3.5
    summary_1008 = f.summary(run_active=True)
    assert [x["counter"] for x in summary_1008["timestamp_faults"]["counters"]] == [5]
    seeded_pattern = f.frame_blob("seeded", filters=["incomplete"],
                                  pattern={"1": "present", "3": "absent"})
    d = framing.decode_sma_frame(seeded_pattern)["meta"]
    assert d["seeds"] and d["incomplete"]["ignored"][0]["counter"] == 5, d["incomplete"]
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
        "hists": {h1: p.store.get(h1).encode().hex(), h2: p.store.get(h2).encode().hex(),
                  **mupix_hists},
        "current_channel": current,
        "seeded": seeded.hex(),
        "raster": raster.hex(),
        "raster_words": raster_words.hex(),
        "raw_seq": int(seeded_seq),
        "raw_event": bytes(raw.data).hex(),
        "seeded_any": seeded_any.hex(),
        "seeded_filtered": seeded_filtered.hex(),
        "seeded_stale": seeded_stale.hex(),
        "seeded_any_nos1": seeded_any_nos1.hex(),
        "seeded_nomatch": seeded_nomatch.hex(),
        "summary_noseeds": json.loads(json.dumps(noseeds)),
        "seeded_mupix_both": seeded_mupix_both.hex(),
        "seeded_mupix_none": seeded_mupix_none.hex(),
        "summary_1008": json.loads(json.dumps(summary_1008)),
        "seeded_pattern": seeded_pattern.hex(),
        "summary_mupix_sync": json.loads(json.dumps(mupix_sync)),
        "trend_mupix_sync": json.loads(json.dumps(trend_sync)),
    }


def test_generate_sma_page_fixtures():
    fx = build()
    OUT.write_text(json.dumps(fx, indent=1))
    assert fx["summary"]["channels"] and len(fx["summary"]["channels"]) == 16
    assert json.loads(json.dumps(fx["seeded"])), "seeded frame present"


if __name__ == "__main__":
    test_generate_sma_page_fixtures()
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
