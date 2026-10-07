"""The "no MuPix data while S1 fires and the run is going" alarm.

Runs without MIDAS: the logic is pure and the ODB layer is driven through a fake client.

The timeline model (:class:`Daq`) follows what pinky does, measured live on 2026-10-07
for the 1000-event sequencer runs: the start transition takes ~6 s, Quads is busy in its
begin-of-run for ~5 s and writes no RCNT, Readout zeroes HIT 0 CNT before Quads starts
the FEBs, clean running lasts ~2 s, the State stays 3 through a ~4.5 s stop transition
during which Quads writes HIT 0 RATE = 0, and HIT 0 CNT stays at the run total
afterwards.
"""

from __future__ import annotations

import datetime
import math
import random

import pytest

from mdqm.dqm import mupix_no_data as ma

STOPPED, PAUSED, RUNNING = 1, 2, 3
CFG = ma.Settings.parse({})


class Daq:
    """The ODB as a function of time, for a sequence of runs."""

    def __init__(self, *, cycle=20.0, start_tr=6.1, clean=2.0, stop_tr=4.6, t0=1.79e9,
                 rate=1.7e6, s1=8e5, blocker=1.0, feb=True, readout=True, dead=False,
                 quads_alive=True, dies_at=None, readout_alive=True, readout_frozen=None):
        self.cycle, self.start_tr, self.clean, self.stop_tr = cycle, start_tr, clean, stop_tr
        self.t0, self.rate, self.s1, self.blocker, self.feb = t0, rate, s1, blocker, feb
        self.readout, self.dead, self.quads_alive, self.dies_at = readout, dead, quads_alive, \
            dies_at
        self.readout_alive, self.readout_frozen = readout_alive, readout_frozen

    def _run(self, t):
        k = math.floor((t - self.t0) / self.cycle)
        return k, self.t0 + k * self.cycle

    def _alive(self, t):
        return not self.dead and (self.dies_at is None or t < self.dies_at)

    def rcnt_at(self, t, start, running, stop):
        """HIT 0 RATE and HIT 0 CNT as last written by Quads at or before ``t``."""
        # Measured: first RCNT after the reset ~1.5 s before State 3 (rate still 0),
        # the next one ~0.5 s before State 3 with the full rate.
        feb_on = start + self.start_tr - 1.5
        feb_off = stop + 2.5
        hits_from = feb_on

        def count(u):
            if u < start + self.start_tr - 1.2:
                return None  # previous run's total, not modelled beyond "large"
            if not self._alive(u):
                end = min(u, feb_off, self.dies_at or u) if not self.dead else u
                return 3 if self.dead else int(self.rate * max(min(end, u) - hits_from, 0))
            return int(self.rate * max(min(u, feb_off) - hits_from, 0))

        rate = self.rate if feb_on + 1 <= t < feb_off and self._alive(t) else 0.0
        if self.dead and feb_on + 1 <= t < feb_off:
            rate = 0.0
        c = count(t)
        return rate, (10_000_000 if c is None else c)

    def observe(self, t):
        k, start = self._run(t)
        running = start + self.start_tr
        stop = running + self.clean
        stopped = stop + self.stop_tr
        if t < running:
            state, tip, run_no = STOPPED, 1, 100 + k
            running_since, stop_time = start - self.cycle + self.start_tr + self.clean \
                + self.stop_tr, start - self.cycle + self.start_tr + self.clean
        elif t < stop:
            state, tip, run_no, running_since, stop_time = RUNNING, 0, 100 + k, running, 0
        elif t < stopped:
            state, tip, run_no, running_since, stop_time = RUNNING, 2, 100 + k, running, stop
        else:
            state, tip, run_no, running_since, stop_time = STOPPED, 0, 100 + k, stopped, stop
        # Quads writes once a second, but not during its own begin-of-run.
        quads_busy = (start - 0.5, start + self.start_tr - 1.2)
        w = math.floor(t)
        if quads_busy[0] <= w < quads_busy[1]:
            w = math.floor(quads_busy[0])
        if not self.quads_alive:
            w = self.t0 - 100
        rate, count = self.rcnt_at(w, start, running, stop)
        if self.dead and t >= running:
            count = 3
        return ma.Observation(
            now=t, run_number=run_no, run_state=state, transition=tip,
            # Start/Stop time binary are whole seconds.
            start_time=float(math.floor(start)),
            stop_time=float(math.floor(stop_time)) or None,
            running_since=float(math.floor(running_since)),
            scint_hz=self.s1, mupix_rate=rate, mupix_count=count, mupix_written=float(w),
            readout_events=(5000 if self.readout else 0),
            # Readout's begin-of-run rewrites its statistics at the start of the start
            # transition; a Readout that is not running leaves an old write behind.
            readout_written=float(math.floor(start)) if self.readout_alive else self.t0 - 999,
            readout_frozen_s=self.readout_frozen if state == RUNNING and tip == 0 else None,
            feb_active=self.feb,
            blocker_open=self.blocker >= 0.5)


def drive(daq, seconds, *, period=5.0, phase=0.0, cfg=CFG, watch=None):
    watch = watch or ma.RateWatch()
    t = daq.t0 + phase
    decisions = []
    while t < daq.t0 + seconds:
        decisions.append((t, watch.step(daq.observe(t), cfg, period)))
        t += period
    return watch, decisions


def fired(decisions):
    return [t for t, d in decisions if d.action == "trigger"]


class TestSettings:
    def test_defaults(self):
        assert CFG.enabled and CFG.stop_sequencer and CFG.check_readout
        assert (CFG.scint_path, CFG.scint_indices) == (
            "/Equipment/WDScalers/Variables/S036", (0, 1, 2, 3, 4))
        assert (CFG.period_s, CFG.alarm_class) == (5.0, "DAQ Alarm")
        assert (CFG.rcnt_path, CFG.rate_index, CFG.count_index) == (
            "/Equipment/Quads/Variables/RCNT", 5, 4)
        assert (CFG.blocker_path, CFG.blocker_index) == ("/Equipment/EPICS/Variables/Measured", 30)
        assert (CFG.scint_min_hz, CFG.mupix_min_hz, CFG.bad_s) == (1000.0, 1000.0, 30.0)

    def test_bad_seconds_is_read_from_the_odb(self):
        assert ma.Settings.parse({"Bad seconds": 12}).bad_s == 12.0

    def test_malformed_values_fall_back(self):
        cfg = ma.Settings.parse({"Bad seconds": "lots", "Scint path": "  ",
                                 "Scint indices": None, "Good seconds": float("nan")})
        assert (cfg.bad_s, cfg.scint_path, cfg.scint_indices, cfg.good_s) == (
            30.0, "/Equipment/WDScalers/Variables/S036", (0, 1, 2, 3, 4), 10.0)

    def test_period_is_clamped(self):
        assert ma.Settings.parse({"Period seconds": 0}).period_s == 1.0
        assert ma.Settings.parse({"Period seconds": 99}).period_s == 15.0

    def test_one_element_index_array_reads_back_as_a_scalar(self):
        assert ma.Settings.parse({"Scint indices": 2}).scint_indices == (2,)

    def test_string_booleans(self):
        assert ma.Settings.parse({"Enabled": "n"}).enabled is False
        assert ma.Settings.parse({"Enabled": "y"}).enabled is True

    def test_clamped(self):
        cfg = ma.Settings.parse({"Bad seconds": -5, "Good seconds": 1e9,
                                 "MuPix min rate Hz": -1})
        assert (cfg.bad_s, cfg.good_s, cfg.mupix_min_hz) == (0.0, 3600.0, 0.0)

    def test_empty_path_switches_a_gate_off(self):
        assert ma.Settings.parse({"Blocker path": ""}).blocker_path == ""

    def test_alarm_name_fits_midas(self):
        assert len(ma.ALARM_NAME) <= 31


class TestSequencerRuns:
    """~10 s runs, 20 s apart, every poll phase."""

    @pytest.mark.parametrize("phase", [0.0, 0.7, 1.3, 2.1, 2.9, 3.6, 4.4])
    def test_healthy_never_fires(self, phase):
        _, ds = drive(Daq(), 3600, phase=phase)
        assert fired(ds) == []
        assert all(d.bad_s == 0 for _, d in ds)

    @pytest.mark.parametrize("clean", [0.2, 1.0, 2.0, 5.7])
    def test_healthy_never_fires_for_any_clean_running_length(self, clean):
        for phase in (0.0, 1.1, 2.5, 3.9):
            assert fired(drive(Daq(clean=clean), 1800, phase=phase)[1]) == []

    @pytest.mark.parametrize("phase", [0.0, 1.3, 2.9, 4.4])
    def test_dead_mupix_fires_within_a_few_runs(self, phase):
        _, ds = drive(Daq(dead=True), 600, phase=phase)
        times = fired(ds)
        assert len(times) == 1
        assert times[0] - Daq().t0 < 4 * 20 + 5  # three ~10 s runs, then the poll

    def test_readout_sending_nothing_fires(self):
        _, ds = drive(Daq(readout=False), 600)
        assert len(fired(ds)) == 1
        assert any(d.reason == "Readout sent no events" for _, d in ds)

    def test_readout_check_can_be_switched_off(self):
        cfg = ma.Settings.parse({"Check Readout events": False})
        assert fired(drive(Daq(readout=False), 600, cfg=cfg)[1]) == []

    @pytest.mark.parametrize("daq", [
        Daq(dead=True, blocker=0.0),
        Daq(dead=True, s1=500),
        Daq(dead=True, feb=False),
    ], ids=["blocker closed", "beam off", "FEB 0 out"])
    def test_gates(self, daq):
        assert fired(drive(daq, 600)[1]) == []

    def test_bad_seconds_setting_is_honoured(self):
        cfg = ma.Settings.parse({"Bad seconds": 90})
        t = fired(drive(Daq(dead=True), 900, cfg=cfg)[1])[0]
        assert t - Daq().t0 > 8 * 20

    def test_quads_frontend_down_fires_as_stale(self):
        _, ds = drive(Daq(quads_alive=False), 600)
        assert len(fired(ds)) == 1
        assert [d for t, d in ds if d.action][0].message.startswith("MuPix counters")

    def test_resets_after_good_runs(self):
        watch, ds = drive(Daq(dead=True), 300)
        assert watch.triggered
        healthy = Daq(t0=Daq().t0)
        resets = []
        t = Daq().t0 + 300
        while t < Daq().t0 + 600:
            d = watch.step(healthy.observe(t), CFG, 5.0)
            resets += [t] if d.action == "reset" else []
            t += 5.0
        assert len(resets) == 1 and not watch.triggered

    def test_message(self):
        _, ds = drive(Daq(dead=True), 600)
        d = [d for _, d in ds if d.action == "trigger"][0]
        assert d.message.startswith("No MuPix data run 1")
        assert len(d.message) <= ma.MAX_MESSAGE


class TestLongRun:
    def test_healthy_long_run_never_fires(self):
        daq = Daq(cycle=1200, clean=1140)
        assert fired(drive(daq, 2400, phase=1.7)[1]) == []

    def test_death_mid_run_fires_after_about_bad_seconds(self):
        t0 = 1.79e9
        daq = Daq(cycle=1200, clean=1140, dies_at=t0 + 300)
        times = fired(drive(daq, 1100, phase=1.7)[1])
        assert len(times) == 1
        # Bad time is counted in poll intervals, so it can fire up to one poll early.
        assert 25 <= times[0] - (t0 + 300) <= 45

    def test_stall_counts_at_most_a_few_polls(self):
        daq = Daq(cycle=1200, clean=1140, dead=True)
        watch = ma.RateWatch()
        watch.step(daq.observe(daq.t0 + 20), CFG, 5.0)
        d = watch.step(daq.observe(daq.t0 + 400), CFG, 5.0)
        assert d.bad_s <= 2 * 15


class TestClassify:
    def test_stop_transition_is_never_judged(self):
        daq = Daq()
        o = daq.observe(daq.t0 + 9.0)  # State 3, TIP 2, Quads writing rate 0
        assert (o.run_state, o.transition) == (RUNNING, 2)
        assert ma.classify(o, CFG).kind == "neutral"

    def test_start_transition_is_never_judged(self):
        daq = Daq()
        assert ma.classify(daq.observe(daq.t0 + 3.0), CFG).kind == "neutral"

    def test_waits_for_a_quads_write_after_running_began(self):
        o = ma.Observation(now=110.5, run_number=1, run_state=RUNNING, transition=0,
                           start_time=104.0, stop_time=None, running_since=110.0, scint_hz=1e6,
                           mupix_rate=0.0, mupix_count=500_000, mupix_written=110.0,
                           readout_events=10, readout_written=104.0, readout_frozen_s=0.0,
                           feb_active=True, blocker_open=True)
        assert ma.classify(o, CFG).reason == "waiting for Quads"
        assert ma.classify(ma.replace(o, mupix_written=111.0, now=111.2), CFG).reason == \
            "MuPix rate low"

    def test_unknown_gates_do_not_block(self):
        daq = Daq(dead=True)
        o = ma.replace(daq.observe(daq.t0 + 7.5), feb_active=None, blocker_open=None)
        assert ma.classify(o, CFG).kind == "bad"

    def test_old_stopped_run_is_not_judged(self):
        daq = Daq(dead=True, cycle=1000)
        o = daq.observe(daq.t0 + 500)
        assert ma.classify(o, CFG).reason == "stopped"

    def test_paused_is_neutral(self):
        daq = Daq(dead=True)
        o = ma.replace(daq.observe(daq.t0 + 7.5), run_state=PAUSED)
        assert ma.classify(o, CFG).kind == "neutral"


class TestDisableAndRearm:
    def test_disable_resets_a_raised_alarm(self):
        watch, _ = drive(Daq(dead=True), 300)
        d = watch.step(Daq(dead=True).observe(Daq().t0 + 300),
                       ma.Settings.parse({"Enabled": False}), 5.0)
        assert (d.verdict, d.action, watch.triggered) == ("disabled", "reset", False)

    def test_rearm_fires_again(self):
        daq = Daq(dead=True)
        watch, ds = drive(daq, 300)
        watch.rearm()
        t = daq.t0 + 300
        again = []
        while t < daq.t0 + 600:
            d = watch.step(daq.observe(t), CFG, 5.0)
            again += [t] if d.action == "trigger" else []
            t += 5.0
        assert len(again) == 1


# -- the ODB layer ---------------------------------------------------------------- #


class FakeClient:
    """The ODB as a dict; settings come from ``self.odb`` like any other key."""

    def __init__(self, odb, written_at):
        self.odb = dict(odb)
        self.written_at = dict(written_at)
        self.sets = {}
        self.alarms = []
        self.msgs = []

    def odb_exists(self, path):
        return path in self.odb

    def odb_get(self, path):
        if path not in self.odb:
            raise KeyError(path)
        return self.odb[path]

    def odb_set(self, path, value):
        self.odb[path] = value
        self.sets[path] = value

    def odb_last_update_time(self, path):
        return datetime.datetime.fromtimestamp(self.written_at[path])

    def trigger_internal_alarm(self, name, message, default_alarm_class="Alarm"):
        self.alarms.append(("trigger", name, message, default_alarm_class))
        self.odb[f"/Alarms/Alarms/{name}/Triggered"] = 1

    def reset_alarm(self, name):
        self.alarms.append(("reset", name))
        self.odb[f"/Alarms/Alarms/{name}/Triggered"] = 0

    def msg(self, message, is_error=False):
        self.msgs.append(message)


S036 = "/Equipment/WDScalers/Variables/S036"
RCNT = "/Equipment/Quads/Variables/RCNT"
SCINTS = [1_000_000, 300_000, 200_000, 100_000, 90_000] + [0] * 14


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def live_client(clock, *, rate=0, count=3, seq_running=True, seq_stopping=False,
                blocker=1.0, scints=SCINTS):
    now = clock.t
    rcnt = [0] * 13
    rcnt[4], rcnt[5] = count, rate
    measured = [0.0] * 40
    measured[30] = blocker
    return FakeClient({
        "/Runinfo/Run number": 3207,
        "/Runinfo/State": RUNNING,
        "/Runinfo/Transition in progress": 0,
        "/Runinfo/Start time binary": int(now - 100),
        "/Runinfo/Stop time binary": 0,
        RCNT: rcnt,
        S036: list(scints),
        "/Equipment/Readout/Statistics/Events sent": 5000,
        "/Equipment/Quads/Settings/DAQ/Links/FEBsActive": [True, True, False, False],
        "/Equipment/EPICS/Variables/Measured": measured,
        "/PySequencer/State/Running": seq_running,
        "/PySequencer/State/Stop after run": seq_stopping,
    }, {RCNT: now, S036: now, "/Runinfo/State": now - 90})


def make(c=None, clock=None, **kw):
    clock = clock or Clock(1.79e9)
    c = c or live_client(clock, **kw)
    return ma.MupixNoData("sma_analyzer", clock=clock), c, clock


def poll_n(alarm, client, clock, n=8):
    """Polls 5 s apart, with Quads and the scaler frontend writing in between."""
    for _ in range(n):
        alarm.maybe_poll(client)
        clock.t += 5
        client.written_at[RCNT] = client.written_at[S036] = clock.t


def settings(client, **values):
    for key, value in values.items():
        client.odb[f"{ma.SETTINGS_ROOT}/{key}"] = value


class TestMupixNoData:
    def test_seeds_its_own_tree_once(self):
        a, c, _ = make()
        assert a.seed(c) == len(ma.DEFAULTS)
        assert c.odb[f"{ma.SETTINGS_ROOT}/Bad seconds"] == 30.0
        c.odb[f"{ma.SETTINGS_ROOT}/Bad seconds"] = 12.0
        assert a.seed(c) == 0 and c.odb[f"{ma.SETTINGS_ROOT}/Bad seconds"] == 12.0

    def test_raises_the_alarm_and_stops_the_sequencer_after_the_run(self):
        a, c, clock = make()
        poll_n(a, c, clock)
        triggers = [x for x in c.alarms if x[0] == "trigger"]
        assert len(triggers) == 1
        assert triggers[0][1] == ma.ALARM_NAME and triggers[0][3] == "DAQ Alarm"
        assert c.sets == {"/PySequencer/Command/Stop after run": True}
        assert c.msgs and "will stop after this run" in c.msgs[0]
        assert a.status()["triggered"] and a.status()["triggers"] == 1

    def test_bad_seconds_from_the_odb(self):
        a, c, clock = make()
        settings(c, **{"Bad seconds": 90.0})
        poll_n(a, c, clock, n=8)
        assert c.alarms == []

    def test_polls_only_every_period(self):
        a, c, clock = make()
        assert a.maybe_poll(c) is not None
        clock.t += 2
        assert a.maybe_poll(c) is None
        clock.t += 3
        assert a.maybe_poll(c) is not None

    def test_leaves_a_sequencer_already_stopping_alone(self):
        a, c, clock = make(seq_stopping=True)
        poll_n(a, c, clock)
        assert c.sets == {}

    def test_leaves_an_idle_sequencer_alone(self):
        a, c, clock = make(seq_running=False)
        poll_n(a, c, clock)
        assert any(x[0] == "trigger" for x in c.alarms) and c.sets == {}

    def test_sequencer_action_can_be_switched_off(self):
        a, c, clock = make()
        settings(c, **{"Stop sequencer after run": False})
        poll_n(a, c, clock)
        assert any(x[0] == "trigger" for x in c.alarms) and c.sets == {}

    def test_alarm_class_from_the_odb(self):
        a, c, clock = make()
        settings(c, **{"Alarm class": "Alarm"})
        poll_n(a, c, clock)
        assert [x[3] for x in c.alarms if x[0] == "trigger"] == ["Alarm"]

    def test_healthy_mupix_never_alarms(self):
        a, c, clock = make(rate=1_700_000, count=10_000_000)
        poll_n(a, c, clock)
        assert c.alarms == [] and c.sets == {}

    def test_closed_blocker_is_read_from_epics(self):
        a, c, _ = make(blocker=0.0)
        assert a.poll(c).reason == "beam blocker closed"

    def test_every_scintillator_must_fire(self):
        a, c, _ = make(scints=[1_000_000, 300_000, 200_000, 100_000, 500] + [0] * 14)
        assert a.poll(c).reason == "scintillators below threshold"

    def test_a_disabled_channel_is_left_out(self):
        a, c, _ = make(scints=[1_000_000, 300_000, 200_000, 100_000, -1] + [0] * 14)
        assert a.poll(c).verdict == "bad"
        a, c, _ = make(scints=[-1] * 19)
        assert a.poll(c).reason == "no scintillator reading"

    def test_stale_scaler_rates_are_no_reading(self):
        a, c, clock = make()
        c.written_at[S036] = clock.t - 60
        assert a.poll(c).reason == "no scintillator reading"

    def test_missing_scaler_bank_is_no_reading(self):
        a, c, _ = make()
        del c.odb[S036]
        assert a.poll(c).reason == "no scintillator reading"

    def test_dry_run_changes_nothing(self):
        clock = Clock(1.79e9)
        c = live_client(clock)
        a = ma.MupixNoData(clock=clock, dry_run=True)
        poll_n(a, c, clock)
        assert a.watch.triggered and c.alarms == [] and c.sets == {}

    def test_adopts_an_alarm_left_raised(self):
        a, c, clock = make(rate=1_700_000, count=10_000_000)
        c.odb[f"/Alarms/Alarms/{ma.ALARM_NAME}/Triggered"] = 1
        poll_n(a, c, clock, n=4)
        assert ("reset", ma.ALARM_NAME) in c.alarms

    def test_reset_by_hand_rearms(self):
        a, c, clock = make()
        poll_n(a, c, clock)
        c.odb[f"/Alarms/Alarms/{ma.ALARM_NAME}/Triggered"] = 0  # shifter pressed Reset
        c.odb["/PySequencer/State/Stop after run"] = True
        poll_n(a, c, clock)
        assert [x[0] for x in c.alarms] == ["trigger", "trigger"]

    def test_odb_failures_never_raise(self):
        class Broken(FakeClient):
            def odb_get(self, path):
                raise RuntimeError("ODB gone")

            def odb_exists(self, path):
                raise RuntimeError("ODB gone")

        a, c, _ = make(c=Broken({}, {}))
        d = a.poll(c)
        assert d is not None and d.verdict == "neutral"

    def test_a_failing_alarm_call_still_stops_the_sequencer(self):
        class NoAlarm(FakeClient):
            def trigger_internal_alarm(self, *a, **k):
                raise RuntimeError("no alarm system")

        a, c, clock = make()
        c.__class__ = NoAlarm
        poll_n(a, c, clock)
        assert c.sets == {"/PySequencer/Command/Stop after run": True}

    def test_status_reports_the_cost(self):
        a, c, _ = make()
        a.poll(c)
        st = a.status()
        assert st["last_poll_ms"] is not None and st["verdict"] == "bad"


def test_random_phases_and_lengths_never_fire_when_healthy():
    rng = random.Random(7)
    for _ in range(40):
        daq = Daq(clean=rng.uniform(0.2, 5.7), stop_tr=rng.uniform(2.6, 7.8),
                  start_tr=rng.uniform(6.2, 7.4), cycle=rng.uniform(18, 40))
        assert fired(drive(daq, 1200, phase=rng.uniform(0, 5))[1]) == []


class TestAnalyzerWiring:
    def test_the_sma_analyzer_runs_it_and_reports_it(self):
        from mdqm.dqm import analyzer as A
        from mdqm.plugins.sma import SmaPlugin

        a = A.Analyzer(lambda s: SmaPlugin(s), rate=20.0)
        assert isinstance(a.mupix_no_data, ma.MupixNoData)
        assert a.mupix_no_data.client_name == "sma_analyzer"
        assert a.status()["mupix_no_data"]["triggered"] is False

    def test_other_plugins_do_not(self):
        from mdqm.dqm import analyzer as A
        from mdqm.plugins.wavedream import WaveDreamPlugin

        try:
            a = A.Analyzer(lambda s: WaveDreamPlugin(s), rate=20.0)
        except ImportError:
            pytest.skip("wavedream plugin needs wdscalers")
        assert a.mupix_no_data is None and a.status()["mupix_no_data"] is None

    def test_the_main_loop_polls_it(self):
        import inspect

        from mdqm.dqm import analyzer as A

        src = inspect.getsource(A.main)
        assert "analyzer.mupix_no_data.maybe_poll(client)" in src
        assert "analyzer.mupix_no_data.seed(client)" in src


class TestReviewFixes:
    def test_readout_not_running_fires(self):
        _, ds = drive(Daq(readout_alive=False), 600)
        assert len(fired(ds)) == 1
        assert any(d.reason == "Readout not running" for _, d in ds)

    def test_readout_stalled_mid_run_fires(self):
        daq = Daq(cycle=1200, clean=1140, readout_frozen=20.0)
        _, ds = drive(daq, 600, phase=1.7)
        assert len(fired(ds)) == 1
        assert any(d.reason == "Readout stalled" for _, d in ds)

    def test_a_readout_paused_less_than_max_age_is_fine(self):
        daq = Daq(cycle=1200, clean=1140, readout_frozen=10.0)
        assert fired(drive(daq, 600, phase=1.7)[1]) == []

    def _resumed(self, since_resume, quads_written):
        resume = 1.79e9 + 500
        return ma.Observation(
            now=resume + since_resume, run_number=7, run_state=RUNNING, transition=0,
            start_time=1.79e9, stop_time=None, running_since=resume, scint_hz=1e6,
            mupix_rate=1.7e6, mupix_count=10**8, mupix_written=quads_written,
            readout_events=10, readout_written=1.79e9, readout_frozen_s=0.0,
            feb_active=True, blocker_open=True)

    def test_resume_after_a_long_pause_waits_for_quads(self):
        # Quads last wrote 60 s ago, before the pause.
        assert ma.classify(self._resumed(1.0, 1.79e9 + 440), CFG).reason == "waiting for Quads"

    def test_resume_then_quads_writes(self):
        assert ma.classify(self._resumed(1.5, 1.79e9 + 501), CFG).kind == "good"

    def test_resume_but_quads_never_writes_is_stale(self):
        assert ma.classify(self._resumed(4.0, 1.79e9 + 440), CFG).reason == \
            "MuPix counters stale"

    def test_no_double_count_when_the_run_number_is_unreadable(self):
        w = ma.RateWatch()
        daq = Daq(dead=True, cycle=1200, clean=1140)
        w.step(daq.observe(daq.t0 + 20), CFG, 5.0)
        before = w.bad_s
        d = w.step(ma.replace(daq.observe(daq.t0 + 25), run_number=None), CFG, 5.0)
        assert d.reason == "run state unreadable"
        d = w.step(daq.observe(daq.t0 + 30), CFG, 5.0)
        assert d.bad_s == pytest.approx(before + 10)

    def test_zero_scintillators_while_running_are_kept_for_the_end_check(self):
        w = ma.RateWatch()
        daq = Daq(dead=True)
        w.step(ma.replace(daq.observe(daq.t0 + 7.0), scint_hz=0.0), CFG, 5.0)
        # Stopped, beam back on now: the run is judged on the 0 Hz it ran with.
        d = w.step(daq.observe(daq.t0 + 15.0), CFG, 5.0)
        assert d.reason == "scintillators below threshold"


class TestPollingAndLogging:
    def test_polls_every_second_around_transitions(self):
        a, c, clock = make()
        c.odb["/Runinfo/Transition in progress"] = 2
        assert a.maybe_poll(c).reason == "transition in progress"
        clock.t += 1.0
        assert a.maybe_poll(c) is not None

    def test_gate_closed_for_minutes_is_reported_once(self, capsys):
        a, c, clock = make(scints=[1_000_000, 300_000, 200_000, 100_000, 500] + [0] * 14)
        poll_n(a, c, clock, n=70)
        err = capsys.readouterr().err
        assert err.count("blind for") == 1 and "scintillators below threshold" in err
        assert a.status()["gate"] == "scintillators below threshold"

    def test_no_rearm_loop_when_the_alarm_system_is_off(self):
        class Off(FakeClient):
            def trigger_internal_alarm(self, name, message, default_alarm_class="Alarm"):
                self.alarms.append(("trigger", name, message, default_alarm_class))

        a, c, clock = make()
        c.__class__ = Off
        poll_n(a, c, clock, n=30)
        assert [x[0] for x in c.alarms] == ["trigger"]
