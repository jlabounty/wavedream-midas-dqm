"""Alarm when the run is going, the scintillators fire, and no MuPix data arrives.

Polled from the ``sma_analyzer`` main loop every ``Period seconds`` (see
:func:`mdqm.dqm.analyzer.main`). It reads only ODB values, never event data, so it costs
about twenty ODB reads per poll and nothing per frame. It is the DQM's because the DQM
can be restarted while data is being taken; the readout frontends should not be.
It has bitten us: the switching board stops receiving hits (HIT 0 CNT stays at 3 for the
whole run) or Readout stops sending events, while the links stay locked, the HV and LV
stay nominal and nothing else on the DAQ goes red.

MIDAS cannot express this as an evaluated alarm. An evaluated alarm compares one key
with one constant, and ``&`` in a condition is a bitwise test, not a logical AND
(``midas/src/alarm.cxx`` ``al_evaluate_condition``). So the condition is evaluated
here and raised as an internal alarm.

What the musip counters do (``quads_config_fe.cpp`` RCNT fill; ``readout_fe.cpp``)
--------------------------------------------------------------------------------

``/Equipment/Quads/Variables/RCNT``, written by the Quads frontend about once a second:

``[5]`` HIT 0 RATE
    MuPix hits per second reaching the switching board from FEB 0 (MuPix only), over
    the last 1 s gate.
``[4]`` HIT 0 CNT
    The hits so far in this run. Readout zeroes it at the start of every run, before
    Quads starts the FEBs, and it stays frozen at the run total after the run ends.

Both are meaningless during the transitions. ``/Runinfo/State`` stays 3 (running)
through the whole stop transition, while Quads has already stopped the FEBs and writes
HIT 0 RATE = 0. During the start transition, HIT 0 CNT is the previous run's total until
the reset. The current ~10 s sequencer runs spend most of their time in these
transitions: only 0.2-5 s is clean running.

Two checks
----------

During the run
    Only during clean running (``State`` 3 and ``Transition in progress`` 0), and only
    once Quads has written RCNT after running began. Bad if HIT 0 RATE <
    ``MuPix min rate Hz``, if Readout has sent no events this run or its count has not
    moved for ``Max MuPix age seconds``, or if RCNT is older than that.
At the end of the run
    Once per run, while stopped. Bad if the run's HIT 0 CNT < ``MuPix min rate Hz`` x
    the run length (start to stop), or if Readout sent no events. This is the check
    that sees the short sequencer runs.

Both also flag a Readout that took no part in the run: its begin-of-run rewrites
``Events sent``, so an older write means it is not running or not connected.

Both checks are skipped (neutral) unless every scintillator in ``Scint indices`` of
``Scint path`` (S1-S5, WD036 channels 0-4 in ``/Equipment/WDScalers/Variables/S036``,
written by the scaler frontend every 5 s) is at or above ``Scint min rate Hz``. Rates
older than ``Max scaler age seconds`` count as no reading. They are also
skipped when the beam blocker is closed, or when FEB 0 is switched off in ``FEBsActive``.
That covers beam off and MuPix deliberately out. Requiring all five, not just S1, keeps
one noisy channel from opening the gate; a closed blocker once left S1 alone at ~56 kHz.
Channels disabled in the scaler settings (-1) are left out. For the end-of-run check, the
rate used is the highest seen while that run was going.

A bad verdict counts the run's time not yet counted, at most ``max(3 x period, 15 s)``
per poll, as bad running time. So a dead 10 s sequencer run adds about 10 s and a long
run adds one poll period per bad poll. A good verdict clears the total. Neutral polls
leave it alone, so the bad time of several runs adds up. The alarm fires at
``Bad seconds`` and resets itself after ``Good seconds`` counted the same way. A
shifter who resets it on the Alarms page re-arms it: it fires again after another
``Bad seconds``.

When it fires, and ``Stop sequencer after run`` is set, it presses "Stop after run" on
the PySequencer: the current run ends normally at its event count and the script then
exits. It never stops a run itself, and never restarts the script.

Settings live in ``/DQM/SMA/MuPix no data`` and are read on every poll.
``mdqm-mupix-no-data`` runs the identical check by hand; see :func:`main`.

This module does not import ``midas`` at module level: the client is passed in and only
the methods named in :class:`MupixNoData` are used, so it is testable without MIDAS.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass, replace
from typing import Any, Final

from . import settings as odb_settings

log = logging.getLogger(__name__)

ALARM_NAME: Final = "MuPix no data"
"""Internal alarm name. MIDAS truncates names at 31 characters."""

SETTINGS_ROOT: Final = "/DQM/SMA/MuPix no data"
"""Its own tree, read only by this poll, not the analyzer's every-cycle settings read."""

STATE_STOPPED: Final = 1
STATE_RUNNING: Final = 3
"""``/Runinfo/State`` values (``midas.h``)."""

MAX_MESSAGE: Final = 79
"""``/Alarms/Alarms/<name>/Alarm message`` is an 80-byte string."""

QUADS_WRITE_GRACE_S: Final = 3.0
"""How long to wait for Quads to write RCNT after running began or the stop began.

Quads writes about once a second, except inside its own transitions. It wrote half a
second before State turned running in every sampled run, and it resumes within a second
of a resume (it does not write while paused)."""

END_OF_RUN_MAX_AGE_S: Final = 120.0
"""A run that ended longer ago than this is not judged (e.g. right after a restart)."""

DEFAULTS: Final[dict[str, Any]] = {
    "Enabled": True,
    "Period seconds": 5.0,
    "Scint path": "/Equipment/WDScalers/Variables/S036",
    "Scint indices": [0, 1, 2, 3, 4],
    "Scint min rate Hz": 1000.0,
    "Max scaler age seconds": 15.0,
    "MuPix min rate Hz": 1000.0,
    "Bad seconds": 30.0,
    "Good seconds": 10.0,
    "Max MuPix age seconds": 15.0,
    "Check Readout events": True,
    "Stop sequencer after run": True,
    "RCNT path": "/Equipment/Quads/Variables/RCNT",
    "RCNT rate index": 5,
    "RCNT count index": 4,
    "Readout events path": "/Equipment/Readout/Statistics/Events sent",
    "FEB active path": "/Equipment/Quads/Settings/DAQ/Links/FEBsActive",
    "FEB active index": 0,
    "Blocker path": "/Equipment/EPICS/Variables/Measured",
    "Blocker index": 30,
    "Sequencer path": "/PySequencer",
    "Alarm class": "DAQ Alarm",
}
"""Seeded into the ODB when absent, never overwritten afterwards."""


@dataclass(frozen=True)
class Settings:
    enabled: bool
    period_s: float
    scint_path: str
    scint_indices: tuple[int, ...]
    scint_min_hz: float
    max_scaler_age_s: float
    mupix_min_hz: float
    bad_s: float
    good_s: float
    max_age_s: float
    check_readout: bool
    stop_sequencer: bool
    rcnt_path: str
    rate_index: int
    count_index: int
    readout_path: str
    feb_path: str
    feb_index: int
    blocker_path: str
    blocker_index: int
    sequencer_path: str
    alarm_class: str

    @classmethod
    def parse(cls, raw: dict | None) -> Settings:
        """Read the ODB directory, falling back to the default for anything malformed.

        Times and rates are clamped to non-negative values and times to at most an
        hour. An empty path switches off the check that uses it (the FEB and blocker
        gates, the Readout check).
        """
        raw = raw if isinstance(raw, dict) else {}

        def get(key, kind):
            value = raw.get(key, DEFAULTS[key])
            try:
                if kind is bool:
                    if isinstance(value, str):
                        # A hand-made STRING key: bool("n") would be True.
                        return value.strip().lower() in ("y", "yes", "true", "1")
                    return bool(value)
                if kind is str:
                    return str(value).strip().rstrip("/")
                result = kind(value)
                if kind is float and not math.isfinite(result):
                    raise ValueError("not finite")
                return result
            except (TypeError, ValueError, OverflowError):
                log.warning("MuPix alarm: bad value %r for %r, using %r",
                            value, key, DEFAULTS[key])
                return DEFAULTS[key]

        def seconds(key):
            return min(max(get(key, float), 0.0), 3600.0)

        def index(key):
            return max(get(key, int), 0)

        def indices(key):
            value = raw.get(key, DEFAULTS[key])
            # MIDAS returns a one-element array as a bare scalar.
            value = value if isinstance(value, list | tuple) else [value]
            try:
                out = tuple(sorted({int(v) for v in value if int(v) >= 0}))
            except (TypeError, ValueError):
                out = ()
            if not out:
                log.warning("MuPix alarm: bad value %r for %r, using %r",
                            value, key, DEFAULTS[key])
                out = tuple(DEFAULTS[key])
            return out

        return cls(
            enabled=get("Enabled", bool),
            period_s=min(max(get("Period seconds", float), 1.0), 15.0),
            scint_path=get("Scint path", str) or DEFAULTS["Scint path"],
            scint_indices=indices("Scint indices"),
            scint_min_hz=max(get("Scint min rate Hz", float), 0.0),
            max_scaler_age_s=seconds("Max scaler age seconds"),
            mupix_min_hz=max(get("MuPix min rate Hz", float), 0.0),
            bad_s=seconds("Bad seconds"),
            good_s=seconds("Good seconds"),
            max_age_s=seconds("Max MuPix age seconds"),
            check_readout=get("Check Readout events", bool),
            stop_sequencer=get("Stop sequencer after run", bool),
            rcnt_path=get("RCNT path", str) or DEFAULTS["RCNT path"],
            rate_index=index("RCNT rate index"),
            count_index=index("RCNT count index"),
            readout_path=get("Readout events path", str),
            feb_path=get("FEB active path", str),
            feb_index=index("FEB active index"),
            blocker_path=get("Blocker path", str),
            blocker_index=index("Blocker index"),
            sequencer_path=get("Sequencer path", str) or DEFAULTS["Sequencer path"],
            alarm_class=get("Alarm class", str) or DEFAULTS["Alarm class"],
        )


@dataclass(frozen=True)
class Observation:
    """Everything one poll looks at. Times are Unix seconds; None = could not read."""

    now: float
    run_number: int | None
    run_state: int | None
    transition: int | None  # /Runinfo/Transition in progress: 0 none, 1 start, 2 stop
    start_time: float | None  # /Runinfo/Start time binary: when the start transition began
    stop_time: float | None  # /Runinfo/Stop time binary: when the stop transition began
    running_since: float | None  # last write of /Runinfo/State
    scint_hz: float | None  # the lowest of the enabled scintillator rates
    mupix_rate: float | None
    mupix_count: int | None
    mupix_written: float | None
    readout_events: int | None
    readout_written: float | None  # last write of the Readout events key
    readout_frozen_s: float | None  # clean running time with Readout events unchanged
    feb_active: bool | None
    blocker_open: bool | None


@dataclass(frozen=True)
class Verdict:
    kind: str  # "neutral", "bad" or "good"
    reason: str
    check: str | None = None  # "run" or "end" when not neutral
    span: tuple[float, float] | None = None  # run time this verdict covers


@dataclass(frozen=True)
class Decision:
    verdict: str  # "neutral", "bad", "good" or "disabled"
    reason: str
    action: str | None  # "trigger", "reset" or None
    message: str
    bad_s: float
    good_s: float


def _khz(hz: float | None) -> str:
    return "?" if hz is None else f"{hz / 1e3:.0f} kHz"


def classify(obs: Observation, cfg: Settings) -> Verdict:
    """The verdict for one poll. ``obs.scint_hz`` is the scintillator rate to use."""
    if obs.run_state is None or obs.transition is None or obs.start_time is None \
            or obs.run_number is None:
        return Verdict("neutral", "run state unreadable")
    if obs.transition != 0:
        return Verdict("neutral", "transition in progress")
    if obs.run_state == STATE_RUNNING:
        check = "run"
    elif obs.run_state == STATE_STOPPED and obs.stop_time is not None \
            and obs.stop_time >= obs.start_time:
        if obs.now - obs.stop_time > END_OF_RUN_MAX_AGE_S:
            return Verdict("neutral", "stopped")
        check = "end"
    else:
        return Verdict("neutral", "not running")

    if obs.feb_active is False:
        return Verdict("neutral", "FEB 0 not active")
    if obs.blocker_open is False:
        return Verdict("neutral", "beam blocker closed")
    if obs.scint_hz is None:
        return Verdict("neutral", "no scintillator reading")
    if obs.scint_hz < cfg.scint_min_hz:
        return Verdict("neutral", "scintillators below threshold")

    end = obs.now if check == "run" else obs.stop_time
    span = (obs.start_time, end)
    if obs.mupix_written is None or obs.mupix_rate is None or obs.mupix_count is None:
        return Verdict("bad", "MuPix counters unreadable", check, span)
    # The counters must have been written after running began (in the run: strictly,
    # since last_written has 1 s resolution) or after the stop began (at the end).
    # Wait for that, but only as long as a live Quads takes: after a long pause the
    # old value is not stale yet, while a dead Quads must still be caught in a 10 s run.
    ref = obs.running_since if check == "run" else obs.stop_time
    stale = obs.now - obs.mupix_written > cfg.max_age_s
    if ref is None:
        return Verdict("neutral", "waiting for Quads")
    if obs.mupix_written < ref or (check == "run" and obs.mupix_written == ref):
        if stale and obs.now - ref > QUADS_WRITE_GRACE_S:
            return Verdict("bad", "MuPix counters stale", check, span)
        return Verdict("neutral", "waiting for Quads")
    if stale:
        return Verdict("bad", "MuPix counters stale", check, span)

    if check == "run" and obs.mupix_rate < cfg.mupix_min_hz:
        return Verdict("bad", "MuPix rate low", check, span)
    if check == "end" and obs.mupix_count < cfg.mupix_min_hz * (obs.stop_time - obs.start_time):
        return Verdict("bad", "MuPix hits low", check, span)

    if cfg.check_readout and cfg.readout_path:
        # Readout's begin-of-run zeroes and rewrites its statistics, so an older write
        # means Readout did not take part in this run (not running, or not connected).
        if obs.readout_written is not None and obs.readout_written < obs.start_time:
            return Verdict("bad", "Readout not running", check, span)
        if obs.readout_events == 0:
            return Verdict("bad", "Readout sent no events", check, span)
        if check == "run" and obs.readout_frozen_s is not None \
                and obs.readout_frozen_s > cfg.max_age_s:
            return Verdict("bad", "Readout stalled", check, span)
    return Verdict("good", "ok", check, span)


GATE_REASONS: Final = frozenset({
    "FEB 0 not active", "beam blocker closed", "no scintillator reading",
    "scintillators below threshold",
})
"""Neutral verdicts that keep the alarm blind while a run is going."""


def _message(v: Verdict, obs: Observation, bad_s: float) -> str:
    where = "" if obs.run_number is None else f" run {obs.run_number}"
    if v.reason == "MuPix rate low":
        text = f"No MuPix data{where}: {obs.mupix_rate:.0f} Hz, scints >= {_khz(obs.scint_hz)}"
    elif v.reason == "MuPix hits low":
        text = f"No MuPix data{where}: {obs.mupix_count} hits, scints >= {_khz(obs.scint_hz)}"
    elif v.reason == "Readout sent no events":
        text = f"No MuPix data{where}: Readout sent 0 events, scints >= {_khz(obs.scint_hz)}"
    elif v.reason == "Readout not running":
        text = f"No MuPix data{where}: Readout not in this run, scints >= {_khz(obs.scint_hz)}"
    elif v.reason == "Readout stalled":
        text = (f"No MuPix data{where}: Readout stalled {obs.readout_frozen_s:.0f} s, "
                f"scints >= {_khz(obs.scint_hz)}")
    elif v.reason == "MuPix counters stale":
        text = (f"MuPix counters (Quads RCNT) not updated for "
                f"{obs.now - obs.mupix_written:.0f} s, scints >= {_khz(obs.scint_hz)}")
    else:
        text = f"MuPix counters unreadable{where}, scints >= {_khz(obs.scint_hz)}"
    return f"{text}, {bad_s:.0f} s"[:MAX_MESSAGE]


class RateWatch:
    """The debounce state machine. Pure: feed it observations, it returns decisions."""

    def __init__(self, triggered: bool = False) -> None:
        self.triggered = triggered
        self.bad_s = 0.0
        self.good_s = 0.0
        self._run: int | None = None
        self._counted_until = 0.0
        self._run_scint: float | None = None  # highest seen while self._run was running
        self._judged_end: int | None = None

    def rearm(self) -> None:
        """The alarm was reset by hand: start counting again from zero."""
        self.triggered = False
        self.bad_s = self.good_s = 0.0

    def _follow_run(self, obs: Observation) -> None:
        if obs.run_number is None:
            return  # unreadable: classify() says neutral; keep the run we know
        if obs.run_number != self._run:
            self._run = obs.run_number
            self._counted_until = obs.start_time or 0.0
            self._run_scint = None
        if obs.run_state == STATE_RUNNING and obs.scint_hz is not None:
            self._run_scint = max(self._run_scint or 0.0, obs.scint_hz)

    def step(self, obs: Observation, cfg: Settings, period_s: float) -> Decision:
        if not cfg.enabled:
            self.bad_s = self.good_s = 0.0
            action = "reset" if self.triggered else None
            self.triggered = False
            return Decision("disabled", "disabled in ODB", action, "", 0.0, 0.0)

        self._follow_run(obs)
        if obs.run_state == STATE_STOPPED and obs.run_number is not None:
            if self._judged_end == obs.run_number:
                return Decision("neutral", "stopped", None, "", self.bad_s, self.good_s)
            # Judge the run on the beam while it ran, not on the beam now.
            if self._run_scint is not None:
                obs = replace(obs, scint_hz=self._run_scint)

        v = classify(obs, cfg)
        if v.kind == "neutral":
            return Decision(v.kind, v.reason, None, "", self.bad_s, self.good_s)
        if v.check == "end":
            self._judged_end = obs.run_number

        # Count the part of this run not counted yet, but never more than a few polls'
        # worth at once: an analyzer that stalled, or started in the middle of a long
        # run, must not turn one bad poll into minutes.
        start, end = v.span
        step = min(max(end - max(self._counted_until, start), 0.0), max(3 * period_s, 15.0))
        self._counted_until = max(self._counted_until, end)

        action = None
        if v.kind == "bad":
            self.bad_s += step
            self.good_s = 0.0
            if not self.triggered and self.bad_s >= cfg.bad_s:
                self.triggered = True
                action = "trigger"
        else:
            self.good_s += step
            if not self.triggered:
                self.bad_s = 0.0
            elif self.good_s >= cfg.good_s:
                self.triggered = False
                self.bad_s = 0.0
                action = "reset"
        message = _message(v, obs, self.bad_s) if v.kind == "bad" else ""
        return Decision(v.kind, v.reason, action, message, self.bad_s, self.good_s)


class MupixNoData:
    """Binds :class:`RateWatch` to a MIDAS client; one per analyzer process.

    The client is passed to every call rather than kept, because the analyzer
    reconnects with a new one after a MIDAS bounce. Client methods used: ``odb_get``,
    ``odb_set``, ``odb_exists``, ``odb_last_update_time``, ``trigger_internal_alarm``,
    ``reset_alarm`` and ``msg``. Every call is wrapped: a failure here is printed and
    never reaches the analyzer loop.
    """

    def __init__(self, client_name: str = "sma_analyzer", root: str = SETTINGS_ROOT,
                 dry_run: bool = False, clock=time.time) -> None:
        self.client_name = client_name
        self.root = root
        self.dry_run = dry_run
        self._clock = clock
        self.watch = RateWatch()
        self._adopted = False
        #: True once the alarm is really raised in the ODB (by us or adopted). Only
        #: then does Triggered = 0 mean a shifter reset it.
        self._raised = False
        self._next_poll = 0.0
        #: (run, Readout events, first seen) during clean running, for "stalled".
        self._readout_seen: tuple[int, int, float] | None = None
        #: Gate reason and since when it has kept the alarm blind during running.
        self._gate: tuple[str, float] | None = None
        self._gate_warned = False
        self.last: Decision | None = None
        self.last_reason: str | None = None
        self.last_poll_at: float | None = None
        self.last_poll_ms: float | None = None
        self.triggers = 0
        self._error: str | None = None

    # -- plumbing ---------------------------------------------------------------

    def _say(self, text: str, error: bool = False) -> None:
        print(f"{self.client_name}: {ALARM_NAME}: {text}",
              file=sys.stderr if error else sys.stdout, flush=True)

    def seed(self, client) -> int:
        """Create any missing setting, leaving existing ones alone."""
        try:
            return odb_settings.seed(client, self.root, DEFAULTS)
        except Exception as exc:  # noqa: BLE001
            self._say(f"could not seed {self.root}: {exc}", error=True)
            return 0

    def read_settings(self, client) -> dict:
        return odb_settings.read(client, self.root, DEFAULTS)

    def _alarm_raised(self, client) -> bool:
        path = f"/Alarms/Alarms/{ALARM_NAME}/Triggered"
        try:
            return bool(client.odb_exists(path) and client.odb_get(path))
        except Exception:
            return False

    @staticmethod
    def _get(client, path: str, index: int | None = None):
        """One ODB value (element ``index`` of an array), or None."""
        if not path:
            return None
        try:
            value = client.odb_get(path)
            if index is not None:
                value = value if isinstance(value, list | tuple) else [value]
                value = value[index]
            return value
        except Exception:
            return None

    @staticmethod
    def _written(client, path: str) -> float | None:
        try:
            return client.odb_last_update_time(path).timestamp()
        except Exception:
            return None

    # -- one poll ---------------------------------------------------------------

    def observe(self, client, cfg: Settings) -> Observation:
        def num(value, kind=float):
            try:
                return None if value is None else kind(value)
            except (TypeError, ValueError):
                return None

        now = self._clock()
        # Disabled channels read -1 and are left out; none left means no reading.
        values = self._get(client, cfg.scint_path)
        values = values if isinstance(values, list | tuple) else \
            ([] if values is None else [values])
        rates = [float(values[i]) for i in cfg.scint_indices
                 if i < len(values) and num(values[i]) is not None and values[i] >= 0]
        scint = min(rates) if rates else None
        written = self._written(client, cfg.scint_path)
        if written is None or now - written > cfg.max_scaler_age_s:
            scint = None

        run_number = num(self._get(client, "/Runinfo/Run number"), int)
        run_state = num(self._get(client, "/Runinfo/State"), int)
        transition = num(self._get(client, "/Runinfo/Transition in progress"), int)
        events = num(self._get(client, cfg.readout_path), int)
        frozen = None
        if run_state == STATE_RUNNING and transition == 0 and None not in (run_number, events):
            seen = self._readout_seen
            if seen is None or seen[0] != run_number or seen[1] != events:
                self._readout_seen = (run_number, events, now)
            frozen = now - self._readout_seen[2]
        else:
            self._readout_seen = None

        rcnt = self._get(client, cfg.rcnt_path)
        rcnt = rcnt if isinstance(rcnt, list | tuple) else None
        rate = num(rcnt[cfg.rate_index]) if rcnt and cfg.rate_index < len(rcnt) else None
        count = num(rcnt[cfg.count_index], int) if rcnt and cfg.count_index < len(rcnt) \
            else None
        feb = self._get(client, cfg.feb_path, cfg.feb_index)
        blocker = num(self._get(client, cfg.blocker_path, cfg.blocker_index))
        return Observation(
            now=now,
            run_number=run_number,
            run_state=run_state,
            transition=transition,
            start_time=num(self._get(client, "/Runinfo/Start time binary")) or None,
            stop_time=num(self._get(client, "/Runinfo/Stop time binary")) or None,
            running_since=self._written(client, "/Runinfo/State"),
            scint_hz=scint,
            mupix_rate=rate,
            mupix_count=count,
            mupix_written=self._written(client, cfg.rcnt_path) if rcnt else None,
            readout_events=events,
            readout_written=self._written(client, cfg.readout_path) if cfg.readout_path
            else None,
            readout_frozen_s=frozen,
            feb_active=None if feb is None else bool(feb),
            blocker_open=None if blocker is None else blocker >= 0.5,
        )

    def maybe_poll(self, client) -> Decision | None:
        """Poll if ``Period seconds`` has passed since the last poll. Never raises."""
        if self._clock() < self._next_poll:
            return None
        return self.poll(client)

    def poll(self, client, raw: dict | None = None) -> Decision | None:
        """One poll. ``raw`` overrides the ODB settings (the CLI). Never raises."""
        t0 = time.perf_counter()
        try:
            cfg = Settings.parse(self.read_settings(client) if raw is None else raw)
            self._next_poll = self._clock() + cfg.period_s
            if not self._adopted:
                # An alarm left raised by an earlier process: adopt it, so that good
                # running resets it.
                self.watch.triggered = self._raised = self._alarm_raised(client)
                self._adopted = True
            elif self.watch.triggered and self._raised and not self._alarm_raised(client):
                self._say("reset by hand; re-armed")
                self._raised = False
                self.watch.rearm()
            obs = self.observe(client, cfg)
            decision = self.watch.step(obs, cfg, cfg.period_s)
            if decision.reason in ("transition in progress", "waiting for Quads"):
                # Look again soon, so the short stopped gap between sequencer runs,
                # where the end-of-run check happens, is never stepped over.
                self._next_poll = min(self._next_poll, self._clock() + 1.0)
            self._watch_gate(obs, decision)
            if decision.verdict != "neutral" and decision.reason != self.last_reason:
                self._say(f"{decision.verdict} ({decision.reason}), "
                          f"bad {decision.bad_s:.0f} s, good {decision.good_s:.0f} s")
                self.last_reason = decision.reason
            self.last = decision
            if decision.action == "trigger":
                self._trigger(client, decision, cfg)
            elif decision.action == "reset":
                self._reset(client)
            self._error = None
            return decision
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            if error != self._error:
                self._error = error
                self._say(f"poll failed: {error}", error=True)
            return None
        finally:
            self.last_poll_at = self._clock()
            self.last_poll_ms = round((time.perf_counter() - t0) * 1e3, 2)

    #: Warn once when a gate keeps the alarm blind for this long while running.
    GATE_WARN_S = 300.0

    def _watch_gate(self, obs: Observation, decision: Decision) -> None:
        """Say so when a gate (scintillators, blocker, FEB) keeps the alarm blind."""
        if obs.run_state != STATE_RUNNING or obs.transition != 0:
            return  # only clean running counts, either way
        if decision.reason not in GATE_REASONS:
            if self._gate_warned:
                self._say("checking again: the gate has opened")
            self._gate, self._gate_warned = None, False
            return
        if self._gate is None or self._gate[0] != decision.reason:
            self._gate = (decision.reason, obs.now)
        elif not self._gate_warned and obs.now - self._gate[1] >= self.GATE_WARN_S:
            self._gate_warned = True
            self._say(f"blind for {obs.now - self._gate[1]:.0f} s of running: "
                      f"{decision.reason}", error=True)

    def status(self) -> dict:
        """For ``dqm::status``."""
        d = self.last
        return {
            "gate": None if self._gate is None else self._gate[0],
            "triggered": self.watch.triggered,
            "verdict": None if d is None else d.verdict,
            "reason": None if d is None else d.reason,
            "bad_s": round(self.watch.bad_s, 1),
            "good_s": round(self.watch.good_s, 1),
            "triggers": self.triggers,
            "last_poll_at": self.last_poll_at,
            "last_poll_ms": self.last_poll_ms,
            "error": self._error,
            "dry_run": self.dry_run,
        }

    # -- actions ----------------------------------------------------------------

    def _trigger(self, client, decision: Decision, cfg: Settings) -> None:
        self.triggers += 1
        self._say(f"TRIGGER{' (dry run)' if self.dry_run else ''}: {decision.message}",
                  error=True)
        if self.dry_run:
            return
        try:
            client.trigger_internal_alarm(ALARM_NAME, decision.message,
                                          default_alarm_class=cfg.alarm_class)
            # Not raised when the alarm system is off or Online Mode is 0; reading it
            # back keeps "Triggered = 0" from looking like a shifter's reset.
            self._raised = self._alarm_raised(client)
        except Exception as exc:  # noqa: BLE001
            self._say(f"could not raise the alarm: {exc}", error=True)
        if cfg.stop_sequencer:
            self._stop_sequencer_after_run(client, cfg.sequencer_path)

    def _stop_sequencer_after_run(self, client, base: str) -> None:
        """Press "Stop after run" on the PySequencer, if a script is running.

        Not "Pause script": the script is what stops each run at its event count, so a
        script paused mid-run would leave the run going until somebody stopped it. With
        "Stop after run" the current run ends normally and the script then exits. The
        operator script takes its next config from the run DB, so restarting it with
        "Start script" loses nothing.
        """
        try:
            running = client.odb_get(f"{base}/State/Running")
            already = client.odb_get(f"{base}/State/Stop after run")
        except Exception:
            self._say(f"no sequencer at {base}, nothing to stop")
            return
        if not running or already:
            return
        try:
            client.odb_set(f"{base}/Command/Stop after run", True)
            client.msg(f"{ALARM_NAME}: the sequencer ({base}) will stop after this run. "
                       "Check MuPix, then press Start script on the Sequencer page.",
                       is_error=True)
        except Exception as exc:  # noqa: BLE001
            self._say(f"could not stop the sequencer after the run: {exc}", error=True)

    def _reset(self, client) -> None:
        self._say(f"MuPix data back, alarm reset{' (dry run)' if self.dry_run else ''}")
        if self.dry_run:
            return
        try:
            client.reset_alarm(ALARM_NAME)
            self._raised = False
        except Exception as exc:  # noqa: BLE001
            self._say(f"could not reset the alarm: {exc}", error=True)


def main(argv: list[str] | None = None) -> int:
    """``mdqm-mupix-no-data``: run the check by hand, against the live ODB.

    ``--once`` prints the settings, what one poll sees and its verdict, and changes
    nothing. ``--watch`` runs the same state machine as the analyzer every ``Period
    seconds``; it raises nothing unless ``--live`` is given. ``Enabled`` is ignored here.
    ``--live`` is for while the analyzer's own check is switched off (``Enabled`` = n):
    two watchers would raise and reset the same alarm. Neither seeds the settings.
    """
    import midas.client  # only here, so the module imports without MIDAS

    parser = argparse.ArgumentParser(prog="mdqm-mupix-no-data", description=main.__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--watch", action="store_true")
    parser.add_argument("--live", action="store_true",
                        help="with --watch: really raise/reset the alarm and stop the sequencer")
    parser.add_argument("--experiment", "-e", default=None, help="MIDAS experiment name")
    parser.add_argument("--client", default="mupix_no_data_check")
    args = parser.parse_args(argv)

    client = midas.client.MidasClient(args.client, expt_name=args.experiment)
    alarm = MupixNoData(client_name=args.client, dry_run=not (args.watch and args.live))
    try:
        while True:
            raw = {**alarm.read_settings(client), "Enabled": True}
            cfg = Settings.parse(raw)
            if args.once:
                t0 = time.perf_counter()
                obs = alarm.observe(client, cfg)
                ms = (time.perf_counter() - t0) * 1e3
                print(f"settings: {cfg}")
                print(f"observation: {obs}")
                print(f"verdict: {classify(obs, cfg)}")
                print(f"cost: {ms:.1f} ms")
                return 0
            alarm.poll(client, raw)
            time.sleep(cfg.period_s)
    except KeyboardInterrupt:
        return 0
    finally:
        client.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
