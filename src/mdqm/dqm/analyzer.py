#!/usr/bin/env python3
"""The DQM analyzer: a MIDAS client that samples events and serves histograms.

    mdqm-analyzer --experiment WDSCALERS --plugin wavedream

What it deliberately does NOT do, because a monitoring process must never be
able to affect data taking:

* **No equipment and no transition callbacks.** Registering either would put
  this client in the run-transition path, where a wedged process delays a run
  start until the watchdog reaps it. That is not hypothetical -- it is exactly
  what the retired DQM publisher did, registering TR_START at sequence 100
  (``docs/REGISTRY.md``). Run state is *polled* from ``/Runinfo`` instead.
* **Never ``GET_ALL``.** The event request is ``GET_NONBLOCKING``, so MIDAS
  overwrites events for this client rather than stalling the producer. One word
  is the difference between a monitor and a throttle, so it is asserted in the
  tests.
* **No unbounded work per event.** Sampling is rate-limited by a token bucket,
  and the analyzer throttles *itself* down if the DAQ starts dropping packets.
  A plugin whose events are cheap may opt out of the bucket ("process all"),
  but never out of the throttle.
* **A CPU budget, for a plugin that declares one** (``Sampling/CPU budget %``,
  the SMA plugin). The analyzer measures its own CPU and analyses as many
  events per second as fit in the budget (`CpuBudget`). The events in between
  are never copied out of the buffer: ``bm_skip_event`` moves this client's
  read pointer to the newest event (`BufferOps`), so a skipped event costs
  nothing whatever the rate. A live peek for shifters, not a lossless record.

One plugin per process. A second detector runs as a second client
(``--plugin sma --client sma_analyzer``) with its own ODB settings tree, its own
event request and its own commands, so neither can starve or reconfigure the
other.

The histogram store lives outside the connection loop on purpose: a MIDAS
restart then costs a reconnect and nothing else, and whoever is watching the
page sees the same accumulated plots afterwards.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import sys
import time
from collections import deque

import midas
import midas.client

from mdqm.dqm import settings as odb_settings
from mdqm.dqm.hist import HistStore
from mdqm.dqm.server import Server

DEFAULT_CLIENT = "wd_analyzer"
#: The DAQ-loss counter watched when a plugin names none of its own.
DROPPED_PATH = "/Equipment/WDWaveforms/Variables/Thread/DroppedPackets"

_stop = False


def _on_signal(_sig, _frm):
    global _stop
    _stop = True


class TokenBucket:
    """Process at most `rate` events per second, discarding the rest.

    The buffer is drained every cycle regardless -- that is free with
    GET_NONBLOCKING and keeps the read pointer current -- but only this many are
    *decoded*. One knob, replacing the three the retired stack had
    (num-events-per-retrieval, period-ms, serialize-every-n-events), which
    interacted in ways nobody could predict from their names.
    """

    def __init__(self, rate: float, cap: float | None = None, clock=time.monotonic):
        self.rate = float(rate)
        #: Most tokens held at once; None: one second's worth (`rate`).
        self.cap = cap
        self._clock = clock
        self._allowance = self.rate if cap is None else float(cap)
        self._last = clock()

    def _refill(self) -> None:
        now = self._clock()
        cap = self.rate if self.cap is None else self.cap
        self._allowance = min(cap, self._allowance + (now - self._last) * self.rate)
        self._last = now

    def allowance(self) -> float:
        self._refill()
        return self._allowance

    def take(self, n: int = 1) -> bool:
        if self.rate <= 0:
            return False
        self._refill()
        if self._allowance < n:
            return False
        self._allowance -= n
        return True


class CpuBudget:
    """Keep this process at or under `budget_pct` % of one core.

    The knob is how many events per second are analysed (`rate`); the measure
    is the process's own CPU time (`time.process_time`: every thread, so the
    buffer reads, the decoding, the fills and the brpc replies all count).
    Every `UPDATE_S` it looks at the last `CONTROL_S` and sets

        rate = analysed_per_s * target / cpu_fraction,   target = HEADROOM * budget

    i.e. the rate at which the measured cost per analysed event would use
    exactly the target. The idle cost (polling, replies) is in the measured
    fraction too, so the rule settles slightly below the target rather than
    above it (the fixed point is (target - idle) / cost per event). It never
    raises the rate by more than `MAX_STEP_UP` times what was actually analysed:
    when the offered rate is below the limit the limit would otherwise run away,
    and a jump in the offered rate would then cost a second of several times the
    budget before the next update. `rate` stays within [`MIN_RATE`, max_rate]
    (0 when the budget or max_rate is 0).

    `clock` and `cpu` are injectable for tests.
    """

    UPDATE_S = 1.0
    CONTROL_S = 2.0
    REPORT_S = 5.0
    HEADROOM = 0.9
    #: One event every 5 s at worst, so the pages never freeze while the
    #: budget is above zero.
    MIN_RATE = 0.2
    MAX_STEP_UP = 2.0
    #: Cost assumed for the first events, before anything is measured.
    GUESS_S_PER_EVENT = 0.010

    def __init__(self, budget_pct: float, max_rate: float, *, clock=time.monotonic,
                 cpu=time.process_time):
        self._clock, self._cpu = clock, cpu
        self.budget_pct = float(budget_pct)
        self.max_rate = float(max_rate)
        self.rate = self._clamp(self._target() / self.GUESS_S_PER_EVENT)
        self.updates = 0
        now = clock()
        self._samples = deque([(now, cpu(), 0)])
        self._last_update = now
        #: CPU fraction of each update interval, for the p95 in the status.
        self._fracs = deque(maxlen=60)

    def _target(self) -> float:
        return self.HEADROOM * self.budget_pct / 100.0

    def _clamp(self, rate: float) -> float:
        if self.budget_pct <= 0 or self.max_rate <= 0:
            return 0.0
        return min(self.max_rate, max(self.MIN_RATE, rate))

    def configure(self, budget_pct: float, max_rate: float) -> None:
        budget_pct, max_rate = float(budget_pct), float(max_rate)
        if (budget_pct, max_rate) == (self.budget_pct, self.max_rate):
            return
        old = self.budget_pct
        self.budget_pct, self.max_rate = budget_pct, max_rate
        # A new budget scales the rate at once rather than a second later.
        self.rate = self._clamp(self.rate * (budget_pct / old) if old > 0
                                else self._target() / self.GUESS_S_PER_EVENT)

    def _span(self, seconds: float):
        """(dt, dcpu, dn) from the sample `seconds` back to the newest one."""
        t1, c1, n1 = self._samples[-1]
        t0, c0, n0 = self._samples[0]
        for t, c, n in self._samples:
            if t1 - t <= seconds + 1e-9:
                t0, c0, n0 = t, c, n
                break
        return t1 - t0, c1 - c0, n1 - n0

    def update(self, n_analysed: int) -> bool:
        """Feed the running count of analysed events; returns True when it acted."""
        now = self._clock()
        if now - self._last_update < self.UPDATE_S:
            return False
        self._last_update = now
        self._samples.append((now, self._cpu(), int(n_analysed)))
        while len(self._samples) > 2 and now - self._samples[1][0] >= self.REPORT_S:
            self._samples.popleft()
        t_prev, c_prev, _n = self._samples[-2]
        if now > t_prev:
            self._fracs.append(max(0.0, (self._samples[-1][1] - c_prev) / (now - t_prev)))
        self.updates += 1

        dt, dcpu, dn = self._span(self.CONTROL_S)
        if dt <= 0:
            return True
        frac = max(0.0, dcpu / dt)
        done = dn / dt
        target = self._target()
        if target <= 0 or self.max_rate <= 0:
            self.rate = 0.0
        elif done > 0 and frac > 0:
            new = done * target / frac
            self.rate = self._clamp(min(new, max(self.MAX_STEP_UP * done, done + 1.0)))
        elif frac > target:
            # Over budget with nothing analysed (replies, polling): nothing to
            # take away but the rate itself.
            self.rate = self._clamp(self.rate / 2.0)
        return True

    def cpu_pct(self) -> float | None:
        """Mean CPU over the last `REPORT_S`, % of one core."""
        dt, dcpu, _dn = self._span(self.REPORT_S)
        return 100.0 * dcpu / dt if dt > 0 else None

    def cpu_pct_p95(self) -> float | None:
        """95th percentile of the per-update CPU, % of one core, over the last minute."""
        if not self._fracs:
            return None
        v = sorted(self._fracs)
        return 100.0 * v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))]


class BufferOps:
    """Skip to the newest event and read the backlog, without copying events.

    The python ``midas.client`` offers neither, so they are called through the
    library it has already loaded. ``bm_skip_event(handle)`` sets this client's
    read pointer to the buffer's write pointer (``midas.cxx`` ``bm_skip_event``:
    ``pclient->read_pointer = pheader->write_pointer``) under the buffer lock:
    O(1), nothing copied, and the next event read is the next one written.
    ``bm_get_buffer_level`` is ``write_pointer - read_pointer`` for this client.
    Both are also RPCs (``RPC_BM_SKIP_EVENT``), so they work through mserver.

    The library is C++ and exports them under their mangled names; `c_*`
    wrappers are tried first in case a MIDAS version adds them. A client object
    with ``skip_event(buf)`` / ``buffer_level(buf)`` methods (a fake, a future
    python client) is used directly. With neither, `skip` falls back to reading
    and discarding (the old cost) and says so in `method`.
    """

    SKIP_SYMBOLS = ("c_bm_skip_event", "_Z13bm_skip_eventi")
    LEVEL_SYMBOLS = ("c_bm_get_buffer_level", "_Z19bm_get_buffer_leveliPi")

    def __init__(self, client, buf):
        import ctypes

        self._ctypes = ctypes
        self.client, self.buf = client, buf
        self._skip = getattr(client, "skip_event", None)
        self._level = getattr(client, "buffer_level", None)
        self.method = "client" if self._skip else None
        lib = getattr(client, "lib", None)
        if self._skip is None and lib is not None:
            f = self._symbol(lib, self.SKIP_SYMBOLS)
            if f is not None:
                f.argtypes = [ctypes.c_int]
                self._skip = lambda b, _f=f: _f(b)
                self.method = "bm_skip_event"
        if self._level is None and lib is not None:
            f = self._symbol(lib, self.LEVEL_SYMBOLS)
            if f is not None:
                f.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]

                def level(b, _f=f):
                    n = ctypes.c_int()
                    _f(b, ctypes.byref(n))
                    return n.value
                self._level = level
        if self._skip is None:
            self.method = "drain"

    @staticmethod
    def _symbol(lib, names):
        for name in names:
            try:
                return lib[name]
            except (AttributeError, TypeError, KeyError):
                continue
        return None

    #: Without bm_skip_event: at most this many events read and dropped per
    #: skip. Bounded so one skip cannot hold the loop (or the brpc replies) off;
    #: its CPU is the process's own, so the budget accounts for it.
    DRAIN_MAX = 200

    def skip(self, receive=None) -> int:
        """Drop the backlog. Returns the events read to do it (0 unless draining).

        `receive(client, buf)` is the analyzer's reader (it turns a truncated
        event into a count instead of an exception); the drain uses it.
        """
        if self.method != "drain":
            self._skip(self.buf)
            return 0
        if not getattr(self, "_announced", False):
            self._announced = True
            print("mdqm: bm_skip_event not found in the MIDAS library; skipping by "
                  "reading and dropping events (costs a copy each; skip_method=drain)",
                  file=sys.stderr, flush=True)
        receive = receive or (lambda c, b: c.receive_event(b, async_flag=True, use_numpy=True))
        n = 0
        while n < self.DRAIN_MAX and receive(self.client, self.buf) is not None:
            n += 1
        return n

    def level(self) -> int | None:
        """Bytes this client has not read yet, or None when unknown."""
        if self._level is None:
            return None
        try:
            return int(self._level(self.buf))
        except Exception:                               # noqa: BLE001
            return None


class Backoff:
    def __init__(self, start=1.0, cap=15.0):
        self.start, self.cap, self.current = start, cap, start

    def reset(self):
        self.current = self.start

    def next(self) -> float:
        d = self.current
        self.current = min(self.cap, self.current * 2)
        return d


class Analyzer:
    """Owns the state that must survive a MIDAS reconnect.

    What the analyzer asks of a plugin beyond ``name``, ``accepts``, ``process``
    and ``status`` is optional, and each absence falls back to what the
    WaveDREAM plugin has always had:

    ``settings_root``, ``settings_defaults``
        The plugin's own ODB tree and its nested defaults (both or neither).
        Without them the plugin gets ``/DQM/Analyzer``.
    ``shape_fingerprint(settings)``
        What, in the settings, changes a histogram's shape. Defaults to
        `settings.binning_fingerprint`.
    ``apply_settings(settings, rebuild)``
        Called on every settings change; ``rebuild`` is True when the shape
        fingerprint changed, and on the first apply. Without it the analyzer
        sets ``plugin.roles`` and calls ``plugin.reconfigure(roles, binning)``.
    ``process_all``
        Bypass the token bucket, when the settings have no
        ``Sampling/process all``.
    ``cpu_budget_pct``
        Sample by CPU budget (`CpuBudget`, `run_once`), when the settings have
        no ``Sampling/CPU budget %``; clamped to 50 (``--no-cpu-budget`` processes all).
    ``sample_burst``
        Consecutive events read per analysis slot when sampling (default 1).
    ``sampling_state``
        Set to a dict of the sampling mode, budget and measured CPU after each
        cycle, for a plugin that reports it.
    ``event_ids``
        One event request per id. Without it, every event (-1).
    ``dropped_path``
        The DAQ-loss counter to watch; empty disables the throttle.
    ``commands()``
        Extra brpc commands, ``{cmd: fn(args) -> framed bytes}``.
    """

    def __init__(self, plugin_factory, *, rate=20.0, buffer_name="SYSTEM",
                 client_name=None, dropped_path=None, clock=time.monotonic,
                 cpu=time.process_time, no_cpu_budget=False):
        self.store = HistStore()
        self.plugin = plugin_factory(self.store)
        #: For the CPU budget only (tests inject fakes); the rest uses real time.
        self._clock, self._cpu = clock, cpu
        self.buffer_name = buffer_name
        self.client_name = (client_name
                            or getattr(self.plugin, "client_name", None) or DEFAULT_CLIENT)
        self.bucket = TokenBucket(rate)
        self.configured_rate = rate
        # Until a settings tree says otherwise, the plugin's own preference.
        self.process_all = bool(getattr(self.plugin, "process_all", False))
        #: The CPU-budget sampler; None for a plugin without a budget (the
        #: WaveDREAM one, whose path below is unchanged) or a budget >= 100 %.
        self.budget: CpuBudget | None = None
        self.cpu_budget_pct = None
        self.sample_burst = max(1, int(getattr(self.plugin, "sample_burst", 1)))
        self._slots = TokenBucket(0.0, clock=clock)
        self._slot_left = 0
        self._slot_active = 0.0
        self._ops: BufferOps | None = None
        self.skips = 0
        self.truncated = 0
        #: Exceptions from plugin.process, by type (the event is dropped).
        self.plugin_errors: dict[str, int] = {}
        #: Offered events of the requested ids, from serial numbers (budget mode).
        self.offered = 0
        self.offered_per_s = None
        self._serials: dict = {}
        self._offer_hist: deque = deque()
        #: --no-cpu-budget: process everything whatever the ODB says (dev only).
        self.unbounded = False
        self.unbounded = bool(no_cpu_budget)
        pct = getattr(self.plugin, "cpu_budget_pct", None)
        if pct is not None:
            self._set_budget(float(pct))
        self.dropped_path = (dropped_path if dropped_path is not None
                             else getattr(self.plugin, "dropped_path", DROPPED_PATH))

        has_root = hasattr(self.plugin, "settings_root")
        has_defaults = hasattr(self.plugin, "settings_defaults")
        if has_root != has_defaults:
            # Half a declaration would put one plugin's keys in another's tree,
            # or read the WaveDREAM defaults from a root that has none of them.
            raise ValueError(f"plugin {self.plugin.name!r} must declare both "
                             "settings_root and settings_defaults, or neither")
        self.settings_root = self.plugin.settings_root if has_root else odb_settings.ROOT
        self.settings_defaults = (self.plugin.settings_defaults if has_defaults
                                  else odb_settings.SECTIONS)

        self.settings = None
        self._settings_shape = None
        self._settings_error = None
        self._settings_checked = 0.0
        self.reconfigures = 0

        self.seen = 0
        self.processed = 0
        self.run_number = None
        self.run_state = None
        self.started_at = time.time()
        self.connected_since = None
        self.reconnects = 0
        self.throttle_events = []
        self.budget_exhausted = 0
        self._dropped_baseline = None
        self._ev_window = []

        self.server = Server(
            self.store,
            status_fn=self.status,
            defs_fn=lambda: {"plugin": self.plugin.name,
                             "histograms": self.store.names()},
            # The plugin owns the frame; the server only forwards it. Guarded
            # because a plugin need not offer one -- only detector-specific
            # plugins have traces to show.
            scope_fn=lambda: (self.plugin.scope_frame(self.run_state == 3)
                              if hasattr(self.plugin, "scope_frame") else None),
            extra=(self.plugin.commands() if hasattr(self.plugin, "commands") else None))

    # -- status --------------------------------------------------------------

    def status(self) -> dict:
        now = time.time()
        recent = [t for t in self._ev_window if now - t < 10.0]
        self._ev_window = recent
        return {
            "client": self.client_name,
            "uptime_s": round(now - self.started_at, 1),
            "connected_since": self.connected_since,
            "reconnects": self.reconnects,
            "run_number": self.run_number,
            "run_active": self.run_state == 3,
            "events_seen": self.seen,
            "events_processed": self.processed,
            "processed_per_s": round(len(recent) / 10.0, 2),
            "rate_limit": self.bucket.rate,
            "configured_rate": self.configured_rate,
            "throttled": self.bucket.rate < self.configured_rate,
            "process_all": self.process_all,
            "sampling_bypassed": self._bypass_bucket(),
            "sampling_mode": self.sampling_mode(),
            "cpu_budget_pct": self.cpu_budget_pct,
            "cpu_pct": None if self.budget is None else _round(self.budget.cpu_pct(), 2),
            "cpu_pct_p95": None if self.budget is None else _round(self.budget.cpu_pct_p95(), 2),
            "analysis_rate_limit": None if self.budget is None else _round(self.budget.rate, 3),
            "skips": self.skips,
            "skip_method": None if self._ops is None else self._ops.method,
            "events_truncated": self.truncated,
            "plugin_errors": dict(self.plugin_errors),
            "cpu_budget_clamped": bool(getattr(self, "budget_clamped", False)),
            "offered_per_s": _round(self.offered_per_s, 3),
            "throttle_events": self.throttle_events[-5:],
            "budget_exhausted": self.budget_exhausted,
            "histograms": len(self.store),
            "reconfigures": self.reconfigures,
            "settings_root": self.settings_root,
            "binning": (self.settings or {}).get("Binning", {}),
            # Read back off the histograms themselves. The settings dict says
            # what was requested; this says what exists, and the two disagreeing
            # is exactly the failure this reports.
            "axes": self.live_axes(),
            "channel_roles": (self.settings or {}).get("Channel roles", {}),
            "server_calls": self.server.calls,
            "server_last_error": self.server.last_error,
            "plugin": self.plugin.status(),
        }

    # -- live configuration --------------------------------------------------

    def apply_settings(self, client, force: bool = False) -> bool:
        """Re-read the plugin's settings tree and adopt any change. Returns True if it did.

        Polled rather than hotlinked. A watch callback would run on the client's
        thread while the brpc handler may be encoding a histogram from the same
        store, and the cost of reading a dozen small keys every couple of seconds
        is far below the cost of getting that locking wrong.
        """
        now = time.time()
        if not force and now - self._settings_checked < 2.0:
            return False
        self._settings_checked = now

        try:
            new = odb_settings.read(client, self.settings_root, self.settings_defaults)
        except Exception:
            return False

        first = self.settings is None
        if not first and odb_settings.fingerprint(new) == odb_settings.fingerprint(self.settings):
            return False

        # `first` is included, not excluded. The plugin's constructor built its
        # histograms from code defaults because it had no ODB to read yet, so the
        # first apply is precisely when the ODB has to be pushed in. Skipping it
        # left the analyzer running on defaults while *reporting* the ODB values
        # in its status -- plots that disagreed with the configuration they
        # claimed, which is worse than plots that are merely wrong.
        #
        # Nothing is recorded until the plugin has taken the change. Recording
        # first and failing after would leave settings that were never applied
        # looking current, and the unchanged fingerprint would stop any retry.
        try:
            shape_fn = getattr(self.plugin, "shape_fingerprint",
                               odb_settings.binning_fingerprint)
            shape = shape_fn(new)
            changed_shape = shape != self._settings_shape
            rebuilt = False
            if hasattr(self.plugin, "apply_settings"):
                # The plugin knows its own settings; it is told every change,
                # and whether it has to rebuild, and rebuilds itself.
                self.plugin.apply_settings(new, changed_shape)
                rebuilt = changed_shape
            else:
                self.plugin.roles = new["Channel roles"]
                if changed_shape and hasattr(self.plugin, "reconfigure"):
                    self.plugin.reconfigure(new["Channel roles"], new["Binning"])
                    rebuilt = True
        except Exception as exc:                       # noqa: BLE001
            # Keep running on what we had and retry on the next poll. Said once
            # per distinct error, not every two seconds.
            error = f"{type(exc).__name__}: {exc}"
            if error != self._settings_error:
                self._settings_error = error
                print(f"{self.client_name}: could not apply {self.settings_root}: {error}",
                      file=sys.stderr, flush=True)
                with contextlib.suppress(Exception):
                    client.msg(f"{self.client_name}: could not apply settings from "
                               f"{self.settings_root} ({error}); still on the previous "
                               f"ones, retrying", is_error=True)
            return False

        self._settings_error = None
        self.settings = new
        self._settings_shape = shape

        sampling = new.get("Sampling", {})
        rate = float(sampling.get("max events per s", self.configured_rate))
        if rate != self.configured_rate:
            self.configured_rate = rate
            # An operator raising the rate also clears a self-throttle, because
            # they have said what they want more recently than we did.
            self.bucket.rate = rate
        if "process all" in sampling:
            self.process_all = bool(sampling["process all"])
        if "CPU budget %" in sampling:
            # After "process all": a tree with both is sampled by the budget.
            try:
                self._set_budget(float(sampling["CPU budget %"]))
            except (TypeError, ValueError):
                pass                        # keep the previous budget

        if rebuilt:
            self.reconfigures += 1
            if first:
                # Startup, not a change: stdout only. A MIDAS message on every
                # analyzer start would teach shifters to ignore the real one.
                print(f"{self.client_name}: applied binning from {self.settings_root}",
                      flush=True)
            else:
                # Best effort: the rebuild has already happened either way.
                with contextlib.suppress(Exception):
                    client.msg(f"{self.client_name}: binning changed; rebuilt "
                               f"{len(self.store)} histograms (counts reset)")
        return True

    # -- the CPU budget --------------------------------------------------------

    #: The most an ODB edit can give the analyzer. More, or no budget at all,
    #: only from the command line (--no-cpu-budget), which is for tests.
    MAX_BUDGET_PCT = 50.0

    def _set_budget(self, pct: float) -> None:
        """Adopt a CPU budget, at most `MAX_BUDGET_PCT`; --no-cpu-budget overrides it."""
        if self.unbounded:
            self.cpu_budget_pct = None
            self.budget = None
            self.process_all = True
            return
        self.budget_clamped = pct > self.MAX_BUDGET_PCT
        pct = max(0.0, min(pct, self.MAX_BUDGET_PCT))
        self.cpu_budget_pct = pct
        self.process_all = False
        if self.budget is None:
            self.budget = CpuBudget(pct, self._max_rate(), clock=self._clock, cpu=self._cpu)
            # One slot's tokens to start with: the first frame is analysed at
            # once, not after the bucket has filled.
            self._slots = TokenBucket(0.0, cap=float(self.sample_burst), clock=self._clock)
            self._slot_left = 0
        else:
            self.budget.configure(pct, self._max_rate())

    def reset_serials(self) -> None:
        """After a reconnect: the serial gap is outage, not frames offered."""
        self._serials.clear()
        self._offer_hist.clear()
        if hasattr(self.plugin, "reset_serial_baseline"):
            self.plugin.reset_serial_baseline()

    def _max_rate(self) -> float:
        # The configured cap, lowered further if the DAQ-health valve fired.
        return min(self.configured_rate, self.bucket.rate)

    def sampling_mode(self) -> str:
        if self.budget is not None:
            return "cpu budget"
        return "process all" if self._bypass_bucket() else "rate"

    def _publish_sampling_state(self) -> None:
        if not hasattr(self.plugin, "sampling_state"):
            return
        b = self.budget
        self.plugin.sampling_state = {
            "mode": self.sampling_mode(),
            "cpu_budget_pct": self.cpu_budget_pct,
            "cpu_pct": None if b is None else b.cpu_pct(),
            "rate_limit": None if b is None else b.rate,
        }

    def live_axes(self) -> dict:
        """The binning the histograms actually have, straight from the objects."""
        out = {}
        for name in self.store.names():
            hist = self.store.get(name)
            meta = getattr(hist, "metadata", None)
            if meta is None:
                continue
            out[name] = [
                {"bins": ax["bins"], "lo": ax["lo"], "hi": ax["hi"]}
                for ax in meta()["axes"]
            ]
        return out

    # -- the DAQ-safety valve -------------------------------------------------

    def check_daq_health(self, client) -> None:
        """Throttle ourselves if the DAQ starts losing packets while we run.

        We cannot prove from inside this process that we are not the cause, so
        the honest response to evidence of stress is to take less, not to
        reason about whether it was our fault. Halves the rate each time, down
        to zero; recovering it is an operator action, never automatic. A
        throttle also re-engages the bucket for a plugin that processes all.
        An empty `dropped_path` turns the check off.
        """
        if not self.dropped_path:
            return
        try:
            dropped = int(client.odb_get(self.dropped_path))
        except Exception:
            return                       # not every experiment has this counter
        if self._dropped_baseline is None:
            self._dropped_baseline = dropped
            return
        if dropped <= self._dropped_baseline:
            return

        delta = dropped - self._dropped_baseline
        self._dropped_baseline = dropped
        new_rate = self.bucket.rate / 2.0
        if new_rate < 0.5:
            new_rate = 0.0
        self.bucket.rate = new_rate
        note = {"at": time.time(), "dropped_delta": delta, "new_rate": new_rate}
        self.throttle_events.append(note)
        # Best effort: if the message cannot be sent the throttle has still
        # happened, and failing here would undo it.
        with contextlib.suppress(Exception):
            client.msg(
                f"{self.client_name}: DAQ dropped {delta} packets; halving my sampling "
                f"rate to {new_rate}/s. I may not be the cause, but monitoring must "
                f"never be. Restart me to restore the configured rate.", is_error=True)

    # -- the loop ------------------------------------------------------------

    def poll_run_state(self, client) -> None:
        """Read run state rather than registering a transition callback."""
        try:
            state = int(client.odb_get("/Runinfo/State"))
            number = int(client.odb_get("/Runinfo/Run number"))
        except Exception:
            return
        if number != self.run_number:
            # A new run: clear what asked to be cleared, and reset per-run state.
            if self.run_number is not None:
                self.store.clear_for_new_run()
            self.run_number = number
        self.run_state = state
        # Plugins that judge "no data while running" need the state; only
        # those that declare the attribute get it.
        if hasattr(self.plugin, "run_active"):
            self.plugin.run_active = state == 3

    #: Decode this many events, then yield to MIDAS before continuing.
    #:
    #: The brpc handler shares this interpreter, so it can only run when this
    #: loop gives up the GIL. Measured at 500 events/s offered: with no yield at
    #: all, wd::status stopped answering entirely -- the analyzer was decoding
    #: perfectly and the pages showed an error. Capping the *work* per cycle
    #: fixed that and cost a third of the throughput (131 ev/s at 39% of a core,
    #: nowhere near CPU-bound). Yielding periodically fixes the same problem
    #: without the cap, because the problem was never how much work there was.
    YIELD_EVERY = 25

    #: ...or after this long, whichever comes first. The count alone assumes
    #: every event costs about the same; one large readout frame can take
    #: milliseconds, and 25 of them would hold the RPC handler off for most of
    #: a second. Checked on drained events too, which cost a copy each.
    YIELD_AFTER_S = 0.05

    def _bypass_bucket(self) -> bool:
        """Process-all skips the bucket until the DAQ-health valve has fired.

        A throttle leaves `bucket.rate` below the configured rate, and from then
        on the bucket applies to this plugin too: the valve must work the same
        whatever the plugin prefers.
        """
        # A rate of zero is an operator saying "decode nothing"; process-all
        # does not override that.
        return (self.process_all and self.configured_rate > 0
                and self.bucket.rate >= self.configured_rate)

    def register_event_requests(self, client, buf) -> list[int]:
        """One request per event id the plugin accepts; every event otherwise.

        Filtering in MIDAS rather than in `accepts` means another detector's
        events are never copied into this process at all.
        """
        ids = sorted(getattr(self.plugin, "event_ids", None) or [-1])
        for event_id in ids:
            client.register_event_request(
                buf, event_id=event_id, trigger_mask=-1,
                # Never GET_ALL: this must not be able to back-pressure a frontend.
                sampling_type=midas.GET_NONBLOCKING)
        return ids

    #: While a sampling slot waits for its events, poll this often...
    SLOT_POLL_MS = 5
    #: ...for this long after the slot opened or last read an event; then at
    #: the normal cycle, so a stopped run costs no more than it did before.
    SLOT_POLL_S = 0.25

    def next_wait_ms(self, cycle_ms: int) -> int:
        """How long the main loop may sit in ``communicate`` before the next cycle.

        Without a CPU budget: `cycle_ms`, as always. With one: short while a
        slot waits for its events (the next one written after a skip), else
        until the next slot's tokens are there, at most `cycle_ms`.
        """
        b = self.budget
        if b is None:
            return cycle_ms
        if self._slot_left > 0:
            if time.monotonic() - self._slot_active > self.SLOT_POLL_S:
                return cycle_ms
            return self.SLOT_POLL_MS
        if b.rate <= 0:
            return cycle_ms
        need = self.sample_burst - self._slots.allowance()
        return int(min(cycle_ms, max(self.SLOT_POLL_MS, 1000.0 * need / b.rate)))

    _TRUNCATED = object()

    def _receive(self, client, buf):
        """One event, None when there is none; `_TRUNCATED` for one too big.

        MIDAS copies what fits, advances the read pointer past the event and
        returns BM_TRUNCATED, which the python client raises. Counting it and
        going on keeps one oversized frame from turning into a reconnect loop.
        """
        try:
            # use_numpy is not optional at these rates. Without it a 33 kB
            # TID_BYTE bank arrives as a tuple of 33,000 Python ints and
            # bank_bytes() has to walk every one of them; wdunpack documents
            # that shape because the offline file reader hands it over too. With
            # it, the same bank is an ndarray and the conversion is a memcpy.
            return client.receive_event(buf, async_flag=True, use_numpy=True)
        except midas.MidasError as exc:
            if getattr(exc, "code", None) != midas.status_codes.get("BM_TRUNCATED"):
                raise
            self.truncated += 1
            grown = self._grow_read_buffer(client, buf)
            if self.truncated == 1 or self.truncated % 1000 == 0 or grown:
                print(f"{self.client_name}: event larger than the read buffer skipped "
                      f"({self.truncated} so far)"
                      + (f"; read buffer grown to {grown >> 20} MiB" if grown else ""),
                      file=sys.stderr, flush=True)
            return self._TRUNCATED

    def _grow_read_buffer(self, client, buf) -> int:
        """After a truncated read, make the next one of that size fit. Returns the new size.

        MIDAS posts an ERROR message for every truncated read (``bm_read_buffer``),
        so leaving the buffer small would flood the message log once per frame.
        The truncated copy still holds the event header, whose data size says
        how much is needed; the python client's one-event buffer is replaced by
        one that large, up to `MAX_EVENT_CAP`. 0 when it cannot (a fake client,
        or an event beyond the cap: those stay truncated and counted).
        """
        import ctypes
        import struct

        bufs = getattr(client, "event_buffers", None)
        old = bufs.get(buf) if isinstance(bufs, dict) else None
        if old is None or ctypes.sizeof(old) < 16:
            return 0
        need = struct.unpack_from("<I", bytes(old[12:16]))[0] + 16 + 100
        if need <= ctypes.sizeof(old) or need > MAX_EVENT_CAP + 100:
            return 0
        size = min(MAX_EVENT_CAP, 1 << (need - 1).bit_length()) + 100
        if size < need:
            return 0
        bufs[buf] = ctypes.create_string_buffer(size)
        return size

    def _process(self, client, event) -> bool:
        """plugin.process, with an error in it counted instead of raised.

        An exception here (a MemoryError on a huge frame, a decoding bug) used
        to leave the connection block and send the analyzer through the MIDAS
        reconnect loop, which is for MIDAS errors. Now the event is dropped,
        the error counted (`plugin_errors` in dqm::status) and reported once
        per distinct type, on stderr and as a MIDAS message.
        """
        if getattr(self.plugin, "wants_raw", False):
            self._attach_raw(client, event)
        try:
            return bool(self.plugin.process(event, run_number=self.run_number))
        except Exception as exc:                        # noqa: BLE001
            kind = type(exc).__name__
            self.plugin_errors[kind] = self.plugin_errors.get(kind, 0) + 1
            if self.plugin_errors[kind] == 1:
                text = (f"{self.client_name}: event dropped, plugin {self.plugin.name!r} "
                        f"raised {kind}: {exc} (counted in dqm::status plugin_errors; "
                        "reported once per error type)")
                print(text, file=sys.stderr, flush=True)
                with contextlib.suppress(Exception):
                    client.msg(text, is_error=True)
            return False

    @staticmethod
    def _attach_raw(client, event) -> None:
        """Give the event the bytes MIDAS copied out of the buffer, as received.

        The python client unpacks from its one-event buffer; right after the
        receive that buffer still holds the event exactly as the frontend wrote
        it (re-packing would zero, for one, bank32a's reserved header bytes).
        Only for a plugin that keeps raw events, and only for analysed ones.
        """
        bufs = getattr(client, "event_buffers", None)
        if not isinstance(bufs, dict) or len(bufs) != 1:
            return
        buf = next(iter(bufs.values()))
        size = getattr(event.header, "event_data_size_bytes", None)
        try:
            n = 16 + int(size)
            event.mdqm_raw = bytes(buf[:n]) if 16 < n <= len(buf) else None
        except Exception:                               # noqa: BLE001
            pass

    def _note_serial(self, event) -> None:
        """Offered events of the requested ids, from the serial numbers read."""
        h = event.header
        eid = getattr(h, "event_id", None)
        ser = getattr(h, "serial_number", None)
        if ser is None:
            return
        last = self._serials.get(eid)
        self._serials[eid] = ser
        if last is not None and ser > last:
            self.offered += ser - last
        elif last is None or ser < last:
            self.offered += 1

    def _behind(self) -> bool:
        """Would reading in order fall behind? Decided by the budget, not the buffer.

        The buffer level counts every event in SYSTEM, the WaveDREAM ones too,
        so it cannot say whether this client is behind on *its* events. The
        serial numbers can: the offered rate of the requested events (serial
        increments over the controller's window) against the rate the budget
        allows. Below it, events are read in order and nothing is skipped;
        above it, each slot starts from the newest event. Unknown (the first
        half second): read in order; the request starts at the write pointer,
        so there is no old backlog to skip.
        """
        b = self.budget
        now = self._clock()
        self._offer_hist.append((now, self.offered))
        while len(self._offer_hist) > 2 and now - self._offer_hist[1][0] >= b.CONTROL_S:
            self._offer_hist.popleft()
        t0, n0 = self._offer_hist[0]
        if now - t0 < 0.5:
            return False
        offered_per_s = (self.offered - n0) / (now - t0)
        self.offered_per_s = offered_per_s
        return offered_per_s > 0.95 * b.rate

    def _skip(self) -> None:
        self._ops.skip(self._receive)
        self.skips += 1

    def _run_budget(self, client, buf) -> int:
        """One cycle under a CPU budget: analyse what the budget allows, skip the rest.

        Analysis happens in slots of `sample_burst` consecutive events (two for
        SMA: a frame and the one after it, for the gap and live fraction). A
        slot starts when the token bucket, filled at the controller's rate,
        holds that many tokens. If the backlog is more than the bucket could
        read in order, the slot first skips to the newest event
        (`BufferOps.skip`): the skipped events are never copied, so the cost
        of the events not analysed is nothing but the writer overwriting
        them, which it does anyway for a non-blocking reader. When the budget
        keeps up with the offered rate there is no backlog, nothing is
        skipped, and every event is analysed in order.
        """
        if self._ops is None or self._ops.client is not client or self._ops.buf != buf:
            self._ops = BufferOps(client, buf)
            self._slot_left = 0
        b = self.budget
        b.max_rate = self._max_rate()
        b.update(self.processed)
        self._slots.rate = b.rate
        # Up to a quarter of a second of tokens: enough to ride out jitter,
        # small enough that the in-order backlog stays fresh.
        self._slots.cap = max(float(self.sample_burst), 0.25 * b.rate)

        read = 0
        last_yield = time.monotonic()
        while not _stop:
            if self._slot_left <= 0:
                if b.rate <= 0:
                    # Budget 0: nothing to read; stay at the write pointer.
                    self._skip()
                    break
                if not self._slots.take(self.sample_burst):
                    break
                if self._behind():
                    self._skip()
                self._slot_left = self.sample_burst
                self._slot_active = time.monotonic()
            event = self._receive(client, buf)
            if event is None:
                break
            if event is self._TRUNCATED:
                continue
            read += 1
            self.seen += 1
            if self.plugin.accepts(event):
                self._note_serial(event)
                self._slot_left -= 1
                self._slot_active = time.monotonic()
                if self._process(client, event):
                    self.processed += 1
                    self._ev_window.append(time.time())
            now = time.monotonic()
            if now - last_yield >= self.YIELD_AFTER_S:
                last_yield = now
                client.communicate(0)
        self._publish_sampling_state()
        return read

    def run_once(self, client, buf, budget_s: float = 2.0) -> int:
        """One cycle: drain the buffer and decode what the bucket allows.

        Draining is unbounded: it is cheap, and stopping early would leave the
        read pointer behind. Decoding is limited by the token bucket (unless the
        plugin processes all), yields to MIDAS every `YIELD_EVERY` events or
        `YIELD_AFTER_S` seconds so the RPC handler is serviced, and has a
        generous wall-clock backstop that should never normally be reached.
        """
        if self.budget is not None:
            return self._run_budget(client, buf)
        drained = 0
        deadline = time.monotonic() + budget_s
        since_yield = 0
        last_yield = time.monotonic()
        decoding = True
        while not _stop:
            event = self._receive(client, buf)
            if event is None:
                break
            if event is self._TRUNCATED:
                continue
            drained += 1
            self.seen += 1
            if (decoding and self.plugin.accepts(event)
                    # Sampled out when refused: draining still matters.
                    and (self._bypass_bucket() or self.bucket.take())):
                if self._process(client, event):
                    self.processed += 1
                    self._ev_window.append(time.time())
                since_yield += 1
                if time.monotonic() > deadline:
                    self.budget_exhausted += 1
                    if self._bypass_bucket():
                        # Process-all: draining the rest undecoded would throw
                        # away frames a short backlog delayed, which is loss the
                        # plugin promises not to have. Leave them in the buffer
                        # for the next cycle instead; the request is
                        # GET_NONBLOCKING, so an unread event can never hold up
                        # the writer (at worst it is overwritten, and the
                        # plugin's serial check counts that).
                        break
                    # The backstop. Keep draining, stop decoding, resume next
                    # cycle. Never break outright -- that would abandon the drain.
                    decoding = False

            now = time.monotonic()
            if since_yield >= self.YIELD_EVERY or now - last_yield >= self.YIELD_AFTER_S:
                since_yield = 0
                last_yield = now
                # Zero timeout: hand control to MIDAS and come straight back.
                # This is what lets a status or histogram request be answered
                # while a burst is being decoded.
                client.communicate(0)
        self._publish_sampling_state()
        return drained

    def serve(self, client, cmd, args, max_len):
        """The brpc callback. Must stay bounded: it runs on an RPC thread."""
        blob = self.server.dispatch(cmd, args)
        if not blob:
            return midas.status_codes["SUCCESS"], b""
        if len(blob) > max_len:
            # The page reads the size from the header and retries with a bigger
            # buffer, so a truncated reply is a protocol step and not a failure.
            blob = blob[:max_len]
        import ctypes
        return midas.status_codes["SUCCESS"], ctypes.create_string_buffer(blob, len(blob))


#: Without --max-event-size and without /Experiment/MAX_EVENT_SIZE.
DEFAULT_EVENT_SIZE = 8 * 1024 * 1024
#: The read buffer is allocated once at this size, whatever the ODB says.
MAX_EVENT_CAP = 64 * 1024 * 1024


def max_event_size(client, requested: int | None) -> int:
    """The size of the one-event read buffer.

    The python client allocates it once (``open_event_buffer``); an event
    larger than it is truncated, which `Analyzer._receive` counts and skips.
    Default: the experiment's ``/Experiment/MAX_EVENT_SIZE`` (no frontend can
    send more), clamped to [8 MiB, 64 MiB] so a huge ODB value cannot make this
    process allocate hundreds of MB. An explicit value is used as given.
    """
    if requested:
        return int(requested)
    try:
        odb = int(client.odb_get("/Experiment/MAX_EVENT_SIZE"))
    except Exception:                                   # noqa: BLE001
        odb = 0
    return min(MAX_EVENT_CAP, max(DEFAULT_EVENT_SIZE, odb))


def _round(x, digits):
    return None if x is None else round(float(x), digits)


def make_plugin_factory(name, roles=None, binning=None):
    if name == "wavedream":
        from mdqm.plugins.wavedream import WaveDreamPlugin
        return lambda store: WaveDreamPlugin(store, roles=roles, binning=binning)
    if name == "sma":
        try:
            from mdqm.plugins.sma import SmaPlugin
        except ImportError as exc:
            raise SystemExit(f"plugin 'sma' cannot be loaded: {exc}") from exc
        return lambda store: SmaPlugin(store)
    raise SystemExit(f"unknown plugin {name!r}; known: wavedream, sma")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default=os.environ.get("MIDAS_EXPT_NAME"))
    ap.add_argument("--client", default=None,
                    help=f"MIDAS client name; default the plugin's, else {DEFAULT_CLIENT}")
    ap.add_argument("--plugin", default="wavedream")
    ap.add_argument("--buffer", default="SYSTEM")
    ap.add_argument("--rate", type=float, default=20.0,
                    help="events per second to decode; the buffer is drained regardless")
    ap.add_argument("--max-event-size", type=int, default=None,
                    help="largest event read whole, bytes; default /Experiment/MAX_EVENT_SIZE "
                         f"(at most {MAX_EVENT_CAP >> 20} MiB, at least {DEFAULT_EVENT_SIZE >> 20} "
                         "MiB); a larger event is counted and skipped, and the buffer grows to fit it")
    ap.add_argument("--cycle-ms", type=int, default=200)
    ap.add_argument("--dropped-path", default=None,
                    help="ODB counter of DAQ losses that throttles this analyzer; "
                         "default the plugin's, empty disables the check")
    ap.add_argument("--no-cpu-budget", action="store_true",
                    help="DEVELOPMENT ONLY: analyse every event in order, ignoring "
                         "Sampling/CPU budget %% (can use a whole core); for comparisons "
                         "with the offline CLI and stress tests")
    ap.add_argument("--decode-budget-s", type=float, default=2.0,
                    help="backstop on how long one decode burst may run; the "
                         "periodic yield, not this, is what keeps RPC answering")
    args = ap.parse_args(argv)

    if not args.experiment:
        print("error: no experiment; pass --experiment or set MIDAS_EXPT_NAME",
              file=sys.stderr)
        return 2

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # Outside the reconnect loop: a MIDAS bounce must not lose what has been
    # accumulated, or the operator watching the page sees their plots reset for
    # a reason that has nothing to do with the data.
    # Built with the defaults; apply_settings() re-reads the ODB and rebuilds
    # once connected, so the ODB is the authority and --rate only seeds it.
    # Without --client, the plugin's own name, so a second plugin does not
    # collide with the WaveDREAM client its pages do not talk to.
    analyzer = Analyzer(make_plugin_factory(args.plugin),
                        rate=args.rate, buffer_name=args.buffer,
                        client_name=args.client, dropped_path=args.dropped_path,
                        no_cpu_budget=args.no_cpu_budget)
    args.client = analyzer.client_name
    backoff = Backoff()

    print(f"{args.client}: plugin={args.plugin} buffer={args.buffer} "
          f"rate={args.rate}/s experiment={args.experiment}", flush=True)
    print(f"{args.client}: {len(analyzer.store)} histograms: "
          f"{', '.join(analyzer.store.names())}", flush=True)

    while not _stop:
        try:
            # The context manager is not stylistic. client.py:186-196 documents
            # that cm_disconnect_experiment() alone does not free RPC *server*
            # resources; disconnect() additionally calls rpc_server_shutdown()
            # and ss_suspend_reset_server_acceptions(). Without it the second
            # register_brpc_callback after a reconnect trips over stale state.
            with midas.client.MidasClient(args.client, expt_name=args.experiment) as client:
                analyzer.connected_since = time.time()
                if analyzer.reconnects:
                    print(f"{args.client}: reconnected", flush=True)
                    analyzer.reset_serials()

                # Seed before anything reads it, so a fresh experiment gets a
                # settings tree an operator can find and edit.
                created = odb_settings.seed(client, analyzer.settings_root,
                                            analyzer.settings_defaults)
                if created:
                    print(f"{args.client}: seeded {created} settings key(s) under "
                          f"{analyzer.settings_root}", flush=True)
                analyzer.apply_settings(client, force=True)

                client.register_brpc_callback(analyzer.serve)
                max_event = max_event_size(client, args.max_event_size)
                buf = client.open_event_buffer(args.buffer, None, max_event)
                analyzer.register_event_requests(client, buf)

                backoff.reset()
                last_health = 0.0
                last_poll = 0.0
                while not _stop:
                    analyzer.apply_settings(client)
                    # Under a CPU budget the loop turns over every few ms while
                    # a slot waits for its events; the run state need not.
                    if (analyzer.budget is None
                            or time.monotonic() - last_poll >= args.cycle_ms / 1000.0):
                        analyzer.poll_run_state(client)
                        last_poll = time.monotonic()
                    analyzer.run_once(client, buf, budget_s=args.decode_budget_s)
                    now = time.time()
                    if now - last_health > 10.0:
                        analyzer.check_daq_health(client)
                        last_health = now
                    client.communicate(analyzer.next_wait_ms(args.cycle_ms))

        except KeyboardInterrupt:
            break
        except Exception as exc:                       # noqa: BLE001
            if _stop:
                break
            delay = backoff.next()
            analyzer.reconnects += 1
            analyzer.connected_since = None
            print(f"{args.client}: lost MIDAS ({type(exc).__name__}: {exc}); "
                  f"retrying in {delay:.0f}s", file=sys.stderr, flush=True)
            for _ in range(int(delay * 10)):
                if _stop:
                    break
                time.sleep(0.1)

    print(f"{args.client}: stopped after {analyzer.processed} events "
          f"({analyzer.reconnects} reconnects)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
