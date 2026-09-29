"""The analyzer's safety properties.

A monitoring process must never be able to affect data taking, and the ways it
could are specific and few: back-pressuring the event buffer, sitting in the run
transition path, or doing unbounded work. Each gets an assertion here, because
each is a one-word edit away.
"""

from __future__ import annotations

import inspect
import json
import time

from mdqm.dqm import analyzer as A
from mdqm.dqm.hist import HistStore


class _FakePlugin:
    name = "fake"
    event_ids = frozenset({401})

    def __init__(self, store):
        self.store = store
        self.processed = []
        self.runs = []

    def accepts(self, event):
        return event.header.event_id in self.event_ids

    def process(self, event, run_number=None):
        self.processed.append(event)
        self.runs.append(run_number)
        return True

    def status(self):
        return {"plugin": self.name, "decoded": len(self.processed)}


class _Event:
    def __init__(self, event_id=401):
        self.header = type("H", (), {"event_id": event_id})()


class _FakeClient:
    def __init__(self, odb=None, dropped_series=None):
        self.odb = odb or {"/Runinfo/State": 1, "/Runinfo/Run number": 7}
        self.events = []
        self.messages = []
        self.requests = []
        self._dropped = list(dropped_series or [])

    def odb_get(self, path):
        if path == A.DROPPED_PATH and self._dropped:
            return self._dropped.pop(0)
        return self.odb[path]

    def receive_event(self, buf, async_flag=True, use_numpy=False):
        # use_numpy is accepted because the analyzer passes it: without it a
        # 33 kB TID_BYTE bank arrives as 33,000 Python ints. A fake that did not
        # accept it would hide that call site changing.
        return self.events.pop(0) if self.events else None

    def msg(self, text, is_error=False):
        self.messages.append(text)

    def register_event_request(self, buf, event_id=-1, trigger_mask=-1, sampling_type=None):
        self.requests.append((event_id, sampling_type))

    def communicate(self, timeout_ms):
        # The time-based yield can fire in any run_once on a slow machine, so
        # every fake has to accept it.
        pass


def _analyzer(rate=1000.0):
    return A.Analyzer(lambda store: _FakePlugin(store), rate=rate)


# --- the three ways this could hurt the DAQ ---------------------------------

def test_the_event_request_is_non_blocking():
    """GET_ALL would let a slow monitor stall the frontend producing the data."""
    import midas

    # main() delegates to the method; the requests themselves are made there.
    assert "analyzer.register_event_requests(client, buf)" in inspect.getsource(A.main)
    assert "register_event_request(" not in inspect.getsource(A.main).replace(
        "register_event_requests(", "")
    src = inspect.getsource(A.Analyzer.register_event_requests)
    # Check what is passed, not what is mentioned: the source says "never
    # GET_ALL" in a comment, and a bare substring test matches that too.
    assert "sampling_type=midas.GET_NONBLOCKING" in src
    assert "sampling_type=midas.GET_ALL" not in src, \
        "one word between a monitor and a throttle"
    assert "GET_RECENT" not in src.replace("# ", ""), "GET_RECENT would drop events silently"
    assert midas.GET_NONBLOCKING != midas.GET_ALL


def test_no_transition_callbacks_or_equipment_are_registered():
    """Either would put this client in the run-transition path.

    That is exactly how the retired publisher could delay a run start: it
    registered TR_START at sequence 100, so a wedged process held up every run
    until the watchdog reaped it.
    """
    src = inspect.getsource(A)
    for forbidden in ("register_transition_callback", "register_transition",
                      "EquipmentBase", "midas.frontend"):
        assert forbidden not in src, f"{forbidden} puts us in the transition path"
    # Run state is polled instead.
    assert "/Runinfo/State" in src


def test_run_state_is_polled_not_hooked():
    a = _analyzer()
    client = _FakeClient({"/Runinfo/State": 3, "/Runinfo/Run number": 42})
    a.poll_run_state(client)
    assert a.run_number == 42
    assert a.run_state == 3
    assert a.status()["run_active"] is True


# --- sampling ----------------------------------------------------------------

def test_the_buffer_is_drained_even_when_sampling_is_refused():
    """Draining is free with GET_NONBLOCKING and keeps the read pointer current."""
    a = _analyzer(rate=0.0)                 # refuse everything
    client = _FakeClient()
    client.events = [_Event() for _ in range(25)]

    drained = a.run_once(client, buf=None)

    assert drained == 25, "every event must be taken off the buffer"
    assert a.seen == 25
    assert a.processed == 0, "and none of them decoded"


def test_the_token_bucket_limits_the_decode_rate():
    bucket = A.TokenBucket(rate=5.0)
    taken = sum(1 for _ in range(100) if bucket.take())
    assert taken <= 6, f"took {taken} in one instant from a 5/s bucket"


def test_the_token_bucket_refills_over_time():
    bucket = A.TokenBucket(rate=50.0)
    while bucket.take():
        pass
    time.sleep(0.1)
    assert bucket.take(), "should have refilled after 100 ms at 50/s"


def test_events_from_other_equipment_are_ignored_cheaply():
    a = _analyzer()
    client = _FakeClient()
    client.events = [_Event(event_id=410) for _ in range(10)]
    a.run_once(client, buf=None)
    assert a.seen == 10
    assert a.processed == 0


# --- the self-throttle -------------------------------------------------------

def test_dropped_packets_halve_the_rate_and_say_so():
    a = _analyzer(rate=20.0)
    client = _FakeClient(dropped_series=[0, 5])

    a.check_daq_health(client)              # establishes the baseline
    assert a.bucket.rate == 20.0
    a.check_daq_health(client)              # 5 packets lost since

    assert a.bucket.rate == 10.0
    assert a.status()["throttled"] is True
    assert client.messages, "a silent throttle is not a diagnosis"
    assert "dropped 5 packets" in client.messages[0]


def test_the_throttle_bottoms_out_at_zero_rather_than_oscillating():
    a = _analyzer(rate=2.0)
    client = _FakeClient(dropped_series=[0, 1, 2, 3, 4, 5])
    for _ in range(6):
        a.check_daq_health(client)
    assert a.bucket.rate == 0.0
    # Still drains, still serves; it just stops decoding.
    client.events = [_Event() for _ in range(3)]
    assert a.run_once(client, buf=None) == 3
    assert a.processed == 0


def test_the_rate_is_never_restored_automatically():
    """Recovering throughput is an operator decision, not the monitor's."""
    a = _analyzer(rate=20.0)
    client = _FakeClient(dropped_series=[0, 1, 1, 1, 1])
    for _ in range(5):
        a.check_daq_health(client)
    assert a.bucket.rate == 10.0, "no further drops, but no automatic recovery either"


def test_a_missing_dropped_counter_is_not_an_error():
    """Not every experiment has this equipment; absence must not break us."""
    a = _analyzer()
    class _NoCounter(_FakeClient):
        def odb_get(self, path):
            if path == A.DROPPED_PATH:
                raise KeyError(path)
            return super().odb_get(path)
    a.check_daq_health(_NoCounter())
    assert a.bucket.rate == a.configured_rate


# --- surviving a MIDAS restart ----------------------------------------------

def test_the_histogram_store_is_created_outside_the_connection_loop():
    """A reconnect must not reset the plots someone is watching."""
    src = inspect.getsource(A.main)
    store_line = src.index("Analyzer(make_plugin_factory")
    loop_line = src.index("while not _stop:")
    assert store_line < loop_line, "the analyzer must outlive any one connection"


def test_the_client_is_used_as_a_context_manager():
    """client.py:186-196: disconnect() frees RPC *server* state that
    cm_disconnect_experiment() alone does not, and without it the second
    register_brpc_callback after a reconnect trips over what is left."""
    src = inspect.getsource(A.main)
    assert "with midas.client.MidasClient(" in src


def test_state_that_should_survive_a_reconnect_does():
    a = _analyzer()
    client = _FakeClient()
    client.events = [_Event() for _ in range(4)]
    a.run_once(client, buf=None)
    before = a.processed

    a.reconnects += 1                       # as the loop would on a failure
    client.events = [_Event() for _ in range(3)]
    a.run_once(client, buf=None)

    assert a.processed == before + 3, "counters and histograms carry across"
    # The plugin keeps filling the same store object rather than a fresh one,
    # which is what makes a MIDAS bounce invisible on the page.
    assert a.plugin.store is a.store
    assert isinstance(a.store, HistStore)


# --- the brpc callback -------------------------------------------------------

def test_serve_returns_bytes_the_page_can_read():
    a = _analyzer()
    status, buf = a.serve(None, "dqm::list", "", 65536)
    assert status == 1
    assert bytes(buf)[:8], "an 8-byte envelope at minimum"


def test_serve_truncates_rather_than_failing_when_the_buffer_is_small():
    """The page reads the true size from the header and retries bigger."""
    a = _analyzer()
    _status, buf = a.serve(None, "dqm::list", "", 4)
    assert len(bytes(buf)) <= 4


def test_serve_never_raises_into_the_rpc_thread():
    a = _analyzer()
    status, buf = a.serve(None, "dqm::histogram", "does-not-exist", 65536)
    assert status == 1
    assert b"no such histogram" in bytes(buf)


def test_an_unknown_namespace_is_left_for_another_handler():
    a = _analyzer()
    _status, buf = a.serve(None, "someoneelse::cmd", "", 65536)
    assert bytes(buf) == b""


# --- run transitions ---------------------------------------------------------

def test_a_new_run_clears_what_asked_to_be_cleared():
    from mdqm.dqm.hist import Axis, Hist1D

    a = _analyzer()
    keep = a.store.add(Hist1D("keep", Axis(4, 0, 1)))
    keep.clear_on_run_start = False
    drop = a.store.add(Hist1D("drop", Axis(4, 0, 1)))
    keep.fill([0.5])
    drop.fill([0.5])

    client = _FakeClient({"/Runinfo/State": 1, "/Runinfo/Run number": 7})
    a.poll_run_state(client)                       # first sight: no clear
    assert drop.entries == 1

    client.odb["/Runinfo/Run number"] = 8
    a.poll_run_state(client)

    assert drop.entries == 0, "run-start histograms reset"
    assert keep.entries == 1, "persistent ones do not"


def test_the_first_run_number_seen_does_not_clear():
    """Starting the analyzer mid-run must not wipe a page someone is reading."""
    from mdqm.dqm.hist import Axis, Hist1D

    a = _analyzer()
    h = a.store.add(Hist1D("h", Axis(4, 0, 1)))
    h.fill([0.5])
    a.poll_run_state(_FakeClient({"/Runinfo/State": 3, "/Runinfo/Run number": 99}))
    assert h.entries == 1


# --- live configuration ------------------------------------------------------

class _SettingsClient(_FakeClient):
    """A fake ODB that supports the settings tree."""

    def __init__(self, values=None):
        super().__init__()
        self.tree = dict(values or {})
        self.messages = []

    def odb_exists(self, path):
        return path in self.tree

    def odb_get(self, path):
        if path in self.tree:
            return self.tree[path]
        return super().odb_get(path)

    def odb_set(self, path, value):
        self.tree[path] = value


def _seeded_client(**overrides):
    from mdqm.dqm import settings as S

    c = _SettingsClient()
    S.seed(c)
    for k, v in overrides.items():
        c.tree[f"{S.ROOT}/{k}"] = v
    return c


def test_settings_are_seeded_once_and_not_overwritten():
    from mdqm.dqm import settings as S

    c = _SettingsClient()
    created = S.seed(c)
    assert created > 0
    c.tree[f"{S.ROOT}/Binning/persistence x bins"] = 999      # operator edit

    assert S.seed(c) == 0, "a second seed must create nothing"
    assert c.tree[f"{S.ROOT}/Binning/persistence x bins"] == 999, \
        "seeding must never overwrite what an operator changed"


def test_the_odb_is_the_authority_for_binning():
    a = _analyzer(rate=20.0)
    c = _seeded_client(**{"Binning/persistence x bins": 64,
                          "Binning/persistence y bins": 20})
    a.apply_settings(c, force=True)
    assert a.settings["Binning"]["persistence x bins"] == 64


def test_changing_the_binning_rebuilds_the_histograms():
    """A histogram with different bins is a different histogram."""
    from mdqm.plugins.wavedream import WaveDreamPlugin

    a = A.Analyzer(lambda s: WaveDreamPlugin(s), rate=20.0)
    c = _seeded_client()
    a.apply_settings(c, force=True)

    pers = a.store.get("wd/persistence_ch00")
    assert pers.x.n == 256, "the seeded default"
    pers.fill([1.0], [-0.5])
    assert pers.entries == 1

    before = a.reconfigures            # startup already applied the ODB once
    c.tree["/DQM/Analyzer/Binning/persistence x bins"] = 64
    a._settings_checked = 0
    assert a.apply_settings(c) is True

    rebuilt = a.store.get("wd/persistence_ch00")
    assert rebuilt.x.n == 64, "the new binning took effect"
    assert rebuilt.entries == 0, "old counts must not be carried into new bins"
    assert a.reconfigures == before + 1
    assert any("binning changed" in m for m in c.messages), \
        "a silent reset would look like data loss"


def test_a_role_change_does_not_throw_away_accumulated_plots():
    from mdqm.plugins.wavedream import WaveDreamPlugin

    a = A.Analyzer(lambda s: WaveDreamPlugin(s), rate=20.0)
    c = _seeded_client()
    a.apply_settings(c, force=True)

    a.store.get("wd/persistence_ch00").fill([1.0], [-0.5])
    assert a.store.get("wd/persistence_ch00").entries == 1

    before = a.reconfigures
    c.tree["/DQM/Analyzer/Channel roles/rf channel"] = 9
    a._settings_checked = 0
    a.apply_settings(c)

    assert a.store.get("wd/persistence_ch00").entries == 1, \
        "moving the RF channel does not change any histogram's shape"
    assert a.reconfigures == before, "no shape changed, so nothing was rebuilt"
    assert a.plugin.roles["rf channel"] == 9, "but the plugin must see it"


def test_the_sampling_rate_can_be_changed_live():
    a = _analyzer(rate=20.0)
    c = _seeded_client(**{"Sampling/max events per s": 250.0})
    a.apply_settings(c, force=True)
    assert a.bucket.rate == 250.0
    assert a.configured_rate == 250.0


def test_raising_the_rate_clears_a_self_throttle():
    """The operator has said what they want more recently than we did."""
    a = _analyzer(rate=20.0)
    c = _seeded_client()
    a.apply_settings(c, force=True)

    a.bucket.rate = 5.0                     # as the throttle would leave it
    c.tree["/DQM/Analyzer/Sampling/max events per s"] = 40.0
    a._settings_checked = 0
    a.apply_settings(c)
    assert a.bucket.rate == 40.0


def test_settings_are_not_re_read_on_every_cycle():
    a = _analyzer()
    c = _seeded_client()
    a.apply_settings(c, force=True)
    assert a.apply_settings(c) is False, "a 2 s floor keeps this off the hot path"


def test_a_broken_settings_tree_does_not_stop_the_analyzer():
    """An ODB nobody can read must leave a working analyzer on defaults.

    read() deliberately swallows a per-key failure and substitutes the default,
    so apply_settings *succeeds* here rather than failing -- it has adopted a
    known-good configuration, which is the outcome that matters. What must not
    happen is an exception reaching the run loop.
    """
    from mdqm.dqm import settings as S

    class _Broken(_SettingsClient):
        def odb_get(self, path):
            raise RuntimeError("ODB unhappy")

    a = _analyzer()
    a.apply_settings(_Broken(), force=True)          # must not raise

    assert a.settings is not None, "it must end up configured, not unconfigured"
    assert a.settings["Binning"]["persistence x bins"] == 256, "on the defaults"
    assert S.read(_Broken())["Binning"]["persistence x bins"] == 256


def test_status_reports_the_live_binning():
    a = _analyzer()
    c = _seeded_client(**{"Binning/amplitude max": 2.5})
    a.apply_settings(c, force=True)
    st = a.status()
    assert st["settings_root"] == "/DQM/Analyzer"
    assert st["binning"]["amplitude max"] == 2.5


# --- yielding to MIDAS while decoding ---------------------------------------

def test_decoding_yields_to_midas_periodically():
    """The brpc handler shares the interpreter and only runs when we let go.

    Measured at 500 ev/s offered: with no yield, wd::status stopped answering
    while the analyzer decoded perfectly -- so the pages reported an error about
    a healthy analyzer. Capping the work per cycle fixed that and cost a third of
    the throughput; yielding fixes it without the cap.
    """
    a = _analyzer(rate=100000.0)

    class _Counting(_FakeClient):
        def __init__(self):
            super().__init__()
            self.yields = 0

        def communicate(self, timeout_ms):
            self.yields += 1

    client = _Counting()
    client.events = [_Event() for _ in range(100)]
    a.run_once(client, buf=None)

    assert a.processed == 100
    expected = 100 // A.Analyzer.YIELD_EVERY
    assert client.yields >= expected - 1, \
        f"yielded {client.yields} times decoding 100 events"


def test_the_backstop_stops_decoding_but_never_the_drain():
    a = _analyzer(rate=100000.0)
    client = _FakeClient()
    client.events = [_Event() for _ in range(50)]
    client.communicate = lambda ms: None

    # A budget already spent: nothing should be decoded, everything drained.
    drained = a.run_once(client, buf=None, budget_s=-1.0)

    assert drained == 50, "the buffer must still be emptied"
    assert a.processed <= 1, "and decoding must stop almost immediately"
    assert a.budget_exhausted >= 1


def test_a_generous_backstop_is_not_normally_reached():
    a = _analyzer(rate=100000.0)
    client = _FakeClient()
    client.events = [_Event() for _ in range(200)]
    client.communicate = lambda ms: None

    a.run_once(client, buf=None, budget_s=2.0)
    assert a.budget_exhausted == 0, "the fake plugin is fast; nothing should trip"
    assert a.processed == 200


def test_the_odb_binning_is_applied_at_startup_not_only_on_change():
    """The bug: the first apply skipped the rebuild.

    The plugin's constructor builds its histograms from code defaults, having no
    ODB to read yet, so the first apply is exactly when the ODB has to be pushed
    in. Skipping it left the analyzer running on defaults while reporting the ODB
    values in its status -- caught on real cosmic data, where the persistence
    plot stayed at the default 110 bins over [-1.0, 0.1] after the ODB had been
    set to 180 over [-1.6, 0.2], and the status claimed the new numbers.
    """
    from mdqm.plugins.wavedream import WaveDreamPlugin

    a = A.Analyzer(lambda s: WaveDreamPlugin(s), rate=20.0)
    assert a.store.get("wd/persistence_ch00").y.n == 110, "the code default"

    c = _seeded_client(**{"Binning/persistence y bins": 180,
                          "Binning/persistence y min": -1.6,
                          "Binning/persistence y max": 0.2})
    a.apply_settings(c, force=True)

    pers = a.store.get("wd/persistence_ch00")
    assert pers.y.n == 180, "the ODB must win at startup, not just on a later edit"
    assert pers.y.lo == -1.6
    assert pers.y.hi == 0.2


def test_status_reports_the_binning_the_histograms_actually_have():
    """So a disagreement between config and reality is visible, not hidden."""
    from mdqm.plugins.wavedream import WaveDreamPlugin

    a = A.Analyzer(lambda s: WaveDreamPlugin(s), rate=20.0)
    c = _seeded_client(**{"Binning/persistence y bins": 44})
    a.apply_settings(c, force=True)

    axes = a.status()["axes"]
    assert "wd/persistence_ch00" in axes
    y = axes["wd/persistence_ch00"][1]
    assert y["bins"] == 44, "read off the histogram, not off the request"


def test_startup_with_an_unseeded_odb_keeps_the_code_defaults():
    a = _analyzer()
    a.apply_settings(_SettingsClient(), force=True)     # nothing seeded
    assert a.settings["Binning"]["persistence y bins"] == 110


# --- a second plugin beside the WaveDREAM one --------------------------------

class _OwnSettingsPlugin(_FakePlugin):
    """A plugin with its own settings tree, as the SMA one will have."""

    name = "own"
    event_ids = frozenset({301})
    settings_root = "/DQM/Own"
    settings_defaults = {
        "Coarse shift": 14,
        "Binning": {"tot bins": 64},
        "Sampling": {"max events per s": 20.0, "process all": True},
    }
    dropped_path = ""
    client_name = "own_analyzer"
    process_all = True                 # before any settings tree is read

    def __init__(self, store):
        super().__init__(store)
        self.applied = []

    def shape_fingerprint(self, settings):
        return str(settings["Binning"])

    def apply_settings(self, settings, rebuild):
        self.applied.append((settings, rebuild))

    def commands(self):
        from mdqm.dqm import framing
        return {"own::echo": lambda args: framing.envelope(
            framing.TAG_JSON, json.dumps({"echo": args}).encode())}


def _own(rate=20.0, **kw):
    return A.Analyzer(lambda store: _OwnSettingsPlugin(store), rate=rate, **kw)


def test_a_plugin_with_its_own_root_never_touches_the_wavedream_tree():
    from mdqm.dqm import settings as S

    a = _own()

    class _Watching(_SettingsClient):
        def __init__(self):
            super().__init__()
            self.read_paths = []

        def odb_get(self, path):
            self.read_paths.append(path)
            return super().odb_get(path)

    c = _Watching()
    created = S.seed(c, a.settings_root, a.settings_defaults)
    assert created == 4
    assert c.tree["/DQM/Own/Coarse shift"] == 14, "a key at the root, not in a section"
    assert c.tree["/DQM/Own/Binning/tot bins"] == 64

    c.tree["/DQM/Own/Coarse shift"] = 13
    a.apply_settings(c, force=True)

    touched = list(c.tree) + c.read_paths
    assert not [p for p in touched if p.startswith(S.ROOT)], \
        "two analyzers in one tree would reconfigure each other"
    assert a.settings["Coarse shift"] == 13
    assert a.status()["settings_root"] == "/DQM/Own"


def test_the_plugin_applies_its_own_settings_and_is_told_when_to_rebuild():
    a = _own()
    c = _SettingsClient()
    from mdqm.dqm import settings as S
    S.seed(c, a.settings_root, a.settings_defaults)

    a.apply_settings(c, force=True)
    assert a.plugin.applied[-1][1] is True, "the first apply rebuilds from the ODB"
    assert a.reconfigures == 1
    assert not hasattr(a.plugin, "roles"), "the WaveDREAM path must not run"

    c.tree["/DQM/Own/Coarse shift"] = 15
    a._settings_checked = 0
    assert a.apply_settings(c) is True
    assert a.plugin.applied[-1] == (a.settings, False), "not a shape change"
    assert a.reconfigures == 1

    c.tree["/DQM/Own/Binning/tot bins"] = 32
    a._settings_checked = 0
    a.apply_settings(c)
    assert a.plugin.applied[-1][1] is True
    assert a.reconfigures == 2
    assert any("binning changed; rebuilt" in m for m in c.messages)


def test_half_a_settings_declaration_is_refused():
    import pytest

    class _Half(_FakePlugin):
        settings_root = "/DQM/Half"

    with pytest.raises(ValueError, match="both"):
        A.Analyzer(lambda store: _Half(store))


def test_process_all_bypasses_the_bucket():
    a = _own(rate=1.0)
    client = _FakeClient()
    client.events = [_Event(301) for _ in range(200)]
    a.run_once(client, buf=None)
    assert a.processed == 200, "every accepted event is decoded"
    assert a.status()["sampling_bypassed"] is True


def test_the_process_all_backstop_leaves_events_for_the_next_cycle():
    """A backlog must delay process-all frames, not drop them undecoded."""
    a = _own(rate=1.0)
    client = _FakeClient()
    client.events = [_Event(301) for _ in range(50)]

    # A budget already spent: stop after the first event, drain nothing more.
    drained = a.run_once(client, buf=None, budget_s=-1.0)
    assert drained == 1 and a.processed == 1
    assert a.budget_exhausted == 1
    assert len(client.events) == 49, "the rest stays in the buffer"

    a.run_once(client, buf=None, budget_s=2.0)
    assert a.processed == 50, "and is decoded next cycle, none lost"


def test_the_process_all_setting_overrides_the_plugin_attribute():
    from mdqm.dqm import settings as S

    a = _own(rate=1.0)
    c = _SettingsClient()
    S.seed(c, a.settings_root, a.settings_defaults)
    c.tree["/DQM/Own/Sampling/process all"] = False
    c.tree["/DQM/Own/Sampling/max events per s"] = 1.0
    a.apply_settings(c, force=True)

    c.events = [_Event(301) for _ in range(100)]
    a.run_once(c, buf=None)
    assert a.processed <= 2, "sampling again: the ODB said so"


def test_a_throttle_re_engages_the_bucket_for_process_all():
    """The DAQ-health valve must work whatever the plugin prefers."""
    a = _own(rate=4.0, dropped_path="/Equipment/X/Dropped")
    client = _FakeClient(dropped_series=[0, 3])
    client.odb["/Equipment/X/Dropped"] = 0

    def odb_get(path, _orig=client.odb_get):
        if path == "/Equipment/X/Dropped":
            return client._dropped.pop(0)
        return _orig(path)
    client.odb_get = odb_get

    a.check_daq_health(client)
    a.check_daq_health(client)
    assert a.bucket.rate == 2.0
    assert a.status()["sampling_bypassed"] is False

    client.events = [_Event(301) for _ in range(100)]
    a.run_once(client, buf=None)
    assert a.processed <= 3, "the bucket applies again once throttled"


def test_the_wavedream_default_still_samples():
    from mdqm.plugins.wavedream import WaveDreamPlugin

    a = A.Analyzer(lambda s: WaveDreamPlugin(s), rate=20.0)
    a.apply_settings(_seeded_client(), force=True)
    assert a.process_all is False
    assert a.status()["sampling_bypassed"] is False
    assert a.bucket.rate == 20.0


def test_an_empty_dropped_path_disables_the_health_check():
    a = _own()
    assert a.dropped_path == "", "the plugin's choice"

    class _Loud(_FakeClient):
        def odb_get(self, path):
            raise AssertionError(f"read {path} with the check disabled")

    a.check_daq_health(_Loud())
    assert a.bucket.rate == a.configured_rate


def test_the_dropped_path_comes_from_the_plugin_or_the_command_line():
    from mdqm.plugins.wavedream import WaveDreamPlugin

    assert A.Analyzer(lambda s: WaveDreamPlugin(s)).dropped_path == A.DROPPED_PATH
    assert _own(dropped_path="/X").dropped_path == "/X"
    assert _analyzer().dropped_path == A.DROPPED_PATH, "legacy plugins keep the WD counter"


def test_one_event_request_per_event_id_all_non_blocking():
    import midas

    from mdqm.plugins.wavedream import WaveDreamPlugin

    client = _FakeClient()
    A.Analyzer(lambda s: WaveDreamPlugin(s)).register_event_requests(client, buf=None)
    assert client.requests == [(1, midas.GET_NONBLOCKING), (401, midas.GET_NONBLOCKING)]

    client = _FakeClient()
    _own().register_event_requests(client, buf=None)
    assert client.requests == [(301, midas.GET_NONBLOCKING)], \
        "the SMA client must not be handed waveform events"

    class _Any(_FakePlugin):
        event_ids = frozenset()

    client = _FakeClient()
    A.Analyzer(lambda s: _Any(s)).register_event_requests(client, buf=None)
    assert client.requests == [(-1, midas.GET_NONBLOCKING)]


def test_decoding_yields_after_a_time_even_below_the_event_count():
    """A few heavy frames must not hold off the RPC handler for their sum."""

    class _Slow(_FakePlugin):
        def process(self, event, run_number=None):
            time.sleep(0.01)
            return super().process(event, run_number)

    a = A.Analyzer(lambda s: _Slow(s), rate=100000.0)
    a.YIELD_AFTER_S = 0.015

    class _Counting(_FakeClient):
        yields = 0

        def communicate(self, timeout_ms):
            self.yields += 1

    client = _Counting()
    client.events = [_Event() for _ in range(6)]
    a.run_once(client, buf=None)

    assert a.processed == 6
    assert A.Analyzer.YIELD_EVERY > 6, "the count alone would never yield here"
    assert client.yields >= 2, f"yielded {client.yields} times over ~60 ms of decoding"
    assert A.Analyzer.YIELD_AFTER_S == 0.05


def test_status_names_the_client_it_runs_as():
    assert _analyzer().status()["client"] == A.DEFAULT_CLIENT
    a = _own(client_name="sma_analyzer")
    assert a.status()["client"] == "sma_analyzer"


def test_messages_carry_the_client_name():
    a = _own(client_name="sma_analyzer", dropped_path="/D")
    client = _FakeClient(dropped_series=[0, 2])
    client.odb_get = lambda path: client._dropped.pop(0)
    a.check_daq_health(client)
    a.check_daq_health(client)
    assert client.messages[0].startswith("sma_analyzer: DAQ dropped 2 packets")


def test_dqm_status_and_defs_alias_the_wd_commands():
    from mdqm.dqm import framing

    a = _own(client_name="sma_analyzer")
    for alias, original in (("dqm::status", "wd::status"), ("dqm::defs", "wd::defs")):
        _st, got = a.serve(None, alias, "", 65536)
        _size, tag, body = framing.parse_envelope(bytes(got))
        assert tag == framing.TAG_JSON, alias
        payload = json.loads(body)
        _st, ref = a.serve(None, original, "", 65536)
        ref_payload = json.loads(framing.parse_envelope(bytes(ref))[2])
        assert payload.keys() == ref_payload.keys(), alias
    assert json.loads(framing.parse_envelope(bytes(
        a.serve(None, "dqm::defs", "", 65536)[1]))[2])["plugin"] == "own"


def test_plugin_commands_are_dispatched():
    from mdqm.dqm import framing

    a = _own()
    _st, got = a.serve(None, "own::echo", "hi", 65536)
    _size, tag, body = framing.parse_envelope(bytes(got))
    assert tag == framing.TAG_JSON
    assert body == b'{"echo": "hi"}'
    # Unknown commands in the plugin's namespace are still left for others.
    assert bytes(a.serve(None, "own::nope", "", 65536)[1]) == b""


def test_a_plugin_command_cannot_shadow_a_builtin():
    from mdqm.dqm.server import Server

    srv = Server(HistStore(), extra={"dqm::list": lambda args: b"hijacked"})
    assert srv.dispatch("dqm::list", "") != b"hijacked"


def test_a_failing_plugin_command_is_reported_not_raised():
    from mdqm.dqm import framing
    from mdqm.dqm.server import Server

    def boom(args):
        raise RuntimeError("frame gone")

    srv = Server(HistStore(), extra={"sma::frame": boom})
    _size, tag, body = framing.parse_envelope(srv.dispatch("sma::frame", ""))
    assert tag == framing.TAG_ERROR
    assert b"frame gone" in body


def test_the_sma_plugin_is_known_and_its_absence_is_explained(monkeypatch):
    import builtins

    import pytest

    real_import = builtins.__import__

    def no_sma(name, *args, **kwargs):
        if name == "mdqm.plugins.sma":
            raise ModuleNotFoundError("No module named 'mdqm.plugins.sma'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_sma)
    with pytest.raises(SystemExit, match="plugin 'sma' cannot be loaded"):
        A.make_plugin_factory("sma")
    with pytest.raises(SystemExit, match="known: wavedream, sma"):
        A.make_plugin_factory("nope")


# --- review follow-ups ---------------------------------------------------------

def test_an_existing_wavedream_tree_gains_no_keys():
    """Upgrading the analyzer must not add anything to a running experiment's tree."""
    from mdqm.dqm import settings as S

    # The tree the WaveDREAM analyzer seeded before plugins had their own roots.
    existing = {f"{S.ROOT}/{section}/{key}": 0 for section, keys in {
        "Channel roles": ["waveform channels", "s1 channel", "rf channel",
                          "nim channels", "nim threshold V", "labels"],
        "Binning": ["persistence x bins", "persistence y bins", "persistence y min",
                    "persistence y max", "amplitude bins", "amplitude min",
                    "amplitude max", "deltat bins", "deltat max s", "phase bins"],
        "Sampling": ["max events per s", "publish history"],
    }.items() for key in keys}
    c = _SettingsClient(existing)
    assert S.seed(c) == 0, "no new key in an experiment that already has the tree"
    assert set(c.tree) == set(existing)
    assert S.seed(_SettingsClient()) == len(existing), "and nothing missing either"


def test_a_process_all_key_added_by_hand_to_the_wavedream_tree_is_ignored():
    from mdqm.plugins.wavedream import WaveDreamPlugin

    a = A.Analyzer(lambda s: WaveDreamPlugin(s), rate=20.0)
    c = _seeded_client(**{"Sampling/process all": True})
    a.apply_settings(c, force=True)
    assert a.process_all is False
    assert "process all" not in a.settings["Sampling"]


def test_rate_zero_stops_decoding_even_with_process_all():
    from mdqm.dqm import settings as S

    a = _own()
    c = _SettingsClient()
    S.seed(c, a.settings_root, a.settings_defaults)
    c.tree["/DQM/Own/Sampling/max events per s"] = 0.0
    a.apply_settings(c, force=True)
    assert a.process_all is True

    c.events = [_Event(301) for _ in range(20)]
    assert a.run_once(c, buf=None) == 20, "still drained"
    assert a.processed == 0, "zero means decode nothing"
    assert a.status()["sampling_bypassed"] is False


def test_a_plugin_that_fails_to_apply_settings_is_retried():
    from mdqm.dqm import settings as S

    a = _own()
    c = _SettingsClient()
    S.seed(c, a.settings_root, a.settings_defaults)
    a.apply_settings(c, force=True)
    before = a.settings

    fail = {"n": 2}
    good = a.plugin.apply_settings

    def flaky(settings, rebuild):
        if fail["n"]:
            fail["n"] -= 1
            raise RuntimeError("rebuild failed")
        good(settings, rebuild)
    a.plugin.apply_settings = flaky

    c.tree["/DQM/Own/Binning/tot bins"] = 32
    for _ in range(2):
        a._settings_checked = 0
        assert a.apply_settings(c) is False
        assert a.settings is before, "nothing recorded until the plugin took it"
    errors = [m for m in c.messages if "could not apply" in m]
    assert len(errors) == 1, "said once, not on every poll"

    a._settings_checked = 0
    assert a.apply_settings(c) is True, "the same change is retried and lands"
    assert a.settings["Binning"]["tot bins"] == 32
    assert a.plugin.applied[-1][1] is True, "still a rebuild: the shape never took"


def test_the_client_name_defaults_to_the_plugin_s():
    assert _own().status()["client"] == "own_analyzer"
    assert _own(client_name="x").client_name == "x"
    assert _analyzer().client_name == A.DEFAULT_CLIENT


def test_a_plugin_command_that_returns_no_bytes_is_an_error_frame():
    from mdqm.dqm import framing
    from mdqm.dqm.server import Server

    srv = Server(HistStore(), extra={"sma::summary": lambda args: {"not": "bytes"}})
    _size, tag, body = framing.parse_envelope(srv.dispatch("sma::summary", ""))
    assert tag == framing.TAG_ERROR
    assert b"not framed bytes" in body
