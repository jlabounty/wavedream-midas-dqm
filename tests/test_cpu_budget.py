"""The CPU budget: the controller, the skip-based reader and the S1 cap.

The analyzer for the SMA must stay within a fixed share of one core however
fast the DAQ sends. These tests drive the controller with a fake clock and a
fake CPU meter (cost = idle + per-event cost x analysed), the budget loop with
a fake buffer that counts what is copied out of it, and the S1 cap with
synthetic frames whose truth is known.
"""

from __future__ import annotations

import numpy as np
import pytest

from mdqm.dqm import analyzer as A
from mdqm.dqm import settings as S
from mdqm.dqm.hist import HistStore
from mdqm.plugins import sma as P
from mdqm.plugins import sma_words as W


class _Time:
    """A fake wall clock and a fake process CPU clock, advanced together."""

    def __init__(self):
        self.t = 1000.0
        self.c = 0.0

    def clock(self):
        return self.t

    def cpu(self):
        return self.c


def _simulate(ctl, tm, offered, cost_s, idle_frac=0.01, seconds=30.0, dt=0.05, n0=0,
              record=None):
    """Run the controller against a process that analyses min(offered, rate)/s.

    Returns the running count of analysed events. `record` gets (t, cpu
    fraction of this step, rate).
    """
    n = float(n0)
    steps = int(round(seconds / dt))
    for _ in range(steps):
        done = min(offered, ctl.rate) * dt
        n += done
        step_cpu = idle_frac * dt + done * cost_s
        tm.t += dt
        tm.c += step_cpu
        ctl.update(int(n))
        if record is not None:
            record.append((tm.t, step_cpu / dt, ctl.rate))
    return n


# --- the controller -------------------------------------------------------------------

def test_the_controller_settles_at_the_budget_when_the_offer_is_larger():
    tm = _Time()
    ctl = A.CpuBudget(20.0, 1000.0, clock=tm.clock, cpu=tm.cpu)
    rec = []
    _simulate(ctl, tm, offered=350.0, cost_s=0.0065, seconds=30.0, record=rec)
    late = [f for t, f, _r in rec if t > tm.t - 20.0]
    assert max(late) <= 0.20, "never above the budget once settled"
    assert np.mean(late) == pytest.approx(0.18, abs=0.01), "90 % headroom, idle included"
    # (0.18 - 0.01) / 0.0065 = 26 events/s
    assert ctl.rate == pytest.approx(26.2, rel=0.03)
    assert ctl.cpu_pct() == pytest.approx(18.0, abs=1.0)


def test_below_the_budget_everything_offered_is_analysed():
    tm = _Time()
    ctl = A.CpuBudget(20.0, 1000.0, clock=tm.clock, cpu=tm.cpu)
    rec = []
    _simulate(ctl, tm, offered=10.0, cost_s=0.0065, seconds=20.0, record=rec)
    assert ctl.rate > 10.0, "the limit sits above the offer: nothing is skipped"
    assert ctl.rate <= 2 * 10.0 + 1e-9, "but not far above it (bounds a jump in the offer)"
    assert rec[-1][1] == pytest.approx(0.01 + 10 * 0.0065, rel=0.01)


def test_a_tenfold_jump_in_the_offer_settles_within_a_few_seconds():
    tm = _Time()
    ctl = A.CpuBudget(20.0, 1000.0, clock=tm.clock, cpu=tm.cpu)
    n = _simulate(ctl, tm, offered=35.0, cost_s=0.0065, seconds=20.0)
    t_step = tm.t
    rec = []
    _simulate(ctl, tm, offered=350.0, cost_s=0.0065, seconds=20.0, n0=n, record=rec)
    over = [t - t_step for t, f, _r in rec if f > 0.21]
    assert not over or max(over) <= 3.0, f"over budget until {max(over):.1f} s after the step"
    peak = max(f for _t, f, _r in rec)
    assert peak <= 0.20 * 2.5, "the transient is bounded by the step-up limit"


def test_a_busier_machine_makes_the_controller_back_off():
    """Contention doubles the CPU per event: the rate halves, the CPU does not grow."""
    tm = _Time()
    ctl = A.CpuBudget(20.0, 1000.0, clock=tm.clock, cpu=tm.cpu)
    n = _simulate(ctl, tm, offered=350.0, cost_s=0.0065, seconds=20.0)
    quiet = ctl.rate
    rec = []
    _simulate(ctl, tm, offered=350.0, cost_s=0.013, seconds=20.0, n0=n, record=rec)
    assert ctl.rate == pytest.approx(quiet * (0.18 - 0.01) / 0.013 / ((0.18 - 0.01) / 0.0065),
                                     rel=0.05)
    assert max(f for t, f, _r in rec if t > tm.t - 15.0) <= 0.20


@pytest.mark.parametrize("budget", [5.0, 50.0])
def test_other_budgets(budget):
    tm = _Time()
    ctl = A.CpuBudget(budget, 1000.0, clock=tm.clock, cpu=tm.cpu)
    rec = []
    _simulate(ctl, tm, offered=350.0, cost_s=0.0065, seconds=30.0, record=rec)
    late = [f for t, f, _r in rec if t > tm.t - 15.0]
    assert max(late) <= budget / 100.0
    assert np.mean(late) == pytest.approx(0.9 * budget / 100.0, rel=0.05)


def test_the_hard_cap_and_a_zero_budget():
    tm = _Time()
    ctl = A.CpuBudget(50.0, 5.0, clock=tm.clock, cpu=tm.cpu)
    _simulate(ctl, tm, offered=350.0, cost_s=0.0065, seconds=10.0)
    assert ctl.rate == 5.0, "max events per s stays a hard upper limit"
    ctl.configure(0.0, 5.0)
    assert ctl.rate == 0.0
    _simulate(ctl, tm, offered=350.0, cost_s=0.0065, seconds=5.0)
    assert ctl.rate == 0.0, "0 % analyses nothing"


def test_idle_cost_above_the_budget_drives_the_rate_to_the_floor():
    tm = _Time()
    ctl = A.CpuBudget(5.0, 1000.0, clock=tm.clock, cpu=tm.cpu)
    _simulate(ctl, tm, offered=350.0, cost_s=0.0065, idle_frac=0.08, seconds=20.0)
    # Near the floor (whole events per window make it hop between 0.2 and ~0.5).
    assert ctl.MIN_RATE <= ctl.rate < 1.0


def test_a_budget_edit_scales_the_rate_at_once():
    tm = _Time()
    ctl = A.CpuBudget(20.0, 1000.0, clock=tm.clock, cpu=tm.cpu)
    _simulate(ctl, tm, offered=350.0, cost_s=0.0065, seconds=10.0)
    r = ctl.rate
    ctl.configure(10.0, 1000.0)
    assert ctl.rate == pytest.approx(r / 2)


# --- the budget loop over a fake buffer ----------------------------------------------------

class _Hdr:
    def __init__(self, serial, size, event_id=301):
        self.event_id = event_id
        self.serial_number = serial
        self.event_data_size_bytes = size


class _Ev:
    def __init__(self, serial, size=320_000, event_id=301):
        self.header = _Hdr(serial, size, event_id)


class _Buffer:
    """A MIDAS buffer as this client sees it: its unread events, in order.

    `copied` counts events handed to Python (receive_event), `skips` the
    bm_skip_event calls; skipping drops the backlog without copying it.
    """

    def __init__(self):
        self.pending = []
        self.serial = 0
        self.copied = 0
        self.skips = 0
        self.requests = []

    def produce(self, n, size=320_000):
        for _ in range(n):
            self.pending.append(_Ev(self.serial, size))
            self.serial += 1

    # the client API the analyzer uses
    def receive_event(self, buf, async_flag=True, use_numpy=False):
        if not self.pending:
            return None
        self.copied += 1
        return self.pending.pop(0)

    def skip_event(self, buf):
        self.skips += 1
        self.pending.clear()

    def buffer_level(self, buf):
        return sum(((e.header.event_data_size_bytes + 16 + 7) // 8) * 8 for e in self.pending)

    def communicate(self, ms):
        pass

    def register_event_request(self, buf, event_id=-1, trigger_mask=-1, sampling_type=None):
        self.requests.append((event_id, sampling_type))


class _BudgetPlugin:
    name = "budgeted"
    event_ids = frozenset({301})
    cpu_budget_pct = 20.0
    sample_burst = 2
    dropped_path = ""

    def __init__(self, store):
        self.store = store
        self.serials = []
        self.sampling_state = None

    def accepts(self, event):
        return event.header.event_id in self.event_ids

    def process(self, event, run_number=None):
        self.serials.append(event.header.serial_number)
        return True

    def status(self):
        return {}


def _budget_analyzer(tm):
    return A.Analyzer(lambda s: _BudgetPlugin(s), rate=1000.0, clock=tm.clock, cpu=tm.cpu)


def _drive(a, buf, tm, offered, seconds, cost_s=0.0065, dt=0.005):
    """Offer `offered` events/s for `seconds`, calling run_once as the main loop would."""
    acc = 0.0
    for _ in range(int(round(seconds / dt))):
        acc += offered * dt
        k = int(acc)
        acc -= k
        buf.produce(k)
        before = a.processed
        a.run_once(buf, buf=0)
        tm.t += dt
        tm.c += 0.01 * dt + (a.processed - before) * cost_s


def test_the_budget_loop_skips_without_copying_and_keeps_pairs_consecutive():
    tm = _Time()
    a = _budget_analyzer(tm)
    buf = _Buffer()
    assert a.sampling_mode() == "cpu budget"
    _drive(a, buf, tm, offered=350.0, seconds=30.0)

    served = a.plugin.serials
    assert buf.copied == len(served), "nothing is copied out of the buffer but what is analysed"
    assert buf.skips > 0 and a.status()["skip_method"] == "client"
    frac = len(served) / buf.serial
    assert 0.03 < frac < 0.15, f"analysed {frac:.1%} of 350/s at 20 % of a core"
    # Slots of two: every analysed frame has its neighbour analysed too.
    s = np.array(served)
    pairs = np.count_nonzero(np.diff(s) == 1)
    assert pairs >= len(s) // 2 - 1
    st = a.status()
    assert st["cpu_pct"] <= 20.0 and st["analysis_rate_limit"] > 0
    assert a.plugin.sampling_state["mode"] == "cpu budget"


def test_when_the_budget_keeps_up_nothing_is_skipped():
    tm = _Time()
    a = _budget_analyzer(tm)
    buf = _Buffer()
    _drive(a, buf, tm, offered=10.0, seconds=20.0)
    s = a.plugin.serials
    assert s == list(range(len(s))), "every frame, in order"
    assert len(s) >= buf.serial - 2
    assert buf.skips <= 1


def test_a_zero_budget_reads_nothing_and_keeps_up_with_the_writer():
    tm = _Time()
    a = _budget_analyzer(tm)
    buf = _Buffer()
    a._set_budget(0.0)
    _drive(a, buf, tm, offered=350.0, seconds=5.0)
    assert a.processed == 0
    assert buf.copied == 0, "idle costs one skip per cycle, never a copy"
    assert buf.skips > 0


def _unbounded(tm):
    return A.Analyzer(lambda s: _BudgetPlugin(s), rate=1000.0, clock=tm.clock, cpu=tm.cpu,
                      no_cpu_budget=True)


def test_no_cpu_budget_is_process_all_and_the_odb_cannot_lift_the_budget_past_50():
    tm = _Time()
    a = _budget_analyzer(tm)
    a._set_budget(100.0)
    assert a.budget is not None and a.budget.budget_pct == 50.0, "the ODB is clamped"
    assert a.status()["cpu_budget_clamped"] is True
    a = _unbounded(tm)
    a._set_budget(20.0)
    assert a.budget is None and a.process_all is True, "--no-cpu-budget wins"
    buf = _Buffer()
    buf.produce(50)
    a.run_once(buf, buf=0)
    assert a.plugin.serials == list(range(50))


def test_the_odb_budget_key_selects_the_mode():
    from test_analyzer import _SettingsClient

    a = A.Analyzer(A.make_plugin_factory("sma"))
    c = _SettingsClient()
    S.seed(c, a.settings_root, a.settings_defaults)
    a.apply_settings(c, force=True)
    assert a.sampling_mode() == "cpu budget" and a.budget.budget_pct == 20.0
    c.tree["/DQM/SMA/Sampling/CPU budget %"] = 150.0
    a._settings_checked = 0
    a.apply_settings(c)
    assert a.sampling_mode() == "cpu budget" and a.budget.budget_pct == 50.0
    c.tree["/DQM/SMA/Sampling/CPU budget %"] = 5.0
    a._settings_checked = 0
    a.apply_settings(c)
    assert a.sampling_mode() == "cpu budget" and a.budget.budget_pct == 5.0


def test_a_truncated_event_is_counted_and_skipped():
    import midas

    class _Trunc(_Buffer):
        def receive_event(self, buf, async_flag=True, use_numpy=False):
            if self.pending and self.pending[0].header.event_data_size_bytes > 10**9:
                self.pending.pop(0)
                raise midas.MidasError(midas.status_codes["BM_TRUNCATED"], "truncated")
            return super().receive_event(buf, async_flag, use_numpy)

    tm = _Time()
    a = _unbounded(tm)
    buf = _Trunc()
    buf.produce(2)
    buf.produce(1, size=2 * 10**9)
    buf.produce(2)
    a.run_once(buf, buf=0)
    assert a.truncated == 1
    assert a.plugin.serials == [0, 1, 3, 4], "the analyzer goes on after it"


def test_the_max_event_size_default_follows_the_odb_within_limits():
    class _C:
        def __init__(self, v):
            self.v = v

        def odb_get(self, path):
            assert path == "/Experiment/MAX_EVENT_SIZE"
            if self.v is None:
                raise KeyError(path)
            return self.v

    assert A.max_event_size(_C(None), None) == A.DEFAULT_EVENT_SIZE
    assert A.max_event_size(_C(32 << 20), None) == 32 << 20
    assert A.max_event_size(_C(1 << 30), None) == A.MAX_EVENT_CAP
    assert A.max_event_size(_C(1 << 20), None) == A.DEFAULT_EVENT_SIZE
    assert A.max_event_size(_C(None), 12345) == 12345, "an explicit value wins"


def test_buffer_ops_finds_the_midas_library_symbols():
    calls = []

    class _Fn:
        def __init__(self, name):
            self.name = name
            self.argtypes = None

        def __call__(self, *args):
            calls.append(self.name)
            if self.name.startswith("_Z19"):
                args[1]._obj.value = 4242
            return 1

    class _Lib:
        def __getitem__(self, name):
            if name.startswith("c_"):
                raise AttributeError(name)
            return _Fn(name)

    class _Client:
        lib = _Lib()

    ops = A.BufferOps(_Client(), 3)
    assert ops.method == "bm_skip_event"
    assert ops.skip() == 0 and calls == ["_Z13bm_skip_eventi"]
    assert ops.level() == 4242


def test_the_wavedream_analyzer_is_untouched_by_the_budget():
    from mdqm.plugins.wavedream import WaveDreamPlugin

    a = A.Analyzer(lambda s: WaveDreamPlugin(s), rate=20.0)
    assert a.budget is None and a.sampling_mode() == "rate"
    assert "CPU budget %" not in S.SAMPLING and "process all" not in S.SAMPLING
    assert S.SAMPLING == {"max events per s": 20.0, "publish history": False}
    assert a.next_wait_ms(200) == 200, "the WD loop keeps its cycle"


# --- the S1 cap ------------------------------------------------------------------------------

def _dense_frame(n_s1=5000, eff=(1.0, 0.7, 0.9, 0.5, 0.8), seed=3, t0=10**12):
    """S1 every ~600 ns, counters k with efficiency eff[k] (random), RF bursts."""
    rng = np.random.default_rng(seed)
    t1 = t0 + np.cumsum(rng.integers(300, 900, n_s1)).astype(np.int64)
    ts, chs = [t1], [np.full(n_s1, 1)]
    for k, c in enumerate((2, 3, 4, 5), start=1):
        hit = rng.random(n_s1) < eff[k]
        ts.append(t1[hit] + rng.integers(0, 8, int(hit.sum())))
        chs.append(np.full(int(hit.sum()), c))
    rf = np.arange(t1[0] - 200, t1[-1] + 200, 20, dtype=np.int64)
    ts.append(rf)
    chs.append(np.full(rf.size, 6))
    t = np.concatenate(ts)
    ch = np.concatenate(chs)
    o = np.argsort(t, kind="stable")
    coarse, fine = W.fields_of(t[o], 14)
    return W.encode(ch[o], np.full(t.size, 30), coarse, fine)


def _prepared(words):
    return W.prepare_frame(words, 14)


def test_the_s1_sample_is_deterministic_and_evenly_spread():
    a = W.even_sample(5000, 2000)
    assert np.array_equal(a, W.even_sample(5000, 2000))
    assert a.size == 2000 and np.all(np.diff(a) > 0)
    assert a[0] < 5000 / 2000 and a[-1] > 5000 - 5000 / 2000
    assert np.array_equal(W.even_sample(10, 20), np.arange(10))


def test_no_cap_below_the_limit_is_bit_identical():
    fr = _prepared(_dense_frame(n_s1=1500))
    full = W.analyse_frame(fr)
    capped = W.analyse_frame(fr, cuts=W.Cuts(max_s1=2000))
    assert capped.sample is None
    for f in ("t_s1", "pattern", "partner_counts", "rf_n", "rf_valid", "seeds"):
        assert np.array_equal(getattr(full, f), getattr(capped, f)), f
    assert np.array_equal(full.rf_phase, capped.rf_phase, equal_nan=True)


def test_the_capped_pattern_is_an_unbiased_sample():
    fr = _prepared(_dense_frame(n_s1=20000))
    full = W.analyse_frame(fr)
    cap = W.analyse_frame(fr, cuts=W.Cuts(max_s1=2000))
    smp = cap.sampled()
    assert smp.t_s1.size == 2000 and cap.n_s1_kept == 20000
    eff_full = np.count_nonzero(full.partner_counts > 0, axis=0) / full.t_s1.size
    eff_cap = np.count_nonzero(smp.partner_counts > 0, axis=0) / smp.t_s1.size
    # Binomial error on 2000 draws: at most ~0.011 (1 sigma) at 0.5.
    assert np.allclose(eff_cap, eff_full, atol=0.035), (eff_cap, eff_full)
    assert np.allclose(eff_full, [1.0, 0.7, 0.9, 0.5, 0.8], atol=0.02)
    # The same S1 hits give the same answers: the sample rows are the full rows.
    rows = np.searchsorted(full.t_s1, smp.t_s1)
    assert np.array_equal(full.pattern[rows], smp.pattern)
    assert np.array_equal(full.rf_valid[rows], smp.rf_valid)
    # Time differences only come from the sampled S1 hits.
    for c, (i, dt) in smp.dt.items():
        fi, fdt = full.dt[c]
        keep = np.isin(fi, rows)
        assert np.array_equal(np.sort(dt), np.sort(fdt[keep])), c


def test_the_cap_keeps_the_seeds_and_the_veto_looks_at_every_s1():
    words = _dense_frame(n_s1=20000)
    fr = _prepared(words)
    full = W.analyse_frame(fr)
    cap = W.analyse_frame(fr, cuts=W.Cuts(max_s1=500))
    assert np.array_equal(cap.t_s1[cap.seeds], full.t_s1[full.seeds]), "same seeds shown"
    assert np.array_equal(cap.rf_phase[cap.seeds], full.rf_phase[full.seeds], equal_nan=True)
    rows = np.searchsorted(full.t_s1, cap.t_s1)
    assert np.array_equal(cap.rf_vetoed, full.rf_vetoed[rows]), \
        "a sampled S1 is vetoed by an unsampled neighbour inside its gate"


def test_the_plugin_fills_the_s1_analyses_from_the_sample():
    words = _dense_frame(n_s1=6000)

    class _E:
        def __init__(self, w, serial):
            self.header = type("H", (), {"event_id": 301, "serial_number": serial})()
            self.banks = {"H000": type("B", (), {"data": np.asarray(w, "<u8").view("<u4")})()}

        def get_bank(self, name):
            return self.banks.get(name)

    p = P.SmaPlugin(HistStore())
    assert p.cfg.cuts.max_s1 == 2000
    assert p.process(_E(words, 1), run_number=1)
    assert p._last.cls == "good"
    h = p.store.get
    assert h("sma/pattern").entries == 2000
    assert h("sma/s1_coinc").entries == 2000
    assert h("sma/s1_spacing_us").entries == 5999, "the spacing uses every S1 hit"
    assert h("sma/words_per_ch").counts[2] == 6000, "per-channel counts use every hit"
    s = p.summary()
    assert s["sampling"]["s1_analysed_frac"] == pytest.approx(2000 / 6000, rel=1e-3)
    eff = {e["counter"]: e["eff"] for e in s["efficiency"]}
    assert eff["S2"] == pytest.approx(0.7, abs=0.04)
    assert eff["S4"] == pytest.approx(0.5, abs=0.04)

    q = P.SmaPlugin(HistStore(), settings={"Cuts": {"max S1 per frame": 0}})
    assert q.cfg.cuts.max_s1 is None, "0 = no cap"
    q.process(_E(words, 1), run_number=1)
    assert q.store.get("sma/pattern").entries == 6000


def test_the_integer_fills_match_the_float_path():
    """Optimisations A and C: bit-identical counts."""
    words = _dense_frame(n_s1=3000)
    fr = _prepared(words)
    p = P.SmaPlugin(HistStore())
    p.process(type("E", (), {
        "header": type("H", (), {"event_id": 301, "serial_number": 0})(),
        "get_bank": lambda self, n: type("B", (), {"data": np.asarray(words, "<u8").view("<u4")})(),
    })(), run_number=1)
    hd = p.store.get("sma/fine_coarse_diff")
    ref = P.Hist2D("ref", hd.x, hd.y, dtype=np.uint64)
    o = fr.order
    P._fill_2d(ref, fr.s_ch.astype(np.float64), (fr.diff_ns[o] >> fr.shift).astype(np.float64))
    assert np.array_equal(ref.counts, hd.counts) and ref.entries == hd.entries

    hp = p.store.get("sma/s1_partner_hits")
    an = W.analyse_frame(fr, p.cfg.roles, p.cfg.cuts).sampled()
    ref = P.Hist2D("ref2", hp.x, hp.y, dtype=np.uint64)
    k = np.broadcast_to(np.arange(an.partner_counts.shape[1], dtype=np.float64),
                        an.partner_counts.shape)
    P._fill_2d(ref, k.ravel(), an.partner_counts.ravel().astype(np.float64))
    assert np.array_equal(ref.counts, hp.counts) and ref.entries == hp.entries


# --- rates under sampling ----------------------------------------------------------------------

def test_the_rate_of_a_lone_frame_ignores_faulty_words_outside_it():
    """A sampled frame has no neighbour to take the overlap out of its span.

    Real 1008 frames start ~1 ms early on a few faulty S5/RF words (a fine
    fault at coarse - fine = -32 ticks decodes one 2^20 ns wrap early). The
    rate must divide by the extent of the consistent hits, or a sampled frame
    reads a few % low -- 25 % at ten times the density.
    """
    from pathlib import Path
    with np.load(Path(__file__).parent / "data" / "sma_run01008_frame.npz") as z:
        words = z["f0_words"]
    fr = W.prepare_frame(words, 14)
    assert fr.timed_span_ns < fr.span_ns, "faulty words widen the kept extent"
    t = fr.timed_extent
    cons = fr.consistent[fr.order]
    assert t.first == fr.s_t[cons][0] and t.last == fr.s_t[cons][-1]

    class _E:
        def __init__(self, serial):
            self.header = type("H", (), {"event_id": 301, "serial_number": serial})()

        def get_bank(self, name):
            return type("B", (), {"data": np.asarray(words, "<u8").view("<u4")})()

    # The rate of a frame is taken when its predecessor was analysed; with a
    # lone frame there is none, so give the plugin the frame's own
    # predecessor extent by hand (what a real preceding frame would leave).
    p = P.SmaPlugin(HistStore())
    p._last_serial = 4
    p._prev_extent = W.Extent(fr.first - 10**7, fr.first - 10**6)
    p._prev_timed = W.Extent(fr.first - 10**7, fr.first - 10**6)
    p.process(_E(5), run_number=None)
    s = p.summary()
    n1 = int(fr.chan[1].size)
    assert s["channels"][1]["rate_hz"] == pytest.approx(n1 / (fr.timed_span_ns * 1e-9), rel=1e-4)


def test_a_slot_waiting_for_data_polls_fast_then_backs_off(monkeypatch):
    """A stopped run must not keep the loop turning every 5 ms."""
    tm = _Time()
    a = _budget_analyzer(tm)
    buf = _Buffer()
    a.run_once(buf, buf=0)                 # opens a slot, nothing to read
    assert a._slot_left > 0
    assert a.next_wait_ms(200) == A.Analyzer.SLOT_POLL_MS
    now = A.time.monotonic()
    monkeypatch.setattr(A.time, "monotonic", lambda: now + A.Analyzer.SLOT_POLL_S + 0.1)
    assert a.next_wait_ms(200) == 200


def test_a_truncated_read_grows_the_read_buffer_once():
    """MIDAS posts an error per truncated read: grow the buffer, do not flood."""
    import ctypes
    import struct

    import midas

    class _Real(_Buffer):
        def __init__(self):
            super().__init__()
            self.event_buffers = {0: ctypes.create_string_buffer(1000)}

        def receive_event(self, buf, async_flag=True, use_numpy=False):
            if not self.pending:
                return None
            ev = self.pending[0]
            size = ev.header.event_data_size_bytes
            b = self.event_buffers[buf]
            if size + 16 > ctypes.sizeof(b):
                self.pending.pop(0)
                b[0:16] = struct.pack("<HHIII", 301, 0, ev.header.serial_number, 0, size)
                raise midas.MidasError(midas.status_codes["BM_TRUNCATED"], "truncated")
            self.copied += 1
            return self.pending.pop(0)

    tm = _Time()
    a = _unbounded(tm)
    buf = _Real()
    buf.produce(3, size=5000)
    a.run_once(buf, buf=0)
    assert a.truncated == 1, "only the first one; the buffer then fits"
    assert ctypes.sizeof(buf.event_buffers[0]) >= 5016
    assert a.plugin.serials == [1, 2]
    buf.produce(1, size=A.MAX_EVENT_CAP * 2)
    a.run_once(buf, buf=0)
    assert a.truncated == 2, "beyond the cap it stays truncated, and counted"


# --- review fixes -----------------------------------------------------------------------------

def test_keeping_up_never_skips_whatever_else_is_in_the_buffer():
    """Skipping is decided by offered (serials) vs allowed rate, not the buffer level."""
    tm = _Time()
    a = _budget_analyzer(tm)
    buf = _Buffer()
    buf.buffer_level = lambda b: 10**9            # SYSTEM full of other detectors' events
    _drive(a, buf, tm, offered=10.0, seconds=20.0)
    assert buf.skips <= 1, "at most the first slot, before any rate is known"
    assert a.plugin.serials == list(range(len(a.plugin.serials)))


def test_a_plugin_error_drops_the_event_and_is_reported_once():
    class _Boom(_BudgetPlugin):
        def process(self, event, run_number=None):
            if event.header.serial_number % 2:
                raise MemoryError("frame too big")
            return super().process(event, run_number)

    class _Msgs(_Buffer):
        def __init__(self):
            super().__init__()
            self.msgs = []

        def msg(self, text, is_error=False):
            self.msgs.append(text)

    tm = _Time()
    a = A.Analyzer(lambda s: _Boom(s), rate=1000.0, clock=tm.clock, cpu=tm.cpu,
                   no_cpu_budget=True)
    buf = _Msgs()
    buf.produce(10)
    a.run_once(buf, buf=0)                         # must not raise
    assert a.plugin_errors == {"MemoryError": 5}
    assert a.plugin.serials == [0, 2, 4, 6, 8]
    assert len(buf.msgs) == 1 and "MemoryError" in buf.msgs[0]
    assert a.status()["plugin_errors"] == {"MemoryError": 5}


def test_the_wavedream_path_also_survives_a_plugin_error():
    import test_analyzer as T

    class _Boom(T._FakePlugin):
        def process(self, event, run_number=None):
            raise ValueError("bad bank")

    a = A.Analyzer(lambda s: _Boom(s), rate=1000.0)
    c = T._FakeClient()
    c.events = [T._Event() for _ in range(3)]
    a.run_once(c, buf=None)
    assert a.plugin_errors == {"ValueError": 3} and a.processed == 0


def test_the_drain_fallback_is_bounded_and_survives_truncation():
    import midas

    class _NoSkip(_Buffer):
        skip_event = None                          # an old MIDAS: no bm_skip_event

        def __init__(self):
            super().__init__()
            self.n = 0

        def receive_event(self, buf, async_flag=True, use_numpy=False):
            self.n += 1
            if self.n == 3:
                self.pending.pop(0)
                raise midas.MidasError(midas.status_codes["BM_TRUNCATED"], "t")
            return super().receive_event(buf, async_flag, use_numpy)

    tm = _Time()
    a = _budget_analyzer(tm)
    buf = _NoSkip()
    buf.produce(1000)
    ops = A.BufferOps(buf, 0)
    assert ops.method == "drain"
    n = ops.skip(a._receive)
    assert n == A.BufferOps.DRAIN_MAX, "bounded"
    assert a.truncated == 1, "a truncated event in the drain is counted, not raised"


def test_oversize_frames_are_counted_not_decoded():
    words = _dense_frame(n_s1=200)

    class _E:
        def __init__(self, w, serial):
            self.header = type("H", (), {"event_id": 301, "serial_number": serial})()
            self._d = np.asarray(w, "<u8").view("<u4")

        def get_bank(self, name):
            return type("B", (), {"data": self._d})()

    p = P.SmaPlugin(HistStore(), settings={"Cuts": {"max words per frame": 500}})
    assert p.cfg.max_words == 500 and words.size > 500
    assert p.process(_E(words, 1), run_number=1) is False
    assert p.frames_oversize == 1 and p.frames == 0
    assert p.store.get("sma/words_per_frame").entries == 0, "nothing decoded"
    s = p.summary()
    assert s["frames"]["oversize"] == 1 and s["frames"]["window"]["oversize"] == 1
    assert s["frames"]["offered"] == 1
    assert "oversize" in [f["code"] for f in s["flags"]]
    assert p.status()["frames_oversize"] == 1
    assert P.SmaPlugin(HistStore()).cfg.max_words == 1 << 20, "default 1 Mi words"


def test_a_reconnect_does_not_count_the_outage_as_offered_frames():
    tm = _Time()
    a = _budget_analyzer(tm)
    buf = _Buffer()
    _drive(a, buf, tm, offered=10.0, seconds=5.0)
    before = a.offered
    buf.serial += 100000                            # an outage
    a.reset_serials()
    _drive(a, buf, tm, offered=10.0, seconds=1.0)
    assert a.offered - before < 20

    p = P.SmaPlugin(HistStore())
    p._last_serial = 5
    p.reset_serial_baseline()
    assert p._last_serial is None
