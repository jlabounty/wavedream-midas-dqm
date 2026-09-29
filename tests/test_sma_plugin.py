"""The SMA plugin: classification, fills, shift check, summary/trend, smaf, commands.

Real frames come from ``tests/data`` (raw H000 words of real readout frames,
see ``generate_sma_golden.py``); synthetic frames are built with
``sma_words.encode`` so the truth is known. No MIDAS needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mdqm.dqm import analyzer as A
from mdqm.dqm import framing
from mdqm.dqm import settings as S
from mdqm.dqm.hist import HistStore
from mdqm.dqm.server import Server
from mdqm.plugins import sma as P
from mdqm.plugins import sma_words as W

DATA = Path(__file__).resolve().parent / "data"


# --- fakes ---------------------------------------------------------------------

class _Bank:
    def __init__(self, data):
        self.data = data


class _Event:
    """What ``receive_event(use_numpy=True)`` hands over: H000 as a uint32 array."""

    def __init__(self, words=None, serial=0, event_id=301, bank="H000"):
        self.header = type("H", (), {"event_id": event_id, "serial_number": serial})()
        self.banks = {}
        if words is not None:
            self.banks[bank] = _Bank(np.asarray(words, dtype="<u8").view("<u4"))

    def get_bank(self, name):
        return self.banks.get(name)


class _Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def _load(name):
    with np.load(DATA / name) as z:
        return [z[f"f{i}_words"] for i in range(len(z["labels"]))]


R682 = _load("sma_run00682_frames.npz")      # 682_00000 #0-3, 682_00005 #0-1
R1008 = _load("sma_run01008_frame.npz")[0]   # first 20000 words of a 40000-word frame
R342 = _load("sma_run00342_frames.npz")      # coarse shift 3; 8000 words each
EXPECTED_CLASS = ["stale", "stale", "good", "good", "good", "good"]


def _plugin(settings=None, clock=None):
    return P.SmaPlugin(HistStore(), settings=settings, clock=clock or _Clock())


def _feed(p, words, serial=0, run=682):
    assert p.process(_Event(words, serial=serial), run_number=run) is True
    return p._last


def _h(p, name):
    return p.store.get(f"sma/{name}")


def synth_frame(t0_ns, n=300, spacing_ns=3000, shift=14, s2=True, seed=1, far=None):
    """Words of a healthy frame whose true times are known.

    S1 on ch 1 every ~spacing, S2-S5 a few ns after it (S2 optional), two RF
    pulses 40 and 60 ns after it on ch 6, and ch 7 in between.
    """
    rng = np.random.default_rng(seed)
    t1 = t0_ns + np.arange(n, dtype=np.int64) * spacing_ns + rng.integers(0, 500, n)
    ts, chs = [t1], [np.full(n, 1)]
    for c in (2, 3, 4, 5):
        if c == 2 and not s2:
            continue
        ts.append(t1 + c)
        chs.append(np.full(n, c))
    for d in (40, 60):
        ts.append(t1 + d)
        chs.append(np.full(n, 6))
    ts.append(t1 + 1500)
    chs.append(np.full(n, 7))
    if far is not None:
        # A second, genuine cluster `far` ns later: 30 S1 hits (a beam trip).
        ts.append(t1[-1] + far + np.arange(30, dtype=np.int64) * 1000)
        chs.append(np.full(30, 1))
    t = np.concatenate(ts)
    ch = np.concatenate(chs)
    o = np.argsort(t, kind="stable")
    t, ch = t[o], ch[o]
    coarse, fine = W.fields_of(t, shift)
    return W.encode(ch, np.full(t.size, 20), coarse, fine)


# --- events --------------------------------------------------------------------

def test_process_rejects_other_events_and_a_missing_bank():
    p = _plugin()
    assert p.process(_Event(R682[4], event_id=401)) is False
    assert p.process(_Event(None)) is False
    assert p.process(_Event(R682[4], bank="H001")) is False
    assert p.frames == 0
    assert p.frames_rejected == 2
    assert not p.accepts(_Event(None, event_id=1))


def test_the_tuple_bank_path_is_accepted():
    p = _plugin()
    ev = _Event(None)
    ev.banks["H000"] = _Bank(tuple(int(x) for x in R682[4].view("<u4")))
    assert p.process(ev, run_number=1) is True
    assert p._last.cls == "good"


# --- the stale-frame rule --------------------------------------------------------

def test_real_frames_are_classified():
    p = _plugin()
    got = [_feed(p, w, serial=i).cls for i, w in enumerate(R682)]
    assert got == EXPECTED_CLASS
    assert _feed(p, R1008, run=1008).cls == "good"
    assert p.frames_stale == 2


def test_the_two_stale_real_frames_fail_for_their_own_reason():
    p = _plugin()
    first = _feed(p, R682[0])
    assert first.cls == "stale" and "S1 fine/coarse agree" in first.reason
    second = _feed(p, R682[1], serial=1)
    assert second.cls == "stale" and "no role" in second.reason


@pytest.mark.parametrize("shift", [12, 13, 15, 16])
def test_the_stale_rule_does_not_depend_on_the_configured_shift(shift):
    """A wrong shift must not make frames stale (old data, hidden) -- it makes the
    genuine ones suspect (wrong time base, flagged), and the stale ones stay stale."""
    p = _plugin({"Coarse shift": shift})
    got = [_feed(p, w, serial=i) for i, w in enumerate(R682)]
    assert [g.cls for g in got] == [c if c == "stale" else "suspect" for c in EXPECTED_CLASS]
    assert "fits coarse shift 14" in got[2].reason
    assert _feed(p, R1008, run=1008).cls == "suspect"


def test_shift_13_data_at_configured_14_is_suspect_not_stale():
    p = _plugin()
    snap = _feed(p, synth_frame(10**12, shift=13))
    assert snap.cls == "suspect" and "fits coarse shift 13" in snap.reason
    assert _plugin({"Coarse shift": 13}).process(
        _Event(synth_frame(10**12, shift=13)), run_number=1)


def test_an_empty_frame_is_its_own_class():
    p = _plugin()
    words = np.full(100, W.FILLER, dtype=np.uint64)
    assert _feed(p, words).cls == "empty"
    assert p.frames_empty == 1
    assert _h(p, "frame_class").counts[3] == 1


def test_a_shift_3_run_at_shift_3_is_good_and_checks_ok():
    p = _plugin({"Coarse shift": 3})
    assert p.cfg.scan == (3, 12, 13, 14, 15, 16)
    assert [_feed(p, w, serial=i, run=342).cls for i, w in enumerate(R342)] == ["good", "good"]
    v = p.shift_verdict()
    assert v["words"] >= 1000
    assert (v["verdict"], v["best"]) == ("ok", 3)
    assert p.summary()["channels"][1]["mismatch_frac"] < 0.01, "the 20 us latch floor"


def test_a_far_wrong_shift_is_suspect_and_names_the_right_one():
    """14 on shift-3 data: the times scatter, the shift check still sees the fields."""
    p = _plugin()
    got = [_feed(p, w, serial=i, run=342) for i, w in enumerate(R342)]
    assert [g.cls for g in got] == ["suspect", "suspect"]
    assert "fits coarse shift 3" in got[0].reason
    assert p.frames_suspect == 2
    assert int(_h(p, "frame_class").counts[4]) == 2
    assert int(_h(p, "word_types").counts[6]) == sum(g.fr.n_trigger for g in got)
    for name in ("words_per_ch", "pattern", "rf_phase_s1", "fine_vs_coarse"):
        assert _h(p, name).entries == 0, name
    v = p.shift_verdict()
    assert (v["verdict"], v["best"], v["configured"]) == ("mismatch", 3, 14)
    s = p.summary(run_active=True)
    codes = [f["code"] for f in s["flags"]]
    assert "shift_mismatch" in codes
    assert "mismatch" not in codes, "no per-channel fine/coarse flags on a wrong time base"
    assert s["frames"]["suspect"] == 2 and s["frames"]["window"]["suspect"] == 2
    r = framing.decode_sma_frame(p.frame_blob("raster"))
    assert r["suspect"] and r["meta"]["class"] == "suspect"


def test_suspect_frames_alone_raise_the_time_base_flag():
    p = _plugin()
    p.cfg.check["shift min words"] = 10**9     # keep the verdict at "insufficient"
    _feed(p, R342[0], run=342)
    flags = {f["code"]: f for f in p.summary()["flags"]}
    assert flags["time_base"]["severity"] == "error"
    assert "shift_mismatch" not in flags


def test_stale_frames_fill_health_but_not_physics():
    p = _plugin()
    _feed(p, R682[0])
    _feed(p, R682[1], serial=1)
    n_words = sum(w.size for w in R682[:2])
    assert _h(p, "word_types").entries == n_words
    assert int(_h(p, "word_types").counts.sum()) == n_words
    trig = sum(W.decode(w)["n_trigger"] for w in R682[:2])
    assert int(_h(p, "word_types").counts[5]) == trig, "bin 4: words of stale frames"
    assert int(_h(p, "stale_per_ch").counts.sum()) == trig
    assert list(_h(p, "frame_class").counts[1:5]) == [0, 2, 0, 0]
    for name in ("words_per_ch", "tot_vs_ch_lsb0", "tot_vs_ch_lsb1", "pattern", "shift_check",
                 "fine_vs_coarse", "frame_span_log10_ms", "frame_gap_ms", "rf_phase_s1",
                 "dt_S2_S1", "rate_vs_ch"):
        assert _h(p, name).entries == 0, name
    assert p._last_good is None
    assert p._last.cls == "stale"


# --- fills -------------------------------------------------------------------------

def test_a_good_frame_fills_what_the_pipeline_computes():
    p = _plugin()
    _feed(p, R1008, run=1008)
    cfg = p.cfg
    fr = W.prepare_frame(R1008, cfg.shift, cfg.cuts.stale_gap_ns, cfg.cuts.latch_margin_ns)
    an = W.analyse_frame(fr, cfg.roles, cfg.cuts)
    kept = int(fr.keep.sum())
    per_ch = np.bincount(fr.s_ch, minlength=16)

    assert int(_h(p, "word_types").counts.sum()) == R1008.size
    assert list(_h(p, "words_per_ch").counts[1:17]) == list(per_ch)
    lsb = [_h(p, f"tot_vs_ch_lsb{k}") for k in (0, 1)]
    assert sum(h.entries for h in lsb) == kept
    assert sum(int(h.counts.sum()) for h in lsb) == kept
    fine_kept = fr.fine[fr.order]
    for ch in (1, 2, 3, 4, 5, 6):
        for k in (0, 1):
            sel = (fr.s_ch == ch) & ((fine_kept & 1) == k)
            h = _h(p, f"tot_ch{ch:02d}_lsb{k}")
            assert int(h.counts.sum()) == int(sel.sum())
            assert np.array_equal(h.counts[1:257], np.bincount(fr.s_tot[sel], minlength=256))
    fvc = _h(p, "fine_vs_coarse").counts
    cons = fr.consistent[fr.order]
    assert list(fvc[1, 1:17]) == list(np.bincount(fr.s_ch[cons], minlength=16))
    assert list(fvc[2, 1:17]) == list(np.bincount(fr.s_ch[~cons], minlength=16))
    assert int(_h(p, "fine_coarse_diff").counts.sum()) == kept

    n1 = an.t_s1.size
    assert _h(p, "pattern").entries == n1
    assert np.array_equal(_h(p, "pattern").counts[1:33], np.bincount(an.pattern, minlength=32))
    hs = _h(p, "shift_check")
    assert hs.entries == int((fr.ch == 1).sum()), "every S1 word, kept or not"
    assert [int(hs.counts[s - cfg.scan[0] + 1]) for s in cfg.scan] == list(an.shift_counts)
    assert _h(p, "rf_npulses").entries == n1
    assert _h(p, "rf_phase_s1").entries == int(an.rf_valid.sum())
    for ch, (_i, dt) in an.dt.items():
        k = list(cfg.roles.counters).index(ch)
        assert _h(p, f"dt_S{k + 1}_S1").entries == dt.size
    for ch, (_i, dt) in an.delayed_dt.items():
        assert _h(p, f"delayed_dt_ch{ch:02d}").entries == dt.size
    assert _h(p, "s1_spacing_us").entries == n1 - 1
    assert list(_h(p, "s1_coinc").counts[1:6]) == list((an.partner_counts > 0).sum(axis=0))
    tb = _h(p, "tot_ge250_per_ch").counts[1:17]
    assert list(tb) == list(np.bincount(fr.s_ch[fr.s_tot >= 250], minlength=16))
    occ = _h(p, "fine_bit_occupancy").counts
    f1 = fine_kept[fr.s_ch == 1]
    for b in (0, 7, 19):
        assert int(occ[1 + b, 2]) == int(((f1 >> np.uint32(b)) & 1).sum()), b
    assert _h(p, "frame_span_log10_ms").entries == 1
    assert _h(p, "frame_gap_ms").entries == 0, "no previous frame, no gap"


def test_a_far_genuine_cluster_is_kept_and_stretches_the_frame():
    p = _plugin()
    snap = _feed(p, synth_frame(10**12, far=3 * 10**9))
    assert snap.cls == "good"
    assert snap.fr.n_rescued == 30
    assert snap.fr.span_ns > 3 * 10**9
    assert int(_h(p, "words_per_ch").counts[2]) == 330


def test_the_overlap_of_consecutive_frames_is_counted_once_in_the_rates():
    p = _plugin()
    a = _feed(p, synth_frame(10**12, seed=1)).fr
    b = _feed(p, synth_frame(a.last - 500_000, seed=2), serial=1).fr
    overlap = a.last - b.first
    assert 0 < overlap < 1_000_000
    s = p.summary()
    assert s["span_s"] == pytest.approx((a.span_ns + b.span_ns) * 1e-9, rel=1e-4)
    # The rates use the frames that follow an analysed one (here b), and the
    # time b shares with a is counted once (with a).
    assert s["covered_s"] == pytest.approx((b.span_ns - overlap) * 1e-9, rel=1e-4)
    b_s1 = int(b.chan[1].size)
    assert s["channels"][1]["rate_hz"] == pytest.approx(b_s1 / s["covered_s"], rel=1e-4)


def test_every_histogram_counts_in_uint64_and_encodes_as_f64():
    p = _plugin()
    _feed(p, R1008, run=1008)
    names = [n for n in p.store.names() if n.startswith("sma/")]
    assert len(names) >= 40
    for n in names:
        h = p.store.get(n)
        assert h.counts.dtype == np.uint64, n
        d = framing.decode_histogram(h.encode())
        assert d["counts"].dtype == np.float64, n
        assert np.array_equal(d["counts"], h.counts.astype(np.float64)), n


def test_consecutive_frames_fill_gap_and_live_and_a_long_gap_resets():
    p = _plugin()
    _feed(p, R682[2], serial=2)
    _feed(p, R682[3], serial=3)
    assert _h(p, "frame_gap_ms").entries == 1
    assert _h(p, "live_fraction").entries == 1
    # 682_00005 starts ~48 s later: a subrun boundary, not a gap. (Serial 4
    # here: the time jump, not a serial jump, has to reset the chain.)
    _feed(p, R682[4], serial=4)
    assert _h(p, "frame_gap_ms").entries == 1
    assert p.gap_resets == 1
    _feed(p, R682[5], serial=5)
    assert _h(p, "frame_gap_ms").entries == 2


def test_only_the_next_serial_continues_the_gap_chain():
    """Sampling: a frame and one sent three later have no gap or live fraction."""
    p = _plugin()
    _feed(p, R682[4], serial=10)
    _feed(p, R682[5], serial=13)                 # 11 and 12 were skipped
    assert _h(p, "frame_gap_ms").entries == 0, "not the frame before it"
    assert _h(p, "live_fraction").entries == 0
    assert p.serial_breaks == 1 and p.gap_resets == 0
    s = p.summary()
    assert s["live_fraction"] is None, "no consecutive pair: null, never a wrong number"
    assert s["frames"]["offered"] == 4 and s["frames"]["processed"] == 2
    assert s["frames"]["window"]["analysed_frac"] == 0.5
    # Rates need a frame and its predecessor too (the first frame after a skip
    # is length-biased): none here, so null rather than a biased number.
    assert s["covered_s"] == 0 and s["channels"][1]["rate_hz"] is None
    _feed(p, R682[4], serial=20)                 # a sampled pair: 20 and 21
    _feed(p, R682[5], serial=21)
    s = p.summary()
    assert s["covered_s"] > 0 and s["channels"][1]["rate_hz"] > 0


def test_a_frame_starting_far_back_resets_the_chain_like_a_long_gap():
    """A replay loop jumps back seconds; real frames overlap by ~1 ms at most."""
    p = _plugin()
    _feed(p, R682[4], serial=1)
    _feed(p, R682[5], serial=2)                  # overlap of ~2 us: a real gap
    assert _h(p, "frame_gap_ms").entries == 1
    _feed(p, R682[4], serial=3)                  # starts ~13 ms before the last one ended
    assert _h(p, "frame_gap_ms").entries == 1
    assert _h(p, "live_fraction").entries == 1
    assert p.gap_resets == 1
    _feed(p, R682[5], serial=4)                  # and the chain goes on from there
    assert _h(p, "frame_gap_ms").entries == 2
    p = _plugin({"Cuts": {"max overlap ms": 100.0}})
    _feed(p, R682[4], serial=1)
    _feed(p, R682[5], serial=2)
    _feed(p, R682[4], serial=3)
    assert _h(p, "frame_gap_ms").entries == 2, "the limit is a setting"


def test_missed_frames_are_counted_from_the_serial_number():
    p = _plugin()
    for s in (10, 11, 15):
        _feed(p, R682[4], serial=s)
    assert p.missed_by_serial == 3
    _feed(p, R682[4], serial=0)          # a replay loop: backwards is not a loss
    assert p.missed_by_serial == 3
    assert p.summary()["frames"]["seen_by_serial"] == 7
    assert p.offered_by_serial == 7, "the first frame counts once, a loop once"


def test_a_new_run_restarts_the_gap_chain_and_the_summary_window():
    p = _plugin()
    _feed(p, R682[4], serial=1, run=1)
    assert p.summary()["frames"]["window"]["frames"] == 1
    _feed(p, R682[5], serial=2, run=2)
    assert _h(p, "frame_gap_ms").entries == 0
    assert p.summary()["frames"]["window"]["frames"] == 1


# --- the shift self-check ----------------------------------------------------------------

def test_shift_13_data_at_configured_14_flags_13():
    clock = _Clock()
    p = _plugin(clock=clock)
    for k in range(5):
        _feed(p, synth_frame(10**12 + k * 10**6, shift=13, seed=k), serial=k)
        clock.t += 0.1
    v = p.shift_verdict()
    assert v["words"] == 1500
    assert v["verdict"] == "mismatch"
    assert v["best"] == 13
    flags = p.summary()["flags"]
    shift_flags = [f for f in flags if f["code"] == "shift_mismatch"]
    assert shift_flags and shift_flags[0]["severity"] == "error"
    assert "Coarse shift = 13" in shift_flags[0]["text"]


def test_real_frames_at_configured_13_flag_14():
    p = _plugin({"Coarse shift": 13})
    _feed(p, R1008, run=1008)
    v = p.shift_verdict()
    assert v["words"] >= 1000
    assert (v["verdict"], v["best"], v["configured"]) == ("mismatch", 14, 13)


def test_real_frames_at_the_right_shift_pass():
    p = _plugin()
    _feed(p, R1008, run=1008)
    v = p.shift_verdict()
    assert (v["verdict"], v["best"]) == ("ok", 14)
    assert not [f for f in p.summary()["flags"] if f["code"].startswith("shift")]


def test_per_channel_fine_coarse_flags_wait_for_the_shift_check():
    p = _plugin()
    _feed(p, R682[4])                       # ~100 S1 words: no verdict yet
    codes = [f["code"] for f in p.summary()["flags"]]
    assert "mismatch" not in codes and "shift_unchecked" in codes
    _feed(p, R1008, run=1008)
    codes = [f["code"] for f in p.summary()["flags"]]
    assert "mismatch" in codes and "shift_unchecked" not in codes


def test_only_the_listed_channels_raise_fine_coarse_flags():
    p = _plugin()
    _feed(p, R1008, run=1008)
    s = p.summary()
    assert s["channels"][7]["mismatch_frac"] > 0.9 and not s["channels"][7]["flagged"]
    texts = [f["text"] for f in s["flags"] if f["code"] == "mismatch"]
    assert any("(ch 5)" in t for t in texts) and not any("(ch 7)" in t for t in texts)
    settings = P._merge(P.SETTINGS_DEFAULTS, {"Self check": {"mismatch flag channels": [7]}})
    assert P.shape_fingerprint(settings) == P.shape_fingerprint(P.SETTINGS_DEFAULTS), \
        "a flag setting must not reset the plots"
    p.apply_settings(settings, rebuild=False)
    texts = [f["text"] for f in p.summary()["flags"] if f["code"] == "mismatch"]
    assert [t for t in texts if "(ch 7)" in t] and not [t for t in texts if "(ch 5)" in t]


def test_too_few_words_is_no_verdict():
    p = _plugin()
    _feed(p, R682[4])
    assert p.shift_verdict()["verdict"] == "insufficient"


def test_stale_frames_stay_out_of_the_shift_check():
    p = _plugin()
    _feed(p, R682[0])
    assert p.shift_verdict()["words"] == 0


def test_the_ring_forgets_after_its_window():
    clock = _Clock()
    p = _plugin(clock=clock)
    _feed(p, R1008, run=1008)
    assert p.shift_verdict()["words"] > 0
    clock.t += 31
    assert p.shift_verdict()["words"] == 0


def test_a_live_shift_edit_is_judged_at_once():
    """Rebuilding for the new shift must not throw away the evidence."""
    p = _plugin()
    _feed(p, R1008, run=1008)
    settings = P._merge(P.SETTINGS_DEFAULTS, {"Coarse shift": 13})
    p.apply_settings(settings, rebuild=True)
    v = p.shift_verdict()
    assert (v["verdict"], v["best"]) == ("mismatch", 14)


# --- settings --------------------------------------------------------------------------------

def _analyzer_with_odb(tree=None):
    from test_analyzer import _SettingsClient

    a = A.Analyzer(A.make_plugin_factory("sma"))
    c = _SettingsClient()
    S.seed(c, a.settings_root, a.settings_defaults)
    c.tree.update(tree or {})
    a.apply_settings(c, force=True)
    return a, c


def test_the_plugin_is_built_by_the_analyzer_with_its_own_tree():
    a, c = _analyzer_with_odb()
    assert isinstance(a.plugin, P.SmaPlugin)
    assert a.settings_root == "/DQM/SMA"
    assert c.tree["/DQM/SMA/Coarse shift"] == 14
    assert c.tree["/DQM/SMA/Channel roles/counters"] == [1, 2, 3, 4, 5]
    assert c.tree["/DQM/SMA/Sampling/CPU budget %"] == 20.0
    assert "/DQM/SMA/Sampling/process all" not in c.tree, "the budget replaced it"
    assert c.tree["/DQM/SMA/Cuts/max S1 per frame"] == 2000
    assert not [p for p in c.tree if p.startswith("/DQM/Analyzer")]
    assert a.process_all is False
    assert a.sampling_mode() == "cpu budget" and a.budget.budget_pct == 20.0
    assert a.dropped_path == ""
    assert sorted(a.plugin.event_ids) == [301]


def test_a_label_change_does_not_rebuild():
    a, c = _analyzer_with_odb()
    a.plugin.process(_Event(R1008, serial=1), run_number=1008)
    before = a.store.get("sma/words_per_ch").entries
    rebuilds = a.plugin.rebuilds

    labels = [""] * 16
    labels[5] = "S5 (t/2!)"
    c.tree["/DQM/SMA/Channel roles/labels"] = labels
    a._settings_checked = 0
    assert a.apply_settings(c) is True
    assert a.plugin.rebuilds == rebuilds
    assert a.store.get("sma/words_per_ch").entries == before
    assert a.plugin.summary()["channels"][5]["label"] == "S5 (t/2!)"


@pytest.mark.parametrize("path,value", [
    ("/DQM/SMA/Coarse shift", 13),
    ("/DQM/SMA/Cuts/coinc window ns", 30),
    ("/DQM/SMA/Binning/gap bins", 50),
    ("/DQM/SMA/Channel roles/delayed", [8, 9]),
])
def test_shift_cut_binning_and_role_changes_rebuild(path, value):
    a, c = _analyzer_with_odb()
    a.plugin.process(_Event(R1008, serial=1), run_number=1008)
    assert a.store.get("sma/words_per_ch").entries > 0
    n = a.reconfigures
    c.tree[path] = value
    a._settings_checked = 0
    a.apply_settings(c)
    assert a.reconfigures == n + 1
    assert a.store.get("sma/words_per_ch").entries == 0
    assert any("rebuilt" in m for m in c.messages)


def test_a_role_change_renames_the_histograms():
    a, c = _analyzer_with_odb({"/DQM/SMA/Channel roles/delayed": [8, 11]})
    assert "sma/delayed_dt_ch11" in a.store
    assert "sma/delayed_dt_ch10" not in a.store


def test_the_shift_change_rebinds_the_diff_axis():
    p = _plugin({"Coarse shift": 16})
    assert _h(p, "fine_coarse_diff").y.n == 16
    p = _plugin({"Coarse shift": 12})
    assert _h(p, "fine_coarse_diff").y.n == 256


def test_garbage_in_the_odb_falls_back_and_is_reported():
    p = _plugin({"Coarse shift": 99, "Cuts": {"seeds": "many"},
                 "Channel roles": {"rf": 17, "counters": []}})
    assert p.cfg.shift == 14
    assert p.cfg.cuts.n_seeds == W.N_SEEDS
    assert p.cfg.roles.rf == 6
    assert p.cfg.roles.counters == (1, 2, 3, 4, 5)
    assert len(p.cfg.errors) == 4
    assert p.status()["settings_errors"] == p.cfg.errors
    assert any(f["code"] == "settings" for f in p.summary()["flags"])


def test_a_single_element_array_from_midas_is_still_a_list():
    p = _plugin({"Channel roles": {"delayed": 9, "labels": "only"}})
    assert p.cfg.roles.delayed == (9,)
    assert p.labels()[0] == "only"


# --- summary and trend ------------------------------------------------------------------------

def test_summary_shape_and_values():
    p = _plugin()
    _feed(p, R1008, run=1008)
    s = json.loads(json.dumps(p.summary(True), allow_nan=False))
    assert set(s) >= {"frames", "channels", "efficiency", "rf", "shift", "flags",
                      "live_fraction", "window_s", "run"}
    assert len(s["channels"]) == 16
    c1 = s["channels"][1]
    assert c1["label"] == "S1" and c1["role"] == "s1"
    assert c1["hits"] == c1["hits_per_frame"] == int((W.prepare_frame(R1008).s_ch == 1).sum())
    assert c1["rate_hz"] is None, "a lone frame: no predecessor, no rate"
    assert s["channels"][6]["label"] == "RF"
    assert s["channels"][7]["label"] == "current"
    assert s["efficiency_kind"] == "timed"
    assert [e["counter"] for e in s["efficiency"]] == ["S2", "S3", "S4", "S5"]
    e = {x["counter"]: x for x in s["efficiency"]}
    assert 0.5 < e["S2"]["eff"] <= 1 and e["S2"]["reason"] is None
    # S5's fine field is broken on this run: no number, a reason.
    assert e["S5"]["eff"] is None and e["S5"]["reason"].startswith("timestamp fault: ")
    assert s["covered_s"] == 0 and s["span_s"] > 0
    assert 0 < s["rf"]["valid_frac"] <= 1
    assert s["frames"]["processed"] == 1
    for f in s["flags"]:
        assert set(f) == {"severity", "code", "text"}
        assert f["severity"] in ("error", "warn", "info")


def test_the_s5_fine_fault_is_an_error_and_the_s3_tot_a_warning():
    p = _plugin()
    _feed(p, R1008, run=1008)
    flags = p.summary()["flags"]
    s5 = [f for f in flags if f["code"] == "mismatch" and "ch 5" in f["text"]]
    assert s5 and s5[0]["severity"] == "error"
    assert not [f for f in flags if f["code"] == "mismatch" and "ch 1)" in f["text"]]


def test_all_stale_frames_are_an_error_some_a_warning():
    p = _plugin()
    _feed(p, R682[0])
    assert [f["severity"] for f in p.summary()["flags"] if f["code"] == "all_stale"] == ["error"]
    _feed(p, R682[2], serial=2)
    flags = p.summary()["flags"]
    assert not [f for f in flags if f["code"] == "all_stale"]
    assert [f["severity"] for f in flags if f["code"] == "stale_frames"] == ["warn"]


def test_no_frames_while_a_run_is_active():
    clock = _Clock()
    p = _plugin(clock=clock)
    assert [f["code"] for f in p.summary(run_active=True)["flags"]] == ["no_frames"]
    _feed(p, R682[4])
    clock.t += 6
    flags = p.summary(run_active=True)["flags"]
    assert [f["severity"] for f in flags if f["code"] == "no_frames"] == ["error"]
    assert not [f for f in p.summary(run_active=False)["flags"] if f["code"] == "no_frames"]


def test_the_analyzer_run_state_wins_over_the_page():
    from test_analyzer import _FakeClient

    a = A.Analyzer(A.make_plugin_factory("sma"))
    a.poll_run_state(_FakeClient({"/Runinfo/State": 3, "/Runinfo/Run number": 5}))
    assert a.plugin.run_active is True
    flags = a.plugin.summary(run_active=False)["flags"]
    assert [f["code"] for f in flags] == ["no_frames"], "the polled state is authoritative"
    a.poll_run_state(_FakeClient({"/Runinfo/State": 1, "/Runinfo/Run number": 5}))
    assert a.plugin.run_active is False
    assert not a.plugin.summary(run_active=True)["flags"]


def test_the_first_settings_apply_posts_no_midas_message():
    a, c = _analyzer_with_odb()
    assert a.reconfigures == 1
    assert c.messages == [], "startup is not a change worth a MIDAS message"
    c.tree["/DQM/SMA/Coarse shift"] = 13
    a._settings_checked = 0
    a.apply_settings(c)
    assert len(c.messages) == 1 and "rebuilt" in c.messages[0]


def test_an_efficiency_drop_is_flagged():
    clock = _Clock()
    p = _plugin(clock=clock)
    t0 = 10**12
    for k in range(60):                 # a minute with S2 present
        _feed(p, synth_frame(t0 + k * 10**7, n=50, seed=k), serial=k)
        clock.t += 1
    assert not [f for f in p.summary()["flags"] if f["code"] == "efficiency_drop"]
    for k in range(60, 95):             # then S2 goes away
        _feed(p, synth_frame(t0 + k * 10**7, n=50, s2=False, seed=k), serial=k)
        clock.t += 1
    drop = [f for f in p.summary()["flags"] if f["code"] == "efficiency_drop"]
    assert len(drop) == 1 and "S2" in drop[0]["text"]


def test_trend_rows_are_seconds_including_empty_ones():
    clock = _Clock(5000.2)
    p = _plugin(clock=clock)
    _feed(p, R682[4], serial=1)
    _feed(p, R682[5], serial=2)
    clock.t += 3                        # two seconds with no frame, then one more
    _feed(p, R682[4], serial=3)
    clock.t += 1
    tr = p.trend()
    rows = tr["rows"]
    assert [r["t"] for r in rows] == [5000, 5001, 5002, 5003]
    assert [r["frames"] for r in rows] == [2, 0, 0, 1]
    r0 = rows[0]
    assert len(r0["rate_hz"]) == 16 and r0["rate_hz"][1] > 0
    assert r0["live"] is not None and 0 < r0["live"] <= 1
    assert len(r0["eff"]) == 4
    assert tr["counters"] == ["S2", "S3", "S4", "S5"]
    assert rows[1]["rate_hz"] is None
    assert [r["t"] for r in p.trend(since=5001)["rows"]] == [5002, 5003]
    json.dumps(tr, allow_nan=False)


def test_trend_withholds_the_efficiency_of_a_counter_with_a_timestamp_fault():
    """Run 682's S5 (ch 5) has ~93 % fine/coarse mismatch: the summary says n/a
    for it, and the trend must not draw it as a counter near 0 % efficient."""
    clock = _Clock(7000.2)
    p = _plugin(clock=clock)
    _feed(p, R682[4], serial=1)
    _feed(p, R682[5], serial=2)
    clock.t += 1
    s = p.summary()
    eff = {e["counter"]: e for e in s["efficiency"]}
    assert eff["S5"]["eff"] is None and "timestamp fault" in eff["S5"]["reason"]
    tr = p.trend()
    row = [r for r in tr["rows"] if r["eff"]][0]
    k5 = tr["counters"].index("S5")
    assert row["eff"][k5] is None
    k2 = tr["counters"].index("S2")
    assert row["eff"][k2] is not None and row["eff"][k2] > 0.3


def test_merged_trend_rows_keep_rates_and_mismatch():
    """Two buckets in one second (an epoch change inside it) merge into one row
    that still has its rates (cover_ns) and its fault masking (mismatch)."""
    clock = _Clock(8000.1)
    p = _plugin(clock=clock)
    _feed(p, R682[4], serial=1)
    p._new_epoch()
    _feed(p, R682[5], serial=2)
    clock.t += 1
    rows = p.trend()["rows"]
    row = [r for r in rows if r["t"] == 8000][0]
    assert row["frames"] == 2
    assert row["rate_hz"] is not None and row["rate_hz"][1] > 0
    assert row["eff"][3] is None                    # S5, as above


def test_trend_is_bounded_to_ten_minutes():
    clock = _Clock(100.0)
    p = _plugin(clock=clock)
    _feed(p, R682[4])
    clock.t += 2000
    _feed(p, R682[4], serial=1)
    clock.t += 1
    rows = p.trend()["rows"]
    assert len(rows) <= P.TREND_S + 1
    assert rows[-1]["frames"] == 1


# --- smaf --------------------------------------------------------------------------------------

def test_smaf_round_trip_and_alignment():
    meta = {"view": "raster", "seeds": [], "label": "x" * 13}
    t = np.array([0, 7, 70000, 2**32 - 1], dtype=np.uint32)
    blob = framing.encode_sma_frame(meta, t, [1, 2, 3, 15], [1, 250, 3, 4], [0, 3, 4, 31],
                                    frame_seq=99, run_number=682, seeded=False, stale=True,
                                    truncated=False)
    d = framing.decode_sma_frame(blob)
    assert framing.SMAF_HEADER.size == 32
    assert d["arrays_offset"] % 8 == 0
    assert d["arrays_offset"] == 32 + ((d["json_len"] + 7) // 8) * 8
    assert len(blob) == d["arrays_offset"] + 7 * 4
    assert d["meta"] == meta
    assert (d["frame_seq"], d["run_number"], d["n_hits"]) == (99, 682, 4)
    assert (d["seeded"], d["stale"], d["truncated"]) == (False, True, False)
    assert list(d["t_rel_ns"]) == list(t)
    assert list(d["ch"]) == [1, 2, 3, 15]
    assert list(d["tot"]) == [1, 250, 3, 4]
    assert list(d["hit_flags"]) == [0, 3, 4, 31]


def test_smaf_refuses_nan_and_ragged_arrays():
    with pytest.raises(ValueError):
        framing.encode_sma_frame({"x": float("nan")}, [], [], [], [])
    with pytest.raises(ValueError):
        framing.encode_sma_frame({}, [1, 2], [1], [1], [1])


def _frame(p, **kw):
    return framing.decode_sma_frame(p.frame_blob(**kw))


def test_the_seeded_view_ships_only_the_seed_windows():
    p = _plugin()
    _feed(p, R1008, run=1008)
    d = _frame(p, view="seeded")
    m = d["meta"]
    assert d["seeded"] and not d["stale"]
    assert 0 < d["n_hits"] < 200, "a few windows of a 16k-hit frame"
    assert np.all(d["hit_flags"] & framing.HIT_IN_SEED)
    assert len(m["seeds"]) == p.cfg.cuts.n_seeds
    for s in m["seeds"]:
        lo, hi = s["hits"]
        t = d["t_rel_ns"][lo:hi].astype(np.int64)
        assert np.all(t >= s["t_rel"] - m["window"]["pre_ns"])
        assert np.all(t <= s["t_rel"] + m["window"]["post_ns"])
        assert s["t_rel"] in set(t[d["ch"][lo:hi] == 1].tolist())
        assert s["pattern"] & 1
    assert np.all(np.diff(d["t_rel_ns"].astype(np.int64)) >= 0)
    assert m["labels"][1] == "S1"
    assert m["n_kept"] == sum(m["per_channel"])


def test_the_raster_ships_the_kept_frame_and_can_drop_channels():
    p = _plugin()
    _feed(p, R1008, run=1008)
    fr = p._last.fr
    d = _frame(p, view="raster")
    assert d["n_hits"] == fr.s_ch.size
    assert np.array_equal(d["ch"], fr.s_ch.astype(np.uint8))
    assert np.array_equal(d["t_rel_ns"].astype(np.int64), fr.s_t - fr.first)
    assert d["meta"]["t0_ns"] == fr.first
    mism = (d["hit_flags"] & framing.HIT_MISMATCH).astype(bool)
    assert np.array_equal(mism, ~fr.consistent[fr.order])
    lsb = (d["hit_flags"] & framing.HIT_FINE_LSB).astype(bool)
    assert np.array_equal(lsb, (fr.fine[fr.order] & 1).astype(bool))

    no7 = _frame(p, view="raster", drop=[7])
    assert 7 not in set(no7["ch"].tolist())
    assert no7["n_hits"] == fr.s_ch.size - int((fr.s_ch == 7).sum())
    assert no7["meta"]["dropped"] == [7]
    assert no7["meta"]["per_channel"][7] > 0, "counts describe the frame, not the selection"

    cut = _frame(p, view="raster", max_hits=100)
    assert cut["truncated"] and cut["n_hits"] == 100
    assert cut["meta"]["n_selected"] == fr.s_ch.size


def test_a_stale_frame_goes_to_the_raster_flagged_and_not_to_the_seeded_view():
    p = _plugin()
    _feed(p, R682[4])
    good_seq = p._last.seq
    _feed(p, R682[1], serial=1)
    r = _frame(p, view="raster")
    assert r["stale"] and r["meta"]["stale"]
    assert np.all(r["hit_flags"] & framing.HIT_STALE)
    assert "no role" in r["meta"]["stale_reason"]
    s = _frame(p, view="seeded")
    assert s["frame_seq"] == good_seq and not s["stale"]


def test_raster_truncation_keeps_the_latest_hits():
    p = _plugin()
    _feed(p, R1008, run=1008)
    full = _frame(p, view="raster")
    cut = _frame(p, view="raster", max_hits=100)
    assert np.array_equal(cut["ch"], full["ch"][-100:])
    t_full = full["meta"]["t0_ns"] + full["t_rel_ns"][-100:].astype(np.int64)
    t_cut = cut["meta"]["t0_ns"] + cut["t_rel_ns"].astype(np.int64)
    assert np.array_equal(t_full, t_cut)
    assert cut["t_rel_ns"][0] == 0, "t0 is the first shipped hit"


def test_a_frame_longer_than_a_u32_of_ns_scales_its_times():
    p = _plugin()
    _feed(p, synth_frame(10**12, far=5_500_000_000))
    fr = p._last.fr
    r = _frame(p, view="raster")
    assert r["time_shift"] == 1 and r["meta"]["time_shift"] == 1
    t = r["meta"]["t0_ns"] + r["t_rel_ns"].astype(np.int64) * 2 ** r["time_shift"]
    assert np.all(np.abs(t - fr.s_t) < 2 ** r["time_shift"])
    assert t[-1] - t[0] > 5 * 10**9, "no clipping at 4.29 s"
    seeded = _frame(p, view="seeded")
    assert seeded["time_shift"] == 0, "a few seed windows are short"


def test_the_seeded_view_keeps_the_last_frame_that_had_seeds():
    p = _plugin()
    first = _feed(p, synth_frame(10**12, seed=1))
    assert first.an.seeds.size > 0
    second = _feed(p, synth_frame(10**12 + 10**7, n=3, seed=2), serial=1)
    assert second.cls == "good" and second.an.seeds.size == 0
    assert _frame(p, view="seeded")["frame_seq"] == first.seq
    assert _frame(p, view="raster")["frame_seq"] == second.seq


def test_a_sparse_counter_does_not_remove_the_seeds():
    words = synth_frame(10**12, seed=3)
    d = W.decode(words)
    keep = (d["ch"] != 5) | (np.cumsum(d["ch"] == 5) <= 2)    # S5: two early hits only
    p = _plugin()
    snap = _feed(p, words[keep])
    assert snap.fr.times(5).size == 2
    assert snap.an.seeds.size == p.cfg.cuts.n_seeds


def test_frames_are_encoded_once_per_frame_view_and_selection():
    p = _plugin()
    _feed(p, R682[4])
    a = p.frame_blob("raster", drop=[7])
    assert p.frame_blob("raster", drop=(7,)) is a
    assert p.frame_blob("raster") is not a
    _feed(p, R682[5], serial=1)
    assert p.frame_blob("raster", drop=[7]) is not a


# --- commands ----------------------------------------------------------------------------------

def _dispatch(p, cmd, args=""):
    srv = Server(p.store, extra=p.commands())
    _size, tag, body = framing.parse_envelope(srv.dispatch(cmd, args))
    return tag, body


def test_commands_through_the_server():
    p = _plugin()
    assert _dispatch(p, "sma::frame") == (framing.TAG_JSON, b'{"no_frame": true}')
    _feed(p, R1008, run=1008)

    tag, body = _dispatch(p, "sma::summary", '{"run_active": true}')
    assert tag == framing.TAG_JSON
    assert json.loads(body)["run_active"] is True

    tag, body = _dispatch(p, "sma::trend", "")
    assert tag == framing.TAG_JSON and "rows" in json.loads(body)

    tag, body = _dispatch(p, "sma::frame", '{"view": "raster", "drop": [7], "max_hits": 5000}')
    assert tag == framing.TAG_SMAF
    d = framing.decode_sma_frame(body)
    assert d["n_hits"] == 5000 and d["truncated"]

    tag, body = _dispatch(p, "sma::frame", '{"view": "sideways"}')
    assert tag == framing.TAG_ERROR and b"view" in body
    tag, _body = _dispatch(p, "sma::summary", "[1]")
    assert tag == framing.TAG_ERROR


def test_the_analyzer_processes_every_frame_and_serves_the_commands():
    """--no-cpu-budget (development only): every frame, in order."""
    from test_analyzer import _FakeClient

    from test_analyzer import _SettingsClient

    a = A.Analyzer(A.make_plugin_factory("sma"), no_cpu_budget=True)
    c = _SettingsClient()
    S.seed(c, a.settings_root, a.settings_defaults)
    a.apply_settings(c, force=True)
    assert a.budget is None and a.process_all is True
    assert a.sampling_mode() == "process all"
    client = _FakeClient()
    client.events = [_Event(R682[4 + (k % 2)], serial=k) for k in range(40)]
    client.events.insert(3, _Event(R682[4], event_id=401))
    a.run_once(client, buf=None)
    assert a.seen == 41
    assert a.processed == 40, "process all: nothing sampled away"
    st = json.loads(framing.parse_envelope(a.server.dispatch("dqm::status", ""))[2])
    assert st["plugin"]["decoded"] == 40
    _s, tag, _b = framing.parse_envelope(a.server.dispatch("sma::frame", ""))
    assert tag == framing.TAG_SMAF


def test_status_is_json_safe():
    p = _plugin()
    json.dumps(p.status(), allow_nan=False)
    _feed(p, R682[0])
    st = p.status()
    assert st["frames_stale"] == 1 and st["last_frame"]["class"] == "stale"
    json.dumps(st, allow_nan=False)
