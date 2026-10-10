"""The epoch repair of a coarse field offset from its fine field (``Cuts/epoch repair``).

Synthetic frames cabled as the DQM's ODB on pinky has it since the S2L/S5
swap: counters S1 1, S2 2, S3 7, S4 4, S5L 12; NIM copies S1L 3, S2L 5, S3L 10,
S4L 11; the NIM offsets and lag nominals give every candidate's delay after
S1. A particle fires every channel at its delay; the board's fault is
modelled where it happens: a channel's coarse field is latched ``lag`` later
than its fine field (plus a per-word spread), and the board puts a word in the
frame of its *coarse* time, so a lagged channel's words of a frame are earlier
in true time than the frame. The fine field is always the true time.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mdqm.dqm import framing
from mdqm.dqm.hist import HistStore
from mdqm.plugins import sma as P
from mdqm.plugins import sma_words as W

T0 = 10**12
TICK = 1 << 14
EPOCH = W.FINE_WRAP_NS
#: The DQM settings on pinky (S2L on input 5, S4 TOT on 4).
LAYOUT = {
    "Channel roles": {"s1": 1, "counters": [1, 2, 7, 4, 12], "rf": 6, "current": -1,
                      "delayed": [-1]},
    "NIM": {"channels": [3, 5, 10, 11, -1], "offset ns": [26, 24, 23, -1, 0],
            "lag nominal ns": [26, 25, 26, 22, 0]},
    "Self check": {"shift min words": 100},
}
#: Each channel's delay after S1 (ns): what LAYOUT's offsets and nominals say.
DELAY = {1: 0, 3: 26, 2: 1, 5: 25, 7: 3, 10: 26, 4: 23, 11: 22, 12: 20}
S2L, S4 = 5, 4


class _Event:
    def __init__(self, words, serial=0):
        self.header = type("H", (), {"event_id": 301, "serial_number": serial,
                                     "timestamp": 0, "trigger_mask": 0})()
        self.banks = {"H000": type("B", (), {"data": np.asarray(words, "<u8").view("<u4")})()}

    def get_bank(self, name):
        return self.banks.get(name)


class _Clock:
    def __call__(self):
        return 1_000_000.0


def with_(*dicts):
    out = {}
    for d in dicts:
        for k, v in (d or {}).items():
            out[k] = {**out.get(k, {}), **v} if isinstance(v, dict) else v
    return out


def make_frames(n_frames=6, span_ns=4_000_000, rate_hz=2e5, lag=None, sd_ticks=0.0,
                random_times=(), random_fine=(), seed=1):
    """Word arrays of ``n_frames`` consecutive frames and the true times.

    ``lag``: {channel: ticks (float, or a function of the true time in ns)}, the
    coarse field's lag; ``sd_ticks`` its spread per word. ``random_times``:
    channels firing at random instead of after S1 (no S1 coincidence).
    ``random_fine``: channels whose fine field is random (a fine-bit fault).
    Returns ``(frames, truth)``: truth = {channel: sorted true times}.
    """
    rng = np.random.default_rng(seed)
    total = n_frames * span_ns
    n = int(rate_hz * total * 1e-9)
    t1 = T0 + np.sort(rng.integers(0, total, n))
    t, ch, tau = [], [], []
    truth = {}
    for c, d in DELAY.items():
        tc = (T0 + np.sort(rng.integers(0, total, n))) if c in random_times else t1 + d
        lg = (lag or {}).get(c)
        if lg is None:
            L = np.zeros(tc.size)
        else:
            L = (lg(tc) if callable(lg) else np.full(tc.size, float(lg))) * TICK
            if sd_ticks:
                L = L + rng.normal(0.0, sd_ticks * TICK, tc.size)
        t.append(tc)
        tau.append(tc + np.round(L).astype(np.int64))
        ch.append(np.full(tc.size, c))
        truth[c] = tc
    t, ch, tau = np.concatenate(t), np.concatenate(ch), np.concatenate(tau)
    coarse = ((tau >> 14) & 0xFFFFFFF).astype(np.uint32)
    fine = (t & (EPOCH - 1)).astype(np.uint32)
    for c in random_fine:
        m = ch == c
        fine[m] = rng.integers(0, EPOCH, int(m.sum())).astype(np.uint32)
    frames = []
    for j in range(n_frames):
        m = (tau >= T0 + j * span_ns) & (tau < T0 + (j + 1) * span_ns)
        o = np.flatnonzero(m)[np.argsort(tau[m], kind="stable")]
        frames.append(W.encode(ch[o], np.full(o.size, 20), coarse[o], fine[o]))
    return frames, truth


def run(frames, settings=None):
    p = P.SmaPlugin(HistStore(), settings=with_(LAYOUT, settings), clock=_Clock())
    fr = []
    for i, words in enumerate(frames):
        assert p.process(_Event(words, i + 1), run_number=4233) is True
        fr.append(p._last.fr)
    return p, fr, p.summary(True)


def chan(s, c):
    return s["channels"][c]


def eff_of(s, c):
    return next(e for e in s["efficiency"] if e["ch"] == c)


def nim_row(s, counter):
    return next(r for r in s["nim"]["counters"] if r["counter"] == counter)


def mismatch_flags(s, c):
    return [f for f in s["flags"]
            if f["code"] == "mismatch" and (f.get("ch") == c or f"(ch {c})" in f["text"])]


def hist_dump(p):
    return {n: np.asarray(p.store.get(n).counts).copy() for n in p.store.names()
            if n.startswith("sma/")}


# --- the pure functions --------------------------------------------------------------

def test_a_zero_offset_repair_is_time_of():
    rng = np.random.default_rng(3)
    coarse = rng.integers(0, 1 << 28, 5000).astype(np.uint32)
    fine = rng.integers(0, EPOCH, 5000).astype(np.uint32)
    D = W.coarse_minus_fine(coarse, fine)
    assert np.array_equal(W.apply_epoch_repair(fine, D, 0), W.time_of(coarse, fine))


def test_every_correction_is_whole_epochs():
    rng = np.random.default_rng(4)
    t = T0 + rng.integers(0, 10**8, 5000)
    coarse = (((t + 75 * TICK) >> 14) & 0xFFFFFFF).astype(np.uint32)
    fine = (t & (EPOCH - 1)).astype(np.uint32)
    D = W.coarse_minus_fine(coarse, fine)
    rep = W.apply_epoch_repair(fine, D, 75 * TICK)
    raw = W.time_of(coarse, fine)
    assert np.all((rep - raw) % EPOCH == 0)
    assert np.array_equal(rep, t & (W.TIME_WRAP_NS - 1)), "the true time, every word"
    assert not np.array_equal(raw, rep), "time_of takes the wrong epoch past 32 ticks"


def test_the_circular_mean_and_R():
    m, r = W.circ_mean_R(np.array([10 * TICK] * 50 + [-10 * TICK] * 50))
    assert r == pytest.approx(np.cos(2 * np.pi * 10 / 64)) and min(m, EPOCH - m) < 1
    m, r = W.circ_mean_R(np.arange(0, EPOCH, TICK))
    assert r < 1e-9
    sums = W.residue_sums(np.array([3] * 4), np.array([5 * TICK + 7] * 4))
    mean, R = W.circ_of_sums(sums)
    assert R[3] == pytest.approx(1.0) and mean[3] == pytest.approx(5.5 * TICK)
    assert np.isnan(mean[0]) and R[0] == 0


def votes(inn, off=None, exp=None):
    """A vote array from per-candidate in-time counts (off time 0, exposure 1000)."""
    n = len(inn)
    return np.array([inn, off or [0] * n, exp or [1000] * n])


def test_a_tie_is_undecided_whatever_the_margin():
    sc = W.sideband_scale(5)
    assert W.decide_epoch(votes([0, 300, 300, 0]), 200, 1, sc).index is None
    assert W.decide_epoch(votes([0, 300, 10, 0]), 200, 5, sc).index == 1
    d = W.decide_epoch(votes([0, 150, 0, 0]), 200, 5, sc)
    assert d.index is None and d.reason == "few", "too few votes"
    assert W.decide_epoch(votes([0, 300, 100, 0]), 200, 5, sc).index is None, "margin 3 < 5"


def test_accidentals_never_decide():
    """In time no higher than the scaled off time: no signal, whatever the count."""
    sc = W.sideband_scale(5)                      # 10 ns / 500 ns
    d = W.decide_epoch(votes([0, 5000, 0], [0, 250_000, 0]), 200, 5, sc)
    assert d.index is None and d.reason == "no signal"
    d = W.decide_epoch(votes([0, 5000, 0], [0, 250_000, 0], [1000, 20_000, 10]), 200, 5, sc)
    assert d.index is None and d.reason == "short", "an unexposed candidate: maybe short frames"


def test_a_candidate_is_judged_per_exposed_word():
    """Exposure 10x lower, same rate: the counts differ, the rates do not."""
    sc = W.sideband_scale(5)
    d = W.decide_epoch(votes([900, 90], [0, 0], [1000, 100]), 200, 5, sc)
    assert d.index is None, "same rate per exposed word: a tie, not 10:1"
    d = W.decide_epoch(votes([900, 9], [0, 0], [1000, 100]), 200, 5, sc)
    assert d.index == 0


def test_the_vote_only_counts_words_inside_the_frame():
    s1 = T0 + np.arange(0, 1_000_000, 1000)           # 1 ms of S1 hits, 1 MHz
    t = s1[100:900] + 25 + EPOCH                      # one epoch late
    v = W.epoch_votes(t, s1, 25, 5, ks=(0, 1, 2))
    assert v[2, 0] == 0 and v[0, 0] == 0, "k=0 puts them 1.05 ms late: outside, not exposed"
    assert v[2, 1] == 800 and v[0, 1] == 800 and v[1, 1] == 0
    assert v[2, 2] == 0


def test_the_correction_rounds_half_up():
    assert W.epoch_correction(0.5 * EPOCH) == -1 and W.epoch_correction(-0.5 * EPOCH) == 0
    assert W.epoch_correction(1.2 * EPOCH) == -1 and W.epoch_correction(-1.6 * EPOCH) == 2


def test_the_candidates_come_from_the_nim_settings():
    assert P.parse_settings(None).epoch_delays() == {}, "unmeasured nominals: no candidate"
    d = P.parse_settings(LAYOUT).epoch_delays()
    assert d == {3: 26, 5: 25, 2: 1, 10: 26, 7: 3, 11: 22, 4: 23}
    assert all(d[c] == DELAY[c] for c in d)


# --- the five cases --------------------------------------------------------------------

def test_healthy_frames_are_left_exactly_as_they_were():
    frames, _ = make_frames()
    p_on, fr_on, s_on = run(frames)
    p_off, fr_off, s_off = run(frames, {"Cuts": {"epoch repair": False}})
    assert all(f.repaired is None for f in fr_on)
    for a, b in zip(fr_on, fr_off, strict=True):
        assert np.array_equal(a.s_t, b.s_t) and np.array_equal(a.time, b.time)
    for name, counts in hist_dump(p_off).items():
        assert np.array_equal(hist_dump(p_on)[name], counts), name
    assert s_on["efficiency"] == s_off["efficiency"]
    assert s_on["nim"]["counters"] == s_off["nim"]["counters"]
    for c in (S2L, S4):
        v = chan(s_on, c)
        assert (v["kind"], v["times"]) == ("healthy", "ok")
        # Monitored all the same: decided, nothing to correct.
        assert v["repair"]["correction"] == 0 and v["repair"]["moved"] == 0
    assert not [f for f in s_on["flags"] if f["code"] in ("mismatch", "nim_pairing")]
    # The frame preparation itself: no offset, the times of time_of.
    plain = W.prepare_frame(frames[0])
    rep = W.prepare_frame(frames[0], repair={})
    assert np.array_equal(plain.s_t, rep.s_t) and rep.repaired is None


def test_an_offset_under_half_an_epoch_moves_nothing_and_says_times_right():
    frames, _ = make_frames(lag={S2L: 7, S4: 7}, sd_ticks=1.0)
    p, frs, s = run(frames)
    p_off, _f, s_off = run(frames, {"Cuts": {"epoch repair": False}})
    for c in (S2L, S4):
        v = chan(s, c)
        assert v["kind"] == "coarse_offset" and v["times"] == "ok"
        assert v["offset_ticks"] == pytest.approx(7, abs=0.6)
        assert v["repair"]["correction"] == 0 and v["repair"]["moved"] == 0
        f = mismatch_flags(s, c)
        assert len(f) == 1 and f[0]["severity"] == "warn" and "times still right" in f[0]["text"]
    for name, counts in hist_dump(p_off).items():
        assert np.array_equal(hist_dump(p)[name], counts), f"{name}: no word moved"
    # The coverage cut takes the frame edge out of the efficiency (115 us here).
    assert eff_of(s, S4)["eff"] > 0.995
    assert eff_of(s, S4)["eff"] > eff_of(s_off, S4)["eff"]
    assert nim_row(s, "S2")["pair_eff"] > 0.995 and nim_row(s, "S4")["pair_eff"] > 0.995


@pytest.mark.parametrize("lag", [40, 75])
def test_an_offset_past_half_an_epoch_is_repaired_by_whole_epochs(lag):
    frames, truth = make_frames(lag={S2L: lag, S4: lag * 1.1}, sd_ticks=6.0)
    p, frs, s = run(frames)
    for fr in frs:
        assert set(fr.repaired) == {S2L, S4}
        raw = W.time_of(fr.coarse, fr.fine)
        d = fr.time - raw
        assert np.all(d % EPOCH == 0), "whole epochs only"
        for c in (S2L, S4):
            m = fr.ch == c
            assert np.isin(fr.time[m], truth[c]).all(), "every repaired word at its true time"
            assert np.count_nonzero(d[m]) > 0
        other = ~np.isin(fr.ch, (S2L, S4))
        assert not d[other].any(), "no other channel touched"
    for c, want in ((S2L, lag), (S4, lag * 1.1)):
        v = chan(s, c)
        assert v["kind"] == "coarse_offset" and v["times"] == "repaired"
        rp = v["repair"]
        assert rp["correction"] == -math.floor(want / 64 + 0.5)
        assert v["offset_ticks"] == pytest.approx(want, abs=2)
        table = {r[0]: r for r in rp["votes"]}
        win = table[rp["correction"]]
        assert win[1] >= 10 * max(win[2], 1), "in time far above off time"
        for nb in (rp["correction"] - 1, rp["correction"] + 1):
            if nb in table and table[nb][3]:
                assert table[nb][1] < 5 * max(table[nb][2], 1), "a neighbour is accidentals"
        assert rp["moved_share"] > 0.3
        f = mismatch_flags(s, c)
        assert len(f) == 1 and f[0]["severity"] == "warn"
        assert "times moved by \u22121 epoch" in f[0]["text"]
        assert "in time" in f[0]["text"] and "off time" in f[0]["text"]
    # The efficiencies are back, and the frame edge does not read as a loss.
    assert eff_of(s, S4)["eff"] > 0.99
    assert nim_row(s, "S2")["pair_eff"] > 0.99 and nim_row(s, "S4")["pair_eff"] > 0.99
    dt = p.store.get("sma/dt_S4_S1")
    ax = dt.x
    peak = ax.lo + (int(np.argmax(dt.counts[1:-1])) + 0.5) * (ax.hi - ax.lo) / ax.n
    assert abs(peak - DELAY[S4]) <= 1, "S4 - S1 back at its delay"
    assert s["timestamp_faults"]["counters"] == []
    assert not [f for f in s["flags"] if f["code"] in ("nim_pairing", "nim_lag")]
    # The same frames with the switch off: votes and verdicts, no repair.
    p_off, frs_off, s_off = run(frames, {"Cuts": {"epoch repair": False}})
    assert all(f.repaired is None for f in frs_off)
    assert chan(s_off, S4)["times"] == "wrong" and eff_of(s_off, S4)["eff"] is None
    assert chan(s_off, S4)["repair"]["would_move"] > 0
    f = mismatch_flags(s_off, S4)
    assert f[0]["severity"] == "error" and "epoch repair is off" in f[0]["text"]
    assert "pairing (S4 + S4L) collapses" in f[0]["text"]
    assert nim_row(s_off, "S4")["pair_eff"] is None and nim_row(s_off, "S4")["pair_eff_reason"]
    assert not [f for f in s_off["flags"] if f["code"] == "nim_pairing"], \
        "the mismatch flag already says it"


def test_no_s1_coincidence_leaves_the_vote_undecided_and_the_times_alone():
    frames, _ = make_frames(lag={S2L: 75}, sd_ticks=6.0, random_times=(S2L,))
    p, frs, s = run(frames)
    assert all(f.repaired is None for f in frs)
    v = chan(s, S2L)
    assert v["kind"] == "coarse_offset" and v["times"] == "wrong"
    assert v["repair"]["correction"] is None and v["repair"]["undecided"] == "no signal"
    f = mismatch_flags(s, S2L)
    assert len(f) == 1 and f[0]["severity"] == "error"
    assert "timestamps wrong" in f[0]["text"] and "no whole-epoch shift" in f[0]["text"]
    assert "0 : 0" not in f[0]["text"]
    assert nim_row(s, "S2")["pair_eff"] is None
    assert not [f for f in s["flags"] if f["code"] == "nim_pairing"]


def test_scattered_fine_bits_are_not_repaired():
    frames, _ = make_frames(random_fine=(S4,))
    p, frs, s = run(frames)
    assert all(not f.repaired for f in frs)
    v = chan(s, S4)
    assert v["kind"] == "scattered" and v["times"] == "wrong" and v["R"] < 0.6
    assert eff_of(s, S4)["eff"] is None and "timestamp fault" in eff_of(s, S4)["reason"]
    f = mismatch_flags(s, S4)
    assert f[0]["severity"] == "error" and "(a fine-bit fault)" in f[0]["text"]
    assert "fine = t/2" not in f[0]["text"]


# --- the run's vote ------------------------------------------------------------------

def test_the_labels_stay_put_when_the_offset_crosses_an_epoch_boundary():
    """The offset drifts from 58 to 70 ticks through 64 (where its residue wraps
    from -6 to +6): one decision, no resync, every word at its true time."""
    span = 4_000_000
    n = 12
    drift = (lambda t: 58 + 12 * (t - T0) / (n * span))
    frames, truth = make_frames(n_frames=n, lag={S2L: drift}, sd_ticks=1.0)
    p, frs, s = run(frames)
    assert p._epoch.resyncs.get(S2L, 0) == 0
    assert chan(s, S2L)["repair"]["correction"] == -1
    for fr in frs:
        m = fr.ch == S2L
        assert np.isin(fr.time[m], truth[S2L]).all()


def test_a_resync_starts_the_vote_again():
    """75 ticks for six frames, then the board is resynchronised (0 ticks): the
    two jump frames confirm it, and from then on nothing is moved. (The frame
    of the change holds words of both offsets and is not judged.)"""
    span = 4_000_000
    frames, truth = make_frames(n_frames=12, lag={S2L: lambda t: np.where(
        t < T0 + 6 * span, 75.0, 0.3)}, sd_ticks=1.0)
    p, frs, s = run(frames)
    assert p._epoch.resyncs.get(S2L) == 1
    for j, fr in enumerate(frs):
        if j == 6:
            continue
        m = fr.ch == S2L
        assert np.isin(fr.time[m], truth[S2L]).all(), j
        assert (S2L in (fr.repaired or {})) == (j < 6), j


def test_a_frame_rebuilt_for_a_page_does_not_vote():
    frames, _ = make_frames(lag={S2L: 75}, sd_ticks=3.0)
    p, frs, s = run(frames)
    before = p._epoch.states[S2L].votes.copy()
    snap = p._snapshot_for(p._last.seq)                  # held: no rebuild
    assert snap is p._last
    p._last = p._last_good = p._last_seeded = None
    p._ring.clear()
    snap = p._snapshot_for(len(frames))                  # rebuilt from the raw ring
    assert snap is not None and S2L in snap.fr.repaired
    assert np.array_equal(p._epoch.states[S2L].votes, before)


def test_a_new_run_resets_the_vote():
    frames, _ = make_frames(lag={S2L: 75}, sd_ticks=3.0)
    p, frs, s = run(frames)
    assert S2L in p._epoch.states
    p.process(_Event(frames[0], 1), run_number=4234)
    assert p._epoch.states[S2L].frames == 1


def test_the_repaired_channels_are_in_the_event_payload():
    frames, _ = make_frames(lag={S2L: 75}, sd_ticks=3.0)
    p, frs, s = run(frames)
    meta = framing.decode_sma_frame(p.frame_blob("raster"))["meta"]
    o, moved = meta["repaired"][str(S2L)]
    assert o == pytest.approx(75 * TICK, abs=2 * TICK) and moved > 0
    healthy, _ = make_frames()
    p, frs, s = run(healthy)
    assert "repaired" not in framing.decode_sma_frame(p.frame_blob("raster"))["meta"]


# --- the review's cases ---------------------------------------------------------------

def truth_share(frs, truth, c):
    m = [np.isin(fr.time[fr.ch == c], truth[c]) for fr in frs]
    return np.concatenate(m).mean()


@pytest.mark.parametrize("lag", [-40, -75, 130, -130])
def test_negative_and_multi_epoch_offsets(lag):
    frames, truth = make_frames(lag={S2L: lag, S4: lag}, sd_ticks=4.0)
    p, frs, s = run(frames)
    for c in (S2L, S4):
        assert truth_share(frs, truth, c) > 0.999
        v = chan(s, c)
        assert v["times"] == "repaired"
        assert v["repair"]["correction"] == -math.floor(lag / 64 + 0.5)
    assert eff_of(s, S4)["eff"] > 0.99
    assert nim_row(s, "S2")["pair_eff"] > 0.99 and nim_row(s, "S4")["pair_eff"] > 0.99


@pytest.mark.parametrize("lag", [50, 75])
def test_frames_shorter_than_the_offset_never_give_a_confident_wrong_epoch(lag):
    """0.9 ms frames at 2 MHz: the words of a frame are 0.8-1.2 ms early, their
    S1 partners in the frame before. A wrong epoch that lands inside the frame
    only meets accidentals and must not win."""
    frames, truth = make_frames(n_frames=12, span_ns=900_000, rate_hz=2e6,
                                lag={S2L: lag, S4: lag}, sd_ticks=3.0)
    p, frs, s = run(frames)
    for c in (S2L, S4):
        v = chan(s, c)
        rp = v["repair"]
        if rp["correction"] is None:
            assert rp["undecided"] == "short" and v["times"] == "wrong"
            assert all(c not in (fr.repaired or {}) for fr in frs)
            f = mismatch_flags(s, c)
            assert "frames shorter than the coarse offset" in f[0]["text"]
            assert f[0]["severity"] == "error"
        else:
            assert rp["correction"] == -1, "only the right epoch may decide"
            assert truth_share(frs, truth, c) > 0.99
    # Never an efficiency of 0 next to "repaired": withheld, with a reason.
    for e in (eff_of(s, S4),):
        assert e["eff"] is None and e["reason"]
    for r in (nim_row(s, "S2"), nim_row(s, "S4")):
        assert r["pair_eff"] is None and r["pair_eff_reason"]


def test_the_frame_edge_is_cut_out_of_the_efficiency():
    """1.2 ms offset, 4 ms frames: without the coverage cut ~30 % of S1 hits
    find no S4 partner in their frame."""
    frames, truth = make_frames(lag={S4: 75}, sd_ticks=3.0)
    p, frs, s = run(frames)
    assert eff_of(s, S4)["eff"] > 0.99
    total = sum(x.n_s1 for x in p._window(60, p._clock()))
    used = sum(int(x.eff_n[3]) for x in p._window(60, p._clock()))
    assert 0.55 < used / total < 0.75


def test_a_low_rate_start_waits_for_the_vote_and_counts_only_decided_frames():
    """12-30 words a frame: the vote decides after a few frames; the frames
    before it count in no efficiency (an undecided frame read 0.86 before)."""
    for rate in (600, 1500):
        frames, truth = make_frames(n_frames=60, span_ns=20_000_000, rate_hz=rate,
                                    lag={S2L: 75, S4: 75}, sd_ticks=3.0)
        p, frs, s = run(frames)
        v = chan(s, S4)
        assert v["times"] == "repaired" and v["repair"]["correction"] == -1
        assert v["repair"]["bad_frames"] > 0
        assert eff_of(s, S4)["eff"] > 0.99


def test_a_low_rate_start_says_it_is_waiting():
    frames, _ = make_frames(n_frames=3, span_ns=20_000_000, rate_hz=600,
                            lag={S2L: 75, S4: 75}, sd_ticks=3.0)
    p, frs, s = run(frames, {"Self check": {"min hits": 20, "shift min words": 10}})
    v = chan(s, S4)
    assert v["repair"]["undecided"] == "few" and v["times"] == "wrong"
    f = mismatch_flags(s, S4)
    assert f and "not enough S1 coincidences yet" in f[0]["text"] and f[0]["severity"] == "warn"
    assert eff_of(s, S4)["eff"] is None


@pytest.mark.parametrize("sd", [0.3, 1.0])
def test_an_offset_of_a_whole_epoch_is_found(sd):
    """64 ticks: the residues look healthy (mismatch about 0 at sd 0.3), only
    the S1 coincidences show it. The monitor vote decides -1 and repairs."""
    frames, truth = make_frames(lag={S2L: 64, S4: 64}, sd_ticks=sd)
    p, frs, s = run(frames)
    for c in (S2L, S4):
        v = chan(s, c)
        assert v["repair"]["correction"] == -1
        assert v["kind"] == "coarse_offset" and v["times"] == "repaired"
        assert truth_share(frs[1:], truth, c) > 0.99
        assert mismatch_flags(s, c), "the shifter is told"
    assert eff_of(s, S4)["eff"] > 0.95


def test_repair_off_counts_the_words_a_repair_would_move():
    """29 ticks with a spread: most words are right, the tail past 32 ticks is
    an epoch late. With the repair off that is not "times still right"."""
    frames, _ = make_frames(lag={S2L: 29}, sd_ticks=3.0)
    p, frs, s = run(frames, {"Cuts": {"epoch repair": False}})
    v = chan(s, S2L)
    assert v["repair"]["correction"] == 0
    assert 0.05 < v["repair"]["would_move_share"] < 0.25
    assert v["times"] == "wrong"
    f = mismatch_flags(s, S2L)
    assert "epoch repair is off" in f[0]["text"] and "times still right" not in f[0]["text"]
    p, frs, s = run(frames)
    assert chan(s, S2L)["times"] == "repaired"
    assert "past half an epoch moved by \u22121 epoch" in mismatch_flags(s, S2L)[0]["text"]


# --- a slip mid-run, impure channels ------------------------------------------------------

def make_frames_extra(spans, rate_hz, lag, sd=0.3, purity=None, dead_after=None, seed=3):
    """Like make_frames, on frames of the given spans (ns), with ``purity``: {ch:
    share of its words with an S1 partner} (the rest at random times) and
    ``dead_after``: {ch: true time from which the channel fires no more}."""
    rng = np.random.default_rng(seed)
    total = int(sum(spans))
    n = int(rate_hz * total * 1e-9)
    t1 = T0 + np.sort(rng.integers(0, total, n))
    t, ch, tau, truth = [], [], [], {}
    for c, d in DELAY.items():
        tc = t1 + d
        if dead_after and c in dead_after:
            tc = tc[tc < dead_after[c]]
        if purity and c in purity:
            extra = int(tc.size * (1 / purity[c] - 1))
            tc = np.sort(np.r_[tc, T0 + rng.integers(0, total, extra)])
        lg = (lag or {}).get(c)
        L = np.zeros(tc.size) if lg is None else (
            (lg(tc) if callable(lg) else np.full(tc.size, float(lg))) * TICK
            + rng.normal(0, sd * TICK, tc.size))
        t.append(tc)
        tau.append(tc + np.round(L).astype(np.int64))
        ch.append(np.full(tc.size, c))
        truth[c] = tc
    t, ch, tau = np.concatenate(t), np.concatenate(ch), np.concatenate(tau)
    coarse = ((tau >> 14) & 0xFFFFFFF).astype(np.uint32)
    fine = (t & (EPOCH - 1)).astype(np.uint32)
    frames, edges = [], T0 + np.r_[0, np.cumsum(spans)].astype(np.int64)
    for j in range(len(spans)):
        m = (tau >= edges[j]) & (tau < edges[j + 1])
        o = np.flatnonzero(m)[np.argsort(tau[m], kind="stable")]
        frames.append(W.encode(ch[o], np.full(o.size, 20), coarse[o], fine[o]))
    return frames, truth


@pytest.mark.parametrize("base", [0.0, 75.0])
@pytest.mark.parametrize("frac", [0.25, 0.5])
def test_a_whole_epoch_slip_mid_run_is_caught(base, frac):
    """S4 healthy (or already repaired at 75 ticks) slips by 64 ticks: its fields
    look as before, only the S1 coincidences move to the next epoch. The recent
    votes catch it within a few frames, the frames before count in no
    efficiency, and from then on the times are right again."""
    n, span = 48, 4_000_000
    tstep = T0 + int(frac * n * span)
    frames, truth = make_frames_extra([span] * n, 1e6, {S4: lambda t: np.where(
        t >= tstep, base + 64.0, base)})
    p, frs, s = run(frames)
    j0 = int(frac * n)
    right = [np.isin(fr.time[fr.ch == S4], truth[S4]).mean() for fr in frs]
    fixed = next(j - j0 for j in range(j0, n) if right[j] > 0.99)
    every = W.EPOCH_MONITOR_EVERY
    assert fixed <= 4 * every, "caught within a few voted frames"
    assert all(r > 0.99 for r in right[j0 + fixed:])
    assert all(r > 0.99 for r in right[:j0]), "nothing changed before the slip"
    # Only the frames before the first voted one after the slip can count.
    wrong_counted = [j for j in range(n) if right[j] < 0.99
                     and not frs[j].epoch.channels[S4].bad_times]
    assert all(j0 <= j < j0 + every for j in wrong_counted), wrong_counted
    assert p._epoch.resyncs.get(S4) == 1
    leak = len(wrong_counted) / n
    assert eff_of(s, S4)["eff"] > 0.99 - leak
    assert nim_row(s, "S4")["pair_eff"] > 0.99 - leak


def test_a_counter_that_dies_mid_run_is_not_called_a_slip():
    """No candidate gets its coincidences: the efficiency shows the drop
    (no frame withheld), so the efficiency_drop alarm can still see it."""
    n, span = 30, 4_000_000
    frames, _ = make_frames_extra([span] * n, 1e6, {}, dead_after={S4: T0 + n * span // 2})
    p, frs, s = run(frames)
    assert not any(fr.epoch.channels[S4].bad_times for fr in frs if S4 in fr.epoch.channels)
    assert p._epoch.resyncs.get(S4, 0) == 0
    assert 0.4 < eff_of(s, S4)["eff"] < 0.6


@pytest.mark.parametrize("rate", [3e6, 5e6])
def test_an_impure_channel_at_high_rate_decides_on_its_excess(rate):
    """30 % of the words have an S1 partner, the rest are random: in time is
    only a few times the off time, but the excess is far above its spread."""
    frames, truth = make_frames_extra([3_000_000] * 8, rate, {S2L: 75, S4: 75}, sd=3.0,
                                      purity={S2L: 0.3, S4: 0.3})
    p, frs, s = run(frames)
    for c in (S2L, S4):
        rp = chan(s, c)["repair"]
        assert rp["correction"] == -1
        if rate > 4e6:
            assert rp["in_time"] < 10 * rp["off_time"], "the old 10x rule would not decide"
        assert np.isin(frs[-1].time[frs[-1].ch == c], truth[c]).mean() > 0.99
