"""TOT + NIM in the SMA plugin: settings, the merge, histograms, summary, flags, the CLI.

Synthetic frames cabled as since run 1015 (``sma_layouts.TOT_1015`` /
``NIM_1015``: S1 1 + S1L 3, S2 2 + 9, S3 7 + 10, S4 4 + 11, S5 5 + 12, RF 6),
built event by event so which hit has which partner is known. The pairing
rule itself is ``tests/test_sma_nim.py``'s (parity with the reco header).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from sma_layouts import NIM_1015, OLD_LAYOUT, TOT_1015, old_layout, words_of
from test_sma_file import write_file

from mdqm.dqm import framing
from mdqm.dqm.hist import HistStore
from mdqm.plugins import sma as P
from mdqm.plugins import sma_nim as N
from mdqm.plugins import sma_words as W
from mdqm.tools import sma_file as S

T0 = 10**12
SPACING = 10_000
N_EV = 300
#: The healthy event: S_k's TOT word at k - 1 ns, its NIM copy 1 ns later, RF.
NORMAL = {**{c: [k] for k, c in enumerate(TOT_1015)},
          **{n: [k + 1] for k, n in enumerate(NIM_1015)}, 6: [40, 60]}


class _Event:
    def __init__(self, words, serial=0):
        self.header = type("H", (), {"event_id": 301, "serial_number": serial,
                                     "timestamp": 0, "trigger_mask": 0})()
        self.banks = {"H000": type("B", (), {"data": np.asarray(words, "<u8").view("<u4")})()}

    def get_bank(self, name):
        return self.banks.get(name)


class _Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def ev(drop=(), **extra):
    """NORMAL without the channels in `drop`; ``ch7=[(3, 20), (23, 20)]`` style
    keyword arguments replace a channel's hits by (dt, ToT) pairs."""
    e = {c: [(d, 20) for d in v] for c, v in NORMAL.items() if c not in drop}
    for key, hits in extra.items():
        e[int(key[2:])] = list(hits)
    return e


def frame(events, fine_lag=None, start=0):
    """Words of one event per entry, SPACING ns apart, from T0 + ``start``."""
    t, ch, tot = [], [], []
    for k, e in enumerate(events):
        for c, hits in e.items():
            for d, code in hits:
                t.append(T0 + start + k * SPACING + d)
                ch.append(c)
                tot.append(code)
    return words_of(t, ch, tot, fine_lag=fine_lag)


#: The shift check needs 1000 S1 words by default before it says "ok", and
#: the NIM flags that compare times wait for it (as the efficiency flags do):
#: one 300-event frame is enough here.
CHECKED = {"Self check": {"shift min words": 100}}
#: The merge is off by default (until the offsets are measured); the tests of
#: the merge turn it on.
MERGE = {"NIM": {"merge": True}}


def with_(*dicts):
    """Settings dicts merged section by section (a later key wins)."""
    out = {}
    for d in dicts:
        for k, v in (d or {}).items():
            out[k] = {**out.get(k, {}), **v} if isinstance(v, dict) else v
    return out


def plugin(settings=None, clock=None, checked=True):
    s = with_(CHECKED if checked else None, settings)
    return P.SmaPlugin(HistStore(), settings=s, clock=clock or _Clock())


def feed(p, words, serial=0):
    assert p.process(_Event(words, serial), run_number=1015) is True
    return p._last


def h(p, name):
    return p.store.get(f"sma/{name}")


def codes(s):
    return {f["code"] for f in s["flags"]}


def nim_row(s, counter):
    return next(r for r in s["nim"]["counters"] if r["counter"] == counter)


# --- settings ---------------------------------------------------------------------------

def test_the_nim_defaults():
    cfg = P.parse_settings(None)
    assert cfg.errors == []
    nim = cfg.nim
    assert nim.channels == NIM_1015 and nim.offsets == (0,) * 5 and nim.nominal == (0,) * 5
    assert not nim.merge and not nim.merge_when_lagged and not cfg.merging, \
        "merge off until the offsets are measured on a clean run >= 1015"
    assert nim.echo == (False, False, True, False, False)
    assert nim.wide_budget == 1024 and nim.lag_every == 4
    assert cfg.check["nim lag min votes"] == 3
    assert nim.cfg == N.NimConfig()
    assert nim.pairs(cfg.roles.counters) == list(zip(range(5), TOT_1015, NIM_1015, strict=True))


@pytest.mark.parametrize("roles, chans, what", [
    ({}, [3, 2, 10, 11, 12], "2 (S2's NIM copy) is counter S2"),
    ({}, [1, 9, 10, 11, 12], "1 (S1's NIM copy) is S1"),
    ({}, [3, 9, 6, 11, 12], "6 (S3's NIM copy) is the RF"),
    ({"current": 9}, NIM_1015, "9 (S2's NIM copy) is current"),
    ({"delayed": [8, 12]}, NIM_1015, "12 (S5's NIM copy) is delayed"),
])
def test_a_nim_channel_with_another_role_turns_nim_off(roles, chans, what):
    p = plugin({"Channel roles": roles, "NIM": {"channels": chans, "merge": True}})
    assert len(p.cfg.errors) == 1, p.cfg.errors
    e = p.cfg.errors[0]
    assert e.startswith("Channel roles look pre-1015: NIM/channels ") and what in e, e
    assert "NIM is off" in e and "docs/SMA-DQM.md" in e
    assert p.cfg.nim.channels == (-1,) * 5 and not p.cfg.nim.active and not p.cfg.merging
    assert not [n for n in p.store.names() if "nim" in n]
    assert "settings" in codes(p.summary())


def test_an_old_pinky_odb_upgraded_with_the_seeded_nim_defaults():
    """The real upgrade: a pre-1015 role tree (counters 1-5, current 7, delayed
    8-10) where the seed created only the missing NIM keys (their defaults)."""
    old = {k: v for k, v in OLD_LAYOUT.items() if k != "NIM"}
    p = plugin(old)
    assert p.cfg.errors == [
        "Channel roles look pre-1015: NIM/channels 3 (S1's NIM copy) is counter S3, "
        "9 (S2's NIM copy) is delayed, 10 (S3's NIM copy) is delayed; NIM is off (no pairing, "
        "no merge). Set the run-1015 roles with the odbedit lines in docs/SMA-DQM.md "
        "('Settings'), or NIM/channels = -1 for the old cabling"]
    assert not p.cfg.nim.active and not p.cfg.merging
    assert p.cfg.role_channels() == {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    assert P.parse_settings(OLD_LAYOUT).errors == [], "the old layout with NIM off is fine"


def test_a_repeated_nim_channel_drops_that_entry_only():
    p = plugin({"NIM": {"channels": [3, 9, 10, 9, 12]}})
    assert len(p.cfg.errors) == 1, p.cfg.errors
    assert "S4's NIM channel 9 is also another counter's NIM channel" in p.cfg.errors[0]
    assert p.cfg.nim.channels == (3, 9, 10, -1, 12)
    assert "sma/nim_dt_S4" not in p.store and "sma/nim_dt_S2" in p.store


@pytest.mark.parametrize("key, value, want", [
    ("channels", [3, 9], NIM_1015),
    ("offset ns", [1, 2], (0,) * 5),
    ("lag nominal ns", [0, 0, 0, 0], (0,) * 5),
    ("channels", [3, 9, 10, 11, 99], NIM_1015),
    ("offset ns", [0, "x", 0, 0, 0], (0,) * 5),
])
def test_a_per_counter_list_of_the_wrong_length_or_value_falls_back(key, value, want):
    p = plugin({"NIM": {key: value}})
    assert len(p.cfg.errors) == 1 and p.cfg.errors[0].startswith(f"NIM/{key}=")
    attr = {"channels": "channels", "offset ns": "offsets", "lag nominal ns": "nominal"}[key]
    assert getattr(p.cfg.nim, attr) == want


def test_per_counter_lists_follow_the_counters():
    """Four counters: the five-entry defaults do not fit and fall back to no NIM."""
    p = plugin({"Channel roles": {"counters": [1, 2, 7, 4]}})
    assert p.cfg.nim.channels == (-1,) * 4 and not p.cfg.merging
    assert [e for e in p.cfg.errors if e.startswith("NIM/channels=")]
    assert not [e for e in p.cfg.errors if "offset" in e], "not judged without a NIM channel"
    ok = plugin({"Channel roles": {"counters": [1, 2, 7, 4]},
                 "NIM": {"channels": [3, 9, 10, 11], "offset ns": [0, 1, 2, 3],
                         "lag nominal ns": [0, 0, 0, 0]}})
    assert ok.cfg.errors == [] and ok.cfg.nim.offsets == (0, 1, 2, 3)


def test_offsets_are_whole_ns_and_options_are_checked():
    p = plugin({"NIM": {"offset ns": [0.4, 2.6, -3, 0, 0], "merge": "n",
                        "time source": "NIM", "pair window ns": 15.0}})
    assert p.cfg.errors == ["NIM/offset ns=[0.4, 2.6, -3, 0, 0]: whole ns only; "
                            "using [0, 3, -3, 0, 0]"], "rounded with a settings note"
    assert p.cfg.nim.offsets == (0, 3, -3, 0, 0)
    assert plugin({"NIM": {"offset ns": [0.0, 3, -3, 0, 0]}}).cfg.errors == []
    assert not p.cfg.nim.merge and not p.cfg.merging
    assert p.cfg.nim.cfg.time_source == "nim" and p.cfg.nim.cfg.pair_window_ns == 15
    q = plugin({"NIM": {"time source": "both", "echo counters": [7], "merge": "perhaps",
                        "lag dominance": 0.5}})
    keys = sorted(e.split("=")[0] for e in q.cfg.errors)
    assert keys == ["NIM/echo counters", "NIM/lag dominance", "NIM/merge", "NIM/time source"]
    assert q.cfg.nim.cfg.time_source == "tot" and not q.cfg.nim.merge
    assert q.cfg.nim.echo == (False, False, True, False, False)


def test_no_nim_copies_at_all():
    for chans in ([-1], [-1] * 5, -1):
        p = plugin({"NIM": {"channels": chans}})
        assert p.cfg.errors == [] and not p.cfg.nim.active and not p.cfg.merging
        assert not [n for n in p.store.names() if "nim" in n or n.endswith("coinc_tot")]
    feed(p, frame([ev() for _ in range(20)]))
    s = p.summary(True)
    assert s["nim"]["counters"] == [] and s["nim_merge"] is False and s["nim_lag_held"] == []
    assert s["roles"]["nim"] == [-1] * 5
    assert p._last.fr.counter_hits is None and p._last.fr.pairing is None


def test_a_nim_setting_changes_the_fingerprint_and_rebuilds():
    base = P.shape_fingerprint({})
    new = {"NIM": {"offset ns": [0, 4, 0, 0, 0]}}
    assert P.shape_fingerprint(new) != base
    assert P.shape_fingerprint({"NIM": {"merge": True}}) != base
    assert P.shape_fingerprint({"Self check": {"nim offset max ns": 9.0}}) == base, \
        "a flag threshold changes no plot"
    assert P.shape_fingerprint({"NIM": {"wide pairs per frame": 0, "lag vote every": 1}}) \
        == base, "the CPU knobs change no plot"
    p = plugin()
    feed(p, frame([ev() for _ in range(50)]))
    assert h(p, "nim_dt_S2").entries == 50
    epoch = p.epoch
    p.apply_settings(new, rebuild=P.shape_fingerprint(new) != base)
    assert p.epoch == epoch + 1 and p.rebuilds == 1
    assert h(p, "nim_dt_S2").entries == 0 and "offset 4 ns" in h(p, "nim_dt_S2").title


# --- roles, labels, the per-channel table -----------------------------------------------

def test_roles_labels_and_pair_of():
    p = plugin()
    feed(p, frame([ev() for _ in range(20)]))
    labels = p.labels()
    assert [labels[n] for n in NIM_1015] == ["S1L", "S2L", "S3L", "S4L", "S5L"]
    s = p.summary(True)
    rows = {c["ch"]: c for c in s["channels"]}
    for c, n in zip(TOT_1015, NIM_1015, strict=True):
        assert rows[n]["role"] == "nim" and rows[n]["pair_of"] == c
        assert rows[c]["pair_of"] == n
    assert rows[6]["pair_of"] is None and rows[8]["role"] == ""
    assert s["roles"] == {"s1": 1, "counters": list(TOT_1015), "rf": 6, "current": -1,
                          "delayed": [], "nim": list(NIM_1015)}


def test_the_stale_rule_does_not_count_nim_words_as_junk():
    """A frame of NIM words alone (no S1): they have a role, so it is not 'stale'."""
    only_nim = frame([{n: [(1, 10)] for n in NIM_1015} for _ in range(200)])
    p = plugin()
    assert feed(p, only_nim).cls == "good"


# --- the merge -------------------------------------------------------------------------

def _seed_times(p, blob=None):
    d = framing.decode_sma_frame(blob or p.frame_blob("seeded"))
    t0 = d["meta"]["t0_ns"]
    return [(t0 + s["t_rel"] - T0) / SPACING for s in d["meta"]["seeds"]], d


@pytest.mark.parametrize("merge", [True, False])
def test_an_s1l_only_hit_is_an_s1_seed_only_with_the_merge(merge):
    events = [ev() for _ in range(N_EV)]
    events[N_EV - 3] = ev(drop=(1,))                 # S1's TOT word missing, S1L there
    p = plugin({"NIM": {"merge": merge}})
    feed(p, frame(events))
    seeds, d = _seed_times(p)
    k = N_EV - 3
    if merge:
        # The merged S1 hit takes the S1L word's time (offset 0): 1 ns after the event.
        assert k + 1 / SPACING in seeds and len(seeds) == W.N_SEEDS
        m = d["meta"]["seeds"][seeds.index(k + 1 / SPACING)]
        assert m["s1_tot"] == 1, "a NIM-only S1 hit has the substituted ToT"
        fr = p._last.fr
        i = int(np.flatnonzero((fr.s_ch == 3) & (fr.s_t == T0 + k * SPACING + 1))[0])
        assert m["s1_word"] == int(fr.word_index[fr.order[i]]), "the seed's word is the S1L word"
        assert p._last.an.t_s1.size == N_EV
    else:
        assert all(abs(x - k) > 0.5 for x in seeds)
        assert p._last.an.t_s1.size == N_EV - 1
    # The event display's seed choices see the same hits: "s1" asked for
    # explicitly (with a filter, so it is chosen at request time) too.
    chosen, _ = _seed_times(p, p.frame_blob("seeded", seed="s1", pattern={"2": "present"}))
    assert (k + 1 / SPACING in chosen) == merge


def test_pattern_efficiency_and_s1_coinc_use_the_merged_hits():
    # Every third event: S2's TOT word missing (its NIM copy there).
    events = [ev(drop=(2,)) if k % 3 == 0 else ev() for k in range(N_EV)]
    p = plugin(MERGE)
    feed(p, frame(events))
    s = p.summary(True)
    eff = {e["counter"]: e["eff"] for e in s["efficiency"]}
    assert eff["S2"] == 1.0, "S2L fills in"
    pat = h(p, "pattern")
    assert pat.counts[1 + 0b11111] == pat.entries == N_EV
    merged, tot_only = h(p, "s1_coinc").counts, h(p, "s1_coinc_tot").counts
    assert merged[2] == N_EV and tot_only[2] == N_EV - N_EV // 3, "the TOT-only twin stays TOT"
    assert h(p, "s1_coinc_tot").entries == N_EV
    assert merged[[1, 3, 4, 5]].tolist() == tot_only[[1, 3, 4, 5]].tolist()
    assert s["nim_merge"] is True and s["nim"]["merging"] is True

    q = plugin({"NIM": {"merge": False}})
    feed(q, frame(events))
    sq = q.summary(True)
    eff = {e["counter"]: e["eff"] for e in sq["efficiency"]}
    assert eff["S2"] == pytest.approx(1 - (N_EV // 3) / N_EV, abs=1e-3)
    assert h(q, "s1_coinc").counts.tolist() == h(q, "s1_coinc_tot").counts.tolist()
    assert sq["nim_merge"] is False
    # The pairing is measured either way.
    assert nim_row(sq, "S2")["pair_eff"] == 1.0 and nim_row(sq, "S2")["nim_only"] == N_EV // 3


def test_an_s3_echo_is_not_paired():
    # S3 (TOT ch 7, ToT 20) at 2 ns, an echo word at its trailing edge (22 ns),
    # an S3L word 1 ns after the echo: 21 ns from S3, outside the window.
    echo_ev = ev(ch7=[(2, 20), (22, 20)], ch10=[(3, 10), (23, 10)])
    events = [echo_ev if k % 10 == 0 else ev() for k in range(N_EV)]
    p = plugin()
    fr = feed(p, frame(events)).fr
    k = 0
    i_echo = int(np.flatnonzero((fr.s_ch == 7) & (fr.s_t == T0 + k * SPACING + 22))[0])
    i_late = int(np.flatnonzero((fr.s_ch == 10) & (fr.s_t == T0 + k * SPACING + 23))[0])
    i_main = int(np.flatnonzero((fr.s_ch == 7) & (fr.s_t == T0 + k * SPACING + 2))[0])
    w = fr.pairing
    assert w.cls[i_echo] == N.ECHO_WORD and w.partner[i_echo] == -1
    assert w.cls[i_late] == N.NIM_ONLY and w.partner[i_late] == -1
    assert w.cls[i_main] == N.PAIRED and fr.s_ch[w.partner[i_main]] == 10
    assert w.partner[w.partner[i_main]] == i_main
    assert w.flags[i_echo] & N.ECHO
    cl = h(p, "nim_classes_S3").counts
    assert cl[1 + 3] == N_EV // 10 and cl[1 + 2] == N_EV // 10, "echo, NIM-only"
    assert cl[1 + 0] == N_EV
    # Without the echo rule the echo word takes the late NIM word.
    q = plugin({"NIM": {"echo counters": [-1]}})
    fq = feed(q, frame(events)).fr
    assert fq.pairing.cls[i_echo] == N.PAIRED and fq.pairing.partner[i_echo] == i_late


def test_the_per_hit_pairing_is_on_the_frame():
    p = plugin()
    fr = feed(p, frame([ev() for _ in range(40)])).fr
    w = fr.pairing
    assert w.cls.shape == w.partner.shape == w.flags.shape == fr.s_ch.shape
    assert set(w.cls[fr.s_ch == 6].tolist()) == {P.NIM_NO_CLASS}
    paired = np.flatnonzero(w.partner >= 0)
    assert paired.size == 2 * 5 * 40
    assert np.array_equal(w.partner[w.partner[paired]], paired)
    for c, n in zip(TOT_1015, NIM_1015, strict=True):
        m = fr.s_ch == c
        assert np.all(fr.s_ch[w.partner[m]] == n)
    assert P.snapshot_bytes(p._last) > 0


def test_a_frame_rebuilt_from_the_raw_ring_is_merged_too():
    events = [ev() for _ in range(N_EV)]
    events[N_EV - 3] = ev(drop=(1,))
    p = plugin(MERGE)
    snap = feed(p, frame(events))
    p._last = p._last_good = p._last_seeded = None
    p._ring.clear()
    again = p._snapshot_for(snap.seq)
    assert again is not snap and again.fr.counter_hits is not None
    assert again.an.t_s1.size == N_EV


# --- the lag fault ---------------------------------------------------------------------

#: Not a multiple of SPACING: the lagged words land between events.
LAG = 155_000


#: One faulted vote is enough for the nim_lag flag (default: 3).
ONE_VOTE = {"Self check": {"nim lag min votes": 1}}


@pytest.mark.parametrize("merge, merge_when_lagged", [(True, False), (True, True), (False, False)])
def test_a_lagged_nim_channel_is_held_back_and_flagged(merge, merge_when_lagged):
    events = [ev() for _ in range(N_EV)]
    p = plugin(with_({"NIM": {"merge": merge, "merge when lagged": merge_when_lagged}}, ONE_VOTE))
    fr = feed(p, frame(events, fine_lag={11: LAG})).fr
    s = p.summary(True)
    r = nim_row(s, "S4")
    assert r["lag"]["faulted"] == 1 and r["lag"]["voted"] == 1
    assert r["lag"]["last_ns"] == pytest.approx(LAG, abs=10) and r["lag"]["last_state"] == "faulted"
    assert nim_row(s, "S2")["lag"]["ok"] == 1, "the other channels vote ok"
    assert "nim_lag" in codes(s)
    lag_h = h(p, "nim_lag_S4")
    assert lag_h.counts[1 + (LAG + 4) // 1024] > 0.9 * N_EV
    assert r["nim_only"] == N_EV, "155 us off: nothing pairs"
    assert r["lag"]["state"] == "faulted" and r["lag"]["state_faulted_frac"] == 1.0
    if not merge:
        # Measured and flagged; nothing is merged, so nothing is "held back".
        assert fr.counter_hits is None
        assert r["lag_held"] == 0 and s["nim_lag_held"] == []
        assert not np.any(fr.pairing.flags & P.NIM_LAG_HELD)
        assert h(p, "nim_classes_S4").counts[1 + 4] == 0
        assert "held back" not in next(f["text"] for f in s["flags"] if f["code"] == "nim_lag")
        assert "nim_pairing" in codes(s), "the pairing is judged with the merge off"
        return
    hits4 = fr.counter_hits[4]
    if merge_when_lagged:
        assert r["lag_held"] == 0 and s["nim_lag_held"] == []
        assert hits4.t.size == 2 * N_EV
        assert "held back" not in next(f["text"] for f in s["flags"] if f["code"] == "nim_lag")
    else:
        assert r["lag_held"] == N_EV and s["nim_lag_held"] == [11]
        assert hits4.t.size == N_EV, "the TOT words only"
        assert h(p, "nim_classes_S4").counts[1 + 4] == N_EV
        held = fr.pairing.flags[fr.s_ch == 11] & P.NIM_LAG_HELD
        assert np.all(held)
        assert fr.counter_hits[2].t.size == N_EV, "S2 unaffected"


def _lag_frames(p, plan, n_quiet=40):
    """Feed frames by plan: "F" a 300-event frame with S4L lagged, "f" a quiet
    one (``n_quiet`` events, too few for a vote) lagged, "o" a quiet one not
    lagged, "O" a 300-event healthy one. Returns the merged S4 hit count per frame."""
    out = []
    for kind in plan:
        j = p._test_frames = getattr(p, "_test_frames", -1) + 1      # 100 ms apart
        n = N_EV if kind.isupper() else n_quiet
        lag = {11: LAG} if kind.lower() == "f" else None
        fr = feed(p, frame([ev() for _ in range(n)], fine_lag=lag, start=j * 10**8)).fr
        out.append(int(fr.counter_hits[4].t.size))
    return out


def test_quiet_frames_after_a_faulted_vote_stay_held_back():
    """S2: the lag fault is whole-file. A frame with too few NIM words for a
    vote ("none") takes the epoch's last decisive vote, so its lagged NIM-only
    hits are not merged (no doubled S4); a later "ok" vote clears the state."""
    p = plugin(with_(MERGE, {"NIM": {"lag vote every": 1}}))
    n = _lag_frames(p, "Fffff")
    assert n == [N_EV, 40, 40, 40, 40], "the TOT words only, not 80"
    r = nim_row(p.summary(True), "S4")
    assert r["lag"]["faulted"] == 1 and r["lag"]["none"] == 4, "one decisive vote"
    assert r["lag"]["state_faulted"] == 5 and r["lag_held"] == N_EV + 4 * 40
    assert r["lag"]["state"] == "faulted" and r["lag"]["epoch_faulted_votes"] == 1
    # A healthy dense frame votes "ok": the next quiet lagged frame is merged.
    assert _lag_frames(p, "Of")[1] == 80
    assert p._nim_mem[3].state == "ok"
    # Without a vote of its own, a quiet frame at the start of an epoch is not held.
    q = plugin(MERGE)
    assert _lag_frames(q, "f") == [80]


def test_the_lag_state_is_cleared_by_a_rebuild():
    p = plugin(MERGE)
    _lag_frames(p, "F")
    assert p._nim_mem[3].state == "faulted"
    new = with_(CHECKED, MERGE, {"NIM": {"offset ns": [0, 0, 0, 1, 0]}})
    p.apply_settings(new, rebuild=True)
    assert p._nim_mem == {}
    assert _lag_frames(p, "f") == [80]


def test_the_lag_vote_runs_every_nth_frame():
    """S5: one vote in "lag vote every" frames once a state is known; the
    frames between take it; nim_lag_Sk is filled from the voted frames only."""
    p = plugin(with_(MERGE, {"NIM": {"lag vote every": 4}}))
    assert _lag_frames(p, "FFFFFFFFF") == [N_EV] * 9, "held in the unvoted frames too"
    r = nim_row(p.summary(True), "S4")
    assert (r["lag"]["faulted"], r["lag"]["skipped"]) == (3, 6), "frames 1, 5, 9 vote"
    assert r["lag"]["state_faulted"] == 9 and r["lag"]["voted"] == 3
    assert h(p, "nim_lag_S4").entries == 3 * N_EV
    assert nim_row(p.summary(True), "S2")["lag"]["ok"] == 3
    # Before any decisive vote every frame votes.
    q = plugin(with_(MERGE, {"NIM": {"lag vote every": 4}}))
    _lag_frames(q, "fff")
    assert nim_row(q.summary(True), "S4")["lag"]["none"] == 3


def test_nim_lag_needs_a_few_faulted_votes():
    """N5: one faulted vote among quiet frames does not raise nim_lag; three do."""
    p = plugin(with_(MERGE, {"NIM": {"lag vote every": 1}}))
    _lag_frames(p, "Fff")
    s = p.summary(True)
    assert "nim_lag" not in codes(s) and nim_row(s, "S4")["lag"]["epoch_faulted_votes"] == 1
    _lag_frames(p, "FF")
    s = p.summary(True)
    f = next(f for f in s["flags"] if f["code"] == "nim_lag")
    assert "in 5 of 5 frames (3 of 3 voted" in f["text"] and "held back" in f["text"]


def test_lag_pairs_with_given_positions_is_the_same():
    rng = np.random.default_rng(3)
    ch = rng.choice([1, 3, 6, 9], 5000)
    coarse = rng.integers(0, 40, 5000).cumsum().astype(np.uint32) // 7
    fine = rng.integers(0, 1 << 20, 5000).astype(np.uint32)
    for n in (3, 9):
        a = N.lag_pairs(ch, coarse, fine, n, 1)
        b = N.lag_pairs(None, coarse, fine, n, 1, ref_idx=np.flatnonzero(ch == 1),
                        nim_idx=np.flatnonzero(ch == n))
        assert a.size and np.array_equal(a, b)


# --- histograms, summary, flags, trend ----------------------------------------------

def test_histograms_and_summary_of_a_healthy_frame():
    events = [ev() for _ in range(N_EV)]
    clk = _Clock()
    p = plugin(clock=clk)
    feed(p, frame(events))
    for k in range(1, 6):
        for stem in ("nim_dt", "nim_dt_wide", "nim_walk", "nim_classes", "nim_width",
                     "nim_candidates", "nim_lag"):
            assert f"sma/{stem}_S{k}" in p.store
    dt = h(p, "nim_dt_S1")
    assert dt.entries == N_EV and dt.counts[1 + 200 + 1] == N_EV, "every NIM word at +1 ns"
    walk = h(p, "nim_walk_S1")
    assert walk.entries == N_EV and walk.counts[1 + 20, 1 + 1 + 20] == N_EV
    assert h(p, "nim_width_S1").counts[1 + 20] == N_EV
    assert h(p, "nim_candidates_S1").counts[1 + 1] == N_EV
    assert h(p, "nim_dt_wide_S1").entries > 0
    s = json.loads(json.dumps(p.summary(True), allow_nan=False))
    r = nim_row(s, "S1")
    assert (r["ch"], r["nim_ch"], r["label"], r["nim_label"]) == (1, 3, "S1", "S1L")
    assert r["paired"] == r["tot_words"] == r["nim_words"] == N_EV
    assert r["pair_eff"] == r["purity"] == 1.0 and r["nim_only_frac"] == 0.0
    assert r["median_dt_ns"] == 1.0 and r["dt_entries"] == N_EV
    assert r["lag"]["ok"] == 1 and r["lag"]["last_ns"] == 1
    assert not codes(s) & {"nim_missing", "nim_pairing", "nim_offset", "nim_lag"}
    clk.t += 2                                       # the second is complete
    t = p.trend()
    assert t["nim_counters"] == ["S1", "S2", "S3", "S4", "S5"]
    assert [r["nim_eff"] for r in t["rows"] if r["frames"]] == [[1.0] * 5]
    assert all(r["nim_eff"] is None for r in t["rows"] if not r["frames"])


def test_nim_missing():
    events = [ev(drop=(12,)) for _ in range(N_EV)]              # no S5L at all
    p = plugin()
    feed(p, frame(events))
    s = p.summary(True)
    miss = [f for f in s["flags"] if f["code"] == "nim_missing"]
    assert len(miss) == 1 and "S5L (ch 12)" in miss[0]["text"]
    assert "nim_pairing" not in codes(s), "one flag for a missing copy"
    # Below min hits: not judged.
    q = plugin()
    feed(q, frame(events[:50]))
    assert "nim_missing" not in codes(q.summary(True))


def test_nim_pairing_and_offset_flags():
    # S2L 8 ns late (window 20: still paired) -> nim_offset; S4L missing in 40 %.
    events = [ev(drop=(11,) if k % 5 < 2 else (), ch9=[(9, 10)]) for k in range(N_EV)]
    p = plugin()
    feed(p, frame(events))
    s = p.summary(True)
    off = [f for f in s["flags"] if f["code"] == "nim_offset"]
    assert len(off) == 1 and "S2L" in off[0]["text"] and "offset ns[1] = 8" in off[0]["text"]
    pa = [f for f in s["flags"] if f["code"] == "nim_pairing"]
    assert len(pa) == 1 and pa[0]["severity"] == "warn" and "S4 (ch 4)" in pa[0]["text"]
    assert nim_row(s, "S4")["pair_eff"] == pytest.approx(0.6, abs=1e-3)
    # The offset applied: the flag goes.
    q = plugin({"NIM": {"offset ns": [0, 8, 0, 0, 0]}})
    feed(q, frame(events))
    sq = q.summary(True)
    assert "nim_offset" not in codes(sq) and nim_row(sq, "S2")["median_dt_ns"] == 0.0


def test_the_time_comparing_nim_flags_wait_for_the_time_base():
    """S3: nim_pairing / nim_offset / nim_lag compare times, so like the
    efficiency flags they need the shift check's "ok"; nim_missing counts hits
    and is judged anyway."""
    events = [ev(drop=(11, 12) if k % 5 < 2 else (12,), ch9=[(9, 10)]) for k in range(N_EV)]
    lagged = frame(events, fine_lag={10: LAG})
    p = plugin(ONE_VOTE, checked=False)            # 300 S1 words: shift "unchecked"
    feed(p, lagged)
    s = p.summary(True)
    assert s["shift"]["verdict"] != "ok"
    assert codes(s) & {"nim_missing", "nim_pairing", "nim_offset", "nim_lag"} == {"nim_missing"}
    q = plugin(ONE_VOTE)
    feed(q, lagged)
    assert codes(q.summary(True)) >= {"nim_missing", "nim_pairing", "nim_offset", "nim_lag"}


def test_a_tot_channel_with_a_timestamp_fault_withholds_the_pairing():
    """S3: S4's TOT words fine/coarse inconsistent (a known timestamp fault):
    its pair efficiency is withheld with the reason, and the mismatch flag (not
    a second nim_pairing alarm blaming the NIM threshold) says the pairing
    collapsed; nim_offset (which compares times) is not judged."""
    events = [ev() for _ in range(N_EV)]
    p = plugin()
    feed(p, frame(events, fine_lag={4: LAG}))
    s = p.summary(True)
    assert [f["ch"] for f in s["timestamp_faults"]["counters"]] == [4]
    r = nim_row(s, "S4")
    assert r["pair_eff"] is None and "timestamp fault" in r["pair_eff_reason"]
    assert not [f for f in s["flags"] if f["code"] in ("nim_pairing", "nim_offset")]
    mm = [f for f in s["flags"] if f["code"] == "mismatch" and "(ch 4)" in f["text"]]
    assert mm and "pairing (S4 + S4L) collapses" in mm[0]["text"]


def test_the_nim_counts_leave_the_summary_after_its_window():
    clk = _Clock()
    p = plugin(clock=clk)
    feed(p, frame([ev() for _ in range(N_EV)]))
    assert nim_row(p.summary(True), "S2")["paired"] == N_EV
    clk.t += p.cfg.check["summary window s"] + 2
    r = nim_row(p.summary(True), "S2")
    assert (r["paired"], r["nim_words"], r["frames"]) == (0, 0, 0)
    assert r["pair_eff"] is None and r["purity"] is None and r["lag"]["voted"] == 0
    assert r["median_dt_ns"] == 1.0, "the median is since the run start, not the window"


def test_the_summary_keys_the_pages_read():
    """The key sets of sma::summary's NIM parts and of the trend, pinned for the pages."""
    clk = _Clock()
    p = plugin(clock=clk)
    feed(p, frame([ev() for _ in range(N_EV)]))
    s = p.summary(True)
    assert set(s["nim"]) == {"active", "merge", "merging", "merge_when_lagged",
                             "lag_vote_every", "pair_window_ns", "time_source", "counters"}
    assert set(s["roles"]) == {"s1", "counters", "rf", "current", "delayed", "nim"}
    assert s["nim_merge"] is False and s["nim_lag_held"] == []
    assert set(s["nim"]["counters"][0]) == {
        "counter", "k", "ch", "nim_ch", "label", "nim_label", "tot_hits", "nim_hits",
        "tot_words", "nim_words", "paired", "tot_only", "nim_only", "echo", "lag_held",
        "shadow", "multi", "frames", "eff_paired", "eff_tot_only", "pair_eff", "purity",
        "nim_only_frac", "median_dt_ns", "dt_entries", "offset_ns", "echo_rule", "lag"}
    assert set(s["nim"]["counters"][0]["lag"]) == {
        "none", "ambiguous", "ok", "faulted", "skipped", "voted", "faulted_frac",
        "state_ok", "state_faulted", "state_faulted_frac", "state", "epoch_votes",
        "epoch_faulted_votes", "nominal_ns", "last_ns", "last_state", "last_age_s"}
    clk.t += 2
    t = p.trend()
    assert "nim_counters" in t and all("nim_eff" in r for r in t["rows"])


def _seed_meta(p, **kw):
    d = framing.decode_sma_frame(p.frame_blob("seeded", **kw))
    t0 = d["meta"]["t0_ns"]
    return {round((t0 + m["t_rel"] - T0) / SPACING, 6): m for m in d["meta"]["seeds"]}


def test_the_rf_phase_of_a_nim_only_s1_seed_uses_its_aligned_time():
    """S6: S1L 6 ns after the event with S1's offset 5: the merged S1 hit of an
    event without its S1 TOT word sits at t' = +1 ns, so its RF phase (last
    pulse - S1) is the TOT seeds' minus 1 ns -- not minus 6 (the raw time)."""
    events = [ev(ch3=[(6, 10)]) for _ in range(N_EV)]
    k = N_EV - 3
    events[k] = ev(drop=(1,), ch3=[(6, 10)])
    p = plugin(with_(MERGE, {"NIM": {"offset ns": [5, 0, 0, 0, 0]}}))
    feed(p, frame(events))
    for kw in ({}, {"seed": "s1", "pattern": {"2": "present"}}):
        seeds = _seed_meta(p, **kw)
        nim_only = seeds[round(k + 1 / SPACING, 6)]
        tot = [m for x, m in seeds.items() if x == int(x)]
        assert tot, kw
        assert nim_only["s1_tot"] == 1
        assert nim_only["rf_n"] == tot[0]["rf_n"] == 2, kw
        assert nim_only["rf_phase"] == pytest.approx(tot[0]["rf_phase"] - 1), kw


def test_an_any_seed_can_start_on_a_nim_only_hit():
    """S6: an event with only the NIM copies: its "any counter" seed is S1's
    merged hit, carried by the S1L word, with the substituted ToT."""
    events = [ev() for _ in range(N_EV)]
    k = N_EV - 3
    events[k] = ev(drop=TOT_1015)
    p = plugin(MERGE)
    fr = feed(p, frame(events)).fr
    seeds = _seed_meta(p, seed="any")
    m = seeds[round(k + 1 / SPACING, 6)]
    assert m["seed_ch"] == 3 and m["seed_tot"] == 1
    i = int(np.flatnonzero((fr.s_ch == 3) & (fr.s_t == T0 + k * SPACING + 1))[0])
    assert m["seed_word"] == int(fr.word_index[fr.order[i]])


def test_time_source_nim_gives_paired_hits_the_aligned_nim_time():
    p = plugin(with_(MERGE, {"NIM": {"time source": "nim", "offset ns": [3, 0, 0, 0, 0]}}))
    fr = feed(p, frame([ev() for _ in range(20)])).fr
    want = T0 + np.arange(20) * SPACING + 1 - 3              # S1L at +1, offset 3
    assert fr.counter_hits[1].t.tolist() == want.tolist()
    assert np.all(fr.s_ch[fr.counter_hits[1].idx] == 1), "the carrier stays the TOT word"
    q = plugin(with_(MERGE, {"NIM": {"offset ns": [3, 0, 0, 0, 0]}}))
    fq = feed(q, frame([ev() for _ in range(20)])).fr
    assert fq.counter_hits[1].t.tolist() == (want + 2).tolist(), "time source tot"


def test_nim_only_s1_hits_stay_out_of_the_rf_phase_vs_tot_plot():
    """N1: a NIM-only S1 hit has no ToT of its own (it is given 1)."""
    events = [ev(drop=(1,)) if k % 4 == 0 else ev() for k in range(N_EV)]
    p = plugin(MERGE)
    feed(p, frame(events))
    hh = h(p, "rf_phase_vs_s1_tot")
    assert hh.entries == N_EV - N_EV // 4
    assert h(p, "rf_phase_s1").entries == N_EV, "the phase itself has every merged S1 hit"
    assert hh.counts[:, 1 + 1].sum() == 0, "no ToT = 1 stripe"


def test_old_layout_frames_have_no_nim():
    p = P.SmaPlugin(HistStore(), settings=old_layout(), clock=_Clock())
    assert not [n for n in p.store.names() if "nim" in n]
    s = p.summary(True)
    assert s["nim"]["counters"] == [] and s["nim_merge"] is False


# --- the offline tool ------------------------------------------------------------------

def _frames_1015(k=4):
    out = []
    for i in range(k):
        events = [ev(drop=(2,)) if j % 3 == 0 else ev() for j in range(N_EV)]
        w = frame(events)
        # each frame 10 ms later on the board
        t_shift = np.uint64(i * 10_000_000)
        dec = W.decode(w)
        t = dec["time"].astype(np.int64) + int(t_shift)
        c, f = W.fields_of(t)
        out.append(W.encode(dec["ch"], dec["tot"], c, f))
    return out


@pytest.mark.parametrize("switch, merged_on", [([], False), (["--merge"], True),
                                               (["--no-merge"], False)])
def test_the_offline_tool_and_merge(tmp_path, capsys, switch, merged_on):
    path = write_file(tmp_path / "run01015_00001.mid", _frames_1015(), run=1015)
    out = tmp_path / "out"
    rc = S.main([str(path), "--out", str(out), "--no-png"] + switch)
    assert rc == S.EXIT_OK
    summary = json.loads((out / "summary.json").read_text())
    no_merge = not merged_on
    assert summary["nim_merge"] is merged_on
    with np.load(out / "hists.npz") as z:
        merged, tot_only = z["sma/s1_coinc"], z["sma/s1_coinc_tot"]
        assert "sma/nim_dt_S2" in z.files
    if no_merge:
        assert merged.tolist() == tot_only.tolist()
    else:
        assert merged[2] > tot_only[2]
    text = capsys.readouterr().out
    assert f"TOT + NIM: merge {'off' if no_merge else 'on'}" in text
    assert "  S2 + S2L (ch 2+9): pair eff 100.0%" in text


def test_build_settings_takes_the_nim_keys():
    s = S.build_settings({"NIM": {"offset ns": [0, 1, 2, 3, 4], "merge": True}})
    assert s["NIM"]["offset ns"] == [0, 1, 2, 3, 4] and s["NIM"]["merge"] is True
    assert S.build_settings()["NIM"]["merge"] is False
    assert S.build_settings(merge=True)["NIM"]["merge"] is True
    assert S.build_settings({"NIM": {"merge": True}}, merge=False)["NIM"]["merge"] is False
    with pytest.raises(KeyError, match="NIM/ofset ns"):
        S.build_settings({"NIM": {"ofset ns": [0] * 5}})


def test_the_nim_page(tmp_path):
    pytest.importorskip("matplotlib")
    p = plugin()
    feed(p, frame([ev() for _ in range(N_EV)]))
    s = S.offline_summary(p)
    fig = S.nim_figure(s, p.store, "test")
    assert fig is not None
    fig.savefig(tmp_path / "nim.png")
    assert (tmp_path / "nim.png").stat().st_size > 0
    q = plugin({"NIM": {"channels": [-1]}})
    assert S.nim_figure(S.offline_summary(q), q.store, "test") is None
