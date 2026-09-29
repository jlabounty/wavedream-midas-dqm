"""The event display's seed choice, odd-event filters, walk-back and staleness.

``sma::frame {"view": "seeded", "seed": ..., "filters": [...]}``: seeds other
than S1 and seeds with an oddity in their window, computed at request time
(``sma_words.select_seeds_by``) from the good frames the plugin keeps (the seed
ring). The default -- S1, no filter -- must stay exactly what it was.

Synthetic frames are built event by event with ``sma_words.encode``, so which
seed has which oddity is known.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from test_sma_plugin import R682, R1008, _Clock, _dispatch, _feed, _plugin, synth_frame

from mdqm.dqm import framing
from mdqm.plugins import sma as P
from mdqm.plugins import sma_words as W

DATA = Path(__file__).resolve().parent / "data"
T0 = 10**12
SPACING = 10_000
#: The healthy event: S1..S5 a few ns apart, two RF pulses in the S1 gate.
NORMAL = {1: [0], 2: [2], 3: [3], 4: [4], 5: [5], 6: [40, 60]}


def build(events, shift=14):
    """Words of a frame with one event per entry, SPACING ns apart.

    An event is ``{ch: [dt, ...]}`` (ToT 20), plus optional ``"tot": {ch: code}``
    (every hit of that channel gets the code) and ``"bad_coarse": [ch...]``
    (those hits' coarse field is two ticks off: fine/coarse inconsistent, but
    the time, which comes from the fine field, is still right).
    """
    t, ch, tot, bad = [], [], [], []
    for k, ev in enumerate(events):
        base = T0 + k * SPACING
        for c, dts in ev.items():
            if not isinstance(c, int):
                continue
            for dt in dts:
                t.append(base + dt)
                ch.append(c)
                tot.append(ev.get("tot", {}).get(c, 20))
                bad.append(c in ev.get("bad_coarse", ()))
    t = np.asarray(t, dtype=np.int64)
    o = np.argsort(t, kind="stable")
    t, ch, tot, bad = t[o], np.asarray(ch)[o], np.asarray(tot)[o], np.asarray(bad)[o]
    coarse, fine = W.fields_of(t, shift)
    coarse = np.where(bad, coarse + 2, coarse).astype(np.uint32)
    return W.encode(ch, tot, coarse, fine)


def ev(**kw):
    e = {c: list(v) for c, v in NORMAL.items()}
    for c in kw.pop("drop", ()):
        e.pop(c)
    e.update(kw)
    return e


def blob_of(p, **kw):
    b = p.frame_blob("seeded", **kw)
    return None if b is None else framing.decode_sma_frame(b)


def seed_times(d):
    """Seed times as event numbers (T0 + k * SPACING)."""
    t0 = d["meta"]["t0_ns"]
    return [(t0 + s["t_rel"] - T0) / SPACING for s in d["meta"]["seeds"]]


# --- the default is unchanged ------------------------------------------------------

def _default_payloads():
    """``(name, what, payload)`` of the default views over the golden cases."""
    clk = _Clock()
    p = _plugin(clock=clk)
    cases = [("r682_2", R682[2]), ("r682_3", R682[3]), ("r682_4", R682[4]), ("r682_5", R682[5]),
             ("r1008", R1008), ("synth_a", synth_frame(10**12, seed=1)),
             ("synth_sparse", synth_frame(10**12 + 10**7, n=3, seed=2))]
    for k, (name, w) in enumerate(cases):
        clk.t += 0.3
        _feed(p, w, serial=k, run=682)
        for words in (None, False, True):
            for args in ({}, {"seed": "s1", "filters": []}, {"seed": "S1", "filters": None},
                         {"seed": "s1", "mupix": "any"}, {"pattern": {}},
                         {"pattern": {"1": "any", "5": "any"}}):
                b = p.frame_blob("seeded", words=words, **args)
                meta = framing.decode_sma_frame(b)["meta"]
                if name == "synth_sparse":
                    # No seeds in the newest frame: the old frame is shown, as
                    # before, and now says so; without that entry it is the old payload.
                    assert meta["stale_view"]["newest_seq"] == p._last.seq
                    b = framing.smaf_drop_meta(b, ["stale_view"])
                else:
                    assert "stale_view" not in meta and "select" not in meta
                yield name, f"seeded/words={words}", b
        yield name, "raster", p.frame_blob("raster", drop=[7], max_hits=60000)


def test_the_default_seeded_view_is_byte_identical_to_before():
    """Two sets of hashes. ``hashes``: taken from the code before the seed choice
    and before MuPix existed; the default payloads with their MuPix part taken
    off (the ``mupix`` meta key and the pixel block) must still be exactly that.
    ``hashes_mupix``: the payloads as sent now, pixel block included, recorded
    when the MuPix lanes were added (tests/data/sma_seeded_default_golden.json)."""
    gold = json.loads((DATA / "sma_seeded_default_golden.json").read_text())
    golden, golden_mp = gold["hashes"], gold["hashes_mupix"]
    n_checked = 0
    for name, what, b in _default_payloads():
        d = framing.decode_sma_frame(b)
        assert d["pixels"] is not None and "mupix" in d["meta"], (name, what)
        bare = framing.smaf_drop_pixels(framing.smaf_drop_meta(b, ["mupix"]))
        assert hashlib.sha256(bare).hexdigest() == golden[f"{name}/{what}"], (name, what)
        assert hashlib.sha256(b).hexdigest() == golden_mp[f"{name}/{what}"], (name, what)
        n_checked += 1
    assert n_checked == 7 * (3 * 6 + 1)


def test_update_then_drop_meta_gives_the_payload_back():
    p = _plugin()
    _feed(p, R682[4])
    b = p.frame_blob("seeded")
    u = framing.smaf_update_meta(b, {"stale_view": {"age_s": 1.5, "reason": "x" * 13}})
    d = framing.decode_sma_frame(u)
    assert d["meta"]["stale_view"]["age_s"] == 1.5
    assert d["arrays_offset"] % 8 == 0
    ref = framing.decode_sma_frame(b)
    for k in ("t_rel_ns", "ch", "tot", "hit_flags", "raw_words", "word_index"):
        assert np.array_equal(d[k], ref[k])
    assert framing.smaf_drop_meta(u, ["stale_view"]) == b


@pytest.mark.parametrize("words", [R682[4], R682[5], R1008])
@pytest.mark.parametrize("max_s1", [None, 100])
def test_s1_without_filters_is_the_analysis_seeds(words, max_s1):
    fr = W.prepare_frame(words)
    cuts = W.Cuts(max_s1=max_s1)
    an = W.analyse_frame(fr, W.Roles(), cuts)
    sel = W.select_seeds_by(fr, W.Roles(), cuts, "s1", ())
    assert sel.idx.size == an.seeds.size > 0
    assert np.array_equal(sel.t, an.t_s1[an.seeds])
    assert np.array_equal(sel.pattern, an.pattern[an.seeds])
    assert np.array_equal(sel.rf_n, an.rf_n[an.seeds])
    assert np.array_equal(sel.rf_valid, an.rf_valid[an.seeds])
    assert np.array_equal(sel.rf_vetoed, an.rf_vetoed[an.seeds])
    assert np.allclose(sel.rf_phase, an.rf_phase[an.seeds], equal_nan=True)
    assert sel.has_s1.all() and (sel.s1_dt == 0).all()
    assert sel.n_candidates >= sel.n_matching == sel.n_examined


# --- seed modes ------------------------------------------------------------------------

def test_parse_seed_and_filters():
    assert W.parse_seed(None) == W.parse_seed("S1") == "s1"
    assert W.parse_seed("any") == "any"
    assert W.parse_seed("ch2") == W.parse_seed(2) == W.parse_seed("CH2") == "ch2"
    for bad in ("ch16", "S2", "", "chx", True, -1):
        with pytest.raises(ValueError):
            W.parse_seed(bad)
    assert W.parse_filters(["rf", "tot", "rf"]) == ("tot", "rf")
    assert W.parse_filters(None) == () == W.parse_filters([])
    with pytest.raises(ValueError):
        W.parse_filters(["odd"])


def test_any_seeds_are_the_first_hit_of_each_counter_cluster():
    # S2 at 0, S3 at +30 and S4 at +60 chain within the 50 ns window: one
    # cluster, seeded by the S2 hit. S5 at +1000 is a cluster of its own, and
    # the RF and ch 8 are no counters.
    e = {2: [0], 3: [30], 4: [60], 5: [1000], 6: [5, 20], 8: [10]}
    fr = W.prepare_frame(build([e]))
    cand = W.seed_candidates(fr, W.Roles(), W.Cuts(), "any")
    assert [int(x) for x in fr.s_t[cand] - T0] == [0, 1000]
    assert [int(x) for x in fr.s_ch[cand]] == [2, 5]
    # A gap of 51 ns splits the chain.
    fr = W.prepare_frame(build([{2: [0], 3: [51]}]))
    assert W.seed_candidates(fr, W.Roles(), W.Cuts(), "any").size == 2


def test_a_frame_without_s1_still_has_any_and_ch2_seeds_with_rf_na():
    p = _plugin()
    _feed(p, build([ev(drop=[1]) for _ in range(12)]))
    assert p._last.cls == "good" and p._last.an.seeds.size == 0
    d = blob_of(p)
    assert d["meta"]["seeds"] == [], "the default S1 view has none"
    for seed in ("any", "ch2"):
        d = blob_of(p, seed=seed)
        seeds = d["meta"]["seeds"]
        assert len(seeds) == 4, seed
        assert seed_times(d) == [k + 2 / SPACING for k in (7, 8, 9, 10)], "S2 is 2 ns in"
        for s in seeds:
            assert s["seed_ch"] == 2
            assert s["rf_na"] is True and s["rf_n"] is None and s["rf_phase"] is None
            assert s["s1_word"] is None and s["s1_dt"] is None and s["s1_tot"] is None
            assert s["pattern"] == 0b11110, "S2..S5 but not S1"
            assert s["odd"] == ["incomplete"]
            lo, hi = s["hits"]
            assert d["ch"][lo:hi].tolist().count(2) >= 1
        sel = d["meta"]["select"]
        assert sel["seed"] == seed and sel["filters"] == []
        assert sel["candidates"] == sel["matching"] == 10
        assert d["meta"]["search"]["frames_searched"] == 1
        assert "stale_view" not in d["meta"]


def test_a_non_s1_seed_borrows_the_rf_of_the_nearest_s1_hit():
    # S2 fires 20 ns before S1 (and a second S1 60 ns after S2 is further).
    e = {1: [20], 2: [0], 3: [3], 4: [4], 5: [5], 6: [60, 80]}
    frame = W.prepare_frame(build([e] * 10))
    sel = W.select_seeds_by(frame, mode="ch2")
    ref = W.select_seeds_by(frame, mode="s1")
    assert sel.has_s1.all() and (sel.s1_dt == 20).all()
    assert np.array_equal(sel.rf_phase, ref.rf_phase)
    assert (sel.rf_phase == 60).all() and sel.rf_valid.all()
    assert (sel.pattern == 0b11111).all()
    assert np.array_equal(sel.t + 20, ref.t)


def test_a_delayed_channel_can_seed():
    events = [ev() for _ in range(10)]
    events[5][8] = [700]
    p = _plugin()
    _feed(p, build(events))
    d = blob_of(p, seed="ch8")
    assert seed_times(d) == [5.07]
    s = d["meta"]["seeds"][0]
    assert s["seed_ch"] == 8 and s["pattern"] == 0 and s["rf_na"]
    assert d["meta"]["select"]["seed_label"] == "ch08 (ch 8)"


# --- filters -----------------------------------------------------------------------------

def _odd_frame():
    """Events 2..9 are candidates; 3 incomplete, 4 mismatch, 5 ToT, 6 RF invalid, 7 vetoed."""
    events = [ev() for _ in range(12)]
    events[3] = ev(drop=[4])
    events[4]["bad_coarse"] = [3]
    events[5]["tot"] = {5: 254}
    events[6][6] = [40]                     # one RF pulse: not a valid gate
    events[7][1] = [0, 30]                  # a second S1 inside the first's gate: vetoed
    return build(events)


@pytest.mark.parametrize("filters, want", [
    (["incomplete"], [3]),
    (["mismatch"], [4]),
    (["tot"], [5]),
    (["rf"], [6, 7]),
    (["incomplete", "tot"], [3, 5]),
    (["incomplete", "mismatch", "tot", "rf"], [4, 5, 6, 7]),     # the latest 4 of 3..7
])
def test_each_filter_and_their_or(filters, want):
    p = _plugin()
    _feed(p, _odd_frame())
    d = blob_of(p, filters=filters)
    # Event 7's second S1 (+30 ns) is a candidate too, but nothing is odd about it.
    assert seed_times(d) == want
    names = {"incomplete": 3, "mismatch": 4, "tot": 5}
    for s, k in zip(d["meta"]["seeds"], seed_times(d), strict=True):
        assert set(filters) & set(s["odd"]), (k, s["odd"])
        for f, n in names.items():
            assert (f in s["odd"]) == (k == n), (k, f, s["odd"])
    sel = d["meta"]["select"]
    assert sel["filters"] == [f for f in W.FILTERS if f in filters]
    assert sel["matching"] >= len(want) and sel["candidates"] >= sel["matching"]


def test_rf_filter_ignores_seeds_without_s1():
    p = _plugin()
    _feed(p, build([ev(drop=[1]) for _ in range(12)]))
    d = blob_of(p, seed="any", filters=["rf"])
    assert d["meta"]["seeds"] == [] and d["meta"]["search"]["no_match"]


def test_mismatch_and_tot_hits_are_flagged_in_the_payload():
    p = _plugin()
    _feed(p, _odd_frame())
    d = blob_of(p, filters=["mismatch", "tot"])
    flags = d["hit_flags"]
    assert (flags & framing.HIT_MISMATCH).any() and (flags & framing.HIT_TOT_CORRUPT).any()
    assert (flags & framing.HIT_IN_SEED).all()


def test_the_candidate_cap_is_reported():
    frame = W.prepare_frame(_odd_frame())
    sel = W.select_seeds_by(frame, filters=["incomplete"], max_candidates=3)
    assert sel.capped and sel.n_examined == 3 and sel.n_candidates > 3
    assert sel.idx.size == 0, "event 3 is not among the latest three"


# --- walk-back, no match, staleness --------------------------------------------------------

def _feed_many(p, clk, frames, dt=1.0):
    out = []
    for k, w in enumerate(frames):
        clk.t += dt
        out.append(_feed(p, w, serial=len(out) + p.frames))
    return out


def test_filters_walk_back_to_the_last_frame_with_a_match():
    clk = _Clock()
    p = _plugin(clock=clk)
    plain = build([ev() for _ in range(12)])
    first = _feed_many(p, clk, [_odd_frame(), plain, plain])[0]
    clk.t += 0.5
    d = blob_of(p, filters=["tot"])
    assert d["frame_seq"] == first.seq
    assert seed_times(d) == [5]
    sv = d["meta"]["stale_view"]
    assert sv["shown_seq"] == first.seq and sv["newest_seq"] == p._last.seq
    assert sv["good_since"] == 2 and sv["age_s"] == 2.5
    assert "ToT \u2265 250" in sv["reason"]
    assert d["meta"]["search"]["frames_searched"] == 3
    # The newest frame has S1 seeds: no stale entry for the default view.
    assert "stale_view" not in blob_of(p)["meta"]


def test_no_match_is_an_explicit_state_on_the_newest_frame():
    clk = _Clock()
    p = _plugin(clock=clk)
    plain = build([ev() for _ in range(12)])
    frames = _feed_many(p, clk, [plain] * 3)
    clk.t += 1
    d = blob_of(p, filters=["tot"])
    assert d["frame_seq"] == frames[-1].seq and d["meta"]["seeds"] == []
    nm = d["meta"]["search"]["no_match"]
    assert nm["frames"] == 3 and nm["oldest_seq"] == frames[0].seq and nm["span_s"] == 3.0
    assert "no S1 (ch 1) seeds with ToT \u2265 250 in the last 3 good frames" == nm["text"]
    assert "stale_view" not in d["meta"]


def test_a_seq_request_searches_that_frame_only():
    clk = _Clock()
    p = _plugin(clock=clk)
    first, second = _feed_many(p, clk, [_odd_frame(), build([ev() for _ in range(12)])])
    d = blob_of(p, filters=["tot"], seq=first.seq)
    assert d["frame_seq"] == first.seq and seed_times(d) == [5]
    assert "stale_view" not in d["meta"]
    d = blob_of(p, filters=["tot"], seq=second.seq)
    assert d["frame_seq"] == second.seq and d["meta"]["search"]["no_match"]["frames"] == 1
    assert p.frame_blob("seeded", filters=["tot"], seq=10**6) is None


def test_the_default_view_says_how_stale_it_is():
    clk = _Clock()
    p = _plugin(clock=clk)
    seeded = _feed_many(p, clk, [build([ev() for _ in range(12)])])[0]
    _feed_many(p, clk, [build([ev(drop=[1]) for _ in range(12)])] * 3, dt=2.0)
    d = blob_of(p)
    assert d["frame_seq"] == seeded.seq
    sv = d["meta"]["stale_view"]
    assert sv == {"shown_seq": seeded.seq, "newest_seq": p._last.seq, "age_s": 6.0,
                  "good_since": 3, "reason": "no S1 hits"}
    # 'any' finds seeds in the newest frame, so it is not stale.
    d = blob_of(p, seed="any")
    assert d["frame_seq"] == p._last.seq and "stale_view" not in d["meta"]


def test_the_reason_names_s1_hits_without_a_complete_window():
    p = _plugin()
    _feed(p, build([ev() for _ in range(12)]))
    _feed(p, synth_frame(10**13, n=3, seed=2), serial=1)
    assert "none of 3 S1 hits has a complete window" == blob_of(p)["meta"]["stale_view"]["reason"]


def test_the_seed_ring_is_bounded_by_frames_and_bytes():
    clk = _Clock()
    p = _plugin({"Sampling": {"seed ring frames": 3}}, clock=clk)
    plain = build([ev() for _ in range(12)])
    frames = _feed_many(p, clk, [plain] * 5)
    st = p.status()["seed_ring"]
    assert st["frames"] == 3 and st["oldest_seq"] == frames[2].seq
    assert st["bytes"] == sum(P.snapshot_bytes(s) for s in frames[2:]) > 0
    assert st["limit_frames"] == 3
    # A byte cap below one frame keeps just the newest.
    p = _plugin({"Sampling": {"seed ring MB": 0.001}}, clock=clk)
    frames = _feed_many(p, clk, [plain] * 3)
    assert [s.seq for s in p._ring] == [frames[-1].seq]
    # Only good frames enter it.
    _feed(p, R682[0], serial=99)
    assert p._last.cls == "stale" and [s.seq for s in p._ring] == [frames[-1].seq]
    # Out-of-range settings fall back and are reported.
    p = _plugin({"Sampling": {"seed ring frames": 1000}})
    assert p.cfg.seed_ring_frames == 8 and p.cfg.errors


def test_selections_are_cached_and_purged_with_their_frames():
    clk = _Clock()
    p = _plugin({"Sampling": {"seed ring frames": 2}}, clock=clk)
    plain = build([ev() for _ in range(12)])
    first = _feed_many(p, clk, [plain])[0]
    a = p.frame_blob("seeded", seed="any")
    sel = p._sel_cache[(first.seq, "any", (), "any", (), ())]
    assert p.frame_blob("seeded", seed="any") == a
    assert p._sel_cache[(first.seq, "any", (), "any", (), ())] is sel
    _feed_many(p, clk, [plain] * 2)
    assert not any(k[0] == first.seq for k in p._sel_cache), "gone with its frame"


# --- the summary flag ---------------------------------------------------------------------------

def test_no_s1_seeds_for_ten_seconds_while_frames_arrive_is_flagged():
    clk = _Clock()
    p = _plugin(clock=clk)
    _feed_many(p, clk, [build([ev() for _ in range(12)])])
    no_s1 = build([ev(drop=[1]) for _ in range(12)])
    _feed_many(p, clk, [no_s1] * 9)
    assert not [f for f in p.summary()["flags"] if f["code"] == "no_seeds"]
    _feed_many(p, clk, [no_s1] * 3)
    flags = [f for f in p.summary()["flags"] if f["code"] == "no_seeds"]
    assert len(flags) == 1 and flags[0]["severity"] == "warn"
    assert "no S1 hits" in flags[0]["text"] and "any counter" in flags[0]["text"]
    # Frames stop: no_frames is the story then, not the seeds.
    clk.t += 30
    assert not [f for f in p.summary()["flags"] if f["code"] == "no_seeds"]
    # S1 back: gone.
    _feed_many(p, clk, [build([ev() for _ in range(12)])])
    assert not [f for f in p.summary()["flags"] if f["code"] == "no_seeds"]


# --- the command -------------------------------------------------------------------------------

def test_the_command_takes_seed_and_filters():
    p = _plugin()
    _feed(p, _odd_frame())
    tag, body = _dispatch(p, "sma::frame", json.dumps({"view": "seeded", "seed": "any",
                                                      "filters": ["tot"]}))
    assert tag == framing.TAG_SMAF
    d = framing.decode_sma_frame(body)
    assert d["meta"]["select"]["seed"] == "any" and seed_times(d) == [5]
    tag, body = _dispatch(p, "sma::frame", json.dumps({"view": "seeded", "seed": "S7"}))
    assert tag == framing.TAG_ERROR and b"seed" in body
    tag, body = _dispatch(p, "sma::frame", json.dumps({"view": "seeded", "filters": ["odd"]}))
    assert tag == framing.TAG_ERROR and b"filter" in body


def test_the_raster_ignores_the_seed_choice():
    p = _plugin()
    _feed(p, R682[4])
    assert p.frame_blob("raster", seed="any", filters=["tot"]) == p.frame_blob("raster")


def test_a_search_out_of_cpu_time_says_so_and_the_next_poll_goes_on(monkeypatch):
    clk = _Clock()
    p = _plugin(clock=clk)
    plain = build([ev() for _ in range(12)])
    first = _feed_many(p, clk, [_odd_frame(), plain, plain])[0]
    monkeypatch.setattr(P, "SEARCH_BUDGET_S", -1.0)    # no time for a second new selection
    d = blob_of(p, filters=["tot"])
    nm = d["meta"]["search"]["no_match"]
    assert nm["partial"] and nm["frames"] == 1 and "(of 3 held; searching on)" in nm["text"]
    assert d["meta"]["search"]["frames_searched"] == 1
    d = blob_of(p, filters=["tot"])
    assert d["meta"]["search"]["frames_searched"] == 2, "the cached one is free, one more is new"
    d = blob_of(p, filters=["tot"])
    assert d["frame_seq"] == first.seq and seed_times(d) == [5]
    assert d["meta"]["stale_view"]["good_since"] == 2


# --- "incomplete pattern" and counters with a known timestamp fault -------------------------

#: Few words suffice for a shift verdict and a per-channel judgement here.
QUICK = {"Self check": {"shift min words": 10, "min hits": 5}}


def _faulty_frame(bad=(5,), n=24, missing=None):
    """n events; the channels in `bad` have fine/coarse inconsistent words in every
    event (their times are still right). `missing`: {event: [channels dropped]}."""
    missing = {3: [5], 6: [3], 9: [5], 12: [5]} if missing is None else missing
    events = []
    for k in range(n):
        e = ev(drop=missing.get(k, ()))
        e["bad_coarse"] = [c for c in bad if c in e]
        events.append(e)
    return build(events)


def _faulty_plugin(bad=(5,), settings=QUICK, **kw):
    clk = _Clock()
    p = _plugin(settings, clock=clk)
    _feed_many(p, clk, [_faulty_frame(bad, **kw)])
    clk.t += 0.5
    return p


def test_the_auto_ignore_is_exactly_the_counters_the_summary_flags():
    # S5 and the RF (a flag channel but no counter) are broken; S3 is not.
    p = _faulty_plugin(bad=(5, 6))
    s = p.summary()
    assert s["shift"]["verdict"] == "ok"
    flagged = {int(f["text"].split("(ch ")[1].split(")")[0])
               for f in s["flags"] if f["code"] == "mismatch"}
    assert flagged == {5, 6}
    tf = s["timestamp_faults"]
    assert tf["judged"] and [f["ch"] for f in tf["counters"]] == [5], "counters only"
    assert p.timestamp_faults() == tf
    d = blob_of(p, filters=["incomplete"], seed="any")
    inc = d["meta"]["incomplete"]
    assert [f["counter"] for f in inc["ignored"]] == [5] and inc["judged"]
    assert inc["label"] == "incomplete pattern (ignoring S5: timestamp fault 100 %)"
    # Only event 6 (no S3) is incomplete now; 3, 9 and 12 lack only S5.
    assert seed_times(d) == [6]
    assert d["meta"]["seeds"][0]["odd"] == ["incomplete", "mismatch"]
    assert d["meta"]["select"]["filter_labels"] == ["incomplete pattern"]
    # The badges agree: a seed missing only S5 has no "incomplete" in any view.
    for s_ in blob_of(p, seed="any", filters=["mismatch"])["meta"]["seeds"]:
        assert "incomplete" not in s_["odd"], s_
    # The same frame, nothing broken: S5's absences count again.
    q = _faulty_plugin(bad=())
    d = blob_of(q, filters=["incomplete"])
    assert seed_times(d) == [3, 6, 9, 12]
    assert d["meta"]["incomplete"] == {"judged": True, "verdict": "ok", "ignored": [],
                                       "note": "", "label": "incomplete pattern"}


def test_a_small_mismatch_below_the_warn_level_is_not_ignored():
    # S5 bad in 1 of 24 events (4 %) is under the 5 % warn level.
    events = [ev() for _ in range(24)]
    events[4]["bad_coarse"] = [5]
    events[9] = ev(drop=[5])
    clk = _Clock()
    p = _plugin(QUICK, clock=clk)
    _feed_many(p, clk, [build(events)])
    assert p.timestamp_faults()["counters"] == []
    assert seed_times(blob_of(p, filters=["incomplete"])) == [9]


def test_without_an_ok_shift_verdict_nothing_is_ignored_and_it_says_so():
    # Default settings: 24 S1 words are far from the 1000 the verdict needs.
    p = _faulty_plugin(settings=None)
    assert p.summary()["shift"]["verdict"] == "insufficient"
    assert not [f for f in p.summary()["flags"] if f["code"] == "mismatch"]
    d = blob_of(p, filters=["incomplete"])
    inc = d["meta"]["incomplete"]
    assert inc["ignored"] == [] and inc["judged"] and inc["verdict"] == "insufficient"
    assert inc["label"] == "incomplete pattern (no counter ignored: shift check insufficient)"
    assert seed_times(d) == [3, 6, 9, 12], "S5's absences count"


def test_all_counters_but_s1_faulted_cannot_be_judged():
    p = _faulty_plugin(bad=(2, 3, 4, 5))
    assert [f["counter"] for f in p.timestamp_faults()["counters"]] == [2, 3, 4, 5]
    d = blob_of(p, filters=["incomplete"])
    inc = d["meta"]["incomplete"]
    assert not inc["judged"] and inc["note"] == "cannot judge: S2, S3, S4, S5 have timestamp faults"
    assert d["meta"]["seeds"] == [], "matches nothing rather than everything"
    nm = d["meta"]["search"]["no_match"]
    assert "incomplete pattern (cannot judge: S2, S3, S4, S5 have timestamp faults)" in nm["text"]
    # No seed carries "incomplete"; the other oddities still work.
    d = blob_of(p, seed="any", filters=["incomplete", "mismatch"])
    assert d["meta"]["seeds"] and all(s["odd"] == ["mismatch"] for s in d["meta"]["seeds"])
    # One counter left besides S1 is enough to judge.
    q = _faulty_plugin(bad=(3, 4, 5), missing={6: [2]})
    assert seed_times(blob_of(q, filters=["incomplete"])) == [6]


def test_a_faulted_s1_is_ignored_like_any_other_counter():
    frame = W.prepare_frame(_faulty_frame(bad=(), missing={5: [1], 8: [4]}))
    sel = W.select_seeds_by(frame, mode="ch2", filters=["incomplete"], ignore=[0])
    assert [round((t - T0) / SPACING, 1) for t in sel.t] == [8.0]
    assert sel.ignore == (0,) and sel.incomplete_judged


# --- the per-counter pattern selector -------------------------------------------------------

def _pattern_frame():
    """Events 2..12 are candidates: 3 no S3, 4 no S2 and S3, 5 no S1 (S2 seeds it),
    7 no S5, the rest complete."""
    events = [ev() for _ in range(14)]
    events[3] = ev(drop=[3])
    events[4] = ev(drop=[2, 3])
    events[5] = ev(drop=[1])
    events[7] = ev(drop=[5])
    return build(events)


def test_parse_pattern():
    assert W.parse_pattern(None) == () == W.parse_pattern({}) == W.parse_pattern({"2": "any"})
    assert W.parse_pattern({"3": "Absent", 1: "present"}) == ((1, "present"), (3, "absent"))
    assert W.parse_pattern({"S5": "absent"}) == ((5, "absent"),)
    for bad in ({"6": "present"}, {"0": "absent"}, {"x": "absent"}, {"1": "maybe"},
                {True: "present"}, ["1"], "1"):
        with pytest.raises(ValueError):
            W.parse_pattern(bad)
    assert W.parse_pattern({"6": "present"}, n_counters=6) == ((6, "present"),)


def _t(sel):
    return [int(round((t - T0) / SPACING)) for t in sel.t]


@pytest.mark.parametrize("mode, pattern, want", [
    ("s1", {"3": "absent"}, [3, 4]),
    ("s1", {"1": "present", "3": "absent"}, [3, 4]),
    ("s1", {"1": "absent"}, []),                       # an S1 seed always has S1
    ("s1", {"2": "absent", "3": "absent"}, [4]),
    ("s1", {"2": "present", "3": "absent"}, [3]),
    ("s1", {"5": "absent"}, [7]),
    ("s1", {"2": "present", "3": "present", "4": "present", "5": "present"}, [9, 10, 11, 12]),
    ("ch2", {"1": "absent"}, [5]),                     # a non-S1 seed: S1 is meaningful
    ("ch2", {"1": "present", "3": "absent"}, [3]),
    ("ch2", {"2": "absent"}, []),
    ("any", {"1": "absent"}, [5]),
    ("any", {"1": "present", "2": "absent"}, [4]),
])
def test_the_pattern_selector_tri_states(mode, pattern, want):
    fr = W.prepare_frame(_pattern_frame())
    sel = W.select_seeds_by(fr, W.Roles(), W.Cuts(n_seeds=4), mode, require=W.parse_pattern(pattern))
    assert _t(sel) == want
    for pat in sel.pattern:
        for k, st in W.parse_pattern(pattern):
            assert bool((int(pat) >> (k - 1)) & 1) == (st == "present")


def test_the_pattern_is_an_and_on_the_or_filters():
    # Event 3: no S3 (incomplete); 5: ToT; 9: no S3 and ToT.
    events = [ev() for _ in range(12)]
    events[3] = ev(drop=[3])
    events[5]["tot"] = {4: 254}
    events[9] = ev(drop=[3], tot={4: 254})
    fr = W.prepare_frame(build(events))
    cuts = W.Cuts(n_seeds=8)
    sel = W.select_seeds_by(fr, cuts=cuts, filters=["tot"], require=((3, "absent"),))
    assert _t(sel) == [9]
    sel = W.select_seeds_by(fr, cuts=cuts, filters=["incomplete", "tot"], require=((3, "present"),))
    assert _t(sel) == [5]
    sel = W.select_seeds_by(fr, cuts=cuts, filters=["incomplete"], require=((3, "absent"),))
    assert _t(sel) == [3, 9]


def test_the_pattern_is_explicit_and_ignores_nothing():
    # S5 has a known fault and is ignored by "incomplete", but "S5 absent" still means absent.
    p = _faulty_plugin(bad=(5,))
    d = blob_of(p, pattern={"5": "absent"})
    assert seed_times(d) == [3, 9, 12]
    sel = d["meta"]["select"]
    assert sel["pattern"] == {"5": "absent"} and sel["pattern_label"] == "S5 absent"
    assert d["meta"]["incomplete"]["ignored"][0]["counter"] == 5
    for s_ in d["meta"]["seeds"]:
        assert not s_["pattern"] & 0b10000 and "incomplete" not in s_["odd"]


def test_the_pattern_walks_back_and_names_itself():
    clk = _Clock()
    p = _plugin(clock=clk)
    plain = build([ev() for _ in range(12)])
    first = _feed_many(p, clk, [_pattern_frame(), plain, plain])[0]
    d = blob_of(p, pattern={"1": "present", "3": "absent"})
    assert d["frame_seq"] == first.seq and seed_times(d) == [3, 4]
    sv = d["meta"]["stale_view"]
    assert sv["good_since"] == 2 and sv["reason"] == "no S1 (ch 1) seeds and S1 present, S3 absent"
    assert d["meta"]["search"]["frames_searched"] == 3
    # With a filter and MuPix too: the no-match text names all of it.
    d = blob_of(p, pattern={"3": "absent"}, filters=["tot"], mupix="both")
    assert d["meta"]["search"]["no_match"]["text"] == (
        "no S1 (ch 1) seeds with ToT \u2265 250 and S3 absent and L1+L2 in time "
        "in the last 3 good frames")
    # By seq: that frame only, and the selection is cached with the pattern in its key.
    d = blob_of(p, pattern={"3": "absent"}, seq=first.seq)
    assert seed_times(d) == [3, 4] and "stale_view" not in d["meta"]
    assert (first.seq, "s1", (), "any", ((3, "absent"),), ()) in p._sel_cache


def test_the_command_takes_a_pattern():
    p = _plugin()
    _feed(p, _pattern_frame())
    tag, body = _dispatch(p, "sma::frame", json.dumps({"view": "seeded", "seed": "ch2",
                                                      "pattern": {"1": "absent"}}))
    assert tag == framing.TAG_SMAF
    d = framing.decode_sma_frame(body)
    assert seed_times(d) == [5 + 2 / SPACING] and d["meta"]["select"]["pattern"] == {"1": "absent"}
    tag, body = _dispatch(p, "sma::frame", json.dumps({"view": "seeded", "pattern": {"7": "absent"}}))
    assert tag == framing.TAG_ERROR and b"pattern" in body
    # The raster ignores it.
    assert p.frame_blob("raster", pattern={"1": "absent"}) == p.frame_blob("raster")


def test_run_1008_ignores_s5_for_incomplete():
    """The real frame: S5's fine = t/2 fault is the only counter fault; ignoring it
    makes "incomplete pattern" select fewer seeds, and none only for lacking S5."""
    clk = _Clock()
    p = _plugin(clock=clk)
    for k in range(3):
        clk.t += 1
        _feed(p, R1008, serial=k, run=1008)
    assert [f["counter"] for f in p.timestamp_faults()["counters"]] == [5]
    d = blob_of(p, filters=["incomplete"])
    fr = p._last.fr
    before = W.select_seeds_by(fr, filters=["incomplete"])
    after = W.select_seeds_by(fr, filters=["incomplete"], ignore=[4])
    assert d["meta"]["select"]["matching"] == after.n_matching < before.n_matching
    assert before.n_matching / before.n_examined > 0.95, "without the ignore nearly all match"
    for s in d["meta"]["seeds"]:
        assert "incomplete" in s["odd"] and (s["pattern"] & 0b01111) != 0b01111
