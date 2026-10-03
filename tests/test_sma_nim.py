"""``sma_nim``: TOT + NIM pairing and the NIM lag vote.

Parity: ``tests/data/nim_pairing_golden.json`` holds hit lists and what the real
reco headers make of them (``tests/cpp/generate_nim_pairing_golden.py``, which
compiles ``tests/cpp/nim_pairing_golden.cpp`` against
``PIPSMSMANimPairing.hh`` and ``PISMAFineOffset.hh`` of
``feature/sma-nim-pairing``). Every case must come out identical: classes,
flags, partners, merged times and order, counts and diagnostics.

How the C++ inputs map onto :func:`sma_nim.pair_counter`:

* One frame holds one counter (TOT id 2004, NIM id 2024, cabled, with
  offsets, so *expected*) plus pass-through hits (id 9999) whose only effect
  is the frame's raw bounds: ``frame_lo``/``frame_hi`` = min/max raw time over
  all hits of the case.
* ``RawHit::index`` (raw position) -> ``idx_tot``/``idx_nim``; each side is
  sorted by (raw time, raw index), the C++ ``GlobalOrder``.
* ``Config::echoVids`` containing the counter -> ``echo=True``;
  ``offsetNs`` -> ``tot_offset_ns``/``nim_offset_ns``.
* ``MergedHit``: ``time`` <-> ``merged.t``; ``rawTotIndex``/``rawNimIndex``
  <-> ``idx_*`` of the carrier and its partner; ``edep`` <-> ``merged.tot``;
  ``tot``/``nimWidth`` -1 without that word. ``NimDiag::dtAligned`` <->
  ``nim_dt``; ``TotDiag`` (non-echo TOT words only) <-> ``n_cand_tot[~echo]``;
  ``PairDiag`` <-> ``pair_dt``/``pair_tot``; ``FrameResult::wide`` (budget
  ``Config::widePairsPerFrame``) <-> :func:`sma_nim.wide_dt` on the raw times.
* A paired NIM word's flags (``flags_nim``) must equal its merged hit's.
* Lag cases: ``PISMAFineOffset::Scan`` with one lag channel and no fallback
  lag (``kNoLag``) <-> :func:`sma_nim.lag_vote`. ``source`` 1 (Vote) is a vote;
  the DQM's "ambiguous"/"none" split is ``n_best >= min pairs``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from mdqm.plugins import sma_nim as N
from mdqm.plugins import sma_words as W

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "data" / "nim_pairing_golden.json"
#: The reco checkout the golden file was made from, when it is on this machine.
RECO = HERE.parent.parent.parent / "scratch" / "worktrees" / "reco_testbeam-sma-nim-pairing"
DOC = json.loads(GOLDEN.read_text())
PAIR_CASES = [c for c in DOC["cases"] if c["kind"] == "pair"]
LAG_CASES = [c for c in DOC["cases"] if c["kind"] == "lag"]
RNG = np.random.default_rng(20261002)


def _arr(values):
    """int64 when every value is integral (the DQM's times), else float64."""
    if all(float(v).is_integer() for v in values):
        return np.array([int(v) for v in values], dtype=np.int64)
    return np.array(values, dtype=np.float64)


def _num(x):
    return int(x) if float(x).is_integer() else float(x)


def run_pair_case(case):
    cfg_c = case["config"]
    hits = case["hits"]
    side = {}
    for role in (0, 1):
        h = sorted((t, idx, tot) for r, t, tot, idx in hits if r == role)
        side[role] = (_arr([x[0] for x in h]), np.array([x[2] for x in h], dtype=np.int16),
                      np.array([x[1] for x in h], dtype=np.int64))
    all_t = [h[1] for h in hits]
    cfg = N.NimConfig(pair_window_ns=_num(cfg_c["window"]), time_source=cfg_c["time_source"],
                      nim_only_tot=_num(cfg_c["nim_only_tot"]), tot_unit_ns=_num(cfg_c["tot_unit"]),
                      echo_late_tot=cfg_c["echo_late_tot"],
                      echo_edge_tol_ns=_num(cfg_c["echo_edge_tol"]))
    (tt, tot_t, it), (tn, tot_n, in_) = side[0], side[1]
    res = N.pair_counter(tt, tot_t, tn, tot_n, cfg,
                         tot_offset_ns=_num(cfg_c["tot_offset"]),
                         nim_offset_ns=_num(cfg_c["nim_offset"]), echo=cfg_c["echo"],
                         frame_lo=_num(min(all_t)), frame_hi=_num(max(all_t)),
                         idx_tot=it, idx_nim=in_)
    tt_raw, tn_raw = side[0][0], side[1][0]
    wide = N.wide_dt(tt_raw, tn_raw, _num(min(all_t)), _num(max(all_t)),
                     cfg_c.get("wide_budget", 0))
    return res, it, in_, wide


def merged_as_cpp(res, idx_tot, idx_nim):
    """The C++ MergedHit fields of every merged hit, in merged order."""
    out = []
    m = res.merged
    for t, tot, cls, src, flags in zip(m.t.tolist(), m.tot.tolist(), m.cls.tolist(),
                                       m.src.tolist(), m.flags.tolist()):
        if cls == N.NIM_ONLY:
            b = src
            out.append({"time": t, "t_tot": None, "t_nim": float(res.t_nim[b]), "tot": -1,
                        "nim_width": int(res.tot_nim[b]), "flags": flags, "raw_tot": -1,
                        "raw_nim": int(idx_nim[b]), "edep": tot})
            continue
        a = src
        b = int(res.partner_tot[a])
        out.append({"time": t, "t_tot": float(res.t_tot[a]),
                    "t_nim": float(res.t_nim[b]) if b >= 0 else None,
                    "tot": int(res.tot_tot[a]), "nim_width": int(res.tot_nim[b]) if b >= 0 else -1,
                    "flags": flags, "raw_tot": int(idx_tot[a]),
                    "raw_nim": int(idx_nim[b]) if b >= 0 else -1, "edep": tot})
    return out


def _same(x, y):
    if x is None or y is None:
        return x is None and y is None
    return float(x) == float(y)


@pytest.mark.parametrize("case", PAIR_CASES, ids=[c["name"] for c in PAIR_CASES])
def test_pairing_matches_reco_header(case):
    res, it, in_, wide = run_pair_case(case)
    got = merged_as_cpp(res, it, in_)
    want = case["merged"]
    assert len(got) == len(want)
    for k, (g, w) in enumerate(zip(got, want)):
        for key in w:
            assert _same(g[key], w[key]), (k, key, g, w)
    assert res.counts() == case["counts"]
    # per NIM word, in NIM order: nearest-TOT dt (aligned), its ToT, the width
    assert len(res.nim_dt) == len(case["nim_diag"])
    for k, (dt, near, width) in enumerate(case["nim_diag"]):
        assert _same(None if np.isnan(res.nim_dt[k]) else res.nim_dt[k], dt), k
        assert int(res.nim_nearest_tot[k]) == near and int(res.tot_nim[k]) == width
    assert res.n_cand_tot[~res.echo].tolist() == case["tot_diag"]
    assert [[float(d), int(t)] for d, t in zip(res.pair_dt, res.pair_tot)] == \
        [[float(d), t] for d, t in case["pair_diag"]]
    # a paired NIM word carries its merged hit's flags (near-edge included)
    pos_nim = {int(r): k for k, r in enumerate(in_)}
    for w in want:
        if w["raw_nim"] >= 0:
            assert int(res.flags_nim[pos_nim[w["raw_nim"]]]) == w["flags"], w
    assert [float(x) for x in wide] == [float(x) for x in case.get("wide", [])]


def test_golden_covers_the_hard_cases():
    """The cases the brief asks for are in the file, and the file is from the
    reviewed header commit."""
    names = {c["name"] for c in DOC["cases"]}
    for n in ("tie_two_tots", "tie_two_nims", "tie_same_time_raw_index", "chain_greedy_vs_mutual",
              "chain_equal_dt_long", "echo_edge_chain", "empty_tot", "empty_nim", "window_edge",
              "nim_only_in_shadow", "near_frame_edge", "faulted_153521", "faulted_881",
              "ambiguous", "too_few", "wrap_at_zero"):
        assert n in names
    for n in ("float_window_boundary", "merged_tie_nim_only_vs_tot", "paired_nim_near_edge",
              "wide_long_frame", "wide_short_frame", "same_coarse_s1_hidden"):
        assert n in names
    assert DOC["reco_commit"].startswith("f25bf7c")
    assert len(PAIR_CASES) >= 20
    assert any(c["config"]["wide_budget"] > 0 and len(c["wide"]) > 0 for c in PAIR_CASES)


def test_golden_headers_unchanged():
    """When the reco headers are on this machine, they are the ones the golden
    file was made from; a changed header skips (regenerate, see tests/cpp)."""
    import hashlib
    alg, midas = RECO / "psm" / "exp" / "alg" / "include", RECO / "pi_midas" / "include"
    paths = {"PIPSMSMANimPairing.hh": alg, "PISMAFineOffset.hh": midas, "PISMAWord.hh": midas}
    if not all((d / n).is_file() for n, d in paths.items()):
        pytest.skip(f"reco headers not found under {RECO}")
    changed = [n for n, d in paths.items()
               if hashlib.sha256((d / n).read_bytes()).hexdigest() != DOC["headers"][n]]
    if changed:
        pytest.skip(f"reco headers changed since the golden file was written: {changed}; "
                    "regenerate it (tests/cpp/README.md)")


@pytest.mark.parametrize("case", LAG_CASES, ids=[c["name"] for c in LAG_CASES])
def test_lag_vote_matches_reco_scan(case):
    c = case["config"]
    w = np.array(case["words"], dtype=np.int64).reshape(-1, 4)
    cfg = N.NimConfig(lag_tol_ns=c["lag_tol"], lag_min_pairs=c["min_pairs"],
                      lag_dominance=c["dominance"], lag_nominal_ns=c["nominal"],
                      lag_next_reference=c["next_reference"])
    v = N.lag_vote(w[:, 0], w[:, 2].astype(np.uint32), w[:, 3].astype(np.uint32), c["nim"],
                   c["s1"], cfg)
    assert v.n_pairs == case["n_pairs"]
    assert np.sort(v.d).tolist() == case["d_sorted"]
    assert (v.n_best, v.n_runner_up) == (case["n_best"], case["n_runner_up"])
    assert v.lag_ns == case["lag"]
    assert v.best_lag_ns == case["best_lag"]
    assert v.runner_up_lag_ns == case["runner_up_lag"]
    assert v.shift_ns == case["shift_ns"]
    voted = case["source"] == 1
    assert (v.state in ("ok", "faulted")) == voted
    assert v.faulted == (voted and case["shift_ns"] != 0)


# ==============================================================================
# Pairing: properties beyond the golden file
# ==============================================================================

def sequential_greedy(a, b, adt, n_tot, n_nim):
    """The C++ loop, literally: sorted candidates, keep a pair when both are free."""
    p_t = np.full(n_tot, -1)
    p_n = np.full(n_nim, -1)
    for k in sorted(range(len(a)), key=lambda k: (adt[k], a[k], b[k])):
        if p_t[a[k]] < 0 and p_n[b[k]] < 0:
            p_t[a[k]], p_n[b[k]] = b[k], a[k]
    return p_t, p_n


@pytest.mark.parametrize("rounds", [1, 2, N.GREEDY_ROUNDS, 1000])
def test_greedy_rounds_equal_the_sequential_loop(rounds):
    for trial in range(40):
        n_t, n_n = RNG.integers(1, 60, size=2)
        tt = np.sort(RNG.integers(0, 600, n_t))
        tn = np.sort(RNG.integers(0, 600, n_n))
        i, j, dt = W.window_pairs(tt, tn, -20, 20)
        want = sequential_greedy(i, j, np.abs(dt), n_t, n_n)
        got = N.greedy_match_by_dt(i, j, dt, n_t, n_n, max_rounds=rounds)
        assert np.array_equal(got[0], want[0]) and np.array_equal(got[1], want[1]), trial


def frame_args(n_tot, n_nim, lo=None, hi=None):
    """The required keywords for a counter alone in its frame (TOT indices first)."""
    return {"frame_lo": lo, "frame_hi": hi, "idx_tot": np.arange(n_tot),
            "idx_nim": n_tot + np.arange(n_nim)}


def test_int64_times_stay_exact():
    """Times near 2^40 (the SMA epoch) with integer offsets stay int64 and exact."""
    base = (1 << 40) - 5000
    tt = base + np.array([0, 100, 200], dtype=np.int64)
    tn = tt + np.array([3, -4, 30])
    res = N.pair_counter(tt, [10, 10, 10], tn, [5, 5, 5], tot_offset_ns=2, nim_offset_ns=1,
                         **frame_args(3, 3))
    assert res.merged.t.dtype == np.int64
    assert res.merged.t.tolist() == sorted([base - 2, base + 98, base + 198, base + 229])
    assert res.cls_tot.tolist() == [N.PAIRED, N.PAIRED, N.TOT_ONLY]
    assert res.nim_dt.tolist() == [4.0, -3.0, 31.0]


def test_merged_hits_without_nim_only():
    res = N.pair_counter([100, 300], [10, 10], [101, 200], [5, 5], **frame_args(2, 2))
    assert res.merged.cls.tolist() == [N.PAIRED, N.NIM_ONLY, N.TOT_ONLY]
    held = res.merged_hits(nim_only=False)
    assert held.t.tolist() == [100, 300] and held.cls.tolist() == [N.PAIRED, N.TOT_ONLY]
    m = N.merge_counter([100, 300], [10, 10], [101, 200], [5, 5], **frame_args(2, 2))
    assert m.t.tolist() == [100, 200, 300] and m.tot.tolist() == [10, 1, 10]


def test_whole_number_float_offsets_keep_int64():
    """An ODB DOUBLE 0.0 or 15.0 keeps the times int64; a fraction makes float64."""
    tt = np.array([100, 300], dtype=np.int64)
    for off in (0.0, 15.0, np.float64(-3.0)):
        res = N.pair_counter(tt, [10, 10], tt + 1, [5, 5], tot_offset_ns=off,
                             nim_offset_ns=off, **frame_args(2, 2))
        assert res.merged.t.dtype == np.int64 and res.t_nim.dtype == np.int64
    res = N.pair_counter(tt, [10, 10], tt + 1, [5, 5], nim_offset_ns=0.5, **frame_args(2, 2))
    assert res.t_nim.dtype == np.float64 and res.t_tot.dtype == np.int64


def test_frame_edges_and_indices_are_required():
    with pytest.raises(TypeError):
        N.pair_counter([100], [10], [101], [5])
    with pytest.raises(ValueError):
        N.pair_counter([100], [10], [101], [5], frame_lo=0, frame_hi=None, idx_tot=[0],
                       idx_nim=[1])
    with pytest.raises(ValueError):
        N.pair_counter([100], [10], [101], [5], frame_lo=0, frame_hi=1000, idx_tot=[0],
                       idx_nim=[])
    # None for both edges: no edge flag
    res = N.pair_counter([5], [10], [6], [5], **frame_args(1, 1))
    assert not (res.flags_tot & N.NEAR_FRAME_EDGE).any()


def test_paired_nim_words_carry_the_pair_flags():
    res = N.pair_counter([5, 500, 600], [10, 10, 10], [6, 501, 700], [5, 5, 5],
                         **frame_args(3, 3, 0, 1000))
    assert res.flags_tot[0] & N.NEAR_FRAME_EDGE
    for a in res.paired:
        assert res.flags_nim[res.partner_tot[a]] == res.flags_tot[a]
    assert res.flags_nim[0] & N.NEAR_FRAME_EDGE
    # the same over a random dense frame, every flag
    tt = np.sort(RNG.integers(0, 5000, 400))
    tn = np.sort(RNG.integers(0, 5000, 400))
    res = N.pair_counter(tt, RNG.integers(1, 60, 400), tn, np.full(400, 9), echo=True,
                         **frame_args(400, 400, 0, 5000))
    a = res.paired
    assert np.array_equal(res.flags_nim[res.partner_tot[a]], res.flags_tot[a])


def brute_wide(tt, tn, lo, hi, budget):
    """WidePairs as the C++ loop writes it, word by word."""
    H = N.HALF_FINE_WRAP_NS
    tt, tn = list(tt), list(tn)
    import bisect
    a = bisect.bisect_left(tn, lo + H)
    b = bisect.bisect_right(tn, hi - H)
    first, last = (a, b) if a < b else (0, len(tn))
    span = max(float(hi - lo), 1.0)
    per = max(1.0, len(tt) * min(1.0, 2.0 * H / span))
    want = max(1, int(budget / per))
    n = last - first
    step = n / min(want, n)
    out, pairs, i = [], 0, 0
    while pairs < budget:
        k = first + int((i + 0.5) * step)
        if k >= last:
            break
        j = bisect.bisect_left(tt, tn[k] - H)
        while j < len(tt) and tt[j] < tn[k] + H:
            out.append(tn[k] - tt[j])
            pairs += 1
            j += 1
        i += 1
    return out


@pytest.mark.parametrize("budget", [1, 37, 500, 10**7])
def test_wide_dt_equals_the_loop_and_stays_bounded(budget):
    for span, n in ((4_000_000, 300), (600_000, 200), (3_000_000, 3000)):
        tt = np.sort(RNG.integers(0, span, n))
        tn = np.sort(RNG.integers(0, span, n))
        got = N.wide_dt(tt, tn, 0, span, budget)
        assert got.tolist() == brute_wide(tt, tn, 0, span, budget)
        # never more than the budget plus one NIM word's window
        per_window = int(np.max(np.searchsorted(tt, tn + N.HALF_FINE_WRAP_NS)
                                - np.searchsorted(tt, tn - N.HALF_FINE_WRAP_NS)))
        assert got.size < budget + per_window + 1
    assert N.wide_dt([], [5], 0, 10, 100).size == 0
    assert N.wide_dt([5], [5], 0, 10, 0).size == 0


def test_config_checks():
    for bad in ({"pair_window_ns": 0}, {"tot_unit_ns": 0}, {"time_source": "both"},
                {"lag_tol_ns": 0}, {"lag_tol_ns": 1024}, {"lag_min_pairs": 0},
                {"lag_dominance": 0.5}, {"lag_dominance": float("inf")},
                {"lag_nominal_ns": 1 << 19}, {"lag_min_pairs": 50.7}, {"echo_late_tot": 12.5},
                {"lag_nominal_ns": 3.5}, {"pair_window_ns": float("nan")},
                {"lag_min_pairs": "50"}):
        with pytest.raises(ValueError):
            N.NimConfig(**bad)
    cfg = N.NimConfig(pair_window_ns=20.0, lag_min_pairs=50.0, tot_unit_ns=2.0,
                      echo_edge_tol_ns=2.5, nim_only_tot=1.0)
    assert type(cfg.pair_window_ns) is int and type(cfg.lag_min_pairs) is int
    assert type(cfg.tot_unit_ns) is int and type(cfg.nim_only_tot) is int
    assert cfg.echo_edge_tol_ns == 2.5


# ==============================================================================
# Lag vote on synthetic coarse/fine fields
# ==============================================================================

SHIFT = 14


def lag_frame(n, lags, delay=20, jitter=3, nim_ch=3, s1_ch=1, t0=7_000_000, gap=3000,
              nim_first=0.0, rng=None):
    """Stream-order (ch, coarse, fine): per event an S1 word at t and a NIM
    word at t + delay whose fine field is moved by a lag drawn from ``lags``."""
    rng = rng or np.random.default_rng(1)
    t = t0 + np.cumsum(rng.integers(50, 2 * gap, n))
    tn = t + delay + np.rint(rng.normal(0, jitter, n)).astype(np.int64)
    lag = rng.choice(np.asarray(lags), n)
    c1, f1 = W.fields_of(t, SHIFT)
    cn, _ = W.fields_of(tn, SHIFT)
    fn = ((tn + lag) & (W.FINE_WRAP_NS - 1)).astype(np.uint32)
    first = rng.random(n) < nim_first
    ch = np.empty(2 * n, dtype=np.int16)
    co = np.empty(2 * n, dtype=np.uint32)
    fi = np.empty(2 * n, dtype=np.uint32)
    s1_at = np.where(first, 1, 0) + 2 * np.arange(n)
    nim_at = np.where(first, 0, 1) + 2 * np.arange(n)
    ch[s1_at], co[s1_at], fi[s1_at] = s1_ch, c1, f1
    ch[nim_at], co[nim_at], fi[nim_at] = nim_ch, cn, fn
    return ch, co, fi


def test_lag_clean():
    v = N.lag_vote(*lag_frame(300, [0]), nim_ch=3)
    assert v.state == "ok" and v.shift_ns == 0
    assert abs(v.lag_ns - 20) <= 1 and v.n_best >= 290 and v.n_runner_up == 0


@pytest.mark.parametrize("fault", [-153521, -881])
def test_lag_faulted(fault):
    v = N.lag_vote(*lag_frame(300, [fault]), nim_ch=3)
    assert v.state == "faulted"
    assert abs(v.lag_ns - (fault + 20)) <= 1 and v.shift_ns == v.lag_ns


def test_lag_ambiguous():
    v = N.lag_vote(*lag_frame(300, [-881] * 3 + [0] * 2), nim_ch=3)
    assert v.state == "ambiguous" and v.lag_ns is None
    assert v.n_best < 2 * v.n_runner_up and v.n_best >= 50
    assert v.shift_ns == 0


def test_lag_too_few_pairs():
    v = N.lag_vote(*lag_frame(30, [-881]), nim_ch=3)
    assert v.state == "none" and v.n_pairs == 30 and v.lag_ns is None
    assert N.lag_vote(np.zeros(0, np.int16), np.zeros(0, np.uint32), np.zeros(0, np.uint32),
                      3).state == "none"


def test_lag_wraps_around_2_20():
    # true delay 0 with jitter: d straddles 0 = 2^20; one window across the wrap
    v = N.lag_vote(*lag_frame(300, [0], delay=0, jitter=4), nim_ch=3)
    assert v.state == "ok" and abs(v.lag_ns) <= 2
    assert (v.d > W.FINE_WRAP_NS // 2).any() and (v.d < 100).any()
    assert v.n_best == 300
    # a lag at the 2^19 fold: still one window, folded to a signed lag
    v = N.lag_vote(*lag_frame(300, [(1 << 19) - 20], delay=20, jitter=4), nim_ch=3)
    assert v.state == "faulted" and v.n_best >= 295
    assert v.lag_ns in range(-(1 << 19), -(1 << 19) + 10) or v.lag_ns in range((1 << 19) - 10, 1 << 19)


def test_lag_nominal_and_circular_distance():
    assert N.lag_shift_ns(-861, -861, 50) == 0
    assert N.lag_shift_ns(-(1 << 19) + 1, (1 << 19) - 1, 50) == 0      # 2 ns apart on the circle
    assert N.lag_shift_ns(-881, 0, 50) == -881
    assert N.lag_shift_ns(None, 0, 50) == 0
    cfg = N.NimConfig(lag_nominal_ns=-861)
    assert N.lag_vote(*lag_frame(300, [-881]), nim_ch=3, cfg=cfg).state == "ok"
    # per channel: the call's nominal wins over the config's
    frame = lag_frame(300, [-881])
    assert N.lag_vote(*frame, nim_ch=3, nominal_ns=-861).state == "ok"
    assert N.lag_vote(*frame, nim_ch=3, cfg=cfg, nominal_ns=0).state == "faulted"
    with pytest.raises(ValueError):
        N.lag_vote(*frame, nim_ch=3, nominal_ns=1 << 19)


def test_lag_pairs_previous_then_next_reference():
    # S1(c=5) NIM(c=5) | NIM(c=6) S1(c=6) | S1(c=7) S1(c=8) NIM(c=7): the last one has
    # no pair (its preceding S1 is c=8, the next S1 does not exist)
    ch = np.array([1, 3, 3, 1, 1, 1, 3])
    co = np.array([5, 5, 6, 6, 7, 8, 7], dtype=np.uint32)
    fi = np.array([100, 130, 200, 190, 0, 0, 5], dtype=np.uint32)
    assert N.lag_pairs(ch, co, fi, 3).tolist() == [30, 10]
    assert N.lag_pairs(ch, co, fi, 3, next_reference=False).tolist() == [30]
    # the fine difference wraps mod 2^20
    assert N.lag_pairs([1, 3], [0, 0], [10, 5], 3).tolist() == [W.FINE_WRAP_NS - 5]


def test_lag_modes_tie_takes_the_lowest_start():
    d = np.array([100] * 10 + [5000] * 10, dtype=np.uint32)
    assert N.lag_modes(d, 100) == (10, 100, 10, 5000)


# ==============================================================================
# Speed (reported, not asserted)
# ==============================================================================

def test_speed_report(capsys):
    """ms per 10k words for the pairing of one counter (TOT + NIM words) and for
    the lag vote; printed with ``-s``, never a failure."""
    n = 5000                                       # 5k TOT + ~5k NIM words
    tt = np.cumsum(RNG.integers(20, 400, n)).astype(np.int64)
    tot = RNG.integers(5, 60, n).astype(np.int16)
    keep = RNG.random(n) < 0.95
    tn = np.sort(tt[keep] + np.rint(RNG.normal(0, 3, keep.sum())).astype(np.int64))
    tw = np.full(tn.size, 15, dtype=np.int16)
    cfg = N.NimConfig()
    kw = frame_args(tt.size, tn.size, int(tt[0]) - 100, int(tt[-1]) + 100)
    N.pair_counter(tt, tot, tn, tw, cfg, echo=True, **kw)
    reps = 20
    t0 = time.perf_counter()
    for _ in range(reps):
        N.pair_counter(tt, tot, tn, tw, cfg, echo=True, **kw)
    pair_ms = (time.perf_counter() - t0) / reps * 1e3 * 10_000 / (tt.size + tn.size)
    ch, co, fi = lag_frame(5000, [-881])
    t0 = time.perf_counter()
    for _ in range(reps):
        N.lag_vote(ch, co, fi, 3)
    lag_ms = (time.perf_counter() - t0) / reps * 1e3 * 10_000 / ch.size
    with capsys.disabled():
        print(f"\n[sma_nim speed] pairing {pair_ms:.2f} ms / 10k words, "
              f"lag vote {lag_ms:.2f} ms / 10k words")
