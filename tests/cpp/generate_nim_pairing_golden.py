"""Write ``tests/data/nim_pairing_golden.json``: hit lists and what the reco headers make of them.

Run inside the ``testbeam-midas`` container (it needs g++ with C++20, nothing
else; stdlib Python only), as your own user so nothing in the tree is root's:

    cd ~/github/pioneer/testbeam-env
    docker exec -u 1000:1000 testbeam-midas bash -lc \\
        "cd /workdir/wavedream-frontends/wavedream-midas-dqm && \\
         python3 tests/cpp/generate_nim_pairing_golden.py \\
         --reco-commit $(git -C scratch/worktrees/reco_testbeam-sma-nim-pairing rev-parse HEAD)"

(The worktree's ``.git`` points at a host path, so git cannot resolve it
inside the container; the commit is read on the host and passed in.)

It writes the cases as text, compiles ``nim_pairing_golden.cpp`` against the
real headers of the ``feature/sma-nim-pairing`` worktree (read only):

* ``psm/exp/alg/include/PIPSMSMANimPairing.hh`` (``Prepare``, ``Pair``),
* ``pi_midas/include/PISMAFineOffset.hh`` + ``PISMAWord.hh`` (``Scan``, the
  lag vote),

runs it, and stores its output with the header commit and hashes. The build
goes to ``scratch/sma-nim-dqm/cpp-build`` (``--build``), never into the repo.
``tests/test_sma_nim.py`` then compares ``mdqm.plugins.sma_nim`` against the
file with numpy and pytest only.

The cases are fixed (seeded), so a rerun on the same headers writes the same
file. Rerun when the headers change, and look at the diff before committing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
WORKDIR = REPO.parent.parent                    # the testbeam-env workspace
#: The default reco checkout (``--reco``): the feature/sma-nim-pairing worktree.
RECO = WORKDIR / "scratch" / "worktrees" / "reco_testbeam-sma-nim-pairing"


def includes(reco: Path):
    """The two include directories of a reco_testbeam checkout."""
    return reco / "psm" / "exp" / "alg" / "include", reco / "pi_midas" / "include"


def headers(reco: Path):
    """The headers the golden file is made from (their hashes go into it)."""
    alg, midas = includes(reco)
    return alg / "PIPSMSMANimPairing.hh", midas / "PISMAFineOffset.hh", midas / "PISMAWord.hh"

OUT = REPO / "tests" / "data" / "nim_pairing_golden.json"
BUILD = WORKDIR / "scratch" / "sma-nim-dqm" / "cpp-build"

TOT, NIM, OTHER = 0, 1, 2
SHIFT = 14
FINE_MASK = (1 << 20) - 1
COARSE_MASK = (1 << 28) - 1


# --- pairing cases -------------------------------------------------------------

def pair(name, tots, nims, others=(), *, window=20, time_source=0, nim_only_tot=1,
         tot_unit=1, echo=0, late=128, edge_tol=3, tot_off=0, nim_off=0, seed=None,
         wide=0, idx=None):
    """One PAIR case. ``tots``/``nims``: (t, tot); ``others``: times. Raw
    indices follow the listed order (TOT, NIM, other), a seeded shuffle, or
    ``idx`` (one per hit, in that order). ``wide``: the wide-dt budget."""
    hits = ([(TOT, t, tot) for t, tot in tots] + [(NIM, t, w) for t, w in nims]
            + [(OTHER, t, 5) for t in others])
    order = list(range(len(hits)))
    if seed is not None:
        random.Random(seed).shuffle(order)
    if idx is not None:
        assert sorted(idx) == order, "idx must be a permutation of the hits"
        order = list(idx)
    head = (f"PAIR {name}\n{window} {time_source} {nim_only_tot} {tot_unit} {echo} {late} "
            f"{edge_tol} {tot_off} {nim_off} {wide}\n{len(hits)}\n")
    return head + "".join(f"{r} {t!r} {v} {i}\n" for (r, t, v), i in zip(hits, order))


def random_pair(name, seed, n, mean_gap, *, eff=0.95, delay=0, jitter=2.0, noise=0.03,
                echo_frac=0.0, late_frac=0.0, quantum=1, **kw):
    """A TOT stream with exponential gaps, its NIM copy (efficiency ``eff``,
    Gaussian jitter), NIM noise, trailing-edge echoes and late words. Times
    are rounded to ``quantum`` ns, so a coarse quantum makes exact ties."""
    rng = random.Random(seed)
    q = lambda x: int(round(x / quantum)) * quantum   # noqa: E731
    t = 1000.0
    tots, nims = [], []
    for _ in range(n):
        t += rng.expovariate(1.0 / mean_gap) + 1
        tot = rng.randint(5, 60)
        tots.append((q(t), tot))
        if rng.random() < eff:
            nims.append((q(t + delay + rng.gauss(0, jitter)), rng.randint(10, 30)))
        if rng.random() < echo_frac:
            tots.append((q(t) + tot + rng.randint(-4, 4), rng.randint(1, 20)))
        if rng.random() < late_frac:
            tots.append((q(t + rng.randint(20, 300)), rng.randint(128, 255)))
        if rng.random() < noise:
            nims.append((q(t + rng.uniform(-3 * mean_gap, 3 * mean_gap)), rng.randint(10, 30)))
    others = [0, q(t + 500)]
    return pair(name, tots, nims, others, seed=seed, **kw)


def pair_cases():
    c = []
    c.append(pair("isolated_pairs", [(100, 20), (500, 30), (900, 25)],
                  [(100, 15), (505, 15), (893, 15)], [0, 2000]))
    c.append(pair("empty_tot", [], [(100, 15), (300, 16)], [0, 1000]))
    c.append(pair("empty_nim", [(100, 20), (300, 21)], [], [0, 1000]))
    c.append(pair("empty_counter", [], [], [0, 1000]))
    c.append(pair("window_edge", [(100, 20), (500, 20), (900, 20), (1300, 20)],
                  [(120, 15), (480, 15), (921, 15), (1279, 15)], [0, 2000]))
    c.append(pair("tie_two_tots", [(100, 20), (110, 20)], [(105, 15)], [0, 1000]))
    c.append(pair("tie_two_nims", [(110, 20)], [(105, 15), (115, 16)], [0, 1000]))
    # equal times told apart by the raw index (the TOT listed second has the lower index)
    c.append(pair("tie_same_time_raw_index", [(100, 20), (100, 21)], [(100, 15)], [0, 1000],
                  seed=3))
    c.append(pair("tie_same_time_nims", [(100, 20)], [(100, 15), (100, 16)], [0, 1000], seed=11))
    # greedy pairs both; one pass of mutual nearest pairs only (a1, b0)
    c.append(pair("chain_greedy_vs_mutual", [(100, 20), (110, 20)], [(106, 15), (119, 15)],
                  [0, 1000]))
    c.append(pair("chain_three", [(100, 20), (112, 20), (125, 20)],
                  [(106, 15), (118, 15), (131, 15)], [0, 1000]))
    # a long chain of equal |dt| (10 ns spacing): more greedy rounds than the vectorised limit
    c.append(pair("chain_equal_dt_long", [(1000 + 10 * k, 5) for k in range(40)],
                  [(1005 + 10 * k, 15) for k in range(40)], [0, 3000], echo=0))
    c.append(pair("echo_late", [(100, 20), (300, 200), (600, 130)],
                  [(101, 15), (303, 15), (598, 15)], [0, 1000], echo=1))
    c.append(pair("echo_edge_chain", [(100, 40), (141, 10), (152, 5), (300, 30), (333, 8),
                                      (345, 8)],
                  [(101, 15), (142, 15), (299, 15)], [0, 1000], echo=1))
    c.append(pair("echo_rule_off_counter", [(100, 40), (141, 10), (300, 200)],
                  [(101, 15), (142, 15), (299, 15)], [0, 1000], echo=0))
    c.append(pair("echo_late_off", [(100, 40), (141, 10), (300, 200)],
                  [(101, 15), (142, 15), (299, 15)], [0, 1000], echo=1, late=0))
    c.append(pair("echo_edge_off", [(100, 40), (141, 10), (300, 200)],
                  [(101, 15), (142, 15), (299, 15)], [0, 1000], echo=1, edge_tol=-1))
    c.append(pair("nim_only_in_shadow", [(100, 100), (1000, 255)],
                  [(102, 15), (150, 15), (200, 15), (201, 15), (400, 15), (1254, 15),
                   (1255, 15), (1256, 15)], [0, 3000]))
    c.append(pair("near_frame_edge", [(10, 20), (35, 20), (500, 20), (975, 20), (995, 20)],
                  [(12, 15), (500, 15), (980, 15), (1000, 15), (3, 15)], [0]))
    # near the edge only through the offsets (frame [0, 1000]): TOT 8 -> 5 + 15 - 20 = 0,
    # TOT 975 -> 972 + 15 + 20 = 1007, while 8 - 20 and 975 + 20 are inside
    c.append(pair("near_frame_edge_offsets", [(8, 20), (60, 20), (500, 20), (975, 20)],
                  [(23, 15), (75, 15), (515, 15), (700, 15)], [0, 1000], tot_off=3, nim_off=15))
    c.append(pair("float_offsets_nim_time", [(100, 20), (130, 20), (500, 20)],
                  [(112, 15), (128, 15), (513, 15), (700, 15)], [0, 1000],
                  tot_off=2.5, nim_off=-9.75, time_source=1))
    # NIM time source reorders the merged hits
    c.append(pair("nim_time_reorders", [(100, 20), (104, 20)], [(118, 15), (90, 15)], [0, 1000],
                  time_source=1))
    c.append(pair("nim_only_tot_and_unit", [(100, 50), (400, 60), (530, 9)],
                  [(101, 15), (140, 15), (175, 15), (520, 15)], [0, 1000],
                  nim_only_tot=2.5, tot_unit=2, echo=1, late=100))
    c.append(pair("window_5", [(100, 20), (200, 20)], [(105, 15), (206, 15)], [0, 1000],
                  window=5))
    # a fractional window and offset landing exactly on the boundary: 107.75 - 0.25 - 100
    # = 7.5 and 200 - (192.75 - 0.25) = 7.5 pair, 307.85 - 0.25 - 300 = 7.6 does not
    c.append(pair("float_window_boundary", [(100, 20), (200, 20), (300, 20)],
                  [(107.75, 15), (192.75, 15), (307.85, 15)], [0, 1000], window=7.5,
                  nim_off=0.25))
    # exact time ties between a NIM-only hit and a TOT hit in the merged list: a second
    # NIM at 100 (the first pairs), and a NIM at the time of a late (echo) word; the raw
    # indices put the NIM-only hit before the TOT hit at 100 in the first case and after
    # it in the second
    c.append(pair("merged_tie_nim_only_vs_tot", [(100, 20), (300, 200)],
                  [(100, 15), (100, 16), (300, 15)], [0, 1000], echo=1,
                  idx=[3, 6, 2, 0, 5, 1, 4]))
    c.append(pair("merged_tie_nim_only_vs_tot_nim_time", [(100, 20), (300, 200)],
                  [(100, 15), (100, 16), (300, 15)], [0, 1000], echo=1, time_source=1,
                  idx=[0, 5, 1, 6, 2, 3, 4]))
    # paired NIM words at the frame edges carry the edge flag of their pair
    c.append(pair("paired_nim_near_edge", [(5, 20), (500, 20), (990, 20)],
                  [(6, 15), (501, 15), (992, 15)], [0, 1000]))
    # the wide dt: a frame of ~4 ms (an interior), a lagged NIM copy, several budgets
    c.append(random_pair("wide_long_frame", 7, 400, 10000.0, delay=-153521, wide=500))
    c.append(random_pair("wide_long_frame_big_budget", 8, 150, 25000.0, wide=1_000_000))
    c.append(random_pair("wide_long_frame_budget_1", 9, 300, 12000.0, wide=1))
    c.append(random_pair("wide_short_frame", 10, 200, 300.0, wide=50))
    c.append(random_pair("wide_dense", 11, 600, 4000.0, delay=-881, wide=1000))
    c.append(random_pair("random_sparse", 1, 120, 400.0))
    c.append(random_pair("random_dense", 2, 300, 30.0, jitter=4.0, noise=0.2))
    c.append(random_pair("random_offsets", 3, 200, 120.0, delay=37, nim_off=37, tot_off=-4,
                         jitter=3.0))
    c.append(random_pair("random_echoes", 4, 250, 150.0, echo_frac=0.2, late_frac=0.1,
                         echo=1))
    c.append(random_pair("random_ties_quantised", 5, 300, 25.0, quantum=5, jitter=5.0,
                         noise=0.3))
    c.append(random_pair("random_low_eff_nim_time", 6, 200, 60.0, eff=0.6, noise=0.4,
                         time_source=1, nim_off=1.5))
    return c


# --- lag cases -----------------------------------------------------------------

def fields(t):
    return (t >> SHIFT) & COARSE_MASK, t & FINE_MASK


def lag(name, words, *, s1=1, nim=3, tol=50.0, min_pairs=50, dominance=2.0, nominal=0,
        next_ref=1):
    head = (f"LAG {name}\n{s1} {nim} {SHIFT} {tol} {min_pairs} {dominance} {nominal} "
            f"{next_ref}\n{len(words)}\n")
    return head + "".join(f"{ch} {tot} {co} {fi}\n" for ch, tot, co, fi in words)


def lag_stream(seed, n, lags, *, delay=20, jitter=3.0, nim=3, nim_first=0.0, extra=True,
               t0=5_000_000):
    """S1 words and NIM words (true time t + delay, fine field moved by a lag
    drawn from ``lags``) in stream order; S2 words in between."""
    rng = random.Random(seed)
    t = t0
    words = []
    for _ in range(n):
        t += int(rng.expovariate(1 / 3000.0)) + 50
        co, fi = fields(t)
        tn = t + delay + int(round(rng.gauss(0, jitter)))
        lagv = rng.choice(lags)
        nco, _ = fields(tn)
        nw = (nim, 12, nco, (tn + lagv) & FINE_MASK)
        sw = (1, 30, co, fi)
        words.extend([nw, sw] if rng.random() < nim_first else [sw, nw])
        if extra and rng.random() < 0.3:
            words.append((2, 25, *fields(t + 5)))
    return words


def lag_cases():
    c = []
    c.append(lag("clean", lag_stream(1, 300, [0])))
    c.append(lag("faulted_153521", lag_stream(2, 300, [-153521])))
    c.append(lag("faulted_881", lag_stream(3, 300, [-881])))
    c.append(lag("ambiguous", lag_stream(4, 300, [-881] * 3 + [0] * 2)))
    c.append(lag("dominant_with_runner_up", lag_stream(5, 300, [-881] * 5 + [0] * 1)))
    c.append(lag("too_few", lag_stream(6, 30, [-881])))
    c.append(lag("wrap_at_zero", lag_stream(7, 200, [0], delay=0, jitter=4.0)))
    c.append(lag("wrap_at_half", lag_stream(8, 200, [(1 << 19) - 20], delay=20, jitter=4.0)))
    c.append(lag("nim_before_s1_next_ref", lag_stream(9, 200, [-881], nim_first=0.7)))
    c.append(lag("nim_before_s1_no_next_ref", lag_stream(9, 200, [-881], nim_first=0.7),
                 next_ref=0))
    # NIM true time in the next coarse tick of S1: no same-coarse S1 for many
    c.append(lag("coarse_mismatch", lag_stream(10, 300, [0], delay=9000, jitter=50.0)))
    c.append(lag("no_s1", [w for w in lag_stream(11, 100, [0]) if w[0] != 1]))
    # an earlier S1 of the same coarse field behind an S1 of another one is not used:
    # NIM(c=5) after S1(c=5), S1(c=6) has no pair (the next S1 is c=7); NIM(c=7) pairs
    # with the S1 after it; NIM(c=6) after S1(c=7) has none
    c.append(lag("same_coarse_s1_hidden", [(1, 30, 5, 100), (1, 30, 6, 200), (3, 12, 5, 130),
                                           (3, 12, 7, 290), (1, 30, 7, 300), (3, 12, 6, 230),
                                           (1, 30, 8, 400), (3, 12, 8, 420)], min_pairs=1))
    c.append(lag("same_coarse_s1_hidden_no_next", [(1, 30, 5, 100), (1, 30, 6, 200),
                                                   (3, 12, 5, 130), (3, 12, 7, 290),
                                                   (1, 30, 7, 300), (1, 30, 8, 400),
                                                   (3, 12, 8, 420)], min_pairs=1, next_ref=0))
    c.append(lag("tie_two_equal_modes", lag_stream(12, 200, [-881, 300])))
    c.append(lag("nominal_set", lag_stream(13, 200, [-881]), nominal=-861, tol=30.0))
    c.append(lag("tol_fraction", lag_stream(14, 200, [-40, 0, 40], jitter=1.0), tol=12.7,
                 min_pairs=20, dominance=1.0))
    return c


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", type=Path, default=BUILD)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--reco", type=Path, default=RECO,
                    help="the reco_testbeam checkout whose headers to use (read only)")
    ap.add_argument("--reco-commit", default=None,
                    help="the worktree's HEAD (git cannot resolve it in the container)")
    args = ap.parse_args(argv)
    args.build.mkdir(parents=True, exist_ok=True)
    exe = args.build / "nim_pairing_golden"
    cmd = ["g++", "-std=c++20", "-O1", "-Wall", "-Wextra",
           *(f"-I{p}" for p in includes(args.reco)), str(HERE / "nim_pairing_golden.cpp"), "-o", str(exe)]
    subprocess.run(cmd, check=True)
    text = "".join(pair_cases() + lag_cases()) + "END\n"
    (args.build / "cases.txt").write_text(text)
    res = subprocess.run([str(exe)], input=text, capture_output=True, text=True, check=True)
    cases = json.loads(res.stdout)
    commit = args.reco_commit
    if commit is None:
        try:
            commit = subprocess.run(["git", "-C", str(args.reco), "rev-parse", "HEAD"], check=True,
                                    capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unknown"
    doc = {
        "about": "PIPSMSMANimPairing::Pair and PISMAFineOffset::Scan (lag role) on fixed "
                 "cases; tests/cpp/generate_nim_pairing_golden.py writes it",
        "reco_commit": commit,
        "headers": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in headers(args.reco)},
        "ids": {"tot": 2004, "nim": 2024, "other": 9999},
        "cases": cases,
    }
    # one case per line: compact, and a regenerated file diffs case by case
    head = json.dumps({k: v for k, v in doc.items() if k != "cases"}, indent=1)[:-2]
    body = ",\n".join(json.dumps(c, separators=(",", ":")) for c in cases)
    args.out.write_text(f'{head},\n "cases": [\n{body}\n]}}\n')
    print(f"{len(cases)} cases -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
