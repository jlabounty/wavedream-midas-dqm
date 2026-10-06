#!/usr/bin/env python3
"""Time the SMA DQM per-frame pipeline (``mdqm.plugins.sma_words``) on real frames.

Reads readout frames (event 301, bank H000) with ``midas.file_reader`` and
``use_numpy=True``, so each bank arrives as the ``uint32`` array the live
analyzer gets, holds them in memory, then times the whole per-frame chain on
one core: words view, decode, fine vs coarse, stale filter, sort, channel
split, fine-bit and mismatch-bit counts, shift scan, RF phase, coincidences,
delayed pairs and seeds. It also prints what the frames look like (stale
words, fine/coarse mismatch per channel, RF valid fraction).

Gate (docs plan): >= 100 frames/s of 40000-word frames on one core.

Channel roles (``--roles``): a file from before run 1015 (its name's run
number) is timed with the old cabling (S1..S5 on ch 1-5, RF 6, current 7,
delayed 8-10: ``tests/sma_layouts.OLD_LAYOUT``) and no NIM copies, so the gate
stays comparable with the numbers from before the 1015 defaults; a run >= 1015
and the synthetic frames with the 1015 defaults. ``--roles old`` / ``1015``
forces one for every input.

``--nim`` adds the TOT + NIM pairing and lag votes of the plugin
(``sma.pair_frame``, the defaults' ``/DQM/SMA/NIM`` with ``merge`` on, every
frame voted: the worst case) to the 1015-cabled inputs; the analysis then runs
on the merged counters. ``--synthetic K`` times K dense run-1015-cabled
synthetic frames (~39000 words, ``tests/sma_layouts``) instead of, or besides,
files; with it the pairing is on::

    PYTHONPATH=src python3 scripts/bench-sma.py --synthetic 40

Run inside testbeam-midas (needs lz4 and the MIDAS python):

    docker exec -u 1000:1000 testbeam-midas bash -lc \\
        'cd /workdir/wavedream-frontends/wavedream-midas-dqm && \\
         PYTHONPATH=src:/software/midas/python taskset -c 2 python3 scripts/bench-sma.py \\
         /workdir/scratch/online/run01008_00001.mid.lz4 \\
         /workdir/scratch/online/run00682_00005.mid.lz4'
"""

from __future__ import annotations

import os

# One core: no BLAS threads behind numpy's back.
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from mdqm.plugins import sma as P  # noqa: E402
from mdqm.plugins import sma_words as W  # noqa: E402

GATE_FPS = 100.0
GATE_WORDS = 40000
#: The cabling before run 1015 (tests/sma_layouts.OLD_ROLES; that module is
#: not importable from an installed checkout's scripts).
OLD_ROLES = W.Roles(s1=1, counters=(1, 2, 3, 4, 5), rf=6, current=7, delayed=(8, 9, 10))
FIRST_1015_RUN = 1015


def run_of(path) -> int | None:
    """The run number in a file name like run01008_00001.mid.lz4; None if absent."""
    import re

    m = re.search(r"run0*(\d+)", Path(path).name)
    return int(m.group(1)) if m else None


def read_frames(path, n_frames, skip=0):
    """``n_frames`` H000 banks (uint32 arrays) after skipping ``skip`` frames."""
    import midas.file_reader

    out = []
    seen = 0
    for ev in midas.file_reader.MidasFile(str(path), use_numpy=True):
        if ev.header.event_id != W.EVID_READOUT:
            continue
        bank = ev.get_bank(W.BANK)
        if bank is None:
            continue
        seen += 1
        if seen <= skip:
            continue
        out.append(np.array(bank.data))
        if len(out) >= n_frames:
            break
    return out


def pipeline(bank, shift, roles, cuts, nim=None):
    """Everything WP3 computes per frame, bar the histogram fills; with `nim`
    (an sma.Config) the TOT + NIM pairing too, the analysis on merged counters.
    The plugin's zero-word check (Cuts/drop zero words, on by default) first."""
    bank, _n_zero, _pos = P.drop_zero_words(bank)
    if bank is None:
        return None
    fr = W.prepare_frame(bank, shift, cuts.stale_gap_ns, cuts.latch_margin_ns)
    occupancy = W.per_channel_bit_counts(fr.ch, fr.fine)
    bad = ~fr.consistent
    xor = W.fine_coarse_xor(fr.coarse[bad], fr.fine[bad], shift)
    mismatch_bits = W.per_channel_bit_counts(fr.ch[bad], xor, W.shared_bits(shift))
    if nim is not None:
        nf = P.pair_frame(fr, nim)
        if nf is not None:
            fr.counter_hits = nf.hits or None
    a = W.analyse_frame(fr, roles, cuts)
    return fr, a, occupancy, mismatch_bits


def time_steps(banks, shift, roles, cuts, nim=None):
    """Per-step mean time [ms], for the profile."""
    steps = {}

    def add(name, dt):
        steps[name] = steps.get(name, 0.0) + dt

    for b in banks:
        t0 = time.perf_counter()
        b, _n_zero, _pos = P.drop_zero_words(b)
        tz = time.perf_counter()
        add("zero words", tz - t0)
        if b is None:
            continue
        t0 = tz
        w = W.words_from_bank(b)
        d = W.decode(w, shift)
        t1 = time.perf_counter()
        add("decode", t1 - t0)
        diff = W.fine_coarse_diff_ns(d["coarse"], d["fine"], shift)
        W.fine_coarse_consistent(diff, shift, cuts.latch_margin_ns)
        t2 = time.perf_counter()
        add("fine-coarse", t2 - t1)
        keep = W.stale_mask(d["time"], cuts.stale_gap_ns)
        t3 = time.perf_counter()
        add("stale+sort", t3 - t2)
        W.split_by_channel(d["ch"][keep])
        t4 = time.perf_counter()
        add("split", t4 - t3)
        W.per_channel_bit_counts(d["ch"], d["fine"])
        t5 = time.perf_counter()
        add("bit counts", t5 - t4)
        fr = W.prepare_frame(b, shift, cuts.stale_gap_ns, cuts.latch_margin_ns)
        t6 = time.perf_counter()
        add("prepare_frame (all above but bits)", t6 - t5)
        if nim is not None:
            nf = P.pair_frame(fr, nim)
            if nf is not None:
                fr.counter_hits = nf.hits or None
            t6b = time.perf_counter()
            add("TOT+NIM pair_frame", t6b - t6)
            t6 = t6b
        W.analyse_frame(fr, roles, cuts)
        t7 = time.perf_counter()
        add("analyse_frame", t7 - t6)
    return {k: 1e3 * v / len(banks) for k, v in steps.items()}


def observe(banks, shift, roles, cuts, label, nim=None):
    words = trig = stale = 0
    per_ch = np.zeros(W.N_CHANNELS, dtype=np.int64)
    bad_ch = np.zeros(W.N_CHANNELS, dtype=np.int64)
    n_s1 = n_valid = n_veto = 0
    shift_counts = np.zeros(len(cuts.shift_scan), dtype=np.int64)
    spans = []
    tot_bad = np.zeros(W.N_CHANNELS, dtype=np.int64)
    for b in banks:
        got = pipeline(b, shift, roles, cuts, nim)
        if got is None:                  # nothing but zero words
            continue
        fr, a, _occ, _mb = got
        words += fr.n_words
        trig += fr.n_trigger
        stale += int(fr.stale_per_ch.sum())
        per_ch += np.bincount(fr.ch.astype(np.intp), minlength=W.N_CHANNELS)
        bad_ch += np.bincount(fr.ch[~fr.consistent].astype(np.intp), minlength=W.N_CHANNELS)
        tot_bad += np.bincount(fr.ch[fr.tot >= cuts.tot_corrupt_min].astype(np.intp),
                               minlength=W.N_CHANNELS)
        n_s1 += a.t_s1.size
        n_valid += int(a.rf_valid.sum())
        n_veto += int(a.rf_vetoed.sum())
        shift_counts += a.shift_counts
        spans.append(fr.span_ns / 1e6)
    print(f"--- {label}: {len(banks)} frames, {words / len(banks):.0f} words/frame, "
          f"{trig / len(banks):.0f} trigger words/frame, median span {np.median(spans):.2f} ms, "
          f"stale words {stale}")
    for c in np.flatnonzero(per_ch):
        print(f"    ch{c:2d}: {per_ch[c] / len(banks):8.1f} words/frame, fine/coarse mismatch "
              f"{bad_ch[c] / per_ch[c]:6.1%}, "
              f"ToT>={cuts.tot_corrupt_min} {tot_bad[c] / per_ch[c]:6.2%}")
    n1 = max(n_s1, 1)
    print(f"    S1 {n_s1}: RF valid {n_valid / n1:.1%}, vetoed {n_veto / n1:.2%}")
    print("    shift scan (kept S1 words consistent): "
          + ", ".join(f"{s}: {c / max(n_s1, 1):.1%}" for s, c in
                      zip(cuts.shift_scan, shift_counts, strict=True)))


def bench(banks, shift, roles, cuts, repeat, nim=None):
    for b in banks[:5]:
        pipeline(b, shift, roles, cuts, nim)         # warm up
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        for b in banks:
            pipeline(b, shift, roles, cuts, nim)
        best = min(best, time.perf_counter() - t0)
    n_words = sum(W.words_from_bank(b).size for b in banks)
    return len(banks) / best, n_words / best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--nim", action="store_true",
                    help="add the TOT + NIM pairing (plugin defaults, merge on) to the "
                         "1015-cabled inputs")
    ap.add_argument("--roles", choices=("auto", "old", "1015"), default="auto",
                    help="channel roles: auto = old before run 1015 (file name), else 1015")
    ap.add_argument("--synthetic", type=int, default=0, metavar="K",
                    help="also time K dense 1015-cabled synthetic frames (pairing on)")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--skip", type=int, default=5,
                    help="frames skipped at the start of each file (subrun-0 stale replay)")
    ap.add_argument("--shift", type=int, default=W.DEFAULT_SHIFT)
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args(argv)
    cuts = W.Cuts()
    try:
        print(f"CPU affinity: {sorted(os.sched_getaffinity(0))}")
    except AttributeError:
        pass
    nim_cfg = P.parse_settings({"NIM": {"merge": True}})

    def roles_for(run):
        old = a.roles == "old" or (a.roles == "auto" and run is not None
                                   and run < FIRST_1015_RUN)
        return (OLD_ROLES, None) if old else (W.Roles(), nim_cfg if a.nim else None)

    sets = [(path.name, read_frames(path, a.frames, a.skip), *roles_for(run_of(path)))
            for path in a.files]
    if a.synthetic:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
        from sma_layouts import dense_1015

        sets.append((f"synthetic 1015 x{a.synthetic}",
                     [dense_1015(seed=k, shift=a.shift).view("<u4") for k in range(a.synthetic)],
                     W.Roles(), nim_cfg))
    if not sets:
        ap.error("give files and/or --synthetic K")
    failed = False
    for name, banks, roles, nim in sets:
        if not banks:
            print(f"{name}: no readout frames")
            continue
        wpf = np.mean([W.words_from_bank(b).size for b in banks])
        cab = "old roles" if roles == OLD_ROLES else "1015 roles"
        observe(banks, a.shift, roles, cuts,
                f"{name} ({cab}{', TOT+NIM' if nim else ''})", nim)
        steps = time_steps(banks, a.shift, roles, cuts, nim)
        print("    per step [ms/frame]: " + ", ".join(f"{k} {v:.2f}" for k, v in steps.items()))
        fps, wps = bench(banks, a.shift, roles, cuts, a.repeat, nim)
        verdict = ""
        if wpf >= 0.9 * GATE_WORDS:
            ok = fps >= GATE_FPS
            failed |= not ok
            verdict = f"  gate >= {GATE_FPS:.0f} frames/s: {'PASS' if ok else 'FAIL'}"
        print(f"    pipeline: {fps:.0f} frames/s, {wps / 1e6:.2f} M words/s "
              f"({1e3 / fps:.2f} ms/frame){verdict}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
