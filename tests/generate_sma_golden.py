"""Write the SMA golden files: real readout frames plus what the references make of them.

Run once, inside the ``testbeam-midas`` container (it needs lz4 and g++):

    docker exec -u 1000:1000 testbeam-midas bash -lc \
        'cd /workdir/wavedream-frontends/wavedream-midas-dqm && \
         python3 tests/generate_sma_golden.py'

The expected values come from the references ``sma_words.py`` vendors, never
from ``mdqm``:

* ``psm-analysis-josh-2026/sma-tot-vs-wd/sma_reader.py`` (imported from its
  checkout, read only): ``iter_frames``, ``decode``/``time_of``,
  ``burst_phase_many`` (both rules), ``gate_veto``, ``pair_frame``
  (``match_to_gates`` of S2-S5, ``pulse_offset``);
* ``main/reco_testbeam/pi_midas/include/PISMAWord.hh``, compiled into a tiny
  program: ``FineCoarseDiffNs``, ``FineCoarseXor``, ``TimeNs`` and
  ``Extent`` of every frame.

``tests/test_sma_golden.py`` compares ``mdqm.plugins.sma_words`` against the
files on the host with numpy and pytest only.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORKDIR = Path("/workdir")
PSM = WORKDIR / "psm-analysis-josh-2026" / "sma-tot-vs-wd"
INCLUDE = WORKDIR / "main" / "reco_testbeam" / "pi_midas" / "include"
ONLINE = WORKDIR / "scratch" / "online"
SHIFT = 14
S1, RF = 1, 6
OTHERS = {"S2": 2, "S3": 3, "S4": 4, "S5": 5}

#: (output file, [(input file, frame indices)])
SETS = (
    ("sma_run00682_frames.npz", [("run00682_00000.mid.lz4", (0, 1, 2, 3)),
                                 ("run00682_00005.mid.lz4", (0, 1))]),
    ("sma_run01008_frame.npz", [("run01008_00001.mid.lz4", (0,))]),
)
#: Words kept of each frame. The 40000-word frames of the current firmware are
#: cut to their first half to keep the files small; they are still real,
#: contiguous stream data.
MAX_WORDS = 20000
#: Stored as a SHA-256 digest only: exact bit slices of the words, or equal
#: to an array stored in full. The test hashes its own array the same way.
DIGEST_ONLY = ("ch", "tot", "fine", "coarse", "order", "cpp_time")

CPP = r"""
#include "PISMAWord.hh"
#include <cstdio>
#include <cstdint>
#include <vector>
// stdin: u32 n, u32 shift, n x u32 coarse, n x u32 fine
// stdout: n x i64 FineCoarseDiffNs, n x u32 FineCoarseXor, n x u64 TimeNs,
//         i64 Extent.first, i64 Extent.last (of the TimeNs, stream order)
int main() {
    std::uint32_t n = 0, shift = 0;
    if (std::fread(&n, 4, 1, stdin) != 1 || std::fread(&shift, 4, 1, stdin) != 1) return 1;
    std::vector<std::uint32_t> c(n), f(n);
    if (std::fread(c.data(), 4, n, stdin) != n || std::fread(f.data(), 4, n, stdin) != n) return 2;
    std::vector<std::int64_t> d(n);
    std::vector<std::uint32_t> x(n);
    std::vector<std::uint64_t> t(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        d[i] = PISMAWord::FineCoarseDiffNs(c[i], f[i], shift);
        x[i] = PISMAWord::FineCoarseXor(c[i], f[i], shift);
        t[i] = PISMAWord::TimeNs(c[i], f[i], shift);
    }
    std::int64_t ext[2] = {0, 0};
    if (n) {
        const auto e = PISMAWord::Extent(t, PISMAWord::TimeBits(shift));
        ext[0] = e.first;
        ext[1] = e.last;
    }
    std::fwrite(d.data(), 8, n, stdout);
    std::fwrite(x.data(), 4, n, stdout);
    std::fwrite(t.data(), 8, n, stdout);
    std::fwrite(ext, 8, 2, stdout);
    return 0;
}
"""


def build_cpp(tmp: Path) -> Path:
    src = tmp / "pisma.cpp"
    exe = tmp / "pisma"
    src.write_text(CPP)
    subprocess.run(["g++", "-std=c++17", "-O2", f"-I{INCLUDE}", str(src), "-o", str(exe)],
                   check=True)
    return exe


def run_cpp(exe: Path, coarse, fine, shift):
    n = coarse.size
    inp = (np.array([n, shift], dtype="<u4").tobytes()
           + coarse.astype("<u4").tobytes() + fine.astype("<u4").tobytes())
    out = subprocess.run([str(exe)], input=inp, capture_output=True, check=True).stdout
    o = 0
    diff = np.frombuffer(out, "<i8", n, o)
    o += 8 * n
    xor = np.frombuffer(out, "<u4", n, o)
    o += 4 * n
    tns = np.frombuffer(out, "<u8", n, o)
    o += 8 * n
    ext = np.frombuffer(out, "<i8", 2, o)
    return diff, xor, tns, ext


def digest(a) -> str:
    """SHA-256 of an array's dtype, shape and bytes (tests/test_sma_golden.py repeats it)."""
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(f"{a.dtype.str}{a.shape}".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def frame_expectations(R, words, exe, prefix):
    out = {f"{prefix}words": words}
    d = R.decode(words, SHIFT)
    for k in ("ch", "tot", "fine", "coarse", "time"):
        out[f"{prefix}{k}"] = d[k]
    out[f"{prefix}n_filler"] = np.array(d["n_filler"])
    out[f"{prefix}n_pixel"] = np.array(d["n_pixel"])
    # The reference's frame processing (sma_reader.process): sort by time, then pair.
    order = np.argsort(d["time"], kind="stable")
    ch, tot, t = d["ch"][order], d["tot"][order], d["time"][order]
    out[f"{prefix}order"] = order
    rf, g = t[ch == RF], t[ch == S1]
    for rule in ("last", "dqm"):
        n, valid, phase, period, lo, pulse_t = R.burst_phase_many(rf, g, rule=rule)
        for k, v in (("n", n), ("valid", valid), ("phase", phase), ("period", period),
                     ("lo", lo), ("pulse_t", pulse_t)):
            out[f"{prefix}bpm_{rule}_{k}"] = v
    vetoed, gap = R.gate_veto(g)
    out[f"{prefix}veto"] = vetoed
    out[f"{prefix}veto_gap"] = gap
    p = R.pair_frame(ch, t, S1, RF, OTHERS, tot=tot, ref_ch=OTHERS["S2"])
    out[f"{prefix}off0"] = p["off0"]
    for rule in ("last", "dqm"):
        for name in OTHERS:
            o = p[rule]["other"][name]
            out[f"{prefix}pair_{rule}_{name}_dt"] = o["dt"]
            out[f"{prefix}pair_{rule}_{name}_phase"] = o["phase"]
    diff, xor, tns, ext = run_cpp(exe, d["coarse"], d["fine"], SHIFT)
    out[f"{prefix}cpp_diff"] = diff
    out[f"{prefix}cpp_xor"] = xor
    out[f"{prefix}cpp_time"] = tns
    out[f"{prefix}cpp_extent"] = ext
    for k in DIGEST_ONLY:
        out[f"{prefix}{k}_sha256"] = np.array(digest(out.pop(f"{prefix}{k}")))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--psm", type=Path, default=PSM, help="sma-tot-vs-wd checkout (read only)")
    ap.add_argument("--online", type=Path, default=ONLINE, help="directory of the run files")
    ap.add_argument("--out", type=Path, default=HERE / "data")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(a.psm))
    import sma_reader as R  # noqa: E402  (the reference, only here)

    a.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        exe = build_cpp(Path(tmp))
        for name, sources in SETS:
            arrays = {"shift": np.array(SHIFT)}
            labels = []
            for fname, wanted in sources:
                last = max(wanted)
                for fi, (serial, _ts, words) in enumerate(R.iter_frames(a.online / fname,
                                                                        evt_max=last + 1)):
                    if fi not in wanted:
                        continue
                    prefix = f"f{len(labels)}_"
                    labels.append(f"{fname}#{fi} serial {serial} words "
                                  f"{min(words.size, MAX_WORDS)}/{words.size}")
                    arrays.update(frame_expectations(R, np.array(words[:MAX_WORDS]), exe,
                                                     prefix))
            arrays["labels"] = np.array(labels)
            path = a.out / name
            np.savez_compressed(path, **arrays)
            print(f"{path}: {len(labels)} frames, {path.stat().st_size / 1024:.0f} kB")


if __name__ == "__main__":
    main()
