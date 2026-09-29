"""Write the MuPix pixel-word golden file: what the references make of real pixel words.

Run once, inside the ``testbeam-midas`` container (it needs g++, and scipy for
the timewalk module):

    docker exec -u 1000:1000 testbeam-midas bash -lc \
        'cd /workdir/wavedream-frontends/wavedream-midas-dqm && \
         /workdir/scratch/sma-dqm-standalone/venv/bin/python tests/generate_mupix_golden.py'

The words are the real readout frames already stored in ``tests/data`` (runs
682 and 1008, see ``generate_sma_golden.py``). The expected values come from
the references ``sma_words.pixel_decode`` follows, never from ``mdqm``:

* ``main/reco_testbeam/pi_midas/include/PIMuPixWord.hh``, compiled into a tiny
  program: ``IsPixel``, ``ChipId``, ``Col``, ``Row``, ``Ts2``, ``Time`` and
  ``Tot(word, 5)`` of every word of the bank (the reco's decoder,
  ``PITMidasMusip.cpp`` ``pixelhit``, is built on these);
* ``psm-analysis-josh-2026/sma-tot-vs-wd/mupix_phase.py`` ``pixel_words``
  (chip, col, row, time in ns) and ``mupix-timewalk/timewalk_lib.py``
  ``pixel_tot`` (the real ToT), imported from the checkout, read only.

The generator checks that the C++ and the two Python references agree with
each other before it writes anything. ``tests/test_mupix_golden.py`` compares
``mdqm.plugins.sma_words`` against the file with numpy and pytest only.
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
PSM = WORKDIR / "psm-analysis-josh-2026"
INCLUDE = WORKDIR / "main" / "reco_testbeam" / "pi_midas" / "include"
SOURCES = ("sma_run00682_frames.npz", "sma_run01008_frame.npz")
TS2_SHIFTS = (5, 4)
#: The first HEAD pixel words of each frame are stored in full (readable
#: failures); every array is also stored as a SHA-256 digest.
HEAD = 40

CPP = r"""
#include "PIMuPixWord.hh"
#include <cstdio>
#include <cstdint>
#include <vector>
// stdin: u32 n, u32 ts2Shift, n x u64 words
// stdout per word: u8 is_pixel, u8 chip, u8 col, u8 row, u8 ts2, u8 tot, u64 time
int main() {
    std::uint32_t n = 0, s = 0;
    if (std::fread(&n, 4, 1, stdin) != 1 || std::fread(&s, 4, 1, stdin) != 1) return 1;
    std::vector<std::uint64_t> w(n);
    if (std::fread(w.data(), 8, n, stdin) != n) return 2;
    for (std::uint32_t i = 0; i < n; ++i) {
        const std::uint64_t x = w[i];
        std::uint8_t b[6] = {
            static_cast<std::uint8_t>(PIMuPixWord::IsPixel(x)),
            static_cast<std::uint8_t>(PIMuPixWord::ChipId(x)),
            static_cast<std::uint8_t>(PIMuPixWord::Col(x)),
            static_cast<std::uint8_t>(PIMuPixWord::Row(x)),
            static_cast<std::uint8_t>(PIMuPixWord::Ts2(x)),
            static_cast<std::uint8_t>(PIMuPixWord::Tot(x, s))};
        const std::uint64_t t = PIMuPixWord::Time(x);
        std::fwrite(b, 1, 6, stdout);
        std::fwrite(&t, 8, 1, stdout);
    }
    return 0;
}
"""


def digest(a) -> str:
    """SHA-256 of an array's dtype, shape and bytes (tests/test_mupix_golden.py repeats it)."""
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(f"{a.dtype.str}{a.shape}".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def build_cpp(tmp: Path) -> Path:
    src, exe = tmp / "pimupix.cpp", tmp / "pimupix"
    src.write_text(CPP)
    subprocess.run(["g++", "-std=c++20", "-O2", f"-I{INCLUDE}", str(src), "-o", str(exe)],
                   check=True)
    return exe


def run_cpp(exe: Path, words, ts2_shift):
    w = np.ascontiguousarray(words, dtype="<u8")
    inp = np.array([w.size, ts2_shift], dtype="<u4").tobytes() + w.tobytes()
    out = subprocess.run([str(exe)], input=inp, capture_output=True, check=True).stdout
    rec = np.frombuffer(out, dtype=np.dtype([("b", "u1", 6), ("t", "<u8")]), count=w.size)
    b = rec["b"]
    return {"is_pixel": b[:, 0].astype(bool), "chip": b[:, 1].copy(), "col": b[:, 2].copy(),
            "row": b[:, 3].copy(), "ts2": b[:, 4].copy(), "tot": b[:, 5].copy(),
            "tick": rec["t"].astype(np.int64)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--psm", type=Path, default=PSM, help="psm-analysis checkout (read only)")
    ap.add_argument("--out", type=Path, default=HERE / "data" / "mupix_golden.npz")
    a = ap.parse_args(argv)
    sys.path[:0] = [str(a.psm / "sma-tot-vs-wd"), str(a.psm / "mupix-timewalk")]
    import mupix_phase as M     # noqa: E402  (the references, only here)
    import timewalk_lib as TW   # noqa: E402

    arrays: dict = {}
    labels = []
    with tempfile.TemporaryDirectory() as tmp:
        exe = build_cpp(Path(tmp))
        for src in SOURCES:
            with np.load(HERE / "data" / src) as z:
                frames = [(i, str(z["labels"][i]), z[f"f{i}_words"])
                          for i in range(len(z["labels"]))]
            for i, label, words in frames:
                p = f"f{len(labels)}_"
                labels.append(f"{src}: {label}")
                c = run_cpp(exe, words, TS2_SHIFTS[0])
                m = c["is_pixel"] & (words != np.uint64(0xFFFFFFFFFFFFFFFF))
                py = M.pixel_words(words)
                pix = words[m]
                # The references agree with each other, or nothing is written.
                assert np.array_equal(py["chip"], c["chip"][m]), label
                assert np.array_equal(py["col"], c["col"][m]), label
                assert np.array_equal(py["row"], c["row"][m]), label
                assert np.array_equal(py["t"], c["tick"][m] * 8), label
                exp = {"word_index": np.flatnonzero(m).astype(np.uint32),
                       "chip": c["chip"][m], "col": c["col"][m], "row": c["row"][m],
                       "ts2": c["ts2"][m], "tick": c["tick"][m], "time": py["t"]}
                for s in TS2_SHIFTS:
                    cs = run_cpp(exe, pix, s)
                    tw = TW.pixel_tot(pix, s).astype(np.uint8)
                    assert np.array_equal(tw, cs["tot"]), (label, s)
                    exp[f"tot{s}"] = cs["tot"]
                # The words themselves stay in their own file: frame i of `src`.
                arrays[f"{p}source"] = np.array(src)
                arrays[f"{p}index"] = np.array(i)
                arrays[f"{p}n_pixel"] = np.array(int(m.sum()))
                for k, v in exp.items():
                    arrays[f"{p}{k}_sha256"] = np.array(digest(v))
                    arrays[f"{p}{k}_head"] = np.asarray(v)[:HEAD]
    arrays["labels"] = np.array(labels)
    arrays["ts2_shifts"] = np.array(TS2_SHIFTS)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **arrays)
    print(f"{a.out}: {len(labels)} frames, "
          f"{sum(int(arrays[f'f{i}_n_pixel']) for i in range(len(labels)))} pixel words, "
          f"{a.out.stat().st_size / 1024:.0f} kB")


if __name__ == "__main__":
    main()
