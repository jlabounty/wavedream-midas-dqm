"""Store raw readout frames of runs the golden set does not cover.

Only the H000 words, no reference values: these frames pin plugin behaviour
(classification, shift check), not the decoder. Run once in ``testbeam-midas``:

    docker exec -u 1000:1000 testbeam-midas bash -lc \\
        'cd /workdir/wavedream-frontends/wavedream-midas-dqm && \\
         PYTHONPATH=src:/software/midas/python python3 tests/generate_sma_extra_frames.py'

* ``sma_run00342_frames.npz``: a coarse-shift-3 run (the firmware before the
  RF gate), frames 1 and 2, each cut to its first 8000 words (still
  contiguous stream data, ~1100 S1 words each).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ONLINE = Path("/workdir/scratch/online")
SETS = (("sma_run00342_frames.npz", "run00342.mid.lz4", 3, (1, 2), 8000),)


def main():
    import midas.file_reader

    from mdqm.plugins import sma_words as W

    for out, fname, shift, wanted, max_words in SETS:
        arrays, labels = {"shift": np.array(shift)}, []
        k = 0
        for ev in midas.file_reader.MidasFile(str(ONLINE / fname), use_numpy=True):
            if ev.header.event_id != W.EVID_READOUT or ev.get_bank(W.BANK) is None:
                continue
            if k in wanted:
                w = W.words_from_bank(np.asarray(ev.get_bank(W.BANK).data))
                arrays[f"f{len(labels)}_words"] = np.array(w[:max_words])
                labels.append(f"{fname}#{k} serial {ev.header.serial_number} words "
                              f"{min(w.size, max_words)}/{w.size}")
            k += 1
            if k > max(wanted):
                break
        arrays["labels"] = np.array(labels)
        path = HERE / "data" / out
        np.savez_compressed(path, **arrays)
        print(f"{path}: {len(labels)} frames, {path.stat().st_size / 1024:.0f} kB")


if __name__ == "__main__":
    main()
