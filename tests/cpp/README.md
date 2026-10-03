# NIM pairing golden file

`tests/data/nim_pairing_golden.json` pins `mdqm.plugins.sma_nim` to the reco
headers it ports. It holds fixed hit lists (ties, chains, echoes, empty sides,
window edges, frame edges, offsets, random streams) and synthetic SMA banks for
the lag vote, together with what the real headers make of them:

- `PIPSMSMANimPairing.hh` (`Prepare`, `Pair` with diagnostics, the wide dt
  with a budget);
- `PISMAFineOffset.hh` and `PISMAWord.hh` (`Scan` with one lag channel).

The headers come from the `feature/sma-nim-pairing` worktree at
`scratch/worktrees/reco_testbeam-sma-nim-pairing`. They are read only. The
file records their commit and SHA-256.

`tests/test_sma_nim.py` compares against the file with numpy and pytest only.
No compiler is needed to run the tests.

## Regenerating

Regenerate when the headers change. Run this from the workspace root on the
host:

```bash
docker exec -u 1000:1000 testbeam-midas bash -lc \
  "cd /workdir/wavedream-frontends/wavedream-midas-dqm && \
   python3 tests/cpp/generate_nim_pairing_golden.py \
   --reco-commit $(git -C scratch/worktrees/reco_testbeam-sma-nim-pairing rev-parse HEAD)"
```

- `generate_nim_pairing_golden.py` uses only the standard library. It writes
  the cases as text, compiles `nim_pairing_golden.cpp` with
  `g++ -std=c++20`, runs it, and writes the JSON with one case per line.
- The binary and `cases.txt` go to `scratch/sma-nim-dqm/cpp-build/` (change
  this with `--build`), never into the repo.
- `--reco <checkout>` takes the headers from another reco_testbeam checkout
  (the default is the scratch worktree, which goes away after the merge).
- When the default worktree exists, `test_golden_headers_unchanged` compares
  its headers with the hashes in the file. It skips, with a message, when they
  differ or are missing: a skip there means "regenerate", not "broken".
- The cases are seeded, so a rerun on unchanged headers writes the same file.
  Look at the diff before committing a new one.
- Use `-u 1000:1000` so nothing in the tree ends up owned by root.
- The commit comes from the host because the worktree's `.git` points at a host
  path, which git cannot resolve inside the container.
