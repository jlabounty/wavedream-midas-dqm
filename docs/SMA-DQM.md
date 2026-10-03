# SMA online DQM: shifter guide

This guide takes you from an idle DAQ to a working SMA monitor, explains every
indicator on the two pages, and gives the manual path for when the monitor is
down. File and line references are to this repository (`src/mdqm/...`,
`pages/...`) as of 2026-09-29.

## What it is

The SMA board (MuSiP trigger/ToT board) sends one readout frame per MIDAS event:
event id 301, bank `H000`, a list of 64-bit words. The SMA DQM is one extra MIDAS
client, `sma_analyzer`, that reads frames from the SYSTEM buffer, decodes them,
fills histograms and a summary, and answers the two custom pages in mhttpd:

| Page (side menu) | What it shows |
|---|---|
| **SMAPlots** | status chips, flags, a per-channel table, and six tabs of plots: Health, ToT / corruption, Timing, RF / delayed, MuPix, Trends (last 10 min) |
| **SMAEvents** | the latest frame: *S1-seeded events* (every channel and both MuPix planes around the latest S1 hits) and *Whole-frame raster* (time vs channel for the whole frame, the MuPix planes as two more rows) |

The same bank also carries the MuPix pixel words (bit 63 clear, about 7300 of
a 40000-word run-1008 frame). The DQM decodes them too: which S1 hits have a
pixel hit on L1 and L2 at the right time, the pixel ToT and occupancy, and
from that an **SMA <-> MuPix time-sync monitor** (see
[MuPix](#mupix-the-pixel-words-in-the-sma-frames)).

It never slows the DAQ down: it reads the buffer without blocking the writer
(`GET_NONBLOCKING`, `Analyzer.register_event_requests` in
`src/mdqm/dqm/analyzer.py`).

**It is a live peek, not a lossless record, and it never uses more than its CPU
budget.** The analyzer measures its own CPU use and analyses only as many
frames per second as fit in `/DQM/SMA/Sampling/CPU budget %` (default **20 %
of one core**, everything included: reading the buffer, decoding, filling the
histograms, answering the pages). At high rate it analyses a sample and skips
the rest; a skipped frame is never read, so it costs nothing however fast the
DAQ runs. The pages always say how much was analysed (**analysed N % of
frames**, see below). Rates, efficiencies and other fractions are not biased by
the sampling; histogram counts are from the analysed sample.

It is a plugin of the same package as the WaveDREAM DQM and runs next to the
WaveDREAM analyzer (`wd_analyzer`) as a separate process with its own ODB tree:
`/DQM/SMA` for SMA, `/DQM/Analyzer` for WaveDREAM (`src/mdqm/plugins/sma.py:548-554`).

Measured on this laptop at one to ten times the rate of run 1008
(`docs/profile-sma-highrate.json`, section "Resource requirements"): the
analyzer stays at its budget at every rate; at today's rate and the default
budget it analyses most of the frames, at ten times the rate about one in ten.

## Start to finish from an idle DAQ

### 1. Install (once per machine)

```bash
cd wavedream-frontends/wavedream-midas-dqm
pip install -e .                                   # mdqm-analyzer, mdqm-register-pages, mdqm-sma-file
pip install -e ../wavedream-scalar-readout         # only needed for the WaveDREAM analyzer
```

`mdqm-sma-file` also needs `lz4` (for `.mid.lz4`) and `matplotlib` (for its PNG).
The MIDAS python package must be importable (`$MIDASSYS/python` on `PYTHONPATH`).

### 2. Register the pages (once, and again after the checkout moves)

```bash
mdqm-register-pages --experiment <EXPT>
```

This writes the `/Custom/SMAPlots` and `/Custom/SMAEvents` keys (and the
WaveDREAM pages) with absolute paths to this checkout
(`src/mdqm/install/manifest.py:117-119`). It is safe to run on every start; it
removes keys of its own that are no longer in the list (for example the old
`/Custom/SMA`). Options (`src/mdqm/install/register_pages.py:207-220`):
`--list`, `--dry-run`, `--check` (every key still points at a readable file),
`--remove`, `--prefix` (to avoid a name clash in a shared experiment).

### 3. Starting and stopping the analyzer

**Where it runs on pinky is not decided yet (TBD).** On the DAQ PC (pinky) or
any machine that can reach the experiment, start it with the script in this
checkout. It needs neither `pip install` nor wavedream-scalar-readout:

```bash
scripts/start-sma-analyzer.sh --experiment <EXPT> --python <python with numpy + midas>
scripts/start-sma-analyzer.sh --experiment <EXPT> --status     # client, nice, memory, tmux, log
scripts/start-sma-analyzer.sh --experiment <EXPT> --stop       # SIGTERM, waits up to 20 s
```

It runs the analyzer in a tmux session (`sma-analyzer`, or
`$WDS_TMUX_PREFIX-sma-analyzer` when that is set, e.g. `wds-sma-analyzer`;
`tmux attach -t sma-analyzer` to watch it) and copies its output to a log
(`~/.local/state/mdqm/<EXPT>-sma_analyzer.log`; `--log FILE` or `--log none`).
It refuses to start a second copy: if a client called `sma_analyzer` is attached
to the experiment, an analyzer with that client name is running on this machine
(for example one retrying after MIDAS went away), or the tmux session still has a
live process. Two analyzers would split the sampled frames between them. Before
starting it checks that the python can import numpy, midas and mdqm under the
memory limit, that the experiment's ODB answers, and whether the MIDAS library
exports `bm_skip_event` (it prints `skip_method`, see below). `--help` lists every
option; the useful ones:

| Option (environment fallback) | Default | What |
|---|---|---|
| `--experiment` (`MIDAS_EXPT_NAME`, `WDS_EXPT_NAME`) | none | MIDAS experiment; `MIDAS_EXPTAB` is honoured |
| `--python` (`MDQM_PYTHON`, `WDS_PYTHON`) | `python3` | runs `-m mdqm.dqm.analyzer` with `PYTHONPATH=<checkout>/src` |
| `--client` | `sma_analyzer` | MIDAS client name |
| `--mem-limit BYTES` | 1073741824 | `prlimit --as`; 0 = none |
| `--cpu-pin N` | off | `taskset -c N` |
| `--foreground` | | run in this terminal (debugging) |
| `--no-tmux` | | detached without tmux, output only to the log |
| `-- ARGS` | | passed to the analyzer, e.g. `-- --no-cpu-budget` |

The script always runs the analyzer as a low-priority guest. The command it
runs, which is also the fallback when the script cannot be used (`mdqm-analyzer`
instead of `python -m mdqm.dqm.analyzer` when the package is pip-installed):

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2 \
    PYTHONPATH=<checkout>/src nice -n 19 ionice -c3 prlimit --as=1073741824 \
    <python> -m mdqm.dqm.analyzer --experiment <EXPT> --plugin sma --client sma_analyzer
```

* The `env` settings: one BLAS/OpenMP thread and two malloc arenas. A pip-installed
  numpy with OpenBLAS otherwise starts a thread per core, each with an 8 MiB stack
  and a 64 MiB arena, and can then fail at import under the address-space limit.

* `nice -n 19`: the DAQ's own processes always get the CPU first.
* `ionice -c3`: idle I/O class (the analyzer writes nothing, but it costs
  nothing to be sure).
* `prlimit --as=1073741824`: a 1 GiB ceiling on its address space. Its virtual
  size is about 250 MB plus the SYSTEM buffer (it maps the buffer and the ODB),
  so 1 GiB is enough for a SYSTEM buffer up to about 600 MB; with a larger one
  raise the limit (`--mem-limit`) or leave `prlimit` out. Past the ceiling an
  allocation fails inside the analyzer instead of the machine starting to swap.
* Optionally `taskset -c N` in front to keep it on one core (`--cpu-pin N`).
* **Do not use a cgroup CPU quota** (systemd `CPUQuota=`, `docker --cpus` on a
  dedicated container, and so on) to hold it down. A quota freezes the process
  when it runs out, possibly while it holds the SYSTEM buffer lock, and then the
  writer (the DAQ) waits. The CPU budget is the limit; `nice` only orders it
  behind the DAQ.

**Once on pinky, before the first start**, check two things.

1. The MIDAS library exports the skip call:
   ```bash
   nm -D /home/pinky/packages/midas/lib/libmidas-c-compat.so | grep -E 'bm_skip_eventi|bm_get_buffer_leveliPi'
   ```
   It should print `_Z13bm_skip_eventi` (and `_Z19bm_get_buffer_leveliPi`). If
   `bm_skip_eventi` is missing, the analyzer still works: `dqm::status` shows
   `skip_method: "drain"` (instead of `"bm_skip_event"`) and it prints a line at
   start. It then drops skipped frames by reading them, at most 200 per skip. That
   costs a copy each, within the same CPU budget. `start-sma-analyzer.sh` makes
   the same check before starting and prints `skip_method bm_skip_event` (or
   warns about `drain`).
2. After a minute of running, the analyzer's virtual size is well under the
   limit: `scripts/start-sma-analyzer.sh --experiment <EXPT> --status` prints its
   VmSize and RLIMIT_AS. It was 210-255 MB here; it must stay well below 1 GiB.
   (Not `pgrep -f 'client sma_analyzer'`: that also matches the tmux server,
   whose command line contains the whole analyzer command.)

The CPU itself is bounded by the budget (`/DQM/SMA/Sampling/CPU budget %`), not
by these. `mdqm-analyzer --help` lists all options. `--max-event-size` defaults
to `/Experiment/MAX_EVENT_SIZE` (clamped to 8-64 MiB). A larger frame is counted
(`events_truncated` in `dqm::status`) and skipped, and the read buffer then grows
to fit such frames (up to 64 MiB), so MIDAS posts its "event truncated" error
once, not once per frame. Leave it running in a `tmux` window (what the start
script does) or under a service. On first start it creates `/DQM/SMA` with
default values ("seeded N settings key(s)"); it never overwrites a key that
already exists. The WaveDREAM analyzer is a separate command:
`mdqm-analyzer --experiment <EXPT> --plugin wavedream --client wd_analyzer`.

**In a restart script** (for example pinky's `restart-all.sh`, which runs the
WaveDREAM `start-analyzer.sh`): start it after the WaveDREAM analyzer, and stop it
in the stop phase before MIDAS goes down. With `WDS_PYTHON`, `WDS_EXPT_NAME` and
`WDS_TMUX_PREFIX` from `wdscalers-env.sh` in the environment the flags are
optional, but explicit is clearer:

```bash
# start phase, after start-analyzer.sh
"$WDS_DQM_PAGES_DIR/scripts/start-sma-analyzer.sh" --experiment bt2026 \
    --python /home/pinky/software/wdscalers-venv/bin/python
# stop phase
"$WDS_DQM_PAGES_DIR/scripts/start-sma-analyzer.sh" --experiment bt2026 --stop
```

A start that finds the analyzer already running exits 1 with a message; a
restart script that should tolerate that can test `--status` (exit 0 = running)
first.

Optional: make it startable from the MIDAS **Programs** page by setting
`/Programs/sma_analyzer/Start command` to
`<checkout>/scripts/start-sma-analyzer.sh --experiment <EXPT> --python <python>`
(it then runs on the mhttpd host, as mhttpd's user, which needs tmux).

To stop it: `scripts/start-sma-analyzer.sh --experiment <EXPT> --stop`,
`Ctrl-C` in its window, or `kill <pid>`. On SIGTERM or SIGINT it deregisters
from MIDAS and exits (the log then ends with "Received Ctrl-C, aborting..." and
"Midas shutdown": MIDAS's own handler does it). Stopping it loses the
accumulated plots.

### 4. Open the pages

In the experiment's mhttpd, pick **SMAPlots** or **SMAEvents** in the side menu
(`?cmd=custom&page=SMAPlots`). The first chip should say
**sma_analyzer connected**. With no run and no data the pages stay empty; that
is not an error.

### 5. Start a run

Nothing else to do. The plots clear themselves at every new run number
(`src/mdqm/dqm/analyzer.py:377-379`). Within a few seconds of beam, the chips show
frames/s, the live fraction and the shift check.

## The SMAPlots page, indicator by indicator

### Status chips (top row)

| Chip | Meaning | What to do |
|---|---|---|
| `sma_analyzer connected` (green) | the analyzer answered in the last poll | nothing |
| `analyzer throttled` (red) | the DAQ-health valve lowered its rate. For SMA this valve is off (no musip loss counter known yet, `sma.py:550-552`), so this should not appear | tell the DQM expert |
| `no analyzer` (red) + `last seen HH:MM:SS` (yellow) | the analyzer did not answer; the page greys out and everything shown is from the time given | see "Analyzer not connected" below |
| `run N` / `run N (stopped)` | run number the analyzer sees; yellow when no run is active | nothing |
| `frames A analysed · B offered · C stale` | A = frames analysed since the analyzer started, B = frames the DAQ sent (counted from the event serial numbers, so skipped frames are counted without being read), C = stale frames | nothing |
| `analysed X % of frames` (blue) | the share of the frames sent in the last 60 s that the analyzer analysed. Below 100 % is normal at high rate: it samples to stay within its CPU budget. Shown as `... since start` when no frame arrived in the last 60 s | nothing; see "Sampling" below |
| `CPU x % / budget y %` | the analyzer's own CPU (last 5 s, % of one core) and its budget. Yellow above 1.1 x the budget, which should not last more than a few seconds | if it stays yellow, tell the DQM expert |
| `N missed (serial gaps)` (yellow) | only with the development flag `--no-cpu-budget` (process everything): frames the analyzer never saw because the buffer overwrote them first. It keeps its count until the analyzer restarts | restart it without the flag (sampling is the normal mode) |
| `N suspect time base` (red) | frames whose hit times make no sense, almost always a wrong coarse shift | see the shift banner |
| `A analysed / B offered frames/s` | analysis rate and the rate the DAQ sends | compare B with the expected beam rate |
| `live X %` | fraction of time covered by frames (1.0 = no dead time between frames), measured on pairs of consecutive frames; `—` when no two consecutive frames were analysed | a sudden drop means frames are missing or short |
| `last frame N s ago` (yellow) | no frame for more than 5 s | check the SMA readout if a run is active |
| `shift 14: ok` (green) / `mismatch, best K` (red) / `no fit` (yellow) / `insufficient` | the coarse-shift self-check (below) | see the shift banner |

The table, flags and chips average over the last 60 s (`Self check/summary window s`).

### Sampling: what "analysed N %" means

Below the chips, when not every frame (or not every S1 hit) was analysed, a
grey note says so, for example:

> Counts are from the analysed sample: 45 % of frames (CPU budget 20 % of a
> core); S1-seeded plots use at most 2000 S1 hits per frame (56 % of S1 hits).
> Rates, fractions and efficiencies are unbiased.

* **Frames.** The analyzer gets `CPU budget %` of one core and fits as many
  frames into it as it can; the rest are skipped unread. Which frames are
  analysed has nothing to do with their content, so every rate (hits per second
  of time covered by the analysed frames), efficiency, mismatch fraction, ToT
  fraction and live fraction is the same as with every frame; only the number
  of entries in each histogram is smaller. The analyzer reads frames in pairs
  of consecutive frames, so the frame gap and the live fraction (which need a
  frame and the one sent just before it) are still measured. The rates use the
  second frame of each pair only: the first is the frame being written when the
  analyzer came back to the buffer, which favours long frames and would read
  the rate low around beam trips.
* **S1 hits.** A frame with more than `Cuts/max S1 per frame` (2000) kept S1
  hits gives the S1-seeded analyses (pattern, efficiency given S1, S2..S5 - S1,
  RF phase, delayed channels) to an evenly spread sample of 2000 of them. A
  run-1008 frame has about 3500. Word counts, per-channel rates, ToT,
  fine/coarse checks and stale words always use every hit.
* An info flag `sampling` repeats this in the flag list. It is information,
  not a warning.
* A low analysed fraction is expected at high rate. It only means the budget is
  too small if the plots fill too slowly to be useful; then the DQM expert can
  raise `/DQM/SMA/Sampling/CPU budget %` (it takes effect within 2 s and resets
  nothing).

### The red banner: coarse shift

The board writes each hit's time twice: a fine field and a coarse field equal to
`time >> shift`. The shift is a board setting that has changed between firmware
versions. If `/DQM/SMA/Coarse shift` does not match the board, every time-derived
number on the page is wrong.

The analyzer checks this continuously: over the last 30 s it counts how many S1
hits have consistent fine and coarse fields at shifts 3 and 12-16
(`sma.py:1011-1038`). When another shift beats the configured one by more than
20 points and reaches 90 %, with at least 1000 S1 words, the page shows:

> **Coarse shift 13 configured, 14 fits better (100.0 % vs 3.9 % of ... S1 words in 30 s) — set /DQM/SMA/Coarse shift = 14**

(flag `shift_mismatch`, `sma.py:1167-1175`; banner `pages/js/dqm-sma.js:576-615`).
A second red banner, **Time base: ...** (flag `time_base`, `sma.py:1176-1181`),
means frames kept less than half of their hits in one time cluster; it has the
same cause and the same fix.

What to do:

1. Click **Edit Coarse shift...** in the banner (or ODB page ->
   `/DQM/SMA/Coarse shift`) and set the value the banner names.
2. The banner clears within a few seconds (measured: 1 s). The plots restart
   from zero, which is expected (a message "rebuilt 45 histograms (counts reset)"
   appears in the MIDAS message bar).
3. Write it in the elog and **tell the SMA expert**: a shift change usually
   means a firmware change on the switching board.

For reference: shift 14 fits the recent runs checked (429, 682, 1008); run 342 and
earlier used 3, runs 367-375 used 15. A switching-board firmware change discussed
on 2026-09-24 would move it to 13 once deployed.

Before 1000 S1 words have been seen the chip says `insufficient` and a blue
`shift_unchecked` note says per-channel checks are waiting (`sma.py:1187-1190`).
`no fit` (flag `shift_no_fit`, `sma.py:1182-1186`) means no scanned shift works
for S1: suspect the S1 channel itself and call the SMA expert.

### Flags (below the banner)

Worst first. Codes as in `sma.py:1152-1232`:

| Flag | Severity | Meaning | What to do |
|---|---|---|---|
| `mismatch` on a channel | error above 50 %, warning above 5 % | that channel's fine and coarse time fields disagree: a timestamp fault on the board. Only shown once the shift check is `ok`, and only for S1-S5 and RF by default | compare with the known faults below; a new channel or a big change goes to the elog and the SMA expert |
| `tot_corrupt` | warning above 5 % | many hits with ToT code >= 250 (a corruption marker) | same |
| `stale_frames` | warning | some frames were old data (not this run's), left out of the plots. Common for the first frames after a run start | nothing unless it persists |
| `all_stale` | error | every frame in 60 s is stale: the board is sending old data, or S1's own timestamps are broken at every shift | call the SMA expert |
| `no_frames` | error (warning before the first frame) | no SMA frame for more than 5 s while a run is active | check that the SMA/musip readout is running and the SMA link is enabled |
| `efficiency_drop` | warning | a counter's efficiency given S1 fell by more than 10 points in the last 30 s compared with the last 10 min | check HV and cabling of that counter |
| `oversize` | warning | a frame had more than `Cuts/max words per frame` words and was not decoded | tell the SMA expert (the readout sent a huge frame) |
| `sampling` | info | not every frame was analysed (CPU budget): histogram counts are from the sample, rates and fractions are not affected | nothing |
| `settings` | warning | a value under `/DQM/SMA` was invalid and its default is used; the text names it. "Channel roles look pre-1015" means a NIM channel is also S1, a counter, the RF, `current` or a `delayed` channel: NIM is then off altogether | fix the ODB value; for "pre-1015", run the odbedit lines under [Settings](#settings-dqmsma) |
| `mupix_sync` | warning | fewer than 30 % of S1 hits have an L1 **and** an L2 pixel hit in time (accidentals taken out) for more than 30 s while S1 fires; or no pixel words at all | see [MuPix time sync](#mupix-time-sync-what-the-flag-means) |
| `mupix_unmapped` | warning | pixel hits on a chip id that is in neither `MuPix/L1 chips` nor `L2 chips` | the plane map does not match the FEB Mapping: tell the MuPix expert, then fix the ODB lists |
| `mupix_skipped` | info | a frame had more pixel hits than `MuPix/max pixel hits per frame`; only the latest were examined | nothing |
| `nim_missing` | warning | a counter's NIM copy (S*k*L) has no hits while its TOT channel has at least `min hits` | check the NIM cable, the discriminator power and `NIM/channels`; see [the four NIM flags](#the-four-nim-flags) |
| `nim_pairing` | error below 50 %, warning below 80 % | pair efficiency (paired / (paired + TOT-only)) is low. Only once the shift check is `ok`, and not for a counter whose TOT channel has a known timestamp fault | if the median NIM - TOT is large, measure the offset; if it is near 0, elog it and tell the SMA expert. See [the four NIM flags](#the-four-nim-flags) |
| `nim_offset` | warning | the median NIM - TOT (since the run start or the last settings change) is more than 5 ns from 0 after `NIM/offset ns`; the text gives the value to set. Same conditions as `nim_pairing` | set `NIM/offset ns`, see [measuring the offsets](#measuring-the-nim-offsets-on-the-first-clean-run) |
| `nim_lag` | warning | the NIM channel's lag state was "faulted" in more than half of the window's frames, after at least 3 frames voted "faulted" (`Self check/nim lag min votes`); only once the shift check is `ok`. The text says whether NIM-only hits were held back from the merge | elog it with the counter; tell the SMA expert (nothing here corrects it) |

**Known timestamp faults (as of 2026-09-28), measured with this DQM on replayed
runs** (channels as cabled before run 1015: S3 on ch 3, the proton current on ch 7):

| Channel | Run 682 | Run 1008 | Status |
|---|---|---|---|
| S5 (ch 5) | 95 % | 93 % | fine = t/2, a known board fault: expect the red S5 flag |
| S3 (ch 3) | 13.5 % | 1.3 % | varies by run |
| RF (ch 6) | 0 % | 31 % | varies by run |
| S2 (ch 2) | 2.7 % | 0.3 % | |
| S4 (ch 4) | 0.6 % | 0.1 % | |
| current (ch 7) | 0 % | 100 % | not flagged (not in `mismatch flag channels`) |

### The table

One row per channel: rate (hits per second of covered time), hits per frame,
fraction of hits with ToT >= 250, fine/coarse mismatch fraction, stale words, and
**eff. given S1**: the fraction of S1 hits with a hit on that counter within
+-50 ns. The efficiency shows **n/a (timestamp fault)** when the counter's
mismatch is above 5 % (`sma.py:1109-1115`): with broken timestamps the
coincidence misses by microseconds and the number would look like a dead
counter. Channels outside `mismatch flag channels` show their mismatch in grey,
"(not flagged)". Footnote: RF valid fraction (S1 hits with a usable RF gate) and
vetoed fraction.

### Tabs

* **Health**: word types, frame classes (good / stale / empty / suspect), words
  per frame, frame span and gap, live fraction, rate per channel, stale words.
* **ToT / corruption**: ToT per channel (split by fine bit 0), ToT >= 250 per
  channel, coarse minus fine, `fine_vs_coarse` (which fine bit disagrees), fine-bit
  occupancy.
* **Timing**: S2..S5 minus S1 time differences, coincidence pattern, partners per
  S1, S1 spacing.
* **RF / delayed**: RF pulses per S1 gate, RF phase, RF period, phase vs S1 ToT,
  delayed channels minus S1 (one plot per entry of `Channel roles/delayed`; none
  by default).
* **MuPix**: above the plots, the share of S1 hits with an L1, an L2 and an
  L1+L2 pixel hit in time, in the sideband, and with the accidentals taken out
  (over the last 60 s), and the time-sync chip. Plots: t(pixel) - t(S1) per
  plane (every pair, [-2560, +2000) ns, 8 ns bins; the in-time window outlined
  green, the sideband grey), the S1-match counts since the last Clear (the
  footnote gives them as fractions), pixel ToT per plane (0-31 counts of 256
  ns), column and row occupancy per plane with one line per chip, pixel hits
  per frame per chip. See [MuPix](#mupix-the-pixel-words-in-the-sma-frames).
* **Trends**: rate per channel, live fraction, efficiency given S1, RF valid
  fraction, the MuPix in-time fractions (L1, L2, L1+L2 accidental-corrected,
  and L1+L2 raw and in the sideband), 1 s rows over the last 10 min. The
  efficiency trend shows the raw value even for a counter the table marks n/a
  (S5 sits near 0 there; ignore it).

Controls: log y / log z, update rate (1 Hz, 0.5 Hz, 0.2 Hz, paused), **Clear**
(zeroes the histograms only; the summary and trends are not affected).
2D plots are drawn as an image (`pages/js/dqm-heatmap.js`): hover a bin for its x, y
and count; they have no zoom or menu buttons, and under/overflow is not drawn (the
footnote gives its count when non-zero). A plot scrolled out of view stops updating
and catches up within a second of scrolling back.

## The SMAEvents page

* **S1-seeded events** (4 Hz): up to 4 *seeds* of the latest good frame, each
  with every channel that has a role in its window (-200 ns .. +3 us). One lane
  per channel, bars from t to t+ToT, hatched red = fine/coarse mismatch, magenta
  marker = ToT >= 250, dashed red line = the seed, black tick on RF = the pulse
  the phase is taken from. Badges per seed: which counters fired (green), RF
  phase/period and whether the gate is valid, hit count, mismatch and ToT
  counts. The *window* selector switches to a +-150 ns prompt view. How the
  seeds are chosen, and the *seed*, *only seeds with*, *MuPix:* and *and
  counters:* controls, are below.
* **MuPix lanes** in every seed panel, whatever the seed: *MuPix L1* and
  *MuPix L2* (and *no plane* when a hit is on a chip the plane map lacks), one
  tick per pixel hit in the panel's window, coloured by its ToT, with the
  in-time window ([-150, +450) ns around the seed) shaded green. A badge per
  seed says which planes had a hit in that window: **L1+L2 ✓**, **L1 ✓**,
  **L2 ✓** or *no MuPix in time* (*MuPix n/a (outside the pixel data)* when the
  window is outside the MuPix part of the frame), and one gives the seed's
  MuPix hit count and their word range. Real hits sit a little before the seed
  line (the peak is at about -30 ns on run 1008) with a tail of low-ToT hits up
  to a few hundred ns later (time walk).
* **Whole-frame raster** (2 Hz): time vs channel for the whole frame, ToT as
  colour, hit count per channel on the right. Drag to zoom in time, double-click
  for the whole frame. *hide the current channel* removes the proton-current
  channel (`Channel roles/current`), which otherwise dominates. Since run 1015 the
  current is not on the SMA (`current` = -1) and the switch is not shown.
  Under the 16 channels, **MuPix L1** and **MuPix L2** rows: every
  pixel hit of the frame (noise included), coloured by pixel ToT on its own
  scale (legend: MuPix ToT 0-31, x 256 ns), the plane's hit count on the
  right; a third row, **no plane**, appears only when a hit is on a chip the
  plane map lacks. *hide MuPix* leaves the rows out (the analyzer then sends no
  pixel hits: -70 kB a reply on a run-1008 frame). The axis starts at the
  frame's first SMA hit; a pixel hit before it (rare) is counted but not drawn.
* Both tabs show the latest *analysed* frame. When the analyzer samples, a note
  after the frame badges says so: "Latest analysed frame; the analyzer analyses
  45 % of frames (CPU budget)".
* **Freeze** holds the current frame so you can point at it; the frame sequence
  number stays fixed until you release it.
* Clicking hits tags them into the **Tagged hits** list under the plots
  (numbered, grouped by frame, with Δt); see
  [Tracking down an odd event](#tracking-down-an-odd-event).
* **Single ▸** (both tabs) freezes if live, then fetches exactly one newer frame for the
  visible tab per press (raster keeps its *hide the current channel* choice, the seeded
  tab its seed and filters); if the analyzer has nothing newer it asks once more after
  300 ms, then says "no newer frame than seq N yet" and keeps the display. **Frozen —
  resume** goes back to live polling.
* A stale or suspect frame carries a red **STALE frame** or **SUSPECT time base** badge
  with the reason (`pages/js/dqm-sma-events.js:440-448`). The seeded view shows only
  good frames.

### How the seeds are chosen

**Default (seed = S1, no box ticked).** While it analyses a frame, the analyzer
takes the latest 4 (`Cuts/seeds`) hits of the S1 channel (`Channel roles/s1`)
whose whole window lies in the part of the frame that every counter and the RF
channel cover (channels with fewer than `Cuts/seed min hits` hits in the frame do
not count) (`sma_words.py:861`). Nothing else is asked of them: no coincidence,
RF, ToT or fine/coarse condition. This is the view as it always was, byte for
byte.

**The *seed* list** (per viewer; kept for the browser tab, not in the ODB):

| Choice | Seeds are | Use it to |
|---|---|---|
| S1 (default) | S1 hits, as above | the normal view |
| S2 .. S5 (and any delayed channel) | hits of that channel (the counters and the delayed channels, names from the ODB labels; RF and the current channel are not offered) | see events a counter fired in, with or without S1 |
| any counter (S1..S5 clusters) | the first hit of each time cluster of the S1..S5 hits, a cluster being hits each within `Cuts/coinc window ns` (50 ns) of the one before | see events **without S1** at all |

For a seed other than S1 the window, the completeness rule and the lanes are the
same, anchored on the seed hit (the time axis reads "ns from the seed"). The
counter boxes S1..S5 light for counters with a hit within +-50 ns of the seed.
The RF phase is borrowed from the S1 hit nearest the seed within +-50 ns ("RF ·
... · from the S1 hit at +3 ns"); with no S1 hit there it says **RF n/a (no
S1)**. These seeds are computed when the page asks, from frames the analyzer
keeps (`sma_words.py:1065`, `sma.py:1809`), not while it fills the plots.

**The *only seeds with* boxes** keep seeds that have *any* of the ticked
oddities (OR):

| Box | A seed has it when |
|---|---|
| incomplete pattern | not every counter S1..S5 has a hit within +-50 ns of it, leaving out counters with a known timestamp fault (below) |
| fine/coarse mismatch | any hit in its window has fine and coarse disagreeing (the hatched red bars) |
| ToT ≥ 250 | any hit in its window has ToT >= `Cuts/tot corrupt` (the magenta markers) |
| RF not valid / vetoed | the S1 hit it takes the RF from has no valid gate, or a vetoed one. A seed with no S1 near it has no RF measurement and does not match (use *incomplete pattern* for those) |

**Incomplete pattern and a counter with a timestamp fault.** A counter whose
fine time is broken (run 1008: S5 has fine = t/2, 93 % of its words
fine/coarse inconsistent) is almost never within +-50 ns of anything, so
without a correction *incomplete pattern* would match nearly every seed (3,527
of 3,540 on run 1008). The analyzer therefore leaves out of that test every
counter that SMAPlots currently flags for a fine/coarse mismatch: more than
`Self check/mismatch warn fraction` (5 %) of its hits inconsistent over the
summary window (60 s), at least `Self check/min hits` hits, a channel in
`Self check/mismatch flag channels`, exactly the rule of the `mismatch` flag
(`sma.py:1849`, `incomplete_rule` at `sma.py:1886`, applied in
`sma_words.py:1579`). The header then reads **seed: S1 (ch 1) with incomplete
pattern (ignoring S5: timestamp fault 93 %)**, each seed's **odd:** badge says
**incomplete pattern (S5 ignored)**, and the ignored counter's box in the
seed's pattern is drawn dashed. Only the pattern test changes: the lanes, the
other boxes and the *and counters:* row below still see the counter as it is.

* While the coarse-shift check is not *ok* (a shift mismatch, no fit, or not
  enough S1 words yet), every channel's mismatch number describes the setting,
  not the board, and nothing is ignored; the header says so: *incomplete pattern
  (no counter ignored: shift check insufficient)*.
* When fewer than two counters are left (e.g. S2..S5 all flagged), there is
  nothing to compare: the header says **incomplete pattern cannot be judged:
  S2, S3, S4, S5 have timestamp faults**, no seed has the oddity, and with only
  that box ticked the red no-match banner names the reason. It does not match
  everything.
* The reply carries this as meta `incomplete` (`judged`, `verdict`, `ignored`
  with each counter's mismatch fraction, `note`, `label`), and `sma::summary`
  as `timestamp_faults`.

**The *MuPix:* selector** is a selection, not an oddity: it is AND-ed with the
boxes (a seed must have one of the ticked oddities *and* pass it).

| MuPix: | A seed passes when, in [-150, +450) ns around it (`MuPix/window lo ns`, `hi ns`) |
|---|---|
| any (default) | always |
| L1+L2 in time | L1 and L2 both have a pixel hit |
| L1 or L2 in time | at least one plane has one |
| none in time | neither has one (events S1 saw and MuPix did not) |

A seed whose window lies outside the MuPix part of the frame says nothing
about MuPix and passes only *any*. The choice is kept per browser tab like
the seed choice, and goes with Freeze and Single.

**The *and counters:* row** asks for a coincidence pattern directly: for each
counter (one selector per entry of `Channel roles/counters`, S1..S5 by default),
*any* (default), *present* (a hit on that counter within +-50 ns,
`Cuts/coinc window ns`, of the seed, i.e. its box lit) or *absent* (no such
hit). It is another AND, on top of the seed choice, the boxes and *MuPix:*
(`sma_words.py:1400`). Examples:

| Row | Finds |
|---|---|
| S1 present, S3 absent | S1 events S3 missed (S3 losses) |
| seed S2, S1 absent | S2 hits with no S1 (for an S1 seed, *S1 absent* never matches) |
| seed any counter, S1 absent, S2 present | counter clusters that start without S1 |
| S3 absent + box *ToT ≥ 250* + MuPix *L1+L2 in time* | all three at once |

Nothing is left out here: the row asks exactly what it says. A counter
SMAPlots flags for a timestamp fault gets a yellow note beside its selector
instead: **S5 has a timestamp fault (93 % fine/coarse mismatch): 'absent' will
match almost everything, 'present' almost nothing**. The row is kept per
browser tab with the other controls, goes with Freeze (re-asks for the frozen
frame), Single and the search back through the seed ring, and appears in the
header (**seed: S1 (ch 1) and S1 present, S3 absent**) and in the banners. The
request argument is `"pattern": {"1": "present", "3": "absent"}` (counters by
number; *any* is left out).

Each seed shows the oddities it has in a yellow **odd:** badge. With a filter
ticked, a MuPix selection, a pattern condition or a seed other than S1, the header adds **seed: ...**
and **matching seeds: n of m candidates in frame seq X** (m = seeds with a complete window in
that frame; the analyzer looks at the latest 20000 of them at most and says so
when it had to stop there). The analyzer looks at the newest good frame first
and, if nothing there matches, back through the last good frames it keeps
(`Sampling/seed ring frames`, default 8, at most `Sampling/seed ring MB` = 24 MB;
8 frames of run 1008 are about 20 MB, about 0.5 s of data at the rig's analysis
rate). It spends at most about 20 ms of CPU per request on this; a longer search
goes on with the next poll. Choosing a seed or ticking a box asks again at once;
while frozen it re-asks for the frozen frame with the new choice.

**The banner above the seeds** says when the frame shown is not the newest
analysed good frame:

* yellow: "Showing frame seq 473 from 0.1 s ago: no ch08 (ch 8) seeds with ToT ≥
  250 in the 1 good frame analysed since (newest seq 474)." In the default view
  the reason is "no S1 hits" or "none of N S1 hits has a complete window", and
  the frame can be old: the default keeps showing the last frame that had S1
  seeds. If that frame's raw event has left the analyzer's raw ring the banner
  adds "Its raw event is no longer held: use the tag".
* red: "No match: no S1 (ch 1) seeds with ToT ≥ 250 in the last 8 good frames
  (0.5 s). Showing the newest analysed frame, seq N, without seeds." The search
  goes on with every new frame.

When good frames keep arriving but none has had an S1 seed for 10 s (`Self
check/no seeds s`), SMAPlots raises a **no_seeds** warning: S1 is absent, dead or
mis-cabled, or `Channel roles/s1` points at the wrong channel. Choose *any
counter* to see what the other counters are doing meanwhile.

## MuPix: the pixel words in the SMA frames

The H000 bank interleaves the MuPix pixel words (bit 63 clear) with the SMA
trigger words. A pixel word holds the chip id (bits 62:58), column (57:50), row
(49:42), TS2 (41:37) and a 37-bit time stamp of 8 ns ticks (36:0)
(`sma_words.pixel_decode`, following `PIMuPixWord.hh` of reco_testbeam; pinned
against it and against the psm-analysis Python on stored real frames by
`tests/test_mupix_golden.py`). The time stamp times 8 ns is on the same 2^40 ns
epoch as the SMA time, so t(pixel) - t(S1) needs no calibration; both are
unwrapped around the frame's middle, so a frame across the 2^40 ns wrap (every
18 min) still subtracts correctly.

* **ToT.** Bits 41:37 are not a ToT but TS2, a second counter latched at the
  end of the pulse. The ToT is `(TS2 - ((time >> 5) & 31)) & 31` counts of 256
  ns (`MuPix/ts2 shift` = 5, from ckdivend2 = 0x1f on every bt2026 run).
* **Planes.** The chip field is a *global* ASIC id set by the switching board
  from `/Equipment/Quads/Settings/DAQ/Links/Mapping`. `MuPix/L1 chips` and
  `L2 chips` say which ids are which plane; the page always shows the id too,
  and a hit on an id in neither list is "chip N (no plane)" and raises
  `mupix_unmapped`. **Default L1 = 0, 1, 2, 3 and L2 = 4, 5, 6, 7**: runs 682 and
  1008 carry the identity Mapping (`[0, 1, ..., 7, 99, ...]` in their
  begin-of-run ODB), and in both the same particle fires chip 0 and 4, 1 and 5,
  2 and 6, 3 and 7 (46-62 % of a chip's prompt hits have one on its partner
  within 30 ns on run 1008, 38-48 % on run 682; at most 1 % and 10 % on any
  other chip), as the reco's readout map
  `bt2026-febmap` has it from run 200 on. Runs up to 186 had Mapping
  `[1, ..., 7, 0]`: L1 = 1, 2, 3, 4 and L2 = 5, 6, 7, 0.
* **In time.** A pixel hit is in time with an S1 hit when t(pixel) - t(S1) is
  in [-150, +450) ns (`MuPix/window lo ns`, `hi ns`): the time walk spreads real
  hits from about -150 ns (large ToT) to +450 ns (small ToT); no walk
  correction. Most pixel hits are not from S1's particle at all: whole-chip
  noise bursts and hot pixels (60-70 % of the hits on run 459). So every
  in-time share is shown with the same share in a **sideband** [-2400, -1800)
  ns (`MuPix/sideband lo ns`, `hi ns`; before the prompt window, never after
  it, where decays and delayed hits are): the chance of an unrelated hit in a
  window that wide. *Accidentals taken out* is `(in - a) / (1 - a)`, `a` the
  sideband share (scaled to the window's width if the two differ).
* **Which S1 hits.** The S1 hits the S1-seeded analyses use (at most
  `Cuts/max S1 per frame`), of those at most `MuPix/max S1 per frame` (500)
  evenly spread, and only those whose window and sideband both lie inside the
  frame's pixel data: the MuPix part of a frame does not cover exactly the SMA
  part (on run 1008 the pixels start ~0.5 ms after the first SMA hit).
* **Bounded cost.** At most `MuPix/max pixel hits per frame` (20000, the
  latest) pixel hits are examined per frame; the rest are counted
  (`mupix_skipped`). The pixel stream is sorted in time when it is not (it has
  been on every frame checked).

**Measured on the replayed runs** (`mdqm-sma-file`, all frames of
`run01008_00001` and `run00682_00005`; `scratch/sma-dqm-mupix/`):

| Run | Plane | t(pixel) - t(S1) peak | FWHM | 10-90 % of the excess | S1 hits in time | sideband | accidentals out |
|---|---|---|---|---|---|---|---|
| 1008 | L1 | -36 ns | 64 ns | -60 .. +20 ns | 92.2 % | 6.7 % | 91.7 % |
| 1008 | L2 | -28 ns | 64 ns | -52 .. +28 ns | 90.8 % | 6.7 % | 90.2 % |
| 1008 | L1+L2 | | | | 88.4 % | 6.1 % | 87.7 % |
| 682 | L1 | -20 ns | 88 ns | -68 .. +28 ns | 90.2 % | 1.1 % | 90.1 % |
| 682 | L2 | -20 ns | 88 ns | -60 .. +36 ns | 88.4 % | 0.9 % | 88.3 % |
| 682 | L1+L2 | | | | 84.7 % | 0.8 % | 84.6 % |

The peak stands ~115 times (1008) and ~550 times (682) above the flat
background, and 99.8 % of the excess is inside the in-time window. The
earlier timewalk study found the uncorrected peak at -50 to -85 ns with a walk
of ~150 ns (runs 429, 459, different MuPix DACs), and 93 % of good WaveDREAM
events with an in-time MuPix hit on runs 480/509 (another selection): the same
picture.

### MuPix time sync: what the flag means

The in-time share only exists because the SMA and MuPix clocks count the same
time. If they stop agreeing, the t(pixel) - t(S1) peak vanishes into the flat
background and the share falls to about the sideband's. The analyzer watches
the L1+L2 share with the accidentals taken out over the last 10 s (`Self
check/mupix sync window s`), with at least 200 analysed S1 hits in it (`mupix
sync min S1`): below 30 % (`mupix sync min fraction`; run 1008 sits at 88 %,
run 682 at 85 %) for more than 30 s (`mupix sync hold s`) it raises the
warning `mupix_sync`, and it clears above 35 % (plus `mupix sync clear margin`).
With no S1 hits (no beam) there is no verdict and no flag. The MuPix tab's chip
says *in sync*, *low for N s*, **SYNC LOST?**, *no verdict* or *MuPix analysis
off*.

When `mupix_sync` is up:

1. Look at the MuPix tab's t(pixel) - t(S1) plots. **No peak in either plane**:
   the time bases disagree. Check the SMA coarse-shift banner first (a wrong
   `Coarse shift` moves every SMA time), then whether the MuPix or SMA readout
   was restarted or reconfigured; write it in the elog and tell the SMA/MuPix
   experts. A peak that has **moved** (not at ~-30 ns) is a time offset: also
   for the experts.
2. **A peak in one plane only**: that plane is dead or off (occupancy plots,
   pixel hits per frame per chip).
3. **Peaks present but small** against the sideband: the beam does not reach
   MuPix (a degrader in, a stopping run) or MuPix sees little of it. Not a
   fault if that is the run plan.
4. "No MuPix pixel words in the SMA frames": the MuPix readout, its link or
   its chips are off.

## NIM copies of the counters (TOT + NIM)

Since run 1015 each scintillator counter reaches the SMA twice: its TOT
channel and a NIM discriminator copy (S1L ch 3, S2L-S5L ch 9-12). The analyzer
pairs the two in every analysed good frame, on every word, with the same rule
as reco (`sma_nim.pair_counter`, a port of `PIPSMSMANimPairing.hh`). A NIM word
pairs with the nearest free TOT word within +-`NIM/pair window ns` (20 ns)
after `NIM/offset ns`. S3's late and edge echo words (`NIM/echo counters`) do
not pair. The analyzer also votes the NIM copy's fine-time lag against S1
(`sma_nim.lag_vote`). It measures the lag and corrects nothing.

Where to look: the **NIM / TOT** tab on SMAPlots (a table per counter above its
plots, and the S1 coincidences with and without the merge), and the NIM lanes
on SMAEvents (S*k*L under each counter, a dark tick joining a TOT word to its
NIM copy, hollow bars for NIM-only words, grey for held-back ones, a hatch for
TOT echo words).

### The merge is off until the offsets are measured

`NIM/merge` is **n** by default. The NIM offsets are not measured yet, and a
NIM copy more than 20 ns off its TOT word pairs with nothing: merging then
counts every particle twice. With the merge off, the pairing plots, the NIM
table, the summary and the four NIM flags all work, and the counters, the
pattern, the efficiencies and the seeds are the TOT words alone. The page says
so in a chip ("NIM merge off", "NIM merge on" when it is on). Turn the merge on only after
[measuring the offsets](#measuring-the-nim-offsets-on-the-first-clean-run).

With `NIM/merge` on, the counters are the merged hits: TOT words (paired,
TOT-only, echo) and NIM-only hits (the aligned NIM time, ToT `NIM/nim only
tot`). Pattern, efficiencies, S2..S5 - S1, RF, delayed, the seeds of SMAEvents
and the MuPix in-time matching (its S1 hits) use them; rates, ToT, fine/coarse,
the stale rule, the shift check and the raster stay on the words.
`rf_phase_vs_s1_tot` leaves the NIM-only S1 hits out (they have no ToT of their
own).

The NIM offsets are the DQM's own constants. The DQM applies no fine-time lag
correction and no per-counter TOT offset, so its offsets differ from reco's
`sma_time_alignment`. Do not copy numbers from one to the other.

### The lag state

The fine-time lag fault is a whole-file state: a NIM channel's fine time is off
by about 150 us or about 0.9 us for the whole file. Each NIM channel keeps its
last decisive vote ("ok" or "faulted") until the next rebuild or run start. A
frame with too few NIM words for a vote (fewer than `NIM/lag min pairs` with an
S1 reference, or no clear winner) takes that state, and so does a frame that is
not voted at all: once a state is known the vote runs in every
`NIM/lag vote every`-th frame (4) of the channel, to save CPU; before that, in
every frame. A later "ok" vote clears a "faulted" state.

The lag is **measured, never corrected**. The only action on it is with the
merge on: the NIM-only hits of a frame whose lag state is "faulted" are held
back from the counters (counted as "lag-held"), unless `NIM/merge when lagged`
is y. With the merge off nothing is held back. `nim_lag_Sk` is filled from the
voted frames only, so its shape is the same as an all-frames plot with about a
quarter of the entries.

### Pair efficiency

Pair efficiency is paired / (paired + TOT-only). Echo words (S3's late and
edge words) are left out of the denominator, as in reco's pair fraction.
smanim leaves the late and echo TOT words out of its TOT sample too, so its
`tot_with_nim` is the same quantity, and its `nim_with_tot` is the purity
(paired / NIM words). The two can differ slightly: smanim pairs mutual
nearest neighbours after its own offset fit and lag removal, the DQM pairs
greedily one to one with the DQM's own offsets.

### What the NIM plots should look like

Plots are per counter with a NIM copy, named `sma/nim_<kind>_S<k>` (k = counter
number, S1 = 1). The two `nim_dt` plots are drawn with log y by default (the
"log y for the Δt plots" switch on the tab). Adapted from the nearline version in
`beamtime2026_pie5` (`docs/SMA_NIM_RECABLING.md`, section 6).

| Plot | Good | Bad, and what it usually means |
|---|---|---|
| `nim_dt_Sk` (NIM minus the nearest TOT word, every NIM word, +-200 ns, 1 ns bins, after `NIM/offset ns`) | one narrow peak at 0, a few bins wide, little flat background. Before the offsets are set the peak sits wherever the cable delay puts it; that is the number to measure | peak away from 0 after the offset is set: the offset is wrong or was not applied. Two peaks: two particles per window, or a swapped cable. A wide peak (several ns): a trigger-level or jitter problem on that channel. No peak: NIM and TOT are not the same counter, or the offset is larger than 200 ns; look at `nim_dt_wide_Sk` |
| `nim_dt_wide_Sk` (raw NIM minus TOT, pairs within +-2^19 ns, 256 ns bins, sampled) | one peak in the central bin: no fine-time offset on the NIM channel. A cable delay of tens of ns does not leave that bin | a peak away from the centre is the lag fault; its position is the lag. Several peaks or a flat spread: a fault that changes within a file (`nim_lag` will usually fire; elog it) |
| `nim_walk_Sk` (NIM minus TOT against the TOT word's ToT, pairs only, +-`pair window`) | a flat band: no walk | a slope or a curve is TOT leading-edge walk. It is measured only, nothing corrects it; elog its size |
| `nim_classes_Sk` (paired, TOT-only, NIM-only, echo, then the sub-counts lag-held, in TOT shadow, multi-candidate) | "paired" is nearly all the hits (pair efficiency above about 95 %); "TOT-only" and "NIM-only" small; "echo" only on S3 | large "NIM-only": the NIM threshold is too low (noise), or the offset is wrong so that real pairs fall outside the window (then "TOT-only" is large too). Large "TOT-only": a dead or high-threshold NIM channel. "echo" on a counter other than S3: the echo rule is only on for `NIM/echo counters` |
| `nim_width_Sk` (ToT field of the NIM word) | one narrow peak: the width is fixed by the discriminator | a wide spread: the channel is not a clean logic pulse |
| `nim_candidates_Sk` (NIM words within the window of each TOT word) | bin 1 holds nearly all TOT words; bin 0 is the TOT-only share | many in bins 2 and up: the window is too wide for the rate, or the NIM channel doubles pulses |
| `nim_lag_Sk` (the lag vote's input, fine minus the S1 word's fine, mod 2^20 ns) | one narrow peak, at the left edge (it wraps, so it can also sit at the right edge) | a peak elsewhere is the lag fault; several peaks or flat, no vote is possible. Filled only in voted frames, so an empty plot means too few S1 coincidences, not a fault |
| `s1_coinc` and `s1_coinc_tot` (S1 hits with each counter in the window; `s1_coinc` is the merged one, `s1_coinc_tot` the TOT words alone) | with the merge off the two are the same and no overlay is drawn. With it on, `s1_coinc` is at or above `s1_coinc_tot` by a small gap, the hits only the NIM copy saw | a large gap: a large NIM-only share (check the table and `nim_classes_Sk`), usually a low NIM threshold or a wrong offset. `s1_coinc` below `s1_coinc_tot` should not happen; tell the SMA expert |

The NIM table above the plots shows, per counter: pair efficiency, purity,
NIM-only share, median NIM - TOT, the lag vote (last state, share of faulted
frames, last lag) and the NIM-only hits held back. All are over the last 60 s.
A healthy counter shows pair efficiency of about 95 % or more and a median near 0.
The median is read from `nim_dt_Sk`, so it runs since the run start or the last
settings change, not over 60 s.

### The four NIM flags

The flags are in the table under [Flags](#flags-below-the-banner). Each one
needs at least `Self check/min hits` words (200) in the 60 s window for the
counter. All but `nim_missing` compare times, so they wait for the shift check
to say `ok`.

* `nim_missing` (warning): the counter's TOT channel has hits and its NIM copy
  has none. Check that the NIM cable is on the right SMA input and the
  discriminator is powered, then `NIM/channels` in the ODB. Nothing is paired
  for that counter meanwhile.
* `nim_pairing` (warning below 80 %, error below 50 %): too few TOT words have
  a NIM word in the window. The text gives the median NIM - TOT. If it is large,
  the offset is not set: do the
  [measurement](#measuring-the-nim-offsets-on-the-first-clean-run). If the
  median is near 0, the NIM threshold or the NIM channel is the problem: elog
  it with the counter, and tell the SMA expert. A counter whose TOT channel has
  a known timestamp fault (the red `mismatch` flag, S5 for now) is not judged.
* `nim_offset` (warning): the median NIM - TOT is more than 5 ns from 0 after
  the offset. The text names the key and the value to set
  (`NIM/offset ns[k-1]` = old offset + the median, rounded). Set it as
  [below](#measuring-the-nim-offsets-on-the-first-clean-run). It is expected
  until the offsets have been measured.
* `nim_lag` (warning): the lag state was "faulted" in more than half of the
  window's frames, after at least 3 faulted votes (`Self check/nim lag min
  votes`). A hardware fault that nothing here fixes. Elog it with the counter
  and tell the SMA expert. With the merge on, the text says how many NIM-only
  hits were held back meanwhile.

### Summary, trends and the manual path

`sma::summary` carries `nim.counters[]` per counter (pair efficiency, purity,
NIM-only share, median dt, lag votes and lag state), `nim_merge`,
`nim_lag_held` and `roles`. Trend: `nim_eff` (pair efficiency) per counter.
Offline, `mdqm-sma-file` runs the same plugin on one file, writes `nim.png`
beside `summary.png`, and takes `--merge` / `--no-merge` (default: no merge).

### Measuring the NIM offsets on the first clean run

Do this once, on the first clean beam run at or after 1015 (S1 and the counters
firing, the shift check `ok`), with `NIM/merge` still **n**.

1. Open SMAPlots, **NIM / TOT** tab. Wait until `nim_dt_Sk` has a few thousand
   entries per counter.
2. For each counter read the position of the `nim_dt_Sk` peak in ns. That is
   its offset. The `nim_offset` flag text gives the same number as the key to
   set, from the median; the median is pulled toward 0 by the flat background,
   so if the peak and the flag disagree by more than a ns or two, use the peak
   and repeat step 4 once.
3. Set the offsets, one line per counter. The list is S1L, S2L, S3L, S4L, S5L
   (index 0-4), whole ns; the values below are placeholders:

   ```bash
   odbedit -e bt2026 -c 'set "/DQM/SMA/NIM/offset ns[0]" <S1L offset>'
   odbedit -e bt2026 -c 'set "/DQM/SMA/NIM/offset ns[1]" <S2L offset>'
   odbedit -e bt2026 -c 'set "/DQM/SMA/NIM/offset ns[2]" <S3L offset>'
   odbedit -e bt2026 -c 'set "/DQM/SMA/NIM/offset ns[3]" <S4L offset>'
   odbedit -e bt2026 -c 'set "/DQM/SMA/NIM/offset ns[4]" <S5L offset>'
   ```

   Every change rebuilds the histograms and zeroes them, so set all five before
   looking again.
4. Wait for the plots to fill again. Check, for every counter:
   * the `nim_dt_Sk` peak is at 0 and the median NIM - TOT in the table is
     within 5 ns of 0 (no `nim_offset` flag);
   * the pair efficiency is **at least about 95 %** (S3: echo words do not count
     against it); the `nim_pairing` flag only fires at 80 %, so look at the
     number, not the flag;
   * no `nim_missing` or `nim_lag` flag.
5. Only then turn the merge on:

   ```bash
   odbedit -e bt2026 -c 'set "/DQM/SMA/NIM/merge" y'
   ```

   The tab's chip changes to "NIM merge on", and `s1_coinc` and `s1_coinc_tot` now
   differ by a small gap. A large gap, or a NIM-only share that is not small,
   means an offset or a threshold is still wrong: turn the merge off again (`n`)
   and repeat from step 2.

Write the measured offsets and the run number in the elog. Redo the measurement
after any change to the NIM cables, thresholds or delays.

## Settings: `/DQM/SMA`

The analyzer re-reads the tree every 2 s (`analyzer.py:231-233`); edits take
effect without a restart. Changing the coarse shift, channel roles (except
labels), anything under Cuts, Binning, MuPix or NIM (except its two CPU knobs) **rebuilds the histograms and zeroes
them** (`SmaPlugin.apply_settings`) and posts a MIDAS message. Labels, Self check
and Sampling never reset a plot. Defaults: `SETTINGS_DEFAULTS` in `sma.py`.

The default channel roles are the cabling since run 1015: 0 clock, 1 S1, 2 S2,
3 S1L, 4 S4, 5 S5, 6 RF, 7 S3 (TOT), 8 WD trigger copy, 9-12 S2L-S5L. The proton
current is no longer on the SMA. The NIM copies (S*k*L) are under `NIM/channels`;
the WD copy has no role. (The roles before run 1015 were S3 on ch 3, the proton
current on ch 7 and delayed channels 8-10.)

**An existing ODB keeps its old roles.** The analyzer only creates missing keys,
so an ODB that already has the keys keeps its values. An ODB upgraded by this
version therefore has the pre-1015 roles next to the new NIM keys: NIM channels
3, 9 and 10 are then a counter and delayed channels. The analyzer turns NIM off
altogether (no pairing, no merge) and raises one `settings` flag, "Channel
roles look pre-1015". To move an existing ODB to the 1015 layout (on pinky;
drop `-e bt2026` if `MIDAS_EXPT_NAME` is set):

```bash
odbedit -e bt2026 -c 'set "/DQM/SMA/Channel roles/counters[2]" 7'
odbedit -e bt2026 -c 'set "/DQM/SMA/Channel roles/current" -1'
odbedit -e bt2026 -c 'set "/DQM/SMA/Channel roles/delayed[*]" -1'
odbedit -e bt2026 -c 'set "/DQM/SMA/Self check/mismatch flag channels[2]" 7'
odbedit -e bt2026 -c 'ls "/DQM/SMA/Channel roles/labels"'
```

The first three lines rebuild the histograms (a role change), and the flag goes
once the third is in. For a channel role, -1 means "none". Check the labels
last: a label set by hand ("current" or "S3" on ch 7, "S3" on ch 3) survives
the edit; clear it (`set ".../labels[7]" ""`) so the role name shows.

To keep analysing a pre-1015 run live instead, switch the NIM copies off with
`odbedit -e bt2026 -c 'set "/DQM/SMA/NIM/channels[*]" -1'` and leave the old
roles.

**The NIM offsets and the merge** are set after the first clean run, not
here: see [Measuring the NIM offsets on the first clean
run](#measuring-the-nim-offsets-on-the-first-clean-run) for the odbedit lines
and the checks before `NIM/merge` goes to y. Both kinds of edit rebuild the
histograms.

| Key | Default | Effect | Resets plots |
|---|---|---|---|
| `Coarse shift` | 14 | board coarse = time >> shift | yes |
| `Channel roles/s1` | 1 | the seed channel | yes |
| `Channel roles/counters` | [1, 2, 7, 4, 5] | S1..S5 in order (first must be the S1 channel); at most 8, the pattern is one byte | yes |
| `Channel roles/rf` | 6 | RF channel | yes |
| `Channel roles/current` | -1 | proton-current channel; -1 = none | yes |
| `Channel roles/delayed` | [-1] | channels paired with S1 in [-1, +10] us; -1 entries are ignored, so [-1] = none (an ODB array cannot be empty) | yes |
| `Channel roles/labels` | 16 x "" | display names; empty = role name or chNN | **no** |
| `Cuts/coinc window ns` | 50 | S1 coincidence window (pattern, efficiency) | yes |
| `Cuts/dt window ns` | 200 | range of the S2..S5 - S1 plots | yes |
| `Cuts/seed pre ns`, `seed post ns` | 200, 3000 | seeded-event window | yes |
| `Cuts/seeds` | 4 | seeds shown per frame | yes |
| `Cuts/rf gate ns`, `rf min pulses`, `rf max pulses` | 125, 2, 4 | RF phase gate | yes |
| `Cuts/latch margin ns` | 3616 | tolerance of the fine/coarse check | yes |
| `Cuts/stale gap ms` | 50 | hits further apart than this split a frame into time clusters; the cluster holding the median is kept | yes |
| `Cuts/tot corrupt` | 250 | ToT code counted as corrupt | yes |
| `Cuts/delayed lo ns`, `delayed hi ns` | -1000, 10000 | delayed-channel window | yes |
| `Cuts/max words per frame` | 1048576 | larger frames (over 8 MiB) are counted as `oversize` (summary, status, a warning flag) and not decoded: one costs ~0.6 s of CPU and hundreds of MB. 0 = no limit | yes |
| `Cuts/max S1 per frame` | 2000 | a frame with more kept S1 hits gives the S1-seeded analyses (pattern, efficiency, S2..S5 - S1, RF, delayed) to an evenly spread sample of this many; 0 = no cap. Bounds the cost of a dense frame | yes |
| `Cuts/max gap s`, `max overlap ms` | 10, 10 | larger jumps between frames count as a loop/run boundary | yes |
| `Cuts/stale frame ...`, `suspect kept fraction` | see `sma.py:145-153` | the stale/suspect frame rules (`sma.py:407-473`) | yes |
| `Binning/...` | `SETTINGS_DEFAULTS` in `sma.py` | histogram ranges and bin counts | yes |
| `Binning/mupix dt min ns`, `mupix dt max ns`, `mupix dt bin ns` | -2560, 2000, 8 | the t(pixel) - t(S1) axes (the default reaches back to the sideband) | yes |
| `Binning/mupix hits per chip max` | 4000 | axis of pixel hits per frame per chip | yes |
| `MuPix/L1 chips`, `L2 chips` | [0, 1, 2, 3], [4, 5, 6, 7] | chip ids (the pixel word's global ASIC id) of each plane; runs up to 186 need [1, 2, 3, 4], [5, 6, 7, 0]. An id in both lists is an error (defaults used) | yes |
| `MuPix/ts2 shift` | 5 | log2(ckdivend2 + 1): ToT = (TS2 - ((time >> shift) & 31)) & 31, 2^shift x 8 ns a count | yes |
| `MuPix/window lo ns`, `window hi ns` | -150, 450 | t(pixel) - t(S1) called in time, half open | yes |
| `MuPix/sideband lo ns`, `sideband hi ns` | -2400, -1800 | the accidentals' window; must end at or before `window lo ns` | yes |
| `MuPix/max pixel hits per frame` | 20000 | pixel hits examined per frame (the latest); 0 = MuPix analysis off (pixel words still counted) | yes |
| `MuPix/max S1 per frame` | 500 | S1 hits per frame matched against the pixels (evenly spread); 0 = all | yes |
| `NIM/channels` | [3, 9, 10, 11, 12] | NIM copy of each counter (per `Channel roles/counters` entry); -1 = none, [-1] alone = no NIM at all. A NIM channel that is also S1, a counter, the RF, `current` or `delayed` turns NIM off (a `settings` flag); a repeated one drops that entry | yes |
| `NIM/offset ns` | [0, 0, 0, 0, 0] | per counter, t'_NIM = t - offset in whole ns (a fraction is rounded, with a `settings` note); set from the `nim_dt` peak | yes |
| `NIM/lag nominal ns` | [0, 0, 0, 0, 0] | per counter, the NIM copy's expected fine - fine(S1) (cable delay, flight); the lag vote is "faulted" beyond `lag tolerance ns` of it | yes |
| `NIM/merge` | n | NIM-only hits join the counters (pattern, efficiencies, seeds, MuPix matching). Turn on once the offsets are measured | yes |
| `NIM/merge when lagged` | n | also merge a channel's NIM-only hits in a frame whose lag state is "faulted" | yes |
| `NIM/pair window ns` | 20 | pair when \|t'_NIM - t'_TOT\| <= this | yes |
| `NIM/time source` | tot | `tot` or `nim`: the time a paired merged hit takes | yes |
| `NIM/nim only tot` | 1 | the ToT code a NIM-only hit gets | yes |
| `NIM/echo counters` | [3] | counter numbers (1 = S1) whose TOT words get the echo rule; -1 = none | yes |
| `NIM/echo late tot`, `echo edge tol ns` | 128, 3 | echo rule: ToT >= late, or a start within +-tol of the previous word's trailing edge | yes |
| `NIM/lag tolerance ns`, `lag min pairs`, `lag dominance` | 50, 50, 2.0 | the lag vote (`sma_nim.decide_lag`) | yes |
| `NIM/lag vote every` | 4 | vote a NIM channel's lag in every this many of its frames once a state is known (1 = every frame); the frames between take the last decisive vote. A CPU knob | **no** |
| `NIM/wide pairs per frame` | 1024 | `nim_dt_wide` pairs per counter and frame, about; 0 = none. A CPU knob | **no** |
| `Self check/shift window s`, `shift margin`, `shift min fraction`, `shift min words` | 30, 0.2, 0.9, 1000 | the shift check | no |
| `Self check/summary window s` | 60 | averaging of chips, table and flags | no |
| `Self check/mismatch warn fraction`, `mismatch error fraction` | 0.05, 0.5 | mismatch flag levels | no |
| `Self check/tot corrupt warn fraction` | 0.05 | `tot_corrupt` level | no |
| `Self check/min hits` | 200 | channels with fewer hits are not judged | no |
| `Self check/no frames s` | 5 | `no_frames` delay | no |
| `Self check/efficiency window s`, `efficiency drop` | 30, 0.1 | `efficiency_drop` | no |
| `Self check/mismatch flag channels` | [1, 2, 4, 5, 6, 7] | channels that can raise mismatch/ToT flags (S1..S5 and RF) | no |
| `Self check/no seeds s` | 10 | `no_seeds` delay: good frames without an S1 seed for this long | no |
| `Self check/nim pairing warn fraction`, `nim pairing error fraction` | 0.8, 0.5 | `nim_pairing` levels (pair efficiency) | no |
| `Self check/nim offset max ns` | 5 | `nim_offset`: \|median NIM - TOT\| above this | no |
| `Self check/nim lag max fraction` | 0.5 | `nim_lag`: share of the window's frames whose lag state is "faulted" above this | no |
| `Self check/nim lag min votes` | 3 | `nim_lag` also needs this many frames voted "faulted" since the run start or the last rebuild | no |
| `Self check/mupix sync min fraction`, `mupix sync hold s`, `mupix sync window s`, `mupix sync min S1`, `mupix sync clear margin` | 0.3, 30, 10, 200, 0.05 | the `mupix_sync` flag (see [MuPix time sync](#mupix-time-sync-what-the-flag-means)) | no |
| `Sampling/CPU budget %` | 20 | the analyzer's CPU, % of one core, everything included; it analyses as many frames as fit and skips the rest unread. At most 50 (a larger value is used as 50, `cpu_budget_clamped` in `dqm::status`). 0: analyse nothing. No budget at all only with the development flag `mdqm-analyzer --no-cpu-budget` | no |
| `Sampling/max events per s` | 1000 | hard cap on analysed frames per second, on top of the budget; 0 means "decode nothing" | no |
| `Sampling/raw ring MB` | 16 | the raw bytes of the last analysed frames, for **Download raw event** (`sma::raw`); about 50 frames of run 1008. Oversize frames are never kept. 0 = none | no |
| `Sampling/seed ring frames` | 8 | good frames kept for the seeded view's other seeds, filters, MuPix selector and counter pattern, searched back when the newest has no match; at most 64 | no |
| `Sampling/seed ring MB` | 24 | the same, at most this many MB (estimated array bytes; the newest good frame is always kept) | no |

The `NIM/` per-counter lists must have one entry per counter; a wrong length is a
settings error and falls back (to the default if it fits, else no NIM copies).
Without any NIM channel the offsets and nominals are not judged.

An ODB seeded by an older version has a `Sampling/process all` key: it is
ignored now (the budget replaced it) and can be deleted.

A key the analyzer seeded once keeps its value when a newer version changes the
default (for example `Binning/span bins`). To go back to the defaults, delete the
key (or all of `/DQM/SMA`) and restart the analyzer; keys it does not know are
ignored.

## Tracking down an odd event

Every frame on SMAEvents carries a **tag** that finds it again in the data
files:

> SMA run 1008 · event 301 serial 4757 · 2026-09-28 07:31:02 UTC · frame seq 4750

It gives the run, the MIDAS event id and **serial number** (the key: serials count up
through a whole run, across subruns), and the event header's time (UTC, 1 s). The
frame seq is only this analyzer's own counter.

**On shift:**

0. To go looking for odd events rather than wait for one, tick one or more
   *only seeds with* boxes on the seeded tab (incomplete pattern, fine/coarse
   mismatch, ToT ≥ 250, RF not valid / vetoed; see
   [How the seeds are chosen](#how-the-seeds-are-chosen)), or set counters to
   *present* / *absent* in the *and counters:* row (e.g. S1 present, S3
   absent), and choose *any counter* as the seed for events without S1. Each seed says which oddities it
   has (**odd:** badge); the header says how many of the frame's seeds match.
   The choice stays for this browser tab and goes with Freeze and Single.
1. **Freeze** the frame (or step with **Single ▸**).
2. Click **Copy tag** in the frame header and paste it into the elog.
3. For hits: hover one to see its line, click it to **tag** it. Click more
   hits to tag them too (on either tab); click a tagged hit again, or its **×**
   in the list, to untag it. The **Tagged hits (N)** list under the plots has
   one row per hit, grouped under its frame's tag: its number, the channel,
   ToT, time, fine/coarse flag, the hit's **word index** in the H000 bank
   (64-bit words, filler and pixel words counted) and the raw word in hex, and
   for every hit after a frame's first, **Δt** in ns from that first tagged
   hit. For example `ch 5 (S5) · ToT 37 · … · word 12345 · 0x8512… · Δt +42 ns`.
   MuPix pixel hits (the ticks on the MuPix lanes and rows) tag the same way:
   `MuPix L1 chip 1 · col 88 · row 140 · ToT 7 (~1792 ns) · t … ns (t_rel … ns)
   · word 36617 · 0x04…`; a chip the plane map lacks reads `MuPix chip 9 (no
   plane)`, a row code of 250 or more `(not on the sensor)`. Their Δt, the
   word range of **Copy all** and its `mdqm-sma-file --words` line include them.
   On the canvas each tagged hit is boxed in black with its list number beside
   it, whenever its frame is on screen (a hit tagged on the seeded tab is also
   marked on the frozen raster of the same frame).
   - The raster has word data only for a frozen frame. A click on a live
     raster **freezes** it, fetches the words once, and then tags the hit (the
     list says so while it waits).
   - The list stays through tab switches, new frames, Freeze/Single and a
     reload of the page (it is kept per browser tab). It holds at most 200
     hits. **Clear all** empties it. **Esc** clears only the hover line.
   - **Copy** on a row copies that line with its frame tag. **Copy all**
     copies the whole list, per frame: its tag, its hits one per line, and the
     command that lists those words from the run file:

     ```
     SMA run 1008 · event 301 serial 355 · 2026-09-28 05:31:08 UTC · frame seq 19617
       #1 · ch 1 (S1) · ToT 10 · t 10585709453 ns (t_rel 0 ns) · fine/coarse ok · word 39501 · 0x810a009dbd451b8d
       #2 · ch 2 (S2) · ToT 2 · t 10585709459 ns (t_rel 6 ns) · fine/coarse ok · word 39502 · 0x8202009dbd451b93 · Δt +6 ns
       mdqm-sma-file --serial 355 --run 1008 --dir <raw dir> --words 39501:39502
     ```

     Replace `<raw dir>` with the directory holding the run's `.mid` files.
4. For a seed, **Copy seed words** gives the tag plus the S1 word and the word
   range of the hits in its window (for a seed on another channel: first that
   seed's own word, then the S1 word it took the RF from, or "—" without one),
   and the word indices of the MuPix pixel hits in the window.
5. If the event matters, click **Download raw event**. That saves
   `sma_run<run>_serial<serial>.mid`, the MIDAS event as the analyzer received it.
   It is a valid one-event MIDAS file (`mdump -x`, `midas.file_reader` and
   `mdqm-sma-file` read it). The analyzer keeps only the last 16 MB of analysed
   frames (`Sampling/raw ring MB`): about 50 run-1008 frames, which is 2 s at
   today's analysis rate. So the page fetches the raw event the moment you
   **Freeze** (or press **Single ▸**) and keeps it while the frame is frozen, and
   Download works for as long as you like. A live, unfrozen frame that has
   already left the ring shows "no longer held — use the tag". The same holds
   for an older frame the seeded view went back to for a seed choice or filter
   (the yellow banner): Download works only while that frame is still in the
   raw ring, and the banner says when it is not.

On a page served over plain http from a hostname (not localhost), browsers
refuse the clipboard API; the page then copies another way, and if that fails
too it shows the text selected in a small box for Ctrl-C.

**Later, in the analysis:**

```bash
mdqm-sma-file --serial 4757 --run 1008 --dir /path/to/raw            # scans run01008_*.mid* in order
mdqm-sma-file --serial 4757 --run 1008 --dir /path --words 12340:12350   # only those words (inclusive)
mdqm-sma-file --serial 4757 run01008_00024.mid.lz4                    # or name the file(s)
```

It prints the tag, the file, subrun and **event position**, a table (word index,
raw hex, ch, ToT, fine, coarse, time, fine/coarse ok or MISMATCH), and writes
`sma_run<run>_serial<serial>.mid` (in `--out`, or `--event-out PATH`). With
`--words` every word in the range is listed, filler and pixel words too (a pixel
word with its chip, column, row, TS2, ToT and time). Without
it, the first 200 trigger words are listed (`--max-lines`). It stops at the first
match; exit 0 found, 1 not found.

**The nearline rec ntuple** numbers entries by the event's **position** in its
subrun file: 0-based among *all* events of the file, the begin-of-run record and
the WaveDREAM events included. It does not use the serial.
`mdqm-sma-file --serial` prints that position, e.g. serial 0 of run 1008 is
position 10 of `run01008_00000`. That agrees with the raw-word study, whose event
list (`scratch/sma-raw-check/examples.json`, made against the rec ntuple) has
`event_index == rec_entry` for all 117 events. So: tag → `mdqm-sma-file
--serial` → subrun and position → rec entry.

On the local rig, `replay.sh` sends the file's own serials and times
(`replay-run.py --keep-header`) and sets the run number from the file name, so
the tags there match the files as well. With `--loop` the serials start again
each pass, which the analyzer treats as a restart. The replayer re-packs events,
so the 4 reserved bytes of each bank header in a downloaded rig event are 0
where the file has whatever the frontend left there; every data word is
identical. At PSI the download is the event exactly as the frontend wrote it.

## Manual path: `mdqm-sma-file` (analyzer or DAQ down)

The same plugin over one file, no MIDAS needed (`src/mdqm/tools/sma_file.py`).
It uses the default settings, not the ODB. The default channel roles are the
run-1015 cabling; for an older run pass its roles, its mismatch flag channels and
no NIM copies (`tests/sma_layouts.OLD_LAYOUT`):

```bash
mdqm-sma-file run01008_00001.mid.lz4 --settings '{"Channel roles": {"s1": 1, "counters": [1, 2, 3, 4, 5], "rf": 6, "current": 7, "delayed": [8, 9, 10]}, "Self check": {"mismatch flag channels": [1, 2, 3, 4, 5, 6]}, "NIM": {"channels": [-1]}}'
```

Without the `NIM` part the old roles collide with the NIM defaults: NIM is
then off anyway, but the `settings` warning stays.
**It is not CPU-budgeted: it analyses
every frame of the file** (it is a batch job, not a guest on the DAQ PC). The
S1 cap (`Cuts/max S1 per frame`) is a cut and applies in both, so the CLI and an
analyzer started with `--no-cpu-budget` on the same frames give identical histograms; an analyzer
that sampled has the same rates and fractions but fewer entries.

```bash
mdqm-sma-file /path/run01008_00001.mid.lz4                    # -> ./sma-file-1008_1/
mdqm-sma-file run00342.mid.lz4 --shift 3 --frames 300 --out /tmp/342
mdqm-sma-file FILE --settings '{"Cuts": {"coinc window ns": 30}}' --no-png
mdqm-sma-file FILE --settings my-sma-settings.json --skip 10 --frames 50
```

Options (`sma_file.py:489-511`): `--shift N` (default 14), `--frames N`,
`--skip N`, `--out DIR` (default `./sma-file-<run>_<subrun>/`), `--settings`
(inline JSON or a JSON file in the `/DQM/SMA` layout; unknown keys are an error),
`--no-png`, `--quiet`, `--merge` / `--no-merge` (`NIM/merge`, default n).

Outputs in the output directory:

| File | Content |
|---|---|
| `hists.npz` | every histogram, identical to what the analyzer holds when it analysed the same frames (`--no-cpu-budget`; verified for runs 682 and 1008) |
| `summary.json` | exactly what the SMAPlots page shows (chips, table, flags, shift check) |
| `trend.json` | the 1 s trend rows |
| `summary.png` | one-page overview |

A text summary goes to the terminal: frame classes, live fraction, shift check,
rates, mismatch per channel, efficiencies, the MuPix in-time fractions, flags.

Exit status (`sma_file.py:71-73`, `:597`): **0** fine (warnings allowed),
**3** at least one error flag (for example a wrong shift, or the S5 mismatch),
**2** bad argument or unreadable file.

If it reports `shift_mismatch`, run it again with the `--shift` it names; that
tells you which shift the board used for that file. Example: run 342 with the
default gives exit 3, "Coarse shift 14 is configured but 3 fits better"; with
`--shift 3` the check is `ok`.

## Troubleshooting

**Analyzer not connected (`no analyzer` chip).** Check the process is running
(`ps aux | grep mdqm-analyzer`) and that `sma_analyzer` is listed on the MIDAS
Programs or Status page (or run `scripts/start-sma-analyzer.sh --experiment <EXPT> --status`).
If not, start it (step 3). Its log says
`lost MIDAS (...); retrying in Ns` when it lost the connection and is retrying
on its own (`analyzer.py:604`).

**The analyzer died after a MIDAS restart.** If the experiment's shared memory
was recreated (all clients stopped, `/dev/shm` files removed, ODB reloaded), or
the analyzer process was frozen for more than 10 s (MIDAS's watchdog removes it),
MIDAS itself aborts the analyzer on its next access ("Cannot continue,
aborting..."). Its retry loop cannot catch that. Start it again (step 3); the
plots start from zero. A restart of mhttpd, the frontends or the logger alone
does not affect it: the pages come back as soon as mhttpd is back, with the
plots intact.

**Stale buffer readers.** The ODB list `/System/Buffers/SYSTEM/Clients` can keep
names of clients that have exited (it is a record, not the live list); the
Buffers page of mhttpd shows who is really attached. A reader that stops reading
(for example a frozen analyzer) does not block the DAQ: its requests are
non-blocking, and MIDAS removes it after its 10 s watchdog timeout.

**After mhttpd restarts, SMAPlots keeps saying "Could not update this tab".**
The plots do update again; the note is left over from the outage. Reload the
page (this is a known page bug as of 2026-09-29).

**The page looks old after an update of this package.** mhttpd tells browsers to
cache `.js` and `.css` files for 24 h. The HTML pages load them with a version
tag (`pages/sma.html:13-18`, `pages/sma-events.html:13-17`, for example
`dqm-sma.js?v=4`); whoever changes a script bumps its `?v=`. As a shifter, a hard
reload (Ctrl-Shift-R) fixes it.

**A raster reply is truncated or slow.** The page asks for up to 512 kB per
`sma::frame` (`pages/js/dqm-brpc.js:26-30`); a whole 40000-word frame is about
230 kB. Replies measured 15-60 ms even with the DAQ flooding the buffer.

**In a container, clients fail with `BM_CORRUPTED, mismatch of buffer name in
shared memory`.** Docker gives containers a 64 MB `/dev/shm` by default. The ODB
plus the SYSTEM buffer plus SYSMSG must fit; a 64 MB SYSTEM buffer does not.
Either make the SYSTEM buffer smaller (`/Experiment/Buffer sizes/SYSTEM`, set
before the buffer is first created) or recreate the container with
`--shm-size`. See `scratch/sma-dqm-standalone/README.md` for the local rig.

**The analyzer analyses fewer frames than the DAQ sends.** That is the design:
it analyses what fits in its CPU budget (`analysed X % of frames`). On a busy
machine each frame costs more CPU, so the same budget analyses fewer frames; the
analyzer backs off rather than taking more. If the plots fill too slowly, the
DQM expert can raise `/DQM/SMA/Sampling/CPU budget %`.

## Resource requirements

**The analyzer never uses more than its CPU budget** (`/DQM/SMA/Sampling/CPU
budget %`, default 20 % of one core). At a high rate it analyses a sample of the
frames. The rate does not change what it costs, only how many of the frames it
analyses.

### Under the CPU budget, 1x to 10x run 1008

Measured on 2026-09-29 (`docs/profile-sma-highrate.json`; scripts and raw data
in `scratch/sma-dqm-profile/highrate/`, `RESULTS.md` there). Run 1008's frames
had their hit times compressed k times (`scratch/sma-dqm-standalone/make_dense.py`:
40000-word frames, k times shorter, rates exactly k times higher). They were
replayed at k x 35 frames/s, so x10 is 350 frames/s, 14 M words/s, 112 MB/s. The
analyzer ran under `nice -n 19 ionice -c3 prlimit --as=1 GiB` with the default
budget of 20 %. Each row is two repeats of 60 s, and the reply latencies are
from dqm::status, sma::summary and sma::frame (raster) probed through mhttpd at
1 Hz each, with SMAPlots and SMAEvents also open in a browser for the "pages
open" figures:

| Load | Offered frames/s | Analyzer CPU, mean / p95 of 1-s samples | Analysed | RSS | Replies p95 (status / summary / raster) | Rates shown vs true |
|---|---|---|---|---|---|---|
| no data | 0 | 1.1 % / 2 % | – | 110-120 MB | 4 / 4 / 10 ms | – |
| x1 (today) | 35 | 18.1 % / 20-22 % | 50-61 % | 109-115 MB | 7-18 / 3-6 / 9-13 ms | within 0.2 % |
| x2 | 71 | 18.0 % / 19-21 % | 32-36 % | 113-118 MB | 3-8 / 3-8 / 9-13 ms | within 0.1 % |
| x5 | 177 | 18.0 % / 19-22 % | 11-15 % | 114-118 MB | 7-14 / 3-15 / 9-18 ms | within 0.1 % |
| x10 | 350 | 18.0 % / 20-22 % | 6-8 % | 115-119 MB | 8-12 / 3-13 / 9 ms | within 0.25 % |

* **CPU**: at the budget's 90 % target at every rate, pages open or closed. The
  p95 of the 1-s samples is at most 2 points above the budget.
* **Other budgets at x10**: 5 % gives 4.6 % of a core (1.2 % of the frames
  analysed), 50 % gives 45 % (21 %). At budget 0 the analyzer only skips: it uses
  0.9-1.0 % of a core at 350 frames/s, including answering the probes.
* **Without a budget** (`--no-cpu-budget`, the old "process all") at x10: 91 % of a core
  (p95 100 %), 46 % of the frames, replies p95 about 60 ms.
* **A rate step** from x1 to x10: the largest 0.5-s CPU sample afterwards was 24 %,
  and the CPU was back within budget 0.6 s after the step.
* **A busy machine** (8 busy loops in the container, the machine 53 % busy): the
  CPU stays at 18 %; each frame costs more, so fewer are analysed (x10: 5.2-5.7 %
  instead of 5.1-7.6 %; x1: 55-58 % instead of 48-75 %). It backs off, it does
  not take more.
* **The DAQ side**: the replayer sends 349.8 frames/s with no analyzer and
  349.6-350.1 frames/s with one at any budget. Its time inside `send_event` is
  0.10-0.16 ms per frame on average, at most 3 ms, with or without an analyzer,
  so it is never held up. Only the 8 busy loops slowed the replayer itself, to
  347 frames/s.
* **Memory**: the analyzer's private memory went from 46 to 48 MB over 40 minutes
  (the 10-minute trend fills in the first ten; this was measured before the raw ring,
  which adds up to `Sampling/raw ring MB` = 16 MB). RSS is 110-121 MB, about 50 MB
  of which is the SYSTEM buffer and ODB mapped into it.
* **Dense frames**: frames of 400000 words (ten times denser, at 35 frames/s)
  stay at the budget, 17.9 %. One frame takes about 56 ms, the S1 cap analyses
  5.6 % of their S1 hits, the slowest reply was 96 ms and RSS peaks at 180 MB.
  Frames of 1.12 M words (9 MB, above an 8 MB MAX_EVENT_SIZE) take about 135 ms
  each; RSS then peaks at 311 MB, under the 1 GiB ceiling.

At today's rate the default budget analyses about 60 % of the frames. One
analysed 1008 frame costs 7-9 ms of CPU live: 5.3 ms in `process()` (with the S1
cap; 6.2 ms without) plus reading, cold caches and replies. Analysing every
frame at 35 frames/s would take a budget of about 30-35 %.

**The MuPix analysis** (measured 2026-09-29 on run 1008, 40000-word frames with
~7300 pixel words; `scratch/sma-dqm-mupix/`): `process()` costs 0.6-0.8 ms more
per frame on one core (5.5-6.4 ms -> 6.3-7.0 ms, three interleaved repeats of
120 frames; about +12 %): decoding the pixel words 0.2 ms, the S1 matching and
the t(pixel) - t(S1) pairs 0.2 ms (at most 500 S1 hits a frame), the
occupancy and ToT fills 0.1-0.2 ms. It is inside the CPU budget like
everything else, so the analyzer does not use more CPU: live, with the replay
at 35 frames/s, it ran at 18.3 % of a core (p95 26 %, budget 20 %) and
analysed 60.5 % of the frames, against 18.1 % and 61.5 % before (the machine
was busier then, so the difference is within the scatter; offline it is ~12 %
fewer frames). Its memory did not change (RSS 203-212 MB, private 132-140 MB
after 10 minutes, the same as before MuPix after an hour); a seed-ring frame
holds ~0.2 MB more for its pixels. The event display's replies grow: a seeded
reply from 2.1 to 2.8 kB, a raster reply from 203 to 272 kB (709 instead of
549 kB with words); *hide MuPix* gives the old 203 kB back. Setting `MuPix/max
pixel hits per frame` to 0 turns the MuPix analysis off.

### Per-frame cost, every frame analysed (no budget)

The measurements below are from before the budget (`docs/profile-sma.json`,
2026-09-29, analyzer analysing every frame). They give the cost of the analysis
itself. Under a budget they set how many frames fit into it, not how much CPU
is used.

Measured on 2026-09-29 on this laptop (Intel Core Ultra 7 356H, 16 logical CPUs
under WSL2, the `testbeam-midas` container capped at 8 CPUs), with real run
files replayed into the local `smadqm` experiment. Every number below is in
`docs/profile-sma.json`; the scripts and raw samples are in the workspace at
`scratch/sma-dqm-profile/` (`RESULTS.md` there explains each measurement). CPU
is given as a percentage of one core. The laptop was shared with other heavy
jobs for part of the day; the numbers quoted here are from the repeats taken
while the rest of the machine was less than 25 % busy, and the effect of load is
described separately.

### The analyzer

At the current beam (run 1008: 40000-word frames at 35 frames/s, 1.4 M words/s)
**one SMA analyzer needs about 25 % of one core and about 115 MB of RAM**
(`analyzer_vs_rate`):

| Data | Frames/s | Analyzer CPU | CPU per frame |
|---|---|---|---|
| no data (run stopped) | 0 | 0.2 % | – |
| run 480 (4000 words) | 11 | 3 % | 2.8 ms |
| run 682 (4000 words) | 77 | 11 % | 1.5 ms |
| run 1008 (40000 words) | 35 | 25 % | 7.4 ms |
| run 1008, twice the rate | 70 | 45 % | 6.4 ms |
| run 1008 | 100 | 66 % | 6.6 ms |

A rule of thumb that fits these rows: **about 0.9 ms per frame plus 0.14 ms per
1000 words**, on one core of this laptop. The design target of 4 M words/s
would take about 60-65 % of a core. The ceiling measured in the stress test
(`docs/stress-sma.json`) is about 140 frames/s of 40000-word frames, i.e.
5.6 M words/s, on one core.

Where the time goes, for a 40000-word frame (`per_frame_breakdown_offline`,
confirmed on the live analyzer with py-spy, `live_pyspy_share`): filling the
histograms about a third, coincidences and time differences 17-25 %, the
stale-hit filter and time sort 13-17 %, fine/coarse checks 13 %, RF phase 5 %,
decoding the words 4 %, and, live only, reading the MIDAS buffer 8-10 %. For small (4000-word)
frames the histogram filling is half of the time: it costs about 1.2 ms per
frame whatever the frame size, because there are 45 histograms to update.

Open pages cost the analyzer little: answering SMAPlots and SMAEvents (raster)
adds about 1.5 % of a core with no data and is within the scatter at 35
frames/s. Building one whole-frame raster takes about 1 ms and is done once per
frame, however many viewers ask for it (`replies_1008`).

**RAM.** The analyzer starts at about 94 MB (Python, numpy, MIDAS, the empty
histograms) and settles at about 115 MB after ten minutes, once the 10-minute
trend is full. About 50 MB of that is the MIDAS SYSTEM buffer and ODB mapped
into the process, which it shares with every other MIDAS client; its own
(private) memory is about 50 MB. The analyzer's own data are small: the 45
histograms hold 0.3 MB, the 10-minute trend 1 MB, the last frames kept for the
event display 2.3 MB, the cached raster 0.2 MB (`static_footprint_1008`).
**No leak was found** (`memory`, `tracemalloc_ring_full`): with run 1008 looped at
35 frames/s and pages open, the analyzer's private memory (resident plus swapped
out) stayed at 46.0 MB from 37 to 67 minutes after its start, +0.03 MB/h. An
analyzer that had been running for 7.75 hours (1.06 million frames) was at
115 MB, the same as after ten minutes.

Without a budget (`--no-cpu-budget`) plan on one core and 256 MB, and do not let it
share a core with heavy batch jobs. When other jobs kept this laptop 90 % busy,
the same frames cost twice the CPU (12-13 ms instead of 6.5 ms per frame). At
100 frames/s the analyzer then fell behind and processed about 70 frames/s (the
rest are counted as "missed (serial gaps)"; the DAQ is never slowed). **With the
default budget, plan on 20 % of a core and 256 MB**, whatever the rate (RSS
about 155 MB with the 16 MB raw-event ring). Raise
the address-space limit if the SYSTEM buffer is larger than about 600 MB.

### Next to the WaveDREAM analyzer

With both analyzers on run 1008 (35 SMA frames/s and about 310 WaveDREAM
events/s, `wd_plus_sma_on_1008`, measured before the budget): SMA 27 % of a core
and 115 MB, WaveDREAM 2-3 % of a core and 108 MB (it samples 20 events/s by
design), mhttpd about 1 %. With the SMA budget at its default the two analyzers
together stay under a quarter of a core at any rate. **Budget one core and 1 GB
of RAM** for the two analyzers, mhttpd and headroom.

### mhttpd and the network, per open page

Each open page costs mhttpd a fraction of a percent of a core
(`mhttpd_pages`, 30 s per point, twice, against a baseline with no page open;
mhttpd idles at 0.6-1 % with the replay running):

| Page, tab | mhttpd CPU per viewer | Data per viewer |
|---|---|---|
| SMAPlots, Health | 0.5 % | 35 kB/s |
| SMAPlots, ToT / corruption | 0.6-0.8 % | 110 kB/s |
| SMAPlots, Timing | 0.4-0.5 % | 26 kB/s |
| SMAPlots, RF / delayed | 0.5 % | 120 kB/s |
| SMAPlots, Trends | 0.2 % | 10 kB/s |
| SMAEvents, seeded (4 Hz) | 0.2-0.25 % | 7 kB/s |
| SMAEvents, raster (2 Hz), current (ch 7, run 1008) hidden | 0.2 % | 390 kB/s |
| SMAEvents, raster (2 Hz), current (ch 7, run 1008) shown | 0.2-0.25 % | 430 kB/s |

These were measured before the NIM copies. With NIM copies configured
(run 1015 on) every frame carries the per-hit TOT + NIM class (smaf v3): the
live raster is 8 instead of 7 bytes a hit, so about 445 instead of 390 kB/s
(+14 %); the partner indices travel only with the word data (the seeded view,
a frozen raster: 24 instead of 19 bytes a hit). The table is to be re-measured
on NIM data.

So three shifters with pages open cost mhttpd 1-2.5 % of a core. The raster
is the only heavy one on the network: about 3.5 Mbit/s per viewer, which
matters only over a slow remote link (set its update to 0.5 Hz there). mhttpd's
own memory grew from about 100 MB to about 170 MB as pages were first opened
and then stayed there; as with the WaveDREAM pages this is allocator working
set, not a leak.

### The browser

One open view costs its browser about 10-15 % of one core
(`browser_per_view`; headless Chromium, which draws in software, so a desktop
browser with a graphics card should need less): SMAPlots Health 15 %, Trends
13 %, SMAEvents seeded 10 %, raster 13-14 %. The JavaScript heap stays at
2-3.5 MB after garbage collection, flat over 35 minutes (the Trends tab grows by
0.6 MB during its first 10 minutes while the trend fills, then stays flat), and
the number of page elements does not grow.

### The offline CLI

`mdqm-sma-file` on one subrun takes 1-4 s and at most 90 MB of RAM (`cli`):
about 2 s for a 1008 or 682 subrun without the PNG, 1 s more with it.

## For developers

### How the CPU budget works

* **Controller** (`CpuBudget` in `src/mdqm/dqm/analyzer.py`). Every second it
  takes the process's CPU time (`time.process_time`: all threads, so buffer
  reads, decoding, fills and brpc replies all count) and the frames analysed
  over the last 2 s, and sets the analysis rate to
  `analysed/s x 0.9 x budget / CPU fraction`: the rate at which the measured
  cost per frame uses 90 % of the budget. The idle cost is inside the measured
  fraction, so it settles a little below 90 %, never above. It raises the rate
  by at most 2x what was actually analysed per step (bounds the overshoot when
  the DAQ rate jumps), never goes below 0.2 frames/s while the budget is above 0,
  and never above `max events per s`.
* **Reading** (`Analyzer._run_budget`, `BufferOps`). Frames are analysed in
  slots of two consecutive frames. A slot skips first only when the budget is
  the limit: when the offered rate of the requested events (from their serial
  numbers, over the last 2 s) is above 95 % of the rate the budget allows.
  Otherwise frames are read in order and none is skipped. (The buffer level
  cannot decide this: it counts every event in SYSTEM, WaveDREAM's too.) The
  skip is `bm_skip_event`, which sets this client's read pointer to the buffer's write
  pointer (`midas.cxx`, `bm_skip_event`: `pclient->read_pointer =
  pheader->write_pointer`): O(1), under the buffer lock, nothing copied. The
  next frame read is the next one written. The python client has no wrapper,
  so it is called through the library already loaded
  (`_Z13bm_skip_eventi`, `_Z19bm_get_buffer_leveliPi` in
  `libmidas-c-compat.so`; measured 1 us per call). The request stays
  `GET_NONBLOCKING`, so the writer is never held up by this client in any case.
* **Why not `GET_RECENT`.** It does not skip to the newest event. In
  `bm_check_requests` (`midas.cxx`) a `GET_RECENT` request only refuses an
  event whose header time stamp is more than 1 s older than now (1 s
  resolution); every event younger than that is still copied out in order,
  exactly as with `GET_NONBLOCKING`. It saves nothing at a steady high rate.
  (Events that match no request are passed over without a copy in either mode;
  the copy happens in `bm_read_buffer` only for requested events.)
* **Offered frames** are counted from the serial numbers of the analysed ones
  (a jump from 10 to 14 means 4 offered), so skipped frames cost nothing to
  count; after a MIDAS reconnect the baseline restarts, so an outage is not
  counted as offered frames. The gap, the live fraction and the rates are only
  computed on a frame whose predecessor (the previous serial) was analysed too.
* **Errors in the plugin** (for example a MemoryError) drop that one event,
  are counted (`plugin_errors` in `dqm::status`) and reported once per error
  type; only MIDAS errors send the analyzer through its reconnect loop.
* **Rates** divide the hits by the time the frames cover, taken from each
  frame's fine/coarse-consistent hits (`Frame.timed_extent`). A few faulty S5/RF
  words (coarse - fine at half a wrap) decode one 2^20 ns wrap early and widen
  the all-hit extent by ~1 ms; with every frame analysed the overlap of
  consecutive frames removed that, but a lone sampled frame has no neighbour,
  and its rates read 4 % low at the 1008 rate and 25 % low at ten times it.
  With the consistent-hit extent, every-frame and every-third-frame rates agree
  to 0.01 %. The span, gap and live-fraction histograms keep the all-hit extent
  (`PISMAWord` / `PITMidasMusip`).
* **WaveDREAM** is unchanged: its settings tree has no `CPU budget %`, so it
  keeps the token bucket at `max events per s` and drains the buffer as before
  (`tests/test_cpu_budget.py::test_the_wavedream_analyzer_is_untouched_by_the_budget`).

* Local rig with replayed run files: `scratch/sma-dqm-standalone/README.md`
  (workspace scratch area, not in this repository).
* High-rate test data: `scratch/sma-dqm-standalone/make_dense.py` (1008 frames with
  their times compressed k times, and `--merge N` for N-times-denser frames); results
  in `docs/profile-sma-highrate.json`.
* Load test (start the analyzer with `--no-cpu-budget`, a development flag: it
  measures the lossless ceiling): `scripts/stress-analyzer.py FILE --event-id 301 --client sma_analyzer
  --status-cmd dqm::status --limit-path "/DQM/SMA/Sampling/max events per s"
  --frame-cmd sma::frame --frame-args '{"view": "raster"}'`; results in
  `docs/stress-sma.json`.
* The smaf wire format, pixel block included: `src/mdqm/dqm/framing.py` ("SMA
  frames", "pixel block"), decoded by `pages/js/dqm-smaframe.js`. The pixel
  block (header flag `SMAF_PIXELS`) comes after the trigger-hit arrays in
  either version and is ignored by a v1/v2 decoder that predates it; `sma::frame
  {"pixels": false}` leaves it out. `framing.smaf_drop_pixels` plus dropping the
  `mupix` meta key gives the payload without MuPix byte for byte, which
  `tests/test_sma_seed_choice.py` checks against the hashes taken before MuPix
  existed (`tests/data/sma_seeded_default_golden.json`, `hashes`; the payloads as
  sent now are `hashes_mupix`).
* Pixel decode golden file: `tests/generate_mupix_golden.py` (in the container:
  compiles `PIMuPixWord.hh`, imports the psm-analysis references), checked by
  `tests/test_mupix_golden.py`.
* Per-frame cost on real frames: `scripts/bench-sma.py FILE` (gate: 100 frames/s of
  40000-word frames on one core).
* Cost of a page for mhttpd, the analyzer and the network: `scripts/profile-page.py
  --driver chromium --container testbeam-midas --url
  'http://localhost:8123/?cmd=custom&page=SMAEvents' --click '#dqm-smaev-tab-raster'
  --analyzer-client sma_analyzer --experiment smadqm --dropped-path '' --tabs 0 1 3
  --repeat 2` (the defaults still profile the WaveDREAM Scalers page).
  Resource figures: `docs/profile-sma.json`, section "Resource requirements" above.
