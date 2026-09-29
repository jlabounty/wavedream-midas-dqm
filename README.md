# midas-dqm

Data-quality monitoring for MIDAS experiments, built as **mhttpd custom pages**
plus (later) one analyzer client. No nginx, no node, no ROOT, no ZMQ, no second
web stack: the pages are plain HTML and JavaScript served by the mhttpd the
experiment already runs, on the port operators already have open.

Two halves, kept apart on purpose:

| generic — works at any MIDAS experiment | experiment-specific |
|---|---|
| `src/mdqm/dqm/`, `src/mdqm/install/` | `src/mdqm/plugins/` |
| `pages/js/dqm-common.js` | `pages/js/dqm-scalars.js` and friends |

## Install

```bash
pip install -e .
mdqm-register-pages --experiment MYEXPT      # writes the /Custom keys
```

Then open the experiment's mhttpd and pick **Scalers** or **Scope** from the side menu.

| page | what it shows | needs |
|---|---|---|
| **Scalers** | rates, thresholds, trends, DAQ health | nothing — reads the ODB and MIDAS history |
| **Scope** | live waveforms | nothing — reads the event buffer through mhttpd |

`--list` shows what would be registered, `--dry-run` says what would change,
`--check` verifies every registered key still resolves to a readable file, and
`--remove` unregisters (only the keys pointing into this checkout).

Registration is idempotent and safe to run on every start — that is how a moved
checkout heals itself.

## SMA (MuSiP trigger/ToT board) DQM

A second analyzer plugin, `src/mdqm/plugins/sma.py`, decodes SMA readout frames
(event 301, bank `H000`) and serves two more pages, **SMAPlots** (flags,
per-channel table, Health / ToT / Timing / RF / MuPix / Trends tabs) and **SMAEvents**
(S1-seeded events and a whole-frame raster). The MuPix pixel words of the same
bank are decoded too: MuPix lanes and rows on SMAEvents, the in-time share of
S1 hits per plane and an SMA <-> MuPix time-sync flag on SMAPlots. It runs as
its own client next to the WaveDREAM one, with its own ODB tree `/DQM/SMA`:

```bash
scripts/start-sma-analyzer.sh --experiment MYEXPT --python /path/to/python   # tmux session; --status, --stop
mdqm-sma-file run01008_00001.mid.lz4          # same plugin over one file, no MIDAS
```

`scripts/start-sma-analyzer.sh` needs no `pip install`: it runs the analyzer
from this checkout as a low-priority guest (one BLAS thread, `nice -n 19`,
`ionice -c3`, 1 GiB `prlimit`), refuses to start a second copy, and has
`--status`, `--stop` (waits up to 20 s for the client to detach) and
`--foreground`. Hooking it into a restart script, and the raw command it runs:
"Starting and stopping the analyzer" in docs/SMA-DQM.md.

**It is a live peek, bounded by a CPU budget, not a lossless record.** The
analyzer measures its own CPU and analyses only as many frames per second as fit
in `/DQM/SMA/Sampling/CPU budget %` (default 20 % of one core, everything
included: reading the buffer, decoding, filling, answering the pages). The
frames in between are skipped without being read (`bm_skip_event` moves the
client's read pointer to the newest event), so they cost nothing at any rate.
The pages say which share was analysed ("analysed N % of frames"); rates,
fractions and efficiencies are unaffected by the sampling, histogram counts are
from the sample. Dense frames are bounded too: the S1-seeded analyses use at
most `Cuts/max S1 per frame` (2000) S1 hits per frame. Measured at 1x-10x run
1008 in `docs/profile-sma-highrate.json`. The ODB cannot set more than 50 %;
`mdqm-analyzer --no-cpu-budget` (development only) restores the old behaviour
(every frame, in order, as long as one core keeps up); the offline
`mdqm-sma-file` is never budgeted. Frames over `Cuts/max words per frame`
(1 Mi words) are counted and not decoded.

The shifter guide, start to finish, is **[docs/SMA-DQM.md](docs/SMA-DQM.md)**:
every chip, flag and banner, the coarse-shift check, the `/DQM/SMA` settings,
the offline `mdqm-sma-file` (outputs `hists.npz`, `summary.json`, `trend.json`,
`summary.png`; exit 0 / 3 on an error flag / 2 on bad input) and troubleshooting.

Tools added with it:

| Tool | What |
|---|---|
| `mdqm-sma-file FILE [--shift N] [--frames N] [--skip N] [--settings JSON] [--out DIR]` | the manual path when the analyzer or DAQ is down |
| `mdqm-sma-file --serial N --run R --dir D [--words A:B]` | find one event by the tag copied from SMAEvents (run + serial): prints its words and event position (= nearline rec-ntuple entry), writes `sma_run<R>_serial<N>.mid`. See "Tracking down an odd event" in docs/SMA-DQM.md |
| `scripts/replay-run.py ... --keep-header` | send the file's own serials and time stamps, so tags from the pages match the file |
| `scripts/replay-run.py ... --event-id 301 --numpy` | `--numpy` (now the default; `--no-numpy` for the old path) sends banks as arrays, 5-6x faster on 40000-word frames; `--event-id` may be repeated |
| `scripts/stress-analyzer.py FILE --event-id 301 --client sma_analyzer --status-cmd dqm::status --limit-path "/DQM/SMA/Sampling/max events per s" --frame-cmd sma::frame --frame-args '{"view": "raster"}'` | load test for either plugin; `--lossless` sets the pass fraction (0.999); counts frames the buffer overwrote via the plugin's serial check. For SMA run the analyzer with `--no-cpu-budget` (otherwise it samples on purpose). Results: `docs/stress-sma.json` |
| `scripts/replay-run.py ... --report-s 5` | prints the achieved rate and the time spent in `send_event` every 5 s: whether a consumer ever held the producer up |
| `scripts/bench-sma.py FILE...` | per-frame cost of the SMA decode and analysis on real frames, one core (gate: 100 frames/s of 40000-word frames) |

## Sharing an experiment

The pages register as `/Custom/<name>` keys holding **absolute** paths, and never
read or write `/Custom/Path`. That is what makes them safe as a guest: another
group's frontend may rewrite `/Custom/Path` on every start (musip's does) without
affecting us at all.

Use `--prefix` to namespace the menu entries if a name might collide:

```bash
mdqm-register-pages --experiment Mu3e --prefix WD
```

The prefix must be space-free, for the same reason the keys are: the key becomes
both `?cmd=custom&page=<key>` and the argument to `mhttpd_init()`.

Registration **refuses** to overwrite a key whose value points outside this
checkout, and says what it found. It never writes `/Custom` as a subtree.

## Configuration

The scaler page discovers boards, banks, labels and history events from the ODB
at load; nothing is compiled in. Optional overrides live in `/DQM/Scalars`,
seeded by the installer. With that subtree absent the page uses identical
built-in defaults and says so, so it works on an experiment nobody has set up.

The one key worth knowing about is `Bank Pattern` — capture 1 is the role
letter, capture 2 the board id. Pointing the page at a different experiment's
naming scheme is a config change, not a code change.

## Development

```bash
python -m pytest                                    # no MIDAS needed
MDQM_NODE=/path/to/node python -m pytest            # + the JavaScript tests
```

The JS tests run the real page code against fixtures captured verbatim from a
live ODB, using a small DOM stub in `tests/js/domstub.js` rather than a browser.
Node is **not** a dependency — the pages have no build step — so those tests skip
where it is missing.

`tests/js/domstub.js` models mhttpd's refresh loop rather than approximating it,
because its contract is where this page is easiest to get wrong. Two rules are
worth knowing before touching a handler:

- **A `modb` watcher's first value fires `onload`, not `onchange`.** mhttpd
  stores it silently and fires `onchange` only on *subsequent* changes. A
  handler wired to `onchange` alone never runs while the ODB is static — which
  is exactly the case when the frontend it is monitoring has died.
- **A `modbvalue`'s `innerHTML` is rewritten every tick**, but `onchange` fires
  only on change. Anything that renders text from a handler is correct for one
  tick and then silently reverts.

Both of those shipped as bugs during development and are now regression tests.

### Working without a detector

`scripts/replay-run.py` feeds a recorded run file into a live event buffer, so
everything downstream of the buffer — the scope page, and later the analyzer —
can be developed and tested against real events on a machine with no hardware
attached:

```bash
scripts/replay-run.py run00201.mid.lz4 --rate 15 --loop --event-id 401
```

It refuses to run while a run is active unless you insist: with the logger
recording, replayed events would be written into the run file as though they
were real data, which is a corrupted dataset nobody would notice until analysis.
With the run stopped, mlogger is not reading the buffer and nothing reaches disk.

`--loop` matters for the time base: the DRS cell-width table rides only the run's
*first* event, so restarting the file is the only way to see one again.

### Seeing the page without a browser

```bash
scripts/shoot.py "http://localhost:8088/?cmd=custom&page=Scalers" out.png \
    --wait-for "document.querySelector('#dqm-root h2')" --console
```

`firefox --screenshot` is not usable here: it fires on the load event, which for
any MIDAS page is long before the content exists — the stock status page
photographs as the word "Loading...". `shoot.py` drives geckodriver over plain
WebDriver HTTP (no selenium dependency), waits for a condition you name, and can
dump the console and any uncaught exceptions. It exits non-zero if the condition
never becomes true, so it works as a test and not only as a camera.

### The one thing that will waste your afternoon

mhttpd stamps `Expires: <now + 24 h>` on anything served through `send_fp()`,
with no `ETag` and no `Last-Modified`. Pages are exempt because their `/Custom`
key contains no dot, which routes them through `show_custom_page()` instead —
but **assets are not**. Bump the `?v=` on the `<script src>` and `<link href>`
when you change a `.js` or `.css` file, or hard-reload (Ctrl-Shift-R).

## Troubleshooting

**`show_custom_page: Cannot open file ... errno 13`** — mhttpd's uid cannot read
the file. `mdqm-register-pages --check` tests readability from *its own* uid,
which is necessary and not sufficient; check the whole path is traversable by
whoever runs mhttpd.

**The side menu shows a page that 404s** — the checkout moved. Re-run
`mdqm-register-pages`; it rewrites values that are recognisably stale copies of
its own.

**`/Custom/Path` appeared as an empty string** — mhttpd creates it itself on the
first unmatched URL once `/Custom` exists. Expected; leave it alone.
