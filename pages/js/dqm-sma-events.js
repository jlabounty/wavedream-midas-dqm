//
// dqm-sma-events.js -- the SMA event display (/Custom/SMAEvents).
//
// Two views of the analyzer's latest frame, from sma::frame:
//
//   Seeded  up to four seeds of the latest good frame, each with every
//           channel that has a role in [-200 ns, +3 us] around it: one lane per
//           channel, each hit a bar from t to t+ToT. This is where a missing
//           counter, an RF burst in the wrong place, or a fine/coarse-corrupt
//           word is seen hit by hit. The seeds are S1 hits by default; the
//           toolbar chooses another channel or "any counter" (the first hit of
//           each counter cluster, so events without S1 show up), and filters
//           that keep only seeds with an oddity in their window. Both are this
//           viewer's own choice (sessionStorage), sent with every request; the
//           analyzer searches back through its last good frames for a match.
//           A banner says when the frame shown is not the newest one, and why.
//           "incomplete pattern" leaves out counters with a known timestamp
//           fault (the analyzer says which, per reply: meta.incomplete). A row
//           of per-counter any / present / absent selectors (S1..S5 within the
//           coincidence window of the seed) is AND-ed with all of that.
//   Raster  the whole latest frame, time vs channel, ToT as colour. This is
//           where a stale buffer, a dead channel or a frame-sized hole is seen.
//
// MuPix: the same bank carries the pixel words (smaf pixel block). The seeded
// view gets two more lanes, "MuPix L1" and "MuPix L2" (a third for chips the
// plane map does not know, when there are any), one tick per pixel hit coloured
// by its ToT, the in-time window shaded, and per seed a badge saying which
// planes had a hit in time; the "MuPix:" selector (any / L1+L2 / L1 or L2 /
// none in time) is AND-ed with the oddity filters. The raster gets the planes
// as extra rows (every pixel hit, noise included; "hide MuPix" drops them).
// A pixel hit is hit number nHits + j of its frame (j in the pixel block), so
// hovering, tagging, Copy all and the word ranges treat it like any other hit.
//
// TOT + NIM (since run 1015 every counter reaches the board twice, its TOT
// word and a NIM discriminator copy "S*k*L"): each NIM copy gets its own lane
// right under its counter, in the counter's colour but lighter. A thin dark
// tick joins a TOT word to the NIM word it is paired with; a NIM-only word
// (what the merge adds, or would add) is a hollow bar, one held back by a lag
// fault is grey (only with the merge on: nothing is held back from a merge
// that is off), and a TOT echo word has a dark back-slanted hatch. The pairing travels
// with the frame (smaf v3, per hit), and so do the roles: nothing here is
// guessed from labels when the analyzer says it. With the merge on, the
// pattern boxes and the per-counter selector are of the merged counters; with
// it off (the default) they are the TOT words alone, and the NIM-only hits are
// shown but not merged -- every legend and tooltip says which. A NIM copy
// arrives a cable delay after its TOT word (tens of ns), so its lane is drawn
// at t - NIM/offset ns, the offsets the frame was paired with (meta.nim_offsets_ns):
// a pair then sits one above the other. "raw times" draws the raw SMA times
// instead; the hover line gives both, and tags, word lists and Δt stay raw.
//
// Drawn on our own <canvas>, not mplot: mplot has no shapes (see dqm-evd.js,
// markEdge), and a 33k-hit raster as 16 scatter series would be both slow and
// the wrong picture. Everything on screen comes from one sma::frame reply, so
// every panel shows the same frame -- and the frame seq in the toolbar lets two
// people on two screens check that they are looking at the same one.
//
// Freeze stops polling and keeps the frame, so it can be pointed at. Single
// freezes too, then fetches exactly one newer frame for the visible tab per
// press -- stepping through frames by hand.
//
// Finding an odd frame again later: every frame carries the analyzer's tag
// (run, event, serial, time, frame seq -- Copy tag puts it in the elog), and
// "Download raw event" saves the frame's MIDAS event as a one-event .mid file
// while the analyzer still holds it (a few seconds; Freeze fetches it at once
// and keeps it). Hovering a hit shows its channel, ToT,
// time and, with word data, its index in the H000 bank and the raw 64-bit
// word. A click tags the hit: it joins the "Tagged hits" list under its
// frame's tag (click it again, or its x, to take it off), is outlined and
// numbered on the canvas while its frame is on screen, and the list keeps it
// across tab switches, new frames and (sessionStorage) a reload. The seeded
// view always carries word data; the raster only for a frozen frame (19
// instead of 7 bytes a hit is not worth it at 2 Hz), which the page asks for
// once on Freeze and with every Single -- so a click on a live raster freezes
// first and tags the hit once its word data is in.
//

(function () {
"use strict";

const LS = "dqm-sma-events-settings";

const TABS = [["seeded", "S1-seeded events"], ["raster", "Whole-frame raster"]];
const RATES = {
  seeded: [["250", "4 Hz"], ["500", "2 Hz"], ["1000", "1 Hz"]],
  raster: [["500", "2 Hz"], ["1000", "1 Hz"], ["2000", "0.5 Hz"]],
};
const PROMPT_NS = 150;
const MAX_HITS = 60000;          // raster reply cap, see fetchFor()
const SUMMARY_EVERY_MS = 10000;  // roles and labels change only on an ODB edit
const SINGLE_RETRY_MS = 300;     // Single asks once more after this if the frame is unchanged
// A raster with words is 19 bytes a hit (24 with the TOT + NIM pairing),
// ~1.2 MB (~1.45 MB) at MAX_HITS: ask for that
// once rather than the 512 kB poll default and a retry.
const WORDS_MAX_REPLY = 2 * 1024 * 1024;
const HIT_SLOP_PX = 4;           // how far from a hit the mouse may be and still point at it
const COPIED_MS = 1500;          // how long a copy button says "Copied"
const RAW_KEEP = 2;              // frozen frames whose raw event the page keeps (one per tab)
const MAX_TAGS = 200;            // tagged hits the list holds
const SS_TAGS = "dqm-sma-events-tagged";   // sessionStorage: the list survives a reload
const SS_SEED = "dqm-sma-events-seed";     // sessionStorage: this viewer's seed choice and filters
// The oddity filters (sma_words.FILTERS), in the order the analyzer uses.
const FILTERS = [
  ["incomplete", "incomplete pattern", "not every counter S1..S5 within the coincidence window of the " +
   "seed. Counters with a known timestamp fault (the SMAPlots fine/coarse flag) are left out, " +
   "since they are almost never in time; the header says which"],
  ["mismatch", "fine/coarse mismatch", "a hit in the window whose fine and coarse fields disagree"],
  ["tot", "ToT ≥ 250", "a hit in the window with a corrupt ToT code (Cuts/tot corrupt)"],
  ["rf", "RF not valid / vetoed", "the S1 hit's RF gate is not valid, or vetoed by another S1 hit"],
];
const TAG_MARK = "#000";         // a tagged hit's outline and number box
// The MuPix selector (sma_words.MUPIX_MODES): AND-ed with the filters above.
const MUPIX_MODES = [
  ["any", "any"], ["both", "L1+L2 in time"], ["either", "L1 or L2 in time"], ["none", "none in time"],
];
// The per-counter pattern selector (sma_words.PATTERN_STATES): AND-ed with everything else.
const PATTERN_STATES = [["any", "any"], ["present", "present"], ["absent", "absent"]];
// One selector per counter (S1..Sn from the roles), at most 8: the pattern is one byte.
const PATTERN_MAX = 8;
const PLANE_LABEL = { 0: "no plane", 1: "MuPix L1", 2: "MuPix L2" };
const IN_TIME_FILL = "rgba(44, 160, 44, 0.16)";   // the in-time window on the MuPix lanes

const LANE_COLOURS = {
  s1: "#1f77b4", counter: ["#1f77b4", "#2ca02c", "#17becf", "#9467bd", "#8c564b", "#bcbd22"],
  rf: "#ff7f0e", current: "#7f7f7f", delayed: "#e377c2",
};
// TOT + NIM hit styles (see hitStyle). A NIM lane's bars are its counter's
// colour, lighter (NIM_LIGHTEN of the way to white), with a 1 px outline in the
// counter's colour darkened by NIM_EDGE_DARKEN: the light fill alone is under
// 2:1 against the lane stripe, the outline is >= 3.6:1 for all six colours.
const NIM_LIGHTEN = 0.35;
const NIM_EDGE_DARKEN = 0.3;
const NIM_MIN_PX = 3;              // a NIM bar's least width (a 10 ns word is ~2 px on the full window)
const PAIR_TICK = "#3a3a3a";       // a TOT word joined to its NIM copy
const HELD_FILL = "#d4d4d4";       // a NIM-only word held back from the merge (lag fault)
const HELD_EDGE = "#6f6f6f";       // 4.6:1 on the lane stripe
const ECHO_HATCH = "rgba(0, 0, 0, 0.55)";   // a TOT echo word: dark, the other way to the mismatch hatch
const NIM_ONLY_FILL = "#ffffff";   // hollow: the outline is the lane edge (counter colour, darker)
const HOLLOW_MIN_PX = 6;           // a hollow/grey bar's least width, so its outline shows
const MISMATCH_ON_HOLLOW = "rgba(208, 0, 0, 0.7)";   // the mismatch hatch on a hollow bar
const PAIR_CLASS_TEXT = {
  0: "paired", 1: "TOT only (no NIM copy in the window)", 2: "echo word (not paired)",
  3: "NIM only",
};

const state = {
  client: "sma_analyzer",
  tab: "seeded",
  paused: false,
  zoom: "full",
  hideCurrent: true,
  hidePixels: false,             // the raster's "hide MuPix"
  rawTimes: false,               // "raw times": the NIM lanes at their raw time, not t - offset
  intervals: { seeded: 250, raster: 500 },
  // Which hits seed the seeded view, and which oddities it is limited to (OR):
  // this viewer's choice, sent with every seeded request (see seedArgs()).
  // pattern: {"<k>": "present"|"absent"} per counter k = 1..n (S1..Sn); "any" is left out.
  seedSel: { seed: "s1", filters: [], mupix: "any", pattern: {} },
  frames: { seeded: null, raster: null },
  lastNewAt: { seeded: null, raster: null },
  summary: null,
  summaryAt: 0,
  view: null,                    // raster x range in ms, null = the whole frame
  drag: null,
  updater: null,
  inflight: null,                // the sma::frame chain in flight (a poll tick or Single)
  singleBusy: false,
  seedPanels: [],
  rasterInfo: null,              // the hover line under the raster
  tagged: [],                    // the tagged hits, in the order they were clicked (see makeTag)
  taggedPending: [],             // {seq, i}: raster clicks waiting for the frozen frame's words
  wordsBusy: { seeded: false, raster: false },   // the word-data refetch in flight
  wordsNote: { seeded: null, raster: null },     // {seq, text}: why a frame has no words
  tagNote: { seeded: null, raster: null },       // {seq, text, cls}: the download outcome
  downloadBusy: false,
  // The raw event of a frozen frame, fetched at Freeze: the analyzer's raw
  // ring holds only the last few seconds of frames, so a frame frozen now may
  // be gone by the time someone presses Download. seq -> ArrayBuffer.
  rawKept: new Map(),
  rawBusy: new Set(),
};

window.addEventListener("load", function () {
  mhttpd_init(mhttpd_getParameterByName("page") || "SMAEvents", 1000);
  restore();
  restoreTags();
  restoreSeedSel();
  build();
  showTab(state.tab);
  window.addEventListener("resize", function () { redraw(); });
  window.addEventListener("keydown", function (e) {
    if (e && e.key === "Escape") clearHover();
  });
});

// ---------------------------------------------------------------------------
// Polling
// ---------------------------------------------------------------------------

/** Retire an updater for good: stop it and drop its visibility listener. */
function retire(u) {
  if (!u) return;
  u.stop();
  document.removeEventListener("visibilitychange", u._onVisible);
}

function startPolling() {
  // A fresh updater each time, never stop()+start() on one: a tick in flight
  // would re-arm after start() and leave two polling chains running.
  retire(state.updater);
  state.updater = null;
  if (state.paused) return;
  const tab = state.tab;
  const u = new BRPC.AutoUpdater(async function () {
    // One request in flight at a time: a Single (or the tick of a retired
    // updater) still waiting for its reply is let finish first.
    await settled();
    if (state.updater !== u || state.paused || state.tab !== tab) return;
    await track(refresh(tab));
  }, state.intervals[tab]);
  u.onError = (e) => setError(e);
  state.updater = u;
  u.start();
}

/** Record p as the request chain in flight; cleared when it settles. */
function track(p) {
  const t = p.finally(function () { if (state.inflight === t) state.inflight = null; });
  state.inflight = t;
  return t;
}

/** Wait until no request chain is in flight (its outcome is its owner's business). */
async function settled() {
  while (state.inflight) {
    try { await state.inflight; } catch (e) { /* reported by whoever started it */ }
  }
}

async function refresh(tab) {
  const frame = await fetchFor(tab);
  if (state.paused || state.tab !== tab) return;   // frozen or switched mid-request
  const prev = state.frames[tab];
  if (!prev || !frame || prev.frameSeq !== frame.frameSeq) state.lastNewAt[tab] = Date.now();
  state.frames[tab] = frame;
  render(tab);
}

/**
 * One sma::frame for this tab's view (plus the summary when it is due).
 * `extra` adds to the request, e.g. {words: true}.
 */
async function fetchFor(tab, extra) {
  if (!state.summary || Date.now() - state.summaryAt > SUMMARY_EVERY_MS) {
    // The roles decide the lanes and which channel "hide current" drops.
    // Without a summary the page still works from the frame's own labels.
    try {
      state.summary = await BRPC.json(state.client, "sma::summary", "");
    } catch (e) { /* keep the last one, or the fallbacks */ }
    state.summaryAt = Date.now();
    try { updateSeedOptions(); } catch (e) { /* the list keeps its fallback labels */ }
  }
  // max_hits: a bound on the reply, so a pathological frame is truncated (and
  // says so in the header) rather than growing the reply without limit. Real
  // 40k-word frames are ~33k hits.
  const args = Object.assign(tab === "raster" ? rasterArgs() : Object.assign({ view: "seeded" }, seedArgs()),
                             extra || {});
  return SMAF.fetchFrame(state.client, args, args.words ? WORDS_MAX_REPLY : undefined);
}

/**
 * Single: freeze (if live) and fetch one newer frame for the visible tab.
 *
 * Guarded so rapid clicks cannot stack requests: the button is disabled and
 * singleBusy set until the reply is in, and a poll tick still in flight is
 * let finish first. "Newer" is the point: a reply with the frame seq already
 * on screen (analyzer idle, or the seeded view's last frame with seeds
 * unchanged) is asked for once more after SINGLE_RETRY_MS, then said so in the
 * toolbar rather than silently redrawn.
 */
async function single() {
  if (state.singleBusy) return;
  state.singleBusy = true;
  const b = document.getElementById("dqm-smaev-single");
  if (b) b.disabled = true;
  try {
    // Single's own request asks for the words, so no separate refetch.
    if (!state.paused) setPaused(true, { words: false });
    await settled();
    await track(singleFetch());
  } catch (e) {
    if (typeof console !== "undefined") console.error("dqm-sma-events single", e);
    setError(e);
  } finally {
    state.singleBusy = false;
    if (b) b.disabled = false;
  }
}

async function singleFetch() {
  const tab = state.tab;
  const shown = state.frames[tab];
  const same = (f) => !!shown && (!f || f.frameSeq === shown.frameSeq);
  // A frame stepped to by hand is one to point at: the raster comes with its
  // words (the seeded view always has them).
  const extra = tab === "raster" ? { words: true } : null;
  let frame = await fetchFor(tab, extra);
  if (same(frame)) {
    await new Promise((res) => window.setTimeout(res, SINGLE_RETRY_MS));
    if (!state.paused || state.tab !== tab) return;
    frame = await fetchFor(tab, extra);
  }
  if (!state.paused || state.tab !== tab) return;   // resumed or switched mid-request
  if (same(frame)) {
    renderStatus(tab, shown);
    singleNote(`no newer frame than seq ${shown.frameSeq} yet`);
    return;
  }
  state.lastNewAt[tab] = Date.now();
  state.frames[tab] = frame;
  render(tab);
  keepRaw(tab);
}

/** The Single outcome note in the toolbar; "" hides it. */
function singleNote(text) {
  const n = document.getElementById("dqm-smaev-singlenote");
  if (!n) return;
  n.textContent = text;
  n.style.display = text ? "" : "none";
}

/** The raster's drop list: the current channel when hidden; nothing when there is none. */
function rasterDrop() {
  const cur = roles().current;
  return state.hideCurrent && cur !== null ? [cur] : [];
}

/** "hide the current channel" only where there is one (the roles may name none). */
function updateCurrentToggle() {
  const lab = document.getElementById("dqm-smaev-hidecurlab");
  if (lab) lab.style.display = roles().current === null ? "none" : "";
}

/** The raster request: the drop list, the reply bound, and "pixels": false when MuPix is hidden. */
function rasterArgs() {
  const a = { view: "raster", drop: rasterDrop(), max_hits: MAX_HITS };
  if (state.hidePixels) a.pixels = false;
  return a;
}

// ---------------------------------------------------------------------------
// Roles and lanes
// ---------------------------------------------------------------------------

/**
 * Channel roles: the analyzer's "roles" block (the ODB's channel map, in every
 * frame and in the summary), else from the summary's channel rows, else from
 * the frame's labels.
 *
 * The frame's block is preferred: it is the map the frame was analysed with.
 * `nim[k]` is counter k's NIM copy (null: none). The label fallback reads the
 * labels the analyzer put in the frame (S1..S5, RF, current are its
 * defaults), so a page whose summary call failed and whose analyzer is older
 * than the roles block still draws sensible lanes. With none of these there
 * are no roles at all (no counters, no RF, no current): nothing is guessed.
 */
function roles() {
  const out = { counters: [], nim: [], s1: null, rf: null, current: null, delayed: [], labels: [] };
  const s = state.summary;
  const f = state.frames[state.tab] || state.frames.seeded || state.frames.raster;
  const fm = (f && f.meta) || {};
  const block = (fm.roles && typeof fm.roles === "object") ? fm.roles : s && s.roles;
  if (block && Array.isArray(block.counters)) {
    out.labels = s && s.channels ? s.channels.map((c) => (c ? c.label : "")) : (fm.labels || []);
    return fromRolesBlock(block, out);
  }
  if (s && s.channels) {
    out.labels = s.channels.map((c) => (c ? c.label : ""));
    // Counter order is S1 first, then the order the efficiency list gives
    // (S2..S5), then any other counter by channel number. The efficiency list
    // is not the whole story: it need not carry S1 itself.
    const s1 = s.channels.find((c) => c && c.role === "s1");
    if (s1) out.counters.push(s1.ch);
    for (const e of s.efficiency || []) {
      if (e && typeof e.ch === "number" && out.counters.indexOf(e.ch) < 0) out.counters.push(e.ch);
    }
    for (const c of s.channels) {
      if (!c) continue;
      if (c.role === "rf") out.rf = c.ch;
      else if (c.role === "current") out.current = c.ch;
      else if (c.role === "delayed") out.delayed.push(c.ch);
      else if ((c.role === "s1" || c.role === "counter") && out.counters.indexOf(c.ch) < 0) {
        out.counters.push(c.ch);
      }
    }
    // A counter's NIM copy: its row's pair_of (an analyzer with NIM copies
    // sends the roles block too, so this is only for a summary without one).
    out.nim = out.counters.map(function (ch) {
      const row = s.channels[ch];
      const n = row && typeof row.pair_of === "number" ? row.pair_of : null;
      const nr = n === null ? null : s.channels[n];
      return nr && nr.role === "nim" ? n : null;
    });
    return out;
  }
  const labels = fm.labels || [];
  out.labels = labels;
  const byS = [];
  labels.forEach(function (l, ch) {
    const m = /^S(\d)$/.exec(l || "");
    if (m) byS.push([Number(m[1]), ch]);
    else if (l === "RF") out.rf = ch;
    else if (l === "current") out.current = ch;
  });
  out.counters = byS.sort((a, b) => a[0] - b[0]).map((x) => x[1]);
  out.nim = out.counters.map(() => null);
  return out;
}

/** The analyzer's roles block into roles(): -1 (none) becomes null, nim stays aligned. */
function fromRolesBlock(b, out) {
  const ch = (x) => (Number.isInteger(x) && x >= 0 && x < 16 ? x : null);
  const nim = Array.isArray(b.nim) ? b.nim : [];
  b.counters.forEach(function (c, k) {
    if (ch(c) === null) return;
    out.counters.push(c);
    const n = ch(nim[k]);
    out.nim.push(n !== null && n !== c ? n : null);
  });
  out.s1 = ch(b.s1);
  out.rf = ch(b.rf);
  out.current = ch(b.current);
  const taken = new Set([...out.counters, ...out.nim.filter((n) => n !== null), out.rf, out.current]);
  out.delayed = (b.delayed || []).map(ch).filter((c) => c !== null && !taken.has(c));
  return out;
}

/**
 * The NIM offsets a frame was paired with, by channel: `out[n]` is the
 * NIM/offset ns of the counter whose NIM copy is channel n, null for every
 * other channel. Null when the frame carries none (no NIM copies, or an
 * analyzer older than meta.nim_offsets_ns): its NIM lanes are drawn raw.
 */
function nimOffsets(frame) {
  const m = (frame && frame.meta) || {};
  const r = m.roles, o = m.nim_offsets_ns;
  if (!r || !Array.isArray(r.counters) || !Array.isArray(r.nim) || !Array.isArray(o)) return null;
  const out = new Array(16).fill(null);
  let any = false;
  r.nim.forEach(function (n, k) {
    if (!Number.isInteger(n) || n < 0 || n >= 16 || n === r.counters[k] || !Number.isFinite(o[k])) return;
    out[n] = o[k];
    any = true;
  });
  return any ? out : null;
}

/** What the views subtract from a hit's time, by channel: nimOffsets(), or null ("raw times"). */
function drawShift(frame) {
  return state.rawTimes ? null : nimOffsets(frame);
}

/** Hit i's time as drawn: t_rel, less its NIM offset when `sh` (drawShift) has one. */
function drawnT(frame, i, sh) {
  return frame.t[i] - ((sh && sh[frame.ch[i]]) || 0);
}

/** Whether the counters are the merged TOT + NIM hits: the frame's word, else the summary's. */
function nimMerge() {
  const f = state.frames[state.tab] || state.frames.seeded || state.frames.raster;
  const m = f && f.meta;
  if (m && typeof m.nim_merge === "boolean") return m.nim_merge;
  return !!(state.summary && state.summary.nim_merge);
}

/** `hex` (#rrggbb) moved fraction `f` of the way to black. */
function darken(hex, f) {
  const m = /^#([0-9a-f]{6})$/i.exec(hex || "");
  if (!m) return hex;
  const v = parseInt(m[1], 16);
  return `#${[16, 8, 0].map((sh) => Math.round(((v >> sh) & 255) * (1 - f))
    .toString(16).padStart(2, "0")).join("")}`;
}

/** `hex` (#rrggbb) moved fraction `f` of the way to white. */
function lighten(hex, f) {
  const m = /^#([0-9a-f]{6})$/i.exec(hex || "");
  if (!m) return hex;
  const v = parseInt(m[1], 16);
  const ch = (sh) => Math.round(((v >> sh) & 255) + (255 - ((v >> sh) & 255)) * f);
  return `#${[16, 8, 0].map((sh) => ch(sh).toString(16).padStart(2, "0")).join("")}`;
}

/**
 * How a hit is drawn, from its v3 cls byte (SMAF.PAIR; undefined before v3):
 * "solid" (paired, TOT-only, no class), "hollow" (NIM-only: merged into the
 * counter), "held" (NIM-only held back by a lag fault: grey) or "echo" (a TOT
 * echo word: hatched).
 */
function hitStyle(b) {
  if (b === undefined || b === null) return "solid";
  const c = b & SMAF.PAIR.CLASS_MASK;
  if (c === SMAF.PAIR.NIM_ONLY) return b & SMAF.PAIR.LAG_HELD ? "held" : "hollow";
  if (c === SMAF.PAIR.ECHO) return "echo";
  return "solid";
}

/** How many counters the pattern selector and the seed's pattern boxes show (0: roles unknown). */
function nPattern(r) {
  return Math.min(PATTERN_MAX, (r || roles()).counters.length);
}

function labelOf(ch, r) {
  const l = (r || roles()).labels[ch];
  return l || `ch${String(ch).padStart(2, "0")}`;
}

/**
 * Seeded-view lanes: counters in order (each with its NIM copy's lane right
 * under it, `nim: true`, `edge` the counter's colour), RF, delayed channels, current, then
 * the MuPix planes when the frame has a pixel block (L1 and L2 always, "no
 * plane" only when a shipped pixel hit is on a chip the plane map lacks).
 * A MuPix lane has `pix` (the plane code) instead of `ch`.
 */
function lanes(frame) {
  const r = roles();
  const out = [];
  r.counters.forEach(function (ch, k) {
    const colour = LANE_COLOURS.counter[k % 6];
    out.push({ ch, label: labelOf(ch, r), colour });
    // Its NIM copy right under it: the pair reads as one block of two lanes.
    const n = r.nim[k];
    if (n !== null && n !== undefined) {
      out.push({ ch: n, label: labelOf(n, r), colour: lighten(colour, NIM_LIGHTEN),
                 edge: darken(colour, NIM_EDGE_DARKEN),
                 nim: true, counter: ch });
    }
  });
  if (r.rf !== null) out.push({ ch: r.rf, label: labelOf(r.rf, r), colour: LANE_COLOURS.rf });
  for (const ch of r.delayed) out.push({ ch, label: labelOf(ch, r), colour: LANE_COLOURS.delayed });
  if (r.current !== null) {
    out.push({ ch: r.current, label: labelOf(r.current, r), colour: LANE_COLOURS.current });
  }
  for (const pl of pixelPlanes(frame)) out.push({ pix: pl, label: PLANE_LABEL[pl], colour: "#555" });
  return out;
}

// -- MuPix pixel hits: hit nHits + j of a frame is pixel j of its pixel block ----------

/** The frame's pixel block, or null (no MuPix in the frame, or hidden). */
function pixOf(frame) { return frame && frame.pixels ? frame.pixels : null; }

/** Hits of the frame, pixel hits included: the valid hit indices are 0 .. nAll - 1. */
function nAll(frame) { return frame.nHits + (frame.pixels ? frame.pixels.n : 0); }

function isPix(frame, i) { return i >= frame.nHits; }

/** Pixel j's time in ns from meta.t0_ns, the SMA hits' reference: one time axis for both. */
function pixRel(frame, j) {
  const m = frame.meta || {};
  const mp = m.mupix || {};
  return (mp.t0_ns - m.t0_ns) + frame.pixels.t[j];
}

/** The planes a frame's MuPix lanes/rows show: [1, 2], and 0 when a shipped hit has no plane. */
function pixelPlanes(frame) {
  const p = pixOf(frame);
  if (!p || !(frame.meta && frame.meta.mupix)) return [];
  for (let j = 0; j < p.n; j++) if (!(p.flags[j] & SMAF.PIX.PLANE_MASK)) return [1, 2, 0];
  return [1, 2];
}

/** A hit's H000 word index (SMA or pixel), or undefined without word data. */
function wordOf(frame, i) {
  if (!isPix(frame, i)) return frame.wordIndex ? frame.wordIndex[i] : undefined;
  const p = frame.pixels;
  return p && p.wordIndex ? p.wordIndex[i - frame.nHits] : undefined;
}

function rawOf(frame, i) {
  if (!isPix(frame, i)) return frame.rawWord ? frame.rawWord[i] : undefined;
  const p = frame.pixels;
  return p && p.rawWord ? p.rawWord[i - frame.nHits] : undefined;
}

/**
 * Where the hit colours stop on viridis. Its last tenth is pale yellow, which
 * all but disappears as a 1.5 px bar on the white and #f4f4f4 lanes, so the
 * largest ToT ends on yellow-green instead.
 */
const HIT_VIRIDIS_TOP = 0.85;

/** A pixel ToT colour, 0..31: viridis over the MuPix range (its own legend ramp). */
function pixColour(tot) {
  return DQM.viridis(Math.min(1, tot / 31) * HIT_VIRIDIS_TOP);
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function render(tab) {
  const frame = state.frames[tab];
  singleNote("");
  refreshTags(tab, frame);
  guard(`dqm-smaev-status`, () => renderStatus(tab, frame));
  // Not guard(): its error text would replace the bar's buttons for good.
  try { renderTagBar(tab, frame); } catch (e) {
    if (typeof console !== "undefined") console.error("dqm-sma-events tag bar", e);
  }
  if (tab === "seeded") renderSeeded(frame);
  else renderRaster(frame);
}

function redraw() { if (state.frames[state.tab] !== undefined) render(state.tab); }

/** Run one block; if it throws, say so in that block and leave the rest alone. */
function guard(id, fn) {
  try { fn(); } catch (e) {
    const node = typeof id === "string" ? document.getElementById(id) : id;
    showError(node, e);
  }
}

function showError(node, e) {
  if (typeof console !== "undefined") console.error("dqm-sma-events", e);
  if (!node) return;
  node.innerHTML = "";
  node.appendChild(el("div", { class: "dqm-error" },
    `could not draw this: ${BRPC.errorText(e)}`));
}

function renderStatus(tab, frame) {
  const chipEl = document.getElementById("dqm-smaev-live");
  const seq = document.getElementById("dqm-smaev-seq");
  seq.textContent = frame ? `frame seq ${frame.frameSeq}` : "frame seq —";
  if (state.paused) {
    chipEl.className = "dqm-chip dqm-sma-frozen";
    chipEl.textContent = "FROZEN";
    return;
  }
  if (!frame) {
    chipEl.className = "dqm-chip yellow";
    chipEl.textContent = "no frame";
    return;
  }
  const age = state.lastNewAt[tab] ? (Date.now() - state.lastNewAt[tab]) / 1000 : 0;
  if (age > 5) {
    chipEl.className = "dqm-chip yellow";
    chipEl.textContent = `no new frame for ${Math.round(age)} s`;
  } else {
    chipEl.className = "dqm-chip green";
    chipEl.textContent = "live";
  }
}

function setError(e) {
  const chipEl = document.getElementById("dqm-smaev-live");
  if (!chipEl) return;
  chipEl.className = "dqm-chip red";
  chipEl.textContent = "no analyzer";
  const diag = document.getElementById(`dqm-smaev-note-${state.tab}`);
  if (diag) {
    diag.className = "dqm-diagnosis red";
    diag.textContent = `'${state.client}' did not answer sma::frame ` +
      `(${BRPC.errorText(e)}). Start it with mdqm-analyzer --plugin sma ` +
      "--client sma_analyzer. " +
      (state.paused ? "Frozen: press Single or resume to try again." : "Retrying every 5 s.");
  }
}

function clearNote(tab) {
  const n = document.getElementById(`dqm-smaev-note-${tab}`);
  if (n) { n.className = ""; n.textContent = ""; }
}

function noFrameNote(tab, text) {
  const n = document.getElementById(`dqm-smaev-note-${tab}`);
  if (n) { n.className = "dqm-diagnosis yellow"; n.textContent = text; }
}

function badge(text, cls) { return el("span", { class: `dqm-sma-badge ${cls || ""}`.trim() }, text); }

/** Header chips shared by both views: which frame this is and what is in it. */
function frameHeader(frame, holder) {
  holder.innerHTML = "";
  const m = frame.meta || {};
  holder.appendChild(badge(`seq ${frame.frameSeq}`));
  holder.appendChild(badge(`run ${frame.run}`));
  if (m.serial !== undefined) holder.appendChild(badge(`serial ${m.serial}`));
  holder.appendChild(badge(`span ${ms(m.span_ns)} ms`));
  holder.appendChild(badge(m.gap_ns === null || m.gap_ns === undefined
    ? "gap —" : `gap ${ms(m.gap_ns)} ms`));
  holder.appendChild(badge(`${num(m.n_words)} words: ${num(m.n_filler)} filler · ` +
    `${num(m.n_pixel)} pixel · ${num(m.n_trigger)} trigger (${num(m.n_kept)} kept)` +
    (m.n_zero ? ` · ${num(m.n_zero)} zero (dropped)` : "")));
  if (frame.suspect || m.suspect) {
    holder.appendChild(badge(`SUSPECT time base${m.stale_reason ? `: ${m.stale_reason}` : ""}`,
                             "red"));
  }
  if (m.n_rescued) holder.appendChild(badge(`${num(m.n_rescued)} rescued`, "blue"));
  if (m.time_shift) holder.appendChild(badge(`times in ${2 ** m.time_shift} ns units`));
  if (frame.stale || m.stale) {
    holder.appendChild(badge(`STALE frame${m.stale_reason ? `: ${m.stale_reason}` : ""}`, "red"));
  } else if (m.class && m.class !== "good" && m.class !== "suspect") {
    // Any other class the analyzer grows (e.g. a suspect time base) is shown
    // verbatim rather than silently treated as good.
    holder.appendChild(badge(`frame class: ${m.class}${m.stale_reason ? ` (${m.stale_reason})` : ""}`,
                             "yellow"));
  }
  if (frame.truncated || m.truncated) {
    holder.appendChild(badge(`truncated: latest ${num(frame.nHits)} of ${num(m.n_selected)} hits shown`,
                             "yellow"));
  }
  if (m.shift !== undefined) holder.appendChild(badge(`coarse shift ${m.shift}`));
  const mp = m.mupix;
  if (mp && mp.per_plane) {
    const pp = mp.per_plane;
    let t = `MuPix ${num(mp.n_examined)} hits: L1 ${num(pp[1])} · L2 ${num(pp[2])}`;
    if (pp[0]) t += ` · ${num(pp[0])} no plane`;
    if (mp.n_skipped) t += ` (${num(mp.n_skipped)} more not examined)`;
    if (mp.truncated) t += `; latest ${num(frame.pixels ? frame.pixels.n : 0)} shown`;
    holder.appendChild(badge(t, pp[0] || mp.truncated ? "yellow" : ""));
  }
  const note = samplingNote();
  if (note) holder.appendChild(el("span", { class: "dqm-footnote dqm-smaev-sampling" }, note));
}

/**
 * "This is one frame in N": when the analyzer samples (its CPU budget), the
 * frame shown is the latest one it analysed, not the latest the DAQ sent.
 * From the summary this page already fetches every SUMMARY_EVERY_MS.
 */
function samplingNote() {
  const s = state.summary || {};
  const smp = s.sampling || {};
  const ok = (x) => typeof x === "number" && Number.isFinite(x);
  // Over the summary window; with nothing offered in it (a stopped run), the
  // share since the analyzer started.
  const since = !ok(smp.analysed_frac) && s.frames && ok(s.frames.analysed_frac);
  const af = since ? s.frames.analysed_frac : smp.analysed_frac;
  if (!ok(af) || af >= 0.999) return "";
  const p = (af * 100).toFixed(af < 0.1 ? 1 : 0);
  return `Latest analysed frame; the analyzer analyses ${p} % of frames` +
         (since ? " since start" : "") + (smp.mode === "cpu budget" ? " (CPU budget)" : "");
}

// -- seeded --------------------------------------------------------------------

function renderSeeded(frame) {
  const holder = document.getElementById("dqm-smaev-seeds");
  const head = document.getElementById("dqm-smaev-seededhead");
  if (!frame) {
    holder.innerHTML = "";
    state.seedPanels = [];
    head.innerHTML = "";
    guard("dqm-smaev-banner", () => renderBanner(null));
    noFrameNote("seeded", `${state.client} has not seen a good SMA frame yet. The seeded view ` +
      "shows only good frames (a stale buffer has no seeds worth showing).");
    return;
  }
  clearNote("seeded");
  guard(head, () => { frameHeader(frame, head); selectHeader(frame, head); });
  guard("dqm-smaev-banner", () => renderBanner(frame));
  try { updatePatternWarnings(); } catch (e) { /* the row keeps its last warnings */ }

  const seeds = (frame.meta && frame.meta.seeds) || [];
  while (state.seedPanels.length > seeds.length) state.seedPanels.pop().wrap.remove();
  while (state.seedPanels.length < seeds.length) state.seedPanels.push(makeSeedPanel(holder));
  const sel = (frame.meta || {}).select;
  if (!seeds.length && !(frame.meta && frame.meta.search && frame.meta.search.no_match)) {
    noFrameNote("seeded", sel
      ? `This frame has no ${sel.seed_label} hit whose whole window lies inside it.`
      : "This frame has no S1 hit whose whole window lies inside it.");
  }
  const ls = lanes(frame);
  updateRawToggle("seeded", frame);
  const nl = document.getElementById("dqm-smaev-nimlegend");
  if (nl) {
    nl.style.display = ls.some((l) => l.nim) ? "" : "none";
    const text = nimLegend(nimMerge(), alignText(frame));
    if (nl.textContent !== text) nl.textContent = text;
  }
  const nums = tagNumbers("seeded", frame);
  seeds.forEach(function (seed, k) {
    const p = state.seedPanels[k];
    // Per seed: one panel that throws shows its error and the others still draw.
    try {
      p.err.textContent = "";
      drawSeed(frame, seed, k, p, ls, nums);
    } catch (e) {
      if (typeof console !== "undefined") console.error("dqm-sma-events seed", k, e);
      p.err.className = "dqm-error";
      p.err.textContent = `seed ${k + 1}: could not draw: ${BRPC.errorText(e)}`;
    }
  });
  renderHints("seeded");
}

/**
 * One seed's panel. Made once and redrawn in place: the buttons in it must
 * survive a 4 Hz redraw, or a click that spans one lands on nothing.
 */
function makeSeedPanel(holder) {
  const wrap = el("div", { class: "dqm-sma-seed" });
  // `content` is redrawn per frame; the copy button beside it is not.
  const content = el("span", { class: "dqm-smaev-seedcontent" });
  const copy = el("button", {
    class: "mbutton dqm-smaev-mini dqm-smaev-copyseed",
    title: "Copy the frame tag, this seed's S1 word, the word range of its window and the " +
           "bank word index of every hit shown in it",
  }, "Copy seed words");
  const head = el("div", { class: "dqm-sma-seedhead" }, content, copy);
  const err = el("div", {});
  const canvas = el("canvas", { class: "dqm-sma-canvas dqm-smaev-pointable" });
  const info = makeHitInfo("seeded");
  wrap.appendChild(head);
  wrap.appendChild(err);
  wrap.appendChild(canvas);
  wrap.appendChild(info.wrap);
  holder.appendChild(wrap);
  const p = { wrap, head: content, copy, err, canvas, info, geom: null, copyText: "" };
  copy.onclick = function () { if (p.copyText) copyWithFeedback(copy, p.copyText); };
  const at = (e) => {
    const xy = mouseXY(canvas, e);
    return seedHitAt(p, xy[0], xy[1]);
  };
  canvas.addEventListener("mousemove", function (e) {
    info.over = true;
    const g = p.geom;
    const i = at(e);
    if (i >= 0 && g) setHover(info, hitText(g.frame, i, "seeded"), tagOf(g.frame));
    else setHover(info, null);
  });
  canvas.addEventListener("mouseleave", function () { info.over = false; setHover(info, null); });
  canvas.addEventListener("click", function (e) {
    const g = p.geom;
    const i = at(e);
    if (i < 0 || !g) return;
    toggleTag("seeded", g.frame, i);
  });
  return p;
}

function seedHits(frame, seed) {
  const h = seed.hits || [0, 0];
  return [Math.max(0, h[0]), Math.min(frame.nHits, h[1])];
}

/** The seed's pixel hits: [pa, pb) in the pixel block (meta.mupix.seeds[k].pix). */
function seedPixels(frame, k) {
  const p = pixOf(frame);
  const ms = (((frame.meta || {}).mupix || {}).seeds || [])[k];
  if (!p || !ms || !ms.pix) return [0, 0];
  return [Math.max(0, ms.pix[0]), Math.min(p.n, ms.pix[1])];
}

/**
 * The seed's MuPix badge: which planes had a hit in the in-time window
 * (counted by the analyzer, meta.mupix.seeds[k]), or why it cannot say (no
 * hit, and the window runs past the frame's pixel data).
 */
function mupixBadge(frame, k) {
  const mp = (frame.meta || {}).mupix;
  const ms = mp && (mp.seeds || [])[k];
  if (!ms) return null;
  const w = mp.window_ns || [];
  const title = `MuPix hits in the in-time window [${w[0]}, ${w[1]}) ns around the seed`;
  const l1 = ms.l1 > 0, l2 = ms.l2 > 0;
  if (!ms.covered && !l1 && !l2) {
    // No hit found, but the window runs past the frame's pixel data: says nothing.
    return el("span", { class: "dqm-sma-badge", title: "The seed's in-time window runs past the " +
                        "MuPix data of this frame" }, "MuPix n/a (outside the pixel data)");
  }
  const text = l1 && l2 ? "L1+L2 ✓" : l1 ? "L1 ✓" : l2 ? "L2 ✓" : "no MuPix in time";
  return el("span", { class: `dqm-sma-badge ${l1 || l2 ? "green" : ""}`.trim(),
                      title: `${title}: L1 ${ms.l1}, L2 ${ms.l2}` }, text);
}

function drawSeed(frame, seed, k, p, ls, nums) {
  const [a, b] = seedHits(frame, seed);
  let nMis = 0, nTot = 0, nOther = 0, nNimOnly = 0, nHeld = 0, nEcho = 0;
  const laneOf = {}, pixLane = {};
  ls.forEach((l, i) => { if (l.pix === undefined) laneOf[l.ch] = i; else pixLane[l.pix] = i; });
  for (let i = a; i < b; i++) {
    if (frame.hitFlags[i] & SMAF.HIT.MISMATCH) nMis++;
    if (frame.hitFlags[i] & SMAF.HIT.TOT_CORRUPT) nTot++;
    if (laneOf[frame.ch[i]] === undefined) nOther++;
    const st = frame.cls ? hitStyle(frame.cls[i]) : "solid";
    if (st === "hollow") nNimOnly++;
    else if (st === "held") nHeld++;
    else if (st === "echo") nEcho++;
  }

  // Header: DOM, not canvas, so it can be read, selected and copied.
  const head = p.head;
  head.innerHTML = "";
  p.geom = null;
  const who = seed.seed_ch === undefined ? `S1` : `${labelOf(seed.seed_ch)} (ch ${seed.seed_ch})`;
  const tot = seed.seed_ch === undefined ? seed.s1_tot : seed.seed_tot;
  head.appendChild(el("span", { class: "dqm-sma-seedtitle" },
    `seed ${k + 1}: ${who} at ${ms(frameOffsetNs(frame) + seed.t_rel)} ms in the frame, ToT ${tot}`));
  const hasNim = roles().nim.some((n) => n !== null);
  const merged = nimMerge() && hasNim;
  const pat = el("span", { class: "dqm-sma-pattern", title: merged
    ? "coincidence pattern of the merged counters (TOT words + NIM-only hits)"
    : hasNim ? "coincidence pattern of the TOT words only (NIM merge off)" : "coincidence pattern" });
  // Unknown roles: as many boxes as the pattern has bits.
  const nCounters = nPattern() || Math.min(PATTERN_MAX, 32 - Math.clz32(seed.pattern || 0));
  const inc = (frame.meta || {}).incomplete;
  const ignored = new Set(((inc && inc.ignored) || []).map((f) => f.counter));
  for (let c = 0; c < nCounters; c++) {
    const lit = (seed.pattern >> c) & 1;
    const attrs = { class: `dqm-sma-pbox${lit ? " lit" : ""}${ignored.has(c + 1) ? " ignored" : ""}` };
    if (ignored.has(c + 1)) attrs.title = `S${c + 1}: timestamp fault, left out of "incomplete pattern"`;
    pat.appendChild(el("span", attrs, `S${c + 1}`));
  }
  head.appendChild(pat);
  head.appendChild(badge(rfText(seed), seed.rf_na ? "" : seed.rf_vetoed ? "yellow" : (seed.rf_valid ? "blue" : "")));
  head.appendChild(badge(`${b - a} hits`));
  if (nMis) head.appendChild(badge(`${nMis} fine/coarse mismatch`, "red"));
  if (nTot) head.appendChild(badge(`${nTot} ToT ≥ ${frame.meta.tot_corrupt || 250}`, "yellow"));
  if (nOther) head.appendChild(badge(`${nOther} on channels without a role`));
  if (nNimOnly) head.appendChild(badge(`${nNimOnly} NIM-only (${merged ? "merged" : "not merged: merge off"})`));
  if (nHeld) head.appendChild(badge(`${nHeld} NIM-only held back (lag fault)`, "yellow"));
  if (nEcho) head.appendChild(badge(`${nEcho} echo word${nEcho === 1 ? "" : "s"}`));
  const mb = mupixBadge(frame, k);
  if (mb) head.appendChild(mb);
  const wr = seed.word_range;
  if (wr) head.appendChild(badge(`words ${wr[0]}–${wr[1]}`));
  const [pa, pb] = seedPixels(frame, k);
  const pw = pixelWords(frame, pa, pb);
  if (pw.length) head.appendChild(badge(`${pb - pa} MuPix hits · words ${pw[0]}–${pw[pw.length - 1]}`));

  p.copyText = seedText(frame, seed, k, a, b);
  p.copy.disabled = !frame.wordIndex;
  if (!frame.wordIndex) p.copy.title = "This analyzer sends no word data";

  const win = (frame.meta && frame.meta.window) || { pre_ns: 200, post_ns: 3000 };
  const range = state.zoom === "prompt" ? [-PROMPT_NS, PROMPT_NS] : [-win.pre_ns, win.post_ns];
  if (seed.odd && seed.odd.length) {
    head.appendChild(badge(`odd: ${seed.odd.map((n) => oddText(frame, n, true)).join(" · ")}`, "yellow"));
  }
  const g = paintSeed(p.canvas, frame, seed, a, b, ls, laneOf, range, nums || tagNumbers("seeded", frame),
                     { lane: pixLane, pa, pb });
  p.geom = Object.assign(g, { frame, seed, a, b, ls, range, pixLane, pa, pb });
}

/** The sorted bank word indices of pixel hits [pa, pb) (empty without word data). */
function pixelWords(frame, pa, pb) {
  const p = pixOf(frame);
  if (!p || !p.wordIndex || pb <= pa) return [];
  return Array.from(p.wordIndex.subarray(pa, pb)).sort((x, y) => x - y);
}

/**
 * What "Copy seed words" copies: the frame tag, the seed's S1 word, the bank
 * word range of its whole window and the word index of every hit shown in it
 * (ascending, the order they sit in the file), on one line for the elog.
 */
function seedText(frame, seed, k, a, b) {
  const wr = seed.word_range;
  const s1 = seed.s1_word === null || seed.s1_word === undefined ? "—" : seed.s1_word;
  // A seed on another channel names its own word, then the S1 word it took the RF from.
  const who = seed.seed_word === undefined || seed.seed_word === seed.s1_word
    ? `S1 word ${s1}`
    : `${labelOf(seed.seed_ch)} word ${seed.seed_word} · S1 word ${s1}`;
  const parts = [tagOf(frame), `seed ${k + 1}: ${who}`,
                 `words ${wr ? `${wr[0]}–${wr[1]}` : "—"} (${b - a} hits in window)`];
  if (frame.wordIndex) {
    const w = Array.from(frame.wordIndex.subarray(a, b)).sort((x, y) => x - y);
    parts.push(`hit words ${w.join(", ")}`);
  }
  const [pa, pb] = seedPixels(frame, k);
  const pw = pixelWords(frame, pa, pb);
  if (pw.length) parts.push(`MuPix pixel words ${pw.join(", ")}`);
  return parts.join(" · ");
}

function rfText(seed) {
  if (seed.rf_na) return "RF n/a (no S1)";
  const parts = ["RF"];
  if (seed.rf_valid) {
    parts.push(`phase ${fmt(seed.rf_phase, 1)} ns`);
    parts.push(`period ${fmt(seed.rf_period, 1)} ns`);
  }
  parts.push(`${seed.rf_n} pulse${seed.rf_n === 1 ? "" : "s"}`);
  parts.push(seed.rf_valid ? "valid" : "not valid");
  if (seed.rf_vetoed) parts.push("vetoed");
  if (seed.s1_dt) parts.push(`from the S1 hit at ${seed.s1_dt > 0 ? "+" : ""}${seed.s1_dt} ns`);
  return parts.join(" · ");
}

const LANE_H = 20;
const MARGIN = { left: 84, right: 12, top: 6, bottom: 34 };

function paintSeed(canvas, frame, seed, a, b, ls, laneOf, range, nums, pix) {
  const H = MARGIN.top + ls.length * LANE_H + MARGIN.bottom;
  const { ctx, W } = setupCanvas(canvas, H);
  const x0 = MARGIN.left, x1 = W - MARGIN.right;
  const X = (t) => x0 + (t - range[0]) / (range[1] - range[0]) * (x1 - x0);

  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, W, H);
  ls.forEach(function (l, i) {
    const y = MARGIN.top + i * LANE_H;
    if (i % 2) { ctx.fillStyle = "#f4f4f4"; ctx.fillRect(x0, y, x1 - x0, LANE_H); }
    ctx.fillStyle = "#333";
    ctx.font = "12px sans-serif";
    ctx.textBaseline = "middle";
    ctx.textAlign = "right";
    ctx.fillText(l.pix === undefined ? `${l.label} (${l.ch})` : l.label, x0 - 6, y + LANE_H / 2);
  });
  const s1Seed = seed.seed_ch === undefined || seed.seed_ch === roles().counters[0];
  timeAxis(ctx, range, X, MARGIN.top + ls.length * LANE_H, s1Seed ? "ns from S1" : "ns from the seed");

  // S1 itself, through every lane: everything else is read against it.
  ctx.strokeStyle = "#d62728";
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(X(0), MARGIN.top);
  ctx.lineTo(X(0), MARGIN.top + ls.length * LANE_H);
  ctx.stroke();
  ctx.setLineDash([]);

  // The last RF pulse the phase was measured from, on the RF lane.
  const rfLane = laneOf[roles().rf];
  if (seed.rf_valid && rfLane !== undefined && Number.isFinite(seed.rf_phase)) {
    // The phase is from the S1 hit, which for another seed sits s1_dt away.
    const x = X((seed.s1_dt || 0) + seed.rf_phase);
    const y = MARGIN.top + rfLane * LANE_H;
    ctx.strokeStyle = "#000";
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y + LANE_H); ctx.stroke();
  }

  ctx.save();
  ctx.beginPath();
  ctx.rect(x0, MARGIN.top, x1 - x0, ls.length * LANE_H);
  ctx.clip();
  const marks = [];
  const cls = frame.cls;
  const sh = drawShift(frame);
  for (let i = a; i < b; i++) {
    const lane = laneOf[frame.ch[i]];
    if (lane === undefined) continue;
    const t = drawnT(frame, i, sh) - seed.t_rel;
    const tot = frame.tot[i];
    if (t + tot < range[0] || t > range[1]) continue;
    const flags = frame.hitFlags[i];
    const px0 = X(t);
    const pw = Math.max(2, X(t + tot) - px0);
    const y = MARGIN.top + lane * LANE_H + 3;
    const h = LANE_H - 6;
    const style = hitStyle(cls ? cls[i] : undefined);
    const hollow = style === "hollow" || style === "held";
    // The width every mark of this hit uses: a NIM bar at least NIM_MIN_PX, a
    // hollow one HOLLOW_MIN_PX (a 10 ns NIM word is 2-3 px on the full window).
    const ww = hollow ? Math.max(HOLLOW_MIN_PX, pw) : ls[lane].nim ? Math.max(NIM_MIN_PX, pw) : pw;
    if (hollow) {
      // A NIM-only word: an outline, so it reads as "only the copy saw it";
      // grey when a lag fault kept it out of the merge. Filled only when its
      // own width reaches the minimum, so a widened one cannot paint over a
      // neighbour on the lane.
      if (pw >= HOLLOW_MIN_PX || style === "held") {
        ctx.fillStyle = style === "held" ? HELD_FILL : NIM_ONLY_FILL;
        ctx.fillRect(px0, y, ww, h);
      }
      ctx.strokeStyle = style === "held" ? HELD_EDGE : (ls[lane].edge || ls[lane].colour);
      ctx.lineWidth = 1.5;
      ctx.strokeRect(px0 + 0.75, y + 0.75, ww - 1.5, h - 1.5);
      ctx.lineWidth = 1;
    } else {
      ctx.fillStyle = ls[lane].colour;
      ctx.fillRect(px0, y, ww, h);
      if (ls[lane].nim) {
        ctx.strokeStyle = ls[lane].edge;
        ctx.strokeRect(px0 + 0.5, y + 0.5, ww - 1, h - 1);
      }
      if (style === "echo") hatch(ctx, px0, y, ww, h, ECHO_HATCH, true);
    }
    if (flags & SMAF.HIT.MISMATCH) {
      // Hatched with a red outline: the time of this hit cannot be trusted,
      // so it must not look like the solid bars around it. (Red hatching on a
      // hollow bar: white would vanish on its white fill.)
      hatch(ctx, px0, y, ww, h, hollow ? MISMATCH_ON_HOLLOW : undefined);
      ctx.strokeStyle = "#d00";
      ctx.lineWidth = 2;
      ctx.strokeRect(px0, y, ww, h);
      ctx.lineWidth = 1;
    }
    if (flags & SMAF.HIT.TOT_CORRUPT) {
      // A ToT of 250+ is a marker value, not a width: flag it where it starts.
      ctx.fillStyle = "#c0c";
      ctx.beginPath();
      ctx.moveTo(px0, y - 3); ctx.lineTo(px0 + 6, y - 3); ctx.lineTo(px0 + 3, y + 3);
      ctx.closePath(); ctx.fill();
    }
    const n = nums.any ? nums.of(i) : undefined;
    if (n !== undefined) marks.push([px0, y, ww, h, n]);
  }
  paintPairTicks(ctx, frame, seed, a, b, ls, laneOf, X, range, sh);
  paintSeedPixels(ctx, frame, seed, pix, X, range, marks, nums);
  for (const [mx, my, mw, mh] of marks) {
    // A tagged hit, boxed in black: its number is its row in "Tagged hits".
    ctx.strokeStyle = TAG_MARK;
    ctx.lineWidth = 2;
    ctx.strokeRect(mx - 2, my - 2, mw + 4, mh + 4);
    ctx.lineWidth = 1;
  }
  ctx.restore();
  // The numbers outside the clip, so one beside a bar at the edge still shows.
  for (const [mx, my, mw, mh, n] of marks) {
    tagMarker(ctx, mx + mw + 3, mx - 3, my + (mh - 13) / 2, n, x1);
  }
  return { x0, x1, laneOf, sh };
}

/**
 * The pair ticks: a thin dark line from each TOT word to the NIM word it is
 * paired with, start to start, across the counter's lane and its NIM lane
 * (adjacent, see lanes()). Only pairs with both words shipped and on screen.
 * With the NIM lane shifted by its offset (`sh`) a tick is all but upright.
 */
function paintPairTicks(ctx, frame, seed, a, b, ls, laneOf, X, range, sh) {
  const pair = frame.pair, cls = frame.cls;
  if (!pair || !cls) return;
  ctx.strokeStyle = PAIR_TICK;
  ctx.lineWidth = 1;
  ctx.beginPath();
  let any = false;
  for (let i = a; i < b; i++) {
    const j = pair[i];
    if (j < a || j >= b || (cls[i] & SMAF.PAIR.NIM_SIDE)) continue;   // drawn once, from the TOT word
    const lt = laneOf[frame.ch[i]], ln = laneOf[frame.ch[j]];
    if (lt === undefined || ln === undefined) continue;
    const ti = drawnT(frame, i, sh) - seed.t_rel, tj = drawnT(frame, j, sh) - seed.t_rel;
    if (ti < range[0] || ti > range[1] || tj < range[0] || tj > range[1]) continue;
    ctx.moveTo(X(ti) + 0.5, MARGIN.top + lt * LANE_H + LANE_H / 2);
    ctx.lineTo(X(tj) + 0.5, MARGIN.top + ln * LANE_H + LANE_H / 2);
    any = true;
  }
  if (any) ctx.stroke();
}

const PIX_TICK_W = 2.5;          // a pixel hit's tick on a MuPix lane or row, px

/**
 * The MuPix lanes of a seed panel: the in-time window shaded on every plane's
 * lane, then one tick per pixel hit of the seed's window, coloured by its ToT.
 * Tagged ticks join `marks` (boxed and numbered by the caller).
 */
function paintSeedPixels(ctx, frame, seed, pix, X, range, marks, nums) {
  const mp = (frame.meta || {}).mupix;
  const p = pixOf(frame);
  if (!mp || !p || !pix) return;
  const w = mp.window_ns || [-150, 450];
  const xa = X(Math.max(range[0], w[0])), xb = X(Math.min(range[1], w[1]));
  for (const pl of Object.keys(pix.lane)) {
    const y = MARGIN.top + pix.lane[pl] * LANE_H;
    if (xb > xa) { ctx.fillStyle = IN_TIME_FILL; ctx.fillRect(xa, y + 1, xb - xa, LANE_H - 2); }
  }
  for (let j = pix.pa; j < pix.pb; j++) {
    const lane = pix.lane[p.flags[j] & SMAF.PIX.PLANE_MASK];
    if (lane === undefined) continue;
    const t = pixRel(frame, j) - seed.t_rel;
    if (t < range[0] || t > range[1]) continue;
    const x = X(t) - PIX_TICK_W / 2;
    const y = MARGIN.top + lane * LANE_H + 3;
    ctx.fillStyle = pixColour(p.tot[j]);
    ctx.fillRect(x, y, PIX_TICK_W, LANE_H - 6);
    const n = nums.any ? nums.of(frame.nHits + j) : undefined;
    if (n !== undefined) marks.push([x, y, PIX_TICK_W, LANE_H - 6, n]);
  }
}

/** The hit under (x, y) on a seed panel, or -1: its bar, or within HIT_SLOP_PX of it. */
function seedHitAt(p, x, y) {
  const g = p.geom;
  if (!g || x < g.x0 || x > g.x1) return -1;
  const lane = Math.floor((y - MARGIN.top) / LANE_H);
  if (lane < 0 || lane >= g.ls.length) return -1;
  const ch = g.ls[lane].ch;
  const f = g.frame;
  const X = (t) => g.x0 + (t - g.range[0]) / (g.range[1] - g.range[0]) * (g.x1 - g.x0);
  if (g.ls[lane].pix !== undefined) {
    // A MuPix lane: the nearest tick of that plane.
    const px = pixOf(f);
    const plane = g.ls[lane].pix;
    let bestJ = -1, bd = HIT_SLOP_PX;
    for (let j = g.pa; px && j < g.pb; j++) {
      if ((px.flags[j] & SMAF.PIX.PLANE_MASK) !== plane) continue;
      const d = Math.abs(X(pixRel(f, j) - g.seed.t_rel) - x);
      if (d <= bd) { bestJ = j; bd = d; }
    }
    return bestJ < 0 ? -1 : f.nHits + bestJ;
  }
  let best = -1, bestD = HIT_SLOP_PX;
  for (let i = g.a; i < g.b; i++) {
    if (f.ch[i] !== ch) continue;
    const t = drawnT(f, i, g.sh) - g.seed.t_rel;     // where paintSeed drew it
    const px0 = X(t);
    const pw = Math.max(2, X(t + f.tot[i]) - px0);
    const d = x < px0 ? px0 - x : x > px0 + pw ? x - px0 - pw : 0;
    if (d <= bestD) { best = i; bestD = d; }        // a tie goes to the later bar, drawn on top
  }
  return best;
}

/**
 * Diagonal hatching over a bar: white "/" lines by default (the fine/coarse
 * mismatch); `colour` and `back` (lines slanted the other way) for the echo words, so the two
 * never look alike.
 */
function hatch(ctx, x, y, w, h, colour, back) {
  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, w, h);
  ctx.clip();
  ctx.strokeStyle = colour || "rgba(255,255,255,0.9)";
  ctx.beginPath();
  for (let d = -h; d < w; d += back ? 4 : 5) {
    if (back) { ctx.moveTo(x + d, y); ctx.lineTo(x + d + h, y + h); }
    else { ctx.moveTo(x + d, y + h); ctx.lineTo(x + d + h, y); }
  }
  ctx.stroke();
  ctx.restore();
}

// -- raster --------------------------------------------------------------------

const ROW_H = 22;
const RASTER_MARGIN = { left: 84, right: 110, top: 6, bottom: 34 };

function renderRaster(frame) {
  // Roles may have come from this frame's labels only (no summary yet).
  try { updateCurrentToggle(); } catch (e) { /* the switch keeps its state */ }
  const head = document.getElementById("dqm-smaev-rasterhead");
  const holder = document.getElementById("dqm-smaev-rasterbox");
  if (!frame) {
    head.innerHTML = "";
    noFrameNote("raster", `${state.client} has not seen an SMA frame yet.`);
    return;
  }
  clearNote("raster");
  updateRawToggle("raster", frame);
  const rn = document.getElementById("dqm-smaev-rasternim");
  if (rn) {
    rn.style.display = frame.cls ? "" : "none";
    const at = alignText(frame);
    const text = `S*k*L rows: the NIM copies${at ? `, ${at}` : ""} · dark mark above a hit: a NIM-only word` +
      (nimMerge() ? " (merged; grey: held back, lag fault)" : " (not merged: NIM merge off)");
    if (rn.textContent !== text) rn.textContent = text;
  }
  guard(head, () => frameHeader(frame, head));
  guard(holder, () => paintRaster(frame));
  renderHints("raster");
}

/** An SMA ToT colour: viridis over codes 0..249; >= 250 (corrupt) magenta. */
function totColour(tot) {
  if (tot >= 250) return "#c0c";
  return DQM.viridis(Math.min(1, tot / 249) * HIT_VIRIDIS_TOP);
}

/**
 * ns from the frame's first kept hit to the first shipped one.
 *
 * The hit and seed times are relative to meta.t0_ns, the first hit *shipped*;
 * with the current channel dropped, a truncated raster (which keeps the latest
 * hits) or a seeded view (which ships only the seed windows) that is not where
 * the frame starts. Times shown to people are from the frame's start, so the
 * same hit has the same time whatever was left out.
 */
function frameOffsetNs(frame) {
  const m = frame.meta || {};
  return Number.isFinite(m.t0_ns) && Number.isFinite(m.frame_first_ns)
    ? m.t0_ns - m.frame_first_ns : 0;
}

function rasterRange(frame) {
  if (state.view) return state.view;
  let hi = frame.meta && frame.meta.span_ns ? frame.meta.span_ns / 1e6 : 0;
  if (frame.nHits) hi = Math.max(hi, (frameOffsetNs(frame) + frame.t[frame.nHits - 1]) / 1e6);
  // The MuPix part of a frame does not cover quite the same time as the SMA
  // part: the axis runs on to the last pixel hit too. (Pixel hits before the
  // first SMA hit, rare, are counted in the row total but left off the axis,
  // which starts at the frame's first kept SMA hit.)
  const px = pixOf(frame);
  if (px && px.n && frame.meta && frame.meta.mupix) {
    hi = Math.max(hi, (frameOffsetNs(frame) + pixRel(frame, px.n - 1)) / 1e6);
  }
  return [0, hi > 0 ? hi : 1];
}

function paintRaster(frame) {
  let canvas = document.getElementById("dqm-smaev-raster");
  if (!canvas || !canvas.getContext) {
    const holder = document.getElementById("dqm-smaev-rasterbox");
    holder.innerHTML = "";
    canvas = el("canvas", { class: "dqm-sma-canvas dqm-smaev-pointable", id: "dqm-smaev-raster" });
    holder.appendChild(canvas);
    wireDrag(canvas);
  }
  const nCh = 16;
  const planes = pixelPlanes(frame);          // MuPix rows under the 16 channels
  const nRows = nCh + planes.length;
  const M = RASTER_MARGIN;
  const H = M.top + nRows * ROW_H + M.bottom;
  const { ctx, W } = setupCanvas(canvas, H);
  const x0 = M.left, x1 = W - M.right;
  const range = rasterRange(frame);
  const X = (t) => x0 + (t - range[0]) / (range[1] - range[0]) * (x1 - x0);
  const r = roles();
  const meta = frame.meta || {};
  const dropped = new Set(meta.dropped || []);

  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, W, H);
  ctx.font = "12px sans-serif";
  ctx.textBaseline = "middle";
  for (let c = 0; c < nCh; c++) {
    const y = M.top + c * ROW_H;
    if (c % 2) { ctx.fillStyle = "#f4f4f4"; ctx.fillRect(x0, y, x1 - x0, ROW_H); }
    ctx.fillStyle = "#333";
    ctx.textAlign = "right";
    ctx.fillText(`${labelOf(c, r)} (${c})`, x0 - 6, y + ROW_H / 2);
    // Every kept hit of the channel, dropped ones included: hiding ch 7's
    // points must not hide the fact that it has 30k of them.
    const n = meta.per_channel ? meta.per_channel[c] : null;
    ctx.textAlign = "left";
    ctx.fillStyle = dropped.has(c) ? "#999" : "#333";
    ctx.fillText(n === null || n === undefined ? "" : `${num(n)}${dropped.has(c) ? " (hidden)" : ""}`,
                 x1 + 6, y + ROW_H / 2);
  }
  const mp = meta.mupix || {};
  planes.forEach(function (pl, k) {
    const y = M.top + (nCh + k) * ROW_H;
    ctx.fillStyle = "#eef6ee";
    ctx.fillRect(x0, y, x1 - x0, ROW_H);
    ctx.fillStyle = "#333";
    ctx.textAlign = "right";
    ctx.fillText(PLANE_LABEL[pl], x0 - 6, y + ROW_H / 2);
    // Every examined pixel hit of the plane, shipped or not (max_hits).
    const n = mp.per_plane ? mp.per_plane[pl] : null;
    ctx.textAlign = "left";
    ctx.fillText(n === null || n === undefined ? "" : num(n), x1 + 6, y + ROW_H / 2);
  });
  timeAxis(ctx, range, X, M.top + nRows * ROW_H, "ms from the frame's first kept hit");

  // One fill per colour bucket, not per hit: 33k fillStyle changes is what
  // makes a canvas raster slow, 33k rects in 33 paths is not.
  const BUCKETS = 32;
  const paths = new Array(BUCKETS + 1);
  const scale = 1e6;
  const off = frameOffsetNs(frame);
  const sh = drawShift(frame);
  for (let i = 0; i < frame.nHits; i++) {
    const t = (off + drawnT(frame, i, sh)) / scale;
    if (t < range[0] || t > range[1]) continue;
    const c = frame.ch[i];
    if (c >= nCh) continue;
    const tot = frame.tot[i];
    const bkt = tot >= 250 ? BUCKETS : Math.min(BUCKETS - 1, Math.floor(tot / 250 * BUCKETS));
    (paths[bkt] = paths[bkt] || []).push(X(t), M.top + c * ROW_H + 4);
  }
  for (let bkt = 0; bkt <= BUCKETS; bkt++) {
    const pts = paths[bkt];
    if (!pts) continue;
    ctx.fillStyle = bkt === BUCKETS ? "#c0c" : totColour((bkt + 0.5) * 250 / BUCKETS);
    ctx.beginPath();
    for (let j = 0; j < pts.length; j += 2) ctx.rect(pts[j], pts[j + 1], 1.5, ROW_H - 8);
    ctx.fill();
  }
  // NIM-only words (smaf v3): a small dark mark above the bar, whose colour
  // stays its ToT -- what the merge adds to each counter, and (grey) what a
  // lag fault held back. Above rather than around the bar: on a whole frame
  // the marks merge into a band whose density is the NIM-only share, without
  // hiding the ToT colours below.
  if (frame.cls) {
    const outl = { hollow: [], held: [] };
    for (let i = 0; i < frame.nHits; i++) {
      const st = hitStyle(frame.cls[i]);
      if (st !== "hollow" && st !== "held") continue;
      const t = (off + drawnT(frame, i, sh)) / scale;
      if (t < range[0] || t > range[1] || frame.ch[i] >= nCh) continue;
      outl[st].push(X(t), M.top + frame.ch[i] * ROW_H + 1);
    }
    for (const [st, colour] of [["hollow", PAIR_TICK], ["held", HELD_EDGE]]) {
      const pts = outl[st];
      if (!pts.length) continue;
      ctx.fillStyle = colour;
      ctx.beginPath();
      for (let j = 0; j < pts.length; j += 2) ctx.rect(pts[j] - 0.5, pts[j + 1], 2.5, 2.5);
      ctx.fill();
    }
  }
  // The MuPix rows: one path per pixel ToT code (32 colours).
  const px = pixOf(frame);
  const rowOf = {};
  planes.forEach((pl, k) => { rowOf[pl] = nCh + k; });
  if (px && planes.length) {
    const pp = new Array(32);
    for (let j = 0; j < px.n; j++) {
      const t = (off + pixRel(frame, j)) / scale;
      if (t < range[0] || t > range[1]) continue;
      const row = rowOf[px.flags[j] & SMAF.PIX.PLANE_MASK];
      if (row === undefined) continue;
      (pp[px.tot[j] & 31] = pp[px.tot[j] & 31] || []).push(X(t), M.top + row * ROW_H + 4);
    }
    for (let tot = 0; tot < 32; tot++) {
      const pts = pp[tot];
      if (!pts) continue;
      ctx.fillStyle = pixColour(tot);
      ctx.beginPath();
      for (let j = 0; j < pts.length; j += 2) ctx.rect(pts[j], pts[j + 1], 1.5, ROW_H - 8);
      ctx.fill();
    }
  }

  const nums = tagNumbers("raster", frame);
  for (let i = 0; nums.any && i < nAll(frame); i++) {
    const n = nums.of(i);
    if (n === undefined) continue;
    const pixel = isPix(frame, i);
    const j = i - frame.nHits;
    const t = (off + (pixel ? pixRel(frame, j) : drawnT(frame, i, sh))) / scale;
    const c = pixel ? rowOf[px.flags[j] & SMAF.PIX.PLANE_MASK] : frame.ch[i];
    if (t < range[0] || t > range[1] || c === undefined || c >= nRows) continue;
    // A tagged hit, boxed in black: its number is its row in "Tagged hits".
    const x = X(t);
    ctx.strokeStyle = TAG_MARK;
    ctx.lineWidth = 2;
    ctx.strokeRect(x - 3, M.top + c * ROW_H + 1, 7.5, ROW_H - 2);
    ctx.lineWidth = 1;
    tagMarker(ctx, x + 6, x - 4, M.top + c * ROW_H + (ROW_H - 13) / 2, n, x1);
  }

  if (state.drag && state.drag.x1 !== undefined) {
    ctx.fillStyle = "rgba(0, 102, 204, 0.15)";
    const a = Math.min(state.drag.x0, state.drag.x1), b = Math.max(state.drag.x0, state.drag.x1);
    ctx.fillRect(a, M.top, b - a, nRows * ROW_H);
  }
  state.rasterGeom = { x0, x1, range, planes, sh };
}

/**
 * The raster hit under (x, y), or -1: the nearest on that row within
 * HIT_SLOP_PX. A plain scan: 33k hits is well under a millisecond, and a hit
 * is looked up per mouse move, not per frame.
 */
function rasterHitAt(frame, x, y) {
  const g = state.rasterGeom;
  if (!g || !frame || x < g.x0 - HIT_SLOP_PX || x > g.x1 + HIT_SLOP_PX) return -1;
  const c = Math.floor((y - RASTER_MARGIN.top) / ROW_H);
  const planes = g.planes || [];
  if (c < 0 || c >= 16 + planes.length) return -1;
  const off = frameOffsetNs(frame);
  const k = (g.x1 - g.x0) / (g.range[1] - g.range[0]);
  let best = -1, bestD = HIT_SLOP_PX;
  if (c >= 16) {
    // A MuPix row: the nearest pixel hit of that plane.
    const px = pixOf(frame);
    const plane = planes[c - 16];
    for (let j = 0; px && j < px.n; j++) {
      if ((px.flags[j] & SMAF.PIX.PLANE_MASK) !== plane) continue;
      const t = (off + pixRel(frame, j)) / 1e6;
      if (t < g.range[0] || t > g.range[1]) continue;
      const d = Math.abs(g.x0 + (t - g.range[0]) * k + 0.75 - x);
      if (d <= bestD) { best = frame.nHits + j; bestD = d; }
    }
    return best;
  }
  for (let i = 0; i < frame.nHits; i++) {
    if (frame.ch[i] !== c) continue;
    const t = (off + drawnT(frame, i, g.sh)) / 1e6;     // where paintRaster drew it
    if (t < g.range[0] || t > g.range[1]) continue;
    const d = Math.abs(g.x0 + (t - g.range[0]) * k + 0.75 - x);   // the mark is 1.5 px wide
    if (d <= bestD) { best = i; bestD = d; }
  }
  return best;
}

/**
 * Drag across the raster to zoom in time; double-click to see the whole frame.
 *
 * The release is watched on the window, not the canvas: a drag that ends
 * outside the canvas would otherwise never finish, and the next mouse move
 * over the raster would keep drawing a selection nobody is making.
 */
function wireDrag(canvas) {
  const xOf = (e) => mouseXY(canvas, e)[0];
  canvas.addEventListener("mousedown", function (e) {
    state.drag = { x0: xOf(e) };
  });
  canvas.addEventListener("mousemove", function (e) {
    const info = state.rasterInfo;
    if (!state.drag) {
      // Not dragging: point at the hit under the mouse.
      const f = state.frames.raster;
      const xy = mouseXY(canvas, e);
      const i = rasterHitAt(f, xy[0], xy[1]);
      if (info) {
        info.over = true;
        if (i >= 0) setHover(info, hitText(f, i, "raster"), tagOf(f));
        else setHover(info, null);
      }
      return;
    }
    state.drag.x1 = xOf(e);
    if (state.frames.raster) guard("dqm-smaev-rasterbox", () => paintRaster(state.frames.raster));
  });
  canvas.addEventListener("mouseleave", function () {
    if (state.rasterInfo) { state.rasterInfo.over = false; setHover(state.rasterInfo, null); }
  });
  if (!state.dragWired) {
    state.dragWired = true;
    window.addEventListener("mouseup", function (e) {
      const xy = mouseXY(canvas, e);
      endDrag(canvas, xy[0], xy[1]);
    });
  }
  canvas.addEventListener("dblclick", function () {
    state.view = null;
    if (state.frames.raster) render("raster");
  });
}

function endDrag(canvas, x1, y1) {
  const d = state.drag;
  state.drag = null;
  const g = state.rasterGeom;
  if (!d) return;
  if (!g || Math.abs(x1 - d.x0) < 4) {        // a click, not a drag: tag the hit under it
    const f = state.frames.raster;
    const i = f && Number.isFinite(y1) ? rasterHitAt(f, x1, y1) : -1;
    if (i >= 0) rasterClick(f, i);
    else if (f) render("raster");
    return;
  }
  const toT = (x) => g.range[0] + (Math.min(Math.max(x, g.x0), g.x1) - g.x0) /
                     (g.x1 - g.x0) * (g.range[1] - g.range[0]);
  state.view = [toT(Math.min(d.x0, x1)), toT(Math.max(d.x0, x1))];
  if (state.frames.raster) render("raster");
}

// -- canvas helpers -------------------------------------------------------------

/** The mouse position in the canvas's CSS pixels. */
function mouseXY(canvas, e) {
  const r = canvas.getBoundingClientRect ? canvas.getBoundingClientRect() : null;
  if (r && typeof e.clientX === "number") {
    return [e.clientX - r.left, typeof e.clientY === "number" ? e.clientY - r.top : NaN];
  }
  return [typeof e.offsetX === "number" ? e.offsetX : 0,
          typeof e.offsetY === "number" ? e.offsetY : NaN];
}

/**
 * Size a canvas for its CSS width at the screen's pixel density.
 *
 * Without the devicePixelRatio factor every line is drawn at CSS resolution
 * and scaled up, which on a laptop panel blurs the 1 ns bars into mush.
 */
function setupCanvas(canvas, cssHeight) {
  const dpr = window.devicePixelRatio || 1;
  const W = Math.max(300, canvas.clientWidth || (canvas.parentNode && canvas.parentNode.clientWidth) || 900);
  canvas.style.height = `${cssHeight}px`;
  if (canvas.width !== Math.round(W * dpr)) canvas.width = Math.round(W * dpr);
  if (canvas.height !== Math.round(cssHeight * dpr)) canvas.height = Math.round(cssHeight * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, W };
}

function timeAxis(ctx, range, X, y, label) {
  ctx.strokeStyle = "#888";
  ctx.fillStyle = "#333";
  ctx.font = "11px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.beginPath();
  ctx.moveTo(X(range[0]), y); ctx.lineTo(X(range[1]), y);
  ctx.stroke();
  const xa = X(range[0]), xb = X(range[1]);
  for (const t of niceTicks(range[0], range[1], 8)) {
    const x = X(t);
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y + 4); ctx.stroke();
    // A label centred on the last tick would hang off the canvas edge.
    ctx.textAlign = x > xb - 15 ? "right" : x < xa + 15 ? "left" : "center";
    ctx.fillText(String(Number(t.toPrecision(6))), x, y + 5);
  }
  // Under the tick labels, not beside them, so neither covers the other.
  ctx.textAlign = "right";
  ctx.fillText(label, xb, y + 19);
}

/**
 * A tagged hit's number in a black box, at x (right of the hit), or ending at
 * xLeft when that would cross xMax (the plot's right edge).
 */
function tagMarker(ctx, x, xLeft, y, n, xMax) {
  const s = String(n);
  const w = 6 + 7 * s.length, h = 13;
  const bx = x + w > xMax ? xLeft - w : x;
  ctx.fillStyle = TAG_MARK;
  ctx.fillRect(bx, y, w, h);
  ctx.fillStyle = "#fff";
  ctx.font = "bold 10px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(s, bx + w / 2, y + h / 2 + 0.5);
}

function niceTicks(lo, hi, n) {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) || 10 * mag;
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) out.push(t);
  return out;
}

// ---------------------------------------------------------------------------
// Finding a frame again: the tag, the raw event, the hit under the mouse
// ---------------------------------------------------------------------------

/**
 * The analyzer's tag for a frame, verbatim (run, event, serial, time, frame
 * seq: what finds it again in the files). An analyzer from before the tag
 * gets a stand-in from the header.
 */
function tagOf(frame) {
  const m = frame.meta || {};
  if (typeof m.tag === "string" && m.tag) return m.tag;
  return `SMA run ${frame.run || "?"} · serial ${serialOf(frame)} · frame seq ${frame.frameSeq}`;
}

function serialOf(frame) {
  const m = frame.meta || {};
  const s = m.event && m.event.serial !== undefined ? m.event.serial : m.serial;
  return s === undefined || s === null ? "?" : s;
}

function rawFileName(frame) {
  return `sma_run${frame.run}_serial${serialOf(frame)}.mid`;
}

/** The tag bar above a tab's frame header: the tag, Copy tag, Download raw event. */
function makeTagBar(tab) {
  const bar = el("div", { class: "dqm-strip dqm-smaev-tagbar", id: `dqm-smaev-tagbar-${tab}` });
  const tag = el("code", { class: "dqm-smaev-tag", id: `dqm-smaev-tag-${tab}` });
  const copy = el("button", {
    class: "mbutton dqm-smaev-mini", id: `dqm-smaev-copytag-${tab}`,
    title: "Copy this frame's tag for the elog: run + serial find the event in the run file",
  }, "Copy tag");
  copy.onclick = function () {
    const f = state.frames[tab];
    if (f) copyWithFeedback(copy, tagOf(f));
  };
  const dl = el("button", { class: "mbutton dqm-smaev-mini", id: `dqm-smaev-download-${tab}` },
                "Download raw event");
  dl.onclick = function () { downloadRaw(tab); };
  const note = el("span", { class: "dqm-smaev-tagnote", id: `dqm-smaev-tagnote-${tab}` });
  bar.appendChild(tag);
  bar.appendChild(copy);
  bar.appendChild(dl);
  bar.appendChild(note);
  bar.style.display = "none";
  return bar;
}

function renderTagBar(tab, frame) {
  const bar = document.getElementById(`dqm-smaev-tagbar-${tab}`);
  if (!bar) return;
  if (!frame) { bar.style.display = "none"; return; }
  bar.style.display = "";
  const m = frame.meta || {};
  document.getElementById(`dqm-smaev-tag-${tab}`).textContent = tagOf(frame);
  const dl = document.getElementById(`dqm-smaev-download-${tab}`);
  const kept = state.rawKept.has(frame.frameSeq);
  const held = m.raw_held === true || kept;
  dl.disabled = !held || state.downloadBusy;
  dl.title = held
    ? `Save this frame's MIDAS event as ${rawFileName(frame)}: a one-event .mid file that ` +
      "mdump and midas.file_reader read" + (kept ? " (kept by this page since Freeze)" : "")
    : (m.raw_held === false
      ? "The analyzer no longer holds this frame's raw event: use the tag to find it in the run file"
      : "This analyzer does not hand out raw events");
  const note = document.getElementById(`dqm-smaev-tagnote-${tab}`);
  const n = state.tagNote[tab];
  if (n && n.seq === frame.frameSeq) {
    note.textContent = n.text;
    note.className = `dqm-smaev-tagnote ${n.cls || ""}`.trim();
  } else if (m.raw_held === false && !kept) {
    note.textContent = "raw event no longer held: use the tag";
    note.className = "dqm-smaev-tagnote";
  } else {
    note.textContent = "";
    note.className = "dqm-smaev-tagnote";
  }
}

function setTagNote(tab, seq, text, cls) {
  state.tagNote[tab] = { seq, text, cls };
  if (state.frames[tab]) renderTagBar(tab, state.frames[tab]);
}

/**
 * Download raw event: sma::raw for the frame on screen, saved as
 * sma_run<run>_serial<serial>.mid. The analyzer keeps only the last few
 * frames' bytes; a frame it has dropped answers with its error, shown beside
 * the button (the tag still finds the event in the run file).
 */
async function downloadRaw(tab) {
  const frame = state.frames[tab];
  if (!frame || state.downloadBusy) return;
  const seq = frame.frameSeq;
  const name = rawFileName(frame);
  state.downloadBusy = true;
  setTagNote(tab, seq, "fetching the raw event…", "");
  try {
    const payload = state.rawKept.get(seq) || await SMAF.fetchRaw(state.client, seq);
    saveBlob(payload, name);
    setTagNote(tab, seq, `saved ${name} (${num(payload.byteLength)} bytes)`, "ok");
  } catch (e) {
    if (typeof console !== "undefined") console.warn("dqm-sma-events raw", e);
    setTagNote(tab, seq, BRPC.errorText(e), "red");
  } finally {
    state.downloadBusy = false;
    if (state.frames[tab]) renderTagBar(tab, state.frames[tab]);
  }
}

function saveBlob(payload, name) {
  const blob = new window.Blob([payload], { type: "application/octet-stream" });
  const url = window.URL.createObjectURL(blob);
  const a = el("a", { href: url, download: name });
  a.style.display = "none";
  const host = document.body || document.getElementById("dqm-root");
  host.appendChild(a);
  a.click();
  a.remove();
  // Revoked later, not at once: some browsers start the download after click() returns.
  window.setTimeout(function () { window.URL.revokeObjectURL(url); }, 10000);
}

// -- clipboard -------------------------------------------------------------------

/**
 * Put `text` on the clipboard. Resolves to how: "clipboard", "execCommand"
 * or "manual".
 *
 * navigator.clipboard exists only in a secure context (https, or localhost),
 * and the DAQ machine's mhttpd is plain http under its hostname -- so there
 * the old execCommand("copy") on a hidden textarea does it, and where even
 * that is refused the text is shown selected in a popup for Ctrl-C.
 */
async function copyText(text) {
  const nav = window.navigator;
  if (window.isSecureContext && nav && nav.clipboard && nav.clipboard.writeText) {
    try {
      await nav.clipboard.writeText(text);
      return "clipboard";
    } catch (e) { /* denied: try the old way */ }
  }
  if (execCopy(text)) return "execCommand";
  showCopyPopup(text);
  return "manual";
}

function execCopy(text) {
  const host = document.body;
  if (!host || typeof document.execCommand !== "function") return false;
  const ta = el("textarea", { readonly: "", "aria-hidden": "true" });
  ta.value = text;
  Object.assign(ta.style, { position: "fixed", top: "0", left: "0", width: "1px",
                            height: "1px", opacity: "0" });
  host.appendChild(ta);
  let ok = false;
  try {
    ta.focus();
    ta.select();
    ok = !!document.execCommand("copy");
  } catch (e) {
    ok = false;
  }
  ta.remove();
  return ok;
}

function showCopyPopup(text) {
  closeCopyPopup();
  const host = document.body || document.getElementById("dqm-root");
  const ta = el("textarea", { class: "dqm-smaev-popuptext", readonly: "", rows: "4" });
  ta.value = text;
  const close = el("button", { class: "mbutton dqm-smaev-mini" }, "Close");
  close.onclick = closeCopyPopup;
  ta.addEventListener("keydown", function (e) { if (e && e.key === "Escape") closeCopyPopup(); });
  host.appendChild(el("div", { class: "dqm-smaev-popup", id: "dqm-smaev-popup", role: "dialog" },
    el("div", {}, "The browser would not copy. The text is selected: press Ctrl-C, then Close."),
    ta, close));
  try { ta.focus(); ta.select(); } catch (e) { /* still readable */ }
}

function closeCopyPopup() {
  const pop = document.getElementById("dqm-smaev-popup");
  if (pop) pop.remove();
}

/** Copy, then say so on the button for a moment (unless it took the popup). */
async function copyWithFeedback(btn, text) {
  if (!btn.dataset.label) btn.dataset.label = btn.textContent;
  const how = await copyText(text);
  if (how === "manual") return how;
  btn.textContent = "Copied ✓";
  btn.classList.add("dqm-smaev-copied");
  if (btn._copiedTimer) window.clearTimeout(btn._copiedTimer);
  btn._copiedTimer = window.setTimeout(function () {
    btn._copiedTimer = null;
    btn.textContent = btn.dataset.label;
    btn.classList.remove("dqm-smaev-copied");
  }, COPIED_MS);
  return how;
}

// -- the hit under the mouse -------------------------------------------------------

/**
 * One hit as a line: channel, ToT, time, the fine/coarse check and, with word
 * data, its H000 word index and raw word. Times are ns: absolute (the frame's
 * t0 plus the hit's) and from the first shipped hit.
 */
function hitText(frame, i, tab) {
  if (isPix(frame, i)) return pixelText(frame, i, tab);
  const c = frame.ch[i];
  const l = roles().labels[c];
  const named = l && !/^ch\d+$/.test(l);
  const m = frame.meta || {};
  const tRel = frame.t[i];
  const tText = (tr) => (Number.isFinite(m.t0_ns) ? `t ${m.t0_ns + tr} ns (t_rel ${tr} ns)` : `t_rel ${tr} ns`);
  const offs = nimOffsets(frame);
  const o = offs ? offs[c] : null;
  // A NIM copy: its raw time and the aligned one, t - NIM/offset ns, whichever is drawn.
  const parts = [`ch ${c}${named ? ` (${l})` : ""}`, `ToT ${frame.tot[i]}`,
    o === null ? tText(tRel)
      : `raw ${tText(tRel)} · aligned ${tText(tRel - o)} (NIM/offset ${o} ns)`,
    `fine/coarse ${frame.hitFlags[i] & SMAF.HIT.MISMATCH ? "MISMATCH" : "ok"}`];
  const pt = pairText(frame, i);
  if (pt) parts.push(pt);
  parts.push(wordText(frame, i, tab));
  return parts.join(" · ");
}

/**
 * A hit's TOT + NIM pairing as words (smaf v3): its class, the partner and
 * NIM - TOT, and the sub-flags. "" for a hit with no class (RF, ...) or a
 * frame without pairing.
 */
function pairText(frame, i) {
  if (!frame.cls) return "";
  const b = frame.cls[i];
  const c = b & SMAF.PAIR.CLASS_MASK;
  if (c === SMAF.PAIR.NONE) return "";
  const nimSide = !!(b & SMAF.PAIR.NIM_SIDE);
  let text = PAIR_CLASS_TEXT[c] || `class ${c}`;
  if (c === SMAF.PAIR.NIM_ONLY) {
    text += b & SMAF.PAIR.LAG_HELD ? " (held back from the merge: lag fault)"
      : nimMerge() ? " (merged into its counter)" : " (merge off)";
  }
  const j = frame.pair ? frame.pair[i] : -1;
  if (c === SMAF.PAIR.PAIRED) {
    if (!frame.pair) {
      text += " (freeze for its partner)";
    } else if (j >= 0) {
      const dt = nimSide ? frame.t[i] - frame.t[j] : frame.t[j] - frame.t[i];
      const offs = nimOffsets(frame);
      const o = offs ? offs[nimSide ? frame.ch[i] : frame.ch[j]] : null;
      const sg = (x) => `${x > 0 ? "+" : ""}${x}`;
      text += ` with ${labelOf(frame.ch[j])} (ch ${frame.ch[j]}), NIM − TOT ${sg(dt)} ns` +
        (o === null ? "" : ` (aligned ${sg(dt - o)} ns)`);
    } else {
      text += " (partner outside what was shipped)";
    }
  }
  const fl = [];
  if (b & SMAF.PAIR.MULTI) fl.push("multi-candidate");
  if (b & SMAF.PAIR.SHADOW) fl.push("in TOT shadow");
  if (b & SMAF.PAIR.EDGE) fl.push("near frame edge");
  if (b & SMAF.PAIR.LAG_HELD) fl.push("lag-held");
  return `${nimSide ? "NIM word" : "TOT word"}: ${text}${fl.length ? ` [${fl.join(", ")}]` : ""}`;
}

/**
 * A pixel hit as a line: plane (or "chip N (no plane)"), chip, column, row,
 * ToT in counts and ns, time (absolute and from the first shipped SMA hit, as
 * for the SMA hits), and its word.
 */
function pixelText(frame, i, tab) {
  const p = frame.pixels;
  const j = i - frame.nHits;
  const m = frame.meta || {};
  const mp = m.mupix || {};
  const plane = p.flags[j] & SMAF.PIX.PLANE_MASK;
  const rel = pixRel(frame, j);
  const totNs = (mp.tot_ns || 256) * p.tot[j];
  const parts = [plane ? `MuPix ${plane === 1 ? "L1" : "L2"} chip ${p.chip[j]}` : `MuPix chip ${p.chip[j]} (no plane)`,
    `col ${p.col[j]}`, `row ${p.row[j]}${p.flags[j] & SMAF.PIX.OFF_SENSOR ? " (not on the sensor)" : ""}`,
    `ToT ${p.tot[j]} (~${totNs} ns)`,
    Number.isFinite(m.t0_ns) ? `t ${m.t0_ns + rel} ns (t_rel ${rel} ns)` : `t_rel ${rel} ns`,
    wordText(frame, i, tab)];
  return parts.join(" · ");
}

function wordText(frame, i, tab) {
  const w = wordOf(frame, i), raw = rawOf(frame, i);
  if (w !== undefined && raw !== undefined) return `word ${w} · ${SMAF.hex64(raw)}`;
  if (!state.paused) return "freeze for word index";
  if (state.wordsBusy[tab]) return "fetching word index…";
  const n = state.wordsNote[tab];
  if (n && n.seq === frame.frameSeq) return `no word index: ${n.text}`;
  return "no word index from this analyzer";
}

/** What the hover line says with no hit under the mouse. */
function hoverHint(tab) {
  const f = state.frames[tab];
  const live = !(f && f.wordIndex) && !state.paused;
  const words = !live ? "" : tab === "raster"
    ? " · a click freezes the frame first, for the word index" : " · freeze for word index";
  return `hover a hit for its channel, time and word; click to tag it, again to untag${words}`;
}

/** The hover line under a canvas (the tagged hits have their own list). */
function makeHitInfo(tab) {
  const hover = el("div", { class: "dqm-smaev-hit dqm-smaev-hover" });
  const wrap = el("div", { class: "dqm-smaev-hitinfo" }, hover);
  const info = { tab, wrap, hover, over: false };
  setHover(info, null);
  return info;
}

function lineNodes(holder, line, tag) {
  holder.innerHTML = "";
  holder.appendChild(el("span", { class: "dqm-smaev-hitline" }, line));
  holder.appendChild(el("span", { class: "dqm-smaev-hittag" }, ` · ${tag}`));
}

function setHover(info, line, tag) {
  if (!line) {
    info.hover.innerHTML = "";
    info.hover.appendChild(el("span", { class: "dqm-smaev-hint" }, hoverHint(info.tab)));
    return;
  }
  lineNodes(info.hover, line, tag);
}

function hitInfos(tab) {
  return (tab === "raster" ? [state.rasterInfo] : state.seedPanels.map((p) => p.info)).filter(Boolean);
}

/** Esc: the hover lines back to their hint (the tagged list stays). */
function clearHover() {
  for (const tab of ["seeded", "raster"]) hitInfos(tab).forEach((info) => setHover(info, null));
}

/** Redraw a tab's view (the tag marks) without touching the toolbar notes. */
function repaint(tab) {
  const f = state.frames[tab];
  if (tab === "seeded") renderSeeded(f);
  else if (f) { guard("dqm-smaev-rasterbox", () => paintRaster(f)); renderHints("raster"); }
}

/** The idle hover hints of a tab (they say whether a click freezes first). */
function renderHints(tab) {
  hitInfos(tab).forEach(function (info) { if (!info.over) setHover(info, null); });
}

// -- tagged hits -------------------------------------------------------------------

/** Which frame this is: run, event id, serial and frame seq, as in its tag. */
function frameKey(frame) {
  const ev = (frame.meta || {}).event || {};
  return [frame.run, ev.id === undefined ? "" : ev.id, serialOf(frame), frame.frameSeq].join("|");
}

function dropKey(frame) { return JSON.stringify((frame.meta || {}).dropped || []); }

/**
 * A hit's identity. With word data it is its H000 word, the same whichever
 * view shows it, so a hit tagged on the seeded tab is marked on the frozen
 * raster too (and a click there untags it); without, the view's own index.
 */
function hitKey(frame, tab, i) {
  const fk = frameKey(frame);
  const w = wordOf(frame, i);
  return w !== undefined ? `${fk}|w${w}` : `${fk}|${tab}|${dropKey(frame)}|${i}`;
}

/**
 * A tagged hit, as plain values (it is kept in sessionStorage): the frame's
 * whole identity at click time, the hit's fields and its line as shown.
 */
function makeTag(tab, frame, i) {
  const m = frame.meta || {};
  const ev = m.event || {};
  const t0 = Number.isFinite(m.t0_ns) ? m.t0_ns : null;
  const pixel = isPix(frame, i);
  const j = i - frame.nHits;
  const tRel = pixel ? pixRel(frame, j) : frame.t[i];
  const w = wordOf(frame, i), raw = rawOf(frame, i);
  return {
    key: hitKey(frame, tab, i), frame: frameKey(frame), tag: tagOf(frame),
    run: frame.run, eventId: ev.id === undefined ? null : ev.id, serial: serialOf(frame),
    utc: Number.isFinite(ev.timestamp) ? new Date(ev.timestamp * 1000).toISOString() : null,
    seq: frame.frameSeq, view: tab, drop: dropKey(frame), i,
    word: w === undefined ? null : w,
    raw: raw === undefined ? null : SMAF.hex64(raw),
    kind: pixel ? "pixel" : "sma",
    ch: pixel ? null : frame.ch[i], tot: pixel ? frame.pixels.tot[j] : frame.tot[i],
    chip: pixel ? frame.pixels.chip[j] : null, tRel,
    tAbs: t0 === null ? null : t0 + tRel,
    line: hitText(frame, i, tab),
  };
}

/** Add hit i of the frame, unless it is there already; false when the list is full. */
function addTag(tab, frame, i) {
  const key = hitKey(frame, tab, i);
  if (state.tagged.some((x) => x.key === key)) return true;
  if (state.tagged.length >= MAX_TAGS) return false;
  state.tagged.push(makeTag(tab, frame, i));
  return true;
}

/** A click on a hit: tag it, or untag it if it is tagged. */
function toggleTag(tab, frame, i) {
  const key = hitKey(frame, tab, i);
  const at = state.tagged.findIndex((x) => x.key === key);
  if (at >= 0) state.tagged.splice(at, 1);
  else if (!addTag(tab, frame, i)) { renderTags(`the list is full (${MAX_TAGS} hits): remove some, or Clear all`); return; }
  tagsChanged();
}

function removeTag(key) {
  state.tagged = state.tagged.filter((x) => x.key !== key);
  tagsChanged();
}

function clearTags() {
  state.tagged = [];
  state.taggedPending = [];
  tagsChanged();
}

function tagsChanged() {
  saveTags();
  renderTags();
  if (state.frames[state.tab] !== undefined) repaint(state.tab);
}

/**
 * A click on a raster hit. The raster's word data comes only with a frozen
 * frame, so a click on a live one freezes (which asks for the words) and the
 * hit is tagged once they are in; a frozen frame that cannot get them (an old
 * analyzer, a frame no longer held) is tagged with what it has.
 */
function rasterClick(f, i) {
  const tried = state.wordsNote.raster;
  if (f.wordIndex || (state.paused && !state.wordsBusy.raster && tried && tried.seq === f.frameSeq)) {
    toggleTag("raster", f, i);
    return;
  }
  const p = state.taggedPending;
  const at = p.findIndex((x) => x.seq === f.frameSeq && x.i === i);
  if (at >= 0) p.splice(at, 1);
  else p.push({ seq: f.frameSeq, i });
  renderTags();
  if (!state.paused) setPaused(true);
  else ensureWords("raster");
}

/** The raster clicks that waited for the words: tagged now, if their frame is still shown. */
function flushPending() {
  const p = state.taggedPending.splice(0);
  if (!p.length) return;
  const f = state.frames.raster;
  let full = false;
  for (const x of p) {
    if (f && f.frameSeq === x.seq && x.i < nAll(f) && !addTag("raster", f, x.i)) full = true;
  }
  saveTags();
  renderTags(full ? `the list is full (${MAX_TAGS} hits): remove some, or Clear all` : "");
  if (state.frames[state.tab] !== undefined) repaint(state.tab);
}

/**
 * A tag made on this frame before it had word data (an analyzer's v1 reply)
 * gains the word once the frame on screen has it, as the hover line does.
 */
function refreshTags(tab, frame) {
  if (!frame || !frame.wordIndex || !state.tagged.length) return;
  const fk = frameKey(frame), dk = dropKey(frame);
  let changed = false;
  const out = [];
  for (const x of state.tagged) {
    if (x.frame !== fk || x.word !== null || x.view !== tab || x.drop !== dk || x.i >= nAll(frame)) {
      out.push(x);
      continue;
    }
    const nx = makeTag(tab, frame, x.i);
    changed = true;
    if (!state.tagged.some((y) => y !== x && y.key === nx.key)) out.push(nx);   // else: tagged twice
  }
  if (!changed) return;
  state.tagged = out;
  saveTags();
  renderTags();
}

/**
 * The tagged hits grouped by frame, groups in the order of their first tag and
 * hits in the order they were tagged, numbered 1, 2, ... down the list (the
 * canvas numbers are these). Each hit after a frame's first carries dt, ns
 * from that first hit (null when the two have no common time base).
 */
function taggedGroups() {
  const groups = [];
  const byFrame = new Map();
  for (const x of state.tagged) {
    let g = byFrame.get(x.frame);
    if (!g) {
      g = { frame: x.frame, tag: x.tag, run: x.run, serial: x.serial, rows: [] };
      byFrame.set(x.frame, g);
      groups.push(g);
    }
    g.rows.push({ x });
  }
  let n = 0;
  for (const g of groups) {
    const first = g.rows[0].x;
    g.rows.forEach(function (r, k) {
      r.n = ++n;
      if (k) r.dt = dtNs(first, r.x);
    });
  }
  return groups;
}

function dtNs(a, b) {
  if (a.tAbs !== null && b.tAbs !== null) return b.tAbs - a.tAbs;
  if (a.view === b.view && a.drop === b.drop) return b.tRel - a.tRel;   // the same shipped t0
  return null;
}

function dtText(dt) {
  return dt === null ? "Δt — (no common time base)" : `Δt ${dt > 0 ? "+" : ""}${dt} ns`;
}

/** A row's hit line, with its Δt after a frame's first. */
function rowLine(r) {
  return r.dt === undefined ? r.x.line : `${r.x.line} · ${dtText(r.dt)}`;
}

/** Where the files have this frame: mdqm-sma-file, with the tagged words' range. */
function fileHint(g) {
  const w = g.rows.map((r) => r.x.word).filter((v) => Number.isInteger(v));
  const words = w.length ? ` --words ${Math.min(...w)}:${Math.max(...w)}` : "";
  return `mdqm-sma-file --serial ${g.serial} --run ${g.run} --dir <raw dir>${words}`;
}

/**
 * What "Copy all" copies: per frame, its tag, then its hits one per line
 * ("#n · line · Δt"), then the mdqm-sma-file line that lists those words from
 * the run file; frames separated by a blank line.
 */
function taggedText() {
  const out = [];
  for (const g of taggedGroups()) {
    if (out.length) out.push("");
    out.push(g.tag);
    for (const r of g.rows) out.push(`  #${r.n} · ${rowLine(r)}`);
    out.push(`  ${fileHint(g)}`);
  }
  return out.join("\n");
}

function makeTaggedPanel() {
  const title = el("span", { class: "dqm-smaev-taggedtitle", id: "dqm-smaev-tagged-title" }, "Tagged hits (0)");
  const copyAll = el("button", {
    class: "mbutton dqm-smaev-mini", id: "dqm-smaev-tagged-copy",
    title: "Copy every tagged hit for the elog: per frame its tag, its hits with Δt, and the " +
           "mdqm-sma-file line that finds them in the run file",
  }, "Copy all");
  copyAll.onclick = function () { if (state.tagged.length) copyWithFeedback(copyAll, taggedText()); };
  const clear = el("button", { class: "mbutton dqm-smaev-mini", id: "dqm-smaev-tagged-clear",
                               title: "Remove every tagged hit from the list" }, "Clear all");
  clear.onclick = clearTags;
  const note = el("span", { class: "dqm-smaev-taggednote", id: "dqm-smaev-tagged-note" });
  const head = el("div", { class: "dqm-smaev-taggedhead" }, title, copyAll, clear, note);
  const list = el("div", { class: "dqm-smaev-taggedlist", id: "dqm-smaev-tagged-list" });
  return el("div", { class: "dqm-smaev-tagged", id: "dqm-smaev-tagged" }, head, list);
}

/**
 * The "Tagged hits" list. Rebuilt only when the list changes, never per frame,
 * so its buttons survive the 4 Hz redraws. `extra` is a one-off note (the
 * list is full).
 */
function renderTags(extra) {
  const title = document.getElementById("dqm-smaev-tagged-title");
  const list = document.getElementById("dqm-smaev-tagged-list");
  if (!title || !list) return;
  const n = state.tagged.length;
  title.textContent = `Tagged hits (${n})`;
  document.getElementById("dqm-smaev-tagged-copy").disabled = !n;
  document.getElementById("dqm-smaev-tagged-clear").disabled = !n && !state.taggedPending.length;
  const notes = [];
  if (state.taggedPending.length) {
    notes.push(`${state.taggedPending.length} raster hit${state.taggedPending.length === 1 ? "" : "s"} ` +
               "tagged once the frozen frame's word data is in…");
  }
  if (extra) notes.push(extra);
  else if (n >= MAX_TAGS) notes.push(`the list holds at most ${MAX_TAGS} hits`);
  const note = document.getElementById("dqm-smaev-tagged-note");
  note.textContent = notes.join(" · ");
  note.className = `dqm-smaev-taggednote${extra || n >= MAX_TAGS ? " full" : ""}`;
  list.innerHTML = "";
  if (!n) {
    list.appendChild(el("div", { class: "dqm-smaev-hint" },
      "Click hits on either tab to tag them; the list keeps them across tabs and new frames."));
    return;
  }
  for (const g of taggedGroups()) {
    list.appendChild(el("div", { class: "dqm-smaev-taggedframe" }, el("code", { class: "dqm-smaev-tag" }, g.tag)));
    for (const r of g.rows) {
      const copy = el("button", { class: "mbutton dqm-smaev-mini dqm-smaev-copyhit",
                                  title: "Copy this line, with the frame tag, for the elog" }, "Copy");
      const text = `${rowLine(r)} · ${r.x.tag}`;
      copy.onclick = function () { copyWithFeedback(copy, text); };
      const x = el("button", { class: "mbutton dqm-smaev-mini dqm-smaev-untag", title: "Untag this hit" }, "×");
      x.onclick = function () { removeTag(r.x.key); };
      list.appendChild(el("div", { class: "dqm-smaev-hit dqm-smaev-taggedrow" },
        el("span", { class: "dqm-smaev-tagnum" }, String(r.n)), copy, x,
        el("span", { class: "dqm-smaev-hitline" }, rowLine(r))));
    }
  }
}

/**
 * The canvas numbers of the tagged hits in this frame, for one view:
 * {any, of(i)}. By word where the frame has word data (a hit tagged on
 * either tab); by index for a tag made on this view without it.
 */
function tagNumbers(tab, frame) {
  const byWord = new Map(), byIndex = new Map();
  if (frame && state.tagged.length) {
    const fk = frameKey(frame), dk = dropKey(frame);
    for (const g of taggedGroups()) {
      if (g.frame !== fk) continue;
      for (const r of g.rows) {
        const x = r.x;
        if (x.word !== null && frame.wordIndex) byWord.set(x.word, r.n);
        else if (x.view === tab && x.drop === dk && x.i < nAll(frame)) byIndex.set(x.i, r.n);
      }
    }
  }
  return {
    any: byWord.size + byIndex.size > 0,
    of: function (i) {
      const w = byWord.size ? wordOf(frame, i) : undefined;
      return w !== undefined && byWord.has(w) ? byWord.get(w) : byIndex.get(i);
    },
  };
}

function saveTags() {
  try { window.sessionStorage.setItem(SS_TAGS, JSON.stringify(state.tagged)); } catch (e) { /* no storage */ }
}

function restoreTags() {
  try {
    const a = JSON.parse(window.sessionStorage.getItem(SS_TAGS) || "[]");
    if (Array.isArray(a)) {
      state.tagged = a.filter((x) => x && typeof x.key === "string" && typeof x.frame === "string" &&
                                     typeof x.line === "string" && typeof x.tag === "string")
                      .slice(0, MAX_TAGS);
    }
  } catch (e) { /* none kept, or unreadable: start empty */ }
}

/**
 * The frozen frame again, with its words, when it came without (a polled
 * raster, or an analyzer that answered v1): once per frame. Chained behind
 * whatever request is in flight, and dropped if the page moved on meanwhile.
 */
async function ensureWords(tab) {
  const f = state.frames[tab];
  if (!f || f.words || !state.paused || state.wordsBusy[tab]) return;
  const tried = state.wordsNote[tab];
  if (tried && tried.seq === f.frameSeq) return;
  state.wordsBusy[tab] = true;
  if (state.tab === tab) renderHints(tab);
  const prev = state.inflight;
  try {
    await track((async function () {
      if (prev) { try { await prev; } catch (e) { /* its owner reports it */ } }
      await wordsFetch(tab, f);
    })());
  } catch (e) {
    state.wordsNote[tab] = { seq: f.frameSeq, text: BRPC.errorText(e) };
  } finally {
    state.wordsBusy[tab] = false;
    if (state.tab === tab && state.frames[tab]) renderHints(tab);
    if (tab === "raster") flushPending();
  }
}

/**
 * Fetch and keep the frozen frame's raw event (see state.rawKept), chained
 * behind any request in flight. A failure is left to Download to report: it
 * asks again and shows the analyzer's answer.
 */
async function keepRaw(tab) {
  const f = state.frames[tab];
  if (!f || !state.paused || (f.meta || {}).raw_held !== true) return;
  const seq = f.frameSeq;
  if (state.rawKept.has(seq) || state.rawBusy.has(seq)) return;
  state.rawBusy.add(seq);
  const prev = state.inflight;
  try {
    await track((async function () {
      if (prev) { try { await prev; } catch (e) { /* its owner reports it */ } }
      const payload = await SMAF.fetchRaw(state.client, seq);
      state.rawKept.set(seq, payload);
      while (state.rawKept.size > RAW_KEEP) state.rawKept.delete(state.rawKept.keys().next().value);
    })());
  } catch (e) {
    /* not held any more, or no sma::raw: Download says so when pressed */
  } finally {
    state.rawBusy.delete(seq);
    if (state.frames[tab]) renderTagBar(tab, state.frames[tab]);
  }
}

async function wordsFetch(tab, f) {
  const m = f.meta || {};
  // The same selection as the frame on screen, so hit i is the same hit.
  const args = tab === "raster"
    ? { view: "raster", drop: m.dropped || [], max_hits: MAX_HITS, words: true, seq: f.frameSeq }
    : Object.assign({ view: "seeded", words: true, seq: f.frameSeq }, seedArgs());
  if (tab === "raster" && !m.mupix) args.pixels = false;     // as shown: no MuPix rows
  const frame = await SMAF.fetchFrame(state.client, args, WORDS_MAX_REPLY);
  const shown = state.frames[tab];
  if (!state.paused || !shown || shown.frameSeq !== f.frameSeq) return;   // moved on
  if (!frame || frame.frameSeq !== f.frameSeq || !frame.words) {
    state.wordsNote[tab] = { seq: f.frameSeq, text: "the analyzer sent none" };
    return;
  }
  state.frames[tab] = frame;
  if (state.tab === tab) render(tab);
}

// ---------------------------------------------------------------------------
// Seed choice, filters and the staleness banner
// ---------------------------------------------------------------------------

/**
 * The seeded request's seed/filters/MuPix/pattern args: always sent, "s1", [],
 * "any" and {} being the default.
 */
function seedArgs() {
  return { seed: state.seedSel.seed, filters: state.seedSel.filters.slice(), mupix: state.seedSel.mupix,
           pattern: Object.assign({}, state.seedSel.pattern) };
}

/**
 * A pattern selector object with only valid "present"/"absent" entries for
 * counters 1..n (default PATTERN_MAX: a stored choice kept until the roles are known).
 */
function cleanPattern(o, n) {
  const out = {};
  if (!o || typeof o !== "object" || Array.isArray(o)) return out;
  const max = n === undefined ? PATTERN_MAX : n;
  for (let k = 1; k <= max; k++) {
    const v = o[String(k)];
    if (v === "present" || v === "absent") out[String(k)] = v;
  }
  return out;
}

/**
 * The counters with a known timestamp fault now: the summary's
 * timestamp_faults (the rule of the SMAPlots fine/coarse flag), else what the
 * last seeded reply said it ignored. [{counter, label, mismatch_frac}].
 */
function faultedCounters() {
  const tf = (state.summary || {}).timestamp_faults;
  if (tf && Array.isArray(tf.counters)) return tf.judged === false ? [] : tf.counters;
  const inc = ((state.frames.seeded || {}).meta || {}).incomplete;
  return inc && Array.isArray(inc.ignored) ? inc.ignored : [];
}

function pctText(f) {
  const x = 100 * f;
  return x < 10 ? `${x.toFixed(1)} %` : `${Math.round(x)} %`;
}

/** The inline warnings of the pattern row: one per counter with a timestamp fault. */
function updatePatternWarnings() {
  updatePatternHead();
  const bad = {};
  for (const f of faultedCounters()) bad[f.counter] = f;
  const r = roles();
  syncPatternBoxes(r);
  for (let k = 1; k <= nPattern(r); k++) {
    const lab = document.getElementById(`dqm-smaev-pattext-${k}`);
    if (lab) lab.textContent = labelOf(r.counters[k - 1], r);
    const w = document.getElementById(`dqm-smaev-patwarn-${k}`);
    if (!w) continue;
    const f = bad[k];
    const text = f ? `${f.label || `S${k}`} has a timestamp fault` +
      (Number.isFinite(f.mismatch_frac) ? ` (${pctText(f.mismatch_frac)} fine/coarse mismatch)` : "") +
      ": 'absent' will match almost everything, 'present' almost nothing" : "";
    w.textContent = text;
    w.style.display = text ? "" : "none";
    const s = document.getElementById(`dqm-smaev-pat-${k}`);
    if (s) s.value = state.seedSel.pattern[String(k)] || "any";
  }
}

function mupixText(mode) {
  const m = MUPIX_MODES.find((x) => x[0] === mode);
  return m ? m[1] : mode;
}

/** The oddity's words in a reply: "incomplete pattern" carries the analyzer's
 *  note on the counters it ignored (meta.incomplete.label), the rest filterText. */
function oddText(frame, name, short) {
  const inc = ((frame || {}).meta || {}).incomplete;
  if (name !== "incomplete" || !inc) return filterText(name);
  if (!short) return inc.label || filterText(name);
  const ign = (inc.ignored || []).map((f) => f.label || `S${f.counter}`);
  return ign.length ? `${filterText(name)} (${ign.join(", ")} ignored)` : filterText(name);
}

function filterText(name) {
  const f = FILTERS.find((x) => x[0] === name);
  if (!f) return name;
  if (name === "tot") {
    const fr = state.frames.seeded;
    const cut = fr && fr.meta && fr.meta.tot_corrupt;
    if (cut) return `ToT ≥ ${cut}`;
  }
  return f[1];
}

/**
 * The seed list: S1 (default), each other counter, the delayed channels (all by
 * the summary's labels, i.e. the ODB roles), then "any counter". RF and the
 * current channel are no seeds worth offering. A stored choice that the roles
 * no longer list stays selectable rather than silently changing.
 */
function seedOptions() {
  const r = roles();
  const out = [["s1", `${labelOf(r.counters.length ? r.counters[0] : 1, r)} (default)`]];
  for (const c of r.counters.slice(1)) out.push([`ch${c}`, `${labelOf(c, r)} (ch ${c})`]);
  for (const c of r.delayed) {
    if (r.counters.indexOf(c) < 0) out.push([`ch${c}`, `${labelOf(c, r)} (ch ${c})`]);
  }
  out.push(["any", "any counter (S1..S5 clusters)"]);
  const cur = state.seedSel.seed;
  if (!out.some((o) => o[0] === cur)) out.push([cur, cur]);
  return out;
}

function seedLabel(seed) {
  if (seed === "any") return "any counter";
  const o = seedOptions().find((x) => x[0] === seed);
  return o ? o[1].replace(/ \(default\)$/, "") : seed;
}

function seededTabLabel() {
  return state.seedSel.seed === "s1" ? TABS[0][1] : `Seeded events: ${seedLabel(state.seedSel.seed)}`;
}

/** An age in seconds as the banner says it: tenths below 10 s. */
function ageText(s) {
  if (!Number.isFinite(s)) return "";
  return s < 10 ? `${s.toFixed(1)} s` : `${Math.round(s)} s`;
}

/** Rebuild the seed list when the roles (the summary) change; the choice is kept. */
function updateSeedOptions() {
  const sel = document.getElementById("dqm-smaev-seedsel");
  if (!sel) return;
  const opts = seedOptions();
  const key = JSON.stringify(opts);
  if (sel._optsKey !== key) {
    sel._optsKey = key;
    sel.innerHTML = "";
    for (const [v, label] of opts) sel.appendChild(el("option", { value: v }, label));
  }
  sel.value = state.seedSel.seed;
  updateCurrentToggle();
  const tab = document.getElementById("dqm-smaev-tab-seeded");
  if (tab) tab.textContent = seededTabLabel();
  const tl = document.getElementById("dqm-smaev-filtertext-tot");
  if (tl) tl.textContent = filterText("tot");
  const ms = document.getElementById("dqm-smaev-mupixsel");
  if (ms) ms.value = state.seedSel.mupix;
  updatePatternWarnings();
}

/**
 * The seed or a filter changed: ask again at once. Live, that is the next poll
 * started now; frozen, the frozen frame is asked for again by its seq with the
 * new choice, so what is on screen stays the frame that was frozen.
 */
function seedChoiceChanged() {
  saveSeedSel();
  updateSeedOptions();
  if (state.tab !== "seeded") return;
  if (!state.paused) { startPolling(); return; }
  const f = state.frames.seeded;
  if (!f) return;
  const prev = state.inflight;
  track((async function () {
    if (prev) { try { await prev; } catch (e) { /* its owner reports it */ } }
    const args = Object.assign({ view: "seeded", seq: f.frameSeq }, seedArgs());
    const frame = await SMAF.fetchFrame(state.client, args);
    if (!state.paused || state.tab !== "seeded") return;
    state.frames.seeded = frame;
    render("seeded");
  })()).catch(function (e) {
    if (typeof console !== "undefined") console.warn("dqm-sma-events reselect", e);
    singleNote(`frame seq ${f.frameSeq}: ${BRPC.errorText(e)}`);
  });
}

function saveSeedSel() {
  try { window.sessionStorage.setItem(SS_SEED, JSON.stringify(state.seedSel)); } catch (e) { /* no storage */ }
}

function restoreSeedSel() {
  try {
    const o = JSON.parse(window.sessionStorage.getItem(SS_SEED) || "{}");
    if (typeof o.seed === "string" && /^(s1|any|ch\d{1,2})$/.test(o.seed)) state.seedSel.seed = o.seed;
    if (Array.isArray(o.filters)) {
      state.seedSel.filters = FILTERS.map((f) => f[0]).filter((f) => o.filters.indexOf(f) >= 0);
    }
    if (MUPIX_MODES.some((m) => m[0] === o.mupix)) state.seedSel.mupix = o.mupix;
    state.seedSel.pattern = cleanPattern(o.pattern);
  } catch (e) { /* none kept, or unreadable: the default */ }
}

/** "matching seeds: n of m candidates in frame seq X", for a chosen seed or filters. */
function selectHeader(frame, holder) {
  const sel = (frame.meta || {}).select;
  if (!sel) return;
  const f = sel.filters && sel.filters.length
    ? ` with ${sel.filters.map((n) => oddText(frame, n)).join(" or ")}` : "";
  const pt = sel.pattern_label ? ` and ${sel.pattern_label}` : "";
  const mp = sel.mupix && sel.mupix !== "any" ? ` and ${sel.mupix_label || mupixText(sel.mupix)}` : "";
  holder.appendChild(badge(`seed: ${sel.seed_label}${f}${pt}${mp}`, "blue"));
  const inc = (frame.meta || {}).incomplete;
  if (inc && inc.judged === false && sel.filters && sel.filters.indexOf("incomplete") >= 0) {
    holder.appendChild(badge(`incomplete pattern cannot be judged: ${inc.note.replace(/^cannot judge: /, "")}`,
                             "yellow"));
  }
  const capped = sel.capped ? ` (the latest ${num(sel.examined)} examined)` : "";
  holder.appendChild(badge(`matching seeds: ${num(sel.matching)} of ${num(sel.candidates)} candidates ` +
                           `in frame seq ${frame.frameSeq}${capped}`,
                           sel.matching ? "" : "yellow"));
}

/**
 * The banner above the seed panels: the frame shown is not the newest analysed
 * good frame (and why), or nothing matched at all. Hidden otherwise.
 */
function renderBanner(frame) {
  const b = document.getElementById("dqm-smaev-banner");
  if (!b) return;
  const m = (frame && frame.meta) || {};
  const nm = m.search && m.search.no_match;
  const sv = m.stale_view;
  let text = "", cls = "";
  if (nm) {
    const span = Number.isFinite(nm.span_s) ? ` (${ageText(nm.span_s)})` : "";
    text = `No match: ${nm.text}${span}. Showing the newest analysed frame, seq ${frame.frameSeq}, ` +
           "without seeds; the search goes on with every new frame.";
    cls = "red";
  } else if (sv) {
    const age = Number.isFinite(sv.age_s) ? `${ageText(sv.age_s)} ago` : "earlier";
    const n = sv.good_since;
    text = `Showing frame seq ${sv.shown_seq} from ${age}: ${sv.reason} in the ${num(n)} good ` +
           `frame${n === 1 ? "" : "s"} analysed since (newest seq ${sv.newest_seq}).`;
    if (m.raw_held === false) text += " Its raw event is no longer held: use the tag.";
    cls = "yellow";
  }
  b.textContent = text;
  b.className = text ? `dqm-smaev-banner dqm-diagnosis ${cls}` : "dqm-smaev-banner";
  b.style.display = text ? "" : "none";
}

// ---------------------------------------------------------------------------
// Chrome
// ---------------------------------------------------------------------------

function build() {
  const r = document.getElementById("dqm-root");
  r.innerHTML = "";

  // The page's controls on a row of their own above the tab strip, so they
  // do not read as more tabs.
  const bar = el("div", { class: "dqm-strip dqm-sma-toolbar", id: "dqm-smaev-toolbar" });
  const freeze = el("button", { class: "mbutton", id: "dqm-smaev-freeze" }, "Freeze");
  freeze.onclick = function () { setPaused(!state.paused); };
  bar.appendChild(freeze);
  const one = el("button", {
    class: "mbutton", id: "dqm-smaev-single",
    title: "Freeze, then fetch one newer frame for this tab. Press again for the next one; " +
           "Frozen — resume goes back to live polling.",
  }, "Single ▸");
  one.onclick = function () { single(); };
  bar.appendChild(one);
  bar.appendChild(el("span", { id: "dqm-smaev-rate" }));
  bar.appendChild(el("span", { class: "dqm-chip", id: "dqm-smaev-live" }, "…"));
  bar.appendChild(el("span", { class: "dqm-chip", id: "dqm-smaev-seq" }, "frame seq —"));
  const note = el("span", { class: "dqm-chip yellow", id: "dqm-smaev-singlenote" });
  note.style.display = "none";
  bar.appendChild(note);
  r.appendChild(bar);
  r.appendChild(tabStrip(TABS.map(([id, label]) => [id, id === "seeded" ? seededTabLabel() : label]),
                         "dqm-smaev", "SMA event views", () => state.tab, showTab));

  // Seeded pane
  const seeded = el("div", { id: "dqm-smaev-pane-seeded", role: "tabpanel",
                             "aria-labelledby": "dqm-smaev-tab-seeded" });
  const sbar = el("div", { class: "dqm-strip" });
  sbar.appendChild(labelled("window", select(
    [["full", "full (-200 ns .. +3 us)"], ["prompt", "prompt (±150 ns)"]], state.zoom,
    function (v) { state.zoom = v; save(); if (state.frames.seeded) render("seeded"); })));
  sbar.appendChild(rawToggle("seeded"));
  const seedSel = select(seedOptions(), state.seedSel.seed, function (v) {
    state.seedSel.seed = v;
    seedChoiceChanged();
  });
  seedSel.id = "dqm-smaev-seedsel";
  seedSel.title = "Which hits seed the events: S1 (the default), another channel, or any " +
                  "counter (the first hit of each S1..S5 cluster: events without S1 too)";
  sbar.appendChild(labelled("seed", seedSel));
  const fbox = el("span", { class: "dqm-chip dqm-smaev-filters", id: "dqm-smaev-filters",
                            title: "Show only seeds with any of the checked oddities in their " +
                                   "window; the analyzer searches back through its last good " +
                                   "frames for them" }, el("span", {}, "only seeds with"));
  for (const [name, label, why] of FILTERS) {
    const box = el("input", { type: "checkbox", id: `dqm-smaev-filter-${name}` });
    box.checked = state.seedSel.filters.indexOf(name) >= 0;
    box.onchange = function () {
      const on = new Set(state.seedSel.filters);
      if (this.checked) on.add(name); else on.delete(name);
      state.seedSel.filters = FILTERS.map((f) => f[0]).filter((f) => on.has(f));
      seedChoiceChanged();
    };
    const lab = el("label", { for: `dqm-smaev-filter-${name}`, title: why },
                   el("span", { id: `dqm-smaev-filtertext-${name}` }, label));
    lab.insertBefore(box, lab.firstChild);
    fbox.appendChild(lab);
  }
  sbar.appendChild(fbox);
  const mupixSel = select(MUPIX_MODES, state.seedSel.mupix, function (v) {
    state.seedSel.mupix = v;
    seedChoiceChanged();
  });
  mupixSel.id = "dqm-smaev-mupixsel";
  mupixSel.title = "MuPix is a selection, not an oddity: it is AND-ed with the boxes. " +
                   "'L1+L2 in time': a pixel hit on both planes in the in-time window around " +
                   "the seed; 'none in time': on neither (seeds outside the MuPix data of the " +
                   "frame match only 'any')";
  sbar.appendChild(labelled("MuPix:", mupixSel));
  sbar.appendChild(el("span", { class: "dqm-footnote" },
    "bars: t to t+ToT · hatched red: fine/coarse mismatch · magenta ▼: ToT ≥ 250 · " +
    "dashed red: the seed · black tick on RF: the pulse the phase is taken from · " +
    "MuPix lanes: one tick per pixel hit, coloured by pixel ToT; green band: the in-time window · " +
    "black box + number: a tagged hit"));
  const nimNote = el("span", { class: "dqm-footnote", id: "dqm-smaev-nimlegend" }, nimLegend(false));
  nimNote.style.display = "none";
  sbar.appendChild(nimNote);
  seeded.appendChild(sbar);
  seeded.appendChild(patternRow());
  seeded.appendChild(el("div", { id: "dqm-smaev-note-seeded" }));
  seeded.appendChild(makeTagBar("seeded"));
  seeded.appendChild(el("div", { class: "dqm-strip dqm-sma-header", id: "dqm-smaev-seededhead" }));
  const banner = el("div", { class: "dqm-smaev-banner", id: "dqm-smaev-banner", role: "status" });
  banner.style.display = "none";
  seeded.appendChild(banner);
  seeded.appendChild(el("div", { id: "dqm-smaev-seeds" }));
  r.appendChild(seeded);

  // Raster pane
  const raster = el("div", { id: "dqm-smaev-pane-raster", role: "tabpanel",
                            "aria-labelledby": "dqm-smaev-tab-raster" });
  const rbar = el("div", { class: "dqm-strip" });
  const box = el("input", { type: "checkbox", id: "dqm-smaev-hidecur" });
  box.checked = state.hideCurrent;
  box.onchange = function () {
    state.hideCurrent = !!this.checked;
    save();
    // Asked for again at once, with the new drop list, rather than waiting a
    // poll: the point of the switch is to see the difference.
    if (!state.paused) startPolling();
  };
  const lab = el("label", { for: "dqm-smaev-hidecur", class: "dqm-chip",
                            id: "dqm-smaev-hidecurlab" }, "hide the current channel");
  lab.insertBefore(box, lab.firstChild);
  rbar.appendChild(lab);
  updateCurrentToggle();
  const pbox = el("input", { type: "checkbox", id: "dqm-smaev-hidepix" });
  pbox.checked = state.hidePixels;
  pbox.onchange = function () {
    state.hidePixels = !!this.checked;
    save();
    if (!state.paused) startPolling();
  };
  const plab = el("label", { for: "dqm-smaev-hidepix", class: "dqm-chip",
                             title: "Leave the MuPix rows out (the analyzer then sends no pixel hits)" },
                  "hide MuPix");
  plab.insertBefore(pbox, plab.firstChild);
  rbar.appendChild(plab);
  rbar.appendChild(rawToggle("raster"));
  rbar.appendChild(el("span", { class: "dqm-footnote" },
    "drag to zoom in time, double-click for the whole frame, click a hit to tag it " +
    "(a live frame is frozen first, for its word data)"));
  const rnim = el("span", { class: "dqm-footnote", id: "dqm-smaev-rasternim" },
    "S*k*L rows: the NIM copies · dark mark above a hit: a NIM-only word (grey: held back, lag fault)");
  rnim.style.display = "none";
  rbar.appendChild(rnim);
  raster.appendChild(rbar);
  raster.appendChild(el("div", { id: "dqm-smaev-note-raster" }));
  raster.appendChild(makeTagBar("raster"));
  raster.appendChild(el("div", { class: "dqm-strip dqm-sma-header", id: "dqm-smaev-rasterhead" }));
  raster.appendChild(el("div", { id: "dqm-smaev-rasterbox" }));
  state.rasterInfo = makeHitInfo("raster");
  raster.appendChild(state.rasterInfo.wrap);
  raster.appendChild(legend());
  r.appendChild(raster);

  // One list for both tabs, outside the panes, so a tab switch keeps it.
  r.appendChild(makeTaggedPanel());
  renderTags();
}

/**
 * "raw times", one checkbox per tab, both bound to state.rawTimes: the NIM
 * lanes/rows at their raw SMA time instead of t - NIM/offset ns. Shown only for
 * a frame that carries its NIM offsets (updateRawToggle); display only, nothing
 * is refetched.
 */
function rawToggle(tab) {
  const box = el("input", { type: "checkbox", id: `dqm-smaev-rawtimes-${tab}` });
  box.checked = state.rawTimes;
  box.onchange = function () {
    state.rawTimes = !!this.checked;
    for (const t of ["seeded", "raster"]) {
      const b = document.getElementById(`dqm-smaev-rawtimes-${t}`);
      if (b) b.checked = state.rawTimes;
    }
    save();
    redraw();
  };
  const lab = el("label", {
    for: `dqm-smaev-rawtimes-${tab}`, class: "dqm-chip", id: `dqm-smaev-rawtimeslab-${tab}`,
    title: "Draw the NIM copies (S*k*L) at their raw SMA time. Off: at t − NIM/offset ns, the " +
           "offsets the frame was paired with, so a NIM copy sits under its TOT word. The hover " +
           "line gives both times; tags and word lists are always raw",
  }, "raw times");
  lab.insertBefore(box, lab.firstChild);
  lab.style.display = "none";
  return lab;
}

/** "raw times" only where it changes something: a frame with NIM offsets. */
function updateRawToggle(tab, frame) {
  const lab = document.getElementById(`dqm-smaev-rawtimeslab-${tab}`);
  if (lab) lab.style.display = nimOffsets(frame) ? "" : "none";
}

/**
 * The per-counter pattern selector: S1..Sn (the counters of the roles) each
 * any / present / absent (a hit within the coincidence window of the seed, or
 * none), AND-ed with the seed choice, the boxes and MuPix. Explicit, so nothing
 * is ignored here; a counter with a known timestamp fault gets a warning beside
 * it instead. The selectors follow the roles (syncPatternBoxes).
 */
function patternRow() {
  const row = el("div", { class: "dqm-strip dqm-smaev-patternrow", id: "dqm-smaev-patternrow" });
  row.appendChild(el("span", { class: "dqm-smaev-patternhead", id: "dqm-smaev-patternhead",
                               title: patternTitle(false) }, "and counters:"));
  syncPatternBoxes(roles(), row);
  return row;
}

/**
 * Where the NIM lanes/rows are drawn, for the legends: at t - NIM/offset ns
 * (the frame's offsets, per copy), at their raw time ("raw times"), or "" when
 * the frame has no offsets (drawn raw, nothing to choose).
 */
function alignText(frame) {
  const offs = nimOffsets(frame);
  if (!offs) return "";
  if (state.rawTimes) return "at their raw time (raw times)";
  const r = (frame.meta || {}).roles || {};
  const each = (r.nim || []).filter((n) => Number.isInteger(n) && offs[n] !== null && offs[n] !== undefined)
    .map((n) => `${labelOf(n)} ${offs[n]}`);
  return `at t − NIM/offset ns (${each.join(", ")})`;
}

/** The NIM line of the seeded view's legend, for the merge on or off; `align`: alignText(). */
function nimLegend(merged, align) {
  return `S*k*L lanes: the counter's NIM copy, lighter, outlined${align ? `, ${align}` : ""} · ` +
    "dark tick: a TOT word and its NIM copy · " + (merged
    ? "hollow: NIM-only (merged into its counter) · grey: NIM-only held back (lag fault) · "
    : "hollow: NIM-only (shown, not merged: NIM merge off, the counters and the pattern are " +
      "the TOT words only) · ") +
    "dark\u00a0\\\u00a0hatch: TOT echo word (not paired)";
}

function patternTitle(merged, hasNim) {
  return "Per counter: a hit within the coincidence window of the seed (present) or none " +
         "(absent). AND-ed with the seed, the oddity boxes and MuPix." +
         (merged ? " Merged: a counter's hits are its TOT words plus its NIM-only hits " +
                   "(/DQM/SMA/NIM/merge), as nearline's."
          : hasNim ? " TOT only: the NIM merge is off (/DQM/SMA/NIM/merge), so a counter's " +
                     "hits are its TOT words; its NIM-only hits are shown but not counted." : "");
}

/** The pattern row's head says whether it selects on the merged counters. */
function updatePatternHead() {
  const h = document.getElementById("dqm-smaev-patternhead");
  if (!h) return;
  const hasNim = roles().nim.some((n) => n !== null);
  const merged = nimMerge() && hasNim;
  const text = merged ? "and counters (merged TOT + NIM):"
    : hasNim ? "and counters (TOT only, merge off):" : "and counters:";
  if (h.textContent !== text) h.textContent = text;
  h.setAttribute("title", patternTitle(merged, hasNim));
}

/**
 * One selector per counter of the roles, rebuilt when their number changes.
 * Roles not known yet (no summary, no frame): none, and a stored choice is
 * kept; once known, choices for counters that no longer exist are dropped.
 */
function syncPatternBoxes(r, rowEl) {
  const row = rowEl || document.getElementById("dqm-smaev-patternrow");
  if (!row) return;
  const n = nPattern(r);
  if (row._nBoxes === n) return;
  row._nBoxes = n;
  while (row.children.length > 1) row.children[row.children.length - 1].remove();
  if (n) state.seedSel.pattern = cleanPattern(state.seedSel.pattern, n);
  for (let k = 1; k <= n; k++) {
    const s = select(PATTERN_STATES, state.seedSel.pattern[String(k)] || "any", function (v) {
      const p = Object.assign({}, state.seedSel.pattern);
      if (v === "present" || v === "absent") p[String(k)] = v; else delete p[String(k)];
      state.seedSel.pattern = cleanPattern(p, nPattern());
      seedChoiceChanged();
    });
    s.id = `dqm-smaev-pat-${k}`;
    const lab = el("label", { class: "dqm-chip dqm-smaev-pat", for: s.id },
                   el("span", { id: `dqm-smaev-pattext-${k}` },
                      labelOf(r.counters[k - 1], r)), s);
    row.appendChild(lab);
    const w = el("span", { class: "dqm-chip yellow dqm-smaev-patwarn", id: `dqm-smaev-patwarn-${k}`,
                           role: "note" });
    w.style.display = "none";
    row.appendChild(w);
  }
  return row;
}

function legend() {
  const stops = [0, 50, 100, 150, 200, 249].map((t) => `${totColour(t)} ${Math.round(t / 249 * 100)}%`);
  const ramp = el("span", { class: "dqm-sma-ramp" });
  ramp.style.background = `linear-gradient(to right, ${stops.join(", ")})`;
  const corrupt = el("span", { class: "dqm-sma-badge" }, "≥ 250");
  corrupt.style.background = "#c0c";
  corrupt.style.color = "#fff";
  const pstops = [0, 8, 16, 24, 31].map((t) => `${pixColour(t)} ${Math.round(t / 31 * 100)}%`);
  const pramp = el("span", { class: "dqm-sma-ramp dqm-sma-ramp-short" });
  pramp.style.background = `linear-gradient(to right, ${pstops.join(", ")})`;
  return el("div", { class: "dqm-sma-legend" }, el("span", {}, "ToT code  0"), ramp,
            el("span", {}, "249"), corrupt,
            el("span", { class: "dqm-sma-legend-gap" }, "MuPix ToT  0"), pramp,
            el("span", {}, "31 (× 256 ns)"));
}

function showTab(tab) {
  state.tab = tab;
  save();
  for (const [id] of TABS) {
    const pane = document.getElementById(`dqm-smaev-pane-${id}`);
    if (pane) pane.style.display = id === tab ? "" : "none";
  }
  markTabs(TABS, "dqm-smaev", tab);
  const holder = document.getElementById("dqm-smaev-rate");
  holder.innerHTML = "";
  holder.appendChild(labelled("update", select(RATES[tab], String(state.intervals[tab]),
    function (v) {
      state.intervals[tab] = Number(v);
      save();
      if (state.updater) state.updater.setInterval(Number(v));
    })));
  // The frame already held for this tab is shown at once (a frozen frame
  // stays visible across tab switches).
  singleNote("");
  if (state.frames[tab]) render(tab);
  else renderStatus(tab, null);
  startPolling();
  if (state.paused) { ensureWords(tab); keepRaw(tab); }
}

/**
 * Freeze or resume. Freezing asks once for the frozen frame's words (a polled
 * raster has none) and keeps its raw event while the analyzer still holds it
 * -- unless `opts.words` is false: Single does both for the frame it fetches.
 */
function setPaused(paused, opts) {
  state.paused = paused;
  const b = document.getElementById("dqm-smaev-freeze");
  b.textContent = paused ? "Frozen — resume" : "Freeze";
  b.className = paused ? "mbutton dqm-sma-frozen" : "mbutton";
  if (paused) { retire(state.updater); state.updater = null; }
  else { singleNote(""); startPolling(); }
  renderStatus(state.tab, state.frames[state.tab]);
  if (state.frames[state.tab]) renderHints(state.tab);      // the hints mention freezing
  if (paused && !(opts && opts.words === false)) { ensureWords(state.tab); keepRaw(state.tab); }
}

// ---------------------------------------------------------------------------

function ms(ns) {
  if (ns === null || ns === undefined || !Number.isFinite(ns)) return "—";
  return (ns / 1e6).toFixed(3);
}

function fmt(x, d) {
  return x === null || x === undefined || !Number.isFinite(x) ? "—" : x.toFixed(d);
}

function num(x) {
  return x === null || x === undefined ? "—" : Number(x).toLocaleString();
}

/**
 * A tab strip (role=tablist) of one role=tab button per [id, label] in `tabs`,
 * with ids `${prefix}-tab-${id}` controlling the panes `${prefix}-pane-${id}`.
 * Left/Right (wrapping), Home and End select and focus the neighbouring tab.
 * `current()` says which tab is shown; `select(id)` shows one. markTabs()
 * then sets the selection state, so the strip always follows showTab().
 */
function tabStrip(tabs, prefix, label, current, select) {
  const bar = el("div", { class: "dqm-sma-tablist", role: "tablist", "aria-label": label });
  for (const [id, text] of tabs) {
    const b = el("button", {
      type: "button", class: "dqm-sma-tab", id: `${prefix}-tab-${id}`, role: "tab",
      "aria-selected": "false", "aria-controls": `${prefix}-pane-${id}`, tabindex: "-1",
    }, text);
    b.onclick = function () { select(id); };
    bar.appendChild(b);
  }
  bar.addEventListener("keydown", function (ev) {
    const n = tabs.length;
    const i = tabs.findIndex(([id]) => id === current());
    const to = { ArrowRight: (i + 1) % n, ArrowLeft: (i - 1 + n) % n, Home: 0, End: n - 1 }[ev.key];
    if (to === undefined) return;
    if (ev.preventDefault) ev.preventDefault();
    const id = tabs[to][0];
    if (id !== current()) select(id);
    const b = document.getElementById(`${prefix}-tab-${id}`);
    if (b) b.focus();
  });
  return bar;
}

/** The selection state of a tabStrip(): only the shown tab is selected and in the tab order. */
function markTabs(tabs, prefix, shown) {
  for (const [id] of tabs) {
    const b = document.getElementById(`${prefix}-tab-${id}`);
    if (!b) continue;
    const on = id === shown;
    b.setAttribute("aria-selected", on ? "true" : "false");
    b.setAttribute("tabindex", on ? "0" : "-1");
    b.className = on ? "dqm-sma-tab active" : "dqm-sma-tab";
  }
}

function save() {
  try {
    window.localStorage.setItem(LS, JSON.stringify({
      client: state.client, tab: state.tab, zoom: state.zoom,
      hideCurrent: state.hideCurrent, hidePixels: state.hidePixels, intervals: state.intervals,
      rawTimes: state.rawTimes,
    }));
  } catch (e) { /* private browsing or quota */ }
}

function restore() {
  try {
    const o = JSON.parse(window.localStorage.getItem(LS) || "{}");
    if (o.client) state.client = o.client;
    if (o.tab && TABS.some(([id]) => id === o.tab)) state.tab = o.tab;
    if (o.zoom === "full" || o.zoom === "prompt") state.zoom = o.zoom;
    if (o.hideCurrent !== undefined) state.hideCurrent = !!o.hideCurrent;
    if (o.hidePixels !== undefined) state.hidePixels = !!o.hidePixels;
    if (o.rawTimes !== undefined) state.rawTimes = !!o.rawTimes;
    if (o.intervals) {
      for (const tab of ["seeded", "raster"]) {
        const v = Number(o.intervals[tab]);
        if (RATES[tab].some(([ms_]) => Number(ms_) === v)) state.intervals[tab] = v;
      }
    }
  } catch (e) { /* defaults are fine */ }
}

function el(tag, attrs, ...children) {
  const e = document.createElement(tag);
  Object.keys(attrs || {}).forEach(function (k) {
    if (k === "class") e.className = attrs[k]; else e.setAttribute(k, attrs[k]);
  });
  children.forEach(function (c) {
    if (c === null || c === undefined) return;
    e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  });
  return e;
}

function select(options, current, onChange) {
  const s = el("select", {});
  options.forEach(function (o) {
    const opt = el("option", { value: o[0] }, o[1]);
    if (o[0] === current) opt.setAttribute("selected", "selected");
    s.appendChild(opt);
  });
  s.value = current;
  s.onchange = function () { onChange(this.value); };
  return s;
}

function labelled(text, node) {
  return el("span", { class: "dqm-chip" }, el("span", {}, text), node);
}

})();
