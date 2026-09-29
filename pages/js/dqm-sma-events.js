//
// dqm-sma-events.js -- the SMA event display (/Custom/SMAEvents).
//
// Two views of the analyzer's latest frame, from sma::frame:
//
//   Seeded  up to four S1 hits of the latest good frame, each with every
//           channel that has a role in [-200 ns, +3 us] around it: one lane per
//           channel, each hit a bar from t to t+ToT. This is where a missing
//           counter, an RF burst in the wrong place, or a fine/coarse-corrupt
//           word is seen hit by hit.
//   Raster  the whole latest frame, time vs channel, ToT as colour. This is
//           where a stale buffer, a dead channel or a frame-sized hole is seen.
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
// A raster with words is 19 bytes a hit, ~1.2 MB at MAX_HITS: ask for that
// once rather than the 512 kB poll default and a retry.
const WORDS_MAX_REPLY = 2 * 1024 * 1024;
const HIT_SLOP_PX = 4;           // how far from a hit the mouse may be and still point at it
const COPIED_MS = 1500;          // how long a copy button says "Copied"
const RAW_KEEP = 2;              // frozen frames whose raw event the page keeps (one per tab)
const MAX_TAGS = 200;            // tagged hits the list holds
const SS_TAGS = "dqm-sma-events-tagged";   // sessionStorage: the list survives a reload
const TAG_MARK = "#000";         // a tagged hit's outline and number box

const LANE_COLOURS = {
  s1: "#1f77b4", counter: ["#1f77b4", "#2ca02c", "#17becf", "#9467bd", "#8c564b", "#bcbd22"],
  rf: "#ff7f0e", current: "#7f7f7f", delayed: "#e377c2",
};

const state = {
  client: "sma_analyzer",
  tab: "seeded",
  paused: false,
  zoom: "full",
  hideCurrent: true,
  intervals: { seeded: 250, raster: 500 },
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
  }
  // max_hits: a bound on the reply, so a pathological frame is truncated (and
  // says so in the header) rather than growing the reply without limit. Real
  // 40k-word frames are ~33k hits.
  const args = Object.assign(tab === "raster" ? { view: "raster", drop: rasterDrop(), max_hits: MAX_HITS }
                                             : { view: "seeded" }, extra || {});
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

function rasterDrop() {
  const cur = roles().current;
  return state.hideCurrent && cur !== null ? [cur] : [];
}

// ---------------------------------------------------------------------------
// Roles and lanes
// ---------------------------------------------------------------------------

/**
 * Channel roles, from the summary (i.e. the ODB), else from the frame's labels.
 *
 * The fallback reads the labels the analyzer put in the frame (S1..S5, RF,
 * current are its defaults), so a page whose summary call failed still draws
 * sensible lanes rather than none.
 */
function roles() {
  const out = { counters: [], rf: null, current: null, delayed: [], labels: [] };
  const s = state.summary;
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
    return out;
  }
  const f = state.frames[state.tab] || state.frames.seeded || state.frames.raster;
  const labels = (f && f.meta && f.meta.labels) || [];
  out.labels = labels;
  const byS = [];
  labels.forEach(function (l, ch) {
    const m = /^S(\d)$/.exec(l || "");
    if (m) byS.push([Number(m[1]), ch]);
    else if (l === "RF") out.rf = ch;
    else if (l === "current") out.current = ch;
  });
  out.counters = byS.sort((a, b) => a[0] - b[0]).map((x) => x[1]);
  if (!labels.length) { out.counters = [1, 2, 3, 4, 5]; out.rf = 6; out.current = 7; }
  return out;
}

function labelOf(ch, r) {
  const l = (r || roles()).labels[ch];
  return l || `ch${String(ch).padStart(2, "0")}`;
}

/** Seeded-view lanes: counters in order, RF, delayed channels, then current. */
function lanes() {
  const r = roles();
  const out = [];
  r.counters.forEach(function (ch, k) {
    out.push({ ch, label: labelOf(ch, r), colour: LANE_COLOURS.counter[k % 6] });
  });
  if (r.rf !== null) out.push({ ch: r.rf, label: labelOf(r.rf, r), colour: LANE_COLOURS.rf });
  for (const ch of r.delayed) out.push({ ch, label: labelOf(ch, r), colour: LANE_COLOURS.delayed });
  if (r.current !== null) {
    out.push({ ch: r.current, label: labelOf(r.current, r), colour: LANE_COLOURS.current });
  }
  return out;
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
    `${num(m.n_pixel)} pixel · ${num(m.n_trigger)} trigger (${num(m.n_kept)} kept)`));
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
    noFrameNote("seeded", `${state.client} has not seen a good SMA frame yet. The seeded view ` +
      "shows only good frames (a stale buffer has no seeds worth showing).");
    return;
  }
  clearNote("seeded");
  guard(head, () => frameHeader(frame, head));

  const seeds = (frame.meta && frame.meta.seeds) || [];
  while (state.seedPanels.length > seeds.length) state.seedPanels.pop().wrap.remove();
  while (state.seedPanels.length < seeds.length) state.seedPanels.push(makeSeedPanel(holder));
  if (!seeds.length) {
    noFrameNote("seeded", "This frame has no S1 hit whose whole window lies inside it.");
  }
  const ls = lanes();
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

function drawSeed(frame, seed, k, p, ls, nums) {
  const [a, b] = seedHits(frame, seed);
  let nMis = 0, nTot = 0, nOther = 0;
  const laneOf = {};
  ls.forEach((l, i) => { laneOf[l.ch] = i; });
  for (let i = a; i < b; i++) {
    if (frame.hitFlags[i] & SMAF.HIT.MISMATCH) nMis++;
    if (frame.hitFlags[i] & SMAF.HIT.TOT_CORRUPT) nTot++;
    if (laneOf[frame.ch[i]] === undefined) nOther++;
  }

  // Header: DOM, not canvas, so it can be read, selected and copied.
  const head = p.head;
  head.innerHTML = "";
  p.geom = null;
  head.appendChild(el("span", { class: "dqm-sma-seedtitle" },
    `seed ${k + 1}: S1 at ${ms(frameOffsetNs(frame) + seed.t_rel)} ms in the frame, ToT ${seed.s1_tot}`));
  const pat = el("span", { class: "dqm-sma-pattern", title: "coincidence pattern" });
  const nCounters = Math.max(roles().counters.length, 5);
  for (let c = 0; c < nCounters; c++) {
    const lit = (seed.pattern >> c) & 1;
    pat.appendChild(el("span", { class: lit ? "dqm-sma-pbox lit" : "dqm-sma-pbox" }, `S${c + 1}`));
  }
  head.appendChild(pat);
  head.appendChild(badge(rfText(seed), seed.rf_vetoed ? "yellow" : (seed.rf_valid ? "blue" : "")));
  head.appendChild(badge(`${b - a} hits`));
  if (nMis) head.appendChild(badge(`${nMis} fine/coarse mismatch`, "red"));
  if (nTot) head.appendChild(badge(`${nTot} ToT ≥ ${frame.meta.tot_corrupt || 250}`, "yellow"));
  if (nOther) head.appendChild(badge(`${nOther} on channels without a role`));
  const wr = seed.word_range;
  if (wr) head.appendChild(badge(`words ${wr[0]}–${wr[1]}`));

  p.copyText = seedText(frame, seed, k, a, b);
  p.copy.disabled = !frame.wordIndex;
  if (!frame.wordIndex) p.copy.title = "This analyzer sends no word data";

  const win = (frame.meta && frame.meta.window) || { pre_ns: 200, post_ns: 3000 };
  const range = state.zoom === "prompt" ? [-PROMPT_NS, PROMPT_NS] : [-win.pre_ns, win.post_ns];
  const g = paintSeed(p.canvas, frame, seed, a, b, ls, laneOf, range, nums || tagNumbers("seeded", frame));
  p.geom = Object.assign(g, { frame, seed, a, b, ls, range });
}

/**
 * What "Copy seed words" copies: the frame tag, the seed's S1 word, the bank
 * word range of its whole window and the word index of every hit shown in it
 * (ascending, the order they sit in the file), on one line for the elog.
 */
function seedText(frame, seed, k, a, b) {
  const wr = seed.word_range;
  const s1 = seed.s1_word === null || seed.s1_word === undefined ? "—" : seed.s1_word;
  const parts = [tagOf(frame), `seed ${k + 1}: S1 word ${s1}`,
                 `words ${wr ? `${wr[0]}–${wr[1]}` : "—"} (${b - a} hits in window)`];
  if (frame.wordIndex) {
    const w = Array.from(frame.wordIndex.subarray(a, b)).sort((x, y) => x - y);
    parts.push(`hit words ${w.join(", ")}`);
  }
  return parts.join(" · ");
}

function rfText(seed) {
  const parts = ["RF"];
  if (seed.rf_valid) {
    parts.push(`phase ${fmt(seed.rf_phase, 1)} ns`);
    parts.push(`period ${fmt(seed.rf_period, 1)} ns`);
  }
  parts.push(`${seed.rf_n} pulse${seed.rf_n === 1 ? "" : "s"}`);
  parts.push(seed.rf_valid ? "valid" : "not valid");
  if (seed.rf_vetoed) parts.push("vetoed");
  return parts.join(" · ");
}

const LANE_H = 20;
const MARGIN = { left: 84, right: 12, top: 6, bottom: 34 };

function paintSeed(canvas, frame, seed, a, b, ls, laneOf, range, nums) {
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
    ctx.fillText(`${l.label} (${l.ch})`, x0 - 6, y + LANE_H / 2);
  });
  timeAxis(ctx, range, X, MARGIN.top + ls.length * LANE_H, "ns from S1");

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
    const x = X(seed.rf_phase);
    const y = MARGIN.top + rfLane * LANE_H;
    ctx.strokeStyle = "#000";
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y + LANE_H); ctx.stroke();
  }

  ctx.save();
  ctx.beginPath();
  ctx.rect(x0, MARGIN.top, x1 - x0, ls.length * LANE_H);
  ctx.clip();
  const marks = [];
  for (let i = a; i < b; i++) {
    const lane = laneOf[frame.ch[i]];
    if (lane === undefined) continue;
    const t = frame.t[i] - seed.t_rel;
    const tot = frame.tot[i];
    if (t + tot < range[0] || t > range[1]) continue;
    const flags = frame.hitFlags[i];
    const px0 = X(t);
    const pw = Math.max(2, X(t + tot) - px0);
    const y = MARGIN.top + lane * LANE_H + 3;
    const h = LANE_H - 6;
    ctx.fillStyle = ls[lane].colour;
    ctx.fillRect(px0, y, pw, h);
    if (flags & SMAF.HIT.MISMATCH) {
      // Hatched with a red outline: the time of this hit cannot be trusted,
      // so it must not look like the solid bars around it.
      hatch(ctx, px0, y, pw, h);
      ctx.strokeStyle = "#d00";
      ctx.lineWidth = 2;
      ctx.strokeRect(px0, y, pw, h);
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
    if (n !== undefined) marks.push([px0, y, pw, h, n]);
  }
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
  return { x0, x1, laneOf };
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
  let best = -1, bestD = HIT_SLOP_PX;
  for (let i = g.a; i < g.b; i++) {
    if (f.ch[i] !== ch) continue;
    const t = f.t[i] - g.seed.t_rel;
    const px0 = X(t);
    const pw = Math.max(2, X(t + f.tot[i]) - px0);
    const d = x < px0 ? px0 - x : x > px0 + pw ? x - px0 - pw : 0;
    if (d <= bestD) { best = i; bestD = d; }        // a tie goes to the later bar, drawn on top
  }
  return best;
}

function hatch(ctx, x, y, w, h) {
  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, w, h);
  ctx.clip();
  ctx.strokeStyle = "rgba(255,255,255,0.9)";
  ctx.beginPath();
  for (let d = -h; d < w; d += 5) { ctx.moveTo(x + d, y + h); ctx.lineTo(x + d + h, y); }
  ctx.stroke();
  ctx.restore();
}

// -- raster --------------------------------------------------------------------

const ROW_H = 22;
const RASTER_MARGIN = { left: 84, right: 110, top: 6, bottom: 34 };

function renderRaster(frame) {
  const head = document.getElementById("dqm-smaev-rasterhead");
  const holder = document.getElementById("dqm-smaev-rasterbox");
  if (!frame) {
    head.innerHTML = "";
    noFrameNote("raster", `${state.client} has not seen an SMA frame yet.`);
    return;
  }
  clearNote("raster");
  guard(head, () => frameHeader(frame, head));
  guard(holder, () => paintRaster(frame));
  renderHints("raster");
}

function totColour(tot) {
  if (tot >= 250) return "#c0c";
  const v = Math.min(1, tot / 249);
  return `hsl(${Math.round((1 - v) * 240)}, 90%, 45%)`;
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
  const M = RASTER_MARGIN;
  const H = M.top + nCh * ROW_H + M.bottom;
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
  timeAxis(ctx, range, X, M.top + nCh * ROW_H, "ms from the frame's first kept hit");

  // One fill per colour bucket, not per hit: 33k fillStyle changes is what
  // makes a canvas raster slow, 33k rects in 33 paths is not.
  const BUCKETS = 32;
  const paths = new Array(BUCKETS + 1);
  const scale = 1e6;
  const off = frameOffsetNs(frame);
  for (let i = 0; i < frame.nHits; i++) {
    const t = (off + frame.t[i]) / scale;
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

  const nums = tagNumbers("raster", frame);
  for (let i = 0; nums.any && i < frame.nHits; i++) {
    const n = nums.of(i);
    if (n === undefined) continue;
    const t = (off + frame.t[i]) / scale;
    const c = frame.ch[i];
    if (t < range[0] || t > range[1] || c >= nCh) continue;
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
    ctx.fillRect(a, M.top, b - a, nCh * ROW_H);
  }
  state.rasterGeom = { x0, x1, range };
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
  if (c < 0 || c >= 16) return -1;
  const off = frameOffsetNs(frame);
  const k = (g.x1 - g.x0) / (g.range[1] - g.range[0]);
  let best = -1, bestD = HIT_SLOP_PX;
  for (let i = 0; i < frame.nHits; i++) {
    if (frame.ch[i] !== c) continue;
    const t = (off + frame.t[i]) / 1e6;
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
  const c = frame.ch[i];
  const l = roles().labels[c];
  const named = l && !/^ch\d+$/.test(l);
  const m = frame.meta || {};
  const tRel = frame.t[i];
  const parts = [`ch ${c}${named ? ` (${l})` : ""}`, `ToT ${frame.tot[i]}`,
    Number.isFinite(m.t0_ns) ? `t ${m.t0_ns + tRel} ns (t_rel ${tRel} ns)` : `t_rel ${tRel} ns`,
    `fine/coarse ${frame.hitFlags[i] & SMAF.HIT.MISMATCH ? "MISMATCH" : "ok"}`,
    wordText(frame, i, tab)];
  return parts.join(" · ");
}

function wordText(frame, i, tab) {
  if (frame.rawWord && frame.wordIndex) {
    return `word ${frame.wordIndex[i]} · ${SMAF.hex64(frame.rawWord[i])}`;
  }
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
  return frame.wordIndex ? `${fk}|w${frame.wordIndex[i]}` : `${fk}|${tab}|${dropKey(frame)}|${i}`;
}

/**
 * A tagged hit, as plain values (it is kept in sessionStorage): the frame's
 * whole identity at click time, the hit's fields and its line as shown.
 */
function makeTag(tab, frame, i) {
  const m = frame.meta || {};
  const ev = m.event || {};
  const t0 = Number.isFinite(m.t0_ns) ? m.t0_ns : null;
  return {
    key: hitKey(frame, tab, i), frame: frameKey(frame), tag: tagOf(frame),
    run: frame.run, eventId: ev.id === undefined ? null : ev.id, serial: serialOf(frame),
    utc: Number.isFinite(ev.timestamp) ? new Date(ev.timestamp * 1000).toISOString() : null,
    seq: frame.frameSeq, view: tab, drop: dropKey(frame), i,
    word: frame.wordIndex ? frame.wordIndex[i] : null,
    raw: frame.rawWord ? SMAF.hex64(frame.rawWord[i]) : null,
    ch: frame.ch[i], tot: frame.tot[i], tRel: frame.t[i],
    tAbs: t0 === null ? null : t0 + frame.t[i],
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
    if (f && f.frameSeq === x.seq && x.i < f.nHits && !addTag("raster", f, x.i)) full = true;
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
    if (x.frame !== fk || x.word !== null || x.view !== tab || x.drop !== dk || x.i >= frame.nHits) {
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
        else if (x.view === tab && x.drop === dk && x.i < frame.nHits) byIndex.set(x.i, r.n);
      }
    }
  }
  const wi = frame && frame.wordIndex;
  return {
    any: byWord.size + byIndex.size > 0,
    of: (i) => (wi && byWord.has(wi[i]) ? byWord.get(wi[i]) : byIndex.get(i)),
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
    : { view: "seeded", words: true, seq: f.frameSeq };
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
// Chrome
// ---------------------------------------------------------------------------

function build() {
  const r = document.getElementById("dqm-root");
  r.innerHTML = "";

  const bar = el("div", { class: "dqm-strip" });
  for (const [id, label] of TABS) {
    const b = el("button", { class: "mbutton dqm-sma-tab", id: `dqm-smaev-tab-${id}` }, label);
    b.onclick = function () { showTab(id); };
    bar.appendChild(b);
  }
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

  // Seeded pane
  const seeded = el("div", { id: "dqm-smaev-pane-seeded" });
  const sbar = el("div", { class: "dqm-strip" });
  sbar.appendChild(labelled("window", select(
    [["full", "full (-200 ns .. +3 us)"], ["prompt", "prompt (±150 ns)"]], state.zoom,
    function (v) { state.zoom = v; save(); if (state.frames.seeded) render("seeded"); })));
  sbar.appendChild(el("span", { class: "dqm-footnote" },
    "bars: t to t+ToT · hatched red: fine/coarse mismatch · magenta ▼: ToT ≥ 250 · " +
    "dashed red: the S1 seed · black tick on RF: the pulse the phase is taken from · " +
    "black box + number: a tagged hit"));
  seeded.appendChild(sbar);
  seeded.appendChild(el("div", { id: "dqm-smaev-note-seeded" }));
  seeded.appendChild(makeTagBar("seeded"));
  seeded.appendChild(el("div", { class: "dqm-strip dqm-sma-header", id: "dqm-smaev-seededhead" }));
  seeded.appendChild(el("div", { id: "dqm-smaev-seeds" }));
  r.appendChild(seeded);

  // Raster pane
  const raster = el("div", { id: "dqm-smaev-pane-raster" });
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
  const lab = el("label", { for: "dqm-smaev-hidecur", class: "dqm-chip" }, "hide the current channel");
  lab.insertBefore(box, lab.firstChild);
  rbar.appendChild(lab);
  rbar.appendChild(el("span", { class: "dqm-footnote" },
    "drag to zoom in time, double-click for the whole frame, click a hit to tag it " +
    "(a live frame is frozen first, for its word data)"));
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

function legend() {
  const stops = [0, 50, 100, 150, 200, 249].map((t) => `${totColour(t)} ${Math.round(t / 249 * 100)}%`);
  const ramp = el("span", { class: "dqm-sma-ramp" });
  ramp.style.background = `linear-gradient(to right, ${stops.join(", ")})`;
  const corrupt = el("span", { class: "dqm-sma-badge" }, "≥ 250");
  corrupt.style.background = "#c0c";
  corrupt.style.color = "#fff";
  return el("div", { class: "dqm-sma-legend" }, el("span", {}, "ToT code  0"), ramp,
            el("span", {}, "249"), corrupt);
}

function showTab(tab) {
  state.tab = tab;
  save();
  for (const [id] of TABS) {
    const b = document.getElementById(`dqm-smaev-tab-${id}`);
    const pane = document.getElementById(`dqm-smaev-pane-${id}`);
    if (b) b.className = id === tab ? "mbutton dqm-sma-tab active" : "mbutton dqm-sma-tab";
    if (pane) pane.style.display = id === tab ? "" : "none";
  }
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

function save() {
  try {
    window.localStorage.setItem(LS, JSON.stringify({
      client: state.client, tab: state.tab, zoom: state.zoom,
      hideCurrent: state.hideCurrent, intervals: state.intervals,
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
