//
// dqm-sma.js -- the SMA accumulated-plots page (/Custom/SMAPlots).
//
// Top to bottom, in the order a shifter needs it:
//
//   1. a status line: is the analyzer there, is it keeping up, which run, and
//      what share of the frames it analyses (it samples to stay within a CPU
//      budget, which is information, not a fault);
//   2. the flags, worst first -- and a coarse-shift mismatch as a banner that
//      cannot be scrolled past, because every fine/coarse number on the page is
//      wrong until it is fixed;
//   3. a per-channel table (rates, ToT >= 250, fine/coarse mismatch, stale,
//      S1-conditional efficiency), with the flagged cells coloured;
//   4. tabs of histograms, and the 10-minute trends. Two MuPix tabs:
//      "MuPix phase space" (reco frame, +x beam-left): a one-line note (track
//      fractions, ToT cuts, stage shift, geometry, chips as drawn), the in-time
//      hit maps of L1 and L2, the S1-seeded tracks x/y, x/x', y/y' -- each a
//      row of all / light / heavy -- their ToT map and states. "MuPix
//      diagnostics": the in-time fractions of S1 hits (L1, L2, L1+L2, with the
//      sideband and the accidentals taken out) and the time-sync state above
//      the timing, ToT and occupancy plots; the per-chip column/row occupancy
//      is drawn per plane, one line per chip.
//      "MuPix pairs (unseeded)": every L1 pixel paired with the nearest-in-time
//      L2 pixel within the window, with or without an S1 hit, as the nearline
//      pairs them: a note (paired fraction, partners, window, cap, cuts, stage),
//      a row "L1 alone | L2 alone" (every pixel of each plane, unseeded), then
//      x/y, x/x', y/y' as rows of all / light / heavy, the L2 - L1 dt with the
//      window outlined, and the partners per L1 pixel.
//      The NIM / TOT tab (since run 1015 each counter has a NIM copy, "S1L"
//      ...) has a table per counter -- pair efficiency, purity, NIM-only share,
//      median NIM - TOT, the lag-fault vote -- above its plots, per counter.
//      A chip in the status line says whether the counters are the merged
//      TOT + NIM hits (pattern, efficiencies), as nearline's are.
//
// The same script draws the MuPix page (/Custom/MuPixPlots, mupix.html): its
// <body data-view="mupix"> picks a VIEWS entry below, which keeps only the MuPix
// tabs, the status chips, the banners and the MuPix flags, and remembers its
// settings under its own localStorage key.
//
// Everything comes from the sma_analyzer client over brpc. Channel names come
// from the summary, which takes them from the ODB (/DQM/SMA/Channel roles), so
// a relabelled or moved counter needs no edit here. Histogram names come from
// dqm::list, so a new histogram in the plugin appears on the page (in the
// Health tab unless tabOf() below says otherwise) without touching this file.
//
// Only the visible tab is polled, and on it only the plots on screen (after
// each has been drawn once). A hidden tab of forty histograms at 1 Hz would be
// forty brpc round trips a second through mhttpd for plots nobody is looking at.
//
// 1D histograms are drawn with mplot; 2D ones with DQMHeatmap (dqm-heatmap.js),
// which draws the bins as one image instead of mplot's fillRect per bin. Either
// is redrawn only when its histogram changed.
//

(function () {
"use strict";

const LS = "dqm-sma-settings";

const TABS = [
  ["health", "Health"],
  ["tot", "ToT / corruption"],
  ["timing", "Timing"],
  ["rf", "RF / delayed"],
  // The phase space first: it is what a shifter watches. The diagnostics tab
  // keeps the old tab's id, so a page that remembered "mupix" opens it.
  // [id, label, short label for a phone-width tab bar]
  ["mupixxy", "MuPix phase space", "MuPix x/y"],
  ["mupixpair", "MuPix pairs (unseeded)", "MuPix pairs"],
  ["mupix", "MuPix diagnostics", "MuPix diag."],
  ["nim", "NIM / TOT"],
  ["trends", "Trends"],
];

/**
 * The pages this script draws, chosen by <body data-view="...">; none is "sma".
 *
 * tabs   the tab ids shown, in TABS order (null: all); the first is the default.
 * store  the localStorage key of the page's settings (tab, scales, rate), so
 *        two pages open side by side do not overwrite each other's tab.
 * page   the menu key for mhttpd_init when the URL has no ?page=.
 * flags  which flags are listed (null: all); the banners are always shown,
 *        since a wrong coarse shift or time base spoils the S1-seeded maps too.
 * table  whether the per-channel table is drawn.
 */
const VIEWS = {
  sma: { tabs: null, store: LS, page: "SMAPlots", flags: null, table: true },
  mupix: { tabs: ["mupixxy", "mupixpair", "mupix"], store: "dqm-sma-mupix-settings", page: "MuPixPlots",
           // The MuPix flags, and those that leave the MuPix plots empty or wrong:
           // no frames, every frame stale, no S1 seeds, a settings error (a bad
           // chip list turns x/y off with one); and the zero frames dropped,
           // which would otherwise be one bright L1 pixel.
           flags: /^(mupix_\w+|settings|no_frames|all_stale|no_seeds|zero_frames)$/, table: false },
};

/** The view named by the page's <body data-view>, else the SMAPlots page. */
function viewOf(doc) {
  const b = doc && doc.body;
  const name = b && typeof b.getAttribute === "function" ? b.getAttribute("data-view") : null;
  return VIEWS[name] || VIEWS.sma;
}

/** The tabs of a view, in TABS order. */
function tabsOf(view) {
  return view.tabs ? TABS.filter(([id]) => view.tabs.indexOf(id) >= 0) : TABS;
}

/**
 * Which tab a histogram belongs on, and in what order.
 *
 * Ordered patterns rather than a name list: the per-channel names (tot_chNN,
 * delayed_dt_chNN, dt_Sk_S1) follow the ODB roles, so they cannot be listed
 * here. Anything unmatched lands at the end of Health rather than nowhere.
 */
const ORDER = {
  health: [/^word_types$/, /^frame_class$/, /^words_per_frame$/, /^trigger_words_per_frame$/,
           /^frame_span/, /^frame_gap_ms$/, /^live_fraction$/, /^rate_vs_ch$/,
           /^words_per_ch$/, /^stale_per_ch$/, /^s1_best_frac$/],
  tot: [/^shift_check$/, /^fine_vs_coarse$/, /^fine_coarse_diff$/, /^fine_bit_occupancy$/,
        /^tot_ge250_per_ch$/, /^tot_vs_ch_lsb/, /^tot_ch\d+/],
  timing: [/^pattern$/, /^s1_coinc$/, /^dt_S\d+_S1$/, /^s1_partner_hits$/, /^s1_spacing_us$/],
  rf: [/^rf_npulses$/, /^rf_phase_s1$/, /^rf_period$/, /^rf_phase_vs_s1_tot$/, /^delayed_dt_ch/],
  // MuPix x/y: the hit maps, then the S1-seeded tracks (all / light / heavy
  // side by side, one row each for x/y, x/x', y/y'), their ToT map and states.
  mupixxy: [/^mupix_hits_xy_L1$/, /^mupix_hits_xy_L2$/,
            /^mupix_track_xy$/, /^mupix_track_xy_light$/, /^mupix_track_xy_heavy$/,
            /^mupix_track_xxp$/, /^mupix_track_xxp_light$/, /^mupix_track_xxp_heavy$/,
            /^mupix_track_yyp$/, /^mupix_track_yyp_light$/, /^mupix_track_yyp_heavy$/,
            /^mupix_track_tot$/, /^mupix_track_state$/, /^mupix_hits_xy_/, /^mupix_track_/],
  // MuPix pairs: each plane alone (L1 | L2), then all / light / heavy for x/y,
  // x/x', y/y', then the L2 - L1 dt and the partners per L1 pixel. Checked
  // before the diagnostics' /^mupix_/.
  mupixpair: [/^mupix_pair_hits_xy_L1$/, /^mupix_pair_hits_xy_L2$/, /^mupix_pair_xy$/, /^mupix_pair_xy_light$/, /^mupix_pair_xy_heavy$/,
              /^mupix_pair_xxp$/, /^mupix_pair_xxp_light$/, /^mupix_pair_xxp_heavy$/,
              /^mupix_pair_yyp$/, /^mupix_pair_yyp_light$/, /^mupix_pair_yyp_heavy$/,
              /^mupix_pair_dt$/, /^mupix_pair_partners$/, /^mupix_pair_/],
  // Every other mupix_ histogram is a diagnostic.
  mupix: [/^mupix_dt_L1$/, /^mupix_dt_L2$/, /^mupix_dt_chip$/, /^mupix_s1_match$/, /^mupix_tot_L1$/, /^mupix_tot_L2$/,
          /^mupix_col_chip/, /^mupix_row_chip/, /^mupix_hits_chip$/, /^mupix_/],
  nim: [/^nim_/, /^s1_coinc_tot$/],
};

/**
 * The NIM tab's order: the merged-vs-TOT coincidences first, then counter by
 * counter (S1, S1L's plots; S2, ...), and within a counter the kinds below --
 * so "is S4L all right" is one block of plots, read top to bottom.
 */
const NIM_KINDS = ["dt", "dt_wide", "classes", "walk", "lag", "width", "candidates"];

/** 1D plots drawn with log y by default (the NIM tab's "log y for Δt" switch). */
const LOG_Y_DEFAULT = [/^sma\/nim_dt_S\d+$/, /^sma\/nim_dt_wide_S\d+$/];

/**
 * The MuPix x/y track maps: linear z by default (the phase-space tab's own switch),
 * so the beam spot and the x/x' band read as they are; the other maps, the hit
 * maps among them, follow the toolbar's log z.
 */
const XY_TRACK_MAP = /^sma\/mupix_track_(xy|xxp|yyp)(_light|_heavy)?$/;

/** The MuPix x/y histograms: the note on them goes in front of the first. */
const XY_ANY = /^sma\/mupix_(hits_xy_L\d|track_\w+)$/;

/** The unseeded pair maps: linear z by default too, with the pairs tab's own switch. */
const PAIR_MAP = /^sma\/mupix_pair_(xy|xxp|yyp)(_light|_heavy)?$/;

/** The pairs tab's histograms: its note goes in front of the first. */
const PAIR_ANY = /^sma\/mupix_pair_\w+$/;

const PAIR_DT = "sma/mupix_pair_dt";

/** t(pixel) - t(S1) by chip (chip x dt): one line per chip, the chips chosen by checkboxes. */
const CHIP_DT = "sma/mupix_dt_chip";

/**
 * Plots laid out side by side in a row of their own, `cols` to a row on a wide
 * screen, one below the other on a phone: the two hit maps; all / light /
 * heavy of each track map. `square`: the plot box is sized so the map's frame
 * is about square (x and y in mm on the same scale).
 */
function rowOf(name) {
  const s = shortName(name);
  if (/^mupix_hits_xy_L\d$/.test(s)) return { key: "xyhits", cols: 2, square: true };
  if (/^mupix_pair_hits_xy_L\d$/.test(s)) return { key: "pairhits", cols: 2, square: true };
  const m = /^mupix_track_(xy|xxp|yyp)(_light|_heavy)?$/.exec(s);
  if (m) return { key: `xy${m[1]}`, cols: 3, square: true };
  const q = /^mupix_pair_(xy|xxp|yyp)(_light|_heavy)?$/.exec(s);
  if (q) return { key: `pair${q[1]}`, cols: 3, square: true };
  return null;
}

/** Plots outside a row that are drawn about square anyway. */
const SQUARE = [/^sma\/mupix_track_tot$/];

function tabOf(name) {
  const s = shortName(name);
  for (const tab of ["tot", "timing", "rf", "mupixxy", "mupixpair", "mupix", "nim"]) {
    if (ORDER[tab].some((re) => re.test(s))) return tab;
  }
  return "health";
}

function rank(tab, name) {
  const s = shortName(name);
  if (tab === "nim") {
    if (s === "s1_coinc_tot") return 0;
    const m = /^nim_(.+)_S(\d+)$/.exec(s);
    const kind = m ? NIM_KINDS.indexOf(m[1]) : -1;
    if (!m) return 10000;
    return 1 + 100 * Number(m[2]) + (kind < 0 ? NIM_KINDS.length : kind);
  }
  const i = ORDER[tab].findIndex((re) => re.test(s));
  return i < 0 ? ORDER[tab].length : i;
}

const COLOURS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b",
                 "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79", "#637939"];

const state = {
  view: VIEWS.sma,         // which page this is (VIEWS), set at load
  tabs: TABS,              // the view's tabs
  client: "sma_analyzer",
  tab: "health",
  intervalMs: 1000,
  logY: false,
  logZ: true,
  nimLogY: true,           // the NIM tab's dt plots on log y (LOG_Y_DEFAULT)
  xyLogZ: false,           // the MuPix x/y track maps on log z (XY_TRACK_MAP)
  xyHead: null,            // the MuPix x/y note and switch, placed by layoutTab
  xyNoteSig: null,         // the note's parts as last drawn
  pairLogZ: false,         // the unseeded pair maps on log z (PAIR_MAP)
  dtChips: null,           // chips drawn on the per-chip dt plot (CHIP_DT); null = all
  dtNorm: false,           // the per-chip dt lines scaled to their own peak
  dtZoom: false,           // the per-chip dt x axis on the in-time window (else the full range)
  pairHead: null,          // the pairs tab's note and switch, placed by layoutTab
  pairNoteSig: null,
  names: null,             // dqm::list, null until the analyzer has answered once
  epoch: null,             // summary epoch; a change means the histograms were rebuilt
  plots: {},               // tab -> [{key, names, wrap, div, mpg, title, foot}]
  meta: {},                // histogram name -> dqm::metadata (titles), fetched once
  summary: null,
  status: null,
  trend: { rows: [], since: null, key: null, graphs: null, labels: [], counters: [], nimCounters: [] },
  summaryUpdater: null,
  tabUpdater: null,
  graphs: new Map(),       // group key -> plot entry, reused across relists (see layoutTab)
  seen: null,              // IntersectionObserver for the plots on screen (null without one)
  table: null,             // {rows, foot}: the table's cells, updated in place
  banners: {},             // banner id -> {wrap, span}
  flagsSig: null,
  lastOkAt: null,
  mupixSig: null,          // the MuPix plane map the MuPix diagnostics tab was laid out for
  nimSig: null,            // the NIM table's rows (counters, channels) as last built
  mergeSig: null,          // the summary's nim_merge the tabs were laid out for
};

window.addEventListener("load", function () {
  state.view = viewOf(document);
  state.tabs = tabsOf(state.view);
  state.tab = state.tabs[0][0];
  mhttpd_init(mhttpd_getParameterByName("page") || state.view.page, 1000);
  restore();
  build();
  state.summaryUpdater = new BRPC.AutoUpdater(refreshSummary, state.intervalMs || 1000);
  state.summaryUpdater.onError = (e) => noAnalyzer(e);
  if (state.intervalMs) state.summaryUpdater.start();
  showTab(state.tab);
});

// ---------------------------------------------------------------------------
// Polling
// ---------------------------------------------------------------------------

async function refreshSummary() {
  const summary = await BRPC.json(state.client, "sma::summary", "");
  // The generic status is secondary: the summary alone is enough to draw the
  // page, so an analyzer that cannot produce it does not blank the table.
  let status = null;
  try { status = await BRPC.json(state.client, "dqm::status", ""); } catch (e) { /* optional */ }
  state.summary = summary;
  state.status = status;
  // The per-plane occupancy plots have one line per chip of the plane: a
  // plane-map edit in the ODB lays the MuPix diagnostics tab out again.
  const sig = JSON.stringify(summary && summary.mupix ? summary.mupix.planes : null);
  if (state.mupixSig !== null && sig !== state.mupixSig && state.plots.mupix) {
    delete state.plots.mupix;
    for (const [k, g] of [...state.graphs]) {
      if (k.indexOf("#") < 0) continue;
      if (state.seen) state.seen.unobserve(g.wrap);
      state.graphs.delete(k);
    }
    if (state.tab === "mupix") layoutTab("mupix");
  }
  state.mupixSig = sig;
  // The merged-vs-TOT overlay exists only with the merge on (groupsFor): a
  // merge edit lays the Timing and NIM tabs out again.
  const msig = summary ? String(summary.nim_merge) : null;
  if (state.mergeSig !== null && msig !== state.mergeSig) {
    for (const tab of ["timing", "nim"]) delete state.plots[tab];
    for (const k of [COINC, COINC_TOT]) {
      const g = state.graphs.get(k);
      if (g && state.seen) state.seen.unobserve(g.wrap);
      state.graphs.delete(k);
    }
    if (state.tab === "timing" || state.tab === "nim") layoutTab(state.tab);
  }
  state.mergeSig = msig;

  // A histogram rebuild (a shift, cut or binning edit in the ODB) starts a new
  // epoch and can change the set of names -- dt_Sk_S1 follows the counter
  // list, delayed_dt_chNN the delayed channels. Relist rather than 404.
  if (state.names === null || (summary && summary.epoch !== state.epoch)) {
    const changed = state.epoch !== null && summary && summary.epoch !== state.epoch;
    state.epoch = summary ? summary.epoch : null;
    await relist(changed);
  }
  render(summary, status);
}

async function relist(epochChanged) {
  const names = await BRPC.list(state.client);
  // Titles and axes can change in a rebuild (a shift edit renames the
  // fine_coarse_diff axis), so they are fetched again.
  if (epochChanged) state.meta = {};
  const same = state.names !== null && names.join("\n") === state.names.join("\n");
  state.names = names;
  // An epoch changes at every run start. The same names mean the same plots:
  // keep them and just feed them the new data. Creating graphs anew each run
  // would leak, since mplot adds a window keydown listener per graph that it
  // never removes.
  if (same && Object.keys(state.plots).length) return;
  state.plots = {};
  if (state.tab !== "trends") layoutTab(state.tab);
}

/** Retire an updater for good: stop it and drop its visibility listener. */
function retire(u) {
  if (!u) return;
  u.stop();
  document.removeEventListener("visibilitychange", u._onVisible);
}

function showTab(tab) {
  if (!state.tabs.some(([id]) => id === tab)) tab = state.tabs[0][0];
  state.tab = tab;
  save();
  for (const [id] of state.tabs) {
    const pane = document.getElementById(`dqm-sma-pane-${id}`);
    if (pane) pane.style.display = id === tab ? "" : "none";
  }
  markTabs(state.tabs, "dqm-sma", tab);
  if (tab === "trends") layoutTrends();
  else layoutTab(tab);
  // The pane was hidden, so every plot reads as off screen. Poll them all
  // until the observer reports again, and make it report: a target observed
  // afresh gets its current state, not just the next change.
  for (const p of state.plots[tab] || []) {
    p.visible = undefined;
    if (state.seen) { state.seen.unobserve(p.wrap); state.seen.observe(p.wrap); }
  }

  // A fresh updater per tab, never stop()+start() on one: a tick already in
  // flight for the old tab would re-arm after start() and leave two polling
  // chains running for the rest of the session.
  retire(state.tabUpdater);
  state.tabUpdater = new BRPC.AutoUpdater(
    tab === "trends" ? refreshTrends : () => refreshTab(tab), state.intervalMs || 1000);
  state.tabUpdater.onError = (e) => tabError(tab, e);
  if (state.intervalMs) state.tabUpdater.start();
}

function setInterval_(ms) {
  state.intervalMs = ms;
  save();
  for (const u of [state.summaryUpdater, state.tabUpdater]) {
    if (!u) continue;
    if (!ms) u.stop();
    else { u.setInterval(ms); if (!u.running) u.start(); }
  }
}

// ---------------------------------------------------------------------------
// Histogram tabs
// ---------------------------------------------------------------------------

/**
 * One graph per histogram, except the per-channel ToT pairs: tot_chNN_lsb0 and
 * _lsb1 are overlaid, because the thing to see is whether they differ (a stuck
 * fine bit 0 shows up as one of the two being empty or shifted).
 */
function groupsFor(tab) {
  const names = (state.names || []).filter((n) => n.startsWith("sma/") && tabOf(n) === tab);
  names.sort((a, b) => rank(tab, a) - rank(tab, b) || (a < b ? -1 : a > b ? 1 : 0));
  const groups = [];
  const byKey = new Map();
  for (const n of names) {
    const occ = /^sma\/mupix_(col|row)_chip$/.exec(n);
    if (occ) {
      // One 2D histogram (chip x column), drawn as one plot per plane with a
      // line per chip: the 1-D occupancy of each chip, side by side.
      for (const [plane, chips] of mupixPlanes()) {
        const key = `${n}#${plane}:${chips.join(",")}`;
        groups.push({ key, names: [n], occupancy: { axis: occ[1], plane, chips } });
      }
      continue;
    }
    if (n === CHIP_DT) {
      groups.push({ key: n, names: [n], chipDt: true });
      continue;
    }
    const m = /^(sma\/tot_ch\d+)_lsb([01])$/.exec(n);
    const key = m ? m[1] : n;
    if (!byKey.has(key)) {
      byKey.set(key, { key, names: [], row: rowOf(n), square: SQUARE.some((re) => re.test(n)) });
      groups.push(byKey.get(key));
    }
    byKey.get(key).names.push(n);
  }
  // S1 coincidences on the merged counters and on the TOT words alone, on one
  // graph: the gap between the two is what the NIM merge adds. Shown on both
  // the Timing and the NIM tab, so each tab has its own graph (key).
  // With the merge off the two are the same curve: no overlay then.
  const all = state.names || [];
  const merging = !(state.summary && state.summary.nim_merge === false);
  if (merging && all.indexOf(COINC_TOT) >= 0 && all.indexOf(COINC) >= 0) {
    for (const g of groups) {
      if (g.key === COINC || g.key === COINC_TOT) g.names = [COINC, COINC_TOT];
    }
  }
  return groups;
}

const COINC = "sma/s1_coinc";
const COINC_TOT = "sma/s1_coinc_tot";

/** A series' legend label on an overlaid graph. */
function seriesLabel(p, n) {
  if (p.names.length > 1 && /^sma\/s1_coinc(_tot)?$/.test(n)) {
    if (n === COINC_TOT) return "TOT only";
    const s = state.summary;
    return s && s.nim_merge === false ? "counters (merge off)" : "merged";
  }
  if (p.names.length > 1) return `fine bit 0 = ${n.slice(-1)}`;
  return shortName(n);
}

/** Whether a 1D histogram is drawn with log y: the toolbar switch, or a NIM dt default. */
function logYFor(name) {
  return state.logY || (state.nimLogY && LOG_Y_DEFAULT.some((re) => re.test(name)));
}

/**
 * Whether a 2D histogram is drawn with log z: the phase-space tab's switch for
 * the track maps, the pairs tab's for the pair maps, else the toolbar's.
 */
function logZFor(name) {
  if (XY_TRACK_MAP.test(name)) return state.xyLogZ;
  if (PAIR_MAP.test(name)) return state.pairLogZ;
  return state.logZ;
}

function layoutTab(tab) {
  if (state.plots[tab] || state.names === null) return;
  const grid = document.getElementById(`dqm-sma-grid-${tab}`);
  if (!grid) return;
  grid.innerHTML = "";
  const plots = [];
  const rows = new Map();
  const groups = groupsFor(tab);
  // The MuPix x/y note goes in front of the x/y plots, the pairs note in front
  // of the pair plots (at the end without them: "XY off" / "pairs off" has to
  // be said somewhere).
  let head = tab === "mupixxy" ? xyHead() : tab === "mupixpair" ? pairHead() : null;
  const headOf = tab === "mupixpair" ? PAIR_ANY : XY_ANY;
  // Where a plot goes: into its row (made at its first plot, so rows keep
  // the plots' order), or straight into the grid.
  const into = function (g) {
    if (head && headOf.test(g.names[0])) { grid.appendChild(head); head = null; }
    if (!g.row) return grid;
    if (!rows.has(g.row.key)) {
      const r = el("div", { class: `dqm-sma-row dqm-sma-row${g.row.cols}`, "data-row": g.row.key });
      grid.appendChild(r);
      rows.set(g.row.key, r);
    }
    return rows.get(g.row.key);
  };
  for (const g of groups) {
    const cached = state.graphs.get(g.key);
    if (cached && cached.names.join() === g.names.join()) {
      into(g).appendChild(cached.wrap);
      cached.title.textContent = titleFor(cached);
      plots.push(cached);
      continue;
    }
    const sq = (g.row && g.row.square) || g.square;
    const wrap = el("div", { class: `dqm-sma-plotwrap${sq ? " dqm-sma-square" : ""}` });
    const title = el("div", { class: "dqm-histtitle" }, titleFor(g));
    const div = el("div", { class: "dqm-plot" });
    const foot = el("div", { class: "dqm-footnote" }, "");
    wrap.appendChild(title);
    wrap.appendChild(div);
    wrap.appendChild(foot);
    into(g).appendChild(wrap);
    // The renderer is made at the first reply, when the dimensions are known:
    // an mplot graph for 1D, a DQMHeatmap for 2D (see drawPlot).
    const entry = Object.assign({ wrap, div, mpg: null, heat: null, title, foot, sig: null }, g);
    wrap._plot = entry;
    watch(wrap);
    state.graphs.set(g.key, entry);
    plots.push(entry);
  }
  if (head) grid.appendChild(head);
  // On the phase-space and pairs tabs the note says why there is nothing.
  if (!plots.length && tab !== "mupixxy" && tab !== "mupixpair") {
    grid.appendChild(el("div", { class: "dqm-note" },
      state.names && state.names.length
        ? "The analyzer has no histograms for this tab."
        : "The analyzer has no histograms yet."));
  }
  state.plots[tab] = plots;
}

/**
 * Plots scrolled out of view are not polled once they have been drawn: most
 * tabs are taller than a screen. rootMargin: a plot about to scroll in is
 * already fetched. Without IntersectionObserver every plot is polled.
 */
function watch(wrap) {
  if (state.seen === null && typeof IntersectionObserver === "function") {
    state.seen = new IntersectionObserver(function (entries) {
      for (const e of entries) if (e.target._plot) e.target._plot.visible = e.isIntersecting;
    }, { rootMargin: "200px 0px" });
  }
  if (state.seen) state.seen.observe(wrap);
}

/** The 1D renderer: an mplot graph, made on the first reply. */
function ensureGraph(p) {
  if (p.mpg) return p.mpg;
  if (p.heat) { p.div.innerHTML = ""; p.heat = null; p.div.heatmap = null; }
  const pair = p.names.length > 1;
  const dtWin = /^sma\/mupix_dt_L[12]$/.test(p.key);
  const pairDt = p.key === PAIR_DT;
  const mpg = new MPlotGraph(p.div, {
    showMenuButtons: true,
    mouseWheelZoom: true,
    title: { text: "" },
    stats: { show: false },
    legend: { show: pair || dtWin || pairDt || !!p.occupancy },
    xAxis: { title: { text: "", textSize: 12 }, textSize: 12 },
    yAxis: { title: { text: "", textSize: 12 }, textSize: 12 },
  });
  p.div.mpg = mpg;
  if (p.occupancy) {
    p.occupancy.chips.forEach(function (c, i) {
      mpg.addPlot({ label: `chip ${c}`, xData: [], yData: [], line: { color: COLOURS[i % COLOURS.length] } });
    });
  } else {
    p.names.forEach(function (n, i) {
      mpg.addPlot({ label: seriesLabel(p, n), xData: [], yData: [], line: { color: COLOURS[i] } });
    });
  }
  if (dtWin) {
    // The in-time window and the sideband, outlined over the histogram.
    mpg.addPlot({ label: "in time", type: "scatter", xData: [], yData: [],
                  line: { draw: true, width: 2, color: "#2ca02c" }, marker: { draw: false } });
    mpg.addPlot({ label: "sideband", type: "scatter", xData: [], yData: [],
                  line: { draw: true, width: 2, color: "#7f7f7f" }, marker: { draw: false } });
  }
  if (pairDt) {
    // The pairing window, outlined over the L2 - L1 dt.
    mpg.addPlot({ label: "pairing window", type: "scatter", xData: [], yData: [],
                  line: { draw: true, width: 2, color: "#2ca02c" }, marker: { draw: false } });
  }
  coalesce(mpg);
  p.mpg = mpg;
  window.setTimeout(function () { mpg.resize(); mpg.draw(); }, 0);
  return mpg;
}

/**
 * One draw per animation frame. mplot's setData() schedules a full draw and so
 * does redraw(): an overlaid pair drew three times per update, a five-line
 * trend six times.
 */
function coalesce(mpg) {
  let queued = false;
  const raf = window.requestAnimationFrame
    ? (f) => window.requestAnimationFrame(f) : (f) => window.setTimeout(f, 0);
  mpg.redraw = function () {
    if (queued) return;
    queued = true;
    raf(function () { queued = false; mpg.draw(); });
  };
  return mpg;
}

/** The 2D renderer, made on the first reply. */
function ensureHeatmap(p) {
  if (p.heat) return p.heat;
  if (p.mpg) { p.div.innerHTML = ""; p.mpg = null; p.div.mpg = null; }
  p.heat = new DQMHeatmap(p.div);
  p.div.heatmap = p.heat;
  sized(p.div);
  return p.heat;
}

/**
 * Redraw a heatmap whose box changed size. DQMHeatmap measures its box once
 * and again only on a window resize; a box in a row of fluid columns also
 * changes when the page around it does (a scroll bar appearing, the note
 * wrapping), and the canvas then hangs over its column, colour-bar labels
 * cut off. The observer fires only on a change, so it costs nothing per poll.
 */
function sized(div) {
  if (typeof ResizeObserver !== "function") return;
  if (!state.boxes) {
    state.boxes = new ResizeObserver(function (entries) {
      for (const e of entries) {
        const hm = e.target.heatmap;
        if (!hm || !hm.cssSize) continue;
        const w = e.target.clientWidth, h = e.target.clientHeight;
        if (!w || !h || (w === hm.cssSize[0] && h === hm.cssSize[1])) continue;
        hm.cssSize = null;
        if (hm.hist) hm.draw();
      }
    });
  }
  state.boxes.observe(div);
}

function titleFor(g) {
  if (g.occupancy) {
    const o = g.occupancy;
    return `${o.axis === "col" ? "Column" : "Row"} occupancy, ${o.plane} (chips ${o.chips.join(", ")}), ` +
           `all pixel hits${o.axis === "row" ? " (rows ≥ 250 are not on the sensor)" : ""}  ` +
           `[${shortName(g.names[0])}]`;
  }
  const m = /tot_ch(\d+)$/.exec(g.key);
  if (m) {
    const ch = Number(m[1]);
    return `ToT, ${labelOf(ch)} (ch ${ch})`;
  }
  if (g.names.length > 1 && g.names[1] === COINC_TOT) {
    return "S1 coincidences, merged TOT + NIM vs TOT only  [s1_coinc, s1_coinc_tot]";
  }
  const meta = state.meta[g.names[0]];
  if (!(meta && meta.title)) return shortName(g.key);
  // The x/y maps' frame, "(bt2026-v4, +x beam-left)", is said once, in the
  // note above them; three-across titles are short enough to stay level.
  const framed = XY_ANY.test(g.names[0]) || PAIR_ANY.test(g.names[0]);
  const t = framed ? meta.title.replace(/\s*\([^()]*\+x beam-left\)$/, "") : meta.title;
  return `${t}  [${shortName(g.names[0])}]`;
}

async function refreshTab(tab) {
  if (state.names === null) return;          // the summary poll lists them first
  layoutTab(tab);
  const plots = state.plots[tab] || [];
  let failed = 0, tried = 0, last = null;
  for (const p of plots) {
    if (state.tab !== tab) return;           // switched away mid-poll
    if (p.visible === false && p.sig !== null) continue;   // off screen, drawn before
    tried++;
    // Per plot, so one histogram that fails to decode or has vanished in a
    // rebuild does not stop the other twenty from updating.
    try {
      await drawPlot(p);
      p.wrap.classList.remove("dqm-sma-plot-error");
    } catch (e) {
      failed++;
      last = e;
      p.wrap.classList.add("dqm-sma-plot-error");
      p.foot.textContent = `could not update: ${BRPC.errorText(e)}`;
      if (/no such histogram/.test(String(e && e.message))) state.names = null;
      // The analyzer itself is gone: asking the other twenty plots the same
      // question only adds twenty timeouts. The updater backs off instead.
      // Same for mhttpd itself being down, which rejects with a plain object.
      if (!(e instanceof Error) || /did not answer/.test(e.message)) throw e;
    }
  }
  if (tried && failed === tried) throw last;
  tabOk(tab);
}

/** [[plane name, chip ids], ...] from the summary (the ODB plane map), L1 and L2. */
function mupixPlanes() {
  const pl = state.summary && state.summary.mupix && state.summary.mupix.planes;
  if (!pl) return [["L1", [0, 1, 2, 3]], ["L2", [4, 5, 6, 7]]];
  return [["L1", pl.L1 || []], ["L2", pl.L2 || []]];
}

/**
 * A per-plane occupancy plot: the chips' columns of the (chip x column/row)
 * histogram, each drawn as a 1-D histogram.
 */
async function drawOccupancy(p) {
  const name = p.names[0];
  if (!(name in state.meta)) {
    try { state.meta[name] = await BRPC.json(state.client, "dqm::metadata", name); }
    catch (e) { state.meta[name] = null; }
  }
  const hist = await BRPC.histogram(state.client, name);
  if (hist.dimensions !== 2) throw new Error(`${name} is not 2-D`);
  const sig = `${DQMHeatmap.checksum(hist.data)}|${hist.entries}|${state.logY}`;
  if (sig === p.sig && p.mpg) return;          // unchanged: nothing to draw
  ensureGraph(p);
  const nx = hist.nBins[0] + 2, ny = hist.nBins[1] + 2;
  applyScale(p.mpg, 1);
  let total = 0;
  p.occupancy.chips.forEach(function (c, i) {
    const col = new Array(ny);
    for (let iy = 0; iy < ny; iy++) col[iy] = hist.data[(c + 1) + iy * nx];
    total += col.reduce((a, b) => a + b, 0);
    BRPC.display({ dimensions: 1, nBins: [hist.nBins[1]], lowEdge: [hist.lowEdge[1]],
                   highEdge: [hist.highEdge[1]], data: col, entries: 0 }, p.mpg, i);
    p.mpg.param.plot[i].line.color = COLOURS[i % COLOURS.length];
  });
  const axes = (state.meta[name] && state.meta[name].axes) || [];
  p.mpg.param.xAxis.title.text = (axes[1] && axes[1].title) || p.occupancy.axis;
  p.mpg.param.yAxis.title.text = "entries";
  p.mpg.redraw();
  p.foot.textContent = `${Math.round(total).toLocaleString()} pixel hits on ${p.occupancy.plane}`;
  p.sig = sig;
}

/**
 * The chips of the per-chip dt plot, in order: L1's, L2's, then any other chip
 * with entries ("no plane"). [[chip, plane name or ""], ...]
 */
function chipDtChips(hist) {
  const out = [], seen = new Set();
  for (const [plane, chips] of mupixPlanes()) {
    for (const c of chips) if (!seen.has(c)) { seen.add(c); out.push([c, plane]); }
  }
  const nx = hist.nBins[0] + 2, ny = hist.nBins[1] + 2;
  for (let c = 0; c < hist.nBins[0]; c++) {
    if (seen.has(c)) continue;
    let n = 0;
    for (let iy = 0; iy < ny && !n; iy++) n = hist.data[(c + 1) + iy * nx];
    if (n) out.push([c, ""]);
  }
  return out;
}

/** Whether chip c is drawn on the per-chip dt plot. */
function chipDtShown(c) {
  return state.dtChips === null || state.dtChips.indexOf(c) >= 0;
}

/** Choose the chips drawn on the per-chip dt plot (null = all) and redraw it at once. */
function setDtChips(chips) {
  state.dtChips = chips;
  save();
  redrawVisible();
}

/**
 * The per-chip dt plot's controls, between its title and the plot: a checkbox
 * per chip in the chip's line colour (the plot's legend), L1 / L2 / all
 * shortcuts and the peak normalisation. Rebuilt when the chip list or the
 * choice changes.
 */
function chipDtBar(p, chips) {
  const sig = JSON.stringify([chips, state.dtChips, state.dtNorm, state.dtZoom]);
  if (p.bar && p.barSig === sig) return;
  if (!p.bar) {
    p.bar = el("div", { class: "dqm-sma-chipbar" });
    p.wrap.insertBefore(p.bar, p.div);
  }
  p.bar.innerHTML = "";
  const boxes = el("div", { class: "dqm-sma-chiprow" });
  for (const [c, plane] of chips) {
    const id = `dqm-sma-dtchip-${c}`;
    const box = el("input", { type: "checkbox", id });
    box.checked = chipDtShown(c);
    box.onchange = function () {
      const cur = state.dtChips === null ? chips.map((x) => x[0]) : state.dtChips.slice();
      const next = this.checked ? cur.concat([c]) : cur.filter((x) => x !== c);
      setDtChips(chips.every(([x]) => next.indexOf(x) >= 0) ? null : next);
    };
    const swatch = el("span", { class: "dqm-sma-swatch" });
    swatch.style.background = chipColour(c);
    boxes.appendChild(el("label", { for: id, class: "dqm-sma-chipbox" }, box, swatch,
                         `chip ${c}${plane ? ` (${plane})` : ""}`));
  }
  const tools = el("div", { class: "dqm-sma-chiprow" }, el("span", {}, "show"));
  const pick = function (text, chosen) {
    const b = el("button", { type: "button", class: "dqm-sma-chipbtn" }, text);
    b.onclick = function () { setDtChips(chosen); };
    tools.appendChild(b);
  };
  for (const [plane, list] of mupixPlanes()) if (list.length) pick(plane, list.slice());
  pick("all", null);
  const norm = el("input", { type: "checkbox", id: "dqm-sma-dtnorm" });
  norm.checked = state.dtNorm;
  norm.onchange = function () {
    state.dtNorm = !!this.checked;
    save();
    redrawVisible();
  };
  tools.appendChild(el("label", { for: "dqm-sma-dtnorm", class: "dqm-sma-chipbox dqm-sma-chipnorm" },
                       norm, "each chip to its peak"));
  const zoom = el("input", { type: "checkbox", id: "dqm-sma-dtzoom" });
  zoom.checked = state.dtZoom;
  zoom.onchange = function () {
    state.dtZoom = !!this.checked;
    save();
    redrawVisible();
  };
  tools.appendChild(el("label", { for: "dqm-sma-dtzoom", class: "dqm-sma-chipbox dqm-sma-chipnorm" },
                       zoom, "zoom on the in-time window"));
  p.bar.appendChild(boxes);
  p.bar.appendChild(tools);
  p.barSig = sig;
}

/** A chip's line colour, the same whichever chips are shown. */
function chipColour(c) {
  return COLOURS[c % COLOURS.length];
}

/**
 * The per-chip t(pixel) - t(S1) plot: the chips' columns of the (chip x dt)
 * histogram, each drawn as an unfilled step line in its own colour (mplot fills
 * a "histogram", and eight filled ones hide each other), with the in-time
 * window and the sideband outlined as on mupix_dt_L1/_L2. The x axis is their
 * full range, so a chip sliding out of the window is seen; "zoom on the in-time
 * window" narrows it to the window +-150 ns, where the chips' peaks sit a few
 * 8 ns bins apart. The checkboxes above it are the legend; the footer gives each
 * drawn chip's entries and peak, and calls out a peak outside the window.
 */
async function drawChipDt(p) {
  const name = p.names[0];
  if (!(name in state.meta)) {
    try { state.meta[name] = await BRPC.json(state.client, "dqm::metadata", name); }
    catch (e) { state.meta[name] = null; }
    p.title.textContent = titleFor(p);
  }
  const hist = await BRPC.histogram(state.client, name);
  if (hist.dimensions !== 2) throw new Error(`${name} is not 2-D`);
  const chips = chipDtChips(hist);
  chipDtBar(p, chips);
  const shown = chips.filter(([c]) => chipDtShown(c));
  const mp = state.summary && state.summary.mupix;
  const series = JSON.stringify(shown);
  const sig = `${DQMHeatmap.checksum(hist.data)}|${hist.entries}|${state.logY}|${state.dtNorm}|${state.dtZoom}|` +
    `${series}|${mp ? JSON.stringify([mp.window_ns, mp.sideband_ns]) : ""}`;
  if (sig === p.sig && p.mpg) return;          // unchanged: nothing to draw
  if (p.mpg && p.series !== series) {          // other chips: other lines
    p.div.innerHTML = "";
    p.mpg = null;
    p.div.mpg = null;
  }
  if (!p.mpg) {
    p.mpg = coalesce(new MPlotGraph(p.div, {
      showMenuButtons: true,
      mouseWheelZoom: true,
      title: { text: "" },
      stats: { show: false },
      legend: { show: false },
      xAxis: { title: { text: "", textSize: 12 }, textSize: 12 },
      yAxis: { title: { text: "", textSize: 12 }, textSize: 12 },
    }));
    p.div.mpg = p.mpg;
    for (const [c, plane] of shown) {
      p.mpg.addPlot({ label: `chip ${c}${plane ? ` (${plane})` : ""}`, type: "scatter",
                      xData: [], yData: [], line: { draw: true, width: 1.5, color: chipColour(c) },
                      marker: { draw: false } });
    }
    p.mpg.addPlot({ label: "in time", type: "scatter", xData: [], yData: [],
                    line: { draw: true, width: 2, color: "#2ca02c" }, marker: { draw: false } });
    p.mpg.addPlot({ label: "sideband", type: "scatter", xData: [], yData: [],
                    line: { draw: true, width: 2, color: "#7f7f7f" }, marker: { draw: false } });
    p.series = series;
    const mpg = p.mpg;
    window.setTimeout(function () { mpg.resize(); mpg.draw(); }, 0);
  }
  const nx = hist.nBins[0] + 2, ny = hist.nBins[1] + 2;
  const lo = hist.lowEdge[1], w = (hist.highEdge[1] - lo) / hist.nBins[1];
  const win = (mp && mp.window_ns) || [-150, 450];
  const side = (mp && mp.sideband_ns) || [-2400, -1800];
  applyScale(p.mpg, 1);
  // Normalised, a log axis starts three decades below the peak, not below one count.
  const floor = state.logY ? (state.dtNorm ? 1e-3 : 0.5) : 0;
  if (state.logY) p.mpg.param.yAxis.min = floor;
  if (state.dtZoom) {
    p.mpg.param.xAxis.min = Math.max(lo, win[0] - 150);
    p.mpg.param.xAxis.max = Math.min(hist.highEdge[1], win[1] + 150);
  } else {
    p.mpg.param.xAxis.min = lo;
    p.mpg.param.xAxis.max = hist.highEdge[1];
  }
  const parts = [], outside = [];
  let top = 1;
  shown.forEach(function ([c], i) {
    // Steps over the in-range bins: two points a bin, at its edges.
    const xs = new Array(2 * (ny - 2)), ys = new Array(2 * (ny - 2));
    let n = 0, peak = 0, at = -1;
    for (let iy = 0; iy < ny; iy++) {
      const v = hist.data[(c + 1) + iy * nx];
      n += v;
      if (iy > 0 && iy < ny - 1 && v > peak) { peak = v; at = iy - 1; }
    }
    const scale = state.dtNorm && peak > 0 ? 1 / peak : 1;
    for (let b = 0; b < ny - 2; b++) {
      const v = Math.max(hist.data[(c + 1) + (b + 1) * nx] * scale, floor);
      xs[2 * b] = lo + b * w;
      xs[2 * b + 1] = lo + (b + 1) * w;
      ys[2 * b] = ys[2 * b + 1] = v;
    }
    if (!state.dtNorm && peak > top) top = peak;
    p.mpg.setData(i, xs, ys);
    const t = Math.round(lo + (at + 0.5) * w);
    if (at >= 0 && (t < win[0] || t >= win[1])) outside.push(c);
    parts.push(`chip ${c}: ${Math.round(n).toLocaleString()}` + (at >= 0 ? `, peak ${t} ns` : ""));
  });
  // The in-time window and the sideband (the last two series).
  const box = (b) => [[b[0], b[0], b[1], b[1]], [floor, top, top, floor]];
  p.mpg.setData(shown.length, ...box(win));
  p.mpg.setData(shown.length + 1, ...box(side));
  const axes = (state.meta[name] && state.meta[name].axes) || [];
  p.mpg.param.xAxis.title.text = (axes[1] && axes[1].title) || "t(pixel) - t(S1) (ns)";
  p.mpg.param.yAxis.title.text = state.dtNorm ? "entries / chip's peak" : "entries";
  p.mpg.redraw();
  const out = outside.length
    ? `PEAK OUTSIDE THE IN-TIME WINDOW: chip${outside.length > 1 ? "s" : ""} ${outside.join(", ")} · ` : "";
  p.foot.textContent = shown.length
    ? `${out}${parts.join(" · ")} · green: in time, grey: sideband`
    : "No chip chosen: tick one above.";
  p.foot.classList.toggle("dqm-sma-foot-warn", outside.length > 0);
  p.sig = sig;
}

/** The pairing window [-w, +w] ns outlined on the L2 - L1 dt (series 1). */
function markPairWindow(p, hist) {
  const w = pairWindow(state.summary && state.summary.pairs);
  if (!w || p.mpg.param.plot.length < 2) return;
  let top = 1;
  for (const v of hist.data) if (v > top) top = v;
  const lo = logYFor(PAIR_DT) ? 0.5 : 0;
  p.mpg.setData(1, [w[0], w[0], w[1], w[1]], [lo, top, top, lo]);
}

/** summary.pairs.window_ns as [lo, hi] ns: a half-width (40 -> [-40, 40]) or a pair; null if absent. */
function pairWindow(pr) {
  const w = pr ? pr.window_ns : null;
  if (typeof w === "number" && Number.isFinite(w)) return [-Math.abs(w), Math.abs(w)];
  if (Array.isArray(w) && w.length === 2 && w.every((x) => typeof x === "number")) return w;
  return null;
}

/** The in-time window and sideband outlines on a t(pixel) - t(S1) plot (series 1 and 2). */
function markMupixWindows(p, hist) {
  const mp = state.summary && state.summary.mupix;
  if (!mp || p.mpg.param.plot.length < 3) return;
  let top = 1;
  for (const v of hist.data) if (v > top) top = v;
  const lo = state.logY ? 0.5 : 0;
  const box = (w) => [[w[0], w[0], w[1], w[1]], [lo, top, top, lo]];
  const [wx, wy] = box(mp.window_ns || [-150, 450]);
  const [sx, sy] = box(mp.sideband_ns || [-2400, -1800]);
  p.mpg.setData(1, wx, wy);
  p.mpg.setData(2, sx, sy);
}

async function drawPlot(p) {
  if (p.occupancy) { await drawOccupancy(p); return; }
  if (p.chipDt) { await drawChipDt(p); return; }
  const hists = [];
  for (let i = 0; i < p.names.length; i++) {
    const name = p.names[i];
    if (!(name in state.meta)) {
      // Titles and axis names, once per histogram. Optional: a musip-style
      // analyzer without dqm::metadata still gets its plots.
      try { state.meta[name] = await BRPC.json(state.client, "dqm::metadata", name); }
      catch (e) { state.meta[name] = null; }
      p.title.textContent = titleFor(p);
    }
    hists.push(await BRPC.histogram(state.client, name));
  }
  const counts = hists.map((h) => h.entries);
  const name0 = p.names[0];

  if (hists[0].dimensions === 2 && hists.length === 1) {
    const hist = hists[0];
    const axes = (state.meta[name0] && state.meta[name0].axes) || [];
    const logZ = logZFor(name0);
    ensureHeatmap(p).setData(hist, {
      logZ,
      xTitle: (axes[0] && axes[0].title) || "",
      yTitle: (axes[1] && axes[1].title) || "",
    });
    const out = p.heat.scale ? p.heat.scale.outside : 0;
    const text = `${counts[0].toLocaleString()} entries` +
      (out > 0 ? ` · ${Math.round(out).toLocaleString()} in under/overflow (not drawn)` : "");
    if (p.foot.textContent !== text) p.foot.textContent = text;
    p.sig = "2d";
    return;
  }

  // 1D: redrawn only when a histogram, the y scale or the MuPix windows changed.
  const mp = state.summary && state.summary.mupix;
  const pw = name0 === PAIR_DT ? JSON.stringify(pairWindow(state.summary && state.summary.pairs)) : "";
  const logY = logYFor(name0);
  const sig = hists.map((h) => `${DQMHeatmap.checksum(h.data)}:${h.entries}`).join(",") +
    `|${logY}|${mp ? JSON.stringify([mp.window_ns, mp.sideband_ns]) : ""}|${pw}|${seriesLabel(p, name0)}`;
  if (sig === p.sig && p.mpg) return;
  ensureGraph(p);
  hists.forEach(function (hist, i) {
    const name = p.names[i];
    applyScale(p.mpg, hist.dimensions, logY);
    if (p.names.length > 1) p.mpg.param.plot[i].label = seriesLabel(p, name);
    BRPC.display(hist, p.mpg, i);
    // mplot resets a histogram's colour in setData; an overlaid pair has to be
    // told apart, so the colour is put back afterwards.
    if (hist.dimensions === 1 && p.names.length > 1) p.mpg.param.plot[i].line.color = COLOURS[i];
    applyAxisTitles(p.mpg, state.meta[name], hist.dimensions);
    if (/^sma\/mupix_dt_L[12]$/.test(name)) markMupixWindows(p, hist);
    if (name === PAIR_DT) markPairWindow(p, hist);
    if (name === "sma/mupix_s1_match") p.matchCounts = Array.from(hist.data);
  });
  p.mpg.redraw();
  const lsb = /_lsb[01]$/.test(name0);
  p.foot.textContent = p.names.length > 1
    ? p.names.map((n, i) => `${lsb ? `lsb${n.slice(-1)}` : seriesLabel(p, n)}: ` +
                            `${counts[i].toLocaleString()}`).join(" · ") + " entries"
    : `${counts[0].toLocaleString()} entries` + (p.matchCounts ? matchText(p.matchCounts) : "") +
      (logY && !state.logY ? " · log y (NIM tab switch)" : "");
  p.sig = sig;
}

/**
 * The fractions the S1-match histogram holds since the last clear: bin 0 the
 * S1 hits judged, 1-3 L1 / L2 / L1+L2 in time, 4-6 in the sideband.
 */
function matchText(c) {
  const b = (k) => c[k + 1] || 0;                 // index 0 is the underflow
  const n = b(0);
  if (!n) return "";
  const f = (k) => `${(100 * b(k) / n).toFixed(1)} %`;
  return ` · since the last clear, of ${n.toLocaleString()} S1 hits: in time L1 ${f(1)}, L2 ${f(2)}, ` +
         `L1+L2 ${f(3)}; sideband L1 ${f(4)}, L2 ${f(5)}, L1+L2 ${f(6)}`;
}

/**
 * Log scales, with an explicit floor.
 *
 * mplot takes the axis minimum from the data, which for a histogram is 0, and
 * a log axis from 0 clamps to 1e-20 -- twenty decades of empty plot. Pinning
 * the minimum at 0.5 starts the axis just below one count; empty bins keep
 * their own colour in a colormap (zeroColor) because they are < 0.5.
 */
function applyScale(mpg, dims, logY) {
  const y = mpg.param.yAxis, z = mpg.param.zAxis || (mpg.param.zAxis = {});
  const ly = logY === undefined ? state.logY : logY;
  if (dims === 1) {
    y.log = ly;
    if (ly) y.min = 0.5; else delete y.min;
    z.log = false;
  } else {
    y.log = false;
    delete y.min;
    z.log = state.logZ;
    if (state.logZ) z.min = 0.5; else delete z.min;
  }
}

function applyAxisTitles(mpg, meta, dims) {
  const axes = (meta && meta.axes) || [];
  if (axes[0]) mpg.param.xAxis.title.text = axes[0].title || "";
  mpg.param.yAxis.title.text = dims === 2 ? ((axes[1] && axes[1].title) || "") : "entries";
}

function redrawVisible() {
  // A scale toggle should not wait a whole poll to take effect.
  if (state.tab === "trends") drawTrends();
  else if (state.tabUpdater && state.intervalMs) {
    retire(state.tabUpdater);
    showTab(state.tab);
  } else {
    refreshTab(state.tab).catch((e) => tabError(state.tab, e));
  }
}

// ---------------------------------------------------------------------------
// Trends
// ---------------------------------------------------------------------------

const TREND_S = 600;

async function refreshTrends() {
  const args = state.trend.since !== null ? JSON.stringify({ since: state.trend.since }) : "";
  const reply = await BRPC.json(state.client, "sma::trend", args);
  if (!reply) return;
  const tr = state.trend;
  // Column k of a row's eff / nim_eff is counter k of the reply's lists: when
  // those change (a counter list edit), the old rows would be relabelled, so
  // they are dropped and the next poll asks for the whole 10 minutes again.
  const cols = JSON.stringify([reply.counters || null, reply.nim_counters || null]);
  if (tr.cols !== undefined && tr.cols !== cols) {
    tr.cols = cols;
    tr.rows = [];
    tr.since = null;
    tr.key = null;
    return;
  }
  tr.cols = cols;
  tr.now = reply.t;
  tr.labels = reply.labels || tr.labels;
  tr.counters = reply.counters || tr.counters;
  tr.nimCounters = reply.nim_counters || tr.nimCounters;
  const seen = new Set(tr.rows.map((r) => r.t));
  for (const r of reply.rows || []) if (!seen.has(r.t)) tr.rows.push(r);
  tr.rows.sort((a, b) => a.t - b.t);
  tr.rows = tr.rows.filter((r) => r.t > reply.t - TREND_S);
  if (tr.rows.length) tr.since = tr.rows[tr.rows.length - 1].t;
  layoutTrends();
  drawTrends();
  tabOk("trends");
}

/** The channels worth a rate line: those with a role (from the summary). */
function trendChannels() {
  const s = state.summary;
  if (!s || !s.channels) return [];
  return s.channels.filter((c) => c.role).map((c) => c.ch);
}

function layoutTrends() {
  const tr = state.trend;
  const chans = trendChannels();
  // Row k of "eff" belongs to counters[k]. S1 given S1 is 1 by definition and
  // not worth a line, whether or not the analyzer sends it.
  const names = tr.counters.length ? tr.counters
    : ((state.summary && state.summary.efficiency) || []).map((e) => e && e.counter);
  const counters = names.map((name, k) => [name, k]).filter(([name]) => name && name !== "S1");
  // A counter whose efficiency the analyzer withholds (a timestamp fault)
  // says so in the legend; its trend points are null and not drawn.
  const withheld = {};
  for (const e of (state.summary && state.summary.efficiency) || []) {
    if (e && e.counter && (e.eff === null || e.eff === undefined)) {
      withheld[e.counter] = String(e.reason || "withheld").split(":")[0];
    }
  }
  // The TOT + NIM pair efficiency: one line per counter with a NIM copy (the
  // summary's NIM rows), column k of each row's nim_eff (S1 first).
  const nimRows = ((state.summary && state.summary.nim && state.summary.nim.counters) || []);
  const nimCols = tr.nimCounters.map((name, k) => [name, k])
    .filter(([name]) => nimRows.some((r) => r.counter === name));
  const nimLabel = (name) => {
    const r = nimRows.find((x) => x.counter === name);
    return r ? `${r.label} + ${r.nim_label}` : name;
  };
  const key = JSON.stringify([chans, counters, chans.map(labelOf), withheld,
                              nimCols.map(([name]) => nimLabel(name))]);
  if (tr.graphs && tr.key === key) return;
  tr.key = key;

  const grid = document.getElementById("dqm-sma-grid-trends");
  if (!grid) return;
  grid.innerHTML = "";
  const specs = [
    { id: "rates", title: "Rate per channel (Hz, log)", log: true,
      series: chans.map((c) => ({ label: labelOf(c), get: (r) => (r.rate_hz ? r.rate_hz[c] : null) })) },
    { id: "live", title: "Live fraction", fraction: true,
      series: [{ label: "live", get: (r) => r.live }] },
    { id: "eff", title: "Efficiency given S1 (counter within the coincidence window)", fraction: true,
      series: counters.map(([name, k]) => ({
        label: withheld[name] ? `${name} (n/a: ${withheld[name]})` : name,
        get: (r) => (r.eff ? r.eff[k] : null) })) },
    { id: "rf", title: "RF valid fraction (S1 hits with a valid RF gate)", fraction: true,
      series: [{ label: "RF valid", get: (r) => r.rf_valid }] },
    { id: "mupix", title: "S1 hits with a MuPix hit in time (accidentals taken out; raw and sideband for L1+L2)",
      fraction: true,
      series: [
        { label: "L1", get: (r) => (r.mupix ? r.mupix.corr[0] : null) },
        { label: "L2", get: (r) => (r.mupix ? r.mupix.corr[1] : null) },
        { label: "L1+L2", get: (r) => (r.mupix ? r.mupix.corr[2] : null) },
        { label: "L1+L2 raw", get: (r) => (r.mupix ? r.mupix.in[2] : null) },
        { label: "L1+L2 sideband", get: (r) => (r.mupix ? r.mupix.side[2] : null) },
      ] },
  ];
  if (nimCols.length) {
    specs.push({ id: "nim", title: "TOT + NIM pair efficiency (TOT words with a NIM copy in the window)",
      fraction: true,
      series: nimCols.map(([name, k]) => ({
        label: nimLabel(name), get: (r) => (r.nim_eff ? r.nim_eff[k] : null) })) });
  }
  tr.graphs = specs.map(function (spec) {
    const wrap = el("div", { class: "dqm-sma-plotwrap" });
    wrap.appendChild(el("div", { class: "dqm-histtitle" }, spec.title));
    const div = el("div", { class: "dqm-plot" });
    wrap.appendChild(div);
    const foot = el("div", { class: "dqm-footnote" }, "");
    wrap.appendChild(foot);
    grid.appendChild(wrap);
    const mpg = new MPlotGraph(div, {
      showMenuButtons: true,
      mouseWheelZoom: true,
      title: { text: "" },
      stats: { show: false },
      legend: { show: spec.series.length > 1 },
      xAxis: { min: -TREND_S / 60, max: 0, title: { text: "minutes ago", textSize: 12 }, textSize: 12 },
      yAxis: { log: !!spec.log, title: { text: "", textSize: 12 }, textSize: 12 },
    });
    coalesce(mpg);
    div.mpg = mpg;
    spec.series.forEach(function (s, i) {
      mpg.addPlot({ label: s.label, type: "scatter",
                    line: { draw: true, width: 1.5, color: COLOURS[i % COLOURS.length] },
                    marker: { draw: false }, xData: [], yData: [] });
    });
    window.setTimeout(function () { mpg.resize(); mpg.draw(); }, 0);
    return Object.assign({ wrap, div, mpg, foot }, spec);
  });
}

function drawTrends() {
  const tr = state.trend;
  if (!tr.graphs) return;
  const now = tr.now || (tr.rows.length ? tr.rows[tr.rows.length - 1].t : 0);
  for (const g of tr.graphs) {
    try {
      let any = false, lo = Infinity, hi = -Infinity;
      g.series.forEach(function (s, i) {
        const xs = [], ys = [];
        for (const r of tr.rows) {
          const v = s.get(r);
          // A second with no frames is a null, and mplot has no notion of a
          // missing point: it would draw it at 0 (or at -inf on the log axis).
          if (v === null || v === undefined || !Number.isFinite(v)) continue;
          if (g.log && v <= 0) continue;
          xs.push((r.t - now) / 60);
          ys.push(v);
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
        if (xs.length) any = true;
        g.mpg.setData(i, xs, ys);
      });
      // mplot autoscales from the data; with none at all its range is
      // Infinity..-Infinity and the axis draws garbage. Pin a sane one.
      // Fractions are pinned to [0, 1]: a 0.2 % wobble autoscaled to full
      // height reads as an alarm. A log axis is pinned around the data, since
      // mplot's 5 % padding below the minimum goes negative and clamps to 1e-20.
      const y = g.mpg.param.yAxis;
      if (g.fraction) { y.min = 0; y.max = 1.05; }
      else if (!any) { y.min = g.log ? 1 : 0; y.max = g.log ? 1e6 : 1; }
      else if (g.log) { y.min = lo / 2; y.max = hi * 2; }
      else { delete y.min; delete y.max; }
      g.mpg.redraw();
      g.foot.textContent = any ? `${tr.rows.length} s of rows` : "no data in the last 10 min";
      g.wrap.classList.remove("dqm-sma-plot-error");
    } catch (e) {
      g.wrap.classList.add("dqm-sma-plot-error");
      g.foot.textContent = `could not draw: ${BRPC.errorText(e)}`;
    }
  }
}

// ---------------------------------------------------------------------------
// Status, banner, flags, table
// ---------------------------------------------------------------------------

function labelOf(ch) {
  const s = state.summary;
  const c = s && s.channels ? s.channels[ch] : null;
  return c && c.label ? c.label : `ch${String(ch).padStart(2, "0")}`;
}

function render(s, status) {
  state.lastOkAt = Date.now();
  setStale(false);
  // Each block on its own: a summary field the page does not expect must cost
  // that block, not the whole top of the page.
  guard("dqm-sma-status", () => renderStatus(s, status));
  guard("dqm-sma-sampling", () => renderSampling(s, status));
  guard("dqm-sma-banner", () => renderBanner(s));
  guard("dqm-sma-flags", () => renderFlags(s));
  guard("dqm-sma-table", () => renderTable(s));
  guard("dqm-sma-mupix", () => renderMupix(s));
  guard("dqm-sma-xynote", () => renderXy(s));
  guard("dqm-sma-pairnote", () => renderPairs(s));
  guard("dqm-sma-nim", () => renderNim(s));
}

// ---------------------------------------------------------------------------
// TOT + NIM
// ---------------------------------------------------------------------------

const LAG_CLASS = { ok: "", faulted: "alarm", ambiguous: "warn", none: "" };
/** Which NIM table cell a NIM flag colours. */
const NIM_FLAG_CELL = { nim_missing: 0, nim_pairing: 1, nim_offset: 4, nim_lag: 5 };
const NIM_COLUMNS = ["counter / NIM copy", "pair efficiency", "purity", "NIM-only share",
                     "median NIM − TOT", "lag vote (last · faulted frames · last lag)",
                     "NIM-only held back"];

/** The flags about one NIM row: those naming its TOT or its NIM channel. */
function nimFlagsOf(s, r) {
  const out = [];
  for (const f of (s && s.flags) || []) {
    if (!/^nim_/.test(f.code)) continue;
    // The analyzer names the channels (ch, nim_ch); an older one only in the text.
    if (typeof f.nim_ch === "number") {
      if (f.ch === r.ch && f.nim_ch === r.nim_ch) out.push(f);
      continue;
    }
    const chans = [];
    const re = /\(ch (\d+)\)/g;
    let m;
    while ((m = re.exec(f.text || "")) !== null) chans.push(Number(m[1]));
    if (chans.indexOf(r.ch) >= 0 || chans.indexOf(r.nim_ch) >= 0) out.push(f);
  }
  return out;
}

/**
 * The lag state ("ok" / "FAULTED"; a frame without a decisive vote of its own
 * takes the last one), the faulted share of the window's frames with a state,
 * and the last voted lag.
 */
function lagText(lg) {
  if (!lg) return "—";
  // A coarse offset shown by the epoch vote: the lag vote pairs on equal
  // coarse fields, i.e. with the wrong S1 word, and its lag means nothing.
  if (lg.na) return `n/a: ${lg.na}`;
  const s0 = lg.state || lg.last_state;
  const st = s0 ? (s0 === "faulted" ? "FAULTED" : s0) : "no vote";
  const nst = (lg.state_ok || 0) + (lg.state_faulted || 0);
  const ff = nst ? `${pct(lg.state_faulted_frac, 0)} of ${num(nst)}`
    : lg.voted ? `${pct(lg.faulted_frac, 0)} of ${num(lg.voted)}` : "no frame voted";
  const last = lg.last_ns === null || lg.last_ns === undefined ? "—" : `${fmtNs(lg.last_ns)}`;
  return `${st} · ${ff} · ${last}`;
}

function fmtNs(ns) {
  const a = Math.abs(ns);
  if (a >= 1e6) return `${(ns / 1e6).toFixed(2)} ms`;
  if (a >= 1e4) return `${(ns / 1e3).toFixed(1)} µs`;
  return `${ns} ns`;
}

/**
 * The NIM tab's head: the merge settings as chips, then one row per counter
 * with a NIM copy. Built once per set of counters, cells updated in place;
 * a NIM flag colours the cell it is about.
 */
function renderNim(s) {
  const holder = document.getElementById("dqm-sma-nim");
  if (!holder) return;
  const nim = s && s.nim;
  if (!nim || !nim.active) {
    holder.innerHTML = "";
    state.nimSig = null;
    holder.appendChild(el("div", { class: "dqm-note" },
      "No NIM copies are configured (/DQM/SMA/NIM/channels): the counters are their TOT words."));
    return;
  }
  const rows = nim.counters || [];
  const sig = JSON.stringify(rows.map((r) => [r.counter, r.ch, r.nim_ch, r.label, r.nim_label]));
  if (state.nimSig !== sig || !holder._cells) {
    holder.innerHTML = "";
    holder.appendChild(el("div", { class: "dqm-strip dqm-sma-nimchips", id: "dqm-sma-nimchips" }));
    const t = el("table", { class: "dqm-table dqm-sma-nimtable" });
    const head = el("tr", {});
    for (const h of NIM_COLUMNS) head.appendChild(el("th", {}, h));
    t.appendChild(head);
    holder._cells = rows.map(function (r) {
      const tr = el("tr", { "data-counter": r.counter });
      const tds = NIM_COLUMNS.map(() => { const td = el("td", {}); tr.appendChild(td); return td; });
      t.appendChild(tr);
      return tds;
    });
    const wrap = el("div", { class: "dqm-sma-nimwrap" }, t);
    holder.appendChild(wrap);
    holder.appendChild(el("div", { class: "dqm-footnote", id: "dqm-sma-nimfoot" }));
    state.nimSig = sig;
  }

  const bar = document.getElementById("dqm-sma-nimchips");
  bar.innerHTML = "";
  bar.appendChild(mergeChip(s));
  bar.appendChild(chip(`pair window ±${nim.pair_window_ns} ns · time from ${nim.time_source}`));
  if (s.nim_merge) {
    bar.appendChild(chip(nim.merge_when_lagged ? "lagged NIM channels merged too"
      : "a lagged NIM channel's NIM-only hits are held back"));
    const held = (s.nim_lag_held || []).map((c) => labelOf(c));
    if (held.length) bar.appendChild(chip(`held back now: ${held.join(", ")}`, "yellow"));
  } else {
    bar.appendChild(chip("NIM-only hits counted, not merged: counters and pattern are TOT only"));
  }

  rows.forEach(function (r, i) {
    const tds = holder._cells[i];
    const marks = {};
    for (const f of nimFlagsOf(s, r)) {
      const k = NIM_FLAG_CELL[f.code];
      if (k === undefined) continue;
      const cls = f.severity === "error" ? "alarm" : "warn";
      if (marks[k] !== "alarm") marks[k] = cls;
    }
    const put = (k, text, extra, title) =>
      setCell(tds[k], text, [extra || "", marks[k] || ""].filter(Boolean).join(" "), title);
    put(0, `${r.label} (ch ${r.ch}) / ${r.nim_label} (ch ${r.nim_ch})`, "",
        `${num(r.tot_words)} TOT words, ${num(r.nim_words)} NIM words in the window` +
        (r.echo_rule ? `; echo rule on (${num(r.echo)} echo words)` : ""));
    // Within the frames' coverage when a channel is epoch-repaired (eff_*):
    // a TOT word whose partner lies in the next frame is left out.
    const ep = r.eff_paired === undefined ? r.paired : r.eff_paired;
    const et = r.eff_tot_only === undefined ? r.tot_only : r.eff_tot_only;
    if ((r.pair_eff === null || r.pair_eff === undefined) && r.pair_eff_reason) {
      put(1, "n/a", "dqm-sma-na", r.pair_eff_reason);
    } else put(1, pct(r.pair_eff, 1), "", `paired / (paired + TOT-only): ${num(ep)} of ` +
        `${num((ep || 0) + (et || 0))} TOT words have a NIM copy` +
        (ep !== r.paired || et !== r.tot_only
          ? " (TOT words inside the frames' coverage of the epoch-repaired channel only)" : ""));
    put(2, pct(r.purity, 1), "", `paired / NIM words: ${num(r.paired)} of ${num(r.nim_words)}`);
    put(3, pct(r.nim_only_frac, 1), "", `${num(r.nim_only)} NIM words without a TOT word, of the ` +
        (s.nim_merge ? "merged hits: what the merge adds"
                     : "hits a merge would make: what it would add (merge off, nothing is merged)"));
    const med = r.median_dt_ns === null || r.median_dt_ns === undefined ? "—"
      : `${r.median_dt_ns > 0 ? "+" : ""}${r.median_dt_ns} ns`;
    put(4, med, "", `after the ${r.offset_ns} ns offset (NIM/offset ns); ${num(r.dt_entries)} entries`);
    const lg = r.lag || {};
    put(5, lagText(lg), marks[5] ? "" : (LAG_CLASS[lg.last_state] || ""),
        `fine-time lag vote against S1: ok ${num(lg.ok)}, faulted ${num(lg.faulted)}, ambiguous ` +
        `${num(lg.ambiguous)}, no vote ${num(lg.none)} frames; nominal ${lg.nominal_ns} ns` +
        (lg.last_age_s === null || lg.last_age_s === undefined ? "" : `; last vote ${secs(lg.last_age_s)} s ago`));
    if (s.nim_merge) {
      put(6, num(r.lag_held), r.lag_held ? "warn" : "",
          "NIM-only hits held back from the merge because the lag state was faulted in their frame");
    } else {
      put(6, "— (merge off)", "dqm-sma-na", "The merge is off, so nothing is held back from it; " +
          "the lag state (previous column) is measured all the same");
    }
  });
  const foot = `Over the last ${Math.round(s.window_s || 0)} s. A healthy counter pairs ≳ 95 % of its ` +
    "TOT words with a median NIM − TOT near 0; a lag fault moves the NIM copy by ~150 µs or ~0.9 µs " +
    "(nim_dt_wide) and is measured only, never corrected. Hover a cell for the counts.";
  const fn = document.getElementById("dqm-sma-nimfoot");
  if (fn.textContent !== foot) fn.textContent = foot;
}

/** "NIM merge on/off": whether the counters (pattern, efficiencies, seeds) include NIM-only hits. */
function mergeChip(s) {
  const on = !!(s && s.nim_merge);
  const c = chip(on ? "NIM merge on" : "NIM merge off", on ? "blue" : "");
  c.setAttribute("title", on
    ? "The counters are the merged TOT + NIM hits (TOT words plus NIM-only hits), as nearline's: " +
      "pattern, coincidences, efficiencies and seeds use them"
    : "The counters are the TOT words only (/DQM/SMA/NIM/merge = n, or no NIM copies)");
  c.id = "dqm-sma-mergechip";
  return c;
}

const SYNC_CLASS = { ok: "green", low: "yellow", flagged: "red", insufficient: "", off: "" };

/**
 * The MuPix diagnostics tab's head: per plane, the share of S1 hits with a pixel hit in
 * the in-time window, in the sideband, and with the accidentals taken out,
 * over the summary window; and the SMA <-> MuPix time-sync monitor's state.
 * Built once, cells updated in place.
 */
function renderMupix(s) {
  const holder = document.getElementById("dqm-sma-mupix");
  if (!holder) return;
  const mp = s && s.mupix;
  if (!mp) { holder.innerHTML = ""; holder._built = false; return; }
  if (!holder._built) {
    holder.innerHTML = "";
    holder.appendChild(el("div", { class: "dqm-strip dqm-sma-mpsync", id: "dqm-sma-mpsync" }));
    const t = el("table", { class: "dqm-table dqm-sma-mptable" });
    const head = el("tr", {});
    for (const h of ["S1 hits with a MuPix hit", "in time", "sideband", "accidentals taken out"]) {
      head.appendChild(el("th", {}, h));
    }
    t.appendChild(head);
    holder._cells = {};
    for (const k of ["L1", "L2", "both"]) {
      const tr = el("tr", { "data-plane": k });
      const tds = [0, 1, 2, 3].map(() => el("td", {}));
      tds.forEach((td) => tr.appendChild(td));
      holder._cells[k] = tds;
      t.appendChild(tr);
    }
    holder.appendChild(t);
    holder.appendChild(el("div", { class: "dqm-footnote", id: "dqm-sma-mpfoot" }));
    holder._built = true;
  }
  const w = mp.window_ns || [], sb = mp.sideband_ns || [];
  const planes = mp.planes || {};
  for (const k of ["L1", "L2", "both"]) {
    const f = (mp.fractions || {})[k] || {};
    const tds = holder._cells[k];
    const name = k === "both" ? "L1 and L2" : `${k} (chips ${(planes[k] || []).join(", ")})`;
    setCell(tds[0], name, "");
    setCell(tds[1], pct(f.in, 1), "");
    setCell(tds[2], pct(f.side, 1), "");
    setCell(tds[3], pct(f.corr, 1), k === "both" ? "dqm-sma-mpmain" : "");
  }
  const sy = mp.sync || {};
  const bar = document.getElementById("dqm-sma-mpsync");
  bar.innerHTML = "";
  const val = sy.value === null || sy.value === undefined ? "" : ` · L1+L2 ${pct(sy.value, 1)}`;
  const thr = sy.threshold === null || sy.threshold === undefined ? "" : ` (warns below ${pct(sy.threshold, 0)})`;
  const stateText = {
    ok: "in sync", low: `low for ${Math.round(sy.low_for_s || 0)} s`, flagged: "SYNC LOST?",
    insufficient: "no verdict (too few S1 hits)", off: "MuPix analysis off",
  }[sy.state] || String(sy.state || "?");
  bar.appendChild(chip(`SMA ↔ MuPix time sync: ${stateText}${val}${thr}`, SYNC_CLASS[sy.state] || ""));
  if (sy.absent) bar.appendChild(chip("no pixel words", "red"));
  bar.appendChild(chip(`${num(mp.n_s1)} S1 hits judged · ${mp.pixels_per_frame === null ? "—" :
    num(Math.round(mp.pixels_per_frame))} pixel words per frame`));
  if (mp.unmapped) bar.appendChild(chip(`${num(mp.unmapped)} hits on chips with no plane`, "yellow"));
  if (mp.skipped_frac) bar.appendChild(chip(`${pct(mp.skipped_frac, 1)} of pixel hits skipped (cap)`, "blue"));
  const foot = `In time: t(pixel) - t(S1) in [${w[0]}, ${w[1]}) ns; sideband [${sb[0]}, ${sb[1]}) ns ` +
    `(the accidentals); over the last ${Math.round(s.window_s || 0)} s, from the analysed S1 hits whose ` +
    "windows lie in the frame's pixel data. The t(pixel) - t(S1) peak sits near 0 while the two time " +
    "bases agree; it vanishes when they do not (see the sync chip).";
  const fn = document.getElementById("dqm-sma-mpfoot");
  if (fn.textContent !== foot) fn.textContent = foot;
}

// ---------------------------------------------------------------------------
// MuPix x/y
// ---------------------------------------------------------------------------

/**
 * The head of the MuPix x/y plots: a one-line note from summary.xy (renderXy)
 * and the switch for log z on the track maps. Made once and kept, so a click on
 * the switch is not lost to a redraw; layoutTab puts it in front of the plots.
 */
function xyHead() {
  if (state.xyHead) return state.xyHead;
  const note = el("div", { class: "dqm-sma-xynote", id: "dqm-sma-xynote" }, "MuPix x/y: …");
  const tools = el("div", { class: "dqm-sma-xytools" },
    checkbox("dqm-sma-xylogz", "log z for the track maps", state.xyLogZ,
      function (v) { state.xyLogZ = v; save(); redrawVisible(); }));
  state.xyHead = el("div", { class: "dqm-sma-xyhead", id: "dqm-sma-xyhead" }, note, tools);
  state.xyHead._note = note;
  if (state.summary) renderXy(state.summary);
  return state.xyHead;
}

/** A shift in mm, signed: "+1.50", "−2.00". */
function fmtShift(x) {
  const v = Number(x) || 0;
  return `${v < 0 ? "−" : "+"}${Math.abs(v).toFixed(2)}`;
}

/** Where a quadrant is drawn on the maps (x to the right is beam-left: the view looking upstream). */
const QUAD_ARROW = ["↘", "↙", "↗", "↖"];   // q0 beam-left bottom ... q3 beam-right top

/**
 * The chips as drawn, per plane, top row then bottom: "L1 ↖3 ↗2 ↙1 ↘0". Its
 * hover text names each chip's quadrant in words.
 */
function quadText(quads) {
  const out = [], tip = [];
  for (const plane of ["L1", "L2"]) {
    const qs = quads.filter((q) => q.plane === plane);
    if (!qs.length) continue;
    const at = (k) => { const q = qs.find((x) => x.quadrant === k); return q ? String(q.chip) : "–"; };
    out.push(`${plane} ${[3, 2, 1, 0].map((k) => `${QUAD_ARROW[k]}${at(k)}`).join(" ")}`);
    for (const q of qs) tip.push(`chip ${q.chip}: ${plane} ${q.where}`);
  }
  return out.length ? [`chips as drawn: ${out.join(" · ")}`, tip.join("\n")] : null;
}

/**
 * The note's parts, [[text, kind, title], ...], kind "" or "warn" (muted
 * warning), "muted" or "label", title an optional hover text. From summary.xy
 * alone; a missing block says so.
 */
function xyNoteParts(xy) {
  if (!xy) return [["MuPix x/y:", "label"], ["no x/y summary from the analyzer", "warn"]];
  if (!xy.enabled) {
    return [["MuPix x/y:", "label"],
            [`XY off: ${xy.off_reason || "MuPix/XY/enable = n, or the MuPix analysis is off"}`,
             xy.off_reason && !/enable = n/.test(xy.off_reason) ? "warn" : ""]];
  }
  const f = xy.fractions || {}, c = xy.cuts || {}, st = xy.stage || {};
  const parts = [["MuPix x/y:", "label"]];
  parts.push([`tracks ${pct(f.track, 1)} of ${num(xy.n_s1)} S1 hits judged, ambiguous ${pct(f.ambiguous, 1)}`, ""]);
  const unit = c.tot_ns ? ` (×${c.tot_ns} ns)` : "";
  parts.push([`light ${pct(xy.light_frac, 1)}, heavy ${pct(xy.heavy_frac, 1)} of the tracks ` +
              `(light: both planes' max ToT ${lightCut(c)}${unit} · heavy: ≥ ${c.tot_heavy_min})`, ""]);
  parts.push(stagePart(st));
  parts.push([`${xy.geometry || "?"}, +x beam-left, seen looking upstream`, ""]);
  const qt = quadText(xy.quadrants || []);
  if (qt) parts.push([qt[0], "", qt[1]]);
  const un = xy.unplaced_chips || [];
  if (un.length) parts.push([`chips with no place: ${un.join(", ")}`, "warn"]);
  if (xy.resets) parts.push([`maps reset ${xy.resets}× by XY edits`, "muted"]);
  return parts;
}

/** The light ToT cut of a summary's cuts block; an older analyzer sends no light min. */
function lightCut(c) {
  return c.tot_light_min == null ? `≤ ${c.tot_light_max}`
    : `${c.tot_light_min}–${c.tot_light_max}`;
}

/** The stage shift as a note part, [text, kind, title], from a summary's stage block. */
function stagePart(st) {
  const sh = st.shift_mm || [0, 0];
  const shift = `x ${fmtShift(sh[0])}, y ${fmtShift(sh[1])} mm`;
  const note = st.note || undefined;
  if (st.applied === false) return ["stage shift off", "", note];
  if (st.source === "missing") return ["stage: missing, (0, 0) mm used", "warn", note];
  if (st.source === "error") return [`stage: read failed, last shift kept (${shift})`, "warn", note];
  if (st.source === "none") return ["stage: no reading yet, (0, 0) mm used", "muted", note];
  const from = { odb: "XY table", file: "the file's begin-of-run ODB", manual: "set by hand" }[st.source] ||
    String(st.source);
  return [`stage shift ${shift} (${from})`, "", note];
}

/** The MuPix x/y note, redrawn only when its text changed. */
function renderXy(s) {
  const head = state.xyHead;
  if (!head) return;
  const parts = xyNoteParts(s ? s.xy : null);
  const sig = JSON.stringify(parts);
  if (sig === state.xyNoteSig) return;
  state.xyNoteSig = sig;
  drawNote(head._note, parts);
}

// ---------------------------------------------------------------------------
// MuPix pairs (unseeded)
// ---------------------------------------------------------------------------

/** The pairs tab's head: a one-line note from summary.pairs and the log-z switch for the pair maps. */
function pairHead() {
  if (state.pairHead) return state.pairHead;
  const note = el("div", { class: "dqm-sma-xynote", id: "dqm-sma-pairnote" }, "MuPix pairs: …");
  const tools = el("div", { class: "dqm-sma-xytools" },
    checkbox("dqm-sma-pairlogz", "log z for the pair maps", state.pairLogZ,
      function (v) { state.pairLogZ = v; save(); redrawVisible(); }));
  state.pairHead = el("div", { class: "dqm-sma-xyhead", id: "dqm-sma-pairhead" }, note, tools);
  state.pairHead._note = note;
  if (state.summary) renderPairs(state.summary);
  return state.pairHead;
}

/** A plain number with `digits` decimals, or a dash. */
function fixed(x, digits) {
  return typeof x === "number" && Number.isFinite(x) ? x.toFixed(digits) : "—";
}

/**
 * The pairs note's parts, as xyNoteParts: [[text, kind, title], ...], from
 * summary.pairs alone.
 */
function pairNoteParts(pr) {
  const label = ["MuPix pairs:", "label"];
  if (!pr) return [label, ["no pairs summary from the analyzer", "warn"]];
  if (!pr.enabled) {
    return [label, [`pairs off: ${pr.off_reason || "MuPix/Pairs/enable = n, or the MuPix analysis is off"}`,
                    pr.off_reason && !/enable = n/.test(pr.off_reason) ? "warn" : ""]];
  }
  const c = pr.cuts || {};
  const w = pairWindow(pr);
  const win = !w ? "—" : w[0] === -w[1] ? `±${w[1]} ns` : `[${w[0]}, ${w[1]}] ns`;
  const parts = [label];
  parts.push([`${pct(pr.paired_frac, 1)} of ${num(pr.n_l1)} L1 pixels paired`, "",
              "L1 pixels with an L2 pixel within the window; each takes the nearest in time, " +
              "ties to the earlier, and one L2 pixel may serve several L1 pixels"]);
  parts.push([`mean partners ${fixed(pr.mean_partners, 2)}`, "",
              "L2 pixels within the window per paired L1 pixel: 1 is no ambiguity " +
              "(every sampled L1 pixel is in mupix_pair_partners)"]);
  parts.push([`window ${win} · at most ${num(pr.max_l1)} L1 pixels per frame`, ""]);
  if (pr.hits) {
    const cap = pr.max_hits ? ` (at most ${num(pr.max_hits)} a plane per frame)` : "";
    parts.push([`L1 / L2 alone: ${num(pr.hits.n_l1)} / ${num(pr.hits.n_l2)} pixels${cap}`, "",
                "The top row: every pixel of each plane on the sensor, paired or not, with no S1 " +
                "(an even sample per frame); the phase-space tab's hit maps are S1-gated"]);
  }
  const unit = c.tot_ns ? ` (×${c.tot_ns} ns)` : "";
  parts.push([`light ${pct(pr.light_frac, 1)}, heavy ${pct(pr.heavy_frac, 1)} of the pairs ` +
              `(light: both pixels' ToT ${lightCut(c)}${unit} · heavy: ≥ ${c.tot_heavy_min})`, ""]);
  parts.push(stagePart(pr.stage || {}));
  if (pr.resets) parts.push([`maps reset ${pr.resets}× by Pairs or ToT-cut edits`, "muted"]);
  parts.push(["pixel pairs, not particles", "muted",
              "No clustering and no S1: a particle that fires several pixels gives several pairs, " +
              "as on the nearline (track_xy, xxp_central, yyp_central)"]);
  return parts;
}

/** The pairs note, redrawn only when its text changed. */
function renderPairs(s) {
  const head = state.pairHead;
  if (!head) return;
  const parts = pairNoteParts(s ? s.pairs : null);
  const sig = JSON.stringify(parts);
  if (sig === state.pairNoteSig) return;
  state.pairNoteSig = sig;
  drawNote(head._note, parts);
}

/** A note from its parts: "label part · part · ...", a hover text on the parts that have one. */
function drawNote(note, parts) {
  note.innerHTML = "";
  parts.forEach(function ([text, kind, title], i) {
    if (i > 1) note.appendChild(document.createTextNode(" · "));
    else if (i === 1) note.appendChild(document.createTextNode(" "));
    const span = el("span", { class: kind ? `dqm-sma-xy-${kind}` : "" }, text);
    if (title) { span.setAttribute("title", title); span.classList.add("dqm-sma-xy-tip"); }
    note.appendChild(span);
  });
}

function guard(id, fn) {
  try { fn(); } catch (e) {
    // The block's DOM is replaced by the error, so whatever was cached about
    // it has to be rebuilt on the next good summary.
    if (id === "dqm-sma-table") state.table = null;
    if (id === "dqm-sma-xynote") state.xyNoteSig = null;
    if (id === "dqm-sma-pairnote") state.pairNoteSig = null;
    if (id === "dqm-sma-flags") state.flagsSig = null;
    if (id === "dqm-sma-banner") state.banners = {};
    if (id === "dqm-sma-nim") state.nimSig = null;
    const node = document.getElementById(id);
    if (node) {
      node.innerHTML = "";
      node.appendChild(el("div", { class: "dqm-error" },
        `could not draw this block: ${BRPC.errorText(e)}`));
    }
    if (typeof console !== "undefined") console.error("dqm-sma", id, e);
  }
}

function chip(text, cls) {
  return el("span", { class: `dqm-chip ${cls || ""}`.trim() }, text);
}

function renderStatus(s, st) {
  const bar = document.getElementById("dqm-sma-status");
  bar.innerHTML = "";
  if (!s) { bar.appendChild(chip("no summary", "yellow")); return; }
  const f = s.frames || {};
  const w = f.window || {};

  if (st && st.throttled) bar.appendChild(chip("analyzer throttled", "red"));
  else bar.appendChild(chip(`${state.client} connected`, "green"));

  const runActive = st ? st.run_active : s.run_active;
  bar.appendChild(chip(s.run !== null && s.run !== undefined
    ? `run ${s.run}${runActive === false ? " (stopped)" : ""}` : "no run",
    runActive === false ? "yellow" : ""));

  const offered = f.offered !== undefined ? f.offered : f.seen_by_serial;
  // Frames of nothing but zero words (a run start) are dropped undecoded.
  bar.appendChild(chip(`frames ${num(f.processed)} analysed · ${num(offered)} offered · ` +
                       `${num(f.stale)} stale` + (f.zero ? ` · ${num(f.zero)} zero` : "")));
  if (f.suspect) bar.appendChild(chip(`${num(f.suspect)} suspect time base`, "red"));

  // Frames the analyzer skipped on purpose (the CPU budget) are information,
  // not a fault: every rate and fraction is per analysed frame. Only when it
  // was told to take every frame ("process all") or a fixed rate is a serial
  // gap a frame lost, and worth a yellow chip.
  const smp = sampling(s, st);
  if (smp.lossMode) {
    if (f.missed_by_serial) {
      bar.appendChild(chip(`${num(f.missed_by_serial)} missed (serial gaps)`, "yellow"));
    }
  } else if (smp.mode === "cpu budget" || (smp.analysed !== null && smp.analysed < FULL)) {
    if (smp.analysed !== null) {
      bar.appendChild(chip(`analysed ${fracText(smp.analysed)} of frames${smp.since}`, "blue"));
    }
  }
  if (smp.mode === "cpu budget" && smp.budget !== null) {
    const over = smp.cpu !== null && smp.cpu > 1.1 * smp.budget;
    bar.appendChild(chip(`CPU ${smp.cpu === null ? "—" : `${smp.cpu.toFixed(1)} %`} / ` +
                         `budget ${fmtPct(smp.budget)}`, over ? "yellow" : ""));
  }

  const perS = st && typeof st.processed_per_s === "number" ? st.processed_per_s : w.per_s;
  const perSText = perS === null || perS === undefined ? "—" : perS.toFixed(1);
  bar.appendChild(chip(smp.offeredPerS === null ? `${perSText} frames/s`
    : `${perSText} analysed / ${smp.offeredPerS.toFixed(1)} offered frames/s`));
  bar.appendChild(chip(`live ${pct(s.live_fraction, 1)}`));
  if (typeof f.last_age_s === "number" && f.last_age_s > 5) {
    bar.appendChild(chip(`last frame ${Math.round(f.last_age_s)} s ago`, "yellow"));
  }

  if (s.nim && s.nim.active) bar.appendChild(mergeChip(s));
  const sh = s.shift || {};
  const verdictCls = { ok: "green", mismatch: "red", "no fit": "yellow" }[sh.verdict] || "";
  const best = sh.best !== null && sh.best !== undefined && sh.best !== sh.configured
    ? `, best ${sh.best}` : "";
  bar.appendChild(chip(`shift ${sh.configured}: ${sh.verdict || "?"}${best}`, verdictCls));
  bar.appendChild(el("span", { class: "dqm-footnote" },
    `averaged over the last ${Math.round(s.window_s || 0)} s`));
}

/** Below this analysed fraction the page says it is looking at a sample. */
const FULL = 0.999;

/**
 * How the analyzer is sampling: from the summary, else from dqm::status.
 *
 * Every field can be null (a plugin run outside the analyzer has no mode, a
 * window with nothing offered has no fraction). lossMode: the analyzer was
 * asked for every frame or a fixed rate, so a serial gap is a frame lost
 * rather than one skipped on purpose.
 */
function sampling(s, st) {
  const sm = (s && s.sampling) || {};
  const f = (s && s.frames) || {};
  const w = f.window || {};
  const first = (...xs) => {
    for (const x of xs) if (typeof x === "number" && Number.isFinite(x)) return x;
    return null;
  };
  const mode = sm.mode || (st && st.sampling_mode) || null;
  // Over the summary window; with nothing offered in it (a stopped run), the
  // share since the analyzer started, said so.
  const recent = first(sm.analysed_frac, w.analysed_frac);
  const total = first(f.analysed_frac, st && st.plugin && st.plugin.analysed_frac);
  return {
    mode,
    lossMode: mode === "process all" || mode === "rate",
    analysed: recent !== null ? recent : total,
    since: recent === null && total !== null ? " since start" : "",
    offeredPerS: first(sm.offered_per_s),
    s1: first(sm.s1_analysed_frac),
    maxS1: first(sm.max_s1_per_frame, st && st.plugin && st.plugin.max_s1_per_frame),
    budget: first(st && st.cpu_budget_pct, sm.cpu_budget_pct),
    cpu: first(st && st.cpu_pct, sm.cpu_pct),
  };
}

/** A fraction as a whole percent, with one decimal below 10 %. */
function fracText(x) {
  return pct(x, x !== null && x !== undefined && x < 0.1 ? 1 : 0);
}

/** A number that already is a percent, e.g. a budget of 20 -> "20 %". */
function fmtPct(x) {
  return x === null || x === undefined || !Number.isFinite(x) ? "—" : `${Number(x.toPrecision(4))} %`;
}

/**
 * One line under the status chips when the histograms are from a sample: of
 * the frames (the CPU budget), or of a frame's S1 hits (Cuts/max S1 per
 * frame). Hidden when the analyzer takes everything.
 */
function renderSampling(s, st) {
  const note = document.getElementById("dqm-sma-sampling");
  const smp = sampling(s, st);
  const framesCut = smp.analysed !== null && smp.analysed < FULL;
  const s1Cut = smp.s1 !== null && smp.s1 < FULL;
  let text = "";
  if (s && (framesCut || s1Cut)) {
    const parts = [];
    if (smp.analysed !== null) {
      parts.push(`${fracText(smp.analysed)} of frames${smp.since}` +
        (smp.mode === "cpu budget" && smp.budget !== null
          ? ` (CPU budget ${fmtPct(smp.budget)} of a core)` : ""));
    }
    if (s1Cut) {
      parts.push(smp.maxS1 !== null
        ? `S1-seeded plots use at most ${num(smp.maxS1)} S1 hits per frame (${pct(smp.s1, 1)} of S1 hits)`
        : `S1-seeded plots use ${pct(smp.s1, 1)} of S1 hits`);
    }
    text = `Counts are from the analysed sample: ${parts.join("; ")}. ` +
           "Rates, fractions and efficiencies are unbiased.";
  }
  if (note.textContent !== text) note.textContent = text;
  note.style.display = text ? "" : "none";
}

/**
 * The banners: a coarse-shift mismatch and a broken time base. The fine/coarse
 * mismatch fractions, the stale-frame rule and the event display's mismatch
 * badges all assume the configured shift, so a wrong one makes half the page
 * lie. Each is a full-width red block with the fix spelled out, above
 * everything else, rather than one flag among many.
 *
 * Updated in place, not rebuilt: the text changes every second (the word
 * count), and a banner rebuilt under the pointer swallows the click on its
 * button.
 */
function renderBanner(s) {
  const holder = document.getElementById("dqm-sma-banner");
  const sh = s && s.shift;
  const flags = (s && s.flags) || [];
  const want = [];
  if (sh && sh.verdict === "mismatch") {
    const frac = {};
    (sh.scan || []).forEach((k, i) => { frac[k] = sh.fractions ? sh.fractions[i] : null; });
    want.push(["dqm-sma-shift-banner",
      `Coarse shift ${sh.configured} configured, ${sh.best} fits better ` +
      `(${pct(frac[sh.best], 1)} vs ${pct(frac[sh.configured], 1)} of ${num(sh.words)} S1 words ` +
      `in ${Math.round(sh.window_s || 0)} s) — set /DQM/SMA/Coarse shift = ${sh.best}`]);
  }
  // A broken time base is the same class of problem -- every time-derived
  // number on the page is suspect -- so it gets the same treatment, with the
  // analyzer's own wording.
  let n = 0;
  for (const f of flags) {
    if (f.code !== "time_base") continue;
    want.push([n ? `dqm-sma-timebase-banner-${n}` : "dqm-sma-timebase-banner", `Time base: ${f.text}`]);
    n++;
  }

  const have = state.banners;
  for (const id of Object.keys(have)) {
    if (!want.some(([w]) => w === id)) { have[id].wrap.remove(); delete have[id]; }
  }
  for (const [id, text] of want) {
    let b = have[id];
    if (!b) {
      const wrap = el("div", { class: "dqm-sma-banner", id, role: "alert" });
      const span = el("span", {}, "");
      const btn = el("button", { class: "mbutton" }, "Edit Coarse shift…");
      // The stock MIDAS editor, not a one-click "fix": the shift is a board
      // setting for the run, and a person should look at the numbers first.
      btn.onclick = function () { dlgOdbEdit("/DQM/SMA/Coarse shift"); };
      wrap.appendChild(span);
      wrap.appendChild(btn);
      holder.appendChild(wrap);
      b = have[id] = { wrap, span };
    }
    if (b.span.textContent !== text) b.span.textContent = text;
  }
  holder.style.display = want.length ? "" : "none";
}

const BANNER_CODES = ["shift_mismatch", "time_base"];

function renderFlags(s) {
  const holder = document.getElementById("dqm-sma-flags");
  const all = (s && s.flags) || [];
  const only = state.view.flags;
  const others = all.filter((f) => BANNER_CODES.indexOf(f.code) < 0);
  const flags = only ? others.filter((f) => only.test(f.code)) : others;
  // On the MuPix page the flags it leaves out are counted, not hidden silently.
  // "the SMA plots page", not its menu key: pinky registers it as WDSMAPlots.
  const hidden = others.length - flags.length;
  const order = { error: 0, warn: 1, info: 2 };
  flags.sort((a, b) => (order[a.severity] ?? 3) - (order[b.severity] ?? 3));
  const sig = JSON.stringify([flags, all.length, hidden]);
  if (sig === state.flagsSig) return;           // unchanged: leave the DOM alone
  state.flagsSig = sig;
  holder.innerHTML = "";
  const more = hidden ? ` ${hidden} other flag${hidden > 1 ? "s" : ""} on the SMA plots page.` : "";
  if (!flags.length) {
    holder.appendChild(el("div", { class: "dqm-diagnosis" },
      only ? `No MuPix flags.${more}` : all.length ? "No other flags." : "No flags."));
    return;
  }
  for (const f of flags) {
    const cls = { error: "red", warn: "yellow" }[f.severity] || "blue";
    const div = el("div", { class: `dqm-diagnosis ${cls} dqm-sma-flag`, "data-code": f.code }, f.text);
    // The TOT + NIM and the MuPix flags point at the tab with the numbers and
    // plots behind them (the MuPix ones are all about the readout and time sync).
    const to = /^nim_/.test(f.code) ? "nim" : /^mupix_/.test(f.code) ? "mupix" : null;
    if (to && state.tabs.some(([id]) => id === to)) {
      const label = TABS.find(([id]) => id === to)[1];   // the full label
      const go = el("button", { type: "button", class: "dqm-sma-flaglink" }, `${label} tab ›`);
      go.onclick = function () { showTab(to); };
      div.appendChild(document.createTextNode(" "));
      div.appendChild(go);
    }
    holder.appendChild(div);
  }
  if (more) holder.appendChild(el("div", { class: "dqm-footnote dqm-sma-flagsmore" }, more.trim()));
}

/** The channel a flag is about: its "ch" field, else "(ch N)" in the text. */
function flagChannel(f) {
  if (typeof f.ch === "number") return f.ch;
  const m = /\(ch (\d+)\)/.exec(f.text || "");
  return m ? Number(m[1]) : null;
}

const FLAG_COLUMN = { mismatch: "mismatch", tot_corrupt: "tot", efficiency_drop: "eff" };
const COLUMNS = [["ch", ""], ["label", "label"], ["role", "label"], ["rate", ""],
                 ["hits/frame", ""], ["ToT ≥ 250", ""], ["fine/coarse mismatch", ""],
                 ["stale words", ""], ["eff. given S1", ""]];

/**
 * The per-channel table, updated cell by cell.
 *
 * Rebuilt only when the number of channels changes. Rebuilding every second
 * made a hover tooltip (the n/a reason) vanish before it could be read.
 */
function renderTable(s) {
  const holder = document.getElementById("dqm-sma-table");
  if (!holder) return;                        // a view without the table (the MuPix page)
  if (!s || !s.channels) { holder.innerHTML = ""; state.table = null; return; }

  const marks = {};                           // "ch:column" -> warn|alarm
  // Only channels the analyzer judges ("flagged": S1, the counters, RF) can
  // carry a per-channel flag; the check keeps a stray "(ch N)" in some other
  // flag's text from colouring an unrelated row.
  const judged = new Set(s.channels.filter((c) => c && c.flagged !== false).map((c) => c.ch));
  for (const f of s.flags || []) {
    const col = FLAG_COLUMN[f.code];
    const ch = flagChannel(f);
    if (!col || ch === null || !judged.has(ch)) continue;
    const cls = f.severity === "error" ? "alarm" : "warn";
    const k = `${ch}:${col}`;
    if (marks[k] !== "alarm") marks[k] = cls;
  }
  const eff = {};
  for (const e of s.efficiency || []) {
    if (e && typeof e.ch === "number") eff[e.ch] = e;
  }
  // A NIM copy's row: its counter's NIM line (pair efficiency, lag vote), and
  // the NIM flags about it colour its cells.
  const nimRow = {};
  for (const r of (s.nim && s.nim.counters) || []) {
    nimRow[r.nim_ch] = r;
    for (const f of nimFlagsOf(s, r)) {
      const col = { nim_pairing: "eff", nim_missing: "role", nim_lag: "role" }[f.code];
      if (!col) continue;
      const k = `${r.nim_ch}:${col}`;
      if (marks[k] !== "alarm") marks[k] = f.severity === "error" ? "alarm" : "warn";
    }
  }

  if (!state.table || state.table.rows.length !== s.channels.length) {
    holder.innerHTML = "";
    const t = el("table", { class: "dqm-table dqm-sma-table" });
    const head = el("tr", {});
    for (const [h, cls] of COLUMNS) head.appendChild(el("th", cls ? { class: cls } : {}, h));
    t.appendChild(head);
    const rows = s.channels.map(function () {
      const tr = el("tr", {});
      const tds = COLUMNS.map(() => { const td = el("td", {}); tr.appendChild(td); return td; });
      t.appendChild(tr);
      return { tr, tds };
    });
    const foot = el("div", { class: "dqm-footnote" }, "");
    const repair = el("div", { class: "dqm-footnote dqm-sma-repair" }, "");
    holder.appendChild(t);
    holder.appendChild(foot);
    holder.appendChild(repair);
    state.table = { rows, foot, repair };
  }

  s.channels.forEach(function (c, i) {
    const { tr, tds } = state.table.rows[i];
    setAttr(tr, "data-ch", String(c.ch));
    setClass(tr, !c.role && !c.hits ? "dqm-sma-quiet" : "");
    const mark = (col) => (col && marks[`${c.ch}:${col}`]) || "";
    const put = (k, text, col, extra, title) =>
      setCell(tds[k], text, [COLUMNS[k][1], extra || "", mark(col)].filter(Boolean).join(" "), title);

    put(0, String(c.ch));
    put(1, c.label || "");
    const nr = c.role === "nim" ? nimRow[c.ch] : null;
    if (nr) {
      const lg = nr.lag || {};
      put(2, `${roleText(c, eff[c.ch])}${lg.last_state === "faulted" && !lg.na ? " · lag FAULTED" : ""}`, "role",
          "", `NIM copy of ${labelOf(c.pair_of)}; lag vote: ${lagText(lg)}`);
    } else {
      put(2, roleText(c, eff[c.ch]));
    }
    put(3, fmtRate(c.rate_hz));
    put(4, c.hits_per_frame === null || c.hits_per_frame === undefined ? "—" : c.hits_per_frame.toFixed(1));
    put(5, pct(c.tot_ge250_frac, 2), "tot");
    if (c.flagged === false && c.mismatch_frac !== null && c.mismatch_frac !== undefined && c.hits) {
      // Not S1, a counter or RF (no role, the current, a delayed channel): the
      // analyzer does not judge its fine/coarse agreement, so a 100 % here is
      // not a fault and must not read like one.
      put(6, `${pct(c.mismatch_frac, 1)} (not flagged)`, null, "dqm-sma-na",
          "Not judged: only S1, the counters and RF are checked for fine/coarse mismatch");
    } else if (offsetVerdict(c)) {
      const v = offsetVerdict(c);
      put(6, v.cell, "mismatch", "", v.line);
    } else {
      put(6, pct(c.mismatch_frac, 1), "mismatch");
    }
    put(7, num(c.stale));
    // S1 given S1 is 1 by definition, so it gets a dash rather than a
    // reassuring 100 % that measures nothing. A counter whose efficiency the
    // analyzer declines to quote (a timestamp fault makes the window
    // meaningless) says n/a and why, inline and in full on hover.
    const e = eff[c.ch];
    if (nr && (nr.pair_eff === null || nr.pair_eff === undefined) && nr.pair_eff_reason) {
      // Withheld: a timestamp fault on either channel of the pair.
      put(8, "pair n/a (timestamp fault)", "eff", "dqm-sma-na", nr.pair_eff_reason);
    } else if (nr) {
      // Not an efficiency given S1: the share of its counter's TOT words it pairs with.
      put(8, `pair ${pct(nr.pair_eff, 1)}`, "eff", "", "TOT + NIM pair efficiency: the share of " +
          `${labelOf(nr.ch)}'s TOT words with this NIM copy within ±${s.nim.pair_window_ns} ns`);
    } else if (!e || c.role === "s1" || e.counter === "S1") {
      put(8, "—", "eff");
    } else if (e.eff === null || e.eff === undefined) {
      const why = e.reason || e.note || e.why || "";
      const short = why ? String(why).split(":")[0] : "";
      put(8, short ? `n/a (${short})` : "n/a", "eff", "dqm-sma-na", why ? String(why) : null);
    } else {
      put(8, pct(e.eff, 1), "eff");
    }
  });

  const rf = s.rf || {};
  const foot = `RF: valid gate for ${pct(rf.valid_frac, 1)} of ${num(rf.n_s1)} S1 hits, vetoed ` +
    `${pct(rf.vetoed_frac, 1)} · rates are hits per second of covered time ` +
    `(${secs(s.covered_s !== undefined ? s.covered_s : s.span_s)} s in the window)` +
    (s.efficiency_kind ? ` · efficiency: ${s.efficiency_kind}` : "");
  if (state.table.foot.textContent !== foot) state.table.foot.textContent = foot;
  renderRepair(s, state.table.repair);
}

const TIMES_TEXT = { ok: "times right", repaired: "times repaired", wrong: "times WRONG",
                     unknown: "times unknown" };

/** Why an epoch vote is undecided (the analyzer's repair.undecided), in words. */
function undecidedText(rp) {
  if (rp.undecided === "few") {
    return `not enough S1 coincidences yet (${Math.max(0, Math.round(rp.excess || 0))} of ${rp.min_votes})`;
  }
  if (rp.undecided === "short") {
    return `frames shorter than the coarse offset (${rp.frame_ms} ms vs ${rp.offset_ms} ms)`;
  }
  return "no whole-epoch shift brings its hits to S1";
}

/**
 * A channel's coarse-offset verdict (the analyzer's kind "coarse_offset"):
 * the mismatch cell's text and one line with its epoch vote, the evidence
 * that the repair makes nothing up. Candidates are labelled by the correction
 * they apply to the word times (−1: one epoch earlier), as the flags say it.
 * null for any other channel.
 */
function offsetVerdict(c) {
  if (!c || c.kind !== "coarse_offset") return null;
  const o = c.offset_ticks;
  const off = o === null || o === undefined ? "?" : `${o > 0 ? "+" : ""}${Math.round(o)} ticks`;
  const times = TIMES_TEXT[c.times] || String(c.times);
  const rp = c.repair;
  let vote = "no nominal delay to S1 for this channel (NIM/lag nominal ns): its epochs are " +
    "not voted on";
  if (rp) {
    const v = (rp.votes || []).map(([k, n, x, e]) =>
      `${epochsText(k)}: ${kilo(n)} in / ${kilo(Math.round(x))} off of ${kilo(e)}`).join(" · ");
    vote = `S1 coincidences per correction (in time / off time of words exposed): ${v || "none yet"}` +
      (rp.correction === null || rp.correction === undefined
        ? ` → undecided: ${undecidedText(rp)}`
        : ` → ${epochsText(rp.correction)}; since then only its neighbours count`);
  }
  let moved = "";
  if (rp && rp.moved) {
    moved = `, ${pct(rp.moved_share, 1)} of hits moved ${epochsText(rp.moved_by)}`;
  } else if (rp && rp.would_move) {
    moved = `, ${pct(rp.would_move_share, 1)} of hits whole epochs off (not repaired)`;
  }
  const line = `${c.label} (ch ${c.ch}): coarse field ${off} from fine (R ${c.R}), ${times}` +
    `${moved}. ${vote}`;
  return { cell: `${pct(c.mismatch_frac, 1)} · coarse ${off} · ${times}`, line };
}

/** A correction in whole epochs as the flags say it: "−1 epoch", "0 epochs", "+2 epochs". */
function epochsText(k) {
  return `${signed(k)} epoch${Math.abs(k) === 1 ? "" : "s"}`;
}

/** Under the table: the epoch repair's switch and each coarse-offset channel's vote. */
function renderRepair(s, holder) {
  if (!holder) return;
  const lines = (s.channels || []).map(offsetVerdict).filter(Boolean).map((v) => v.line);
  const er = s.epoch_repair;
  if (lines.length && er) {
    lines.unshift(`Epoch repair (Cuts/epoch repair) ${er.enabled ? "on" : "OFF"}: a channel ` +
      "whose coarse field is offset from its fine field gets its times moved by whole 2^20 ns " +
      "epochs, chosen by its S1 coincidences in time against an off-time sideband (words " +
      "whose moved time falls inside the frame only).");
  }
  const text = lines.join("\n");
  if (holder.getAttribute("data-text") === text) return;
  holder.setAttribute("data-text", text);
  holder.innerHTML = "";
  for (const l of lines) holder.appendChild(el("div", {}, l));
}

function kilo(n) {
  if (n === null || n === undefined) return "—";
  for (const [d, u] of [[1e6, "M"], [1e3, "k"]]) {
    if (n >= d) return `${(n / d).toFixed(n / d < 10 ? 1 : 0)}${u}`;
  }
  return String(n);
}

function signed(k) {
  return k > 0 ? `+${k}` : k < 0 ? `−${-k}` : "0";
}

function setCell(td, text, cls, title) {
  if (td.textContent !== text) td.textContent = text;
  setClass(td, cls);
  if (title) setAttr(td, "title", title);
  else if (td.getAttribute("title")) td.removeAttribute("title");
}

function setClass(node, cls) { if (node.className !== cls) node.className = cls; }

function setAttr(node, k, v) { if (node.getAttribute(k) !== v) node.setAttribute(k, v); }

function roleText(c, e) {
  if (c.role === "s1") return "S1 (seed)";
  if (c.role === "nim") {
    return typeof c.pair_of === "number" ? `NIM copy of ${labelOf(c.pair_of)}` : "NIM copy";
  }
  if (e && e.counter) return e.counter;
  return c.role || "";
}

function tabError(tab, e) {
  const note = document.getElementById(`dqm-sma-tabnote-${tab}`);
  if (note) {
    note.className = "dqm-diagnosis red";
    note.textContent = `Could not update this tab: ${BRPC.errorText(e)}`;
  }
}

/** A successful poll: whatever went wrong before is over, so stop saying so. */
function tabOk(tab) {
  const note = document.getElementById(`dqm-sma-tabnote-${tab}`);
  if (note && note.textContent) { note.className = ""; note.textContent = ""; }
}

/**
 * The analyzer stopped answering. The table, flags and banners still hold the
 * last summary, which is worth keeping on screen (it is what was true when it
 * went away) but must not read as current -- "No flags." from a dead analyzer
 * is the most misleading thing this page could show. Greyed, and dated.
 */
function setStale(stale) {
  for (const id of ["dqm-sma-banner", "dqm-sma-sampling", "dqm-sma-flags", "dqm-sma-table",
                    ...state.tabs.map(([t]) => `dqm-sma-pane-${t}`)]) {
    const node = document.getElementById(id);
    if (node) node.classList.toggle("dqm-sma-stale", stale);
  }
}

function noAnalyzer(e) {
  const bar = document.getElementById("dqm-sma-status");
  if (!bar) return;
  setStale(true);
  bar.innerHTML = "";
  bar.appendChild(chip("no analyzer", "red"));
  bar.appendChild(chip(state.lastOkAt
    ? `last seen ${clock(state.lastOkAt)} — everything below is from then`
    : "never answered", "yellow"));
  bar.appendChild(el("span", { class: "dqm-diagnosis red" },
    `'${state.client}' did not answer sma::summary (${BRPC.errorText(e)}). ` +
    "These plots are filled by the SMA analyzer client; start it with " +
    "mdqm-analyzer --plugin sma --client sma_analyzer. Retrying every 5 s."));
}

// ---------------------------------------------------------------------------
// Chrome
// ---------------------------------------------------------------------------

function build() {
  const r = document.getElementById("dqm-root");
  r.innerHTML = "";

  r.appendChild(el("div", { id: "dqm-sma-banner" }));
  r.appendChild(el("div", { class: "dqm-strip", id: "dqm-sma-status" }, chip("…")));
  const note = el("div", { id: "dqm-sma-sampling", class: "dqm-footnote dqm-sma-sampling" });
  note.style.display = "none";
  r.appendChild(note);
  r.appendChild(el("div", { id: "dqm-sma-flags", class: "dqm-sma-flags" }));
  if (state.view.table) r.appendChild(el("div", { id: "dqm-sma-table", class: "dqm-panel" }));

  // The page's controls on a row of their own above the tab strip, so they
  // do not read as more tabs.
  const bar = el("div", { class: "dqm-strip dqm-sma-toolbar", id: "dqm-sma-toolbar" });
  bar.appendChild(checkbox("dqm-sma-logy", "log y (1D)", state.logY,
    function (v) { state.logY = v; save(); redrawVisible(); }));
  bar.appendChild(checkbox("dqm-sma-logz", "log z (2D)", state.logZ,
    function (v) { state.logZ = v; save(); redrawVisible(); }));
  bar.appendChild(labelled("update", select(
    [["1000", "1 Hz"], ["2000", "0.5 Hz"], ["5000", "0.2 Hz"], ["0", "paused"]],
    String(state.intervalMs), function (v) { setInterval_(Number(v)); })));
  const clear = el("button", { class: "mbutton", id: "dqm-sma-clear" }, "Clear");
  clear.onclick = function () {
    dlgConfirm("Clear every accumulated SMA histogram? The trends and the summary are not " +
               "affected.", function (yes) {
      if (!yes) return;
      BRPC.json(state.client, "dqm::clear", "sma")
        .then(() => refreshTab(state.tab === "trends" ? "health" : state.tab))
        .catch((e) => tabError(state.tab, e));
    });
  };
  bar.appendChild(clear);
  r.appendChild(bar);
  r.appendChild(tabStrip(state.tabs, "dqm-sma", state.view.tabs ? "MuPix plots" : "SMA plots",
                         () => state.tab, showTab));

  for (const [id] of state.tabs) {
    const pane = el("div", { id: `dqm-sma-pane-${id}`, class: "dqm-sma-pane", role: "tabpanel",
                             "aria-labelledby": `dqm-sma-tab-${id}` });
    pane.appendChild(el("div", { id: `dqm-sma-tabnote-${id}` }));
    if (id === "mupix") pane.appendChild(el("div", { id: "dqm-sma-mupix", class: "dqm-panel" }));
    if (id === "nim") {
      pane.appendChild(el("div", { id: "dqm-sma-nim", class: "dqm-panel" }));
      // Built once, outside the per-second redraw, so a click on it is not lost.
      pane.appendChild(el("div", { class: "dqm-strip dqm-sma-nimtools" },
        checkbox("dqm-sma-nimlogy", "log y for the Δt plots (nim_dt, nim_dt_wide)", state.nimLogY,
          function (v) { state.nimLogY = v; save(); redrawVisible(); })));
    }
    pane.appendChild(el("div", { class: "dqm-grid", id: `dqm-sma-grid-${id}` }));
    r.appendChild(pane);
  }
}

// ---------------------------------------------------------------------------

function shortName(name) {
  return name.indexOf("/") >= 0 ? name.slice(name.indexOf("/") + 1) : name;
}

function num(x) {
  return x === null || x === undefined ? "—" : Number(x).toLocaleString();
}

function pct(x, digits) {
  if (x === null || x === undefined || !Number.isFinite(x)) return "—";
  return `${(x * 100).toFixed(digits === undefined ? 1 : digits)} %`;
}

function clock(ms) {
  const d = new Date(ms);
  return [d.getHours(), d.getMinutes(), d.getSeconds()].map((x) => String(x).padStart(2, "0")).join(":");
}

function secs(x) {
  return x === null || x === undefined || !Number.isFinite(x) ? "—" : x.toFixed(2);
}

function fmtRate(hz) {
  if (hz === null || hz === undefined || !Number.isFinite(hz)) return "—";
  if (hz >= 1e6) return `${(hz / 1e6).toFixed(2)} MHz`;
  if (hz >= 1e3) return `${(hz / 1e3).toFixed(1)} kHz`;
  return `${hz.toFixed(0)} Hz`;
}

/**
 * A tab strip (role=tablist) of one role=tab button per [id, label] in `tabs`,
 * with ids `${prefix}-tab-${id}` controlling the panes `${prefix}-pane-${id}`.
 * Left/Right (wrapping), Home and End select and focus the neighbouring tab.
 * `current()` says which tab is shown; `select(id)` shows one. markTabs()
 * then sets the selection state, so the strip always follows showTab().
 * An optional third entry is a short label, shown instead on a narrow window
 * (dqm-sma.css); the full one stays the tab's accessible name.
 */
function tabStrip(tabs, prefix, label, current, select) {
  const bar = el("div", { class: "dqm-sma-tablist", role: "tablist", "aria-label": label });
  for (const [id, text, short] of tabs) {
    const b = el("button", {
      type: "button", class: "dqm-sma-tab", id: `${prefix}-tab-${id}`, role: "tab",
      "aria-selected": "false", "aria-controls": `${prefix}-pane-${id}`, tabindex: "-1",
    }, short ? el("span", { class: "dqm-sma-tabfull" }, text) : text);
    if (short) {
      b.appendChild(el("span", { class: "dqm-sma-tabshort", "aria-hidden": "true" }, short));
      b.setAttribute("title", text);
    }
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
    window.localStorage.setItem(state.view.store, JSON.stringify({
      client: state.client, tab: state.tab, intervalMs: state.intervalMs,
      logY: state.logY, logZ: state.logZ, nimLogY: state.nimLogY, xyLogZ: state.xyLogZ,
      pairLogZ: state.pairLogZ, dtChips: state.dtChips, dtNorm: state.dtNorm, dtZoom: state.dtZoom,
    }));
  } catch (e) { /* private browsing or quota */ }
}

function restore() {
  try {
    const o = JSON.parse(window.localStorage.getItem(state.view.store) || "{}");
    if (o.client) state.client = o.client;
    if (o.tab && state.tabs.some(([id]) => id === o.tab)) state.tab = o.tab;
    if (o.intervalMs !== undefined) state.intervalMs = Number(o.intervalMs);
    if (o.logY !== undefined) state.logY = !!o.logY;
    if (o.logZ !== undefined) state.logZ = !!o.logZ;
    if (o.nimLogY !== undefined) state.nimLogY = !!o.nimLogY;
    if (o.xyLogZ !== undefined) state.xyLogZ = !!o.xyLogZ;
    if (o.pairLogZ !== undefined) state.pairLogZ = !!o.pairLogZ;
    if (Array.isArray(o.dtChips)) state.dtChips = o.dtChips.filter((c) => Number.isInteger(c));
    if (o.dtNorm !== undefined) state.dtNorm = !!o.dtNorm;
    if (o.dtZoom !== undefined) state.dtZoom = !!o.dtZoom;
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

function checkbox(id, text, checked, onChange) {
  const box = el("input", { type: "checkbox", id: id });
  box.checked = checked;
  box.onchange = function () { onChange(!!this.checked); };
  const lab = el("label", { for: id, class: "dqm-chip" }, text);
  lab.insertBefore(box, lab.firstChild);
  return lab;
}

})();
