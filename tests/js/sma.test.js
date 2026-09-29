//
// The SMA page (/Custom/SMAPlots), headless, against what the real plugin sends.
//
// tests/generate_sma_page_fixtures.py runs SmaPlugin over the stored real
// frames and saves its sma::summary, sma::trend and two encoded histograms; the
// transport below wraps them in brpc envelopes exactly as mhttpd hands them to
// the page, so BRPC.call, the JSON decoding and the histogram decoder all run
// for real.
//

const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");

const { El, runPage } = require("./domstub.js");

const PAGES = path.join(__dirname, "..", "..", "pages", "js");
globalThis.DQM = require(path.join(PAGES, "dqm-common.js"));
globalThis.BRPC = require(path.join(PAGES, "dqm-brpc.js"));
globalThis.DQMHeatmap = require(path.join(PAGES, "dqm-heatmap.js"));

const SMA = path.join(PAGES, "dqm-sma.js");
const FX = JSON.parse(fs.readFileSync(path.join(__dirname, "sma-summary-fixture.json"), "utf8"));

function envelope(tag, bytes) {
  const out = new Uint8Array(8 + bytes.length);
  new DataView(out.buffer).setUint32(0, out.length, true);
  for (let i = 0; i < 4; i++) out[4 + i] = tag.charCodeAt(i);
  out.set(bytes, 8);
  return out.buffer;
}
const json = (o) => envelope("json", new TextEncoder().encode(JSON.stringify(o)));
const clone = (o) => JSON.parse(JSON.stringify(o));

/**
 * An analyzer. `over` replaces any command's handler: (args) -> ArrayBuffer,
 * or throws to make that one call fail.
 */
function analyzer(over = {}) {
  const calls = [];
  const handlers = Object.assign({
    "sma::summary": () => json(FX.summary),
    "dqm::status": () => json(FX.status),
    "sma::trend": () => json(FX.trend),
    "dqm::list": () => envelope("list", new TextEncoder().encode(FX.hist_names.join("\n"))),
    "dqm::metadata": (name) => json({ name, title: `title of ${name}`,
                                      axes: [{ title: "x axis" }, { title: "y axis" }] }),
    "dqm::histogram": (name) => {
      const dims = FX.hist_dims[name];
      const hex = FX.hists[name] ||
        (dims === 2 ? FX.hists["sma/tot_vs_ch_lsb0"] : FX.hists["sma/words_per_ch"]);
      return envelope("hist", Buffer.from(hex, "hex"));
    },
    "dqm::clear": () => json({ cleared: 45 }),
  }, over);
  const brpc = (params) => {
    calls.push({ cmd: params.cmd, args: params.args });
    const h = handlers[params.cmd];
    if (!h) return Promise.resolve(envelope("err ", new TextEncoder().encode("unknown")));
    try { return Promise.resolve(h(params.args)); } catch (e) { return Promise.reject(e); }
  };
  return { calls, brpc, handlers };
}

async function settle(page, rounds = 4) {
  for (let r = 0; r < rounds; r++) {
    page.flushTimers();
    for (let i = 0; i < 30; i++) await new Promise((res) => setImmediate(res));
  }
}

async function boot(over, stored, before) {
  globalThis.__alerts = [];
  globalThis.localStorage = {
    _d: stored ? { "dqm-sma-settings": JSON.stringify(stored) } : {},
    getItem(k) { return this._d[k] || null; }, setItem(k, v) { this._d[k] = v; },
  };
  const an = analyzer(over);
  const page = runPage(SMA, { brpc: an.brpc });
  if (before) before(page);
  globalThis.dlgConfirm = (text, cb) => { page.calls.push({ method: "dlgConfirm", params: text }); cb(true); };
  await page.load();
  await settle(page);
  page.an = an;
  return page;
}

const byId = (page, id) => page.doc.getElementById(id);
const histCalls = (page, from = 0) =>
  page.an.calls.slice(from).filter((c) => c.cmd === "dqm::histogram").map((c) => c.args);

// ---------------------------------------------------------------------------

test("the summary table has a row for every channel, labelled from the summary", async () => {
  const page = await boot();
  const table = byId(page, "dqm-sma-table");
  const rows = table.findAll((e) => e.tagName === "TR" && e.attrs["data-ch"] !== undefined);
  assert.strictEqual(rows.length, 16);
  FX.summary.channels.forEach(function (c, i) {
    const cells = rows[i].byTag("td").map((td) => td.textContent);
    assert.strictEqual(cells[0], String(c.ch));
    assert.strictEqual(cells[1], c.label, `ch ${c.ch} label`);
  });
  // S1 is ch 1 with 5.6 kHz in the fixture.
  const s1 = rows[1].byTag("td").map((td) => td.textContent);
  assert.ok(/kHz/.test(s1[3]), s1.join(" | "));
  assert.deepStrictEqual(globalThis.__alerts, []);
});

test("labels are the ODB's, not built in", async () => {
  const s = clone(FX.summary);
  s.channels[3].label = "Scint-3 top";
  s.efficiency[2].label = "Scint-3 top";
  const page = await boot({ "sma::summary": () => json(s) });
  const row = byId(page, "dqm-sma-table").find((e) => e.attrs && e.attrs["data-ch"] === "3");
  assert.strictEqual(row.byTag("td")[1].textContent, "Scint-3 top");
});

test("flagged cells are coloured by severity", async () => {
  const page = await boot();
  const row = (ch) => byId(page, "dqm-sma-table").find((e) => e.attrs && e.attrs["data-ch"] === String(ch));
  const mismatchCell = (ch) => row(ch).byTag("td")[6];
  // The fixture's flags: S5 (ch 5) error, S3 (ch 3) warn.
  assert.ok(mismatchCell(5).classList.contains("alarm"), "S5 93 % mismatch is an error");
  assert.ok(mismatchCell(3).classList.contains("warn"), "S3 12.6 % mismatch is a warning");
  assert.ok(!mismatchCell(1).classList.contains("warn") && !mismatchCell(1).classList.contains("alarm"));
  // S1-conditional efficiency for S2..S5, and a dash for S1 itself.
  assert.strictEqual(row(1).byTag("td")[8].textContent, "—");
  assert.strictEqual(row(2).byTag("td")[8].textContent, "69.1 %");
  // A counter with a timestamp fault gets no efficiency, and says why.
  const s5 = row(5).byTag("td")[8];
  assert.strictEqual(s5.textContent, "n/a (timestamp fault)", "the reason inline, in short");
  assert.ok(/timestamp fault/.test(s5.getAttribute("title")), s5.getAttribute("title"));
});

test("the flags are listed worst first", async () => {
  const page = await boot();
  const flags = byId(page, "dqm-sma-flags").byClass("dqm-sma-flag");
  assert.strictEqual(flags.length, FX.summary.flags.length);
  assert.ok(flags[0].classList.contains("red"), "the error comes first");
  assert.ok(/S5/.test(flags[0].textContent));
});

test("no shift banner when the shift fits", async () => {
  const page = await boot();
  assert.strictEqual(byId(page, "dqm-sma-shift-banner"), null);
  assert.ok(/shift 14: ok/.test(byId(page, "dqm-sma-status").textContent));
});

test("a shift mismatch is a banner that says what to set", async () => {
  const page = await boot({ "sma::summary": () => json(FX.summary_shift13) });
  const banner = byId(page, "dqm-sma-shift-banner");
  assert.ok(banner, "the mismatch must be impossible to miss");
  const text = banner.textContent;
  assert.ok(/Coarse shift 13 configured, 14 fits better/.test(text), text);
  const sh = FX.summary_shift13.shift;
  const fr = (k) => `${(sh.fractions[sh.scan.indexOf(k)] * 100).toFixed(1)} %`;
  assert.ok(text.includes(`${fr(14)} vs ${fr(13)}`), text);
  assert.ok(/set \/DQM\/SMA\/Coarse shift = 14/.test(text), text);
  // Not repeated in the flag list, and the status chip agrees.
  assert.ok(!byId(page, "dqm-sma-flags").textContent.includes("fits better"));
  assert.ok(/shift 13: mismatch, best 14/.test(byId(page, "dqm-sma-status").textContent));
  banner.byTag("button")[0].onclick();
  assert.ok(page.calls.some((c) => c.method === "dlgOdbEdit" && c.params === "/DQM/SMA/Coarse shift"));
});

test("a broken time base gets the same banner, and a suspect-frame chip", async () => {
  const page = await boot({ "sma::summary": () => json(FX.summary_timebase) });
  const banner = byId(page, "dqm-sma-timebase-banner");
  assert.ok(banner, "time_base must be impossible to miss");
  const flag = FX.summary_timebase.flags.find((f) => f.code === "time_base");
  assert.ok(banner.textContent.includes(flag.text), banner.textContent);
  assert.ok(!byId(page, "dqm-sma-flags").textContent.includes(flag.text), "not repeated below");
  assert.strictEqual(byId(page, "dqm-sma-shift-banner"), null, "the shift check has no verdict");
  const status = byId(page, "dqm-sma-status").textContent;
  assert.ok(new RegExp(`${FX.summary_timebase.frames.suspect} suspect time base`).test(status), status);
});

test("only channels the analyzer judges are coloured", async () => {
  const s = clone(FX.summary);
  s.flags.push({ severity: "error", code: "mismatch", text: "ch09 (ch 9): made up" });
  const page = await boot({ "sma::summary": () => json(s) });
  const row = byId(page, "dqm-sma-table").find((e) => e.attrs && e.attrs["data-ch"] === "9");
  assert.ok(!row.byTag("td")[6].classList.contains("alarm"), "ch 9 is not flagged=true");
});

test("only the visible tab is polled", async () => {
  const page = await boot();
  const health = histCalls(page);
  assert.ok(health.length > 0, "the Health tab is shown first and polled");
  assert.ok(health.includes("sma/word_types"));
  for (const n of health) {
    assert.ok(!/^sma\/(tot_|fine_|dt_|rf_|delayed_|pattern|s1_coinc|s1_partner|s1_spacing|shift_check)/.test(n),
      `${n} is not on the Health tab`);
  }

  const mark = page.an.calls.length;
  byId(page, "dqm-sma-tab-timing").onclick();
  await settle(page);
  const timing = histCalls(page, mark);
  assert.ok(timing.includes("sma/dt_S2_S1") && timing.includes("sma/pattern"), timing.join(","));
  for (const n of timing) {
    assert.ok(/^sma\/(dt_S\d_S1|pattern|s1_coinc|s1_partner_hits|s1_spacing_us)$/.test(n),
      `${n} polled while the Timing tab is shown`);
  }
  assert.ok(!page.an.calls.slice(mark).some((c) => c.cmd === "sma::trend"));
  assert.ok(page.an.calls.slice(mark).some((c) => c.cmd === "sma::summary"),
    "the summary keeps polling whatever the tab");
});

test("the tab strip is a tablist: aria-selected follows the tab, arrows move it", async () => {
  const page = await boot();
  const ids = ["health", "tot", "timing", "rf", "mupix", "trends"];
  const tab = (id) => byId(page, `dqm-sma-tab-${id}`);
  const strip = tab("health").parent;
  assert.strictEqual(strip.getAttribute("role"), "tablist");
  assert.ok(!strip.find((e) => e.id === "dqm-sma-clear"), "the controls are not in the tab strip");
  for (const id of ids) {
    assert.strictEqual(tab(id).getAttribute("role"), "tab");
    assert.strictEqual(tab(id).getAttribute("aria-controls"), `dqm-sma-pane-${id}`);
    assert.ok(!tab(id).classList.contains("mbutton"), "a tab is not styled as a midas button");
    const pane = byId(page, `dqm-sma-pane-${id}`);
    assert.strictEqual(pane.getAttribute("role"), "tabpanel");
    assert.strictEqual(pane.getAttribute("aria-labelledby"), `dqm-sma-tab-${id}`);
  }
  const selected = () => ids.filter((id) => tab(id).getAttribute("aria-selected") === "true");
  const inOrder = () => ids.filter((id) => tab(id).getAttribute("tabindex") === "0");
  assert.deepStrictEqual(selected(), ["health"]);
  assert.deepStrictEqual(inOrder(), ["health"], "only the selected tab is in the tab order");

  tab("rf").onclick();
  await settle(page);
  assert.deepStrictEqual(selected(), ["rf"]);
  assert.deepStrictEqual(inOrder(), ["rf"]);

  const key = (k) => { let prevented = false;
    strip.dispatch("keydown", { key: k, target: page.doc.getElementById("dqm-sma-tab-rf"),
                                preventDefault() { prevented = true; } });
    return prevented; };
  assert.ok(key("ArrowRight"));
  assert.deepStrictEqual(selected(), ["mupix"]);
  assert.strictEqual(El.focused, tab("mupix"), "the new tab has the focus");
  key("ArrowRight"); key("ArrowRight");
  assert.deepStrictEqual(selected(), ["health"], "Right wraps past the last tab");
  key("ArrowLeft");
  assert.deepStrictEqual(selected(), ["trends"], "Left wraps past the first tab");
  key("Home");
  assert.deepStrictEqual(selected(), ["health"]);
  key("End");
  assert.deepStrictEqual(selected(), ["trends"]);
  assert.ok(!key("a"), "other keys are left alone");
  assert.deepStrictEqual(selected(), ["trends"]);
  assert.strictEqual(JSON.parse(globalThis.localStorage._d["dqm-sma-settings"]).tab, "trends",
                     "the keyboard choice is remembered like a click");

  // Keyboard switching polls only the tab it lands on.
  key("Home"); key("ArrowRight"); key("ArrowRight");      // -> timing
  await settle(page);
  assert.deepStrictEqual(selected(), ["timing"]);
  assert.strictEqual(byId(page, "dqm-sma-pane-timing").style.display, "");
  assert.strictEqual(byId(page, "dqm-sma-pane-health").style.display, "none");
  const mark = page.an.calls.length;
  await settle(page);
  const polled = histCalls(page, mark);
  assert.ok(polled.length > 0, "the Timing tab is polled");
  for (const n of polled) {
    assert.ok(/^sma\/(dt_S\d_S1|pattern|s1_coinc|s1_partner_hits|s1_spacing_us)$/.test(n),
      `${n} polled while the Timing tab is shown`);
  }
});

test("the ToT pairs are overlaid, one graph per channel", async () => {
  const page = await boot();
  byId(page, "dqm-sma-tab-tot").onclick();
  await settle(page);
  const grid = byId(page, "dqm-sma-grid-tot");
  const titles = grid.byClass("dqm-histtitle").map((t) => t.textContent);
  assert.ok(titles.includes("ToT, S1 (ch 1)"), titles.join(" | "));
  const s1 = grid.byClass("dqm-plot")[titles.indexOf("ToT, S1 (ch 1)")].mpg;
  assert.strictEqual(s1.param.plot.length, 2, "lsb0 and lsb1 on one graph");
  assert.notStrictEqual(s1.param.plot[0].line.color, s1.param.plot[1].line.color);
});

test("2D plots are heatmaps with log z by default; 1D plots are mplot with log y on request", async () => {
  const page = await boot();
  const divs = byId(page, "dqm-sma-grid-health").byClass("dqm-plot");
  const twoD = divs.find((d) => d.heatmap);
  const oneD = divs.find((d) => d.mpg && d.mpg.param.plot[0].type === "histogram");
  assert.ok(twoD && oneD);
  assert.ok(!twoD.mpg, "no mplot colormap");
  const hm = twoD.heatmap;
  assert.strictEqual(hm.logZ, true, "log z is the default");
  assert.strictEqual(hm.scale.min, 0.5, "one count sits just above the bottom of the scale");
  assert.strictEqual(hm.image.width, 16, "one pixel per in-range bin, under/overflow not drawn");
  assert.strictEqual(oneD.mpg.param.yAxis.log, false);

  const box = byId(page, "dqm-sma-logy");
  box.checked = true;
  box.onchange.call(box);
  await settle(page);
  assert.strictEqual(oneD.mpg.param.yAxis.log, true);
  assert.strictEqual(oneD.mpg.param.yAxis.min, 0.5, "a log axis from 0 would clamp to 1e-20");

  const draws = hm.draws;
  const z = byId(page, "dqm-sma-logz");
  z.checked = false;
  z.onchange.call(z);
  await settle(page);
  assert.strictEqual(hm.logZ, false);
  assert.strictEqual(hm.scale.min, 0);
  assert.strictEqual(hm.draws, draws + 1, "redrawn once for the new scale");
});

test("an unchanged histogram is fetched but not redrawn", async () => {
  const page = await boot();
  const divs = byId(page, "dqm-sma-grid-health").byClass("dqm-plot");
  const hm = divs.find((d) => d.heatmap).heatmap;
  const g = divs.find((d) => d.mpg).mpg;
  const before = [hm.draws, g.draws, g.data.length && g.data[0]];
  const mark = page.an.calls.length;
  await settle(page, 3);
  assert.ok(histCalls(page, mark).length > 0, "still polled");
  assert.strictEqual(hm.draws, before[0], "same bins: the heatmap draws nothing");
  assert.strictEqual(g.draws, before[1], "same bins: mplot is not asked to draw");
});

test("plots scrolled out of view are not polled once drawn", async () => {
  const observers = [];
  const page = await boot(undefined, undefined, function () {
    globalThis.IntersectionObserver = class {
      constructor(cb) { this.cb = cb; this.targets = []; observers.push(this); }
      observe(t) { this.targets.push(t); }
      unobserve(t) { this.targets = this.targets.filter((x) => x !== t); }
    };
  });
  try {
    const io = observers[0];
    assert.ok(io && io.targets.length > 5, "every plot is watched");
    const wraps = byId(page, "dqm-sma-grid-health").byClass("dqm-sma-plotwrap");
    const name = (w) => /\[(.*)\]/.exec(w.byClass("dqm-histtitle")[0].textContent)[1];
    // Only the first two on screen.
    io.cb(wraps.map((w, i) => ({ target: w, isIntersecting: i < 2 })));
    const mark = page.an.calls.length;
    await settle(page, 2);
    const asked = new Set(histCalls(page, mark).map((n) => n.replace(/^sma\//, "")));
    assert.deepStrictEqual([...asked].sort(), wraps.slice(0, 2).map(name).sort());
    // Scrolled: the next one comes into view and is polled again.
    io.cb([{ target: wraps[5], isIntersecting: true }]);
    const mark2 = page.an.calls.length;
    await settle(page, 1);
    assert.ok(histCalls(page, mark2).includes(`sma/${name(wraps[5])}`));

    // Back from another tab: the pane was hidden, so its plots are observed
    // afresh (which makes the observer report their current state) and all
    // polled until it has.
    byId(page, "dqm-sma-tab-timing").onclick();
    await settle(page, 1);
    const seen = io.targets.length;
    const mark3 = page.an.calls.length;
    byId(page, "dqm-sma-tab-health").onclick();
    assert.strictEqual(io.targets.length, seen, "re-observed, not observed twice");
    assert.strictEqual(io.targets.slice(-wraps.length).every((t) => wraps.includes(t)), true);
    await settle(page, 1);
    assert.deepStrictEqual([...new Set(histCalls(page, mark3))].sort(), wraps.map((w) => `sma/${name(w)}`).sort());
  } finally {
    delete globalThis.IntersectionObserver;
  }
});

test("every plot shows its entries", async () => {
  const page = await boot();
  const foots = byId(page, "dqm-sma-grid-health").byClass("dqm-footnote").map((f) => f.textContent);
  assert.ok(foots.length > 5);
  for (const f of foots) assert.ok(/entries/.test(f), f);
});

test("a histogram that fails is isolated to its own plot", async () => {
  const page = await boot({
    "dqm::histogram": (name) => {
      if (name === "sma/word_types") throw new Error("decoder exploded");
      const hex = FX.hist_dims[name] === 2 ? FX.hists["sma/tot_vs_ch_lsb0"] : FX.hists["sma/words_per_ch"];
      return envelope("hist", Buffer.from(hex, "hex"));
    },
  });
  const wraps = byId(page, "dqm-sma-grid-health").byClass("dqm-sma-plotwrap");
  const bad = wraps.filter((w) => w.classList.contains("dqm-sma-plot-error"));
  assert.strictEqual(bad.length, 1);
  assert.ok(/decoder exploded/.test(bad[0].textContent));
  const good = wraps.filter((w) => !w.classList.contains("dqm-sma-plot-error"));
  assert.strictEqual(good.length, wraps.length - 1);
  for (const w of good) assert.ok(/entries/.test(w.textContent), "the others still update");
});

test("a malformed summary block is isolated to that block", async () => {
  const s = clone(FX.summary);
  s.channels[4] = null;
  const page = await boot({ "sma::summary": () => json(s) });
  assert.ok(/could not draw this block/.test(byId(page, "dqm-sma-table").textContent));
  assert.ok(/connected/.test(byId(page, "dqm-sma-status").textContent), "the status still draws");
  assert.ok(byId(page, "dqm-sma-flags").byClass("dqm-sma-flag").length > 0, "the flags still draw");
});

test("the trends tab plots rates, live fraction, efficiency and RF valid", async () => {
  const page = await boot();
  byId(page, "dqm-sma-tab-trends").onclick();
  await settle(page);
  assert.ok(page.an.calls.some((c) => c.cmd === "sma::trend"));
  const graphs = byId(page, "dqm-sma-grid-trends").byClass("dqm-plot").map((d) => d.mpg);
  assert.strictEqual(graphs.length, 5);
  assert.deepStrictEqual(graphs[4].param.plot.map((p) => p.label),
                         ["L1", "L2", "L1+L2", "L1+L2 raw", "L1+L2 sideband"], "the MuPix in-time trend");
  const roleChans = FX.summary.channels.filter((c) => c.role).length;
  assert.strictEqual(graphs[0].param.plot.length, roleChans, "one rate line per channel with a role");
  assert.strictEqual(graphs[0].param.plot[0].label, FX.summary.channels.find((c) => c.role).label);
  assert.strictEqual(graphs[2].param.plot.length,
    FX.trend.counters.filter((c) => c !== "S1").length, "S2..S5, never S1 given S1");
  assert.deepStrictEqual(graphs[2].param.plot.map((p) => p.label.split(" ")[0]), ["S2", "S3", "S4", "S5"]);
  for (const g of graphs) {
    assert.ok(g.data[0] && g.data[0].x.length > 0, "each trend has points");
    assert.ok(g.data[0].x.every((x) => x <= 0 && x >= -10), "x is minutes ago");
  }
  // Incremental: the second poll asks only for rows after the last one.
  await settle(page);
  const trendArgs = page.an.calls.filter((c) => c.cmd === "sma::trend").map((c) => c.args);
  assert.ok(trendArgs.length >= 2);
  assert.ok(/since/.test(trendArgs[trendArgs.length - 1]));
});

test("Clear asks first, then clears the SMA histograms", async () => {
  const page = await boot();
  byId(page, "dqm-sma-clear").onclick();
  await settle(page, 1);
  assert.ok(page.calls.some((c) => c.method === "dlgConfirm"));
  const clear = page.an.calls.find((c) => c.cmd === "dqm::clear");
  assert.ok(clear, "dqm::clear was sent");
  assert.strictEqual(clear.args, "sma");
});

test("the current channel's mismatch is shown as not judged, not as a fault", async () => {
  const page = await boot();
  const row = byId(page, "dqm-sma-table").find((e) => e.attrs && e.attrs["data-ch"] === "7");
  const cell = row.byTag("td")[6];
  assert.ok(/\(not flagged\)$/.test(cell.textContent), cell.textContent);
  assert.ok(cell.classList.contains("dqm-sma-na"));
  assert.ok(!cell.classList.contains("alarm") && !cell.classList.contains("warn"));
});

test("the table and banner are updated in place, so a tooltip or a click survives a poll", async () => {
  let n = 0;
  const page = await boot({ "sma::summary": () => {
    const s = clone(FX.summary_shift13);
    s.shift.words += n++;                     // the banner text changes every poll
    s.channels[1].rate_hz += n;
    return json(s);
  } });
  const banner = byId(page, "dqm-sma-shift-banner");
  const td = byId(page, "dqm-sma-table").find((e) => e.attrs && e.attrs["data-ch"] === "1").byTag("td")[3];
  const text = banner.textContent, rate = td.textContent;
  await settle(page, 3);
  assert.strictEqual(byId(page, "dqm-sma-shift-banner"), banner, "same banner element");
  assert.notStrictEqual(banner.textContent, text, "with new text");
  assert.strictEqual(banner.getAttribute("role"), "alert");
  const td2 = byId(page, "dqm-sma-table").find((e) => e.attrs && e.attrs["data-ch"] === "1").byTag("td")[3];
  assert.strictEqual(td2, td, "same cell element");
  assert.notStrictEqual(td2.textContent, rate);
});

test("an analyzer that stops answering greys the page and dates it; recovery restores it", async () => {
  let up = true;
  const page = await boot({ "sma::summary": () => {
    if (!up) throw new Error("sma_analyzer did not answer sma::summary (status 103)");
    return json(FX.summary);
  } });
  const table = byId(page, "dqm-sma-table"), flags = byId(page, "dqm-sma-flags");
  assert.ok(!table.classList.contains("dqm-sma-stale"));

  up = false;
  await settle(page, 2);
  const status = byId(page, "dqm-sma-status").textContent;
  assert.ok(/no analyzer/.test(status), status);
  assert.ok(/last seen \d\d:\d\d:\d\d/.test(status), status);
  assert.ok(table.classList.contains("dqm-sma-stale"), "the table is marked not live");
  assert.ok(flags.classList.contains("dqm-sma-stale"), "so are the flags");
  assert.ok(byId(page, "dqm-sma-pane-health").classList.contains("dqm-sma-stale"));
  assert.strictEqual(table.findAll((e) => e.attrs && e.attrs["data-ch"] !== undefined).length, 16,
    "the last values stay on screen");

  up = true;
  await settle(page, 2);
  assert.ok(!table.classList.contains("dqm-sma-stale"), "fresh again");
  assert.ok(/connected/.test(byId(page, "dqm-sma-status").textContent));
});

test("a dead analyzer is asked once per tab poll, not once per plot", async () => {
  let up = true;
  const page = await boot({ "dqm::histogram": (name) => {
    if (!up) throw new Error("sma_analyzer did not answer dqm::histogram (status 103)");
    const hex = FX.hist_dims[name] === 2 ? FX.hists["sma/tot_vs_ch_lsb0"] : FX.hists["sma/words_per_ch"];
    return envelope("hist", Buffer.from(hex, "hex"));
  } });
  up = false;
  const mark = page.an.calls.length;
  await settle(page, 1);
  const asked = histCalls(page, mark).length;
  assert.ok(asked >= 1 && asked <= 1, `${asked} dqm::histogram calls in one tab poll against a dead analyzer`);
});

test("graphs are reused across epochs when the histograms are the same", async () => {
  let epoch = 0, made = 0;
  const page = await boot({ "sma::summary": () => json(Object.assign(clone(FX.summary), { epoch })) },
    undefined, function () {
      const Orig = globalThis.MPlotGraph;
      globalThis.MPlotGraph = class extends Orig { constructor(...a) { super(...a); made++; } };
    });
  const before = made;
  const graphs = byId(page, "dqm-sma-grid-health").byClass("dqm-plot").map((d) => d.mpg);
  assert.ok(before > 5);
  for (epoch = 1; epoch <= 5; epoch++) await settle(page, 1);
  assert.strictEqual(made, before, "no new MPlotGraph for a run change");
  const after = byId(page, "dqm-sma-grid-health").byClass("dqm-plot").map((d) => d.mpg);
  assert.deepStrictEqual(after, graphs);
  assert.ok(page.an.calls.filter((c) => c.cmd === "dqm::list").length >= 2, "but it did relist");
});

test("mhttpd going away is said in words, and the note clears when it is back", async () => {
  let down = false;
  const page = await boot({ "dqm::histogram": (name) => {
    // What mjsonrpc rejects with when mhttpd is restarting: not an Error.
    if (down) throw { request: { method: "brpc" }, xhr: { readyState: 4, status: 0, statusText: "" } };
    const hex = FX.hist_dims[name] === 2 ? FX.hists["sma/tot_vs_ch_lsb0"] : FX.hists["sma/words_per_ch"];
    return envelope("hist", Buffer.from(hex, "hex"));
  } });
  down = true;
  const mark = page.an.calls.length;
  await settle(page, 1);
  const note = byId(page, "dqm-sma-tabnote-health");
  assert.ok(/mhttpd is not reachable/.test(note.textContent), note.textContent);
  assert.ok(!/object Object/.test(byId(page, "dqm-root").textContent), "never [object Object]");
  assert.ok(histCalls(page, mark).length <= 1, "one attempt per poll while mhttpd is down");
  down = false;
  await settle(page, 2);
  assert.strictEqual(note.textContent, "", "cleared on the next good poll");
});

test("errors of every shape read as text", () => {
  const t = BRPC.errorText;
  assert.strictEqual(t(new Error("boom")), "boom");
  assert.strictEqual(t({ xhr: { readyState: 4, status: 503, statusText: "Service Unavailable" } }),
    "mhttpd answered HTTP 503 (Service Unavailable)");
  assert.strictEqual(t({ result: { status: 103 } }), "MIDAS status 103");
  assert.strictEqual(t({ error: { code: -32601, message: "Method not found" } }),
    "JSON-RPC error -32601 Method not found");
  assert.ok(!/object Object/.test(t({ weird: 1 })));
});

test("a counter with a timestamp fault is not drawn as a near-zero efficiency", async () => {
  const page = await boot();
  byId(page, "dqm-sma-tab-trends").onclick();
  await settle(page);
  const g = byId(page, "dqm-sma-grid-trends").byClass("dqm-plot")[2].mpg;
  const labels = g.param.plot.map((p) => p.label);
  const k5 = labels.findIndex((l) => /^S5/.test(l));
  assert.strictEqual(labels[k5], "S5 (n/a: timestamp fault)", labels.join(" | "));
  assert.strictEqual(g.data[k5].y.length, 0, "the backend's nulls are not plotted");
  const k2 = labels.indexOf("S2");
  assert.ok(g.data[k2].y.length > 0, "a healthy counter still has its line");
});

test("no analyzer is said plainly", async () => {
  const page = await boot({
    "sma::summary": () => { throw new Error("sma_analyzer did not answer sma::summary (status 103)"); },
  });
  const status = byId(page, "dqm-sma-status").textContent;
  assert.ok(/no analyzer/.test(status), status);
  assert.ok(/mdqm-analyzer --plugin sma/.test(status), status);
});

// ---------------------------------------------------------------------------
// Sampling: the CPU budget skips frames on purpose, and the page says so as
// information, not as missed frames.
// ---------------------------------------------------------------------------

const statusChips = (page) => byId(page, "dqm-sma-status").byClass("dqm-chip");
const chipText = (page, re) => statusChips(page).find((c) => re.test(c.textContent));

test("under the CPU budget: analysed share as information, CPU against budget, no missed chip", async () => {
  const page = await boot({ "sma::summary": () => json(FX.summary_budget),
                            "dqm::status": () => json(FX.status_budget) });
  const f = FX.summary_budget.frames;
  const frames = chipText(page, /^frames /);
  assert.strictEqual(frames.textContent,
    `frames ${f.processed} analysed · ${f.offered} offered · ${f.stale} stale`);
  const analysed = chipText(page, /^analysed /);
  assert.ok(analysed, "the analysed share is shown");
  assert.strictEqual(analysed.textContent, "analysed 45 % of frames");
  assert.ok(analysed.classList.contains("blue"), "information, not a warning");
  assert.ok(!chipText(page, /missed/), "skipped on purpose is not missed");
  assert.ok(!statusChips(page).some((c) => c.classList.contains("yellow")), "nothing yellow");

  const cpu = chipText(page, /^CPU /);
  assert.strictEqual(cpu.textContent, "CPU 19.4 % / budget 20 %");
  assert.ok(!cpu.classList.contains("yellow"), "19.4 % is within a 20 % budget");
  const rate = chipText(page, /frames\/s$/);
  assert.strictEqual(rate.textContent,
    `1.8 analysed / ${FX.summary_budget.sampling.offered_per_s.toFixed(1)} offered frames/s`);

  const note = byId(page, "dqm-sma-sampling");
  assert.notStrictEqual(note.style.display, "none");
  const s1 = FX.summary_budget.sampling.s1_analysed_frac;
  assert.strictEqual(note.textContent,
    "Counts are from the analysed sample: 45 % of frames (CPU budget 20 % of a core); " +
    `S1-seeded plots use at most 100 S1 hits per frame (${(s1 * 100).toFixed(1)} % of S1 hits). ` +
    "Rates, fractions and efficiencies are unbiased.");

  const flag = byId(page, "dqm-sma-flags").find((e) => e.attrs && e.attrs["data-code"] === "sampling");
  assert.ok(flag, "the sampling flag is listed");
  assert.ok(flag.classList.contains("blue"), "as information");
  assert.ok(/CPU budget of 20 % of a core/.test(flag.textContent), flag.textContent);
  assert.deepStrictEqual(globalThis.__alerts, []);
});

test("a CPU well over its budget is yellow", async () => {
  const st = clone(FX.status_budget);
  st.cpu_pct = 23.0;                          // > 1.1 x 20 %
  const page = await boot({ "sma::summary": () => json(FX.summary_budget),
                            "dqm::status": () => json(st) });
  const cpu = chipText(page, /^CPU /);
  assert.strictEqual(cpu.textContent, "CPU 23.0 % / budget 20 %");
  assert.ok(cpu.classList.contains("yellow"));
});

test("with every frame asked for, a serial gap is a real loss and stays yellow", async () => {
  const page = await boot({ "sma::summary": () => json(FX.summary_lossy),
                            "dqm::status": () => json(FX.status_lossy) });
  const missed = chipText(page, /missed/);
  assert.ok(missed, "lost frames are said");
  assert.strictEqual(missed.textContent, "5 missed (serial gaps)");
  assert.ok(missed.classList.contains("yellow"));
  assert.ok(!chipText(page, /^analysed /), "not dressed up as sampling");
  assert.ok(!chipText(page, /^CPU /), "no budget, no CPU chip");
  // The histograms are still from fewer frames than were sent: the note says
  // so, without a budget it does not have.
  const note = byId(page, "dqm-sma-sampling").textContent;
  assert.strictEqual(note, "Counts are from the analysed sample: 71 % of frames. " +
                           "Rates, fractions and efficiencies are unbiased.");
});

test("no live fraction (no two consecutive frames analysed) is a dash", async () => {
  assert.strictEqual(FX.summary_nolive.live_fraction, null);
  const page = await boot({ "sma::summary": () => json(FX.summary_nolive),
                            "dqm::status": () => json(FX.status_nolive) });
  assert.strictEqual(chipText(page, /^live /).textContent, "live —");
  assert.strictEqual(chipText(page, /^analysed /).textContent, "analysed 53 % of frames");
  assert.ok(chipText(page, /^CPU /).classList.contains("yellow"), "26 % against a 20 % budget");
  assert.ok(!/NaN|undefined|null/.test(byId(page, "dqm-sma-status").textContent));
});

test("taking every frame and every S1 hit: no note", async () => {
  const s = clone(FX.summary_lossy);
  s.sampling.analysed_frac = 1.0;
  s.frames.window.analysed_frac = 1.0;
  s.frames.missed_by_serial = 0;
  s.flags = s.flags.filter((f) => f.code !== "sampling");
  const page = await boot({ "sma::summary": () => json(s), "dqm::status": () => json(FX.status_lossy) });
  const note = byId(page, "dqm-sma-sampling");
  assert.strictEqual(note.textContent, "");
  assert.strictEqual(note.style.display, "none");
  assert.ok(!chipText(page, /^analysed |missed/));
});

test("an older summary without the sampling fields still draws the old chips", async () => {
  const s = clone(FX.summary);
  delete s.sampling;
  delete s.frames.offered;
  delete s.frames.window.analysed_frac;
  delete s.frames.analysed_frac;
  const st = clone(FX.status);
  delete st.plugin.analysed_frac;
  const page = await boot({ "sma::summary": () => json(s), "dqm::status": () => json(st) });
  assert.strictEqual(chipText(page, /^frames /).textContent,
    `frames ${s.frames.processed} analysed · ${s.frames.seen_by_serial} offered · ${s.frames.stale} stale`);
  assert.strictEqual(chipText(page, /frames\/s$/).textContent, `${FX.status.processed_per_s.toFixed(1)} frames/s`);
  assert.strictEqual(byId(page, "dqm-sma-sampling").style.display, "none");
});

test("with nothing offered in the window (a stopped run), the share since start is shown", async () => {
  const s = clone(FX.summary_budget);
  s.sampling.analysed_frac = null;
  s.sampling.offered_per_s = null;
  s.frames.window.analysed_frac = null;
  const page = await boot({ "sma::summary": () => json(s), "dqm::status": () => json(FX.status_budget) });
  assert.strictEqual(chipText(page, /^analysed /).textContent, "analysed 45 % of frames since start");
  assert.strictEqual(chipText(page, /frames\/s$/).textContent, "1.8 frames/s", "no offered rate to quote");
  assert.ok(/^Counts are from the analysed sample: 45 % of frames since start \(CPU budget 20 % of a core\)/
    .test(byId(page, "dqm-sma-sampling").textContent));
  assert.ok(!/—/.test(chipText(page, /^analysed /).textContent), "never 'analysed — of frames'");
});

// -- the MuPix tab ------------------------------------------------------------------------

test("the MuPix tab: in-time fractions per plane, the sync state, and its plots", async () => {
  const page = await boot();
  await settle(page);
  byId(page, "dqm-sma-tab-mupix").onclick();
  const mark = page.an.calls.length;
  await settle(page);
  const mp = FX.summary.mupix;
  const box = byId(page, "dqm-sma-mupix");
  const rows = box.byTag("tr").slice(1).map((tr) => tr.byTag("td").map((td) => td.textContent));
  const p = (x) => `${(x * 100).toFixed(1)} %`;
  assert.deepStrictEqual(rows[0], [`L1 (chips ${mp.planes.L1.join(", ")})`, p(mp.fractions.L1.in),
                                   p(mp.fractions.L1.side), p(mp.fractions.L1.corr)]);
  assert.deepStrictEqual(rows[2].slice(1), [p(mp.fractions.both.in), p(mp.fractions.both.side),
                                            p(mp.fractions.both.corr)]);
  const sync = byId(page, "dqm-sma-mpsync").byClass("dqm-chip")[0];
  assert.ok(/time sync: in sync · L1\+L2 \d+\.\d %/.test(sync.textContent), sync.textContent);
  assert.ok(sync.classList.contains("green"));
  assert.ok(/\[-150, 450\) ns; sideband \[-2400, -1800\) ns/.test(box.textContent), box.textContent);

  // Only the MuPix histograms, dt first, and the occupancy drawn per plane, one line per chip.
  const grid = byId(page, "dqm-sma-grid-mupix");
  const titles = grid.byClass("dqm-histtitle").map((t) => t.textContent);
  assert.ok(/\[mupix_dt_L1\]$/.test(titles[0]) && /\[mupix_dt_L2\]$/.test(titles[1]), titles.join(" | "));
  const occ = titles.filter((t) => /occupancy, L[12] \(chips/.test(t));
  assert.strictEqual(occ.length, 4, "column and row, L1 and L2");
  const graphs = grid.byClass("dqm-plot").map((d) => d.mpg);
  const dt = graphs[0];
  assert.deepStrictEqual(dt.param.plot.map((q) => q.label), ["mupix_dt_L1", "in time", "sideband"]);
  assert.deepStrictEqual(dt.data[1].x, [-150, -150, 450, 450], "the window outlined");
  assert.deepStrictEqual(dt.data[2].x, [-2400, -2400, -1800, -1800]);
  assert.ok(dt.data[1].y[1] >= Math.max(...dt.data[0].y), "as high as the peak");
  const col = graphs[titles.indexOf(occ[0])];
  assert.deepStrictEqual(col.param.plot.map((q) => q.label), mp.planes.L1.map((c) => `chip ${c}`));
  assert.ok(col.data[0].y.length === 258 && col.data[0].y.some((v) => v > 0),
            "chip 0's columns, from the 2D histogram");
  const match = grid.byClass("dqm-sma-plotwrap").find((w) => /mupix_s1_match/.test(w.textContent));
  assert.ok(/since the last clear, of [\d,]+ S1 hits: in time L1 \d+\.\d %/.test(match.textContent), match.textContent);
  assert.ok(page.an.calls.slice(mark + 1).filter((c) => c.cmd === "dqm::histogram")
    .every((c) => /^sma\/mupix_/.test(c.args)),
            "only the MuPix tab's histograms are asked for");
});

test("a lost MuPix time sync is a red chip and a warning flag", async () => {
  const page = await boot({ "sma::summary": () => json(FX.summary_mupix_sync) });
  byId(page, "dqm-sma-tab-mupix").onclick();
  await settle(page);
  const sync = byId(page, "dqm-sma-mpsync").byClass("dqm-chip")[0];
  assert.ok(/SYNC LOST\?/.test(sync.textContent) && sync.classList.contains("red"), sync.textContent);
  const flag = byId(page, "dqm-sma-flags").byClass("dqm-sma-flag").find((f) => f.attrs["data-code"] === "mupix_sync");
  assert.ok(flag && flag.classList.contains("yellow"), "a warning");
  assert.ok(/lost sync/.test(flag.textContent));
});

test("a plane-map edit lays the MuPix tab out again", async () => {
  let s = FX.summary;
  const page = await boot({ "sma::summary": () => json(s) });
  byId(page, "dqm-sma-tab-mupix").onclick();
  await settle(page);
  const titles = () => byId(page, "dqm-sma-grid-mupix").byClass("dqm-histtitle").map((t) => t.textContent);
  assert.ok(titles().some((t) => /occupancy, L1 \(chips 0, 1, 2, 3\)/.test(t)));
  s = clone(FX.summary);
  s.mupix.planes = { L1: [1, 2, 3, 4], L2: [5, 6, 7, 0] };
  await settle(page);
  assert.ok(titles().some((t) => /occupancy, L1 \(chips 1, 2, 3, 4\)/.test(t)), titles().join(" | "));
  assert.ok(!titles().some((t) => /chips 0, 1, 2, 3/.test(t)));
});

test("a reply that grows between the truncated call and its retry still arrives", async () => {
  // The 10-minute trend is ~270 kB, over the 256 kB first guess, and gains a
  // few bytes every second: a retry asking for exactly the size in the first
  // header came back truncated again, and the Trends tab never loaded.
  let n = 0;
  const asked = [];
  const saved = globalThis.mjsonrpc_call;
  globalThis.mjsonrpc_call = (method, params) => {
    asked.push(params.max_reply_length);
    const full = json({ rows: "x".repeat(270000 + 40 * n++) });
    return Promise.resolve(full.slice(0, Math.min(full.byteLength, params.max_reply_length + 23)));
  };
  try {
    const reply = await BRPC.json("sma_analyzer", "sma::trend", "");
    assert.ok(reply.rows.length >= 270000);
    assert.strictEqual(asked.length, 2);
  } finally {
    globalThis.mjsonrpc_call = saved;
  }
});
