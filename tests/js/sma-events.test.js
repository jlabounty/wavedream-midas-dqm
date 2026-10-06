//
// The SMA event display (/Custom/SMAEvents), headless, on real plugin frames.
//
// The seeded and raster payloads in sma-summary-fixture.json are what
// SmaPlugin.frame_blob sends for a real run-682 frame, so the decoder, the
// lanes and the badges all run on the analyzer's actual output. Canvas drawing
// is checked through the recording 2D context in domstub.js.
//

const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");

const { El, runPage } = require("./domstub.js");

const PAGES = path.join(__dirname, "..", "..", "pages", "js");
globalThis.DQM = require(path.join(PAGES, "dqm-common.js"));
globalThis.BRPC = require(path.join(PAGES, "dqm-brpc.js"));
globalThis.SMAF = require(path.join(PAGES, "dqm-smaframe.js"));

const EVENTS = path.join(PAGES, "dqm-sma-events.js");
const FX = JSON.parse(fs.readFileSync(path.join(__dirname, "sma-summary-fixture.json"), "utf8"));
const SEEDED = SMAF.decode(new Uint8Array(Buffer.from(FX.seeded, "hex")));
const RASTER = SMAF.decode(new Uint8Array(Buffer.from(FX.raster, "hex")));

function envelope(tag, bytes) {
  const out = new Uint8Array(8 + bytes.length);
  new DataView(out.buffer).setUint32(0, out.length, true);
  for (let i = 0; i < 4; i++) out[4 + i] = tag.charCodeAt(i);
  out.set(bytes, 8);
  return out.buffer;
}
const json = (o) => envelope("json", new TextEncoder().encode(JSON.stringify(o)));

function analyzer(over = {}) {
  const calls = [];
  const handlers = Object.assign({
    "sma::summary": () => json(FX.summary),
    "sma::frame": (args) => {
      const a = JSON.parse(args || "{}");
      return envelope("smaf", Buffer.from(a.view === "raster" ? FX.raster : FX.seeded, "hex"));
    },
  }, over);
  const brpc = (params) => {
    calls.push({ cmd: params.cmd, args: params.args });
    const h = handlers[params.cmd];
    if (!h) return Promise.resolve(envelope("err ", new TextEncoder().encode("unknown")));
    try { return Promise.resolve(h(params.args)); } catch (e) { return Promise.reject(e); }
  };
  return { calls, brpc };
}

async function settle(page, rounds = 3) {
  for (let r = 0; r < rounds; r++) {
    page.flushTimers();
    for (let i = 0; i < 30; i++) await new Promise((res) => setImmediate(res));
  }
}

async function boot(over, stored, dpr, opts) {
  globalThis.__alerts = [];
  globalThis.localStorage = {
    _d: stored ? { "dqm-sma-events-settings": JSON.stringify(stored) } : {},
    getItem(k) { return this._d[k] || null; }, setItem(k, v) { this._d[k] = v; },
  };
  // The tagged-hit list's store: a fresh tab unless a test hands one over (a reload).
  globalThis.sessionStorage = (opts && opts.session) || {
    _d: {}, getItem(k) { return this._d[k] || null; }, setItem(k, v) { this._d[k] = v; },
  };
  const an = analyzer(over);
  const page = runPage(EVENTS, { brpc: an.brpc }, opts || {});
  if (dpr) globalThis.window.devicePixelRatio = dpr;
  else delete globalThis.window.devicePixelRatio;
  await page.load();
  await settle(page);
  page.an = an;
  return page;
}

const byId = (page, id) => page.doc.getElementById(id);
const frameCalls = (page, from = 0) =>
  page.an.calls.slice(from).filter((c) => c.cmd === "sma::frame").map((c) => JSON.parse(c.args));
/** The ops of the most recent paint only (every paint starts with setTransform). */
function lastPaint(ctx) {
  let i = ctx.ops.length - 1;
  while (i >= 0 && ctx.ops[i][0] !== "setTransform") i--;
  const ops = ctx.ops.slice(Math.max(0, i));
  return {
    ops,
    count: (name, style) => ops.filter((o) => o[0] === name && (style === undefined || o[2] === style)).length,
    texts: () => ops.filter((o) => o[0] === "fillText").map((o) => String(o[1][0])),
  };
}
const popcount = (x) => { let n = 0; while (x) { n += x & 1; x >>= 1; } return n; };

// ---------------------------------------------------------------------------

test("the fixture frames are what the tests assume", () => {
  assert.ok(SEEDED.seeded && SEEDED.meta.seeds.length >= 2, "a seeded frame with seeds");
  assert.ok(!RASTER.seeded && RASTER.nHits > 100, "a raster frame");
  assert.ok(SEEDED.words && SEEDED.version === 2, "the seeded view is v2, with words");
  assert.ok(!RASTER.words && RASTER.version === 1, "the polled raster is v1");
  assert.deepStrictEqual(RASTER.meta.dropped, [FX.current_channel]);
});

test("the seeded view draws one panel per seed, with lanes and a pattern", async () => {
  const page = await boot();
  assert.ok(frameCalls(page).every((a) => a.view === "seeded"), "only the visible tab is asked for");
  const seeds = SEEDED.meta.seeds;
  const panels = byId(page, "dqm-smaev-seeds").byClass("dqm-sma-seed");
  assert.strictEqual(panels.length, seeds.length);
  panels.forEach(function (p, k) {
    const boxes = p.byClass("dqm-sma-pbox");
    assert.strictEqual(boxes.length, 5, "S1..S5");
    assert.deepStrictEqual(boxes.map((b) => b.textContent), ["S1", "S2", "S3", "S4", "S5"]);
    assert.strictEqual(p.byClass("lit").length, popcount(seeds[k].pattern), `seed ${k} pattern`);
    assert.ok(/RF/.test(p.textContent), "per-seed RF text");
    assert.ok(new RegExp(`${seeds[k].rf_n} pulse`).test(p.textContent));
    const canvas = p.byTag("canvas")[0];
    const ctx = canvas.getContext("2d");
    const labels = ctx.texts();
    // The fixture is old-layout data (S1..S5 on ch 1-5, RF 6, the current channel).
    for (const lane of ["S1 (1)", "S2 (2)", "S5 (5)", "RF (6)", `current (${FX.current_channel})`]) {
      assert.ok(labels.includes(lane), `lane ${lane} missing: ${labels.join(",")}`);
    }
    assert.ok(ctx.count("fillRect") > 10, "hit bars were drawn");
  });
  assert.deepStrictEqual(globalThis.__alerts, []);
  assert.strictEqual(byId(page, "dqm-smaev-seq").textContent, `frame seq ${SEEDED.frameSeq}`);
});

test("the tab strip is a tablist: aria-selected follows the tab, arrows move it", async () => {
  const page = await boot();
  const tab = (id) => byId(page, `dqm-smaev-tab-${id}`);
  const strip = tab("seeded").parent;
  assert.strictEqual(strip.getAttribute("role"), "tablist");
  for (const id of ["dqm-smaev-freeze", "dqm-smaev-single", "dqm-smaev-rate"]) {
    assert.ok(!strip.find((e) => e.id === id), `${id} is not in the tab strip`);
  }
  for (const id of ["seeded", "raster"]) {
    assert.strictEqual(tab(id).getAttribute("role"), "tab");
    assert.strictEqual(tab(id).getAttribute("aria-controls"), `dqm-smaev-pane-${id}`);
    assert.strictEqual(byId(page, `dqm-smaev-pane-${id}`).getAttribute("role"), "tabpanel");
  }
  // The Freeze and Single buttons stay midas buttons.
  assert.ok(byId(page, "dqm-smaev-freeze").classList.contains("mbutton"));
  assert.ok(byId(page, "dqm-smaev-single").classList.contains("mbutton"));
  const sel = () => ["seeded", "raster"].map((id) => tab(id).getAttribute("aria-selected"));
  assert.deepStrictEqual(sel(), ["true", "false"]);
  assert.strictEqual(tab("seeded").getAttribute("tabindex"), "0");
  assert.strictEqual(tab("raster").getAttribute("tabindex"), "-1");

  const key = (k) => strip.dispatch("keydown", { key: k, preventDefault() {} });
  let mark = page.an.calls.length;
  key("ArrowRight");
  await settle(page);
  assert.deepStrictEqual(sel(), ["false", "true"]);
  assert.ok(El.focused === tab("raster"), "the raster tab has the focus");
  assert.strictEqual(byId(page, "dqm-smaev-pane-seeded").style.display, "none");
  let asks = frameCalls(page, mark);
  assert.ok(asks.length > 0 && asks.every((a) => a.view !== "seeded"),
            `only the raster is asked for: ${JSON.stringify(asks.map((a) => a.view))}`);

  key("ArrowRight");                                   // wraps
  await settle(page);
  assert.deepStrictEqual(sel(), ["true", "false"]);
  key("ArrowLeft");
  assert.deepStrictEqual(sel(), ["false", "true"]);
  key("Home");
  assert.deepStrictEqual(sel(), ["true", "false"]);
  mark = page.an.calls.length;
  await settle(page);
  asks = frameCalls(page, mark);
  assert.ok(asks.length > 0 && asks.every((a) => a.view === "seeded"), "back to the seeded view only");

  tab("raster").onclick();
  assert.deepStrictEqual(sel(), ["false", "true"], "a click selects too");
  assert.strictEqual(JSON.parse(globalThis.localStorage._d["dqm-sma-events-settings"]).tab, "raster");
});

test("mismatched and ToT-corrupt hits get badges and their own marks", async () => {
  // The stored frame's seeds happen to hold no corrupt word, so three hits of
  // the first seed are flagged in the bytes the analyzer "sends".
  const raw = Buffer.from(FX.seeded, "hex");
  const s0 = SEEDED.meta.seeds[0];
  const flagsAt = SEEDED.offsets.hitFlags;
  raw[flagsAt + s0.hits[0]] |= SMAF.HIT.MISMATCH;
  raw[flagsAt + s0.hits[0] + 1] |= SMAF.HIT.MISMATCH;
  raw[flagsAt + s0.hits[0] + 2] |= SMAF.HIT.TOT_CORRUPT;
  const page = await boot({ "sma::frame": () => envelope("smaf", raw) });

  const first = byId(page, "dqm-smaev-seeds").byClass("dqm-sma-seed")[0];
  const badges = first.byClass("dqm-sma-badge").map((b) => [b.textContent, b.className]);
  assert.ok(badges.some(([t, c]) => /^2 fine\/coarse mismatch/.test(t) && /red/.test(c)),
    JSON.stringify(badges));
  assert.ok(badges.some(([t]) => /^1 ToT ≥ 250/.test(t)), JSON.stringify(badges));
  const paint = lastPaint(first.byTag("canvas")[0].getContext("2d"));
  assert.strictEqual(paint.count("strokeRect", "#d00"), 2, "a red outline per mismatched hit");

  // The other seeds are untouched and carry no such badge.
  const second = byId(page, "dqm-smaev-seeds").byClass("dqm-sma-seed")[1];
  assert.ok(!/mismatch/.test(second.textContent));
});

test("the prompt zoom redraws on +-150 ns", async () => {
  const page = await boot();
  const sel = byId(page, "dqm-smaev-pane-seeded").byTag("select")[0];
  sel.value = "prompt";
  sel.onchange.call(sel);
  const ticks = lastPaint(byId(page, "dqm-smaev-seeds").byTag("canvas")[0].getContext("2d")).texts();
  assert.ok(ticks.includes("-150") || ticks.includes("-100"), ticks.join(","));
  assert.ok(!ticks.includes("3000"), "the full-window ticks are gone");
});

test("the canvas follows devicePixelRatio", async () => {
  const page = await boot(undefined, undefined, 2);
  const canvas = byId(page, "dqm-smaev-seeds").byTag("canvas")[0];
  assert.strictEqual(canvas.width, 1800, "900 css px at dpr 2");
  const st = canvas.getContext("2d").ops.find((o) => o[0] === "setTransform");
  assert.deepStrictEqual(st[1], [2, 0, 0, 2, 0, 0]);
});

test("the raster hides the current channel by asking the analyzer not to send it", async () => {
  const page = await boot(undefined, { tab: "raster" });
  const asks = frameCalls(page);
  assert.ok(asks.length > 0);
  for (const a of asks) {
    assert.strictEqual(a.view, "raster");
    assert.deepStrictEqual(a.drop, [FX.current_channel], "the drop list comes from the summary's current role");
  }
  const box = byId(page, "dqm-smaev-hidecur");
  box.checked = false;
  box.onchange.call(box);
  const mark = page.an.calls.length;
  await settle(page);
  const after = frameCalls(page, mark);
  assert.ok(after.length > 0);
  assert.deepStrictEqual(after[0].drop, []);
});

test("the raster decodes and draws the frame, header and per-channel counts", async () => {
  const page = await boot(undefined, { tab: "raster" });
  const head = byId(page, "dqm-smaev-rasterhead").textContent;
  assert.ok(head.includes(`seq ${RASTER.frameSeq}`), head);
  assert.ok(head.includes(`run ${RASTER.run}`), head);
  assert.ok(/words: .* filler .* pixel .* trigger/.test(head), head);
  assert.ok(/span \d+\.\d{3} ms/.test(head), head);

  const ctx = lastPaint(byId(page, "dqm-smaev-raster").getContext("2d"));
  const texts = ctx.texts();
  const per = RASTER.meta.per_channel;
  assert.ok(texts.includes(`${per[7].toLocaleString()} (hidden)`),
    "the hidden channel still reports its count");
  assert.ok(texts.includes(per[1].toLocaleString()), "S1's count at the right edge");
  // Every hit went into a rect, bucketed by ToT colour into a few fills: the
  // SMA hits, then the MuPix pixel hits on their two rows.
  assert.ok(RASTER.pixels && RASTER.pixels.n > 100, "the fixture raster carries pixel hits");
  // (A pixel hit before the frame's first kept SMA hit is off the axis.)
  const mp = RASTER.meta.mupix;
  let nPix = 0;
  for (let j = 0; j < RASTER.pixels.n; j++) {
    if (mp.t0_ns - RASTER.meta.frame_first_ns + RASTER.pixels.t[j] >= 0) nPix++;
  }
  assert.ok(nPix > 100 && nPix <= RASTER.pixels.n);
  assert.strictEqual(ctx.count("rect"), RASTER.nHits + nPix, "one rect per hit");
  assert.ok(ctx.count("fill") < 40 + 32, "one fill per colour bucket, not per hit");
  assert.strictEqual(byId(page, "dqm-smaev-pane-raster").byClass("dqm-sma-ramp").length, 2,
                     "the SMA ToT legend and the MuPix ToT legend");
  // The MuPix rows under the 16 channels, with their plane totals at the right.
  assert.ok(texts.includes("MuPix L1") && texts.includes("MuPix L2"), texts.join(","));
  assert.ok(texts.includes(mp.per_plane[1].toLocaleString()), "L1's pixel count");
  assert.ok(!texts.includes("no plane"), "no third row without an unmapped chip");
  assert.ok(/MuPix \d+ hits: L1 \d+ · L2 \d+/.test(head), head);
});

test("a suspect frame says so, with its rescued hits, on a frame-start axis", async () => {
  const sus = SMAF.decode(new Uint8Array(Buffer.from(FX.raster_suspect, "hex")));
  assert.ok(sus.suspect, "the run-342-at-shift-14 fixture frame is suspect");
  const page = await boot({ "sma::frame": () => envelope("smaf", Buffer.from(FX.raster_suspect, "hex")) },
                          { tab: "raster" });
  const head = byId(page, "dqm-smaev-rasterhead");
  const red = head.byClass("dqm-sma-badge").filter((b) => b.classList.contains("red"));
  assert.ok(red.some((b) => /^SUSPECT time base/.test(b.textContent)),
    head.byClass("dqm-sma-badge").map((b) => b.textContent).join(" | "));
  if (sus.meta.n_rescued) assert.ok(head.textContent.includes(`${sus.meta.n_rescued.toLocaleString()} rescued`));
  if (sus.timeShift) assert.ok(head.textContent.includes(`times in ${2 ** sus.timeShift} ns units`));
  assert.ok(lastPaint(byId(page, "dqm-smaev-raster").getContext("2d")).count("rect") > 0);
});

test("raster times are from the frame start, not the first shipped hit", async () => {
  // Dropping ch 7 moves t0_ns (the first *shipped* hit) later than the frame's
  // first kept hit; the axis must not shift with it.
  const off = RASTER.meta.t0_ns - RASTER.meta.frame_first_ns;
  assert.ok(off > 0, "the fixture raster's first shipped hit is not the frame's first");
  const page = await boot(undefined, { tab: "raster" });
  const rects = lastPaint(byId(page, "dqm-smaev-raster").getContext("2d")).ops
    .filter((o) => o[0] === "rect").map((o) => o[1][0]);
  // Canvas 900 css px, margins 84 left / 110 right; the axis runs to the
  // later of the frame span and the last hit, measured from the frame start.
  const x0 = 84, x1 = 900 - 110;
  const hi = Math.max(RASTER.meta.span_ns, off + RASTER.t[RASTER.nHits - 1]);
  const want = x0 + off / hi * (x1 - x0);
  assert.ok(Math.abs(Math.min(...rects) - want) < 0.5,
    `first hit at x=${Math.min(...rects)}, expected ${want} (not ${x0})`);
});

test("the seed time is from the frame start, not from the first shipped hit", async () => {
  const page = await boot();
  const off = SEEDED.meta.t0_ns - SEEDED.meta.frame_first_ns;
  assert.ok(off > 0, "the seeded view ships only the windows, so t0 is late");
  const titles = byId(page, "dqm-smaev-seeds").byClass("dqm-sma-seedtitle").map((t) => t.textContent);
  SEEDED.meta.seeds.forEach(function (seed, k) {
    const want = ((off + seed.t_rel) / 1e6).toFixed(3);
    assert.ok(titles[k].includes(`S1 at ${want} ms in the frame`), `${titles[k]} (want ${want})`);
  });
});

test("the raster asks for a bounded reply", async () => {
  const page = await boot(undefined, { tab: "raster" });
  for (const a of frameCalls(page)) assert.strictEqual(a.max_hits, 60000);
});

test("a raster with time_shift > 0 is drawn on the true time axis", async () => {
  const CASES = JSON.parse(fs.readFileSync(path.join(__dirname, "smaframe-cases.json"), "utf8")).cases;
  const c = CASES.find((x) => x.expect.time_shift > 0 && x.expect.n_hits > 100 && !x.expect.seeded);
  assert.ok(c, "smaframe-cases.json has a real raster with a time shift");
  const f = SMAF.decode(new Uint8Array(Buffer.from(c.payload_hex, "hex")));
  const page = await boot({ "sma::frame": () => envelope("smaf", Buffer.from(c.payload_hex, "hex")) },
                          { tab: "raster" });
  const head = byId(page, "dqm-smaev-rasterhead").textContent;
  assert.ok(head.includes(`times in ${2 ** f.timeShift} ns units`), head);
  const rects = lastPaint(byId(page, "dqm-smaev-raster").getContext("2d")).ops
    .filter((o) => o[0] === "rect").map((o) => o[1][0]);
  const x0 = 84, x1 = 900 - 110;
  const off = f.meta.t0_ns - f.meta.frame_first_ns;
  const hi = Math.max(f.meta.span_ns || 0, off + f.tRaw[f.nHits - 1] * 2 ** f.timeShift);
  const want = x0 + (off + f.tRaw[f.nHits - 1] * 2 ** f.timeShift) / hi * (x1 - x0);
  assert.ok(Math.abs(Math.max(...rects) - want) < 0.5,
    `last hit at ${Math.max(...rects)}, want ${want}: the 2^k scale must be applied`);
});

test("a drag released outside the canvas still ends, and zooms", async () => {
  const page = await boot(undefined, { tab: "raster" });
  const canvas = byId(page, "dqm-smaev-raster");
  canvas.dispatch("mousedown", { clientX: 200 });
  canvas.dispatch("mousemove", { clientX: 400 });
  page.windowEvent("mouseup", { clientX: 2000 });          // far outside the canvas
  const selection = "rgba(0, 102, 204, 0.15)";
  const ctx = canvas.getContext("2d");
  const mark = ctx.ops.length;
  canvas.dispatch("mousemove", { clientX: 500 });           // no button held any more
  assert.strictEqual(ctx.ops.slice(mark).filter((o) => o[2] === selection).length, 0,
    "no selection is drawn after the release");
  // Tick labels are the texts below the 16 channel rows and the 2 MuPix rows.
  const rows = 16 + 2;
  const ticks = () => lastPaint(ctx).ops
    .filter((o) => o[0] === "fillText" && o[1][2] > 6 + rows * 22 && o[1][2] < 6 + rows * 22 + 12)
    .map((o) => Number(o[1][0]));
  const zoomed = ticks();
  assert.ok(zoomed.length > 2 && zoomed[0] > 0, `zoomed away from the frame start: ${zoomed}`);
  canvas.dispatch("dblclick");
  assert.strictEqual(ticks()[0], 0, "double-click shows the whole frame again");
});

test("Freeze stops polling and keeps the frame", async () => {
  const page = await boot();
  const seq = byId(page, "dqm-smaev-seq").textContent;
  byId(page, "dqm-smaev-freeze").onclick();
  assert.strictEqual(byId(page, "dqm-smaev-live").textContent, "FROZEN");
  const mark = page.an.calls.length;
  await settle(page, 5);
  assert.strictEqual(frameCalls(page, mark).length, 0, "no polling while frozen");
  assert.strictEqual(byId(page, "dqm-smaev-seq").textContent, seq, "the frame is kept");
  assert.strictEqual(byId(page, "dqm-smaev-seeds").byClass("dqm-sma-seed").length,
                     SEEDED.meta.seeds.length);

  // Switching tab while frozen does not poll either.
  byId(page, "dqm-smaev-tab-raster").onclick();
  await settle(page, 2);
  assert.strictEqual(frameCalls(page, mark).length, 0);

  byId(page, "dqm-smaev-freeze").onclick();
  await settle(page);
  assert.ok(frameCalls(page, mark).length > 0, "resumes");
});

test("a panel that throws shows its error and the others still draw", async () => {
  const bad = JSON.parse(JSON.stringify(SEEDED.meta));
  bad.seeds[1].hits = null;       // seedHits() would throw on this
  bad.seeds[1].pattern = { toString() { throw new Error("pattern exploded"); } };
  const real = SMAF.fetchFrame;
  SMAF.fetchFrame = async () => Object.assign({}, SEEDED, { meta: bad });
  try {
    const page = await boot();
    const panels = byId(page, "dqm-smaev-seeds").byClass("dqm-sma-seed");
    assert.strictEqual(panels.length, bad.seeds.length);
    const errs = panels.filter((p) => /could not draw/.test(p.textContent));
    assert.strictEqual(errs.length, 1, "exactly the broken seed says so");
    assert.ok(/seed 2/.test(errs[0].textContent));
    const drawn = panels.filter((p) => p.byTag("canvas")[0].getContext("2d").count("fillRect") > 10);
    assert.strictEqual(drawn.length, bad.seeds.length - 1, "the other seeds are drawn");
  } finally {
    SMAF.fetchFrame = real;
  }
});

test("no analyzer is said plainly", async () => {
  const page = await boot({
    "sma::frame": () => { throw new Error("sma_analyzer did not answer sma::frame (status 103)"); },
    "sma::summary": () => { throw new Error("nope"); },
  });
  assert.strictEqual(byId(page, "dqm-smaev-live").textContent, "no analyzer");
  assert.ok(/mdqm-analyzer --plugin sma/.test(byId(page, "dqm-smaev-note-seeded").textContent));
});

test("no frame yet is explained, not drawn blank", async () => {
  const page = await boot({ "sma::frame": () => json({ no_frame: true }) });
  assert.strictEqual(byId(page, "dqm-smaev-live").textContent, "no frame");
  assert.ok(/good SMA frame/.test(byId(page, "dqm-smaev-note-seeded").textContent));
});

// -- Single --------------------------------------------------------------------

/** The fixture payload with its frame seq (header bytes 8..15) replaced. */
function withSeq(hex, seq) {
  const raw = Buffer.from(hex, "hex");
  raw.writeBigUInt64LE(BigInt(seq), 8);
  return raw;
}

/** An analyzer whose every sma::frame reply is a new frame (seq 1000, 1001, ...). */
function advancing() {
  let seq = 1000;
  return {
    "sma::frame": (args) => {
      const a = JSON.parse(args || "{}");
      return envelope("smaf", withSeq(a.view === "raster" ? FX.raster : FX.seeded, seq++));
    },
  };
}

/** A handler whose replies are held until release() is called. */
function held() {
  const waiting = [];
  let seq = 2000;
  return {
    waiting,
    handler: (args) => new Promise((res) => {
      const a = JSON.parse(args || "{}");
      waiting.push(() => res(envelope("smaf", withSeq(a.view === "raster" ? FX.raster : FX.seeded, seq++))));
    }),
    release() { waiting.splice(0).forEach((f) => f()); },
  };
}

const seqText = (page) => byId(page, "dqm-smaev-seq").textContent;

test("Single from live freezes and fetches exactly one new frame per press", async () => {
  const page = await boot(advancing());
  const single = byId(page, "dqm-smaev-single");
  assert.ok(single.getAttribute("title"), "a tooltip");
  assert.strictEqual(single.textContent, "Single ▸");
  const before = seqText(page);

  let mark = page.an.calls.length;
  single.onclick();
  await settle(page, 5);
  assert.strictEqual(byId(page, "dqm-smaev-live").textContent, "FROZEN");
  assert.strictEqual(byId(page, "dqm-smaev-freeze").textContent, "Frozen — resume");
  let asks = frameCalls(page, mark);
  assert.strictEqual(asks.length, 1, "exactly one request");
  assert.strictEqual(asks[0].view, "seeded");
  const first = seqText(page);
  assert.notStrictEqual(first, before, "the new frame is shown");
  assert.ok(!single.disabled, "enabled again once the reply is in");

  mark = page.an.calls.length;
  await settle(page, 5);
  assert.strictEqual(frameCalls(page, mark).length, 0, "no polling afterwards");

  single.onclick();
  await settle(page, 5);
  assert.strictEqual(frameCalls(page, mark).length, 1, "a second press, one more request");
  const n = (t) => Number(/(\d+)$/.exec(t)[1]);
  assert.strictEqual(n(seqText(page)), n(first) + 1, "the next frame");
  assert.strictEqual(byId(page, "dqm-smaev-live").textContent, "FROZEN", "still frozen");

  // Freeze/resume goes back to polling as before.
  byId(page, "dqm-smaev-freeze").onclick();
  mark = page.an.calls.length;
  await settle(page, 3);
  assert.ok(frameCalls(page, mark).length > 1, "polling again");
  assert.strictEqual(byId(page, "dqm-smaev-freeze").textContent, "Freeze");
});

test("Single with no newer frame says so and keeps the display", async () => {
  const page = await boot();                     // every reply is the same frame
  const seq = seqText(page);
  const canvas = byId(page, "dqm-smaev-seeds").byTag("canvas")[0];
  const nOps = canvas.getContext("2d").ops.length;
  const note = byId(page, "dqm-smaev-singlenote");
  assert.strictEqual(note.style.display, "none", "hidden until needed");

  const mark = page.an.calls.length;
  byId(page, "dqm-smaev-single").onclick();
  await settle(page, 5);
  assert.strictEqual(frameCalls(page, mark).length, 2, "one request plus one retry, no more");
  assert.strictEqual(note.textContent, `no newer frame than seq ${SEEDED.frameSeq} yet`);
  assert.notStrictEqual(note.style.display, "none");
  assert.strictEqual(seqText(page), seq, "the frame seq is unchanged");
  assert.strictEqual(canvas.getContext("2d").ops.length, nOps, "not redrawn");
  assert.strictEqual(byId(page, "dqm-smaev-seeds").byClass("dqm-sma-seed").length,
                     SEEDED.meta.seeds.length);

  // Resuming clears the note.
  byId(page, "dqm-smaev-freeze").onclick();
  await settle(page);
  assert.strictEqual(note.style.display, "none");
});

test("clicks while a Single is in flight do not start a second request", async () => {
  const h = held();
  let holding = false;
  const page = await boot({
    "sma::frame": (args) => (holding ? h.handler(args)
                                     : envelope("smaf", Buffer.from(FX.seeded, "hex"))),
  });
  byId(page, "dqm-smaev-freeze").onclick();           // quiet: no poll tick in flight
  await settle(page);
  holding = true;                                     // from here on every reply waits
  const mark = page.an.calls.length;
  const single = byId(page, "dqm-smaev-single");
  single.onclick();
  await settle(page, 2);
  assert.ok(single.disabled, "disabled while in flight");
  single.onclick(); single.onclick(); single.onclick();
  await settle(page, 2);
  assert.strictEqual(frameCalls(page, mark).length, 1, "one request, however many clicks");
  h.release();
  await settle(page, 3);
  assert.ok(!single.disabled);
  assert.strictEqual(seqText(page), "frame seq 2000");
  assert.strictEqual(frameCalls(page, mark).length, 1);
});

test("Single waits for a poll tick in flight rather than stacking on it", async () => {
  const h = held();
  let holding = false;
  const page = await boot({
    "sma::frame": (args) => (holding ? h.handler(args)
                                     : envelope("smaf", Buffer.from(FX.seeded, "hex"))),
  });
  holding = true;
  let mark = page.an.calls.length;
  page.flushTimers();                                 // the next poll tick starts and waits
  for (let i = 0; i < 30; i++) await new Promise((res) => setImmediate(res));
  assert.strictEqual(frameCalls(page, mark).length, 1, "a tick in flight");

  byId(page, "dqm-smaev-single").onclick();
  for (let i = 0; i < 30; i++) await new Promise((res) => setImmediate(res));
  assert.strictEqual(frameCalls(page, mark).length, 1, "Single has not started its own yet");
  assert.strictEqual(byId(page, "dqm-smaev-live").textContent, "FROZEN");

  h.release();                                         // the tick's reply (dropped: frozen)
  await settle(page, 2);
  assert.strictEqual(frameCalls(page, mark).length, 2, "then exactly one for Single");
  h.release();
  await settle(page, 3);
  assert.strictEqual(seqText(page), "frame seq 2001", "Single's frame, not the tick's");
  assert.strictEqual(frameCalls(page, mark).length, 2, "and nothing more");
});

test("Single on the raster tab keeps the drop and max_hits args", async () => {
  const page = await boot(advancing(), { tab: "raster" });
  const box = byId(page, "dqm-smaev-hidecur");
  let mark = page.an.calls.length;
  byId(page, "dqm-smaev-single").onclick();
  await settle(page, 5);
  let asks = frameCalls(page, mark);
  assert.strictEqual(asks.length, 1);
  // A frame stepped to by hand comes with its words, in the same request.
  assert.deepStrictEqual(asks[0], { view: "raster", drop: [FX.current_channel], max_hits: 60000, words: true });
  assert.ok(byId(page, "dqm-smaev-rasterhead").textContent.includes(`seq ${Number(/(\d+)$/.exec(seqText(page))[1])}`));

  // The switch while frozen changes what the next Single asks for, not polling.
  box.checked = false;
  box.onchange.call(box);
  mark = page.an.calls.length;
  await settle(page, 3);
  assert.strictEqual(frameCalls(page, mark).length, 0, "still frozen");
  byId(page, "dqm-smaev-single").onclick();
  await settle(page, 5);
  asks = frameCalls(page, mark);
  assert.strictEqual(asks.length, 1);
  assert.deepStrictEqual(asks[0], { view: "raster", drop: [], max_hits: 60000, words: true });
});

test("switching tab while frozen, Single fetches for the new tab", async () => {
  const page = await boot(advancing());
  byId(page, "dqm-smaev-freeze").onclick();
  byId(page, "dqm-smaev-tab-raster").onclick();
  const mark = page.an.calls.length;
  await settle(page, 2);
  assert.strictEqual(frameCalls(page, mark).length, 0);
  byId(page, "dqm-smaev-single").onclick();
  await settle(page, 5);
  const asks = frameCalls(page, mark);
  assert.strictEqual(asks.length, 1);
  assert.strictEqual(asks[0].view, "raster");
  assert.ok(byId(page, "dqm-smaev-rasterhead").textContent.includes("seq "), "the raster is drawn");
});

test("a failed Single is reported and the button comes back", async () => {
  let fail = false;
  const page = await boot({
    "sma::frame": () => {
      if (fail) throw new Error("sma_analyzer did not answer sma::frame (status 103)");
      return envelope("smaf", Buffer.from(FX.seeded, "hex"));
    },
  });
  fail = true;
  const single = byId(page, "dqm-smaev-single");
  single.onclick();
  await settle(page, 5);
  assert.strictEqual(byId(page, "dqm-smaev-live").textContent, "no analyzer");
  assert.ok(/status 103/.test(byId(page, "dqm-smaev-note-seeded").textContent));
  assert.ok(/press Single/.test(byId(page, "dqm-smaev-note-seeded").textContent));
  assert.ok(!single.disabled);
});

// -- sampling --------------------------------------------------------------------

const samplingNote = (page, id) => byId(page, id).byClass("dqm-smaev-sampling").map((n) => n.textContent);

test("under the CPU budget the header says the frame is the latest analysed one", async () => {
  const page = await boot({ "sma::summary": () => json(FX.summary_budget) });
  assert.deepStrictEqual(samplingNote(page, "dqm-smaev-seededhead"),
    ["Latest analysed frame; the analyzer analyses 45 % of frames (CPU budget)"]);
  byId(page, "dqm-smaev-tab-raster").onclick();
  await settle(page);
  assert.deepStrictEqual(samplingNote(page, "dqm-smaev-rasterhead"),
    ["Latest analysed frame; the analyzer analyses 45 % of frames (CPU budget)"]);
  // From the summary it already fetches: no dqm::status poll for it.
  assert.ok(!page.an.calls.some((c) => c.cmd === "dqm::status"));
});

test("without a budget the note has no budget, and with every frame there is none", async () => {
  const lossy = await boot({ "sma::summary": () => json(FX.summary_lossy) });
  assert.deepStrictEqual(samplingNote(lossy, "dqm-smaev-seededhead"),
    ["Latest analysed frame; the analyzer analyses 71 % of frames"]);

  const s = JSON.parse(JSON.stringify(FX.summary_budget));
  s.sampling.analysed_frac = 1.0;
  const all = await boot({ "sma::summary": () => json(s) });
  assert.deepStrictEqual(samplingNote(all, "dqm-smaev-seededhead"), []);

  const none = await boot({ "sma::summary": () => { throw new Error("nope"); } });
  assert.deepStrictEqual(samplingNote(none, "dqm-smaev-seededhead"), [], "no summary, no claim");
});

test("with nothing offered in the window the note falls back to the share since start", async () => {
  const s = JSON.parse(JSON.stringify(FX.summary_budget));
  s.sampling.analysed_frac = null;
  const page = await boot({ "sma::summary": () => json(s) });
  assert.deepStrictEqual(samplingNote(page, "dqm-smaev-seededhead"),
    ["Latest analysed frame; the analyzer analyses 45 % of frames since start (CPU budget)"]);
});

// -- finding a frame again: tag, raw event, hit lines --------------------------------

const RASTER_WORDS = SMAF.decode(new Uint8Array(Buffer.from(FX.raster_words, "hex")));
const RAW_EVENT = Buffer.from(FX.raw_event, "hex");

/** Let promise chains run without firing timers (settle() would reset "Copied"). */
async function drain(n = 30) {
  for (let i = 0; i < n; i++) await new Promise((res) => setImmediate(res));
}

/** A smaf payload (hex) with its JSON metadata changed by fn, re-encoded. */
function withMeta(hex, fn) {
  const raw = Buffer.from(hex, "hex");
  const f = SMAF.decode(new Uint8Array(raw));
  const meta = JSON.parse(JSON.stringify(f.meta));
  fn(meta);
  const js = Buffer.from(JSON.stringify(meta));
  const at = 32 + Math.ceil(js.length / 8) * 8;
  const out = Buffer.alloc(at + raw.length - f.arraysOffset);
  raw.copy(out, 0, 0, 32);
  out.writeUInt32LE(js.length, 20);
  js.copy(out, 32);
  raw.copy(out, at, f.arraysOffset);
  return out;
}

/** An analyzer that also answers sma::raw, and the words refetch by seq. */
function withRaw(extra = {}) {
  return Object.assign({
    "sma::frame": (args) => {
      const a = JSON.parse(args || "{}");
      if (a.view === "raster") {
        return envelope("smaf", Buffer.from(a.words && a.seq === RASTER.frameSeq
          ? FX.raster_words : FX.raster, "hex"));
      }
      return envelope("smaf", Buffer.from(FX.seeded, "hex"));
    },
    "sma::raw": (args) => {
      const a = JSON.parse(args || "{}");
      if (a.seq === FX.raw_seq) return envelope("mevt", RAW_EVENT);
      return envelope("err ", new TextEncoder().encode(`frame ${a.seq} no longer held — use the tag`));
    },
  }, extra);
}

const TAG_RE = /^SMA run 682 · event 301 serial \d+ · 2026-09-21 \d\d:\d\d:\d\d UTC · frame seq \d+$/;

/** The label of each lane on a seed canvas, from what was drawn: {ch: laneIndex}. */
function laneMap(ctx) {
  const out = {};
  for (const o of lastPaint(ctx).ops) {
    if (o[0] !== "fillText") continue;
    const m = /\((\d+)\)$/.exec(String(o[1][0]));
    if (m && o[1][1] === 84 - 6) out[Number(m[1])] = Math.round((o[1][2] - 6 - 10) / 20);
  }
  return out;
}

/** The hit line the page must show for hit i, built here from the spec, not the page. */
function expectLine(f, i, labels) {
  const c = f.ch[i];
  const l = labels[c];
  const t = f.t[i];
  const words = f.rawWord
    ? `word ${f.wordIndex[i]} · 0x${f.rawWord[i].toString(16).padStart(16, "0")}`
    : "freeze for word index";
  return `ch ${c}${/^ch\d+$/.test(l) ? "" : ` (${l})`} · ToT ${f.tot[i]} · ` +
         `t ${f.meta.t0_ns + t} ns (t_rel ${t} ns) · ` +
         `fine/coarse ${f.hitFlags[i] & SMAF.HIT.MISMATCH ? "MISMATCH" : "ok"} · ${words}`;
}

/** A seed-0 hit with no other hit of its lane within 30 px, and where to point at it. */
function isolatedSeedHit(page) {
  const seed = SEEDED.meta.seeds[0];
  const canvas = byId(page, "dqm-smaev-seeds").byTag("canvas")[0];
  const lanes = laneMap(canvas.getContext("2d"));
  const X = (t) => 84 + (t + 200) / 3200 * (888 - 84);
  const [a, b] = seed.hits;
  for (let i = a; i < b; i++) {
    const lane = lanes[SEEDED.ch[i]];
    if (lane === undefined) continue;
    const x = X(SEEDED.t[i] - seed.t_rel);
    let alone = true;
    for (let j = a; j < b; j++) {
      if (j !== i && SEEDED.ch[j] === SEEDED.ch[i] &&
          Math.abs(X(SEEDED.t[j] - seed.t_rel) - x) < 30) alone = false;
    }
    if (alone) return { i, canvas, ev: { clientX: x + 1, clientY: 6 + lane * 20 + 10 } };
  }
  throw new Error("no isolated hit in seed 0");
}

/** The same for the raster: a hit alone on its row within 10 px. */
function isolatedRasterHit(f) {
  const off = f.meta.t0_ns - f.meta.frame_first_ns;
  const hi = Math.max(f.meta.span_ns / 1e6, (off + f.t[f.nHits - 1]) / 1e6);
  const X = (i) => 84 + ((off + f.t[i]) / 1e6) / hi * (790 - 84) + 0.75;
  const byCh = {};
  for (let i = 0; i < f.nHits; i++) (byCh[f.ch[i]] = byCh[f.ch[i]] || []).push(i);
  for (const ch of [1, 2, 3, 4, 5]) {
    for (const i of byCh[ch] || []) {
      if ((byCh[ch] || []).every((j) => j === i || Math.abs(X(j) - X(i)) > 10)) {
        return { i, ev: { clientX: X(i), clientY: 6 + ch * 22 + 11 } };
      }
    }
  }
  throw new Error("no isolated raster hit");
}

/** Every raster hit alone on its row (S1..S5) within 10 px, and where to point at each. */
function isolatedRasterHits(f) {
  const off = f.meta.t0_ns - f.meta.frame_first_ns;
  const hi = Math.max(f.meta.span_ns / 1e6, (off + f.t[f.nHits - 1]) / 1e6);
  const X = (i) => 84 + ((off + f.t[i]) / 1e6) / hi * (790 - 84) + 0.75;
  const byCh = {};
  for (let i = 0; i < f.nHits; i++) (byCh[f.ch[i]] = byCh[f.ch[i]] || []).push(i);
  const out = [];
  for (const ch of [1, 2, 3, 4, 5]) {
    for (const i of byCh[ch] || []) {
      if ((byCh[ch] || []).every((j) => j === i || Math.abs(X(j) - X(i)) > 10)) {
        out.push({ i, ev: { clientX: X(i), clientY: 6 + ch * 22 + 11 } });
      }
    }
  }
  assert.ok(out.length >= 3, "the fixture raster has isolated hits");
  return out;
}

/**
 * The hits of seed k a click picks out unambiguously (no other bar of the lane
 * within the 4 px slop of the click point), skipping `skipCh`, with where the
 * page draws each one's bar.
 */
function clickableSeedHits(page, k = 0, skipCh = []) {
  const seed = SEEDED.meta.seeds[k];
  const canvas = byId(page, "dqm-smaev-seeds").byTag("canvas")[k];
  const lanes = laneMap(canvas.getContext("2d"));
  const X = (t) => 84 + (t + 200) / 3200 * (888 - 84);
  const bar = (j) => {
    const px0 = X(SEEDED.t[j] - seed.t_rel);
    return [px0, Math.max(2, X(SEEDED.t[j] - seed.t_rel + SEEDED.tot[j]) - px0)];
  };
  const [a, b] = seed.hits;
  const out = [];
  for (let i = a; i < b; i++) {
    const lane = lanes[SEEDED.ch[i]];
    if (lane === undefined || skipCh.includes(SEEDED.ch[i])) continue;
    const [px0, pw] = bar(i);
    const x = px0 + Math.min(1, pw / 2);
    const near = [];
    for (let j = a; j < b; j++) {
      if (SEEDED.ch[j] !== SEEDED.ch[i]) continue;
      const [q0, qw] = bar(j);
      const d = x < q0 ? q0 - x : x > q0 + qw ? x - q0 - qw : 0;
      if (d <= 4) near.push(j);
    }
    if (near.length === 1) {
      out.push({ i, canvas, px0, pw, lane, ev: { clientX: x, clientY: 6 + lane * 20 + 10 } });
    }
  }
  return out;
}

const taggedRows = (page) => byId(page, "dqm-smaev-tagged-list").byClass("dqm-smaev-taggedrow");
const rowNum = (row) => row.byClass("dqm-smaev-tagnum")[0].textContent;
const rowLine = (row) => row.byClass("dqm-smaev-hitline")[0].textContent;
const taggedFrameTags = (page) => byId(page, "dqm-smaev-tagged-list").byClass("dqm-smaev-taggedframe")
  .map((e) => e.textContent);
/** The numbers drawn beside tagged hits in a canvas's last paint (white text: only they are). */
const markerTexts = (canvas) => lastPaint(canvas.getContext("2d")).ops
  .filter((o) => o[0] === "fillText" && o[2] === "#fff").map((o) => String(o[1][0]));
const markers = (canvas) => lastPaint(canvas.getContext("2d")).ops
  .filter((o) => o[0] === "fillText" && o[2] === "#fff").map((o) => ({ n: String(o[1][0]), x: o[1][1], y: o[1][2] }));

test("the tag bar shows the analyzer's tag verbatim, and Copy tag copies it", async () => {
  const page = await boot(withRaw(), undefined, undefined, { secure: true });
  const tag = SEEDED.meta.tag;
  assert.match(tag, TAG_RE);
  assert.strictEqual(byId(page, "dqm-smaev-tag-seeded").textContent, tag);
  const btn = byId(page, "dqm-smaev-copytag-seeded");
  assert.strictEqual(btn.textContent, "Copy tag");
  btn.onclick();
  await drain();
  assert.deepStrictEqual(page.clipboard, [tag], "navigator.clipboard in a secure context");
  assert.deepStrictEqual(page.copied, [], "no execCommand fallback needed");
  assert.strictEqual(btn.textContent, "Copied ✓");
  page.flushTimers();
  assert.strictEqual(btn.textContent, "Copy tag", "back after a moment");
});

test("in an insecure context (plain http) copying falls back to execCommand", async () => {
  const page = await boot(withRaw());                       // isSecureContext false
  byId(page, "dqm-smaev-copytag-seeded").onclick();
  await drain();
  assert.deepStrictEqual(page.clipboard, [], "navigator.clipboard is not used over http");
  assert.deepStrictEqual(page.copied, [SEEDED.meta.tag]);
  assert.strictEqual(page.doc.body.byTag("textarea").length, 0, "the helper textarea is removed");
  assert.strictEqual(byId(page, "dqm-smaev-copytag-seeded").textContent, "Copied ✓");
});

test("a rejected clipboard falls back too, and a refused execCommand shows the text", async () => {
  const rej = await boot(withRaw(), undefined, undefined, { secure: true, clipboard: "reject" });
  byId(rej, "dqm-smaev-copytag-seeded").onclick();
  await drain();
  assert.deepStrictEqual(rej.copied, [SEEDED.meta.tag], "rejected writeText -> execCommand");

  const page = await boot(withRaw(), undefined, undefined, { execCommand: false });
  const btn = byId(page, "dqm-smaev-copytag-seeded");
  btn.onclick();
  await drain();
  const pop = byId(page, "dqm-smaev-popup");
  assert.ok(pop, "a popup with the text");
  const ta = pop.byTag("textarea")[0];
  assert.strictEqual(ta.value, SEEDED.meta.tag);
  assert.ok(/Ctrl-C/.test(pop.textContent));
  assert.strictEqual(btn.textContent, "Copy tag", "not claimed as copied");
  pop.byTag("button")[0].onclick();
  assert.ok(byId(page, "dqm-smaev-popup") === null, "Close removes it");
});

test("Download raw event saves the sma::raw bytes as sma_run<run>_serial<serial>.mid", async () => {
  const page = await boot(withRaw());
  const dl = byId(page, "dqm-smaev-download-seeded");
  assert.strictEqual(dl.textContent, "Download raw event");
  assert.ok(!dl.disabled, "raw_held: enabled");
  dl.onclick();
  await drain();
  const raws = page.an.calls.filter((c) => c.cmd === "sma::raw");
  assert.deepStrictEqual(raws.map((c) => JSON.parse(c.args)), [{ seq: SEEDED.frameSeq }]);
  assert.strictEqual(page.blobs.length, 1);
  const { url, blob } = page.blobs[0];
  assert.strictEqual(blob.type, "application/octet-stream");
  const bytes = Buffer.from(await blob.arrayBuffer());
  assert.ok(bytes.equals(RAW_EVENT), "exactly the event bytes, no envelope");
  // A MIDAS event header: u16 id, u16 trigger mask, u32 serial, u32 time, u32 size.
  assert.strictEqual(bytes.readUInt16LE(0), SEEDED.meta.event.id);
  assert.strictEqual(bytes.readUInt32LE(4), SEEDED.meta.event.serial);
  assert.strictEqual(bytes.readUInt32LE(12) + 16, bytes.length);
  const name = `sma_run682_serial${SEEDED.meta.event.serial}.mid`;
  const link = page.clicks.find((e) => e.tagName === "A");
  assert.ok(link, "a download link was clicked");
  assert.strictEqual(link.getAttribute("download"), name);
  assert.strictEqual(link.getAttribute("href"), url);
  assert.strictEqual(page.doc.body.byTag("a").length, 0, "and removed again");
  assert.strictEqual(byId(page, "dqm-smaev-tagnote-seeded").textContent,
                     `saved ${name} (${RAW_EVENT.length.toLocaleString()} bytes)`);
  page.flushTimers();
  assert.deepStrictEqual(page.revoked, [url], "the object URL is released");
});

test("a raw event the analyzer no longer holds says so beside the button", async () => {
  const page = await boot(withRaw({
    "sma::raw": (args) => envelope("err ", new TextEncoder().encode(
      `frame ${JSON.parse(args).seq} no longer held — use the tag`)),
  }));
  byId(page, "dqm-smaev-download-seeded").onclick();
  await drain();
  assert.strictEqual(page.blobs.length, 0);
  const note = byId(page, "dqm-smaev-tagnote-seeded");
  assert.strictEqual(note.textContent, `frame ${SEEDED.frameSeq} no longer held — use the tag`);
  assert.ok(note.classList.contains("red"));
  assert.ok(!byId(page, "dqm-smaev-download-seeded").disabled, "can be tried again");
});

test("raw_held false greys the download out", async () => {
  const off = withMeta(FX.seeded, (m) => { m.raw_held = false; });
  const page = await boot({ "sma::frame": () => envelope("smaf", off) });
  const dl = byId(page, "dqm-smaev-download-seeded");
  assert.ok(dl.disabled);
  assert.ok(/no longer holds/.test(dl.getAttribute("title") || dl.title));
  assert.strictEqual(byId(page, "dqm-smaev-tagnote-seeded").textContent,
                     "raw event no longer held: use the tag");
});

test("hovering a seeded hit shows its line with word and raw word; a click tags it, again untags", async () => {
  const page = await boot(withRaw());
  const { i, canvas, ev } = isolatedSeedHit(page);
  const labels = SEEDED.meta.labels;
  const line = expectLine(SEEDED, i, labels);
  assert.match(line, /^ch \d+ \((S\d|RF|current)\) · ToT \d+ · t \d+ ns \(t_rel \d+ ns\) · fine\/coarse (ok|MISMATCH) · word \d+ · 0x[0-9a-f]{16}$/);
  const panel = byId(page, "dqm-smaev-seeds").byClass("dqm-sma-seed")[0];
  const hover = panel.byClass("dqm-smaev-hover")[0];
  assert.ok(/hover a hit/.test(hover.textContent), "a hint until then");
  canvas.dispatch("mousemove", ev);
  assert.strictEqual(hover.textContent, `${line} · ${SEEDED.meta.tag}`);
  canvas.dispatch("mouseleave", {});
  assert.ok(/hover a hit/.test(hover.textContent));

  assert.strictEqual(byId(page, "dqm-smaev-tagged-title").textContent, "Tagged hits (0)");
  assert.ok(byId(page, "dqm-smaev-tagged-copy").disabled, "nothing to copy yet");
  canvas.dispatch("click", ev);
  assert.strictEqual(byId(page, "dqm-smaev-tagged-title").textContent, "Tagged hits (1)");
  const rows = taggedRows(page);
  assert.strictEqual(rows.length, 1);
  assert.strictEqual(rowNum(rows[0]), "1");
  assert.strictEqual(rowLine(rows[0]), line, "the first row of a frame has no Δt");
  assert.deepStrictEqual(taggedFrameTags(page), [SEEDED.meta.tag], "listed under its frame tag");
  const paint = lastPaint(canvas.getContext("2d"));
  assert.strictEqual(paint.count("strokeRect", "#000"), 1, "the tagged hit is boxed");
  assert.deepStrictEqual(markerTexts(canvas), ["1"], "and numbered");
  rows[0].byClass("dqm-smaev-copyhit")[0].onclick();
  await drain();
  assert.deepStrictEqual(page.copied, [`${line} · ${SEEDED.meta.tag}`], "the old pinned-line text");

  // The tag stays across a redraw of the same frame, and a second click untags.
  await settle(page);
  assert.strictEqual(taggedRows(page).length, 1);
  canvas.dispatch("click", ev);
  assert.strictEqual(taggedRows(page).length, 0);
  assert.strictEqual(byId(page, "dqm-smaev-tagged-title").textContent, "Tagged hits (0)");
  assert.strictEqual(lastPaint(canvas.getContext("2d")).count("strokeRect", "#000"), 0);
  assert.deepStrictEqual(markerTexts(canvas), []);
});

test("Copy seed words copies the tag, the S1 word, the window's range and every hit's word", async () => {
  const page = await boot(withRaw());
  const seed = SEEDED.meta.seeds[1];
  assert.ok(seed.word_range && Number.isInteger(seed.s1_word), "the fixture seed carries words");
  const [a, b] = seed.hits;
  const words = Array.from(SEEDED.wordIndex.slice(a, b)).sort((x, y) => x - y);
  const [pa, pb] = SEEDED.meta.mupix.seeds[1].pix;
  const pwords = Array.from(SEEDED.pixels.wordIndex.slice(pa, pb)).sort((x, y) => x - y);
  assert.ok(pwords.length > 0, "the fixture seed has MuPix hits");
  const want = `${SEEDED.meta.tag} · seed 2: S1 word ${seed.s1_word} · ` +
               `words ${seed.word_range[0]}–${seed.word_range[1]} (${b - a} hits in window) · ` +
               `hit words ${words.join(", ")} · MuPix pixel words ${pwords.join(", ")}`;
  const panel = byId(page, "dqm-smaev-seeds").byClass("dqm-sma-seed")[1];
  assert.ok(panel.textContent.includes(`words ${seed.word_range[0]}–${seed.word_range[1]}`), "a badge");
  const btn = panel.byClass("dqm-smaev-copyseed")[0];
  btn.onclick();
  await drain();
  assert.deepStrictEqual(page.copied, [want]);
  assert.strictEqual(btn.textContent, "Copied ✓");
  // The button is not rebuilt by a redraw, so a click that spans one still lands.
  await settle(page);
  assert.ok(byId(page, "dqm-smaev-seeds").byClass("dqm-sma-seed")[1].byClass("dqm-smaev-copyseed")[0] === btn,
    "the same copy button");
});

test("a click on the live raster freezes, fetches the words once, then tags the hit", async () => {
  const page = await boot(withRaw(), { tab: "raster" });
  const canvas = byId(page, "dqm-smaev-raster");
  const [h1, h2] = isolatedRasterHits(RASTER);
  const hover = byId(page, "dqm-smaev-pane-raster").byClass("dqm-smaev-hover")[0];
  assert.ok(/a click freezes the frame first/.test(hover.textContent), hover.textContent);
  canvas.dispatch("mousemove", h1.ev);
  const live = expectLine(RASTER, h1.i, RASTER.meta.labels);
  assert.ok(live.endsWith(" · freeze for word index"), live);
  assert.strictEqual(hover.textContent, `${live} · ${RASTER.meta.tag}`);

  // mousedown + release in place is a click, not a drag.
  const mark = page.an.calls.length;
  canvas.dispatch("mousedown", h1.ev);
  page.windowEvent("mouseup", h1.ev);
  assert.strictEqual(byId(page, "dqm-smaev-live").textContent, "FROZEN", "frozen at once");
  assert.strictEqual(taggedRows(page).length, 0, "not tagged before the words are in");
  assert.ok(/tagged once the frozen frame's word data is in/.test(byId(page, "dqm-smaev-tagged-note").textContent));
  await settle(page, 4);
  assert.deepStrictEqual(frameCalls(page, mark), [{ view: "raster", drop: [7], max_hits: 60000, words: true,
                                                    seq: RASTER.frameSeq }], "the frame on screen, once, with words");
  assert.strictEqual(RASTER_WORDS.nHits, RASTER.nHits, "the same hits");
  const frozen1 = expectLine(RASTER_WORDS, h1.i, RASTER.meta.labels);
  assert.match(frozen1, / · word \d+ · 0x[0-9a-f]{16}$/);
  let rows = taggedRows(page);
  assert.strictEqual(rows.length, 1);
  assert.strictEqual(rowLine(rows[0]), frozen1, "tagged with its word");
  assert.strictEqual(byId(page, "dqm-smaev-tagged-note").textContent, "");
  canvas.dispatch("mousemove", h1.ev);
  assert.strictEqual(hover.textContent, `${frozen1} · ${RASTER.meta.tag}`);
  assert.strictEqual(lastPaint(canvas.getContext("2d")).count("strokeRect", "#000"), 1, "boxed");
  assert.deepStrictEqual(markerTexts(canvas), ["1"]);

  // Frozen with words: the next click tags at once, with its Δt.
  canvas.dispatch("mousedown", h2.ev);
  page.windowEvent("mouseup", h2.ev);
  rows = taggedRows(page);
  assert.strictEqual(rows.length, 2);
  const dt = RASTER_WORDS.t[h2.i] - RASTER_WORDS.t[h1.i];
  assert.strictEqual(rowLine(rows[1]),
    `${expectLine(RASTER_WORDS, h2.i, RASTER.meta.labels)} · Δt ${dt > 0 ? "+" : ""}${dt} ns`);
  assert.deepStrictEqual(markerTexts(canvas).sort(), ["1", "2"]);

  // Nothing more while frozen: the words are asked for once.
  const again = page.an.calls.length;
  await settle(page, 3);
  assert.strictEqual(frameCalls(page, again).length, 0);
});

test("a frozen raster whose frame is gone says why it has no word index", async () => {
  const page = await boot(withRaw({
    "sma::frame": (args) => {
      const a = JSON.parse(args || "{}");
      if (a.seq !== undefined) {
        return envelope("err ", new TextEncoder().encode(`frame ${a.seq} no longer held — use the tag`));
      }
      return envelope("smaf", Buffer.from(FX.raster, "hex"));
    },
  }), { tab: "raster" });
  byId(page, "dqm-smaev-freeze").onclick();
  await settle(page, 4);
  const { i, ev } = isolatedRasterHit(RASTER);
  byId(page, "dqm-smaev-raster").dispatch("mousemove", ev);
  const hover = byId(page, "dqm-smaev-pane-raster").byClass("dqm-smaev-hover")[0];
  assert.ok(hover.textContent.includes(
    `fine/coarse ${RASTER.hitFlags[i] & 1 ? "MISMATCH" : "ok"} · no word index: frame ${RASTER.frameSeq} no longer held — use the tag`),
    hover.textContent);
  assert.strictEqual(byId(page, "dqm-smaev-live").textContent, "FROZEN", "not an analyzer failure");
});

test("an analyzer without tags or raw events still gets a usable tag bar", async () => {
  const old = withMeta(FX.seeded, (m) => { delete m.tag; delete m.raw_held; delete m.event; });
  const page = await boot({ "sma::frame": () => envelope("smaf", old) });
  assert.strictEqual(byId(page, "dqm-smaev-tag-seeded").textContent,
                     `SMA run 682 · serial ${SEEDED.meta.serial} · frame seq ${SEEDED.frameSeq}`);
  const dl = byId(page, "dqm-smaev-download-seeded");
  assert.ok(dl.disabled, "no raw_held: no download");
  assert.ok(/does not hand out raw events/.test(dl.getAttribute("title") || dl.title));
});

test("Freeze keeps the raw event, so Download works after the analyzer drops it", async () => {
  let held = true;
  const page = await boot(withRaw({
    "sma::raw": (args) => {
      const a = JSON.parse(args || "{}");
      if (held && a.seq === FX.raw_seq) return envelope("mevt", RAW_EVENT);
      return envelope("err ", new TextEncoder().encode(`frame ${a.seq} no longer held — use the tag`));
    },
  }));
  const mark = page.an.calls.length;
  byId(page, "dqm-smaev-freeze").onclick();
  await settle(page, 3);
  const raws = page.an.calls.slice(mark).filter((c) => c.cmd === "sma::raw");
  assert.deepStrictEqual(raws.map((c) => JSON.parse(c.args)), [{ seq: SEEDED.frameSeq }], "fetched once at Freeze");
  assert.strictEqual(page.blobs.length, 0, "kept, not saved");
  assert.ok(/kept by this page/.test(byId(page, "dqm-smaev-download-seeded").title));

  held = false;                                  // the analyzer's ring moved on
  byId(page, "dqm-smaev-download-seeded").onclick();
  await drain();
  assert.strictEqual(page.blobs.length, 1);
  assert.ok(Buffer.from(await page.blobs[0].blob.arrayBuffer()).equals(RAW_EVENT));
  assert.strictEqual(page.an.calls.slice(mark).filter((c) => c.cmd === "sma::raw").length, 1,
                     "no second request");
});

test("Single keeps the raw event of the frame it steps to, not of the one it left", async () => {
  let next = false;
  const page = await boot(withRaw({
    "sma::frame": () => envelope("smaf", next ? withSeq(FX.seeded, SEEDED.frameSeq + 1)
                                              : Buffer.from(FX.seeded, "hex")),
    "sma::raw": () => envelope("mevt", RAW_EVENT),
  }));
  next = true;
  const mark = page.an.calls.length;
  byId(page, "dqm-smaev-single").onclick();
  await settle(page, 5);
  const raws = page.an.calls.slice(mark).filter((c) => c.cmd === "sma::raw").map((c) => JSON.parse(c.args));
  assert.deepStrictEqual(raws, [{ seq: SEEDED.frameSeq + 1 }]);
});

// -- tagging several hits ------------------------------------------------------------

/** Where the page puts a one-digit number box beside a seeded bar (see tagMarker). */
function seedMarkerAt(h) {
  const w = 13, x1 = 888;
  const bx = h.px0 + h.pw + 3 + w > x1 ? h.px0 - 3 - w : h.px0 + h.pw + 3;
  return { x: bx + w / 2, y: 6 + h.lane * 20 + 3 + 0.5 + 6.5 + 0.5 };
}

/** The seeded fixture as a different frame: another serial, seq and tag. */
function otherFrame(serial, seq) {
  const raw = withMeta(FX.seeded, (m) => {
    m.serial = serial;
    m.event.serial = serial;
    m.seq = seq;
    m.tag = m.tag.replace(/serial \d+/, `serial ${serial}`).replace(/frame seq \d+$/, `frame seq ${seq}`);
  });
  raw.writeBigUInt64LE(BigInt(seq), 8);
  return raw;
}

test("tagging three hits numbers them 1-3 on the canvas and in the list; removing one renumbers both", async () => {
  const page = await boot(withRaw());
  const hits = clickableSeedHits(page, 0).slice(0, 3);
  assert.strictEqual(hits.length, 3, "three clickable hits in seed 1");
  hits.forEach((h) => h.canvas.dispatch("click", h.ev));
  const canvas = hits[0].canvas;
  let rows = taggedRows(page);
  assert.deepStrictEqual(rows.map(rowNum), ["1", "2", "3"]);
  const labels = SEEDED.meta.labels;
  const dt = (h) => {
    const d = SEEDED.t[h.i] - SEEDED.t[hits[0].i];
    return ` · Δt ${d > 0 ? "+" : ""}${d} ns`;
  };
  assert.deepStrictEqual(rows.map(rowLine), [expectLine(SEEDED, hits[0].i, labels),
    expectLine(SEEDED, hits[1].i, labels) + dt(hits[1]), expectLine(SEEDED, hits[2].i, labels) + dt(hits[2])]);
  const at = (ms) => ms.map((m) => [m.n, Math.round(m.x * 10) / 10, Math.round(m.y * 10) / 10]);
  const want = (list) => list.map((h, k) => {
    const p = seedMarkerAt(h);
    return [String(k + 1), Math.round(p.x * 10) / 10, Math.round(p.y * 10) / 10];
  });
  assert.deepStrictEqual(at(markers(canvas)), want(hits), "each number beside its own hit");
  assert.strictEqual(lastPaint(canvas.getContext("2d")).count("strokeRect", "#000"), 3);

  // Remove #2 with its x: #3 becomes #2, on the canvas too.
  rows[1].byClass("dqm-smaev-untag")[0].onclick();
  rows = taggedRows(page);
  assert.deepStrictEqual(rows.map(rowNum), ["1", "2"]);
  assert.strictEqual(rowLine(rows[1]), expectLine(SEEDED, hits[2].i, labels) + dt(hits[2]));
  assert.deepStrictEqual(at(markers(canvas)), want([hits[0], hits[2]]));
  assert.strictEqual(byId(page, "dqm-smaev-tagged-title").textContent, "Tagged hits (2)");

  // Removing the first by clicking it again: the Δt is now from the new first.
  canvas.dispatch("click", hits[0].ev);
  rows = taggedRows(page);
  assert.deepStrictEqual(rows.map(rowNum), ["1"]);
  assert.strictEqual(rowLine(rows[0]), expectLine(SEEDED, hits[2].i, labels), "no Δt on a frame's first");
  assert.deepStrictEqual(at(markers(canvas)), want([hits[2]]));

  byId(page, "dqm-smaev-tagged-clear").onclick();
  assert.strictEqual(taggedRows(page).length, 0);
  assert.deepStrictEqual(markerTexts(canvas), []);
});

test("tags survive new frames, group by frame with Δt per frame, and Copy all has the exact text", async () => {
  let next = false;
  const B = otherFrame(29, 27);
  const page = await boot(withRaw({
    "sma::frame": () => envelope("smaf", next ? B : Buffer.from(FX.seeded, "hex")),
  }));
  const [h1, h2] = clickableSeedHits(page, 0);
  const h3 = clickableSeedHits(page, 1)[0];
  h1.canvas.dispatch("click", h1.ev);
  h2.canvas.dispatch("click", h2.ev);

  next = true;                                          // live polling moves on to frame B
  await settle(page, 3);
  assert.strictEqual(seqText(page), "frame seq 27");
  assert.strictEqual(taggedRows(page).length, 2, "a new frame does not clear the list");
  assert.deepStrictEqual(markerTexts(h1.canvas), [], "frame A's tags are not drawn on frame B");
  h3.canvas.dispatch("click", h3.ev);

  const tagA = SEEDED.meta.tag;
  const tagB = tagA.replace("serial 28", "serial 29").replace("frame seq 26", "frame seq 27");
  assert.deepStrictEqual(taggedFrameTags(page), [tagA, tagB], "one group per frame");
  const rows = taggedRows(page);
  assert.deepStrictEqual(rows.map(rowNum), ["1", "2", "3"]);
  assert.deepStrictEqual(markerTexts(h3.canvas), ["3"], "B's hit, numbered as in the list");
  assert.deepStrictEqual(markerTexts(h1.canvas), []);

  const labels = SEEDED.meta.labels;
  const d = SEEDED.t[h2.i] - SEEDED.t[h1.i];
  const wA = [SEEDED.wordIndex[h1.i], SEEDED.wordIndex[h2.i]];
  const want = [
    tagA,
    `  #1 · ${expectLine(SEEDED, h1.i, labels)}`,
    `  #2 · ${expectLine(SEEDED, h2.i, labels)} · Δt ${d > 0 ? "+" : ""}${d} ns`,
    `  mdqm-sma-file --serial 28 --run 682 --dir <raw dir> --words ${Math.min(...wA)}:${Math.max(...wA)}`,
    "",
    tagB,
    `  #3 · ${expectLine(SEEDED, h3.i, labels)}`,
    `  mdqm-sma-file --serial 29 --run 682 --dir <raw dir> --words ${SEEDED.wordIndex[h3.i]}:${SEEDED.wordIndex[h3.i]}`,
  ].join("\n");
  const btn = byId(page, "dqm-smaev-tagged-copy");
  btn.onclick();
  await drain();
  assert.deepStrictEqual(page.copied, [want]);
  assert.strictEqual(btn.textContent, "Copied ✓");

  // Each tag keeps its frame's whole identity from click time.
  const kept = JSON.parse(globalThis.sessionStorage.getItem("dqm-sma-events-tagged"));
  assert.deepStrictEqual(kept.map((x) => [x.run, x.eventId, x.serial, x.seq, x.utc]),
    [[682, 301, 28, 26, "2026-09-21T14:13:20.000Z"], [682, 301, 28, 26, "2026-09-21T14:13:20.000Z"],
     [682, 301, 29, 27, "2026-09-21T14:13:20.000Z"]]);

  // Back on frame A (a Single or the analyzer's next frame): its marks return.
  next = false;
  await settle(page, 3);
  assert.deepStrictEqual(markerTexts(h1.canvas).sort(), ["1", "2"]);
});

test("the list survives a tab switch, and a seeded tag is marked (and untagged) on the frozen raster", async () => {
  const page = await boot(withRaw());
  const h = clickableSeedHits(page, 0, [FX.current_channel])[0];
  h.canvas.dispatch("click", h.ev);
  const word = SEEDED.wordIndex[h.i];

  byId(page, "dqm-smaev-tab-raster").onclick();
  await settle(page, 3);
  assert.strictEqual(taggedRows(page).length, 1, "kept across the switch");
  assert.ok(byId(page, "dqm-smaev-tagged").parent !== byId(page, "dqm-smaev-pane-seeded"),
    "the list is outside the panes, so it shows on both tabs");
  const canvas = byId(page, "dqm-smaev-raster");
  assert.deepStrictEqual(markerTexts(canvas), [], "the live raster has no words to find the hit by");

  byId(page, "dqm-smaev-freeze").onclick();
  await settle(page, 4);
  const j = Array.from(RASTER_WORDS.wordIndex).indexOf(word);
  assert.ok(j >= 0, "the seeded hit is in the raster");
  assert.deepStrictEqual(markerTexts(canvas), ["1"], "found by its H000 word");

  // A click on the same hit in the raster is the same hit: it untags.
  const off = RASTER.meta.t0_ns - RASTER.meta.frame_first_ns;
  const hi = Math.max(RASTER.meta.span_ns / 1e6, (off + RASTER.t[RASTER.nHits - 1]) / 1e6);
  const ev = { clientX: 84 + ((off + RASTER.t[j]) / 1e6) / hi * (790 - 84) + 0.75,
               clientY: 6 + RASTER.ch[j] * 22 + 11 };
  canvas.dispatch("mousedown", ev);
  page.windowEvent("mouseup", ev);
  assert.strictEqual(taggedRows(page).length, 0);

  byId(page, "dqm-smaev-tab-seeded").onclick();
  await settle(page);
  assert.deepStrictEqual(markerTexts(h.canvas), []);
});

test("Esc clears the hover line but not the list", async () => {
  const page = await boot(withRaw());
  const [h] = clickableSeedHits(page, 0);
  h.canvas.dispatch("mousemove", h.ev);
  h.canvas.dispatch("click", h.ev);
  const hover = byId(page, "dqm-smaev-seeds").byClass("dqm-smaev-hover")[0];
  assert.ok(/^ch /.test(hover.textContent));
  page.windowEvent("keydown", { key: "Escape" });
  assert.ok(/hover a hit/.test(hover.textContent), "the hover is back to its hint");
  assert.strictEqual(taggedRows(page).length, 1, "the list is kept");
});

test("the list is capped, says so, and a reload keeps it (sessionStorage)", async () => {
  // 199 tags of some other frame, as a reload would find them.
  const filler = Array.from({ length: 199 }, (_, k) => ({
    key: `x|${k}`, frame: "682|301|1|1", tag: "SMA run 682 · event 301 serial 1 · frame seq 1",
    run: 682, eventId: 301, serial: 1, seq: 1, utc: null, view: "seeded", drop: "[]", i: k,
    word: 100 + k, raw: null, ch: 1, tot: 1, tRel: k, tAbs: k, line: `hit ${k}`,
  }));
  const session = { _d: { "dqm-sma-events-tagged": JSON.stringify(filler) },
                    getItem(k) { return this._d[k] || null; }, setItem(k, v) { this._d[k] = v; } };
  const page = await boot(withRaw(), undefined, undefined, { session });
  assert.strictEqual(byId(page, "dqm-smaev-tagged-title").textContent, "Tagged hits (199)", "restored");
  const [h1, h2] = clickableSeedHits(page, 0);
  h1.canvas.dispatch("click", h1.ev);
  assert.strictEqual(byId(page, "dqm-smaev-tagged-title").textContent, "Tagged hits (200)");
  assert.deepStrictEqual(markerTexts(h1.canvas), ["200"]);
  assert.ok(/at most 200/.test(byId(page, "dqm-smaev-tagged-note").textContent));
  h2.canvas.dispatch("click", h2.ev);
  assert.strictEqual(taggedRows(page).length, 200, "no 201st");
  const note = byId(page, "dqm-smaev-tagged-note");
  assert.ok(/the list is full \(200 hits\)/.test(note.textContent), note.textContent);
  assert.ok(note.classList.contains("full"));
  // Untagging one makes room again.
  h1.canvas.dispatch("click", h1.ev);
  h2.canvas.dispatch("click", h2.ev);
  assert.strictEqual(taggedRows(page).length, 200);
  assert.deepStrictEqual(markerTexts(h2.canvas), ["200"]);

  // A reload with the same session storage finds the same list.
  const again = await boot(withRaw(), undefined, undefined, { session });
  assert.strictEqual(taggedRows(again).length, 200);
  assert.strictEqual(rowLine(taggedRows(again)[199]), expectLine(SEEDED, h2.i, SEEDED.meta.labels));
  assert.deepStrictEqual(markerTexts(clickableSeedHits(again, 0)[1].canvas), ["200"]);
});

test("without sessionStorage the page still tags", async () => {
  const page = await boot(withRaw());
  globalThis.sessionStorage = { getItem() { throw new Error("SecurityError"); },
                                setItem() { throw new Error("SecurityError"); } };
  const [h] = clickableSeedHits(page, 0);
  h.canvas.dispatch("click", h.ev);
  assert.strictEqual(taggedRows(page).length, 1);
});

// -- seed choice, filters, the staleness banner --------------------------------------

const D = (name) => SMAF.decode(new Uint8Array(Buffer.from(FX[name], "hex")));
const STALE = D("seeded_stale");
const NOMATCH = D("seeded_nomatch");
const ANY_NOS1 = D("seeded_any_nos1");
const FILTERED = D("seeded_filtered");

/** An analyzer answering the seeded view with a fixture chosen by the request args. */
function seededBy(pick) {
  return {
    "sma::frame": (args) => {
      const a = JSON.parse(args || "{}");
      if (a.view === "raster") return envelope("smaf", Buffer.from(FX.raster, "hex"));
      return envelope("smaf", Buffer.from(FX[pick(a)], "hex"));
    },
  };
}

const session = (d) => ({ _d: d || {}, getItem(k) { return this._d[k] || null; },
                          setItem(k, v) { this._d[k] = v; } });
const banner = (page) => byId(page, "dqm-smaev-banner");
const ageText = (x) => (x < 10 ? `${x.toFixed(1)} s` : `${Math.round(x)} s`);

test("the seeded fixtures carry what the page is tested on", () => {
  assert.ok(STALE.meta.stale_view && STALE.meta.stale_view.good_since === 12);
  assert.ok(NOMATCH.meta.search.no_match && NOMATCH.meta.seeds.length === 0);
  assert.ok(ANY_NOS1.meta.seeds.every((s) => s.rf_na && s.seed_ch !== 1));
  assert.ok(FILTERED.meta.select && FILTERED.meta.seeds.every((s) => s.odd.length));
  assert.ok(!SEEDED.meta.stale_view && !SEEDED.meta.select, "the default frame is as before");
});

test("the default request sends seed s1 and no filters; the dropdown lists the roles", async () => {
  const page = await boot();
  for (const a of frameCalls(page)) {
    assert.strictEqual(a.seed, "s1");
    assert.deepStrictEqual(a.filters, []);
  }
  const sel = byId(page, "dqm-smaev-seedsel");
  assert.strictEqual(sel.value, "s1");
  const opts = sel.byTag("option").map((o) => [o.getAttribute("value"), o.textContent]);
  assert.deepStrictEqual(opts.map((o) => o[0]), ["s1", "ch2", "ch3", "ch4", "ch5", "ch8", "ch9", "ch10", "any"]);
  assert.strictEqual(opts[0][1], "S1 (default)");
  assert.strictEqual(opts[1][1], "S2 (ch 2)");
  assert.ok(!opts.some((o) => o[0] === "ch6" || o[0] === `ch${FX.current_channel}`), "no RF, no current");
  assert.strictEqual(byId(page, "dqm-smaev-tab-seeded").textContent, "S1-seeded events");
  assert.strictEqual(banner(page).style.display, "none", "a fresh frame has no banner");
});

test("choosing a seed and filters asks at once with them, and they persist per viewer", async () => {
  const ss = session();
  const page = await boot(seededBy((a) => (a.seed === "any" ? "seeded_any" : "seeded")),
                          undefined, undefined, { session: ss });
  const sel = byId(page, "dqm-smaev-seedsel");
  let mark = page.an.calls.length;
  sel.value = "any";
  sel.onchange.call(sel);
  await settle(page);
  let asks = frameCalls(page, mark);
  assert.ok(asks.length > 0 && asks.every((a) => a.seed === "any"), JSON.stringify(asks));
  assert.strictEqual(byId(page, "dqm-smaev-tab-seeded").textContent, "Seeded events: any counter");
  const head = byId(page, "dqm-smaev-seededhead").textContent;
  const m = D("seeded_any").meta.select;
  assert.ok(head.includes(`matching seeds: ${m.matching.toLocaleString()} of ${m.candidates.toLocaleString()} ` +
                          `candidates in frame seq ${D("seeded_any").frameSeq}`), head);

  mark = page.an.calls.length;
  for (const name of ["rf", "tot"]) {
    const box = byId(page, `dqm-smaev-filter-${name}`);
    box.checked = true;
    box.onchange.call(box);
  }
  await settle(page);
  asks = frameCalls(page, mark);
  assert.deepStrictEqual(asks[asks.length - 1].filters, ["tot", "rf"], "in the analyzer's order");
  assert.deepStrictEqual(JSON.parse(ss._d["dqm-sma-events-seed"]),
                         { seed: "any", filters: ["tot", "rf"], mupix: "any", pattern: {} });

  // A reload in the same tab keeps the choice.
  const again = await boot(undefined, undefined, undefined, { session: ss });
  const first = frameCalls(again)[0];
  assert.strictEqual(first.seed, "any");
  assert.deepStrictEqual(first.filters, ["tot", "rf"]);
  assert.strictEqual(byId(again, "dqm-smaev-seedsel").value, "any");
  assert.ok(byId(again, "dqm-smaev-filter-tot").checked && byId(again, "dqm-smaev-filter-rf").checked);
  assert.ok(!byId(again, "dqm-smaev-filter-mismatch").checked);
});

test("a stored choice that is garbage falls back to S1, and no storage is fine", async () => {
  const page = await boot(undefined, undefined, undefined,
                          { session: session({ "dqm-sma-events-seed": '{"seed":"x;y","filters":["odd","tot"]}' }) });
  const a = frameCalls(page)[0];
  assert.strictEqual(a.seed, "s1");
  assert.deepStrictEqual(a.filters, ["tot"]);
  const broken = { getItem() { throw new Error("SecurityError"); }, setItem() { throw new Error("SecurityError"); } };
  const p2 = await boot(undefined, undefined, undefined, { session: broken });
  const box = byId(p2, "dqm-smaev-filter-mismatch");
  box.checked = true;
  box.onchange.call(box);
  await settle(p2);
  assert.deepStrictEqual(frameCalls(p2).pop().filters, ["mismatch"]);
});

test("a stale frame gets the banner, a fresh one hides it", async () => {
  let pick = "seeded_stale";
  const page = await boot(seededBy(() => pick));
  const b = banner(page);
  const sv = STALE.meta.stale_view;
  assert.notStrictEqual(b.style.display, "none");
  assert.ok(b.classList.contains("yellow"));
  assert.strictEqual(b.textContent,
    `Showing frame seq ${sv.shown_seq} from ${ageText(sv.age_s)} ago: no S1 hits in the 12 good ` +
    `frames analysed since (newest seq ${sv.newest_seq}).`);
  pick = "seeded";
  await settle(page);
  assert.strictEqual(b.style.display, "none");
  assert.strictEqual(b.textContent, "");
});

test("a walked-back frame whose raw event is gone says so in the banner", async () => {
  const raw = withMeta(FX.seeded_stale, (m) => { m.raw_held = false; });
  const page = await boot({ "sma::frame": () => envelope("smaf", raw) });
  assert.ok(/raw event is no longer held: use the tag/.test(banner(page).textContent));
  assert.ok(byId(page, "dqm-smaev-download-seeded").disabled);
});

test("no match is its own state: a red banner, no panels, 0 matching", async () => {
  const page = await boot(seededBy(() => "seeded_nomatch"));
  const b = banner(page);
  const nm = NOMATCH.meta.search.no_match;
  assert.ok(b.classList.contains("red"));
  assert.ok(b.textContent.startsWith(`No match: ${nm.text} (${ageText(nm.span_s)}). Showing the newest ` +
                                     `analysed frame, seq ${NOMATCH.frameSeq}`), b.textContent);
  assert.strictEqual(byId(page, "dqm-smaev-seeds").byClass("dqm-sma-seed").length, 0);
  assert.ok(/matching seeds: 0 of/.test(byId(page, "dqm-smaev-seededhead").textContent));
  assert.strictEqual(byId(page, "dqm-smaev-note-seeded").textContent, "", "the banner says it, once");
  // The tag bar still works on the frame shown.
  assert.strictEqual(byId(page, "dqm-smaev-tag-seeded").textContent, NOMATCH.meta.tag);
});

test("a seed without S1 names its channel, says RF n/a, and copies its own word", async () => {
  const page = await boot(seededBy(() => "seeded_any_nos1"));
  const panels = byId(page, "dqm-smaev-seeds").byClass("dqm-sma-seed");
  assert.strictEqual(panels.length, ANY_NOS1.meta.seeds.length);
  const s0 = ANY_NOS1.meta.seeds[0];
  const lab = ANY_NOS1.meta.labels[s0.seed_ch];
  const title = panels[0].byClass("dqm-sma-seedtitle")[0].textContent;
  assert.ok(title.startsWith(`seed 1: ${lab} (ch ${s0.seed_ch}) at `) && title.endsWith(`ToT ${s0.seed_tot}`), title);
  assert.ok(panels[0].textContent.includes("RF n/a (no S1)"));
  assert.ok(panels[0].textContent.includes("odd: incomplete pattern"));
  assert.ok(!panels[0].byClass("lit").some((b) => b.textContent === "S1"), "S1 not lit");
  assert.ok(lastPaint(panels[0].byTag("canvas")[0].getContext("2d")).texts().includes("ns from the seed"));
  const btn = panels[0].byClass("dqm-smaev-copyseed")[0];
  btn.onclick();
  await drain();
  assert.ok(page.copied[0].includes(`seed 1: ${lab} word ${s0.seed_word} · S1 word —`), page.copied[0]);
});

test("filtered seeds show which oddity they have", async () => {
  const page = await boot(seededBy(() => "seeded_filtered"));
  const panels = byId(page, "dqm-smaev-seeds").byClass("dqm-sma-seed");
  assert.strictEqual(panels.length, FILTERED.meta.seeds.length);
  FILTERED.meta.seeds.forEach(function (s, k) {
    assert.ok(/odd: /.test(panels[k].textContent), panels[k].textContent);
  });
  const head = byId(page, "dqm-smaev-seededhead").textContent;
  assert.ok(head.includes("seed: S1 (ch 1) with ToT ≥ 250 or RF not valid / vetoed"), head);
});

test("Single sends the current seed and filters", async () => {
  const page = await boot(advancing(), undefined, undefined,
                          { session: session({ "dqm-sma-events-seed": '{"seed":"ch2","filters":["mismatch"]}' }) });
  const mark = page.an.calls.length;
  byId(page, "dqm-smaev-single").onclick();
  await settle(page, 5);
  const asks = frameCalls(page, mark);
  assert.strictEqual(asks.length, 1);
  assert.strictEqual(asks[0].seed, "ch2");
  assert.deepStrictEqual(asks[0].filters, ["mismatch"]);
});

test("changing a filter while frozen re-asks for the frozen frame by its seq", async () => {
  const page = await boot();
  byId(page, "dqm-smaev-freeze").onclick();
  await settle(page);
  const mark = page.an.calls.length;
  const box = byId(page, "dqm-smaev-filter-incomplete");
  box.checked = true;
  box.onchange.call(box);
  await settle(page, 5);
  const asks = frameCalls(page, mark);
  assert.strictEqual(asks.length, 1, JSON.stringify(asks));
  assert.strictEqual(asks[0].seq, SEEDED.frameSeq);
  assert.deepStrictEqual(asks[0].filters, ["incomplete"]);
  assert.strictEqual(byId(page, "dqm-smaev-live").textContent, "FROZEN", "still frozen");
});

// -- MuPix: seeded lanes and badges, the selector, raster rows, tagging pixel hits ----------

const IN_TIME_FILL = "rgba(44, 160, 44, 0.16)";
const MP_BOTH = D("seeded_mupix_both");
const MP_NONE = D("seeded_mupix_none");

/** The hover line of pixel j, built here from the layout and the spec, not the page. */
function expectPixelLine(f, j) {
  const p = f.pixels, mp = f.meta.mupix;
  const plane = p.flags[j] & 3;
  const rel = mp.t0_ns - f.meta.t0_ns + p.t[j];
  const words = p.rawWord
    ? `word ${p.wordIndex[j]} · 0x${p.rawWord[j].toString(16).padStart(16, "0")}`
    : "freeze for word index";
  return `${plane ? `MuPix L${plane} chip ${p.chip[j]}` : `MuPix chip ${p.chip[j]} (no plane)`} · ` +
         `col ${p.col[j]} · row ${p.row[j]}${p.flags[j] & 4 ? " (not on the sensor)" : ""} · ` +
         `ToT ${p.tot[j]} (~${mp.tot_ns * p.tot[j]} ns) · t ${f.meta.t0_ns + rel} ns (t_rel ${rel} ns) · ${words}`;
}

/** The MuPix lanes of a seed canvas, from the labels drawn: {1: lane, 2: lane}. */
function pixLanes(ctx) {
  const out = {};
  for (const o of lastPaint(ctx).ops) {
    if (o[0] !== "fillText" || o[1][1] !== 84 - 6) continue;
    const m = /^MuPix L([12])$/.exec(String(o[1][0]));
    if (m) out[Number(m[1])] = Math.round((o[1][2] - 6 - 10) / 20);
  }
  return out;
}

test("the MuPix fixtures carry what the page is tested on", () => {
  assert.ok(SEEDED.pixels && SEEDED.pixels.words && SEEDED.pixels.n > 0, "seeded: a pixel block with words");
  assert.strictEqual(SEEDED.meta.mupix.seeds.length, SEEDED.meta.seeds.length);
  assert.ok(RASTER.pixels && !RASTER.pixels.words, "the polled raster's pixels have no words");
  assert.ok(RASTER_WORDS.pixels.words, "the frozen raster's pixels have words");
  assert.strictEqual(MP_BOTH.meta.select.mupix, "both");
  assert.ok(MP_BOTH.meta.mupix.seeds.every((s) => s.l1 > 0 && s.l2 > 0));
  assert.ok(MP_NONE.meta.mupix.seeds.every((s) => s.covered && s.l1 === 0 && s.l2 === 0));
});

test("every seed panel has MuPix L1 and L2 lanes, one tick per pixel hit, the in-time band and a badge", async () => {
  const page = await boot();
  const panels = byId(page, "dqm-smaev-seeds").byClass("dqm-sma-seed");
  const mps = SEEDED.meta.mupix.seeds;
  panels.forEach(function (p, k) {
    const ctx = p.byTag("canvas")[0].getContext("2d");
    const paint = lastPaint(ctx);
    const texts = paint.texts();
    assert.ok(texts.includes("MuPix L1") && texts.includes("MuPix L2"), texts.join(","));
    assert.strictEqual(paint.count("fillRect", IN_TIME_FILL), 2, "the in-time window on both lanes");
    const [pa, pb] = mps[k].pix;
    const ticks = paint.ops.filter((o) => o[0] === "fillRect" && o[1][2] === 2.5);
    assert.strictEqual(ticks.length, pb - pa, `seed ${k}: a tick per pixel hit`);
    const lanes = pixLanes(ctx);
    for (let j = pa; j < pb; j++) {
      const y = 6 + lanes[SEEDED.pixels.flags[j] & 3] * 20 + 3;
      assert.ok(ticks.some((o) => o[1][1] === y), `pixel ${j} on its plane's lane`);
    }
    const want = mps[k].l1 && mps[k].l2 ? "L1+L2 ✓" : mps[k].l1 ? "L1 ✓" : mps[k].l2 ? "L2 ✓" : "no MuPix in time";
    const badge = p.byClass("dqm-sma-badge").find((b) => b.textContent === want);
    assert.ok(badge, `seed ${k}: badge ${want}`);
    assert.ok(/MuPix hits · words \d+–\d+/.test(p.textContent), "the seed's pixel words");
  });
  assert.deepStrictEqual(globalThis.__alerts, []);
});

test("a seed past the pixel data without a hit says MuPix n/a; with one, a hit is a hit", async () => {
  const f = withMeta(FX.seeded, (m) => {
    m.mupix.seeds[0].covered = false; m.mupix.seeds[0].l1 = 0; m.mupix.seeds[0].l2 = 0;
    m.mupix.seeds[1].l1 = 0; m.mupix.seeds[1].l2 = 0;
    m.mupix.seeds[2].covered = false; m.mupix.seeds[2].l1 = 1; m.mupix.seeds[2].l2 = 0;
  });
  const page = await boot({ "sma::frame": () => envelope("smaf", f) });
  const panels = byId(page, "dqm-smaev-seeds").byClass("dqm-sma-seed");
  assert.ok(panels[0].textContent.includes("MuPix n/a (outside the pixel data)"));
  assert.ok(panels[1].textContent.includes("no MuPix in time"));
  assert.ok(panels[2].byClass("dqm-sma-badge").some((b) => b.textContent === "L1 ✓"));
});

test("hovering a MuPix tick shows the pixel hit; a click tags it and Copy all finds its word", async () => {
  const page = await boot(withRaw());
  const canvas = byId(page, "dqm-smaev-seeds").byTag("canvas")[0];
  const ctx = canvas.getContext("2d");
  const [pa, pb] = SEEDED.meta.mupix.seeds[0].pix;
  const lanes = pixLanes(ctx);
  const seed = SEEDED.meta.seeds[0];
  const X = (t) => 84 + (t + 200) / 3200 * (888 - 84);
  // A pixel hit alone on its lane within 10 px.
  let j = -1, ev = null;
  for (let q = pa; q < pb && j < 0; q++) {
    const pl = SEEDED.pixels.flags[q] & 3;
    const x = X(SEEDED.meta.mupix.t0_ns - SEEDED.meta.t0_ns + SEEDED.pixels.t[q] - seed.t_rel);
    const alone = [...Array(pb - pa).keys()].map((r) => r + pa).every((r) => r === q ||
      (SEEDED.pixels.flags[r] & 3) !== pl ||
      Math.abs(X(SEEDED.meta.mupix.t0_ns - SEEDED.meta.t0_ns + SEEDED.pixels.t[r] - seed.t_rel) - x) > 10);
    if (alone) { j = q; ev = { clientX: x, clientY: 6 + lanes[pl] * 20 + 10 }; }
  }
  assert.ok(j >= 0, "an isolated pixel hit in seed 0");
  const line = expectPixelLine(SEEDED, j);
  assert.match(line, /^MuPix L[12] chip \d+ · col \d+ · row \d+ · ToT \d+ \(~\d+ ns\) · t \d+ ns \(t_rel -?\d+ ns\) · word \d+ · 0x[0-9a-f]{16}$/);
  const hover = byId(page, "dqm-smaev-seeds").byClass("dqm-sma-seed")[0].byClass("dqm-smaev-hover")[0];
  canvas.dispatch("mousemove", ev);
  assert.strictEqual(hover.textContent, `${line} · ${SEEDED.meta.tag}`);
  canvas.dispatch("click", ev);
  const rows = taggedRows(page);
  assert.strictEqual(rows.length, 1);
  assert.strictEqual(rowLine(rows[0]), line);
  assert.strictEqual(lastPaint(ctx).count("strokeRect", "#000"), 1, "the tagged tick is boxed");
  // With an SMA hit of the same frame: Δt from the first, and the word range spans both.
  const h = clickableSeedHits(page, 0)[0];
  h.canvas.dispatch("click", h.ev);
  const btn = byId(page, "dqm-smaev-tagged-copy");
  btn.onclick();
  await drain();
  const w = [SEEDED.pixels.wordIndex[j], SEEDED.wordIndex[h.i]];
  const tRelPix = SEEDED.meta.mupix.t0_ns - SEEDED.meta.t0_ns + SEEDED.pixels.t[j];
  const d = SEEDED.t[h.i] - tRelPix;
  assert.deepStrictEqual(page.copied, [[
    SEEDED.meta.tag,
    `  #1 · ${line}`,
    `  #2 · ${expectLine(SEEDED, h.i, SEEDED.meta.labels)} · Δt ${d > 0 ? "+" : ""}${d} ns`,
    `  mdqm-sma-file --serial ${SEEDED.meta.serial} --run 682 --dir <raw dir> --words ${Math.min(...w)}:${Math.max(...w)}`,
  ].join("\n")]);
  // Untagged by a second click on it.
  canvas.dispatch("click", ev);
  assert.strictEqual(taggedRows(page).length, 1);
});

test("the MuPix selector: any by default, both asks at once, is AND-ed with the boxes and persists", async () => {
  const ss = session();
  const page = await boot(seededBy((a) => (a.mupix === "both" ? "seeded_mupix_both" : "seeded")),
                          undefined, undefined, { session: ss });
  assert.strictEqual(frameCalls(page)[0].mupix, "any", "the default is sent explicitly");
  const sel = byId(page, "dqm-smaev-mupixsel");
  assert.deepStrictEqual(sel.byTag("option").map((o) => o.textContent),
                         ["any", "L1+L2 in time", "L1 or L2 in time", "none in time"]);
  let mark = page.an.calls.length;
  sel.value = "both";
  sel.onchange.call(sel);
  await settle(page);
  let asks = frameCalls(page, mark);
  assert.ok(asks.length && asks.every((a) => a.mupix === "both" && a.seed === "s1"), JSON.stringify(asks));
  const head = byId(page, "dqm-smaev-seededhead").textContent;
  assert.ok(head.includes("seed: S1 (ch 1) and L1+L2 in time"), head);
  mark = page.an.calls.length;
  const box = byId(page, "dqm-smaev-filter-incomplete");
  box.checked = true;
  box.onchange.call(box);
  await settle(page);
  asks = frameCalls(page, mark);
  assert.deepStrictEqual([asks[asks.length - 1].filters, asks[asks.length - 1].mupix], [["incomplete"], "both"]);
  assert.deepStrictEqual(JSON.parse(ss._d["dqm-sma-events-seed"]),
                         { seed: "s1", filters: ["incomplete"], mupix: "both", pattern: {} });
  const again = await boot(undefined, undefined, undefined, { session: ss });
  assert.strictEqual(frameCalls(again)[0].mupix, "both");
  assert.strictEqual(byId(again, "dqm-smaev-mupixsel").value, "both");
  const bad = await boot(undefined, undefined, undefined,
                         { session: session({ "dqm-sma-events-seed": '{"seed":"s1","mupix":"L1"}' }) });
  assert.strictEqual(frameCalls(bad)[0].mupix, "any", "garbage falls back to any");
});

test("no match with the MuPix selector keeps its banner", async () => {
  const nm = withMeta(FX.seeded_nomatch, (m) => { m.select.mupix = "none"; m.select.mupix_label = "no MuPix hit in time"; });
  const page = await boot({ "sma::frame": () => envelope("smaf", nm) });
  assert.ok(/^No match/.test(banner(page).textContent), banner(page).textContent);
  assert.ok(byId(page, "dqm-smaev-seededhead").textContent.includes("and no MuPix hit in time"));
});

/** The raster frame without its pixel block, as the analyzer sends it with "pixels": false. */
function rasterWithoutPixels() {
  const raw = Buffer.from(FX.raster, "hex");
  const out = Buffer.from(raw.subarray(0, RASTER.offsets.hitFlags + RASTER.nHits));
  out[1] &= ~SMAF.FLAGS.PIXELS;
  return withMeta(out.toString("hex"), (m) => { delete m.mupix; });
}

test("the raster's hide MuPix asks for no pixels, drops the rows and persists", async () => {
  const bare = rasterWithoutPixels();
  const page = await boot({
    "sma::frame": (args) => envelope("smaf", JSON.parse(args).pixels === false ? bare : Buffer.from(FX.raster, "hex")),
  }, { tab: "raster" });
  assert.ok(frameCalls(page).every((a) => a.pixels === undefined), "shown by default: nothing extra asked");
  let texts = lastPaint(byId(page, "dqm-smaev-raster").getContext("2d")).texts();
  assert.ok(texts.includes("MuPix L1"));
  const box = byId(page, "dqm-smaev-hidepix");
  box.checked = true;
  box.onchange.call(box);
  const mark = page.an.calls.length;
  await settle(page);
  const asks = frameCalls(page, mark);
  assert.ok(asks.length && asks.every((a) => a.pixels === false && a.drop[0] === 7), JSON.stringify(asks));
  texts = lastPaint(byId(page, "dqm-smaev-raster").getContext("2d")).texts();
  assert.ok(!texts.includes("MuPix L1"), "no MuPix rows");
  assert.strictEqual(JSON.parse(globalThis.localStorage._d["dqm-sma-events-settings"]).hidePixels, true);
});

test("a pixel hit on the live raster: a click freezes, fetches the words once, and tags it", async () => {
  const page = await boot(withRaw(), { tab: "raster" });
  const canvas = byId(page, "dqm-smaev-raster");
  const f = RASTER, p = RASTER.pixels, mp = RASTER.meta.mupix;
  const off = f.meta.t0_ns - f.meta.frame_first_ns;
  const hi = Math.max(f.meta.span_ns / 1e6, (off + f.t[f.nHits - 1]) / 1e6,
                      (mp.t0_ns - f.meta.frame_first_ns + p.t[p.n - 1]) / 1e6);
  const X = (j) => 84 + ((mp.t0_ns - f.meta.frame_first_ns + p.t[j]) / 1e6) / hi * (790 - 84) + 0.75;
  let j = -1;
  for (let q = 0; q < p.n && j < 0; q++) {
    if ((p.flags[q] & 3) !== 1 || X(q) < 84) continue;
    let alone = true;
    for (let r = 0; r < p.n; r++) if (r !== q && (p.flags[r] & 3) === 1 && Math.abs(X(r) - X(q)) < 10) alone = false;
    if (alone) j = q;
  }
  assert.ok(j >= 0, "an isolated L1 hit on the raster");
  const ev = { clientX: X(j), clientY: 6 + 16 * 22 + 11 };      // the first row under the 16 channels
  canvas.dispatch("mousemove", ev);
  const hover = byId(page, "dqm-smaev-pane-raster").byClass("dqm-smaev-hover")[0];
  assert.strictEqual(hover.textContent, `${expectPixelLine(RASTER, j)} · ${RASTER.meta.tag}`);
  canvas.dispatch("mousedown", ev);
  page.windowEvent("mouseup", ev);
  assert.strictEqual(byId(page, "dqm-smaev-live").textContent, "FROZEN");
  await settle(page, 4);
  const rows = taggedRows(page);
  assert.strictEqual(rows.length, 1);
  assert.strictEqual(rowLine(rows[0]), expectPixelLine(RASTER_WORDS, j), "tagged with its word");
  const paint = lastPaint(canvas.getContext("2d"));
  assert.ok(paint.count("strokeRect", "#000") >= 1, "boxed on the frozen raster");
});

// -- the per-counter pattern selector and "incomplete pattern" with faulted counters --------

const PATTERN = D("seeded_pattern");
const pickPattern = (a) => (a.pattern && Object.keys(a.pattern).length ? "seeded_pattern" : "seeded");

test("the pattern fixture carries what the page is tested on", () => {
  assert.deepStrictEqual(PATTERN.meta.select.pattern, { 1: "present", 3: "absent" });
  assert.deepStrictEqual(PATTERN.meta.incomplete.ignored.map((f) => f.counter), [5]);
  assert.ok(PATTERN.meta.seeds.length && PATTERN.meta.seeds.every((s) => !(s.pattern & 4) && (s.pattern & 1)));
  assert.deepStrictEqual(FX.summary_1008.timestamp_faults.counters.map((f) => f.counter), [5]);
});

test("the pattern row: any by default, tri-states ask at once, persist and reload", async () => {
  const ss = session();
  const page = await boot(seededBy(pickPattern), undefined, undefined, { session: ss });
  assert.ok(frameCalls(page).every((a) => JSON.stringify(a.pattern) === "{}"), "the default is sent: {}");
  const labels = [1, 2, 3, 4, 5].map((k) => byId(page, `dqm-smaev-pattext-${k}`).textContent);
  assert.deepStrictEqual(labels, ["S1", "S2", "S3", "S4", "S5"]);
  const s1 = byId(page, "dqm-smaev-pat-1");
  assert.deepStrictEqual(s1.byTag("option").map((o) => o.getAttribute("value")), ["any", "present", "absent"]);
  assert.strictEqual(s1.value, "any");
  let mark = page.an.calls.length;
  s1.value = "present";
  s1.onchange.call(s1);
  const s3 = byId(page, "dqm-smaev-pat-3");
  s3.value = "absent";
  s3.onchange.call(s3);
  await settle(page);
  let asks = frameCalls(page, mark);
  assert.deepStrictEqual(asks[asks.length - 1].pattern, { 1: "present", 3: "absent" });
  assert.strictEqual(asks[asks.length - 1].seed, "s1");
  assert.deepStrictEqual(JSON.parse(ss._d["dqm-sma-events-seed"]),
                         { seed: "s1", filters: [], mupix: "any", pattern: { 1: "present", 3: "absent" } });
  // Back to any: the key goes.
  mark = page.an.calls.length;
  s1.value = "any";
  s1.onchange.call(s1);
  await settle(page);
  asks = frameCalls(page, mark);
  assert.deepStrictEqual(asks[asks.length - 1].pattern, { 3: "absent" });
  // A reload keeps it; a stored garbage pattern is cleaned.
  const again = await boot(seededBy(pickPattern), undefined, undefined, { session: ss });
  assert.deepStrictEqual(frameCalls(again)[0].pattern, { 3: "absent" });
  assert.strictEqual(byId(again, "dqm-smaev-pat-3").value, "absent");
  assert.strictEqual(byId(again, "dqm-smaev-pat-1").value, "any");
  const bad = await boot(undefined, undefined, undefined, { session: session({
    "dqm-sma-events-seed": '{"seed":"ch2","pattern":{"1":"absent","3":"maybe","9":"present","x":1}}' }) });
  assert.deepStrictEqual(frameCalls(bad)[0].pattern, { 1: "absent" });
  assert.strictEqual(frameCalls(bad)[0].seed, "ch2");
  const arr = await boot(undefined, undefined, undefined, { session: session({
    "dqm-sma-events-seed": '{"seed":"s1","pattern":["present"]}' }) });
  assert.deepStrictEqual(frameCalls(arr)[0].pattern, {});
});

test("the header names the pattern and the counters incomplete ignores; badges agree", async () => {
  const ss = session({ "dqm-sma-events-seed": JSON.stringify(
    { seed: "s1", filters: ["incomplete"], mupix: "any", pattern: { 1: "present", 3: "absent" } }) });
  const page = await boot(Object.assign(seededBy(pickPattern), { "sma::summary": () => json(FX.summary_1008) }),
                          undefined, undefined, { session: ss });
  const head = byId(page, "dqm-smaev-seededhead").textContent;
  assert.ok(head.includes("seed: S1 (ch 1) with incomplete pattern (ignoring S5: timestamp fault 93 %) " +
                          "and S1 present, S3 absent"), head);
  const panels = byId(page, "dqm-smaev-seeds").byClass("dqm-sma-seed");
  assert.strictEqual(panels.length, PATTERN.meta.seeds.length);
  PATTERN.meta.seeds.forEach(function (s, k) {
    const t = panels[k].textContent;
    if (s.odd.indexOf("incomplete") >= 0) assert.ok(t.includes("incomplete pattern (S5 ignored)"), t);
    const boxes = panels[k].byClass("dqm-sma-pbox");
    assert.ok(boxes[4].classList.contains("ignored"), "S5 drawn as ignored");
    assert.ok(!boxes[2].classList.contains("ignored") && !boxes[2].classList.contains("lit"), "S3 absent");
    assert.ok(boxes[0].classList.contains("lit"), "S1 present");
  });
  // Nothing ignored (e.g. the shift is not judged): plain words, no dashed box.
  const plain = withMeta(FX.seeded_pattern, (m) => {
    m.incomplete = { judged: true, verdict: "insufficient", ignored: [],
                     note: "no counter ignored: shift check insufficient",
                     label: "incomplete pattern (no counter ignored: shift check insufficient)" };
  });
  const p2 = await boot({ "sma::frame": () => envelope("smaf", plain) });
  assert.ok(byId(p2, "dqm-smaev-seededhead").textContent.includes(
    "with incomplete pattern (no counter ignored: shift check insufficient)"));
  const pan = byId(p2, "dqm-smaev-seeds").byClass("dqm-sma-seed");
  assert.ok(pan.every((p) => !p.byClass("ignored").length));
  assert.ok(pan.every((p) => !p.textContent.includes("ignored)")));
});

test("incomplete that cannot be judged says so in the header", async () => {
  const cj = withMeta(FX.seeded_pattern, (m) => {
    m.incomplete.judged = false;
    m.incomplete.note = "cannot judge: S2, S3, S4, S5 have timestamp faults";
    m.incomplete.label = `incomplete pattern (${m.incomplete.note})`;
    m.seeds = [];
    m.search.no_match = { frames: 8, oldest_seq: 1, partial: false, span_s: 2.0,
                          text: `no S1 (ch 1) seeds with ${m.incomplete.label} in the last 8 good frames` };
  });
  const page = await boot({ "sma::frame": () => envelope("smaf", cj) });
  const head = byId(page, "dqm-smaev-seededhead").textContent;
  assert.ok(head.includes("incomplete pattern cannot be judged: S2, S3, S4, S5 have timestamp faults"), head);
  assert.ok(/^No match: no S1 \(ch 1\) seeds with incomplete pattern \(cannot judge/.test(banner(page).textContent));
});

test("a counter with a timestamp fault gets its warning beside its selector", async () => {
  const page = await boot({ "sma::summary": () => json(FX.summary_1008) });
  const w5 = byId(page, "dqm-smaev-patwarn-5");
  assert.notStrictEqual(w5.style.display, "none");
  assert.strictEqual(w5.textContent, "S5 has a timestamp fault (93 % fine/coarse mismatch): " +
                     "'absent' will match almost everything, 'present' almost nothing");
  for (const k of [1, 2, 3, 4]) {
    assert.strictEqual(byId(page, `dqm-smaev-patwarn-${k}`).style.display, "none", `S${k}`);
  }
  // Run 682's summary: S3 (13 %) and S5.
  const p2 = await boot();
  assert.deepStrictEqual([1, 2, 3, 4, 5].filter((k) => byId(p2, `dqm-smaev-patwarn-${k}`).style.display !== "none"),
                         [3, 5]);
  // The shift not judged: no warnings.
  const p3 = await boot({ "sma::summary": () => json(FX.summary_shift13) });
  assert.ok([1, 2, 3, 4, 5].every((k) => byId(p3, `dqm-smaev-patwarn-${k}`).style.display === "none"));
});

test("Single and a frozen re-ask send the pattern", async () => {
  const page = await boot(advancing(), undefined, undefined, { session: session({
    "dqm-sma-events-seed": '{"seed":"ch2","filters":["mismatch"],"mupix":"both","pattern":{"1":"absent"}}' }) });
  let mark = page.an.calls.length;
  byId(page, "dqm-smaev-single").onclick();
  await settle(page, 5);
  let asks = frameCalls(page, mark);
  assert.strictEqual(asks.length, 1);
  assert.deepStrictEqual([asks[0].seed, asks[0].filters, asks[0].mupix, asks[0].pattern],
                         ["ch2", ["mismatch"], "both", { 1: "absent" }]);
  mark = page.an.calls.length;
  const s4 = byId(page, "dqm-smaev-pat-4");
  s4.value = "present";
  s4.onchange.call(s4);
  await settle(page, 5);
  asks = frameCalls(page, mark);
  assert.strictEqual(asks.length, 1, JSON.stringify(asks));
  assert.ok(Number.isInteger(asks[0].seq), "the frozen frame, by its seq");
  assert.deepStrictEqual(asks[0].pattern, { 1: "absent", 4: "present" });
  assert.strictEqual(byId(page, "dqm-smaev-live").textContent, "FROZEN");
});

// --- the run-1015 cabling: no current channel, the counters from the roles ------------------

test("the 1015 fixtures carry what the page is tested on", () => {
  const roles = FX.summary_1015.channels.map((c) => c.role);
  assert.ok(roles.indexOf("current") < 0, "no current channel");
  assert.strictEqual(FX.summary_1015.channels[7].role, "counter", "S3 on ch 7");
  assert.strictEqual(FX.summary_1015.channels[7].label, "S3");
  assert.ok(roles.indexOf("delayed") < 0, "no delayed channel");
  assert.deepStrictEqual(FX.summary_3counters.efficiency.map((e) => e.ch), [2, 4]);
});

/**
 * An analyzer with the 1015 summary (or `summary`, with its own seeded frame
 * `seeded`); the frames carry the same roles block as the summary.
 */
function cabled1015(summary, seeded) {
  return {
    "sma::summary": () => json(summary || FX.summary_1015),
    "sma::frame": (args) => {
      const a = JSON.parse(args || "{}");
      return envelope("smaf", Buffer.from(a.view === "raster" ? FX.raster_1015
        : (seeded || FX.seeded_1015), "hex"));
    },
  };
}

test("no current channel: the raster drops nothing and the switch is hidden", async () => {
  const page = await boot(cabled1015(), { tab: "raster", hideCurrent: true });
  const asks = frameCalls(page);
  assert.ok(asks.length > 0);
  for (const a of asks) assert.deepStrictEqual(a.drop, [], "nothing to hide");
  const lab = byId(page, "dqm-smaev-hidecurlab");
  assert.strictEqual(lab.style.display, "none", "no 'hide the current channel' without one");
  // The old-layout summary has one: the switch is back.
  const old = await boot(undefined, { tab: "raster" });
  assert.strictEqual(byId(old, "dqm-smaev-hidecurlab").style.display, "");
});

test("no current channel: the seeded lanes have none, S3 is ch 7", async () => {
  const page = await boot(cabled1015());
  const ctx = byId(page, "dqm-smaev-seeds").byTag("canvas")[0].getContext("2d");
  const labels = ctx.texts();
  assert.ok(labels.includes("S3 (7)"), labels.join(","));
  assert.ok(!labels.some((l) => /^current/.test(l)), labels.join(","));
  const pat = [1, 2, 3, 4, 5].map((k) => byId(page, `dqm-smaev-pattext-${k}`).textContent);
  assert.deepStrictEqual(pat, ["S1", "S2", "S3", "S4", "S5"]);
  assert.ok(byId(page, "dqm-smaev-pat-6") === null, "no dqm-smaev-pat-6");
});

test("the pattern selector has one box per counter of the roles", async () => {
  const ss = session({ "dqm-sma-events-seed": JSON.stringify(
    { seed: "s1", filters: [], mupix: "any", pattern: { 2: "present", 4: "absent" } }) });
  const page = await boot(cabled1015(FX.summary_3counters, FX.seeded_3counters), undefined, undefined,
                          { session: ss });
  const row = byId(page, "dqm-smaev-patternrow");
  assert.strictEqual(row.byClass("dqm-smaev-pat").length, 3, "S1..S3");
  assert.deepStrictEqual([1, 2, 3].map((k) => byId(page, `dqm-smaev-pattext-${k}`).textContent),
                         ["S1", "S2", "S3"]);
  assert.ok(byId(page, "dqm-smaev-pat-4") === null, "no dqm-smaev-pat-4");
  // A stored choice for a counter that no longer exists is not sent.
  for (const a of frameCalls(page)) assert.deepStrictEqual(a.pattern, { 2: "present" });
  assert.strictEqual(byId(page, "dqm-smaev-pat-2").value, "present");
});

test("no summary and no frame: no counters are guessed", async () => {
  const page = await boot({
    "sma::summary": () => { throw new Error("analyzer down"); },
    "sma::frame": () => { throw new Error("analyzer down"); },
  });
  assert.strictEqual(byId(page, "dqm-smaev-patternrow").byClass("dqm-smaev-pat").length, 0);
  assert.ok(byId(page, "dqm-smaev-pat-1") === null, "no dqm-smaev-pat-1");
  assert.strictEqual(byId(page, "dqm-smaev-hidecurlab").style.display, "none");
  // Then a frame arrives: its labels are the roles.
  const later = await boot({ "sma::summary": () => { throw new Error("no summary"); } });
  assert.deepStrictEqual([1, 2, 3, 4, 5].map((k) => byId(later, `dqm-smaev-pattext-${k}`).textContent),
                         ["S1", "S2", "S3", "S4", "S5"]);
});

// --- TOT + NIM: NIM lanes, class styling, pair ticks (smaf v3) --------------------------------

const SEEDED_NIM = SMAF.decode(new Uint8Array(Buffer.from(FX.seeded_nim, "hex")));
const RASTER_NIM = SMAF.decode(new Uint8Array(Buffer.from(FX.raster_nim, "hex")));
const P = SMAF.PAIR;
const clone = (o) => JSON.parse(JSON.stringify(o));

/** An analyzer cabled as since run 1015, with NIM copies: frames are smaf v3. */
function nimEvents(over = {}, seededHex) {
  return Object.assign({
    "sma::summary": () => json(FX.summary_nim),
    "sma::frame": (args) => {
      const a = JSON.parse(args || "{}");
      return envelope("smaf", Buffer.from(a.view === "raster" ? FX.raster_nim
        : (seededHex || FX.seeded_nim), "hex"));
    },
  }, over);
}

/** The page's lighten(), from its spec: each channel moved f of the way to 255. */
function lighter(hex, f) {
  const v = parseInt(hex.slice(1), 16);
  return `#${[16, 8, 0].map((sh) => Math.round(((v >> sh) & 255) + (255 - ((v >> sh) & 255)) * f)
    .toString(16).padStart(2, "0")).join("")}`;
}

/** The page's darken(), from its spec: each channel moved f of the way to 0. */
function darker(hex, f) {
  const v = parseInt(hex.slice(1), 16);
  return `#${[16, 8, 0].map((sh) => Math.round(((v >> sh) & 255) * (1 - f))
    .toString(16).padStart(2, "0")).join("")}`;
}
/** WCAG contrast of two #rrggbb colours. */
function contrast(a, b) {
  const L = (h) => {
    const c = [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255)
      .map((x) => (x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4));
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
  };
  const [hi, lo] = [L(a), L(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}
const COUNTER_COLOURS = ["#1f77b4", "#2ca02c", "#17becf", "#9467bd", "#8c564b", "#bcbd22"];
const NIM_CH = [3, 9, 10, 11, 12];

/** Seed k's hits [a, b) on screen in the full window, by class. */
function seedClasses(f, k) {
  const seed = f.meta.seeds[k];
  const [a, b] = seed.hits;
  const out = { held: [], hollow: [], echo: [], nimSolid: {}, nimHollow: {}, ticks: 0 };
  for (let i = a; i < b; i++) {
    const t = f.t[i] - seed.t_rel;
    if (t + f.tot[i] < -200 || t > 3000) continue;
    const c = f.cls[i] & P.CLASS_MASK;
    if (c === P.NIM_ONLY) out[f.cls[i] & P.LAG_HELD ? "held" : "hollow"].push(i);
    const k = NIM_CH.indexOf(f.ch[i]);
    if (k >= 0 && c !== P.NIM_ONLY) out.nimSolid[k] = (out.nimSolid[k] || 0) + 1;
    if (k >= 0 && c === P.NIM_ONLY && !(f.cls[i] & P.LAG_HELD)) out.nimHollow[k] = (out.nimHollow[k] || 0) + 1;
    if (c === P.ECHO) out.echo.push(i);
    const j = f.pair[i];
    if (j >= a && j < b && !(f.cls[i] & P.NIM_SIDE) && t >= -200 && f.t[j] - seed.t_rel <= 3000) out.ticks++;
  }
  return out;
}

/** Where to point at hit i of seed k on its canvas (its bar's start, mid-lane). */
function pointAt(page, f, k, i) {
  const canvas = byId(page, "dqm-smaev-seeds").byTag("canvas")[k];
  const lane = laneMap(canvas.getContext("2d"))[f.ch[i]];
  const x = 84 + (f.t[i] - f.meta.seeds[k].t_rel + 200) / 3200 * (888 - 84);
  return { canvas, ev: { clientX: x + 1, clientY: 6 + lane * 20 + 10 } };
}

test("the NIM fixture frame is what the tests assume", () => {
  assert.strictEqual(SEEDED_NIM.version, 3);
  assert.ok(SEEDED_NIM.pairing && SEEDED_NIM.words);
  assert.deepStrictEqual(SEEDED_NIM.meta.roles.nim, [3, 9, 10, 11, 12]);
  assert.strictEqual(SEEDED_NIM.meta.nim_merge, true);
  assert.strictEqual(RASTER_NIM.version, 3);
  const all = SEEDED_NIM.meta.seeds.map((_, k) => seedClasses(SEEDED_NIM, k));
  assert.ok(all.some((c) => c.held.length) && all.some((c) => c.hollow.length) &&
            all.some((c) => c.echo.length) && all.some((c) => c.ticks), JSON.stringify(all));
});

test("each NIM copy has its own lane right under its counter, in its colour but lighter", async () => {
  const page = await boot(nimEvents());
  const canvas = byId(page, "dqm-smaev-seeds").byTag("canvas")[0];
  const lanes = laneMap(canvas.getContext("2d"));
  const order = Object.entries(lanes).sort((x, y) => x[1] - y[1]).map(([ch]) => Number(ch));
  assert.deepStrictEqual(order.slice(0, 11), [1, 3, 2, 9, 7, 10, 4, 11, 5, 12, 6],
                         "S1 S1L S2 S2L S3 S3L S4 S4L S5 S5L RF");
  const labels = canvas.getContext("2d").texts();
  for (const l of ["S1L (3)", "S3L (10)", "S4L (11)"]) assert.ok(labels.includes(l), labels.join(","));
  const paint = lastPaint(canvas.getContext("2d"));
  const nimFill = lighter("#1f77b4", 0.35);
  assert.ok(paint.count("fillRect", nimFill) > 0, `S1L bars in ${nimFill}`);
  assert.ok(paint.count("strokeRect", darker("#1f77b4", 0.3)) > 0, "outlined in the darker counter colour");
  assert.ok(paint.count("fillRect", "#1f77b4") > 0, "S1 bars in the counter colour");
  // The outline carries the contrast on the lane stripe: >= 3:1 for every counter colour.
  for (const c of COUNTER_COLOURS) {
    assert.ok(contrast(darker(c, 0.3), "#f4f4f4") >= 3, `${c}: ${contrast(darker(c, 0.3), "#f4f4f4")}`);
  }
  assert.ok(contrast("#6f6f6f", "#f4f4f4") >= 3, "the held-back outline");
  assert.ok(!byId(page, "dqm-smaev-nimlegend").style.display, "the NIM legend is shown");
});

test("hit styles by class: NIM-only hollow, lag-held grey, echo cross-hatched, pair ticks", async () => {
  const page = await boot(nimEvents());
  const canvases = byId(page, "dqm-smaev-seeds").byTag("canvas");
  let held = 0, hollow = 0, echo = 0, ticks = 0;
  SEEDED_NIM.meta.seeds.forEach(function (seed, k) {
    const want = seedClasses(SEEDED_NIM, k);
    const paint = lastPaint(canvases[k].getContext("2d"));
    assert.strictEqual(paint.count("strokeRect", "#6f6f6f"), want.held.length, `seed ${k}: grey outlines`);
    assert.strictEqual(paint.count("fillRect", "#d4d4d4"), want.held.length, `seed ${k}: grey fill`);
    NIM_CH.forEach(function (_ch, j) {
      // Solid NIM bars: light fill + dark outline; hollow ones: the outline only
      // (a 10 ns word is narrower than the hollow minimum, so it is not filled).
      const solid = want.nimSolid[j] || 0, hol = want.nimHollow[j] || 0;
      assert.strictEqual(paint.count("fillRect", lighter(COUNTER_COLOURS[j], 0.35)), solid, `seed ${k} NIM ${j} fill`);
      assert.strictEqual(paint.count("strokeRect", darker(COUNTER_COLOURS[j], 0.3)), solid + hol,
                         `seed ${k} NIM ${j} outlines`);
    });
    assert.strictEqual(paint.count("stroke", "rgba(0, 0, 0, 0.55)"), want.echo.length, `seed ${k}: echo hatch`);
    assert.strictEqual(paint.count("stroke", "#3a3a3a"), want.ticks ? 1 : 0, `seed ${k}: one tick path`);
    if (want.ticks) {
      // One moveTo/lineTo per pair, after the tick colour is set.
      const ops = paint.ops;
      const at = ops.findIndex((o) => o[0] === "stroke" && o[2] === "#3a3a3a");
      let begin = at;
      while (begin > 0 && ops[begin][0] !== "beginPath") begin--;
      assert.strictEqual(ops.slice(begin, at).filter((o) => o[0] === "lineTo").length, want.ticks);
    }
    held += want.held.length; hollow += want.hollow.length; echo += want.echo.length; ticks += want.ticks;
    assert.strictEqual(paint.count("strokeRect", "#d00"),
      (() => { let n = 0; const [a, b] = seed.hits;
        for (let i = a; i < b; i++) if (SEEDED_NIM.hitFlags[i] & SMAF.HIT.MISMATCH) n++; return n; })(),
      "the red mismatch outline only for mismatches, never for an echo");
  });
  assert.ok(held && hollow && echo && ticks);
  const head = byId(page, "dqm-smaev-seeds").textContent;
  assert.ok(/NIM-only held back \(lag fault\)/.test(head) && /NIM-only \(merged\)/.test(head), head);
});

test("a frame without pairing (v2) draws no ticks and no NIM styles", async () => {
  const page = await boot();
  for (const c of byId(page, "dqm-smaev-seeds").byTag("canvas")) {
    const paint = lastPaint(c.getContext("2d"));
    assert.strictEqual(paint.count("stroke", "#3a3a3a"), 0);
    assert.strictEqual(paint.count("strokeRect", "#6f6f6f"), 0);
  }
  assert.strictEqual(byId(page, "dqm-smaev-nimlegend").style.display, "none");
});

test("hovering says a hit's class, its partner and NIM - TOT, and its flags", async () => {
  const page = await boot(nimEvents());
  const f = SEEDED_NIM;
  const k = f.meta.seeds.findIndex((_, s) => seedClasses(f, s).held.length);
  const i = seedClasses(f, k).held[0];
  const panel = byId(page, "dqm-smaev-seeds").byClass("dqm-sma-seed")[k];
  const hover = panel.byClass("dqm-smaev-hover")[0];
  let { canvas, ev } = pointAt(page, f, k, i);
  canvas.dispatch("mousemove", ev);
  assert.match(hover.textContent, /^ch 11 \(S4L\) · ToT \d+ · .* · NIM word: NIM only \(held back from the merge: lag fault\) \[(.*, )?lag-held\] · word \d+/);
  // A paired TOT word: its NIM copy and the time between them.
  const [a, b] = f.meta.seeds[k].hits;
  let tot = -1;
  for (let x = a; x < b && tot < 0; x++) {
    if ((f.cls[x] & P.CLASS_MASK) === P.PAIRED && !(f.cls[x] & P.NIM_SIDE) && f.pair[x] >= 0 &&
        f.ch[x] !== 1) tot = x;
  }
  ({ canvas, ev } = pointAt(page, f, k, tot));
  canvas.dispatch("mousemove", ev);
  const j = f.pair[tot];
  const dt = f.t[j] - f.t[tot];
  const nimLab = f.meta.labels[f.ch[j]];
  assert.ok(hover.textContent.includes(
    `TOT word: paired with ${nimLab} (ch ${f.ch[j]}), NIM − TOT ${dt > 0 ? "+" : ""}${dt} ns`), hover.textContent);
});

test("roles come from the frame's roles block, even without a summary", async () => {
  const page = await boot(nimEvents({ "sma::summary": () => { throw new Error("no summary"); } }));
  const labels = byId(page, "dqm-smaev-seeds").byTag("canvas")[0].getContext("2d").texts();
  assert.ok(labels.includes("S1L (3)") && labels.includes("S3 (7)"), labels.join(","));
  // The frame's block wins over the summary's when they disagree (a frame analysed before an edit).
  const s = clone(FX.summary_nim);
  s.roles.nim = [-1, -1, -1, -1, -1];
  const p2 = await boot(nimEvents({ "sma::summary": () => json(s) }));
  const l2 = byId(p2, "dqm-smaev-seeds").byTag("canvas")[0].getContext("2d").texts();
  assert.ok(l2.includes("S4L (11)"), l2.join(","));
  // No block in the frame (an older analyzer or no NIM copies): the summary's.
  const noRoles = withMeta(FX.seeded_nim, (m) => { delete m.roles; delete m.nim_merge; });
  const p3 = await boot(nimEvents({ "sma::summary": () => json(s) }, noRoles.toString("hex")));
  const l3 = byId(p3, "dqm-smaev-seeds").byTag("canvas")[0].getContext("2d").texts();
  assert.ok(!l3.some((l) => /^S\dL/.test(l)), "the summary says no NIM copies");
});

test("the pattern selector and the seed's pattern say merged when the merge is on", async () => {
  const page = await boot(nimEvents());
  const head = byId(page, "dqm-smaev-patternhead");
  assert.strictEqual(head.textContent, "and counters (merged TOT + NIM):");
  assert.ok(/Merged: a counter's hits are its TOT words plus its NIM-only hits/.test(head.getAttribute("title")));
  const pat = byId(page, "dqm-smaev-seeds").byClass("dqm-sma-pattern")[0];
  assert.ok(/merged counters/.test(pat.getAttribute("title")));
  assert.ok(/merged into its counter/.test(byId(page, "dqm-smaev-nimlegend").textContent));
});

test("merge off (the default): TOT-only pattern, NIM-only shown but not merged, nothing grey", async () => {
  const off = SMAF.decode(new Uint8Array(Buffer.from(FX.seeded_nim_off, "hex")));
  assert.strictEqual(off.meta.nim_merge, false);
  const page = await boot(nimEvents({ "sma::summary": () => json(FX.summary_nim_off) }, FX.seeded_nim_off));
  const head = byId(page, "dqm-smaev-patternhead");
  assert.strictEqual(head.textContent, "and counters (TOT only, merge off):");
  assert.ok(/TOT only: the NIM merge is off/.test(head.getAttribute("title")));
  assert.strictEqual(byId(page, "dqm-smaev-seeds").byClass("dqm-sma-pattern")[0].getAttribute("title"),
                     "coincidence pattern of the TOT words only (NIM merge off)");
  const legend = byId(page, "dqm-smaev-nimlegend").textContent;
  assert.ok(/shown, not merged: NIM merge off/.test(legend) && !/held back/.test(legend), legend);
  const text = byId(page, "dqm-smaev-seeds").textContent;
  assert.ok(!/held back/.test(text), "no held-back badge with nothing merged");
  assert.ok(/NIM-only \(not merged: merge off\)/.test(text), text);
  for (const c of byId(page, "dqm-smaev-seeds").byTag("canvas")) {
    assert.strictEqual(lastPaint(c.getContext("2d")).count("fillRect", "#d4d4d4"), 0, "no grey bars");
  }
  // A NIM-only hit's line says it is not merged.
  const k = off.meta.seeds.findIndex((_, j) => seedClasses(off, j).hollow.length);
  const i = seedClasses(off, k).hollow[0];
  const { canvas, ev } = pointAt(page, off, k, i);
  canvas.dispatch("mousemove", ev);
  const hover = byId(page, "dqm-smaev-seeds").byClass("dqm-sma-seed")[k].byClass("dqm-smaev-hover")[0];
  assert.ok(/NIM word: NIM only \(merge off\)/.test(hover.textContent), hover.textContent);
});

test("the raster labels the NIM rows and marks the NIM-only words; colour stays ToT", async () => {
  const page = await boot(nimEvents(), { tab: "raster" });
  const ctx = byId(page, "dqm-smaev-raster").getContext("2d");
  const paint = lastPaint(ctx);
  const texts = paint.texts();
  for (const l of ["S1L (3)", "S4L (11)", "S5L (12)"]) assert.ok(texts.includes(l), texts.join(","));
  let hollow = 0, held = 0;
  for (let i = 0; i < RASTER_NIM.nHits; i++) {
    if ((RASTER_NIM.cls[i] & P.CLASS_MASK) !== P.NIM_ONLY) continue;
    if (RASTER_NIM.cls[i] & P.LAG_HELD) held++; else hollow++;
  }
  assert.ok(hollow > 0 && held > 0);
  assert.strictEqual(paint.count("fill", "#3a3a3a"), 1, "one path of NIM-only marks");
  assert.strictEqual(paint.count("fill", "#6f6f6f"), 1, "one path of held-back marks");
  const rects = paint.ops.filter((o) => o[0] === "rect").length;
  assert.ok(rects >= RASTER_NIM.nHits + hollow + held, "every hit plus each mark");
  assert.ok(!byId(page, "dqm-smaev-rasternim").style.display, "the raster's NIM note is shown");
  // The live raster carries the classes only: its partners come with the words (Freeze).
  assert.strictEqual(RASTER_NIM.pair, null);
  assert.ok(/merged; grey: held back/.test(byId(page, "dqm-smaev-rasternim").textContent));
});

test("with no roles block anywhere, the NIM lanes come from the channel rows' pair_of", async () => {
  const s = clone(FX.summary_nim);
  delete s.roles;
  const bare = withMeta(FX.seeded_nim, (m) => { delete m.roles; });
  const page = await boot(nimEvents({ "sma::summary": () => json(s) }, bare.toString("hex")));
  const lanes = laneMap(byId(page, "dqm-smaev-seeds").byTag("canvas")[0].getContext("2d"));
  const order = Object.entries(lanes).sort((x, y) => x[1] - y[1]).map(([ch]) => Number(ch));
  assert.deepStrictEqual(order.slice(0, 10), [1, 3, 2, 9, 7, 10, 4, 11, 5, 12]);
});

// --- NIM lanes at t - NIM/offset ns (meta.nim_offsets_ns), "raw times" -------------------------

const OFFS = [25, 73, 11, 20, 19];
/** The seeded NIM frame with `offsets` as its NIM/offset ns (per counter, S1..S5). */
const withOffsets = (hex, offsets) => withMeta(hex, (m) => { m.nim_offsets_ns = offsets; });
/** Where the full seed window puts t (ns from the seed) on a 900 px canvas. */
const seedX = (t) => 84 + (t + 200) / 3200 * (888 - 84);

/** The x of every pair tick of a paint: [[xTOT, xNIM], ...]. */
function tickXs(paint) {
  const ops = paint.ops;
  const at = ops.findIndex((o) => o[0] === "stroke" && o[2] === "#3a3a3a");
  if (at < 0) return [];
  let begin = at;
  while (begin > 0 && ops[begin][0] !== "beginPath") begin--;
  const out = [];
  for (let i = begin; i < at; i++) {
    if (ops[i][0] === "moveTo" && ops[i + 1] && ops[i + 1][0] === "lineTo") out.push([ops[i][1][0], ops[i + 1][1][0]]);
  }
  return out;
}

test("the frame's NIM offsets: the fixtures carry them, a frame without NIM copies none", () => {
  assert.deepStrictEqual(SEEDED_NIM.meta.nim_offsets_ns, [0, 0, 0, 0, 0]);
  assert.strictEqual(SEEDED_NIM.meta.nim_offsets_ns.length, SEEDED_NIM.meta.roles.counters.length);
  assert.strictEqual(SEEDED.meta.nim_offsets_ns, undefined);
});

test("NIM lanes are drawn at t - NIM/offset ns: bars move, the pair ticks stand upright", async () => {
  // The fixture's NIM copies are 2 +- 1 ns after their TOT words: offsets of 2 align them.
  const hex = withOffsets(FX.seeded_nim, [2, 2, 2, 2, 2]).toString("hex");
  const page = await boot(nimEvents({}, hex));
  const canvases = byId(page, "dqm-smaev-seeds").byTag("canvas");
  const f = SEEDED_NIM;
  let n = 0;
  f.meta.seeds.forEach(function (seed, k) {
    const ticks = tickXs(lastPaint(canvases[k].getContext("2d")));
    n += ticks.length;
    for (const [xt, xn] of ticks) assert.ok(Math.abs(xn - xt) <= 1 / 3200 * 804 + 1e-9, `seed ${k}: ${xt} ${xn}`);
  });
  assert.ok(n > 0, "some pair ticks");
  // "raw times": the ticks slant by NIM - TOT again (2 ns on average).
  const box = byId(page, "dqm-smaev-rawtimes-seeded");
  box.checked = true;
  box.onchange.call(box);
  let sum = 0, m = 0;
  for (const c of byId(page, "dqm-smaev-seeds").byTag("canvas")) {
    for (const [xt, xn] of tickXs(lastPaint(c.getContext("2d")))) { sum += xn - xt; m++; }
  }
  assert.ok(m === n && Math.abs(sum / m - 2 / 3200 * 804) < 0.1, `mean slant ${sum / m} px`);
});

test("each NIM bar sits at its own counter's offset; TOT bars stay where they were", async () => {
  const page = await boot(nimEvents({}, withOffsets(FX.seeded_nim, OFFS).toString("hex")));
  const f = SEEDED_NIM;
  const seed = f.meta.seeds[0];
  const [a, b] = seed.hits;
  const paint = lastPaint(byId(page, "dqm-smaev-seeds").byTag("canvas")[0].getContext("2d"));
  const xsOf = (colour) => paint.ops.filter((o) => o[0] === "fillRect" && o[2] === colour)
    .map((o) => o[1][0].toFixed(6)).sort();
  let lanesWithBars = 0;
  NIM_CH.forEach(function (ch, k) {
    const want = [], tot = [];
    for (let i = a; i < b; i++) {
      const c = f.cls[i] & P.CLASS_MASK;
      if (f.ch[i] === ch && c !== P.NIM_ONLY) {
        const t = f.t[i] - OFFS[k] - seed.t_rel;
        if (!(t + f.tot[i] < -200 || t > 3000)) want.push(seedX(t).toFixed(6));
      }
      if (f.ch[i] === f.meta.roles.counters[k] && c !== P.ECHO) {
        const t = f.t[i] - seed.t_rel;
        if (!(t + f.tot[i] < -200 || t > 3000)) tot.push(seedX(t).toFixed(6));
      }
    }
    if (want.length) lanesWithBars++;     // S4L's words are all NIM-only (its lag fault)
    assert.deepStrictEqual(xsOf(lighter(COUNTER_COLOURS[k], 0.35)), want.sort(), `S${k + 1}L at t - ${OFFS[k]} ns`);
    const drawn = new Set(xsOf(COUNTER_COLOURS[k]));
    assert.ok(tot.every((x) => drawn.has(x)), `S${k + 1}'s TOT bars at their raw time`);
  });
  assert.ok(lanesWithBars >= 3, `${lanesWithBars} NIM lanes with solid bars`);
  const legend = byId(page, "dqm-smaev-nimlegend").textContent;
  assert.ok(legend.includes("at t − NIM/offset ns (S1L 25, S2L 73, S3L 11, S4L 20, S5L 19)"), legend);
});

test("hovering a NIM hit finds it where it is drawn and gives the raw and the aligned time", async () => {
  const page = await boot(nimEvents({}, withOffsets(FX.seeded_nim, OFFS).toString("hex")));
  const f = SEEDED_NIM;
  const t0 = f.meta.t0_ns;
  // A paired S2L word (offset 73 ns, 18 px) alone on its lane within 30 px either way.
  let k = -1, i = -1;
  for (let s = 0; s < f.meta.seeds.length && i < 0; s++) {
    const [a, b] = f.meta.seeds[s].hits;
    const on = [];
    for (let x = a; x < b; x++) if (f.ch[x] === 9) on.push(x);
    for (const x of on) {
      const c = f.cls[x] & P.CLASS_MASK, tr = f.t[x] - 73 - f.meta.seeds[s].t_rel;
      if (c !== P.PAIRED || f.pair[x] < 0 || tr < 0 || tr > 2800) continue;
      if (on.every((y) => y === x || Math.abs(f.t[y] - f.t[x]) > 120)) { k = s; i = x; break; }
    }
  }
  assert.ok(i >= 0, "an isolated paired S2L word");
  const canvas = byId(page, "dqm-smaev-seeds").byTag("canvas")[k];
  const lane = laneMap(canvas.getContext("2d"))[9];
  const hover = byId(page, "dqm-smaev-seeds").byClass("dqm-sma-seed")[k].byClass("dqm-smaev-hover")[0];
  const y = 6 + lane * 20 + 10;
  const tr = f.t[i];
  canvas.dispatch("mousemove", { clientX: seedX(tr - 73 - f.meta.seeds[k].t_rel) + 1, clientY: y });
  assert.ok(hover.textContent.startsWith(`ch 9 (S2L) · ToT ${f.tot[i]} · raw t ${t0 + tr} ns (t_rel ${tr} ns) · ` +
    `aligned t ${t0 + tr - 73} ns (t_rel ${tr - 73} ns) (NIM/offset 73 ns) · `), hover.textContent);
  const j = f.pair[i];
  const dt = f.t[i] - f.t[j];
  assert.ok(hover.textContent.includes(`NIM − TOT ${dt > 0 ? "+" : ""}${dt} ns (aligned ${dt - 73 > 0 ? "+" : ""}${dt - 73} ns)`),
    hover.textContent);
  // Its raw position (18 px to the right) is empty lane now.
  canvas.dispatch("mousemove", { clientX: seedX(tr - f.meta.seeds[k].t_rel) + 1, clientY: y });
  assert.ok(!/^ch 9 /.test(hover.textContent), hover.textContent);
  // "raw times": it is found at its raw position again, and the line still gives both.
  const box = byId(page, "dqm-smaev-rawtimes-seeded");
  box.checked = true;
  box.onchange.call(box);
  canvas.dispatch("mousemove", { clientX: seedX(tr - f.meta.seeds[k].t_rel) + 1, clientY: y });
  assert.ok(hover.textContent.includes(`raw t ${t0 + tr} ns`) && hover.textContent.includes("aligned t "), hover.textContent);
  // A TOT word's line is unchanged: one time.
  const tot = f.pair[i];
  canvas.dispatch("mousemove", { clientX: seedX(f.t[tot] - f.meta.seeds[k].t_rel) + 1,
                                 clientY: 6 + laneMap(canvas.getContext("2d"))[f.ch[tot]] * 20 + 10 });
  assert.ok(hover.textContent.includes(` · t ${t0 + f.t[tot]} ns (t_rel ${f.t[tot]} ns) · `) &&
            !hover.textContent.includes("aligned t"), hover.textContent);
});

test("raw times: off by default, remembered, shared by both tabs, hidden without offsets", async () => {
  const hex = withOffsets(FX.seeded_nim, OFFS).toString("hex");
  const page = await boot(nimEvents({}, hex));
  const lab = byId(page, "dqm-smaev-rawtimeslab-seeded");
  const box = byId(page, "dqm-smaev-rawtimes-seeded");
  assert.ok(!lab.style.display && box.checked === false, "shown, off");
  assert.ok(/at their raw SMA time/.test(lab.getAttribute("title")));
  box.checked = true;
  box.onchange.call(box);
  assert.strictEqual(JSON.parse(globalThis.localStorage._d["dqm-sma-events-settings"]).rawTimes, true);
  assert.ok(byId(page, "dqm-smaev-rawtimes-raster").checked === true, "the raster's box follows");
  assert.ok(/S\*k\*L lanes: the counter's NIM copy, lighter, outlined, at their raw time \(raw times\) · /
    .test(byId(page, "dqm-smaev-nimlegend").textContent));
  // Raw: the S2L bars at their raw time.
  const f = SEEDED_NIM, seed = f.meta.seeds[0];
  const paint = lastPaint(byId(page, "dqm-smaev-seeds").byTag("canvas")[0].getContext("2d"));
  const xs = new Set(paint.ops.filter((o) => o[0] === "fillRect" && o[2] === lighter(COUNTER_COLOURS[1], 0.35))
    .map((o) => o[1][0].toFixed(6)));
  let raw = 0;
  for (let i = seed.hits[0]; i < seed.hits[1]; i++) {
    if (f.ch[i] === 9 && xs.has(seedX(f.t[i] - seed.t_rel).toFixed(6))) raw++;
  }
  assert.ok(raw > 0 && raw === xs.size, `${raw} of ${xs.size} S2L bars raw`);
  // A reload remembers it.
  const p2 = await boot(nimEvents({}, hex), { rawTimes: true });
  assert.ok(byId(p2, "dqm-smaev-rawtimes-seeded").checked === true, "restored");
  // No offsets in the frame (no NIM copies, an older analyzer): hidden, drawn raw.
  const p3 = await boot();
  assert.strictEqual(byId(p3, "dqm-smaev-rawtimeslab-seeded").style.display, "none");
  const old = withMeta(FX.seeded_nim, (m) => { delete m.nim_offsets_ns; });
  const p4 = await boot(nimEvents({}, old.toString("hex")));
  assert.strictEqual(byId(p4, "dqm-smaev-rawtimeslab-seeded").style.display, "none");
  assert.ok(!/NIM\/offset/.test(byId(p4, "dqm-smaev-nimlegend").textContent));
});

test("the raster draws the NIM rows at t - NIM/offset ns too, and points at them there", async () => {
  // 1 ms on S2L: ~24 px on the whole-frame raster, enough to see and to point at.
  const offs = [0, 1000000, 0, 0, 0];
  const rhex = withOffsets(FX.raster_nim, offs).toString("hex");
  const page = await boot(nimEvents({
    "sma::frame": () => envelope("smaf", Buffer.from(rhex, "hex")),
  }), { tab: "raster" });
  const ctx = byId(page, "dqm-smaev-raster").getContext("2d");
  const rowXs = (paint, ch) => paint.ops.filter((o) => o[0] === "rect" && o[1][1] === 6 + ch * 22 + 4).map((o) => o[1][0]);
  const al = lastPaint(ctx);
  const s2l = rowXs(al, 9), s2 = rowXs(al, 2);
  const box = byId(page, "dqm-smaev-rawtimes-raster");
  assert.ok(!byId(page, "dqm-smaev-rawtimeslab-raster").style.display, "shown on the raster");
  assert.ok(/S\*k\*L rows: the NIM copies, at t − NIM\/offset ns \(S1L 0, S2L 1000000/.test(
    byId(page, "dqm-smaev-rasternim").textContent));
  box.checked = true;
  box.onchange.call(box);
  const raw = lastPaint(ctx);
  const r2l = rowXs(raw, 9), r2 = rowXs(raw, 2);
  assert.deepStrictEqual(r2, s2, "S2 (TOT) unchanged");
  const f = RASTER_NIM;
  const off = f.meta.t0_ns - f.meta.frame_first_ns;
  const hi = Math.max(f.meta.span_ns / 1e6, (off + f.t[f.nHits - 1]) / 1e6);
  const px = 1 / hi * (790 - 84);                  // px per ms
  // Every S2L mark 1 ms (px) to the left of its raw place, less the ones the axis start clips.
  const shifted = new Set(s2l.map((x) => (x + px).toFixed(4)));
  const back = r2l.filter((x) => shifted.has(x.toFixed(4)));
  assert.ok(back.length === s2l.length && r2l.length - s2l.length <= 30, `${back.length} ${s2l.length} ${r2l.length}`);
  // Pointing: the S2L hit found is one drawn under the mouse (its aligned time), on
  // a raster too dense to have a hit alone on its row.
  box.checked = false;
  box.onchange.call(box);
  const X = (t) => 84 + ((off + t) / 1e6) / hi * (790 - 84) + 0.75;
  const hover = byId(page, "dqm-smaev-pane-raster").byClass("dqm-smaev-hover")[0];
  let n = 0;
  for (let i = 0; i < f.nHits && n < 20; i++) {
    if (f.ch[i] !== 9 || f.t[i] + off < 2e6) continue;
    const x = X(f.t[i] - 1e6);
    byId(page, "dqm-smaev-raster").dispatch("mousemove", { clientX: x, clientY: 6 + 9 * 22 + 11 });
    const m = /^ch 9 \(S2L\) · .* · aligned t -?\d+ ns \(t_rel (-?\d+) ns\) \(NIM\/offset 1000000 ns\)/.exec(hover.textContent);
    assert.ok(m && Math.abs(X(Number(m[1])) - x) <= 4 + 1e-6, hover.textContent);
    n++;
  }
  assert.strictEqual(n, 20);
});
