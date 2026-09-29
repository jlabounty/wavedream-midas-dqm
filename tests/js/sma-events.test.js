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

const { runPage } = require("./domstub.js");

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
    for (const lane of ["S1 (1)", "S2 (2)", "S5 (5)", "RF (6)", "current (7)"]) {
      assert.ok(labels.includes(lane), `lane ${lane} missing: ${labels.join(",")}`);
    }
    assert.ok(ctx.count("fillRect") > 10, "hit bars were drawn");
  });
  assert.deepStrictEqual(globalThis.__alerts, []);
  assert.strictEqual(byId(page, "dqm-smaev-seq").textContent, `frame seq ${SEEDED.frameSeq}`);
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
    assert.deepStrictEqual(a.drop, [7], "the drop list comes from the summary's current role");
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
  // Every hit went into a rect, bucketed by ToT colour into a few fills.
  assert.strictEqual(ctx.count("rect"), RASTER.nHits, "one rect per hit");
  assert.ok(ctx.count("fill") < 40, "one fill per colour bucket, not per hit");
  assert.ok(byId(page, "dqm-smaev-pane-raster").byClass("dqm-sma-ramp").length === 1, "ToT legend");
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
  // Tick labels are the texts below the 16 rows (y > 6 + 16 * 22).
  const ticks = () => lastPaint(ctx).ops
    .filter((o) => o[0] === "fillText" && o[1][2] > 6 + 16 * 22 && o[1][2] < 6 + 16 * 22 + 12)
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
  assert.deepStrictEqual(asks[0], { view: "raster", drop: [7], max_hits: 60000, words: true });
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
  assert.strictEqual(byId(page, "dqm-smaev-popup"), null, "Close removes it");
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
  const want = `${SEEDED.meta.tag} · seed 2: S1 word ${seed.s1_word} · ` +
               `words ${seed.word_range[0]}–${seed.word_range[1]} (${b - a} hits in window) · ` +
               `hit words ${words.join(", ")}`;
  const panel = byId(page, "dqm-smaev-seeds").byClass("dqm-sma-seed")[1];
  assert.ok(panel.textContent.includes(`words ${seed.word_range[0]}–${seed.word_range[1]}`), "a badge");
  const btn = panel.byClass("dqm-smaev-copyseed")[0];
  btn.onclick();
  await drain();
  assert.deepStrictEqual(page.copied, [want]);
  assert.strictEqual(btn.textContent, "Copied ✓");
  // The button is not rebuilt by a redraw, so a click that spans one still lands.
  await settle(page);
  assert.strictEqual(byId(page, "dqm-smaev-seeds").byClass("dqm-sma-seed")[1].byClass("dqm-smaev-copyseed")[0], btn);
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
  assert.notStrictEqual(byId(page, "dqm-smaev-tagged").parent, byId(page, "dqm-smaev-pane-seeded"),
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
