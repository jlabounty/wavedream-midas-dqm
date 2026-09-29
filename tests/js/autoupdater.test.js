//
// BRPC.AutoUpdater: at most one polling chain, whatever happens around it.
//
// Every analyzer page (the WaveDream Waveforms and EventDisplay pages as well
// as the SMA ones) polls through this class, so a second chain is a bug at the
// beamtime, not only on the new pages: each chain is another request per
// interval through mhttpd, and they multiplied with every tab hide/show.
//

const test = require("node:test");
const assert = require("node:assert");
const path = require("node:path");

/** A document with a hidden flag, and a window whose timers we fire by hand. */
function world() {
  const listeners = {};
  const doc = {
    hidden: false,
    addEventListener(n, f) { (listeners[n] = listeners[n] || []).push(f); },
    removeEventListener(n, f) { listeners[n] = (listeners[n] || []).filter((g) => g !== f); },
    setHidden(h) { this.hidden = h; (listeners.visibilitychange || []).forEach((f) => f()); },
  };
  let next = 1;
  const timers = new Map();
  const win = {
    setTimeout(fn, ms) { const id = next++; timers.set(id, { fn, ms }); return id; },
    clearTimeout(id) { timers.delete(id); },
  };
  globalThis.document = doc;
  globalThis.window = win;
  return {
    doc, timers,
    fire() { const t = [...timers.values()]; timers.clear(); t.forEach((x) => x.fn()); },
  };
}

const BRPC = require(path.join(__dirname, "..", "..", "pages", "js", "dqm-brpc.js"));

/** An update whose requests stay in flight until released. */
function slowUpdate() {
  const s = { inFlight: 0, maxInFlight: 0, calls: 0, pending: [] };
  s.fn = () => {
    s.calls++;
    s.inFlight++;
    s.maxInFlight = Math.max(s.maxInFlight, s.inFlight);
    return new Promise((resolve, reject) => s.pending.push({ resolve, reject }));
  };
  s.release = async (fail) => {
    const p = s.pending.splice(0);
    p.forEach((x) => { s.inFlight--; if (fail) x.reject(new Error("did not answer")); else x.resolve(); });
    for (let i = 0; i < 5; i++) await Promise.resolve();
  };
  return s;
}

test("hide/show during an in-flight request does not start a second chain", async () => {
  const w = world();
  const s = slowUpdate();
  const u = new BRPC.AutoUpdater(s.fn, 1000);
  u.start();
  assert.strictEqual(s.calls, 1);
  for (let k = 0; k < 10; k++) { w.doc.setHidden(true); w.doc.setHidden(false); }
  assert.strictEqual(s.calls, 1, "a toggle while a request is in flight must not send another");
  await s.release();
  assert.strictEqual(w.timers.size, 1, "exactly one re-arm");
  for (let k = 0; k < 5; k++) {
    w.fire();
    for (let j = 0; j < 3; j++) { w.doc.setHidden(true); w.doc.setHidden(false); }
    await s.release();
    assert.ok(w.timers.size <= 1, `round ${k}: ${w.timers.size} timers pending`);
  }
  assert.strictEqual(s.maxInFlight, 1);
});

test("stop() + start() during an in-flight request keeps one chain", async () => {
  const w = world();
  const s = slowUpdate();
  const u = new BRPC.AutoUpdater(s.fn, 1000);
  u.start();
  u.stop(); u.start(); u.stop(); u.start();
  assert.strictEqual(s.calls, 1);
  await s.release();
  assert.strictEqual(w.timers.size, 1);
  w.fire();
  assert.strictEqual(s.calls, 2);
  await s.release();
  assert.strictEqual(w.timers.size, 1);
  assert.strictEqual(s.maxInFlight, 1);
});

test("stopped while in flight, it does not re-arm", async () => {
  const w = world();
  const s = slowUpdate();
  const u = new BRPC.AutoUpdater(s.fn, 1000);
  u.start();
  u.stop();
  await s.release();
  assert.strictEqual(w.timers.size, 0);
});

test("a failing analyzer backs off to 5 s, with one chain, however often it is poked", async () => {
  const w = world();
  const s = slowUpdate();
  const u = new BRPC.AutoUpdater(s.fn, 1000);
  let errors = 0;
  u.onError = () => { errors++; };
  const orig = console.error; console.error = () => {};
  try {
    u.start();
    for (let k = 0; k < 4; k++) {
      w.doc.setHidden(true); w.doc.setHidden(false);
      await s.release(true);
      const t = [...w.timers.values()];
      assert.strictEqual(t.length, 1, `round ${k}`);
      assert.strictEqual(t[0].ms, 5000, "back off, do not hammer");
      w.fire();
    }
  } finally { console.error = orig; }
  assert.strictEqual(errors, 4);
  assert.strictEqual(s.maxInFlight, 1);
});

test("hidden: no polling; visible again: one immediate tick", async () => {
  const w = world();
  const s = slowUpdate();
  const u = new BRPC.AutoUpdater(s.fn, 1000);
  u.start();
  await s.release();
  w.doc.hidden = true;
  w.fire();                                   // the re-arm fires while hidden
  assert.strictEqual(s.calls, 1, "nothing is fetched for a hidden tab");
  assert.strictEqual(w.timers.size, 0);
  w.doc.setHidden(false);
  assert.strictEqual(s.calls, 2, "comes back at once");
  w.doc.setHidden(true); w.doc.setHidden(false);
  assert.strictEqual(s.calls, 2, "and only once");
});
