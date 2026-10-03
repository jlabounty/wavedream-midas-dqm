//
// The JavaScript smaf decoder against Python's encoder.
//
// Same arrangement as scopeframe.test.js: tests/generate_smaframe_cases.py runs
// the real plugin over stored real frames and writes the bytes it would send,
// plus what framing.decode_sma_frame reads back. A decoder that disagrees does
// not fail in the browser, it draws hits on the wrong lane.
//

const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");

const PAGES = path.join(__dirname, "..", "..", "pages", "js");
globalThis.BRPC = require(path.join(PAGES, "dqm-brpc.js"));
const SMAF = require(path.join(PAGES, "dqm-smaframe.js"));

const CASES = path.join(__dirname, "smaframe-cases.json");
const cases = JSON.parse(fs.readFileSync(CASES, "utf8")).cases;

function bufferOf(hex) {
  const b = Buffer.from(hex, "hex");
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}

function check(f, e, name) {
  assert.strictEqual(f.version, e.version, `${name}: version`);
  assert.strictEqual(f.flags, e.flags, `${name}: flags`);
  assert.strictEqual(f.nHits, e.n_hits, `${name}: n_hits`);
  assert.strictEqual(f.frameSeq, e.frame_seq, `${name}: frame_seq`);
  assert.strictEqual(f.run, e.run_number, `${name}: run`);
  assert.strictEqual(f.jsonLen, e.json_len, `${name}: json_len`);
  assert.strictEqual(f.arraysOffset, e.arrays_offset, `${name}: arrays offset`);
  assert.strictEqual(f.seeded, e.seeded, `${name}: seeded`);
  assert.strictEqual(f.stale, e.stale, `${name}: stale`);
  assert.strictEqual(f.truncated, e.truncated, `${name}: truncated`);
  if (e.suspect !== undefined) assert.strictEqual(f.suspect, e.suspect, `${name}: suspect`);
  assert.deepStrictEqual(f.meta, e.meta, `${name}: meta`);
  // Cases may give the wire values (t_rel_ns) and the time shift k; the
  // decoded times are the wire values scaled by 2^k either way.
  const k = e.time_shift !== undefined ? e.time_shift : 0;
  assert.strictEqual(f.timeShift, k, `${name}: time shift`);
  if (e.t_ns !== undefined) assert.deepStrictEqual(Array.from(f.t), e.t_ns, `${name}: t_ns`);
  assert.deepStrictEqual(Array.from(f.tRaw), e.t_rel_ns, `${name}: t_rel_ns (wire)`);
  assert.deepStrictEqual(Array.from(f.t), e.t_rel_ns.map((x) => x * 2 ** k), `${name}: times`);
  assert.deepStrictEqual(Array.from(f.ch), e.ch, `${name}: ch`);
  assert.deepStrictEqual(Array.from(f.tot), e.tot, `${name}: tot`);
  assert.deepStrictEqual(Array.from(f.hitFlags), e.hit_flags, `${name}: hit flags`);
  assert.strictEqual(f.words, e.words, `${name}: words`);
  if (e.words) {
    assert.ok(f.rawWord instanceof BigUint64Array, `${name}: raw words are u64`);
    assert.deepStrictEqual(Array.from(f.rawWord, (w) => w.toString(16).padStart(16, "0")),
                           e.raw_words, `${name}: raw words`);
    assert.deepStrictEqual(Array.from(f.wordIndex), e.word_index, `${name}: word index`);
  } else {
    assert.strictEqual(f.rawWord, null, `${name}: no raw words in v1`);
    assert.strictEqual(f.wordIndex, null, `${name}: no word index in v1`);
  }
  if (e.pairing !== undefined) {
    assert.strictEqual(f.pairing, e.pairing, `${name}: pairing`);
    if (e.pairing) {
      if (e.pair === null) {
        assert.strictEqual(f.pair, null, `${name}: no partners without words`);
      } else {
        assert.ok(f.pair instanceof Int32Array, `${name}: pair is i32`);
        assert.deepStrictEqual(Array.from(f.pair), e.pair, `${name}: pair`);
      }
      assert.deepStrictEqual(Array.from(f.cls), e.cls, `${name}: cls`);
    } else {
      assert.strictEqual(f.pair, null, `${name}: no pair before v3`);
      assert.strictEqual(f.cls, null, `${name}: no cls before v3`);
    }
  }
  checkPixels(f, e, name);
}

function checkPixels(f, e, name) {
  if (e.pixels === undefined) return;
  if (e.pixels === null) {
    assert.strictEqual(f.pixels, null, `${name}: no pixel block`);
    assert.ok(!(f.flags & SMAF.FLAGS.PIXELS), `${name}: no PIXELS flag`);
    return;
  }
  const p = f.pixels, x = e.pixels;
  assert.ok(f.flags & SMAF.FLAGS.PIXELS, `${name}: PIXELS flag`);
  assert.strictEqual(p.n, x.n, `${name}: pixel count`);
  assert.strictEqual(p.timeShift, x.time_shift, `${name}: pixel time shift`);
  assert.strictEqual(p.words, x.words, `${name}: pixel words`);
  assert.strictEqual(p.offset, x.offset, `${name}: pixel block offset`);
  assert.deepStrictEqual(Array.from(p.tRaw), x.t_rel, `${name}: pixel t_rel (wire)`);
  assert.deepStrictEqual(Array.from(p.t), x.t_rel.map((v) => v * 2 ** x.time_shift), `${name}: pixel t`);
  for (const k of ["chip", "col", "row", "tot", "flags"]) {
    assert.deepStrictEqual(Array.from(p[k]), x[k], `${name}: pixel ${k}`);
  }
  if (x.words) {
    assert.deepStrictEqual(Array.from(p.rawWord, (w) => w.toString(16).padStart(16, "0")),
                           x.raw_words, `${name}: pixel raw words`);
    assert.deepStrictEqual(Array.from(p.wordIndex), x.word_index, `${name}: pixel word index`);
  } else {
    assert.strictEqual(p.rawWord, null, `${name}: no pixel raw words`);
    assert.strictEqual(p.wordIndex, null, `${name}: no pixel word index`);
  }
}

for (const c of cases) {
  test(`decodes Python's ${c.name}`, () => {
    check(SMAF.decode(bufferOf(c.payload_hex)), c.expect, c.name);
  });

  for (const shift of [3, 4]) {
    test(`decodes ${c.name} from a view at byte ${shift}`, () => {
      // A reply sliced out of a bigger buffer at an offset: at 3 a Uint32Array
      // would throw, at 4 a BigUint64Array would (the u32 arrays are fine
      // there), so the decoder must copy exactly the arrays that do not fit.
      const raw = Buffer.from(c.payload_hex, "hex");
      const big = new Uint8Array(raw.length + shift);
      big.set(raw, shift);
      check(SMAF.decode(new Uint8Array(big.buffer, shift, raw.length)), c.expect, c.name);
    });
  }
}

test("v1, v2 and v3 are all among the cases", () => {
  const versions = new Set(cases.map((c) => c.expect.version));
  assert.deepStrictEqual([...versions].sort(), [1, 2, 3]);
  const v3 = cases.filter((c) => c.expect.version === 3);
  assert.deepStrictEqual([...new Set(v3.map((c) => c.expect.words))].sort(), [false, true],
                         "v3 with and without words");
});

test("v3: the live raster carries classes only, 8 bytes a hit", () => {
  const c = cases.find((x) => x.expect.version === 3 && x.expect.n_hits > 1000);
  assert.ok(c, "a real v3 raster");
  const f = SMAF.decode(bufferOf(c.payload_hex));
  assert.ok(f.pairing && !f.words && f.pair === null && f.cls.length === f.nHits);
  assert.strictEqual(f.offsets.cls - f.offsets.t, 4 * f.nHits + 3 * f.nHits);
});

test("v3: pair is a view when aligned, -1 survives, partners point back at each other", () => {
  const c = cases.find((x) => x.expect.version === 3 && x.expect.words && x.expect.n_hits >= 300);
  assert.ok(c, "a real v3 raster with words");
  const ab = bufferOf(c.payload_hex);
  const f = SMAF.decode(ab);
  assert.strictEqual(f.pair.buffer, ab, "i32 pair: a view");
  assert.ok(f.pairing && f.words);
  let pairs = 0, none = 0;
  for (let i = 0; i < f.nHits; i++) {
    const j = f.pair[i];
    if (j < 0) { none++; continue; }
    pairs++;
    assert.strictEqual(f.pair[j], i, `hit ${i}'s partner ${j} points back`);
    assert.strictEqual(f.cls[i] & SMAF.PAIR.CLASS_MASK, SMAF.PAIR.PAIRED, "a partner means paired");
    assert.notStrictEqual(f.cls[i] & SMAF.PAIR.NIM_SIDE, f.cls[j] & SMAF.PAIR.NIM_SIDE,
                          "a pair is one TOT word and one NIM word");
  }
  assert.ok(pairs > 50 && none > 0, `${pairs} paired, ${none} without a partner`);
  // At an odd offset the i32 array is copied, with the same values.
  const raw = Buffer.from(c.payload_hex, "hex");
  const big = new Uint8Array(raw.length + 3);
  big.set(raw, 3);
  const g = SMAF.decode(new Uint8Array(big.buffer, 3, raw.length));
  assert.notStrictEqual(g.pair.buffer, big.buffer);
  assert.deepStrictEqual(Array.from(g.pair), Array.from(f.pair));
});

test("v3: the cls byte packs the class and the sub-flags", () => {
  const c = cases.find((x) => /NIM-only lag-held, echo/.test(x.name));
  const f = SMAF.decode(bufferOf(c.payload_hex));
  const P = SMAF.PAIR;
  const cl = Array.from(f.cls, (b) => b & P.CLASS_MASK);
  assert.deepStrictEqual(cl, [P.PAIRED, P.PAIRED, P.ECHO, P.NIM_ONLY, P.TOT_ONLY]);
  assert.ok(f.cls[3] & P.LAG_HELD && f.cls[3] & P.NIM_SIDE);
  assert.ok(f.cls[1] & P.NIM_SIDE && !(f.cls[0] & P.NIM_SIDE));
  assert.deepStrictEqual(Array.from(f.pair), [1, 0, -1, -1, -1]);
  assert.strictEqual(SMAF.hex64(f.rawWord[3]), "0xffffffffffffffff", "words travel with v3 too");
});

test("v2 arrays are views too when aligned, and copied only where they must be", () => {
  const c = cases.find((x) => x.expect.words && x.expect.n_hits > 100);
  assert.ok(c, "a real raster with words");
  const ab = bufferOf(c.payload_hex);
  const f = SMAF.decode(ab);
  assert.strictEqual(f.rawWord.buffer, ab, "u64 words: a view");
  assert.strictEqual(f.wordIndex.buffer, ab, "word index: a view");
  assert.strictEqual(f.t.buffer, ab);
  const raw = Buffer.from(c.payload_hex, "hex");
  const big = new Uint8Array(raw.length + 4);
  big.set(raw, 4);
  const g = SMAF.decode(new Uint8Array(big.buffer, 4, raw.length));
  assert.notStrictEqual(g.rawWord.buffer, big.buffer, "4-aligned only: the u64 array is copied");
  assert.strictEqual(g.wordIndex.buffer, big.buffer, "the u32 arrays are not");
});

test("hex64 prints 16 digits, top bit and small words included", () => {
  assert.strictEqual(SMAF.hex64(0x8512n << 48n), "0x8512000000000000");
  assert.strictEqual(SMAF.hex64(1n), "0x0000000000000001");
  assert.strictEqual(SMAF.hex64(0xFFFFFFFFFFFFFFFFn), "0xffffffffffffffff");
  const c = cases.find((x) => /past 2\^63/.test(x.name));
  const f = SMAF.decode(bufferOf(c.payload_hex));
  assert.strictEqual(SMAF.hex64(f.rawWord[0]), "0x85123456789abcde");
  assert.strictEqual(SMAF.hex64(f.rawWord[3]), "0x0020000000000001", "past 2^53 stays exact");
  assert.strictEqual(f.wordIndex[2], 4294967295);
});

test("the arrays are views, not copies, when aligned", () => {
  const c = cases.find((x) => x.expect.n_hits > 100);
  const ab = bufferOf(c.payload_hex);
  const f = SMAF.decode(ab);
  assert.strictEqual(f.t.buffer, ab, "a 33k-hit raster must not be copied per poll");
  assert.strictEqual(f.ch.buffer, ab);
});

test("a time shift scales the times without overflowing u32", () => {
  const c = cases.find((x) => x.expect.n_hits > 10 && !x.expect.time_shift);
  const ab = bufferOf(c.payload_hex);
  new DataView(ab).setUint16(2, 4, true);
  const f = SMAF.decode(ab);
  assert.strictEqual(f.timeShift, 4);
  f.tRaw.forEach((v, i) => assert.strictEqual(f.t[i], v * 16));
  const big = new DataView(bufferOf(c.payload_hex));
  big.setUint16(2, 8, true);
  big.setUint32(f.offsets.t, 0xFFFFFFFF, true);            // after the u64 words in v2
  assert.strictEqual(SMAF.decode(big.buffer).t[0], 0xFFFFFFFF * 256, "past 2^32 ns stays exact");
});

test("an unknown version is refused, not misread", () => {
  const ab = bufferOf(cases[0].payload_hex);
  new DataView(ab).setUint8(0, 4);
  assert.throws(() => SMAF.decode(ab), /unknown smaf version 4/);
});

test("a version that disagrees with the PAIRING flag is refused", () => {
  const v1 = cases.find((x) => x.expect.version === 1);
  const ab = bufferOf(v1.payload_hex);
  new DataView(ab).setUint8(0, 3);                       // v3 claimed, no PAIRING flag
  assert.throws(() => SMAF.decode(ab), /smaf version 3 with flags/);
  const v3 = cases.find((x) => x.expect.version === 3 && x.expect.words);
  const ab3 = bufferOf(v3.payload_hex);
  new DataView(ab3).setUint8(0, 2);                      // v2 claimed, PAIRING set
  assert.throws(() => SMAF.decode(ab3), /smaf version 2 with flags/);
});

test("a version that disagrees with the WORDS flag is refused", () => {
  const v1 = cases.find((x) => x.expect.version === 1);
  const ab = bufferOf(v1.payload_hex);
  new DataView(ab).setUint8(0, 2);                       // v2 claimed, no WORDS flag
  assert.throws(() => SMAF.decode(ab), /smaf version 2 with flags/);
  const v2 = cases.find((x) => x.expect.version === 2);
  const ab2 = bufferOf(v2.payload_hex);
  new DataView(ab2).setUint8(0, 1);                      // v1 claimed, WORDS set
  assert.throws(() => SMAF.decode(ab2), /smaf version 1 with flags/);
});

test("a truncated payload is refused", () => {
  const c = cases.find((x) => x.expect.n_hits > 10);
  const ab = bufferOf(c.payload_hex).slice(0, c.expect.arrays_offset + 10);
  assert.throws(() => SMAF.decode(ab), /short smaf payload/);
});

test("the flag constants agree with framing.py", () => {
  const py = fs.readFileSync(
    path.join(__dirname, "..", "..", "src", "mdqm", "dqm", "framing.py"), "utf8");
  const want = {
    SMAF_SEEDED: SMAF.FLAGS.SEEDED, SMAF_STALE: SMAF.FLAGS.STALE,
    SMAF_TRUNCATED: SMAF.FLAGS.TRUNCATED, SMAF_SUSPECT: SMAF.FLAGS.SUSPECT,
    SMAF_WORDS: SMAF.FLAGS.WORDS, SMAF_PAIRING: SMAF.FLAGS.PAIRING, HIT_MISMATCH: SMAF.HIT.MISMATCH,
    PAIR_MULTI: SMAF.PAIR.MULTI, PAIR_SHADOW: SMAF.PAIR.SHADOW, PAIR_EDGE: SMAF.PAIR.EDGE,
    PAIR_LAG_HELD: SMAF.PAIR.LAG_HELD, PAIR_NIM_SIDE: SMAF.PAIR.NIM_SIDE,
    HIT_TOT_CORRUPT: SMAF.HIT.TOT_CORRUPT, HIT_FINE_LSB: SMAF.HIT.FINE_LSB,
    HIT_IN_SEED: SMAF.HIT.IN_SEED, HIT_STALE: SMAF.HIT.STALE,
  };
  for (const [name, value] of Object.entries(want)) {
    const m = new RegExp(`^${name} = 1 << (\\d+)`, "m").exec(py);
    assert.ok(m, `${name} not found in framing.py`);
    assert.strictEqual(1 << Number(m[1]), value, name);
  }
  assert.ok(/^PAIR_CLASS_MASK = 0x07/m.test(py) && SMAF.PAIR.CLASS_MASK === 7);
  assert.ok(new RegExp(`^PAIR_NONE = ${SMAF.PAIR.NONE}\\b`, "m").test(py));
  // The classes are sma_nim's.
  const nim = fs.readFileSync(
    path.join(__dirname, "..", "..", "src", "mdqm", "plugins", "sma_nim.py"), "utf8");
  for (const [name, v] of [["PAIRED", SMAF.PAIR.PAIRED], ["TOT_ONLY", SMAF.PAIR.TOT_ONLY],
                           ["ECHO_WORD", SMAF.PAIR.ECHO], ["NIM_ONLY", SMAF.PAIR.NIM_ONLY]]) {
    assert.ok(new RegExp(`^${name} = ${v}$`, "m").test(nim), `sma_nim.${name} = ${v}`);
  }
});

test("every case ran", () => {
  assert.ok(cases.length >= 20, `only ${cases.length} cases`);
});

test("pixel blocks come after v1, v2 and v3, with and without words, and old fields are untouched", () => {
  const px = cases.filter((c) => c.expect.pixels);
  assert.deepStrictEqual([...new Set(px.map((c) => c.expect.version))].sort(), [1, 2, 3]);
  assert.deepStrictEqual([...new Set(px.map((c) => c.expect.pixels.words))].sort(), [false, true]);
  // The same real frame with and without the block: the trigger hits read the same.
  const a = cases.find((c) => c.name === "real 4k-word frame, seeded (v2, words)");
  const b = cases.find((c) => /seeded \(v2, words\), no pixel block/.test(c.name));
  const fa = SMAF.decode(bufferOf(a.payload_hex)), fb = SMAF.decode(bufferOf(b.payload_hex));
  assert.ok(fa.pixels && fa.pixels.n > 0 && fb.pixels === null);
  assert.deepStrictEqual(Array.from(fa.t), Array.from(fb.t));
  assert.deepStrictEqual(Array.from(fa.ch), Array.from(fb.ch));
});

test("a truncated pixel block is refused, and an unknown block version too", () => {
  const c = cases.find((x) => x.expect.pixels && x.expect.pixels.n > 100);
  const at = c.expect.pixels.offset;
  assert.throws(() => SMAF.decode(bufferOf(c.payload_hex).slice(0, at + 20)), /short smaf pixel block/);
  const ab = bufferOf(c.payload_hex);
  new DataView(ab).setUint8(at + 6, 9);
  assert.throws(() => SMAF.decode(ab), /unknown smaf pixel block version 9/);
});

test("the pixel flag constants agree with framing.py", () => {
  const py = fs.readFileSync(
    path.join(__dirname, "..", "..", "src", "mdqm", "dqm", "framing.py"), "utf8");
  assert.ok(new RegExp(`^SMAF_PIXELS = 1 << ${Math.log2(SMAF.FLAGS.PIXELS)}`, "m").test(py));
  assert.ok(new RegExp(`^PIX_OFF_SENSOR = 1 << ${Math.log2(SMAF.PIX.OFF_SENSOR)}`, "m").test(py));
  assert.ok(new RegExp(`^PIX_IN_SEED = 1 << ${Math.log2(SMAF.PIX.IN_SEED)}`, "m").test(py));
  assert.ok(/^PIX_PLANE_MASK = 0x3/m.test(py) && SMAF.PIX.PLANE_MASK === 3);
});
