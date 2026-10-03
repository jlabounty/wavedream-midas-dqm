//
// dqm-smaframe.js -- the SMA event-display frame ("smaf" v1, v2 and v3, with
// or without the MuPix pixel block), decoded.
//
// Mirror of mdqm/dqm/framing.py's encode_sma_frame. Keep the two in step: the
// cross-language test in tests/js/smaframe.test.js decodes Python's bytes with
// this function, so a divergence fails a test rather than misdrawing a hit on
// the wrong channel -- which on an event display is the worst kind of bug,
// because it looks exactly like a detector problem.
//
// Layout (offsets from the payload start, envelope already removed):
//
//     0   u8   version = 1, 2 with per-hit words (flag WORDS), 3 with the
//               per-hit TOT + NIM pairing (flag PAIRING; words optional)
//     1   u8   flags        SEEDED | STALE | TRUNCATED | SUSPECT | WORDS | PIXELS | PAIRING
//     2   u16  time shift k (0 = reserved in the first v1 encoder): the u32
//               times are in units of 2^k ns, t_rel_ns = t << k
//     4   u32  n_hits       n
//     8   u64  frame_seq
//     16  u32  run
//     20  u32  json_len
//     24  u64  reserved
//     32  JSON metadata, json_len bytes, NUL-padded to a multiple of 8
//     A = 32 + ceil8(json_len)
//
//   version 1 (7 bytes a hit)         version 2 (19 bytes a hit)
//     A        u32 t_rel[n]             A        u64 raw_word[n]
//     A + 4n   u8  ch[n]                A + 8n   u32 t_rel[n]
//     A + 5n   u8  tot[n]               A + 12n  u32 word_index[n]
//     A + 6n   u8  hit_flags[n]         A + 16n  u8  ch[n], tot[n], hit_flags[n]
//
//   version 3 without words (8 bytes)   version 3 with words (24 bytes a hit)
//     A        u32 t_rel[n]             A        u64 raw_word[n]
//     A + 4n   u8  ch, tot, hit_flags,  A + 8n   u32 t_rel[n]
//              cls [n each]             A + 12n  u32 word_index[n]
//                                       A + 16n  i32 pair[n]
//                                       A + 20n  u8  ch, tot, hit_flags, cls [n each]
//
// pair (with the words only: the live raster needs the classes, not the
// partners): the index in this hit list of the word this hit is paired with
// (TOT word <-> its NIM copy), -1 when none or not shipped. cls: bits 2:0 the class
// (PAIR.PAIRED 0, TOT_ONLY 1, ECHO 2, NIM_ONLY 3, NONE 7 = not a paired
// counter's word), then MULTI, SHADOW, EDGE, LAG_HELD, NIM_SIDE (see PAIR).
//
// hit time = meta.t0_ns + t_rel * 2^k. raw_word is the 64-bit word as the
// board sent it and word_index its position in the H000 bank (in 64-bit words,
// filler and pixel words included): what finds the hit again in the file.
//
// The MuPix pixel block (header flag PIXELS), in any version, after the hit
// arrays at P = their end rounded up to 8 (an old decoder simply ignores it):
//
//     P      u32 n_pix m, u16 pixel time shift kp, u8 block version 1, u8 block flags
//     Q = P + 8
//     no words (9 bytes a pixel)        words (block flag WORDS, 21 bytes a pixel)
//       Q       u32 t_rel[m]              Q        u64 raw_word[m]
//       Q + 4m  u8 chip, col, row,        Q + 8m   u32 t_rel[m]
//               tot, flags [m each]       Q + 12m  u32 word_index[m]
//                                         Q + 16m  u8 chip, col, row, tot, flags [m each]
//
// pixel time = meta.mupix.t0_ns + t_rel * 2^kp; tot is the real ToT (0-31,
// meta.mupix.tot_ns a count); flags: bits 1:0 the plane (0 none, 1 L1, 2 L2),
// OFF_SENSOR (row >= 250), IN_SEED.
//
// The hit arrays are wrapped as typed-array *views* rather than copied or
// parsed: a whole-frame raster is ~33k hits, and the whole point of the binary
// format is that the page does not touch each one before drawing it.
//

(function (root) {
"use strict";

const VERSION = 1;
const VERSION_WORDS = 2;
const VERSION_PAIRING = 3;
const HEADER_BYTES = 32;

const FLAGS = { SEEDED: 1 << 0, STALE: 1 << 1, TRUNCATED: 1 << 2, SUSPECT: 1 << 3, WORDS: 1 << 4,
                PIXELS: 1 << 5, PAIRING: 1 << 6 };
// The v3 cls byte (framing.py PAIR_*): the class in bits 2:0, then the sub-flags.
const PAIR = {
  CLASS_MASK: 0x07,
  PAIRED: 0,              // a TOT word with its NIM copy (or that NIM copy)
  TOT_ONLY: 1,            // a TOT word with no NIM copy in the window
  ECHO: 2,                // a TOT echo word (late, or on the previous word's edge): not paired
  NIM_ONLY: 3,            // a NIM copy with no TOT word: what the merge adds
  NONE: 7,                // not a paired counter's word (RF, S1 without a copy, ...)
  MULTI: 1 << 3,          // more than one candidate in the window
  SHADOW: 1 << 4,         // in the previous TOT word's shadow
  EDGE: 1 << 5,           // near the frame's first/last hit
  LAG_HELD: 1 << 6,       // a NIM-only word held back from the merge (lag fault)
  NIM_SIDE: 1 << 7,       // the word is a NIM copy (else a TOT word)
};
const PIX = { PLANE_MASK: 0x3, OFF_SENSOR: 1 << 2, IN_SEED: 1 << 3 };
const PIXB = { VERSION: 1, WORDS: 1 << 0, HEADER_BYTES: 8 };
const HIT = {
  MISMATCH: 1 << 0,       // fine and coarse disagree
  TOT_CORRUPT: 1 << 1,    // ToT >= the corrupt threshold
  FINE_LSB: 1 << 2,       // fine bit 0 set
  IN_SEED: 1 << 3,        // inside a seed window
  STALE: 1 << 4,          // belongs to a stale frame
};

// Uint32Array reads in the platform's byte order and the wire is
// little-endian. Every machine a browser runs on is little-endian, but the
// check costs nothing and the fallback keeps a big-endian reader correct.
const LITTLE_ENDIAN = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;

/**
 * n u32 at payload offset `at`: a view when aligned, else a copy.
 * `base` is the payload's offset in `buffer`; alignment is of base + at.
 */
function u32Array(buffer, base, dv, at, n) {
  if (n === 0) return new Uint32Array(0);
  if ((base + at) % 4 === 0 && LITTLE_ENDIAN) return new Uint32Array(buffer, base + at, n);
  const out = new Uint32Array(n);
  for (let i = 0; i < n; i++) out[i] = dv.getUint32(at + 4 * i, true);
  return out;
}

/** The same for i32, as an Int32Array. */
function i32Array(buffer, base, dv, at, n) {
  if (n === 0) return new Int32Array(0);
  if ((base + at) % 4 === 0 && LITTLE_ENDIAN) return new Int32Array(buffer, base + at, n);
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) out[i] = dv.getInt32(at + 4 * i, true);
  return out;
}

/** The same for u64, as a BigUint64Array (a view needs an 8-aligned offset). */
function u64Array(buffer, base, dv, at, n) {
  if (n === 0) return new BigUint64Array(0);
  if ((base + at) % 8 === 0 && LITTLE_ENDIAN) return new BigUint64Array(buffer, base + at, n);
  if (LITTLE_ENDIAN) return new BigUint64Array(buffer.slice(base + at, base + at + 8 * n));
  const out = new BigUint64Array(n);
  for (let i = 0; i < n; i++) out[i] = dv.getBigUint64(at + 8 * i, true);
  return out;
}

/**
 * Decode one smaf payload, version 1, 2 or 3.
 *
 * Accepts an ArrayBuffer or any ArrayBufferView (a Node Buffer, a Uint8Array
 * slice of a bigger reply). A view can start at any byte offset, and a
 * Uint32Array (BigUint64Array) over an offset that is not a multiple of 4 (8)
 * throws -- so a multi-byte array that does not land on its boundary in the
 * underlying buffer is copied out. BRPC.call hands over a fresh slice starting
 * at 0, where the arrays are aligned by construction and nothing is copied.
 *
 * `rawWord` (BigUint64Array) and `wordIndex` (Uint32Array) are null without
 * words; `cls` (Uint8Array) is null without pairing, `pair` (Int32Array)
 * without pairing or without words.
 */
function decode(input) {
  let buffer, base, length;
  if (input instanceof ArrayBuffer) {
    buffer = input; base = 0; length = input.byteLength;
  } else if (input && input.buffer instanceof ArrayBuffer) {
    buffer = input.buffer; base = input.byteOffset; length = input.byteLength;
  } else {
    throw new Error("smaf: expected an ArrayBuffer or a typed-array view");
  }
  if (length < HEADER_BYTES) throw new Error(`short smaf payload: ${length} bytes`);

  const dv = new DataView(buffer, base, length);
  const LE = true;
  const version = dv.getUint8(0);
  if (version !== VERSION && version !== VERSION_WORDS && version !== VERSION_PAIRING) {
    throw new Error(`unknown smaf version ${version}`);
  }
  const flags = dv.getUint8(1);
  // v1: neither; v2: words; v3: pairing, with or without words.
  const words = !!(flags & FLAGS.WORDS);
  const pairing = version === VERSION_PAIRING;
  if (pairing !== !!(flags & FLAGS.PAIRING) || (!pairing && words !== (version === VERSION_WORDS))) {
    throw new Error(`smaf version ${version} with flags 0x${flags.toString(16)}`);
  }
  // Coarser time units let a frame longer than 4.29 s (a stale buffer can
  // span minutes) still travel as u32. Zero, the usual case, means ns.
  const timeShift = dv.getUint16(2, LE);
  const nHits = dv.getUint32(4, LE);
  // 2^53 is ~285 years of frames at 1 kHz; a Number is exact for any real seq.
  const frameSeq = Number(dv.getBigUint64(8, LE));
  const run = dv.getUint32(16, LE);
  const jsonLen = dv.getUint32(20, LE);

  if (HEADER_BYTES + jsonLen > length) {
    throw new Error(`short smaf payload: ${length} bytes, JSON block needs ${jsonLen}`);
  }
  const text = new TextDecoder().decode(new Uint8Array(buffer, base + HEADER_BYTES, jsonLen));
  const meta = jsonLen ? JSON.parse(text) : {};

  let at = HEADER_BYTES + jsonLen;
  if (at % 8) at += 8 - (at % 8);
  const perHit = 7 + (words ? 12 : 0) + (pairing ? 1 : 0) + (pairing && words ? 4 : 0);
  if (length < at + perHit * nHits) {
    throw new Error(`short smaf payload: ${length} bytes for ${nHits} hits at ${at}`);
  }

  // Payload offsets of each array (the tests poke bytes through these).
  const offsets = words
    ? { rawWord: at, t: at + 8 * nHits, wordIndex: at + 12 * nHits, ch: at + 16 * nHits }
    : { rawWord: null, t: at, wordIndex: null, ch: at + 4 * nHits };
  offsets.pair = null;
  offsets.cls = null;
  if (pairing && words) {
    offsets.pair = offsets.ch;
    offsets.ch += 4 * nHits;
  }
  offsets.tot = offsets.ch + nHits;
  offsets.hitFlags = offsets.ch + 2 * nHits;
  if (pairing) offsets.cls = offsets.ch + 3 * nHits;

  let t = u32Array(buffer, base, dv, offsets.t, nHits);
  const tRaw = t;
  if (timeShift) {
    // value << k can pass 2^32, which a Uint32Array cannot hold; a double is
    // exact to 2^53 ns (104 days), far beyond any frame.
    const f = 2 ** timeShift;
    t = new Float64Array(nHits);
    for (let i = 0; i < nHits; i++) t[i] = tRaw[i] * f;
  }
  const rawWord = words ? u64Array(buffer, base, dv, offsets.rawWord, nHits) : null;
  const wordIndex = words ? u32Array(buffer, base, dv, offsets.wordIndex, nHits) : null;
  const ch = new Uint8Array(buffer, base + offsets.ch, nHits);
  const tot = new Uint8Array(buffer, base + offsets.tot, nHits);
  const hitFlags = new Uint8Array(buffer, base + offsets.hitFlags, nHits);
  const pair = pairing && words ? i32Array(buffer, base, dv, offsets.pair, nHits) : null;
  const cls = pairing ? new Uint8Array(buffer, base + offsets.cls, nHits) : null;

  const end = offsets.hitFlags + nHits + (pairing ? nHits : 0);
  const pixels = flags & FLAGS.PIXELS
    ? decodePixels(buffer, base, dv, length, end + ((8 - (end % 8)) % 8)) : null;

  return {
    version, flags, timeShift, nHits, frameSeq, run, jsonLen, arraysOffset: at, offsets, pixels,
    seeded: !!(flags & FLAGS.SEEDED),
    stale: !!(flags & FLAGS.STALE),
    truncated: !!(flags & FLAGS.TRUNCATED),
    // No usable time base: most hits fell outside the time clusters, which
    // usually means the coarse shift is wrong rather than the board.
    suspect: !!(flags & FLAGS.SUSPECT),
    words, pairing,
    meta, t, tRaw, ch, tot, hitFlags, rawWord, wordIndex, pair, cls,
  };
}

/**
 * The pixel block at payload offset `at`: {n, timeShift, words, offset, t (ns
 * from meta.mupix.t0_ns), tRaw, chip, col, row, tot, flags, rawWord, wordIndex}.
 */
function decodePixels(buffer, base, dv, length, at) {
  if (length < at + PIXB.HEADER_BYTES) {
    throw new Error(`short smaf payload: no room for the pixel block at ${at}`);
  }
  const n = dv.getUint32(at, true);
  const timeShift = dv.getUint16(at + 4, true);
  const version = dv.getUint8(at + 6);
  if (version !== PIXB.VERSION) throw new Error(`unknown smaf pixel block version ${version}`);
  const words = !!(dv.getUint8(at + 7) & PIXB.WORDS);
  const q = at + PIXB.HEADER_BYTES;
  if (length < q + (words ? 21 : 9) * n) {
    throw new Error(`short smaf pixel block: ${length - q} bytes for ${n} pixel hits`);
  }
  const o = words ? { raw: q, t: q + 8 * n, wi: q + 12 * n, b: q + 16 * n } : { t: q, b: q + 4 * n };
  const tRaw = u32Array(buffer, base, dv, o.t, n);
  let t = tRaw;
  if (timeShift) {
    const f = 2 ** timeShift;
    t = new Float64Array(n);
    for (let i = 0; i < n; i++) t[i] = tRaw[i] * f;
  }
  const bytes = (j) => new Uint8Array(buffer, base + o.b + j * n, n);
  return {
    n, timeShift, words, offset: at, t, tRaw,
    chip: bytes(0), col: bytes(1), row: bytes(2), tot: bytes(3), flags: bytes(4),
    rawWord: words ? u64Array(buffer, base, dv, o.raw, n) : null,
    wordIndex: words ? u32Array(buffer, base, dv, o.wi, n) : null,
  };
}

/** A raw 64-bit word as the page shows it: 0x and 16 hex digits. */
function hex64(w) {
  return `0x${BigInt.asUintN(64, BigInt(w)).toString(16).padStart(16, "0")}`;
}

/**
 * Ask an analyzer for its latest frame. Returns null when there is none yet.
 *
 * `args` is sent as JSON, e.g. {view: "raster", drop: [7]}, or with
 * {seq: N, words: true} for frame N again with its words. A "json" reply is
 * the analyzer's {"no_frame": true}: not an error, there has simply been no
 * frame of that kind since it started. An "err " reply (frame N no longer
 * held) throws with the analyzer's text.
 */
async function fetchFrame(client, args, maxLength) {
  const { tag, payload } = await root.BRPC.call(client, "sma::frame", JSON.stringify(args || {}),
                                                maxLength);
  if (tag === "err") throw new Error(root.BRPC.textOf(payload));
  if (tag === "json" || !payload.byteLength) return null;
  if (tag !== "smaf") throw new Error(`sma::frame answered with tag '${tag}', not smaf`);
  return decode(payload);
}

/**
 * The raw MIDAS event of frame `seq` (sma::raw): an ArrayBuffer holding the
 * 16-byte event header and its banks, which on its own is a valid one-event
 * .mid file. Throws with the analyzer's text when it no longer holds the frame.
 */
async function fetchRaw(client, seq) {
  const { tag, payload } = await root.BRPC.call(client, "sma::raw", JSON.stringify({ seq }));
  if (tag === "err") throw new Error(root.BRPC.textOf(payload));
  if (tag !== "mevt") throw new Error(`sma::raw answered with tag '${tag}', not mevt`);
  return payload;
}

const SMAF = { decode, fetchFrame, fetchRaw, hex64, FLAGS, HIT, PIX, PAIR, VERSION, VERSION_WORDS,
               VERSION_PAIRING, HEADER_BYTES };
root.SMAF = SMAF;
if (typeof module !== "undefined" && module.exports) module.exports = SMAF;

})(typeof globalThis !== "undefined" ? globalThis : this);
