//
// dqm-smaframe.js -- the SMA event-display frame ("smaf" v1 and v2), decoded.
//
// Mirror of mdqm/dqm/framing.py's encode_sma_frame. Keep the two in step: the
// cross-language test in tests/js/smaframe.test.js decodes Python's bytes with
// this function, so a divergence fails a test rather than misdrawing a hit on
// the wrong channel -- which on an event display is the worst kind of bug,
// because it looks exactly like a detector problem.
//
// Layout (offsets from the payload start, envelope already removed):
//
//     0   u8   version = 1, or 2 with per-hit words (flag WORDS)
//     1   u8   flags        SEEDED | STALE | TRUNCATED | SUSPECT | WORDS
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
// hit time = meta.t0_ns + t_rel * 2^k. raw_word is the 64-bit word as the
// board sent it and word_index its position in the H000 bank (in 64-bit words,
// filler and pixel words included): what finds the hit again in the file.
//
// The hit arrays are wrapped as typed-array *views* rather than copied or
// parsed: a whole-frame raster is ~33k hits, and the whole point of the binary
// format is that the page does not touch each one before drawing it.
//

(function (root) {
"use strict";

const VERSION = 1;
const VERSION_WORDS = 2;
const HEADER_BYTES = 32;

const FLAGS = { SEEDED: 1 << 0, STALE: 1 << 1, TRUNCATED: 1 << 2, SUSPECT: 1 << 3, WORDS: 1 << 4 };
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
 * Decode one smaf payload, version 1 or 2.
 *
 * Accepts an ArrayBuffer or any ArrayBufferView (a Node Buffer, a Uint8Array
 * slice of a bigger reply). A view can start at any byte offset, and a
 * Uint32Array (BigUint64Array) over an offset that is not a multiple of 4 (8)
 * throws -- so a multi-byte array that does not land on its boundary in the
 * underlying buffer is copied out. BRPC.call hands over a fresh slice starting
 * at 0, where the arrays are aligned by construction and nothing is copied.
 *
 * `rawWord` (BigUint64Array) and `wordIndex` (Uint32Array) are null in v1.
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
  if (version !== VERSION && version !== VERSION_WORDS) {
    throw new Error(`unknown smaf version ${version}`);
  }
  const flags = dv.getUint8(1);
  const words = version === VERSION_WORDS;
  if (words !== !!(flags & FLAGS.WORDS)) {
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
  const perHit = words ? 19 : 7;
  if (length < at + perHit * nHits) {
    throw new Error(`short smaf payload: ${length} bytes for ${nHits} hits at ${at}`);
  }

  // Payload offsets of each array (the tests poke bytes through these).
  const offsets = words
    ? { rawWord: at, t: at + 8 * nHits, wordIndex: at + 12 * nHits, ch: at + 16 * nHits }
    : { rawWord: null, t: at, wordIndex: null, ch: at + 4 * nHits };
  offsets.tot = offsets.ch + nHits;
  offsets.hitFlags = offsets.ch + 2 * nHits;

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

  return {
    version, flags, timeShift, nHits, frameSeq, run, jsonLen, arraysOffset: at, offsets,
    seeded: !!(flags & FLAGS.SEEDED),
    stale: !!(flags & FLAGS.STALE),
    truncated: !!(flags & FLAGS.TRUNCATED),
    // No usable time base: most hits fell outside the time clusters, which
    // usually means the coarse shift is wrong rather than the board.
    suspect: !!(flags & FLAGS.SUSPECT),
    words,
    meta, t, tRaw, ch, tot, hitFlags, rawWord, wordIndex,
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

const SMAF = { decode, fetchFrame, fetchRaw, hex64, FLAGS, HIT, VERSION, VERSION_WORDS, HEADER_BYTES };
root.SMAF = SMAF;
if (typeof module !== "undefined" && module.exports) module.exports = SMAF;

})(typeof globalThis !== "undefined" ? globalThis : this);
