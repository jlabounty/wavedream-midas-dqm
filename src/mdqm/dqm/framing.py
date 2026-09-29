"""The binary wire format between the analyzer and the browser.

Deliberately byte-compatible with musip's DQM, whose JavaScript decoder
(``musip/custom/onlineDQM.js``) this was written against. That costs nothing --
the format is fully specified by the decoder we would have had to read anyway --
and buys two things: musip's page can drive our analyzer unmodified, and our
generic browser page can drive theirs.

Everything is little-endian.

Envelope
--------
Every reply is an 8-byte header followed by a payload::

    u32  total size, including these 8 bytes
    char tag[4]                      e.g. b"hist", b"list", b"json"

Eight bytes precisely so the payload starts 8-aligned and a ``float64`` in it
cannot be misaligned.

    A trap worth knowing: musip's C++ enum stores ``hist`` as 0x74736968 while
    its JavaScript reads the four bytes big-endian and compares 0x68697374. Same
    bytes, opposite integer conventions. Write ``b"hist"`` and never think about
    it in integers.

Histogram payload
-----------------
::

    u8   version = 1
    u8   type                    index into musip's object_type variant
    u8   dimensions
    u8   abscissa_size[dimensions]   float width of the axis edges, 4 or 8
    u8   ordinate_size               bin content width, 4 or 8
    <align 4>
    u32  n_bins[dimensions]          NOT counting under/overflow
    <align abscissa_size, per axis>
    f    low_edge, high_edge         per axis
    <align 8>
    u64  entries
    <align ordinate_size>
    data[prod(n_bins[d] + 2)]        under- and overflow included, x fastest

The alignment is relative to the start of the *payload*, because the page slices
the envelope off (``rpc.slice(8)``) before handing the rest to a typed-array
constructor, which throws on a misaligned offset.
"""

from __future__ import annotations

import struct

import numpy as np

HEADER = struct.Struct("<I4s")
HEADER_SIZE = HEADER.size

# Tags. The first three are musip's; the rest are ours.
TAG_LIST = b"list"
TAG_HIST = b"hist"
TAG_META = b"meta"
TAG_JSON = b"json"
TAG_SCOPE = b"scop"
TAG_SMAF = b"smaf"                      # one SMA readout frame; see plugins/sma.py
TAG_MEVT = b"mevt"                      # one raw MIDAS event (header + banks), a .mid file
TAG_ERROR = b"err "

#: Index into musip's ``PlotCollection::object_type`` variant. Only the ones we
#: emit are named; the numbering is theirs and must not be renumbered.
TYPE_1D_F32 = 0
TYPE_1D_F64 = 1
TYPE_2D_F32 = 2
TYPE_1D_U32 = 3
TYPE_2D_U32 = 4
TYPE_2D_F64 = 6

_DTYPE_FOR_TYPE = {
    TYPE_1D_F32: np.float32,
    TYPE_1D_F64: np.float64,
    TYPE_2D_F32: np.float32,
    TYPE_1D_U32: np.uint32,
    TYPE_2D_U32: np.uint32,
    TYPE_2D_F64: np.float64,
}


def _align(buf: bytearray, alignment: int) -> None:
    """Pad to the next multiple of `alignment`, counting from the payload start."""
    remainder = len(buf) % alignment
    if remainder:
        buf.extend(b"\x00" * (alignment - remainder))


def envelope(tag: bytes, payload: bytes) -> bytes:
    """Wrap a payload in the 8-byte header the page expects."""
    if len(tag) != 4:
        raise ValueError(f"tag must be exactly 4 bytes, got {tag!r}")
    return HEADER.pack(HEADER_SIZE + len(payload), tag) + payload


def parse_envelope(blob: bytes) -> tuple[int, bytes, bytes]:
    """Inverse of `envelope`, for tests and for talking to another analyzer."""
    if len(blob) < HEADER_SIZE:
        raise ValueError(f"short read: {len(blob)} bytes cannot hold an 8-byte header")
    size, tag = HEADER.unpack_from(blob, 0)
    return size, tag, blob[HEADER_SIZE:size]


def encode_histogram(
    counts: np.ndarray,
    edges: list[tuple[float, float]],
    entries: int,
    *,
    ordinate_size: int = 4,
    abscissa_size: int = 8,
) -> bytes:
    """Encode a histogram, under/overflow bins included.

    `counts` has shape (nx+2,) or (ny+2, nx+2) -- the +2 per axis is the
    under/overflow pair, which the format carries and the page strips.
    """
    dims = counts.ndim
    if dims not in (1, 2):
        raise ValueError(f"only 1D and 2D histograms are encodable, got {dims}D")
    if len(edges) != dims:
        raise ValueError(f"{dims}D histogram needs {dims} edge pairs, got {len(edges)}")

    if np.issubdtype(counts.dtype, np.integer) and counts.dtype.itemsize > 4:
        # The format's only integer ordinate is u32. Wider counters travel as
        # f64, which is exact to 2**53 and which every decoder reads by its
        # ordinate size alone; narrowing to u32 would wrap silently instead.
        counts = counts.astype(np.float64)
        ordinate_size = 8

    is_int = np.issubdtype(counts.dtype, np.integer)
    if dims == 1:
        htype = TYPE_1D_U32 if is_int else (TYPE_1D_F32 if ordinate_size == 4 else TYPE_1D_F64)
    else:
        htype = TYPE_2D_U32 if is_int else (TYPE_2D_F32 if ordinate_size == 4 else TYPE_2D_F64)
    if is_int and ordinate_size != 4:
        raise ValueError("integer histograms are u32; ordinate_size must be 4")

    # numberOfBins excludes under/overflow; counts includes it.
    n_bins = [n - 2 for n in reversed(counts.shape)]   # x first, as the format wants
    if any(n < 1 for n in n_bins):
        raise ValueError(f"counts shape {counts.shape} is too small for under/overflow")

    buf = bytearray()
    buf.append(1)                       # version
    buf.append(htype)
    buf.append(dims)
    for _ in range(dims):
        buf.append(abscissa_size)
    buf.append(ordinate_size)

    _align(buf, 4)
    for n in n_bins:
        buf.extend(struct.pack("<I", n))

    edge_fmt = "<d" if abscissa_size == 8 else "<f"
    for lo, hi in edges:
        _align(buf, abscissa_size)
        buf.extend(struct.pack(edge_fmt, float(lo)))
        buf.extend(struct.pack(edge_fmt, float(hi)))

    _align(buf, 8)
    buf.extend(struct.pack("<Q", int(entries)))

    _align(buf, ordinate_size)
    # Row-major with x fastest, which is what the page's
    # `zData[j + i*nx]` indexing expects and what C++ writes.
    buf.extend(np.ascontiguousarray(counts, dtype=_DTYPE_FOR_TYPE[htype]).tobytes())
    return bytes(buf)


def decode_histogram(payload: bytes) -> dict:
    """Decode `encode_histogram`'s output. A Python mirror of the JS decoder.

    Exists so the round trip can be asserted without a JavaScript engine; the
    cross-language test additionally checks the bytes against musip's own
    decoder, which is what makes the compatibility claim real rather than
    self-referential.
    """
    off = 0

    def align(to: int) -> None:
        nonlocal off
        rem = off % to
        if rem:
            off += to - rem

    version = payload[off]
    off += 1
    if version != 1:
        raise ValueError(f"unknown histogram version {version}")
    htype = payload[off]
    off += 1
    dims = payload[off]
    off += 1
    abscissa = [payload[off + i] for i in range(dims)]
    off += dims
    ordinate = payload[off]
    off += 1

    align(4)
    n_bins = list(struct.unpack_from(f"<{dims}I", payload, off))
    off += 4 * dims

    lo_edges, hi_edges = [], []
    for size in abscissa:
        align(size)
        fmt = "<d" if size == 8 else "<f"
        lo_edges.append(struct.unpack_from(fmt, payload, off)[0])
        off += size
        hi_edges.append(struct.unpack_from(fmt, payload, off)[0])
        off += size

    align(8)
    entries = struct.unpack_from("<Q", payload, off)[0]
    off += 8

    align(ordinate)
    total = 1
    for n in n_bins:
        total *= n + 2
    data = np.frombuffer(payload, dtype=_DTYPE_FOR_TYPE[htype], count=total, offset=off)
    shape = tuple(n + 2 for n in reversed(n_bins))
    return {
        "type": htype,
        "n_bins": n_bins,
        "low_edge": lo_edges,
        "high_edge": hi_edges,
        "entries": entries,
        "counts": data.reshape(shape),
    }


# ---------------------------------------------------------------------------
# Scope frames
# ---------------------------------------------------------------------------
#
# One triggered event: every channel's samples plus the quantities derived from
# *that* event, in a single reply.
#
# The single reply is the whole point. An event display exists to be pointed at
# -- "channel 3 looks odd on this one" -- so the traces on screen and the phase
# printed beside them have to come from the same event, and every screen has to
# be showing the same event as every other. Both are guaranteed by construction
# if there is one frame, and by nothing at all if the page assembles traces from
# one source and numbers from another.
#
# Samples travel as the native int16 they came off the wire as, with the volts
# scale in the channel header. That is 32 kB for sixteen channels against about
# 150 kB of JSON, and the browser does one multiply while copying into a
# Float32Array instead of parsing.

SCOPE_HEADER = struct.Struct("<IIQIIIIQffIIII")
"""64 bytes: version, nChannels, frameSeq, run, event, trigger, triggerType,
timestampTicks, boardTempC, nominalPs, flags, nDerived, boardId, reserved."""

CHANNEL_HEADER = struct.Struct("<HHHBBfI")
"""16 bytes: channel, firstBin, nSamples, encoding, decoded, scale, pad."""

DERIVED_ENTRY = struct.Struct("<16sd")
"""24 bytes: a NUL-padded name and a float64."""

SCOPE_VERSION = 1

#: `flags` bits.
SCOPE_HAVE_WIDTHS = 1 << 0
SCOPE_RUN_ACTIVE = 1 << 1
SCOPE_WIDTHS_CACHED = 1 << 2

VOLTS_SCALE = 1e-4
"""Encoding mode 0: sample * this = volts. Matches wdunpack's VOLTAGE_SCALE."""


def encode_scope_frame(
    channels: list[dict],
    *,
    frame_seq: int = 0,
    run_number: int = 0,
    event_number: int = 0,
    trigger_number: int = 0,
    trigger_type: int = 0,
    timestamp_ticks: int = 0,
    board_temp_c: float = 0.0,
    nominal_ps: float = 0.0,
    board_id: int = 0,
    have_widths: bool = False,
    widths_cached: bool = False,
    run_active: bool = False,
    derived: dict[str, float] | None = None,
) -> bytes:
    """Encode one event.

    Each entry of `channels` is
    ``{"channel", "first_bin", "samples" (int16 array or None), "encoding"}``;
    a `samples` of None means the channel could not be decoded, which is
    carried through rather than dropped so the page can grey that panel out
    instead of silently omitting it.
    """
    derived = derived or {}
    flags = 0
    if have_widths:
        flags |= SCOPE_HAVE_WIDTHS
    if run_active:
        flags |= SCOPE_RUN_ACTIVE
    if widths_cached:
        flags |= SCOPE_WIDTHS_CACHED

    buf = bytearray()
    buf.extend(SCOPE_HEADER.pack(
        SCOPE_VERSION, len(channels), int(frame_seq),
        int(run_number), int(event_number),
        int(trigger_number), int(trigger_type),
        int(timestamp_ticks),
        float(board_temp_c), float(nominal_ps),
        flags, len(derived), int(board_id), 0))

    for ch in channels:
        samples = ch.get("samples")
        decoded = samples is not None
        arr = np.ascontiguousarray(samples, dtype="<i2") if decoded else np.empty(0, "<i2")
        buf.extend(CHANNEL_HEADER.pack(
            int(ch["channel"]) & 0xFFFF,
            int(ch.get("first_bin", 0)) & 0xFFFF,
            int(arr.size) & 0xFFFF,
            int(ch.get("encoding", 0)) & 0xFF,
            1 if decoded else 0,
            float(ch.get("scale", VOLTS_SCALE)),
            0))
        buf.extend(arr.tobytes())
        _align(buf, 8)

    for name, value in derived.items():
        buf.extend(DERIVED_ENTRY.pack(name.encode()[:16], float(value)))

    return bytes(buf)


def decode_scope_frame(payload: bytes) -> dict:
    """Decode `encode_scope_frame`. A Python mirror, for tests and for reuse."""
    (version, n_channels, frame_seq, run_number, event_number,
     trigger_number, trigger_type, timestamp_ticks,
     board_temp_c, nominal_ps, flags, n_derived, board_id,
     _reserved) = SCOPE_HEADER.unpack_from(payload, 0)
    if version != SCOPE_VERSION:
        raise ValueError(f"unknown scope frame version {version}")

    off = SCOPE_HEADER.size
    channels = []
    for _ in range(n_channels):
        (channel, first_bin, n_samples, encoding, decoded, scale,
         _pad) = CHANNEL_HEADER.unpack_from(payload, off)
        off += CHANNEL_HEADER.size
        samples = None
        if decoded:
            samples = np.frombuffer(payload, dtype="<i2", count=n_samples, offset=off)
        off += n_samples * 2
        rem = off % 8
        if rem:
            off += 8 - rem
        channels.append({
            "channel": channel, "first_bin": first_bin, "encoding": encoding,
            "decoded": bool(decoded), "scale": scale, "samples": samples,
        })

    derived = {}
    for _ in range(n_derived):
        raw_name, value = DERIVED_ENTRY.unpack_from(payload, off)
        off += DERIVED_ENTRY.size
        derived[raw_name.rstrip(b"\x00").decode()] = value

    return {
        "frame_seq": frame_seq, "run_number": run_number,
        "event_number": event_number, "trigger_number": trigger_number,
        "trigger_type": trigger_type, "timestamp_ticks": timestamp_ticks,
        "board_temp_c": board_temp_c, "nominal_ps": nominal_ps,
        "board_id": board_id,
        "have_widths": bool(flags & SCOPE_HAVE_WIDTHS),
        "run_active": bool(flags & SCOPE_RUN_ACTIVE),
        "widths_cached": bool(flags & SCOPE_WIDTHS_CACHED),
        "channels": channels, "derived": derived,
    }


# ---------------------------------------------------------------------------
# SMA frames (tag b"smaf")
# ---------------------------------------------------------------------------
#
# One SMA readout frame for the event display: a fixed header, a JSON block
# with everything that is per frame or per seed (small, and changes shape as the
# display grows), then one entry per hit in four flat arrays. The hits dominate
# -- a 40000-word frame is ~33k hits -- so they travel as 7 bytes each instead
# of ~40 bytes of JSON, and the page wraps them in typed arrays without parsing.
#
# Layout, offsets relative to the payload start (the envelope sliced off, as
# for every other tag; JS typed arrays throw on a misaligned offset)::
#
#     0   u8   version = 1
#     1   u8   flags        SMAF_SEEDED | SMAF_STALE | SMAF_TRUNCATED | SMAF_SUSPECT
#     2   u16  time_shift   k: a hit's time is meta["t0_ns"] + t_rel_ns * 2^k
#     4   u32  n_hits       n
#     8   u64  frame_seq    the plugin's frame counter (the cache/freeze key)
#     16  u32  run          run number, 0 when unknown
#     20  u32  json_len     bytes of UTF-8 JSON (ASCII in practice)
#     24  u64  reserved = 0
#     32  JSON metadata, json_len bytes, then NUL padding to a multiple of 8
#     A = 32 + ceil8(json_len)
#     A        u32 t_rel_ns[n]   (hit time - meta["t0_ns"]) >> k; ascending
#     A + 4n   u8  ch[n]
#     A + 5n   u8  tot[n]
#     A + 6n   u8  hit_flags[n]  HIT_* bits below
#     total = A + 7n
#
# ``t_rel_ns`` starts 8-aligned and is the only multi-byte array, which is why
# it comes first; the byte arrays need no alignment. ``t0_ns`` is the first
# *shipped* hit. A u32 of ns covers 4.29 s, and real frames with a beam trip in
# them span more (5-6 s were measured), so the encoder picks the smallest
# ``time_shift`` k that fits the shipped span: k = 0, exact ns, for every ordinary
# frame; otherwise the times lose their k low bits. In JavaScript multiply by
# ``2 ** k`` -- ``<<`` works on 32-bit signed integers and would overflow.
# Seed times in the JSON (``seeds[].t_rel``) are ns from ``t0_ns``, never shifted.
#
# Version 2 (header flag SMAF_WORDS, version byte 2) adds, per hit, the raw
# 64-bit word as received and its index in the H000 bank (counted in 64-bit
# words, filler and pixel words included), so a hit on the page can be found
# again in the file. The u64 array goes first for alignment::
#
#     A        u64 raw_word[n]
#     A + 8n   u32 t_rel_ns[n]
#     A + 12n  u32 word_index[n]
#     A + 16n  u8  ch[n];  A + 17n u8 tot[n];  A + 18n u8 hit_flags[n]
#     total = A + 19n
#
# Without word data the payload is exactly version 1 (7 bytes a hit). The
# seeded view is always v2 (a few hundred hits); the raster only on request
# (``{"words": true}``): 19 instead of 7 bytes a hit is +170 %, 0.6 MB instead
# of 0.23 MB for a 40000-word frame at 2 Hz, so the page asks for it once, for a
# frozen frame.
#
# The pixel block (header flag SMAF_PIXELS) carries the frame's MuPix pixel hits
# after the trigger-hit arrays, in either version. It is an addition at the end
# that the version byte does not announce, so a v1/v2 decoder that knows nothing
# of it reads the same trigger hits and ignores the tail. P is the end of the
# hit arrays rounded up to a multiple of 8::
#
#     P       u32 n_pix       m
#     P + 4   u16 pix_shift   kp: a pixel's time is meta["mupix"]["t0_ns"] + t * 2^kp
#     P + 6   u8  block version = 1
#     P + 7   u8  block flags  PIXB_WORDS: raw words and word indices follow
#     Q = P + 8
#     without words                    with words (PIXB_WORDS)
#     Q        u32 t_rel[m]            Q         u64 raw_word[m]
#     Q + 4m   u8  chip[m]             Q + 8m    u32 t_rel[m]
#     Q + 5m   u8  col[m]              Q + 12m   u32 word_index[m]
#     Q + 6m   u8  row[m]              Q + 16m   u8 chip, col, row, tot, flags [m each]
#     Q + 7m   u8  tot[m]
#     Q + 8m   u8  pix_flags[m]
#     9 bytes a pixel hit              21 bytes a pixel hit
#
# ``tot`` is the real ToT (0-31, meta["mupix"]["tot_ns"] ns a count), not the
# raw TS2 field. ``pix_flags``: bits 1:0 the plane (PIX_PLANE_MASK: 0 none, 1
# L1, 2 L2), PIX_OFF_SENSOR (row >= 250), PIX_IN_SEED. t_rel ascending. The
# pixel words carry the same word indices as the trigger words (one bank).

SMAF_HEADER = struct.Struct("<BBHIQIIQ")
"""32 bytes: version, flags, timeShift, nHits, frameSeq, run, jsonLen, reserved."""

SMAF_VERSION = 1
SMAF_VERSION_WORDS = 2

#: Header `flags` bits.
SMAF_SEEDED = 1 << 0        # the seeded view (hits inside seed windows only); else raster
SMAF_STALE = 1 << 1         # the frame was classified stale (replayed/old buffer)
SMAF_TRUNCATED = 1 << 2     # more hits than max_hits; the latest ones were kept
SMAF_SUSPECT = 1 << 3       # no usable time base (most hits outside the time clusters)
SMAF_WORDS = 1 << 4         # v2: raw words and word indices follow (see above)
SMAF_PIXELS = 1 << 5        # a MuPix pixel block follows the hit arrays (see above)

#: Pixel block flags and version.
PIXB_VERSION = 1
PIXB_WORDS = 1 << 0
PIXB_HEADER = struct.Struct("<IHBB")
"""8 bytes: n_pix, pix_shift, block version, block flags."""

#: Per-pixel flag bits.
PIX_PLANE_MASK = 0x3        # bits 1:0: 0 no plane, 1 L1, 2 L2
PIX_OFF_SENSOR = 1 << 2     # row >= 250: not a row of the sensor
PIX_IN_SEED = 1 << 3        # inside at least one seed window

#: Per-hit flag bits.
HIT_MISMATCH = 1 << 0       # fine and coarse disagree beyond one coarse tick + margin
HIT_TOT_CORRUPT = 1 << 1    # ToT >= the corrupt threshold (254/255 markers)
HIT_FINE_LSB = 1 << 2       # fine bit 0 set
HIT_IN_SEED = 1 << 3        # inside at least one seed window
HIT_STALE = 1 << 4          # the hit belongs to a stale frame


def encode_sma_frame(
    meta: dict,
    t_rel_ns,
    ch,
    tot,
    hit_flags,
    *,
    frame_seq: int = 0,
    run_number: int = 0,
    seeded: bool = False,
    stale: bool = False,
    truncated: bool = False,
    suspect: bool = False,
    time_shift: int = 0,
    raw_words=None,
    word_index=None,
    pixels: dict | None = None,
) -> bytes:
    """Encode one SMA frame; the arrays must all have the same length.

    With `raw_words` and `word_index` (both or neither) the payload is version
    2 (`SMAF_WORDS`), else version 1. `pixels`: the pixel block, ``{t_rel,
    time_shift, chip, col, row, tot, flags}`` and optionally ``raw_words`` and
    ``word_index`` (both or neither), all of one length (`SMAF_PIXELS`).

    `meta` must be JSON-serialisable without NaN (JavaScript's ``JSON.parse``
    rejects the ``NaN`` Python would write); the plugin maps NaN to null.
    """
    import json

    t = np.ascontiguousarray(t_rel_ns, dtype="<u4")
    c = np.ascontiguousarray(ch, dtype=np.uint8)
    k = np.ascontiguousarray(tot, dtype=np.uint8)
    f = np.ascontiguousarray(hit_flags, dtype=np.uint8)
    n = t.size
    if not (c.size == k.size == f.size == n):
        raise ValueError(f"hit arrays differ in length: {n}, {c.size}, {k.size}, {f.size}")
    blob = json.dumps(meta, separators=(",", ":"), allow_nan=False).encode()
    words = (raw_words is not None) or (word_index is not None)
    if words:
        if raw_words is None or word_index is None:
            raise ValueError("raw_words and word_index go together")
        rw = np.ascontiguousarray(raw_words, dtype="<u8")
        wi = np.ascontiguousarray(word_index, dtype="<u4")
        if not rw.size == wi.size == n:
            raise ValueError(f"word arrays differ in length: {n}, {rw.size}, {wi.size}")

    if not 0 <= int(time_shift) <= 32:
        raise ValueError(f"time_shift must be 0..32, got {time_shift}")
    flags = ((SMAF_SEEDED if seeded else 0) | (SMAF_STALE if stale else 0)
             | (SMAF_TRUNCATED if truncated else 0) | (SMAF_SUSPECT if suspect else 0)
             | (SMAF_WORDS if words else 0) | (SMAF_PIXELS if pixels is not None else 0))
    version = SMAF_VERSION_WORDS if words else SMAF_VERSION
    buf = bytearray(SMAF_HEADER.pack(version, flags, int(time_shift), n, int(frame_seq),
                                     int(run_number) & 0xFFFFFFFF, len(blob), 0))
    buf.extend(blob)
    _align(buf, 8)
    if words:
        buf.extend(rw.tobytes())
        buf.extend(t.tobytes())
        buf.extend(wi.tobytes())
    else:
        buf.extend(t.tobytes())
    buf.extend(c.tobytes())
    buf.extend(k.tobytes())
    buf.extend(f.tobytes())
    if pixels is not None:
        _align(buf, 8)
        buf.extend(_pixel_block(pixels))
    return bytes(buf)


def _pixel_block(px: dict) -> bytes:
    t = np.ascontiguousarray(px["t_rel"], dtype="<u4")
    m = t.size
    cols = [np.ascontiguousarray(px[k], dtype=np.uint8) for k in ("chip", "col", "row", "tot",
                                                                   "flags")]
    if any(x.size != m for x in cols):
        raise ValueError(f"pixel arrays differ in length: {m}, {[x.size for x in cols]}")
    has_rw, has_wi = px.get("raw_words") is not None, px.get("word_index") is not None
    if has_rw != has_wi:
        raise ValueError("pixel raw_words and word_index go together")
    shift = int(px.get("time_shift", 0))
    if not 0 <= shift <= 32:
        raise ValueError(f"pixel time_shift must be 0..32, got {shift}")
    out = bytearray(PIXB_HEADER.pack(m, shift, PIXB_VERSION, PIXB_WORDS if has_rw else 0))
    if has_rw:
        rw = np.ascontiguousarray(px["raw_words"], dtype="<u8")
        wi = np.ascontiguousarray(px["word_index"], dtype="<u4")
        if not rw.size == wi.size == m:
            raise ValueError(f"pixel word arrays differ in length: {m}, {rw.size}, {wi.size}")
        out.extend(rw.tobytes())
        out.extend(t.tobytes())
        out.extend(wi.tobytes())
    else:
        out.extend(t.tobytes())
    for x in cols:
        out.extend(x.tobytes())
    return bytes(out)


def _pixel_block_offset(payload: bytes) -> tuple[int, int] | None:
    """``(start of the pixel block, end of the hit arrays)``; None without a block."""
    (version, flags, _ts, n, _seq, _run, json_len, _r) = SMAF_HEADER.unpack_from(payload, 0)
    if not flags & SMAF_PIXELS:
        return None
    off = SMAF_HEADER.size + json_len
    off += (-off) % 8
    end = off + (19 if version == SMAF_VERSION_WORDS else 7) * n
    return end + (-end) % 8, end


def smaf_drop_pixels(payload: bytes) -> bytes:
    """`payload` without its pixel block: the flag cleared and the block cut off.

    With the ``mupix`` meta key dropped too (``smaf_drop_meta``), a payload made
    with pixels is byte for byte the one made without.
    """
    at = _pixel_block_offset(payload)
    if at is None:
        return bytes(payload)
    buf = bytearray(payload[:at[1]])
    buf[1] &= ~SMAF_PIXELS & 0xFF
    return bytes(buf)


def decode_pixel_block(payload: bytes, at: int) -> dict:
    """The pixel block at payload offset `at` (a Python mirror of the page's decoder)."""
    if len(payload) < at + PIXB_HEADER.size:
        raise ValueError(f"short smaf payload: no room for the pixel block at {at}")
    m, shift, version, bflags = PIXB_HEADER.unpack_from(payload, at)
    if version != PIXB_VERSION:
        raise ValueError(f"unknown smaf pixel block version {version}")
    words = bool(bflags & PIXB_WORDS)
    q = at + PIXB_HEADER.size
    need = (21 if words else 9) * m
    if len(payload) < q + need:
        raise ValueError(f"short smaf pixel block: {len(payload) - q} bytes for {m} pixel hits")
    rw = wi = None
    if words:
        rw = np.frombuffer(payload, dtype="<u8", count=m, offset=q)
        t = np.frombuffer(payload, dtype="<u4", count=m, offset=q + 8 * m)
        wi = np.frombuffer(payload, dtype="<u4", count=m, offset=q + 12 * m)
        b0 = q + 16 * m
    else:
        t = np.frombuffer(payload, dtype="<u4", count=m, offset=q)
        b0 = q + 4 * m
    cols = {k: np.frombuffer(payload, dtype=np.uint8, count=m, offset=b0 + j * m)
            for j, k in enumerate(("chip", "col", "row", "tot", "flags"))}
    return {"n": m, "time_shift": shift, "offset": at, "words": words, "t_rel": t,
            "raw_words": rw, "word_index": wi, **cols}


def smaf_update_meta(payload: bytes, extra: dict) -> bytes:
    """`payload` with `extra` merged into its JSON metadata; the hit arrays untouched.

    The binary layout is unchanged: only ``json_len`` and the NUL padding move,
    and the arrays are copied as they are. New keys go after the existing ones,
    so ``smaf_update_meta(smaf_update_meta(p, {"k": v}), {})`` with "k" removed
    again gives back `p` byte for byte (see ``smaf_drop_meta``). Used for what
    changes per request (the seeded view's staleness) on a payload encoded once.
    """
    return _smaf_rewrite_meta(payload, lambda meta: meta.update(extra))


def smaf_drop_meta(payload: bytes, keys) -> bytes:
    """`payload` without the metadata `keys` (the inverse of `smaf_update_meta`)."""
    def drop(meta):
        for k in keys:
            meta.pop(k, None)
    return _smaf_rewrite_meta(payload, drop)


def _smaf_rewrite_meta(payload: bytes, edit) -> bytes:
    import json

    head = SMAF_HEADER.unpack_from(payload, 0)
    json_len = head[6]
    off = SMAF_HEADER.size
    meta = json.loads(bytes(payload[off:off + json_len]).decode())
    arrays = off + json_len
    arrays += (-arrays) % 8
    edit(meta)
    blob = json.dumps(meta, separators=(",", ":"), allow_nan=False).encode()
    buf = bytearray(SMAF_HEADER.pack(*head[:6], len(blob), head[7]))
    buf.extend(blob)
    _align(buf, 8)
    buf.extend(memoryview(payload)[arrays:])
    return bytes(buf)


def decode_sma_frame(payload: bytes) -> dict:
    """Decode `encode_sma_frame`. A Python mirror of the page's decoder."""
    import json

    (version, flags, time_shift, n, frame_seq, run_number, json_len,
     _r1) = SMAF_HEADER.unpack_from(payload, 0)
    if version not in (SMAF_VERSION, SMAF_VERSION_WORDS):
        raise ValueError(f"unknown smaf version {version}")
    words = version == SMAF_VERSION_WORDS
    if words != bool(flags & SMAF_WORDS):
        raise ValueError(f"smaf version {version} with flags {flags:#x}")
    off = SMAF_HEADER.size
    meta = json.loads(bytes(payload[off:off + json_len]).decode())
    off += json_len
    rem = off % 8
    if rem:
        off += 8 - rem
    per_hit = 19 if words else 7
    if len(payload) < off + per_hit * n:
        raise ValueError(f"short smaf payload: {len(payload)} bytes for {n} hits at {off}")
    rw = wi = None
    if words:
        rw = np.frombuffer(payload, dtype="<u8", count=n, offset=off)
        t = np.frombuffer(payload, dtype="<u4", count=n, offset=off + 8 * n)
        wi = np.frombuffer(payload, dtype="<u4", count=n, offset=off + 12 * n)
        b0 = off + 16 * n
    else:
        t = np.frombuffer(payload, dtype="<u4", count=n, offset=off)
        b0 = off + 4 * n
    c = np.frombuffer(payload, dtype=np.uint8, count=n, offset=b0)
    k = np.frombuffer(payload, dtype=np.uint8, count=n, offset=b0 + n)
    f = np.frombuffer(payload, dtype=np.uint8, count=n, offset=b0 + 2 * n)
    pix = None
    if flags & SMAF_PIXELS:
        end = b0 + 3 * n
        pix = decode_pixel_block(payload, end + (-end) % 8)
    return {
        "version": version, "flags": flags, "n_hits": n, "frame_seq": frame_seq,
        "run_number": run_number, "json_len": json_len, "arrays_offset": off,
        "seeded": bool(flags & SMAF_SEEDED), "stale": bool(flags & SMAF_STALE),
        "truncated": bool(flags & SMAF_TRUNCATED),
        "suspect": bool(flags & SMAF_SUSPECT), "time_shift": time_shift,
        "meta": meta, "t_rel_ns": t, "ch": c, "tot": k, "hit_flags": f,
        "words": words, "raw_words": rw, "word_index": wi, "pixels": pix,
    }
