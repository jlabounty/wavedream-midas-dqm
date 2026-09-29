"""A small MIDAS-free reader for ``.mid``, ``.mid.lz4`` and ``.mid.gz`` files.

The analyzer receives its events from ``midas.client`` (``receive_event(...,
use_numpy=True)``); the offline tools read the same events from a file. This
module hands over objects with the same attributes the plugins use, so a plugin
cannot tell the two apart:

* ``event.header.event_id``, ``trigger_mask``, ``serial_number``, ``timestamp``,
  ``event_data_size_bytes`` and ``is_midas_internal_event()``;
* ``event.banks`` (name -> `Bank`) and ``event.get_bank(name)``, where
  ``bank.data`` is a numpy array of the bank's TID type (raw ``bytes`` for the
  types MIDAS itself leaves raw), exactly as ``midas.event.Bank`` with
  ``use_numpy=True``.

No MIDAS installation is needed; ``lz4`` is needed only for ``.lz4`` files.

File format (MIDAS wiki, "Event Structure"): a sequence of events, each a
16-byte header ``<HHIII`` (event id, trigger mask, serial, unix time, data
size) and ``data size`` bytes of body. A body starts with an 8-byte bank header
``<II`` (size of all banks, flags) and then the banks:

==========  =====  ================================  ============
format      flags  bank header                       header bytes
==========  =====  ================================  ============
bank16      0x01   name[4], type u16, size u16       8
bank32      0x11   name[4], type u32, size u32       12
bank32a     0x31   name[4], type u32, size u32, pad  16
==========  =====  ================================  ============

Every bank's data is padded to a multiple of 8 bytes (``ALIGN8`` in
``bk_iterate``); bank32a additionally pads the header to 16 bytes so the data
itself is 8-byte aligned. The begin-of-run, end-of-run and message events
(ids 0x8000, 0x8001, 0x8002) carry an ODB dump or a message instead of banks;
mlogger writes the run number into the serial of the begin-of-run event.
"""

from __future__ import annotations

import gzip
import re
import struct
from pathlib import Path

import numpy as np

EVENT_HEADER = struct.Struct("<HHIII")
BANK_HEADER = struct.Struct("<II")
BANK16 = struct.Struct("<4sHH")
BANK32 = struct.Struct("<4sII")

EVID_BOR = 0x8000
EVID_EOR = 0x8001
EVID_MSG = 0x8002
INTERNAL_IDS = frozenset({EVID_BOR, EVID_EOR, EVID_MSG})

FLAG_32BIT = 0x10
FLAG_ALIGN64 = 0x20

#: A MIDAS event is at most a few hundred MB (MAX_EVENT_SIZE); anything larger
#: means we are not reading a MIDAS file, or reading one at the wrong offset.
MAX_EVENT_BYTES = 1 << 30

#: TID -> numpy dtype, as ``midas.tid_np_formats`` (None: raw bytes).
TID_DTYPES: dict[int, np.dtype | None] = {
    1: np.dtype("<u1"),    # TID_BYTE / UINT8
    2: np.dtype("<i1"),    # TID_SBYTE / INT8
    3: np.dtype("<u1"),    # TID_CHAR
    4: np.dtype("<u2"),    # TID_WORD / UINT16
    5: np.dtype("<i2"),    # TID_SHORT / INT16
    6: np.dtype("<u4"),    # TID_DWORD / UINT32
    7: np.dtype("<i4"),    # TID_INT / INT32
    8: np.dtype("<u4"),    # TID_BOOL (converted to bool, as MIDAS does)
    9: np.dtype("<f4"),    # TID_FLOAT
    10: np.dtype("<f8"),   # TID_DOUBLE
    11: np.dtype("<u4"),   # TID_BITFIELD
    17: np.dtype("<i8"),   # TID_INT64
    18: np.dtype("<u8"),   # TID_QWORD / UINT64
}
TID_BOOL = 8


class EventHeader:
    """The 16-byte event header; attribute names as ``midas.event.EventHeader``."""

    __slots__ = ("event_id", "trigger_mask", "serial_number", "timestamp",
                 "event_data_size_bytes")

    def __init__(self, event_id, trigger_mask, serial_number, timestamp, event_data_size_bytes):
        self.event_id = event_id
        self.trigger_mask = trigger_mask
        self.serial_number = serial_number
        self.timestamp = timestamp
        self.event_data_size_bytes = event_data_size_bytes

    def is_bor_event(self) -> bool:
        return self.event_id == EVID_BOR

    def is_eor_event(self) -> bool:
        return self.event_id == EVID_EOR

    def is_msg_event(self) -> bool:
        return self.event_id == EVID_MSG

    def is_midas_internal_event(self) -> bool:
        return self.event_id in INTERNAL_IDS


class Bank:
    """One bank; ``data`` is a numpy array (or raw bytes for string/struct types)."""

    __slots__ = ("name", "type", "size_bytes", "data")

    def __init__(self, name, type, size_bytes, data):  # noqa: A002 - MIDAS's name
        self.name = name
        self.type = type
        self.size_bytes = size_bytes
        self.data = data


class Event:
    """One event: header, banks, and for internal events the raw body."""

    __slots__ = ("header", "banks", "non_bank_data", "flags", "bank_error", "position", "raw")

    def __init__(self, header: EventHeader):
        self.header = header
        #: 0-based position among ALL events of the file (BOR and other
        #: detectors' events included) -- the entry number of the nearline rec
        #: ntuple for that subrun. None for events not read from a file.
        self.position: int | None = None
        #: The event's bytes (header + body), when the reader keeps them.
        self.raw: bytes | None = None
        self.banks: dict[str, Bank] = {}
        self.non_bank_data: bytes | None = None
        self.flags = 0
        #: Set when the bank list is malformed; the banks before it are kept.
        self.bank_error: str | None = None

    def get_bank(self, name: str) -> Bank | None:
        return self.banks.get(name)

    def bank_exists(self, name: str) -> bool:
        return name in self.banks


def bank_data(btype: int, raw: bytes | memoryview):
    """Bank payload -> numpy array of its TID, like ``use_numpy=True``."""
    dt = TID_DTYPES.get(btype)
    if dt is None:
        return bytes(raw)
    n = len(raw) // dt.itemsize
    arr = np.frombuffer(raw, dtype=dt, count=n)
    if btype == TID_BOOL:
        arr = arr.astype(np.bool_)
    return arr


def parse_banks(event: Event, body: bytes) -> None:
    """Fill ``event.banks`` from an event body (bank16, bank32 or bank32a)."""
    if len(body) < BANK_HEADER.size:
        event.bank_error = f"body of {len(body)} bytes has no bank header"
        return
    all_size, flags = BANK_HEADER.unpack_from(body, 0)
    event.flags = flags
    if flags & FLAG_ALIGN64:
        hdr, fmt = 16, BANK32
    elif flags & FLAG_32BIT:
        hdr, fmt = 12, BANK32
    else:
        hdr, fmt = 8, BANK16
    end = BANK_HEADER.size + all_size
    if end > len(body):
        event.bank_error = f"bank list of {all_size} bytes overruns a {len(body)}-byte body"
        end = len(body)
    pos = BANK_HEADER.size
    view = memoryview(body)
    while pos + hdr <= end:
        name_b, btype, size = fmt.unpack_from(body, pos)
        start = pos + hdr
        if start + size > end:
            event.bank_error = (f"bank {name_b!r} of {size} bytes at offset {pos} overruns "
                                f"the bank list ({end} bytes)")
            break
        name = name_b.decode("ascii", "replace")
        event.banks[name] = Bank(name, btype, size, bank_data(btype, view[start:start + size]))
        pos = start + ((size + 7) & ~7)


def open_raw(path: str | Path):
    """A binary file object for plain, ``.lz4`` or ``.gz`` MIDAS files."""
    path = str(path)
    if path.endswith(".lz4"):
        try:
            import lz4.frame
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(f"{path}: reading .lz4 needs the 'lz4' package "
                               "(pip install lz4, or the mdqm[offline] extra)") from exc
        return lz4.frame.open(path, "rb")
    if path.endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")  # noqa: SIM115 - closed by MidasFile


_NAME_RE = re.compile(r"run(\d+)(?:_(\d+))?")


def run_subrun_from_name(path: str | Path) -> tuple[int | None, int | None]:
    """``run01008_00001.mid.lz4`` -> (1008, 1); (None, None) when the name has neither."""
    m = _NAME_RE.search(Path(path).name)
    if not m:
        return None, None
    return int(m.group(1)), (int(m.group(2)) if m.group(2) is not None else None)


class MidasFile:
    """Stream the events of one file.

    ``event_ids``: only these events have their banks unpacked and are yielded
    (the others are still read past; lz4 cannot seek). Internal events
    (BOR/EOR/messages) are yielded only with ``include_internal=True``, but the
    begin-of-run serial is always recorded as `bor_run_number`.

    After iteration, `truncated` says whether the file ended inside an event,
    `events_read` counts every event header read, and `last_timestamp` is the
    time of the last event of any kind. Every yielded event carries its
    `position` (0-based among all events of the file); with ``keep_raw`` also its
    bytes in `raw`.
    """

    def __init__(self, path: str | Path, event_ids=None, include_internal: bool = False,
                 keep_raw: bool = False,
                 chunk: int = 1 << 23):
        self.path = str(path)
        self.event_ids = None if event_ids is None else frozenset(int(x) for x in event_ids)
        self.include_internal = include_internal
        #: Keep each yielded event's bytes in ``event.raw`` (for writing it out).
        self.keep_raw = bool(keep_raw)
        self.chunk = int(chunk)
        self.truncated = False
        self.events_read = 0
        self.bor_run_number: int | None = None
        self.first_timestamp: int | None = None
        self.last_timestamp: int | None = None

    def __iter__(self):
        return self.events()

    def events(self):
        fh = open_raw(self.path)
        try:
            yield from self._events(fh)
        finally:
            fh.close()

    def _events(self, fh):
        buf = bytearray()
        off = 0
        hsz = EVENT_HEADER.size
        eof = False

        def fill(need: int) -> bool:
            """Make at least `need` bytes available from `off`; False at end of file."""
            nonlocal buf, off, eof
            while len(buf) - off < need:
                if eof:
                    return False
                more = fh.read(max(self.chunk, need))
                if not more:
                    eof = True
                    return False
                if off:
                    del buf[:off]
                    off = 0
                buf += more
            return True

        while True:
            if not fill(hsz):
                if len(buf) - off:
                    self.truncated = True
                return
            eid, tmask, serial, ts, dsz = EVENT_HEADER.unpack_from(buf, off)
            if dsz > MAX_EVENT_BYTES:
                raise ValueError(f"{self.path}: implausible event size {dsz} bytes after "
                                 f"{self.events_read} events; not a MIDAS file?")
            if not fill(hsz + dsz):
                self.truncated = True
                return
            self.events_read += 1
            if self.first_timestamp is None:
                self.first_timestamp = ts
            self.last_timestamp = ts
            header = EventHeader(eid, tmask, serial, ts, dsz)
            start = off + hsz
            off = start + dsz

            if eid in INTERNAL_IDS:
                if eid == EVID_BOR and self.bor_run_number is None:
                    self.bor_run_number = serial
                if self.include_internal:
                    ev = Event(header)
                    ev.position = self.events_read - 1
                    ev.non_bank_data = bytes(buf[start:start + dsz])
                    if self.keep_raw:
                        ev.raw = bytes(buf[start - hsz:start + dsz])
                    yield ev
                continue
            if self.event_ids is not None and eid not in self.event_ids:
                continue
            ev = Event(header)
            ev.position = self.events_read - 1
            parse_banks(ev, bytes(buf[start:start + dsz]))
            if self.keep_raw:
                ev.raw = bytes(buf[start - hsz:start + dsz])
            yield ev


# ---------------------------------------------------------------------------
# writing (tests and fixtures)
# ---------------------------------------------------------------------------

def encode_event(event_id: int, serial: int, timestamp: int, banks, *, fmt: str = "bank32a",
                 trigger_mask: int = 0) -> bytes:
    """Bytes of one event holding `banks` = [(name, tid, array-or-bytes), ...].

    ``fmt`` is ``bank16``, ``bank32`` or ``bank32a``. Used by the tests to build
    files by hand; kept here so the layout lives next to the reader.
    """
    flags = {"bank16": 0x01, "bank32": 0x11, "bank32a": 0x31}[fmt]
    body = bytearray()
    for name, tid, data in banks:
        raw = data.tobytes() if isinstance(data, np.ndarray) else bytes(data)
        nb = name.encode("ascii")
        if len(nb) != 4:
            raise ValueError(f"bank name must be 4 characters, got {name!r}")
        if fmt == "bank16":
            body += BANK16.pack(nb, tid, len(raw))
        elif fmt == "bank32":
            body += BANK32.pack(nb, tid, len(raw))
        else:
            body += BANK32.pack(nb, tid, len(raw)) + b"\0" * 4
        body += raw + b"\0" * (((len(raw) + 7) & ~7) - len(raw))
    body = BANK_HEADER.pack(len(body), flags) + body
    return EVENT_HEADER.pack(event_id, trigger_mask, serial, timestamp, len(body)) + body


def encode_internal(event_id: int, serial: int, timestamp: int, payload: bytes) -> bytes:
    """A BOR/EOR/message event: header and a raw payload (an ODB dump, a message)."""
    return EVENT_HEADER.pack(event_id, 0x494D, serial, timestamp, len(payload)) + payload
