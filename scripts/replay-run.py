#!/usr/bin/env python3
"""Replay a recorded MIDAS run file into a live event buffer.

The point is to be able to develop and test everything downstream of the event
buffer -- the scope page, the analyzer, the histogram plumbing -- on a machine
with no detector attached, against events that are real rather than synthetic.

    scripts/replay-run.py run00201.mid.lz4 --rate 20 --loop

Reads the file, re-stamps each event's serial number and timestamp so consumers
see a plausible live stream, and sends it at a chosen rate. With
``--keep-header`` the file's own serial numbers and time stamps are sent
unchanged, so a tag copied from the pages (run, serial) finds the same event in
the file; with ``--loop`` the serials then start again at every pass, which the
SMA plugin treats as a restart (not a gap, and not frames offered).

Safety
------
This injects events into a shared buffer, so it refuses to run while a run is
active unless you insist: with the logger recording, replayed events would be
written into the run file as though they were real data, which is a corrupted
dataset that nobody would notice until analysis. With the run stopped, mlogger
is not reading the buffer and nothing reaches disk.

It also declares no equipment and registers no transition callbacks, so it
cannot delay a run start or stop however wedged it gets.

If it appears to hang
---------------------
``send_event`` ends in ``bm_flush_cache(..., BM_WAIT)``, which blocks until the
buffer has room -- and blocks *uninterruptibly*, because the signal handler
cannot run while the interpreter is inside that C call. The cause is almost
always a **stale buffer client**: a consumer that died without detaching still
pins the buffer's read pointer, so space is never reclaimed and no amount of
waiting helps.

Compare ``/System/Buffers/SYSTEM/Clients`` with ``/System/Clients``; a name in
the first and not the second is a stale reader. ``odbedit -c cleanup`` does not
remove them. Stopping every MIDAS client (``stop-midas.sh``) and starting again
does.

Note that ``/System/Buffers/SYSTEM/filled`` is a stale ODB *reflection*, not a
live measurement -- nothing updates it while the buffer is idle, so a reading of
99% on a healthy buffer is normal and is not evidence of anything. The check
below therefore warns rather than refuses.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

import midas
import midas.client
import midas.file_reader

_stop = False


def _on_signal(_sig, _frm):
    global _stop
    _stop = True


def run_state(client) -> int:
    try:
        return int(client.odb_get("/Runinfo/State"))
    except Exception:
        return -1


def warn_about_stale_readers(client, buffer_name: str) -> None:
    """Name any dead client still attached to the buffer, before we block on it.

    A stale reader pins the read pointer, so the buffer fills and send_event
    blocks forever inside BM_WAIT where the signal handler cannot reach it. That
    is a confusing way to hang, and the cause is cheap to look up first.
    """
    try:
        attached = set(_client_names(client, f"/System/Buffers/{buffer_name}/Clients"))
        alive = set(_client_names(client, "/System/Clients"))
    except Exception:
        return
    stale = sorted(n for n in attached - alive if n)
    if not stale:
        return
    print(f"warning: {len(stale)} client(s) are attached to {buffer_name} but are not "
          f"running: {', '.join(stale)}", file=sys.stderr)
    print("         A dead reader pins the buffer's read pointer, and send_event will "
          "block forever waiting for space it will never get.", file=sys.stderr)
    print("         `odbedit -c cleanup` does not clear these; stop every MIDAS client "
          "and start again if this hangs.", file=sys.stderr)


def _client_names(client, path: str) -> list[str]:
    """The `Name` of every subdirectory of `path`."""
    names = []
    entries = client.odb_get(path) or {}
    for key, value in entries.items():
        if key.endswith("/key") or not isinstance(value, dict):
            continue
        name = value.get("Name")
        if isinstance(name, str):
            names.append(name)
    return names


def replay(path: Path, client, buf, rate: float, limit: int | None,
           loop: bool, event_ids: set[int] | None, verbose: bool,
           use_numpy: bool = True, report_s: float = 0.0, keep_header: bool = False) -> int:
    """Send events from `path` at `rate` per second. Returns how many were sent.

    With `report_s`, prints one line every `report_s` seconds: the events sent
    and the rate achieved in that interval, and the time spent inside
    ``send_event`` (mean and max per event). The last two show whether the
    producer was ever held up by the buffer, i.e. by a consumer.
    """
    interval = 1.0 / rate if rate > 0 else 0.0
    sent = 0
    serial = 0
    t_next = time.time()
    rep = {"t": time.time(), "n": 0, "send_s": 0.0, "send_max": 0.0}

    while not _stop:
        # Reopened each pass: MidasFile is a one-shot iterator, and reopening is
        # also what makes --loop replay the run's first event again, which is
        # the only event carrying the full DRS calibration table.
        #
        # use_numpy hands banks over as ndarrays, not tuples of Python ints. A
        # 40000-word TID_DWORD bank is 40000 int objects as a tuple, and both
        # unpacking it and struct-packing it again in send_event are per-word
        # Python loops; as an array the pack is one memmove (midas/event.py,
        # the `np_buf = bank.data.tobytes()` branch).
        f = midas.file_reader.MidasFile(str(path), use_numpy=use_numpy)
        for event in f:
            if _stop:
                break
            # Begin/end-of-run records and messages are midas' own bookkeeping.
            # Replaying them would announce run transitions that are not
            # happening.
            if event.header.is_midas_internal_event():
                continue
            if event_ids is not None and event.header.event_id not in event_ids:
                continue

            if not keep_header:
                serial += 1
                event.header.serial_number = serial
                event.header.timestamp = int(time.time())
            t_send = time.perf_counter()
            client.send_event(buf, event)
            d_send = time.perf_counter() - t_send
            sent += 1
            if report_s:
                rep["n"] += 1
                rep["send_s"] += d_send
                rep["send_max"] = max(rep["send_max"], d_send)
                now = time.time()
                if now - rep["t"] >= report_s:
                    dt = now - rep["t"]
                    print(f"report t={now:.3f} sent={sent} rate={rep['n'] / dt:.2f} "
                          f"send_mean_ms={1e3 * rep['send_s'] / rep['n']:.3f} "
                          f"send_max_ms={1e3 * rep['send_max']:.2f}", flush=True)
                    rep.update(t=now, n=0, send_s=0.0, send_max=0.0)

            if verbose and sent % 100 == 0:
                print(f"  sent {sent}", flush=True)
            if limit is not None and sent >= limit:
                return sent

            if interval:
                t_next += interval
                delay = t_next - time.time()
                if delay > 0:
                    time.sleep(delay)
                else:
                    # Behind schedule: give up the lost time rather than
                    # sprinting to catch up, which would defeat the point of
                    # asking for a rate.
                    t_next = time.time()

        if not loop:
            return sent
        client.communicate(10)
    return sent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_file", type=Path)
    ap.add_argument("--experiment", default=os.environ.get("MIDAS_EXPT_NAME"))
    ap.add_argument("--buffer", default="SYSTEM")
    ap.add_argument("--client-name", default="wd_replay")
    ap.add_argument("--rate", type=float, default=20.0,
                    help="events per second; 0 means as fast as possible")
    ap.add_argument("--limit", type=int, default=None, help="stop after N events")
    ap.add_argument("--loop", action="store_true", help="start again at the end")
    ap.add_argument("--event-id", type=int, action="append", default=None,
                    help="only replay these event ids (repeatable)")
    ap.add_argument("--max-event-size", type=int, default=8 * 1024 * 1024)
    ap.add_argument("--allow-during-run", action="store_true",
                    help="inject even with a run active -- see the safety note")
    ap.add_argument("--numpy", action=argparse.BooleanOptionalAction, default=True,
                    help="read banks as numpy arrays (default; --no-numpy gives the "
                         "slow tuple path, useful only to compare)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--keep-header", action="store_true",
                    help="send the file's own serial numbers and time stamps (tags from the "
                         "pages then match the file); default: renumber from 1, stamp now")
    ap.add_argument("--report-s", type=float, default=0.0,
                    help="print the achieved rate and the time spent in send_event every "
                         "this many seconds (0: never)")
    args = ap.parse_args(argv)

    if not args.run_file.is_file():
        print(f"error: no such run file: {args.run_file}", file=sys.stderr)
        return 2
    if not args.experiment:
        print("error: no experiment; pass --experiment or set MIDAS_EXPT_NAME",
              file=sys.stderr)
        return 2

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    with midas.client.MidasClient(args.client_name, expt_name=args.experiment) as client:
        state = run_state(client)
        if state == 3 and not args.allow_during_run:
            print("error: a run is active. Replayed events would be written into the "
                  "run file as though they were real data.\n"
                  "       Stop the run, or pass --allow-during-run if you are certain.",
                  file=sys.stderr)
            return 1

        buf = client.open_event_buffer(args.buffer, None, args.max_event_size)
        warn_about_stale_readers(client, args.buffer)

        if not args.quiet:
            ids = "all" if args.event_id is None else args.event_id
            print(f"replaying {args.run_file.name} into {args.buffer} "
                  f"of {args.experiment}")
            print(f"  rate {args.rate or 'unthrottled'} ev/s, event ids {ids}"
                  f"{', looping' if args.loop else ''}")

        t0 = time.time()
        sent = replay(args.run_file, client, buf,
                      args.rate, args.limit, args.loop,
                      set(args.event_id) if args.event_id else None,
                      not args.quiet, args.numpy, args.report_s, args.keep_header)
        dt = time.time() - t0

    if not args.quiet:
        achieved = sent / dt if dt else 0.0
        print(f"sent {sent} events in {dt:.1f}s ({achieved:.1f} ev/s)")
        # A warning, not an error: the stream is still valid, it is just slower
        # than the consumer test assumed. Matters when the rate is the variable.
        if args.rate > 0 and achieved < 0.95 * args.rate:
            print(f"warning: asked for {args.rate:g} ev/s but achieved {achieved:.1f}; "
                  "the replay itself is the bottleneck"
                  + ("" if args.numpy else " (try --numpy)"), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
