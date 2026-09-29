#!/usr/bin/env python3
"""How fast can the DQM chain actually process waveforms?

Replays a recorded run into the event buffer at increasing rates and measures
what the analyzer sustains, so the answer is a measurement rather than an
estimate. Reports, per offered rate:

    offered      what the replay actually delivered
    processed    what the analyzer decoded and histogrammed
    cpu          analyzer CPU, as a fraction of one core
    mhttpd       mhttpd CPU, since it is shared with run control
    lost         offered - processed, i.e. events sampled away
    frame p50/95 (with --frame-cmd) round-trip latency of that brpc while under load

The same script drives the SMA analyzer: --event-id, --status-cmd, --limit-path
and --client select which stream, status call, rate-limit key and client are
measured. The defaults are the WaveDREAM ones.

Losing events is not a failure. The analyzer samples deliberately: it drains the
buffer every cycle with GET_NONBLOCKING so it can never back-pressure a frontend,
and decodes only what its rate limit allows. What this measures is where the
*limit* has to sit before it starts to bite.
"""

from __future__ import annotations

import argparse
import json
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path

REPLAY = Path(__file__).resolve().parent / "replay-run.py"
CLK = 100.0


class NotAnswered(RuntimeError):
    """The analyzer did not answer in time -- a measurement, not an error."""


def pid_of(name: str) -> int | None:
    out = subprocess.run(["pgrep", "-x", name], capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


def analyzer_client(comm: str, cmd: list[str]) -> tuple[bool, str | None] | None:
    """(True, --client value or None) if this process is an mdqm analyzer, else None.

    ``comm`` is "python3" for ``python -m mdqm.dqm.analyzer`` but the script's
    own name, "mdqm-analyzer", for the installed console script (the kernel
    names a shebang process after the script), so both are accepted.
    """
    if not (comm.startswith("python") or comm == "mdqm-analyzer"):
        return None
    if not any("mdqm.dqm.analyzer" in a or "mdqm-analyzer" in a for a in cmd):
        return None
    named = None
    for i, a in enumerate(cmd):
        if a == "--client" and i + 1 < len(cmd):
            named = cmd[i + 1]
        elif a.startswith("--client="):
            named = a.split("=", 1)[1]
    return True, named


def analyzer_pid(client: str | None = None) -> int | None:
    """The analyzer is a python process, so match on its command line.

    With two analyzers in one experiment (wd_analyzer and sma_analyzer) "the
    first one" is a coin toss that would measure the wrong process's CPU, so
    prefer the one started with ``--client <client>``. An analyzer started
    without --client has no name on its command line; it is accepted only when
    it is the sole candidate.
    """
    candidates = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            comm = (entry / "comm").read_text().strip()
            cmd = (entry / "cmdline").read_bytes().decode(errors="ignore").split("\0")
        except OSError:
            continue
        found = analyzer_client(comm, cmd)
        if found is not None:
            candidates.append((int(entry.name), found[1]))
    if client:
        for pid, named in candidates:
            if named == client:
                return pid
    unnamed = [pid for pid, named in candidates if named is None]
    if len(unnamed) == 1 and len(candidates) == 1:
        return unnamed[0]
    return None


def cpu_seconds(pid: int) -> float:
    parts = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    return (int(parts[11]) + int(parts[12])) / CLK


def rss_mb(pid: int) -> float:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return 0.0


def brpc(client: str, cmd: str, port: int, max_len: int = 262144, args: str = ""):
    payload = json.dumps({"jsonrpc": "2.0", "method": "brpc", "id": 1,
                          "params": {"client_name": client, "cmd": cmd, "args": args,
                                     "max_reply_length": max_len}})
    raw = subprocess.run(
        ["curl", "-s", "-H", "Content-Type: application/json", "--data-binary", payload,
         f"http://localhost:{port}?mjsonrpc"], capture_output=True).stdout
    return decode_reply(cmd, raw)


def decode_reply(cmd: str, raw: bytes):
    """A brpc reply: the parsed JSON for a ``json`` envelope, else the raw payload.

    Binary replies (``smaf``, ``scop``, ``hist``) are only timed, not decoded.
    """
    if len(raw) < 8:
        raise RuntimeError(f"short reply to {cmd}: {len(raw)} bytes")
    # mhttpd answers with JSON-RPC text when it could not reach the client at
    # all -- which at high rates means the analyzer did not get scheduled in
    # time to answer. That is a result worth reporting, not a crash.
    if raw[:1] == b"{":
        raise NotAnswered(raw[:200].decode(errors="replace"))
    size, tag = struct.unpack_from("<I4s", raw, 0)
    if size > len(raw) or size < 8:
        raise NotAnswered(f"implausible envelope size {size} in {len(raw)} bytes")
    if tag == b"json":
        return json.loads(raw[8:size])
    return raw[8:size]


def set_odb(expt: str, path: str, value) -> None:
    subprocess.run(["odbedit", "-e", expt, "-q", "-c", f"set '{path}' {value}"],
                   capture_output=True)


def kill_replays() -> None:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if not (entry / "comm").read_text().strip().startswith("python"):
                continue
            if "replay-run" in (entry / "cmdline").read_bytes().decode(errors="ignore"):
                subprocess.run(["kill", "-9", entry.name], capture_output=True)
        except OSError:
            continue


def _pct(values: list[float], q: int) -> float | None:
    """Percentile of a short list; None when there were no samples."""
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[q - 1]


def missed_per_s(s0: dict, s1: dict, dt: float) -> float | None:
    """Serial-number gaps per second from the plugin's status, if it counts them."""
    p0, p1 = s0.get("plugin") or {}, s1.get("plugin") or {}
    if "missed_by_serial" not in p0 or "missed_by_serial" not in p1:
        return None
    return (p1["missed_by_serial"] - p0["missed_by_serial"]) / dt


def measure(run_file: Path, rate: float, seconds: float, client: str, expt: str,
            port: int, event_id: int, status_cmd: str,
            frame_cmd: str | None, frame_args: str = "") -> dict:
    apid, mpid = analyzer_pid(client), pid_of("mhttpd")
    kill_replays()
    time.sleep(1)

    proc = subprocess.Popen(
        [sys.executable, str(REPLAY), str(run_file), "--loop", "--event-id", str(event_id),
         "--experiment", expt, "--quiet", "--rate", str(rate) if rate > 0 else "0"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    unanswered = 0
    frame_ms: list[float] = []
    try:
        time.sleep(4)                              # let it reach steady state
        s0 = brpc(client, status_cmd, port)
        c0, m0, t0 = cpu_seconds(apid), cpu_seconds(mpid), time.time()
        # Poll while the load runs, so an analyzer that stops answering under
        # load is caught rather than only sampled at the ends.
        deadline = time.time() + seconds
        while time.time() < deadline:
            time.sleep(2.0)
            try:
                brpc(client, status_cmd, port)
            except NotAnswered:
                unanswered += 1
            if frame_cmd:
                # A frame reply is large (a whole waveform frame as JSON), so
                # its latency under load is the number the page's refresh rate
                # has to live with, not the cheap status call's.
                t = time.time()
                try:
                    brpc(client, frame_cmd, port, max_len=8 * 1024 * 1024,
                         args=frame_args)
                    frame_ms.append((time.time() - t) * 1e3)
                except NotAnswered:
                    unanswered += 1
        s1 = brpc(client, status_cmd, port)
        c1, m1, t1 = cpu_seconds(apid), cpu_seconds(mpid), time.time()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        kill_replays()

    dt = t1 - t0
    seen = (s1["events_seen"] - s0["events_seen"]) / dt
    missed = missed_per_s(s0, s1, dt)
    return {
        "asked": rate,
        # events_seen counts everything taken off the buffer. A reader that
        # falls behind never sees the events the buffer overwrote, so when the
        # plugin counts serial-number gaps (the replay numbers every event),
        # what the producer delivered is processed + missed.
        "offered": seen if missed is None else max(
            seen, (s1["events_processed"] - s0["events_processed"]) / dt + missed),
        "seen": seen,
        "missed_by_serial": missed,
        "processed": (s1["events_processed"] - s0["events_processed"]) / dt,
        "analyzer_cpu": (c1 - c0) / dt,
        "mhttpd_cpu": (m1 - m0) / dt,
        "analyzer_rss": rss_mb(apid),
        "throttled": s1.get("throttled", False),
        "unanswered": unanswered,
        "frame_p50_ms": _pct(frame_ms, 50),
        "frame_p95_ms": _pct(frame_ms, 95),
        "frame_calls": len(frame_ms),
        "budget_exhausted": s1.get("budget_exhausted", 0) - s0.get("budget_exhausted", 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_file", type=Path)
    ap.add_argument("--rates", type=float, nargs="+",
                    default=[20, 100, 200, 500, 1000, 0])
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--client", default="wd_analyzer",
                    help="analyzer client name (SMA: sma_analyzer)")
    ap.add_argument("--event-id", type=int, default=401,
                    help="event id to replay (WaveDREAM 401, SMA 301)")
    ap.add_argument("--status-cmd", default="wd::status",
                    help="brpc command returning events_seen/events_processed "
                         "(the SMA analyzer also answers dqm::status)")
    ap.add_argument("--limit-path", default="/DQM/Analyzer/Sampling/max events per s",
                    help="ODB key of the analyzer's max-rate setting "
                         "(SMA: /DQM/SMA/Sampling/max events per s)")
    ap.add_argument("--frame-cmd", default=None,
                    help="also time this brpc (e.g. sma::frame) at each rate step")
    ap.add_argument("--frame-args", default="",
                    help="args for --frame-cmd, e.g. '{\"view\": \"raster\"}' for the "
                         "SMA whole-frame raster (default: the command's own default)")
    ap.add_argument("--lossless", type=float, default=0.999,
                    help="processed/offered fraction counted as lossless")
    ap.add_argument("--experiment", default="WDSCALERS")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--limit", type=float, default=5000.0,
                    help="analyzer rate limit to set for the test, via the ODB")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if not args.run_file.is_file():
        print(f"error: no such run file: {args.run_file}", file=sys.stderr)
        return 2
    if analyzer_pid(args.client) is None:
        print(f"error: no analyzer process found for client {args.client}",
              file=sys.stderr)
        return 2

    # Raise the analyzer's own limit out of the way, through the ODB, so the
    # ceiling this finds is the code's rather than the configured cap.
    set_odb(args.experiment, args.limit_path, args.limit)
    time.sleep(4)

    print(f"analyzer rate limit set to {args.limit}/s for the test\n", flush=True)
    print(f"{'asked':>7} {'offered':>9} {'processed':>10} {'lost':>7} "
          f"{'analyzer':>9} {'mhttpd':>8} {'RSS MB':>8}  notes", flush=True)
    results = []
    for rate in args.rates:
        try:
            r = measure(args.run_file, rate, args.seconds, args.client,
                        args.experiment, args.port, args.event_id,
                        args.status_cmd, args.frame_cmd, args.frame_args)
        except NotAnswered as exc:
            label = "max" if rate == 0 else f"{rate:.0f}"
            print(f"{label:>7} {'-':>9} {'-':>10} {'-':>7} {'-':>9} {'-':>8} "
                  f"{'-':>8}  analyzer did not answer: {exc}", flush=True)
            continue
        results.append(r)
        lost = max(0.0, r["offered"] - r["processed"])
        pct = (100.0 * lost / r["offered"]) if r["offered"] else 0.0
        label = "max" if rate == 0 else f"{rate:.0f}"
        print(f"{label:>7} {r['offered']:>9.1f} {r['processed']:>10.1f} "
              f"{pct:>6.1f}% {r['analyzer_cpu'] * 100:>8.1f}% "
              f"{r['mhttpd_cpu'] * 100:>7.1f}% {r['analyzer_rss']:>8.1f}"
              + ("  THROTTLED" if r["throttled"] else "")
              + (f"  {r['unanswered']} status calls unanswered" if r["unanswered"] else "")
              + (f"  budget hit {r['budget_exhausted']}x" if r["budget_exhausted"] else "")
              + (f"  frame p50 {r['frame_p50_ms']:.0f} / p95 {r['frame_p95_ms']:.0f} ms"
                 f" (n={r['frame_calls']})" if r["frame_p50_ms"] is not None else ""),
              flush=True)

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))
        print(f"\nwrote {args.json}")

    # 99.9 %, not 95 %: "lossless" is the claim the SMA analyzer makes, and a
    # 5 % loss would hide a real drop of whole frames.
    ok = [r for r in results if r["offered"] > 0
          and r["processed"] >= args.lossless * r["offered"]]
    if ok:
        best = max(ok, key=lambda r: r["processed"])
        print(f"\nlossless (>={100 * args.lossless:g}% processed) up to at least {best['processed']:.0f} ev/s "
              f"at {best['analyzer_cpu'] * 100:.0f}% of one core")
    saturated = [r for r in results if r["offered"] > 0
                 and r["processed"] < args.lossless * r["offered"]]
    if saturated:
        first = min(saturated, key=lambda r: r["offered"])
        print(f"first falls behind when offered {first['offered']:.0f} ev/s "
              f"(processed {first['processed']:.0f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
