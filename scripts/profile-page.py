#!/usr/bin/env python3
"""Measure what a custom page costs the machine that is taking data.

A monitoring page must never affect data taking, and on this deployment the
process to watch is not ours -- it is mhttpd, which also serves run control.
Every ODB read the page makes is work in that process, so "the page has no
backend" is not by itself an argument that it is free.

    scripts/profile-page.py --tabs 0 1 2 5 --seconds 30

Reports, per tab count: mhttpd CPU as a fraction of one core, its RSS, and the
DAQ's own dropped-packet counter, against a baseline measured first with no
page open at all.

The same script profiles the SMA pages (defaults are the WaveDREAM Scalers
page's). For a brpc page the analyzer answering it does work too, and the
reply bytes are what a remote viewer pulls over the network:

    scripts/profile-page.py --driver chromium --container testbeam-midas \\
        --url 'http://localhost:8123/?cmd=custom&page=SMAEvents' \\
        --click '#dqm-smaev-tab-raster' --analyzer-client sma_analyzer \\
        --experiment smadqm --dropped-path '' --tabs 0 1 3 --repeat 2

* ``--driver chromium`` uses playwright's headless chromium instead of
  geckodriver/firefox, and then also reports the bytes each viewer receives
  (Chrome DevTools ``Network.loadingFinished``), per second.
* ``--container NAME`` reads ``/proc`` inside a docker container (mhttpd and
  the analyzer run there and are invisible to the host's ``/proc``).
* ``--click SELECTOR`` (repeatable) is clicked in order after each page loads:
  a tab button, a checkbox.
* ``--analyzer-client NAME`` also measures that brpc client's CPU and RSS.
* ``--repeat N`` measures every tab count N times (each against its own
  0-tab baseline when 0 is in ``--tabs``).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

CLK = 100.0  # kernel USER_HZ; /proc/<pid>/stat ticks


# ---------------------------------------------------------------------------
# /proc, here or inside a container
# ---------------------------------------------------------------------------

def parse_ticks(stat_text: str) -> float:
    """utime + stime from a /proc/<pid>/stat line, in seconds."""
    parts = stat_text.rsplit(") ", 1)[1].split()
    return (int(parts[11]) + int(parts[12])) / CLK


def parse_rss_mb(status_text: str) -> float:
    for line in status_text.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024
    return 0.0


class Proc:
    """Reads /proc and finds pids locally, or inside ``docker exec <container>``."""

    def __init__(self, container: str | None = None, run=subprocess.run):
        self.container = container
        self._run = run

    def _cmd(self, argv: list[str]) -> list[str]:
        return (["docker", "exec", self.container] + argv) if self.container else argv

    def read(self, path: str) -> str:
        if not self.container:
            return Path(path).read_text()
        return self._run(self._cmd(["cat", path]), capture_output=True, text=True,
                         check=True).stdout

    def pid_of(self, name: str) -> int | None:
        out = self._run(self._cmd(["pgrep", "-x", name]), capture_output=True,
                        text=True).stdout.split()
        return int(out[0]) if out else None

    def pid_of_client(self, client: str) -> int | None:
        """The process started with ``--client <client>`` (an mdqm analyzer)."""
        out = self._run(self._cmd(["pgrep", "-f", "--", f"--client {client}"]),
                        capture_output=True, text=True).stdout.split()
        return int(out[0]) if out else None

    def cpu_ticks(self, pid: int) -> float:
        return parse_ticks(self.read(f"/proc/{pid}/stat"))

    def rss_mb(self, pid: int) -> float:
        return parse_rss_mb(self.read(f"/proc/{pid}/status"))


# Kept for callers of the old module-level helpers.
def pid_of(name: str) -> int | None:
    return Proc().pid_of(name)


def cpu_ticks(pid: int) -> float:
    return Proc().cpu_ticks(pid)


def rss_mb(pid: int) -> float:
    return Proc().rss_mb(pid)


def odb_int(expt: str, path: str) -> int | None:
    out = subprocess.run(["odbedit", "-e", expt, "-q", "-c", f"ls -v '{path}'"],
                         capture_output=True, text=True).stdout.strip()
    try:
        return int(out.split()[0], 0)
    except (ValueError, IndexError):
        return None


def loadavg() -> float | None:
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except OSError:
        return None


# ---------------------------------------------------------------------------
# viewers
# ---------------------------------------------------------------------------

class GeckoViewers:
    """N firefox sessions through geckodriver (the original harness)."""

    def __init__(self, driver: str):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from shoot import Session  # noqa: E402
        self._Session = Session
        self._driver = driver
        self.sessions = []

    def open(self, n: int, url: str, clicks: list[str]) -> None:
        for _ in range(n):
            s = self._Session(self._driver, width=1200, height=900)
            s.goto(url)
            for sel in clicks:
                time.sleep(1.0)
                s.script(f"document.querySelector({json.dumps(sel)}).click()")
            self.sessions.append(s)

    def wait(self, seconds: float) -> None:
        time.sleep(seconds)

    def bytes_received(self) -> list[int] | None:
        return None

    def close(self) -> None:
        for s in self.sessions:
            s.close()
        self.sessions = []


class ChromiumViewers:
    """N headless chromium pages (one browser context each, like N people).

    Reply bytes come from the DevTools protocol: ``Network.loadingFinished``
    carries the bytes received per request, headers included.
    """

    def __init__(self, pw):
        self._browser = pw.chromium.launch()
        self.pages = []
        self._rx: list[int] = []

    def open(self, n: int, url: str, clicks: list[str]) -> None:
        for _ in range(n):
            ctx = self._browser.new_context(viewport={"width": 1200, "height": 900})
            page = ctx.new_page()
            k = len(self._rx)
            self._rx.append(0)
            cdp = ctx.new_cdp_session(page)
            cdp.send("Network.enable")

            def on_finished(ev, k=k):
                self._rx[k] += int(ev.get("encodedDataLength") or 0)
            cdp.on("Network.loadingFinished", on_finished)
            page.goto(url)
            for sel in clicks:
                page.wait_for_timeout(1000)
                page.click(sel)
            self.pages.append((ctx, page, cdp))

    def wait(self, seconds: float) -> None:
        # Pumps the event loop, so the byte counters stay current; time.sleep
        # would leave the DevTools events queued until the next call.
        if self.pages:
            self.pages[0][1].wait_for_timeout(seconds * 1000)
        else:
            time.sleep(seconds)

    def bytes_received(self) -> list[int] | None:
        return list(self._rx)

    def close(self) -> None:
        for ctx, _page, _cdp in self.pages:
            ctx.close()
        self.pages = []
        self._rx = []

    def shutdown(self) -> None:
        self.close()
        self._browser.close()


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

def measure(proc: Proc, pid: int, seconds: float, expt: str, dropped_path: str | None,
            viewers=None, apid: int | None = None) -> dict:
    d0 = odb_int(expt, dropped_path) if dropped_path else None
    rx0 = viewers.bytes_received() if viewers else None
    l0 = loadavg()
    c0, a0, t0 = proc.cpu_ticks(pid), (proc.cpu_ticks(apid) if apid else None), time.time()
    if viewers:
        viewers.wait(seconds)
    else:
        time.sleep(seconds)
    c1, a1, t1 = proc.cpu_ticks(pid), (proc.cpu_ticks(apid) if apid else None), time.time()
    rx1 = viewers.bytes_received() if viewers else None
    d1 = odb_int(expt, dropped_path) if dropped_path else None
    dt = t1 - t0
    out = {
        "cpu_cores": (c1 - c0) / dt,
        "rss_mb": proc.rss_mb(pid),
        "dropped_delta": (None if d0 is None or d1 is None else d1 - d0),
    }
    if apid:
        out["analyzer_cpu_cores"] = (a1 - a0) / dt
        out["analyzer_rss_mb"] = proc.rss_mb(apid)
    if rx0 is not None and rx1 is not None and len(rx0) == len(rx1) and rx1:
        per = [(b - a) / dt for a, b in zip(rx0, rx1, strict=True)]
        out["bytes_per_s_per_viewer"] = sum(per) / len(per)
    out["load1_before"], out["load1_after"] = l0, loadavg()
    return out


def marginal_cost(results: list[dict], key: str = "cpu_cores") -> float | None:
    """(worst - baseline) / tabs, per repeat, averaged; None without a baseline."""
    costs = []
    for rep in sorted({r.get("repeat", 1) for r in results}):
        rr = [r for r in results if r.get("repeat", 1) == rep and key in r]
        base = [r for r in rr if r["tabs"] == 0]
        per_tab = [r for r in rr if r["tabs"]]
        if not base or not per_tab:
            continue
        worst = max(per_tab, key=lambda r: r["tabs"])
        costs.append((worst[key] - base[0][key]) / worst["tabs"])
    return sum(costs) / len(costs) if costs else None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8088/?cmd=custom&page=Scalers")
    ap.add_argument("--process", default="mhttpd", help="process to profile")
    ap.add_argument("--experiment", default="WDSCALERS")
    ap.add_argument("--dropped-path",
                    default="/Equipment/WDWaveforms/Variables/Thread/DroppedPackets",
                    help="the DAQ's own loss counter; the gate that matters ('' = none)")
    ap.add_argument("--tabs", type=int, nargs="+", default=[0, 1, 2, 5],
                    help="numbers of open browser tabs (viewers) to measure")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--driver", choices=["gecko", "chromium"], default="gecko",
                    help="gecko: geckodriver + firefox; chromium: playwright, adds bytes/s")
    ap.add_argument("--container", default=None,
                    help="docker container whose /proc holds the processes")
    ap.add_argument("--click", action="append", default=[],
                    help="CSS selector to click after load (repeatable, in order)")
    ap.add_argument("--analyzer-client", default=None,
                    help="also measure the process started with --client NAME")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--settle", type=float, default=6.0,
                    help="seconds between opening the pages and starting the clock")
    ap.add_argument("--label", default=None, help="stored in every JSON row")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    proc = Proc(args.container)

    pid = proc.pid_of(args.process)
    if not pid:
        print(f"error: {args.process} is not running", file=sys.stderr)
        return 2
    apid = None
    if args.analyzer_client:
        apid = proc.pid_of_client(args.analyzer_client)
        if not apid:
            print(f"error: no process started with --client {args.analyzer_client}",
                  file=sys.stderr)
            return 2

    pw_cm = None
    if args.driver == "gecko":
        driver = shutil.which("geckodriver")
        if not driver:
            print("error: geckodriver not found", file=sys.stderr)
            return 2
        viewers = GeckoViewers(driver)
    else:
        from playwright.sync_api import sync_playwright
        pw_cm = sync_playwright()
        viewers = ChromiumViewers(pw_cm.__enter__())

    extra = bool(apid) or args.driver == "chromium"
    print(f"profiling {args.process} (pid {pid}) for {args.seconds:.0f}s per point\n")
    print(f"{'tabs':>5} {'CPU (cores)':>12} {'vs idle':>9} {'RSS MB':>8} {'dropped':>8}"
          + (f" {'analyzer':>9} {'kB/s/viewer':>12} {'load':>5}" if extra else ""))

    results = []
    try:
        for rep in range(1, args.repeat + 1):
            baseline = None
            for n in args.tabs:
                try:
                    viewers.open(n, args.url, args.click)
                    if n:
                        # Let the pages finish discovery before the clock starts, so we
                        # measure steady-state polling and not one-off page build.
                        viewers.wait(args.settle)

                    r = measure(proc, pid, args.seconds, args.experiment, args.dropped_path,
                                viewers if n else None, apid)
                    r["tabs"] = n
                    if args.repeat > 1 or args.label:
                        r["repeat"] = rep
                    if args.label:
                        r["label"] = args.label
                    results.append(r)
                    if baseline is None:
                        baseline = r["cpu_cores"]
                    delta = r["cpu_cores"] - baseline
                    dropped = "-" if r["dropped_delta"] is None else str(r["dropped_delta"])
                    line = (f"{n:>5} {r['cpu_cores']:>12.4f} {delta:>+9.4f} "
                            f"{r['rss_mb']:>8.1f} {dropped:>8}")
                    if extra:
                        a = r.get("analyzer_cpu_cores")
                        b = r.get("bytes_per_s_per_viewer")
                        line += (f" {'-' if a is None else f'{100 * a:.1f}%':>9}"
                                 f" {'-' if b is None else f'{b / 1e3:.1f}':>12}"
                                 f" {r['load1_after'] or 0:>5.1f}")
                    print(line, flush=True)
                finally:
                    viewers.close()
    finally:
        if pw_cm is not None:
            viewers.shutdown()
            pw_cm.__exit__(None, None, None)

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))
        print(f"\nwrote {args.json}")

    cost = marginal_cost(results)
    if cost is not None:
        print(f"\nmarginal cost: {cost * 100:.2f}% of one core per open tab")
        acost = marginal_cost(results, "analyzer_cpu_cores")
        if acost is not None:
            print(f"analyzer: {acost * 100:.2f}% of one core per open tab")
        bad = [r for r in results if r["dropped_delta"]]
        if bad:
            print("WARNING: the DAQ dropped packets during this measurement.")
            return 1
        if args.dropped_path:
            print("no packets dropped at any tab count")
    return 0


if __name__ == "__main__":
    sys.exit(main())
