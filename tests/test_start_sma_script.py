"""scripts/start-sma-analyzer.sh: option handling and the duplicate-start guards.

Runs the real script with a fake odbedit and a fake python on PATH, in --dry-run
(checks + the command it would run, nothing started), so no MIDAS is needed.
"""
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "start-sma-analyzer.sh"

FAKE_ODBEDIT = textwrap.dedent("""\
    #!/bin/bash
    # Prints /System/Clients the way `odbedit -q -c "ls -lr /System/Clients"` does,
    # from "$FAKE_CLIENTS" (lines: pid host name...). FAKE_ODB_DOWN=1: unreachable.
    [ -n "${FAKE_ODB_DOWN:-}" ] && { echo "Experiment not defined"; exit 1; }
    echo "Key name                        Type    #Val  Size  Last Opn Mode Value"
    echo "---------------------------------------------------------------------------"
    echo "Clients                         DIR"
    while read -r pid host name; do
        [ -n "$pid" ] || continue
        printf '    %-28sDIR\\n' "$pid"
        printf '        Name                    STRING  1     32    1h   0  R     %s\\n' "$name"
        printf '        Host                    STRING  1     256   1h   0  R     %s\\n' "$host"
    done < "${FAKE_CLIENTS:-/dev/null}"
    """)

# Stands in for the python preflight, which prints the skip method last.
FAKE_PYTHON = "#!/bin/sh\necho bm_skip_event\n"


@pytest.fixture
def rig(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, text in (("odbedit", FAKE_ODBEDIT), ("fakepy", FAKE_PYTHON)):
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    midassys = tmp_path / "midas"
    (midassys / "lib").mkdir(parents=True)
    (midassys / "lib" / "libmidas-c-compat.so").write_text("")
    clients = tmp_path / "clients"
    clients.write_text("")
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "MIDASSYS": str(midassys),
        "MIDAS_EXPT_NAME": "TESTEXPT",
        "MDQM_PYTHON": "fakepy",
        "FAKE_CLIENTS": str(clients),
    }

    def run(*args, **extra_env):
        e = dict(env)
        for k, v in extra_env.items():
            if v is None:
                e.pop(k, None)
            else:
                e[k] = v
        return subprocess.run(["bash", str(SCRIPT), *args], env=e, capture_output=True,
                              text=True, timeout=60)

    run.clients = clients
    run.tmp = tmp_path
    return run


def command_of(proc):
    assert proc.returncode == 0, proc.stderr
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("command:")]
    assert lines, proc.stdout
    return lines[0].split()[1:]


def dead_pid():
    pid = 4_000_000
    while os.path.exists(f"/proc/{pid}"):
        pid += 1
    return pid


def test_syntax_and_help(rig):
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0
    proc = rig("--help")
    assert proc.returncode == 0
    for word in ("--stop", "--status", "--foreground", "--mem-limit", "--cpu-pin", "--session"):
        assert word in proc.stdout


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_shellcheck_clean():
    proc = subprocess.run(["shellcheck", str(SCRIPT)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout


def test_default_command(rig):
    proc = rig("--dry-run")
    cmd = command_of(proc)
    for var in ("OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
                "MALLOC_ARENA_MAX=2"):
        assert var in cmd
    assert cmd[cmd.index("nice"):cmd.index("nice") + 3] == ["nice", "-n", "19"]
    assert "--as=1073741824" in cmd
    assert "taskset" not in cmd
    pp = next(c for c in cmd if c.startswith("PYTHONPATH="))
    assert pp.split("=", 1)[1].split(":")[0] == str(SCRIPT.parents[1] / "src")
    tail = cmd[cmd.index("-m"):]
    assert tail == ["-m", "mdqm.dqm.analyzer", "--experiment", "TESTEXPT", "--plugin", "sma",
                    "--client", "sma_analyzer"]
    assert "session: sma-analyzer" in proc.stdout
    assert "TESTEXPT-sma_analyzer.log" in proc.stdout


def test_options_and_passthrough(rig):
    cmd = command_of(rig("--dry-run", "--cpu-pin", "3", "--mem-limit=0", "--client", "sma2",
                         "--", "--rate", "5", "--no-cpu-budget"))
    assert cmd[cmd.index("taskset"):cmd.index("taskset") + 3] == ["taskset", "-c", "3"]
    assert not any(c.startswith("--as=") for c in cmd)
    assert cmd[cmd.index("--client") + 1] == "sma2"
    assert cmd[-3:] == ["--rate", "5", "--no-cpu-budget"]


def test_environment_fallbacks(rig):
    # WDS_* (wavedream-scalar-readout's environment) when the MIDAS/MDQM ones are unset.
    proc = rig("--dry-run", MIDAS_EXPT_NAME=None, MDQM_PYTHON=None, WDS_EXPT_NAME="WDEXPT",
               WDS_PYTHON="fakepy", WDS_TMUX_PREFIX="wds")
    cmd = command_of(proc)
    assert cmd[cmd.index("--experiment") + 1] == "WDEXPT"
    assert cmd[cmd.index("-m") - 1].endswith("/fakepy")
    assert "session: wds-sma-analyzer" in proc.stdout
    # Flags win over the environment.
    proc = rig("--dry-run", "--experiment", "other", "--session", "mine", "--log", "none")
    cmd = command_of(proc)
    assert cmd[cmd.index("--experiment") + 1] == "other"
    assert "session: mine" in proc.stdout and "log: none" in proc.stdout


def test_errors(rig):
    proc = rig("--dry-run", MIDAS_EXPT_NAME=None)
    assert proc.returncode != 0 and "no experiment" in proc.stderr
    proc = rig("--bogus")
    assert proc.returncode != 0 and "unknown option" in proc.stderr
    proc = rig("--stop=yes")
    assert proc.returncode != 0 and "takes no value" in proc.stderr
    proc = rig("--dry-run", "--mem-limit", "1G")
    assert proc.returncode != 0 and "--mem-limit" in proc.stderr
    proc = rig("--dry-run", "--python", "no-such-python")
    assert proc.returncode != 0 and "not found" in proc.stderr


def test_refuses_when_client_attached(rig):
    rig.clients.write_text(f"{os.getpid()} localhost sma_analyzer\n")
    proc = rig("--dry-run")
    assert proc.returncode == 1
    assert "already attached" in proc.stderr and "--stop" in proc.stderr
    # Attached from another host: the PID cannot be checked here, so it counts.
    rig.clients.write_text("123 daq-other-host sma_analyzer\n")
    proc = rig("--dry-run")
    assert proc.returncode == 1 and "daq-other-host" in proc.stderr


def test_stale_entry_is_ignored(rig):
    # A client killed without deregistering stays listed; its PID is gone.
    rig.clients.write_text(f"{dead_pid()} localhost sma_analyzer\n"
                           f"{os.getpid()} localhost wd_analyzer\n")
    assert rig("--dry-run").returncode == 0
    proc = rig("--status")
    assert proc.returncode == 1 and "not attached" in proc.stdout


def test_refuses_when_odb_unreadable(rig):
    proc = rig("--dry-run", FAKE_ODB_DOWN="1")
    assert proc.returncode == 1 and "Refusing to start blind" in proc.stderr
    assert rig("--status", FAKE_ODB_DOWN="1").returncode == 2


def test_refuses_when_an_unattached_analyzer_runs(rig):
    # An analyzer in its reconnect loop is not in /System/Clients but still counts.
    fake = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)",
                             "mdqm.dqm.analyzer", "--experiment", "TESTEXPT",
                             "--client", "sma_analyzer"])
    try:
        proc = rig("--dry-run")
        assert proc.returncode == 1 and "already running here" in proc.stderr
        # A different experiment or client name is someone else's.
        assert rig("--dry-run", "--experiment", "OTHER").returncode == 0
        assert rig("--dry-run", "--client", "sma_other").returncode == 0
    finally:
        fake.kill()
        fake.wait()


def test_stop_when_nothing_runs(rig):
    proc = rig("--stop")
    assert proc.returncode == 0 and "was not running" in proc.stdout
