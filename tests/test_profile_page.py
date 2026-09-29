"""scripts/profile-page.py: WaveDREAM defaults unchanged, /proc parsing, container reads."""
import importlib.util
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "profile-page.py"
spec = importlib.util.spec_from_file_location("profile_page", SCRIPT)
pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pp)


def test_wavedream_defaults_are_unchanged():
    a = pp.build_parser().parse_args([])
    assert a.url == "http://localhost:8088/?cmd=custom&page=Scalers"
    assert a.process == "mhttpd"
    assert a.experiment == "WDSCALERS"
    assert a.dropped_path == "/Equipment/WDWaveforms/Variables/Thread/DroppedPackets"
    assert a.tabs == [0, 1, 2, 5]
    assert a.seconds == 30.0
    assert a.json is None
    # The additions default to the original behaviour.
    assert a.driver == "gecko"
    assert a.container is None
    assert a.click == []
    assert a.analyzer_client is None
    assert a.repeat == 1
    assert a.settle == 6.0


def test_stat_ticks_survive_a_command_name_with_spaces_and_parens():
    fields = ["S"] + ["0"] * 10 + ["250", "50"] + ["0"] * 30
    line = "1234 (mdqm (x) y) " + " ".join(fields)
    assert pp.parse_ticks(line) == 3.0


def test_rss_is_read_from_status():
    text = "Name:\tmhttpd\nVmPeak:\t 999 kB\nVmRSS:\t  51200 kB\nThreads:\t4\n"
    assert pp.parse_rss_mb(text) == 50.0
    assert pp.parse_rss_mb("Name:\tx\n") == 0.0


class FakeRun:
    def __init__(self, stdout=""):
        self.calls = []
        self.stdout = stdout

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout=self.stdout, stderr="")


def test_container_reads_go_through_docker_exec():
    run = FakeRun("VmRSS:\t 2048 kB\n")
    p = pp.Proc("testbeam-midas", run=run)
    assert p.rss_mb(42) == 2.0
    assert run.calls[-1] == ["docker", "exec", "testbeam-midas", "cat", "/proc/42/status"]
    run.stdout = "77\n78\n"
    assert p.pid_of("mhttpd") == 77
    assert run.calls[-1] == ["docker", "exec", "testbeam-midas", "pgrep", "-x", "mhttpd"]
    assert p.pid_of_client("sma_analyzer") == 77
    assert run.calls[-1][-4:] == ["pgrep", "-f", "--", "--client sma_analyzer"]


def test_local_reads_do_not_use_docker(tmp_path):
    run = FakeRun("5\n")
    p = pp.Proc(None, run=run)
    f = tmp_path / "status"
    f.write_text("VmRSS:\t 1024 kB\n")
    assert p.read(str(f)).startswith("VmRSS")
    assert p.pid_of("mhttpd") == 5
    assert run.calls[-1] == ["pgrep", "-x", "mhttpd"]


def test_no_pid_when_nothing_matches():
    assert pp.Proc("c", run=FakeRun("")).pid_of("mhttpd") is None


def test_marginal_cost_matches_the_original_single_run_formula():
    rows = [{"tabs": 0, "cpu_cores": 0.005}, {"tabs": 1, "cpu_cores": 0.008},
            {"tabs": 5, "cpu_cores": 0.025}]
    assert abs(pp.marginal_cost(rows) - (0.025 - 0.005) / 5) < 1e-12


def test_marginal_cost_averages_repeats_against_their_own_baseline():
    rows = [{"tabs": 0, "cpu_cores": 0.01, "repeat": 1}, {"tabs": 2, "cpu_cores": 0.03, "repeat": 1},
            {"tabs": 0, "cpu_cores": 0.02, "repeat": 2}, {"tabs": 2, "cpu_cores": 0.06, "repeat": 2}]
    assert abs(pp.marginal_cost(rows) - (0.01 + 0.02) / 2) < 1e-12
    assert pp.marginal_cost([{"tabs": 1, "cpu_cores": 0.1}]) is None
    assert pp.marginal_cost(rows, "analyzer_cpu_cores") is None
