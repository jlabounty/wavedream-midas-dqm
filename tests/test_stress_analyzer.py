"""scripts/stress-analyzer.py finds the analyzer however it was started."""
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stress-analyzer.py"
spec = importlib.util.spec_from_file_location("stress_analyzer", SCRIPT)
stress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stress)


def test_console_script_is_found():
    # The installed entry point: the kernel names the process after the script.
    cmd = ["/venv/bin/python3", "/venv/bin/mdqm-analyzer", "--plugin", "sma",
           "--client", "sma_analyzer"]
    assert stress.analyzer_client("mdqm-analyzer", cmd) == (True, "sma_analyzer")


def test_python_module_is_found():
    cmd = ["python3", "-m", "mdqm.dqm.analyzer", "--client=wd_analyzer"]
    assert stress.analyzer_client("python3", cmd) == (True, "wd_analyzer")
    assert stress.analyzer_client("python3", ["python3", "-m", "mdqm.dqm.analyzer"]) == (True, None)


def test_other_processes_are_ignored():
    assert stress.analyzer_client("python3", ["python3", "replay-run.py", "x"]) is None
    assert stress.analyzer_client("bash", ["bash", "-c", "mdqm-analyzer"]) is None


def test_binary_reply_is_returned_raw():
    import struct
    payload = b"\xec\x00\x01binary smaf bytes"
    raw = struct.pack("<I4s", 8 + len(payload), b"smaf") + payload
    assert stress.decode_reply("sma::frame", raw) == payload


def test_json_reply_is_parsed():
    import struct
    body = b'{"events_seen": 3}'
    raw = struct.pack("<I4s", 8 + len(body), b"json") + body
    assert stress.decode_reply("dqm::status", raw) == {"events_seen": 3}


def test_unanswered_is_reported():
    import pytest
    with pytest.raises(stress.NotAnswered):
        stress.decode_reply("dqm::status", b'{"jsonrpc": "2.0", "error": {}}')


def test_missed_frames_come_from_the_plugin_serial_check():
    s0 = {"plugin": {"missed_by_serial": 10}}
    s1 = {"plugin": {"missed_by_serial": 70}}
    assert stress.missed_per_s(s0, s1, 30.0) == 2.0
    # The WaveDREAM plugin does not count serial gaps: no correction.
    assert stress.missed_per_s({"plugin": {}}, {"plugin": {}}, 30.0) is None
    assert stress.missed_per_s({}, {}, 30.0) is None
