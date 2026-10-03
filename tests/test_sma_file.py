"""The offline path: ``mdqm.tools.midasfile`` and ``mdqm-sma-file``.

The MIDAS files are built by hand here (``midasfile.encode_event``, checked
against a byte layout written out in full below), so no MIDAS is needed. The
lz4 test needs ``lz4``, the PNG test ``matplotlib``; both skip without them.
"""

from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from mdqm.dqm.hist import HistStore
from mdqm.plugins import sma as P
from mdqm.plugins import sma_words as W
from mdqm.tools import midasfile as MF
from mdqm.tools import sma_file as S

DATA = Path(__file__).resolve().parent / "data"
T0 = 1_790_000_000


def _golden(name):
    with np.load(DATA / name) as z:
        return [z[f"f{i}_words"] for i in range(len(z["labels"]))]


def h000(words):
    """An H000 bank as the FEB writes it: 64-bit words as TID_DWORD pairs."""
    return ("H000", 6, np.asarray(words, dtype="<u8").view("<u4"))


def write_file(path, frames, *, run=682, fmt="bank32a", extra=(), eor=True):
    """A MIDAS file: BOR (serial = run), one id-301 event per frame, EOR."""
    parts = [MF.encode_internal(MF.EVID_BOR, run, T0, b"<odb/>\n\0")]
    for i, words in enumerate(frames):
        parts.append(MF.encode_event(301, 100 + i, T0 + i // 50, [h000(words)], fmt=fmt))
    parts += list(extra)
    if eor:
        parts.append(MF.encode_internal(MF.EVID_EOR, run, T0 + len(frames) // 50,
                                        b"<odb/>\n\0"))
    blob = b"".join(parts)
    path = Path(path)
    if path.suffix == ".lz4":
        import lz4.frame
        with lz4.frame.open(path, "wb") as fh:
            fh.write(blob)
    elif path.suffix == ".gz":
        with gzip.open(path, "wb") as fh:
            fh.write(blob)
    else:
        path.write_bytes(blob)
    return path


def synth_frame(t0_ns, n=300, spacing_ns=3000, shift=14, seed=1):
    """A healthy frame: S1 every ~spacing, S2-S5 a few ns later, RF, current."""
    rng = np.random.default_rng(seed)
    t1 = t0_ns + np.arange(n, dtype=np.int64) * spacing_ns + rng.integers(0, 500, n)
    ts = [t1] + [t1 + c for c in (2, 3, 4, 5)] + [t1 + 40, t1 + 60, t1 + 1500]
    chs = [np.full(n, c) for c in (1, 2, 3, 4, 5, 6, 6, 7)]
    t = np.concatenate(ts)
    ch = np.concatenate(chs)
    o = np.argsort(t, kind="stable")
    coarse, fine = W.fields_of(t[o], shift)
    return W.encode(ch[o], np.full(t.size, 20), coarse, fine)


def synth_frames(k=8, shift=14):
    # One frame every 2 ms of board time, each ~0.9 ms long.
    return [synth_frame(10**9 + i * 2 * 10**6, shift=shift, seed=i) for i in range(k)]


class _FakeEvent:
    """What the analyzer gets from receive_event(use_numpy=True); as test_sma_plugin."""

    def __init__(self, words, serial):
        self.header = type("H", (), {"event_id": 301, "serial_number": serial})()
        self.banks = {"H000": type("B", (), {"data": np.asarray(words, "<u8").view("<u4")})()}

    def get_bank(self, name):
        return self.banks.get(name)


# --- the reader -------------------------------------------------------------------

def test_encode_event_matches_the_bank32a_layout_byte_for_byte():
    data = np.array([0x11223344, 0x55667788, 0x99AABBCC], dtype="<u4")      # 12 bytes
    got = MF.encode_event(301, 7, T0, [("H000", 6, data)], fmt="bank32a", trigger_mask=2)
    body = (struct.pack("<II", 16 + 16, 0x31)                # all-bank size, flags
            + b"H000" + struct.pack("<III", 6, 12, 0)         # name, TID_DWORD, size, pad
            + data.tobytes() + b"\0" * 4)                     # data padded to 16
    assert got == struct.pack("<HHIII", 301, 2, 7, T0, len(body)) + body


@pytest.mark.parametrize("fmt", ["bank16", "bank32", "bank32a"])
def test_reader_bank_formats_types_and_padding(tmp_path, fmt):
    banks = [("ABYT", 1, np.arange(3, dtype="<u1")),            # 3 bytes: 5 of padding
             ("AFLT", 9, np.array([1.5, -2.0], dtype="<f4")),
             ("ASHT", 5, np.array([-3, 4, 5], dtype="<i2")),    # 6 bytes
             ("ABOO", 8, np.array([0, 1, 7], dtype="<u4")),
             ("ASTR", 12, b"hello\0"),
             ("AQWD", 18, np.array([2**63 + 5], dtype="<u8"))]
    ev = MF.encode_event(5, 1, T0, banks, fmt=fmt)
    path = tmp_path / "x.mid"
    path.write_bytes(MF.encode_internal(MF.EVID_BOR, 42, T0, b"odb") + ev + ev)
    events = list(MF.MidasFile(path))
    assert len(events) == 2
    e = events[0]
    assert e.header.event_id == 5 and e.header.serial_number == 1 and e.header.timestamp == T0
    assert e.flags == {"bank16": 0x01, "bank32": 0x11, "bank32a": 0x31}[fmt]
    assert list(e.banks) == [b[0] for b in banks]
    assert e.bank_error is None
    np.testing.assert_array_equal(e.get_bank("ABYT").data, [0, 1, 2])
    assert e.get_bank("ABYT").data.dtype == np.uint8
    np.testing.assert_array_equal(e.get_bank("AFLT").data, [1.5, -2.0])
    np.testing.assert_array_equal(e.get_bank("ASHT").data, [-3, 4, 5])
    assert e.get_bank("ABOO").data.dtype == np.bool_
    np.testing.assert_array_equal(e.get_bank("ABOO").data, [False, True, True])
    assert e.get_bank("ASTR").data == b"hello\0"
    assert int(e.get_bank("AQWD").data[0]) == 2**63 + 5
    assert e.get_bank("ASHT").size_bytes == 6 and e.get_bank("ASHT").type == 5
    assert e.get_bank("NONE") is None


def test_reader_skips_internal_events_and_filters_ids(tmp_path):
    frames = _golden("sma_run00682_frames.npz")[:2]
    other = MF.encode_event(401, 9, T0, [("WDXX", 6, np.arange(4, dtype="<u4"))])
    msg = MF.encode_internal(MF.EVID_MSG, 0, T0, b"a message\0")
    path = write_file(tmp_path / "run00682_00005.mid", frames, extra=[other, msg])

    r = MF.MidasFile(path)
    ids = [e.header.event_id for e in r]
    assert ids == [301, 301, 401]
    assert r.bor_run_number == 682
    assert r.events_read == 6 and not r.truncated

    r = MF.MidasFile(path, event_ids={301})
    assert [e.header.serial_number for e in r] == [100, 101]

    r = MF.MidasFile(path, event_ids={301}, include_internal=True)
    got = [(e.header.event_id, e.header.is_midas_internal_event()) for e in r]
    assert got == [(0x8000, True), (301, False), (301, False), (0x8002, True), (0x8001, True)]


def test_reader_on_golden_frames_gives_the_words_back(tmp_path):
    frames = _golden("sma_run00682_frames.npz") + _golden("sma_run01008_frame.npz")
    path = write_file(tmp_path / "g.mid", frames)
    got = [e.get_bank("H000") for e in MF.MidasFile(path, event_ids={301})]
    assert len(got) == len(frames)
    for bank, words in zip(got, frames, strict=True):
        assert bank.data.dtype == np.uint32 and bank.type == 6
        np.testing.assert_array_equal(bank.data.view("<u8"), words)


def test_reader_reports_a_truncated_file(tmp_path):
    frames = _golden("sma_run00682_frames.npz")[:3]
    full = write_file(tmp_path / "full.mid", frames, eor=False).read_bytes()
    cut = tmp_path / "cut.mid"
    cut.write_bytes(full[:-100])
    r = MF.MidasFile(cut, event_ids={301})
    assert len(list(r)) == 2
    assert r.truncated


def test_reader_rejects_a_non_midas_file(tmp_path):
    p = tmp_path / "junk.mid"
    p.write_bytes(b"\xff" * 64)
    with pytest.raises(ValueError, match="implausible event size"):
        list(MF.MidasFile(p))


def test_reader_reports_a_bank_overrunning_its_event(tmp_path):
    ev = bytearray(MF.encode_event(301, 1, T0, [("H000", 6, np.arange(4, dtype="<u4"))]))
    struct.pack_into("<I", ev, 16 + 8 + 8, 1000)              # the bank's size field
    p = tmp_path / "bad.mid"
    p.write_bytes(bytes(ev))
    (e,) = list(MF.MidasFile(p))
    assert e.get_bank("H000") is None and "overruns" in e.bank_error


def test_gzip_file(tmp_path):
    frames = _golden("sma_run00682_frames.npz")[:2]
    path = write_file(tmp_path / "run00682_00005.mid.gz", frames)
    assert len(list(MF.MidasFile(path, event_ids={301}))) == 2


def test_lz4_file(tmp_path):
    pytest.importorskip("lz4")
    frames = _golden("sma_run00682_frames.npz")
    path = write_file(tmp_path / "run00682_00005.mid.lz4", frames)
    r = MF.MidasFile(path, event_ids={301}, chunk=4096)       # many refills
    got = [e.get_bank("H000").data.view("<u8") for e in r]
    assert r.bor_run_number == 682 and not r.truncated
    for a, b in zip(got, frames, strict=True):
        np.testing.assert_array_equal(a, b)


def test_run_subrun_from_name():
    assert MF.run_subrun_from_name("/x/run01008_00001.mid.lz4") == (1008, 1)
    assert MF.run_subrun_from_name("run00682.mid") == (682, None)
    assert MF.run_subrun_from_name("data.mid") == (None, None)


# --- the adapter is faithful --------------------------------------------------------

def test_file_events_fill_exactly_what_daemon_events_fill(tmp_path):
    """The same frames through the file reader and through the analyzer's event shape."""
    frames = _golden("sma_run00682_frames.npz") + synth_frames(4)
    path = write_file(tmp_path / "run00682_00000.mid", frames)
    settings = S.build_settings()

    a, _ = S.feed(MF.MidasFile(path), settings, 682, clock=S.DataClock(T0))

    b = S.make_plugin(settings, S.DataClock(T0))
    for i, words in enumerate(frames):
        assert b.process(_FakeEvent(words, 100 + i), run_number=682)

    assert a.store.names() == b.store.names()
    for name in a.store.names():
        ha, hb = a.store.get(name), b.store.get(name)
        np.testing.assert_array_equal(ha.counts, hb.counts, err_msg=name)
        assert ha.entries == hb.entries, name
    assert a.frames == b.frames == len(frames)
    assert a.frames_stale == b.frames_stale == 2


# --- settings -----------------------------------------------------------------------

def test_build_settings_is_the_defaults_plus_overrides():
    s = S.build_settings()
    assert s == P.SETTINGS_DEFAULTS and s is not P.SETTINGS_DEFAULTS
    s = S.build_settings({"Cuts": {"coinc window ns": 30}}, shift=3)
    assert s["Cuts"]["coinc window ns"] == 30 and s["Coarse shift"] == 3
    assert s["Cuts"]["dt window ns"] == P.SETTINGS_DEFAULTS["Cuts"]["dt window ns"]
    assert P.SETTINGS_DEFAULTS["Coarse shift"] == W.DEFAULT_SHIFT          # untouched
    with pytest.raises(KeyError, match="Cuts/coinc windw ns"):
        S.build_settings({"Cuts": {"coinc windw ns": 30}})
    with pytest.raises(KeyError, match="directory"):
        S.build_settings({"Cuts": 3})


def test_offline_summary_never_flags_no_frames():
    clock = S.DataClock(T0)
    p = S.make_plugin(S.build_settings(), clock)
    assert "no_frames" not in {f["code"] for f in S.offline_summary(p)["flags"]}
    p.process(_FakeEvent(synth_frame(10**9), 1), run_number=1)
    clock.t += 100                                   # long after the last frame
    s = S.offline_summary(p)
    assert s["run_active"] is False
    assert "no_frames" not in {f["code"] for f in s["flags"]}
    # The same plugin, asked as the page asks during a run, does flag it.
    live = S.command_json(p, "sma::summary", {"run_active": True})
    assert "no_frames" in {f["code"] for f in live["flags"]}


# --- the CLI ------------------------------------------------------------------------

def _run_cli(tmp_path, frames, *args, name="run00682_00005.mid", run=682):
    path = write_file(tmp_path / name, frames, run=run)
    out = tmp_path / "out"
    rc = S.main([str(path), "--out", str(out), *args])
    return rc, out


def test_cli_healthy_file(tmp_path, capsys):
    # The synthetic frames are cabled without NIM copies (every NIM channel of
    # the defaults but ch 3 would be flagged nim_missing).
    rc, out = _run_cli(tmp_path, synth_frames(8), "--no-png", "--settings",
                       '{"NIM": {"channels": [-1]}}')
    assert rc == S.EXIT_OK
    assert sorted(p.name for p in out.iterdir()) == ["hists.npz", "summary.json", "trend.json"]

    summary = json.loads((out / "summary.json").read_text())
    assert summary["run"] == 682 and summary["run_active"] is False
    assert summary["frames"]["processed"] == 8
    assert summary["shift"]["verdict"] == "ok"
    assert not [f for f in summary["flags"] if f["severity"] == "error"]

    with np.load(out / "hists.npz") as z:
        keys = set(z.files)
        meta = json.loads(str(z["__meta__"]))
        wt = z["sma/word_types"]
        assert int(z["sma/word_types.entries"]) == sum(f.size for f in synth_frames(8))
        np.testing.assert_array_equal(z["sma/rate_vs_ch.axes"][0], [16, -0.5, 15.5])
        assert z["sma/rate_vs_ch"].shape == (140 + 2, 16 + 2)
    names = {k for k in keys if "." not in k and k != "__meta__"}
    assert "sma/word_types" in names and "sma/dt_S2_S1" in names and "sma/shift_check" in names
    for n in names:
        assert {f"{n}.entries", f"{n}.dropped", f"{n}.axes"} <= keys
    assert wt[3] == sum(f.size for f in synth_frames(8))           # all "trigger kept"
    assert meta["run"] == 682 and meta["subrun"] == 5 and meta["frames_fed"] == 8
    assert {h["name"] for h in meta["histograms"]} == names

    trend = json.loads((out / "trend.json").read_text())
    assert trend["counters"] == ["S2", "S3", "S4", "S5"]
    text = capsys.readouterr().out
    assert "run 682, 8 SMA frames processed" in text and "flags: none" in text
    assert "suspect 0" in text and "timed efficiency given S1: S2 100.0%" in text


def test_withheld_efficiency_shows_na_with_the_reason(capsys):
    """A counter with a fine-bit fault gets eff null + reason; the CLI says n/a (why)."""
    p = S.make_plugin(S.build_settings(), S.DataClock(T0))
    for i, words in enumerate(synth_frames(8)):
        w = np.asarray(words, dtype=np.uint64).copy()
        s2 = ((w >> np.uint64(56)) & np.uint64(0xF)) == 2
        fine = w & np.uint64(0xFFFFF)
        w[s2] = (w[s2] & ~np.uint64(0xFFFFF)) | ((fine[s2] >> np.uint64(1)) ^ np.uint64(1 << 19))
        p.process(_FakeEvent(w, i), run_number=1)
    summary = S.offline_summary(p)
    e2 = summary["efficiency"][0]
    assert e2["counter"] == "S2" and e2["eff"] is None and e2["reason"]
    stats = S.FeedStats(frames_fed=8, t_first=T0, t_last=T0)
    text = S.text_summary(summary, stats, "f", 0.1)
    assert f"S2 n/a ({e2['reason']})" in text
    page = " ".join(ln.strip() for ln in S.figure_text(summary))     # wrapped lines
    assert f"S2 n/a ({e2['reason']})" in page
    assert [r[0] for r in S.counter_roles(summary)] == ["S1", "S2", "S3", "S4", "S5"]


def test_cli_summary_json_is_exactly_the_command_output(tmp_path):
    frames = synth_frames(6)
    rc, out = _run_cli(tmp_path, frames, "--no-png", "--quiet")
    assert rc == 0
    plugin, _ = S.feed(MF.MidasFile(tmp_path / "run00682_00005.mid"), S.build_settings(), 682)
    assert json.loads((out / "summary.json").read_text()) == S.offline_summary(plugin)


def test_cli_wrong_shift_is_an_error_exit(tmp_path, capsys):
    rc, out = _run_cli(tmp_path, synth_frames(8), "--no-png", "--shift", "12")
    assert rc == S.EXIT_ERROR_FLAG
    summary = json.loads((out / "summary.json").read_text())
    codes = {f["code"] for f in summary["flags"] if f["severity"] == "error"}
    assert "shift_mismatch" in codes
    assert summary["shift"]["configured"] == 12 and summary["shift"]["best"] == 14
    assert "ERROR shift_mismatch" in capsys.readouterr().out


def test_cli_settings_frames_and_skip(tmp_path):
    rc, out = _run_cli(tmp_path, synth_frames(8), "--no-png", "--quiet", "--skip", "2",
                       "--frames", "3", "--settings", '{"Coarse shift": 14}')
    assert rc == 0
    s = json.loads((out / "summary.json").read_text())
    assert s["frames"]["processed"] == 3
    # The serial baseline starts at the first processed frame: skipping is not a loss.
    assert s["frames"]["missed_by_serial"] == 0
    with np.load(out / "hists.npz") as z:
        meta = json.loads(str(z["__meta__"]))
    assert (meta["frames_seen"], meta["frames_skipped"], meta["frames_fed"]) == (6, 2, 3)


def test_cli_settings_file(tmp_path):
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"Coarse shift": 12}))
    rc, out = _run_cli(tmp_path, synth_frames(8), "--no-png", "--quiet", "--settings", str(f))
    assert rc == S.EXIT_ERROR_FLAG
    assert json.loads((out / "summary.json").read_text())["shift"]["configured"] == 12


def test_cli_usage_errors(tmp_path, capsys):
    assert S.main([str(tmp_path / "missing.mid")]) == S.EXIT_USAGE
    rc, _ = _run_cli(tmp_path, synth_frames(1), "--settings", '{"Nope": 1}')
    assert rc == S.EXIT_USAGE
    assert "unknown setting 'Nope'" in capsys.readouterr().err
    rc, _ = _run_cli(tmp_path, synth_frames(1), "--shift", "99")
    assert rc == S.EXIT_USAGE


def test_cli_default_out_dir(tmp_path, monkeypatch):
    path = write_file(tmp_path / "run01008_00001.mid", synth_frames(2), run=1008)
    monkeypatch.chdir(tmp_path)
    assert S.main([str(path), "--no-png", "--quiet"]) == 0
    assert (tmp_path / "sma-file-1008_1" / "summary.json").is_file()


def test_cli_file_without_sma_frames(tmp_path, capsys):
    rc, out = _run_cli(tmp_path, [], "--no-png")
    assert rc == 0
    assert json.loads((out / "summary.json").read_text())["frames"]["processed"] == 0
    assert "no SMA readout frames" in capsys.readouterr().out


def test_cli_png(tmp_path):
    pytest.importorskip("matplotlib")
    rc, out = _run_cli(tmp_path, synth_frames(8))
    assert rc == 0
    assert (out / "summary.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_summary_figure_draws_with_errors_and_no_data():
    pytest.importorskip("matplotlib")
    p = S.make_plugin(S.build_settings(shift=12), S.DataClock(T0))
    for i, words in enumerate(synth_frames(8)):
        p.process(_FakeEvent(words, i), run_number=1)
    fig = S.summary_figure(S.offline_summary(p), p.store, "t")
    fig.canvas.draw()
    empty = S.make_plugin(S.build_settings(), S.DataClock(T0))
    S.summary_figure(S.offline_summary(empty), empty.store, "t").canvas.draw()


def test_hist_arrays_cover_every_histogram():
    store = HistStore()
    p = P.SmaPlugin(store)
    arrays = S.hist_arrays(p.store)
    for name in store.names():
        assert arrays[name].shape == store.get(name).counts.shape
