"""WaveDream: turn events into the quantities worth histogramming.

The only experiment-specific module in the analyzer. Decoding is reused from
``wdscalers.wdunpack``, which is the reference implementation of these bank
formats; what lives here is the *analysis* -- the pulse quantities and the RF
phase measurement.

Those algorithms and their tuned constants are ported from the retired C++ DQM
plugin (``wd_dqm_plugin/src/.../decode/wd_decode.cpp`` and
``stages/wd_rf_phase_stage.cpp``), which is the only place they ever existed.
They are reproduced deliberately faithfully, including the cuts, because the
numbers were arrived at by looking at real beam and not by reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mdqm.dqm import framing
from mdqm.dqm.framing import VOLTS_SCALE
from mdqm.dqm.hist import Axis, Hist1D, Hist2D

# --- tuned constants, from the retired plugin --------------------------------
# Changing any of these changes what the plots mean; they are not free knobs.
BASELINE_SAMPLES = 100          # wd_amplitude_spectrum_stage.cpp
MIN_S1_AMPLITUDE_V = 0.01       # wd_rf_phase_stage.cpp:30
MIN_RF_PEAK_TO_PEAK_V = 0.02    # :31
MAX_PERIOD_JITTER_FRAC = 0.25   # :32
MIN_RF_CROSSINGS = 5            # :90
MIN_RF_PERIOD_SAMPLES = 2.0     # :95
EDGE_FRACTION = 0.5             # the 50% leading edge

DAQ_CLOCK_HZ = 80e6

_decode_event = None


def _decoder():
    """The reference decoder, with a useful message if it is not installed.

    Imported lazily and by hand rather than at module scope so that the failure
    is a sentence an operator can act on instead of a ModuleNotFoundError from
    three frames down. The decoders deliberately live in the readout repository
    -- they are the reference implementation of these bank formats and a second
    copy here would fork on the first bug fix.
    """
    global _decode_event
    if _decode_event is None:
        try:
            from wdscalers.wdunpack import decode_event
        except ImportError as exc:
            raise ImportError(
                "the WaveDream decoders are not importable: "
                f"{exc}.\n"
                "They live in the wavedream-scalar-readout repository, which this "
                "plugin depends on. Install it into the same environment:\n"
                "    pip install -e /path/to/wavedream-scalar-readout\n"
                "or install this package with its extra:\n"
                "    pip install -e '.[wavedream]'"
            ) from exc
        _decode_event = decode_event
    return _decode_event


def rising_crossings(v: np.ndarray) -> np.ndarray:
    """Sample positions where `v` crosses its own mean going upwards.

    Linear interpolation between the bracketing samples, so the returned
    positions are fractional. Port of `wd_dqm::risingCrossings`.
    """
    if v.size < 4:
        return np.empty(0)
    d = np.asarray(v, dtype=np.float64) - float(np.mean(v))
    a, b = d[:-1], d[1:]
    hit = np.flatnonzero((a < 0) & (b >= 0))
    if hit.size == 0:
        return np.empty(0)
    return hit + a[hit] / (a[hit] - b[hit])


@dataclass
class Pulse:
    """The 50% leading edge of a negative-going pulse, and its amplitude."""

    baseline: float
    amplitude: float
    time: float          # sample position, or -1 if no usable edge
    min_index: int

    @property
    def valid(self) -> bool:
        return self.time >= 0


def leading_edge(v: np.ndarray, baseline_samples: int = BASELINE_SAMPLES,
                 frac: float = EDGE_FRACTION) -> Pulse | None:
    """Port of `wd_dqm::leadingEdgeTime`, amplitude and baseline included.

    Searches *backwards* from the minimum for the fraction-of-amplitude
    crossing, which is what makes it robust against ringing after the pulse.
    """
    v = np.asarray(v, dtype=np.float64)
    if baseline_samples < 1 or v.size < baseline_samples + 2:
        return None

    baseline = float(np.mean(v[:baseline_samples]))
    min_index = int(np.argmin(v))
    amplitude = baseline - float(v[min_index])
    if min_index == 0:
        # The minimum is the first sample: there is no rising edge in the record
        # to measure, so report the amplitude but no time.
        return Pulse(baseline, amplitude, -1.0, min_index)

    thr = baseline - frac * amplitude
    seg = v[: min_index + 1]
    crossed = np.flatnonzero((seg[:-1] >= thr) & (seg[1:] < thr))
    if crossed.size == 0:
        return Pulse(baseline, amplitude, -1.0, min_index)

    i = int(crossed[-1])          # the last crossing before the minimum
    t = i + (seg[i] - thr) / (seg[i] - seg[i + 1])
    return Pulse(baseline, amplitude, float(t), min_index)


@dataclass
class RfResult:
    valid: bool = False
    period_samples: float = 0.0
    phase_deg: float = 0.0
    s1_time: float = 0.0
    s1_amplitude: float = 0.0
    reason: str = ""


def rf_phase(s1: np.ndarray, rf: np.ndarray,
             baseline_samples: int = BASELINE_SAMPLES) -> RfResult:
    """S1's leading edge as a phase within the RF period.

    Every rejection carries a reason, which the retired version did not: a plot
    that is simply empty tells nobody whether the RF is missing, aperiodic, or
    the S1 pulse is too small to trust.
    """
    rf = np.asarray(rf, dtype=np.float64)
    if rf.size == 0:
        return RfResult(reason="no RF channel")
    if float(np.max(rf) - np.min(rf)) < MIN_RF_PEAK_TO_PEAK_V:
        return RfResult(reason="RF peak-to-peak below threshold")

    crossings = rising_crossings(rf)
    if crossings.size < MIN_RF_CROSSINGS:
        return RfResult(reason=f"only {crossings.size} RF crossings")

    diffs = np.diff(crossings)
    period = float(np.mean(diffs))
    if period < MIN_RF_PERIOD_SAMPLES:
        return RfResult(reason="RF period below two samples")
    if np.any(np.abs(diffs - period) > MAX_PERIOD_JITTER_FRAC * period):
        return RfResult(reason="RF period jitter too large")

    pulse = leading_edge(s1, baseline_samples, EDGE_FRACTION)
    if pulse is None or not pulse.valid:
        return RfResult(reason="no S1 leading edge")
    if pulse.amplitude < MIN_S1_AMPLITUDE_V:
        return RfResult(reason="S1 amplitude below threshold")

    frac = (pulse.time - float(crossings[0])) / period % 1.0
    return RfResult(True, period, 360.0 * frac, pulse.time, pulse.amplitude)


class WaveDreamPlugin:
    """Decode waveform events and fill the standard plot set."""

    name = "wavedream"
    #: Both the current IDs and the pre-2026-07-30 ones, for the same reason
    #: wdunpack accepts both: a run file records its IDs but not who wrote it.
    event_ids = frozenset({401, 1})
    #: The WaveDREAM frontend's loss counter: the analyzer halves its own rate
    #: when this rises.
    dropped_path = "/Equipment/WDWaveforms/Variables/Thread/DroppedPackets"

    def __init__(self, store, roles: dict | None = None, binning: dict | None = None):
        self.store = store
        self.last_rf = None
        self.roles = roles or {}
        self.binning = binning or {}
        self.widths = None            # the run's DRS cell-width table
        self.widths_from_run = None
        self.last_ticks = None
        self.frame = None             # the most recent decoded event
        self.rf_rejections: dict[str, int] = {}
        self.decoded = 0
        self._build()

    # -- setup ---------------------------------------------------------------

    def _role(self, key, default):
        v = self.roles.get(key, default)
        return v if v is not None else default

    def reconfigure(self, roles: dict, binning: dict) -> None:
        """Adopt new roles and binning, rebuilding the histograms they define.

        The rebuild *resets* the affected plots, and that is correct rather than
        unfortunate: a histogram with different bins is a different histogram,
        and carrying old counts into new bins would silently mix two binnings in
        one plot -- a plot that looks fine and means nothing. The page shows the
        entry count, so the reset is visible.

        Plots whose shape did not change are left alone, so moving a channel role
        does not throw away an afternoon of persistence.
        """
        self.roles = roles or {}
        self.binning = binning or {}
        for name in self.store.names():
            if name.startswith("wd/"):
                self.store.remove(name)
        self._build()

    def _build(self):
        b = self.binning
        chans = list(self._role("waveform channels", [0, 1, 2, 3, 4]))

        for ch in chans:
            self.store.add(Hist2D(
                f"wd/persistence_ch{ch:02d}",
                Axis(int(b.get("persistence x bins", 256)), 0, 1024, "sample"),
                Axis(int(b.get("persistence y bins", 110)),
                     float(b.get("persistence y min", -1.0)),
                     float(b.get("persistence y max", 0.1)), "V"),
                title=f"Persistence, channel {ch}"))
            self.store.add(Hist1D(
                f"wd/amplitude_ch{ch:02d}",
                Axis(int(b.get("amplitude bins", 200)),
                     float(b.get("amplitude min", 0.0)),
                     float(b.get("amplitude max", 1.0)), "baseline - min (V)"),
                title=f"Amplitude, channel {ch}"))

        self.store.add(Hist1D(
            "wd/deltat",
            Axis(int(b.get("deltat bins", 200)), 0.0,
                 float(b.get("deltat max s", 0.1)), "s"),
            title="Time between sampled events"))
        self.store.add(Hist1D(
            "wd/rf_phase",
            Axis(int(b.get("phase bins", 72)), 0.0, 360.0, "phase (deg)"),
            title="S1 leading edge vs RF"))
        self.store.add(Hist2D(
            "wd/amp_vs_phase",
            Axis(int(b.get("phase bins", 72)), 0.0, 360.0, "phase (deg)"),
            Axis(int(b.get("amplitude bins", 100)),
                 float(b.get("amplitude min", 0.0)),
                 float(b.get("amplitude max", 1.0)), "baseline - min (V)"),
            title="S1 amplitude vs RF phase"))

    # -- per event -----------------------------------------------------------

    def accepts(self, event) -> bool:
        return event.header.event_id in self.event_ids

    def process(self, event, run_number=None) -> bool:
        """Decode and fill. Returns True if the event was ours and usable."""
        decode_event = _decoder()

        # A new run invalidates the width table: a calibration from the previous
        # run is not this run's, and silently reusing it would put a wrong time
        # axis on everything downstream.
        if run_number is not None and run_number != self.widths_from_run:
            if self.widths_from_run is not None:
                self.widths = None
            self.widths_from_run = run_number

        frame = decode_event(event, self.widths)
        if frame is None:
            return False
        if frame.cell_widths is not None:
            self.widths = frame.cell_widths

        self.frame = frame
        self.decoded += 1
        self._fill(frame)
        return True

    def _fill(self, frame):
        chans = list(self._role("waveform channels", [0, 1, 2, 3, 4]))

        for ch in chans:
            v = frame.waveforms.get(ch)
            if v is None:                 # undecodable encoding mode
                continue
            v = np.asarray(v, dtype=np.float64)

            pers = self.store.get(f"wd/persistence_ch{ch:02d}")
            if pers is not None:
                pers.fill(np.arange(v.size, dtype=np.float64), v)

            amp = self.store.get(f"wd/amplitude_ch{ch:02d}")
            if amp is not None:
                pulse = leading_edge(v)
                if pulse is not None:
                    amp.fill([pulse.amplitude])

        self._fill_deltat(frame)
        self._fill_rf(frame)

    def _fill_deltat(self, frame):
        ticks = frame.timestamp_ticks
        if ticks is None:
            return
        if self.last_ticks is not None and ticks > self.last_ticks:
            dt = (ticks - self.last_ticks) / DAQ_CLOCK_HZ
            h = self.store.get("wd/deltat")
            if h is not None:
                h.fill([dt])
        # ticks <= last_ticks means the counter was zeroed at a run start (or we
        # sampled out of order). Re-baseline rather than filling a nonsense gap.
        self.last_ticks = ticks

    def _fill_rf(self, frame):
        s1_ch = int(self._role("s1 channel", 0))
        rf_ch = int(self._role("rf channel", 5))
        s1 = frame.waveforms.get(s1_ch)
        rf = frame.waveforms.get(rf_ch)
        if s1 is None or rf is None:
            self.rf_rejections["missing S1 or RF channel"] = \
                self.rf_rejections.get("missing S1 or RF channel", 0) + 1
            return

        result = rf_phase(np.asarray(s1), np.asarray(rf))
        if not result.valid:
            self.rf_rejections[result.reason] = self.rf_rejections.get(result.reason, 0) + 1
            return

        h = self.store.get("wd/rf_phase")
        if h is not None:
            h.fill([result.phase_deg])
        h2 = self.store.get("wd/amp_vs_phase")
        if h2 is not None:
            h2.fill([result.phase_deg], [result.s1_amplitude])
        self.last_rf = result

    # -- the scope frame -----------------------------------------------------

    def scope_frame(self, run_active: bool = False) -> bytes | None:
        """The most recent event, with the quantities derived from *that* event.

        One reply, deliberately. An event display is a thing people point at --
        "channel 3 looks odd on this one" -- so the traces and the phase printed
        beside them must come from the same event, and every screen must show the
        same event as every other. Both hold by construction with a single frame
        and by nothing at all if the page fetches traces and numbers separately.

        Samples go back as the int16 they arrived as. `decode_drsv` divided them
        by 1e4 to get volts, so multiplying back and rounding recovers them
        exactly -- an int16 is exactly representable in float32, so the division
        is the only rounding and it is undone. Sending float32 instead would
        double the payload to say the same thing.
        """
        f = self.frame
        if f is None:
            return None

        channels = []
        for ch in sorted(f.waveforms):
            volts = f.waveforms[ch]
            samples = None
            if volts is not None:
                samples = np.rint(np.asarray(volts, dtype=np.float64) / VOLTS_SCALE)
                samples = samples.astype(np.int16)
            channels.append({
                "channel": int(ch),
                "first_bin": int(f.first_bin.get(ch, 0)),
                "samples": samples,
                # 0 is the plain mode; anything else is why samples is None.
                "encoding": 0 if volts is not None else 11,
                "scale": VOLTS_SCALE,
            })

        derived = {}
        rf = getattr(self, "last_rf", None)
        if rf is not None and rf.valid:
            derived["rf_phase_deg"] = rf.phase_deg
            derived["rf_period_smp"] = rf.period_samples
            derived["s1_time_smp"] = rf.s1_time
            derived["s1_amp_v"] = rf.s1_amplitude
        # Per-channel amplitude, so the panels can be labelled without the page
        # re-deriving anything and possibly disagreeing with the histograms.
        for ch in list(self._role("waveform channels", [0, 1, 2, 3, 4]))[:8]:
            volts = f.waveforms.get(ch)
            if volts is None:
                continue
            pulse = leading_edge(np.asarray(volts, dtype=np.float64))
            if pulse is not None:
                derived[f"amp_ch{int(ch):02d}"] = pulse.amplitude

        return framing.encode_scope_frame(
            channels,
            frame_seq=self.decoded,
            run_number=self.widths_from_run or 0,
            event_number=f.event_number or 0,
            trigger_number=f.trigger_number or 0,
            timestamp_ticks=f.timestamp_ticks or 0,
            board_temp_c=f.temperature_c or 0.0,
            nominal_ps=(f.nominal_width_s or 0.0) * 1e12,
            board_id=f.board_id or 0,
            have_widths=f.cell_widths is not None,
            widths_cached=False,
            run_active=run_active,
            derived=derived,
        )

    # -- reporting -----------------------------------------------------------

    def status(self) -> dict:
        f = self.frame
        return {
            "plugin": self.name,
            "decoded": self.decoded,
            "have_drs_widths": self.widths is not None,
            "drs_widths_from_run": self.widths_from_run,
            "rf_rejections": dict(self.rf_rejections),
            "last_event": None if f is None else {
                "event_number": f.event_number,
                "board_id": f.board_id,
                "temperature_c": f.temperature_c,
                "channels": sorted(int(c) for c in f.waveforms),
                "undecoded": sorted(int(c) for c, w in f.waveforms.items() if w is None),
            },
        }
