"""The analyzer's live configuration, in the ODB.

Under ``/DQM/Analyzer``, deliberately, and **not** under ``/Equipment/...``:
the analyzer registers no equipment on purpose, because equipment would put it
in the run-transition path where a wedged monitoring process can delay a run
start. A settings tree under ``/Equipment/WDAnalyzer`` would imply equipment
that does not exist. ``/DQM/Analyzer`` sits beside ``/DQM/Scalars``, which the
scaler page already uses.

Everything here is seeded when absent and never overwritten, so an operator's
edits survive a restart. Changing a value takes effect within a couple of
seconds without restarting anything -- which for binning means the affected
histograms are rebuilt and therefore reset, because a histogram with different
bins is a different histogram and pretending otherwise would silently mix two
binnings in one plot.

The module-level defaults below are the WaveDREAM analyzer's. A second plugin
running as its own client brings its own root and defaults, and passes them to
`seed` and `read`; two analyzers sharing one tree would each rebuild on the
other's edits and fight over the sampling rate.
"""

from __future__ import annotations

import json

ROOT = "/DQM/Analyzer"

#: Which channels are what. Read by the analyzer and by the event display, from
#: one place -- the retired stack kept this in a JSON file that the C++ stages
#: and the browser loaded independently and could disagree about.
CHANNEL_ROLES: dict[str, object] = {
    "waveform channels": [0, 1, 2, 3, 4],
    "s1 channel": 0,
    "rf channel": 5,
    "nim channels": [7, 8, 9, 10, 11, 12, 13, 14, 15],
    "nim threshold V": 0.1,
    #: Per-channel names, 18 entries. Empty means "use ch NN".
    "labels": [""] * 18,
}

#: Histogram binning. Defaults are the values the retired C++ stages used, which
#: are the record of what operators found useful -- not arbitrary starting points.
BINNING: dict[str, object] = {
    "persistence x bins": 256,
    "persistence y bins": 110,
    "persistence y min": -1.0,
    "persistence y max": 0.1,
    "amplitude bins": 200,
    "amplitude min": 0.0,
    "amplitude max": 1.0,
    "deltat bins": 200,
    "deltat max s": 0.1,
    "phase bins": 72,
}

#: How hard the analyzer works. One knob, replacing the retired stack's three.
#:
#: Deliberately no "process all" here. A plugin whose events are cheap declares
#: that key in its own defaults; for this one it would be a switch that makes
#: Python decode every waveform, one mistaken edit away.
SAMPLING: dict[str, object] = {
    "max events per s": 20.0,
    "publish history": False,
}

SECTIONS = {
    "Channel roles": CHANNEL_ROLES,
    "Binning": BINNING,
    "Sampling": SAMPLING,
}


def seed(client, root: str = ROOT, sections: dict | None = None) -> int:
    """Create any missing key, without disturbing one that exists.

    `sections` is a nested dict of defaults: a dict value is an ODB directory,
    anything else is a key. Defaults to the WaveDREAM tree under `ROOT`.

    Written key by key rather than as a subtree dict: ``odb_set`` defaults to
    ``remove_unspecified_keys=True``, so handing it a whole section would delete
    anything an operator had added under it.
    """
    created = 0
    for path, value in _leaves(root, SECTIONS if sections is None else sections):
        if client.odb_exists(path):
            continue
        client.odb_set(path, value)
        created += 1
    return created


def _leaves(prefix: str, tree: dict):
    """(ODB path, default) for every key in a nested defaults dict."""
    for key, value in tree.items():
        path = f"{prefix}/{key}"
        if isinstance(value, dict):
            yield from _leaves(path, value)
        else:
            yield path, value


def _as_list(value):
    """MIDAS collapses a one-element array to a scalar on the way out."""
    if value is None:
        return []
    return list(value) if isinstance(value, list | tuple) else [value]


def read(client, root: str = ROOT, sections: dict | None = None) -> dict[str, dict]:
    """The current settings, with built-in defaults for anything missing.

    Mirrors the shape of `sections` (the WaveDREAM tree when omitted). Only keys
    that have a default are read, so a stray key an operator added does nothing
    -- including a ``Sampling/process all`` added to a tree that does not
    declare it.

    Never raises: the analyzer must keep running with an ODB that somebody has
    half-edited, and falling back to a known default is better than stopping.
    """
    return _read_tree(client, root, SECTIONS if sections is None else sections)


def _read_tree(client, prefix: str, defaults: dict) -> dict:
    out: dict = {}
    for key, default in defaults.items():
        path = f"{prefix}/{key}"
        if isinstance(default, dict):
            out[key] = _read_tree(client, path, default)
            continue
        out[key] = default
        try:
            got = client.odb_get(path)
        except Exception:
            continue
        if got is None:
            continue
        out[key] = _as_list(got) if isinstance(default, list) else got
    return out


def fingerprint(settings: dict) -> str:
    """A stable digest, for spotting a change without diffing by hand."""
    return json.dumps(settings, sort_keys=True, default=str)


def binning_fingerprint(settings: dict) -> str:
    """Only the parts that change the *shape* of a WaveDREAM histogram.

    The default shape test; a plugin with other settings supplies its own as
    ``shape_fingerprint(settings)``.

    Separate from the whole-settings digest on purpose: moving a channel role
    should not throw away accumulated plots, while changing a bin count has to.
    """
    return json.dumps({
        "Binning": settings.get("Binning", {}),
        # The channel list decides which persistence and amplitude histograms
        # exist at all, so it belongs here too.
        "waveform channels": settings.get("Channel roles", {}).get("waveform channels"),
    }, sort_keys=True, default=str)
