"""What gets registered into ``/Custom``, and the rules a name has to satisfy.

This is the single source of truth for the page set. ``register_pages.py``
writes what is listed here; ``tests/test_manifest.py`` asserts every rule below
holds, because each one is a way mhttpd will silently serve the wrong thing (or
nothing) rather than report an error.

Why registration works the way it does
--------------------------------------
``add_custom_path()`` (midas ``progs/mhttpd.cxx:3602``) returns a value that
starts with ``/`` unchanged -- ``/Custom/Path`` is only prepended to *relative*
names. And a plain static request such as ``GET /dqm-common.js``
(``mhttpd.cxx:13146``) is resolved by looking for the ODB key
``/Custom/dqm-common.js``, then ``...&``, then ``...!``, and serving whatever
file that key names.

So every page and every asset is registered as its own ``/Custom/<name>`` key
holding an **absolute** path, and ``/Custom/Path`` is never read or written.
That is what lets these pages coexist as a guest in somebody else's experiment:
musip's ``quads_config_fe.cpp:668`` rewrites ``/Custom/Path`` on every start,
and it does not matter to us at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: Repository root, derived rather than hardcoded -- the checkout lives in a
#: different place on every machine (``~/josh`` here, ``~/software/wavedream-
#: frontends`` on Pinky), and the ODB values are absolute, per-machine state.
REPO_ROOT = Path(__file__).resolve().parents[3]
PAGES_DIR = REPO_ROOT / "pages"

#: MIDAS ``NAME_LENGTH``. A longer key is truncated, not rejected.
MAX_KEY_LENGTH = 31

#: mhttpd's ``interprete()`` intercepts any request path *containing* one of
#: these as a substring, before ``/Custom`` is ever consulted. An asset named
#: after one of them is unreachable.
RESERVED_SUBSTRINGS = (
    "midas.js",
    "midas.css",
    "mhttpd.js",
    "mhttpd.css",
    "controls.js",
    "obsolete.js",
)

#: After the ``/Custom`` lookup misses, mhttpd falls back to ``send_resource()``,
#: which serves anything in ``$MIDASSYS/resources``. A key that collides with a
#: stock resource name still wins (the /Custom lookup comes first), but the
#: reverse -- expecting to override a stock resource -- does not work, and the
#: ambiguity is not worth having. Checked at test time against the live
#: resources directory when one is available.
STOCK_RESOURCE_HINT = ("mplot.js", "mhistory.js", "mihistory.js", "eqtable.js", "filesrw.js")

#: ``show_custom_page()`` reads the file into a ``char*``, truncates at the first
#: NUL, and rewrites legacy ``<odb ...>`` tags. Neither is wanted in our files.
FORBIDDEN_CONTENT = ("<odb ",)


@dataclass(frozen=True)
class Entry:
    """One ``/Custom`` key.

    ``key``      ODB key name, *without* the ``!`` suffix.
    ``path``     file to serve, relative to ``pages/``.
    ``menu``     True for a page that should appear in the mhttpd side menu.
    ``summary``  what it is, for ``--list`` output.
    """

    key: str
    path: str
    menu: bool
    summary: str

    @property
    def odb_name(self) -> str:
        """The key as written to the ODB.

        A trailing ``!`` hides the entry from the side menu -- that is the only
        thing the suffix does. (External links use a ``link:https://...``
        *value*, not this suffix; the two are widely confused.)
        """
        return self.key if self.menu else self.key + "!"

    def resolve(self, pages_dir: Path) -> Path:
        return (pages_dir / self.path).resolve()


# Pages carry no dot in their key. That is not cosmetic: a dot-less key routes
# through show_custom_page(), which sends no cache headers at all, while a
# dotted key routes through send_file() -> send_fp(), which stamps
# "Expires: <now + 24h>" with no ETag and no Last-Modified. So the HTML is
# always fresh and only the assets need ?v= busting.
#
# Keys are also space-free and free of regex metacharacters: the key is both
# "?cmd=custom&page=<key>" and the argument to mhttpd_init('<key>'), and the
# sidenav highlight does current_page.search(item) -- an *unescaped* regex.
# Only what exists and works is listed. A page added to the menu before it
# does anything is worse than no page: an operator who opens it and finds it
# broken stops trusting the whole set. Later stages append here.
ENTRIES: tuple[Entry, ...] = (
    Entry("Scalers", "scalars.html", True, "scaler rates, thresholds and trends (no backend)"),
    Entry("Scope", "scope.html", True, "live waveforms from the event buffer (no backend)"),
    Entry("Waveforms", "waveforms.html", True, "accumulated plots (needs the analyzer)"),
    Entry("EventDisplay", "evd.html", True, "one event, every channel (needs the analyzer)"),
    Entry("dqm-common.js", "js/dqm-common.js", False, "shared discovery and RPC helpers"),
    Entry("dqm-scalars.js", "js/dqm-scalars.js", False, "scaler page"),
    Entry("dqm-wdbanks.js", "js/dqm-wdbanks.js", False, "WaveDream bank decoding in the browser"),
    Entry("dqm-scope.js", "js/dqm-scope.js", False, "scope page"),
    Entry("dqm-brpc.js", "js/dqm-brpc.js", False, "talking to an analyzer over binary RPC"),
    Entry("dqm-waveforms.js", "js/dqm-waveforms.js", False, "accumulated plots page"),
    Entry("dqm-evd.js", "js/dqm-evd.js", False, "event display page"),
    Entry("SMAPlots", "sma.html", True, "SMA plots, flags and trends (needs the SMA analyzer)"),
    Entry("SMAEvents", "sma-events.html", True,
          "SMA S1-seeded events and whole-frame raster (needs the SMA analyzer)"),
    Entry("dqm-sma.js", "js/dqm-sma.js", False, "SMA plots page"),
    Entry("dqm-sma-events.js", "js/dqm-sma-events.js", False, "SMA event display page"),
    Entry("dqm-smaframe.js", "js/dqm-smaframe.js", False, "SMA frame (smaf) decoder"),
    Entry("dqm-heatmap.js", "js/dqm-heatmap.js", False, "2D histograms drawn as an image"),
    Entry("dqm.css", "css/dqm.css", False, "the little that midas.css does not cover"),
    Entry("dqm-sma.css", "css/dqm-sma.css", False, "SMA page styles"),
)

#: Config subtree for the pages themselves. Deliberately *not* under /Custom --
#: mhttpd renders any /Custom subdirectory as a sidenav submenu, unconditionally,
#: with no way to hide it. And deliberately not under /Equipment/<eq>/Settings --
#: the pages must be able to describe equipment they do not own, and to keep
#: working when that equipment is absent entirely.
CONFIG_ROOT = "/DQM/Scalars"

_KEY_OK = re.compile(r"^[A-Za-z0-9_.-]+$")


def check_key(name: str) -> list[str]:
    """Return the reasons ``name`` is unusable as a ``/Custom`` key. Empty is good."""
    problems: list[str] = []
    if len(name) > MAX_KEY_LENGTH:
        problems.append(f"longer than MIDAS NAME_LENGTH ({len(name)} > {MAX_KEY_LENGTH})")
    if not _KEY_OK.match(name):
        problems.append("contains a space or a regex metacharacter")
    for reserved in RESERVED_SUBSTRINGS:
        if reserved in name:
            problems.append(f"contains {reserved!r}, which mhttpd intercepts before /Custom")
    return problems


def check_entry(entry: Entry) -> list[str]:
    problems = check_key(entry.key)
    if entry.menu and "." in entry.key:
        problems.append("menu page keys must contain no dot, or mhttpd caches them for 24 h")
    if not entry.menu and "." not in entry.key:
        problems.append("asset keys should keep their extension so get_content_type() works")
    return problems


def pages(pages_dir: Path | None = None) -> list[tuple[str, Path, Entry]]:
    """The manifest as ``(odb_name, absolute_path, entry)``."""
    base = Path(pages_dir) if pages_dir else PAGES_DIR
    return [(e.odb_name, e.resolve(base), e) for e in ENTRIES]
