"""The manifest rules are all ways mhttpd fails silently, so they get a test.

Nothing here needs MIDAS running.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mdqm.install.config_defaults import DEFAULTS
from mdqm.install.manifest import (
    ENTRIES,
    FORBIDDEN_CONTENT,
    MAX_KEY_LENGTH,
    RESERVED_SUBSTRINGS,
    check_entry,
    check_key,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e.key)
def test_entry_name_is_legal(entry):
    assert check_entry(entry) == []


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e.key)
def test_entry_file_exists_and_is_readable(entry):
    path = entry.resolve(REPO / "pages")
    assert path.is_file(), f"{entry.key} names a file that does not exist: {path}"
    assert os.access(path, os.R_OK)


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e.key)
def test_no_forbidden_content(entry):
    """show_custom_page() rewrites legacy <odb ...> tags and truncates at NUL."""
    text = entry.resolve(REPO / "pages").read_bytes()
    assert b"\x00" not in text, f"{entry.key} contains a NUL; mhttpd truncates there"
    for token in FORBIDDEN_CONTENT:
        assert token.encode() not in text, (
            f"{entry.key} contains {token!r}, which show_custom_page() will rewrite. "
            f"Split the string if it is genuinely needed."
        )


def test_menu_keys_have_no_dot():
    """A dotted key is served by send_fp(), which sets Expires: +24h.

    Menu pages must stay dot-less so an edit is visible on reload.
    """
    for entry in ENTRIES:
        if entry.menu:
            assert "." not in entry.key


def test_asset_keys_are_hidden_from_the_menu():
    for entry in ENTRIES:
        if not entry.menu:
            assert entry.odb_name.endswith("!")


def test_keys_are_unique():
    names = [e.odb_name for e in ENTRIES]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("reserved", RESERVED_SUBSTRINGS)
def test_reserved_substrings_are_rejected(reserved):
    """mhttpd's interprete() grabs these before /Custom is consulted."""
    assert check_key(f"my-{reserved}") != []


def test_long_and_spacey_keys_are_rejected():
    assert check_key("x" * (MAX_KEY_LENGTH + 1)) != []
    assert check_key("WD Scalers") != []
    assert check_key("Scalers") == []


def test_html_references_only_registered_assets():
    """Every local <script src>/<link href> must be something we register.

    An unregistered name falls through to send_resource() and either 404s or,
    worse, silently serves a stock MIDAS file of the same name.
    """
    import re

    registered = {e.key for e in ENTRIES}
    stock = {
        "midas.js", "midas.css", "mhttpd.js", "mhttpd.css", "controls.js",
        "mplot.js", "mhistory.js", "mihistory.js", "eqtable.js", "filesrw.js",
    }
    ref = re.compile(r'(?:src|href)="([^"]+)"')
    for entry in ENTRIES:
        if not entry.path.endswith(".html"):
            continue
        text = entry.resolve(REPO / "pages").read_text()
        for target in ref.findall(text):
            if target.startswith(("http://", "https://", "?", "#", "/")):
                continue
            name = target.split("?")[0]          # strip the ?v= cache buster
            assert name in registered or name in stock, (
                f"{entry.key} references {name!r}, which is neither registered "
                f"in the manifest nor a stock MIDAS resource"
            )


def test_js_defaults_match_python_defaults():
    """The JS copy is what runs when /DQM/Scalars is absent, so it must agree."""
    import json
    import re

    text = (REPO / "pages" / "js" / "dqm-common.js").read_text()
    m = re.search(r"const DEFAULTS = (\{.*?\n\});", text, re.S)
    assert m, "could not find `const DEFAULTS = {...};` in dqm-common.js"
    js = json.loads(re.sub(r",(\s*\})", r"\1", m.group(1)))

    assert set(js) == set(DEFAULTS), (
        f"only in JS: {set(js) - set(DEFAULTS)}; only in Python: {set(DEFAULTS) - set(js)}"
    )
    for key, want in DEFAULTS.items():
        got = js[key]
        if isinstance(want, float):
            assert float(got) == pytest.approx(want), key
        else:
            assert got == want, key


@pytest.mark.parametrize("prefix", ["", "WD"])
def test_no_menu_key_is_a_substring_of_another(prefix):
    """mhttpd's sidenav highlights item when current_page.search(item) hits.

    The key is an unescaped regex there, and with no metacharacters (check_key)
    that is a substring test, tried with the key and with its lower case. A
    menu key inside another lights both entries up on the longer one's page
    (the old "SMA" did on SMAEvents). The prefix is what pinky registers with.
    """
    keys = [prefix + e.key for e in ENTRIES if e.menu]
    for a in keys:
        for b in keys:
            if a == b:
                continue
            assert a not in b and a.lower() not in b, (
                f"menu key {a!r} is inside {b!r}: mhttpd would highlight both on {b!r}")


def test_the_mupix_page_is_in_the_menu():
    entry = next((e for e in ENTRIES if e.key == "MuPixPlots"), None)
    assert entry is not None, "the MuPix page has its own sidenav entry"
    assert entry.menu and entry.path == "mupix.html"
    assert "SMA analyzer" in entry.summary
    # It is the SMAPlots page's script in its MuPix view, with the same assets.
    html = (REPO / "pages" / "mupix.html").read_text()
    sma = (REPO / "pages" / "sma.html").read_text()
    assert 'data-view="mupix"' in html
    import re
    assets = re.compile(r'(?:src|href)="(dqm[^"]+)"')
    assert assets.findall(html) == assets.findall(sma), "same scripts and ?v= strings as sma.html"
