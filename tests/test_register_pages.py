"""Registration behaviour, against a fake ODB. No MIDAS needed.

The cases that matter are the refusals: this script runs on every start, in
experiments we share with other groups, so "does nothing surprising" is the
whole specification.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mdqm.install import register_pages as rp
from mdqm.install.manifest import pages

REPO = Path(__file__).resolve().parents[1]
PAGES_DIR = REPO / "pages"


class FakeClient:
    """Just enough midas.client.MidasClient to exercise register_pages.

    `odb_get` models a *directory* read as well as a leaf read, because that is
    what MIDAS does and what prune() relies on: reading "/Custom" returns its
    children rather than raising. A fake that only did leaves made prune look
    like it worked when it had in fact found nothing.
    """

    def __init__(self, initial=None):
        self.odb = dict(initial or {})
        self.writes = []
        self.deletes = []

    def odb_exists(self, path):
        return path in self.odb

    def odb_get(self, path):
        if path in self.odb:
            return self.odb[path]
        prefix = path.rstrip("/") + "/"
        children = {k[len(prefix):]: v for k, v in self.odb.items()
                    if k.startswith(prefix) and "/" not in k[len(prefix):]}
        if children:
            return children
        raise KeyError(path)

    def odb_set(self, path, value):
        self.odb[path] = value
        self.writes.append((path, value))

    def odb_delete(self, path):
        del self.odb[path]
        self.deletes.append(path)


@pytest.fixture
def entries():
    return pages(PAGES_DIR)


def test_manifest_validates_against_the_real_files(entries):
    assert rp.validate(entries) == []


def test_creates_missing_keys(entries):
    c = FakeClient()
    assert rp.register(c, entries, PAGES_DIR, replace=False, dry_run=False) == rp.EXIT_OK
    assert len(c.writes) == len(entries)
    for odb_name, path, _ in entries:
        assert c.odb[f"/Custom/{odb_name}"] == str(path)


def test_is_idempotent(entries):
    c = FakeClient()
    rp.register(c, entries, PAGES_DIR, replace=False, dry_run=False)
    c.writes.clear()
    assert rp.register(c, entries, PAGES_DIR, replace=False, dry_run=False) == rp.EXIT_OK
    assert c.writes == [], "a second run must write nothing"


def test_dry_run_writes_nothing(entries):
    c = FakeClient()
    assert rp.register(c, entries, PAGES_DIR, replace=False, dry_run=True) == rp.EXIT_OK
    assert c.writes == []
    assert c.odb == {}


def test_moved_checkout_heals_itself(entries):
    """A stale value that is still recognisably ours gets rewritten silently."""
    odb_name, path, _ = entries[0]
    c = FakeClient({f"/Custom/{odb_name}":
                    "/somewhere/else/wavedream-midas-dqm/pages/scalars.html"})
    assert rp.register(c, entries, PAGES_DIR, replace=False, dry_run=False) == rp.EXIT_OK
    assert c.odb[f"/Custom/{odb_name}"] == str(path)


def test_any_other_pages_dir_is_not_a_moved_checkout(entries):
    """`.../pages/scalars.html` of another repo is somebody else's page."""
    odb_name, _path, _ = entries[0]
    foreign = "/somewhere/else/mdqm/pages/scalars.html"
    c = FakeClient({f"/Custom/{odb_name}": foreign})
    assert rp.register(c, entries, PAGES_DIR, replace=False, dry_run=False) == rp.EXIT_REFUSED
    assert c.odb[f"/Custom/{odb_name}"] == foreign


def test_refuses_a_foreign_value(entries, capsys):
    """The joined-mode guard: never take over somebody else's menu entry."""
    odb_name, _path, _ = entries[0]
    foreign = "/home/musip/musip/custom/quads.html"
    c = FakeClient({f"/Custom/{odb_name}": foreign})

    rc = rp.register(c, entries, PAGES_DIR, replace=False, dry_run=False)

    assert rc == rp.EXIT_REFUSED
    assert c.odb[f"/Custom/{odb_name}"] == foreign, "the foreign value must survive"
    err = capsys.readouterr().err
    assert foreign in err, "the refusal must name what it found"


def test_replace_overrides_the_refusal(entries):
    odb_name, path, _ = entries[0]
    c = FakeClient({f"/Custom/{odb_name}": "/home/musip/musip/custom/quads.html"})
    assert rp.register(c, entries, PAGES_DIR, replace=True, dry_run=False) == rp.EXIT_OK
    assert c.odb[f"/Custom/{odb_name}"] == str(path)


def test_refusal_does_not_stop_the_other_keys(entries):
    """One collision must not leave the rest of the page set unregistered."""
    odb_name, _path, _ = entries[0]
    c = FakeClient({f"/Custom/{odb_name}": "/home/musip/musip/custom/quads.html"})
    rp.register(c, entries, PAGES_DIR, replace=False, dry_run=False)
    assert len(c.writes) == len(entries) - 1


def test_remove_only_deletes_our_keys(entries):
    c = FakeClient()
    rp.register(c, entries, PAGES_DIR, replace=False, dry_run=False)
    foreign_name = entries[0][0]
    c.odb[f"/Custom/{foreign_name}"] = "/home/musip/musip/custom/quads.html"

    rp.unregister(c, entries, PAGES_DIR, dry_run=False)

    assert f"/Custom/{foreign_name}" in c.odb, "a foreign key must not be deleted"
    assert len(c.deletes) == len(entries) - 1


def test_registration_never_touches_custom_path(entries):
    """The single most important property in a shared experiment.

    Values are absolute, so /Custom/Path is irrelevant to us -- and musip's
    frontend rewrites it on every start. We must neither read nor write it, and
    we must never write /Custom as a subtree (odb_set would then default to
    remove_unspecified_keys=True and delete their keys).
    """
    c = FakeClient({"/Custom/Path": "/home/musip/musip/custom",
                    "/Custom/Quads": "Quads/quad_basics.html"})

    rp.register(c, entries, PAGES_DIR, replace=False, dry_run=False)

    assert c.odb["/Custom/Path"] == "/home/musip/musip/custom"
    assert c.odb["/Custom/Quads"] == "Quads/quad_basics.html"
    for path, _value in c.writes:
        assert path != "/Custom", "never write the /Custom subtree as a whole"
        assert path.startswith("/Custom/")
        assert path.count("/") == 2, f"{path} is not a flat top-level key"


def test_all_registered_values_are_absolute(entries):
    """A relative value would be resolved against /Custom/Path, which we do not own."""
    for _odb_name, path, _ in entries:
        assert str(path).startswith("/")


def test_check_reports_unreadable_and_missing(entries, capsys):
    c = FakeClient()
    rp.register(c, entries, PAGES_DIR, replace=False, dry_run=False)
    assert rp.check(c, entries) == rp.EXIT_OK

    odb_name = entries[0][0]
    c.odb[f"/Custom/{odb_name}"] = "/nonexistent/scalars.html"
    assert rp.check(c, entries) == rp.EXIT_REFUSED
    assert "UNREADABLE" in capsys.readouterr().out


class TestPruningStaleKeys:
    """Renaming must not leave the old entries behind.

    Changing the menu prefix used to orphan every page key: they still pointed at
    real files, so they still worked, and the side menu grew a duplicate of each.
    """

    def _client_with(self, extra):
        c = FakeClient()
        rp.register(c, pages(PAGES_DIR), PAGES_DIR, replace=False, dry_run=False)
        c.odb.update(extra)
        c.writes.clear()
        return c

    def test_a_stale_key_of_ours_is_removed(self):
        entries = pages(PAGES_DIR)
        stale = str(entries[0][1])            # a real file in our checkout
        c = self._client_with({"/Custom/OldName": stale})

        assert rp.prune(c, entries, PAGES_DIR, dry_run=False) == 1
        assert "/Custom/OldName" not in c.odb

    def test_another_tenant_s_key_is_never_touched(self):
        entries = pages(PAGES_DIR)
        c = self._client_with({"/Custom/Quads": "Quads/quad_basics.html",
                               "/Custom/Path": "/home/musip/musip/custom"})

        assert rp.prune(c, entries, PAGES_DIR, dry_run=False) == 0
        assert c.odb["/Custom/Quads"] == "Quads/quad_basics.html"
        assert c.odb["/Custom/Path"] == "/home/musip/musip/custom"

    def test_what_was_just_registered_is_kept(self):
        entries = pages(PAGES_DIR)
        c = self._client_with({})
        assert rp.prune(c, entries, PAGES_DIR, dry_run=False) == 0
        for odb_name, _p, _e in entries:
            assert f"/Custom/{odb_name}" in c.odb

    def test_dry_run_removes_nothing(self):
        entries = pages(PAGES_DIR)
        stale = str(entries[0][1])
        c = self._client_with({"/Custom/OldName": stale})
        assert rp.prune(c, entries, PAGES_DIR, dry_run=True) == 1
        assert "/Custom/OldName" in c.odb

    def test_a_prefix_change_leaves_exactly_the_new_names(self):
        """The scenario this exists for."""
        unprefixed = pages(PAGES_DIR)
        c = FakeClient()
        rp.register(c, unprefixed, PAGES_DIR, replace=False, dry_run=False)

        prefixed = [((("WD" + n) if e.menu else n), p, e) for n, p, e in unprefixed]
        rp.register(c, prefixed, PAGES_DIR, replace=False, dry_run=False)
        rp.prune(c, prefixed, PAGES_DIR, dry_run=False)

        menu = sorted(k.rsplit("/", 1)[1] for k in c.odb if not k.endswith("!"))
        assert all(m.startswith("WD") for m in menu), menu


# --------------------------------------------------------------------------
# The pinky incident (2026-10-03).
#
# `python -m mdqm.install.register_pages --experiment bt2026 --prefix WD`, run
# from the checkout root on pinky, deleted six of MuSiP's /Custom keys. Their
# values are *relative* paths, which mhttpd resolves against /Custom/Path.
# main() took the repo root as "our pages directory", and _is_ours() resolved
# the relative value against the current directory, so `lvds.html` became
# `<root>/lvds.html`: "inside our checkout", hence "ours", hence pruned.
#
# These tests go through main(), so the pages directory is computed exactly as
# on pinky, and run it from three working directories.
# --------------------------------------------------------------------------

#: MuSiP's keys on pinky. Relative: mhttpd resolves them against /Custom/Path.
MUSIP_RELATIVE = {
    "lvds": "lvds.html",
    "Quads": "Quads/quad_basics.html",
    "MuTRiG": "Mutrig/TimingScint.html",
    "DQM": "onlineDQM.html",
    "Quads_new": "Quads/quads_svgBased.html",
    "Scintillators SMA board": "Quads/sma_svgBased.html",
}

#: Other tenants' absolute keys, outside our checkout. The last one contains
#: "/pages/" and ends in .html, which the old fallback match took for ours.
FOREIGN_ABSOLUTE = {
    "Path": "/home/pinky/bt2026/musip/custom",
    "msetpoint": "/home/pinky/bt2026/beamtime2026_pie5/custom/msetpoint.html",
    "caenhv": "/home/pinky/bt2026/beamtime2026_pie5/custom/caenhv.html",
    "rundb": "/home/pinky/bt2026/beamtime2026_pie5/custom/rundb.html",
    "OtherPages": "/home/x/other/pages/foo.html",
    "OtherScalars": "/home/x/other/pages/scalars.html",
}


class _FakeContextClient(FakeClient):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _pinky_custom(stale_value):
    """pinky's /Custom as it was: foreign keys, our WD keys, one stale key of ours."""
    odb = {f"/Custom/{k}": v for k, v in {**MUSIP_RELATIVE, **FOREIGN_ABSOLUTE}.items()}
    for odb_name, path, entry in pages(PAGES_DIR):
        odb[f"/Custom/{('WD' + odb_name) if entry.menu else odb_name}"] = str(path)
    odb["/Custom/Scalers"] = stale_value           # left over from an unprefixed run
    return odb


def _run_main(monkeypatch, client, argv):
    """main() against `client`, with a stand-in midas.client module."""
    import sys
    import types

    mod = types.ModuleType("midas.client")
    mod.MidasClient = lambda *a, **k: client
    pkg = types.ModuleType("midas")
    pkg.client = mod
    monkeypatch.setitem(sys.modules, "midas", pkg)
    monkeypatch.setitem(sys.modules, "midas.client", mod)
    return rp.main(["--experiment", "bt2026", *argv])


def _custom(client):
    return {k: v for k, v in client.odb.items() if k.startswith("/Custom/")}


@pytest.fixture(params=["root", "pages", "tmp"])
def cwd(request, monkeypatch, tmp_path):
    where = {"root": REPO, "pages": PAGES_DIR, "tmp": tmp_path}[request.param]
    monkeypatch.chdir(where)
    return where


class TestPinkyIncident:
    STALE = str(PAGES_DIR / "scalars.html")

    def test_prune_removes_only_our_stale_key(self, cwd, monkeypatch):
        c = _FakeContextClient(_pinky_custom(self.STALE))
        before = _custom(c)

        assert _run_main(monkeypatch, c, ["--prefix", "WD", "--no-config"]) == rp.EXIT_OK

        assert c.deletes == ["/Custom/Scalers"], c.deletes
        after = _custom(c)
        for key, value in before.items():
            if key != "/Custom/Scalers":
                assert after.get(key) == value, f"{key} changed or vanished"

    def test_foreign_keys_survive_a_moved_checkout_stale_key(self, cwd, monkeypatch):
        """The heal case still works: an old checkout's absolute value is ours."""
        moved = "/home/pinky/old/wavedream-midas-dqm/pages/scalars.html"
        c = _FakeContextClient(_pinky_custom(moved))

        assert _run_main(monkeypatch, c, ["--prefix", "WD", "--no-config"]) == rp.EXIT_OK

        assert c.deletes == ["/Custom/Scalers"], c.deletes
        for k, v in {**MUSIP_RELATIVE, **FOREIGN_ABSOLUTE}.items():
            assert c.odb[f"/Custom/{k}"] == v

    def test_dry_run_writes_and_deletes_nothing(self, cwd, monkeypatch, capsys):
        c = _FakeContextClient(_pinky_custom(self.STALE))
        before = dict(c.odb)

        assert _run_main(monkeypatch, c, ["--prefix", "WD", "--dry-run"]) == rp.EXIT_OK

        assert c.odb == before and c.writes == [] and c.deletes == []
        out = capsys.readouterr().out
        assert "Scalers" in out and "stale, ours" in out
        for k in MUSIP_RELATIVE:
            assert f"- {k} " not in out and f"- {k:22s}" not in out, out

    def test_remove_leaves_foreign_keys_alone(self, cwd, monkeypatch):
        c = _FakeContextClient(_pinky_custom(self.STALE))

        assert _run_main(monkeypatch, c, ["--prefix", "WD", "--remove"]) == rp.EXIT_OK

        for k, v in {**MUSIP_RELATIVE, **FOREIGN_ABSOLUTE}.items():
            assert c.odb[f"/Custom/{k}"] == v
        assert not any(k.startswith("/Custom/WD") for k in c.odb)

    def test_remove_never_deletes_a_relative_value_at_our_name(self, cwd, monkeypatch):
        """Somebody else's relative key that happens to share one of our names."""
        c = _FakeContextClient({"/Custom/WDScalers": "scalars.html",
                                "/Custom/dqm.css": "css/dqm.css"})

        _run_main(monkeypatch, c, ["--prefix", "WD", "--remove"])

        assert c.deletes == []

    def test_register_refuses_a_relative_value_at_our_name(self, cwd, monkeypatch):
        """A relative value is resolved by mhttpd against /Custom/Path: never ours."""
        c = _FakeContextClient({"/Custom/WDScalers": "scalars.html",
                                "/Custom/lvds": "lvds.html"})

        rc = _run_main(monkeypatch, c, ["--prefix", "WD", "--no-config"])

        assert rc == rp.EXIT_REFUSED
        assert c.odb["/Custom/WDScalers"] == "scalars.html"
        assert c.odb["/Custom/lvds"] == "lvds.html"
        assert c.deletes == [], "a refused register must not prune"

    def test_replace_writes_only_our_manifest_names(self, cwd, monkeypatch):
        """--replace overwrites a foreign value at one of OUR names, nothing else."""
        c = _FakeContextClient(_pinky_custom(self.STALE))
        c.odb["/Custom/WDScalers"] = "/home/musip/musip/custom/quads.html"
        ours = {f"/Custom/{('WD' + n) if e.menu else n}" for n, _p, e in pages(PAGES_DIR)}

        assert _run_main(monkeypatch, c,
                         ["--prefix", "WD", "--replace", "--no-config"]) == rp.EXIT_OK

        assert {p for p, _v in c.writes} <= ours
        assert c.deletes == ["/Custom/Scalers"]
        for k, v in {**MUSIP_RELATIVE, **FOREIGN_ABSOLUTE}.items():
            assert c.odb[f"/Custom/{k}"] == v


class TestIsOurs:
    @pytest.mark.parametrize("value", [
        *MUSIP_RELATIVE.values(),
        "scalars.html", "pages/scalars.html", "js/dqm-sma.js",
        "/home/x/other/pages/foo.html",
        "/home/x/other/pages/scalars.html",             # right file, wrong repo
        "/home/x/wavedream-midas-dqm/pages/notours.html",  # right repo, not in manifest
        "/home/x/wavedream-midas-dqm/pages/sub/scalars.html",
        "link:https://example.org/pages/scalars.html",
        "",
    ])
    def test_not_ours(self, value, cwd):
        assert not rp._is_ours(value, PAGES_DIR)

    @pytest.mark.parametrize("value", [
        str(PAGES_DIR / "scalars.html"),
        str(PAGES_DIR / "js" / "dqm-sma.js"),
        "/home/pinky/old/wavedream-midas-dqm/pages/scalars.html",
        "/somewhere/wavedream-midas-dqm/pages/js/dqm-sma.js",
    ])
    def test_ours(self, value, cwd):
        assert rp._is_ours(value, PAGES_DIR)

    def test_main_uses_the_real_pages_dir(self, monkeypatch):
        seen = {}
        orig = rp.prune

        def spy(client, entries, pages_dir, dry_run):
            seen["pages_dir"] = pages_dir
            return orig(client, entries, pages_dir, dry_run)

        monkeypatch.setattr(rp, "prune", spy)
        _run_main(monkeypatch, _FakeContextClient(), ["--no-config"])
        assert seen["pages_dir"] == PAGES_DIR
