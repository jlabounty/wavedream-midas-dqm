#!/usr/bin/env python3
"""Register the DQM pages into a MIDAS experiment's ``/Custom``.

Knows nothing about any particular experiment: an installation that has never
heard of wavedream-scalar-readout must still be able to run this. It needs only
``midas.client`` on ``PYTHONPATH`` and an experiment to talk to.

    mdqm-register-pages --list
    mdqm-register-pages --dry-run
    mdqm-register-pages --prefix WD
    mdqm-register-pages --check

Idempotent by design, and safe to call on every start -- which is how a moved
checkout heals itself.

The one rule worth reading before editing this file
---------------------------------------------------
Keys are written **one at a time, by full path**. Never
``odb_set("/Custom", {...})``: ``odb_set`` defaults to
``remove_unspecified_keys=True``, so passing a dict for the whole subtree would
delete every key it did not mention -- in a shared experiment that is somebody
else's ``Path``, ``Quads``, ``MuTRiG`` and ``LVDS``, i.e. this script would
break another group's DAQ. Same trap as ``EquipmentBase(default_settings=...)``,
in a worse place.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from mdqm.install.manifest import CONFIG_ROOT, ENTRIES, PAGES_DIR, check_entry, check_key, pages

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_BAD_MANIFEST = 2


def _fail(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)


def validate(entries) -> list[str]:
    """Manifest problems that must stop us before we touch the ODB."""
    problems = []
    for _odb_name, path, entry in entries:
        for p in check_entry(entry):
            problems.append(f"{entry.key}: {p}")
        if not path.is_file():
            problems.append(f"{entry.key}: no such file: {path}")
        elif not os.access(path, os.R_OK):
            problems.append(f"{entry.key}: not readable: {path}")
    return problems


def register(client, entries, pages_dir: Path, replace: bool, dry_run: bool) -> int:
    """Create or update the /Custom keys. Returns a process exit code."""
    refused = 0
    for odb_name, path, entry in entries:
        full = f"/Custom/{odb_name}"
        target = str(path)
        existing = client.odb_get(full) if client.odb_exists(full) else None

        if existing == target:
            print(f"  = {entry.key:22s} already registered")
            continue

        if existing is None:
            action = "+"
        elif _is_ours(existing, pages_dir) or replace:
            action = "~"
        else:
            # Somebody else owns this name. Do not take it over silently: in a
            # shared experiment that would replace a working menu entry of
            # theirs with one of ours, and the first anyone would know is that
            # their page stopped loading.
            _fail(
                f"{full} already exists and is not ours (a relative value, or "
                f"an absolute one outside {pages_dir}):\n"
                f"    existing: {existing}\n"
                f"    ours:     {target}\n"
                f"    Use --prefix to pick a different name, or --replace if you "
                f"are certain the existing entry is stale."
            )
            refused += 1
            continue

        if dry_run:
            print(f"  {action} {entry.key:22s} {target}   (dry run)")
        else:
            client.odb_set(full, target)
            print(f"  {action} {entry.key:22s} {target}")

    return EXIT_REFUSED if refused else EXIT_OK


def prune(client, entries, pages_dir: Path, dry_run: bool) -> int:
    """Remove /Custom keys that are ours but no longer wanted.

    Without this, changing the menu prefix leaves the old entries behind: they
    still point at real files, so they still work, and the side menu grows a
    duplicate of every page. Renaming should not litter.

    Only keys whose value is an absolute path into our checkout (see
    ``_is_ours``) are considered, so another tenant's entries are never
    touched, and only those absent from the manifest we just wrote -- so this
    cannot remove what it has just registered.
    """
    wanted = {odb_name for odb_name, _p, _e in entries}
    removed = 0
    try:
        existing = client.odb_get("/Custom") or {}
    except Exception:
        return 0

    for key, value in existing.items():
        if key.endswith("/key") or not isinstance(value, str):
            continue
        if key in wanted or key == "Path":
            continue
        # Relative values (MuSiP's lvds.html, Quads/..., resolved by mhttpd
        # against /Custom/Path) and absolute ones outside our checkout are
        # somebody else's; _is_ours() says no to both.
        if not _is_ours(value, pages_dir):
            continue
        if dry_run:
            print(f"  - {key:22s} stale, ours   (dry run)")
        else:
            client.odb_delete(f"/Custom/{key}")
            print(f"  - {key:22s} stale, ours")
        removed += 1
    return removed


#: Directory name of this repository. A value under ``.../<this>/pages/`` that
#: names one of our manifest files is recognised even after the checkout moved.
REPO_DIR_NAME = "wavedream-midas-dqm"


def _is_ours(value: str, pages_dir: Path) -> bool:
    """True only if ``value`` is an absolute path to one of our files.

    A key is ours when its value is **absolute** and either

    * lies inside ``pages_dir`` (the real ``pages/`` directory), or
    * ends in ``<...>/wavedream-midas-dqm/pages/<a manifest entry path>``:
      the same file in a checkout that has since moved, which is what lets a
      moved checkout heal itself.

    A **relative** value is never ours. mhttpd resolves it against
    ``/Custom/Path``, which belongs to whoever runs the experiment (MuSiP on
    pinky); we only ever write absolute values. Resolving a relative value with
    ``Path.resolve()`` would anchor it to the *current working directory*, and
    that is how ``lvds.html`` once counted as ours and was pruned. The ``Path``
    key itself is never ours either (callers skip it by name).
    """
    if not isinstance(value, str) or not value.startswith("/"):
        return False
    path = Path(value)
    pages_dir = Path(pages_dir)
    try:
        path.resolve().relative_to(pages_dir.resolve())
        return True
    except (ValueError, OSError):
        pass
    parts = path.parts
    repo_names = {REPO_DIR_NAME, pages_dir.resolve().parent.name}
    for entry in ENTRIES:
        tail = Path(entry.path).parts
        n = len(tail)
        if (len(parts) >= n + 3 and parts[-n:] == tail and parts[-n - 1] == "pages"
                and parts[-n - 2] in repo_names):
            return True
    return False


def unregister(client, entries, pages_dir: Path, dry_run: bool) -> int:
    """Remove only the keys whose value points into our checkout."""
    for odb_name, _path, entry in entries:
        full = f"/Custom/{odb_name}"
        if not client.odb_exists(full):
            continue
        existing = client.odb_get(full)
        if not _is_ours(existing, pages_dir):
            print(f"  ! {entry.key:22s} left alone, not ours: {existing}")
            continue
        if dry_run:
            print(f"  - {entry.key:22s} (dry run)")
        else:
            client.odb_delete(full)
            print(f"  - {entry.key:22s}")
    return EXIT_OK


def check(client, entries) -> int:
    """Verify every registered key still resolves to a readable file."""
    bad = 0
    for odb_name, _path, entry in entries:
        full = f"/Custom/{odb_name}"
        if not client.odb_exists(full):
            print(f"  ? {entry.key:22s} not registered")
            bad += 1
            continue
        value = client.odb_get(full)
        if os.access(value, os.R_OK):
            print(f"  = {entry.key:22s} {value}")
        else:
            # The uid that matters is mhttpd's, not ours -- this check is
            # necessary and not sufficient. See README troubleshooting.
            print(f"  x {entry.key:22s} UNREADABLE {value}")
            bad += 1
    return EXIT_REFUSED if bad else EXIT_OK


def seed_config(client, config_root: str, dry_run: bool) -> None:
    """Create the page config subtree, create-if-absent, never overwriting.

    The page works without this -- it falls back to the same values as built-in
    defaults and says so. Seeding just makes them editable from the ODB browser.
    """
    from mdqm.install.config_defaults import DEFAULTS

    for key, value in DEFAULTS.items():
        full = f"{config_root}/{key}"
        if client.odb_exists(full):
            continue
        if dry_run:
            print(f"  + {full} = {value!r}   (dry run)")
        else:
            client.odb_set(full, value)
            print(f"  + {full} = {value!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default=os.environ.get("MIDAS_EXPT_NAME"),
                    help="MIDAS experiment name (default: $MIDAS_EXPT_NAME)")
    ap.add_argument("--pages-dir", default=None,
                    help="override the pages/ directory (default: this checkout)")
    ap.add_argument("--prefix", default=os.environ.get("WDS_CUSTOM_PAGE_PREFIX", ""),
                    help="prefix for menu entries, e.g. 'WD' when sharing an experiment. "
                         "Must be space-free for the same reason the keys are.")
    ap.add_argument("--client-name", default="mdqm_register")
    ap.add_argument("--list", action="store_true", help="show the manifest and exit")
    ap.add_argument("--replace", action="store_true", help="overwrite foreign values")
    ap.add_argument("--remove", action="store_true", help="unregister our keys")
    ap.add_argument("--check", action="store_true", help="verify registered keys resolve")
    ap.add_argument("--no-config", action="store_true", help="skip seeding the config subtree")
    ap.add_argument("--dry-run", action="store_true", help="say what would happen")
    args = ap.parse_args(argv)

    pages_dir = Path(args.pages_dir).resolve() if args.pages_dir else None
    entries = pages(pages_dir)
    # The real pages/ directory. (It once was entries[0][1].parents[1], which is
    # the repo *root*: entries[0] is pages/scalars.html. With that, and run
    # from the root, relative values of other groups counted as ours.)
    pages_dir = pages_dir or PAGES_DIR.resolve()

    # Apply the prefix to menu entries only. Assets are fetched by the literal
    # name the HTML asks for, so prefixing them would break every <script src>.
    if args.prefix:
        for bad in check_key(args.prefix):
            _fail(f"--prefix {args.prefix!r}: {bad}")
            return EXIT_BAD_MANIFEST
        entries = [
            ((args.prefix + n) if e.menu else n, p, e) for n, p, e in entries
        ]

    if args.list:
        for odb_name, path, entry in entries:
            mark = " " if entry.menu else "!"
            print(f"  {mark} /Custom/{odb_name:24s} -> {path}")
            print(f"      {entry.summary}")
        return EXIT_OK

    problems = validate(entries)
    if problems:
        for p in problems:
            _fail(p)
        return EXIT_BAD_MANIFEST

    if not args.experiment:
        _fail("no experiment: pass --experiment or set MIDAS_EXPT_NAME")
        return EXIT_BAD_MANIFEST

    import midas.client

    with midas.client.MidasClient(args.client_name, expt_name=args.experiment) as client:
        if args.check:
            return check(client, entries)
        if args.remove:
            return unregister(client, entries, pages_dir, args.dry_run)
        rc = register(client, entries, pages_dir, args.replace, args.dry_run)
        if rc == EXIT_OK:
            prune(client, entries, pages_dir, args.dry_run)
            if not args.no_config:
                seed_config(client, CONFIG_ROOT, args.dry_run)
        return rc


if __name__ == "__main__":
    sys.exit(main())
