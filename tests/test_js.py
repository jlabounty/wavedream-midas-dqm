"""Run the node tests for the pure JS helpers, if node is available.

Node is not a dependency of this project and must not become one: the pages
themselves have no build step and load plain <script src>. So this is a skip,
not a failure, where node is missing -- but where it exists the JS gets the same
scrutiny as the Python.

Point $MDQM_NODE at a node binary to use a specific one.
"""

from __future__ import annotations

import os
import re
import resource
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: Address-space cap for the node test run (as `ulimit -v 3000000`).
NODE_VMEM = 3_000_000 * 1024


def find_node() -> str | None:
    explicit = os.environ.get("MDQM_NODE")
    if explicit:
        return explicit if Path(explicit).is_file() else None
    return shutil.which("node") or shutil.which("nodejs")


def test_js_helpers():
    node = find_node()
    if not node:
        pytest.skip("no node on PATH; set $MDQM_NODE to run the JS tests")

    # Capped: a failing assert on the cyclic stub DOM once grew node to 16 GB
    # and the OOM killer took the session with it. Over the cap node dies with
    # an out-of-memory error, which fails this test instead.
    proc = subprocess.run(
        [node, "--max-old-space-size=1500", "--test", "tests/js/"],
        cwd=REPO, capture_output=True, text=True, timeout=120,
        preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_AS, (NODE_VMEM, NODE_VMEM)),
    )
    if proc.returncode != 0:
        pytest.fail(f"node --test failed:\n{proc.stdout}\n{proc.stderr}")


# -- no equality asserts on stub DOM nodes --------------------------------------------------
#
# node's assert builds its failure message by inspecting both operands, and the
# stub DOM (tests/js/domstub.js) is a large cyclic graph: a *failing*
# strictEqual on two nodes grew node by 1.6 GB in a second, a deepStrictEqual
# without bound. Identity is checked with assert.ok(a === b, "label") instead,
# which prints only the label. This is a lint on the source, by pattern: an
# operand that ends in something that returns a node or a list of them.

EQUALITY = re.compile(r"\bassert\.(deepStrictEqual|strictEqual|notStrictEqual|notDeepStrictEqual"
                      r"|deepEqual|notDeepEqual|equal|notEqual)\(")
DOMISH = re.compile(r"(\bEl\.focused|\.parent|\.children(\[[^\]]*\])?|\bbyId\([^()]*\)"
                    r"|\.(byClass|byTag|findAll)\([^()]*\)(\[[^\]]*\])?|\.find\(.*\)"
                    r"|\.(wrap|div|grid|head|firstChild)|getElementById\([^()]*\))$")


def _operands(src: str, start: int) -> list[str]:
    """The top-level comma-separated arguments of the call whose "(" is at start - 1."""
    depth, args, cur, i, quote = 0, [], [], start, None
    while i < len(src):
        c = src[i]
        if quote:
            cur.append(c)
            if c == "\\":
                cur.append(src[i + 1])
                i += 1
            elif c == quote:
                quote = None
        elif c in "\"'`":
            quote = c
            cur.append(c)
        elif c in "([{":
            depth += 1
            cur.append(c)
        elif c in ")]}":
            if depth == 0:
                args.append("".join(cur).strip())
                return args
            depth -= 1
            cur.append(c)
        elif c == "," and depth == 0:
            args.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
        i += 1
    return args


def dom_equality_sites(src: str) -> list[str]:
    out = []
    for m in EQUALITY.finditer(src):
        ops = _operands(src, m.end())[:2]
        if any(DOMISH.search(op) for op in ops):
            line = src.count("\n", 0, m.start()) + 1
            out.append(f"line {line}: {m.group(0)}{', '.join(ops)})")
    return out


def test_the_dom_lint_catches_what_it_should():
    bad = ['assert.strictEqual(El.focused, tab("x"));',
           'assert.strictEqual(grid.children[0], byId(page, "dqm-sma-xyhead"));',
           "assert.deepStrictEqual(grid.children, [head]);",
           'assert.notStrictEqual(byId(page, "a").parent, byId(page, "b"), "msg");',
           "assert.strictEqual(w.parent, grid, `${n} is not in a row`);"]
    good = ['assert.ok(El.focused === tab("x"), "focus");',
            'assert.strictEqual(byId(page, "a").getAttribute("role"), "tab");',
            'assert.strictEqual(grid.children.length, 1);',
            'assert.deepStrictEqual(rows.map((r) => r.attrs["data-row"]), ["a", "b"]);',
            'assert.strictEqual(byId(page, "x").textContent, "a, b");']
    for s in bad:
        assert dom_equality_sites(s), s
    for s in good:
        assert not dom_equality_sites(s), s


def test_no_equality_assert_on_stub_dom_nodes():
    hits = []
    for f in sorted((REPO / "tests" / "js").glob("*.test.js")):
        hits += [f"{f.name} {h}" for h in dom_equality_sites(f.read_text())]
    assert not hits, ("equality asserts on stub DOM nodes (use assert.ok(a === b, 'label') or "
                      "compare lengths):\n  " + "\n  ".join(hits))
