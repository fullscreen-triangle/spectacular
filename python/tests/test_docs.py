"""The documentation's examples run, and its tables of keys match the compiler.

Each page's ```python blocks run in order, in one namespace, in a scratch directory, so
a page can build on its earlier blocks as a reader would. A block preceded by
`<!-- no-test -->`, or holding a cell magic, is skipped. Every view a block makes must
compile without problems: a broken example would otherwise show its problems in the cell
and pass.
"""

import re
import textwrap
from pathlib import Path

import pytest

import spectacular as sp
from spectacular import _view

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
DESCRIBE = ROOT / "crates" / "spectacular" / "src" / "describe.rs"

pytestmark = pytest.mark.skipif(not DOCS.is_dir(), reason="the documentation is in the repository, not the package")

FENCE = re.compile(r"(?P<skip><!-- no-test -->\s*\n)?(?P<indent>[ \t]*)```python\n(?P<body>.*?)\n(?P=indent)```", re.S)


def blocks(page: Path):
    for m in FENCE.finditer(page.read_text("utf-8")):
        body = textwrap.dedent("\n".join(line[len(m["indent"]):] for line in m["body"].splitlines()))
        if m["skip"] or body.lstrip().startswith("%%"):
            continue
        yield body


PAGES = sorted(p for p in DOCS.rglob("*.md") if any(True for _ in blocks(p)))


def test_pages_with_examples_are_found():
    names = {p.relative_to(DOCS).as_posix() for p in PAGES}
    assert {"quickstart.md", "guide/chart-types.md", "for-ai.md"} <= names


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.relative_to(DOCS).as_posix())
def test_examples_run(page, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    load = _view._load

    def strict(spec, data):
        out = load(spec, data)
        assert not out[3], f"an example's description has problems: {out[3]}"
        return out

    monkeypatch.setattr(_view, "_load", strict)
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError:  # figures need a browser; without one, check the description only
        monkeypatch.setattr(sp, "render", lambda spec, out, data=None, **_: sp.validate(spec, data) or Path(out))
    namespace = {"__name__": "__docs__"}
    for i, body in enumerate(blocks(page)):
        try:
            exec(compile(body, f"{page.name}[{i}]", "exec"), namespace)
        except Exception as e:  # name the block, not just the line
            raise AssertionError(f"{page.relative_to(DOCS)} block {i} failed:\n{body}") from e


def norm(key: str) -> str:
    return "".join(c for c in key.lower() if c.isalnum())


def rust_table(name: str):
    """{canonical: {aliases}} from a `const NAME: &[(&str, &[&str])]` (or `&[&str]`) table."""
    text = DESCRIBE.read_text("utf-8")
    # A table on one line, else one spread over lines and closed by "];" on its own.
    body = re.search(rf"const {name}: [^=\n]+= &\[([^\n]*)\];", text) or re.search(rf"const {name}: [^=]+= &\[(.*?)\n\];", text, re.S)
    entries = re.findall(r'\("([^"]+)", &\[([^\]]*)\]\)', body.group(1))
    if entries:
        return {c: set(re.findall(r'"([^"]+)"', a)) for c, a in entries}
    names = re.findall(r'"([^"]+)"', body.group(1))
    return {names[0]: set(names)}


def docs_table(name: str):
    text = (DOCS / "reference" / "keys.md").read_text("utf-8")
    table = text.split(f"<!-- aliases: {name} -->", 1)[1].strip().split("\n\n", 1)[0]
    out = {}
    for row in table.splitlines()[2:]:
        cells = [c.strip() for c in row.strip("|").split("|")]
        key = re.findall(r"`([^`]+)`", cells[0])[0]
        out[key] = {norm(key)} | {norm(a) for a in re.findall(r"`([^`]+)`", cells[1])}
    return out


@pytest.mark.skipif(not DESCRIBE.is_file(), reason="needs the Rust source")
@pytest.mark.parametrize("table", ["TOP", "LISTS", "CHART", "CHART_TYPES"])
def test_the_keys_reference_lists_exactly_what_the_compiler_accepts(table):
    rust, docs = rust_table(table), docs_table(table)
    assert set(docs) == set(rust), "keys in one and not the other"
    for key, aliases in rust.items():
        assert docs[key] == aliases, f"{key}: documented {sorted(docs[key])}, compiler {sorted(aliases)}"
