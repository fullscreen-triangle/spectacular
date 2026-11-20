"""Validation and figures from Python scripts and pipelines.

`validate` runs the native command that ships in the wheel. `render` draws the views in
headless Chromium: through Playwright for Python when it is installed
(`pip install "spectacular-charts[render]"`, then `python -m playwright install chromium`), else
through the Node command of a spectacular checkout.
"""

from __future__ import annotations

import json
import math
import re
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ._spec import SpecLike, load_spec


class SpectacularError(Exception):
    """The command refused a spec (`problems` lists them by path) or could not run."""

    def __init__(self, message: str, problems: Optional[List[Dict[str, str]]] = None, exit: Optional[int] = None):
        super().__init__(message)
        self.problems = problems or []
        self.exit = exit


def validate(spec: SpecLike, data: Any = None) -> List[Dict[str, str]]:
    """The problems of a description or spec, each {"path", "message"}; [] when it holds.

    With `data` (or a local data file named in it) the columns are checked too. Paths
    point into what was given, e.g. "/charts/1/x-axis".
    """
    from . import _native
    from ._data import column_types
    from ._spec import is_spec

    description, base = load_spec(spec)
    types, rows = column_types(data) if data is not None else (None, None)
    if data is not None and is_spec(description):
        description = {**description, "data": {"columns": "external"}}  # the table is the data given
    _, problems = _native.compile_description(description, types, rows, base)
    return problems


_CSS_PX = {"px": 1.0, "mm": 96 / 25.4, "in": 96.0}
_BROWSER_ARGS = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]


def render(
    spec: SpecLike,
    out: Union[str, "os.PathLike[str]"],
    data: Any = None,
    *,
    width: Optional[float] = None,
    unit: Optional[str] = None,
    dpi: Optional[float] = None,
) -> Path:
    """Draws a description or spec to an SVG, PDF or PNG file (by `out`'s extension).

    Size comes from these arguments, else the spec's output.figure ("figure" in a
    description), else 960 px wide; `dpi` defaults to 300 for mm and in, 192 for px.
    """
    from ._data import DescriptionError, compile_spec, column_types

    path = Path(out).resolve()
    fmt = path.suffix[1:].lower()
    if fmt not in ("svg", "pdf", "png"):
        raise SpectacularError(f"cannot write {path.name}: the file must end in .svg, .pdf or .png", exit=2)
    description, base = load_spec(spec)
    types, rows = column_types(data) if data is not None else (None, None)
    try:
        compiled = compile_spec(description, base, types, rows)
    except DescriptionError as e:
        raise SpectacularError(str(e), e.problems, exit=1) from None
    figure = (compiled.get("output") or {}).get("figure") or {}
    u = unit or figure.get("unit") or "px"
    if u not in _CSS_PX:
        raise SpectacularError(f"unit must be mm, in or px, not {u!r}", exit=2)
    size = (width or figure.get("width") or 960, u, dpi or figure.get("dpi") or (192 if u == "px" else 300))
    try:
        import playwright.sync_api  # noqa: F401
    except ImportError:
        return _render_node(compiled, base, data, path, width, unit, dpi)
    # A notebook kernel runs an asyncio loop, inside which Playwright's sync API refuses to
    # start; a thread of its own has none.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_render_browser, compiled, base, data, path, fmt, *size).result()


def _render_browser(spec: Dict[str, Any], base: Path, data: Any, path: Path, fmt: str, width: float, unit: str, dpi: float) -> Path:
    from playwright.sync_api import sync_playwright

    from ._html import to_html

    css_width = round(width * _CSS_PX[unit])
    cwd = Path.cwd()
    try:
        os.chdir(base)  # a relative data file resolves against the spec, as everywhere else
        document = to_html(spec, data, figure=True)
    finally:
        os.chdir(cwd)
    with sync_playwright() as p:
        browser = p.chromium.launch(args=_BROWSER_ARGS)
        try:
            page = browser.new_page(viewport={"width": css_width, "height": 800}, device_scale_factor=dpi / 96, color_scheme="light")
            page.set_content(document)
            page.wait_for_function("document.body.dataset.state !== 'loading'", timeout=120_000)
            if page.evaluate("document.body.dataset.state") == "error":
                raise SpectacularError(page.inner_text("#spectacular"), exit=1)
            page.evaluate("new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))")
            root = page.locator("#spectacular .sv-root")
            height = math.ceil(root.bounding_box()["height"])
            if fmt == "svg":
                svg = page.evaluate("window.spectacular.toSVG()")
                if unit != "px":  # the root's width and height become physical lengths
                    def size(m: "re.Match[str]") -> str:
                        return f' {m.group(1)}="{round(float(m.group(2)) / _CSS_PX[unit], 3):g}{unit}"'
                    head, rest = svg.split("<svg", 1)
                    tag, body = rest.split(">", 1)
                    tag = re.sub(r' (width|height)="([\d.]+)"', size, tag, count=2)
                    svg = f"{head}<svg{tag}>{body}"
                path.write_text(svg, encoding="utf-8")
            elif fmt == "png":
                root.screenshot(path=str(path))
            else:
                # Printing lays the page out again, and a cleared WebGL canvas would print
                # blank, so canvases are fixed as images first.
                page.evaluate("""() => { for (const c of document.querySelectorAll('canvas')) {
                    const img = document.createElement('img'); img.src = c.toDataURL('image/png');
                    img.style.cssText = c.style.cssText;
                    Object.assign(img.style, { width: c.clientWidth + 'px', height: c.clientHeight + 'px', display: 'block' });
                    c.replaceWith(img); } }""")
                page.emulate_media(media="screen")
                page.pdf(path=str(path), width=f"{css_width}px", height=f"{height}px", print_background=True,
                         page_ranges="1", margin={"top": "0", "right": "0", "bottom": "0", "left": "0"})
        finally:
            browser.close()
    return path


def _node_command() -> List[str]:
    """`node <checkout>/cli/spectacular.mjs`; SPECTACULAR_CLI names the script elsewhere."""
    script = os.environ.get("SPECTACULAR_CLI")
    found = Path(script) if script else Path(__file__).resolve().parents[3] / "cli" / "spectacular.mjs"
    node = shutil.which("node")
    if not found.is_file() or not node:
        raise SpectacularError(
            "render draws figures in headless Chromium: pip install \"spectacular-charts[render]\" and run "
            "`python -m playwright install chromium` (or use a spectacular checkout with Node).",
            exit=2,
        )
    return [node, str(found)]


def _render_node(spec: Dict[str, Any], base: Path, data: Any, path: Path, width: Optional[float], unit: Optional[str], dpi: Optional[float]) -> Path:
    from ._data import write_csv

    args = ["render", "-o", str(path)]
    for flag, value in (("--width", width), ("--unit", unit), ("--dpi", dpi)):
        if value is not None:
            args += [flag, str(value)]
    with tempfile.TemporaryDirectory() as tmp:
        cwd = base
        if data is not None:
            types = write_csv(data, spec, Path(tmp) / "data.csv")
            spec["data"] = {"url": "data.csv"}
            fields = spec.setdefault("fields", {})
            for name, kind in types.items():
                fields.setdefault(name, {}).setdefault("type", kind)
            cwd = Path(tmp)
        proc = subprocess.run([*_node_command(), *args, "--json", "-"], input=json.dumps(spec), cwd=cwd,
                              capture_output=True, text=True, encoding="utf-8")
    lines = [json.loads(line) for line in proc.stdout.splitlines() if line.strip().startswith("{")]
    result = lines[0] if lines else {}
    if proc.returncode != 0:
        raise SpectacularError(result.get("error") or proc.stderr.strip() or f"render exited {proc.returncode}",
                               result.get("problems"), proc.returncode)
    return path
