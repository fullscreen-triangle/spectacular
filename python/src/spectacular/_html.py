"""Standalone interactive pages: one HTML file holding the runtime, the spec and the data."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any, Optional, Union

from ._data import payload
from ._spec import SpecLike, static_file


def _wrap(value: Any) -> Any:
    """Binary values as {"$b64": ...}, which the page unwraps."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"$b64": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, dict):
        return {k: _wrap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def to_html(spec: SpecLike, data: Any = None, *, title: Optional[str] = None, figure: bool = False) -> str:
    """A complete HTML document with the linked views, working offline in any browser.

    Brushing works as in a notebook; the page reports nothing back. It embeds the runtime
    (about 1 MB) and the data, so keep it to tables you would also send by email.
    `figure=True` drops the page margin and the status line, for drawing figures.
    """
    spec, columns, file = payload(spec, data)
    # "</" cannot appear inside a <script> element; both escapes keep the meaning.
    runtime = static_file("spectacular.js").read_text("utf-8").replace("</script", "<\\/script")
    wire = {"spec": spec, "columns": columns, "file": file, "figure": figure}
    data_json = json.dumps(_wrap(wire)).replace("</", "<\\/")
    heading = html.escape(title or spec.get("title") or "spectacular")
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{heading}</title>\n"
        f"<style>:root {{ color-scheme: light dark; }} body {{ margin: 0; padding: {0 if figure else 8}px; background: Canvas; }}</style>\n"
        "</head>\n<body data-state=\"loading\">\n<div id=\"spectacular\"></div>\n"
        f"<script type=\"application/json\" id=\"spectacular-payload\">{data_json}</script>\n"
        f"<script type=\"module\">\n{runtime}\n"
        "globalThis.spectacularNotebook.boot(document.getElementById(\"spectacular\"),\n"
        "  JSON.parse(document.getElementById(\"spectacular-payload\").textContent));\n"
        "</script>\n</body>\n</html>\n"
    )


def save_html(spec: SpecLike, path: Union[str, "Path"], data: Any = None, *, title: Optional[str] = None) -> Path:
    """Writes `to_html(...)` to `path` and returns it."""
    out = Path(path)
    out.write_text(to_html(spec, data, title=title), encoding="utf-8")
    return out


def iframe(document: str, height: int = 480) -> str:
    """An iframe holding `document`; the page grows the frame to fit its views."""
    return (
        f'<iframe srcdoc="{html.escape(document, quote=True)}" '
        f'style="width: 100%; height: {height}px; border: 0;" loading="eager"></iframe>'
    )
