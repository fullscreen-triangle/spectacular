"""Figures and validation through the `spectacular` command (Node), for scripts and pipelines."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ._data import write_csv
from ._spec import SpecLike, load_spec


class SpectacularError(Exception):
    """The command refused a spec (`problems` lists them by path) or could not run."""

    def __init__(self, message: str, problems: Optional[List[Dict[str, str]]] = None, exit: Optional[int] = None):
        super().__init__(message)
        self.problems = problems or []
        self.exit = exit


def command() -> List[str]:
    """`node <checkout>/cli/spectacular.mjs`. SPECTACULAR_CLI names the script when this
    package is not installed from a spectacular checkout."""
    script = os.environ.get("SPECTACULAR_CLI")
    candidates = [Path(script)] if script else [Path(__file__).resolve().parents[3] / "cli" / "spectacular.mjs"]
    found = next((c for c in candidates if c.is_file()), None)
    if not found:
        raise SpectacularError(
            "render and validate run the spectacular command, which was not found. Install this "
            "package from a spectacular checkout (pip install -e python) or set SPECTACULAR_CLI "
            "to its cli/spectacular.mjs."
        )
    node = shutil.which("node")
    if not node:
        raise SpectacularError("render and validate need Node.js 22.18 or later on the PATH")
    return [node, str(found)]


def _run(args: List[str], spec: SpecLike, data: Any) -> tuple:
    spec, base = load_spec(spec)
    with tempfile.TemporaryDirectory() as tmp:
        cwd = base
        if data is not None:
            types = write_csv(data, spec, Path(tmp) / "data.csv")
            spec["data"] = {"url": "data.csv"}
            fields = spec.setdefault("fields", {})
            for name, kind in types.items():
                fields.setdefault(name, {}).setdefault("type", kind)
            cwd = Path(tmp)
        proc = subprocess.run(
            [*command(), *args, "--json", "-"],
            input=json.dumps(spec), cwd=cwd, capture_output=True, text=True, encoding="utf-8",
        )
    lines = [json.loads(line) for line in proc.stdout.splitlines() if line.strip().startswith("{")]
    return proc.returncode, (lines[0] if lines else {}), proc.stderr.strip()


def validate(spec: SpecLike, data: Any = None) -> List[Dict[str, str]]:
    """The spec's problems, each {"path", "message"}; an empty list when it is valid.

    With `data` (or a local data.url) the fields are checked against the data too.
    """
    code, out, err = _run(["validate"], spec, data)
    if code not in (0, 1):
        raise SpectacularError(out.get("error") or err or f"validate exited {code}", exit=code)
    return out.get("problems", [])


def render(
    spec: SpecLike,
    out: Union[str, "os.PathLike[str]"],
    data: Any = None,
    *,
    width: Optional[float] = None,
    unit: Optional[str] = None,
    dpi: Optional[float] = None,
) -> Path:
    """Draws the spec to an SVG, PDF or PNG file (by `out`'s extension) and returns its path.

    Size comes from these arguments, else the spec's output.figure, else 960 px wide.
    It starts headless Chromium, so it takes a few seconds.
    """
    path = Path(out).resolve()
    args = ["render", "-o", str(path)]
    for flag, value in (("--width", width), ("--unit", unit), ("--dpi", dpi)):
        if value is not None:
            args += [flag, str(value)]
    code, result, err = _run(args, spec, data)
    if code != 0:
        raise SpectacularError(result.get("error") or err or f"render exited {code}", result.get("problems"), code)
    return path
