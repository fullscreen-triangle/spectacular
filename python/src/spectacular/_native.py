"""The native `spectacular` command that ships in the wheel, built from crates/spectacular.

It compiles chart descriptions into view specs and validates specs, so neither needs
Node or a browser.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import sysconfig
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EXE = "spectacular.exe" if os.name == "nt" else "spectacular"


def _is_native(path: Path) -> bool:
    """True for the Rust command (the Node one, if linked on the PATH, answers otherwise)."""
    try:
        out = subprocess.run([str(path), "--version"], capture_output=True, text=True, timeout=10)
    except OSError:
        return False
    return out.returncode == 0 and out.stdout.startswith("spectacular ")


@lru_cache(maxsize=1)
def binary() -> Optional[Path]:
    """The native command: SPECTACULAR_BIN, this interpreter's scripts directory (where the
    wheel installs it), the user scripts directory, a checkout's build, then the PATH."""
    candidates = []
    if os.environ.get("SPECTACULAR_BIN"):
        candidates.append(Path(os.environ["SPECTACULAR_BIN"]))
    candidates.append(Path(sysconfig.get_path("scripts")) / EXE)
    try:
        candidates.append(Path(sysconfig.get_path("scripts", f"{os.name}_user")) / EXE)
    except KeyError:  # pragma: no cover - schemes vary by platform
        pass
    candidates.append(Path(sys.executable).parent / EXE)
    checkout = Path(__file__).resolve().parents[3] / "target"
    candidates += [checkout / "release" / EXE, checkout / "debug" / EXE]
    found = shutil.which("spectacular")
    if found:
        candidates.append(Path(found))
    return next((c for c in candidates if c.is_file() and _is_native(c)), None)


def run(args: List[str], stdin: Optional[str] = None) -> Tuple[int, List[Dict[str, Any]], str]:
    """Runs the command with --json; returns (exit code, JSON lines, stderr)."""
    exe = binary()
    if exe is None:
        from ._cli import SpectacularError

        raise SpectacularError(
            "the native spectacular command was not found. It ships in the wheel "
            "(pip install spectacular-charts); from a checkout, run `cargo build --release -p spectacular` "
            "or set SPECTACULAR_BIN."
        )
    proc = subprocess.run([str(exe), *args, "--json"], input=stdin, capture_output=True, text=True, encoding="utf-8")
    lines = [json.loads(l) for l in proc.stdout.splitlines() if l.strip().startswith("{")]
    return proc.returncode, lines, proc.stderr.strip()


def compile_description(
    description: Any, columns: Optional[Dict[str, str]] = None, rows: Optional[int] = None, base: Optional[Path] = None
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, str]]]:
    """(spec, problems) for a chart description. `columns` maps the data's column names to
    "quantitative" or "nominal"; without them, the description's own data file is read."""
    envelope = {"description": description, "columns": columns, "rows": rows, "base": str(base or Path.cwd())}
    code, lines, err = run(["compile", "--envelope", "-"], json.dumps(envelope))
    out = lines[0] if lines else {}
    if code == 2 or not lines:
        from ._cli import SpectacularError

        raise SpectacularError(out.get("error") or err or f"compile exited {code}", exit=code)
    return out.get("spec"), out.get("problems", [])
