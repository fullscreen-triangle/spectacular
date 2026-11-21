"""Specs as Python sees them: loading one, and the JSON Schema it is checked against."""

from __future__ import annotations

import copy
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

SPEC_VERSION = "spectacular/v0"

# The schema marks every property that names a field of the data with this description.
_FIELD = "the name of a field in the data"

#: A view spec or a chart description: a dict, a list of charts, a path to a JSON file,
#: or JSON text.
SpecLike = Union[Dict[str, Any], List[Any], str, "os.PathLike[str]"]


def static_file(name: str) -> Path:
    path = Path(__file__).parent / "static" / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. It is built from the spectacular checkout with "
            "`npm run build:python`, then `pip install -e python` (or a wheel built after it)."
        )
    return path


@lru_cache(maxsize=1)
def _schema() -> Dict[str, Any]:
    return json.loads(static_file("spectacular.schema.json").read_text("utf-8"))


def schema() -> Dict[str, Any]:
    """The JSON Schema (draft 2020-12) for specs, generated from the validator's own rules."""
    return copy.deepcopy(_schema())


@lru_cache(maxsize=1)
def field_keys() -> Dict[str, Tuple[str, ...]]:
    """For each view type, the properties that name a field: {"scatter": ("x", "y", "color"), ...}."""
    out: Dict[str, Tuple[str, ...]] = {}
    for view in _schema()["properties"]["views"]["items"]["oneOf"]:
        props = view["properties"]
        out[props["type"]["const"]] = tuple(k for k, p in props.items() if p.get("description") == _FIELD)
    return out


def load_spec(spec: SpecLike) -> Tuple[Any, Path]:
    """A spec or description as new data, and the directory a relative data file resolves
    against.

    `spec` is a dict, a list of charts, a path to a JSON file, or JSON text. A dict, list
    or text resolves the data file against the current directory; a file, against its own.
    """
    if isinstance(spec, (dict, list)):
        return copy.deepcopy(spec), Path.cwd()
    if isinstance(spec, str) and spec.lstrip()[:1] in ("{", "["):
        return json.loads(spec), Path.cwd()
    if isinstance(spec, (str, os.PathLike)):
        path = Path(spec)
        return json.loads(path.read_text("utf-8")), path.resolve().parent
    raise TypeError(f"a spec is a dict, a path to a spec file, or JSON text, not {type(spec).__name__}")


def is_spec(value: Any) -> bool:
    """True for a full view spec: every view has an id and a type (the native compiler's
    test, so both sides agree on what passes through unchanged)."""
    views = value.get("views") if isinstance(value, dict) else None
    types = set(field_keys())
    return bool(views) and all(
        isinstance(v, dict) and isinstance(v.get("id"), str) and v.get("type") in types for v in views
    )


def fields_used(spec: Dict[str, Any]) -> List[str]:
    """The fields a spec reads: its key, those its views name, then those it declares,
    without repeats."""
    keys = field_keys()
    names: List[str] = []
    if isinstance(spec.get("key"), str):
        names.append(spec["key"])
    for view in spec.get("views") or []:
        if not isinstance(view, dict):
            continue
        for key in keys.get(view.get("type"), ()):
            name = view.get(key)
            if isinstance(name, str) and name not in names:
                names.append(name)
    for name in spec.get("fields") or {}:
        if name not in names:
            names.append(name)
    return names
