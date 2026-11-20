"""Tables from Python, turned into the columns the views read.

A column travels as bytes: little-endian float64 for a quantitative field, or uint32
codes plus their labels for a nominal one. That is what the browser's crossfilter takes
without parsing, so a million-row DataFrame costs one copy, not a CSV round trip.
"""

from __future__ import annotations

import array
import math
import numbers
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import unquote

from ._spec import SpecLike, fields_used, is_spec, load_spec

try:  # numpy is optional; with it, numeric columns are converted without a Python loop
    import numpy as _np
except ImportError:  # pragma: no cover - exercised only without numpy
    _np = None

Wire = Dict[str, Any]


def columns_of(data: Any) -> Dict[str, Any]:
    """Name -> sequence for a pandas or polars DataFrame, a dict of columns, rows, or a numpy record array."""
    if hasattr(data, "columns") and hasattr(data, "iloc"):  # pandas
        return {str(c): data[c] for c in data.columns}
    if hasattr(data, "columns") and hasattr(data, "get_column"):  # polars
        return {str(c): data.get_column(c) for c in data.columns}
    if isinstance(data, Mapping):
        return {str(k): v for k, v in data.items()}
    if _np is not None and isinstance(data, _np.ndarray) and data.dtype.names:
        return {name: data[name] for name in data.dtype.names}
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)) and all(isinstance(r, Mapping) for r in data):
        names: List[str] = []
        for row in data:
            names.extend(str(k) for k in row if str(k) not in names)
        return {n: [row.get(n) for row in data] for n in names}
    raise TypeError(
        "data is a pandas or polars DataFrame, a dict of columns, a list of row dicts, "
        f"or a numpy record array, not {type(data).__name__}"
    )


def _missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or type(v).__name__ in ("NAType", "NaTType")


def _is_number(v: Any) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, bool) and type(v).__name__ != "bool_"


def _label(v: Any) -> str:
    if _missing(v):
        return ""
    if isinstance(v, bool) or type(v).__name__ == "bool_":
        return "true" if v else "false"
    return str(v)


def _float64(values: List[Any]) -> bytes:
    out = array.array("d", (math.nan if _missing(v) else float(v) for v in values))
    if sys.byteorder == "big":  # pragma: no cover
        out.byteswap()
    return out.tobytes()


def _uint32(values: List[int]) -> bytes:
    code = "I" if array.array("I").itemsize == 4 else "L"
    out = array.array(code, values)
    if sys.byteorder == "big":  # pragma: no cover
        out.byteswap()
    return out.tobytes()


def _nominal(values: List[Any]) -> Wire:
    index: Dict[str, int] = {}
    codes = []
    for v in values:
        key = _label(v)
        code = index.setdefault(key, len(index))
        codes.append(code)
    return {"type": "nominal", "codes": _uint32(codes), "labels": list(index)}


def column_type(values: Any) -> str:
    """The type `encode_column` will send a column as, when the spec declares none."""
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    if _np is not None and isinstance(values, _np.ndarray):
        if values.dtype.kind in "iufMm":  # dates are refused later, by encode_column
            return "quantitative"
        values = values.tolist()
    return "quantitative" if all(_missing(v) or _is_number(v) for v in values) else "nominal"


def column_types(data: Any) -> Tuple[Dict[str, str], int]:
    """Every column's type, in the data's order, and the number of rows."""
    columns = columns_of(data)
    rows = len(next(iter(columns.values()))) if columns else 0
    return {name: column_type(values) for name, values in columns.items()}, rows


def encode_column(values: Any, declared: Optional[str] = None) -> Wire:
    """One column on the wire. `declared` is the spec's type for it, if any."""
    if hasattr(values, "to_numpy"):  # pandas and polars Series
        values = values.to_numpy()
    if _np is not None and isinstance(values, _np.ndarray):
        kind = values.dtype.kind
        if kind in "Mm":
            raise TypeError(
                "dates and durations have no axis type yet; convert them to numbers first, "
                "e.g. (t - t.min()).dt.total_seconds()"
            )
        if kind in "iuf" and declared != "nominal":
            return {"type": "quantitative", "values": values.astype("<f8").tobytes()}
        values = values.tolist()
    values = list(values)
    if declared == "quantitative":
        def number(v: Any) -> float:
            try:
                return math.nan if _missing(v) else float(v)
            except (TypeError, ValueError):
                return math.nan
        return {"type": "quantitative", "values": _float64([number(v) for v in values])}
    if declared != "nominal" and all(_missing(v) or _is_number(v) for v in values):
        return {"type": "quantitative", "values": _float64(values)}
    return _nominal(values)


def encode(data: Any, spec: Dict[str, Any]) -> Dict[str, Wire]:
    """The columns `spec` reads, from `data`. A field the data lacks is left out, and the
    page then reports it by its path in the spec, as for any other data source."""
    columns = columns_of(data)
    declared = {name: (d or {}).get("type") for name, d in (spec.get("fields") or {}).items() if isinstance(d, dict)}
    return {name: encode_column(columns[name], declared.get(name)) for name in fields_used(spec) if name in columns}


_REMOTE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.I)


class DescriptionError(ValueError):
    """A chart description (or spec) the compiler refused; `problems` says where and why."""

    def __init__(self, problems: List[Dict[str, str]]):
        self.problems = problems
        lines = "\n".join(f"  {p['path'] or '/'}: {p['message']}" for p in problems)
        super().__init__(f"the chart description has {len(problems)} problem(s):\n{lines}")


def compile_spec(description: Any, base: Path, columns: Optional[Dict[str, str]] = None, rows: Optional[int] = None) -> Dict[str, Any]:
    """A view spec from a description, through the native command. A full spec passes
    through it too; without the command, a full spec is used as it is."""
    from . import _native

    if columns is not None and is_spec(description):
        description = {**description, "data": {"columns": "external"}}  # the table is the data given
    if _native.binary() is None and is_spec(description):
        return description
    spec, problems = _native.compile_description(description, columns, rows, base)
    if problems or spec is None:
        raise DescriptionError(problems)
    return spec


def payload(spec: SpecLike, data: Any = None) -> Tuple[Dict[str, Any], Optional[Dict[str, Wire]], Optional[Wire]]:
    """(spec, columns, file) for the page, from a description or a spec.

    With `data`, the description is compiled against its columns, which then replace the
    spec's data. Without it, the description's own data file is read: a local file has its
    bytes sent along, so the page needs no server; a remote URL is fetched by the page.
    """
    description, base = load_spec(spec)
    if data is not None:
        types, rows = column_types(data)
        spec = compile_spec(description, base, types, rows)
        columns = encode(data, spec)
        spec["data"] = {"columns": "external"}
        return spec, columns, None
    spec = compile_spec(description, base)
    source = spec.get("data")
    url = source.get("url") if isinstance(source, dict) else None
    if isinstance(url, str) and not _REMOTE.match(url) and not url.startswith("data:"):
        path = base / unquote(re.split(r"[?#]", url, 1)[0])
        if not path.is_file():
            raise FileNotFoundError(f"data.url {url!r}: no file at {path}")
        return spec, None, {"name": url, "bytes": path.read_bytes()}
    return spec, None, None


def types_of(columns: Dict[str, Wire]) -> Dict[str, str]:
    return {name: c["type"] for name, c in columns.items()}


def write_csv(data: Any, spec: Dict[str, Any], path: Path) -> Dict[str, str]:
    """Writes the fields `spec` reads to a CSV file and returns the type chosen for each,
    so a spec rendered from the file reads them the same way a notebook view would."""
    import csv

    columns = columns_of(data)
    names = [n for n in fields_used(spec) if n in columns]
    wire = encode(data, spec)
    lists = []
    for name in names:
        values = columns[name]
        if hasattr(values, "to_numpy"):
            values = values.to_numpy()
        lists.append(values.tolist() if hasattr(values, "tolist") else list(values))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(names)
        for row in zip(*lists):
            writer.writerow(
                "" if _missing(v) else (_label(v) if wire[n]["type"] == "nominal" else repr(float(v)) if _is_number(v) else v)
                for n, v in zip(names, row)
            )
    return types_of(wire)
