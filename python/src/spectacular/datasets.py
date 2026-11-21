"""Small datasets for trying spectacular, used by the examples in its documentation.

    import spectacular as sp

    df = sp.datasets.plate()      # 2,400 optical-density readings of a growth plate

Each function returns a pandas DataFrame when pandas is installed, and a list of row dicts
otherwise; both are accepted as `data=`. Pass `frame=False` for the list of rows. The
numbers are generated (or, for the bundle tables, were generated) from fixed seeds: they
are realistic in shape, not measurements.
"""

from __future__ import annotations

import csv
import json
import math
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).parent / "data"

Rows = List[Dict[str, Any]]


def _frame(rows: Rows, frame: bool) -> Any:
    if frame:
        try:
            import pandas as pd

            return pd.DataFrame(rows)
        except ImportError:
            pass
    return [dict(r) for r in rows]


@lru_cache(maxsize=1)
def _plate() -> Rows:
    rng = random.Random(7)
    rows: Rows = []
    for row in "ABCDEFGH":
        for col in range(1, 13):
            medium = "rich" if col <= 6 else "minimal"
            rate = rng.gauss(0.55 if medium == "rich" else 0.35, 0.06)
            lag = rng.uniform(1.0, 3.0)
            for k in range(25):
                t = k * 0.5
                od = 0.05 + 1.1 / (1 + math.exp(-rate * 2 * (t - lag - 3)))
                rows.append({"well": f"{row}{col}", "row": row, "column": col, "medium": medium,
                             "time_h": t, "od600": round(od + rng.gauss(0, 0.01), 4), "rate": round(rate, 4)})
    return rows


def plate(frame: bool = True) -> Any:
    """A 96-well growth plate read every 30 minutes for 12 hours: 2,400 rows.

    Columns: `well` ("A1".."H12"), `row` ("A".."H"), `column` (1..12, a plate position),
    `medium` ("rich" in columns 1-6, "minimal" in 7-12), `time_h`, `od600` (optical
    density) and `rate` (each well's growth rate, per hour).
    """
    return _frame(_plate(), frame)


@lru_cache(maxsize=1)
def _lorenz() -> Rows:
    sigma, rho, beta, dt, steps = 10.0, 28.0, 8.0 / 3.0, 0.002, 150_000

    def f(x: float, y: float, z: float):
        return sigma * (y - x), x * (rho - z) - y, x * y - beta * z

    x, y, z = 1.0, 1.0, 1.0
    rows: Rows = []
    for i in range(1, steps + 1):
        a = f(x, y, z)
        b = f(x + dt / 2 * a[0], y + dt / 2 * a[1], z + dt / 2 * a[2])
        c = f(x + dt / 2 * b[0], y + dt / 2 * b[1], z + dt / 2 * b[2])
        d = f(x + dt * c[0], y + dt * c[1], z + dt * c[2])
        x += dt / 6 * (a[0] + 2 * b[0] + 2 * c[0] + d[0])
        y += dt / 6 * (a[1] + 2 * b[1] + 2 * c[1] + d[1])
        z += dt / 6 * (a[2] + 2 * b[2] + 2 * c[2] + d[2])
        if i % 5 == 0 and i * dt > 1:
            v = f(x, y, z)
            rows.append({"t": round(i * dt, 2), "x": round(x, 4), "y": round(y, 4), "z": round(z, 4),
                         "speed": round(math.hypot(*v), 3), "lobe": "left" if x < 0 else "right"})
    return rows


def lorenz(frame: bool = True) -> Any:
    """29,900 states of the Lorenz system (sigma 10, rho 28, beta 8/3), every 0.01 time units.

    Columns: `t`, `x`, `y`, `z`, `speed` (the length of the velocity) and `lobe` ("left"
    or "right": the sign of x). Computed on first use, which takes a few seconds.
    """
    return _frame(_lorenz(), frame)


def bundle(frame: bool = True) -> Any:
    """260 rows of a view bundle, the shape a provenance pipeline writes: one line per
    slot of a record, with the way it got its value and the outcome.

    Columns: `row` (the row's id), `record`, `kind`, `parent`, `slot`, `step`, `way`,
    `graph`, `basis`, `status`, `outcome`, `reason`, `label`, `blocker`, `value` (text),
    and the quantities `low`, `high` (the two ends of a value), `margin` (nats) and
    `sources`. A category with no value holds "none"; a number with none is missing.
    """
    return _frame(json.loads((_HERE / "bundle.rows.json").read_text("utf-8")), frame)


def lineage(frame: bool = True) -> Any:
    """695 links of rows passing from one step of a pipeline to the next: `row`, `slot`,
    `from_step` and `step`. Some rows are measured again after a decision, a cycle."""
    with open(_HERE / "bundle.lineage.csv", newline="", encoding="utf-8") as f:
        return _frame(list(csv.DictReader(f)), frame)
