"""Draws every image in the documentation from the descriptions below.

    python docs/figures.py            # all of them, into docs/assets/
    python docs/figures.py strip      # only those whose name contains "strip"

Needs the package with its render extra (`pip install -e "python[render]"` and
`python -m playwright install chromium`). Rerun it after a change to how charts look,
and commit the images: the PyPI page shows them from the repository.
"""

import sys
from pathlib import Path

import spectacular as sp

ASSETS = Path(__file__).parent / "assets"
DATA = {
    "plate": sp.datasets.plate,
    "lorenz": sp.datasets.lorenz,
    "bundle": sp.datasets.bundle,
    "lineage": sp.datasets.lineage,
}
OUTCOMES = {
    "published": "#1baf7a", "empty": "#c3c2b7", "open": "#6250d6", "missing": "#eda100",
    "blocked": "#e15759", "defect": "#e87ba4", "not-asked": "#9aa0ae",
}
BUNDLE = {
    "domains": {
        "way": ["stated", "supplied", "in-place", "tool", "rule", "node", "asked", "held"],
        "outcome": list(OUTCOMES),
    },
    "palettes": {"outcome": OUTCOMES},
    "key": "row",
}

# name: (data, description, width in CSS px)
FIGURES = {
    "hero": ("plate", {
        "title": "A 96-well growth plate, read every 30 minutes",
        "theme": "light",
        "columns": 3,
        "labels": {"time_h": "time (h)", "od600": "OD600", "rate": "growth rate (1/h)"},
        "palettes": {"medium": {"rich": "#4e79a7", "minimal": "#f28e2b"}},
        "charts": [
            {"x-axis": "time_h", "y-axis": "od600", "color": "medium", "title": "Growth curves", "span": 2},
            {"x": "medium", "title": "Wells per medium"},
            {"x": "rate", "bins": 40, "title": "Growth rates"},
            {"x": "medium", "y": "rate", "title": "Rate by medium"},
            {"x": "od600", "color": "slateblue", "title": "All readings"},
        ],
    }, 1100),
    "scatter": ("lorenz", {"x": "x", "y": "z", "color": "lobe", "title": "Lorenz attractor, x against z", "theme": "light"}, 620),
    "histogram": ("lorenz", {"x": "speed", "bins": 80, "title": "Speed", "theme": "light", "color": "teal"}, 620),
    "density": ("lorenz", {"x": "x", "y": "z", "type": "density", "title": "Density of states", "theme": "light"}, 620),
    "scatter3d": ("lorenz", {"x": "x", "y": "y", "z": "z", "color": "speed", "title": "Phase space", "theme": "light"}, 620),
    "bars": ("bundle", {**BUNDLE, "x": "way", "title": "Rows by way, in a declared order", "theme": "light"}, 620),
    "split-bars": ("bundle", {**BUNDLE, "x": "way", "color": "outcome", "title": "Each way split into outcomes", "theme": "light"}, 620),
    "strip": ("bundle", {**BUNDLE, "x": "outcome", "y": "margin", "title": "Margins by outcome", "theme": "light"}, 620),
    "matrix": ("bundle", {**BUNDLE, "x": "label", "y": "blocker", "title": "Refusals: label by blocker", "theme": "light"}, 620),
    "interval": ("bundle", {**BUNDLE, "low": "low", "high": "high", "y": "slot", "color": "outcome", "scale": "each",
                            "title": "Quantities from low end to high end", "theme": "light"}, 620),
    "sankey": ("bundle", {**BUNDLE, "from": "way", "to": "outcome", "title": "Way to outcome", "theme": "light"}, 620),
    "layers": ("lineage", {"domains": {"step": ["parse", "resolve", "measure", "decide", "publish"]},
                           "from": "from_step", "to": "step", "type": "layers", "title": "Rows passing from step to step",
                           "theme": "light"}, 620),
    "arc": ("lineage", {"from": "from_step", "to": "step", "type": "arc", "title": "Steps on a line", "theme": "light"}, 620),
    "network3d": ("lineage", {"from": "from_step", "to": "step", "type": "network", "title": "Steps in 3D", "theme": "light"}, 620),
    "dark": ("plate", {
        "title": "The same charts, dark", "theme": "dark", "columns": 2,
        "charts": [{"x": "time_h", "y": "od600", "color": "medium"}, {"x": "rate", "bins": 40}],
    }, 900),
}


def main(only: str = "") -> None:
    ASSETS.mkdir(exist_ok=True)
    tables = {}
    for name, (data, description, width) in FIGURES.items():
        if only not in name:
            continue
        table = tables.setdefault(data, DATA[data]())
        out = sp.render(description, ASSETS / f"{name}.png", data=table, width=width, unit="px", dpi=192)
        print(f"{out.relative_to(ASSETS.parent)}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
