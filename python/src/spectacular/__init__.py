"""spectacular from Python: linked, crossfiltered charts described in JSON, no plotting code.

    import spectacular as sp

    w = sp.view({
        "title": "Growth plate",
        "charts": [
            {"x-axis": "time_h", "y-axis": "od600", "color": "medium"},   # a scatter
            {"x": "rate", "color": "teal"},                                # a histogram
            {"x": "medium"},                                               # bars
        ],
    }, data=df)
    w.selection                       # what was brushed, in data terms
    w.selected_data()                 # the brushed rows of df

Or a cell holding only the description, after `import spectacular`:

    %%spectacular df
    {"charts": [{"x": "time_h", "y": "od600"}, {"x": "rate"}]}

The chart type follows from the columns; `color` takes a column, a colour or a list of
colours. Full view specs (`spectacular-specifications.md`) are accepted everywhere a
description is. `save_html` writes the views as an offline page; `render` draws a figure.
"""

from . import datasets
from ._cli import SpectacularError, render, validate
from ._data import DescriptionError
from ._magic import load_ipython_extension
from ._html import save_html, to_html
from ._spec import SPEC_VERSION, fields_used, load_spec, schema
from ._view import HTMLView, View, show, view

__version__ = "0.2.1"

#: Another name for `view`, for those who think in charts.
chart = view

try:  # inside IPython, `import spectacular` makes %%spectacular available
    from IPython import get_ipython as _get_ipython

    if _get_ipython() is not None:
        load_ipython_extension(_get_ipython())
except ImportError:  # pragma: no cover - IPython is optional
    pass

__all__ = [
    "SPEC_VERSION",
    "DescriptionError",
    "chart",
    "datasets",
    "load_ipython_extension",
    "HTMLView",
    "SpectacularError",
    "View",
    "fields_used",
    "load_spec",
    "render",
    "save_html",
    "schema",
    "show",
    "to_html",
    "validate",
    "view",
]
