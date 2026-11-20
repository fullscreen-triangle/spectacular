"""spectacular from Python: linked scientific views in Jupyter, as HTML pages, and as figures.

    import spectacular as sp

    spec = {
        "title": "Growth plate",
        "views": [
            {"id": "curves", "type": "scatter", "x": "time_h", "y": "od600", "color": "medium"},
            {"id": "rate", "type": "histogram", "field": "rate"},
        ],
    }
    w = sp.view(spec, data=df)        # interactive views in a notebook cell
    w.selection                       # the brushes, in data terms
    w.selected_data()                 # the brushed rows of df

    sp.save_html(spec, "plate.html", data=df)    # the same views as one offline page
    sp.render(spec, "plate.svg", data=df)        # a figure (needs Node and the checkout)

A spec is the JSON `spectacular-specifications.md` describes, as a dict, a path or JSON
text. With `data=`, its columns replace the spec's `data`; without it, the spec's own data
is used (a local `data.url` is read from disk).
"""

from ._cli import SpectacularError, render, validate
from ._html import save_html, to_html
from ._spec import SPEC_VERSION, fields_used, load_spec, schema
from ._view import HTMLView, View, show, view

__version__ = "0.2.0"

__all__ = [
    "SPEC_VERSION",
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
