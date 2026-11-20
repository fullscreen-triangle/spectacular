# spectacular for Python

Linked 2D and 3D views of one table, in a Jupyter notebook, as an offline HTML page, or
as an SVG/PDF/PNG figure. Brushing a range in one view filters all the others. The
filtering runs in the browser, in a WebAssembly crossfilter, so it stays interactive at
millions of rows.

```python
import spectacular as sp

spec = {
    "title": "Growth plate",
    "fields": {"time_h": {"title": "time (h)"}, "od600": {"title": "OD600"}},
    "views": [
        {"id": "curves", "type": "scatter", "x": "time_h", "y": "od600", "color": "medium", "span": 2},
        {"id": "rate", "type": "histogram", "field": "rate", "bins": 40},
        {"id": "medium", "type": "bars", "field": "medium"},
    ],
    "layout": {"columns": 2},
}

w = sp.view(spec, data=df)   # last expression in a cell: the views appear below it
```

After brushing, in a later cell:

```python
w.selection        # {"selected": 412, "rows": 2400, "filters": [{"view": "rate", "field": "rate", "range": [0.31, 0.42]}]}
w.selected_rows    # numpy array of row indices into df
w.selected_data()  # df.iloc[w.selected_rows]
w.on_select(lambda s: print(s["selected"]))   # called after each brush settles
w.save_svg("plate.svg")                        # what the view shows now, brushes included
```

The spec is the JSON described in
[`spectacular-specifications.md`](../spectacular-specifications.md), as a dict, a path to
a spec file, or JSON text. Programs can generate it; `sp.schema()` returns its
JSON Schema.

## Data

`data=` takes a pandas or polars DataFrame, a dict of columns, a list of row dicts, or a
numpy record array. Only the columns the views name (and those in `fields`) are sent, as
binary columns: numbers as float64, everything else as categories. Declare a numeric
code (a well number, a detector id) as `"nominal"` in `fields`. Dates have no axis type
yet; convert them to numbers first.

Without `data=`, the spec's own `data` is used. A local `data.url` (CSV, TSV, JSON, or
JSON for Linked Data (JSON-LD) with a `pointer`) is read from disk, relative to the spec
file, or to the current directory for a dict.

## Other ways out

| Call | Gives | Needs |
|---|---|---|
| `sp.view(spec, data)` | interactive views in a notebook, selection readable in Python | `anywidget` |
| `sp.show(spec, data)` | the same, displayed now | `anywidget` |
| `sp.save_html(spec, "x.html", data)` | one offline page with the views | nothing |
| `sp.render(spec, "x.svg", data)` | a figure: `.svg`, `.pdf` or `.png` | Node.js and the spectacular checkout |
| `sp.validate(spec, data)` | the spec's problems, each `{"path", "message"}` | Node.js and the checkout |

Without `anywidget`, `sp.view` shows the views in an iframe instead; brushing still
works, but nothing comes back to Python.

The views take the notebook's theme (JupyterLab, Visual Studio Code and Colab dark themes
are detected). A spec can fix it with `"theme": "light" | "dark" | "paper"`; `paper` is for
print: white, no panel borders, darker axes.

## Install

From a spectacular checkout, after the JavaScript build:

```bash
npm run build:wasm && npm install && npm run build   # builds python/src/spectacular/static too
pip install -e "python[notebook]"
```

`npm run build:python` rebuilds only the notebook runtime. `pip wheel python` (or
`uv build python`) makes a wheel that carries it, for machines without the checkout;
`render` and `validate` still need the checkout there, or `SPECTACULAR_CLI` pointing at
its `cli/spectacular.mjs`.
