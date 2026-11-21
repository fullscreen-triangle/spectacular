# Notebooks

## Showing charts

A view shows itself when it is the last expression of a cell:

```python
import spectacular as sp

df = sp.datasets.plate()
sp.view({"x": "time_h", "y": "od600", "color": "medium"}, data=df)
```

To show charts from the middle of a cell, or from a loop, use `sp.show`, which displays
them at once and returns the view:

```python
for medium in ["rich", "minimal"]:
    sp.show({"title": medium, "x": "rate"}, data=df[df.medium == medium])
```

`sp.chart` is another name for `sp.view`.

## The `%%spectacular` cell magic

`import spectacular` registers a cell magic in IPython. A cell that starts with
`%%spectacular` holds only a description, in JSON:

```python
%%spectacular df
{"title": "Growth plate",
 "labels": {"od600": "OD600"},
 "charts": [
   {"x-axis": "time_h", "y-axis": "od600", "color": "medium"},
   {"x": "rate", "color": "teal"},
   {"x": "medium", "wide": true}
 ]}
```

- The line after `%%spectacular` is any Python expression that gives the table:
  `%%spectacular df[df.time_h > 2]` works.
- `--as w` stores the view in `w`: `%%spectacular df --as w`.
- With no table, the description names a file: `"data": "readings.csv"`.
- The body is JSON. Python's literal syntax (`True`, single quotes, trailing commas) is
  accepted too, so a dict pasted from Python works.

Without importing spectacular first, `%load_ext spectacular` registers the magic.

## Interactive or static

| Installed | `sp.view` gives | Brushing | Selection in Python |
|---|---|---|---|
| `spectacular-charts[notebook]` (anywidget) | a `View` widget | yes | yes |
| `spectacular-charts` only | an `HTMLView`, in an iframe | yes | no |

The widget is an [anywidget](https://anywidget.dev). Anywidget runs in JupyterLab,
Jupyter Notebook 7, Visual Studio Code, Google Colab and other frontends with Jupyter
widgets. The project's own end-to-end test drives JupyterLab: it opens a notebook, draws
a view, brushes it and reads the selection back in the kernel.

The runtime (the WebAssembly crossfilter, its worker and the views) is carried inside
the package and sent with the widget, so charts draw without an internet connection.

## Saving a notebook with its charts

A widget's charts come from the running kernel, so a notebook saved and reopened later
shows no chart until the cell runs again. `sp.HTMLView` draws the same charts as an
offline page in an iframe, and that page is saved in the notebook file:

```python
sp.HTMLView({"x": "rate"}, data=df, height=360)
```

Jupyter shows it again when the notebook is reopened and trusted. Each one carries the
runtime (about 1 MB) and the data, so keep it to small tables. Sites that only display
notebooks, such as GitHub, do not run scripts; for those, save an image with
`sp.render(...)` and show that.

## Large tables in a notebook

Only the columns the charts use are sent, as binary arrays: 8 bytes per number and 4 per
category per row. A million rows and four numeric columns is about 32 MB, sent once.
After that, brushing is computed in the browser and only the selection comes back. Pass
`track_rows=False` to `sp.view` to stop the row positions coming back too, when you only
need the filters.

Two numbers above 200,000 rows become a density instead of a scatter; see
[Data](data.md#large-tables).
