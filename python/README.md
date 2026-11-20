# spectacular for Python

Linked, crossfiltered charts in Jupyter, described in JSON instead of written as plotting
code. Brush a range in one chart and every other chart filters. The filtering runs in the
browser, in a WebAssembly crossfilter, so it stays interactive at millions of rows.

```python
import spectacular        # once; it also turns on the %%spectacular cell magic
```

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

That cell draws a scatter of optical density (OD600) over time coloured by medium, a teal histogram of
growth rates, and a bar chart of media across the full width, all linked. Nothing in it is
code: the chart types follow from the columns.

## Describing charts

A description is JSON (or a Python dict). The keys are forgiving: `"x-axis"`,
`"X axis"` and `"x_axis"` are one key, and column names match without regard to case or
spacing.

| Key | Means |
|---|---|
| `x`, `y`, `z` (or `x-axis`, `y-axis`, `z-axis`) | columns on each axis |
| `from`, `to` | columns of an edge table: a node to another |
| `low`, `high` | the two ends of a quantity, drawn as an interval |
| `color` (or `colour`) | a column to colour by, one colour (`"teal"`, `"#4e79a7"`), or a list of colours |
| `palette` (or `colors`) | the colours for the categories, in order of first appearance |
| `type` (or `chart`) | ask for a chart type instead of the inferred one |
| `title`, `bins`, `size`, `height`, `wide` | panel title, histogram bins, point size, plot height, full width |
| `scale` | on an interval: `"each"` for a scale per band (quantities in different units) |
| `brush` | `false` to stop the chart filtering the others |

The chart type follows from the columns when `type` is not given:

| Columns | Chart |
|---|---|
| one number | histogram |
| one category | bars (click to select); with a category `color`, each bar split into it |
| two numbers | scatter (a density above 200,000 rows) |
| a number and a category | strip: one row of points per category |
| two categories | matrix: a cell per pair, shaded by its rows |
| `low` and `high` | interval, banded by a category given as `y` |
| three numbers | 3D scatter |
| `from` and `to` | sankey (`"type": "arc"`, `"network"` or `"layers"` for the others) |

Around the charts:

| Key | Means |
|---|---|
| `charts` | the list of charts (or give a list at the top, or one chart's keys at the top) |
| `title`, `theme` | the heading; `"light"`, `"dark"` or `"paper"` (default: the notebook's) |
| `labels` | axis labels by column, e.g. `{"od600": "OD600"}` |
| `categorical` | columns of numbers that are categories, e.g. `["well"]` |
| `domains` | a column's categories in order, e.g. `{"way": ["stated", "supplied", "tool"]}`; empty ones keep their place |
| `palettes` | a column's colours, everywhere it is drawn: a list, or by name, e.g. `{"outcome": {"published": "#1baf7a"}}` |
| `key` | the column whose values name rows, e.g. `"row"` |
| `columns` | charts per row |
| `data` | a CSV, TSV or JSON file, when no table is given in Python |
| `figure` | print size for `render`, e.g. `{"format": "pdf", "width": 180, "unit": "mm"}` |

A palette given with `color` for a category column becomes that column's own, so every
chart of it uses the same colours. Mistakes come back at the key you wrote, with a suggestion:
`/charts/0/x-axis: no column "od60"; did you mean "od600"?`. In a notebook they show in
the cell, where the charts would have been.

Full view specs, as [`spectacular-specifications.md`](../spectacular-specifications.md)
describes them, are accepted everywhere a description is. `spectacular.schema()` returns
their JSON Schema, for programs that generate specs.

## From Python

```python
import spectacular as sp

w = sp.view(description, data=df)   # the cell's last expression: the charts appear
w.selection                         # {"selected": 412, "rows": 2400, "filters": [{"view": "rate", "field": "rate", "range": [0.31, 0.42]}]}
w.selected_data()                   # the brushed rows of df
w.on_select(lambda s: print(s["selected"]))   # after each brush settles
w.update(description)               # draw a new description over the same data
w.save_svg("plate.svg")             # what the charts show now, brushes included
w.selected_keys                     # with a "key": the key values of the brushed rows
w.select(["r001", "r007"])          # with a "key": show rows chosen elsewhere, in every chart

sp.validate(description, data=df)   # the problems, each {"path", "message"}; [] when it holds
sp.save_html(description, "plate.html", data=df)   # one offline page with the linked charts
sp.render(description, "plate.pdf", data=df)       # a figure: .svg, .pdf or .png
```

`%%spectacular df --as w` keeps the charts in `w`; the line takes any expression
(`df[df.medium == "rich"]`). `data=` takes a pandas or polars DataFrame, a dict of
columns, a list of row dicts, or a numpy record array. Only the columns the charts use are
sent, as binary arrays. Dates have no axis type yet; convert them to numbers first.

## Install

```bash
pip install spectacular-charts               # the charts, descriptions, offline pages
pip install "spectacular-charts[notebook]"   # + interactive charts in Jupyter (anywidget)
pip install "spectacular-charts[render]"     # + figures: then `python -m playwright install chromium`
```

The import name is `spectacular`. The wheel carries everything it needs: the browser
runtime and the native `spectacular` command (compiled Rust), which compiles descriptions
and validates them without Node. The command is on the `PATH` after installing:

```bash
spectacular compile chart.json --data readings.csv   # the view spec a description makes
spectacular validate spec.json                         # problems by path, exit 1 if any
spectacular html chart.json --data readings.csv -o readings.html
```

### From a checkout

The wheels are built by `.github/workflows/python-wheels.yml` for Linux, macOS and
Windows. To build one here, with Rust, `wasm-pack` and Node installed:

```bash
npm run build:wasm && npm install && npm run build   # the browser runtime, into python/src/spectacular/static
cd python && maturin build --release --out dist      # a wheel for this machine
pip install dist/*.whl
```

`maturin develop --release` installs it into the active environment for working on it.
The runtime is generated and not committed, so `pip install` straight from the Git
repository needs that build first.
