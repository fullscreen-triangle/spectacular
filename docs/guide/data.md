# Data

## What `data=` takes

| Type | Example |
|---|---|
| pandas DataFrame | `data=df` |
| polars DataFrame | `data=pl.read_csv("x.csv")` |
| dict of columns | `data={"t": [0, 1, 2], "v": [0.1, 0.4, 0.9]}` |
| list of row dicts | `data=[{"t": 0, "v": 0.1}, {"t": 1, "v": 0.4}]` |
| numpy record array | `data=np.rec.fromarrays([t, v], names="t,v")` |

All columns of a table must have the same length. Only the columns the charts use (and
the `key`) are sent to the page.

Without `data=`, the description can name a file instead, relative to the description
file or the working directory:

<!-- no-test -->
```python
sp.save_html({"data": "readings.csv", "x": "od600"}, "readings.html")
```

The file is CSV, TSV or JSON (an array of row objects), read by its extension. Its bytes
are sent along with the page, so the page needs no server. A URL starting with
`https://` is fetched by the page instead.

## Numbers and categories

Each column is sent as one of two types, and the type decides the chart:

| Column | Type | Sent as |
|---|---|---|
| integers, floats | number (quantitative) | 64-bit floats |
| strings, booleans, mixed | category (nominal) | 32-bit codes plus the list of labels |
| numbers named in `categorical`, `domains` or `palettes` | category | codes and labels, the numbers written as text |

Booleans become the categories `"true"` and `"false"`.

**Missing values.** `None`, `NaN` and pandas' `NA` are missing. A missing number travels
as NaN; a missing category becomes the empty category `""`. To show missing categories
under a name, fill them first: `df.fillna({"blocker": "none"})`.

**Dates** have no axis type yet. Convert them to numbers first, for example hours since
the start:

```python
import pandas as pd

runs = pd.DataFrame({"at": pd.to_datetime(["2026-11-01 08:00", "2026-11-01 09:30"]), "od600": [0.05, 0.12]})
runs["hours"] = (runs["at"] - runs["at"].min()).dt.total_seconds() / 3600
```

A date column passed as it is raises `TypeError` with that advice.

## Large tables

The crossfilter keeps every column sorted and updates only the rows whose state a brush
changes. Measured on 10 million rows with four dimensions, dragging a brush takes about
7 ms per step in WebAssembly (the project's `README.md` has the full benchmark). In a
notebook the limits are rather:

- **Sending the table.** 8 bytes per number and 4 per category, per row, once.
- **Drawing points.** Above 200,000 rows, two numbers become a density, which draws in the
  same time at any size. Ask for `"type": "scatter"` to keep points; they are drawn with
  WebGL at 2 pixels.
- **Reading back rows.** `selected_rows` sends 4 bytes per selected row after each brush.
  `sp.view(..., track_rows=False)` turns that off.

## Sample tables

`sp.datasets` holds the tables these pages use. Each returns a DataFrame, or a list of
row dicts with `frame=False` or without pandas:

| Function | Rows | Columns |
|---|---|---|
| `plate()` | 2,400 | `well`, `row`, `column`, `medium`, `time_h`, `od600`, `rate`: a 96-well growth plate |
| `lorenz()` | 29,900 | `t`, `x`, `y`, `z`, `speed`, `lobe`: the Lorenz system, computed on first use |
| `bundle()` | 260 | `row`, `record`, `slot`, `step`, `way`, `outcome`, `label`, `blocker`, `low`, `high`, `margin`, `sources`, …: a provenance bundle |
| `lineage()` | 695 | `row`, `slot`, `from_step`, `step`: rows passing between the steps of a pipeline |

The numbers are generated from fixed seeds. Their shapes are realistic, but they are
not measurements.
