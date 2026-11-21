# Describing charts

A **description** says which columns to draw, and optionally how. It is a Python dict, a
JSON string, or the path of a JSON file. The rest is worked out from your table.

```python
import spectacular as sp

df = sp.datasets.plate()
description = {
    "title": "Growth plate",
    "labels": {"od600": "OD600"},
    "charts": [
        {"x": "time_h", "y": "od600", "color": "medium"},
        {"x": "rate", "color": "teal", "bins": 40},
        {"x": "medium"},
    ],
}
sp.view(description, data=df)
```

## Three shapes

All three of these are descriptions:

=== "One chart"

    ```python
    sp.view({"x": "rate", "title": "Growth rates"}, data=df)
    ```

    The chart's keys sit at the top, beside settings such as `theme`.

=== "A list of charts"

    ```python
    sp.view([{"x": "rate"}, {"x": "medium"}], data=df)
    ```

=== "An object with charts"

    ```python
    sp.view({"title": "Plate", "columns": 2, "charts": [{"x": "rate"}, {"x": "medium"}]}, data=df)
    ```

    The list can also be called `views`, `plots`, `graphs`, `figures` or `panels`.

## Forgiving keys

Keys are compared without case, spaces, dashes or underscores, so `"x-axis"`, `"X axis"`,
`"x_axis"` and `"xAxis"` are one key. Many keys have other names too: `colour` and `hue`
for `color`, `source` for `from`, `min` for `low`. The
[Description keys](../reference/keys.md) page lists them all.

Writing the same key twice is a problem (`"x" and "x-axis" say the same thing; keep one`),
and so is an unknown key (`unknown key "colr"; did you mean "color"?`).

## Forgiving column names

A column name is first matched exactly, then without case, spaces, dashes or
underscores: `"time h"` finds `time_h`, and `"OD600"` finds `od600`. A name that matches
nothing comes back with the nearest column:

```python
sp.validate({"x": "od60"}, data=df)
# [{"path": "/x", "message": "no column \"od60\"; did you mean \"od600\"?"}]
```

If a name matches two columns loosely (`Time` and `time`), write it exactly.

## Columns decide the chart

Each column is a **number** or a **category**. Numbers are numeric columns of the table;
everything else (text, booleans) is a category. Name numeric columns in `categorical` to
treat them as categories, for example well numbers or plate columns:

```python
sp.view({"categorical": ["column"], "x": "column"}, data=df)   # bars, not a histogram
```

Without a `type`, the chart is chosen from what it names:

| The chart names | Chart |
|---|---|
| `low` and `high` | interval, banded by `y` (or `x`) |
| `from` and `to` | sankey |
| `x`, `y` and `z` | 3D scatter |
| `x` and `y`, both numbers | scatter (a density beyond 200,000 rows, unless coloured by a column) |
| `x` and `y`, both categories | matrix |
| `x` and `y`, a number and a category | strip (the number across, whichever way round) |
| one column, a number | histogram |
| one column, a category | bars; with a category `color`, each bar split into it |

`"type"` asks for a type instead, for example `{"from": "a", "to": "b", "type": "arc"}`.
The types and their other names are on the [Chart types](chart-types.md) page.

## What a chart can say

| Key | Means |
|---|---|
| `x`, `y`, `z` | the columns on each axis |
| `from`, `to` | the two ends of an edge, for arc, sankey, layers and network charts |
| `low`, `high` | the two ends of a quantity, for an interval |
| `color` | a column to colour by, one colour (`"teal"`, `"#4e79a7"`), or a list of colours |
| `palette` | colours for the categories, in order |
| `type` | the chart type, instead of the inferred one |
| `title` | the chart's title; by default it is made from the columns |
| `bins` | a histogram's bins (default 60); a density's grid (default `[128, 128]`) |
| `size` | the point size of a scatter, strip or 3D scatter, in pixels |
| `height` | the plot's height, in pixels |
| `span`, `wide` | how many grid columns the chart spans; `"wide": true` spans the whole row |
| `brush` | `false` to stop the chart filtering the others |
| `scale` | on an interval: `"each"` gives each band its own scale (quantities in different units) |
| `order` | on an arc chart: nodes ordered by `"count"` (default), `"name"` or `"data"` |
| `id` | the chart's id in selections; by default made from its title |

## What the whole description can say

| Key | Means |
|---|---|
| `charts` | the list of charts |
| `title` | the heading above the charts |
| `theme` | `"light"`, `"dark"`, `"paper"` (print) or `"auto"` (the notebook's, the default) |
| `columns` | charts per row (default: as many charts as there are, up to 3) |
| `labels` | axis labels by column: `{"od600": "OD600"}` |
| `categorical` | numeric columns to treat as categories |
| `domains` | a column's categories in order: `{"medium": ["rich", "minimal"]}` |
| `palettes` | a column's colours, everywhere it is drawn: a list, or by name |
| `key` | the column whose values name rows, for selecting by key |
| `data` | a CSV, TSV or JSON file, when no table is given from Python |
| `figure` | print size for `render`: `{"format": "pdf", "width": 180, "unit": "mm"}` |
| `description` | a note kept with the spec; not drawn |

`domains` and `palettes` are covered in [Colours, orders and themes](colours.md), `key` in
[Brushing and selections](selections.md), and `figure` in [Figures and pages](figures.md).

## Problems

A description that cannot be drawn is not drawn. Each problem has a `path` into what
you wrote, and a `message`:

```python
problems = sp.validate({"charts": [{"x": "time_h", "y": "od60"}, {"x": "rate", "colr": "teal"}]}, data=df)
for p in problems:
    print(p["path"], p["message"])
# /charts/0/y no column "od60"; did you mean "od600"?
# /charts/1/colr unknown key "colr"; did you mean "color"?
```

In a notebook, `sp.view` shows the problems in the cell where the charts would have been.
`sp.render` and `sp.save_html` raise an error that lists them (`DescriptionError` or
`SpectacularError`, both with a `.problems` list).

## A description from a file

A path to a `.json` file works wherever a dict does. Relative `data` paths in it resolve
against the file's folder:

<!-- no-test -->
```json
{
  "title": "Growth plate",
  "data": "readings.csv",
  "charts": [{"x": "time_h", "y": "od600", "color": "medium"}, {"x": "rate"}]
}
```

<!-- no-test -->
```python
sp.save_html("plate.chart.json", "plate.html")
```
