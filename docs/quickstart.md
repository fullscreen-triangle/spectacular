# Quickstart

This page builds the linked charts of a growth plate step by step. Run it in JupyterLab
or Jupyter Notebook, after `pip install "spectacular-charts[notebook]"`.

## 1. A table

Any table will do. The package carries a few for trying it out; `plate` is 96 wells of
bacteria in two media, read every 30 minutes for 12 hours:

```python
import spectacular as sp

df = sp.datasets.plate()
df.head()
```

| well | row | column | medium | time_h | od600 | rate |
|---|---|---|---|---|---|---|
| A1 | A | 1 | rich | 0.0 | 0.0589 | 0.5346 |
| A1 | A | 1 | rich | 0.5 | 0.0676 | 0.5346 |
| … | | | | | | |

## 2. One chart

Name a column. One number makes a histogram:

```python
sp.view({"x": "rate"}, data=df)
```

The type follows from the columns you name:

| You name | You get |
|---|---|
| one number | a histogram |
| one category | bars |
| two numbers | a scatter |
| a number and a category | a strip: a row of points per category |
| two categories | a matrix: a cell per pair |

## 3. Linked charts

Put several charts in a list under `"charts"`. They share the table, so brushing one
filters the rest:

```python
w = sp.view({
    "title": "Growth plate",
    "labels": {"od600": "OD600", "time_h": "time (h)"},
    "charts": [
        {"x-axis": "time_h", "y-axis": "od600", "color": "medium", "wide": True},
        {"x": "rate", "color": "teal", "title": "Growth rates"},
        {"x": "medium"},
    ],
}, data=df)
w
```

- Drag across the histogram of growth rates. The scatter keeps only the wells growing at
  those rates, and the bars count them per medium.
- Click a bar to keep one medium, and click it again to let it go.
- Click a chart without dragging to clear its brush.

## 4. Read the selection back

The view object knows what you selected:

```python
w.selection         # {"selected": 600, "rows": 2400, "filters": [{"view": "growth-rates", "field": "rate", "range": [0.31, 0.42]}]}
w.selected_rows     # the row positions in df that pass every brush
w.selected_data()   # those rows of df, as a DataFrame
```

To run code after each brush, register a callback:

```python
w.on_select(lambda s: print(s["selected"], "rows selected"))
```

## 5. Keep it

```python
sp.save_html({"x": "time_h", "y": "od600", "color": "medium"}, "plate.html", data=df)
```

The page is one file holding the charts, the runtime and the data. It works offline and
needs no server.

For a paper, draw a figure at print size (this needs the `render` extra):

```python
sp.render({"x": "time_h", "y": "od600", "color": "medium",
           "figure": {"format": "pdf", "width": 85, "unit": "mm"}}, "plate.pdf", data=df)
```

## 6. Without Python code: the cell magic

`import spectacular` adds a `%%spectacular` cell magic to the notebook. The cell holds
only the description, and the line names the table:

```python
%%spectacular df
{"title": "Growth plate",
 "charts": [
   {"x-axis": "time_h", "y-axis": "od600", "color": "medium"},
   {"x": "rate", "color": "teal"},
   {"x": "medium"}
 ]}
```

`%%spectacular df --as w` also stores the charts in `w`, so you can read the selection.

## Next

- [Describing charts](guide/describing.md): everything a description can say.
- [Chart types](guide/chart-types.md): a picture and a description for each type.
- [Brushing and selections](guide/selections.md): reading, reacting to and setting selections.
