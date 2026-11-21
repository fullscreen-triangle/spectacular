# Brushing and selections

Every chart in a view reads the same table. Brushing a chart filters the rows, and
every chart redraws from the rows that pass **all** the brushes.

| Chart | How to brush | Selects |
|---|---|---|
| scatter | drag a rectangle | rows inside it |
| histogram, strip | drag across | rows in the range |
| interval | drag across (not with `"scale": "each"`) | rows whose interval overlaps the range |
| bars | click bars | rows of the chosen categories |
| matrix | click cells, or row and column labels | rows of the chosen pairs |
| sankey, layers, arc | click links or nodes | rows of the chosen links |
| network3d | click nodes | rows of the links touching them |
| density, scatter3d | — | they show the others' brushes but do not brush |

Click a chart without dragging to clear its brush. `"brush": false` on a chart stops it
filtering the others.

## Reading the selection

`sp.view` returns a `View`. Its selection follows the page as you brush:

```python
import spectacular as sp

df = sp.datasets.plate()
w = sp.view({"charts": [{"x": "time_h", "y": "od600"}, {"x": "rate", "id": "rate"}, {"x": "medium"}]}, data=df)
w
```

After dragging across the histogram and clicking the `rich` bar:

<!-- no-test -->
```python
w.selection
# {"selected": 380, "rows": 2400,
#  "filters": [{"view": "rate", "field": "rate", "range": [0.52, 0.61]},
#              {"view": "medium", "field": "medium", "values": ["rich"]}]}

w.selected_rows     # array([  0,   1,   2, ...]): positions in df, 0-based
w.selected_data()   # those rows of df, as a DataFrame
```

- `selection` holds the brushes **in data terms**: a range of a number, or a list of
  categories. Each filter names its chart's `id` (made from the title unless you give one)
  and its column.
- `selected_rows` are the positions of the passing rows in the data you gave, as a
  numpy array (or a list without numpy). They are empty until the view has been drawn.
- `selected_data()` returns those rows in the type you passed: a DataFrame, a dict of
  columns, or a list of rows. When the description reads a file (`"data": "x.csv"`)
  instead of a table from Python, use `selected_rows`.

To stop the page sending row positions (for very large tables, where you only need the
filters), pass `track_rows=False`.

## Reacting to a brush

`on_select` calls a function after each brush settles:

```python
def report(selection):
    print(f"{selection['selected']} of {selection['rows']} rows")

handler = w.on_select(report)
```

`w.unobserve(handler, names="selection")` removes it. The callback runs in the kernel, so
it can update another chart, refit a model on `w.selected_data()`, or write a file.

## Naming rows: `key`

A `key` is a column whose values identify rows, such as a sample id or a record's
Internationalized Resource Identifier (IRI).
With one, the view also reports the **key values** of the selected rows, and can select
rows by key from Python:

```python
bundle = sp.datasets.bundle()
w = sp.view({"key": "row", "charts": [{"x": "outcome"}, {"x": "outcome", "y": "margin"}]}, data=bundle)
w
```

<!-- no-test -->
```python
w.selected_keys            # ["r002", "r007", ...]: the key of each selected row
w.select(["r001", "r007"]) # show these rows, in every chart
w.select(None)             # drop that selection again
```

`select(keys)` adds one more filter on top of the brushes, so the charts show the rows
both chose. It is how another tool (a table, a search, a model) points at rows in the
charts. The page answers after the cell finishes running, and `selection` then includes
it. Keys are compared as text.

`select` without a `key` in the description raises `ValueError`.

## Replacing the charts

`update` draws a new description, new data, or both, in the same output. Brushes are
cleared:

```python
w = sp.view({"x": "rate"}, data=df)
w.update({"x": "od600"})                                   # another chart, same data
w.update(data=df[df.medium == "rich"])                     # same charts, fewer rows
```

## Saving what you see

`w.save_svg("brushed.svg")` writes the charts as they are now, brushes included, as one
SVG document. See [Figures and pages](figures.md).

## Outside a notebook

The `spectacular serve` command of the Node CLI shows a spec in a browser and prints
each selection as a JSON line on standard output, for programs that are not Python. See
[Command line](command-line.md#the-node-command).
