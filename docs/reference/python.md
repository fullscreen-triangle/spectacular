# Python API

```python
import spectacular as sp
```

Everywhere below, **`spec`** is a description or a view spec, given as any of:

- a dict, or a list of charts;
- a JSON string;
- the path of a JSON file (a `str` or `Path`).

Relative data paths in a file resolve against the file's folder. **`data`** is a pandas
or polars DataFrame, a dict of columns, a list of row dicts, or a numpy record array (see
[Data](../guide/data.md)); when given, it replaces any data the spec names.

## Showing charts

### `view(spec, data=None, **kwargs)`

Linked charts to display in a notebook, as a cell's last expression or with `display()`.
Returns a [`View`](#view) when anywidget is installed (the `notebook` extra), else an
[`HTMLView`](#htmlview). Keyword arguments go to the class: `track_rows` for `View`,
`height` for `HTMLView`.

A description with problems does not raise: the view shows the problems in the cell.

`chart` is the same function.

### `show(spec, data=None, **kwargs)`

Displays the charts at once and returns the view, for use inside loops and functions.

### `View`

An [anywidget](https://anywidget.dev) widget holding linked charts of one table.

<!-- no-test -->
```python
View(spec, data=None, *, track_rows=True)
```

| Member | Type | |
|---|---|---|
| `selection` | `dict` | `{"selected": int, "rows": int, "filters": [...]}`; each filter is `{"view", "field", "range": [lo, hi]}` or `{"view", "field", "values": [...]}`. Updated after each brush settles. |
| `selected_rows` | numpy array (or list) | 0-based positions of the rows passing every brush. Empty until drawn, and always with `track_rows=False`. |
| `selected_data()` | same type as `data` | the selected rows of the table given. Raises `ValueError` when the data came from a file. |
| `selected_keys` | `list[str]` | the `key` values of the selected rows; `[]` without a key. |
| `select(keys)` | | selects rows by `key` value, as one more filter; `None` clears it. Raises `ValueError` without a key. |
| `on_select(callback)` | handler | calls `callback(selection)` after each brush; remove with `w.unobserve(handler, names="selection")`. |
| `update(spec=None, data=None)` | | redraws with a new spec, new data or both; clears brushes. |
| `save_svg(path)` | | writes the charts as shown, brushes included, once the page answers. The view must be displayed. |
| `saved` | `list[Path]` | files `save_svg` has written. |
| `problems` | `list[dict]` | the problems that stopped the charts being drawn, each `{"path", "message"}`. |
| `spec` | `dict` | the compiled view spec. |

`select` and `save_svg` send a message to the page, which answers after the current cell
finishes. Read `selection`, `selected_keys` or `saved` in a later cell.

### `HTMLView`

<!-- no-test -->
```python
HTMLView(spec, data=None, height=480)
```

The charts as a self-contained page in an iframe. Brushing works, nothing is reported
back, and the output is saved in the notebook file. `problems` and `document` (the HTML)
are attributes.

## Files

### `render(spec, out, data=None, *, width=None, unit=None, dpi=None) -> Path`

Draws the charts, unbrushed, to `out`: `.svg`, `.pdf` or `.png` by extension. `width` in
`unit` (`"mm"`, `"in"` or `"px"`); `dpi` for PNG (default 300 for mm and in, 192 for px).
Unset values come from the spec's `figure`, then 960 px. Uses Playwright for Python when
installed (`render` extra), else the Node command in a checkout.

Raises `SpectacularError` with `exit` 1 for a spec with problems (listed in
`problems`), and 2 for a bad file name or unit, or when neither Playwright nor a checkout
with Node is available. Through the Node command, 3 means the browser failed; through
Playwright, its own errors pass through.

### `save_html(spec, path, data=None, *, title=None) -> Path`

Writes an offline HTML page with the charts, the runtime and the data. Returns `path`.

### `to_html(spec, data=None, *, title=None, figure=False) -> str`

The same page as a string. `figure=True` drops the margin and status line.

## Checking

### `validate(spec, data=None) -> list[dict]`

The problems of a description or spec, each `{"path": str, "message": str}`; `[]` when it
holds. With `data`, or a local file named in the spec, columns are checked too. Paths
point into what was given, such as `/charts/1/x-axis`.

### `schema() -> dict`

The JSON Schema of view specs.

### `load_spec(spec) -> (value, base)`

The parsed description or spec, and the folder its relative paths resolve against.

### `fields_used(spec) -> list[str]`

The fields a view spec reads: its `key`, the fields its views name, then those it
declares.

### `SPEC_VERSION`

`"spectacular/v0"`.

## Errors

### `DescriptionError(ValueError)`

A description the compiler refused. `problems` lists them. Raised by `save_html` and
`to_html`.

### `SpectacularError(Exception)`

Raised by `render`. `problems` lists the spec's problems and `exit` holds the command's
exit code.

## Notebook magic

### `load_ipython_extension(shell)`

Registers `%%spectacular`. `import spectacular` calls it inside IPython, and
`%load_ext spectacular` calls it explicitly. See
[Notebooks](../guide/notebooks.md#the-spectacular-cell-magic).

## Sample data: `spectacular.datasets`

| Function | Returns |
|---|---|
| `plate(frame=True)` | 2,400 rows of a 96-well growth plate |
| `lorenz(frame=True)` | 29,900 states of the Lorenz system |
| `bundle(frame=True)` | 260 rows of a provenance bundle |
| `lineage(frame=True)` | 695 links between pipeline steps |

Each returns a pandas DataFrame, or a list of row dicts with `frame=False` or without
pandas. [Data](../guide/data.md#sample-tables) lists their columns.

## The `spectacular` command

The wheel installs it next to the Python package; see [Command line](../guide/command-line.md).
