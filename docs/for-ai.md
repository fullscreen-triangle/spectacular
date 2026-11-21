# spectacular-charts for AI agents

This page is written for language models and agents that write chart descriptions, or
call `spectacular` on a user's behalf. It is complete on its own: everything needed to
produce a valid description first time is here. People will find the same material,
with pictures, in the [Guide](guide/describing.md).

Plain-text copy: <https://raw.githubusercontent.com/fullscreen-triangle/spectacular/rebuild/docs/for-ai.md>.
Index of every page: [`llms.txt`](llms.txt).

## 1. What it is, in one paragraph

`spectacular-charts` (import name `spectacular`) draws **linked, crossfiltered charts of
one table**. Brushing a range or clicking a category in any chart filters every other
chart. You write a **description**: JSON naming columns. A compiler, shipped as a native
command in the wheel, turns it into a **view spec**: it infers each chart's type from the
column types, matches names loosely, and validates against the table. Problems come back
as `{"path", "message"}` pairs whose paths point into what you wrote. Output is an
interactive notebook widget, an offline HTML file, or an SVG, PDF or PNG figure.

## 2. When to choose it

Choose it when the user wants to **explore** a table interactively, or needs **linked**
charts, where selecting in one filters the others. It also fits when the user wants the
brushed rows back in Python, has large tables (millions of rows filter interactively), or
wants charts without writing plotting code.

Prefer another tool when the user needs dates on an axis (or convert them to numbers
first), log scales, or error bars computed from data. The same goes for facets (small
multiples by a column), annotations, and chart types not in §5.

## 3. Install and import

```bash
pip install "spectacular-charts[notebook]"           # + interactive widgets (anywidget)
pip install "spectacular-charts[notebook,render]"    # + figures; then: python -m playwright install chromium
```

```python
import spectacular as sp     # never `import spectacular_charts`
```

## 4. The procedure that works

1. Know the table's column names and which are numbers and which are categories. If
   unsure, inspect the table's column types first.
2. Write the description (§5–§8).
3. Call `sp.validate(description, data=df)`. It returns `[]` or a list of problems.
4. Fix each problem at its `path` and validate again. Messages say what to do and often
   suggest the right name (`did you mean "od600"?`).
5. Show it with `sp.view(description, data=df)`, or write a file (§11).

Do not guess at keys outside the tables below; unknown keys are errors, not ignored.

## 5. Chart types and inference

A chart is an object naming columns. When `type` is absent, the **first** matching rule
decides:

| # | The chart names | Type |
|---|---|---|
| 1 | `low` or `high` | `interval` (needs both), banded by the category in `y` (or `x`) |
| 2 | `from` or `to` | `sankey` (needs both) |
| 3 | `x`, `y`, `z` | `scatter3d` (three numbers) |
| 4 | `x`, `y`, both numbers | `scatter`; `density` if rows > 200,000 and no `color` column |
| 5 | `x`, `y`, both categories | `matrix` |
| 6 | `x`, `y`, one number and one category | `strip` (number across, category down) |
| 7 | one column (`field`, `x` or `y`), a number | `histogram` |
| 8 | one column, a category | `bars`; with a category `color`, split bars |

A column is a **number** if it is numeric in the table, and a **category** otherwise
(strings, booleans), or if named in `categorical`, `domains` or `palettes`.

Set `type` to override. Accepted values (and aliases): `scatter`, `histogram`, `bars`,
`density` (`heatmap`), `scatter3d` (`3d`), `strip` (`jitter`, `beeswarm`), `matrix`
(`crosstab`), `interval` (`range`, `errorbars`, `forest`), `sankey` (`flow`), `arc`,
`network3d` (`network`, `graph`), `layers` (`dag`, `pipeline`, `steps`).

The column requirements of each type:

| Type | Requires | Brush |
|---|---|---|
| `scatter` | `x`, `y` numbers | rectangle |
| `density` | `x`, `y` numbers | none |
| `scatter3d` | `x`, `y`, `z` numbers | none |
| `histogram` | one number | range |
| `bars` | one category; optional `color` category | click |
| `strip` | one number, one category | range |
| `matrix` | two categories | click cell or label |
| `interval` | `low`, `high` numbers; band category in `y` | range (none with `scale: "each"`) |
| `sankey`, `arc`, `layers`, `network3d` | `from`, `to` categories (an edge table: one row per edge) | click edge or node |

Edge charts read rows as edges: rows with the same (`from`, `to`) pair form one edge whose
weight is the row count. All charts in one description read **the same table**.

## 6. Keys

Keys are matched after removing case, spaces, `-` and `_` (`"X axis"` = `x`). Two names
for one key in the same object is an error.

**Chart keys:**

| Key | Value |
|---|---|
| `x`, `y`, `z` | column names |
| `field` | column name (one-column charts; same as `x`) |
| `from`, `to` | column names (edge charts) |
| `low`, `high` | column names (interval) |
| `color` | a column name, **or** one colour, **or** a list of colours |
| `palette` | list of colours for the categories (do not combine with a list `color`) |
| `type` | §5 |
| `title` | string; default derived from the columns |
| `bins` | integer (histogram, default 60); `[nx, ny]` (density, default `[128, 128]`) |
| `size` | point diameter in px (scatter, strip, scatter3d) |
| `height` | plot height in px |
| `span` | grid columns spanned (integer) |
| `wide` | `true` spans the full row (boolean only) |
| `brush` | `false` stops the chart filtering others |
| `scale` | interval only: `"shared"` (default) or `"each"` (a scale per band) |
| `order` | arc only: `"count"` (default), `"name"`, `"data"` |
| `id` | string id used in selections; default slug of the title |

**Top-level keys:**

| Key | Value |
|---|---|
| `charts` | list of chart objects (aliases: `views`, `plots`, `panels`) |
| `title` | string |
| `theme` | `"auto"` (default), `"light"`, `"dark"`, `"paper"` (print) |
| `columns` | integer 1–12, charts per row; default min(3, number of charts) |
| `labels` | `{column: label}` for axes and legends |
| `categorical` | `[column, ...]` numeric columns to treat as categories |
| `domains` | `{column: [category, ...]}` category order; listed-but-empty categories stay visible |
| `palettes` | `{column: [colour, ...]}` or `{column: {category: colour}}`, at most 32 colours |
| `key` | column name whose values identify rows |
| `data` | file path or URL (CSV, TSV, JSON rows) when no `data=` is passed |
| `figure` | `{"format": "svg"\|"pdf"\|"png", "width": n, "unit": "mm"\|"in"\|"px", "dpi": n}`; `format` required |
| `description` | string, not drawn |

Three shapes are accepted: `{"charts": [...], ...top-level keys}`; a bare list `[chart,
...]`; or a single chart's keys at the top level mixed with top-level keys.

The [keys reference](reference/keys.md) lists every alias.

## 7. Colours

- A colour is `#rgb`, `#rrggbb`, or a CSS colour name (case-insensitive). Nothing else:
  no `rgb()`, no colour-map names.
- `color: "<column>"` colours by that column. The table's columns win over colour
  names: `"color": "orange"` colours by a column named `orange` if one exists.
- What takes what: scatter, strip, scatter3d and interval take a column (category or
  number), one colour or a list. Bars take a category column (split bars), one colour or
  a list. Histograms take one colour only. Density, matrix and edge charts take none.
  Breaking this is an error.
- Precedence for a category's colour: chart's list (≤10) > the column's `palettes` entry
  (≤32) > Tableau 10.
- A list given as `color`/`palette` on a chart coloured by a category column becomes that
  column's palette for every chart, unless `palettes` names one.
- Number columns as `color` use a continuous viridis scale.

## 8. Recipes

```python
import spectacular as sp
df = sp.datasets.plate()   # columns: well, row, column, medium, time_h, od600, rate
```

Distribution of one number, split by a category, linked:

```python
sp.view({"charts": [{"x": "rate", "bins": 40}, {"x": "medium"}, {"x": "rate", "y": "medium"}]}, data=df)
```

Relationship of two numbers, coloured by a category, with a fixed colour per category:

```python
sp.view({"palettes": {"medium": {"rich": "#4e79a7", "minimal": "#f28e2b"}},
         "x": "time_h", "y": "od600", "color": "medium"}, data=df)
```

Two categories against each other, and counts split by a second category:

```python
bundle = sp.datasets.bundle()
sp.view({"charts": [{"x": "label", "y": "blocker"}, {"x": "way", "color": "outcome"}]}, data=bundle)
```

Ranges per item with different units per band:

```python
sp.view({"low": "low", "high": "high", "y": "slot", "scale": "each"}, data=bundle)
```

A pipeline of steps from an edge table, in a fixed step order:

```python
lineage = sp.datasets.lineage()   # row, slot, from_step, step
sp.view({"domains": {"step": ["parse", "resolve", "measure", "decide", "publish"]},
         "from": "from_step", "to": "step", "type": "layers"}, data=lineage)
```

Rows identified by id, so other code can select them:

```python
w = sp.view({"key": "row", "charts": [{"x": "outcome"}, {"x": "outcome", "y": "margin"}]}, data=bundle)
```

A print figure (needs the `render` extra):

```python
sp.render({"theme": "paper", "x": "time_h", "y": "od600", "color": "medium"},
          "growth.pdf", data=df, width=85, unit="mm")
```

## 9. Reading selections back (notebook widget)

<!-- no-test -->
```python
w = sp.view(description, data=df)    # display it; the user brushes
```

In a later cell (the page updates these asynchronously):

| Expression | Gives |
|---|---|
| `w.selection` | `{"selected": n, "rows": n, "filters": [{"view": id, "field": col, "range": [lo, hi]} or {"view", "field", "values": [...]}]}` |
| `w.selected_rows` | numpy array of 0-based row positions passing every brush |
| `w.selected_data()` | those rows, in the type of `data` |
| `w.selected_keys` | key values of selected rows (needs `key`) |
| `w.select(["r001", "r007"])` | selects rows by key in every chart; `w.select(None)` clears |
| `w.on_select(fn)` | `fn(selection)` after each brush |

Do not read these in the same cell that displays the view; they are empty until the
page reports.

## 10. Errors you will see, and the fix

| Message (at `path`) | Fix |
|---|---|
| `no column "X"; did you mean "Y"?` | use `Y` |
| `no column "X"; the data has a, b, c, ...` | pick a listed column |
| `unknown key "K"; did you mean "C"?` | use `C` |
| `"a" and "b" say the same thing; keep one` | remove one |
| `say what to draw: ...` | add `x` (or `from`/`to`, `low`/`high`) |
| `a histogram has one colour; ...` / `colouring by a column ... give a histogram one colour` | one colour string, or use bars/strip |
| `bars split by a column need a category; "c" is a number` | add `c` to `categorical` |
| `a T takes no colour setting` | remove `color`/`palette` from density, matrix or edge charts |
| `unknown chart type ...; one of ...` | a type from §5 |
| `/figure/format: one of svg, pdf, png` | add `format` to `figure` |
| `TypeError: dates and durations have no axis type yet` | convert to numbers, e.g. `(t - t.min()).dt.total_seconds() / 3600` |

`sp.view` shows problems in the cell rather than raising. `save_html`/`to_html` raise
`DescriptionError`, and `render` raises `SpectacularError`; both have `.problems`.

## 11. Outputs

| Call | Result |
|---|---|
| `sp.view(desc, data=df)` | interactive widget (as the cell's last expression) |
| `sp.show(desc, data=df)` | displays now, returns the widget |
| `sp.save_html(desc, "out.html", data=df)` | one offline interactive HTML file (~1 MB + data) |
| `sp.to_html(desc, data=df)` | the same as a string |
| `sp.render(desc, "out.svg" \| ".pdf" \| ".png", data=df, width=, unit=, dpi=)` | figure, unbrushed |
| `w.save_svg("now.svg")` | SVG of a displayed widget, brushes included (written after the cell ends) |
| `%%spectacular df` cell with a JSON body | a widget; `--as w` stores it |

Without Python: `spectacular compile desc.json --data table.csv [--json]` prints the
spec; `spectacular validate spec.json --json`; `spectacular html desc.json --data
table.csv -o out.html`. Exit codes: 0 ok, 1 problems, 2 usage or I/O.

## 12. Rules

- Use only the keys and values in this page. Do not invent options (`log`, `facet`,
  `tooltip`, `stack`, `opacity`, `xlim`, `sort` on bars): they are errors.
- Prefer descriptions to full specs unless the user asks for a spec. A full spec (every
  view with `id` and an exact `type`) is accepted everywhere; see
  [View specs](reference/spec.md).
- Name columns exactly as in the table when you know them; the loose matching is for
  people.
- Use `domains` whenever categories have a natural order (stages, grades, outcomes); use
  `palettes` whenever a category has a conventional colour, so every chart agrees.
- Keep one table per description. For an edge chart plus row charts, the edge table must
  carry the columns the row charts need, or use two descriptions.
- Validate before showing or rendering, and report remaining problems verbatim.
