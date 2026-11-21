# Spectacular: specification for programs and agents

This file is for whatever, in **another** repository, needs a scientific visualisation:
an ordinary program (a lab-automation runtime, a pipeline step, a notebook, a CI job)
or an AI agent writing specs for one. It says what spectacular does, how to hand it a
task, the complete spec format, and what it does not do yet. Everything here describes
the code on branch `rebuild` of
[fullscreen-triangle/spectacular](https://github.com/fullscreen-triangle/spectacular).
If this file and the code disagree, the code wins: `packages/viz/src/spec.ts` is the
contract, and `spectacular.schema.json` is generated from it.

**Programs** need §2 and §8: one command-line tool, JSON in and out, fixed exit codes.
They never link against spectacular's code. **Python programs and Jupyter notebooks**
import the `spectacular` package instead (§8.6). **People** can skip the full spec: a
*chart description* (§4.7) names columns, and the chart types, ids and layout follow.

## 1. What it is

Spectacular renders **linked views** of one table. Brushing a range in any brushable
view filters every other view immediately. It is built for tables of thousands to about
ten million rows:

- a Rust crossfilter (`crates/chigutiro`), compiled to WebAssembly (WASM), runs in a
  Web Worker;
- WebGL2 draws the 2D point views and three.js draws the 3D view;
- SVG draws the histograms and bars, with D3 for axes and brushes.

You never write rendering code. You write a **view spec**: a JSON file naming the data,
the fields and the views. The renderer is deterministic, and it rejects an invalid spec
with problems that each name the exact JSON path at fault.

Use it when someone needs to **explore** a table: find which rows explain a peak,
see how a cut on one variable changes the others, look at a 3D structure, or compare
categories. It also draws the same views to an SVG, PDF or PNG file for a paper or a report
(§8.3). Do not use it for data that is not one flat table.

## 2. The ways in

| You have | Do this |
|---|---|
| A program in any language that writes a spec and a data file | The **`spectacular` command**, §8: `validate`, `render`, `serve`. |
| Python: a Jupyter notebook, a script, a pipeline step | **`pip install spectacular-charts`**, §8.6: views in the notebook, selections back in Python. |
| A chart description, and no Node | The native **`spectacular`** command in that wheel (or `cargo build --release -p spectacular`): `compile`, `validate`, `html`, `schema`, §4.7. |
| A spec plus a data file, and a person who wants to look | `spectacular serve <spec>`, §8.4, or the dev server, §2.1. |
| A TypeScript or JavaScript web app | **Embed** with `mount()`, §7. |
| Only a spec to check | `spectacular validate <spec>`, §2.2. |

Setup, once per machine, from the spectacular checkout. It needs Rust with the
`wasm32-unknown-unknown` target, `wasm-pack`, and Node 22.18 or later (the command runs
TypeScript sources directly):

```bash
npm run build:wasm     # builds packages/chigutiro-wasm; must come before npm install
npm install
npm run build          # the viewer that render and serve use (packages/viz/dist)
npx playwright install chromium   # only for render
```

Then call `node <checkout>/cli/spectacular.mjs <command>`. Inside the checkout,
`npm run spectacular -- <command>` does the same, and `npm link` puts `spectacular` on
the `PATH`.

### 2.1 Viewer mode with the dev server

For working on spectacular itself, or for a quick look without the command:

1. Put the spec and its data file side by side under `examples/local/<your-project>/`.
   That directory is gitignored, so private data never gets committed. In the dev
   server, files under `examples/` are served at the site root; the production build
   leaves them out.
2. In the spec, write `"data": { "url": "<data file name>" }`. A relative URL resolves
   against the spec's own URL.
3. Start the dev server: `npm run dev` (Vite, default `http://localhost:5173`).
4. Open `http://localhost:5173/?spec=/local/<your-project>/<name>.spec.json`.

`?spec=` also accepts an absolute URL, for example a spec served by your own repo's
server. The server must then send Cross-Origin Resource Sharing (CORS) headers for
both the spec and the data. `spectacular serve` (§8.4) needs none of this: it serves a
spec from wherever it is.

**Signals for automation.** `document.body.dataset.state` starts as `"loading"` and
becomes `"ready"` or `"error"`. On `"error"`, the element `#error` holds the message,
including every spec problem as `/path: message`. On `"ready"`, `window.spectacular` is
the handle described in §7.

### 2.2 Validate before you open anything

```bash
spectacular validate path/to/spec.json [more.spec.json ...]
# or, in the checkout: npm run validate -- path/to/spec.json
```

It checks the structure. When `data.url` names a local file beside the spec, or the
data is inline, it also loads the data and checks that every field exists and has the
type each view needs. It reports every problem in one run:

```
x.spec.json /views/0/field: histogram.field needs a quantitative field; "lobe" is nominal
x.spec.json /views/1/colour: unknown property "colour"; did you mean "color"?
x.spec.json /views/2/field: no field "nope" in the data
x.spec.json /views/3/type: unknown view type "pie"; one of scatter, histogram, bars, density, scatter3d, arc, sankey, network3d
```

Exit codes: `0` all valid, `1` problems found, `2` usage or I/O error. When a spec's data
is a remote URL or `"external"`, only the structure is checked, and the output says
`valid (structure only)`. `--json` prints one object per spec instead (§8.2).

**Run it after every edit to a spec.** Unknown properties are errors, not ignored, so
a typo cannot quietly draw a different figure.

## 3. A complete example

`examples/lorenz.spec.json`, which reads `examples/lorenz.csv` (29,900 states of the
Lorenz system):

```json
{
  "version": "spectacular/v0",
  "title": "Lorenz attractor",
  "description": "Brush time or speed to see where on the attractor those states lie.",
  "data": { "url": "lorenz.csv" },
  "fields": {
    "t": { "type": "quantitative", "title": "time" },
    "speed": { "type": "quantitative", "title": "|dx/dt|" },
    "lobe": { "type": "nominal" }
  },
  "layout": { "columns": 2 },
  "views": [
    { "id": "attractor", "type": "scatter3d", "title": "Phase space", "x": "x", "y": "y", "z": "z", "color": "speed", "height": 380 },
    { "id": "xz", "type": "scatter", "title": "x–z projection", "x": "x", "y": "z", "color": "lobe", "height": 380 },
    { "id": "time", "type": "histogram", "title": "Time", "field": "t", "bins": 150, "span": 2 },
    { "id": "speed", "type": "histogram", "title": "Speed", "field": "speed", "bins": 80 },
    { "id": "lobe", "type": "bars", "title": "Lobe", "field": "lobe" }
  ]
}
```

`examples/synthetic.spec.json` shows every view type, using `"columns": "external"`
data.

## 4. Spec reference

<!-- --8<-- [start:spec] -->
### 4.1 Top level

| Property | Type | Required | Meaning |
|---|---|---|---|
| `version` | `"spectacular/v0"` | no | If present, it must be exactly this. Include it. |
| `$schema` | string | no | Ignored by the renderer. Point it at `spectacular.schema.json` for editor checking. |
| `title` | string | no | Heading above the views. |
| `description` | string | no | Not displayed; use it for provenance. |
| `theme` | `"auto"` \| `"light"` \| `"dark"` \| `"paper"` | no | Default `"auto"`: follow the page's colour scheme (in a notebook, the notebook's theme). `"paper"` is for print: white, no panel borders, darker axes. |
| `data` | object | **yes** | §4.2. |
| `fields` | object | no | Field types, axis titles, category orders and colours, §4.3. |
| `key` | string | no | A field whose values identify rows. Selections then name rows by it, and rows can be selected from outside by it (§7, §8.6). It loads as a category and uses one crossfilter dimension. |
| `views` | array | **yes** | At least one view, §4.5. |
| `layout` | object | no | `{ "columns": n }`, n an integer from 1 to 12. The default is the smaller of 3 and the number of views. |
| `output` | object | no | `{ "interactive": bool, "figure": { "format": "svg" \| "pdf" \| "png", "width": n, "unit"?: "mm" \| "in" \| "px", "dpi"?: n } }`. `figure` sets the default size for `spectacular render` (§8.3). `interactive` is accepted and not yet used. |

Any other property is an error.

### 4.2 `data`: give exactly one of these

| Form | Use |
|---|---|
| `{ "url": "file.csv", "format"?: ..., "pointer"?: ... }` | A file. Relative URLs resolve against the spec. |
| `{ "values": [ { "a": 1, "k": "x" }, ... ] }` | Inline rows, for small tables. |
| `{ "columns": "external" }` | Columns passed by the program calling `mount()` (§7). The fast path for large data. |

With `url`:
- **`format`** is `"csv"`, `"tsv"` or `"json"`. The default comes from the extension:
  `.json` and `.jsonld` read as JSON, `.tsv` as TSV, anything else as CSV. Set it when
  the URL has no telling extension, such as an API endpoint.
- **`pointer`** is a JSON Pointer, as defined in Request for Comments (RFC) 6901, to the
  array of row objects inside a JSON document. Without it, the document itself must be
  that array. A run log or JSON for Linked Data (JSON-LD) record usually nests its rows: `{ "url": "run.jsonld", "pointer": "/executions" }`
  reads the objects in the document's `executions` array as rows. Escape `/` in a key as
  `~1` and `~` as `~0`.
- Each row's fields must be plain values. A nested object or array in a field is not
  flattened; give the program that writes the file a flat column for anything to plot.

Only the fields that views use, or that `fields` declares, are loaded. Other columns
are ignored.

**Size guidance.** Parsing CSV in the browser is the slowest step. Keep `url` and
`values` data to about a million rows. For more, pass typed arrays as external columns.

### 4.3 `fields`

`"fields": { "<name>": { "type": "quantitative" | "nominal", "title": "axis label", "domain": [...], "palette": [...] } }`

- **Type inference** when `type` is omitted: a field is `quantitative` if every
  non-empty value parses as a finite number, and `nominal` otherwise. Declare the type
  whenever you know it. A numeric code such as a detector id is **nominal**: declare
  it, or it will be inferred as quantitative.
- **Missing values.** An empty quantitative cell becomes NaN, and NaN rows are excluded
  by any active range brush on that field. An empty nominal cell becomes the category
  `""`.
- `title` labels axes. It defaults to the field name.
- **`domain`** lists a nominal field's categories in order, each once. Every view of
  the field draws them in that order, also a category with no rows (bars keep its band,
  a sankey its place). Values the list does not name follow it, in order of first
  appearance.
- **`palette`** gives a nominal field's colours, used by every view that draws the
  field's categories: bars, split bars, a scatter or strip coloured by it, and sankey
  and layers nodes. Either a list (1 to 32 colours, in the order of `domain`, or of
  first appearance) or category name to colour; categories it does not name take the
  default colours in turn. A view's own `palette` wins over it. Colours belong to
  categories, so a category keeps its colour when filters remove others.
- A field with a `domain` or a `palette` is nominal; declaring it `quantitative` as well
  is an error.

### 4.4 Field types each view needs

| View | Quantitative | Nominal | Either |
|---|---|---|---|
| `scatter` | `x`, `y` | | `color` |
| `histogram` | `field` | | |
| `bars` | | `field`, `color` | |
| `density` | `x`, `y` | | |
| `scatter3d` | `x`, `y`, `z` | | `color` |
| `strip` | `x` | `y` | `color` |
| `matrix` | | `x`, `y` | |
| `interval` | `low`, `high` | `label` | `color` |
| `arc`, `sankey`, `network3d`, `layers` | | `source`, `target` | |

### 4.5 Views

Every view has `id` (required, unique, a non-empty string), `type` (required), and
optional `title` (defaults to `id`), `height` (plot height in CSS pixels) and `span`
(grid columns occupied, capped at `layout.columns`; default 1).

| `type` | Properties | Brush | Defaults |
|---|---|---|---|
| `scatter` | `x`, `y`, `color?`, `brush?`, `pointSize?`, `fill?`, `palette?` | `"xy"` (rectangle) or `"none"` | brush `"xy"`, height 320 |
| `histogram` | `field`, `bins?`, `brush?`, `fill?` | `"x"` (range) or `"none"` | bins 60, brush `"x"`, height 200 |
| `bars` | `field`, `color?`, `brush?`, `palette?` | `"select"` (click a category to toggle it) or `"none"` | brush `"select"`, height 22 px per category, between 120 and 420 |
| `density` | `x`, `y`, `bins?` (`[xBins, yBins]`) | none | bins `[128, 128]`, height 300 |
| `scatter3d` | `x`, `y`, `z`, `color?`, `pointSize?`, `fill?`, `palette?` | none (drag to orbit, scroll to zoom) | height 360 |
| `strip` | `x` (the value), `y` (the category), `color?`, `brush?`, `pointSize?`, `fill?`, `palette?` | `"x"` (range of the value) or `"none"` | brush `"x"`, height 30 px per category, between 120 and 480 |
| `matrix` | `x`, `y`, `brush?` | `"select"` (click a cell, or a row or column label) or `"none"` | brush `"select"`, height 26 px per `y` category, between 80 and 480 |
| `interval` | `low`, `high`, `label?`, `color?`, `scale?` (`"shared"` or `"label"`), `brush?`, `fill?`, `palette?` | `"x"` (rows overlapping a range) or `"none"`; none with `scale: "label"` | scale `"shared"`, brush `"x"`, height 8 px per row, between 120 and 520 |
| `arc` | `source`, `target`, `order?` (`"count"`, `"name"` or `"data"`), `brush?` | `"select"` (click an arc or a node) or `"none"` | order `"count"`, brush `"select"`, height 300 |
| `sankey` | `source`, `target`, `brush?` | `"select"` (click a flow, or a node on either side) or `"none"` | brush `"select"`, height 360 |
| `network3d` | `source`, `target`, `brush?` | `"select"` (click a node) or `"none"`; drag to orbit, scroll to zoom | brush `"select"`, height 420 |
| `layers` | `source`, `target`, `brush?` | `"select"` (click a flow or a node) or `"none"` | brush `"select"`, height 360 |

Constraints:
- **`bins`:** an integer from 1 to 10,000 for histograms; each of the pair an integer
  from 1 to 1,024 for density.
- **`height`, `span` and `pointSize`:** positive numbers.
- **`fill`:** one colour for the marks when no `color` field is given (for a histogram,
  its bars): `"#rrggbb"`, `"#rgb"` or a CSS colour name such as `"steelblue"`.
- **`palette`:** 1 to 10 such colours for the categories, in order of first appearance;
  they repeat beyond that. It colours a `color` field's categories, and a bar chart's bars.
- **Brushes:** only `scatter`, `histogram`, `bars`, `arc`, `sankey` and `network3d`
  accept a `brush` property. On any other view it is an error.

What each view draws:
- **`scatter`:** rows that pass every brush are drawn coloured, on top of the rest in
  grey. When fewer than 20,000 rows are selected, selected points are drawn twice as
  large. Axes are rounded outward ("niced") from the data extent.
- **`histogram`:** equal-width bins over the field's minimum to maximum. Bars count rows
  passing every *other* view's brush. Light context bars behind them show the
  unfiltered distribution while it fits on the same axis; they are hidden when it would
  not fit. The count axis follows the filtered counts.
- **`bars`:** one horizontal bar per category, in order of first appearance in the data,
  coloured from Tableau 10 (repeating after 10). It counts like a histogram.
- **`density`:** a grid of counts over the rows passing every brush, coloured with
  inferno on a log scale. Empty bins are transparent. Prefer it to `scatter` for
  hundreds of thousands of rows or more, where points saturate into a solid blob.
- **`scatter3d`:** each axis is scaled separately to fill a cube, so the aspect ratio is
  **not** preserved. It shows the selection made in other views.
- **Edge views (`arc`, `sankey`, `network3d`):** each row is one edge from `source` to
  `target`. Rows with the same pair are one edge, weighted by how many of them pass every
  *other* view's brush. Each edge view has one crossfilter dimension over the pairs, so
  clicking a node selects every edge that touches it, which a filter on either field
  alone cannot express. Selections are reported as `"source → target"` values.
- **`arc`:** nodes on one line, ordered by the rows touching them unless `order` says
  otherwise; every edge is an arc above the line, as thick as its rows. Light context
  arcs show the unfiltered edges while another view filters.
- **`sankey`:** sources on the left and targets on the right, ordered by their totals;
  each flow is as wide as its rows. A node that is both appears on both sides. Labels
  are shown where a node is tall enough; every node names itself on hover.
- **`network3d`:** nodes placed by a three-dimensional force layout that starts from a
  fixed arrangement, so the same data always gives the same picture. Nodes are sized
  by the rows that touch them; edges fade as the filters empty them. The 24 busiest
  nodes are labelled.

- **`bars` with `color`:** each bar split into the categories of a second nominal field,
  stacked in that field's colours, with a key. Clicking a bar still selects its whole
  category; the selection reports categories of `field`.
- **`strip`:** one band per category of `y`, each row a point at its value of `x`,
  spread across the band so equal values stay visible. Without `color` it is coloured by
  its category when that field declares a `palette`.
- **`matrix`:** one cell per pair of categories that occurs, shaded by the rows holding
  it after the other views' filters (light to dark, square-root scale), with the count
  written in cells large enough. A cell emptied by filters keeps a faint outline.
- **`interval`:** each row a line from `low` to `high` on a lane of its own, grouped in
  bands by `label` and ordered by low end within a band; rows missing either end are
  left out. With `scale: "label"` each band has its own scale, for quantities in
  different units, with its ends printed at the band's corners. A brush keeps the rows
  whose interval overlaps the range (low end at or below its top, high end at or above
  its bottom), reported as `overlaps`.
- **`layers`:** a directed graph left to right: each node one layer after the furthest
  node leading to it, nodes ordered within a layer to keep flows short, flows as wide as
  their rows. An edge that closes a cycle (found by depth-first search) loops back under
  the graph, dashed. For pipelines of steps and rows passing between them.

**`color`:** a nominal field uses Tableau 10 by category, or the field's `palette`; a quantitative field uses
viridis over the field's extent. Without `color`, points use one colour. A key under the
view's title names the colours: swatches for up to 12 categories (then "+N more"), or the
viridis ramp between the field's minimum and maximum. A density view's key shows its
inferno ramp from 1 to the fullest bin's count.

**Axes.** Every 2D view has light gridlines under its data, tick labels sized to read
(integers without decimals, values from 10,000 up as 12k or 3.4M), and axis titles
centred along each axis. Titles come from `fields.<name>.title`, else the field name.

### 4.6 Limits

- **32 crossfilter dimensions per spec.** The validator reports a spec that needs more.
  - 2: a scatter or an interval with a brush, and density.
  - 1: histogram, bars, a strip with a brush, matrix, arc, sankey, network3d, layers, and
    a `key`.
  - 0: scatter3d, and a scatter, strip or interval without a brush.
- **Memory.** Each dimension holds its own copy of its field: 8 bytes per row for
  numbers, 4 for category codes, plus 4 bytes per row for the sort order. On top of
  that come 5 bytes per row of filter state, and the page's own copies for drawing.
  Measured: 10 million rows with four dimensions used 631 MiB of WASM memory. Budget
  views accordingly at that scale.
- **Rows:** at most 4,294,967,295 (the crossfilter indexes rows with 32 bits). The
  demo's synthetic generator caps itself at 20 million.

<!-- --8<-- [end:spec] -->

### 4.7 Chart descriptions: the short form

A description says which columns to chart; the compiler in `crates/spectacular` turns it
into a spec. It is what the Python package and the native command accept, and a full spec
passes through it unchanged, so either form works everywhere a description does.

```json
{"title": "Growth plate", "labels": {"od600": "OD600"}, "categorical": ["well"], "columns": 2,
 "charts": [
   {"x-axis": "time_h", "y-axis": "od600", "color": "medium", "palette": ["teal", "orange"]},
   {"x": "rate", "color": "slateblue"},
   {"x": "medium", "wide": true}
 ]}
```

- **Forgiving keys and names.** Keys match without regard to case, spaces, dashes or
  underscores, and have aliases (`x-axis`, `colour`, `chart` for `type`, `from` and `to`
  for `source` and `target`). Column names match the same way; a name that matches no
  column gets a suggestion.
- **Chart types from the columns**, when `type` is absent:

  | Columns | Chart |
  |---|---|
  | one number | `histogram` |
  | one category | `bars` (with a category `color`: split bars) |
  | two numbers | `scatter`, or `density` above 200,000 rows |
  | a number and a category | `strip` (the value across, the categories down) |
  | two categories | `matrix` |
  | three numbers | `scatter3d` |
  | `low` and `high` (and a category as `y`) | `interval` |
  | `from` and `to` | `sankey`; `"type": "arc"`, `"network"` or `"layers"` for the others |

- **Colours.** `color` is a column to colour by, one colour, or a list of colours;
  `palette` lists category colours. A histogram takes one colour (`fill`); bars a palette,
  or a category column to split by; point charts, strips and intervals all three. A
  palette given with a category column becomes that field's `palette` (§4.3), so every
  chart of the column shows the same colours. These become the spec's `color`, `fill`
  and `palette` (§4.5).
- **`scale`** on an interval: `"each"` (or `"label"`) for a scale per band, `"shared"` for
  one axis.
- **Around the charts:** `charts` holds them (or give a list at the top, or one chart's
  keys at the top). Beside it:
  - `title` and `theme`;
  - `labels`: an axis label per column;
  - `categorical`: numeric columns that are categories;
  - `domains`: a column's categories in order, and `palettes`: a column's colours (a
    list or by name); both become `fields` (§4.3);
  - `key`: the column naming rows;
  - `columns`: charts per row;
  - `data`: a file, when no table is given otherwise;
  - `figure`: §8.3's size.
- **Ids and titles** are made from the columns (`"od600 vs time_h"`, id
  `od600-vs-time-h`), unless given.
- **Problems** point at the description's own keys, e.g.
  `/charts/0/x-axis: no column "od60"; did you mean "od600"?`, and a spec's problems
  found after compiling are mapped back the same way.

```bash
spectacular compile chart.json --data readings.csv --json   # {"ok", "spec", "problems"}
spectacular validate spec.json --json                         # as §8.2, without Node
spectacular html chart.json --data readings.csv -o page.html  # one offline page
```

The native command's `validate` checks exactly what the Node command's does:
`cli/conformance.mjs` runs both on seeded random mutations of valid specs and requires
the same problems, rows and field types from each.

## 5. How linking works

The numbers you read back mean the following:

- **A row is selected when it passes every active brush.** The status line shows
  "N of M rows selected".
- **Each view brushes its own copy of its fields.** Two views on the same field do not
  overwrite each other's ranges.
- **A histogram or bars view ignores its own brush.** It keeps its full shape while
  narrowing the others. A density view has no brush, so it reflects all of them.
- **Ranges are inclusive at both ends**, in data units. A scatter rectangle is two
  ranges, one on x and one on y, applied together. Clicking empty space in a brushable
  view clears its brush. In bars, deselecting the last chosen category clears the
  filter.
- **Updates are coalesced.** While the worker is busy, newer brush positions replace
  older ones, so a fast drag never queues stale work.
- **Views follow their container's width.** When it changes by 20 px or more (a resized
  window, a notebook laying out its output), the views are drawn again for the new width
  and **every brush is cleared**. A program reading selections then sees one with no
  filters.

Measured cost, 10 million rows: about 7–9 ms per brush step while dragging, and about
150–250 ms when a change moves most of the table, such as clearing a narrow filter. The
README has the details.

## 6. Choosing views for common tasks

| Question | Views |
|---|---|
| Which rows make this peak? | `histogram` of the variable with a brush, plus views of the variables you suspect |
| How do two variables relate, under cuts on others? | `scatter` up to about 200,000 rows; `density` beyond; histograms of the cut variables |
| Is there 3D structure (trajectories, embeddings, positions)? | `scatter3d`, plus a `scatter` projection with a brush to select a region |
| How do groups differ? | `bars` of the group field (click to isolate) and histograms coloured by the same field |
| Where in time did something happen? | `histogram` of time with many bins (100–300) and a brush |

Make IDs short and stable. A test or a later edit can then find a view by
`[data-view="<id>"]`.

## 7. Embedding with `mount()`

```ts
import { mount, validate, SpecError, type Spec } from "@spectacular/viz";

// columns only for data: { columns: "external" }; e.g. px: Float64Array, species: string[]
const handle = await mount(element, spec, { columns: { px, species } });
handle.onUpdate((state) => console.log(state.selectedCount, state.rows));
handle.destroy();
```

- `mount` validates first. On problems it rejects with a `SpecError` whose `problems` is
  `[{ path, message }]`.
- External columns can be `Float64Array` or any array of numbers (quantitative), an
  array of strings (nominal), or `{ codes, labels }` (pre-encoded nominal). All columns
  must have the same length.
- The handle exposes:
  - `table`: the loaded columns, with `rows` and `columns`;
  - `engine.state`: `mask` (1 byte per row, 1 = selected), `selectedCount`, `rows`,
    `counts` and `baseline` (per-group counts, filtered and unfiltered), and `elapsed`
    (crossfilter milliseconds for the last update);
  - `onUpdate(listener)`;
  - `selection()`: the active brushes in data terms, `{ selected, rows, filters }`, the
    same object `spectacular serve` prints (§8.4);
  - `selectedRows()`: the selected row indices (0-based, in data order) as a
    `Uint32Array`;
  - `selectedKeys()`: the spec `key` values of the selected rows;
  - `select(keys)`: selects rows from outside by `key`, as one more filter on top of the
    brushes (`null` clears it), and returns how many of the keys the data holds. A page
    that holds the selection elsewhere passes its row ids here, and every view shows them;
  - `toSVG()`: the views as they are now, including any brushes, as a standalone SVG
    document (a string), built the same way as `render`'s SVG (§8.3);
  - `destroy()`.
- The package is not published to npm. Inside this repository it is the npm workspace
  in `packages/viz` (package name @spectacular/viz), and its entry point is TypeScript
  source (`packages/viz/src/index.ts`), so the
  consuming app must bundle it (Vite works). The worker and WASM files are resolved
  with `new URL(..., import.meta.url)`.

## 8. Driving it from another program

A program talks to spectacular through one command, `cli/spectacular.mjs`. It writes a
spec and a data file, runs a command, and reads JSON from stdout and an exit code. That
is the whole interface, so the calling language does not matter.
`examples/programs/growth_plate.py` is a complete caller in standard-library Python: it
writes plate-reader readings, validates, renders a PDF and a PNG, then serves the views
and prints the wells a person brushes.

### 8.1 Commands and exit codes

```
spectacular validate <spec.json|-> [...]          check specs and their local data
spectacular render <spec.json|-> -o <out.svg|.pdf|.png> [--width N] [--unit mm|in|px] [--dpi N]
spectacular serve <spec.json> [--port N] [--open] [--indices] [--no-watch]
spectacular schema                                print the JSON Schema for specs
```

| Exit | Meaning | What to do |
|---|---|---|
| `0` | Done. | |
| `1` | The spec or its data has problems. | Read `problems`, fix the spec or the data file, retry. |
| `2` | Usage or I/O: a bad flag, an unreadable file, an unsupported output extension, or the viewer is not built. | Fix the call or the setup (§2). |
| `3` | Rendering failed in the browser. | Read `error`; retrying the same spec will fail the same way. |

`-` in place of a spec file reads the spec from stdin; a relative `data.url` then
resolves against the working directory. `serve` needs a real file, because it watches
it.

### 8.2 Machine output (`--json`)

With `--json`, every command prints one JSON object per line on stdout and nothing else
there. Human-readable messages go to stderr.

- **validate:** one line per spec:
  `{"spec":"x.spec.json","ok":true,"problems":[],"rows":29900,"fields":{"t":"quantitative","lobe":"nominal"}}`.
  `rows` and `fields` are present only when the data was loaded (a local file or inline
  values).
- **render:**
  `{"ok":true,"file":"/abs/fig.png","format":"png","width":960,"height":1060,"pixels":[1920,2119]}`.
  `width` and `height` are in CSS pixels; `pixels` (PNG only) is the image size.
- **Any failure:** `{"ok":false,"error":"...","exit":1}`, with `problems` added when the
  spec was invalid.
- **serve:** a stream of events, described in §8.4.

`spectacular schema` prints a JSON Schema (draft 2020-12) for specs. It is committed as
`spectacular.schema.json`. A program can use it to check a spec before writing it, or to
generate one. The schema covers structure only; `validate` also checks fields against
the data and the 32-dimension limit.

### 8.3 `render`: figures to files

```bash
spectacular render growth.spec.json -o growth.svg
spectacular render growth.spec.json -o growth.pdf
spectacular render growth.spec.json -o growth.png --width 900 --unit px --dpi 192
```

- The output format comes from the extension: `.svg`, `.pdf` or `.png`.
- **Size:** each flag overrides one value of the spec's `output.figure`. Without either,
  the figure is 960 CSS px wide. Units: `mm` and `in` are physical; 1 CSS px is 1/96 in.
  `dpi` defaults to 300 for `mm` and `in` and to 192 (2×) for `px`. The height follows
  from the views' heights.
- **SVG:** a standalone document. Its `width` and `height` carry the physical unit
  (`180mm`), so it keeps its size when placed in a paper, and the `viewBox` is in CSS
  px. Titles and labels are `<text>`; axes, histograms, bars and panel frames are
  shapes with their colours written in, so the file looks the same in a browser,
  Inkscape or Illustrator. A 2D scatter of up to 50,000 rows is one `<circle>` per point.
  Larger scatters, density grids and 3D views are embedded PNG images at the chosen
  dpi (density keeps square bins: `image-rendering: pixelated`).
- **PDF:** a single page exactly the figure's size. Text, axes, histograms and bars stay
  vector. Point and density views are embedded as images at the chosen dpi.
- What is drawn is the unfiltered state, with the spec's title and without the
  interactive status line and hints.
- The command validates first: an invalid spec exits `1` without starting a browser.
  Rendering takes a few seconds, mostly starting Chromium.

### 8.4 `serve`: interactive views that report back

```bash
spectacular serve growth.spec.json --json --indices --open
```

`serve` starts a web server on 127.0.0.1 only (a random port unless `--port`). It serves
the spec and the files in its directory and below, under a random path prefix; the URL
contains that prefix. `--open` opens the URL in the default browser. It runs until
stopped (ctrl-c, or the signals `SIGINT` and `SIGTERM`). With `--json` it prints events:

```json
{"event":"listening","url":"http://127.0.0.1:53111/?spec=/s/3f…/growth.spec.json&channel&indices"}
{"event":"selection","selected":2985,"rows":29900,"filters":[{"view":"time","field":"t","range":[60.7,90.54]}],"indices":[6070,6071,...]}
{"event":"selection","selected":12941,"rows":29900,"filters":[{"view":"lobe","field":"lobe","values":["left"]}]}
{"event":"reload","file":"readings.csv"}
{"event":"problems","problems":[{"path":"/views/0/field","message":"..."}]}
```

- **`selection`** is printed after each brush settles (150 ms without change). A row is
  selected when it passes every filter. A filter is one of these:
  - `range`: inclusive at both ends, in data units, never matching a missing value;
  - `values`: the chosen categories of the field (bars, split bars), `"a × b"` pairs (a
    matrix), or `"source → target"` (edge views);
  - `overlaps: [a, b]`, for an interval: `field` is the low end and `to` the high end, and
    an open end is `null`.

  Rows selected from outside by key show as a filter with `view` `""` and the key field.
  With a spec `key`, the event also lists the selected rows' `keys`. `filters` is empty when nothing is brushed. A program can apply the
  filters to its own copy of the data and get exactly the selected rows.
- **`indices`** (only with `--indices`) lists the selected rows, 0-based, in data order.
  At millions of rows this is large; prefer `filters` there.
- **`reload`:** the spec, or the local data file it names, changed on disk. Open pages
  reload, and brushes are lost. A program can append readings to the data file while a
  run is in progress, and the views follow (`--no-watch` turns this off).
- **`problems`** is printed once at start if the spec is invalid. The server still runs,
  the page shows the problems, and it reloads once the file is fixed.

The URL is the access check. Any page holding it can read the spec's directory, so do
not paste it anywhere shared.

### 8.5 Driving the page yourself

There is no programmatic "brush view X to [a, b]" call yet. To script brushes, drive the
page the way a user would, as `cli/test.mjs` does with Playwright:

```js
await page.goto(url);   // the URL from serve's "listening" event
await page.waitForFunction(() => document.body.dataset.state !== "loading");
const box = await page.locator('[data-view="mass"] .sv-brush .overlay').boundingBox();
await page.mouse.move(box.x + box.width * 0.3, box.y + box.height / 2);
await page.mouse.down();
await page.mouse.move(box.x + box.width * 0.5, box.y + box.height / 2, { steps: 5 });
await page.mouse.up();
const n = await page.evaluate(() => window.spectacular.engine.state.selectedCount);
```

In headless Chromium, WebGL runs in software (SwiftShader). Launch with
`--use-angle=swiftshader --enable-unsafe-swiftshader`. Software rendering competes with
the worker for CPU, so crossfilter timings measured there are not performance figures.
In this repository's runs, one category filter on 1M rows took 42–388 ms in headless
Chromium against 12–18 ms in Node. For timings, run `npm run test:wasm` or
`cargo run --release -p chigutiro --example bench`.

### 8.6 From Python and Jupyter

The `spectacular` Python package (in `python/`, published as `spectacular-charts`; the
import name is `spectacular`) shows the views in a notebook and reads the brushes back
into Python. Its wheel carries the browser runtime and the native command (§4.7), so it
needs neither Node nor a checkout: `pip install "spectacular-charts[notebook]"`. From a
checkout, see `python/README.md`.

A notebook cell can hold nothing but a description:

```
%%spectacular df --as w
{"charts": [{"x-axis": "time_h", "y-axis": "od600", "color": "medium"}, {"x": "rate"}]}
```

```python
import spectacular as sp

w = sp.view(spec, data=df)    # a description or a spec; as a cell's last expression, the views appear
w.selection                   # {"selected": 412, "rows": 2400, "filters": [...]}, as in §8.4
w.selected_rows               # numpy array of row indices into df
w.selected_data()             # df.iloc[w.selected_rows]
w.on_select(callback)         # called with the selection after each brush settles
w.update(spec=new_spec)       # redraw with a new spec or data= (brushes are cleared)
w.save_svg("now.svg")         # what the view shows now, brushes included (§8.3's SVG)
w.selected_keys               # the `key` values of the selected rows
w.select(["r001", "r007"])    # select rows by key from Python, on top of the brushes; None clears
w.problems                    # the spec's problems, as the page reported them

sp.save_html(spec, "views.html", data=df)   # one offline page; nothing is reported back
sp.render(spec, "fig.pdf", data=df)         # §8.3, in headless Chromium (see below)
sp.validate(spec, data=df)                  # [{"path", "message"}], by the native command
sp.schema()                                 # the JSON Schema of §8.2
```

- **`spec`** is a description (§4.7) or a spec: a dict, a list of charts, a path to a JSON
  file, or JSON text. A program can build it from its own rules; nothing in it is
  Python-specific.
- **`data=`** is a pandas or polars DataFrame, a dict of columns, a list of row dicts, or
  a numpy record array. It replaces the spec's `data`. Only the fields the views name and
  the ones `fields` declares are sent, as binary columns: integer and float columns as
  float64, everything else as categories (booleans as `"true"`/`"false"`, missing values
  as NaN or `""`). A type declared in `fields` wins. Dates are refused; convert them to
  numbers first.
- **Without `data=`**, the spec's own data is used. A local `data.url` is read from disk
  (relative to the spec file, or the working directory for a dict or text) and sent to
  the page, so no server is involved.
- **Problems** show in the cell, where the views would be, each with its path in what
  was given, and in `w.problems`; `sp.validate` returns them without drawing.
- **`render`** draws through Playwright for Python when it is installed
  (`pip install "spectacular-charts[render]"`, then `python -m playwright install
  chromium`), else through the Node command of a checkout.
- **Selections** arrive 150 ms after a brush settles, while the kernel is idle: read them
  in a later cell, or in `on_select`. `track_rows=False` stops sending row indices,
  for tables of millions of rows where only the filters are wanted.
- **`save_svg`** is answered by the page, so the view must be displayed. The file is
  written after the cell finishes; `w.saved` lists the files written.
- **Without anywidget**, `sp.view` returns the same views as an iframe page: brushing
  works, nothing comes back to Python.
- **Theme:** the views follow JupyterLab's, Visual Studio Code's and Colab's light or dark theme
  unless the spec sets one.

The runtime is one JavaScript file in the package (about 1 MB, with the crossfilter's
WASM inside). If a front end refuses Web Workers, the crossfilter runs on the page's own
thread instead: the same answers, with slower brushing at millions of rows.
`examples/notebooks/spectacular-in-jupyter.ipynb` walks through all of it.

## 9. Not supported yet

Do not write specs that rely on these; the validator will reject most of them:

- **Rendering a brushed state.** `render` draws the unfiltered views; a selection cannot
  be set in the spec.
- **Arrow or Parquet input.** Use CSV, TSV, JSON rows, inline values or external
  columns.
- **Uncertainty as error bars on points.** A quantity with two ends is an `interval`;
  a value with a resolution has no mark of its own yet.
- **One chart per value of a field** (facets, small multiples): repeat the description
  per value and filter the table, or use one description per table.
- **Data transforms.** There are no derived fields, aggregation, joins or multiple
  datasets. Compute them in your repo first and write the result to the data file.
- **Initial brushes in the spec**, or a programmatic brush API.
- **Display options:** pan or zoom, log scales, explicit axis domains, histogram ranges,
  and tooltips outside the edge views.
- **More selection:** lasso, brushing in 3D, and brushing a density view.

## 10. Checklist before you hand a spec over

1. `spectacular validate <spec>` exits 0, and the summary lists every field with the
   type you intended.
2. Numeric codes and identifiers are declared `nominal`.
3. Above about 200,000 rows you used `density` and kept `scatter` for regions you can
   brush down to.
4. The dimension count is within 32 (§4.6). The validator checks this.
5. In viewer mode, the page reached `data-state="ready"`, and the status line shows the
   row count you expect.
6. A program checks the exit code of every call, not just the output.
