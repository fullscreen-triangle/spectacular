# Spectacular: specification for AI agents

This file is for AI agents working in **other** repositories that need an interactive
scientific visualisation. It says what spectacular does, how to hand it a task, the
complete spec format, and what it does not do yet. Everything here describes the code on
branch `rebuild` of [fullscreen-triangle/spectacular](https://github.com/fullscreen-triangle/spectacular). If this file and the code
disagree, the code wins: `packages/viz/src/spec.ts` is the contract.

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
categories. Do not use it for a static figure in a paper (export is not built yet, see
§9), or for data that is not one flat table.

## 2. The three ways in

| You have | Do this |
|---|---|
| A spec plus a CSV, TSV or JSON file | **Viewer mode**, §2.1. No code. |
| A TypeScript or JavaScript web app | **Embed** with `mount()`, §7. |
| Only a spec to check | **Validate** from the command line, §2.2. |

Setup, once per machine, from the spectacular checkout. It needs Rust with the
`wasm32-unknown-unknown` target, `wasm-pack`, and Node 20 or later:

```bash
npm run build:wasm     # builds packages/chigutiro-wasm; must come before npm install
npm install
```

### 2.1 Viewer mode

1. Put the spec and its data file side by side under `examples/local/<your-project>/`.
   That directory is gitignored, so private data never gets committed. Files under
   `examples/` are served at the site root.
2. In the spec, write `"data": { "url": "<data file name>" }`. A relative URL resolves
   against the spec's own URL.
3. Start the dev server: `npm run dev` (Vite, default `http://localhost:5173`).
4. Open `http://localhost:5173/?spec=/local/<your-project>/<name>.spec.json`.

`?spec=` also accepts an absolute URL, for example a spec served by your own repo's
server. The server must then send Cross-Origin Resource Sharing (CORS) headers for
both the spec and the data.

**Signals for automation.** `document.body.dataset.state` starts as `"loading"` and
becomes `"ready"` or `"error"`. On `"error"`, the element `#error` holds the message,
including every spec problem as `/path: message`. On `"ready"`, `window.spectacular` is
the handle described in §7.

### 2.2 Validate before you open anything

```bash
npm run validate -- path/to/spec.json [more.spec.json ...]
# or: node scripts/validate.mjs path/to/spec.json
```

It checks the structure. When `data.url` names a local file beside the spec, or the
data is inline, it also loads the data and checks that every field exists and has the
type each view needs. It reports every problem in one run:

```
x.spec.json /views/0/field: histogram.field needs a quantitative field; "lobe" is nominal
x.spec.json /views/1/colour: unknown property "colour"; did you mean "color"?
x.spec.json /views/2/field: no field "nope" in the data
x.spec.json /views/3/type: unknown view type "pie"; one of scatter, histogram, bars, density, scatter3d
```

Exit codes: `0` all valid, `1` problems found, `2` usage or I/O error. When a spec's data
is a remote URL or `"external"`, only the structure is checked, and the output says
`valid (structure only)`.

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

### 4.1 Top level

| Property | Type | Required | Meaning |
|---|---|---|---|
| `version` | `"spectacular/v0"` | no | If present, it must be exactly this. Include it. |
| `$schema` | string | no | Ignored by the renderer. |
| `title` | string | no | Heading above the views. |
| `description` | string | no | Not displayed; use it for provenance. |
| `data` | object | **yes** | §4.2. |
| `fields` | object | no | Field types and axis titles, §4.3. |
| `views` | array | **yes** | At least one view, §4.5. |
| `layout` | object | no | `{ "columns": n }`, n an integer from 1 to 12. The default is the smaller of 3 and the number of views. |
| `output` | object | no | `{ "interactive": bool, "figure": {...} }`. It is validated, but **nothing renders `figure` yet** (§9). |

Any other property is an error.

### 4.2 `data`: give exactly one of these

| Form | Use |
|---|---|
| `{ "url": "file.csv" }` | CSV. `.tsv` is read as tab-separated; `.json` must be an array of row objects. Relative URLs resolve against the spec. |
| `{ "values": [ { "a": 1, "k": "x" }, ... ] }` | Inline rows, for small tables. |
| `{ "columns": "external" }` | Columns passed by the program calling `mount()` (§7). The fast path for large data. |

Only the fields that views use, or that `fields` declares, are loaded. Other columns
are ignored.

**Size guidance.** Parsing CSV in the browser is the slowest step. Keep `url` and
`values` data to about a million rows. For more, pass typed arrays as external columns.

### 4.3 `fields`

`"fields": { "<name>": { "type": "quantitative" | "nominal", "title": "axis label" } }`

- **Type inference** when `type` is omitted: a field is `quantitative` if every
  non-empty value parses as a finite number, and `nominal` otherwise. Declare the type
  whenever you know it. A numeric code such as a detector id is **nominal**: declare
  it, or it will be inferred as quantitative.
- **Missing values.** An empty quantitative cell becomes NaN, and NaN rows are excluded
  by any active range brush on that field. An empty nominal cell becomes the category
  `""`.
- `title` labels axes. It defaults to the field name.

### 4.4 Field types each view needs

| View | Quantitative | Nominal | Either |
|---|---|---|---|
| `scatter` | `x`, `y` | | `color` |
| `histogram` | `field` | | |
| `bars` | | `field` | |
| `density` | `x`, `y` | | |
| `scatter3d` | `x`, `y`, `z` | | `color` |

### 4.5 Views

Every view has `id` (required, unique, a non-empty string), `type` (required), and
optional `title` (defaults to `id`), `height` (plot height in CSS pixels) and `span`
(grid columns occupied, capped at `layout.columns`; default 1).

| `type` | Properties | Brush | Defaults |
|---|---|---|---|
| `scatter` | `x`, `y`, `color?`, `brush?`, `pointSize?` | `"xy"` (rectangle) or `"none"` | brush `"xy"`, height 320 |
| `histogram` | `field`, `bins?`, `brush?` | `"x"` (range) or `"none"` | bins 60, brush `"x"`, height 200 |
| `bars` | `field`, `brush?` | `"select"` (click a category to toggle it) or `"none"` | brush `"select"`, height 22 px per category, between 120 and 420 |
| `density` | `x`, `y`, `bins?` (`[xBins, yBins]`) | none | bins `[128, 128]`, height 300 |
| `scatter3d` | `x`, `y`, `z`, `color?`, `pointSize?` | none (drag to orbit, scroll to zoom) | height 360 |

Constraints:
- **`bins`:** an integer from 1 to 10,000 for histograms; each of the pair an integer
  from 1 to 1,024 for density.
- **`height`, `span` and `pointSize`:** positive numbers.
- **Brushes:** only `scatter`, `histogram` and `bars` accept a `brush` property. On
  any other view it is an error.

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

**`color`:** a nominal field uses Tableau 10 by category; a quantitative field uses
viridis over the field's extent. Without `color`, points use one colour.

### 4.6 Limits

- **32 crossfilter dimensions per spec.** A scatter with a brush uses 2, density 2,
  histogram 1, bars 1, and scatter3d 0. The validator reports a spec that exceeds the
  limit.
- **Memory.** Each dimension holds its own copy of its field: 8 bytes per row for
  numbers, 4 for category codes, plus 4 bytes per row for the sort order. On top of
  that come 5 bytes per row of filter state, and the page's own copies for drawing.
  Measured: 10 million rows with four dimensions used 631 MiB of WASM memory. Budget
  views accordingly at that scale.
- **Rows:** at most 4,294,967,295 (the crossfilter indexes rows with 32 bits). The
  demo's synthetic generator caps itself at 20 million.

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
  - `destroy()`.
- The package is not published to npm. Inside this repository it is the npm workspace
  in `packages/viz` (package name @spectacular/viz), and its entry point is TypeScript
  source (`packages/viz/src/index.ts`), so the
  consuming app must bundle it (Vite works). The worker and WASM files are resolved
  with `new URL(..., import.meta.url)`.

## 8. Driving it headlessly

There is no programmatic "brush view X to [a, b]" call yet. Drive the page the way a
user would. `packages/viz/test/e2e.mjs` does exactly this with Playwright:

```js
await page.goto("http://localhost:5173/?spec=/local/proj/x.spec.json");
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

## 9. Not supported yet

Do not write specs that rely on these; the validator will reject most of them:

- **Publication figure export.** `output.figure` is accepted and validated, but nothing
  renders SVG, PDF or PNG yet. For now, take a screenshot through §8.
- **Arrow or Parquet input.** Use CSV, TSV, JSON rows, inline values or external
  columns.
- **Data transforms.** There are no derived fields, aggregation, joins or multiple
  datasets. Compute them in your repo first and write the result to the data file.
- **Initial brushes in the spec**, or a programmatic brush API.
- **Display options:** pan or zoom, log scales, explicit axis domains, histogram ranges,
  colour legends and tooltips.
- **More selection:** lasso, brushing in 3D, and brushing a density view.

## 10. Checklist before you hand a spec over

1. `npm run validate -- <spec>` exits 0, and the summary lists every field with the
   type you intended.
2. Numeric codes and identifiers are declared `nominal`.
3. Above about 200,000 rows you used `density` and kept `scatter` for regions you can
   brush down to.
4. The dimension count is within 32 (§4.6). The validator checks this.
5. In viewer mode, the page reached `data-state="ready"`, and the status line shows the
   row count you expect.
