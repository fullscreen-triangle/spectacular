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
They never link against spectacular's code.

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
x.spec.json /views/3/type: unknown view type "pie"; one of scatter, histogram, bars, density, scatter3d
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

### 4.1 Top level

| Property | Type | Required | Meaning |
|---|---|---|---|
| `version` | `"spectacular/v0"` | no | If present, it must be exactly this. Include it. |
| `$schema` | string | no | Ignored by the renderer. Point it at `spectacular.schema.json` for editor checking. |
| `title` | string | no | Heading above the views. |
| `description` | string | no | Not displayed; use it for provenance. |
| `data` | object | **yes** | §4.2. |
| `fields` | object | no | Field types and axis titles, §4.3. |
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
  - `selection()`: the active brushes in data terms, `{ selected, rows, filters }`, the
    same object `spectacular serve` prints (§8.4);
  - `selectedRows()`: the selected row indices (0-based, in data order) as a
    `Uint32Array`;
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
  selected when it passes every filter. A filter is either `range` (inclusive at both
  ends, in data units, never matching a missing value) or `values` (the chosen
  categories). `filters` is empty when nothing is brushed. A program can apply the
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

## 9. Not supported yet

Do not write specs that rely on these; the validator will reject most of them:

- **Rendering a brushed state.** `render` draws the unfiltered views; a selection cannot
  be set in the spec.
- **Arrow or Parquet input.** Use CSV, TSV, JSON rows, inline values or external
  columns.
- **Uncertainty.** There are no error bars or bands. Put a value and its resolution in
  two columns, and plot each.
- **Data transforms.** There are no derived fields, aggregation, joins or multiple
  datasets. Compute them in your repo first and write the result to the data file.
- **Initial brushes in the spec**, or a programmatic brush API.
- **Display options:** pan or zoom, log scales, explicit axis domains, histogram ranges,
  colour legends and tooltips.
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
