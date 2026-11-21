# Spectacular

Linked 2D and 3D scientific views over millions of rows. Brushing any view filters all the
others. A Rust crossfilter (`chigutiro`), compiled to WebAssembly, runs in a Web Worker
behind a TypeScript renderer.

Other projects never call the renderer directly. They hand it a **view spec**: a JSON
file naming the data, the fields and the views. A script, a build step or a model can
write one. The renderer is deterministic, and it rejects an invalid spec with errors that
point at the exact path.

## In Jupyter and Python

`pip install "spectacular-charts[notebook]"`, `import spectacular`, then describe the
charts instead of coding them. A cell holding only a description draws linked charts:

```
%%spectacular df
{"title": "Growth plate",
 "charts": [
   {"x-axis": "time_h", "y-axis": "od600", "color": "medium"},
   {"x": "rate", "color": "teal"},
   {"x": "medium"}
 ]}
```

Two numbers make a scatter, one number a histogram, one category bars; keys and column
names are matched loosely, and a misspelt column comes back as
`/charts/0/x-axis: no column "od60"; did you mean "od600"?`. From Python:

```python
import spectacular as sp

w = sp.view(description, data=df)  # the same charts; full view specs work too
w.selection                        # what you brushed, in data terms
w.selected_data()                  # ...and as the rows of df
w.save_svg("brushed.svg")          # what the charts show now

sp.save_html(description, "views.html", data=df)   # one offline page
sp.render(description, "fig.pdf", data=df)         # a figure at print size
```

The wheel carries the browser runtime and a native `spectacular` command, built from
`crates/spectacular`, which compiles descriptions and validates specs without Node. Its
validator is a port of the TypeScript one, and `npm run test:conformance` holds the two to
the same answers.

**Documentation: <https://fullscreen-triangle.github.io/spectacular/>** (built from
[`docs/`](docs/) by `npm run docs:build`): a quickstart, a guide with a picture of every
chart type, the Python API and every description key. [`docs/for-ai.md`](docs/for-ai.md)
is the description language on one page for agents, and [`docs/llms.txt`](docs/llms.txt)
indexes the site for them. [`python/README.md`](python/README.md) is the PyPI page, and
[`examples/notebooks/spectacular-in-jupyter.ipynb`](examples/notebooks/spectacular-in-jupyter.ipynb)
a notebook to start from.

## Handing spectacular a task

**Programs and AI agents in other repos: read
[`spectacular-specifications.md`](spectacular-specifications.md).** It is the complete
guide to the spec, the command, limits and what is not built yet.

A program in any language drives one command and reads JSON lines and exit codes
(0 ok, 1 spec or data problems, 2 usage or I/O, 3 rendering failed):

```bash
spectacular validate run.spec.json --json          # problems by JSON path; row count and field types
spectacular render run.spec.json -o fig.svg        # SVG, PDF or PNG, sized by output.figure or flags
spectacular serve run.spec.json --json --indices   # linked views on 127.0.0.1; prints each selection
spectacular schema > spectacular.schema.json       # JSON Schema for specs
```

`spectacular` is `node cli/spectacular.mjs` (or `npm link` it). `serve` prints
`{"event":"selection","selected":…,"filters":[{"view":"time","field":"t","range":[60.7,90.54]}]}`
after each brush and reloads the page when the spec or data file changes, so a running
experiment can append readings and the views follow.
[`examples/programs/growth_plate.py`](examples/programs/growth_plate.py) is a complete
caller in standard-library Python.

```json
{
  "version": "spectacular/v0",
  "title": "Synthetic collision events",
  "data": { "url": "events.csv" },
  "fields": { "mass": { "type": "quantitative", "title": "invariant mass (GeV)" } },
  "layout": { "columns": 3 },
  "views": [
    { "id": "momentum", "type": "scatter",   "x": "px", "y": "py", "color": "species" },
    { "id": "density",  "type": "density",   "x": "px", "y": "py", "bins": [192, 192] },
    { "id": "space",    "type": "scatter3d", "x": "px", "y": "py", "z": "pz" },
    { "id": "mass",     "type": "histogram", "field": "mass", "bins": 120, "span": 2 },
    { "id": "species",  "type": "bars",      "field": "species" }
  ]
}
```

| View | Reads | Brush |
|---|---|---|
| `scatter` | `x`, `y`, optional `color` | rectangle (`"xy"`) |
| `histogram` | one quantitative `field` | range (`"x"`) |
| `bars` | one nominal `field` | click to toggle categories (`"select"`) |
| `density` | `x`, `y`, binned on a grid | — |
| `scatter3d` | `x`, `y`, `z`, optional `color` | — (shows other views' filters) |
| `arc` | edges: nominal `source` and `target` | click an arc or a node (`"select"`) |
| `sankey` | edges: nominal `source` and `target` | click a flow or a node (`"select"`) |
| `network3d` | edges: nominal `source` and `target`, laid out in 3D | click a node (`"select"`) |
| `layers` | edges as a directed graph in layers, left to right | click a flow or a node (`"select"`) |
| `strip` | a number `x` against a category `y` | range of the value (`"x"`) |
| `matrix` | two categories, a cell per pair | click a cell or a row or column label (`"select"`) |
| `interval` | `low` to `high` per row, banded by a `label` | rows overlapping a range (`"x"`) |

`bars` take a second category as `color` and split each bar into it. A field can declare
its categories' order (`fields.<f>.domain`) and colours (`fields.<f>.palette`, a list or by
name), which every view of it follows. A spec `key` names rows: selections report it, and
rows can be selected from outside by it. `examples/bundle.desc.json` and
`examples/lineage.desc.json` use all of these on tables shaped like a view bundle's.

`data` is `{ "url" }` (CSV, TSV or JSON rows), `{ "values": [...] }`, or
`{ "columns": "external" }` with columns passed in by the embedding program. With `url`,
`format` overrides the extension and `pointer` (a JSON Pointer such as `"/executions"`)
reads rows nested inside a JSON or JSON for Linked Data (JSON-LD) document.
`examples/synthetic.spec.json` and `examples/lorenz.spec.json` (with `lorenz.csv`) are
complete specs. `"theme"` is `"auto"` (follow the page), `"light"`, `"dark"` or `"paper"`
(white, borderless, for print). `validate(spec)` returns
`[{ path, message }]`, for example `/views/2/field: histogram.field needs a quantitative
field; "species" is nominal`.

## Embedding

```ts
import { mount } from "@spectacular/viz";
const handle = await mount(element, spec, { columns });  // columns only for "external" data
handle.onUpdate((s) => console.log(s.selectedCount, handle.selection().filters));
const svg = handle.toSVG();      // the views as they are now, as a standalone SVG document
handle.destroy();
```

## Develop

Needs Rust with the `wasm32-unknown-unknown` target, `wasm-pack`, Node 22.18+, and Python
3.9+ for the Python package.

```bash
npm run build:wasm          # crates/chigutiro-wasm -> packages/chigutiro-wasm (generated; first)
npm install
npm run build               # the viewer (packages/viz/dist) and the notebook runtime (python/)
npm run dev                 # demo at http://localhost:5173/?rows=2000000
npm test                    # Rust tests, WASM-vs-JS check, spec/data/planner tests
npm run test:e2e -w @spectacular/viz   # headless Chromium: brushes, clicks, pixels
npm run test:cli            # the command as a program uses it: exit codes, render, serve
npm run test:conformance    # the Rust and TypeScript validators agree (npm run build:native first)
npm run test:python         # the Python package (maturin develop, or pip install -e "python[test]", first)
SPECTACULAR_PYTHON=<python with jupyterlab> npm run test:jupyter   # a real JupyterLab: view, brush, read back
cargo run --release -p chigutiro --example bench   # 10M-row timings
```

## How the crossfilter stays fast

Each dimension keeps its rows sorted by key, and each row carries a bitmask of the
dimensions currently rejecting it. A brush move computes which sorted positions entered
or left its range, then touches only those rows. That updates their bits, the bins of
every group that doesn't ignore the dimension, and the selected mask. A group ignores its
own view's brush, so a histogram keeps its shape while it filters the other views. When
more than an eighth of the rows cross, one sequential rescan replaces the row-by-row
update. Correctness is checked against a brute-force model over thousands of random
filter sequences, on both paths.

Measured 2026-10-22, 10M rows, four dimensions, two histograms, category counts and a
256×256 density grid:

| | native | WASM (Node) |
|---|---|---|
| drag a brush (≈0.7% of rows cross per step) | 8–9 ms median | 7 ms median |
| clear a narrow filter (73% cross) | ≈250 ms | ≈150 ms per rescan |
| build dimensions | 2.5–3.3 s | 2.0 s |

The page keeps a copy of the selected mask. The worker sends only the rows that flipped,
or the whole mask when that is smaller, and the GPU gets one byte per point per update.

## Status

Working: the crossfilter and its WebAssembly (WASM) bindings, the spec, validator and
JSON Schema, and CSV/TSV/JSON/JSON-LD/inline/external data. There are twelve view types:
WebGL2 scatter and strip, three.js 3D scatter, density grid, histogram, bars (also split
by a second category), matrix and interval, and the arc, sankey, network3d and layers edge
views. Fields can declare a category order and colours, and a key names rows. Axes have gridlines and readable tick labels, colour has a key,
and four themes cover the web, dark pages and print. The `spectacular` command validates
specs, renders SVG, PDF and PNG figures at physical size, and serves a viewer that
reports selections. The Python package shows the views in Jupyter, with selections
readable from Python, and writes offline pages and figures. The demo and end-to-end
browser tests run against the production build, and a JupyterLab test drives a notebook.

Next:
- Rendering a brushed state from the command line.
- Apache Arrow input (the inter-process communication (IPC) file format).
- Pan and zoom.
- Lasso selection.
- Faster large changes (cached per-row bins, f32 keys, radix-sorted builds).

The 2025 implementation (a natural-language-to-D3 pipeline) is preserved at tag `legacy-2025`.
