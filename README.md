# Spectacular

Linked 2D and 3D scientific views over millions of rows. Brushing any view filters all the
others. A Rust crossfilter (`chigutiro`), compiled to WebAssembly, runs in a Web Worker
behind a TypeScript renderer.

Other projects never call the renderer directly. They hand it a **view spec**: a JSON
file naming the data, the fields and the views. A script, a build step or a model can
write one. The renderer is deterministic, and it rejects an invalid spec with errors that
point at the exact path.

## Handing spectacular a task

**AI agents in other repos: read [`spectacular-specifications.md`](spectacular-specifications.md).**
It is the complete guide to the spec, viewer mode, validation, limits and what is not
built yet.

Put a spec and its data under `examples/local/<project>/` (gitignored), validate it,
then open it in the viewer:

```bash
npm run validate -- examples/local/proj/x.spec.json
npm run dev     # then http://localhost:5173/?spec=/local/proj/x.spec.json
```

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

`data` is `{ "url" }` (CSV, TSV or JSON rows), `{ "values": [...] }`, or
`{ "columns": "external" }` with columns passed in by the embedding program.
`examples/synthetic.spec.json` and `examples/lorenz.spec.json` (with `lorenz.csv`) are
complete specs. `validate(spec)` returns
`[{ path, message }]`, for example `/views/2/field: histogram.field needs a quantitative
field; "species" is nominal`.

## Embedding

```ts
import { mount } from "@spectacular/viz";
const handle = await mount(element, spec, { columns });  // columns only for "external" data
handle.onUpdate((s) => console.log(s.selectedCount));
handle.destroy();
```

## Develop

Needs Rust with the `wasm32-unknown-unknown` target, `wasm-pack`, and Node 20+.

```bash
npm run build:wasm          # crates/chigutiro-wasm -> packages/chigutiro-wasm (generated; first)
npm install
npm run dev                 # demo at http://localhost:5173/?rows=2000000
npm test                    # Rust tests, WASM-vs-JS check, spec/data/planner tests
npm run test:e2e -w @spectacular/viz   # headless Chromium: brushes, clicks, pixels
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

Working: the crossfilter and its WASM bindings, the spec and validator, CSV/JSON/inline/
external data, and five view types (WebGL2 scatter, three.js 3D scatter, density grid,
histogram, bars). The demo and an end-to-end browser test run against the production
build.

Next:
- Publication figures: render a spec headlessly to SVG, PDF or PNG at physical size.
- Arrow IPC input.
- Pan and zoom.
- Lasso selection.
- Faster large changes (cached per-row bins, f32 keys, radix-sorted builds).

The 2025 implementation (an NL→D3 pipeline) is preserved at tag `legacy-2025`.
