# Changelog

All notable changes to `spectacular-charts`. Versions follow [semantic versioning](https://semver.org);
until 1.0, a minor version may change the description language.

## 0.2.0 (2026-11-21)

### Added

- **Chart descriptions.** Charts are described in JSON (`{"x": "time_h", "y": "od600"}`) instead
  of written as view specs; the chart type follows from the columns. Keys and column names
  match loosely, and mistakes come back at the key written, with a suggestion.
- **The native `spectacular` command**, compiled from Rust and carried in the wheel:
  `compile`, `validate`, `html` and `schema`, with no Node needed.
- **Views for categories and quantities:** `strip`, `matrix`, `interval` and split `bars`, plus
  the edge views `arc`, `sankey`, `network3d` and `layers`.
- **`domains`, `palettes` and `key`** in descriptions: a column's category order and colours,
  followed by every chart of it, and a column that names rows.
- `View.selected_keys` and `View.select(keys)`: read the key values of brushed rows, and select
  rows chosen elsewhere.
- `spectacular.datasets`: `plate`, `lorenz`, `bundle` and `lineage`, the tables the
  documentation uses.
- A documentation site, and a guide written for AI agents (`for-ai.md`, `llms.txt`).

### Changed

- The distribution is named `spectacular-charts`; the import name stays `spectacular`.
- `render` draws with Playwright for Python when it is installed, in a thread of its own so it
  works inside a running notebook.

### Fixed

- A description's `key` column is now sent with the data even when no chart draws it.
  Before, a `key` with a DataFrame failed with `no field "row" in the data`.

## 0.1.0

- First version. `View` (an anywidget) showed linked views in Jupyter, with the selection
  readable from Python. `to_html` and `save_html` wrote offline pages, `render` drew SVG, PDF
  and PNG figures through the Node command, and `validate` and `schema` checked view specs.
