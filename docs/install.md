# Install

```bash
pip install "spectacular-charts[notebook]"
```

The distribution is called **`spectacular-charts`**. The name you import is
**`spectacular`**:

```python
import spectacular as sp
```

It needs Python 3.9 or newer. Wheels are built for Linux, macOS and Windows; each carries
the browser runtime and a native `spectacular` command compiled from Rust, so nothing
else has to be installed for descriptions, validation and offline pages.

## What each extra adds

| Install | Gives you |
|---|---|
| `pip install spectacular-charts` | descriptions, `validate`, `save_html` (offline pages), the `spectacular` command. No dependencies. |
| `pip install "spectacular-charts[notebook]"` | interactive charts in Jupyter, with the selection readable from Python (adds [anywidget](https://anywidget.dev)). |
| `pip install "spectacular-charts[render]"` | figures: `render` to SVG, PDF or PNG (adds Playwright). Then run `python -m playwright install chromium` once. |

Combine them as `"spectacular-charts[notebook,render]"`.

!!! tip "Without anywidget"
    `sp.view` still works without the `notebook` extra: the charts are drawn in an iframe
    and brushing works, but nothing is reported back to Python. Install the extra to read
    `w.selection` and `w.selected_data()`.

## Check the install

```python
import spectacular as sp

print(sp.__version__)
print(sp.validate({"x": "rate"}, data=sp.datasets.plate()))   # [] means it holds
```

From a terminal, the command the wheel carries:

```bash
spectacular --version
spectacular schema > spectacular.schema.json
```

## Working on spectacular itself

The runtime that draws the charts is built from TypeScript and Rust, and is not
committed, so `pip install` straight from the Git repository needs a build first.
With Rust (and its `wasm32-unknown-unknown` target), `wasm-pack`, Node 22.18+ and
[maturin](https://www.maturin.rs):

```bash
git clone -b rebuild https://github.com/fullscreen-triangle/spectacular
cd spectacular
npm run build:wasm && npm install && npm run build   # the runtime, into python/src/spectacular/static
cd python && maturin develop --release               # install into the active environment
```

`maturin build --release --out dist` builds a wheel instead. The project's `README.md`
lists the test suites.

!!! warning "An old `spectacular` in the same environment"
    PyPI's `spectacular` is an unrelated project. If it is installed in the same
    environment, it shadows this one; `pip uninstall spectacular` removes it.
