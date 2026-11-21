# Troubleshooting

## The cell shows problems instead of charts

The description could not be drawn. Each problem names the key it is about:

| Message | Fix |
|---|---|
| `no column "od60"; did you mean "od600"?` | the column name; it is matched without case or spacing, but not misspelt |
| `unknown key "colr"; did you mean "color"?` | the key; [Description keys](reference/keys.md) lists them all |
| `"x" and "x-axis" say the same thing; keep one` | two names for one key in one chart |
| `colouring by a column ("medium") works on point charts and bars; give a histogram one colour instead` | a histogram takes one colour; split it with a second chart instead |
| `bars split by a column need a category; "rate" is a number` | name the column in `categorical`, or colour by a category |
| `say what to draw: "x" (and "y", "z"), or "from" and "to"` | a chart named no columns |
| `"Time" could be any of "Time", "time"; write the name exactly` | two columns differ only by case or spacing |
| `/figure/format: one of svg, pdf, png` | `figure` needs a `format` |

`sp.validate(description, data=df)` returns the same problems as a list, without
drawing anything.

## The charts are blank, or never appear

- **`sp.view` printed `HTMLView` or a plain repr instead of charts.** The frontend does
  not run widgets. Install the `notebook` extra in the *kernel's* environment and restart
  the kernel; in classic Jupyter Notebook 6 and older, widgets also need
  that frontend's widget extension installed.
- **A chart shows axes but no points.** All rows may be filtered out by another chart's
  brush: click that chart without dragging to clear it.
- **A reopened notebook shows nothing.** Widget output needs the kernel; run the cell
  again, or see [Saving a notebook with its charts](guide/notebooks.md#saving-a-notebook-with-its-charts).

## `import spectacular` imports something else

PyPI's `spectacular` is a different project. If it is installed, it shadows this one:

```bash
pip uninstall spectacular
pip install --force-reinstall spectacular-charts
```

`sp.__file__` shows which package was imported.

## `render` fails

- **`render draws figures in headless Chromium: pip install ...`**: install the `render`
  extra and the browser: `pip install "spectacular-charts[render]"`, then
  `python -m playwright install chromium`.
- **`Executable doesn't exist at ...`** (from Playwright): the browser was not installed
  for this environment; run `python -m playwright install chromium` with the same Python.
- **`NotImplementedError` in a notebook on Windows** (0.2.0 only): ipykernel 6 switches
  asyncio to the selector event loop, which cannot start the browser. Upgrade with
  `pip install -U spectacular-charts`; from 0.2.1, `render` uses an event loop of its own.
- **Slow**: each figure starts a browser, a few seconds each. Draw many charts into one
  figure with `charts` rather than many figures.

## `TypeError: dates and durations have no axis type yet`

Convert the column to numbers first, such as hours since the start; see
[Data](guide/data.md#numbers-and-categories).

## The native command is not found

`spectacular` is installed into the environment's scripts folder (bin on Linux and
macOS, Scripts on Windows). Activate the environment, or call it by its full path. The Python package
looks for it in this order:

1. the `SPECTACULAR_BIN` environment variable;
2. the scripts folder of the running Python;
3. the user scripts folder;
4. next to the Python executable;
5. a `target/` build in a checkout;
6. `PATH`.

## Selections stay empty

- `selection` is filled by the page after you brush. Read it in a *later* cell than the
  one that showed the view.
- `selected_rows` is empty with `track_rows=False`.
- `selected_keys` needs a `key` in the description.

## Still stuck

Open an issue at
[github.com/fullscreen-triangle/spectacular/issues](https://github.com/fullscreen-triangle/spectacular/issues)
with the description, `sp.__version__`, and the output of `sp.validate(description, data=df)`.
