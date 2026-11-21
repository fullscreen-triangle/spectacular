# Figures and pages

Three ways to keep charts outside a running notebook:

| Function | Writes | Interactive | Needs |
|---|---|---|---|
| `sp.render(desc, "fig.pdf", data=df)` | SVG, PDF or PNG, at a size you choose | no | the `render` extra |
| `w.save_svg("now.svg")` | SVG of a displayed view, brushes included | no | a displayed `View` |
| `sp.save_html(desc, "page.html", data=df)` | one HTML file with charts, runtime and data | yes | nothing |

## Figures with `render`

```python
import spectacular as sp

df = sp.datasets.plate()
sp.render({"theme": "paper", "x": "time_h", "y": "od600", "color": "medium"},
          "growth.pdf", data=df, width=85, unit="mm")
```

The format follows the file's extension: `.svg`, `.pdf` or `.png`. `render` returns the
path it wrote.

### Size

Figures are sized for the page they go on, not the screen:

| Argument | Means | Default |
|---|---|---|
| `width` | the figure's width | 960 |
| `unit` | `"mm"`, `"in"` or `"px"` | `"px"` |
| `dpi` | pixels per inch for PNG | 300 for mm and in, 192 for px |

The height follows from the charts. A one-column journal figure is about 85 mm wide, a
two-column one about 180 mm. The size can also live in the description, so that the
description alone says how its figure looks:

```python
description = {
    "theme": "paper",
    "figure": {"format": "pdf", "width": 180, "unit": "mm"},
    "columns": 2,
    "charts": [{"x": "time_h", "y": "od600", "color": "medium"}, {"x": "rate"}],
}
sp.render(description, "figure-2.pdf", data=df)
```

`figure` needs a `format` (`"svg"`, `"pdf"` or `"png"`), which `spectacular render` uses;
`sp.render` writes the format of the file name it is given. Arguments passed to
`render` take precedence over `figure`.

### What `render` draws

`render` draws the charts with no brushes, in a headless Chromium through Playwright.
After `pip install "spectacular-charts[render]"`, install the browser once with
`python -m playwright install chromium`. It takes a few seconds per figure, most of it
starting the browser.

SVG keeps text as text, and a scatter of up to about 50,000 points as one circle each, so
it can be edited in Inkscape or Illustrator. For LaTeX with pdflatex, use PDF.

!!! note "Theme for print"
    Use `"theme": "paper"` for manuscripts: white, no panel borders, darker axes. The
    other themes draw the panels as they look on screen.

## What you see: `save_svg`

A displayed view can save itself as it is now, brushes included:

```python
w = sp.view({"charts": [{"x": "time_h", "y": "od600"}, {"x": "rate"}]}, data=df)
w
```

<!-- no-test -->
```python
w.save_svg("selected-wells.svg")   # after brushing
```

The page draws the SVG, so the view must be on screen. The file is written when the
page answers, just after the cell finishes; `w.saved` lists the files written so far.

## Offline pages: `save_html` and `to_html`

```python
sp.save_html({"title": "Growth plate", "charts": [
    {"x": "time_h", "y": "od600", "color": "medium"}, {"x": "rate"}, {"x": "medium"},
]}, "plate.html", data=df)
```

The page holds the runtime (about 1 MB), the description and the columns the charts use.
It opens in any modern browser with no server and no network, so it can be emailed,
attached to a report or put on a static site. Brushing works; nothing is reported back.

`sp.to_html(...)` returns the same page as a string, for a web framework or a report
generator. `title=` sets the browser tab's title.

Very large tables make large pages: a million rows of four numbers is 32 MB of data,
and more once written into the page as text. For those, publish a figure, or keep the data in a file next to a spec and use
`spectacular serve` (see [Command line](command-line.md#the-node-command)).
