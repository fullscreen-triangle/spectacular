# Descriptions and specs

spectacular has two languages for the same charts:

| | Description | View spec |
|---|---|---|
| Written by | people, notebooks, models | programs, build steps, other tools |
| Keys | forgiving: `x-axis`, `colour`, `hue` | exact: `x`, `color`, `field` |
| Chart type | inferred from the columns | always stated |
| Columns | matched loosely | named exactly |
| Defined by | the compiler in the `spectacular` command | `packages/viz/src/spec.ts`, and its JSON Schema |
| Problems at | the description's own keys | JSON paths into the spec |

A description is compiled into a spec before anything is drawn. Every function that takes
a description also takes a spec, and a spec passes through the compiler unchanged, apart
from being validated.

## Seeing the spec of a description

```console
$ spectacular compile chart.json --data readings.csv
```

prints the spec (see [Command line](command-line.md#compile)). Compiling once and keeping
the spec is useful when a description has settled and you want the exact form under
version control, or want to hand it to the Node command.

## When to write a spec

Write a spec when a program produces the charts and should state exactly what it means.
Examples are a pipeline that emits a figure for every run, a tool in another language,
or a model whose output you validate against a schema. Write a description when a person
writes it, or when the columns' types should decide the charts.

A minimal spec:

```python
import spectacular as sp

spec = {
    "version": "spectacular/v0",
    "data": {"columns": "external"},
    "views": [
        {"id": "growth", "type": "scatter", "x": "time_h", "y": "od600", "color": "medium"},
        {"id": "rate", "type": "histogram", "field": "rate", "bins": 40},
    ],
}
sp.view(spec, data=sp.datasets.plate())
```

`{"columns": "external"}` means "the table passed in". With a file it is
`{"url": "readings.csv"}`. Passing `data=` always replaces the spec's own data.

## The schema

`sp.schema()` (or `spectacular schema`) returns the JSON Schema for specs. Give it to a
model's structured output, to an editor for completion, or to a validator in another
language:

```python
schema = sp.schema()
sorted(schema["properties"])
```

The schema catches the shape of a spec. `sp.validate(spec, data=df)` also checks what a
schema cannot: that the fields exist in the data, that a histogram's field is a number,
and that a palette has at most 32 colours.

## The full reference

The spec format, every view's properties, the data sources (CSV, TSV, JSON, JSON for Linked Data (JSON-LD) with
a JSON Pointer, inline values) and the limits are documented in
[View specs](../reference/spec.md).
