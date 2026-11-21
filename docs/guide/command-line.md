# Command line

The wheel installs a `spectacular` command, compiled from Rust. It does what the Python
package does with descriptions, with no Python code and no Node, so a shell script, a
Makefile or a program in any language can use it.

```bash
spectacular compile chart.json --data readings.csv     # a description -> a view spec
spectacular validate spec.json                         # check a spec, and its fields against its data
spectacular html chart.json --data readings.csv -o readings.html   # one offline page
spectacular schema > spectacular.schema.json           # the JSON Schema for view specs
spectacular --version
```

| Exit code | Means |
|---|---|
| 0 | done |
| 1 | the description, spec or data has problems |
| 2 | usage or I/O error (a missing file, an unknown option) |

`-` in place of a file reads standard input; relative paths then resolve against the
working directory.

## compile

Given a description in a file named chart.json:

```json
{"title": "Plate", "charts": [{"x": "time_h", "y": "od600", "color": "medium"}, {"x": "medium"}]}
```

`spectacular compile chart.json --data readings.csv` prints the view spec:

```json
{
  "version": "spectacular/v0",
  "title": "Plate",
  "data": { "url": "readings.csv" },
  "views": [
    { "id": "od600-vs-time-h", "type": "scatter", "title": "od600 vs time_h", "x": "time_h", "y": "od600", "color": "medium" },
    { "id": "medium", "type": "bars", "title": "medium", "field": "medium" }
  ]
}
```

With `--data`, column names are checked and matched against the file's columns. Without
it, the description's own `data` file is used if it names one.

## Problems, for people and for programs

```console
$ spectacular compile bad.json --data readings.csv
bad.json /x: no column "od60"; did you mean "od600"?
$ echo $?
1
```

`--json` prints one JSON object per line on standard output instead:

```json
{"ok":false,"problems":[{"path":"/x","message":"no column \"od60\"; did you mean \"od600\"?"}]}
```

```json
{"spec":"spec.json","ok":true,"problems":[],"rows":3,"fields":{"time_h":"quantitative","od600":"quantitative","medium":"nominal"}}
```

The second line is `validate --json` on a spec that holds: it also reports the row count
and the type of each field it read.

## Driving it from another program

The command reads files or standard input and writes JSON, so any language can call it.
From Python's standard library, for example:

<!-- no-test -->
```python
import json, subprocess

description = {"x": "time_h", "y": "od600"}
result = subprocess.run(["spectacular", "compile", "-", "--json"],
                        input=json.dumps(description), capture_output=True, text=True)
```

`compile --envelope` reads `{"description", "columns": {name: type}, "rows", "base"}` on
standard input, for a caller that holds the table itself and knows its column types.
The Python package calls the command that way.

## The Node command

The repository also has a Node command, `cli/spectacular.mjs`, which adds what needs a
browser:

| Command | Does |
|---|---|
| `render spec.json -o fig.pdf` | draws SVG, PDF or PNG at physical size (`--width 180 --unit mm`) |
| `serve spec.json --json --indices` | shows the charts on 127.0.0.1 and prints each selection as a JSON line |

It runs from a checkout of the repository and reads view specs (which `compile` writes).
`serve` reloads the page when the spec or the data file changes, so a running experiment
can append readings and the charts follow. Its URL carries an access token: do not
share it. The repository's
[`spectacular-specifications.md`](https://github.com/fullscreen-triangle/spectacular/blob/rebuild/spectacular-specifications.md)
documents both commands in full.
