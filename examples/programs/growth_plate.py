"""A lab program driving spectacular, with the Python standard library only.

It simulates optical-density readings of a 96-well growth plate, writes them as CSV next
to a view spec, validates the spec, renders figures, and with --serve opens the linked
views and prints which wells a person brushes:

    python examples/programs/growth_plate.py OUT_DIR [--serve]

Every call goes through the `spectacular` CLI and reads its JSON lines and exit codes
(0 ok, 1 the spec or its data has problems, 2 usage or I/O, 3 rendering failed). Set
SPECTACULAR to the CLI's path when this file is copied out of the repo.
"""

import csv
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

CLI = os.environ.get("SPECTACULAR", str(Path(__file__).resolve().parents[2] / "cli" / "spectacular.mjs"))


def spectacular(*args, stdin=None):
    """Runs one CLI command with --json; returns (exit code, parsed output lines)."""
    proc = subprocess.run(["node", CLI, *args, "--json"], input=stdin, capture_output=True, text=True)
    return proc.returncode, [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def readings(seed=7):
    """Logistic growth per well; one reading every 30 min for 12 h."""
    rng = random.Random(seed)
    for row in "ABCDEFGH":
        for col in range(1, 13):
            medium = "rich" if col <= 6 else "minimal"
            rate = rng.gauss(0.55 if medium == "rich" else 0.35, 0.06)
            lag = rng.uniform(1.0, 3.0)
            for k in range(25):
                t = k * 0.5
                od = 0.05 + 1.1 / (1 + math.exp(-rate * 2 * (t - lag - 3)))
                yield {"well": f"{row}{col}", "row": row, "column": col, "medium": medium,
                       "time_h": t, "od600": round(od + rng.gauss(0, 0.01), 4), "rate": round(rate, 4)}


SPEC = {
    "version": "spectacular/v0",
    "title": "Growth plate: OD600 over 12 h",
    "data": {"url": "readings.csv"},
    "fields": {
        "time_h": {"title": "time (h)"},
        "od600": {"title": "OD600"},
        "rate": {"title": "growth rate (1/h)"},
        "column": {"type": "quantitative", "title": "plate column"},
    },
    "layout": {"columns": 2},
    "views": [
        {"id": "curves", "type": "scatter", "x": "time_h", "y": "od600", "color": "medium", "span": 2},
        {"id": "rate", "type": "histogram", "field": "rate", "bins": 40},
        {"id": "medium", "type": "bars", "field": "medium"},
    ],
    "output": {"figure": {"format": "pdf", "width": 180, "unit": "mm"}},
}


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rows = list(readings())
    with open(out / "readings.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    spec_file = out / "growth.spec.json"
    spec_file.write_text(json.dumps(SPEC, indent=2))

    code, [result] = spectacular("validate", str(spec_file))
    if code != 0:
        for p in result.get("problems", []):
            print(f"spec problem at {p['path']}: {p['message']}")
        sys.exit(code)
    print(f"valid: {result['rows']} rows")

    # The size comes from output.figure; flags would override it.
    for name in ("growth.svg", "growth.pdf", "growth.png"):
        code, [result] = spectacular("render", str(spec_file), "-o", str(out / name))
        if code != 0:
            sys.exit(f"render failed ({code}): {result['error']}")
        print(f"wrote {result['file']} ({result['width']}x{result['height']} CSS px)")

    if "--serve" in sys.argv:
        serve(spec_file, rows)


def serve(spec_file, rows):
    """Prints the wells behind each selection until the viewer is closed with ctrl-c."""
    proc = subprocess.Popen(["node", CLI, "serve", str(spec_file), "--json", "--indices", "--open"],
                            stdout=subprocess.PIPE, text=True)
    try:
        for line in proc.stdout:
            event = json.loads(line)
            if event["event"] == "listening":
                print(f"brush the views at {event['url']}")
            elif event["event"] == "selection":
                wells = sorted({rows[i]["well"] for i in event["indices"]})
                print(f"{event['selected']} readings from {len(wells)} wells: {', '.join(wells[:12])}"
                      + (" ..." if len(wells) > 12 else ""))
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
