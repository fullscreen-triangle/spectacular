"""`%%spectacular`: a notebook cell that is nothing but a chart description.

    %%spectacular df
    {"title": "Growth plate",
     "charts": [
       {"x-axis": "time_h", "y-axis": "od600", "color": "medium"},
       {"x": "rate", "color": "teal"},
       {"x": "medium"}
     ]}

The line names the data (any expression: `df`, `runs[3]`, `pd.read_csv("x.csv")`), or
nothing when the description names its own data file. `--as w` keeps the views in `w`.
The cell is JSON; Python's own literal syntax (single quotes, True, trailing commas) is
accepted too.
"""

from __future__ import annotations

import ast
import json
import shlex
from typing import Any, Optional


def parse_description(text: str) -> Any:
    """The cell as JSON, or else as a Python literal; a JSON error is reported if neither."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as first:
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            raise ValueError(f"the cell is not a JSON description: {first.msg} at line {first.lineno}, column {first.colno}") from None


def spectacular_cell(line: str, cell: str, local_ns: Optional[dict] = None) -> Any:
    from IPython import get_ipython

    from ._view import view

    args = shlex.split(line, posix=True)
    name = None
    if "--as" in args:
        i = args.index("--as")
        if i + 1 >= len(args):
            raise ValueError("--as needs a variable name")
        name = args[i + 1]
        del args[i : i + 2]
    shell = get_ipython()
    data = shell.ev(" ".join(args)) if args else None
    views = view(parse_description(cell), data)
    if name:
        shell.user_ns[name] = views
    return views


def load_ipython_extension(shell: Any) -> None:
    """`%load_ext spectacular` (also done by `import spectacular` inside IPython)."""
    shell.register_magic_function(spectacular_cell, magic_kind="cell", magic_name="spectacular")
