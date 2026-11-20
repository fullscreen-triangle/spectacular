"""Linked views in a notebook cell, with the selection readable from Python."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from ._data import payload
from ._html import iframe, to_html
from ._spec import SpecLike, static_file

try:
    import anywidget
    import traitlets
except ImportError:  # pragma: no cover - exercised only without anywidget
    anywidget = None

try:
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None


class HTMLView:
    """The views as a self-contained page in an iframe: what a notebook gets without
    anywidget. Brushing works; the selection stays in the page."""

    def __init__(self, spec: SpecLike, data: Any = None, height: int = 480):
        self.document = to_html(spec, data)
        self.height = height

    def _repr_html_(self) -> str:
        return iframe(self.document, self.height)


if anywidget is not None:

    class View(anywidget.AnyWidget):
        """Linked views of one table, drawn in the notebook.

        Brushing in the views updates `selection` (the filters, in data terms) and
        `selected_rows` (row indices into the data) here in Python.
        """

        _esm = static_file("spectacular.js")

        spec = traitlets.Dict().tag(sync=True)
        columns = traitlets.Dict(allow_none=True, default_value=None).tag(sync=True)
        file = traitlets.Dict(allow_none=True, default_value=None).tag(sync=True)
        track_rows = traitlets.Bool(True).tag(sync=True)
        #: {"selected": n, "rows": n, "filters": [{"view", "field", "range" | "values"}]}
        selection = traitlets.Dict().tag(sync=True)
        #: The spec's problems as the page found them, each {"path", "message"}.
        problems = traitlets.List().tag(sync=True)
        # Binary state arrives from the page as a memoryview; Bytes takes only bytes.
        _rows = traitlets.Bytes(b"").tag(sync=True, from_json=lambda value, _widget: bytes(value or b""))

        def __init__(self, spec: SpecLike, data: Any = None, *, track_rows: bool = True, **kwargs: Any):
            s, columns, file = payload(spec, data)
            super().__init__(spec=s, columns=columns, file=file, track_rows=track_rows, **kwargs)
            self._data = data
            self._svg: Dict[str, Path] = {}
            self.saved: List[Path] = []
            self.on_msg(self._message)

        def update(self, spec: Optional[SpecLike] = None, data: Any = None) -> None:
            """Redraws with a new spec, new data, or both. Brushes are cleared."""
            s, columns, file = payload(spec if spec is not None else self.spec, data if data is not None else self._data)
            if data is not None:
                self._data = data
            with self.hold_sync():
                self.spec, self.columns, self.file = s, columns, file

        @property
        def selected_rows(self) -> Any:
            """Indices (0-based, in data order) of the rows passing every brush: a numpy
            array when numpy is installed, else a list. Empty until the view is drawn."""
            if _np is not None:
                return _np.frombuffer(self._rows, dtype="<u4").copy()
            import array

            out = array.array("I" if array.array("I").itemsize == 4 else "L")
            out.frombytes(self._rows)
            return out.tolist()

        def selected_data(self) -> Any:
            """The selected rows of the data this view was given, in its own type."""
            data, rows = self._data, self.selected_rows
            if data is None:
                raise ValueError("selected_data needs the table passed as data=; with a data.url, use selected_rows")
            if hasattr(data, "iloc"):
                return data.iloc[rows]
            if hasattr(data, "get_column"):
                return data[list(rows)]
            if isinstance(data, dict):
                return {k: [v[i] for i in rows] for k, v in data.items()}
            if _np is not None and isinstance(data, _np.ndarray):
                return data[rows]
            return [data[i] for i in rows]

        def on_select(self, callback: Callable[[Dict[str, Any]], None]) -> Callable[[Any], None]:
            """Calls `callback(selection)` after each brush settles. Returns the handler,
            which `unobserve(handler, names="selection")` removes."""
            def handler(change: Any) -> None:
                callback(change["new"])
            self.observe(handler, names="selection")
            return handler

        def save_svg(self, path: Union[str, Path]) -> None:
            """Saves what the view shows now, brushes included, as an SVG file.

            The page draws it, so the view must be displayed. The file is written when the
            page answers, after this cell finishes; `saved` lists the files written.
            """
            key = uuid.uuid4().hex
            self._svg[key] = Path(path)
            self.send({"type": "svg", "id": key})

        def _message(self, _widget: Any, content: Dict[str, Any], _buffers: Any) -> None:
            if content.get("type") != "svg":
                return
            path = self._svg.pop(content.get("id", ""), None)
            if path is not None and content.get("svg"):
                path.write_text(content["svg"], encoding="utf-8")
                self.saved.append(path)

else:  # pragma: no cover
    View = None  # type: ignore[assignment,misc]


def view(spec: SpecLike, data: Any = None, **kwargs: Any) -> Any:
    """Linked views to display in a notebook (as a cell's last expression, or with display()).

    With anywidget installed this is a `View`, whose selection Python can read. Without it,
    an `HTMLView`: the same views in an iframe, with nothing reported back.
    """
    if View is not None:
        return View(spec, data, **kwargs)
    return HTMLView(spec, data, **{k: v for k, v in kwargs.items() if k == "height"})


def show(spec: SpecLike, data: Any = None, **kwargs: Any) -> Any:
    """Displays the views now and returns them (see `view`)."""
    from IPython.display import display

    v = view(spec, data, **kwargs)
    display(v)
    return v
