"""The Python side: specs, data on the wire, pages, the widget's Python half, and the command."""

import array
import json
import math
import shutil
from pathlib import Path

import pytest

import spectacular as sp
from spectacular._data import encode, encode_column, payload

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

REPO = Path(__file__).resolve().parents[2]
LORENZ = REPO / "examples" / "lorenz.spec.json"


def floats(wire):
    return array.array("d", wire["values"]).tolist()


def codes(wire):
    return array.array("I", wire["codes"]).tolist()


def spec_for(*views, **extra):
    return {"views": list(views), **extra}


def test_fields_come_from_the_schema_for_every_view_type():
    spec = spec_for(
        {"id": "s", "type": "scatter", "x": "a", "y": "b", "color": "c"},
        {"id": "e", "type": "sankey", "source": "from", "target": "to"},
        fields={"declared": {"type": "nominal"}, "a": {}},
    )
    assert sp.fields_used(spec) == ["a", "b", "c", "from", "to", "declared"]
    assert "theme" in sp.schema()["properties"]


def test_numbers_travel_as_float64_and_the_rest_as_categories():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [0.5, None, 2.5], "k": ["a", "b", "a"], "ok": [True, False, True]})
    wire = encode(df, spec_for({"id": "s", "type": "scatter", "x": "x", "y": "y", "color": "k"}, {"id": "b", "type": "bars", "field": "ok"}))
    assert floats(wire["x"]) == [1.0, 2.0, 3.0]
    assert floats(wire["y"])[0] == 0.5 and math.isnan(floats(wire["y"])[1])
    assert wire["k"]["labels"] == ["a", "b"] and codes(wire["k"]) == [0, 1, 0]
    assert wire["ok"]["labels"] == ["true", "false"]


def test_only_the_fields_the_views_read_are_sent():
    df = pd.DataFrame({"x": [1.0], "unused": ["big text"]})
    assert list(encode(df, spec_for({"id": "h", "type": "histogram", "field": "x"}))) == ["x"]


def test_declared_types_win_over_the_data():
    wire = encode({"well": [1, 2, 1], "od": ["0.1", "bad", None]}, spec_for(
        {"id": "b", "type": "bars", "field": "well"}, {"id": "h", "type": "histogram", "field": "od"},
        fields={"well": {"type": "nominal"}, "od": {"type": "quantitative"}},
    ))
    assert wire["well"]["type"] == "nominal" and wire["well"]["labels"] == ["1", "2"]
    od = floats(wire["od"])
    assert od[0] == 0.1 and math.isnan(od[1]) and math.isnan(od[2])


def test_rows_dicts_and_record_arrays_are_tables_too():
    spec = spec_for({"id": "h", "type": "histogram", "field": "v"})
    assert floats(encode([{"v": 1}, {"v": 2.5}], spec)["v"]) == [1.0, 2.5]
    rec = np.array([(1.0,), (4.0,)], dtype=[("v", "f8")])
    assert floats(encode(rec, spec)["v"]) == [1.0, 4.0]


def test_pandas_missing_values_in_object_columns():
    s = pd.Series([1, pd.NA, 3], dtype="Int64")
    assert encode_column(s.astype(object))["type"] == "quantitative"


def test_dates_are_refused_with_a_way_forward():
    with pytest.raises(TypeError, match="total_seconds"):
        encode_column(pd.Series(pd.to_datetime(["2026-01-01"])))


def test_data_replaces_the_spec_data_and_a_local_url_travels_as_bytes(tmp_path):
    spec, columns, file = payload(spec_for({"id": "h", "type": "histogram", "field": "v"}, data={"url": "x.csv"}), {"v": [1.0]})
    assert spec["data"] == {"columns": "external"} and file is None and "v" in columns

    (tmp_path / "d.csv").write_bytes(b"v\n1\n2\n")
    (tmp_path / "s.json").write_text(json.dumps(spec_for({"id": "h", "type": "histogram", "field": "v"}, data={"url": "d.csv"})))
    spec, columns, file = payload(tmp_path / "s.json")
    assert columns is None and file == {"name": "d.csv", "bytes": b"v\n1\n2\n"}

    with pytest.raises(sp.SpectacularError, match="nope.csv"):
        payload({"data": "nope.csv", "x": "v"})
    remote = payload(spec_for({"id": "h", "type": "histogram", "field": "v"}, data={"url": "https://example.org/x.csv"}))
    assert remote[1] is None and remote[2] is None


def test_a_page_embeds_the_runtime_and_survives_script_like_labels():
    html = sp.to_html(spec_for({"id": "b", "type": "bars", "field": "k"}), {"k": ["</script><b>", "x"]})
    assert html.startswith("<!doctype html>") and "spectacularNotebook.boot" in html
    payload_text = html.split('id="spectacular-payload">', 1)[1].split("</script>", 1)[0]
    assert json.loads(payload_text)["columns"]["k"]["labels"][0] == "</script><b>"


def test_the_widget_reads_selections_and_slices_its_data():
    df = pd.DataFrame({"v": [5.0, 6.0, 7.0, 8.0]})
    w = sp.View(spec_for({"id": "h", "type": "histogram", "field": "v"}), df)
    assert w.spec["data"] == {"columns": "external"} and w.selected_rows.tolist() == []
    w._rows = np.array([1, 3], dtype="<u4").tobytes()
    assert w.selected_rows.tolist() == [1, 3]
    assert w.selected_data()["v"].tolist() == [6.0, 8.0]
    seen = []
    w.on_select(seen.append)
    w.selection = {"selected": 2, "rows": 4, "filters": []}
    assert seen == [{"selected": 2, "rows": 4, "filters": []}]


def test_state_from_the_page_is_accepted_as_the_kernel_receives_it():
    # ipywidgets hands binary buffers to Python as memoryviews.
    w = sp.View(spec_for({"id": "h", "type": "histogram", "field": "v"}), {"v": [1.0, 2.0]})
    w.set_state({"selection": {"selected": 1, "rows": 2, "filters": []}, "_rows": memoryview(b"\x01\x00\x00\x00")})
    assert w.selection["selected"] == 1 and w.selected_rows.tolist() == [1]


def test_save_svg_writes_the_file_when_the_page_answers(tmp_path, monkeypatch):
    w = sp.View(spec_for({"id": "h", "type": "histogram", "field": "v"}), {"v": [1.0]})
    sent = []
    monkeypatch.setattr(w, "send", sent.append)
    w.save_svg(tmp_path / "fig.svg")
    w._message(w, {"type": "svg", "id": sent[0]["id"], "svg": "<svg/>"}, [])
    assert (tmp_path / "fig.svg").read_text() == "<svg/>" and w.saved == [tmp_path / "fig.svg"]


PLATE = pd.DataFrame({
    "Time (h)": [0.0, 0.5, 1.0, 1.5],
    "od600": [0.05, 0.1, 0.2, 0.4],
    "medium": ["rich", "minimal", "rich", "minimal"],
    "well": [1, 2, 1, 2],
})


def test_a_description_becomes_linked_views():
    w = sp.view({
        "title": "Plate",
        "categorical": ["well"],
        "labels": {"od600": "OD600"},
        "charts": [
            {"X axis": "time h", "y-axis": "OD600", "Colour": "medium", "palette": ["teal", "orange"]},
            {"x": "od600", "color": "tomato"},
            {"x": "well"},
        ],
    }, PLATE)
    views = w.spec["views"]
    assert [v["type"] for v in views] == ["scatter", "histogram", "bars"]
    assert views[0]["x"] == "Time (h)" and views[0]["color"] == "medium" and views[0]["palette"] == ["teal", "orange"]
    assert views[1]["fill"] == "tomato" and views[1]["title"] == "OD600"
    assert w.spec["fields"]["well"]["type"] == "nominal" and w.columns["well"]["type"] == "nominal"
    assert w.problems == [] and sp.chart is sp.view


def test_a_list_of_charts_is_a_description_too():
    w = sp.view([{"x": "od600"}, {"x": "medium"}], PLATE)
    assert [v["type"] for v in w.spec["views"]] == ["histogram", "bars"]


def test_description_problems_show_where_the_charts_would_be():
    w = sp.view({"charts": [{"x": "od60"}, {"x": "medium", "color": "od600"}, {"x": "od600", "colr": "red"}]}, PLATE)
    assert w.spec == {} and w.columns is None
    assert [p["path"] for p in w.problems] == ["/charts/0/x", "/charts/1/color", "/charts/2/colr"]
    assert w.problems[0]["message"] == 'no column "od60"; did you mean "od600"?'
    w.update(spec={"x": "od600"})
    assert w.problems == [] and w.spec["views"][0]["type"] == "histogram"


BUNDLE = REPO / "examples" / "bundle.desc.json"


def test_the_bundle_description_compiles_to_the_newer_views():
    rows = json.loads((REPO / "examples" / "bundle.rows.json").read_text())
    w = sp.view(BUNDLE, rows)
    assert w.problems == []
    types = [v["type"] for v in w.spec["views"]]
    assert types == ["bars", "sankey", "strip", "matrix", "interval", "scatter", "bars"]
    assert w.spec["key"] == "row" and w.spec["views"][0]["color"] == "outcome"
    assert w.spec["fields"]["way"]["domain"][:3] == ["stated", "supplied", "in-place"]
    assert w.spec["fields"]["outcome"]["palette"]["published"] == "#1baf7a"
    # The key travels as a column, so selections can be named by it.
    assert "row" in w.columns


def test_selecting_by_key_goes_to_the_page_and_keys_come_back(monkeypatch):
    w = sp.View({"key": "id", "x": "v"}, {"id": ["a", "b", "c"], "v": [1.0, 2.0, 3.0]})
    assert set(w.columns) == {"id", "v"}  # the key column travels though no chart draws it
    sent = []
    monkeypatch.setattr(w, "send", sent.append)
    w.select(["a", "c"])
    w.select(None)
    assert [m["keys"] for m in sent] == [["a", "c"], None] and all(m["type"] == "select" for m in sent)
    w.set_state({"_keys": ["a", "c"]})
    assert w.selected_keys == ["a", "c"]
    with pytest.raises(ValueError, match="key"):
        sp.View({"x": "v"}, {"v": [1.0]}).select(["a"])


def test_the_cell_magic_takes_only_a_description():
    from IPython.core.interactiveshell import InteractiveShell

    shell = InteractiveShell.instance()
    sp.load_ipython_extension(shell)
    shell.user_ns["plate"] = PLATE
    w = shell.run_cell_magic("spectacular", "plate --as views", '{"x-axis": "time h", "y-axis": "od600"}')
    assert shell.user_ns["views"] is w and w.spec["views"][0]["type"] == "scatter"
    # Python's literal syntax works too, and the data can be any expression.
    w = shell.run_cell_magic("spectacular", "plate[plate.well == 1]", "{'x': 'od600', 'color': 'teal',}")
    assert w.spec["views"][0]["fill"] == "teal" and len(w._data) == 2
    with pytest.raises(ValueError, match="not a JSON description"):
        shell.run_cell_magic("spectacular", "plate", '{"x": }')


def test_validate_reports_problems_against_the_data():
    problems = sp.validate(spec_for({"id": "h", "type": "histogram", "field": "k"}), {"k": ["a", "b"]})
    assert problems == [{"path": "/views/0/field", "message": 'histogram.field needs a quantitative field; "k" is nominal'}]
    assert sp.validate(LORENZ) == []


def test_validate_speaks_in_the_description_s_own_paths():
    assert sp.validate({"charts": [{"x-axis": "nope"}]}, PLATE) == [
        {"path": "/charts/0/x-axis", "message": 'no column "nope"; the data has Time (h), od600, medium, well'}
    ]


def _can_render():
    try:
        import playwright.sync_api  # noqa: F401
        return True
    except ImportError:
        return shutil.which("node") is not None and (REPO / "packages/viz/dist/index.html").exists()


@pytest.mark.skipif(not _can_render(), reason="needs Playwright for Python, or Node and the built viewer")
def test_render_draws_a_figure_from_a_dataframe(tmp_path):
    df = pd.DataFrame({"well": [1, 2, 3, 1], "od": [0.1, 0.4, 0.9, 0.2]})
    out = sp.render(spec_for({"id": "s", "type": "scatter", "x": "od", "y": "od", "color": "well"},
                             fields={"well": {"type": "nominal"}}), tmp_path / "fig.svg", df)
    svg = out.read_text()
    assert svg.count("<circle ") == 4 and ">well</span>" not in svg and "well" in svg
    with pytest.raises(sp.SpectacularError) as err:
        sp.render(spec_for({"id": "h", "type": "histogram", "field": "nope"}), tmp_path / "x.png", df)
    assert err.value.exit == 1 and err.value.problems[0]["path"] == "/views/0/field"


@pytest.mark.skipif(not _can_render(), reason="needs Playwright for Python, or Node and the built viewer")
def test_render_works_where_an_event_loop_runs(tmp_path):
    # A Jupyter kernel runs an asyncio loop; rendering must not depend on there being none.
    import asyncio

    async def in_a_kernel():
        return sp.render({"x": "od600"}, tmp_path / "loop.svg", PLATE)

    assert asyncio.run(in_a_kernel()).read_text().startswith("<?xml")
