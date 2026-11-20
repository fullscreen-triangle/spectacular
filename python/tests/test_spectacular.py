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

    with pytest.raises(FileNotFoundError, match="nope.csv"):
        payload({"data": {"url": "nope.csv"}, "views": []})
    assert payload({"data": {"url": "https://example.org/x.csv"}, "views": []})[2] is None


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


needs_node = pytest.mark.skipif(shutil.which("node") is None or not (REPO / "packages/viz/dist/index.html").exists(),
                                reason="needs Node and the built viewer")


@needs_node
def test_validate_reports_problems_against_the_data():
    problems = sp.validate(spec_for({"id": "h", "type": "histogram", "field": "k"}), {"k": ["a", "b"]})
    assert problems == [{"path": "/views/0/field", "message": 'histogram.field needs a quantitative field; "k" is nominal'}]
    assert sp.validate(LORENZ) == []


@needs_node
def test_render_draws_a_figure_from_a_dataframe(tmp_path):
    df = pd.DataFrame({"well": [1, 2, 3, 1], "od": [0.1, 0.4, 0.9, 0.2]})
    out = sp.render(spec_for({"id": "s", "type": "scatter", "x": "od", "y": "od", "color": "well"},
                             fields={"well": {"type": "nominal"}}), tmp_path / "fig.svg", df)
    svg = out.read_text()
    assert svg.count("<circle ") == 4 and ">well</span>" not in svg and "well" in svg
    with pytest.raises(sp.SpectacularError) as err:
        sp.render(spec_for({"id": "h", "type": "histogram", "field": "nope"}), tmp_path / "x.png", df)
    assert err.value.exit == 1 and err.value.problems[0]["path"] == "/views/0/field"
