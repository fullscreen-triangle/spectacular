import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { loadTable, fieldTypes } from "../src/data";
import { plan } from "../src/mount";
import { validate, type Spec } from "../src/spec";

const example = JSON.parse(readFileSync(new URL("../../../examples/synthetic.spec.json", import.meta.url), "utf8")) as Spec;

const paths = (spec: unknown, types?: Record<string, "quantitative" | "nominal">) =>
  validate(spec, types).map((p) => p.path);

describe("validate", () => {
  it("accepts the shipped example", () => {
    expect(validate(example)).toEqual([]);
  });

  it("points at each problem by path", () => {
    const bad = {
      version: "spectacular/v9",
      data: { url: "a.csv", values: [] },
      views: [
        { id: "a", type: "scatter", x: "x" },
        { id: "a", type: "pie", field: "k" },
        { id: "c", type: "histogram", field: "x", bins: 0, brush: "xy" },
        { id: "d", type: "density", x: "x", y: "y", bins: [10] },
      ],
      output: { figure: { format: "gif" } },
    };
    expect(paths(bad)).toEqual([
      "/version", "/data", "/views/0/y", "/views/1/id", "/views/1/type",
      "/views/2/brush", "/views/2/bins", "/views/3/bins", "/output/figure/format", "/output/figure/width",
    ]);
  });

  it("checks fields against the loaded data", () => {
    const spec = { data: { values: [] }, views: [{ id: "h", type: "histogram", field: "species" }, { id: "b", type: "bars", field: "nope" }] };
    const problems = validate(spec, { species: "nominal" });
    expect(problems.map((p) => p.path)).toEqual(["/views/0/field", "/views/1/field"]);
    expect(problems[0]!.message).toMatch(/needs a quantitative field; "species" is nominal/);
  });

  it("reports unknown properties, suggesting the near miss", () => {
    const spec = {
      data: { values: [] }, colums: 2, layout: { colums: 2 },
      views: [{ id: "s", type: "scatter", x: "a", y: "b", colour: "k" }, { id: "d", type: "density", x: "a", y: "b", brush: "xy" }],
    };
    const problems = validate(spec);
    expect(problems.map((p) => p.path)).toEqual(["/colums", "/views/0/colour", "/views/1/brush", "/layout/colums"]);
    expect(problems[1]!.message).toBe('unknown property "colour"; did you mean "color"?');
  });

  it("rejects non-objects without throwing", () => {
    expect(paths(null)).toEqual([""]);
    expect(paths({ data: { values: [] }, views: [] })).toEqual(["/views"]);
  });
});

describe("loadTable", () => {
  const spec: Spec = {
    data: { url: "t.csv" },
    views: [{ id: "s", type: "scatter", x: "a", y: "b", color: "k" }],
  };
  const csv = "a,b,k,unused\n1,2.5,x,z\n,3,y,z\n4,-1,x,z\n";
  const fetcher = (async () => new Response(csv)) as typeof fetch;

  it("reads only the fields the views use, inferring their types", async () => {
    const t = await loadTable(spec, undefined, fetcher);
    expect(t.rows).toBe(3);
    expect(fieldTypes(t)).toEqual({ a: "quantitative", b: "quantitative", k: "nominal" });
    const a = t.columns.get("a")!;
    expect(a.type === "quantitative" && Array.from(a.values)).toEqual([1, NaN, 4]);
    const k = t.columns.get("k")!;
    expect(k.type === "nominal" && [Array.from(k.codes), k.labels]).toEqual([[0, 1, 0], ["x", "y"]]);
  });

  it("takes external columns and checks their lengths agree", async () => {
    const ext: Spec = { ...spec, data: { columns: "external" } };
    const t = await loadTable(ext, { a: new Float64Array([1, 2]), b: [3, 4], k: ["p", "q"] });
    expect(fieldTypes(t)).toEqual({ a: "quantitative", b: "quantitative", k: "nominal" });
    await expect(loadTable(ext, { a: [1, 2], b: [3], k: ["p", "q"] })).rejects.toThrow(/has 1 rows/);
  });
});

describe("plan", () => {
  it("gives each brushing view its own dimensions", async () => {
    const t = await loadTable(example, {
      px: [1, 2], py: [1, 2], pz: [1, 2], mass: [1, 2], energy: [1, 2], species: ["a", "b"],
    });
    const p = plan(example, t);
    // scatter 2 + density 2 + mass 1 + species 1 + energy 1; the 3D view brushes nothing.
    expect(p.dims).toHaveLength(7);
    expect(p.groups.map((g) => g.kind)).toEqual(["grid", "histogram", "category", "histogram"]);
    expect(p.allocations[0]!.dims).toEqual({ px: 0, py: 1 });
    expect(p.allocations[2]).toEqual({ dims: {} });
  });
});

describe("data from other programs", () => {
  const views = [{ id: "s", type: "histogram", field: "step" }];

  it("reads rows under a JSON Pointer, with the format from .jsonld", async () => {
    const doc = { "@id": "urn:run", "a/b": { runs: [{ step: 1 }, { step: 2 }, { step: 4 }] } };
    const spec = { data: { url: "run.jsonld", pointer: "/a~1b/runs" }, views } as Spec;
    expect(validate(spec)).toEqual([]);
    const table = await loadTable(spec, undefined, async () => new Response(JSON.stringify(doc)));
    expect(table.rows).toBe(3);
  });

  it("lets data.format override the extension", async () => {
    const spec = { data: { url: "/api/readings", format: "tsv" }, views } as Spec;
    const table = await loadTable(spec, undefined, async () => new Response("step\tother\n3\tx\n5\ty\n"));
    expect(fieldTypes(table)).toEqual({ step: "quantitative" });
  });

  it("rejects bad data options by path", () => {
    expect(paths({ data: { url: "a", format: "xlsx", pointer: "rows", fromat: 1 }, views })).toEqual([
      "/data/fromat", "/data/format", "/data/pointer",
    ]);
    expect(paths({ data: { values: [], pointer: "/x" }, views })).toEqual(["/data/pointer"]);
  });

  it("checks figure units and dpi", () => {
    expect(paths({ data: { values: [] }, views, output: { interactive: "yes", figure: { format: "svg", width: 80, unit: "cm", dpi: 0 } } }))
      .toEqual(["/output/interactive", "/output/figure/format", "/output/figure/unit", "/output/figure/dpi"]);
  });
});
