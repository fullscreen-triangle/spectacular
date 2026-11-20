import { describe, expect, it } from "vitest";
import { fieldTypes, loadTable, type Column } from "../src/data";
import { validate, type Spec } from "../src/spec";
import { colorsFor, crossPairs, edgePairs, PALETTE } from "../src/views/common";
import { layering } from "../src/views/layers";

const values = (rows: Record<string, unknown>[]) => ({ values: rows });
const labelsOf = (c: Column | undefined) => (c?.type === "nominal" ? c.labels : []);
const codesOf = (c: Column | undefined) => (c?.type === "nominal" ? Array.from(c.codes) : []);

describe("a declared domain", () => {
  const spec = {
    data: values([{ way: "tool" }, { way: "stated" }, { way: "none" }, { way: "tool" }]),
    fields: { way: { domain: ["stated", "supplied", "tool"] } },
    views: [{ id: "b", type: "bars", field: "way" }],
  } as Spec;

  it("orders the categories, keeps empty ones, and puts others after", async () => {
    const t = await loadTable(spec);
    expect(labelsOf(t.columns.get("way"))).toEqual(["stated", "supplied", "tool", "none"]);
    expect(codesOf(t.columns.get("way"))).toEqual([2, 0, 3, 2]);
  });

  it("orders columns a program passed already coded", async () => {
    const t = await loadTable({ ...spec, data: { columns: "external" } } as Spec, {
      way: { codes: [0, 1, 2, 0], labels: ["tool", "stated", "none"] },
    });
    expect(labelsOf(t.columns.get("way"))).toEqual(["stated", "supplied", "tool", "none"]);
    expect(codesOf(t.columns.get("way"))).toEqual([2, 0, 3, 2]);
  });

  it("makes a field nominal, and is checked", () => {
    expect(validate({ ...spec, fields: { way: { domain: ["a", "a"] } } })).toEqual([
      { path: "/fields/way/domain", message: "a list of category names, each once" },
    ]);
    expect(validate({ ...spec, fields: { way: { type: "quantitative", domain: ["a"] } } })[0]?.path).toBe("/fields/way");
    expect(validate(spec, { way: "nominal" })).toEqual([]);
  });
});

describe("a field's palette", () => {
  const table = { rows: 3, columns: new Map<string, Column>([["o", { type: "nominal", codes: new Uint32Array([0, 1, 2]), labels: ["a", "b", "c"] }]]) };
  const ctx = (palette: unknown) => ({ table, fieldDef: () => ({ palette }) as never });

  it("colours categories by position or by name, the rest from the defaults", () => {
    expect(colorsFor(ctx(["red", "#00f"]), "o")).toEqual(["#ff0000", "#0000ff"]);
    expect(colorsFor(ctx({ b: "teal" }), "o")).toEqual([PALETTE[0], "#008080", PALETTE[1]]);
  });

  it("gives way to a view's own palette", () => {
    expect(colorsFor(ctx({ b: "teal" }), "o", { palette: ["gold"] })).toEqual(["#ffd700"]);
  });

  it("is checked by position or name", () => {
    const s = { data: values([]), views: [{ id: "b", type: "bars", field: "o" }] };
    expect(validate({ ...s, fields: { o: { palette: { a: "teal", b: "nope" } } } })).toEqual([
      { path: "/fields/o/palette/b", message: 'a colour: "#rrggbb", "#rgb" or a CSS colour name such as "steelblue"' },
    ]);
  });
});

describe("JSON rows and keys", () => {
  it("take a table's columns from every row", async () => {
    const t = await loadTable({ data: values([{ a: 1 }, { b: "x" }]), views: [{ id: "h", type: "histogram", field: "a" }, { id: "k", type: "bars", field: "b" }] } as Spec);
    expect(fieldTypes(t)).toEqual({ a: "quantitative", b: "nominal" });
    expect(labelsOf(t.columns.get("b"))).toEqual(["", "x"]);
  });

  it("load a key as a category even when it looks like numbers", async () => {
    const spec = { key: "id", data: values([{ id: 7, v: 1 }, { id: 9, v: 2 }]), views: [{ id: "h", type: "histogram", field: "v" }] } as Spec;
    const t = await loadTable(spec);
    expect(labelsOf(t.columns.get("id"))).toEqual(["7", "9"]);
    expect(validate(spec, fieldTypes(t))).toEqual([]);
    expect(validate({ ...spec, key: "nope" }, fieldTypes(t))).toEqual([{ path: "/key", message: 'no field "nope" in the data' }]);
  });
});

describe("pairs and layers", () => {
  const table = {
    rows: 5,
    columns: new Map<string, Column>([
      ["a", { type: "nominal", codes: new Uint32Array([0, 0, 1, 1, 0]), labels: ["p", "q"] }],
      ["b", { type: "nominal", codes: new Uint32Array([0, 1, 0, 0, 0]), labels: ["q", "r"] }],
    ]),
  };

  it("keep each field's own categories in cross pairs", () => {
    const c = crossPairs(table, "a", "b");
    expect(c.pairs).toEqual([[0, 0], [0, 1], [1, 0]]);
    expect(Array.from(c.codes)).toEqual([0, 1, 2, 2, 0]);
    expect(c.labels).toEqual(["p × q", "p × r", "q × q"]);
  });

  it("layer a graph by its longest paths and break cycles where they close", () => {
    const steps = {
      rows: 5,
      columns: new Map<string, Column>([
        ["from", { type: "nominal", codes: new Uint32Array([0, 1, 2, 0, 3]), labels: ["parse", "resolve", "measure", "decide"] }],
        ["to", { type: "nominal", codes: new Uint32Array([0, 1, 2, 1, 1]), labels: ["resolve", "measure", "decide"] }],
      ]),
    };
    const p = edgePairs(steps, "from", "to");
    const { layer, back } = layering(p);
    const at = (name: string) => layer[p.nodes.indexOf(name)];
    expect([at("parse"), at("resolve"), at("measure"), at("decide")]).toEqual([0, 1, 2, 3]);
    expect([...back].map((k) => p.labels[k])).toEqual(["decide → measure"]);
  });
});
