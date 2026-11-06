import { describe, expect, it } from "vitest";
import { nominal, type Table } from "../src/data";
import { plan } from "../src/mount";
import { validate, type Spec } from "../src/spec";
import { edgePairs, incident, nodeTotals } from "../src/views/common";
import { layout3d } from "../src/views/network3d";

/** Six transfers between four projects; one pair repeats. */
const table = (): Table => ({
  rows: 6,
  columns: new Map([
    ["from", nominal(["conspirator", "conspirator", "conspirator", "musande", "musande", "conspirator"])],
    ["to", nominal(["sighthound", "verum", "sighthound", "lavoisier", "conspirator", "verum"])],
  ]),
});

describe("edge views", () => {
  it("validate their two nominal fields, brush and order", () => {
    const spec = {
      data: { values: [] },
      views: [
        { id: "a", type: "arc", source: "from", target: "to", order: "name" },
        { id: "s", type: "sankey", source: "from", target: "to", brush: "none" },
        { id: "n", type: "network3d", source: "from" },
        { id: "x", type: "arc", source: "from", target: "to", order: "size", brush: "xy" },
      ],
    };
    expect(validate(spec, { from: "nominal", to: "nominal" }).map((p) => p.path)).toEqual([
      "/views/2/target", "/views/3/brush", "/views/3/order",
    ]);
    const typed = validate({ data: { values: [] }, views: [{ id: "a", type: "sankey", source: "n", target: "to" }] }, { n: "quantitative", to: "nominal" });
    expect(typed[0]!.message).toMatch(/needs a nominal field/);
  });

  it("join repeated (source, target) rows into one pair", () => {
    const p = edgePairs(table(), "from", "to");
    expect(p.nodes).toEqual(["conspirator", "sighthound", "verum", "musande", "lavoisier"]);
    expect(p.labels).toEqual(["conspirator → sighthound", "conspirator → verum", "musande → lavoisier", "musande → conspirator"]);
    expect(Array.from(p.codes)).toEqual([0, 1, 0, 2, 3, 1]);
  });

  it("select a node's edges on either side", () => {
    const p = edgePairs(table(), "from", "to");
    const conspirator = p.nodes.indexOf("conspirator");
    expect(incident(p, conspirator)).toEqual([0, 1, 3]);
    expect(incident(p, conspirator, "source")).toEqual([0, 1]);
    expect(incident(p, conspirator, "target")).toEqual([3]);
    const t = nodeTotals(p, [2, 2, 1, 1]);
    expect([t.out[conspirator], t.into[conspirator]]).toEqual([4, 1]);
  });

  it("each take one dimension and one count group over pairs", () => {
    const spec: Spec = {
      data: { values: [] },
      views: [
        { id: "a", type: "arc", source: "from", target: "to" },
        { id: "s", type: "sankey", source: "from", target: "to" },
        { id: "b", type: "bars", field: "to" },
      ],
    };
    const { dims, groups, allocations } = plan(spec, table());
    expect(dims).toHaveLength(3);
    expect(dims[0]).toMatchObject({ kind: "categorical", cardinality: 4 });
    expect(groups.map((g) => g.kind)).toEqual(["category", "category", "category"]);
    expect(allocations[0]!.pairs!.labels).toHaveLength(4);
    expect(Object.keys(allocations[1]!.dims)).toEqual(["from → to"]);
  });

  it("lay out a network the same way every time, inside the unit sphere", () => {
    const p = edgePairs(table(), "from", "to");
    const a = layout3d(p, [2, 2, 1, 1]);
    const b = layout3d(p, [2, 2, 1, 1]);
    expect(Array.from(a)).toEqual(Array.from(b));
    for (let i = 0; i < p.nodes.length; i++) expect(Math.hypot(a[i * 3]!, a[i * 3 + 1]!, a[i * 3 + 2]!)).toBeLessThanOrEqual(1 + 1e-6);
    // The pair joined twice sits closer than two nodes with no edge between them.
    const d = (i: number, j: number) => Math.hypot(a[i * 3]! - a[j * 3]!, a[i * 3 + 1]! - a[j * 3 + 1]!, a[i * 3 + 2]! - a[j * 3 + 2]!);
    expect(d(0, 1)).toBeLessThan(d(1, 4));
  });
});
