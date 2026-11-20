// node examples/bundle.mjs
// Writes bundle.rows.json and bundle.lineage.csv: synthetic tables in the shape hinde
// writes from a view bundle (docs/programme/view-in-spectacular.md in hinde). One line
// per row of the bundle; a nominal column with no value holds "none", a quantitative
// one null. The numbers are made up, from a fixed seed, to exercise the charts.
import { writeFileSync } from "node:fs";

let seed = 20261120;
const rand = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 2 ** 32);
const pick = (list, weights) => {
  let r = rand() * weights.reduce((a, b) => a + b, 0);
  for (let i = 0; i < list.length; i++) if ((r -= weights[i]) < 0) return list[i];
  return list[list.length - 1];
};

const ways = ["stated", "supplied", "in-place", "tool", "rule", "node", "asked", "held"];
const steps = { stated: "parse", supplied: "resolve", "in-place": "resolve", tool: "measure", rule: "decide", node: "decide", asked: "ask", held: "hold" };
const outcomes = ["published", "empty", "open", "missing", "blocked", "defect", "not-asked"];
const slots = ["title", "licence", "publisher", "temporal.start", "temporal.end", "spatial.bbox", "mass", "retention", "concentration", "theme"];
const quantities = { mass: [12, 3], retention: [4.2, 0.8], concentration: [0.35, 0.1], "temporal.start": [2019, 2], "temporal.end": [2023, 2] };
const labels = ["unit-unknown", "no-source", "conflict", "out-of-range"];
const blockers = ["parser", "registry", "network", "rule"];

const rows = [];
for (let i = 0; i < 260; i++) {
  const record = `dataset/${1 + Math.floor(rand() * 6)}`;
  const slot = pick(slots, [3, 2, 2, 1, 1, 1, 2, 2, 2, 1]);
  const way = slot in quantities ? pick(ways, [1, 1, 1, 4, 3, 2, 1, 0]) : pick(ways, [5, 3, 2, 1, 1, 1, 1, 1]);
  const outcome = way === "asked" ? pick(outcomes, [1, 0, 3, 0, 0, 0, 3]) : pick(outcomes, [12, 2, 2, 2, 2, 1, 0]);
  const refused = outcome === "blocked" || outcome === "defect";
  const decided = way === "rule" || way === "node";
  const q = quantities[slot];
  const centre = q ? q[0] + (rand() - 0.5) * 2 * q[1] : null;
  const half = q ? q[1] * (0.05 + rand() * 0.4) : null;
  rows.push({
    row: `r${String(i + 1).padStart(3, "0")}`,
    record,
    kind: slot.startsWith("temporal") || slot === "spatial.bbox" ? "extent" : "dataset",
    parent: rand() < 0.3 ? record : "none",
    slot,
    step: steps[way],
    way,
    graph: way === "stated" ? "g:source" : "g:derived",
    basis: way === "stated" ? "stated" : way === "supplied" ? "supplied" : decided ? "derived" : q ? "estimated" : "derived",
    status: `K3:${outcome}`,
    outcome,
    reason: refused ? pick(["unparseable", "ambiguous", "stale"], [1, 1, 1]) : "none",
    label: refused ? pick(labels, [3, 2, 2, 1]) : "none",
    blocker: refused ? pick(blockers, [2, 3, 1, 2]) : "none",
    value: outcome === "published" ? (q ? centre.toFixed(2) : `${slot}-${i}`) : "none",
    low: outcome === "published" && q ? +(centre - half).toFixed(3) : null,
    high: outcome === "published" && q ? +(centre + half).toFixed(3) : null,
    margin: decided ? +(Math.abs(Math.log(rand() + 1e-3)) * 1.4).toFixed(3) : null,
    sources: decided || way === "tool" ? 1 + Math.floor(rand() * 5) : 0,
  });
}
writeFileSync(new URL("bundle.rows.json", import.meta.url), JSON.stringify(rows, null, 1) + "\n");

// The lineage: each row passes from step to step; a few are measured again after a
// decision, which closes a cycle.
const order = ["parse", "resolve", "measure", "decide", "publish"];
const lines = ["row,slot,from_step,step"];
for (const r of rows) {
  const last = r.step === "ask" || r.step === "hold" ? 1 : Math.max(1, order.indexOf(r.step));
  let at = 0;
  while (at < Math.min(last + 1, order.length - 1)) {
    const next = at + (rand() < 0.15 && at + 2 <= last ? 2 : 1);
    lines.push(`${r.row},${r.slot},${order[at]},${order[next]}`);
    if (order[next] === "decide" && rand() < 0.12) lines.push(`${r.row},${r.slot},decide,measure`);
    at = next;
  }
}
writeFileSync(new URL("bundle.lineage.csv", import.meta.url), lines.join("\n") + "\n");
console.log(`${rows.length} rows, ${lines.length - 1} lineage links`);
