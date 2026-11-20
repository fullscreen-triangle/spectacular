// node cli/conformance.mjs [cases] [seed]   (build the native command first:
//   cargo build --release -p spectacular)
// The Rust validator (crates/spectacular) must answer exactly as the TypeScript one
// (packages/viz/src/spec.ts, through cli/check.mjs) does: same problems, in the same
// order, with the same paths and messages, and the same rows and field types. Cases are
// valid specs and seeded random mutations of them.
import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { check } from "./check.mjs";

const repo = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const exe = join(repo, "target", "release", process.platform === "win32" ? "spectacular.exe" : "spectacular");
if (!existsSync(exe)) {
  console.error(`no ${exe}; run: cargo build --release -p spectacular`);
  process.exit(2);
}
const count = Number(process.argv[2] ?? 600);
let seed = Number(process.argv[3] ?? 20261120) >>> 0;
const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
const pick = (list) => list[Math.floor(rand() * list.length)];

const rows = [
  { a: 1, b: 2.5, c: "3", k: "x", s: "p", t: "q", w: 7 },
  { a: 4, b: -1, c: "", k: "y", s: "q", t: "p", w: true },
  { a: null, b: 0, c: "1e3", k: "x", s: "p", t: "r", w: "n/a" },
];
const csv = "a,b,k,s,t\n1,2,x,p,q\n3,,y,q,p\n";
const bases = [
  { version: "spectacular/v0", title: "All views", theme: "dark", data: { values: rows }, layout: { columns: 3 },
    fields: { k: { type: "nominal", title: "kind" }, c: { type: "quantitative" } },
    views: [
      { id: "s", type: "scatter", x: "a", y: "b", color: "k", brush: "xy", pointSize: 3, fill: "teal", palette: ["red", "#00f"] },
      { id: "h", type: "histogram", field: "b", bins: 20, fill: "#abc", span: 2, height: 150 },
      { id: "b", type: "bars", field: "k", palette: ["gold"] },
      { id: "d", type: "density", x: "a", y: "c", bins: [16, 16] },
      { id: "3", type: "scatter3d", x: "a", y: "b", z: "c", color: "a" },
      { id: "arc", type: "arc", source: "s", target: "t", order: "name" },
      { id: "sk", type: "sankey", source: "s", target: "t", brush: "none" },
      { id: "net", type: "network3d", source: "s", target: "t" },
    ],
    output: { interactive: true, figure: { format: "pdf", width: 180, unit: "mm", dpi: 300 } } },
  { data: { url: "table.csv" }, views: [{ id: "h", type: "histogram", field: "a" }, { id: "k", type: "bars", field: "k" }] },
  { data: { url: "table.csv", format: "csv" }, views: [{ id: "s", type: "scatter", x: "a", y: "b" }] },
  { data: { columns: "external" }, views: [{ id: "h", type: "histogram", field: "a" }] },
  { data: { values: [] }, views: [{ id: "h", type: "histogram", field: "a" }] },
  // Domains and colours declared on fields, a key, and the views over categories.
  { title: "Rows of a bundle", key: "id", data: { values: rows.map((r, i) => ({ ...r, id: `r${i}`, lo: i, hi: i + 2 })) },
    fields: {
      k: { domain: ["y", "x", "z"], palette: { x: "teal", y: "#e15759" } },
      s: { type: "nominal", domain: ["q", "p"], palette: ["gold", "#123456"] },
      t: { palette: ["red"] },
    },
    views: [
      { id: "split", type: "bars", field: "k", color: "s" },
      { id: "strip", type: "strip", x: "b", y: "k", color: "s", pointSize: 2, fill: "navy" },
      { id: "grid", type: "matrix", x: "s", y: "t", brush: "select" },
      { id: "span", type: "interval", low: "lo", high: "hi", label: "k", color: "a", brush: "x", palette: ["red"] },
      { id: "own", type: "interval", low: "lo", high: "hi", label: "s", scale: "label", brush: "none" },
      { id: "flow", type: "layers", source: "s", target: "t" },
    ] },
  // JSON rows need not all carry the same properties.
  { data: { values: [{ a: 1 }, { b: "x" }, { a: 2, b: "y" }] },
    views: [{ id: "h", type: "histogram", field: "a" }, { id: "k", type: "bars", field: "b" }, { id: "strip", type: "strip", x: "a", y: "b", brush: "none" }] },
];
const junk = [null, 0, -1, 1.5, 2, 33, 1025, 10001, "x", "", "#zzz", "teal", "TEAL", "#abc", "#abcd", [], [1, 2], ["red"],
  ["red", "nope"], [0, 0], [16, 2000], {}, true, false, "none", "xy", "x", "select", "count", "a", "k", "s", "nope",
  "density", "pie", "spectacular/v1", "paper", "csv", "xlsx", "/rows", "rows", "external", "mm", "cm", "svg", "gif",
  "strip", "matrix", "interval", "layers", "id", "lo", "hi", "label", "shared", ["x", "x"], ["y", "z"], { x: "red" }, { x: "nope" }, ["red", 3]];
const typos = ["colr", "feild", "binz", "tpye", "titel", "colour", "point_size", "Fill", "pallete", "extra", "brsh", "domian", "lable", "kye", "hgih"];

function paths(v, at = []) {
  const out = [at];
  if (v && typeof v === "object") for (const k of Object.keys(v)) out.push(...paths(v[k], [...at, k]));
  return out;
}
function mutate(spec) {
  const s = structuredClone(spec);
  for (let n = 1 + Math.floor(rand() * 3); n > 0; n--) {
    const all = paths(s).filter((p) => p.length > 0 && !(p[0] === "data" && p[1] === "values" && p.length > 2));
    if (!all.length) break;
    const p = pick(all);
    const parent = p.slice(0, -1).reduce((o, k) => o[k], s);
    const key = p[p.length - 1];
    const op = rand();
    // A view replaced by null makes the TypeScript loader throw a JavaScript error
    // message that Rust has no reason to copy; that one case is left out.
    const isView = p.length === 2 && p[0] === "views";
    if (op < 0.3) Array.isArray(parent) ? parent.splice(Number(key), 1) : delete parent[key];
    else if (op < 0.75) parent[key] = structuredClone(isView ? pick(junk.filter((j) => j !== null)) : pick(junk));
    else if (op < 0.9 && parent && typeof parent === "object" && !Array.isArray(parent)) parent[pick(typos)] = structuredClone(pick(junk));
    else if (Array.isArray(s.views) && s.views.length) s.views.push(structuredClone(pick(s.views)));
  }
  return s;
}

const dir = mkdtempSync(join(tmpdir(), "spectacular-conformance-"));
writeFileSync(join(dir, "table.csv"), csv);
const cases = [...bases, ...Array.from({ length: count }, () => mutate(pick(bases)))];
const files = cases.map((c, i) => {
  const f = join(dir, `case-${i}.json`);
  writeFileSync(f, JSON.stringify(c));
  return f;
});
// In batches: Windows caps a command line at 32 KB.
const native = [];
for (let i = 0; i < files.length; i += 100) {
  const run = spawnSync(exe, ["validate", "--json", ...files.slice(i, i + 100)], { encoding: "utf8", maxBuffer: 1 << 28 });
  if (run.error) throw run.error;
  native.push(...run.stdout.trim().split("\n").map((l) => JSON.parse(l)));
}
// A file the system cannot read is reported in the system's own words, which differ
// between Node and Rust (and by locale); for those only the path is compared.
const os = (problems) => problems?.map((p) =>
  p.path === "/data" && (/^E[A-Z]+:/.test(p.message) || p.message.startsWith("cannot read ")) ? { path: p.path, message: "(system error)" } : p);
let failures = 0;
for (let i = 0; i < cases.length; i++) {
  const ts = await check(cases[i], dir);
  const want = { ok: ts.problems.length === 0, problems: os(ts.problems), rows: ts.rows, fields: ts.fields };
  const got = { ok: native[i]?.ok, problems: os(native[i]?.problems), rows: native[i]?.rows, fields: native[i]?.fields };
  if (JSON.stringify(want) !== JSON.stringify(got)) {
    if (failures++ < 5) console.log(`MISMATCH case ${i}: ${JSON.stringify(cases[i])}\n  ts:   ${JSON.stringify(want)}\n  rust: ${JSON.stringify(got)}`);
  }
}
const problems = cases.reduce((n, _, i) => n + (native[i]?.problems?.length ?? 0), 0);
rmSync(dir, { recursive: true, force: true });
console.log(`${cases.length} specs, ${problems} problems between them: ${failures ? `${failures} differ` : "the two validators agree on every one"}`);
process.exit(failures ? 1 : 0);
