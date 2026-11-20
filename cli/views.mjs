// node cli/views.mjs [shots-dir]   (needs `npm run build:python` and
//   `cargo build --release -p spectacular`)
// The views made for categories and quantities (split bars, strip, matrix, interval,
// layers) and selecting rows by key, driven in Chromium on standalone pages that the
// native command writes from the example descriptions. Every selection is checked
// against a brute-force count over the table files.
import { spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync, copyFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { chromium } from "playwright";

const repo = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const exe = join(repo, "target", "release", process.platform === "win32" ? "spectacular.exe" : "spectacular");
const shots = process.argv[2];
const dir = mkdtempSync(join(tmpdir(), "spectacular-views-"));
let failures = 0;
const ok = (cond, what) => {
  console.log(`${cond ? "ok  " : "FAIL"} ${what}`);
  if (!cond) failures++;
};

for (const f of ["bundle.rows.json", "bundle.lineage.csv", "bundle.desc.json", "lineage.desc.json"]) copyFileSync(join(repo, "examples", f), join(dir, f));
writeFileSync(join(dir, "spans.desc.json"), JSON.stringify({
  data: "bundle.rows.json", key: "row", charts: [{ low: "low", high: "high", title: "spans" }, { x: "slot" }],
}));
const page = (name) => {
  const r = spawnSync(exe, ["html", `${name}.desc.json`, "-o", `${name}.html`], { cwd: dir, encoding: "utf8" });
  if (r.status !== 0) throw new Error(`${name}: ${r.stderr}`);
  return pathToFileURL(join(dir, `${name}.html`)).href;
};
const rows = JSON.parse(readFileSync(join(dir, "bundle.rows.json"), "utf8"));
const lineage = readFileSync(join(dir, "bundle.lineage.csv"), "utf8").trim().split("\n").slice(1).map((l) => l.split(","));

const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
try {
  const p = await browser.newPage({ viewport: { width: 1300, height: 1000 } });
  const errors = [];
  p.on("pageerror", (e) => errors.push(e.message));
  const open = async (url) => {
    await p.goto(url);
    await p.waitForFunction(() => document.body.dataset.state && document.body.dataset.state !== "loading", null, { timeout: 60_000 });
    ok(await p.evaluate(() => document.body.dataset.state) === "ready", `${url.split("/").pop()} draws`);
  };
  /** Waits until the selection has `n` filters and the page has caught up with them. */
  const settled = async (n) => {
    await p.waitForFunction((n) => window.spectacular.selection().filters.length === n, n, { timeout: 15_000 });
    await p.waitForTimeout(250);
    return p.evaluate(() => window.spectacular.selection());
  };
  const view = (id) => `[data-view="${id}"]`;
  const drag = async (selector, from, to) => {
    const box = await p.locator(`${selector} .sv-brush .overlay`).boundingBox();
    await p.mouse.move(box.x + box.width * from, box.y + box.height / 2);
    await p.mouse.down();
    await p.mouse.move(box.x + box.width * to, box.y + box.height / 2, { steps: 6 });
    await p.mouse.up();
  };

  await open(page("bundle"));
  if (shots) await p.screenshot({ path: join(shots, "views-bundle.png"), fullPage: true });

  // Split bars: the bands follow the declared domain; clicking one selects its category.
  const ticks = await p.locator(`${view("each-way-split-into-outcomes")} .sv-axis .tick text`).allTextContents();
  const bands = ticks.filter((t) => Number.isNaN(Number(t))); // the category axis, not the counts
  ok(bands.slice(0, 8).join() === "stated,supplied,in-place,tool,rule,node,asked,held", `split bars keep the declared order (${bands.slice(0, 8).join(", ")})`);
  await p.locator(`${view("each-way-split-into-outcomes")} .sv-row`).nth(3).click();
  let s = await settled(1);
  const tool = rows.filter((r) => r.way === "tool").length;
  ok(s.filters[0].field === "way" && s.filters[0].values.join() === "tool" && s.selected === tool,
    `clicking a split bar selects its category (${s.selected} rows; the table has ${tool})`);
  await p.locator(`${view("each-way-split-into-outcomes")} .sv-row`).nth(3).click();
  await settled(0);

  // Matrix: a cell selects its pair.
  await p.locator(`${view("refusals-label-by-blocker")} g.sv-hit:not(.sv-axis)`).nth(1).click();
  s = await settled(1);
  const [label, blocker] = s.filters[0].values[0].split(" × ");
  const pair = rows.filter((r) => r.label === label && r.blocker === blocker).length;
  ok(s.selected === pair && pair > 0, `a matrix cell selects its pair (${label} × ${blocker}: ${s.selected} rows; the table has ${pair})`);
  await p.locator(`${view("refusals-label-by-blocker")} g.sv-hit:not(.sv-axis)`).nth(1).click();
  await settled(0);

  // Strip: brushing the value keeps the rows in range, of every category.
  await drag(view("margins-by-outcome"), 0.05, 0.4);
  s = await settled(1);
  const [a, z] = s.filters[0].range;
  const inRange = rows.filter((r) => r.margin !== null && r.margin >= a && r.margin <= z).length;
  ok(s.filters[0].field === "margin" && s.selected === inRange, `brushing a strip keeps margins in [${a.toFixed(2)}, ${z.toFixed(2)}] (${s.selected}; the table has ${inRange})`);
  await p.evaluate(() => window.spectacular.engine.filter({ dim: [...window.spectacular.engine.filters.keys()][0], clear: true }));
  await settled(0);

  // Selecting by key from outside, as a page holding the bundle would.
  const wanted = ["r001", "r005", "r009", "nope"];
  const matched = await p.evaluate((k) => window.spectacular.select(k), wanted);
  s = await settled(1);
  const keys = await p.evaluate(() => window.spectacular.selectedKeys());
  ok(matched === 3 && s.selected === 3 && keys.join() === "r001,r005,r009" && s.filters[0].view === "" && s.filters[0].field === "row",
    `select() by key shows exactly those rows (${keys.join(", ")}; "nope" is not in the data)`);
  await p.evaluate(() => window.spectacular.select(null));
  await settled(0);

  // Interval on a shared scale: brushing keeps the rows whose interval overlaps the range.
  await open(page("spans"));
  await drag(view("spans"), 0.0, 0.02);
  s = await settled(1);
  const [lo, hi] = s.filters[0].overlaps;
  const overlapping = rows.filter((r) => r.low !== null && r.high !== null && r.low <= hi && r.high >= lo).length;
  ok(s.filters[0].field === "low" && s.filters[0].to === "high" && s.selected === overlapping && overlapping > 0,
    `brushing intervals keeps those overlapping [${lo.toFixed(2)}, ${hi.toFixed(2)}] (${s.selected}; the table has ${overlapping})`);

  // Layers: a node selects every flow that touches it.
  await open(page("lineage"));
  if (shots) await p.screenshot({ path: join(shots, "views-lineage.png"), fullPage: true });
  await p.locator(`${view("the-steps-left-to-right")} g.sv-hit`).filter({ hasText: /^measure$/ }).locator("rect").click();
  s = await settled(1);
  const touching = lineage.filter(([, , from, to]) => from === "measure" || to === "measure").length;
  ok(s.selected === touching && s.filters[0].values.every((v) => v.includes("measure")),
    `a layers node selects the flows through it (${s.selected} links; the table has ${touching})`);

  ok(errors.length === 0, `no page errors${errors.length ? ": " + errors.join("; ") : ""}`);
} catch (e) {
  ok(false, e.stack);
} finally {
  await browser.close();
  rmSync(dir, { recursive: true, force: true, maxRetries: 5 });
}
console.log(failures ? `${failures} failed` : "PASS");
process.exit(failures ? 1 : 0);
