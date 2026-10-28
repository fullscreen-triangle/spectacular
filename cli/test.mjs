// node cli/test.mjs   (build the viewer first: npm run build)
// Drives the CLI as another program would: spawns it, reads its JSON lines and exit codes.
import { spawn, spawnSync } from "node:child_process";
import { copyFileSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath, pathToFileURL } from "node:url";
import { chromium } from "playwright";

const repo = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const cli = join(repo, "cli/spectacular.mjs");
const dir = mkdtempSync(join(tmpdir(), "spectacular-cli-"));
let failures = 0;
const ok = (cond, what) => {
  console.log(`${cond ? "ok  " : "FAIL"} ${what}`);
  if (!cond) failures++;
};
const run = (args, input) => {
  const r = spawnSync(process.execPath, [cli, ...args], { cwd: dir, input, encoding: "utf8", timeout: 180_000 });
  return { code: r.status, out: r.stdout, err: r.stderr, json: r.stdout.trim().split("\n").filter(Boolean).map((l) => { try { return JSON.parse(l); } catch { return undefined; } }) };
};

for (const f of ["lorenz.spec.json", "lorenz.csv"]) copyFileSync(join(repo, "examples", f), join(dir, f));
const lorenz = JSON.parse(readFileSync(join(dir, "lorenz.spec.json"), "utf8"));

try {
  // The committed schema is what `schema` prints.
  ok(run(["schema"]).out === readFileSync(join(repo, "spectacular.schema.json"), "utf8"), "spectacular.schema.json is current");

  let r = run(["validate", "--json", "lorenz.spec.json"]);
  ok(r.code === 0 && r.json[0]?.ok && r.json[0].rows === 29_900, "validate: a good spec exits 0 with its row count");
  r = run(["validate", "--json", "-"], JSON.stringify({ ...lorenz, views: [{ id: "h", type: "histogram", field: "lobe", bnis: 4 }] }));
  ok(r.code === 1 && r.json[0]?.problems.map((p) => p.path).join() === "/views/0/bnis,/views/0/field", "validate: stdin spec with problems exits 1 and names each path");
  ok(run(["validate", "missing.json"]).code === 2, "validate: an unreadable file exits 2");

  // A JSON-LD document whose rows sit under a key, as a run log would write it.
  writeFileSync(join(dir, "run.jsonld"), JSON.stringify({ "@id": "urn:run", executions: [
    { pass: 0, step_index: 0, event: "done" }, { pass: 0, step_index: 1, event: "incubated" }, { pass: 1, step_index: 1, event: "incubated" },
  ] }));
  const runSpec = { data: { url: "run.jsonld", pointer: "/executions" }, views: [{ id: "event", type: "bars", field: "event" }, { id: "step", type: "histogram", field: "step_index" }] };
  r = run(["validate", "--json", "-"], JSON.stringify(runSpec));
  ok(r.code === 0 && r.json[0]?.rows === 3 && r.json[0].fields.event === "nominal", "validate: data.pointer reads rows inside a JSON-LD document");
  r = run(["validate", "--json", "-"], JSON.stringify({ ...runSpec, data: { url: "run.jsonld", pointer: "/nothing" } }));
  ok(r.code === 1 && r.json[0]?.problems[0]?.message.includes("nothing at"), "validate: a pointer to nothing is a problem");

  r = run(["render", "--json", "lorenz.spec.json", "-o", "fig.png", "--width", "600", "--dpi", "192"]);
  const png = r.code === 0 ? readFileSync(join(dir, "fig.png")) : Buffer.alloc(0);
  ok(r.code === 0 && png.readUInt32BE(16) === 1200 && r.json[0]?.pixels[0] === 1200, `render png: 600 px at 2x is 1200 pixels wide (${png.length ? png.readUInt32BE(16) : "no file"})`);
  r = run(["render", "--json", "-", "-o", "fig.pdf"], JSON.stringify({ ...lorenz, output: { figure: { format: "pdf", width: 180, unit: "mm" } } }));
  const pdf = r.code === 0 ? readFileSync(join(dir, "fig.pdf"), "latin1") : "";
  const box = /\/MediaBox\s*\[0 0 ([\d.]+)/.exec(pdf);
  ok(r.code === 0 && pdf.startsWith("%PDF") && (pdf.match(/\/Type\s*\/Page[^s]/g) ?? []).length === 1 && Math.abs(+box[1] / 72 * 25.4 - 180) < 0.5,
    `render pdf: one page, 180 mm wide from output.figure (${box ? (+box[1] / 72 * 25.4).toFixed(1) + " mm" : "no MediaBox"})`);
  ok(run(["render", "-", "-o", "x.png"], JSON.stringify({ ...lorenz, data: { url: "gone.csv" } })).code === 1, "render: missing data exits 1 before starting a browser");
  ok(run(["render", "lorenz.spec.json", "-o", "x.gif"]).code === 2, "render: an unsupported output format exits 2");
  r = run(["render", "--json", "lorenz.spec.json", "-o", "fig.svg", "--width", "180", "--unit", "mm"]);
  const svg = r.code === 0 ? readFileSync(join(dir, "fig.svg"), "utf8") : "";
  const count = (re) => (svg.match(re) ?? []).length;
  ok(svg.startsWith("<?xml") && /<svg[^>]* width="179\.\d+mm"/.test(svg) && !svg.includes("currentColor"),
    "render svg: a standalone document sized in mm, with no page-dependent colours");
  ok(count(/<circle /g) === 29_900 && count(/<image /g) === 1 && svg.includes(">Lorenz attractor</text>"),
    `render svg: the 2D scatter is ${count(/<circle /g)} vector points, the 3D view an image, titles are text`);

  // serve: brush in a real page, read the selection back from stdout.
  const child = spawn(process.execPath, [cli, "serve", "--json", "--indices", "lorenz.spec.json"], { cwd: dir });
  const lines = createInterface({ input: child.stdout });
  const events = [];
  const waiters = [];
  lines.on("line", (l) => {
    const e = JSON.parse(l);
    events.push(e);
    waiters.splice(0).forEach((w) => w());
  });
  const next = async (event, ms = 30_000) => {
    const t = Date.now() + ms;
    for (;;) {
      const i = events.findIndex((e) => e.event === event);
      if (i >= 0) return events.splice(i, 1)[0];
      if (Date.now() > t) throw new Error(`no "${event}" event within ${ms} ms`);
      await new Promise((w) => { waiters.push(w); setTimeout(w, 250); });
    }
  };
  const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
  try {
    if (svg) {
      const p = await browser.newPage();
      await p.goto(pathToFileURL(join(dir, "fig.svg")).href);
      ok(await p.evaluate(() => document.documentElement.tagName === "svg" && !document.querySelector("parsererror")), "render svg: the file parses as SVG on its own");
      await p.close();
    }
    const { url } = await next("listening");
    const u = new URL(url);
    ok(u.hostname === "127.0.0.1", "serve: binds loopback");
    const token = u.searchParams.get("spec").split("/")[2];
    ok((await fetch(`${u.origin}/s/${"0".repeat(token.length)}/lorenz.csv`)).status === 404 && (await fetch(`${u.origin}/lorenz.csv`)).status === 404
      && (await fetch(`${u.origin}/s/${token}/lorenz.csv`)).ok, "serve: data is reachable only under the random prefix");
    ok((await fetch(`${u.origin}/s/${token}/..%2f..%2fpackage.json`)).status === 404, "serve: paths cannot leave the spec's directory");

    const page = await browser.newPage({ viewport: { width: 1200, height: 1000 } });
    await page.goto(url);
    await page.waitForFunction(() => document.body.dataset.state === "ready", null, { timeout: 60_000 });
    const tb = await page.locator('[data-view="time"] .sv-brush .overlay').boundingBox();
    await page.mouse.move(tb.x + tb.width * 0.2, tb.y + tb.height / 2);
    await page.mouse.down();
    await page.mouse.move(tb.x + tb.width * 0.3, tb.y + tb.height / 2, { steps: 5 });
    await page.mouse.up();
    let sel = await next("selection");
    while (events.some((e) => e.event === "selection")) sel = await next("selection"); // the last one is the settled state
    const shown = await page.evaluate(() => window.spectacular.engine.state.selectedCount);
    const f = sel.filters[0];
    ok(sel.rows === 29_900 && sel.selected === shown && sel.selected > 0 && sel.selected < 29_900, `serve: selection event matches the page (${sel.selected} rows)`);
    ok(sel.filters.length === 1 && f.view === "time" && f.field === "t" && f.range[0] < f.range[1], `serve: the filter is reported in data terms (t in [${f?.range?.map((v) => v.toFixed(2))}])`);
    // The program can apply the filter to its own rows and get the same answer.
    const t = readFileSync(join(dir, "lorenz.csv"), "utf8").trim().split("\n");
    const col = t[0].split(",").indexOf("t");
    const mine = t.slice(1).flatMap((l, i) => { const v = +l.split(",")[col]; return v >= f.range[0] && v <= f.range[1] ? [i] : []; });
    ok(mine.length === sel.selected && sel.indices.length === sel.selected && mine.every((v, i) => v === sel.indices[i]), "serve: applying the reported range to the CSV gives exactly the reported indices");

    // A changed data file reloads the page.
    writeFileSync(join(dir, "lorenz.csv"), readFileSync(join(dir, "lorenz.csv")));
    const reload = await next("reload");
    await page.waitForFunction(() => document.body.dataset.state === "ready" && !window.spectacular.engine.filters.size, null, { timeout: 60_000 });
    ok(reload.file === "lorenz.csv", "serve: rewriting the data file reloads the page");

    // A broken spec still serves, and the page says what is wrong.
    writeFileSync(join(dir, "lorenz.spec.json"), JSON.stringify({ ...lorenz, views: [{ id: "a", type: "histogram", field: "nope" }] }));
    await next("reload");
    await page.waitForFunction(() => document.body.dataset.state === "error", null, { timeout: 60_000 });
    ok((await page.textContent("#error")).includes('no field "nope"'), "serve: an invalid spec shows its problem in the page");
  } finally {
    await browser.close();
    const exited = new Promise((r) => child.once("exit", r));
    child.kill();
    await exited; // it watches the directory removed below
  }
} catch (e) {
  ok(false, e.stack);
} finally {
  rmSync(dir, { recursive: true, force: true, maxRetries: 5 });
}
console.log(failures ? `${failures} failed` : "PASS");
process.exit(failures ? 1 : 0);
