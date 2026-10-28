// node test/e2e.mjs [rows] [screenshot-dir]   (run `vite build` first)
// Drives the built demo in headless Chromium: real brushes, real WebGL, real worker.
import { chromium } from "playwright";
import { preview } from "vite";

const rows = Number(process.argv[2] ?? 1_000_000);
const shots = process.argv[3];
const server = await preview({ root: new URL("..", import.meta.url).pathname.replace(/^\/(\w:)/, "$1"), preview: { port: 4317 }, logLevel: "warn" });
const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
const page = await browser.newPage({ viewport: { width: 1400, height: 1100 } });
const problems = [];
page.on("pageerror", (e) => problems.push(`page error: ${e.message}`));
page.on("console", (m) => m.type() === "error" && problems.push(`console: ${m.text()}`));

const fail = async (msg) => {
  console.error(`FAIL: ${msg}`);
  for (const p of problems) console.error(`  ${p}`);
  await browser.close();
  await server.close();
  process.exit(1);
};

const t0 = Date.now();
await page.goto(`http://localhost:4317/?rows=${rows}`);
try {
  await page.waitForFunction(() => document.getElementById("note")?.textContent?.includes("indexed") || document.getElementById("error")?.textContent, null, { timeout: 180_000 });
} catch {
  await fail("demo did not finish loading");
}
const err = await page.textContent("#error");
if (err) await fail(`mount failed: ${err}`);
console.log(`loaded ${rows.toLocaleString()} rows in ${((Date.now() - t0) / 1000).toFixed(1)} s: ${await page.textContent("#note")}`);

const status = () => page.textContent(".sv-status");
const consistent = () =>
  page.evaluate(() => {
    const s = window.spectacular.engine.state;
    let sum = 0;
    for (const v of s.mask) sum += v;
    return { mask: sum, count: s.selectedCount };
  });
const settle = async (before) => {
  await page.waitForFunction((b) => document.querySelector(".sv-status")?.textContent !== b, before, { timeout: 30_000 });
  await page.waitForTimeout(300); // let coalesced follow-ups land
};

// Drag a brush across the middle of the mass histogram.
let before = await status();
const box = await page.locator('[data-view="mass"] .sv-brush .overlay').boundingBox();
await page.mouse.move(box.x + box.width * 0.35, box.y + box.height / 2);
await page.mouse.down();
for (let i = 1; i <= 10; i++) await page.mouse.move(box.x + box.width * (0.35 + i * 0.02), box.y + box.height / 2);
await page.mouse.up();
await settle(before);
const afterBrush = await status();
console.log(`after mass brush: ${afterBrush}`);
let c = await consistent();
if (c.mask !== c.count) await fail(`mask has ${c.mask} rows set but the worker reports ${c.count}`);
if (c.count === 0 || c.count >= rows) await fail(`brush selected ${c.count} of ${rows}`);

// Toggle one species.
before = await status();
await page.locator('[data-view="species"] .sv-row').first().click();
await settle(before);
console.log(`after species click: ${await status()}`);
c = await consistent();
if (c.mask !== c.count) await fail(`mask has ${c.mask} rows set but the worker reports ${c.count}`);

// The WebGL views must actually have drawn.
for (const id of ["momentum", "space"]) {
  const lit = await page.evaluate((id) => {
    const src = document.querySelector(`[data-view="${id}"] canvas`);
    const c = document.createElement("canvas");
    c.width = src.width; c.height = src.height;
    const g = c.getContext("2d");
    g.drawImage(src, 0, 0);
    const d = g.getImageData(0, 0, c.width, c.height).data;
    let painted = 0, coloured = 0;
    for (let i = 0; i < d.length; i += 4) {
      if (d[i + 3] === 0) continue;
      painted++;
      // Unselected points are grey; selected ones carry the species palette.
      if (Math.max(d[i], d[i + 1], d[i + 2]) - Math.min(d[i], d[i + 1], d[i + 2]) > 40) coloured++;
    }
    return { painted, coloured };
  }, id);
  console.log(`${id}: ${lit.painted} painted pixels, ${lit.coloured} coloured`);
  if (lit.painted < 100) await fail(`${id} canvas is blank`);
  if (lit.coloured < 20) await fail(`${id} does not show the selected rows`);
}

if (shots) {
  await page.screenshot({ path: `${shots}/demo.png`, fullPage: true });
  console.log(`screenshot: ${shots}/demo.png`);
}

// Viewer mode, as another repo would use it: a spec file whose data.url is a CSV beside it.
await page.goto("http://localhost:4317/?spec=/lorenz.spec.json");
await page.waitForFunction(() => document.body.dataset.state !== "loading", null, { timeout: 60_000 });
if ((await page.evaluate(() => document.body.dataset.state)) !== "ready") await fail(`viewer: ${await page.textContent("#error")}`);
const rowsLoaded = await page.evaluate(() => window.spectacular.table.rows);
if (rowsLoaded !== 29_900) await fail(`viewer loaded ${rowsLoaded} rows, expected 29900`);
before = await status();
const tb = await page.locator('[data-view="time"] .sv-brush .overlay').boundingBox();
await page.mouse.move(tb.x + tb.width * 0.2, tb.y + tb.height / 2);
await page.mouse.down();
await page.mouse.move(tb.x + tb.width * 0.3, tb.y + tb.height / 2, { steps: 5 });
await page.mouse.up();
await settle(before);
console.log(`viewer: ${await status()}`);
c = await consistent();
if (c.mask !== c.count || c.count === 0 || c.count >= rowsLoaded) await fail(`viewer brush: mask ${c.mask}, count ${c.count}`);
if (shots) {
  await page.screenshot({ path: `${shots}/viewer.png`, fullPage: true });
  console.log(`screenshot: ${shots}/viewer.png`);
}

// A broken spec must fail visibly, with the offending path.
await page.goto("http://localhost:4317/?spec=/does-not-exist.spec.json");
await page.waitForFunction(() => document.body.dataset.state !== "loading", null, { timeout: 30_000 });
if ((await page.evaluate(() => document.body.dataset.state)) !== "error") await fail("a missing spec did not report an error");
problems.length = 0; // the 404 above is expected

if (problems.length) await fail("errors were logged");
console.log("PASS");
await browser.close();
await server.close();
