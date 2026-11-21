// SPECTACULAR_PYTHON=<python with jupyterlab, anywidget, pandas and spectacular> node python/tests/jupyter.mjs [shots-dir]
// A real JupyterLab: a cell that is only a JSON description (%%spectacular) shows linked
// views; brush them in the browser, then run cells that read the selection back in
// Python and save an SVG; a misspelt column shows its problem in the cell.
import { spawn, spawnSync } from "node:child_process";
import { randomBytes } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { chromium } from "playwright";

const python = process.env.SPECTACULAR_PYTHON ?? "python";
const shots = process.argv[2];
const dir = mkdtempSync(join(tmpdir(), "spectacular-jupyter-"));
const token = randomBytes(16).toString("hex");
const port = await new Promise((ok) => {
  const s = createServer().listen(0, "127.0.0.1", () => {
    const p = s.address().port;
    s.close(() => ok(p));
  });
});
let failures = 0;
const ok = (cond, what) => {
  console.log(`${cond ? "ok  " : "FAIL"} ${what}`);
  if (!cond) failures++;
};

const code = (source) => ({ cell_type: "code", metadata: {}, execution_count: null, outputs: [], source });
writeFileSync(join(dir, "demo.ipynb"), JSON.stringify({
  nbformat: 4, nbformat_minor: 5, metadata: { kernelspec: { name: "python3", display_name: "Python 3", language: "python" } },
  cells: [
    code([
      "import numpy as np, pandas as pd, spectacular as sp\n",
      "rng = np.random.default_rng(1)\n",
      "n = 20_000\n",
      "df = pd.DataFrame({'x': rng.normal(size=n), 'y': rng.normal(size=n), 'group': rng.choice(['a', 'b', 'c'], n)})\n",
      "df['r'] = np.hypot(df.x, df.y)",
    ]),
    code([
      "%%spectacular df --as w\n",
      "{\"title\": \"From a notebook\", \"columns\": 2,\n",
      " \"charts\": [\n",
      "   {\"x-axis\": \"X\", \"y-axis\": \"Y\", \"colour\": \"group\", \"palette\": [\"teal\", \"orange\", \"crimson\"]},\n",
      "   {\"x\": \"r\", \"bins\": 40},\n",
      "   {\"x\": \"group\", \"wide\": true}\n",
      " ]}",
    ]),
    code([
      "sel = w.selection\n",
      "rows = w.selected_data()\n",
      "print('SEL', sel['selected'], len(w.selected_rows), len(rows), sel['filters'][0]['view'],\n",
      "      round(sel['filters'][0]['range'][0], 6), round(sel['filters'][0]['range'][1], 6), rows['r'].min(), rows['r'].max())",
    ]),
    code(["w.save_svg('from-notebook.svg')"]),
    code(["%%spectacular df\n", "{\"charts\": [{\"x\": \"grop\"}]}"]),
    // render from inside the kernel: on Windows ipykernel sets the selector event loop,
    // under which 0.2.0 could not start the browser.
    code([
      "try:\n",
      "    import playwright.async_api\n",
      "except ImportError:\n",
      "    print('RENDER skipped: no playwright in this kernel')\n",
      "else:\n",
      "    out = sp.render({'x': 'r', 'bins': 30}, 'rendered.png', data=df)\n",
      "    print('RENDER', out.stat().st_size)",
    ]),
  ],
}));

const lab = spawn(python, ["-m", "jupyter", "lab", "--no-browser", `--port=${port}`, `--ServerApp.token=${token}`,
  `--ServerApp.root_dir=${dir}`, "--LabApp.expose_app_in_browser=True", "--ServerApp.ip=127.0.0.1"],
  {
    cwd: dir, stdio: ["ignore", "ignore", "pipe"],
    // Settings and workspaces of its own, so the theme switch below never touches the user's.
    env: { ...process.env, JUPYTERLAB_SETTINGS_DIR: join(dir, ".settings"), JUPYTERLAB_WORKSPACES_DIR: join(dir, ".workspaces") },
  });
let labLog = "";
lab.stderr.on("data", (d) => (labLog += d));
const base = `http://127.0.0.1:${port}`;
for (let i = 0; ; i++) {
  try {
    if ((await fetch(`${base}/api/status?token=${token}`)).ok) break;
  } catch {}
  if (i > 120) throw new Error(`JupyterLab did not start:\n${labLog}`);
  await new Promise((r) => setTimeout(r, 500));
}

const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
try {
  const page = await browser.newPage({ viewport: { width: 1300, height: 1100 } });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(`${base}/lab/workspaces/spectacular-test/tree/demo.ipynb?reset&token=${token}`);
  await page.waitForFunction(() => window.jupyterapp?.shell, null, { timeout: 120_000 });
  await page.evaluate(() => {
    window.__nb = () => [...window.jupyterapp.shell.widgets("main")].find((w) => w.context?.path === "demo.ipynb");
  });
  try {
    await page.waitForFunction(() => window.__nb()?.context?.sessionContext?.session?.kernel, null, { timeout: 120_000 });
  } catch (e) {
    const state = await page.evaluate(() => ({
      app: typeof window.jupyterapp,
      widget: window.__nb()?.title?.label,
      session: !!window.__nb()?.context?.sessionContext?.session,
      dialog: document.querySelector(".jp-Dialog")?.innerText,
    }));
    if (shots) await page.screenshot({ path: join(shots, "jupyter-stuck.png") });
    throw new Error(`no kernel: ${JSON.stringify(state)}\n${labLog.slice(-2000)}`);
  }
  await page.waitForFunction(() => window.__nb().context.sessionContext.session.kernel.status === "idle", null, { timeout: 120_000 });
  // Cells marked by index, so each check looks only at its own cell's output.
  const tag = () => page.evaluate(() => window.__nb().content.widgets.forEach((c, i) => (c.node.dataset.svCell = String(i))));
  await tag();
  const inCell = (i, selector) => page.locator(`[data-sv-cell="${i}"] ${selector}`);

  const run = async (index) => {
    await page.evaluate((i) => {
      const panel = window.__nb();
      window.jupyterapp.shell.activateById(panel.id);
      panel.content.activeCellIndex = i;
      return window.jupyterapp.commands.execute("notebook:run-cell");
    }, index);
    await page.waitForFunction((i) => {
      const cell = window.__nb().content.widgets[i];
      return cell.model.executionCount != null &&
        window.__nb().context.sessionContext.session.kernel.status === "idle";
    }, index, { timeout: 120_000 });
  };
  const output = (index) => page.evaluate((i) => window.__nb().content.widgets[i].node.querySelector(".jp-OutputArea")?.innerText ?? "", index);

  await run(0);
  await run(1);
  const views = inCell(1, ".sv-root");
  await views.waitFor({ timeout: 60_000 });
  await page.waitForFunction(() => document.querySelector('[data-sv-cell="1"] .sv-status')?.textContent?.includes("20"), null, { timeout: 60_000 });
  const kinds = await inCell(1, "[data-view]").evaluateAll((els) => els.map((e) => e.dataset.view));
  ok(kinds.join() === "y-vs-x,r,group", `a description cell shows its charts, types chosen from the columns (${kinds.join(", ")}; ${await inCell(1, ".sv-status").textContent()})`);
  ok(await views.getAttribute("data-theme") === "light", "they take the notebook's light theme");
  const swatch = await inCell(1, '[data-view="y-vs-x"] .sv-swatch').first().evaluate((e) => getComputedStyle(e).backgroundColor);
  ok(swatch === "rgb(0, 128, 128)", `the palette it names colours the categories (${swatch})`);

  const box = await inCell(1, '[data-view="r"] .sv-brush .overlay').boundingBox();
  await page.mouse.move(box.x + box.width * 0.1, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.3, box.y + box.height / 2, { steps: 6 });
  await page.mouse.up();
  await page.waitForTimeout(1200); // the page publishes 150 ms after the brush settles
  const status = await inCell(1, ".sv-status").textContent();
  const shown = Number(status.split(" ")[0].replace(/[^\d]/g, ""));
  if (shots) await page.screenshot({ path: join(shots, "jupyter-light.png"), fullPage: true });

  await run(2);
  const text = await output(2);
  const [, selected, indices, sliced, view, lo, hi, min, max] = text.trim().split(/\s+/);
  ok(Number(selected) === shown && shown > 0 && shown < 20_000, `Python sees the brushed count the page shows (${selected} of 20000)`);
  ok(Number(indices) === shown && Number(sliced) === shown, "selected_rows and selected_data() hold exactly those rows");
  ok(view === "r" && +min >= +lo && +max <= +hi, `every selected row lies in the reported range (r in [${lo}, ${hi}], data ${(+min).toFixed(3)}..${(+max).toFixed(3)})`);

  await run(3);
  const svgFile = join(dir, "from-notebook.svg");
  for (let i = 0; i < 40 && !existsSync(svgFile); i++) await page.waitForTimeout(250);
  const svg = existsSync(svgFile) ? readFileSync(svgFile, "utf8") : "";
  ok(svg.includes("<svg") && (svg.match(/<circle /g) ?? []).length === 20_000 && svg.includes("From a notebook"),
    "save_svg writes what the view shows: 20000 vector points and the title");

  await run(4);
  await inCell(4, ".sv-problems").waitFor({ timeout: 30_000 });
  const problem = await inCell(4, ".sv-problems").innerText();
  ok(problem.includes("/charts/0/x") && problem.includes('did you mean "group"'), `a misspelt column shows its problem in the cell (${problem.replace(/\s+/g, " ").trim()})`);

  await run(5);
  const rendered = (await output(5)).trim();
  if (rendered.startsWith("RENDER skipped")) console.log(`skip ${rendered}`);
  else ok(/^RENDER \d{4,}$/.test(rendered), `sp.render draws a figure from inside the kernel (${rendered.slice(0, 160)})`);

  // A dark JupyterLab gets dark views.
  await page.evaluate(() => window.jupyterapp.commands.execute("apputils:change-theme", { theme: "JupyterLab Dark" }));
  await page.waitForFunction(() => document.body.dataset.jpThemeLight === "false", null, { timeout: 30_000 });
  await run(1);
  await page.waitForFunction(() => document.querySelector('[data-sv-cell="1"] .sv-root')?.dataset.theme === "dark", null, { timeout: 60_000 });
  ok(true, "after switching JupyterLab to dark, the views draw dark");
  if (shots) await page.screenshot({ path: join(shots, "jupyter-dark.png"), fullPage: true });
  await page.evaluate(() => window.jupyterapp.commands.execute("apputils:change-theme", { theme: "JupyterLab Light" }));
  ok(errors.length === 0, `no page errors${errors.length ? ": " + errors.join("; ") : ""}`);
} catch (e) {
  ok(false, e.stack);
} finally {
  await browser.close();
  // Shutting down through the API stops the kernels too; killing the process would not
  // on Windows, where children outlive their parent.
  const exited = new Promise((r) => lab.once("exit", r));
  await fetch(`${base}/api/shutdown?token=${token}`, { method: "POST" }).catch(() => {});
  if (await Promise.race([exited.then(() => true), new Promise((r) => setTimeout(() => r(false), 15_000))]) === false) {
    if (process.platform === "win32") spawnSync("taskkill", ["/pid", String(lab.pid), "/T", "/F"]);
    else lab.kill();
  }
  await new Promise((r) => setTimeout(r, 1000));
  try {
    rmSync(dir, { recursive: true, force: true, maxRetries: 10, retryDelay: 300 });
  } catch {
    console.log(`(left ${dir}: the kernel still held it)`);
  }
}
console.log(failures ? `${failures} failed` : "PASS");
process.exit(failures ? 1 : 0);
