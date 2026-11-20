// SPECTACULAR_PYTHON=<python with jupyterlab, anywidget, pandas and spectacular> node python/tests/jupyter.mjs [shots-dir]
// A real JupyterLab: run a cell that shows views, brush them in the browser, then run
// cells that read the selection back in Python and save an SVG.
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
      "df['r'] = np.hypot(df.x, df.y)\n",
      "spec = {'title': 'From a notebook', 'layout': {'columns': 2}, 'views': [\n",
      "    {'id': 'pts', 'type': 'scatter', 'x': 'x', 'y': 'y', 'color': 'group'},\n",
      "    {'id': 'r', 'type': 'histogram', 'field': 'r', 'bins': 40},\n",
      "    {'id': 'g', 'type': 'bars', 'field': 'group', 'span': 2}]}\n",
      "w = sp.view(spec, data=df)\n",
      "w",
    ]),
    code([
      "sel = w.selection\n",
      "rows = w.selected_data()\n",
      "print('SEL', sel['selected'], len(w.selected_rows), len(rows), sel['filters'][0]['view'],\n",
      "      round(sel['filters'][0]['range'][0], 6), round(sel['filters'][0]['range'][1], 6), rows['r'].min(), rows['r'].max())",
    ]),
    code(["w.save_svg('from-notebook.svg')"]),
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
  const views = page.locator(".jp-OutputArea .sv-root");
  await views.waitFor({ timeout: 60_000 });
  await page.waitForFunction(() => document.querySelector(".jp-OutputArea .sv-status")?.textContent?.includes("20"), null, { timeout: 60_000 });
  ok(true, `the views appear in the cell's output (${await page.textContent(".jp-OutputArea .sv-status")})`);
  ok(await views.getAttribute("data-theme") === "light", "they take the notebook's light theme");

  const box = await page.locator('.jp-OutputArea [data-view="r"] .sv-brush .overlay').boundingBox();
  await page.mouse.move(box.x + box.width * 0.1, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.3, box.y + box.height / 2, { steps: 6 });
  await page.mouse.up();
  await page.waitForTimeout(1200); // the page publishes 150 ms after the brush settles
  const status = await page.textContent(".jp-OutputArea .sv-status");
  const shown = Number(status.split(" ")[0].replace(/[^\d]/g, ""));
  if (shots) await page.screenshot({ path: join(shots, "jupyter-light.png"), fullPage: true });

  await run(1);
  const text = await output(1);
  const [, selected, indices, sliced, view, lo, hi, min, max] = text.trim().split(/\s+/);
  ok(Number(selected) === shown && shown > 0 && shown < 20_000, `Python sees the brushed count the page shows (${selected} of 20000)`);
  ok(Number(indices) === shown && Number(sliced) === shown, "selected_rows and selected_data() hold exactly those rows");
  ok(view === "r" && +min >= +lo && +max <= +hi, `every selected row lies in the reported range (r in [${lo}, ${hi}], data ${(+min).toFixed(3)}..${(+max).toFixed(3)})`);

  await run(2);
  const svgFile = join(dir, "from-notebook.svg");
  for (let i = 0; i < 40 && !existsSync(svgFile); i++) await page.waitForTimeout(250);
  const svg = existsSync(svgFile) ? readFileSync(svgFile, "utf8") : "";
  ok(svg.includes("<svg") && (svg.match(/<circle /g) ?? []).length === 20_000 && svg.includes("From a notebook"),
    "save_svg writes what the view shows: 20000 vector points and the title");

  // A dark JupyterLab gets dark views.
  await page.evaluate(() => window.jupyterapp.commands.execute("apputils:change-theme", { theme: "JupyterLab Dark" }));
  await page.waitForFunction(() => document.body.dataset.jpThemeLight === "false", null, { timeout: 30_000 });
  await run(0);
  await page.waitForFunction(() => document.querySelector(".jp-OutputArea .sv-root")?.dataset.theme === "dark", null, { timeout: 60_000 });
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
