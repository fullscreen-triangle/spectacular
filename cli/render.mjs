// Renders a spec to a PNG or PDF file through the built viewer in headless Chromium.
import { writeFile } from "node:fs/promises";
import { extname } from "node:path";
import { startServer } from "./server.mjs";

const CSS_PX = { px: 1, mm: 96 / 25.4, in: 96 };
// Chromium without a GPU draws WebGL through SwiftShader.
const BROWSER_ARGS = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"];

/**
 * Size comes from the flags, else the spec's output.figure, else 960 px wide; each flag
 * overrides one value. dpi defaults to 300 for mm and in, and to 2x (192) for px.
 * Returns { file, format, width, height, pixels? } in CSS px; throws with `exit` set.
 */
export async function render({ spec, specFile, specText, out, width, unit, dpi, timeout = 120_000 }) {
  const format = extname(out).slice(1).toLowerCase();
  if (format !== "png" && format !== "pdf")
    throw Object.assign(new Error(`cannot write "${out}": the output must end in .png or .pdf`), { exit: 2 });
  const fig = spec.output?.figure ?? {};
  const u = unit ?? fig.unit ?? "px";
  if (!(u in CSS_PX)) throw Object.assign(new Error(`--unit must be mm, in or px, not "${u}"`), { exit: 2 });
  const cssWidth = Math.round((width ?? fig.width ?? 960) * CSS_PX[u]);
  const scale = (dpi ?? fig.dpi ?? (u === "px" ? 192 : 300)) / 96;

  const { chromium } = await import("playwright").catch(() => {
    throw Object.assign(new Error("rendering needs Playwright: run `npm install` and `npx playwright install chromium`"), { exit: 2 });
  });
  const server = await startServer({ specFile, specText });
  let browser;
  try {
    browser = await chromium.launch({ args: BROWSER_ARGS });
    const page = await browser.newPage({ viewport: { width: cssWidth, height: 800 }, deviceScaleFactor: scale, colorScheme: "light" });
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(`${server.origin}/?figure&spec=${server.specPath}`);
    await page.waitForFunction(() => document.body.dataset.state && document.body.dataset.state !== "loading", null, { timeout });
    if ((await page.evaluate(() => document.body.dataset.state)) === "error")
      throw Object.assign(new Error((await page.textContent("#error")) || "the viewer reported an error"), { exit: 1 });
    // Let every view draw its first frame, then fix WebGL output as images so a PDF
    // keeps it (printing re-lays out the page, and a cleared canvas would print blank).
    await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
    if (errors.length) throw Object.assign(new Error(errors.join("\n")), { exit: 3 });
    const app = page.locator("#app");
    const box = await app.boundingBox();
    const height = Math.ceil(box.height);
    if (format === "png") {
      await writeFile(out, await app.screenshot({ type: "png" }));
      return { file: out, format, width: cssWidth, height, pixels: [Math.round(box.width * scale), Math.round(box.height * scale)] };
    }
    await page.evaluate(() => {
      for (const c of document.querySelectorAll("#app canvas")) {
        const img = document.createElement("img");
        img.src = c.toDataURL("image/png");
        img.style.cssText = c.style.cssText; // keeps image-rendering and any positioning
        Object.assign(img.style, { width: `${c.clientWidth}px`, height: `${c.clientHeight}px`, display: "block" });
        c.replaceWith(img);
      }
    });
    await page.emulateMedia({ media: "screen" });
    await page.pdf({ path: out, width: `${cssWidth}px`, height: `${height}px`, printBackground: true, pageRanges: "1",
      margin: { top: 0, right: 0, bottom: 0, left: 0 } });
    return { file: out, format, width: cssWidth, height };
  } catch (e) {
    if (e.exit === undefined) e.exit = 3;
    throw e;
  } finally {
    await browser?.close();
    await server.close();
  }
}
