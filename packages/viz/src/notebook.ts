/**
 * The single-file build used from Python: an anywidget front end for Jupyter, and the
 * boot code for the standalone HTML pages `spectacular.to_html` writes. The crossfilter
 * worker and its WASM module are inlined, so this one file is the whole runtime.
 */
import type { ExternalColumn } from "./data";
import InlineWorker from "./engine/worker-inline.ts?worker&inline";
import { localWorker } from "./engine/local";
import { ensureStyle, mount, type Handle } from "./mount";
import { SpecError, type Problem, type Spec, type Theme } from "./spec";

type Binary = DataView | ArrayBuffer | Uint8Array;

/** A column as Python sends it: little-endian float64 values, or uint32 codes with labels. */
export type WireColumn =
  | { type: "quantitative"; values: Binary }
  | { type: "nominal"; codes: Binary; labels: string[] };

export interface Payload {
  spec: Spec;
  /** Columns from a DataFrame, dict or rows; they replace the spec's data. */
  columns?: Record<string, WireColumn> | null;
  /** The bytes of the local file the spec's data.url names. */
  file?: { name: string; bytes: Binary } | null;
}

/** A copy in a buffer of its own, so typed arrays over it are aligned. */
const buffer = (b: Binary): ArrayBuffer =>
  b instanceof ArrayBuffer ? b : (b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer);

function decode(columns: Record<string, WireColumn>): Record<string, ExternalColumn> {
  const out: Record<string, ExternalColumn> = {};
  for (const [name, c] of Object.entries(columns))
    out[name] = c.type === "quantitative"
      ? new Float64Array(buffer(c.values))
      : { codes: new Uint32Array(buffer(c.codes)), labels: c.labels };
  return out;
}

/** The notebook's own theme, so a dark JupyterLab or VS Code gets dark views. */
export function hostTheme(): Theme {
  const body = document.body;
  const lab = body?.dataset.jpThemeLight;
  if (lab === "false") return "dark";
  if (lab === "true") return "light";
  if (body?.classList.contains("vscode-dark") || body?.classList.contains("vscode-high-contrast")) return "dark";
  if (body?.classList.contains("vscode-light")) return "light";
  if (document.documentElement.getAttribute("theme") === "dark") return "dark"; // Colab
  return globalThis.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function problemBox(problems: Problem[]): HTMLElement {
  ensureStyle();
  const root = document.createElement("div");
  root.className = "sv-root";
  root.dataset.theme = hostTheme();
  const box = document.createElement("div");
  box.className = "sv-problems";
  const head = document.createElement("b");
  head.textContent = `The charts were not drawn: ${problems.length === 1 ? "1 problem" : `${problems.length} problems`}`;
  box.append(head);
  for (const p of problems) {
    const line = document.createElement("div");
    const at = document.createElement("code");
    at.textContent = p.path || "/";
    line.append(at, document.createTextNode(` ${p.message}`));
    box.append(line);
  }
  root.append(box);
  return root;
}

/** Mounts a payload into `el`, or shows its problems there. */
export async function show(el: HTMLElement, payload: Payload): Promise<{ handle?: Handle; problems: Problem[] }> {
  let spec = payload.spec;
  let columns: Record<string, ExternalColumn> | undefined;
  let fetcher: typeof fetch | undefined;
  if (payload.columns) {
    spec = { ...spec, data: { columns: "external" } };
    columns = decode(payload.columns);
  } else if (payload.file) {
    const body = buffer(payload.file.bytes);
    fetcher = async () => new Response(body.slice(0));
  }
  const container = document.createElement("div");
  el.replaceChildren(container);
  let worker: Worker;
  try {
    worker = new InlineWorker();
  } catch {
    worker = localWorker();
  }
  try {
    const handle = await mount(container, spec, { columns, fetch: fetcher, worker, fallbackWorker: localWorker, theme: hostTheme() });
    return { handle, problems: [] };
  } catch (e) {
    worker.terminate();
    const problems = e instanceof SpecError ? e.problems : [{ path: "", message: e instanceof Error ? e.message : String(e) }];
    el.replaceChildren(problemBox(problems));
    return { problems };
  }
}

/** The part of anywidget's model this module uses. */
interface Model {
  get(name: string): any;
  set(name: string, value: unknown): void;
  save_changes(): void;
  on(event: string, callback: (...args: any[]) => void): void;
  off(event?: string, callback?: (...args: any[]) => void): void;
  send(content: unknown): void;
}

/**
 * anywidget's entry point. It draws the payload, publishes each settled selection to the
 * Python side (`selection`, and the selected row indices in `_rows`), redraws when the
 * spec or data change, and answers requests for an SVG of what is shown.
 */
function render({ model, el }: { model: Model; el: HTMLElement }) {
  let handle: Handle | undefined;
  let generation = 0;
  let timer = 0;
  let queued = false;

  const publish = () => {
    if (!handle) return;
    model.set("selection", handle.selection());
    if (model.get("track_rows")) model.set("_rows", new DataView(handle.selectedRows().buffer));
    model.save_changes();
  };
  const draw = async () => {
    const mine = ++generation;
    handle?.destroy();
    handle = undefined;
    // A description Python's compiler refused arrives as problems with no spec.
    const refused = model.get("problems") as Problem[] | undefined;
    if (!model.get("spec")?.views && refused?.length) {
      el.replaceChildren(problemBox(refused));
      return;
    }
    const result = await show(el, { spec: model.get("spec"), columns: model.get("columns"), file: model.get("file") });
    if (mine !== generation) return result.handle?.destroy();
    handle = result.handle;
    model.set("problems", result.problems);
    if (handle) {
      handle.onUpdate(() => {
        clearTimeout(timer);
        timer = window.setTimeout(publish, 150);
      });
      publish();
    } else model.save_changes();
  };
  // Python may change several traits at once; draw once for all of them.
  const redraw = () => {
    if (queued) return;
    queued = true;
    queueMicrotask(() => {
      queued = false;
      void draw();
    });
  };
  const onMessage = (msg: { type?: string; id?: string }) => {
    if (msg?.type === "svg") model.send({ type: "svg", id: msg.id, svg: handle ? handle.toSVG() : null });
  };
  model.on("change:spec", redraw);
  model.on("change:columns", redraw);
  model.on("change:file", redraw);
  model.on("msg:custom", onMessage);
  void draw();
  return () => {
    generation++;
    clearTimeout(timer);
    handle?.destroy();
    model.off("change:spec", redraw);
    model.off("change:columns", redraw);
    model.off("change:file", redraw);
    model.off("msg:custom", onMessage);
  };
}

/** Undoes the base64 wrapping `to_html` gives binary values: `{"$b64": "..."}`. */
function unwrap(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(unwrap);
  if (value && typeof value === "object") {
    const o = value as Record<string, unknown>;
    if (typeof o.$b64 === "string") {
      const text = atob(o.$b64);
      const bytes = new Uint8Array(text.length);
      for (let i = 0; i < text.length; i++) bytes[i] = text.charCodeAt(i);
      return bytes;
    }
    return Object.fromEntries(Object.entries(o).map(([k, v]) => [k, unwrap(v)]));
  }
  return value;
}

/**
 * Starts a standalone page written by `to_html`. Inside an iframe (how notebooks without
 * anywidget show it), the frame is resized to fit the views.
 */
export async function boot(el: HTMLElement, payload: unknown) {
  // A figure (for spectacular.render) leaves out the status line and interaction hints.
  if ((payload as { figure?: boolean })?.figure) el.classList.add("sv-figure");
  const result = await show(el, unwrap(payload) as Payload);
  let frame: HTMLElement | null = null;
  try {
    frame = window.frameElement as HTMLElement | null;
  } catch {
    // A cross-origin frame: it keeps the height its host gave it.
  }
  if (frame) {
    const fit = () => (frame!.style.height = `${document.documentElement.scrollHeight}px`);
    fit();
    new ResizeObserver(fit).observe(document.body);
  }
  (window as unknown as { spectacular?: unknown }).spectacular = result.handle;
  document.body.dataset.state = result.handle ? "ready" : "error";
  return result;
}

(globalThis as unknown as { spectacularNotebook: unknown }).spectacularNotebook = { boot, show };

export { render };
export default { render };
