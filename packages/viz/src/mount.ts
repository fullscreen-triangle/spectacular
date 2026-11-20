import { fieldTypes, loadTable, type ExternalColumn, type Table } from "./data";
import { Engine, type EngineState } from "./engine/client";
import type { DimInit, GroupInit } from "./engine/protocol";
import { SpecError, validate, type Spec, type Theme, type View } from "./spec";
import { arc } from "./views/arc";
import { bars } from "./views/bars";
import { crossPairs, edgePairs, type Allocation, type Panel, type ViewContext } from "./views/common";
import { density } from "./views/density";
import { histogram } from "./views/histogram";
import { interval } from "./views/interval";
import { layers } from "./views/layers";
import { matrix } from "./views/matrix";
import { network3d } from "./views/network3d";
import { sankey } from "./views/sankey";
import { scatter } from "./views/scatter";
import { scatter3d } from "./views/scatter3d";
import { strip } from "./views/strip";
import { STYLE } from "./style";
import { toSVG } from "./export";

export interface MountOptions {
  /** Columns for `data: { columns: "external" }`. */
  columns?: Record<string, ExternalColumn>;
  fetch?: typeof fetch;
  /** Supply a worker (e.g. from a bundler that needs its own import form). */
  worker?: Worker;
  /** Used when `worker` fails to start, e.g. a host that refuses workers. */
  fallbackWorker?: () => Worker;
  /** The theme when the spec's is "auto" or absent, e.g. one detected from a notebook. */
  theme?: Theme;
}

/**
 * One active brush, in data terms: a closed range on a quantitative field, categories
 * of a nominal one, or for an interval view the range its rows overlap (low end at or
 * below `overlaps[1]`, high end at or above `overlaps[0]`; `field` is the low end and
 * `to` the high end). A selection made from outside by key has `view` "" and the key
 * field.
 */
export interface Filter {
  view: string;
  field: string;
  range?: [number, number];
  values?: string[];
  to?: string;
  overlaps?: [number, number];
}

/** What is selected, in terms another program can apply to its own copy of the data. */
export interface Selection {
  selected: number;
  rows: number;
  filters: Filter[];
}

export interface Handle {
  readonly table: Table;
  readonly engine: Engine;
  /** Fires after every crossfilter update. */
  onUpdate(listener: (state: EngineState) => void): () => void;
  /** The brushes behind the current state. A row is selected when it passes every filter. */
  selection(): Selection;
  /** Indices (0-based, in data order) of the selected rows. */
  selectedRows(): Uint32Array;
  /** The spec's `key` values of the selected rows ([] without a key). */
  selectedKeys(): string[];
  /**
   * Selects rows from outside by their `key` values, as one more filter on top of the
   * brushes; null clears it. Returns how many of the keys the data holds. Needs a `key`.
   */
  select(keys: Iterable<string> | null): number;
  /**
   * The views as they are now, as a standalone SVG document. Text, axes, bars and 2D
   * points (up to 50,000 rows) are vectors; density grids, larger point clouds and 3D
   * views are embedded images at the page's pixel ratio.
   */
  toSVG(): string;
  destroy(): void;
}

interface Plan {
  dims: DimInit[];
  groups: GroupInit[];
  allocations: Allocation[];
  /** The dimension of the spec's key, when it has one. */
  key?: number;
}

/**
 * Each brushing view gets dimensions of its own, even on a field another view also
 * brushes. A group then ignores exactly its own view's brush, which is what makes a
 * histogram keep its full shape while it filters everything else.
 */
export function plan(spec: Spec, table: Table): Plan {
  const dims: DimInit[] = [];
  const groups: GroupInit[] = [];
  const pairDim = (codes: Uint32Array, cardinality: number): number => {
    dims.push({ kind: "categorical", codes, cardinality: Math.max(1, cardinality) });
    return dims.length - 1;
  };
  const dim = (field: string): number => {
    const c = table.columns.get(field)!;
    dims.push(c.type === "quantitative"
      ? { kind: "numeric", values: c.values }
      : { kind: "categorical", codes: c.codes, cardinality: c.labels.length });
    return dims.length - 1;
  };
  const allocations = spec.views.map((v: View): Allocation => {
    switch (v.type) {
      case "scatter":
        return (v.brush ?? "xy") === "xy" ? { dims: { [v.x]: dim(v.x), [v.y]: dim(v.y) } } : { dims: {} };
      case "histogram": {
        const d = dim(v.field);
        groups.push({ kind: "histogram", dim: d, bins: v.bins ?? 60 });
        return { dims: { [v.field]: d }, group: groups.length - 1 };
      }
      case "bars": {
        if (!v.color) {
          const d = dim(v.field);
          groups.push({ kind: "category", dim: d });
          return { dims: { [v.field]: d }, group: groups.length - 1 };
        }
        // Split bars count (field, color) pairs; a bar's filter is all its pairs.
        const cross = crossPairs(table, v.field, v.color);
        const d = pairDim(cross.codes, cross.pairs.length);
        groups.push({ kind: "category", dim: d });
        return { dims: { [v.field]: d }, group: groups.length - 1, cross };
      }
      case "strip":
        return (v.brush ?? "x") === "x" ? { dims: { [v.x]: dim(v.x) } } : { dims: {} };
      case "matrix": {
        const cross = crossPairs(table, v.x, v.y);
        const d = pairDim(cross.codes, cross.pairs.length);
        groups.push({ kind: "category", dim: d });
        return { dims: { [`${v.x} × ${v.y}`]: d }, group: groups.length - 1, cross };
      }
      case "interval":
        return (v.brush ?? "x") === "x" && v.scale !== "label" ? { dims: { [v.low]: dim(v.low), [v.high]: dim(v.high) } } : { dims: {} };
      case "density": {
        const [xBins, yBins] = v.bins ?? [128, 128];
        const x = dim(v.x), y = dim(v.y);
        groups.push({ kind: "grid", x, y, xBins, yBins });
        return { dims: { [v.x]: x, [v.y]: y }, group: groups.length - 1 };
      }
      case "scatter3d":
        return { dims: {} };
      case "arc":
      case "sankey":
      case "network3d":
      case "layers": {
        // One dimension over (source, target) pairs: a node's edges are a set of
        // pair codes, so selecting a node is one categorical filter.
        const pairs = edgePairs(table, v.source, v.target);
        const d = pairDim(pairs.codes, pairs.pairs.length);
        groups.push({ kind: "category", dim: d });
        return { dims: { [`${v.source} → ${v.target}`]: d }, group: groups.length - 1, pairs };
      }
    }
  });
  // A key has a dimension of its own, so rows can be selected from outside by key.
  const key = spec.key && table.columns.get(spec.key)?.type === "nominal" ? dim(spec.key) : undefined;
  if (dims.length > 32)
    throw new Error(`this spec needs ${dims.length} crossfilter dimensions; the limit is 32 (each brushing view uses one per brushed field)`);
  return { dims, groups, allocations, key };
}

const BUILDERS: { [K in View["type"]]: (ctx: ViewContext<Extract<View, { type: K }>>) => Panel } = {
  scatter, histogram, bars, density, scatter3d, strip, matrix, interval, arc, sankey, network3d, layers,
};

let styled = false;

/** Adds spectacular's stylesheet to the page once. */
export function ensureStyle() {
  if (styled || typeof document === "undefined") return;
  const style = document.createElement("style");
  style.textContent = STYLE;
  document.head.append(style);
  styled = true;
}

export async function mount(el: HTMLElement, spec: Spec, opts: MountOptions = {}): Promise<Handle> {
  let problems = validate(spec);
  if (problems.length) throw new SpecError(problems);
  const table = await loadTable(spec, opts.columns, opts.fetch);
  problems = validate(spec, fieldTypes(table));
  if (problems.length) throw new SpecError(problems);

  const { dims, groups, allocations, key } = plan(spec, table);
  let engine: Engine;
  try {
    engine = await Engine.start(table.rows, dims, groups, opts.worker);
  } catch (e) {
    if (!opts.fallbackWorker) throw e;
    engine = await Engine.start(table.rows, dims, groups, opts.fallbackWorker());
  }

  ensureStyle();
  el.classList.add("sv-root");
  el.dataset.theme = spec.theme && spec.theme !== "auto" ? spec.theme : opts.theme ?? "auto";
  el.replaceChildren();
  if (spec.title) {
    const h = document.createElement("div");
    h.className = "sv-heading";
    h.textContent = spec.title;
    el.append(h);
  }
  const status = document.createElement("div");
  status.className = "sv-status";
  const grid = document.createElement("div");
  grid.className = "sv-grid";
  const columns = Math.max(1, spec.layout?.columns ?? Math.min(3, spec.views.length));
  grid.style.gridTemplateColumns = `repeat(${columns}, minmax(0, 1fr))`;
  el.append(status, grid);

  const gap = 12;
  const titleOf = (field: string) => spec.fields?.[field]?.title ?? field;
  const fieldDef = (field: string) => spec.fields?.[field];
  /** The width inside the root's padding; 0 while the element is not laid out. */
  const inner = () => {
    const cs = getComputedStyle(el);
    return el.clientWidth - (parseFloat(cs.paddingLeft) || 0) - (parseFloat(cs.paddingRight) || 0);
  };
  const build = (width: number) => {
    const cell = (Math.max(320, width) - gap * (columns - 1)) / columns;
    return spec.views.map((view, i) => {
      const span = Math.min(columns, view.span ?? 1);
      const make = BUILDERS[view.type] as (ctx: ViewContext) => Panel;
      const panel = make({ view, table, engine, alloc: allocations[i]!, width: cell * span + gap * (span - 1) - 2, titleOf, fieldDef });
      panel.el.style.gridColumn = `span ${span}`;
      grid.append(panel.el);
      return panel;
    });
  };
  let builtWidth = inner() || 960;
  let panels = build(builtWidth);

  // Views are drawn for a width, so a container that changes width (a resized window, a
  // notebook output that is laid out after the views were built) gets them drawn again.
  // Brushes do not survive that, so the filters are cleared with them.
  let relayout = 0;
  const observer = typeof ResizeObserver === "undefined" ? undefined : new ResizeObserver(() => {
    const width = inner();
    if (width <= 0 || Math.abs(width - builtWidth) < 20) return;
    clearTimeout(relayout);
    relayout = window.setTimeout(() => {
      builtWidth = inner();
      // A selection made from outside (by key) is not a brush, so it stays.
      for (const dim of [...engine.filters.keys()]) if (dim !== key) engine.filter({ dim, clear: true });
      panels.forEach((p) => p.destroy());
      grid.replaceChildren();
      panels = build(builtWidth);
    }, 150);
  });
  observer?.observe(el);

  const fmt = new Intl.NumberFormat();
  const report = (s: EngineState) => {
    status.textContent = `${fmt.format(s.selectedCount)} of ${fmt.format(s.rows)} rows selected` +
      (s.elapsed ? ` · crossfilter ${s.elapsed.toFixed(1)} ms` : "");
  };
  report(engine.state);
  engine.on((s) => {
    for (const p of panels) p.update(s);
    report(s);
  });

  /** What each dimension's filter means in data terms. */
  interface Owner {
    view: string;
    field: string;
    /** Names of a pair dimension's codes, when they are not the field's own categories. */
    labels?: string[];
    /** An interval view's two dimensions, reported together as one overlap. */
    interval?: { low: string; high: string; lowDim: number; highDim: number };
  }
  const owners = new Map<number, Owner>();
  allocations.forEach((a, i) => {
    const v = spec.views[i]!;
    // Pair dimensions report categories: an edge's "source → target", a matrix cell's
    // "x × y", and a split bar its category of the bar's own field.
    const barLabels = (() => {
      if (!a.cross || v.type !== "bars") return undefined;
      const c = table.columns.get(v.field);
      return c?.type === "nominal" ? a.cross.pairs.map(([j]) => c.labels[j]!) : undefined;
    })();
    const labels = a.pairs?.labels ?? barLabels ?? a.cross?.labels;
    const interval = v.type === "interval" && a.dims[v.low] !== undefined && a.dims[v.high] !== undefined
      ? { low: v.low, high: v.high, lowDim: a.dims[v.low]!, highDim: a.dims[v.high]! }
      : undefined;
    for (const [field, d] of Object.entries(a.dims)) owners.set(d, { view: v.id, field, labels, interval });
  });
  if (key !== undefined) owners.set(key, { view: "", field: spec.key! });
  const finite = (n: number) => (Number.isFinite(n) ? n : null) as number;
  const selection = (): Selection => {
    const filters: Filter[] = [];
    const reported = new Set<string>();
    for (const [d, op] of [...engine.filters].sort(([a], [b]) => a - b)) {
      const { view, field, labels, interval } = owners.get(d)!;
      if (interval) {
        if (reported.has(view)) continue;
        reported.add(view);
        const low = engine.filters.get(interval.lowDim), high = engine.filters.get(interval.highDim);
        const top = low && "range" in low ? low.range[1] : Infinity;
        const bottom = high && "range" in high ? high.range[0] : -Infinity;
        filters.push({ view, field: interval.low, to: interval.high, overlaps: [finite(bottom), finite(top)] });
        continue;
      }
      if ("range" in op) {
        filters.push({ view, field, range: [Math.min(...op.range), Math.max(...op.range)] });
        continue;
      }
      const c = table.columns.get(field);
      const name = (k: number) => labels?.[k] ?? (c?.type === "nominal" ? c.labels[k]! : String(k));
      filters.push({ view, field, values: "codes" in op ? [...new Set(op.codes.map(name))] : [] });
    }
    return { selected: engine.state.selectedCount, rows: engine.state.rows, filters };
  };
  const keyColumn = key !== undefined ? table.columns.get(spec.key!) : undefined;
  const keyIndex = keyColumn?.type === "nominal" ? new Map(keyColumn.labels.map((l, i) => [l, i])) : undefined;

  return {
    table,
    engine,
    onUpdate: (l) => engine.on(l),
    selection,
    toSVG() {
      const vectors = new Map<Element, SVGElement>();
      for (const p of panels) {
        const v = p.vector?.();
        if (v) vectors.set(v.plot, v.content);
      }
      return toSVG(el, vectors);
    },
    selectedRows() {
      const m = engine.state.mask;
      const out = new Uint32Array(engine.state.selectedCount);
      for (let r = 0, k = 0; r < m.length; r++) if (m[r]) out[k++] = r;
      return out;
    },
    selectedKeys() {
      if (keyColumn?.type !== "nominal") return [];
      const m = engine.state.mask;
      const out: string[] = [];
      for (let r = 0; r < m.length; r++) if (m[r]) out.push(keyColumn.labels[keyColumn.codes[r]!]!);
      return out;
    },
    select(keys) {
      if (key === undefined || !keyIndex) throw new Error("select() needs a spec with a \"key\" field");
      if (keys === null) {
        engine.filter({ dim: key, clear: true });
        return 0;
      }
      const codes = [...new Set(keys)].flatMap((k) => (keyIndex.has(k) ? [keyIndex.get(k)!] : []));
      engine.filter({ dim: key, codes });
      return codes.length;
    },
    destroy() {
      observer?.disconnect();
      clearTimeout(relayout);
      engine.dispose();
      panels.forEach((p) => p.destroy());
      el.replaceChildren();
    },
  };
}
