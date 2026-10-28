import { fieldTypes, loadTable, type ExternalColumn, type Table } from "./data";
import { Engine, type EngineState } from "./engine/client";
import type { DimInit, GroupInit } from "./engine/protocol";
import { SpecError, validate, type Spec, type View } from "./spec";
import { bars } from "./views/bars";
import type { Allocation, Panel, ViewContext } from "./views/common";
import { density } from "./views/density";
import { histogram } from "./views/histogram";
import { scatter } from "./views/scatter";
import { scatter3d } from "./views/scatter3d";
import { STYLE } from "./style";
import { toSVG } from "./export";

export interface MountOptions {
  /** Columns for `data: { columns: "external" }`. */
  columns?: Record<string, ExternalColumn>;
  fetch?: typeof fetch;
  /** Supply a worker (e.g. from a bundler that needs its own import form). */
  worker?: Worker;
}

/** One active brush, in data terms: a closed range on a quantitative field, or categories. */
export interface Filter {
  view: string;
  field: string;
  range?: [number, number];
  values?: string[];
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
}

/**
 * Each brushing view gets dimensions of its own, even on a field another view also
 * brushes. A group then ignores exactly its own view's brush, which is what makes a
 * histogram keep its full shape while it filters everything else.
 */
export function plan(spec: Spec, table: Table): Plan {
  const dims: DimInit[] = [];
  const groups: GroupInit[] = [];
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
        const d = dim(v.field);
        groups.push({ kind: "category", dim: d });
        return { dims: { [v.field]: d }, group: groups.length - 1 };
      }
      case "density": {
        const [xBins, yBins] = v.bins ?? [128, 128];
        const x = dim(v.x), y = dim(v.y);
        groups.push({ kind: "grid", x, y, xBins, yBins });
        return { dims: { [v.x]: x, [v.y]: y }, group: groups.length - 1 };
      }
      case "scatter3d":
        return { dims: {} };
    }
  });
  if (dims.length > 32)
    throw new Error(`this spec needs ${dims.length} crossfilter dimensions; the limit is 32 (each brushing view uses one per brushed field)`);
  return { dims, groups, allocations };
}

const BUILDERS: { [K in View["type"]]: (ctx: ViewContext<Extract<View, { type: K }>>) => Panel } = {
  scatter, histogram, bars, density, scatter3d,
};

let styled = false;

export async function mount(el: HTMLElement, spec: Spec, opts: MountOptions = {}): Promise<Handle> {
  let problems = validate(spec);
  if (problems.length) throw new SpecError(problems);
  const table = await loadTable(spec, opts.columns, opts.fetch);
  problems = validate(spec, fieldTypes(table));
  if (problems.length) throw new SpecError(problems);

  const { dims, groups, allocations } = plan(spec, table);
  const engine = await Engine.start(table.rows, dims, groups, opts.worker);

  if (!styled && typeof document !== "undefined") {
    const style = document.createElement("style");
    style.textContent = STYLE;
    document.head.append(style);
    styled = true;
  }
  el.classList.add("sv-root");
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
  const cell = (Math.max(320, el.clientWidth || 960) - gap * (columns - 1)) / columns;
  const titleOf = (field: string) => spec.fields?.[field]?.title ?? field;
  const panels = spec.views.map((view, i) => {
    const span = Math.min(columns, view.span ?? 1);
    const build = BUILDERS[view.type] as (ctx: ViewContext) => Panel;
    const panel = build({ view, table, engine, alloc: allocations[i]!, width: cell * span + gap * (span - 1) - 2, titleOf });
    panel.el.style.gridColumn = `span ${span}`;
    grid.append(panel.el);
    return panel;
  });

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

  const owners = new Map<number, { view: string; field: string }>();
  allocations.forEach((a, i) => {
    for (const [field, d] of Object.entries(a.dims)) owners.set(d, { view: spec.views[i]!.id, field });
  });
  const selection = (): Selection => ({
    selected: engine.state.selectedCount,
    rows: engine.state.rows,
    filters: [...engine.filters].sort(([a], [b]) => a - b).map(([d, op]): Filter => {
      const owner = owners.get(d)!;
      if ("range" in op) return { ...owner, range: [Math.min(...op.range), Math.max(...op.range)] };
      const c = table.columns.get(owner.field)!;
      return { ...owner, values: "codes" in op && c.type === "nominal" ? op.codes.map((k) => c.labels[k]!) : [] };
    }),
  });

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
    destroy() {
      engine.dispose();
      panels.forEach((p) => p.destroy());
      el.replaceChildren();
    },
  };
}
