import { axisBottom, axisLeft, format, interpolateViridis, schemeTableau10, select, type NumberValue, type ScaleLinear, type Selection } from "d3";
import type { Column, Table } from "../data";
import type { Engine, EngineState } from "../engine/client";
import type { View } from "../spec";

export interface Panel {
  readonly el: HTMLElement;
  update(state: EngineState): void;
  /**
   * Vector content for SVG export, replacing the canvases in `plot`, in its coordinates.
   * Views without it are exported as images of their canvases.
   */
  vector?(): { plot: HTMLElement; content: SVGElement } | undefined;
  destroy(): void;
}

/** What a view was given by the planner: its crossfilter dimensions and groups. */
export interface Allocation {
  dims: Record<string, number>;
  group?: number;
  /** For edge views: the pairs their one dimension filters on. */
  pairs?: Pairs;
}

export interface ViewContext<V extends View = View> {
  view: V;
  table: Table;
  engine: Engine;
  alloc: Allocation;
  width: number;
  titleOf(field: string): string;
}

// Room for tick labels plus an axis title on the bottom and left.
export const MARGIN = { top: 10, right: 16, bottom: 42, left: 62 };
export const PALETTE = schemeTableau10;

export interface Frame {
  root: HTMLDivElement;
  /** The plot area: views place canvases here, under the SVG overlay. */
  plot: HTMLDivElement;
  svg: Selection<SVGSVGElement, unknown, null, undefined>;
  /** A group translated to the plot area's origin. */
  g: Selection<SVGGElement, unknown, null, undefined>;
  /** The same origin in an SVG layer under the plot's canvases, for gridlines. */
  under: Selection<SVGGElement, unknown, null, undefined>;
  w: number;
  h: number;
}

export function frame(ctx: ViewContext, height = 300): Frame {
  const root = document.createElement("div");
  root.className = "sv-panel";
  root.dataset.view = ctx.view.id;
  const title = document.createElement("div");
  title.className = "sv-title";
  title.textContent = ctx.view.title ?? ctx.view.id;
  root.append(title);

  const stage = document.createElement("div");
  stage.className = "sv-stage";
  stage.style.height = `${height + MARGIN.top + MARGIN.bottom}px`;
  root.append(stage);

  const w = Math.max(80, ctx.width - MARGIN.left - MARGIN.right);
  const h = height;
  const plot = document.createElement("div");
  plot.className = "sv-plot";
  Object.assign(plot.style, { left: `${MARGIN.left}px`, top: `${MARGIN.top}px`, width: `${w}px`, height: `${h}px` });
  const under = select(stage)
    .append("svg")
    .attr("class", "sv-underlay")
    .attr("width", w + MARGIN.left + MARGIN.right)
    .attr("height", h + MARGIN.top + MARGIN.bottom)
    .append("g")
    .attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);
  stage.append(plot);

  const svg = select(stage)
    .append("svg")
    .attr("class", "sv-overlay")
    .attr("width", w + MARGIN.left + MARGIN.right)
    .attr("height", h + MARGIN.top + MARGIN.bottom);
  const g = svg.append("g").attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);
  return { root, plot, svg, g, under, w, h };
}

/**
 * The count axis follows the filtered counts. The unfiltered context is drawn only when
 * the axis can hold it; clipped context bars would read as false totals.
 */
export function elastic(focus: number, context: number): { max: number; context: boolean } {
  if (context <= Math.max(focus, 1) * 1.25) return { max: Math.max(context, focus, 1), context: true };
  return { max: Math.max(focus, 1), context: false };
}

let clips = 0;

/** Adds a clip path the size of the plot area and returns its id. */
export function clip(f: Pick<Frame, "svg" | "w" | "h">): string {
  const id = `sv-clip-${++clips}`;
  f.svg.append("defs").append("clipPath").attr("id", id).append("rect").attr("width", f.w).attr("height", f.h);
  return id;
}

/** Axis titles, centred along each axis, outside the tick labels. */
export function axisLabels(f: Frame, x: string, y: string) {
  f.g.append("text").attr("class", "sv-axis-label").attr("x", f.w / 2).attr("y", f.h + 36).attr("text-anchor", "middle").text(x);
  f.g
    .append("text")
    .attr("class", "sv-axis-label")
    .attr("transform", "rotate(-90)")
    .attr("x", -f.h / 2)
    .attr("y", -MARGIN.left + 14)
    .attr("text-anchor", "middle")
    .text(y);
}

/**
 * Tick labels a reader can take in at a glance: integers without decimals, values from
 * 10,000 up with SI prefixes (12k, 3.4M), and otherwise the fewest digits that tell ticks apart.
 */
export function tickFormat(scale: ScaleLinear<number, number>, count: number): (d: NumberValue) => string {
  const ticks = scale.ticks(count);
  const largest = Math.max(0, ...ticks.map(Math.abs));
  const f = largest >= 1e4 ? format("~s") : ticks.every(Number.isInteger) ? format("d") : scale.tickFormat(count);
  return (d) => f(+d);
}

const xTicks = (w: number) => Math.max(2, Math.round(w / 90));
const yTicks = (h: number) => Math.max(2, Math.round(h / 45));

/** A bottom axis with light vertical gridlines across the plot area. */
export function xAxis(f: Frame, scale: ScaleLinear<number, number>, grid = true) {
  const n = xTicks(f.w);
  if (grid)
    f.under.append("g").attr("class", "sv-gridlines").attr("transform", `translate(0,${f.h})`)
      .call(axisBottom(scale).ticks(n).tickSize(-f.h).tickFormat(() => "")).call((g) => g.select(".domain").remove());
  f.g.append("g").attr("class", "sv-axis").attr("transform", `translate(0,${f.h})`)
    .call(axisBottom(scale).ticks(n).tickFormat(tickFormat(scale, n)).tickSizeOuter(0).tickPadding(6));
}

/**
 * A left axis with horizontal gridlines. It returns a redraw for views whose domain
 * follows the data (histogram counts), which call it after changing the scale.
 */
export function yAxis(f: Frame, scale: ScaleLinear<number, number>, count = yTicks(f.h)): () => void {
  const grid = f.under.append("g").attr("class", "sv-gridlines");
  const axis = f.g.append("g").attr("class", "sv-axis");
  const draw = () => {
    grid.call(axisLeft(scale).ticks(count).tickSize(-f.w).tickFormat(() => "")).call((g) => g.select(".domain").remove());
    axis.call(axisLeft(scale).ticks(count).tickFormat(tickFormat(scale, count)).tickSizeOuter(0).tickPadding(6));
  };
  draw();
  return draw;
}

/**
 * The colour key for a point view coloured by `field`: swatches for a nominal field, a
 * viridis bar for a quantitative one. It is HTML and inline SVG, so SVG export keeps it.
 */
export function colorKey(table: Table, field: string | undefined, title: string): HTMLElement | null {
  if (!field) return null;
  const c = table.columns.get(field);
  if (!c) return null;
  const key = document.createElement("div");
  key.className = "sv-key";
  const name = document.createElement("span");
  name.className = "sv-key-title";
  name.textContent = title;
  key.append(name);
  if (c.type === "nominal") {
    const shown = c.labels.slice(0, 12);
    shown.forEach((label, i) => {
      const item = document.createElement("span");
      item.className = "sv-key-item";
      const sw = document.createElement("span");
      sw.className = "sv-swatch";
      sw.style.backgroundColor = PALETTE[i % PALETTE.length]!;
      item.append(sw, document.createTextNode(label === "" ? "(empty)" : label));
      key.append(item);
    });
    if (c.labels.length > shown.length) {
      const more = document.createElement("span");
      more.className = "sv-key-item";
      more.textContent = `+${c.labels.length - shown.length} more`;
      key.append(more);
    }
  } else {
    const fmt = format(".3~g");
    key.append(document.createTextNode(fmt(c.extent[0])), gradient(interpolateViridis), document.createTextNode(fmt(c.extent[1])));
  }
  return key;
}

let gradients = 0;

/** A small horizontal colour ramp as inline SVG. */
export function gradient(interpolate: (t: number) => string, width = 96, height = 8): SVGSVGElement {
  const id = `sv-ramp-${++gradients}`;
  const svg = select(document.createElementNS("http://www.w3.org/2000/svg", "svg"))
    .attr("class", "sv-ramp").attr("width", width).attr("height", height);
  svg.append("defs").append("linearGradient").attr("id", id)
    .selectAll("stop").data([0, 0.25, 0.5, 0.75, 1]).join("stop")
    .attr("offset", (t) => `${t * 100}%`).attr("stop-color", (t) => interpolate(t));
  svg.append("rect").attr("width", width).attr("height", height).attr("rx", 2).attr("fill", `url(#${id})`);
  return svg.node()!;
}

/** Puts a key under a panel's title. */
export function addKey(root: HTMLElement, key: HTMLElement | null) {
  if (key) root.insertBefore(key, root.children[1] ?? null);
}

export function numeric(table: Table, field: string): Extract<Column, { type: "quantitative" }> {
  const c = table.columns.get(field);
  if (!c || c.type !== "quantitative") throw new Error(`field "${field}" is not quantitative`);
  return c;
}

export function categorical(table: Table, field: string): Extract<Column, { type: "nominal" }> {
  const c = table.columns.get(field);
  if (!c || c.type !== "nominal") throw new Error(`field "${field}" is not nominal`);
  return c;
}

/** Points stay readable at any row count: alpha falls as density rises. */
export function pointAlpha(rows: number): number {
  return Math.min(0.85, Math.max(0.02, 40_000 / Math.max(rows, 1)));
}

/** A small selection is drawn larger, so a few hundred brushed rows stand out of millions. */
export function selectedScale(selected: number, rows: number): number {
  return selected < rows && selected < 20_000 ? 2 : 1;
}

export function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

/** Colour attribute for a point view: palette index for nominal, 0..1 for quantitative. */
export function colorChannel(table: Table, field: string | undefined): { mode: 0 | 1 | 2; data: Float32Array } {
  const n = table.rows;
  if (!field) return { mode: 0, data: new Float32Array(n) };
  const c = table.columns.get(field);
  if (!c) throw new Error(`no field "${field}"`);
  if (c.type === "nominal") return { mode: 1, data: Float32Array.from(c.codes, (k) => k % PALETTE.length) };
  const [lo, hi] = c.extent;
  const span = hi > lo ? hi - lo : 1;
  return { mode: 2, data: Float32Array.from(c.values, (v) => (Number.isNaN(v) ? 0 : (v - lo) / span)) };
}

/** Float32 positions relative to the domain centre, so large offsets keep their precision. */
export function centred(values: Float64Array, extent: [number, number]): { data: Float32Array; centre: number } {
  const centre = (extent[0] + extent[1]) / 2;
  return { data: Float32Array.from(values, (v) => v - centre), centre };
}

export const COLOR_GLSL = /* glsl */ `
uniform int u_colorMode;
uniform vec3 u_palette[10];
uniform vec3 u_base;
vec3 viridis(float t) {
  t = clamp(t, 0.0, 1.0);
  const vec3 c0 = vec3(0.2777, 0.0054, 0.3341), c1 = vec3(0.1050, 1.4046, 1.3846),
             c2 = vec3(-0.3309, 0.2148, 0.0951), c3 = vec3(-4.6342, -5.7991, -19.3324),
             c4 = vec3(6.2283, 14.1799, 56.6906), c5 = vec3(4.7764, -13.7451, -65.3530),
             c6 = vec3(-5.4355, 4.6459, 26.3124);
  return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}
vec3 pointColor(float c) {
  if (u_colorMode == 1) return u_palette[int(c + 0.5)];
  if (u_colorMode == 2) return viridis(c);
  return u_base;
}`;

/**
 * The edges of an edge view: every distinct (source, target) pair, the nodes they join,
 * and for each row the code of its pair. A pair is the unit an edge view filters on.
 */
export interface Pairs {
  nodes: string[];
  /** For each pair code, its [source node, target node]. */
  pairs: [number, number][];
  /** For each row, its pair code. */
  codes: Uint32Array;
  /** "source → target", per pair code, for selections reported in data terms. */
  labels: string[];
}

export function edgePairs(table: Table, source: string, target: string): Pairs {
  const s = categorical(table, source);
  const t = categorical(table, target);
  const nodes: string[] = [];
  const nodeOf = new Map<string, number>();
  const node = (label: string) => {
    let i = nodeOf.get(label);
    if (i === undefined) {
      i = nodes.length;
      nodes.push(label);
      nodeOf.set(label, i);
    }
    return i;
  };
  const pairs: [number, number][] = [];
  const pairOf = new Map<number, number>();
  const codes = new Uint32Array(table.rows);
  for (let r = 0; r < table.rows; r++) {
    const a = node(s.labels[s.codes[r]!]!);
    const b = node(t.labels[t.codes[r]!]!);
    const key = a * 1_000_003 + b;
    let k = pairOf.get(key);
    if (k === undefined) {
      k = pairs.length;
      pairs.push([a, b]);
      pairOf.set(key, k);
    }
    codes[r] = k;
  }
  return { nodes, pairs, codes, labels: pairs.map(([a, b]) => `${nodes[a]} → ${nodes[b]}`) };
}

/**
 * The selection of an edge view: a set of pair codes. Clicking an edge toggles it;
 * clicking a node toggles every edge that touches it (all on if any is off). The view's
 * own group ignores this filter, so the view keeps its full shape while it filters the
 * other views.
 */
export class EdgeSelection {
  readonly chosen = new Set<number>();
  constructor(private engine: Engine, private dim: number | undefined, private changed: () => void) {}
  get active(): boolean {
    return this.dim !== undefined;
  }
  toggle(codes: number[]) {
    if (this.dim === undefined || codes.length === 0) return;
    const allOn = codes.every((k) => this.chosen.has(k));
    for (const k of codes) {
      if (allOn) this.chosen.delete(k);
      else this.chosen.add(k);
    }
    this.engine.filter(this.chosen.size ? { dim: this.dim, codes: [...this.chosen] } : { dim: this.dim, clear: true });
    this.changed();
  }
  /** Shown at full strength: chosen edges, or every edge when nothing is chosen. */
  shows(code: number): boolean {
    return this.chosen.size === 0 || this.chosen.has(code);
  }
}

/** Node totals from per-pair counts: rows leaving, rows arriving. */
export function nodeTotals(p: Pairs, counts: ArrayLike<number>): { out: Float64Array; into: Float64Array } {
  const out = new Float64Array(p.nodes.length);
  const into = new Float64Array(p.nodes.length);
  p.pairs.forEach(([a, b], k) => {
    out[a]! += counts[k] ?? 0;
    into[b]! += counts[k] ?? 0;
  });
  return { out, into };
}

/** Pair codes touching a node, as source, as target, or either. */
export function incident(p: Pairs, node: number, side: "source" | "target" | "either" = "either"): number[] {
  const out: number[] = [];
  p.pairs.forEach(([a, b], k) => {
    if ((side !== "target" && a === node) || (side !== "source" && b === node)) out.push(k);
  });
  return out;
}

/** A small tooltip that follows the pointer inside a panel. */
export function tooltip(root: HTMLElement): { show(e: MouseEvent, text: string): void; hide(): void } {
  const tip = document.createElement("div");
  tip.className = "sv-tip";
  tip.hidden = true;
  root.style.position = "relative";
  root.append(tip);
  return {
    show(e, text) {
      const r = root.getBoundingClientRect();
      tip.textContent = text;
      tip.hidden = false;
      tip.style.left = `${Math.min(e.clientX - r.left + 12, r.width - 220)}px`;
      tip.style.top = `${e.clientY - r.top + 12}px`;
    },
    hide() {
      tip.hidden = true;
    },
  };
}
