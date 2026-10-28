import { schemeTableau10, select, type Selection } from "d3";
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
}

export interface ViewContext<V extends View = View> {
  view: V;
  table: Table;
  engine: Engine;
  alloc: Allocation;
  width: number;
  titleOf(field: string): string;
}

export const MARGIN = { top: 8, right: 12, bottom: 34, left: 52 };
export const PALETTE = schemeTableau10;

export interface Frame {
  root: HTMLDivElement;
  /** The plot area: views place canvases here, under the SVG overlay. */
  plot: HTMLDivElement;
  svg: Selection<SVGSVGElement, unknown, null, undefined>;
  /** A group translated to the plot area's origin. */
  g: Selection<SVGGElement, unknown, null, undefined>;
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
  stage.append(plot);

  const svg = select(stage)
    .append("svg")
    .attr("class", "sv-overlay")
    .attr("width", w + MARGIN.left + MARGIN.right)
    .attr("height", h + MARGIN.top + MARGIN.bottom);
  const g = svg.append("g").attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);
  return { root, plot, svg, g, w, h };
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

export function axisLabels(f: Frame, x: string, y: string) {
  f.g.append("text").attr("class", "sv-axis-label").attr("x", f.w).attr("y", f.h + 30).attr("text-anchor", "end").text(x);
  f.g
    .append("text")
    .attr("class", "sv-axis-label")
    .attr("transform", "rotate(-90)")
    .attr("x", 0)
    .attr("y", -40)
    .attr("text-anchor", "end")
    .text(y);
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
