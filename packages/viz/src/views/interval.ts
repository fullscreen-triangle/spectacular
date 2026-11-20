import { axisLeft, brushX, format, interpolateViridis, scaleLinear, type ScaleLinear } from "d3";
import type { EngineState } from "../engine/client";
import { SVG_NS } from "../export";
import type { IntervalView } from "../spec";
import {
  addKey, axisLabels, categorical, colorChannel, colorKey, colorsFor, frame, hex, labelRoom, numeric, pointAlpha, shiftLeft, xAxis,
  type Panel, type ViewContext,
} from "./common";
import { VECTOR_POINTS } from "./scatter";

/**
 * Each row as a quantity from `low` to `high`: a line across, on a lane of its own.
 * With `label`, lanes are grouped in one band per category, in the field's order; in
 * each band rows are ordered by their low end. Rows passing every brush are drawn in
 * colour over the rest in grey. Brushing a range keeps the rows whose interval overlaps
 * it: low end at or below the range's top, high end at or above its bottom. With
 * `scale: "label"`, each band has a scale of its own (quantities in different units),
 * its ends printed at the band's corners, and there is no brush.
 */
export function interval(ctx: ViewContext<IntervalView>): Panel {
  const { view, table, engine, alloc } = ctx;
  const lo = numeric(table, view.low), hi = numeric(table, view.high);
  const cat = view.label ? categorical(table, view.label) : undefined;
  const bands = cat ? cat.labels.length : 1;

  // Lanes: rows with both ends, by band, then by low end.
  const order = Array.from({ length: table.rows }, (_, i) => i)
    .filter((i) => !Number.isNaN(lo.values[i]!) && !Number.isNaN(hi.values[i]!))
    .sort((a, b) => (cat ? cat.codes[a]! - cat.codes[b]! : 0) || lo.values[a]! - lo.values[b]! || a - b);
  const lane = new Float64Array(table.rows).fill(NaN);
  const bandStart = new Array<number>(bands + 1).fill(0);
  order.forEach((r, k) => (lane[r] = k));
  for (const r of order) bandStart[(cat ? cat.codes[r]! : 0) + 1]!++;
  for (let b = 1; b <= bands; b++) bandStart[b]! += bandStart[b - 1]!;

  const room = cat ? labelRoom(cat.labels) : undefined;
  const raw = frame(ctx, view.height ?? Math.min(520, Math.max(120, order.length * 8 + bands * 6)));
  const f = room ? shiftLeft(raw, room.width) : raw;
  const lowEnd = Math.min(lo.extent[0], hi.extent[0]), highEnd = Math.max(lo.extent[1], hi.extent[1]);
  const shared = scaleLinear().domain([lowEnd, highEnd]).range([0, f.w]).nice();
  const y = scaleLinear().domain([0, Math.max(1, order.length)]).range([0, f.h]);
  const own = view.scale === "label" && cat !== undefined;
  // Per band: the scale of its own rows.
  const bandScale: ScaleLinear<number, number>[] = Array.from({ length: bands }, () => shared);
  if (own) {
    const ends = Array.from({ length: bands }, () => [Infinity, -Infinity]);
    for (const r of order) {
      const e = ends[cat.codes[r]!]!;
      e[0] = Math.min(e[0]!, lo.values[r]!, hi.values[r]!);
      e[1] = Math.max(e[1]!, lo.values[r]!, hi.values[r]!);
    }
    ends.forEach(([a, z], b) => {
      if (a! <= z!) bandScale[b] = scaleLinear().domain(a === z ? [a! - 1, z! + 1] : [a!, z!]).range([0, f.w]).nice();
    });
  }
  const x = (r: number, v: number) => (own ? bandScale[cat.codes[r]!]! : shared)(v);
  if (own) {
    const fmt = format("~g");
    const ends = f.g.append("g").attr("class", "sv-axis");
    cat.labels.forEach((_, b) => {
      if (bandStart[b + 1]! === bandStart[b]!) return;
      const [a, z] = bandScale[b]!.domain() as [number, number];
      const yy = y(bandStart[b + 1]!) - 2;
      ends.append("text").attr("class", "sv-band").attr("x", 2).attr("y", yy).text(fmt(a));
      ends.append("text").attr("class", "sv-band").attr("x", f.w - 2).attr("y", yy).attr("text-anchor", "end").text(fmt(z));
    });
    axisLabels(f, `${ctx.titleOf(view.low)} – ${ctx.titleOf(view.high)}, each band on its own scale`, "");
  } else {
    xAxis(f, shared);
    axisLabels(f, `${ctx.titleOf(view.low)} – ${ctx.titleOf(view.high)}`, "");
  }
  if (cat && room) {
    const middles = cat.labels.map((_, b) => y((bandStart[b]! + bandStart[b + 1]!) / 2));
    const axis = f.g.append("g").attr("class", "sv-axis");
    axis.call(axisLeft(scaleLinear().range([0, f.h])).tickValues([]).tickSizeOuter(0));
    axis.selectAll("text.sv-band").data(cat.labels.map((_, b) => b)).join("text").attr("class", "sv-band")
      .attr("x", -8).attr("y", (b) => middles[b]!).attr("dy", "0.35em").attr("text-anchor", "end")
      .attr("display", (b) => (bandStart[b + 1]! > bandStart[b]! ? null : "none"))
      .text((b) => room.short(cat.labels[b]!))
      .append("title").text((b) => cat.labels[b]!);
    f.under.append("g").attr("class", "sv-gridlines").selectAll("line").data(bandStart.slice(1, -1)).join("line")
      .attr("x1", 0).attr("x2", f.w).attr("y1", (s) => y(s)).attr("y2", (s) => y(s)).attr("stroke", "var(--sv-grid)");
  }

  const palette = colorsFor(ctx, view.color, view);
  const base = view.fill ? hex(view.fill) : palette[0]!;
  addKey(f.root, colorKey(table, view.color, view.color ? ctx.titleOf(view.color) : "", palette));
  const color = colorChannel(table, view.color, palette.length);
  const fillOf = (i: number) =>
    color.mode === 1 ? palette[color.data[i]!]! : color.mode === 2 ? interpolateViridis(color.data[i]!) : base;
  const thickness = Math.max(1, Math.min(6, (f.h / Math.max(1, order.length)) * 0.6));

  const canvas = document.createElement("canvas");
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(f.w * dpr);
  canvas.height = Math.round(f.h * dpr);
  Object.assign(canvas.style, { width: `${f.w}px`, height: `${f.h}px` });
  f.plot.append(canvas);
  const c2d = canvas.getContext("2d")!;
  c2d.scale(dpr, dpr);
  c2d.lineCap = "round";

  const draw = (state: EngineState) => {
    c2d.clearRect(0, 0, f.w, f.h);
    c2d.lineWidth = thickness;
    for (const pass of [0, 1]) {
      c2d.globalAlpha = pass ? Math.max(0.35, pointAlpha(state.selectedCount)) : 0.25;
      for (const r of order) {
        if (state.mask[r] !== pass) continue;
        c2d.strokeStyle = pass ? fillOf(r) : "rgb(140,140,153)";
        const yy = y(lane[r]! + 0.5);
        c2d.beginPath();
        c2d.moveTo(x(r, lo.values[r]!), yy);
        c2d.lineTo(x(r, hi.values[r]!), yy);
        c2d.stroke();
      }
    }
    c2d.globalAlpha = 1;
  };
  draw(engine.state);

  if ((view.brush ?? "x") === "x" && !own && alloc.dims[view.low] !== undefined) {
    const dl = alloc.dims[view.low]!, dh = alloc.dims[view.high]!;
    const b = brushX().extent([[0, 0], [f.w, f.h]]).on("brush end", (event) => {
      const s = event.selection as [number, number] | null;
      if (!s) {
        engine.filter({ dim: dl, clear: true });
        engine.filter({ dim: dh, clear: true });
        return;
      }
      const [a, z] = [shared.invert(s[0]), shared.invert(s[1])];
      engine.filter({ dim: dl, range: [-Infinity, z] });
      engine.filter({ dim: dh, range: [a, Infinity] });
    });
    f.g.append("g").attr("class", "sv-brush").call(b);
  }

  /** Lines for SVG export, the same two passes as the canvas. */
  const vector = () => {
    if (order.length > VECTOR_POINTS) return undefined;
    const s = engine.state;
    const content = document.createElementNS(SVG_NS, "g");
    content.setAttribute("stroke-linecap", "round");
    content.setAttribute("stroke-width", thickness.toFixed(2));
    for (const pass of [0, 1]) {
      const g = document.createElementNS(SVG_NS, "g");
      g.setAttribute("stroke-opacity", (pass ? Math.max(0.35, pointAlpha(s.selectedCount)) : 0.25).toFixed(3));
      for (const r of order) {
        if (s.mask[r] !== pass) continue;
        const l = document.createElementNS(SVG_NS, "line");
        const yy = y(lane[r]! + 0.5).toFixed(2);
        l.setAttribute("x1", x(r, lo.values[r]!).toFixed(2));
        l.setAttribute("x2", x(r, hi.values[r]!).toFixed(2));
        l.setAttribute("y1", yy);
        l.setAttribute("y2", yy);
        l.setAttribute("stroke", pass ? fillOf(r) : "rgb(140,140,153)");
        g.append(l);
      }
      content.append(g);
    }
    return { plot: f.plot, content };
  };
  return { el: f.root, update: draw, vector, destroy() {} };
}
