import { brushX, max, scaleLinear } from "d3";
import type { EngineState } from "../engine/client";
import type { HistogramView } from "../spec";
import { axisLabels, clip, elastic, frame, hex, xAxis, yAxis, type Panel, type ViewContext } from "./common";

/**
 * Bars show rows passing every *other* filter; the light bars behind them show the
 * unfiltered distribution, so the brush's own effect never hides its context. The count
 * axis follows the filtered bars, so a small selection stays legible; the context is
 * hidden while it would outgrow the axis.
 */
export function histogram(ctx: ViewContext<HistogramView>): Panel {
  const { view, engine, alloc } = ctx;
  const f = frame(ctx, view.height ?? 200);
  const group = alloc.group!;
  const [min, width, bins] = engine.state.binnings[group] as [number, number, number];
  const x = scaleLinear().domain([min, min + width * bins]).range([0, f.w]);
  const y = scaleLinear().range([f.h, 0]);
  xAxis(f, x, false);
  const redrawY = yAxis(f, y, 4);
  axisLabels(f, ctx.titleOf(view.field), "count");

  const bw = Math.max(1, f.w / bins - (bins < 80 ? 1 : 0));
  const layer = f.g.append("g").attr("clip-path", `url(#${clip(f)})`);
  const bars = (cls: string, counts: Uint32Array) =>
    layer
      .append("g")
      .attr("class", cls)
      .selectAll("rect")
      .data(Array.from(counts))
      .join("rect")
      .attr("x", (_, i) => x(min + i * width))
      .attr("width", bw);
  const context = bars("sv-context", engine.state.baseline[group]!);
  const focus = bars("sv-focus", engine.state.counts[group]!);
  if (view.fill) focus.style("fill", hex(view.fill));

  const paint = (state: EngineState) => {
    const counts = state.counts[group]!;
    const top = elastic(max(counts) ?? 0, max(state.baseline[group]!) ?? 0);
    y.domain([0, top.max]).nice();
    redrawY();
    const height = (d: number) => Math.max(0, f.h - y(d));
    context.attr("display", top.context ? null : "none").attr("y", (d) => f.h - height(d)).attr("height", height);
    focus.data(Array.from(counts)).attr("y", (d) => f.h - height(d)).attr("height", height);
  };
  paint(engine.state);

  if ((view.brush ?? "x") === "x") {
    const dim = alloc.dims[view.field]!;
    const b = brushX().extent([[0, 0], [f.w, f.h]]).on("brush end", (event) => {
      const s = event.selection as [number, number] | null;
      engine.filter(s ? { dim, range: [x.invert(s[0]), x.invert(s[1])] } : { dim, clear: true });
    });
    f.g.append("g").attr("class", "sv-brush").call(b);
  }

  return { el: f.root, update: paint, destroy() {} };
}
