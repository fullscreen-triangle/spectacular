import { axisBottom, axisLeft, format, max, scaleBand, scaleLinear } from "d3";
import type { EngineState } from "../engine/client";
import type { BarsView } from "../spec";
import { PALETTE, categorical, clip, elastic, frame, type Panel, type ViewContext } from "./common";

/**
 * Horizontal bars per category. Clicking toggles a category in this view's filter. As in
 * histograms, the count axis follows the filtered bars and the context hides when it
 * would outgrow the axis.
 */
export function bars(ctx: ViewContext<BarsView>): Panel {
  const { view, table, engine, alloc } = ctx;
  const col = categorical(table, view.field);
  const group = alloc.group!;
  const f = frame(ctx, view.height ?? Math.min(420, Math.max(120, col.labels.length * 22)));
  const labelWidth = Math.min(160, 8 + 7 * Math.max(...col.labels.map((l) => l.length)));
  // Long labels keep their end: in IRIs and paths that is the part that differs.
  const fit = Math.floor((labelWidth - 8) / 7);
  const short = (l: string) => (l.length <= fit ? l : "…" + l.slice(l.length - fit + 1));
  f.g.attr("transform", `translate(${labelWidth},8)`);
  const w = f.w - (labelWidth - 52);
  const yb = scaleBand<number>().domain(col.labels.map((_, i) => i)).range([0, f.h]).padding(0.15);
  const x = scaleLinear().range([0, w]);
  const xAxis = f.g.append("g").attr("transform", `translate(0,${f.h})`);
  f.g.append("g")
    .call(axisLeft(yb).tickFormat((i) => short(col.labels[i]!)))
    .call((a) => a.selectAll<SVGGElement, number>(".tick").append("title").text((i) => col.labels[i]!));

  const chosen = new Set<number>();
  const rows = f.g
    .append("g")
    .attr("clip-path", `url(#${clip({ ...f, w })})`)
    .selectAll("g.sv-row")
    .data(col.labels.map((_, i) => i))
    .join("g")
    .attr("class", "sv-row")
    .attr("transform", (i) => `translate(0,${yb(i)})`);
  const context = rows.append("rect").attr("class", "sv-context").attr("height", yb.bandwidth());
  const focus = rows
    .append("rect")
    .attr("class", "sv-focus")
    .attr("height", yb.bandwidth())
    .attr("fill", (i) => PALETTE[i % PALETTE.length]!);

  const paint = (state: EngineState) => {
    const counts = state.counts[group]!;
    const top = elastic(max(counts) ?? 0, max(state.baseline[group]!) ?? 0);
    x.domain([0, top.max]).nice();
    xAxis.call(axisBottom(x).ticks(4).tickFormat(format("~s")));
    context.attr("display", top.context ? null : "none").attr("width", (i) => x(state.baseline[group]![i]!));
    focus
      .attr("width", (i) => x(counts[i]!))
      .attr("opacity", (i) => (chosen.size === 0 || chosen.has(i) ? 1 : 0.25));
  };
  paint(engine.state);

  if ((view.brush ?? "select") === "select") {
    const dim = alloc.dims[view.field]!;
    rows.style("cursor", "pointer").on("click", (_, i) => {
      if (chosen.has(i)) chosen.delete(i);
      else chosen.add(i);
      engine.filter(chosen.size ? { dim, codes: [...chosen] } : { dim, clear: true });
      paint(engine.state);
    });
  }
  return { el: f.root, update: paint, destroy() {} };
}
