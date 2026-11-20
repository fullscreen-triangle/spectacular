import { axisBottom, axisLeft, max, scaleBand, scaleLinear, type Selection } from "d3";
import type { EngineState } from "../engine/client";
import type { BarsView } from "../spec";
import {
  MARGIN, addKey, categorical, clip, colorKey, colorsFor, elastic, frame, tickFormat, tooltip, type Panel, type ViewContext,
} from "./common";

/**
 * Horizontal bars per category, in the field's domain order when it declares one (a
 * category with no rows keeps its band). With `color`, each bar is split into the
 * categories of a second field, stacked in that field's colours. Clicking a bar toggles
 * its category in this view's filter. As in histograms, the count axis follows the
 * filtered bars and the context hides when it would outgrow the axis.
 */
export function bars(ctx: ViewContext<BarsView>): Panel {
  const { view, table, engine, alloc } = ctx;
  const col = categorical(table, view.field);
  const cross = view.color ? alloc.cross : undefined;
  const colors = cross ? colorsFor(ctx, view.color, view) : colorsFor(ctx, view.field, view);
  const group = alloc.group!;
  const f = frame(ctx, view.height ?? Math.min(420, Math.max(120, col.labels.length * 22)));
  if (cross) addKey(f.root, colorKey(table, view.color, ctx.titleOf(view.color!), colors));
  const tip = tooltip(f.root);
  const labelWidth = Math.min(160, 8 + 7 * Math.max(...col.labels.map((l) => l.length)));
  // Long labels keep their end: in IRIs and paths that is the part that differs.
  const fit = Math.floor((labelWidth - 8) / 7);
  const short = (l: string) => (l.length <= fit ? l : "…" + l.slice(l.length - fit + 1));
  f.g.attr("transform", `translate(${labelWidth},${MARGIN.top})`);
  const w = f.w - (labelWidth - MARGIN.left);
  const yb = scaleBand<number>().domain(col.labels.map((_, i) => i)).range([0, f.h]).padding(0.15);
  const x = scaleLinear().range([0, w]);
  const xAxis = f.g.append("g").attr("class", "sv-axis").attr("transform", `translate(0,${f.h})`);
  f.g.append("g").attr("class", "sv-axis")
    .call(axisLeft(yb).tickFormat((i) => short(col.labels[i]!)).tickSizeOuter(0).tickPadding(6))
    .call((a) => a.selectAll<SVGGElement, number>(".tick").append("title").text((i) => col.labels[i]!));

  // The pair codes in each bar, in the second field's order.
  const segments: number[][] = col.labels.map(() => []);
  cross?.pairs.forEach(([i], k) => segments[i]!.push(k));
  for (const s of segments) s.sort((a, b) => cross!.pairs[a]![1] - cross!.pairs[b]![1]);
  const totals = (counts: Uint32Array) =>
    cross ? segments.map((s) => s.reduce((n, k) => n + counts[k]!, 0)) : Array.from(counts);
  const colorLabels = view.color ? categorical(table, view.color).labels : [];

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
  // One rect per bar, or per segment of a split bar; each is bound to its code.
  type Rects = Selection<SVGRectElement, number, any, unknown>;
  const focus = (cross
    ? rows.selectAll("rect.sv-segment").data((i) => segments[i]!).join("rect")
      .attr("class", "sv-segment").attr("height", yb.bandwidth())
      .attr("fill", (k) => colors[cross.pairs[k]![1] % colors.length]!)
    : rows.append("rect").attr("class", "sv-focus").attr("height", yb.bandwidth())
      .attr("fill", (i) => colors[i % colors.length]!)) as unknown as Rects;

  let counts = engine.state.counts[group]!;
  const paint = (state: EngineState) => {
    counts = state.counts[group]!;
    const now = totals(counts), base = totals(state.baseline[group]!);
    const top = elastic(max(now) ?? 0, max(base) ?? 0);
    x.domain([0, top.max]).nice();
    xAxis.call(axisBottom(x).ticks(4).tickFormat(tickFormat(x, 4)).tickSizeOuter(0).tickPadding(6));
    context.attr("display", top.context ? null : "none").attr("width", (i) => x(base[i]!));
    const shown = (i: number) => (chosen.size === 0 || chosen.has(i) ? 1 : 0.25);
    if (cross) {
      const at = new Float64Array(cross.pairs.length);
      for (const s of segments) {
        let acc = 0;
        for (const k of s) {
          at[k] = acc;
          acc += counts[k]!;
        }
      }
      focus
        .attr("x", (k) => x(at[k]!))
        .attr("width", (k) => x(at[k]! + counts[k]!) - x(at[k]!))
        .attr("opacity", (k) => shown(cross.pairs[k]![0]));
    } else {
      focus
        .attr("width", (i) => x(counts[i]!))
        .attr("opacity", shown);
    }
  };
  paint(engine.state);
  if (cross)
    focus
      .on("mousemove", (e: MouseEvent, k) => {
        const [i, j] = cross.pairs[k]!;
        tip.show(e, `${col.labels[i]} · ${colorLabels[j]}: ${counts[k]} rows`);
      })
      .on("mouseleave", () => tip.hide());

  if ((view.brush ?? "select") === "select") {
    const dim = Object.values(alloc.dims)[0]!;
    rows.style("cursor", "pointer").on("click", (_, i) => {
      if (chosen.has(i)) chosen.delete(i);
      else chosen.add(i);
      const codes = cross ? [...chosen].flatMap((c) => segments[c]!) : [...chosen];
      engine.filter(chosen.size ? { dim, codes } : { dim, clear: true });
      paint(engine.state);
    });
  }
  return { el: f.root, update: paint, destroy() {} };
}
