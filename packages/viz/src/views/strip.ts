import { axisLeft, brushX, scaleBand, scaleLinear } from "d3";
import type { StripView } from "../spec";
import { axisLabels, categorical, frame, hasFieldColors, labelRoom, numeric, shiftLeft, xAxis, type Panel, type ViewContext } from "./common";
import { pointLayer } from "./scatter";

/** A spread in [-0.5, 0.5) from a row's index, the same on every draw. */
const jitter = (i: number) => ((Math.imul(i + 1, 2654435761) >>> 0) / 2 ** 32) - 0.5;

/**
 * A number against a category: each category a band, each row a point at its value,
 * spread across the band so that rows with equal values stay visible. Points are
 * coloured and selected as in a scatter (by the category itself when its field declares
 * a palette); brushing a range of the value filters the other views.
 */
export function strip(ctx: ViewContext<StripView>): Panel {
  const { view, table, engine, alloc } = ctx;
  const cat = categorical(table, view.y);
  const n = cat.labels.length;
  const room = labelRoom(cat.labels);
  const f = shiftLeft(frame(ctx, view.height ?? Math.min(480, Math.max(120, n * 30))), room.width);
  const xs = numeric(table, view.x);
  const x = scaleLinear().domain(xs.extent).range([0, f.w]).nice();
  // Band i spans [i, i + 1] from the top; its points fill the middle 70%.
  const y = scaleLinear().domain([0, Math.max(1, n)]).range([0, f.h]);
  const yv = Float64Array.from(cat.codes, (c, i) => c + 0.5 + 0.7 * jitter(i));
  xAxis(f, x);
  const bands = scaleBand<number>().domain(cat.labels.map((_, i) => i)).range([0, f.h]);
  f.g.append("g").attr("class", "sv-axis")
    .call(axisLeft(bands).tickFormat((i) => room.short(cat.labels[i]!)).tickSizeOuter(0).tickPadding(6))
    .call((a) => a.selectAll<SVGGElement, number>(".tick").append("title").text((i) => cat.labels[i]!));
  f.under.append("g").attr("class", "sv-gridlines").selectAll("line").data(cat.labels.slice(1).map((_, i) => i + 1)).join("line")
    .attr("x1", 0).attr("x2", f.w).attr("y1", (i) => y(i)).attr("y2", (i) => y(i)).attr("stroke", "var(--sv-grid)");
  axisLabels(f, ctx.titleOf(view.x), "");
  // Without its own colour, a strip takes its category's declared colours.
  const color = view.color ?? (hasFieldColors(ctx, view.y) && !view.fill ? view.y : undefined);
  const layer = pointLayer({ ...ctx, view: { ...view, color } }, f, xs.values, yv, x, y);

  if ((view.brush ?? "x") === "x") {
    const dim = alloc.dims[view.x]!;
    const b = brushX().extent([[0, 0], [f.w, f.h]]).on("brush end", (event) => {
      const s = event.selection as [number, number] | null;
      engine.filter(s ? { dim, range: [x.invert(s[0]), x.invert(s[1])] } : { dim, clear: true });
    });
    f.g.append("g").attr("class", "sv-brush").call(b);
  }
  return { el: f.root, update: layer.draw, vector: layer.vector, destroy: layer.destroy };
}
