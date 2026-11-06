import { max, scalePoint, scaleSqrt } from "d3";
import type { EngineState } from "../engine/client";
import type { ArcView } from "../spec";
import { EdgeSelection, frame, incident, nodeTotals, tooltip, type Panel, type ViewContext } from "./common";

/** Room under the line for the node labels, which are set at an angle. */
const LABEL_SPACE = 92;

/**
 * Nodes on one line and every edge as an arc above it, as thick as the rows it holds
 * after the other views' filters. Arcs leave from the left end of each pair, so the
 * direction is in the tooltip rather than the shape. Clicking an arc selects its rows;
 * clicking a node selects every arc that touches it.
 */
export function arc(ctx: ViewContext<ArcView>): Panel {
  const { view, engine, alloc } = ctx;
  const p = alloc.pairs!;
  const group = alloc.group!;
  const f = frame(ctx, view.height ?? 300);
  const tip = tooltip(f.root);
  const base = engine.state.baseline[group]!;
  const totals = nodeTotals(p, base);
  const weight = (i: number) => totals.out[i]! + totals.into[i]!;
  const order = p.nodes.map((_, i) => i);
  const how = view.order ?? "count";
  if (how === "count") order.sort((a, b) => weight(b) - weight(a) || p.nodes[a]!.localeCompare(p.nodes[b]!));
  else if (how === "name") order.sort((a, b) => p.nodes[a]!.localeCompare(p.nodes[b]!));

  const line = Math.max(40, f.h - LABEL_SPACE);
  const x = scalePoint<number>().domain(order).range([0, f.w]).padding(0.5);
  const width = scaleSqrt().domain([0, Math.max(1, max(base) ?? 1)]).range([0, 12]);
  const radius = scaleSqrt().domain([0, Math.max(1, max(order.map(weight)) ?? 1)]).range([2.5, 9]);
  const squash = line / Math.max(1, f.w / 2);
  const path = (k: number) => {
    const [a, b] = p.pairs[k]!;
    const xa = x(a)!, xb = x(b)!;
    const r = Math.abs(xb - xa) / 2;
    return `M${xa},${line} A${r},${r * squash} 0 0,${xa < xb ? 1 : 0} ${xb},${line}`;
  };

  const sel = new EdgeSelection(engine, view.brush === "none" ? undefined : alloc.dims[Object.keys(alloc.dims)[0]!], () => paint(engine.state));
  const codes = p.pairs.map((_, k) => k).sort((a, b) => base[b]! - base[a]!);
  const context = f.g.append("g").selectAll("path").data(codes).join("path").attr("class", "sv-edge-context").attr("d", path)
    .attr("stroke-width", (k) => Math.max(1, width(base[k]!)));
  const edges = f.g.append("g").selectAll("path").data(codes).join("path").attr("class", "sv-edge").attr("d", path);
  const hits = f.g.append("g").selectAll("path").data(codes).join("path").attr("class", "sv-hit").attr("d", path)
    .attr("fill", "none").attr("stroke", "transparent").attr("stroke-width", 8);
  f.g.append("line").attr("x1", 0).attr("x2", f.w).attr("y1", line).attr("y2", line).attr("class", "sv-edge-context");
  const nodes = f.g.append("g").selectAll("circle").data(order).join("circle").attr("class", "sv-node sv-hit")
    .attr("cx", (i) => x(i)!).attr("cy", line);
  const labels = f.g.append("g").selectAll("text").data(order).join("text").attr("class", "sv-node-label")
    .attr("text-anchor", "end")
    .attr("transform", (i) => `translate(${x(i)! + 3},${line + 12}) rotate(-55)`)
    .text((i) => (p.nodes[i]!.length > 22 ? p.nodes[i]!.slice(0, 21) + "…" : p.nodes[i]!));

  let counts: ArrayLike<number> = base;
  const paint = (state: EngineState) => {
    counts = state.counts[group]!;
    const filtered = state.selectedCount < state.rows;
    context.attr("display", filtered ? null : "none");
    edges
      .attr("display", (k) => (counts[k]! > 0 ? null : "none"))
      .attr("stroke-width", (k) => Math.max(1, width(counts[k]!)))
      .attr("stroke-opacity", (k) => (sel.shows(k) ? 0.75 : 0.18))
      .classed("sv-chosen", (k) => sel.chosen.has(k));
    const t = nodeTotals(p, counts);
    nodes
      .attr("r", (i) => radius(t.out[i]! + t.into[i]!))
      .classed("sv-chosen", (i) => incident(p, i).some((k) => sel.chosen.has(k)));
    labels.classed("sv-quiet", (i) => t.out[i]! + t.into[i]! === 0);
  };
  paint(engine.state);

  hits
    .on("mousemove", (e: MouseEvent, k) => tip.show(e, `${p.labels[k]}: ${counts[k]} of ${base[k]} rows`))
    .on("mouseleave", () => tip.hide())
    .on("click", (_, k) => sel.toggle([k]));
  nodes
    .on("mousemove", (e: MouseEvent, i) => {
      const t = nodeTotals(p, counts);
      tip.show(e, `${p.nodes[i]}: ${t.out[i]} out, ${t.into[i]} in`);
    })
    .on("mouseleave", () => tip.hide())
    .on("click", (_, i) => sel.toggle(incident(p, i)));
  if (!sel.active) {
    hits.style("cursor", null);
    nodes.style("cursor", null);
  }
  return { el: f.root, update: paint, destroy() {} };
}
