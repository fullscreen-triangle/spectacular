import { axisBottom, axisLeft, format, interpolateBlues, max, scaleBand, scaleSequentialSqrt } from "d3";
import type { EngineState } from "../engine/client";
import type { MatrixView } from "../spec";
import {
  EdgeSelection, addKey, categorical, frame, gradient, labelRoom, shiftLeft, tooltip, type Panel, type ViewContext,
} from "./common";

/** Light to dark, leaving out the near-white end so a one-row cell still shows. */
const shade = (t: number) => interpolateBlues(0.15 + 0.85 * t);

/**
 * Two nominal fields as a grid: one row per category of `y`, one column per category
 * of `x`, each cell shaded by the rows holding that pair after the other views'
 * filters. Clicking a cell selects its rows; clicking a row or column label selects the
 * whole row or column. The grid keeps its full shape while it filters the other views.
 */
export function matrix(ctx: ViewContext<MatrixView>): Panel {
  const { view, table, engine, alloc } = ctx;
  const cx = categorical(table, view.x), cy = categorical(table, view.y);
  const cross = alloc.cross!;
  const group = alloc.group!;
  const room = labelRoom(cy.labels);
  const f = shiftLeft(frame(ctx, view.height ?? Math.min(480, Math.max(80, cy.labels.length * 26))), room.width);
  const tip = tooltip(f.root);
  const xb = scaleBand<number>().domain(cx.labels.map((_, i) => i)).range([0, f.w]).padding(0.04);
  const yb = scaleBand<number>().domain(cy.labels.map((_, i) => i)).range([0, f.h]).padding(0.04);
  const fit = Math.max(1, Math.floor(xb.bandwidth() / 7));
  const shortX = (l: string) => (l.length <= fit ? l : l.slice(0, Math.max(1, fit - 1)) + "…");
  const color = scaleSequentialSqrt(shade);
  const counted = format("~s");

  // The key: rows per cell, on the same square-root scale as the cells.
  const key = document.createElement("div");
  key.className = "sv-key";
  const keyTitle = document.createElement("span");
  keyTitle.className = "sv-key-title";
  keyTitle.textContent = "rows per cell";
  const top = document.createTextNode("");
  key.append(keyTitle, document.createTextNode("0"), gradient(shade), top);
  addKey(f.root, key);

  const sel = new EdgeSelection(engine, view.brush === "none" ? undefined : Object.values(alloc.dims)[0], () => paint(engine.state));
  const inRow = (j: number) => cross.pairs.flatMap(([, b], k) => (b === j ? [k] : []));
  const inColumn = (i: number) => cross.pairs.flatMap(([a], k) => (a === i ? [k] : []));
  const base = engine.state.baseline[group]!;

  f.under.append("rect").attr("width", f.w).attr("height", f.h).attr("fill", "var(--sv-grid)");
  const xAxisG = f.g.append("g").attr("class", "sv-axis sv-hit").attr("transform", `translate(0,${f.h})`)
    .call(axisBottom(xb).tickFormat((i) => shortX(cx.labels[i]!)).tickSizeOuter(0).tickPadding(6));
  const yAxisG = f.g.append("g").attr("class", "sv-axis sv-hit")
    .call(axisLeft(yb).tickFormat((j) => room.short(cy.labels[j]!)).tickSizeOuter(0).tickPadding(6));
  xAxisG.selectAll<SVGGElement, number>(".tick").append("title").text((i) => cx.labels[i]!);
  yAxisG.selectAll<SVGGElement, number>(".tick").append("title").text((j) => cy.labels[j]!);
  xAxisG.selectAll<SVGGElement, number>(".tick").on("click", (_, i) => sel.toggle(inColumn(i)));
  yAxisG.selectAll<SVGGElement, number>(".tick").on("click", (_, j) => sel.toggle(inRow(j)));
  f.g.append("text").attr("class", "sv-axis-label").attr("x", f.w / 2).attr("y", f.h + 36).attr("text-anchor", "middle").text(ctx.titleOf(view.x));

  const cells = f.g.append("g").selectAll<SVGGElement, number>("g").data(cross.pairs.map((_, k) => k)).join("g")
    .attr("class", "sv-hit")
    .attr("transform", (k) => `translate(${xb(cross.pairs[k]![0])},${yb(cross.pairs[k]![1])})`);
  const rects = cells.append("rect").attr("width", xb.bandwidth()).attr("height", yb.bandwidth()).attr("rx", 2);
  const labels = cells.append("text").attr("class", "sv-cell-label").attr("x", xb.bandwidth() / 2).attr("y", yb.bandwidth() / 2)
    .attr("dy", "0.35em").attr("text-anchor", "middle")
    .attr("display", xb.bandwidth() >= 26 && yb.bandwidth() >= 13 ? null : "none");
  let counts: Uint32Array = base;
  cells
    .on("mousemove", (e: MouseEvent, k) => tip.show(e, `${cross.labels[k]}: ${counts[k]} of ${base[k]} rows`))
    .on("mouseleave", () => tip.hide())
    .on("click", (_, k) => sel.toggle([k]));

  const paint = (state: EngineState) => {
    counts = state.counts[group]!;
    const most = Math.max(1, max(counts) ?? 0);
    color.domain([0, most]);
    top.textContent = counted(most);
    rects
      .attr("fill", (k) => (counts[k]! > 0 ? color(counts[k]!) : "none"))
      .attr("stroke", (k) => (sel.chosen.has(k) ? "var(--sv-accent)" : counts[k]! === 0 && base[k]! > 0 ? "var(--sv-context)" : "none"))
      .attr("stroke-width", (k) => (sel.chosen.has(k) ? 2 : 1))
      .attr("opacity", (k) => (sel.shows(k) ? 1 : 0.35));
    labels
      .text((k) => (counts[k]! > 0 ? counted(counts[k]!) : ""))
      .attr("fill", (k) => (counts[k]! / most > 0.45 ? "#fff" : "#1d1f24"));
  };
  paint(engine.state);
  return { el: f.root, update: paint, destroy() {} };
}
