import type { EngineState } from "../engine/client";
import type { SankeyView } from "../spec";
import { EdgeSelection, frame, incident, nodeTotals, tooltip, type Panel, type ViewContext } from "./common";

const NODE_W = 10;
const GAP = 4;

/**
 * Sources on the left, targets on the right, each flow as wide as the rows it holds
 * after the other views' filters. A node that is both a source and a target appears on
 * both sides. Clicking a flow selects its rows; clicking a node selects all its flows on
 * that side.
 */
export function sankey(ctx: ViewContext<SankeyView>): Panel {
  const { view, engine, alloc } = ctx;
  const p = alloc.pairs!;
  const group = alloc.group!;
  const f = frame(ctx, view.height ?? 360);
  const tip = tooltip(f.root);
  const base = engine.state.baseline[group]!;
  const bt = nodeTotals(p, base);
  const left = p.nodes.map((_, i) => i).filter((i) => bt.out[i]! > 0).sort((a, b) => bt.out[b]! - bt.out[a]!);
  const right = p.nodes.map((_, i) => i).filter((i) => bt.into[i]! > 0).sort((a, b) => bt.into[b]! - bt.into[a]!);
  const labelW = Math.min(150, Math.max(60, f.w * 0.22));
  const x0 = labelW, x1 = f.w - labelW;
  const sel = new EdgeSelection(engine, view.brush === "none" ? undefined : alloc.dims[Object.keys(alloc.dims)[0]!], () => paint(engine.state));
  const short = (s: string) => (s.length > 20 ? s.slice(0, 19) + "…" : s);

  const flowsG = f.g.append("g");
  const leftG = f.g.append("g");
  const rightG = f.g.append("g");
  let counts: ArrayLike<number> = base;

  /** Stack nodes in a column by their totals; returns each node's top and height. */
  const column = (ids: number[], total: Float64Array) => {
    const sum = ids.reduce((s, i) => s + total[i]!, 0);
    const avail = Math.max(10, f.h - GAP * Math.max(0, ids.length - 1));
    const k = sum > 0 ? avail / sum : 0;
    const top = new Map<number, number>(), h = new Map<number, number>();
    let y = 0;
    for (const i of ids) {
      const hh = Math.max(total[i]! > 0 ? 1 : 0.5, total[i]! * k);
      top.set(i, y);
      h.set(i, hh);
      y += hh + GAP;
    }
    return { top, h, k };
  };

  const paint = (state: EngineState) => {
    counts = state.counts[group]!;
    const t = nodeTotals(p, counts);
    const L = column(left, t.out), R = column(right, t.into);
    const k = Math.min(L.k, R.k);
    // Flows leave each source in the order of their targets, and arrive likewise.
    const outAt = new Map<number, number>(left.map((i) => [i, L.top.get(i)!]));
    const inAt = new Map<number, number>(right.map((i) => [i, R.top.get(i)!]));
    const rank = new Map(right.map((i, r) => [i, r]));
    const lrank = new Map(left.map((i, r) => [i, r]));
    const live = p.pairs.map((_, c) => c).filter((c) => counts[c]! > 0);
    const geom = new Map<number, { y0: number; y1: number; w: number }>();
    for (const c of [...live].sort((a, b) => rank.get(p.pairs[a]![1])! - rank.get(p.pairs[b]![1])!)) {
      const [a] = p.pairs[c]!;
      const w = counts[c]! * k;
      const y0 = outAt.get(a)! + w / 2;
      outAt.set(a, outAt.get(a)! + w);
      geom.set(c, { y0, y1: 0, w });
    }
    for (const c of [...live].sort((a, b) => lrank.get(p.pairs[a]![0])! - lrank.get(p.pairs[b]![0])!)) {
      const b = p.pairs[c]![1];
      const g = geom.get(c)!;
      g.y1 = inAt.get(b)! + g.w / 2;
      inAt.set(b, inAt.get(b)! + g.w);
    }
    const mid = (x0 + NODE_W + x1) / 2;
    flowsG.selectAll<SVGPathElement, number>("path").data(live, (c) => String(c)).join("path")
      .attr("class", (c) => "sv-flow sv-hit" + (sel.chosen.has(c) ? " sv-chosen" : ""))
      .attr("d", (c) => { const g = geom.get(c)!; return `M${x0 + NODE_W},${g.y0} C${mid},${g.y0} ${mid},${g.y1} ${x1},${g.y1}`; })
      .attr("stroke-width", (c) => Math.max(1, geom.get(c)!.w))
      .attr("stroke-opacity", (c) => (sel.shows(c) ? null : 0.12))
      .on("mousemove", (e: MouseEvent, c) => tip.show(e, `${p.labels[c]}: ${counts[c]} of ${base[c]} rows`))
      .on("mouseleave", () => tip.hide())
      .on("click", (_, c) => sel.toggle([c]));
    const side = (g: typeof leftG, ids: number[], col: { top: Map<number, number>; h: Map<number, number> }, isLeft: boolean, tot: Float64Array) => {
      const nodes = g.selectAll<SVGGElement, number>("g").data(ids, (i) => String(i)).join((enter) => {
        const n = enter.append("g").attr("class", "sv-hit");
        n.append("rect").attr("class", "sv-bar").attr("width", NODE_W);
        n.append("text").attr("class", "sv-node-label").attr("dy", "0.35em");
        return n;
      });
      nodes.attr("transform", (i) => `translate(${isLeft ? x0 : x1},${col.top.get(i)})`);
      nodes.select("rect").attr("height", (i) => col.h.get(i)!)
        .classed("sv-chosen", (i) => incident(p, i, isLeft ? "source" : "target").some((c) => sel.chosen.has(c)));
      nodes.select("text").attr("x", isLeft ? -6 : NODE_W + 6).attr("y", (i) => col.h.get(i)! / 2)
        .attr("text-anchor", isLeft ? "end" : "start").classed("sv-quiet", (i) => tot[i]! === 0).text((i) => short(p.nodes[i]!))
        // A label needs room; a thin node still names itself on hover.
        .attr("display", (i) => (col.h.get(i)! >= 9 ? null : "none"));
      nodes
        .on("mousemove", (e: MouseEvent, i) => tip.show(e, `${p.nodes[i]}: ${tot[i]} ${isLeft ? "out" : "in"}`))
        .on("mouseleave", () => tip.hide())
        .on("click", (_, i) => sel.toggle(incident(p, i, isLeft ? "source" : "target")));
    };
    side(leftG, left, L, true, t.out);
    side(rightG, right, R, false, t.into);
  };
  paint(engine.state);
  return { el: f.root, update: paint, destroy() {} };
}
