import type { EngineState } from "../engine/client";
import type { LayersView } from "../spec";
import {
  EdgeSelection, colorsFor, frame, hasFieldColors, incident, nodeTotals, tooltip, type Pairs, type Panel, type ViewContext,
} from "./common";

const NODE_W = 10;
const GAP = 6;
const LABEL = 14;

/**
 * Layers for a directed graph: each node one layer after the furthest node that leads to
 * it. Edges that close a cycle (found by depth-first search) and self-loops do not count
 * for the layering. Returns each node's layer and the edges that go backwards.
 */
export function layering(p: Pairs): { layer: Int32Array; back: Set<number> } {
  const n = p.nodes.length;
  const out: number[][] = Array.from({ length: n }, () => []);
  p.pairs.forEach(([a, b], k) => out[a]!.push(k));
  const back = new Set<number>();
  const state = new Uint8Array(n); // 0 unseen, 1 on the path, 2 done
  for (let s = 0; s < n; s++) {
    if (state[s]) continue;
    const stack: [number, number][] = [[s, 0]];
    state[s] = 1;
    while (stack.length) {
      const top = stack[stack.length - 1]!;
      const [v, i] = top;
      if (i < out[v]!.length) {
        top[1]++;
        const k = out[v]![i]!;
        const w = p.pairs[k]![1];
        if (state[w] === 1) back.add(k);
        else if (state[w] === 0) {
          state[w] = 1;
          stack.push([w, 0]);
        }
      } else {
        state[v] = 2;
        stack.pop();
      }
    }
  }
  // Longest path over the acyclic rest, in topological order.
  const indeg = new Int32Array(n);
  p.pairs.forEach(([, b], k) => { if (!back.has(k)) indeg[b]!++; });
  const layer = new Int32Array(n);
  const queue = [...Array(n).keys()].filter((v) => indeg[v] === 0);
  while (queue.length) {
    const v = queue.shift()!;
    for (const k of out[v]!) {
      if (back.has(k)) continue;
      const w = p.pairs[k]![1];
      layer[w] = Math.max(layer[w]!, layer[v]! + 1);
      if (--indeg[w]! === 0) queue.push(w);
    }
  }
  return { layer, back };
}

/**
 * A directed graph in layers, left to right, for pipelines of steps and rows passing
 * between them. Within a layer, nodes are ordered to keep flows short (a few barycentre
 * sweeps) and sized by the rows through them; each flow is as wide as the rows it holds
 * after the other views' filters. Clicking a flow selects its rows; clicking a node
 * selects every flow that touches it.
 */
export function layers(ctx: ViewContext<LayersView>): Panel {
  const { view, engine, alloc } = ctx;
  const p = alloc.pairs!;
  const group = alloc.group!;
  const f = frame(ctx, view.height ?? 360);
  const tip = tooltip(f.root);
  const base = engine.state.baseline[group]!;
  const bt = nodeTotals(p, base);
  const through = (i: number) => Math.max(bt.out[i]!, bt.into[i]!);
  const { layer, back } = layering(p);
  const depth = Math.max(0, ...layer) + 1;
  const columns: number[][] = Array.from({ length: depth }, () => []);
  p.nodes.forEach((_, i) => { if (through(i) > 0) columns[layer[i]!]!.push(i); });

  // Order within layers by the mean position of neighbours, sweeping down and up.
  const pos = new Float64Array(p.nodes.length);
  const place = () => columns.forEach((c) => c.forEach((v, i) => (pos[v] = i)));
  place();
  const neighbours = (v: number, before: boolean) =>
    p.pairs.flatMap(([a, b], k) => (back.has(k) ? [] : before ? (b === v ? [a] : []) : a === v ? [b] : []));
  for (let sweep = 0; sweep < 4; sweep++) {
    const down = sweep % 2 === 0;
    const range = down ? [...columns.keys()].slice(1) : [...columns.keys()].reverse().slice(1);
    for (const l of range) {
      const mean = (v: number) => {
        const ns = neighbours(v, down);
        return ns.length ? ns.reduce((s, u) => s + pos[u]!, 0) / ns.length : pos[v]!;
      };
      columns[l]!.sort((a, b) => mean(a) - mean(b));
      place();
    }
  }

  // Columns across the width; labels sit right of each node, so leave room at the end.
  const labelW = Math.min(110, Math.max(50, f.w * 0.1));
  const xOf = (l: number) => (depth === 1 ? 0 : (l / (depth - 1)) * (f.w - labelW - NODE_W));
  const tallest = Math.max(1, ...columns.map((c) => c.reduce((s, v) => s + through(v), 0)));
  const fullest = Math.max(1, ...columns.map((c) => c.length));
  // Every node has its label above it, so each takes LABEL more height.
  const k = Math.max(0, (f.h - LABEL - (GAP + LABEL) * (fullest - 1)) / tallest);
  const top = new Float64Array(p.nodes.length), height = new Float64Array(p.nodes.length);
  for (const c of columns) {
    const used = c.reduce((s, v) => s + Math.max(2, through(v) * k), 0) + (GAP + LABEL) * (c.length - 1);
    let y = LABEL + (f.h - LABEL - used) / 2;
    for (const v of c) {
      top[v] = y;
      height[v] = Math.max(2, through(v) * k);
      y += height[v]! + GAP + LABEL;
    }
  }

  const colorOf = (() => {
    const field = hasFieldColors(ctx, view.source) ? view.source : hasFieldColors(ctx, view.target) ? view.target : undefined;
    if (!field) return () => null;
    const col = ctx.table.columns.get(field);
    const colors = colorsFor(ctx, field);
    const byName = new Map(col?.type === "nominal" ? col.labels.map((l, i) => [l, colors[i % colors.length]!]) : []);
    return (v: number) => byName.get(p.nodes[v]!) ?? null;
  })();

  const sel = new EdgeSelection(engine, view.brush === "none" ? undefined : Object.values(alloc.dims)[0], () => paint(engine.state));
  const flowsG = f.g.append("g");
  const nodesG = f.g.append("g");
  let counts: ArrayLike<number> = base;
  const short = (s: string) => (s.length > 22 ? s.slice(0, 21) + "…" : s);

  const paint = (state: EngineState) => {
    counts = state.counts[group]!;
    const live = p.pairs.map((_, c) => c).filter((c) => counts[c]! > 0 && p.pairs[c]![0] !== p.pairs[c]![1]);
    // Flows leave a node in the order of their targets' heights, and arrive likewise.
    const outAt = new Float64Array(top), inAt = new Float64Array(top);
    const geom = new Map<number, { y0: number; y1: number; w: number }>();
    for (const c of [...live].sort((a, b) => pos[p.pairs[a]![1]]! - pos[p.pairs[b]![1]]! || top[p.pairs[a]![1]]! - top[p.pairs[b]![1]]!)) {
      const [a] = p.pairs[c]!;
      const w = Math.max(1, counts[c]! * k);
      geom.set(c, { y0: outAt[a]! + w / 2, y1: 0, w });
      outAt[a]! += w;
    }
    for (const c of [...live].sort((a, b) => top[p.pairs[a]![0]]! - top[p.pairs[b]![0]]!)) {
      const b = p.pairs[c]![1];
      const g = geom.get(c)!;
      g.y1 = inAt[b]! + g.w / 2;
      inAt[b]! += g.w;
    }
    const path = (c: number) => {
      const [a, b] = p.pairs[c]!;
      const g = geom.get(c)!;
      const xa = xOf(layer[a]!) + NODE_W, xb = xOf(layer[b]!);
      if (back.has(c)) {
        // A flow that closes a cycle loops back under the graph.
        const under = f.h + 4;
        return `M${xa},${g.y0} C${xa + 40},${g.y0} ${xa + 40},${under} ${(xa + xb) / 2},${under} S${xb - 40},${g.y1} ${xb},${g.y1}`;
      }
      const mid = (xa + xb) / 2;
      return `M${xa},${g.y0} C${mid},${g.y0} ${mid},${g.y1} ${xb},${g.y1}`;
    };
    flowsG.selectAll<SVGPathElement, number>("path").data(live, (c) => String(c)).join("path")
      .attr("class", (c) => "sv-flow sv-hit" + (sel.chosen.has(c) ? " sv-chosen" : ""))
      .attr("d", path)
      .attr("stroke-width", (c) => geom.get(c)!.w)
      .attr("stroke-dasharray", (c) => (back.has(c) ? "4 3" : null))
      .attr("stroke-opacity", (c) => (sel.shows(c) ? null : 0.12))
      .on("mousemove", (e: MouseEvent, c) => tip.show(e, `${p.labels[c]}: ${counts[c]} of ${base[c]} rows`))
      .on("mouseleave", () => tip.hide())
      .on("click", (_, c) => sel.toggle([c]));
    const t = nodeTotals(p, counts);
    const ids = columns.flat();
    const nodes = nodesG.selectAll<SVGGElement, number>("g").data(ids, (i) => String(i)).join((enter) => {
      const g = enter.append("g").attr("class", "sv-hit");
      g.append("rect").attr("class", "sv-bar").attr("width", NODE_W);
      g.append("text").attr("class", "sv-node-label").attr("x", 0).attr("y", -5);
      return g;
    });
    nodes.attr("transform", (i) => `translate(${xOf(layer[i]!)},${top[i]})`);
    nodes.select("rect").attr("height", (i) => height[i]!)
      .style("fill", (i) => colorOf(i))
      .classed("sv-chosen", (i) => incident(p, i).some((c) => sel.chosen.has(c)));
    nodes.select("text").text((i) => short(p.nodes[i]!))
      .classed("sv-quiet", (i) => Math.max(t.out[i]!, t.into[i]!) === 0);
    nodes
      .on("mousemove", (e: MouseEvent, i) => tip.show(e, `${p.nodes[i]}: ${t.into[i]} in, ${t.out[i]} out`))
      .on("mouseleave", () => tip.hide())
      .on("click", (_, i) => sel.toggle(incident(p, i)));
  };
  paint(engine.state);
  return { el: f.root, update: paint, destroy() {} };
}
