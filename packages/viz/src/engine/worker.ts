/// <reference lib="webworker" />
import init, { Crossfilter, wasmMemory } from "chigutiro-wasm";
import type { Request, Response } from "./protocol";

let cf: Crossfilter | undefined;
let groups = 0;
const ready = init();

const post = (msg: Response, transfer: Transferable[] = []) =>
  (self as unknown as DedicatedWorkerGlobalScope).postMessage(msg, transfer);

function counts(): Uint32Array[] {
  return Array.from({ length: groups }, (_, g) => cf!.groupCounts(g));
}

self.onmessage = async (event: MessageEvent<Request>) => {
  await ready;
  const req = event.data;
  try {
    if (req.type === "init") {
      cf?.free();
      cf = new Crossfilter(req.rows);
      for (const d of req.dims) {
        if (d.kind === "numeric") cf.addNumeric(d.values);
        else cf.addCategorical(d.codes, d.cardinality);
      }
      for (const g of req.groups) {
        if (g.kind === "histogram") cf.addHistogram(g.dim, g.bins, g.min, g.max);
        else if (g.kind === "category") cf.addCategoryCounts(g.dim);
        else cf.addGrid(g.x, g.y, g.xBins, g.yBins);
      }
      groups = req.groups.length;
      const binnings = Array.from({ length: groups }, (_, g) => Array.from(cf!.groupBinning(g)));
      const c = counts();
      post({ type: "ready", binnings, counts: c, selectedCount: cf.selectedCount() }, c.map((a) => a.buffer));
      return;
    }

    if (!cf) throw new Error("filter before init");
    const t0 = performance.now();
    // Ops are applied in order; a mask is sent instead of the changed list once the
    // list would be larger than the mask itself.
    const flipped = new Set<number>();
    let wholeMask = false;
    for (const op of req.ops) {
      if ("range" in op) cf.filterRange(op.dim, op.range[0], op.range[1]);
      else if ("codes" in op) cf.filterCodes(op.dim, Uint32Array.from(op.codes));
      else cf.clearFilter(op.dim);
      const n = cf.changedLen();
      if (wholeMask || n * 4 > cf.rows()) { wholeMask = true; continue; }
      const changed = new Uint32Array(wasmMemory().buffer, cf.changedPtr(), n);
      // A row flipped twice across ops is back where it started.
      for (const r of changed) if (!flipped.delete(r)) flipped.add(r);
    }
    const elapsed = performance.now() - t0;
    const c = counts();
    const transfer: Transferable[] = c.map((a) => a.buffer);
    let msg: Response;
    if (wholeMask) {
      const mask = new Uint8Array(wasmMemory().buffer, cf.selectedPtr(), cf.rows()).slice();
      transfer.push(mask.buffer);
      msg = { type: "update", seq: req.seq, mask, counts: c, selectedCount: cf.selectedCount(), elapsed };
    } else {
      const changed = Uint32Array.from(flipped);
      transfer.push(changed.buffer);
      msg = { type: "update", seq: req.seq, changed, counts: c, selectedCount: cf.selectedCount(), elapsed };
    }
    post(msg, transfer);
  } catch (e) {
    post({ type: "error", message: e instanceof Error ? e.message : String(e) });
  }
};
