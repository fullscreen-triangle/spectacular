import type { DimInit, FilterOp, GroupInit, Request, Response } from "./protocol";

export interface EngineState {
  /** Per row: 1 when the row passes every filter. */
  mask: Uint8Array;
  counts: Uint32Array[];
  /** Counts before any filter, for drawing context behind the filtered bars. */
  baseline: Uint32Array[];
  binnings: number[][];
  selectedCount: number;
  rows: number;
  /** Crossfilter time for the last update, in ms. */
  elapsed: number;
}

type Listener = (state: EngineState) => void;

/**
 * The page's handle on the crossfilter worker. Filter calls are coalesced: while one
 * request is in flight, later calls only replace the pending op for their dimension,
 * so a fast brush never queues work the user has already moved past.
 */
export class Engine {
  private seq = 0;
  private inFlight = false;
  private pending = new Map<number, FilterOp>();
  private sent: FilterOp[] = [];
  private listeners = new Set<Listener>();
  readonly state: EngineState;
  /** The filter on each dimension that the current `state` reflects (cleared ones absent). */
  readonly filters = new Map<number, FilterOp>();

  private constructor(private worker: Worker, rows: number, ready: Extract<Response, { type: "ready" }>) {
    this.state = {
      mask: new Uint8Array(rows).fill(1),
      counts: ready.counts,
      baseline: ready.counts.map((c) => c.slice()),
      binnings: ready.binnings,
      selectedCount: ready.selectedCount,
      rows,
      elapsed: 0,
    };
    worker.onmessage = (e: MessageEvent<Response>) => this.receive(e.data);
  }

  static start(rows: number, dims: DimInit[], groups: GroupInit[], worker?: Worker): Promise<Engine> {
    const w = worker ?? new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    return new Promise((resolve, reject) => {
      w.onmessage = (e: MessageEvent<Response>) => {
        if (e.data.type === "ready") resolve(new Engine(w, rows, e.data));
        else if (e.data.type === "error") reject(new Error(e.data.message));
      };
      w.onerror = (e) => reject(new Error(e.message));
      // Columns are copied, not transferred: the page keeps its own for rendering.
      w.postMessage({ type: "init", rows, dims, groups } satisfies Request);
    });
  }

  on(listener: Listener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  filter(op: FilterOp) {
    this.pending.set(op.dim, op);
    this.flush();
  }

  dispose() {
    this.worker.terminate();
    this.listeners.clear();
  }

  private flush() {
    if (this.inFlight || this.pending.size === 0) return;
    const ops = [...this.pending.values()];
    this.pending.clear();
    this.sent = ops;
    this.inFlight = true;
    this.worker.postMessage({ type: "filter", seq: ++this.seq, ops } satisfies Request);
  }

  private receive(msg: Response) {
    if (msg.type === "error") {
      this.inFlight = false;
      console.error("spectacular engine:", msg.message);
      this.flush();
      return;
    }
    if (msg.type !== "update") return;
    const s = this.state;
    if (msg.mask) s.mask.set(msg.mask);
    else if (msg.changed) for (const r of msg.changed) s.mask[r] = s.mask[r]! ^ 1;
    s.counts = msg.counts;
    s.selectedCount = msg.selectedCount;
    s.elapsed = msg.elapsed;
    for (const op of this.sent) "clear" in op ? this.filters.delete(op.dim) : this.filters.set(op.dim, op);
    this.inFlight = false;
    for (const l of this.listeners) l(s);
    this.flush();
  }
}
