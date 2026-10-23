/** Messages between the page and the crossfilter worker. */

export type DimInit =
  | { kind: "numeric"; values: Float64Array }
  | { kind: "categorical"; codes: Uint32Array; cardinality: number };

export type GroupInit =
  | { kind: "histogram"; dim: number; bins: number; min?: number; max?: number }
  | { kind: "category"; dim: number }
  | { kind: "grid"; x: number; y: number; xBins: number; yBins: number };

export type FilterOp =
  | { dim: number; range: [number, number] }
  | { dim: number; codes: number[] }
  | { dim: number; clear: true };

export type Request =
  | { type: "init"; rows: number; dims: DimInit[]; groups: GroupInit[] }
  | { type: "filter"; seq: number; ops: FilterOp[] };

export interface Counts {
  counts: Uint32Array[];
  selectedCount: number;
}

export type Response =
  | ({ type: "ready"; binnings: number[][] } & Counts)
  | ({
      type: "update";
      seq: number;
      /** Rows whose selected state flipped, when that is cheaper than the whole mask. */
      changed?: Uint32Array;
      mask?: Uint8Array;
      /** Time spent inside the crossfilter, in ms. */
      elapsed: number;
    } & Counts)
  | { type: "error"; message: string };
