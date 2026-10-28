import { dsvFormat } from "d3";
import type { FieldType, Spec } from "./spec.ts";
import { fieldsUsed } from "./spec.ts";

export type Column =
  | { type: "quantitative"; values: Float64Array; extent: [number, number] }
  | { type: "nominal"; codes: Uint32Array; labels: string[] };

export interface Table {
  rows: number;
  columns: Map<string, Column>;
}

/** What an embedding program may pass for `data: { columns: "external" }`. */
export type ExternalColumn = ArrayLike<number> | readonly string[] | { codes: ArrayLike<number>; labels: string[] };

export function quantitative(values: Float64Array): Column {
  let lo = Infinity, hi = -Infinity;
  for (const v of values) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return { type: "quantitative", values, extent: lo <= hi ? [lo, hi] : [0, 1] };
}

export function nominal(raw: ArrayLike<unknown>): Column {
  const index = new Map<string, number>();
  const labels: string[] = [];
  const codes = new Uint32Array(raw.length);
  for (let i = 0; i < raw.length; i++) {
    const key = raw[i] == null ? "" : String(raw[i]);
    let c = index.get(key);
    if (c === undefined) {
      c = labels.length;
      index.set(key, c);
      labels.push(key);
    }
    codes[i] = c;
  }
  return { type: "nominal", codes, labels };
}

/** Quantitative when every non-empty value parses as a number. */
function infer(raw: ArrayLike<unknown>): FieldType {
  for (let i = 0; i < raw.length; i++) {
    const v = raw[i];
    if (v == null || v === "") continue;
    if (typeof v === "number") continue;
    if (typeof v === "string" && v.trim() !== "" && Number.isFinite(Number(v))) continue;
    return "nominal";
  }
  return "quantitative";
}

function toNumbers(raw: ArrayLike<unknown>): Float64Array {
  const out = new Float64Array(raw.length);
  for (let i = 0; i < raw.length; i++) {
    const v = raw[i];
    out[i] = v == null || v === "" ? NaN : Number(v);
  }
  return out;
}

/** The fields the spec reads: those its views use, plus those it declares. */
export function neededFields(spec: Spec): string[] {
  const names = new Set(spec.views.flatMap((v) => fieldsUsed(v).map(([f]) => f)));
  for (const f of Object.keys(spec.fields ?? {})) names.add(f);
  return [...names];
}

function fromRows(spec: Spec, rows: Record<string, unknown>[]): Table {
  const columns = new Map<string, Column>();
  for (const name of neededFields(spec)) {
    if (rows.length && !(name in rows[0]!)) continue;
    const raw = rows.map((r) => r[name]);
    const type = spec.fields?.[name]?.type ?? infer(raw);
    columns.set(name, type === "quantitative" ? quantitative(toNumbers(raw)) : nominal(raw));
  }
  return { rows: rows.length, columns };
}

function fromExternal(spec: Spec, external: Record<string, ExternalColumn>): Table {
  const columns = new Map<string, Column>();
  let rows = -1;
  for (const name of neededFields(spec)) {
    const col = external[name];
    if (col === undefined) continue;
    let column: Column;
    if ("codes" in col && "labels" in col) {
      column = { type: "nominal", codes: Uint32Array.from(col.codes), labels: col.labels };
    } else if (spec.fields?.[name]?.type === "nominal" || (col.length > 0 && typeof col[0] === "string")) {
      column = nominal(col as ArrayLike<unknown>);
    } else {
      column = quantitative(col instanceof Float64Array ? col : Float64Array.from(col as ArrayLike<number>));
    }
    const n = column.type === "quantitative" ? column.values.length : column.codes.length;
    if (rows >= 0 && n !== rows) throw new Error(`column "${name}" has ${n} rows, others have ${rows}`);
    rows = n;
    columns.set(name, column);
  }
  return { rows: Math.max(rows, 0), columns };
}

export async function loadTable(
  spec: Spec,
  external?: Record<string, ExternalColumn>,
  fetcher: typeof fetch = fetch,
): Promise<Table> {
  const data = spec.data;
  if ("columns" in data) {
    if (!external) throw new Error('data.columns is "external" but no columns were passed to mount()');
    return fromExternal(spec, external);
  }
  if ("values" in data) return fromRows(spec, data.values);
  const res = await fetcher(data.url);
  if (!res.ok) throw new Error(`fetching ${data.url}: ${res.status} ${res.statusText}`);
  const text = await res.text();
  if (/\.json(\?|$)/i.test(data.url)) return fromRows(spec, JSON.parse(text));
  const sep = /\.tsv(\?|$)/i.test(data.url) ? "\t" : ",";
  return fromRows(spec, dsvFormat(sep).parse(text) as unknown as Record<string, unknown>[]);
}

export function fieldTypes(table: Table): Record<string, FieldType> {
  return Object.fromEntries([...table.columns].map(([name, c]) => [name, c.type]));
}
