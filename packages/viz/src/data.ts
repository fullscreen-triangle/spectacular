import { dsvFormat } from "d3";
import type { FieldType, Spec } from "./spec.ts";
import { declaredType, fieldsUsed } from "./spec.ts";

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

/** The fields the spec reads: those its views use, its key, then those it declares. */
export function neededFields(spec: Spec): string[] {
  const names = new Set(spec.views.flatMap((v) => fieldsUsed(v).map(([f]) => f)));
  if (typeof spec.key === "string" && spec.key) names.add(spec.key);
  for (const f of Object.keys(spec.fields ?? {})) names.add(f);
  return [...names];
}

/** Every property any row has, in order of first appearance. */
function propertiesOf(rows: Record<string, unknown>[]): string[] {
  const names = new Set<string>();
  for (const r of rows) if (r && typeof r === "object") for (const k of Object.keys(r)) names.add(k);
  return [...names];
}

/**
 * A table's columns: a CSV's `header`, or for JSON rows every property any row has, so
 * rows need not all carry the same properties (a row without one holds a missing value).
 */
function fromRows(spec: Spec, rows: Record<string, unknown>[], header = propertiesOf(rows)): Table {
  const columns = new Map<string, Column>();
  for (const name of neededFields(spec)) {
    if (!header.includes(name)) continue;
    const raw = rows.map((r) => r?.[name]);
    // A key identifies rows, so it is a category even when its values look like numbers.
    const type = declaredType(spec.fields?.[name]) ?? (name === spec.key ? "nominal" : infer(raw));
    columns.set(name, type === "quantitative" ? quantitative(toNumbers(raw)) : nominal(raw));
  }
  return withDomains(spec, { rows: rows.length, columns });
}

/**
 * Puts each nominal field with a declared domain in that order: the domain's categories
 * first, with or without rows, then any other values in their first-appearance order.
 */
function withDomains(spec: Spec, table: Table): Table {
  for (const [name, c] of table.columns) {
    const domain = spec.fields?.[name]?.domain;
    if (c.type !== "nominal" || !Array.isArray(domain)) continue;
    const labels = domain.filter((d): d is string => typeof d === "string");
    const at = new Map(labels.map((l, i) => [l, i]));
    for (const l of c.labels)
      if (!at.has(l)) {
        at.set(l, labels.length);
        labels.push(l);
      }
    const remap = c.labels.map((l) => at.get(l)!);
    table.columns.set(name, { type: "nominal", codes: Uint32Array.from(c.codes, (k) => remap[k]!), labels });
  }
  return table;
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
    } else if ((declaredType(spec.fields?.[name]) ?? (name === spec.key ? "nominal" : undefined)) === "nominal" || (col.length > 0 && typeof col[0] === "string")) {
      column = nominal(col as ArrayLike<unknown>);
    } else {
      column = quantitative(col instanceof Float64Array ? col : Float64Array.from(col as ArrayLike<number>));
    }
    const n = column.type === "quantitative" ? column.values.length : column.codes.length;
    if (rows >= 0 && n !== rows) throw new Error(`column "${name}" has ${n} rows, others have ${rows}`);
    rows = n;
    columns.set(name, column);
  }
  return withDomains(spec, { rows: Math.max(rows, 0), columns });
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
  const format = data.format ?? (/\.json(ld)?(\?|#|$)/i.test(data.url) ? "json" : /\.tsv(\?|#|$)/i.test(data.url) ? "tsv" : "csv");
  if (format === "json") return fromRows(spec, rowsAt(JSON.parse(text), data.pointer ?? ""));
  const parsed = dsvFormat(format === "tsv" ? "\t" : ",").parse(text);
  return fromRows(spec, parsed as unknown as Record<string, unknown>[], parsed.columns);
}

/** Follows an RFC 6901 JSON Pointer to the array of row objects inside a document. */
export function rowsAt(doc: unknown, pointer: string): Record<string, unknown>[] {
  let at: unknown = doc;
  for (const raw of pointer === "" ? [] : pointer.slice(1).split("/")) {
    const key = raw.replace(/~1/g, "/").replace(/~0/g, "~");
    at = typeof at === "object" && at !== null ? (at as Record<string, unknown>)[key] : undefined;
    if (at === undefined) throw new Error(`data.pointer ${JSON.stringify(pointer)}: nothing at "${key}"`);
  }
  if (!Array.isArray(at)) throw new Error(`data.pointer ${JSON.stringify(pointer || "(document)")} must lead to an array of row objects`);
  return at as Record<string, unknown>[];
}

export function fieldTypes(table: Table): Record<string, FieldType> {
  return Object.fromEntries([...table.columns].map(([name, c]) => [name, c.type]));
}
