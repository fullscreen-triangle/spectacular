/**
 * The view spec: what another repo hands spectacular to get a figure.
 *
 * A spec names data, the fields to read from it, and a set of linked views. Brushing
 * any brushable view filters every other view through the crossfilter. A spec is plain
 * JSON, so a script, a build step, or a model can write one; `validate` says exactly
 * what is wrong with one that does not hold.
 */

import { MAX_FIELD_PALETTE, MAX_PALETTE, isColor } from "./colors.ts";

export const SPEC_VERSION = "spectacular/v0";

export type FieldType = "quantitative" | "nominal";

export interface FieldDef {
  type?: FieldType;
  /** Axis label; defaults to the field name. */
  title?: string;
  /**
   * A nominal field's categories, in order. They keep their place even with no rows;
   * values the list does not name follow it, in order of first appearance.
   */
  domain?: string[];
  /**
   * A nominal field's colours, used by every view that colours by it: a list (in the
   * order of `domain`, or of first appearance) or category name to colour.
   */
  palette?: string[] | Record<string, string>;
}

export type DataFormat = "csv" | "tsv" | "json";

export type DataSource =
  /**
   * CSV, TSV or JSON. `format` defaults from the extension (.json/.jsonld: json, .tsv:
   * tsv, otherwise csv). `pointer` is an RFC 6901 JSON Pointer to the array of row
   * objects inside a JSON document, e.g. "/executions"; the default is the document.
   */
  | { url: string; format?: DataFormat; pointer?: string }
  /** Inline rows. */
  | { values: Record<string, unknown>[] }
  /** Columns supplied by the embedding program through `mount(..., { columns })`. */
  | { columns: "external" };

interface ViewBase {
  id: string;
  title?: string;
  /** Pixel height of the plot area. */
  height?: number;
  /** Columns this view spans in the grid layout. */
  span?: number;
}

/** Colours a view may set: one for all its marks, or one per category. */
interface Colors {
  /** One colour for the marks when no `color` field is given: "#rrggbb", "#rgb" or a CSS name. */
  fill?: string;
  /** Category colours, in order of first appearance (1 to 10; they repeat). */
  palette?: string[];
}

export interface ScatterView extends ViewBase, Colors {
  type: "scatter";
  x: string;
  y: string;
  color?: string;
  /** "xy" lets the view filter the others with a rectangle brush. */
  brush?: "xy" | "none";
  pointSize?: number;
}

export interface HistogramView extends ViewBase {
  type: "histogram";
  fill?: string;
  field: string;
  bins?: number;
  brush?: "x" | "none";
}

export interface BarsView extends ViewBase {
  type: "bars";
  palette?: string[];
  field: string;
  /** A second nominal field: each bar is split into its categories, stacked. */
  color?: string;
  /** "select" toggles categories in the filter by clicking. */
  brush?: "select" | "none";
}

export interface DensityView extends ViewBase {
  type: "density";
  x: string;
  y: string;
  bins?: [number, number];
}

export interface Scatter3DView extends ViewBase, Colors {
  type: "scatter3d";
  x: string;
  y: string;
  z: string;
  color?: string;
  pointSize?: number;
}

/**
 * A number against a category: one row of points per category, the value across,
 * spread a little vertically so equal values stay visible. Brushing a range of the
 * value filters the other views.
 */
export interface StripView extends ViewBase, Colors {
  type: "strip";
  /** The value: quantitative. */
  x: string;
  /** The category: nominal. */
  y: string;
  color?: string;
  brush?: "x" | "none";
  pointSize?: number;
}

/**
 * Two nominal fields as a grid, each cell coloured by the rows holding that pair after
 * the other views' filters. Clicking a cell selects its pair; clicking a row or column
 * label selects all its cells.
 */
export interface MatrixView extends ViewBase {
  type: "matrix";
  x: string;
  y: string;
  brush?: "select" | "none";
}

/**
 * Each row as a quantity from `low` to `high`, grouped in bands by `label`. Brushing a
 * range keeps the rows whose interval overlaps it.
 */
export interface IntervalView extends ViewBase, Colors {
  type: "interval";
  low: string;
  high: string;
  label?: string;
  color?: string;
  /**
   * "shared" (default): one axis for every band. "label": each band on its own scale,
   * for quantities in different units; there is then no brush.
   */
  scale?: "shared" | "label";
  brush?: "x" | "none";
}

/**
 * Views over edges: each row is one edge from `source` to `target` (two nominal fields),
 * and edges between the same two nodes are drawn as one, weighted by how many rows it
 * holds. Clicking an edge selects its rows; clicking a node selects every edge that
 * touches it. Each view keeps its own full shape while it filters the others.
 */
interface EdgeViewBase extends ViewBase {
  source: string;
  target: string;
  /** "select" toggles edges and nodes in the filter by clicking. */
  brush?: "select" | "none";
}

/** Nodes on one line, edges as arcs above it. */
export interface ArcView extends EdgeViewBase {
  type: "arc";
  /** Node order along the line: by rows touching the node (default), by name, or as first seen. */
  order?: "count" | "name" | "data";
}

/** Two columns, sources and targets, joined by flows as wide as their row counts. */
export interface SankeyView extends EdgeViewBase {
  type: "sankey";
}

/** Nodes placed in 3D by a force layout, edges as lines; orbitable. */
export interface Network3DView extends EdgeViewBase {
  type: "network3d";
}

/**
 * A directed graph in layers, left to right: each node one layer after the furthest
 * node that leads to it (a cycle is broken where it closes), edges as flows as wide as
 * their rows. For pipelines of steps, and rows passing from step to step.
 */
export interface LayersView extends EdgeViewBase {
  type: "layers";
}

export type View =
  | ScatterView | HistogramView | BarsView | DensityView | Scatter3DView | StripView | MatrixView | IntervalView
  | ArcView | SankeyView | Network3DView | LayersView;

/** True for the views whose rows are edges. */
export const isEdgeView = (v: View): v is ArcView | SankeyView | Network3DView | LayersView =>
  v.type === "arc" || v.type === "sankey" || v.type === "network3d" || v.type === "layers";
export type ViewType = View["type"];

export interface FigureOutput {
  format: "svg" | "pdf" | "png";
  /** Physical width; publication figures are sized for the page, not the screen. */
  width: number;
  unit?: "mm" | "in" | "px";
  dpi?: number;
}

/**
 * "auto" follows the host page's colour scheme; "light" and "dark" fix it; "paper" is
 * for print: white, no panel borders, darker axes.
 */
export type Theme = "auto" | "light" | "dark" | "paper";

export interface Spec {
  $schema?: string;
  version?: typeof SPEC_VERSION;
  title?: string;
  description?: string;
  theme?: Theme;
  data: DataSource;
  fields?: Record<string, FieldDef>;
  /**
   * A field whose values identify rows. Selections then report those values, and rows
   * can be selected from outside by them (`Handle.select`).
   */
  key?: string;
  views: View[];
  layout?: { columns?: number };
  output?: { interactive?: boolean; figure?: FigureOutput };
}

export interface Problem {
  /** JSON-pointer-like path to the offending value, e.g. `/views/2/x`. */
  path: string;
  message: string;
}

const VIEW_FIELDS: Record<ViewType, { required: string[]; optional: string[] }> = {
  scatter: { required: ["x", "y"], optional: ["color"] },
  histogram: { required: ["field"], optional: [] },
  bars: { required: ["field"], optional: ["color"] },
  density: { required: ["x", "y"], optional: [] },
  scatter3d: { required: ["x", "y", "z"], optional: ["color"] },
  strip: { required: ["x", "y"], optional: ["color"] },
  matrix: { required: ["x", "y"], optional: [] },
  interval: { required: ["low", "high"], optional: ["label", "color"] },
  arc: { required: ["source", "target"], optional: [] },
  sankey: { required: ["source", "target"], optional: [] },
  network3d: { required: ["source", "target"], optional: [] },
  layers: { required: ["source", "target"], optional: [] },
};

const NEEDS: Record<ViewType, Record<string, FieldType | undefined>> = {
  scatter: { x: "quantitative", y: "quantitative", color: undefined },
  histogram: { field: "quantitative" },
  bars: { field: "nominal", color: "nominal" },
  density: { x: "quantitative", y: "quantitative" },
  scatter3d: { x: "quantitative", y: "quantitative", z: "quantitative", color: undefined },
  strip: { x: "quantitative", y: "nominal", color: undefined },
  matrix: { x: "nominal", y: "nominal" },
  interval: { low: "quantitative", high: "quantitative", label: "nominal", color: undefined },
  arc: { source: "nominal", target: "nominal" },
  sankey: { source: "nominal", target: "nominal" },
  network3d: { source: "nominal", target: "nominal" },
  layers: { source: "nominal", target: "nominal" },
};

const BRUSHES: Record<ViewType, string[]> = {
  scatter: ["xy", "none"],
  histogram: ["x", "none"],
  bars: ["select", "none"],
  density: [],
  scatter3d: [],
  strip: ["x", "none"],
  matrix: ["select", "none"],
  interval: ["x", "none"],
  arc: ["select", "none"],
  sankey: ["select", "none"],
  network3d: ["select", "none"],
  layers: ["select", "none"],
};

const COMMON_VIEW_KEYS = ["id", "type", "title", "height", "span"];
const EXTRA_VIEW_KEYS: Record<ViewType, string[]> = {
  scatter: ["brush", "pointSize", "fill", "palette"],
  histogram: ["brush", "bins", "fill"],
  bars: ["brush", "palette"],
  density: ["bins"],
  scatter3d: ["pointSize", "fill", "palette"],
  strip: ["brush", "pointSize", "fill", "palette"],
  matrix: ["brush"],
  interval: ["brush", "scale", "fill", "palette"],
  arc: ["brush", "order"],
  sankey: ["brush"],
  network3d: ["brush"],
  layers: ["brush"],
};
const TOP_KEYS = ["$schema", "version", "title", "description", "theme", "data", "fields", "key", "views", "layout", "output"];
const FIELD_KEYS = ["type", "title", "domain", "palette"];
const THEMES: Theme[] = ["auto", "light", "dark", "paper"];
const DATA_KEYS = { url: ["url", "format", "pointer"], values: ["values"], columns: ["columns"] };
const DATA_FORMATS: DataFormat[] = ["csv", "tsv", "json"];
const FIGURE_FORMATS: FigureOutput["format"][] = ["svg", "pdf", "png"];
const FIGURE_UNITS = ["mm", "in", "px"];
const COLOR_RULE = 'a colour: "#rrggbb", "#rgb" or a CSS colour name such as "steelblue"';

function distance(a: string, b: string): number {
  const d = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    let prev = d[0]!;
    d[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const cur = d[j]!;
      d[j] = Math.min(d[j]! + 1, d[j - 1]! + 1, prev + (a[i - 1] === b[j - 1] ? 0 : 1));
      prev = cur;
    }
  }
  return d[b.length]!;
}

/** Unknown properties are errors: a typo silently ignored would draw a different figure. */
function unknownKeys(obj: Record<string, unknown>, allowed: string[], at: string, bad: (path: string, message: string) => void) {
  for (const key of Object.keys(obj)) {
    if (allowed.includes(key)) continue;
    const near = allowed.filter((k) => distance(k.toLowerCase(), key.toLowerCase()) <= 2);
    bad(`${at}/${key}`, `unknown property "${key}"` + (near.length ? `; did you mean "${near[0]}"?` : `; allowed: ${allowed.join(", ")}`));
  }
}

const isObject = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

export const MAX_DIMENSIONS = 32;

/**
 * Crossfilter dimensions a view needs: one per brushed field, plus the fields a
 * histogram, bars or density view bins (those exist even without a brush).
 */
export function dimensionsNeeded(view: View): number {
  switch (view.type) {
    case "scatter": return (view.brush ?? "xy") === "xy" ? 2 : 0;
    case "histogram": case "bars": return 1;
    case "density": return 2;
    case "scatter3d": return 0;
    case "strip": return (view.brush ?? "x") === "x" ? 1 : 0;
    case "matrix": return 1;
    case "interval": return (view.brush ?? "x") === "x" && view.scale !== "label" ? 2 : 0;
    case "arc": case "sankey": case "network3d": case "layers": return 1;
  }
}

/**
 * The type a declared field loads as: its `type`, else nominal when it has a domain or
 * a palette (those are for categories), else whatever the data says.
 */
export function declaredType(def: FieldDef | undefined): FieldType | undefined {
  if (!def) return undefined;
  if (def.type !== undefined && def.type !== null) return def.type === "quantitative" ? "quantitative" : "nominal";
  return def.domain !== undefined || def.palette !== undefined ? "nominal" : undefined;
}

/** Every field a view reads, with the type it needs (undefined: any type). */
export function fieldsUsed(view: View): [string, FieldType | undefined][] {
  const needs = NEEDS[view.type];
  if (!needs) return []; // an unknown type is reported by validate; it reads no fields
  return Object.keys(needs)
    .map((k) => [(view as unknown as Record<string, unknown>)[k], needs[k]] as const)
    .filter((e): e is readonly [string, FieldType | undefined] => typeof e[0] === "string")
    .map(([f, t]) => [f, t]);
}

/**
 * Checks a spec's structure. `types` (known field types, e.g. inferred from loaded
 * data) also enables checks that each view's fields have the type it needs.
 */
export function validate(input: unknown, types?: Record<string, FieldType>): Problem[] {
  const problems: Problem[] = [];
  const bad = (path: string, message: string) => problems.push({ path, message });
  if (!isObject(input)) return [{ path: "", message: "a spec must be a JSON object" }];
  const spec = input;
  unknownKeys(spec, TOP_KEYS, "", bad);

  if (spec.version !== undefined && spec.version !== SPEC_VERSION)
    bad("/version", `unsupported version ${JSON.stringify(spec.version)}; this build reads "${SPEC_VERSION}"`);
  if (spec.theme !== undefined && !THEMES.includes(spec.theme as Theme)) bad("/theme", `one of ${THEMES.join(", ")}`);

  const data = spec.data;
  if (!isObject(data)) bad("/data", "required: { url } | { values } | { columns: \"external\" }");
  else {
    const kinds = ["url", "values", "columns"].filter((k) => k in data);
    if (kinds.length !== 1) bad("/data", `give exactly one of url, values, columns (got ${kinds.join(", ") || "none"})`);
    else if ("url" in data) {
      unknownKeys(data, DATA_KEYS.url, "/data", bad);
      if (typeof data.url !== "string") bad("/data/url", "must be a string");
      if (data.format !== undefined && !DATA_FORMATS.includes(data.format as DataFormat))
        bad("/data/format", `one of ${DATA_FORMATS.join(", ")}`);
      if (data.pointer !== undefined && !(typeof data.pointer === "string" && (data.pointer === "" || data.pointer.startsWith("/"))))
        bad("/data/pointer", 'a JSON Pointer: "" or a string starting with "/"');
    } else if ("values" in data) {
      unknownKeys(data, DATA_KEYS.values, "/data", bad);
      if (!(Array.isArray(data.values) && data.values.every(isObject))) bad("/data/values", "must be an array of row objects");
    } else {
      unknownKeys(data, DATA_KEYS.columns, "/data", bad);
      if (data.columns !== "external") bad("/data/columns", 'must be "external"');
    }
  }

  const declared: Record<string, FieldType> = { ...types };
  if (spec.fields !== undefined) {
    if (!isObject(spec.fields)) bad("/fields", "must be an object of field name to definition");
    else
      for (const [name, def] of Object.entries(spec.fields)) {
        if (!isObject(def)) { bad(`/fields/${name}`, "must be an object"); continue; }
        const at = `/fields/${name}`;
        unknownKeys(def, FIELD_KEYS, at, bad);
        if (def.type !== undefined) {
          if (def.type !== "quantitative" && def.type !== "nominal")
            bad(`${at}/type`, 'must be "quantitative" or "nominal"');
          else declared[name] = def.type;
        }
        if (def.domain !== undefined) {
          const d = def.domain;
          if (!Array.isArray(d) || d.length === 0 || !d.every((c) => typeof c === "string") || new Set(d).size !== d.length)
            bad(`${at}/domain`, "a list of category names, each once");
        }
        if (def.palette !== undefined) {
          const p = def.palette;
          const entries = Array.isArray(p) ? p.map((c, k) => [String(k), c] as const) : isObject(p) ? Object.entries(p) : [];
          if (entries.length < 1 || entries.length > MAX_FIELD_PALETTE)
            bad(`${at}/palette`, `a list of 1 to ${MAX_FIELD_PALETTE} colours, or category name to colour`);
          else for (const [k, c] of entries) if (!isColor(c)) bad(`${at}/palette/${k}`, COLOR_RULE);
        }
        // A domain or a palette makes a field nominal.
        if (def.domain !== undefined || def.palette !== undefined) {
          if (def.type === "quantitative") bad(at, "a domain or a palette is for a nominal field; this one is quantitative");
          else if (def.type === undefined) declared[name] = "nominal";
        }
      }
  }

  if (spec.key !== undefined) {
    if (typeof spec.key !== "string" || spec.key === "") bad("/key", "the name of a field whose values identify rows");
    else if (types && !(spec.key in types)) bad("/key", `no field "${spec.key}" in the data`);
  }

  if (!Array.isArray(spec.views) || spec.views.length === 0) {
    bad("/views", "required: a non-empty array of views");
    return problems;
  }
  const ids = new Set<string>();
  spec.views.forEach((v: unknown, i: number) => {
    const at = `/views/${i}`;
    if (!isObject(v)) return bad(at, "must be an object");
    if (typeof v.id !== "string" || v.id === "") bad(`${at}/id`, "required: a non-empty string");
    else if (ids.has(v.id)) bad(`${at}/id`, `duplicate id "${v.id}"`);
    else ids.add(v.id);
    const type = v.type as ViewType;
    if (!(type in VIEW_FIELDS))
      return bad(`${at}/type`, `unknown view type ${JSON.stringify(v.type)}; one of ${Object.keys(VIEW_FIELDS).join(", ")}`);
    const { required, optional } = VIEW_FIELDS[type];
    unknownKeys(v, [...COMMON_VIEW_KEYS, ...required, ...optional, ...EXTRA_VIEW_KEYS[type]], at, bad);
    for (const key of required)
      if (typeof v[key] !== "string") bad(`${at}/${key}`, `required: the name of a field`);
    for (const key of [...required, ...optional]) {
      const field = v[key];
      if (typeof field !== "string") continue;
      const need = NEEDS[type][key];
      const have = declared[field];
      if (types && !(field in types)) bad(`${at}/${key}`, `no field "${field}" in the data`);
      else if (need && have && need !== have) bad(`${at}/${key}`, `${type}.${key} needs a ${need} field; "${field}" is ${have}`);
    }
    // Views without a brush already reported "brush" as an unknown property.
    if (v.brush !== undefined && BRUSHES[type].length && !BRUSHES[type].includes(v.brush as string))
      bad(`${at}/brush`, `one of ${BRUSHES[type].join(", ")}`);
    if (v.bins !== undefined) {
      const ok = type === "density"
        ? Array.isArray(v.bins) && v.bins.length === 2 && v.bins.every((b) => Number.isInteger(b) && b > 0 && b <= 1024)
        : Number.isInteger(v.bins) && (v.bins as number) > 0 && (v.bins as number) <= 10_000;
      if (!ok) bad(`${at}/bins`, type === "density" ? "[xBins, yBins], integers in 1..1024" : "an integer in 1..10000");
    }
    if (v.order !== undefined && !["count", "name", "data"].includes(v.order as string))
      bad(`${at}/order`, "one of count, name, data");
    if (v.scale !== undefined && !["shared", "label"].includes(v.scale as string)) bad(`${at}/scale`, "one of shared, label");
    if (v.scale === "label" && v.brush === "x") bad(`${at}/brush`, 'a brush needs one shared scale; with "scale": "label" it is "none"');
    if (v.fill !== undefined && !isColor(v.fill)) bad(`${at}/fill`, COLOR_RULE);
    if (v.palette !== undefined) {
      if (!Array.isArray(v.palette) || v.palette.length < 1 || v.palette.length > MAX_PALETTE)
        bad(`${at}/palette`, `a list of 1 to ${MAX_PALETTE} colours`);
      else v.palette.forEach((c, k) => { if (!isColor(c)) bad(`${at}/palette/${k}`, COLOR_RULE); });
    }
    for (const key of ["height", "span", "pointSize"] as const)
      if (v[key] !== undefined && !(typeof v[key] === "number" && (v[key] as number) > 0)) bad(`${at}/${key}`, "must be a positive number");
  });

  if (spec.layout !== undefined) {
    if (!isObject(spec.layout)) bad("/layout", "must be an object");
    else {
      unknownKeys(spec.layout, ["columns"], "/layout", bad);
      const c = spec.layout.columns;
      if (c !== undefined && !(Number.isInteger(c) && (c as number) >= 1 && (c as number) <= 12)) bad("/layout/columns", "an integer in 1..12");
    }
  }
  if (spec.output !== undefined) {
    if (!isObject(spec.output)) bad("/output", "must be an object");
    else {
      unknownKeys(spec.output, ["interactive", "figure"], "/output", bad);
      if (spec.output.interactive !== undefined && typeof spec.output.interactive !== "boolean")
        bad("/output/interactive", "must be true or false");
    }
  }
  const views = spec.views.filter((v: unknown): v is View => isObject(v) && (v.type as string) in VIEW_FIELDS);
  const dims = views.reduce((n: number, v: View) => n + dimensionsNeeded(v), typeof spec.key === "string" && spec.key ? 1 : 0);
  if (dims > MAX_DIMENSIONS)
    bad("/views", `these views need ${dims} crossfilter dimensions; the limit is ${MAX_DIMENSIONS} (scatter with brush 2, interval with brush 2, density 2, histogram 1, bars 1, strip with brush 1, matrix 1, arc 1, sankey 1, network3d 1, layers 1, scatter3d 0, and 1 for a key)`);

  const figure = isObject(spec.output) ? spec.output.figure : undefined;
  if (figure !== undefined) {
    if (!isObject(figure)) bad("/output/figure", "must be an object");
    else {
      unknownKeys(figure, ["format", "width", "unit", "dpi"], "/output/figure", bad);
      if (!FIGURE_FORMATS.includes(figure.format as FigureOutput["format"]))
        bad("/output/figure/format", `one of ${FIGURE_FORMATS.join(", ")}`);
      if (!(typeof figure.width === "number" && figure.width > 0)) bad("/output/figure/width", "required: a positive number");
      if (figure.unit !== undefined && !FIGURE_UNITS.includes(figure.unit as "mm")) bad("/output/figure/unit", `one of ${FIGURE_UNITS.join(", ")}`);
      if (figure.dpi !== undefined && !(typeof figure.dpi === "number" && figure.dpi > 0)) bad("/output/figure/dpi", "must be a positive number");
    }
  }
  return problems;
}

/**
 * A JSON Schema (draft 2020-12) for specs, built from the same tables `validate` reads,
 * so programs in any language can check or autocomplete a spec. It covers structure
 * only; `validate` also checks fields against the data and the dimension limit.
 */
export function jsonSchema(): Record<string, unknown> {
  const str = { type: "string" };
  const field = { type: "string", minLength: 1, description: "the name of a field in the data" };
  const positive = { type: "number", exclusiveMinimum: 0 };
  const color = { type: "string", description: "a colour: #rrggbb, #rgb or a CSS colour name" };
  const extras: Record<string, Record<string, unknown>> = {
    brush: {},
    pointSize: positive,
    bins: { type: "integer", minimum: 1, maximum: 10_000 },
    order: { enum: ["count", "name", "data"] },
    scale: { enum: ["shared", "label"] },
    fill: color,
    palette: { type: "array", items: color, minItems: 1, maxItems: MAX_PALETTE },
  };
  const view = (type: ViewType) => {
    const { required, optional } = VIEW_FIELDS[type];
    const properties: Record<string, unknown> = {
      id: { type: "string", minLength: 1 },
      type: { const: type },
      title: str,
      height: positive,
      span: positive,
    };
    for (const k of [...required, ...optional]) properties[k] = field;
    for (const k of EXTRA_VIEW_KEYS[type]) properties[k] = k === "brush" ? { enum: BRUSHES[type] } : extras[k];
    if (type === "density")
      properties.bins = { type: "array", items: { type: "integer", minimum: 1, maximum: 1024 }, minItems: 2, maxItems: 2 };
    return { type: "object", required: ["id", "type", ...required], properties, additionalProperties: false };
  };
  const closed = (properties: Record<string, unknown>, required: string[]) =>
    ({ type: "object", required, properties, additionalProperties: false });
  return {
    $schema: "https://json-schema.org/draft/2020-12/schema",
    title: `spectacular view spec (${SPEC_VERSION})`,
    ...closed({
      $schema: str,
      version: { const: SPEC_VERSION },
      title: str,
      description: str,
      theme: { enum: THEMES },
      data: {
        oneOf: [
          closed({ url: str, format: { enum: DATA_FORMATS }, pointer: { type: "string", pattern: "^(/.*)?$" } }, ["url"]),
          closed({ values: { type: "array", items: { type: "object" } } }, ["values"]),
          closed({ columns: { const: "external" } }, ["columns"]),
        ],
      },
      fields: {
        type: "object",
        additionalProperties: closed({
          type: { enum: ["quantitative", "nominal"] },
          title: str,
          domain: { type: "array", items: str, minItems: 1, uniqueItems: true },
          palette: {
            oneOf: [
              { type: "array", items: color, minItems: 1, maxItems: MAX_FIELD_PALETTE },
              { type: "object", additionalProperties: color, minProperties: 1, maxProperties: MAX_FIELD_PALETTE },
            ],
          },
        }, []),
      },
      key: { type: "string", minLength: 1, description: "a field whose values identify rows" },
      views: { type: "array", minItems: 1, items: { oneOf: (Object.keys(VIEW_FIELDS) as ViewType[]).map(view) } },
      layout: closed({ columns: { type: "integer", minimum: 1, maximum: 12 } }, []),
      output: closed({
        interactive: { type: "boolean" },
        figure: closed({ format: { enum: FIGURE_FORMATS }, width: positive, unit: { enum: FIGURE_UNITS }, dpi: positive }, ["format", "width"]),
      }, []),
    }, ["data", "views"]),
  };
}

export class SpecError extends Error {
  // A plain field, not a parameter property, so Node can run this file by stripping types.
  readonly problems: Problem[];
  constructor(problems: Problem[]) {
    super("invalid spec:\n" + problems.map((p) => `  ${p.path || "/"}: ${p.message}`).join("\n"));
    this.problems = problems;
  }
}
