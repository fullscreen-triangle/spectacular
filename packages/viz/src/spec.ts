/**
 * The view spec: what another repo hands spectacular to get a figure.
 *
 * A spec names data, the fields to read from it, and a set of linked views. Brushing
 * any brushable view filters every other view through the crossfilter. A spec is plain
 * JSON, so a script, a build step, or a model can write one; `validate` says exactly
 * what is wrong with one that does not hold.
 */

export const SPEC_VERSION = "spectacular/v0";

export type FieldType = "quantitative" | "nominal";

export interface FieldDef {
  type?: FieldType;
  /** Axis label; defaults to the field name. */
  title?: string;
}

export type DataSource =
  /** CSV (or TSV by extension) or a JSON array of row objects. */
  | { url: string }
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

export interface ScatterView extends ViewBase {
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
  field: string;
  bins?: number;
  brush?: "x" | "none";
}

export interface BarsView extends ViewBase {
  type: "bars";
  field: string;
  /** "select" toggles categories in the filter by clicking. */
  brush?: "select" | "none";
}

export interface DensityView extends ViewBase {
  type: "density";
  x: string;
  y: string;
  bins?: [number, number];
}

export interface Scatter3DView extends ViewBase {
  type: "scatter3d";
  x: string;
  y: string;
  z: string;
  color?: string;
  pointSize?: number;
}

export type View = ScatterView | HistogramView | BarsView | DensityView | Scatter3DView;
export type ViewType = View["type"];

export interface FigureOutput {
  format: "svg" | "png" | "pdf";
  /** Physical width; publication figures are sized for the page, not the screen. */
  width: number;
  unit?: "mm" | "in" | "px";
  dpi?: number;
}

export interface Spec {
  $schema?: string;
  version?: typeof SPEC_VERSION;
  title?: string;
  description?: string;
  data: DataSource;
  fields?: Record<string, FieldDef>;
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
  bars: { required: ["field"], optional: [] },
  density: { required: ["x", "y"], optional: [] },
  scatter3d: { required: ["x", "y", "z"], optional: ["color"] },
};

const NEEDS: Record<ViewType, Record<string, FieldType | undefined>> = {
  scatter: { x: "quantitative", y: "quantitative", color: undefined },
  histogram: { field: "quantitative" },
  bars: { field: "nominal" },
  density: { x: "quantitative", y: "quantitative" },
  scatter3d: { x: "quantitative", y: "quantitative", z: "quantitative", color: undefined },
};

const BRUSHES: Record<ViewType, string[]> = {
  scatter: ["xy", "none"],
  histogram: ["x", "none"],
  bars: ["select", "none"],
  density: [],
  scatter3d: [],
};

const COMMON_VIEW_KEYS = ["id", "type", "title", "height", "span"];
const EXTRA_VIEW_KEYS: Record<ViewType, string[]> = {
  scatter: ["brush", "pointSize"],
  histogram: ["brush", "bins"],
  bars: ["brush"],
  density: ["bins"],
  scatter3d: ["pointSize"],
};
const TOP_KEYS = ["$schema", "version", "title", "description", "data", "fields", "views", "layout", "output"];

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
  }
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

  const data = spec.data;
  if (!isObject(data)) bad("/data", "required: { url } | { values } | { columns: \"external\" }");
  else {
    const kinds = ["url", "values", "columns"].filter((k) => k in data);
    if (kinds.length !== 1) bad("/data", `give exactly one of url, values, columns (got ${kinds.join(", ") || "none"})`);
    else if ("url" in data && typeof data.url !== "string") bad("/data/url", "must be a string");
    else if ("values" in data && !(Array.isArray(data.values) && data.values.every(isObject)))
      bad("/data/values", "must be an array of row objects");
    else if ("columns" in data && data.columns !== "external") bad("/data/columns", 'must be "external"');
  }

  const declared: Record<string, FieldType> = { ...types };
  if (spec.fields !== undefined) {
    if (!isObject(spec.fields)) bad("/fields", "must be an object of field name to definition");
    else
      for (const [name, def] of Object.entries(spec.fields)) {
        if (!isObject(def)) { bad(`/fields/${name}`, "must be an object"); continue; }
        unknownKeys(def, ["type", "title"], `/fields/${name}`, bad);
        if (def.type !== undefined) {
          if (def.type !== "quantitative" && def.type !== "nominal")
            bad(`/fields/${name}/type`, 'must be "quantitative" or "nominal"');
          else declared[name] = def.type;
        }
      }
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
    else unknownKeys(spec.output, ["interactive", "figure"], "/output", bad);
  }
  const views = spec.views.filter((v: unknown): v is View => isObject(v) && (v.type as string) in VIEW_FIELDS);
  const dims = views.reduce((n: number, v: View) => n + dimensionsNeeded(v), 0);
  if (dims > MAX_DIMENSIONS)
    bad("/views", `these views need ${dims} crossfilter dimensions; the limit is ${MAX_DIMENSIONS} (scatter with brush 2, density 2, histogram 1, bars 1, scatter3d 0)`);

  const figure = isObject(spec.output) ? spec.output.figure : undefined;
  if (figure !== undefined) {
    if (!isObject(figure)) bad("/output/figure", "must be an object");
    else {
      unknownKeys(figure, ["format", "width", "unit", "dpi"], "/output/figure", bad);
      if (!["svg", "png", "pdf"].includes(figure.format as string)) bad("/output/figure/format", "one of svg, png, pdf");
      if (!(typeof figure.width === "number" && figure.width > 0)) bad("/output/figure/width", "required: a positive number");
    }
  }
  return problems;
}

export class SpecError extends Error {
  // A plain field, not a parameter property, so Node can run this file by stripping types.
  readonly problems: Problem[];
  constructor(problems: Problem[]) {
    super("invalid spec:\n" + problems.map((p) => `  ${p.path || "/"}: ${p.message}`).join("\n"));
    this.problems = problems;
  }
}
