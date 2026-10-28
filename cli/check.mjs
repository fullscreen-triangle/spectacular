// Checks a spec's structure, then, when data.url names a local file, loads that file and
// checks every field the views read exists with the type they need.
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { validate } from "../packages/viz/src/spec.ts";
import { fieldTypes, loadTable } from "../packages/viz/src/data.ts";

/** The local file a spec's data.url names, or undefined for inline data and remote URLs. */
export function localData(spec, baseDir) {
  const url = spec?.data?.url;
  if (typeof url !== "string" || /^[a-z][a-z0-9+.-]*:/i.test(url)) return undefined;
  return resolve(baseDir, decodeURIComponent(url.replace(/[?#].*$/, "")));
}

/**
 * Returns { problems: [{ path, message }], rows?, fields? }; rows and fields are present
 * when the data was loaded. `baseDir` is where a relative data.url resolves.
 */
export async function check(spec, baseDir) {
  let problems = validate(spec);
  // Check fields against the data even when the structure has problems, so one run reports
  // everything; skip it only when the data itself is what is wrong.
  if (problems.some((p) => p.path.startsWith("/data") || p.path === "/views" || p.path === "")) return { problems };
  let table;
  try {
    const file = localData(spec, baseDir);
    if (file && !existsSync(file)) return { problems: [...problems, { path: "/data/url", message: `no file at ${file}` }] };
    if (file) table = await loadTable(spec, undefined, async () => new Response(readFileSync(file)));
    else if (Array.isArray(spec.data.values)) table = await loadTable(spec);
  } catch (e) {
    return { problems: [...problems, { path: "/data", message: e.message }] };
  }
  if (!table) return { problems };
  return { problems: validate(spec, fieldTypes(table)), rows: table.rows, fields: fieldTypes(table) };
}
