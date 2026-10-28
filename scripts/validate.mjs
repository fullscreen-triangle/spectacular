#!/usr/bin/env node
// node scripts/validate.mjs <spec.json> [<spec.json> ...]
// Checks each spec's structure, then, when data.url names a local file beside the spec,
// loads that file and checks every field the views read exists with the type they need.
// Exit 0: all valid. 1: problems found (printed as "file /path: message"). 2: usage or I/O.
import { readFileSync, existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { validate } from "../packages/viz/src/spec.ts";
import { loadTable, fieldTypes } from "../packages/viz/src/data.ts";

const files = process.argv.slice(2);
if (!files.length) {
  console.error("usage: node scripts/validate.mjs <spec.json> [...]");
  process.exit(2);
}
let failed = false;
for (const file of files) {
  let spec;
  try {
    spec = JSON.parse(readFileSync(file, "utf8"));
  } catch (e) {
    console.error(`${file}: ${e.message}`);
    process.exit(2);
  }
  let problems = validate(spec);
  let summary = "structure only";
  // Check fields against the data even when the structure has problems, so one run reports
  // everything; skip it only when the data itself is what is wrong.
  const dataOk = !problems.some((p) => p.path.startsWith("/data") || p.path === "/views");
  try {
    let table;
    const url = spec?.data?.url;
    if (dataOk && typeof url === "string" && !/^[a-z]+:/i.test(url)) {
      const path = resolve(dirname(file), url);
      if (!existsSync(path)) problems.push({ path: "/data/url", message: `no file at ${path}` });
      else table = await loadTable(spec, undefined, async () => new Response(readFileSync(path)));
    } else if (dataOk && Array.isArray(spec?.data?.values)) table = await loadTable(spec);
    if (table) {
      problems = validate(spec, fieldTypes(table));
      summary = `${table.rows} rows; ` + Object.entries(fieldTypes(table)).map(([f, t]) => `${f}:${t}`).join(", ");
    }
  } catch {
    // A view too malformed to plan; the structural problems already say why.
  }
  if (problems.length) {
    failed = true;
    for (const p of problems) console.log(`${file} ${p.path || "/"}: ${p.message}`);
  } else console.log(`${file}: valid (${summary})`);
}
process.exit(failed ? 1 : 0);
