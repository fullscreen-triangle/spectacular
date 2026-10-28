#!/usr/bin/env node
// The command-line interface other programs drive spectacular through. It runs the
// TypeScript sources directly, so it needs Node 22.18+ (type stripping).
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { spawn } from "node:child_process";
import { parseArgs } from "node:util";
import { jsonSchema } from "../packages/viz/src/spec.ts";
import { check } from "./check.mjs";

const USAGE = `usage: spectacular <command> [options]

  validate <spec.json|-> [...]   check each spec, and its fields against local data
  render <spec.json|-> -o <out.svg|.pdf|.png> [--width N] [--unit mm|in|px] [--dpi N]
                                 draw the spec to a file; flags override output.figure
  serve <spec.json> [--port N] [--open] [--indices] [--no-watch]
                                 open the spec interactively; report each selection,
                                 and reload pages when the spec or its data changes
  schema                         print the JSON Schema for specs

  --json      machine-readable output: one JSON object per line on stdout
  "-"         read the spec from stdin; a relative data.url resolves against the cwd

exit codes: 0 ok, 1 the spec or its data has problems, 2 usage or I/O error,
            3 rendering failed`;

const commands = { validate: validateCmd, render: renderCmd, serve: serveCmd, schema: schemaCmd };
const [command, ...rest] = process.argv.slice(2);
if (!(command in commands)) {
  console.error(command && command !== "--help" && command !== "-h" ? `unknown command "${command}"\n\n${USAGE}` : USAGE);
  process.exit(command === "--help" || command === "-h" ? 0 : 2);
}
let args;
try {
  args = parseArgs({
    args: rest,
    allowPositionals: true,
    options: {
      json: { type: "boolean" },
      out: { type: "string", short: "o" },
      width: { type: "string" },
      unit: { type: "string" },
      dpi: { type: "string" },
      port: { type: "string" },
      open: { type: "boolean" },
      indices: { type: "boolean" },
      "no-watch": { type: "boolean" },
    },
  });
} catch (e) {
  console.error(`${e.message}\n\n${USAGE}`);
  process.exit(2);
}
const json = !!args.values.json;
const emit = (obj) => process.stdout.write(JSON.stringify(obj) + "\n");
process.exitCode = await commands[command](args.positionals, args.values);

/** Reads a spec from a file or "-" (stdin). Returns { spec, specFile, specText } or exits 2. */
function readSpec(arg) {
  const fromStdin = arg === "-";
  const specFile = fromStdin ? resolve("stdin.spec.json") : resolve(arg);
  let specText;
  try {
    specText = readFileSync(fromStdin ? 0 : specFile, "utf8");
    return { spec: JSON.parse(specText), specFile, specText: fromStdin ? specText : undefined, label: fromStdin ? "-" : arg };
  } catch (e) {
    fail(`${arg}: ${e.message}`, 2);
  }
}

function fail(message, code, extra = {}) {
  if (json) emit({ ok: false, error: message, exit: code, ...extra });
  else console.error(message);
  process.exit(code);
}

function number(name, value) {
  if (value === undefined) return undefined;
  const n = Number(value);
  if (!(n > 0)) fail(`--${name} must be a positive number`, 2);
  return n;
}

async function validateCmd(files) {
  if (!files.length) fail(USAGE, 2);
  let worst = 0;
  for (const arg of files) {
    const { spec, specFile, label } = readSpec(arg);
    const result = await check(spec, resolve(specFile, ".."));
    const ok = result.problems.length === 0;
    if (!ok) worst = 1;
    if (json) emit({ spec: label, ok, ...result });
    else if (ok)
      console.log(`${label}: valid (${result.rows === undefined ? "structure only" : `${result.rows} rows; ` +
        Object.entries(result.fields).map(([f, t]) => `${f}:${t}`).join(", ")})`);
    else for (const p of result.problems) console.log(`${label} ${p.path || "/"}: ${p.message}`);
  }
  return worst;
}

async function renderCmd([arg, ...extra], opts) {
  if (!arg || extra.length || !opts.out) fail(USAGE, 2);
  const { spec, specFile, specText, label } = readSpec(arg);
  const { problems } = await check(spec, resolve(specFile, ".."));
  if (problems.length) {
    if (!json) for (const p of problems) console.error(`${label} ${p.path || "/"}: ${p.message}`);
    fail(`${label}: not rendered, the spec has ${problems.length} problem(s)`, 1, { problems });
  }
  const { render } = await import("./render.mjs");
  try {
    const result = await render({
      spec, specFile, specText, out: resolve(opts.out),
      width: number("width", opts.width), unit: opts.unit, dpi: number("dpi", opts.dpi),
    });
    if (json) emit({ ok: true, ...result });
    else console.log(`wrote ${opts.out} (${result.width}×${result.height} CSS px${result.pixels ? `, ${result.pixels.join("×")} pixels` : ""})`);
    return 0;
  } catch (e) {
    fail(`${label}: ${e.message}`, e.exit ?? 3);
  }
}

async function serveCmd([arg, ...extra], opts) {
  if (!arg || arg === "-" || extra.length) fail(`serve needs a spec file (not stdin)\n\n${USAGE}`, 2);
  const { spec, specFile, label } = readSpec(arg);
  const { problems } = await check(spec, resolve(specFile, ".."));
  if (problems.length) {
    // Still serve: the page shows the problems, and --watch reloads once they are fixed.
    if (json) emit({ event: "problems", problems });
    else for (const p of problems) console.error(`${label} ${p.path || "/"}: ${p.message}`);
  }
  const { startServer } = await import("./server.mjs");
  let server;
  try {
    server = await startServer({
      specFile,
      port: opts.port === undefined ? 0 : Number(opts.port),
      watch: !opts["no-watch"],
      onSelection: (s) => {
        if (json) emit({ event: "selection", ...s });
        else console.log(`${s.selected} of ${s.rows} rows selected` + s.filters.map((f) =>
          `; ${f.view}.${f.field} ${f.range ? `in [${f.range.map((v) => +v.toPrecision(6)).join(", ")}]` : `in {${f.values.join(", ")}}`}`).join(""));
      },
      onReload: (file) => (json ? emit({ event: "reload", file }) : console.log(`${file} changed; reloading`)),
    });
  } catch (e) {
    fail(e.message, e.exit ?? 2);
  }
  const url = `${server.origin}/?spec=${server.specPath}&channel${opts.indices ? "&indices" : ""}`;
  if (json) emit({ event: "listening", url });
  else console.log(`serving ${label} at ${url}\n(loopback only; ctrl-c to stop)`);
  if (opts.open) {
    const [cmd, ...a] = process.platform === "win32" ? ["rundll32", "url.dll,FileProtocolHandler", url]
      : process.platform === "darwin" ? ["open", url] : ["xdg-open", url];
    spawn(cmd, a, { stdio: "ignore", detached: true }).on("error", () => {}).unref();
  }
  const stop = () => server.close().then(() => process.exit(0));
  process.on("SIGINT", stop);
  process.on("SIGTERM", stop);
  await new Promise(() => {}); // until a signal
}

function schemaCmd() {
  process.stdout.write(JSON.stringify(jsonSchema(), null, 2) + "\n");
  return 0;
}
