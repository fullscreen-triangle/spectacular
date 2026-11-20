// The single-file build for Python: `npm run build:python` writes
// python/src/spectacular/static/spectacular.js (the anywidget module and the runtime of
// standalone pages, with the worker and WASM inlined) and the spec's JSON Schema beside it.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";
import { jsonSchema } from "./src/spec.ts";

const VIRTUAL = "virtual:chigutiro-wasm-base64";

function inlineWasm(): Plugin {
  return {
    name: "spectacular-inline-wasm",
    resolveId: (id) => (id === VIRTUAL ? "\0" + VIRTUAL : null),
    load(id) {
      if (id !== "\0" + VIRTUAL) return null;
      const file = fileURLToPath(new URL("../chigutiro-wasm/chigutiro_bg.wasm", import.meta.url));
      return `export default ${JSON.stringify(readFileSync(file).toString("base64"))};`;
    },
  };
}

function schemaFile(): Plugin {
  return {
    name: "spectacular-schema",
    generateBundle(_, bundle) {
      this.emitFile({ type: "asset", fileName: "spectacular.schema.json", source: JSON.stringify(jsonSchema(), null, 2) + "\n" });
      // Engine.start's default worker is emitted as a file, but this build always passes
      // its inline worker, so that file would only be dead weight in the package.
      for (const name of Object.keys(bundle)) if (name.startsWith("assets/worker")) delete bundle[name];
    },
  };
}

export default defineConfig({
  publicDir: false,
  plugins: [inlineWasm(), schemaFile()],
  worker: { format: "es", plugins: () => [inlineWasm()] },
  build: {
    outDir: "../../python/src/spectacular/static",
    emptyOutDir: true,
    minify: true,
    lib: { entry: "src/notebook.ts", formats: ["es"], fileName: () => "spectacular.js" },
    // Library mode keeps whitespace in ES output unless the output step is told otherwise.
    rolldownOptions: { output: { minify: true } },
  },
});
