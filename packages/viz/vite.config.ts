import { defineConfig } from "vite";

export default defineConfig({
  // In dev, specs and their data in examples/ are served at the root:
  // /?spec=/lorenz.spec.json. The build leaves them out, so private data dropped in
  // examples/local/ never lands in dist/; `spectacular serve` hands a spec its own files.
  publicDir: "../../examples",
  build: { copyPublicDir: false },
  server: { fs: { allow: ["../.."] } },
  worker: { format: "es" },
  optimizeDeps: { exclude: ["chigutiro-wasm"] },
  test: { include: ["test/**/*.test.ts"] },
} as Parameters<typeof defineConfig>[0]);
