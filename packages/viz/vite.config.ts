import { defineConfig } from "vite";

export default defineConfig({
  // Specs and their data in examples/ are served at the root: /?spec=/lorenz.spec.json
  publicDir: "../../examples",
  server: { fs: { allow: ["../.."] } },
  worker: { format: "es" },
  optimizeDeps: { exclude: ["chigutiro-wasm"] },
  test: { include: ["test/**/*.test.ts"] },
} as Parameters<typeof defineConfig>[0]);
