import { defineConfig } from "vite";

export default defineConfig({
  server: { fs: { allow: ["../.."] } },
  worker: { format: "es" },
  optimizeDeps: { exclude: ["chigutiro-wasm"] },
  test: { include: ["test/**/*.test.ts"] },
} as Parameters<typeof defineConfig>[0]);
