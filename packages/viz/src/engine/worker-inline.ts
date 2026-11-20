/// <reference lib="webworker" />
import init from "chigutiro-wasm";
import { serve, type Port } from "./backend";
import { wasmBytes } from "./wasm-inline";

// The single-file bundle (notebooks, standalone HTML) carries the WASM in itself.
serve(self as unknown as Port, init({ module_or_path: wasmBytes() }));
