/// <reference lib="webworker" />
import init from "chigutiro-wasm";
import { serve, type Port } from "./backend";

// The WASM file is fetched from beside the bundle.
serve(self as unknown as Port, init());
