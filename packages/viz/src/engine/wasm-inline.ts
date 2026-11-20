import base64 from "virtual:chigutiro-wasm-base64";

/** The crossfilter's WASM module, embedded by the single-file build as base64. */
export function wasmBytes(): Uint8Array {
  const text = atob(base64);
  const bytes = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) bytes[i] = text.charCodeAt(i);
  return bytes;
}
