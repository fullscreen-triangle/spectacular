import init from "chigutiro-wasm";
import { serve, type Port } from "./backend";
import type { Request, Response } from "./protocol";
import { wasmBytes } from "./wasm-inline";

let ready: Promise<unknown> | undefined;

/**
 * A stand-in for the crossfilter worker that runs on the page's own thread, for hosts
 * that refuse workers (some notebook front ends do). It is slower to brush at millions
 * of rows, because filtering then shares the thread with drawing, but it is the same
 * engine with the same answers.
 */
export function localWorker(): Worker {
  ready ??= init({ module_or_path: wasmBytes() });
  const page = { onmessage: null, onerror: null } as unknown as Worker;
  let closed = false;
  const backend: Port = {
    onmessage: null,
    postMessage: (msg: Response) => {
      if (!closed) setTimeout(() => page.onmessage?.(new MessageEvent("message", { data: msg })));
    },
  };
  serve(backend, ready);
  page.postMessage = (msg: Request) => {
    if (!closed) setTimeout(() => backend.onmessage?.(new MessageEvent("message", { data: msg })));
  };
  page.terminate = () => { closed = true; };
  return page;
}
