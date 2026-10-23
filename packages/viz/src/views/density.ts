import { axisBottom, axisLeft, interpolateInferno, rgb, scaleLinear } from "d3";
import type { EngineState } from "../engine/client";
import type { DensityView } from "../spec";
import { axisLabels, frame, type Panel, type ViewContext } from "./common";

const LUT = Array.from({ length: 256 }, (_, i) => {
  const c = rgb(interpolateInferno(i / 255));
  return [c.r, c.g, c.b] as const;
});

/**
 * A binned density of two fields over the rows passing every filter. It stays exact
 * at any row count, where a scatter of millions of points saturates.
 */
export function density(ctx: ViewContext<DensityView>): Panel {
  const { view, engine, alloc } = ctx;
  const f = frame(ctx, view.height ?? 300);
  const group = alloc.group!;
  const [xMin, xW, xBins, yMin, yW, yBins] = engine.state.binnings[group] as number[] as [number, number, number, number, number, number];
  const x = scaleLinear().domain([xMin, xMin + xW * xBins]).range([0, f.w]);
  const y = scaleLinear().domain([yMin, yMin + yW * yBins]).range([f.h, 0]);
  f.g.append("g").attr("transform", `translate(0,${f.h})`).call(axisBottom(x).ticks(Math.max(2, f.w / 80)));
  f.g.append("g").call(axisLeft(y).ticks(Math.max(2, f.h / 50)));
  axisLabels(f, ctx.titleOf(view.x), ctx.titleOf(view.y));

  const canvas = document.createElement("canvas");
  canvas.width = xBins;
  canvas.height = yBins;
  Object.assign(canvas.style, { width: `${f.w}px`, height: `${f.h}px`, imageRendering: "pixelated" });
  f.plot.append(canvas);
  const c2d = canvas.getContext("2d")!;
  const image = c2d.createImageData(xBins, yBins);

  const draw = (state: EngineState) => {
    const counts = state.counts[group]!;
    let top = 0;
    for (const c of counts) if (c > top) top = c;
    const scale = top > 0 ? 255 / Math.log1p(top) : 0;
    for (let by = 0; by < yBins; by++) {
      const dst = (yBins - 1 - by) * xBins; // canvas rows run downwards
      for (let bx = 0; bx < xBins; bx++) {
        const c = counts[by * xBins + bx]!;
        const o = (dst + bx) * 4;
        if (c === 0) { image.data[o + 3] = 0; continue; }
        const [r, g, b] = LUT[Math.min(255, Math.round(Math.log1p(c) * scale))]!;
        image.data[o] = r; image.data[o + 1] = g; image.data[o + 2] = b; image.data[o + 3] = 255;
      }
    }
    c2d.putImageData(image, 0, 0);
  };
  draw(engine.state);
  return { el: f.root, update: draw, destroy() {} };
}
