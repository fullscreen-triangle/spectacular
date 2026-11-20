import { brush, interpolateViridis, scaleLinear, type ScaleLinear } from "d3";
import type { EngineState } from "../engine/client";
import { SVG_NS } from "../export";
import type { ScatterView, View } from "../spec";
import {
  COLOR_GLSL, addKey, axisLabels, centred, colorChannel, colorKey, frame, colorsFor, hex, hexToRgb, numeric, pointAlpha, SHADER_COLORS, selectedScale, shaderPalette,
  xAxis, yAxis,
  type Frame, type Panel, type ViewContext,
} from "./common";

const VERT = /* glsl */ `#version 300 es
in float a_x;
in float a_y;
in float a_sel;
in float a_c;
uniform vec2 u_scale;
uniform vec2 u_offset;
uniform float u_pass;
uniform float u_size;
uniform float u_alpha;
uniform float u_dimAlpha;
uniform float u_selScale;
out vec4 v_color;
${COLOR_GLSL}
void main() {
  bool sel = a_sel > 0.5;
  if ((u_pass > 0.5) != sel) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
  gl_Position = vec4(a_x * u_scale.x + u_offset.x, a_y * u_scale.y + u_offset.y, 0.0, 1.0);
  gl_PointSize = sel ? u_size * u_selScale : max(1.0, u_size - 1.0);
  v_color = sel ? vec4(pointColor(a_c), u_alpha) : vec4(0.55, 0.55, 0.6, u_dimAlpha);
}`;

const FRAG = /* glsl */ `#version 300 es
precision mediump float;
in vec4 v_color;
out vec4 color;
void main() {
  vec2 d = gl_PointCoord - 0.5;
  if (dot(d, d) > 0.25) discard;
  color = vec4(v_color.rgb * v_color.a, v_color.a);
}`;

/** Up to this many rows, SVG export draws circles; above it, an image of the canvas. */
export const VECTOR_POINTS = 50_000;

export function program(gl: WebGL2RenderingContext, vert: string, frag: string): WebGLProgram {
  const compile = (type: number, src: string) => {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "shader");
    return s;
  };
  const p = gl.createProgram()!;
  gl.attachShader(p, compile(gl.VERTEX_SHADER, vert));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, frag));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? "link");
  return p;
}

export function scatter(ctx: ViewContext<ScatterView>): Panel {
  const { view, table, engine, alloc } = ctx;
  const f = frame(ctx, view.height ?? 320);
  const xs = numeric(table, view.x), ys = numeric(table, view.y);
  const x = scaleLinear().domain(xs.extent).range([0, f.w]).nice();
  const y = scaleLinear().domain(ys.extent).range([f.h, 0]).nice();
  xAxis(f, x);
  yAxis(f, y);
  axisLabels(f, ctx.titleOf(view.x), ctx.titleOf(view.y));
  const layer = pointLayer(ctx, f, xs.values, ys.values, x, y);

  if ((view.brush ?? "xy") === "xy") {
    const dx = alloc.dims[view.x]!, dy = alloc.dims[view.y]!;
    const b = brush().extent([[0, 0], [f.w, f.h]]).on("brush end", (event) => {
      const s = event.selection as [[number, number], [number, number]] | null;
      if (!s) {
        engine.filter({ dim: dx, clear: true });
        engine.filter({ dim: dy, clear: true });
        return;
      }
      engine.filter({ dim: dx, range: [x.invert(s[0][0]), x.invert(s[1][0])] });
      engine.filter({ dim: dy, range: [y.invert(s[1][1]), y.invert(s[0][1])] });
    });
    f.g.append("g").attr("class", "sv-brush").call(b);
  }
  return { el: f.root, update: layer.draw, vector: layer.vector, destroy: layer.destroy };
}

/** The views that draw rows as points, and the properties the point layer reads. */
type PointView = View & { color?: string; fill?: string; palette?: string[]; pointSize?: number };

/**
 * Rows as points in a frame, at `xv`/`yv` through the scales `x`/`y`, drawn by WebGL in
 * two passes (rows outside the selection in grey, then the selection in colour), with
 * the colour key and vector content for SVG export. The scatter and strip views share it.
 */
export function pointLayer(ctx: ViewContext<PointView>, f: Frame, xv: Float64Array, yv: Float64Array,
  x: ScaleLinear<number, number>, y: ScaleLinear<number, number>) {
  const { view, table, engine } = ctx;
  const palette = colorsFor(ctx, view.color, view);
  const base = view.fill ? hex(view.fill) : palette[0]!;
  addKey(f.root, colorKey(table, view.color, view.color ? ctx.titleOf(view.color) : "", palette));

  const canvas = document.createElement("canvas");
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(f.w * dpr);
  canvas.height = Math.round(f.h * dpr);
  canvas.style.width = `${f.w}px`;
  canvas.style.height = `${f.h}px`;
  f.plot.append(canvas);
  const gl = canvas.getContext("webgl2", { premultipliedAlpha: true, antialias: false, preserveDrawingBuffer: true });
  if (!gl) throw new Error("WebGL2 is not available");
  const prog = program(gl, VERT, FRAG);
  gl.useProgram(prog);

  const vao = gl.createVertexArray()!;
  gl.bindVertexArray(vao);
  const attr = (name: string, data: ArrayBufferView, type: number, normalized = false) => {
    const buf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, name);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 1, type, normalized, 0, 0);
    return buf;
  };
  const [x0, x1] = x.domain() as [number, number], [y0, y1] = y.domain() as [number, number];
  const px = centred(xv, [x0, x1]), py = centred(yv, [y0, y1]);
  attr("a_x", px.data, gl.FLOAT);
  attr("a_y", py.data, gl.FLOAT);
  const color = colorChannel(table, view.color, Math.min(palette.length, SHADER_COLORS));
  attr("a_c", color.data, gl.FLOAT);
  const selBuf = attr("a_sel", engine.state.mask, gl.UNSIGNED_BYTE); // 0/1, not normalised

  // Map centred data to clip space through the scale domains (a scale may run downwards).
  const sx = 2 / (x1 - x0), sy = 2 / (y1 - y0);
  const flip = y(y0) < y(y1) ? -1 : 1; // a band axis puts its first category at the top
  const u = (n: string) => gl.getUniformLocation(prog, n);
  gl.uniform2f(u("u_scale"), sx, sy * flip);
  gl.uniform2f(u("u_offset"), (px.centre - x0) * sx - 1, flip * ((py.centre - y0) * sy - 1));
  const size = view.pointSize ?? (table.rows > 200_000 ? 2 : 4); // diameter in CSS px
  gl.uniform1f(u("u_size"), size * dpr);
  gl.uniform1i(u("u_colorMode"), color.mode);
  gl.uniform3fv(u("u_palette"), shaderPalette(palette));
  gl.uniform3fv(u("u_base"), hexToRgb(base));
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

  const draw = (state: EngineState) => {
    gl.bindBuffer(gl.ARRAY_BUFFER, selBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, state.mask);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.uniform1f(u("u_alpha"), pointAlpha(state.selectedCount));
    gl.uniform1f(u("u_dimAlpha"), Math.min(0.15, pointAlpha(state.rows)));
    gl.uniform1f(u("u_selScale"), selectedScale(state.selectedCount, state.rows));
    for (const pass of [0, 1]) {
      gl.uniform1f(u("u_pass"), pass);
      gl.drawArrays(gl.POINTS, 0, table.rows);
    }
  };
  draw(engine.state);

  /** The same two passes as the shader: unselected rows in grey, then selected ones. */
  const vector = () => {
    if (table.rows > VECTOR_POINTS) return undefined;
    const s = engine.state;
    const content = document.createElementNS(SVG_NS, "g");
    const fillOf = (i: number) =>
      color.mode === 1 ? palette[color.data[i]!]! : color.mode === 2 ? interpolateViridis(color.data[i]!) : base;
    const passes = [
      { sel: 0, r: Math.max(1, size * dpr - 1) / dpr / 2, alpha: Math.min(0.15, pointAlpha(s.rows)), fill: () => "rgb(140,140,153)" },
      { sel: 1, r: (size * selectedScale(s.selectedCount, s.rows)) / 2, alpha: pointAlpha(s.selectedCount), fill: fillOf },
    ];
    for (const pass of passes) {
      const groups = new Map<string, SVGGElement>();
      for (let i = 0; i < table.rows; i++) {
        const vx = xv[i]!, vy = yv[i]!;
        if (s.mask[i] !== pass.sel || Number.isNaN(vx) || Number.isNaN(vy)) continue;
        const fill = pass.fill(i);
        let g = groups.get(fill);
        if (!g) {
          g = document.createElementNS(SVG_NS, "g");
          g.setAttribute("fill", fill);
          g.setAttribute("fill-opacity", pass.alpha.toFixed(3));
          groups.set(fill, g);
          content.append(g);
        }
        const c = document.createElementNS(SVG_NS, "circle");
        c.setAttribute("cx", x(vx).toFixed(2));
        c.setAttribute("cy", y(vy).toFixed(2));
        c.setAttribute("r", pass.r.toFixed(2));
        g.append(c);
      }
    }
    return { plot: f.plot, content };
  };

  return { draw, vector, destroy: () => void gl.getExtension("WEBGL_lose_context")?.loseContext() };
}
