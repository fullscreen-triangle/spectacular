import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { EngineState } from "../engine/client";
import type { Scatter3DView } from "../spec";
import { COLOR_GLSL, PALETTE, colorChannel, hexToRgb, numeric, pointAlpha, selectedScale, type Panel, type ViewContext } from "./common";

const VERT = /* glsl */ `
in float a_sel;
in float a_c;
uniform float u_size;
uniform float u_alpha;
uniform float u_pass;
uniform float u_dimAlpha;
uniform float u_selScale;
out vec4 v_color;
${COLOR_GLSL}
void main() {
  bool sel = a_sel > 0.5;
  if ((u_pass > 0.5) != sel) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = u_size * (sel ? u_selScale : 0.7) * (3.0 / -mv.z);
  v_color = sel ? vec4(pointColor(a_c), u_alpha) : vec4(0.55, 0.55, 0.6, u_dimAlpha);
}`;

const FRAG = /* glsl */ `
in vec4 v_color;
out vec4 color;
void main() {
  vec2 d = gl_PointCoord - 0.5;
  if (dot(d, d) > 0.25) discard;
  color = vec4(v_color.rgb * v_color.a, v_color.a);
}`;

/** Points in a unit cube, orbitable. It shows the other views' filters; it does not brush. */
export function scatter3d(ctx: ViewContext<Scatter3DView>): Panel {
  const { view, table, engine } = ctx;
  const height = view.height ?? 360;
  const root = document.createElement("div");
  root.className = "sv-panel";
  root.dataset.view = view.id;
  const title = document.createElement("div");
  title.className = "sv-title";
  title.textContent = view.title ?? view.id;
  const legend = document.createElement("div");
  legend.className = "sv-legend";
  legend.textContent = `x: ${ctx.titleOf(view.x)} · y: ${ctx.titleOf(view.y)} · z: ${ctx.titleOf(view.z)} · drag to orbit, scroll to zoom`;
  root.append(title);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(ctx.width, height);
  root.append(renderer.domElement, legend);

  const n = table.rows;
  const cols = [numeric(table, view.x), numeric(table, view.y), numeric(table, view.z)];
  const pos = new Float32Array(n * 3);
  cols.forEach((c, k) => {
    const [lo, hi] = c.extent;
    const s = hi > lo ? 2 / (hi - lo) : 1;
    for (let i = 0; i < n; i++) pos[i * 3 + k] = (c.values[i]! - lo) * s - 1;
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  const color = colorChannel(table, view.color);
  geometry.setAttribute("a_c", new THREE.BufferAttribute(color.data, 1));
  const sel = new THREE.BufferAttribute(engine.state.mask.slice(), 1, false); // 0/1, not normalised
  sel.setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute("a_sel", sel);

  const uniforms = {
    u_size: { value: view.pointSize ?? (n > 200_000 ? 2 : 4) },
    u_alpha: { value: pointAlpha(engine.state.selectedCount) },
    u_pass: { value: 0 },
    u_dimAlpha: { value: Math.min(0.1, pointAlpha(n)) },
    u_selScale: { value: selectedScale(engine.state.selectedCount, n) },
    u_colorMode: { value: color.mode },
    u_palette: { value: PALETTE.map((h) => new THREE.Vector3(...hexToRgb(h))) },
    u_base: { value: new THREE.Vector3(...hexToRgb(PALETTE[0]!)) },
  };
  const material = (pass: number) =>
    new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: VERT,
      fragmentShader: FRAG,
      uniforms: { ...uniforms, u_pass: { value: pass } },
      transparent: true,
      depthWrite: false,
      blending: THREE.CustomBlending,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneMinusSrcAlphaFactor,
    });
  const scene = new THREE.Scene();
  const dim = new THREE.Points(geometry, material(0));
  const lit = new THREE.Points(geometry, material(1));
  lit.renderOrder = 1;
  scene.add(dim, lit);
  scene.add(new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(2, 2, 2)), new THREE.LineBasicMaterial({ color: 0x888888 })));
  scene.add(new THREE.AxesHelper(1.2).translateX(-1).translateY(-1).translateZ(-1));

  const camera = new THREE.PerspectiveCamera(45, ctx.width / height, 0.01, 100);
  camera.position.set(3.2, 2.4, 3.2);
  const controls = new OrbitControls(camera, renderer.domElement);
  const render = () => renderer.render(scene, camera);
  controls.addEventListener("change", render);
  render();

  return {
    el: root,
    update(state: EngineState) {
      (sel.array as Uint8Array).set(state.mask);
      sel.needsUpdate = true;
      for (const p of [dim, lit]) {
        const u = (p.material as THREE.ShaderMaterial).uniforms;
        u.u_alpha!.value = pointAlpha(state.selectedCount);
        u.u_selScale!.value = selectedScale(state.selectedCount, state.rows);
      }
      render();
    },
    destroy() {
      controls.dispose();
      geometry.dispose();
      renderer.dispose();
    },
  };
}
