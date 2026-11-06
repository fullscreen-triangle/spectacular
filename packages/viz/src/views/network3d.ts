import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { EngineState } from "../engine/client";
import type { Network3DView } from "../spec";
import { EdgeSelection, PALETTE, hexToRgb, incident, nodeTotals, tooltip, type Pairs, type Panel, type ViewContext } from "./common";

const FOCUS = new THREE.Color(...hexToRgb(PALETTE[0]!));
const ACCENT = new THREE.Color(...hexToRgb(PALETTE[2]!));
const QUIET = new THREE.Color(0.6, 0.62, 0.68);

/**
 * Fruchterman–Reingold in three dimensions, from a fixed start (a golden spiral on the
 * sphere), so the same data always gives the same layout. Edges pull by the log of the
 * rows they hold; the result is scaled into the unit sphere.
 */
export function layout3d(p: Pairs, weights: ArrayLike<number>, iterations = 300): Float32Array {
  const n = p.nodes.length;
  const pos = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    const y = 1 - (2 * (i + 0.5)) / Math.max(1, n);
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const phi = i * 2.399963229728653;
    pos[i * 3] = Math.cos(phi) * r;
    pos[i * 3 + 1] = y;
    pos[i * 3 + 2] = Math.sin(phi) * r;
  }
  const und = new Map<number, number>();
  p.pairs.forEach(([a, b], k) => {
    if (a === b) return;
    const key = Math.min(a, b) * 1_000_003 + Math.max(a, b);
    und.set(key, (und.get(key) ?? 0) + (weights[k] ?? 0));
  });
  const edges = [...und].map(([key, w]) => [Math.floor(key / 1_000_003), key % 1_000_003, 1 + Math.log1p(w)] as const);
  const kk = 2 / Math.cbrt(Math.max(1, n));
  const disp = new Float64Array(n * 3);
  for (let it = 0; it < iterations; it++) {
    const temp = 0.12 * (1 - it / iterations) + 0.004;
    disp.fill(0);
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++) {
        const dx = pos[i * 3]! - pos[j * 3]!, dy = pos[i * 3 + 1]! - pos[j * 3 + 1]!, dz = pos[i * 3 + 2]! - pos[j * 3 + 2]!;
        const d2 = dx * dx + dy * dy + dz * dz + 1e-6;
        const f = (kk * kk) / d2;
        disp[i * 3]! += dx * f; disp[i * 3 + 1]! += dy * f; disp[i * 3 + 2]! += dz * f;
        disp[j * 3]! -= dx * f; disp[j * 3 + 1]! -= dy * f; disp[j * 3 + 2]! -= dz * f;
      }
    for (const [a, b, w] of edges) {
      const dx = pos[a * 3]! - pos[b * 3]!, dy = pos[a * 3 + 1]! - pos[b * 3 + 1]!, dz = pos[a * 3 + 2]! - pos[b * 3 + 2]!;
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-9;
      const f = (d * w) / kk;
      disp[a * 3]! -= dx * f; disp[a * 3 + 1]! -= dy * f; disp[a * 3 + 2]! -= dz * f;
      disp[b * 3]! += dx * f; disp[b * 3 + 1]! += dy * f; disp[b * 3 + 2]! += dz * f;
    }
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < 3; c++) disp[i * 3 + c]! -= pos[i * 3 + c]! * 0.6; // gravity keeps islands in view
      const dx = disp[i * 3]!, dy = disp[i * 3 + 1]!, dz = disp[i * 3 + 2]!;
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-9;
      const step = Math.min(len, temp) / len;
      pos[i * 3]! += dx * step; pos[i * 3 + 1]! += dy * step; pos[i * 3 + 2]! += dz * step;
    }
  }
  // A hub with many leaves packs the core tight while the leaves fly out. Compressing
  // the radius (r -> r^0.55) opens the core and keeps the leaves in view.
  let far = 1e-9;
  for (let i = 0; i < n; i++) far = Math.max(far, Math.hypot(pos[i * 3]!, pos[i * 3 + 1]!, pos[i * 3 + 2]!));
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const r = Math.hypot(pos[i * 3]!, pos[i * 3 + 1]!, pos[i * 3 + 2]!) / far;
    const k = r > 0 ? Math.pow(r, 0.55) / r / far : 0;
    for (let c = 0; c < 3; c++) out[i * 3 + c] = pos[i * 3 + c]! * k;
  }
  return out;
}

/**
 * The edge table as a network in 3D: nodes as spheres sized by the rows that touch them
 * after the other views' filters, edges as lines that fade when the filters empty them.
 * Drag to orbit, scroll to zoom; click a node to select every edge that touches it.
 */
export function network3d(ctx: ViewContext<Network3DView>): Panel {
  const { view, engine, alloc } = ctx;
  const p = alloc.pairs!;
  const group = alloc.group!;
  const height = view.height ?? 420;
  const root = document.createElement("div");
  root.className = "sv-panel";
  root.dataset.view = view.id;
  const title = document.createElement("div");
  title.className = "sv-title";
  title.textContent = view.title ?? view.id;
  const legend = document.createElement("div");
  legend.className = "sv-legend";
  legend.textContent = `${ctx.titleOf(view.source)} → ${ctx.titleOf(view.target)} · ${p.nodes.length} nodes, ${p.pairs.length} edges`;
  const hint = document.createElement("span");
  hint.className = "sv-hint";
  hint.textContent = " · drag to orbit, scroll to zoom, click a node to select its edges";
  legend.append(hint);
  const stage = document.createElement("div");
  stage.style.position = "relative";
  root.append(title, stage, legend);
  const tip = tooltip(stage);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setSize(ctx.width, height);
  stage.append(renderer.domElement);

  const base = engine.state.baseline[group]!;
  const pos = layout3d(p, base);
  const n = p.nodes.length;
  const scene = new THREE.Scene();
  const sphere = new THREE.SphereGeometry(1, 16, 12);
  const mesh = new THREE.InstancedMesh(sphere, new THREE.MeshBasicMaterial(), n);
  scene.add(mesh);
  const linePos = new Float32Array(p.pairs.length * 6);
  p.pairs.forEach(([a, b], k) => {
    linePos.set(pos.subarray(a * 3, a * 3 + 3), k * 6);
    linePos.set(pos.subarray(b * 3, b * 3 + 3), k * 6 + 3);
  });
  const lineGeo = new THREE.BufferGeometry();
  lineGeo.setAttribute("position", new THREE.BufferAttribute(linePos, 3));
  const lineCol = new THREE.BufferAttribute(new Float32Array(p.pairs.length * 6), 3);
  lineCol.setUsage(THREE.DynamicDrawUsage);
  lineGeo.setAttribute("color", lineCol);
  const lines = new THREE.LineSegments(lineGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.85 }));
  scene.add(lines);

  const camera = new THREE.PerspectiveCamera(45, ctx.width / height, 0.01, 100);
  camera.position.set(0, 0.4, 3.1);
  const controls = new OrbitControls(camera, renderer.domElement);

  // Labels for the busiest nodes; every node names itself on hover.
  const busiest = new Set(
    p.nodes.map((_, i) => i).sort((a, b) => {
      const t = nodeTotals(p, base);
      return t.out[b]! + t.into[b]! - (t.out[a]! + t.into[a]!);
    }).slice(0, 24),
  );
  const labels = new Map<number, HTMLDivElement>();
  for (const i of busiest) {
    const d = document.createElement("div");
    d.className = "sv-label3d";
    d.textContent = p.nodes[i]!;
    stage.append(d);
    labels.set(i, d);
  }
  const v = new THREE.Vector3();
  const placeLabels = () => {
    for (const [i, d] of labels) {
      v.set(pos[i * 3]!, pos[i * 3 + 1]!, pos[i * 3 + 2]!).project(camera);
      d.style.left = `${((v.x + 1) / 2) * ctx.width}px`;
      d.style.top = `${((1 - v.y) / 2) * height}px`;
      d.style.display = v.z < 1 ? "" : "none";
    }
  };
  const render = () => {
    renderer.render(scene, camera);
    placeLabels();
  };
  controls.addEventListener("change", render);

  const sel = new EdgeSelection(engine, view.brush === "none" ? undefined : alloc.dims[Object.keys(alloc.dims)[0]!], () => paint(engine.state));
  let totals = nodeTotals(p, base);
  const m = new THREE.Matrix4();
  const paint = (state: EngineState) => {
    const counts = state.counts[group]!;
    totals = nodeTotals(p, counts);
    const most = Math.max(1, ...p.nodes.map((_, i) => totals.out[i]! + totals.into[i]!));
    for (let i = 0; i < n; i++) {
      const t = totals.out[i]! + totals.into[i]!;
      const s = 0.018 + 0.06 * Math.sqrt(t / most);
      m.makeScale(s, s, s).setPosition(pos[i * 3]!, pos[i * 3 + 1]!, pos[i * 3 + 2]!);
      mesh.setMatrixAt(i, m);
      const chosen = incident(p, i).some((k) => sel.chosen.has(k));
      mesh.setColorAt(i, chosen ? ACCENT : t > 0 ? FOCUS : QUIET);
      labels.get(i)?.classList.toggle("sv-quiet", t === 0);
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    const top = Math.max(1, ...Array.from(counts));
    const arr = lineCol.array as Float32Array;
    p.pairs.forEach((_, k) => {
      const c = counts[k]!;
      const col = sel.chosen.has(k) ? ACCENT : c > 0 ? FOCUS : QUIET;
      const strength = c > 0 ? 0.35 + 0.65 * Math.sqrt(c / top) : 0.12;
      const shown = sel.shows(k) ? strength : strength * 0.3;
      for (let e = 0; e < 2; e++) {
        arr[k * 6 + e * 3] = col.r * shown + (1 - shown) * 0.5;
        arr[k * 6 + e * 3 + 1] = col.g * shown + (1 - shown) * 0.5;
        arr[k * 6 + e * 3 + 2] = col.b * shown + (1 - shown) * 0.5;
      }
    });
    lineCol.needsUpdate = true;
    render();
  };
  paint(engine.state);

  const ray = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  const pick = (e: PointerEvent | MouseEvent): number | undefined => {
    const r = renderer.domElement.getBoundingClientRect();
    ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    return ray.intersectObject(mesh)[0]?.instanceId;
  };
  let down: [number, number] | null = null;
  const el = renderer.domElement;
  el.addEventListener("pointerdown", (e) => { down = [e.clientX, e.clientY]; });
  el.addEventListener("pointerup", (e) => {
    if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 4) {
      const i = pick(e);
      if (i !== undefined) sel.toggle(incident(p, i));
    }
    down = null;
  });
  el.addEventListener("mousemove", (e) => {
    const i = pick(e);
    if (i === undefined) { tip.hide(); el.style.cursor = ""; return; }
    el.style.cursor = sel.active ? "pointer" : "";
    tip.show(e, `${p.nodes[i]}: ${totals.out[i]} out, ${totals.into[i]} in`);
  });
  el.addEventListener("mouseleave", () => tip.hide());

  return {
    el: root,
    update: paint,
    destroy() {
      controls.dispose();
      sphere.dispose();
      lineGeo.dispose();
      renderer.dispose();
    },
  };
}
