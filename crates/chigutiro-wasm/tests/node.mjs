// node crates/chigutiro-wasm/tests/node.mjs [rows-for-timing]
// Checks the WASM build against a plain-JS brute force, then times brush moves.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { initSync, Crossfilter, wasmMemory } from "../../../packages/chigutiro-wasm/chigutiro.js";

const pkg = new URL("../../../packages/chigutiro-wasm/", import.meta.url);
initSync({ module: readFileSync(fileURLToPath(new URL("chigutiro_bg.wasm", pkg))) });

let s = 0x9e3779b97f4a7c15n;
const unit = () => {
  s ^= s >> 12n; s ^= (s << 25n) & 0xffffffffffffffffn; s ^= s >> 27n;
  return Number(((s * 0x2545f4914f6cdd1dn) & 0xffffffffffffffffn) >> 11n) / 2 ** 53;
};
const gauss = () => Math.sqrt(-2 * Math.log(Math.max(unit(), 1e-300))) * Math.cos(2 * Math.PI * unit());

function table(n) {
  const x = new Float64Array(n), y = new Float64Array(n), k = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    x[i] = i % 50 === 0 ? NaN : gauss();
    y[i] = 0.6 * (x[i] || 0) + 0.8 * gauss();
    k[i] = Math.floor(unit() * 12);
  }
  return { x, y, k };
}

const selected = (cf) => new Uint8Array(wasmMemory().buffer, cf.selectedPtr(), cf.rows());

// --- correctness ---------------------------------------------------------------
{
  const n = 200_000, { x, y, k } = table(n);
  const cf = new Crossfilter(n);
  const dx = cf.addNumeric(x), dy = cf.addNumeric(y), dk = cf.addCategorical(k, 12);
  const hx = cf.addHistogram(dx, 40), ck = cf.addCategoryCounts(dk);
  const filters = [null, null, null];
  const passes = (d, i) => {
    const f = filters[d];
    if (!f) return true;
    if (d === dk) return f.includes(k[i]);
    const v = d === dx ? x[i] : y[i];
    return f[0] <= v && v <= f[1];
  };
  const [min, width, bins] = cf.groupBinning(hx);
  for (let step = 0; step < 40; step++) {
    const d = step % 3;
    const before = selected(cf).slice();
    if (step % 7 === 6) { filters[d] = null; cf.clearFilter(d); }
    else if (d === dk) { filters[d] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11].filter(() => unit() < 0.5); cf.filterCodes(d, Uint32Array.from(filters[d])); }
    else { const lo = gauss(), hi = lo + unit() * 2; filters[d] = [lo, hi]; cf.filterRange(d, lo, hi); }

    const sel = selected(cf), wantH = new Uint32Array(bins), wantK = new Uint32Array(12);
    let count = 0;
    for (let i = 0; i < n; i++) {
      const p = [passes(0, i), passes(1, i), passes(2, i)];
      const all = p[0] && p[1] && p[2];
      if (sel[i] !== (all ? 1 : 0)) throw new Error(`step ${step}: row ${i} selected=${sel[i]} want ${all}`);
      count += all;
      if (p[1] && p[2] && !Number.isNaN(x[i]) && x[i] >= min && x[i] <= min + width * bins) wantH[Math.min(bins - 1, Math.floor((x[i] - min) / width))]++;
      if (p[0] && p[1]) wantK[k[i]]++;
    }
    if (cf.selectedCount() !== count) throw new Error(`step ${step}: count`);
    if (String(cf.groupCounts(hx)) !== String(wantH)) throw new Error(`step ${step}: histogram`);
    if (String(cf.groupCounts(ck)) !== String(wantK)) throw new Error(`step ${step}: category counts`);
    const changed = new Uint32Array(wasmMemory().buffer, cf.changedPtr(), cf.changedLen());
    const flipped = before.reduce((acc, b, i) => acc + (b !== sel[i]), 0);
    if (changed.length !== flipped) throw new Error(`step ${step}: changed ${changed.length} vs ${flipped}`);
  }
  console.log(`correctness: 40 filter changes on ${n} rows match the JS brute force`);
}

// --- timing --------------------------------------------------------------------
{
  const n = Number(process.argv[2] ?? 10_000_000), { x, y, k } = table(n);
  const z = new Float64Array(n);
  for (let i = 0; i < n; i++) z[i] = (x[i] || 0) * y[i] + 0.3 * gauss();
  let t = performance.now();
  const cf = new Crossfilter(n);
  const dx = cf.addNumeric(x), dy = cf.addNumeric(y), dz = cf.addNumeric(z), dk = cf.addCategorical(k, 12);
  const build = performance.now() - t;
  t = performance.now();
  cf.addHistogram(dx, 60); cf.addHistogram(dz, 60); cf.addCategoryCounts(dk); cf.addGrid(dx, dy, 256, 256);
  const groups = performance.now() - t;
  console.log(`rows ${n}: build dimensions ${build.toFixed(0)} ms, groups ${groups.toFixed(0)} ms, wasm memory ${(wasmMemory().buffer.byteLength / 2 ** 20).toFixed(0)} MiB`);

  const time = (label, steps, fn) => {
    const ts = [];
    for (let i = 0; i < steps; i++) { const t0 = performance.now(); fn(i); ts.push(performance.now() - t0); }
    ts.sort((a, b) => a - b);
    console.log(`${label.padEnd(34)} median ${ts[ts.length >> 1].toFixed(2)} ms  p95 ${ts[Math.floor(ts.length * 0.95)].toFixed(2)} ms  max ${ts[ts.length - 1].toFixed(2)} ms`);
  };
  time("drag x-brush (200 steps)", 200, (i) => cf.filterRange(dx, -1 + i * 0.005, i * 0.005));
  cf.filterRange(dy, -0.5, 1.5);
  cf.filterCodes(dk, Uint32Array.from([0, 2, 4, 6, 8, 10]));
  time("drag z-brush, 3 filters on", 200, (i) => cf.filterRange(dz, -1 + i * 0.005, i * 0.005));
  time("clear a narrow filter (worst case)", 10, () => { cf.clearFilter(dz); cf.filterRange(dz, -0.2, 0.2); });
}
