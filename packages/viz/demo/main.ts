import { mount, type Spec } from "../src";
import spec from "../../../examples/synthetic.spec.json";

const SPECIES = ["electron", "muon", "pion"];

/** Correlated momenta per species, plus a resonance at 91 GeV over a falling background. */
function generate(n: number) {
  const px = new Float64Array(n), py = new Float64Array(n), pz = new Float64Array(n);
  const mass = new Float64Array(n), energy = new Float64Array(n);
  const species: string[] = new Array(n);
  let s = 0x2545f491;
  const unit = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32);
  const gauss = () => Math.sqrt(-2 * Math.log(unit() || 1e-12)) * Math.cos(2 * Math.PI * unit());
  for (let i = 0; i < n; i++) {
    const k = unit() < 0.5 ? 2 : unit() < 0.6 ? 0 : 1;
    species[i] = SPECIES[k]!;
    const spread = [6, 10, 4][k]!;
    const a = gauss() * spread, b = gauss() * spread;
    px[i] = a + 0.4 * b + (k === 1 ? 8 : 0);
    py[i] = 0.7 * a - 0.5 * b;
    pz[i] = gauss() * 20 + (k === 0 ? 10 : -5) + 0.3 * px[i]!;
    mass[i] = unit() < 0.15 ? 91.2 + gauss() * 2.5 : 40 + -Math.log(unit() || 1e-12) * 25;
    energy[i] = Math.hypot(px[i]!, py[i]!, pz[i]!) + Math.abs(gauss()) * 2;
  }
  return { px, py, pz, mass, energy, species };
}

const params = new URLSearchParams(location.search);
const rows = Math.min(Number(params.get("rows") ?? 1_000_000), 20_000_000);
const note = document.getElementById("note")!;
note.textContent = `Generating ${rows.toLocaleString()} rows… (set ?rows= to change)`;

const t0 = performance.now();
const columns = generate(rows);
const t1 = performance.now();
mount(document.getElementById("app")!, spec as Spec, { columns })
  .then((h) => {
    const t2 = performance.now();
    note.textContent = `${rows.toLocaleString()} rows · generated in ${(t1 - t0).toFixed(0)} ms · loaded and indexed in ${(t2 - t1).toFixed(0)} ms · brush the scatter or histograms, click species bars`;
    (window as unknown as { spectacular: unknown }).spectacular = h;
  })
  .catch((e: Error) => {
    document.getElementById("error")!.textContent = e.message;
    note.textContent = "";
  });
