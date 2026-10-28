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
const note = document.getElementById("note")!;
const app = document.getElementById("app")!;
const expose = (h: unknown) => ((window as unknown as { spectacular: unknown }).spectacular = h);
/** For automation: body[data-state] goes from "loading" to "ready" or "error". */
const settle = (state: "ready" | "error", message = "") => {
  document.body.dataset.state = state;
  if (state === "error") {
    document.getElementById("error")!.textContent = message;
    note.textContent = "";
  }
};
document.body.dataset.state = "loading";

const specUrl = params.get("spec");
if (specUrl) {
  // Viewer mode: ?spec=<url of a spec JSON>. A relative data.url resolves against the spec's URL.
  const base = new URL(specUrl, location.href);
  note.textContent = `Loading ${base.pathname}…`;
  fetch(base)
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`fetching ${base}: ${r.status} ${r.statusText}`))))
    .then((s: Spec) => {
      if (s?.data && "url" in s.data && typeof s.data.url === "string") s = { ...s, data: { url: new URL(s.data.url, base).href } };
      const t0 = performance.now();
      return mount(app, s).then((h) => {
        note.textContent = `${base.pathname} · ${h.table.rows.toLocaleString()} rows · loaded and indexed in ${(performance.now() - t0).toFixed(0)} ms`;
        expose(h);
        settle("ready");
      });
    })
    .catch((e: Error) => settle("error", e.message));
} else {
  const rows = Math.min(Number(params.get("rows") ?? 1_000_000), 20_000_000);
  note.textContent = `Generating ${rows.toLocaleString()} rows… (set ?rows= to change)`;
  const t0 = performance.now();
  const columns = generate(rows);
  const t1 = performance.now();
  mount(app, spec as Spec, { columns })
    .then((h) => {
      const t2 = performance.now();
      note.textContent = `${rows.toLocaleString()} rows · generated in ${(t1 - t0).toFixed(0)} ms · loaded and indexed in ${(t2 - t1).toFixed(0)} ms · brush the scatter or histograms, click species bars`;
      expose(h);
      settle("ready");
    })
    .catch((e: Error) => settle("error", e.message));
}
