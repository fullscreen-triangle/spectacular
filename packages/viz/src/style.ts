/**
 * Injected once per page. Colours follow `color-scheme`: under theme "auto" the host page
 * decides, and "light", "dark" and "paper" fix it on the root.
 */
export const STYLE = /* css */ `
.sv-root {
  --sv-bg: light-dark(#f5f6f8, #0f1115);
  --sv-fg: light-dark(#1d1f24, #e6e7ea);
  --sv-muted: light-dark(#5f6472, #a0a6b4);
  --sv-axis: light-dark(#9096a3, #5c6270);
  --sv-grid: light-dark(#eef0f3, #1f2229);
  --sv-panel: light-dark(#ffffff, #171a20);
  --sv-border: light-dark(#e3e5ea, #2a2d35);
  --sv-context: light-dark(#dfe2e8, #2f333c);
  --sv-focus: light-dark(#4e79a7, #6f9bd1);
  --sv-accent: light-dark(#e15759, #f28e8c);
  box-sizing: border-box;
  padding: 14px;
  border-radius: 10px;
  background: var(--sv-bg);
  color: var(--sv-fg);
  font: 12px/1.4 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
.sv-root[data-theme="light"] { color-scheme: light; }
.sv-root[data-theme="dark"] { color-scheme: dark; }
.sv-root[data-theme="paper"] {
  color-scheme: light;
  --sv-bg: #fff; --sv-panel: #fff; --sv-border: transparent;
  --sv-fg: #111; --sv-muted: #333; --sv-axis: #333; --sv-grid: #eeeeee;
}
.sv-root *, .sv-root *::before, .sv-root *::after { box-sizing: border-box; }
.sv-figure .sv-status, .sv-figure .sv-hint { display: none; }
.sv-heading { font-size: 16px; font-weight: 600; letter-spacing: -0.01em; margin: 0 0 4px; }
.sv-status { color: var(--sv-muted); margin: 0 0 10px; font-variant-numeric: tabular-nums; }
.sv-grid { display: grid; gap: 12px; }
.sv-panel { background: var(--sv-panel); border: 1px solid var(--sv-border); border-radius: 8px; padding: 10px 0 6px; min-width: 0; }
.sv-title { font-weight: 600; padding: 0 12px 4px; }
.sv-legend { color: var(--sv-muted); padding: 2px 12px; }
.sv-key { display: flex; flex-wrap: wrap; align-items: center; gap: 3px 10px; padding: 0 12px 6px; color: var(--sv-muted); font-size: 11px; }
.sv-key-title { color: var(--sv-fg); font-weight: 500; }
.sv-key-item { display: inline-flex; align-items: center; gap: 4px; }
.sv-swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; }
.sv-ramp { display: block; }
.sv-stage { position: relative; }
.sv-plot { position: absolute; }
.sv-plot canvas { display: block; }
.sv-overlay, .sv-underlay { position: absolute; left: 0; top: 0; overflow: visible; }
.sv-underlay { pointer-events: none; }
.sv-overlay .domain, .sv-overlay .tick line { stroke: var(--sv-axis); }
.sv-overlay .tick text { fill: var(--sv-muted); font-family: inherit; font-size: 11px; font-variant-numeric: tabular-nums; }
.sv-gridlines .tick line { stroke: var(--sv-grid); }
.sv-axis-label { fill: var(--sv-fg); font-size: 11.5px; font-weight: 500; }
.sv-context rect, rect.sv-context { fill: var(--sv-context); }
.sv-focus rect { fill: var(--sv-focus); }
.sv-brush .selection { fill: var(--sv-fg); fill-opacity: 0.1; stroke: var(--sv-fg); stroke-dasharray: 4 3; }
.sv-edge { fill: none; stroke: var(--sv-focus); stroke-linecap: round; }
.sv-edge.sv-chosen { stroke: var(--sv-accent); }
.sv-edge-context { fill: none; stroke: var(--sv-context); }
.sv-node { fill: var(--sv-focus); stroke: var(--sv-panel); stroke-width: 1.5; }
.sv-node.sv-chosen { fill: var(--sv-accent); }
.sv-node-label { fill: var(--sv-fg); font-size: 10px; }
.sv-node-label.sv-quiet { fill: var(--sv-muted); }
.sv-flow { fill: none; stroke: var(--sv-focus); stroke-opacity: 0.45; }
.sv-flow.sv-chosen { stroke: var(--sv-accent); stroke-opacity: 0.75; }
.sv-flow:hover, .sv-edge:hover { stroke-opacity: 0.9; }
.sv-bar { fill: var(--sv-focus); }
.sv-bar.sv-chosen { fill: var(--sv-accent); }
.sv-hit { cursor: pointer; }
.sv-tip { position: absolute; pointer-events: none; z-index: 2; max-width: 240px; padding: 4px 7px; border-radius: 4px;
  background: var(--sv-fg); color: var(--sv-panel); font-variant-numeric: tabular-nums; }
.sv-label3d { position: absolute; pointer-events: none; font-size: 10px; color: var(--sv-fg); white-space: nowrap;
  transform: translate(6px, -50%); text-shadow: 0 0 3px var(--sv-panel); }
.sv-label3d.sv-quiet { color: var(--sv-muted); }
.sv-problems { padding: 10px 12px; border: 1px solid var(--sv-accent); border-radius: 8px; background: var(--sv-panel); }
.sv-problems b { display: block; margin-bottom: 4px; }
.sv-problems code { font: 11px ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--sv-accent); }
`;
