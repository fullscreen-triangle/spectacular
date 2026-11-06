/** Injected once per page. Colours follow `color-scheme`, so the host page decides. */
export const STYLE = /* css */ `
.sv-root {
  --sv-fg: light-dark(#1d1f24, #e6e7ea);
  --sv-muted: light-dark(#6b7080, #9aa0ae);
  --sv-panel: light-dark(#ffffff, #17191e);
  --sv-border: light-dark(#e3e5ea, #2a2d35);
  --sv-context: light-dark(#dfe2e8, #2f333c);
  --sv-focus: light-dark(#4e79a7, #6f9bd1);
  --sv-accent: light-dark(#e15759, #f28e8c);
  color: var(--sv-fg);
  font: 12px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;
}
.sv-heading { font-size: 15px; font-weight: 600; margin: 0 0 4px; }
.sv-status { color: var(--sv-muted); margin: 0 0 10px; font-variant-numeric: tabular-nums; }
.sv-grid { display: grid; gap: 12px; }
.sv-panel { background: var(--sv-panel); border: 1px solid var(--sv-border); border-radius: 8px; padding: 8px 0 4px; min-width: 0; }
.sv-title { font-weight: 600; padding: 0 12px 4px; }
.sv-legend { color: var(--sv-muted); padding: 2px 12px; }
.sv-stage { position: relative; }
.sv-plot { position: absolute; }
.sv-plot canvas { display: block; }
.sv-overlay { position: absolute; left: 0; top: 0; overflow: visible; }
.sv-overlay .domain, .sv-overlay .tick line { stroke: var(--sv-muted); }
.sv-overlay .tick text { fill: var(--sv-muted); }
.sv-axis-label { fill: var(--sv-fg); font-size: 11px; }
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
`;
