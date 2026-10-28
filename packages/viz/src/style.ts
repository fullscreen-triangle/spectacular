/** Injected once per page. Colours follow `color-scheme`, so the host page decides. */
export const STYLE = /* css */ `
.sv-root {
  --sv-fg: light-dark(#1d1f24, #e6e7ea);
  --sv-muted: light-dark(#6b7080, #9aa0ae);
  --sv-panel: light-dark(#ffffff, #17191e);
  --sv-border: light-dark(#e3e5ea, #2a2d35);
  --sv-context: light-dark(#dfe2e8, #2f333c);
  --sv-focus: light-dark(#4e79a7, #6f9bd1);
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
`;
