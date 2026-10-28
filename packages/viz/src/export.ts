/**
 * Mounted views as one standalone SVG document. Panels become rects, HTML text becomes
 * <text>, the views' own SVG is copied with its computed styles written in (so it looks
 * the same outside this page, in Inkscape or Illustrator), and canvases become embedded
 * PNG images unless a view supplies vector content for its plot area.
 */

export const SVG_NS = "http://www.w3.org/2000/svg";

/** Presentation properties copied from computed style onto the exported SVG elements. */
const INHERITED = ["fill", "fill-opacity", "stroke", "stroke-width", "stroke-opacity", "stroke-dasharray",
  "font-family", "font-size", "font-weight", "font-style", "text-anchor"];
const OWN = ["opacity"];
/** d3-brush hit areas: invisible, and meaningless outside the page. */
const POINTER_ONLY = ".overlay, .handle";

const transparent = (c: string) => c === "transparent" || /^rgba\(.*,\s*0\)$/.test(c);

function el<K extends keyof SVGElementTagNameMap>(name: K, attrs: Record<string, string | number>): SVGElementTagNameMap[K] {
  const e = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  return e;
}

const r2 = (v: number) => Math.round(v * 100) / 100;

let measure: CanvasRenderingContext2D | null | undefined;
function ascent(style: CSSStyleDeclaration, text: string): number {
  measure ??= document.createElement("canvas").getContext("2d");
  const size = parseFloat(style.fontSize);
  if (!measure) return size * 0.8;
  measure.font = `${style.fontStyle} ${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
  return measure.measureText(text).fontBoundingBoxAscent || size * 0.8;
}

/** A copy of a live SVG element whose look no longer depends on this page's CSS. */
function inlineStyles(source: SVGSVGElement): SVGSVGElement {
  const copy = source.cloneNode(true) as SVGSVGElement;
  const originals = [source, ...source.querySelectorAll("*")];
  const copies = [copy, ...copy.querySelectorAll("*")];
  const drop: Element[] = [];
  originals.forEach((o, i) => {
    const c = copies[i]!;
    const cs = getComputedStyle(o);
    if (cs.display === "none" || o.matches(POINTER_ONLY)) return drop.push(c);
    const parent = o === source ? undefined : getComputedStyle(o.parentElement!);
    // Write a value where the element states its own (d3 uses currentColor, which means
    // something else outside the page) or where it differs from what it would inherit.
    for (const p of INHERITED) {
      const v = cs.getPropertyValue(p);
      if (o.hasAttribute(p) || !parent || parent.getPropertyValue(p) !== v) c.setAttribute(p, v);
    }
    for (const p of OWN) if (cs.getPropertyValue(p) !== "1") c.setAttribute(p, cs.getPropertyValue(p));
    c.removeAttribute("style");
  });
  for (const d of drop) d.remove();
  return copy;
}

/**
 * Serialises what `root` shows. `vectors` maps a view's plot element to SVG content (in
 * that element's coordinates) that replaces its canvases.
 */
export function toSVG(root: HTMLElement, vectors: Map<Element, SVGElement> = new Map()): string {
  const origin = root.getBoundingClientRect();
  const W = Math.ceil(origin.width), H = Math.ceil(origin.height);
  const rootStyle = getComputedStyle(root);
  const svg = el("svg", { xmlns: SVG_NS, width: W, height: H, viewBox: `0 0 ${W} ${H}`, color: rootStyle.color });

  let bg = "";
  for (let e: Element | null = root; e && !bg; e = e.parentElement) {
    const c = getComputedStyle(e).backgroundColor;
    if (!transparent(c)) bg = c;
  }
  svg.append(el("rect", { width: W, height: H, fill: bg || "#fff" }));

  const at = (r: DOMRect) => ({ x: r2(r.left - origin.left), y: r2(r.top - origin.top), width: r2(r.width), height: r2(r.height) });

  const walk = (node: Element, parent: SVGElement) => {
    const style = getComputedStyle(node);
    if (style.display === "none" || style.visibility === "hidden") return;
    if (node instanceof SVGSVGElement) {
      const copy = inlineStyles(node);
      const box = at(node.getBoundingClientRect());
      copy.setAttribute("x", String(box.x));
      copy.setAttribute("y", String(box.y));
      copy.setAttribute("overflow", "visible");
      parent.append(copy);
      return;
    }
    const box = at(node.getBoundingClientRect());
    const vector = vectors.get(node);
    if (vector) {
      const g = el("g", { transform: `translate(${box.x},${box.y})` });
      g.append(vector);
      parent.append(g);
      return;
    }
    if (node instanceof HTMLCanvasElement) {
      const img = el("image", { ...box, href: node.toDataURL("image/png"), preserveAspectRatio: "none" });
      if (style.imageRendering === "pixelated") {
        img.setAttribute("image-rendering", "optimizeSpeed");
        img.setAttribute("style", "image-rendering:pixelated");
      }
      parent.append(img);
      return;
    }
    const border = parseFloat(style.borderTopWidth) || 0;
    if (!transparent(style.backgroundColor) || border > 0) {
      parent.append(el("rect", {
        x: box.x + border / 2, y: box.y + border / 2, width: box.width - border, height: box.height - border,
        rx: parseFloat(style.borderTopLeftRadius) || 0,
        fill: transparent(style.backgroundColor) ? "none" : style.backgroundColor,
        stroke: border > 0 ? style.borderTopColor : "none",
        "stroke-width": border,
      }));
    }
    for (const child of node.childNodes) {
      if (child instanceof Element) walk(child, parent);
      else if (child.nodeType === Node.TEXT_NODE && child.textContent!.trim()) {
        const range = document.createRange();
        range.selectNodeContents(child);
        const r = range.getClientRects()[0];
        if (!r) continue;
        const text = child.textContent!.replace(/\s+/g, " ").trim();
        const t = el("text", {
          x: r2(r.left - origin.left),
          y: r2(r.top - origin.top + ascent(style, text)),
          fill: style.color,
          "font-family": style.fontFamily,
          "font-size": style.fontSize,
          "font-weight": style.fontWeight,
        });
        t.textContent = text;
        parent.append(t);
      }
    }
  };
  for (const child of root.children) walk(child, svg);
  return '<?xml version="1.0" encoding="UTF-8"?>\n' + new XMLSerializer().serializeToString(svg) + "\n";
}
