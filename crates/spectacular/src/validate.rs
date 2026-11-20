//! The view-spec validator: a port of `validate` in `packages/viz/src/spec.ts`, kept
//! line for line so the conformance test can hold both to the same problems.

use std::collections::HashMap;

use serde_json::{Map, Value};

use crate::colors::{is_color, MAX_PALETTE};
use crate::{distance, FieldType, Problem};

pub const SPEC_VERSION: &str = "spectacular/v0";
pub const MAX_DIMENSIONS: usize = 32;

pub const VIEW_TYPES: &[&str] = &["scatter", "histogram", "bars", "density", "scatter3d", "arc", "sankey", "network3d"];

/// Required and optional field properties per view type.
pub fn view_fields(t: &str) -> (&'static [&'static str], &'static [&'static str]) {
    match t {
        "scatter" => (&["x", "y"], &["color"]),
        "histogram" | "bars" => (&["field"], &[]),
        "density" => (&["x", "y"], &[]),
        "scatter3d" => (&["x", "y", "z"], &["color"]),
        _ => (&["source", "target"], &[]),
    }
}

/// The type a view needs in a field property (None: any type).
pub fn needs(t: &str, key: &str) -> Option<FieldType> {
    use FieldType::*;
    match (t, key) {
        ("scatter" | "scatter3d", "color") => None,
        ("bars", _) => Some(Nominal),
        ("arc" | "sankey" | "network3d", _) => Some(Nominal),
        _ => Some(Quantitative),
    }
}

fn brushes(t: &str) -> &'static [&'static str] {
    match t {
        "scatter" => &["xy", "none"],
        "histogram" => &["x", "none"],
        "density" | "scatter3d" => &[],
        _ => &["select", "none"],
    }
}

const COMMON_VIEW_KEYS: &[&str] = &["id", "type", "title", "height", "span"];

fn extra_view_keys(t: &str) -> &'static [&'static str] {
    match t {
        "scatter" => &["brush", "pointSize", "fill", "palette"],
        "histogram" => &["brush", "bins", "fill"],
        "bars" => &["brush", "palette"],
        "density" => &["bins"],
        "scatter3d" => &["pointSize", "fill", "palette"],
        "arc" => &["brush", "order"],
        _ => &["brush"],
    }
}

const TOP_KEYS: &[&str] = &["$schema", "version", "title", "description", "theme", "data", "fields", "views", "layout", "output"];
const THEMES: &[&str] = &["auto", "light", "dark", "paper"];
const DATA_FORMATS: &[&str] = &["csv", "tsv", "json"];
const FIGURE_FORMATS: &[&str] = &["svg", "pdf", "png"];
const FIGURE_UNITS: &[&str] = &["mm", "in", "px"];
const COLOR_RULE: &str = "a colour: \"#rrggbb\", \"#rgb\" or a CSS colour name such as \"steelblue\"";

/// Unknown properties are errors: a typo silently ignored would draw a different figure.
fn unknown_keys(obj: &Map<String, Value>, allowed: &[&str], at: &str, out: &mut Vec<Problem>) {
    for key in obj.keys() {
        if allowed.contains(&key.as_str()) {
            continue;
        }
        let near = allowed.iter().find(|k| distance(&k.to_lowercase(), &key.to_lowercase()) <= 2);
        let hint = match near {
            Some(k) => format!("; did you mean \"{k}\"?"),
            None => format!("; allowed: {}", allowed.join(", ")),
        };
        out.push(Problem::new(format!("{at}/{key}"), format!("unknown property \"{key}\"{hint}")));
    }
}

fn positive(v: &Value) -> bool {
    v.as_f64().is_some_and(|n| n > 0.0)
}

fn integer(v: &Value) -> Option<f64> {
    v.as_f64().filter(|n| n.fract() == 0.0 && n.is_finite())
}

fn in_list(v: &Value, list: &[&str]) -> bool {
    v.as_str().is_some_and(|s| list.contains(&s))
}

/// Crossfilter dimensions a view needs (see `dimensionsNeeded` in spec.ts).
pub fn dimensions_needed(view: &Map<String, Value>) -> usize {
    match view.get("type").and_then(Value::as_str) {
        Some("scatter") => {
            if view.get("brush").and_then(Value::as_str).unwrap_or("xy") == "xy" { 2 } else { 0 }
        }
        Some("histogram" | "bars" | "arc" | "sankey" | "network3d") => 1,
        Some("density") => 2,
        _ => 0,
    }
}

/// The fields a spec reads, in the browser's order: those its views name (in each view
/// type's field order), then those `fields` declares.
pub fn needed_fields(spec: &Value) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for v in spec.get("views").and_then(Value::as_array).into_iter().flatten() {
        let Some(t) = v.get("type").and_then(Value::as_str).filter(|t| VIEW_TYPES.contains(t)) else { continue };
        let (required, optional) = view_fields(t);
        for key in required.iter().chain(optional) {
            if let Some(f) = v.get(*key).and_then(Value::as_str) {
                if !names.iter().any(|n| n == f) {
                    names.push(f.to_string());
                }
            }
        }
    }
    for name in spec.get("fields").and_then(Value::as_object).into_iter().flat_map(|f| f.keys()) {
        if !names.contains(name) {
            names.push(name.clone());
        }
    }
    names
}

/// Checks a spec's structure. `types` (known field types, e.g. from the data) also
/// enables the checks that each view's fields exist and have the type it needs.
pub fn validate(input: &Value, types: Option<&HashMap<String, FieldType>>) -> Vec<Problem> {
    let mut out = Vec::new();
    let Some(spec) = input.as_object() else {
        return vec![Problem::new("", "a spec must be a JSON object")];
    };
    unknown_keys(spec, TOP_KEYS, "", &mut out);

    if let Some(v) = spec.get("version") {
        if v.as_str() != Some(SPEC_VERSION) {
            out.push(Problem::new("/version", format!("unsupported version {v}; this build reads \"{SPEC_VERSION}\"")));
        }
    }
    if let Some(t) = spec.get("theme") {
        if !in_list(t, THEMES) {
            out.push(Problem::new("/theme", format!("one of {}", THEMES.join(", "))));
        }
    }

    match spec.get("data").and_then(Value::as_object) {
        None => out.push(Problem::new("/data", "required: { url } | { values } | { columns: \"external\" }")),
        Some(data) => {
            let kinds: Vec<&str> = ["url", "values", "columns"].into_iter().filter(|k| data.contains_key(*k)).collect();
            if kinds.len() != 1 {
                let got = if kinds.is_empty() { "none".to_string() } else { kinds.join(", ") };
                out.push(Problem::new("/data", format!("give exactly one of url, values, columns (got {got})")));
            } else if data.contains_key("url") {
                unknown_keys(data, &["url", "format", "pointer"], "/data", &mut out);
                if !data["url"].is_string() {
                    out.push(Problem::new("/data/url", "must be a string"));
                }
                if let Some(f) = data.get("format") {
                    if !in_list(f, DATA_FORMATS) {
                        out.push(Problem::new("/data/format", format!("one of {}", DATA_FORMATS.join(", "))));
                    }
                }
                if let Some(p) = data.get("pointer") {
                    if !p.as_str().is_some_and(|s| s.is_empty() || s.starts_with('/')) {
                        out.push(Problem::new("/data/pointer", "a JSON Pointer: \"\" or a string starting with \"/\""));
                    }
                }
            } else if data.contains_key("values") {
                unknown_keys(data, &["values"], "/data", &mut out);
                if !data["values"].as_array().is_some_and(|rows| rows.iter().all(Value::is_object)) {
                    out.push(Problem::new("/data/values", "must be an array of row objects"));
                }
            } else {
                unknown_keys(data, &["columns"], "/data", &mut out);
                if data["columns"].as_str() != Some("external") {
                    out.push(Problem::new("/data/columns", "must be \"external\""));
                }
            }
        }
    }

    let mut declared: HashMap<String, FieldType> = types.cloned().unwrap_or_default();
    if let Some(fields) = spec.get("fields") {
        match fields.as_object() {
            None => out.push(Problem::new("/fields", "must be an object of field name to definition")),
            Some(fields) => {
                for (name, def) in fields {
                    let Some(def) = def.as_object() else {
                        out.push(Problem::new(format!("/fields/{name}"), "must be an object"));
                        continue;
                    };
                    unknown_keys(def, &["type", "title"], &format!("/fields/{name}"), &mut out);
                    if let Some(t) = def.get("type") {
                        match t.as_str().and_then(FieldType::parse) {
                            Some(ft) => {
                                declared.insert(name.clone(), ft);
                            }
                            None => out.push(Problem::new(format!("/fields/{name}/type"), "must be \"quantitative\" or \"nominal\"")),
                        }
                    }
                }
            }
        }
    }

    let views = match spec.get("views").and_then(Value::as_array) {
        Some(v) if !v.is_empty() => v,
        _ => {
            out.push(Problem::new("/views", "required: a non-empty array of views"));
            return out;
        }
    };
    let mut ids: Vec<String> = Vec::new();
    for (i, v) in views.iter().enumerate() {
        let at = format!("/views/{i}");
        let Some(v) = v.as_object() else {
            out.push(Problem::new(at, "must be an object"));
            continue;
        };
        match v.get("id").and_then(Value::as_str) {
            Some(id) if !id.is_empty() => {
                if ids.iter().any(|x| x == id) {
                    out.push(Problem::new(format!("{at}/id"), format!("duplicate id \"{id}\"")));
                } else {
                    ids.push(id.to_string());
                }
            }
            _ => out.push(Problem::new(format!("{at}/id"), "required: a non-empty string")),
        }
        let t = match v.get("type").and_then(Value::as_str) {
            Some(t) if VIEW_TYPES.contains(&t) => t,
            other => {
                let shown = match (other, v.get("type")) {
                    (_, Some(raw)) => raw.to_string(),
                    (_, None) => "undefined".to_string(),
                };
                out.push(Problem::new(format!("{at}/type"), format!("unknown view type {shown}; one of {}", VIEW_TYPES.join(", "))));
                continue;
            }
        };
        let (required, optional) = view_fields(t);
        let allowed: Vec<&str> = COMMON_VIEW_KEYS.iter().chain(required).chain(optional).chain(extra_view_keys(t)).copied().collect();
        unknown_keys(v, &allowed, &at, &mut out);
        for key in required {
            if !v.get(*key).is_some_and(Value::is_string) {
                out.push(Problem::new(format!("{at}/{key}"), "required: the name of a field"));
            }
        }
        for key in required.iter().chain(optional) {
            let Some(field) = v.get(*key).and_then(Value::as_str) else { continue };
            let need = needs(t, key);
            let have = declared.get(field);
            if types.is_some_and(|ty| !ty.contains_key(field)) {
                out.push(Problem::new(format!("{at}/{key}"), format!("no field \"{field}\" in the data")));
            } else if let (Some(need), Some(have)) = (need, have) {
                if need != *have {
                    out.push(Problem::new(
                        format!("{at}/{key}"),
                        format!("{t}.{key} needs a {} field; \"{field}\" is {}", need.as_str(), have.as_str()),
                    ));
                }
            }
        }
        if let Some(b) = v.get("brush") {
            let list = brushes(t);
            if !list.is_empty() && !in_list(b, list) {
                out.push(Problem::new(format!("{at}/brush"), format!("one of {}", list.join(", "))));
            }
        }
        if let Some(bins) = v.get("bins") {
            let ok = if t == "density" {
                bins.as_array().is_some_and(|b| b.len() == 2 && b.iter().all(|x| integer(x).is_some_and(|n| n > 0.0 && n <= 1024.0)))
            } else {
                integer(bins).is_some_and(|n| n > 0.0 && n <= 10_000.0)
            };
            if !ok {
                let rule = if t == "density" { "[xBins, yBins], integers in 1..1024" } else { "an integer in 1..10000" };
                out.push(Problem::new(format!("{at}/bins"), rule));
            }
        }
        if let Some(o) = v.get("order") {
            if !in_list(o, &["count", "name", "data"]) {
                out.push(Problem::new(format!("{at}/order"), "one of count, name, data"));
            }
        }
        if let Some(f) = v.get("fill") {
            if !f.as_str().is_some_and(is_color) {
                out.push(Problem::new(format!("{at}/fill"), COLOR_RULE));
            }
        }
        if let Some(p) = v.get("palette") {
            match p.as_array() {
                Some(list) if !list.is_empty() && list.len() <= MAX_PALETTE => {
                    for (k, c) in list.iter().enumerate() {
                        if !c.as_str().is_some_and(is_color) {
                            out.push(Problem::new(format!("{at}/palette/{k}"), COLOR_RULE));
                        }
                    }
                }
                _ => out.push(Problem::new(format!("{at}/palette"), format!("a list of 1 to {MAX_PALETTE} colours"))),
            }
        }
        for key in ["height", "span", "pointSize"] {
            if let Some(x) = v.get(key) {
                if !positive(x) {
                    out.push(Problem::new(format!("{at}/{key}"), "must be a positive number"));
                }
            }
        }
    }

    if let Some(layout) = spec.get("layout") {
        match layout.as_object() {
            None => out.push(Problem::new("/layout", "must be an object")),
            Some(layout) => {
                unknown_keys(layout, &["columns"], "/layout", &mut out);
                if let Some(c) = layout.get("columns") {
                    if !integer(c).is_some_and(|n| (1.0..=12.0).contains(&n)) {
                        out.push(Problem::new("/layout/columns", "an integer in 1..12"));
                    }
                }
            }
        }
    }
    if let Some(output) = spec.get("output") {
        match output.as_object() {
            None => out.push(Problem::new("/output", "must be an object")),
            Some(output) => {
                unknown_keys(output, &["interactive", "figure"], "/output", &mut out);
                if output.get("interactive").is_some_and(|v| !v.is_boolean()) {
                    out.push(Problem::new("/output/interactive", "must be true or false"));
                }
            }
        }
    }
    let dims: usize = views
        .iter()
        .filter_map(Value::as_object)
        .filter(|v| v.get("type").and_then(Value::as_str).is_some_and(|t| VIEW_TYPES.contains(&t)))
        .map(dimensions_needed)
        .sum();
    if dims > MAX_DIMENSIONS {
        out.push(Problem::new(
            "/views",
            format!("these views need {dims} crossfilter dimensions; the limit is {MAX_DIMENSIONS} (scatter with brush 2, density 2, histogram 1, bars 1, arc 1, sankey 1, network3d 1, scatter3d 0)"),
        ));
    }

    if let Some(figure) = spec.get("output").and_then(Value::as_object).and_then(|o| o.get("figure")) {
        match figure.as_object() {
            None => out.push(Problem::new("/output/figure", "must be an object")),
            Some(figure) => {
                unknown_keys(figure, &["format", "width", "unit", "dpi"], "/output/figure", &mut out);
                if !figure.get("format").is_some_and(|f| in_list(f, FIGURE_FORMATS)) {
                    out.push(Problem::new("/output/figure/format", format!("one of {}", FIGURE_FORMATS.join(", "))));
                }
                if !figure.get("width").is_some_and(positive) {
                    out.push(Problem::new("/output/figure/width", "required: a positive number"));
                }
                if figure.get("unit").is_some_and(|u| !in_list(u, FIGURE_UNITS)) {
                    out.push(Problem::new("/output/figure/unit", format!("one of {}", FIGURE_UNITS.join(", "))));
                }
                if figure.get("dpi").is_some_and(|d| !positive(d)) {
                    out.push(Problem::new("/output/figure/dpi", "must be a positive number"));
                }
            }
        }
    }
    out
}
