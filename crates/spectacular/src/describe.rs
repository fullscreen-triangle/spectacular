//! Chart descriptions: the forgiving way to ask for views.
//!
//! A description names columns and, if it likes, a chart type and colours; everything
//! else is worked out from the data:
//!
//! ```json
//! {"title": "Growth plate", "labels": {"od600": "OD600"}, "charts": [
//!   {"x-axis": "time_h", "y-axis": "od600", "color": "medium"},
//!   {"x": "rate", "color": "teal"},
//!   {"x": "medium"}
//! ]}
//! ```
//!
//! Keys are matched without regard to case, spaces, dashes or underscores ("X axis",
//! "x_axis" and "x-axis" are one key), and so are column names. The chart type follows
//! from the columns: two numbers make a scatter (a density beyond 200,000 rows), three a
//! 3D scatter, one number a histogram, one category a bar chart, "from"/"to" a sankey.
//! `color` is a column to colour by, a colour for the marks, or a list of colours.
//!
//! A full view spec (every view has an `id` and a known `type`) passes through unchanged.
//! Problems point at the description's own keys, e.g. `/charts/1/x-axis`.

use std::collections::HashMap;

use serde_json::{json, Map, Value};

use crate::colors::is_color;
use crate::validate::{validate, SPEC_VERSION, VIEW_TYPES};
use crate::{distance, Columns, FieldType, Problem};

/// Beyond this many rows, two numeric columns are drawn as a density, not points.
pub const DENSITY_ROWS: usize = 200_000;

/// The result: a view spec when the description holds, and its problems.
#[derive(Debug)]
pub struct Compiled {
    pub spec: Option<Value>,
    pub problems: Vec<Problem>,
}

/// Keys without case, spaces, dashes or underscores.
fn norm(key: &str) -> String {
    key.chars().filter(|c| c.is_alphanumeric()).flat_map(char::to_lowercase).collect()
}

const TOP: &[(&str, &[&str])] = &[
    ("title", &["title", "heading"]),
    ("description", &["description", "about", "notes"]),
    ("data", &["data", "file", "csv"]),
    ("theme", &["theme", "style"]),
    ("columns", &["columns", "grid", "percolumn", "chartsperrow", "perrow"]),
    ("layout", &["layout"]),
    ("labels", &["labels", "titles", "axislabels", "names"]),
    ("categorical", &["categorical", "nominal", "categories", "categoricalcolumns", "groups"]),
    ("domains", &["domains", "domain", "levels", "categoryorder", "orders"]),
    ("palettes", &["palettes", "fieldcolors", "fieldcolours", "categorycolors", "categorycolours"]),
    ("key", &["key", "rowkey", "keycolumn", "idcolumn"]),
    ("figure", &["figure"]),
    ("output", &["output"]),
    ("fields", &["fields"]),
    ("version", &["version"]),
    ("$schema", &["schema"]),
];

const LISTS: &[&str] = &["charts", "views", "plots", "graphs", "figures", "panels"];

const CHART: &[(&str, &[&str])] = &[
    ("x", &["x", "xaxis", "horizontal", "across"]),
    ("y", &["y", "yaxis", "vertical", "up"]),
    ("z", &["z", "zaxis", "depth"]),
    ("field", &["field", "column", "value", "values"]),
    ("from", &["from", "source", "start"]),
    ("to", &["to", "target", "end"]),
    ("low", &["low", "lower", "lowerbound", "lowend", "min", "minimum"]),
    ("high", &["high", "upper", "upperbound", "highend", "max", "maximum"]),
    ("color", &["color", "colour", "colorby", "colourby", "hue", "fill"]),
    ("palette", &["palette", "colors", "colours"]),
    ("title", &["title", "name", "label", "caption"]),
    ("type", &["type", "chart", "kind", "as", "charttype", "plot", "mark"]),
    ("bins", &["bins", "buckets"]),
    ("size", &["size", "pointsize", "dotsize"]),
    ("height", &["height"]),
    ("span", &["span"]),
    ("wide", &["wide", "fullwidth"]),
    ("brush", &["brush", "select", "selectable", "interactive"]),
    ("order", &["order", "sort"]),
    ("scale", &["scale", "scales", "axis"]),
    ("id", &["id"]),
];

const CHART_TYPES: &[(&str, &[&str])] = &[
    ("scatter", &["scatter", "points", "point", "dots", "xy", "scatterplot"]),
    ("histogram", &["histogram", "hist", "distribution"]),
    ("bars", &["bars", "bar", "barchart", "counts", "count", "categories"]),
    ("density", &["density", "heatmap", "heat", "hexbin", "2dhistogram", "histogram2d"]),
    ("scatter3d", &["scatter3d", "3d", "points3d", "cloud", "pointcloud"]),
    ("arc", &["arc", "arcs", "arcdiagram"]),
    ("sankey", &["sankey", "flow", "flows"]),
    ("network3d", &["network3d", "network", "graph", "graph3d"]),
    ("strip", &["strip", "stripplot", "jitter", "beeswarm", "swarm"]),
    ("matrix", &["matrix", "grid", "crosstab", "contingency", "table"]),
    ("interval", &["interval", "intervals", "range", "ranges", "errorbar", "errorbars", "forest"]),
    ("layers", &["layers", "layered", "dag", "pipeline", "steps", "flowchart"]),
];

fn canonical(table: &[(&'static str, &[&str])], key: &str) -> Option<&'static str> {
    let k = norm(key);
    table.iter().find(|(_, aliases)| aliases.contains(&k.as_str())).map(|(c, _)| *c)
}

fn suggest(table: &[(&'static str, &[&str])], key: &str) -> String {
    let k = norm(key);
    let mut near: Option<(&str, usize)> = None;
    for (c, aliases) in table {
        for a in aliases.iter() {
            let d = distance(a, &k);
            if d <= 2 && near.is_none_or(|(_, best)| d < best) {
                near = Some((c, d));
            }
        }
    }
    match near {
        Some((c, _)) => format!("unknown key \"{key}\"; did you mean \"{c}\"?"),
        None => format!("unknown key \"{key}\"; known: {}", table.iter().map(|(c, _)| *c).collect::<Vec<_>>().join(", ")),
    }
}

/// True when `v` is already a view spec: every view has an id and a known type.
fn is_spec(v: &Value) -> bool {
    v.get("views").and_then(Value::as_array).is_some_and(|views| {
        !views.is_empty()
            && views.iter().all(|w| {
                w.get("id").is_some_and(Value::is_string) && w.get("type").and_then(Value::as_str).is_some_and(|t| VIEW_TYPES.contains(&t))
            })
    })
}

/// The data's columns, for looking names up.
struct Lookup<'a> {
    columns: Option<&'a Columns>,
}

impl Lookup<'_> {
    /// The data's own name for `name`, matched exactly, then without case or spacing.
    fn resolve(&self, name: &str) -> Result<String, String> {
        let Some(cols) = self.columns else { return Ok(name.to_string()) };
        if cols.iter().any(|(c, _)| c == name) {
            return Ok(name.to_string());
        }
        let n = norm(name);
        let matches: Vec<&String> = cols.iter().filter(|(c, _)| norm(c) == n).map(|(c, _)| c).collect();
        match matches.as_slice() {
            [one] => Ok((*one).clone()),
            [] => {
                let near = cols
                    .iter()
                    .map(|(c, _)| (c, distance(&norm(c), &n)))
                    .filter(|(_, d)| *d <= 2.max(n.len() / 3))
                    .min_by_key(|(_, d)| *d);
                Err(match near {
                    Some((c, _)) => format!("no column \"{name}\"; did you mean \"{c}\"?"),
                    None => {
                        let list: Vec<&str> = cols.iter().take(12).map(|(c, _)| c.as_str()).collect();
                        let more = if cols.len() > 12 { ", ..." } else { "" };
                        format!("no column \"{name}\"; the data has {}{more}", list.join(", "))
                    }
                })
            }
            many => Err(format!(
                "\"{name}\" could be any of {}; write the name exactly",
                many.iter().map(|c| format!("\"{c}\"")).collect::<Vec<_>>().join(", ")
            )),
        }
    }

    fn type_of(&self, name: &str, categorical: &[String]) -> Option<FieldType> {
        if categorical.iter().any(|c| c == name) {
            return Some(FieldType::Nominal);
        }
        self.columns.and_then(|cols| cols.iter().find(|(c, _)| c == name).map(|(_, t)| *t))
    }
}

fn slug(text: &str) -> String {
    let mut out = String::new();
    for c in text.chars().flat_map(char::to_lowercase) {
        if c.is_alphanumeric() {
            out.push(c);
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    let s = out.trim_matches('-').to_string();
    if s.is_empty() { "chart".into() } else { s }
}

/// One chart's view, and where each of its spec keys came from in the description.
struct Built {
    view: Map<String, Value>,
    origin: HashMap<String, String>,
    wide: bool,
}

struct Context<'a> {
    lookup: Lookup<'a>,
    categorical: Vec<String>,
    labels: HashMap<String, String>,
    rows: Option<usize>,
}

impl Context<'_> {
    fn label(&self, column: &str) -> String {
        self.labels.get(column).cloned().unwrap_or_else(|| column.to_string())
    }

    /// `single`: the chart's keys share the object with the description's own settings
    /// ("data", "theme", ...), which are then not the chart's to report.
    fn build(&self, chart: &Map<String, Value>, at: &str, single: bool, out: &mut Vec<Problem>) -> Option<Built> {
        let mut keys: HashMap<&'static str, (&String, &Value)> = HashMap::new();
        let before = out.len();
        for (k, v) in chart {
            if single && canonical(CHART, k).is_none() && canonical(TOP, k).is_some() {
                continue;
            }
            match canonical(CHART, k) {
                Some(c) => {
                    if let Some((first, _)) = keys.get(c) {
                        out.push(Problem::new(format!("{at}/{k}"), format!("\"{k}\" and \"{first}\" say the same thing; keep one")));
                    } else {
                        keys.insert(c, (k, v));
                    }
                }
                None => out.push(Problem::new(format!("{at}/{k}"), suggest(CHART, k))),
            }
        }
        let path = |c: &str| keys.get(c).map(|(k, _)| format!("{at}/{k}")).unwrap_or_else(|| at.to_string());
        let mut origin: HashMap<String, String> = HashMap::new();

        // Columns named by the chart.
        let column = |c: &'static str, out: &mut Vec<Problem>| -> Option<String> {
            let (k, v) = keys.get(c)?;
            match v.as_str() {
                Some(name) => match self.lookup.resolve(name) {
                    Ok(found) => Some(found),
                    Err(message) => {
                        out.push(Problem::new(format!("{at}/{k}"), message));
                        None
                    }
                },
                None => {
                    out.push(Problem::new(format!("{at}/{k}"), "the name of a column"));
                    None
                }
            }
        };
        let x = column("x", out);
        let y = column("y", out);
        let z = column("z", out);
        let field = column("field", out);
        let from = column("from", out);
        let to = column("to", out);
        let low = column("low", out);
        let high = column("high", out);

        // Colour: `color` is a column to colour by, one colour, or a list of colours;
        // `palette` is a list of colours for the categories.
        let mut color_field = None;
        let mut fill = None;
        let mut palette: Option<Vec<Value>> = None;
        let colours = |v: &Value| v.as_array().filter(|l| !l.is_empty() && l.iter().all(|c| c.as_str().is_some_and(is_color))).cloned();
        const LIST: &str = "a list of colours, e.g. [\"#4e79a7\", \"tomato\"]";
        if let Some((k, v)) = keys.get("color") {
            let p = format!("{at}/{k}");
            match v {
                // A name the data has is a column; otherwise a colour, if it is one.
                Value::String(s) => match self.lookup.resolve(s) {
                    Ok(found) if self.lookup.columns.is_some() || !is_color(s) => color_field = Some(found),
                    Ok(_) => fill = Some(s.clone()),
                    Err(_) if is_color(s) => fill = Some(s.clone()),
                    Err(message) => out.push(Problem::new(p, format!("{message}, and not a colour either"))),
                },
                Value::Array(_) => match colours(v) {
                    Some(list) => palette = Some(list),
                    None => out.push(Problem::new(p, LIST)),
                },
                _ => out.push(Problem::new(p, "a column name, a colour, or a list of colours")),
            }
        }
        if let Some((k, v)) = keys.get("palette") {
            match colours(v) {
                Some(_) if palette.is_some() => out.push(Problem::new(format!("{at}/{k}"), "colours are already given as \"color\"; keep one")),
                Some(list) => palette = Some(list),
                None => out.push(Problem::new(format!("{at}/{k}"), LIST)),
            }
        }

        let categorical = &self.categorical;
        let ty = |c: &Option<String>| c.as_ref().and_then(|n| self.lookup.type_of(n, categorical));
        let quantitative = |c: &Option<String>| ty(c) != Some(FieldType::Nominal);

        // The chart type: as asked, or from the columns.
        let asked = match keys.get("type") {
            Some((k, v)) => match v.as_str().and_then(|s| canonical(CHART_TYPES, s)) {
                Some(t) => Some(t),
                None => {
                    out.push(Problem::new(
                        format!("{at}/{k}"),
                        format!("unknown chart type {v}; one of {}", CHART_TYPES.iter().map(|(t, _)| *t).collect::<Vec<_>>().join(", ")),
                    ));
                    return None;
                }
            },
            None => None,
        };
        if out.len() > before {
            return None;
        }
        let one = field.clone().or_else(|| x.clone()).or_else(|| y.clone());
        let one_key = if field.is_some() { "field" } else if x.is_some() { "x" } else { "y" };
        let kind = match asked {
            Some(t) => t,
            None if low.is_some() || high.is_some() => "interval",
            None if from.is_some() || to.is_some() => "sankey",
            None if x.is_some() && y.is_some() && z.is_some() => "scatter3d",
            None if x.is_some() && y.is_some() => match (quantitative(&x), quantitative(&y)) {
                (true, true) => {
                    if self.rows.is_some_and(|r| r > DENSITY_ROWS) && color_field.is_none() { "density" } else { "scatter" }
                }
                (false, false) => "matrix",
                _ => "strip",
            },
            None if one.is_some() => {
                if quantitative(&one) { "histogram" } else { "bars" }
            }
            None => {
                out.push(Problem::new(at.to_string(), "say what to draw: \"x\" (and \"y\", \"z\"), or \"from\" and \"to\""));
                return None;
            }
        };

        let mut view = Map::new();
        view.insert("id".into(), Value::Null); // filled in below, kept first
        view.insert("type".into(), json!(kind));
        view.insert("title".into(), Value::Null); // filled in below, kept third
        let put = |view: &mut Map<String, Value>, origin: &mut HashMap<String, String>, key: &str, value: Value, from_key: &str| {
            view.insert(key.to_string(), value);
            if let Some((k, _)) = keys.get(from_key) {
                origin.insert(key.to_string(), (*k).clone());
            }
        };
        let default_title;
        match kind {
            "scatter" | "density" => {
                let (Some(xx), Some(yy)) = (x.clone(), y.clone()) else {
                    out.push(Problem::new(at.to_string(), format!("a {kind} needs \"x\" and \"y\"")));
                    return None;
                };
                default_title = format!("{} vs {}", self.label(&yy), self.label(&xx));
                put(&mut view, &mut origin, "x", json!(xx), "x");
                put(&mut view, &mut origin, "y", json!(yy), "y");
            }
            "scatter3d" => {
                let (Some(xx), Some(yy), Some(zz)) = (x.clone(), y.clone(), z.clone()) else {
                    out.push(Problem::new(at.to_string(), "a 3D scatter needs \"x\", \"y\" and \"z\""));
                    return None;
                };
                default_title = format!("{}, {}, {}", self.label(&xx), self.label(&yy), self.label(&zz));
                put(&mut view, &mut origin, "x", json!(xx), "x");
                put(&mut view, &mut origin, "y", json!(yy), "y");
                put(&mut view, &mut origin, "z", json!(zz), "z");
            }
            "histogram" | "bars" => {
                let Some(f) = one.clone() else {
                    out.push(Problem::new(at.to_string(), format!("a {kind} needs one column, as \"x\"")));
                    return None;
                };
                default_title = match (&color_field, kind) {
                    (Some(c), "bars") => format!("{} by {}", self.label(&f), self.label(c)),
                    _ => self.label(&f),
                };
                put(&mut view, &mut origin, "field", json!(f), one_key);
            }
            "strip" => {
                let (Some(xx), Some(yy)) = (x.clone(), y.clone()) else {
                    out.push(Problem::new(at.to_string(), "a strip needs a number and a category, as \"x\" and \"y\""));
                    return None;
                };
                // The value runs across and the categories down, whichever way round they came.
                let swap = !quantitative(&x) && quantitative(&y);
                let ((value, value_key), (category, category_key)) = if swap { ((yy, "y"), (xx, "x")) } else { ((xx, "x"), (yy, "y")) };
                default_title = format!("{} by {}", self.label(&value), self.label(&category));
                put(&mut view, &mut origin, "x", json!(value), value_key);
                put(&mut view, &mut origin, "y", json!(category), category_key);
            }
            "matrix" => {
                let (Some(xx), Some(yy)) = (x.clone(), y.clone()) else {
                    out.push(Problem::new(at.to_string(), "a matrix needs two categories, as \"x\" and \"y\""));
                    return None;
                };
                default_title = format!("{} by {}", self.label(&yy), self.label(&xx));
                put(&mut view, &mut origin, "x", json!(xx), "x");
                put(&mut view, &mut origin, "y", json!(yy), "y");
            }
            "interval" => {
                let (Some(l), Some(h)) = (low.clone(), high.clone()) else {
                    out.push(Problem::new(at.to_string(), "an interval needs \"low\" and \"high\" columns"));
                    return None;
                };
                put(&mut view, &mut origin, "low", json!(l), "low");
                put(&mut view, &mut origin, "high", json!(h), "high");
                // The bands: whichever of y, x or field names a column.
                let band = [("y", &y), ("x", &x), ("field", &field)].into_iter().find_map(|(k, c)| c.clone().map(|c| (k, c)));
                default_title = match &band {
                    Some((_, b)) => format!("{} – {} by {}", self.label(&l), self.label(&h), self.label(b)),
                    None => format!("{} – {}", self.label(&l), self.label(&h)),
                };
                if let Some((k, b)) = band {
                    put(&mut view, &mut origin, "label", json!(b), k);
                }
            }
            _ => {
                let (Some(s), Some(t)) = (from.clone(), to.clone()) else {
                    out.push(Problem::new(at.to_string(), format!("a {kind} needs \"from\" and \"to\" columns")));
                    return None;
                };
                default_title = format!("{} → {}", self.label(&s), self.label(&t));
                put(&mut view, &mut origin, "source", json!(s), "from");
                put(&mut view, &mut origin, "target", json!(t), "to");
            }
        }

        // Colours, where the chart type can take them.
        let palette_key = if keys.contains_key("palette") { "palette" } else { "color" };
        let colour_path = path("color");
        match kind {
            "scatter" | "scatter3d" | "strip" | "interval" => {
                if let Some(c) = color_field {
                    put(&mut view, &mut origin, "color", json!(c), "color");
                }
                if let Some(f) = fill {
                    put(&mut view, &mut origin, "fill", json!(f), "color");
                }
                if let Some(p) = palette {
                    put(&mut view, &mut origin, "palette", Value::Array(p), palette_key);
                }
            }
            "bars" if color_field.is_some() => {
                // Bars coloured by a category are split into it.
                let c = color_field.unwrap_or_default();
                if ty(&Some(c.clone())) == Some(FieldType::Quantitative) {
                    out.push(Problem::new(colour_path, format!("bars split by a column need a category; \"{c}\" is a number")));
                    return None;
                }
                put(&mut view, &mut origin, "color", json!(c), "color");
                if let Some(p) = palette {
                    put(&mut view, &mut origin, "palette", Value::Array(p), palette_key);
                }
            }
            "histogram" if color_field.is_some() => {
                out.push(Problem::new(
                    colour_path,
                    format!("colouring by a column (\"{}\") works on point charts and bars; give a histogram one colour instead, e.g. \"teal\"", color_field.unwrap_or_default()),
                ));
                return None;
            }
            "histogram" => match (fill, palette) {
                (Some(f), _) => put(&mut view, &mut origin, "fill", json!(f), "color"),
                (None, Some(p)) if p.len() == 1 => put(&mut view, &mut origin, "fill", p[0].clone(), palette_key),
                (None, Some(_)) => {
                    out.push(Problem::new(path(palette_key), "a histogram has one colour; give one, e.g. \"teal\""));
                    return None;
                }
                (None, None) => {}
            },
            "bars" => match (fill, palette) {
                (Some(f), _) => put(&mut view, &mut origin, "palette", json!([f]), "color"),
                (None, Some(p)) => put(&mut view, &mut origin, "palette", Value::Array(p), palette_key),
                (None, None) => {}
            },
            _ if color_field.is_some() || fill.is_some() || palette.is_some() => {
                out.push(Problem::new(path(if keys.contains_key("color") { "color" } else { "palette" }), format!("a {kind} takes no colour setting")));
                return None;
            }
            _ => {}
        }

        let title = keys.get("title").and_then(|(_, v)| v.as_str()).map(str::to_string);
        if let Some((k, v)) = keys.get("title") {
            if v.as_str().is_none() {
                out.push(Problem::new(format!("{at}/{k}"), "a title is text"));
                return None;
            }
        }
        view.insert("title".into(), json!(title.clone().unwrap_or_else(|| default_title.clone())));
        if let Some((k, _)) = keys.get("title") {
            origin.insert("title".into(), (*k).clone());
        }
        if let Some((k, v)) = keys.get("scale") {
            // "each" and the like mean one scale per band; "one" and the like a shared one.
            let value = match v.as_str().map(norm).as_deref() {
                Some("label" | "each" | "own" | "separate" | "perlabel" | "perband" | "band") => json!("label"),
                Some("shared" | "one" | "common" | "same") => json!("shared"),
                _ => (*v).clone(),
            };
            view.insert("scale".into(), value);
            origin.insert("scale".into(), (*k).clone());
        }
        for (canon, spec_key) in [("bins", "bins"), ("size", "pointSize"), ("height", "height"), ("span", "span"), ("order", "order")] {
            if let Some((_, v)) = keys.get(canon) {
                put(&mut view, &mut origin, spec_key, (*v).clone(), canon);
            }
        }
        if let Some((k, v)) = keys.get("brush") {
            let value = match v {
                Value::Bool(true) => None,
                Value::Bool(false) => Some(json!("none")),
                other => Some((*other).clone()),
            };
            if let Some(value) = value {
                view.insert("brush".into(), value);
                origin.insert("brush".into(), (*k).clone());
            }
        }
        let wide = match keys.get("wide") {
            Some((_, Value::Bool(b))) => *b,
            Some((k, _)) => {
                out.push(Problem::new(format!("{at}/{k}"), "true or false"));
                return None;
            }
            None => false,
        };
        let id = match keys.get("id").and_then(|(_, v)| v.as_str()) {
            Some(id) => {
                origin.insert("id".into(), keys["id"].0.clone());
                id.to_string()
            }
            None => slug(title.as_deref().unwrap_or(&default_title)),
        };
        view.insert("id".into(), json!(id));
        Some(Built { view, origin, wide })
    }
}

/// Compiles a description (or passes a spec through) and validates the result against
/// `columns` when they are known. `rows` lets large tables get density views.
pub fn compile(desc: &Value, columns: Option<&Columns>, rows: Option<usize>) -> Compiled {
    let types: Option<HashMap<String, FieldType>> = columns.map(|c| c.iter().cloned().collect());
    if is_spec(desc) {
        let problems = validate(desc, types.as_ref());
        return Compiled { spec: problems.is_empty().then(|| desc.clone()), problems };
    }

    let mut out = Vec::new();
    let empty = Map::new();
    // Where the charts are: a list, a list under a key, or one chart at the top.
    let (top, charts, list_at): (&Map<String, Value>, Vec<(&Map<String, Value>, String)>, String) = match desc {
        Value::Array(items) => {
            let mut charts = Vec::new();
            for (i, item) in items.iter().enumerate() {
                match item.as_object() {
                    Some(o) => charts.push((o, format!("/{i}"))),
                    None => out.push(Problem::new(format!("/{i}"), "a chart is an object, e.g. {\"x\": \"column A\"}")),
                }
            }
            (&empty, charts, String::new())
        }
        Value::Object(o) => match o.iter().find(|(k, _)| LISTS.contains(&norm(k).as_str())) {
            Some((k, list)) => {
                let mut charts = Vec::new();
                match list {
                    Value::Array(items) => {
                        for (i, item) in items.iter().enumerate() {
                            match item.as_object() {
                                Some(c) => charts.push((c, format!("/{k}/{i}"))),
                                None => out.push(Problem::new(format!("/{k}/{i}"), "a chart is an object, e.g. {\"x\": \"column A\"}")),
                            }
                        }
                    }
                    Value::Object(c) => charts.push((c, format!("/{k}"))),
                    _ => out.push(Problem::new(format!("/{k}"), "a list of charts")),
                }
                (o, charts, format!("/{k}"))
            }
            None => (o, vec![(o, String::new())], String::new()),
        },
        _ => {
            return Compiled { spec: None, problems: vec![Problem::new("", "describe charts as a JSON object, or a list of them")] };
        }
    };
    let single = !top.is_empty() && charts.len() == 1 && std::ptr::eq(charts[0].0, top);

    // Top-level settings.
    let mut tops: HashMap<&'static str, (&String, &Value)> = HashMap::new();
    for (k, v) in top {
        if LISTS.contains(&norm(k).as_str()) {
            continue;
        }
        match canonical(TOP, k) {
            Some(c) if !(single && c == "title") => {
                tops.insert(c, (k, v));
            }
            Some(_) => {}
            None if single && canonical(CHART, k).is_some() => {}
            None => out.push(Problem::new(format!("/{k}"), suggest(TOP, k))),
        }
    }
    let lookup = Lookup { columns };
    let mut categorical = Vec::new();
    if let Some((k, v)) = tops.get("categorical") {
        match v.as_array() {
            Some(list) => {
                for (i, name) in list.iter().enumerate() {
                    match name.as_str().map(|n| lookup.resolve(n)) {
                        Some(Ok(c)) => categorical.push(c),
                        Some(Err(m)) => out.push(Problem::new(format!("/{k}/{i}"), m)),
                        None => out.push(Problem::new(format!("/{k}/{i}"), "the name of a column")),
                    }
                }
            }
            None => out.push(Problem::new(format!("/{k}"), "a list of column names to treat as categories")),
        }
    }
    let mut labels = HashMap::new();
    if let Some((k, v)) = tops.get("labels") {
        match v.as_object() {
            Some(map) => {
                for (name, label) in map {
                    match (lookup.resolve(name), label.as_str()) {
                        (Ok(c), Some(l)) => {
                            labels.insert(c, l.to_string());
                        }
                        (Err(m), _) => out.push(Problem::new(format!("/{k}/{name}"), m)),
                        (_, None) => out.push(Problem::new(format!("/{k}/{name}"), "a label is text")),
                    }
                }
            }
            None => out.push(Problem::new(format!("/{k}"), "column name to axis label, e.g. {\"od600\": \"OD600\"}")),
        }
    }

    // Declared category orders and colours, by column; both make a column a category.
    let mut declared: Vec<(String, &'static str, Value, String)> = Vec::new(); // (column, "domain" | "palette", value, path)
    for (top, spec_key, example) in [
        ("domains", "domain", "column name to its categories in order, e.g. {\"way\": [\"stated\", \"supplied\"]}"),
        ("palettes", "palette", "column name to its colours, e.g. {\"outcome\": {\"published\": \"#1baf7a\"}}"),
    ] {
        let Some((k, v)) = tops.get(top) else { continue };
        let Some(map) = v.as_object() else {
            out.push(Problem::new(format!("/{k}"), example));
            continue;
        };
        for (name, value) in map {
            match lookup.resolve(name) {
                Ok(c) => {
                    if !categorical.contains(&c) {
                        categorical.push(c.clone());
                    }
                    declared.push((c, spec_key, value.clone(), format!("/{k}/{name}")));
                }
                Err(m) => out.push(Problem::new(format!("/{k}/{name}"), m)),
            }
        }
    }
    let key = match tops.get("key") {
        Some((k, Value::String(name))) => match lookup.resolve(name) {
            Ok(c) => Some((c, format!("/{k}"))),
            Err(m) => {
                out.push(Problem::new(format!("/{k}"), m));
                None
            }
        },
        Some((k, _)) => {
            out.push(Problem::new(format!("/{k}"), "the name of the column whose values identify rows"));
            None
        }
        None => None,
    };

    let ctx = Context { lookup, categorical, labels, rows };
    let mut built = Vec::new();
    for (chart, at) in &charts {
        if let Some(b) = ctx.build(chart, at, single, &mut out) {
            built.push((b, at.clone()));
        }
    }
    if charts.is_empty() && out.is_empty() {
        out.push(Problem::new(list_at.clone(), "no charts: add at least one, e.g. {\"x\": \"column A\"}"));
    }
    if !out.is_empty() {
        return Compiled { spec: None, problems: out };
    }

    // A palette given with a category to colour by becomes that column's own (unless
    // `palettes` names one), so every chart of the column shows the same colours: the
    // bars, a scatter's key, a sankey's nodes.
    for (b, at) in &built {
        let (Some(Value::String(c)), Some(p)) = (b.view.get("color"), b.view.get("palette")) else { continue };
        let nominal = ctx.lookup.type_of(c, &ctx.categorical) == Some(FieldType::Nominal);
        if nominal && !declared.iter().any(|(d, k, _, _)| d == c && *k == "palette") {
            let from = b.origin.get("palette").map(|k| format!("{at}/{k}")).unwrap_or_else(|| at.clone());
            declared.push((c.clone(), "palette", p.clone(), from));
        }
    }

    // Unique ids, in chart order.
    let mut seen: HashMap<String, usize> = HashMap::new();
    for (b, _) in built.iter_mut() {
        let id = b.view["id"].as_str().unwrap_or("chart").to_string();
        let n = seen.entry(id.clone()).or_insert(0);
        *n += 1;
        if *n > 1 && !b.origin.contains_key("id") {
            b.view.insert("id".into(), json!(format!("{id}-{n}")));
        }
    }

    let mut spec = Map::new();
    spec.insert("version".into(), json!(SPEC_VERSION));
    let mut back: HashMap<String, String> = HashMap::new(); // spec path prefix -> description path
    for key in ["title", "description", "theme"] {
        if let Some((k, v)) = tops.get(key) {
            spec.insert(key.into(), (*v).clone());
            back.insert(format!("/{key}"), format!("/{k}"));
        }
    }
    match tops.get("data") {
        Some((k, Value::String(url))) => {
            spec.insert("data".into(), json!({ "url": url }));
            back.insert("/data".into(), format!("/{k}"));
        }
        Some((k, other)) => {
            spec.insert("data".into(), (*other).clone());
            back.insert("/data".into(), format!("/{k}"));
        }
        None => {
            spec.insert("data".into(), json!({ "columns": "external" }));
        }
    }
    let mut fields = match tops.get("fields") {
        Some((_, Value::Object(f))) => f.clone(),
        _ => Map::new(),
    };
    for c in &ctx.categorical {
        let entry = fields.entry(c.clone()).or_insert_with(|| json!({}));
        if let Some(o) = entry.as_object_mut() {
            o.insert("type".into(), json!("nominal"));
        }
    }
    for (c, l) in &ctx.labels {
        let entry = fields.entry(c.clone()).or_insert_with(|| json!({}));
        if let Some(o) = entry.as_object_mut() {
            o.insert("title".into(), json!(l));
        }
    }
    for (c, spec_key, value, from) in &declared {
        let entry = fields.entry(c.clone()).or_insert_with(|| json!({}));
        if let Some(o) = entry.as_object_mut() {
            o.insert((*spec_key).into(), value.clone());
            back.insert(format!("/fields/{c}/{spec_key}"), from.clone());
        }
    }
    if !fields.is_empty() {
        spec.insert("fields".into(), Value::Object(fields));
    }
    if let Some((k, from)) = key {
        spec.insert("key".into(), json!(k));
        back.insert("/key".into(), from);
    }
    let grid = match tops.get("columns") {
        Some((k, v)) => {
            back.insert("/layout/columns".into(), format!("/{k}"));
            Some((*v).clone())
        }
        None => tops.get("layout").and_then(|(_, v)| v.get("columns")).cloned(),
    };
    let per_row = grid.as_ref().and_then(Value::as_u64).unwrap_or(built.len().min(3) as u64).max(1);
    let views: Vec<Value> = built
        .iter()
        .enumerate()
        .map(|(i, (b, at))| {
            let mut v = b.view.clone();
            if b.wide {
                v.insert("span".into(), json!(per_row));
            }
            back.insert(format!("/views/{i}"), at.clone());
            for (spec_key, desc_key) in &b.origin {
                back.insert(format!("/views/{i}/{spec_key}"), format!("{at}/{desc_key}"));
            }
            Value::Object(v)
        })
        .collect();
    spec.insert("views".into(), Value::Array(views));
    match (grid, tops.get("layout")) {
        (Some(g), _) => {
            spec.insert("layout".into(), json!({ "columns": g }));
        }
        (None, Some((k, v))) => {
            spec.insert("layout".into(), (*v).clone());
            back.insert("/layout".into(), format!("/{k}"));
        }
        _ => {}
    }
    if let Some((k, v)) = tops.get("output") {
        spec.insert("output".into(), (*v).clone());
        back.insert("/output".into(), format!("/{k}"));
    } else if let Some((k, v)) = tops.get("figure") {
        spec.insert("output".into(), json!({ "figure": v }));
        back.insert("/output/figure".into(), format!("/{k}"));
    }
    for key in ["version", "$schema"] {
        if let Some((_, v)) = tops.get(key) {
            spec.insert(key.into(), (*v).clone());
        }
    }

    let spec = Value::Object(spec);
    let problems: Vec<Problem> = validate(&spec, types.as_ref())
        .into_iter()
        .map(|p| Problem::new(translate(&p.path, &back), p.message))
        .collect();
    Compiled { spec: problems.is_empty().then_some(spec), problems }
}

/// A spec path in the description's terms: the longest mapped prefix, rewritten.
fn translate(path: &str, back: &HashMap<String, String>) -> String {
    let mut best: Option<(&String, &String)> = None;
    for (from, to) in back {
        let fits = path == from || path.starts_with(&format!("{from}/"));
        if fits && best.is_none_or(|(b, _)| from.len() > b.len()) {
            best = Some((from, to));
        }
    }
    match best {
        Some((from, to)) => format!("{to}{}", &path[from.len()..]),
        None => path.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cols() -> Columns {
        vec![
            ("Time (h)".into(), FieldType::Quantitative),
            ("od600".into(), FieldType::Quantitative),
            ("medium".into(), FieldType::Nominal),
            ("well".into(), FieldType::Quantitative),
            ("rate".into(), FieldType::Quantitative),
            ("src".into(), FieldType::Nominal),
            ("dst".into(), FieldType::Nominal),
        ]
    }

    fn compile_ok(desc: Value) -> Value {
        let c = compile(&desc, Some(&cols()), Some(100));
        assert!(c.problems.is_empty(), "{:?}", c.problems);
        c.spec.unwrap()
    }

    #[test]
    fn one_chart_with_friendly_keys() {
        let s = compile_ok(json!({"X axis": "time h", "y-axis": "OD600", "Colour": "medium"}));
        assert_eq!(s["views"][0], json!({"id": "od600-vs-time-h", "type": "scatter", "title": "od600 vs Time (h)", "x": "Time (h)", "y": "od600", "color": "medium"}));
        assert_eq!(s["data"], json!({"columns": "external"}));
    }

    #[test]
    fn chart_types_follow_the_columns() {
        let s = compile_ok(json!({"title": "Plate", "categorical": ["well"], "labels": {"od600": "OD600"}, "columns": 2, "charts": [
            {"x": "rate", "color": "teal"},
            {"x": "well"},
            {"x": "medium", "colours": ["red", "#00f"]},
            {"x": "rate", "y": "od600", "z": "Time (h)"},
            {"from": "src", "to": "dst", "wide": true},
        ]}));
        let types: Vec<&str> = s["views"].as_array().unwrap().iter().map(|v| v["type"].as_str().unwrap()).collect();
        assert_eq!(types, ["histogram", "bars", "bars", "scatter3d", "sankey"]);
        assert_eq!(s["views"][0]["fill"], "teal");
        assert_eq!(s["views"][2]["palette"], json!(["red", "#00f"]));
        assert_eq!(s["views"][4]["span"], 2);
        assert_eq!(s["fields"]["well"]["type"], "nominal");
        assert_eq!(s["fields"]["od600"]["title"], "OD600");
        assert_eq!(s["layout"], json!({"columns": 2}));
    }

    #[test]
    fn a_column_keeps_its_colours_across_charts() {
        let s = compile_ok(json!([{"x": "rate", "y": "od600", "color": "medium", "palette": ["teal", "gold"]}, {"x": "medium"}, {"x": "medium", "color": "red"}]));
        // The palette becomes the field's, which every chart of it uses; a chart's own colour still wins.
        assert_eq!(s["fields"]["medium"]["palette"], json!(["teal", "gold"]));
        assert!(s["views"][1].get("palette").is_none());
        assert_eq!(s["views"][2]["palette"], json!(["red"]));
    }

    #[test]
    fn categories_and_numbers_choose_the_newer_charts() {
        let s = compile_ok(json!([
            {"x": "medium", "y": "rate"},
            {"x": "src", "y": "dst"},
            {"low": "rate", "high": "od600", "y": "medium", "color": "medium"},
            {"x": "medium", "color": "src"},
            {"from": "src", "to": "dst", "type": "pipeline"},
        ]));
        let v = s["views"].as_array().unwrap();
        let types: Vec<&str> = v.iter().map(|v| v["type"].as_str().unwrap()).collect();
        assert_eq!(types, ["strip", "matrix", "interval", "bars", "layers"]);
        // The strip puts the value across and the categories down, whichever way it was asked.
        assert_eq!((&v[0]["x"], &v[0]["y"], &v[0]["title"]), (&json!("rate"), &json!("medium"), &json!("rate by medium")));
        assert_eq!((&v[2]["low"], &v[2]["high"], &v[2]["label"]), (&json!("rate"), &json!("od600"), &json!("medium")));
        assert_eq!((&v[3]["field"], &v[3]["color"]), (&json!("medium"), &json!("src")));
    }

    #[test]
    fn domains_palettes_and_a_key_are_declared_once() {
        let s = compile_ok(json!({
            "key": "WELL",
            "domains": {"Medium": ["rich", "minimal", "none"]},
            "palettes": {"src": {"a": "teal"}, "medium": ["gold", "navy", "grey"]},
            "charts": [{"x": "medium"}, {"x": "rate", "y": "od600", "color": "medium", "palette": ["red"]}]
        }));
        assert_eq!(s["key"], "well");
        assert_eq!(s["fields"]["medium"]["domain"], json!(["rich", "minimal", "none"]));
        assert_eq!(s["fields"]["medium"]["palette"], json!(["gold", "navy", "grey"]), "a declared palette is not replaced by a chart's");
        assert_eq!(s["fields"]["src"]["palette"], json!({"a": "teal"}));
        let c = compile(&json!({"palettes": {"medium": ["teal", "nope"]}, "x": "medium"}), Some(&cols()), None);
        assert_eq!(c.problems, vec![Problem::new("/palettes/medium/1", "a colour: \"#rrggbb\", \"#rgb\" or a CSS colour name such as \"steelblue\"")]);
    }

    #[test]
    fn one_chart_can_carry_the_description_settings() {
        let s = compile_ok(json!({"data": "plate.csv", "theme": "paper", "title": "Rates", "x": "rate"}));
        assert_eq!(s["data"], json!({"url": "plate.csv"}));
        assert_eq!(s["theme"], "paper");
        assert_eq!(s["views"][0]["title"], "Rates");
    }

    #[test]
    fn large_tables_get_densities() {
        let c = compile(&json!({"x": "rate", "y": "od600"}), Some(&cols()), Some(1_000_000));
        assert_eq!(c.spec.unwrap()["views"][0]["type"], "density");
    }

    #[test]
    fn problems_point_at_the_description() {
        let c = compile(&json!({"charts": [{"x-axis": "tim", "y": "od600"}, {"x": "medium", "color": "rate"}, {"x": "rate", "bins": 0, "colr": "red"}]}), Some(&cols()), None);
        let got: Vec<(String, String)> = c.problems.iter().map(|p| (p.path.clone(), p.message.clone())).collect();
        assert_eq!(got[0], ("/charts/0/x-axis".into(), "no column \"tim\"; did you mean \"Time (h)\"?".into()));
        assert_eq!(got[1], ("/charts/1/color".into(), "bars split by a column need a category; \"rate\" is a number".into()));
        assert_eq!(got[2], ("/charts/2/colr".into(), "unknown key \"colr\"; did you mean \"color\"?".into()));
        assert!(c.spec.is_none());

        let c = compile(&json!({"charts": [{"x": "rate", "bins": 0}]}), Some(&cols()), None);
        assert_eq!(c.problems, vec![Problem::new("/charts/0/bins", "an integer in 1..10000")]);
    }

    #[test]
    fn specs_pass_through() {
        let spec = json!({"data": {"url": "x.csv"}, "views": [{"id": "h", "type": "histogram", "field": "rate"}]});
        assert_eq!(compile(&spec, Some(&cols()), None).spec.unwrap(), spec);
    }
}
