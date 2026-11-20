//! A data file's columns and their types, read the way the browser reads them
//! (`loadTable` in `packages/viz/src/data.ts`): CSV and TSV as d3-dsv parses them, JSON
//! rows (optionally under a JSON Pointer), and a column is quantitative when every
//! non-empty value is a finite number.

use std::path::Path;

use serde_json::Value;

use crate::{Columns, FieldType};

/// The columns a table has, with inferred types, and how many rows it holds.
#[derive(Debug, Clone)]
pub struct Table {
    pub columns: Columns,
    pub rows: usize,
    /// True when the columns come from a header (CSV, TSV), so they are known even
    /// without rows. JSON rows name their fields only in their first row.
    pub header: bool,
}

/// `Number(s)` in JavaScript, for the strings a CSV holds: trimmed, decimal or
/// 0x/0o/0b integers; anything else is not a number.
fn js_number(s: &str) -> Option<f64> {
    let t = s.trim();
    if t.is_empty() {
        return Some(0.0);
    }
    for (prefix, radix) in [("0x", 16), ("0X", 16), ("0o", 8), ("0O", 8), ("0b", 2), ("0B", 2)] {
        if let Some(rest) = t.strip_prefix(prefix) {
            return u64::from_str_radix(rest, radix).ok().map(|n| n as f64);
        }
    }
    // Rust also reads "inf" and "nan", which JavaScript does not; neither is finite, and
    // only finite numbers count below, so both sides agree.
    if t.chars().any(|c| c.is_ascii_alphabetic() && !matches!(c, 'e' | 'E')) {
        return None;
    }
    t.parse::<f64>().ok()
}

/// A text cell counts as a number when it is not blank and parses to a finite number.
fn numeric_text(s: &str) -> bool {
    !s.trim().is_empty() && js_number(s).is_some_and(f64::is_finite)
}

fn infer<'a>(values: impl Iterator<Item = Option<&'a Value>>) -> FieldType {
    for v in values {
        match v {
            None | Some(Value::Null) => continue,
            Some(Value::String(s)) if s.is_empty() => continue,
            Some(Value::Number(_)) => continue,
            Some(Value::String(s)) if numeric_text(s) => continue,
            _ => return FieldType::Nominal,
        }
    }
    FieldType::Quantitative
}

/// Parses delimiter-separated text as d3-dsv does: quotes around fields, "" for a quote
/// inside them, and \n or \r\n between records. The first record is the header.
pub fn parse_dsv(text: &str, sep: char) -> Vec<Vec<String>> {
    let mut records = Vec::new();
    let mut record = Vec::new();
    let mut field = String::new();
    let mut chars = text.chars().peekable();
    let mut quoted = false;
    let mut at_start = true;
    while let Some(c) = chars.next() {
        if quoted {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    quoted = false;
                }
            } else {
                field.push(c);
            }
            continue;
        }
        match c {
            '"' if at_start => quoted = true,
            c if c == sep => {
                record.push(std::mem::take(&mut field));
                at_start = true;
                continue;
            }
            '\r' if chars.peek() == Some(&'\n') => {}
            '\n' | '\r' => {
                record.push(std::mem::take(&mut field));
                records.push(std::mem::take(&mut record));
                at_start = true;
                continue;
            }
            c => field.push(c),
        }
        at_start = false;
    }
    if !field.is_empty() || !record.is_empty() {
        record.push(field);
        records.push(record);
    }
    records
}

fn from_dsv(text: &str, sep: char) -> Table {
    let mut records = parse_dsv(text, sep).into_iter();
    let header = records.next().unwrap_or_default();
    let rows: Vec<Vec<String>> = records.collect();
    let columns = header
        .iter()
        .enumerate()
        .map(|(k, name)| {
            let values: Vec<Value> = rows.iter().map(|r| r.get(k).cloned().map(Value::String).unwrap_or(Value::Null)).collect();
            (name.clone(), infer(values.iter().map(Some)))
        })
        .collect();
    Table { columns, rows: rows.len(), header: true }
}

/// Follows an RFC 6901 JSON Pointer.
fn pointer<'a>(doc: &'a Value, pointer: &str) -> Result<&'a Value, String> {
    let mut at = doc;
    if pointer.is_empty() {
        return Ok(at);
    }
    for raw in pointer[1..].split('/') {
        let key = raw.replace("~1", "/").replace("~0", "~");
        let next = match at {
            Value::Object(m) => m.get(&key),
            Value::Array(a) => key.parse::<usize>().ok().and_then(|i| a.get(i)),
            _ => None,
        };
        at = next.ok_or_else(|| format!("data.pointer {pointer:?}: nothing at \"{key}\""))?;
    }
    Ok(at)
}

/// Rows from JSON. As in the browser, the fields a table has are every property any row
/// has, in order of first appearance; a row without one holds a missing value.
fn from_json(doc: &Value, at: &str) -> Result<Table, String> {
    let rows = pointer(doc, at)?.as_array().ok_or_else(|| {
        let shown = if at.is_empty() { "(document)" } else { at };
        format!("data.pointer {shown:?} must lead to an array of row objects")
    })?;
    let mut names: Vec<String> = Vec::new();
    for row in rows.iter().filter_map(Value::as_object) {
        for k in row.keys() {
            if !names.contains(k) {
                names.push(k.clone());
            }
        }
    }
    let columns = names
        .into_iter()
        .map(|name| {
            let t = infer(rows.iter().map(|r| r.get(&name)));
            (name, t)
        })
        .collect();
    Ok(Table { columns, rows: rows.len(), header: false })
}

/// The format a URL implies when the spec does not say: .json/.jsonld, .tsv, else CSV.
pub fn format_of(url: &str) -> &'static str {
    let path = url.split(['?', '#']).next().unwrap_or(url).to_ascii_lowercase();
    if path.ends_with(".json") || path.ends_with(".jsonld") {
        "json"
    } else if path.ends_with(".tsv") {
        "tsv"
    } else {
        "csv"
    }
}

/// Reads `file` as a spec's `data` describes it (`format`, `pointer`).
pub fn read(file: &Path, url: &str, format: Option<&str>, at: Option<&str>) -> Result<Table, String> {
    let text = std::fs::read_to_string(file).map_err(|e| format!("cannot read {}: {e}", file.display()))?;
    match format.unwrap_or_else(|| format_of(url)) {
        "json" => {
            let doc: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", file.display()))?;
            from_json(&doc, at.unwrap_or(""))
        }
        "tsv" => Ok(from_dsv(&text, '\t')),
        _ => Ok(from_dsv(&text, ',')),
    }
}

/// Columns of inline `values` rows.
pub fn from_values(rows: &Value) -> Option<Table> {
    rows.is_array().then(|| from_json(rows, "").ok()).flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsv_follows_d3() {
        let r = parse_dsv("a,b\r\n\"x, y\",\"say \"\"hi\"\"\"\n1,\n", ',');
        assert_eq!(r, vec![vec!["a", "b"], vec!["x, y", "say \"hi\""], vec!["1", ""]]);
    }

    #[test]
    fn numbers_as_javascript_reads_them() {
        for (s, n) in [(" 12 ", true), ("1e3", true), ("0x1F", true), ("-.5", true), ("inf", false), ("NaN", false), ("12a", false), ("1,5", false)] {
            assert_eq!(numeric_text(s), n, "{s}");
        }
    }

    #[test]
    fn types_and_header_only_files() {
        let t = from_dsv("t,lobe,empty\n1,left,\n2.5,right,\n", ',');
        assert_eq!(t.rows, 2);
        assert_eq!(t.columns, vec![("t".into(), FieldType::Quantitative), ("lobe".into(), FieldType::Nominal), ("empty".into(), FieldType::Quantitative)]);
        let h = from_dsv("a,b\n", ',');
        assert_eq!((h.rows, h.columns.len()), (0, 2));
    }
}
