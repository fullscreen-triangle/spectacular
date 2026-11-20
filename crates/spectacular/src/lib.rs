//! The native side of spectacular: what a program needs without a browser.
//!
//! - [`describe`] compiles a chart *description* (`{"x-axis": "column A", "color": "group"}`)
//!   into a view spec, resolving column names and choosing chart types from the data.
//! - [`validate`] checks a view spec exactly as `packages/viz/src/spec.ts` does; the
//!   conformance test in `cli/conformance.mjs` holds the two to the same answers.
//! - [`table`] reads a CSV, TSV or JSON data file's columns and their types.
//! - [`page`] writes a standalone HTML page around the browser runtime.

pub mod colors;
pub mod describe;
pub mod page;
pub mod table;
pub mod validate;

use serde_json::{json, Value};

/// A problem with a spec or a description: where it is, and what is wrong.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Problem {
    /// A JSON-Pointer-like path to the offending value, e.g. `/views/2/x`.
    pub path: String,
    pub message: String,
}

impl Problem {
    pub fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Problem { path: path.into(), message: message.into() }
    }

    pub fn to_json(&self) -> Value {
        json!({ "path": self.path, "message": self.message })
    }
}

/// The two field types a view can need.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldType {
    Quantitative,
    Nominal,
}

impl FieldType {
    pub fn as_str(self) -> &'static str {
        match self {
            FieldType::Quantitative => "quantitative",
            FieldType::Nominal => "nominal",
        }
    }

    pub fn parse(s: &str) -> Option<FieldType> {
        match s {
            "quantitative" => Some(FieldType::Quantitative),
            "nominal" => Some(FieldType::Nominal),
            _ => None,
        }
    }
}

/// Column names and types in the data's own order.
pub type Columns = Vec<(String, FieldType)>;

/// The JSON Schema for specs, generated from the TypeScript contract and committed.
pub const SCHEMA: &str = include_str!("../../../spectacular.schema.json");

/// Levenshtein distance, as the TypeScript validator computes it for "did you mean".
pub fn distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut d: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut prev = d[0];
        d[0] = i;
        for j in 1..=b.len() {
            let cur = d[j];
            d[j] = (d[j] + 1).min(d[j - 1] + 1).min(prev + usize::from(a[i - 1] != b[j - 1]));
            prev = cur;
        }
    }
    d[b.len()]
}
