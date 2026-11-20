//! `spectacular`: chart descriptions in, view specs, problems and pages out.
//!
//! The Python package ships this binary and calls it to compile descriptions; programs
//! in any language can call it the same way. `validate` and `schema` keep the contract of
//! the Node command in `cli/spectacular.mjs`.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use serde_json::{json, Map, Value};
use spectacular::describe::compile;
use spectacular::table::{self, Table};
use spectacular::validate::{needed_fields, validate};
use spectacular::{page, Columns, FieldType, Problem, SCHEMA};

const USAGE: &str = "usage: spectacular <command> [options]

  compile <description.json|-> [--data FILE]   a chart description -> a view spec
  validate <spec.json|-> [...]                 check view specs, and their fields against local data
  html <description|spec> [--data FILE] -o OUT.html
                                               one offline page with the linked views
  schema                                       the JSON Schema for view specs

  --json      machine-readable output: one JSON object per line on stdout
  \"-\"         read from stdin; relative paths then resolve against the working directory

compile --envelope reads {\"description\", \"columns\": {name: type}, \"rows\", \"base\"} instead,
which is how the Python package describes a DataFrame it holds.

exit codes: 0 ok, 1 the description, spec or data has problems, 2 usage or I/O error";

struct Fail(u8, String);

type Res<T> = Result<T, Fail>;

fn io(message: impl Into<String>) -> Fail {
    Fail(2, message.into())
}

struct Args {
    command: String,
    positionals: Vec<String>,
    json: bool,
    envelope: bool,
    data: Option<String>,
    out: Option<String>,
}

fn parse(argv: Vec<String>) -> Res<Args> {
    let mut it = argv.into_iter();
    let command = it.next().ok_or_else(|| io(USAGE))?;
    let mut a = Args { command, positionals: vec![], json: false, envelope: false, data: None, out: None };
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--json" => a.json = true,
            "--envelope" => a.envelope = true,
            "--data" => a.data = Some(it.next().ok_or_else(|| io("--data needs a file"))?),
            "-o" | "--out" => a.out = Some(it.next().ok_or_else(|| io("-o needs a file"))?),
            s if s.starts_with("--") => return Err(io(format!("unknown option {s}\n\n{USAGE}"))),
            _ => a.positionals.push(arg),
        }
    }
    Ok(a)
}

/// Reads JSON from a file or "-", with the directory relative paths resolve against.
fn read_json(arg: &str) -> Res<(Value, PathBuf)> {
    let (text, base) = if arg == "-" {
        let mut s = String::new();
        std::io::stdin().read_to_string(&mut s).map_err(|e| io(format!("stdin: {e}")))?;
        (s, std::env::current_dir().unwrap_or_default())
    } else {
        let path = Path::new(arg);
        let text = std::fs::read_to_string(path).map_err(|e| io(format!("{arg}: {e}")))?;
        // absolute, not canonicalize: no \?\ prefix on Windows, as Node prints paths.
        let base = std::path::absolute(path).ok().and_then(|p| p.parent().map(Path::to_path_buf)).unwrap_or_default();
        (text, base)
    };
    let value = serde_json::from_str(&text).map_err(|e| io(format!("{arg}: {e}")))?;
    Ok((value, base))
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(b) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(b);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The local file a spec's data.url names, as the Node command decides it: anything
/// with a scheme ("https:", or even "C:") is remote.
fn local_data(spec: &Value, base: &Path) -> Option<PathBuf> {
    let url = spec.get("data")?.get("url")?.as_str()?;
    let scheme = url.find(':').is_some_and(|i| {
        let s = &url[..i];
        s.chars().next().is_some_and(|c| c.is_ascii_alphabetic()) && s.chars().all(|c| c.is_ascii_alphanumeric() || "+.-".contains(c))
    });
    if scheme {
        return None;
    }
    let path = url.split(['?', '#']).next().unwrap_or(url);
    let joined = base.join(percent_decode(path));
    Some(std::path::absolute(&joined).unwrap_or(joined))
}

/// The table a spec's data holds, when it can be read here.
fn spec_table(spec: &Value, base: &Path) -> Result<Option<(Table, Option<PathBuf>)>, Problem> {
    let data = spec.get("data");
    if let Some(file) = local_data(spec, base) {
        if !file.exists() {
            return Err(Problem::new("/data/url", format!("no file at {}", file.display())));
        }
        let d = data.unwrap();
        let t = table::read(&file, d["url"].as_str().unwrap_or(""), d.get("format").and_then(Value::as_str), d.get("pointer").and_then(Value::as_str))
            .map_err(|m| Problem::new("/data", m))?;
        return Ok(Some((t, Some(file))));
    }
    Ok(data.and_then(|d| d.get("values")).and_then(table::from_values).map(|t| (t, None)))
}

/// The fields the browser would load for `spec` from `t`, with their types: the needed
/// fields the table has, a declared type winning over the inferred one.
fn loaded(spec: &Value, t: &Table) -> Columns {
    // As the browser does (`declaredType` in spec.ts): any declared type other than
    // "quantitative" loads as nominal, even an invalid one (which the validator reports);
    // with no type, a domain or a palette makes a field nominal, and so does being the key.
    let key = spec.get("key").and_then(Value::as_str);
    let declared = |name: &str| {
        let def = spec.get("fields").and_then(|f| f.get(name));
        if let Some(t) = def.and_then(|d| d.get("type")).filter(|t| !t.is_null()) {
            return Some(if t.as_str() == Some("quantitative") { FieldType::Quantitative } else { FieldType::Nominal });
        }
        let categorical = def.is_some_and(|d| d.get("domain").is_some() || d.get("palette").is_some());
        (categorical || key == Some(name)).then_some(FieldType::Nominal)
    };
    needed_fields(spec)
        .into_iter()
        .filter_map(|name| {
            let inferred = t.columns.iter().find(|(c, _)| *c == name)?.1;
            let ty = declared(&name).unwrap_or(inferred);
            Some((name, ty))
        })
        .collect()
}

fn problems_json(problems: &[Problem]) -> Value {
    Value::Array(problems.iter().map(Problem::to_json).collect())
}

fn fields_json(columns: &Columns) -> Value {
    Value::Object(columns.iter().map(|(n, ty)| (n.clone(), json!(ty.as_str()))).collect())
}

/// `validate`: the Node command's checks and output, from Rust.
fn check(spec: &Value, base: &Path) -> (Vec<Problem>, Option<(usize, Columns)>) {
    let mut problems = validate(spec, None);
    if problems.iter().any(|p| p.path.starts_with("/data") || p.path == "/views" || p.path.is_empty()) {
        return (problems, None);
    }
    match spec_table(spec, base) {
        Err(p) => {
            problems.push(p);
            (problems, None)
        }
        Ok(None) => (problems, None),
        Ok(Some((t, _))) => {
            let columns = loaded(spec, &t);
            let types: HashMap<String, FieldType> = columns.iter().cloned().collect();
            (validate(spec, Some(&types)), Some((t.rows, columns)))
        }
    }
}

fn cmd_validate(a: &Args) -> Res<u8> {
    if a.positionals.is_empty() {
        return Err(io(USAGE));
    }
    let mut worst = 0;
    for arg in &a.positionals {
        let (spec, base) = read_json(arg)?;
        let (problems, table) = check(&spec, &base);
        let ok = problems.is_empty();
        if !ok {
            worst = 1;
        }
        if a.json {
            let mut o = Map::new();
            o.insert("spec".into(), json!(arg));
            o.insert("ok".into(), json!(ok));
            o.insert("problems".into(), problems_json(&problems));
            if let Some((rows, columns)) = &table {
                o.insert("rows".into(), json!(rows));
                o.insert("fields".into(), fields_json(columns));
            }
            println!("{}", Value::Object(o));
        } else if ok {
            let summary = match &table {
                Some((rows, columns)) => format!(
                    "{} rows; {}",
                    rows,
                    columns.iter().map(|(n, ty)| format!("{n}:{}", ty.as_str())).collect::<Vec<_>>().join(", ")
                ),
                None => "structure only".into(),
            };
            println!("{arg}: valid ({summary})");
        } else {
            for p in &problems {
                println!("{arg} {}: {}", if p.path.is_empty() { "/" } else { &p.path }, p.message);
            }
        }
    }
    Ok(worst)
}

/// What `compile` works from: the description, the data's columns, and where files are.
struct Input {
    description: Value,
    columns: Option<Columns>,
    rows: Option<usize>,
    base: PathBuf,
    /// The data file and its URL in the spec, when the data is a local file.
    file: Option<(PathBuf, String)>,
}

fn columns_from(v: &Value) -> Res<Columns> {
    let map = v.as_object().ok_or_else(|| io("columns: an object of column name to \"quantitative\" or \"nominal\""))?;
    map.iter()
        .map(|(n, t)| {
            t.as_str()
                .and_then(FieldType::parse)
                .map(|ty| (n.clone(), ty))
                .ok_or_else(|| io(format!("columns.{n}: \"quantitative\" or \"nominal\"")))
        })
        .collect()
}

fn input(a: &Args) -> Res<Input> {
    let arg = a.positionals.first().ok_or_else(|| io(USAGE))?;
    let (value, base) = read_json(arg)?;
    if a.envelope {
        let base = value.get("base").and_then(Value::as_str).map(PathBuf::from).unwrap_or(base);
        let columns = value.get("columns").filter(|c| !c.is_null()).map(columns_from).transpose()?;
        let rows = value.get("rows").and_then(Value::as_u64).map(|r| r as usize);
        let description = value.get("description").cloned().ok_or_else(|| io("the envelope has no \"description\""))?;
        return finish(Input { description, columns, rows, base, file: None }, a);
    }
    finish(Input { description: value, columns: None, rows: None, base, file: None }, a)
}

/// Fills in the columns from a data file: --data, or the description's own data.
fn finish(mut i: Input, a: &Args) -> Res<Input> {
    if i.columns.is_some() {
        return Ok(i);
    }
    let named = a.data.clone().map(|d| (PathBuf::from(&d), d)).or_else(|| {
        let url = data_url(&i.description)?;
        local_data(&json!({ "data": { "url": url } }), &i.base).map(|p| (p, url))
    });
    if let Some((path, url)) = named {
        let (format, pointer) = data_options(&i.description);
        let t = table::read(&path, &url, format.as_deref(), pointer.as_deref()).map_err(io)?;
        i.columns = Some(t.columns);
        i.rows = Some(t.rows);
        let name = if a.data.is_some() { path.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or(url) } else { url };
        i.file = Some((path, name));
    }
    Ok(i)
}

/// The description's data: a string, or { url } (also under "file" or "csv").
fn data_url(desc: &Value) -> Option<String> {
    let o = desc.as_object()?;
    let d = o.iter().find(|(k, _)| matches!(k.to_lowercase().as_str(), "data" | "file" | "csv")).map(|(_, v)| v)?;
    d.as_str().map(str::to_string).or_else(|| d.get("url").and_then(Value::as_str).map(str::to_string))
}

fn data_options(desc: &Value) -> (Option<String>, Option<String>) {
    let d = desc.get("data");
    let get = |k: &str| d.and_then(|d| d.get(k)).and_then(Value::as_str).map(str::to_string);
    (get("format"), get("pointer"))
}

fn report(problems: &[Problem], json_out: bool, label: &str) {
    if json_out {
        println!("{}", json!({ "ok": false, "problems": problems_json(problems) }));
    } else {
        for p in problems {
            eprintln!("{label} {}: {}", if p.path.is_empty() { "/" } else { &p.path }, p.message);
        }
    }
}

fn cmd_compile(a: &Args) -> Res<u8> {
    let i = input(a)?;
    let c = compile(&i.description, i.columns.as_ref(), i.rows);
    let label = a.positionals[0].as_str();
    match c.spec {
        Some(mut spec) => {
            if let Some((_, name)) = &i.file {
                if a.data.is_some() {
                    spec["data"] = json!({ "url": name });
                }
            }
            if a.json {
                println!("{}", json!({ "ok": true, "spec": spec, "problems": [] }));
            } else {
                println!("{}", serde_json::to_string_pretty(&spec).unwrap());
            }
            Ok(0)
        }
        None => {
            report(&c.problems, a.json, label);
            Ok(1)
        }
    }
}

fn cmd_html(a: &Args) -> Res<u8> {
    let out = a.out.clone().ok_or_else(|| io(format!("html needs -o OUT.html\n\n{USAGE}")))?;
    let i = input(a)?;
    let c = compile(&i.description, i.columns.as_ref(), i.rows);
    let Some(mut spec) = c.spec else {
        report(&c.problems, a.json, &a.positionals[0]);
        return Ok(1);
    };
    let bytes;
    let file = match &i.file {
        Some((path, name)) => {
            bytes = std::fs::read(path).map_err(|e| io(format!("{}: {e}", path.display())))?;
            spec["data"] = json!({ "url": name });
            Some((name.as_str(), bytes.as_slice()))
        }
        None if spec["data"].get("columns").is_some() => {
            return Err(io("the description names no data; give --data FILE or \"data\": \"file.csv\""));
        }
        None => None,
    };
    let html = page::html(&spec, file).map_err(io)?;
    std::fs::write(&out, html).map_err(|e| io(format!("{out}: {e}")))?;
    if a.json {
        println!("{}", json!({ "ok": true, "file": out }));
    } else {
        println!("wrote {out}");
    }
    Ok(0)
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.first().is_some_and(|a| a == "--version" || a == "-V") {
        println!("spectacular {}", env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    }
    if argv.is_empty() || argv[0] == "--help" || argv[0] == "-h" {
        println!("{USAGE}");
        return ExitCode::from(if argv.is_empty() { 2 } else { 0 });
    }
    let json_out = argv.iter().any(|a| a == "--json");
    let result = parse(argv).and_then(|a| match a.command.as_str() {
        "compile" => cmd_compile(&a),
        "validate" => cmd_validate(&a),
        "html" => cmd_html(&a),
        "schema" => {
            print!("{SCHEMA}");
            Ok(0)
        }
        other => Err(io(format!("unknown command \"{other}\"\n\n{USAGE}"))),
    });
    match result {
        Ok(code) => ExitCode::from(code),
        Err(Fail(code, message)) => {
            if json_out {
                println!("{}", json!({ "ok": false, "error": message, "exit": code }));
            } else {
                eprintln!("{message}");
            }
            ExitCode::from(code)
        }
    }
}
