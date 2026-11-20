//! Standalone pages: one HTML file holding the browser runtime, a spec and its data,
//! the same page `spectacular.to_html` writes from Python.

use serde_json::{json, Value};

/// The browser runtime, embedded at build time; empty when it was not built first.
pub const RUNTIME: &str = include_str!(concat!(env!("OUT_DIR"), "/runtime.js"));

fn base64(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = (u32::from(chunk[0]) << 16) | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8) | u32::from(*chunk.get(2).unwrap_or(&0));
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(T[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// The page for `spec`; `file` is (data.url, bytes) when the data is a local file.
pub fn html(spec: &Value, file: Option<(&str, &[u8])>) -> Result<String, String> {
    if RUNTIME.is_empty() {
        return Err("this build of spectacular has no browser runtime; build it with `npm run build:python` first".into());
    }
    let file = file.map(|(name, bytes)| json!({ "name": name, "bytes": { "$b64": base64(bytes) } }));
    let payload = json!({ "spec": spec, "columns": null, "file": file });
    let title = spec.get("title").and_then(Value::as_str).unwrap_or("spectacular");
    // "</" cannot appear inside a <script> element; both escapes keep the meaning.
    let runtime = RUNTIME.replace("</script", "<\\/script");
    let data = payload.to_string().replace("</", "<\\/");
    Ok(format!(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<title>{}</title>\n\
         <style>:root {{ color-scheme: light dark; }} body {{ margin: 0; padding: 8px; background: Canvas; }}</style>\n\
         </head>\n<body data-state=\"loading\">\n<div id=\"spectacular\"></div>\n\
         <script type=\"application/json\" id=\"spectacular-payload\">{data}</script>\n\
         <script type=\"module\">\n{runtime}\n\
         globalThis.spectacularNotebook.boot(document.getElementById(\"spectacular\"),\n  \
         JSON.parse(document.getElementById(\"spectacular-payload\").textContent));\n\
         </script>\n</body>\n</html>\n",
        escape_html(title)
    ))
}

#[cfg(test)]
mod tests {
    #[test]
    fn base64_matches_the_standard() {
        assert_eq!(super::base64(b"hello world!?"), "aGVsbG8gd29ybGQhPw==");
        assert_eq!(super::base64(b"ab"), "YWI=");
    }
}
