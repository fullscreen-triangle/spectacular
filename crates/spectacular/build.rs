// Embeds the browser runtime (built by `npm run build:python`) so `spectacular html` can
// write standalone pages. Without it the binary still compiles and validates.
use std::path::Path;

fn main() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../python/src/spectacular/static/spectacular.js");
    let out = Path::new(&std::env::var("OUT_DIR").unwrap()).join("runtime.js");
    println!("cargo:rerun-if-changed={}", src.display());
    let runtime = std::fs::read_to_string(&src).unwrap_or_default();
    std::fs::write(out, runtime).unwrap();
}
