//! Integration test: the compile/load/execute pipeline is exposed as a library
//! API (`rtorch::formula::run`), not just the CLI. This is the P1 requirement.
//!
//! It compiles `examples/formula_verify.cpp` on the fly (needs g++) and runs it
//! through `rtorch::formula::run`, asserting the numeric output.

use std::path::Path;

fn gpp_available() -> bool {
    if std::env::var_os("RTORCH_GXX").is_some() {
        return true;
    }
    std::process::Command::new("where")
        .arg("g++")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[test]
fn formula_run_pipeline_works_as_library() {
    if !gpp_available() {
        return;
    }
    let formula = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/formula_verify.cpp");
    if !Path::new(formula).exists() {
        return;
    }
    // Build a small input (5 float32 values) and run via the library API.
    let vals: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let mut bytes = Vec::with_capacity(vals.len() * 4);
    for v in &vals {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let inputs = vec![bytes];

    let out = rtorch::formula::run(Path::new(formula), &[], &inputs, 0).expect("formula::run should succeed");
    let s = String::from_utf8_lossy(&out);
    assert!(s.contains("[verify] n=5"), "library pipeline output = {s}");
}
