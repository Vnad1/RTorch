// CLI execution tests — run the real rtorch.exe and assert behavior + exit codes.
use std::process::Command;

fn rtorch() -> &'static str { env!("CARGO_BIN_EXE_rtorch") }

fn gpp_available() -> bool {
    if std::env::var_os("RTORCH_GXX").is_some() { return true; }
    Command::new("where").arg("g++").output().map(|o| o.status.success()).unwrap_or(false)
}

#[test]
fn version_flag() {
    let out = Command::new(rtorch()).arg("--version").output().unwrap();
    assert!(out.status.success(), "status={:?}", out.status);
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.starts_with("rtorch "), "stdout={s}");
}

#[test]
fn help_exits_zero() {
    let out = Command::new(rtorch()).arg("--help").output().unwrap();
    assert!(out.status.success());
    // usage() prints to stderr.
    let s = String::from_utf8_lossy(&out.stderr);
    assert!(s.contains("usage: rtorch"), "stderr={s}");
}

#[test]
fn no_args_is_usage_exit_2() {
    let out = Command::new(rtorch()).output().unwrap();
    assert_eq!(out.status.code(), Some(2), "stderr={}", String::from_utf8_lossy(&out.stderr));
    let e = String::from_utf8_lossy(&out.stderr);
    assert!(e.contains("usage: rtorch"), "stderr={e}");
}

#[test]
fn missing_formula_exit_1_no_panic() {
    let out = Command::new(rtorch()).arg("definitely_not_here.cpp").arg("--device").arg("cpu").output().unwrap();
    assert_eq!(out.status.code(), Some(1), "stderr={}", String::from_utf8_lossy(&out.stderr));
    let e = String::from_utf8_lossy(&out.stderr).to_lowercase();
    assert!(e.contains("rtorch:"), "stderr={e}");
    assert!(!e.contains("panicked"), "should not panic: {e}");
}

#[test]
fn formula_smoke_cpu() {
    // Full pipeline: compile the formula with g++, run it on CPU. Skipped if no
    // g++ is available (keeps the suite green without a toolchain). Input is
    // self-generated so the test does not depend on a committed .bin fixture.
    if !gpp_available() { return; }
    let formula = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/formula_verify.cpp");
    if !std::path::Path::new(formula).exists() { return; }

    // Generate a small float32 input (1000 values) in a temp file.
    let dir = std::env::temp_dir();
    let input = dir.join(format!("rtorch_cli_test_{}.bin", std::process::id()));
    let vals: Vec<f32> = (0..1000).map(|i| i as f32 * 0.5).collect();
    let mut bytes = Vec::with_capacity(vals.len() * 4);
    for v in &vals { bytes.extend_from_slice(&v.to_le_bytes()); }
    std::fs::write(&input, &bytes).unwrap();

    let out = Command::new(rtorch())
        .arg(formula).arg("--input").arg(&input).arg("--device").arg("cpu")
        .output().unwrap();
    let _ = std::fs::remove_file(&input);
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "status={:?} stderr={}", out.status, String::from_utf8_lossy(&out.stderr));
    assert!(s.contains("[verify] n=1000"), "stdout={s}");
}

#[test]
fn gpu_unavailable_reports_not_crash() {
    // If the engine DLL or GPU is unavailable, the GPU path must fail cleanly
    // (non-zero, no panic/access-violation), not crash.
    let out = Command::new(rtorch()).arg("--vk-smoke").output().unwrap();
    let e = String::from_utf8_lossy(&out.stderr).to_lowercase();
    // Either it PASSes (GPU present) or it fails cleanly with a message.
    assert!(out.status.code().is_some(), "should exit, not crash");
    assert!(!e.contains("panicked") && !e.contains("access violation"), "stderr={e}");
}
