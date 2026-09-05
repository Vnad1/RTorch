// CLI execution tests — run the real rtorch.exe and assert behavior + exit codes.
use std::process::Command;

fn rtorch() -> &'static str {
    env!("CARGO_BIN_EXE_rtorch")
}

fn gpp_available() -> bool {
    if std::env::var_os("RTORCH_GXX").is_some() {
        return true;
    }
    Command::new("where")
        .arg("g++")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
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
    assert_eq!(
        out.status.code(),
        Some(2),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let e = String::from_utf8_lossy(&out.stderr);
    assert!(e.contains("usage: rtorch"), "stderr={e}");
}

#[test]
fn missing_formula_exit_1_no_panic() {
    let out = Command::new(rtorch())
        .arg("definitely_not_here.cpp")
        .arg("--device")
        .arg("cpu")
        .output()
        .unwrap();
    assert_eq!(
        out.status.code(),
        Some(1),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let e = String::from_utf8_lossy(&out.stderr).to_lowercase();
    assert!(e.contains("rtorch:"), "stderr={e}");
    assert!(!e.contains("panicked"), "should not panic: {e}");
}

#[test]
fn formula_smoke_cpu() {
    // Full pipeline: compile the formula with g++, run it on CPU. Skipped if no
    // g++ is available (keeps the suite green without a toolchain). Input is
    // self-generated so the test does not depend on a committed .bin fixture.
    if !gpp_available() {
        return;
    }
    let formula = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/formula_verify.cpp");
    if !std::path::Path::new(formula).exists() {
        return;
    }

    // Generate a small float32 input (1000 values) in a temp file.
    let dir = std::env::temp_dir();
    let input = dir.join(format!("rtorch_cli_test_{}.bin", std::process::id()));
    let vals: Vec<f32> = (0..1000).map(|i| i as f32 * 0.5).collect();
    let mut bytes = Vec::with_capacity(vals.len() * 4);
    for v in &vals {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(&input, &bytes).unwrap();

    let out = Command::new(rtorch())
        .arg(formula)
        .arg("--input")
        .arg(&input)
        .arg("--device")
        .arg("cpu")
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&input);
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "status={:?} stderr={}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
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
    assert!(
        !e.contains("panicked") && !e.contains("access violation"),
        "stderr={e}"
    );
}

// Build a formula .cpp into a pre-built DLL with g++ -shared, then run
// `rtorch.exe <formula.dll> --input ...`, asserting RTorch references an
// already-compiled formula DLL (no on-the-fly compile). Requires g++.
#[test]
fn references_prebuilt_formula_dll() {
    if !gpp_available() {
        return;
    }
    let formula_cpp = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/formula_verify.cpp");
    if !std::path::Path::new(formula_cpp).exists() {
        return;
    }
    // Compile a stable DLL next to the test exe (cwd).
    let dir = std::env::temp_dir();
    let dll = dir.join(format!("rtorch_formula_test_{}.dll", std::process::id()));
    let include_dir = std::path::Path::new(formula_cpp).parent().unwrap();
    let comp = Command::new("g++")
        .arg("-shared")
        .arg("-std=c++17")
        .arg("-O2")
        .arg("-static")
        .arg("-static-libgcc")
        .arg("-static-libstdc++")
        .arg("-I")
        .arg(include_dir)
        .arg(formula_cpp)
        .arg("-o")
        .arg(&dll)
        .output();
    match comp {
        Ok(c) if c.status.success() => {}
        other => {
            let _ = other;
            // g++ may need its bin on PATH for collect2; if it fails, skip rather
            // than fail the whole suite (some machines lack a usable g++).
            return;
        }
    }

    // Small input (5 floats).
    let input = dir.join(format!("rtorch_formula_input_{}.bin", std::process::id()));
    let vals: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let mut bytes = Vec::new();
    for v in &vals {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(&input, &bytes).unwrap();

    // A whitelist allowing the prebuilt DLL (trust gate passes it silently).
    let dll_canon = std::fs::canonicalize(&dll).unwrap_or_else(|_| dll.clone());
    let dll_json = dll_canon.display().to_string().replace('\\', "\\\\");
    let wl_json = dir.join(format!("wl_{}.json", std::process::id()));
    std::fs::write(&wl_json, format!(r#"{{ "allowed": ["{}"] }}"#, dll_json)).unwrap();
    let dll_whitelist = wl_json.to_string_lossy().to_string();

    let out = Command::new(rtorch())
        // The prebuilt formula DLL must be trusted (whitelist) so this test (which
        // deliberately loads a compiled .dll) is allowed through the trust gate.
        .env("RTORCH_WHITELIST", &dll_whitelist)
        .arg(&dll)
        .arg("--input")
        .arg(&input)
        .arg("--device")
        .arg("cpu")
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&dll);
    let _ = std::fs::remove_file(&wl_json);
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "status={:?} stderr={}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(s.contains("[verify] n=5"), "stdout={s} (DLL reference path)");
}

#[test]
fn rtw_whitelist_default_deny_then_allow() {
    // Build a valid kernel `.rtw` from an existing example formula.
    let formula = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/formula_verify.cpp");
    if !std::path::Path::new(formula).exists() {
        return; // example not present; skip
    }
    let dir = std::env::temp_dir().join(format!("rtorch_wl_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let rtw_out = dir.join("whitelist.rtw");
    // Pack it with --pack into an .rtw kernel container.
    let pack = Command::new(rtorch())
        .arg("--pack")
        .arg(formula)
        .arg("-o")
        .arg(&rtw_out)
        .output()
        .unwrap();
    assert!(pack.status.success(), "pack failed: {}", String::from_utf8_lossy(&pack.stderr));
    assert!(rtw_out.exists(), "no .rtw produced");

    // (A) Non-whitelisted, non-interactive: rejected (fail-closed, no hang).
    let deny = Command::new(rtorch())
        .env("RTORCH_TRUST_PROMPT", "0")
        .arg(&rtw_out)
        .arg("--device")
        .arg("cpu")
        .output()
        .unwrap();
    let deny_err = String::from_utf8_lossy(&deny.stderr);
    assert!(
        deny_err.contains("not trusted") || deny_err.contains("trusted"),
        "expected trust rejection, stderr={deny_err}"
    );
    assert_eq!(deny.status.code(), Some(1), "deny should exit 1, stderr={deny_err}");

    // (B) Allow: with RTORCH_WHITELIST listing the exact canonical path, it loads.
    let canon = std::fs::canonicalize(&rtw_out).unwrap();
    // JSON escapes backslashes (`\` -> `\\`).
    let escaped = canon.display().to_string().replace('\\', "\\\\");
    let wl_path = dir.join("whitelist.json");
    std::fs::write(&wl_path, format!(r#"{{ "allowed": ["{}"] }}"#, escaped)).unwrap();

    let allow = Command::new(rtorch())
        .env("RTORCH_WHITELIST", &wl_path)
        .arg(&rtw_out)
        .arg("--device")
        .arg("cpu")
        .output()
        .unwrap();
    let allow_err = String::from_utf8_lossy(&allow.stderr);
    assert!(
        allow.status.success(),
        "allow should succeed, stderr={allow_err}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

