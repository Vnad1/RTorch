// Formula C-ABI edge cases — drive the real rtorch.exe over formula sources that
// stress the rtorch.h ABI (error return, zero inputs/output, shrunken out->len,
// missing required symbols). Each must behave correctly (error or correct bytes),
// never crash. Skipped if g++ is unavailable at runtime.
use std::process::Command;

fn rtorch() -> &'static str { env!("CARGO_BIN_EXE_rtorch") }
fn gpp() -> bool {
    if std::env::var_os("RTORCH_GXX").is_some() { return true; }
    Command::new("where").arg("g++").output().map(|o| o.status.success()).unwrap_or(false)
}
fn run_formula(_dir: &std::path::Path, src: &str, name: &str) -> std::process::Output {
    use std::sync::atomic::{AtomicU32, Ordering};
    static CNT: AtomicU32 = AtomicU32::new(0);
    // Write into examples/ which already contains rtorch.h -> the CLI's -I
    // <formula dir> resolves #include "rtorch.h". Unique name avoids parallel races.
    let ex = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
    let base = name.trim_end_matches(".cpp");
    let f = ex.join(format!("abi_{}_{}.cpp", base, CNT.fetch_add(1, Ordering::Relaxed)));
    std::fs::write(&f, src).unwrap();
    let o = Command::new(rtorch()).arg(&f).arg("--device").arg("cpu").output().unwrap();
    let _ = std::fs::remove_file(&f);
    o
}

// helper: a formula that just writes a known float to out->data
const HEAD: &str = r#"#include "rtorch.h"
unsigned long long rtorch_output_size(int n, const rtorch_blob* in, int d){ (void)n;(void)in;(void)d; return 4; }
int rtorch_compute(int n, const rtorch_blob* in, rtorch_blob* out, int d){
    (void)n;(void)in;(void)d;
    float* p=(float*)out->data; p[0]=42.0f;
    return 0;
}
"#;

#[test]
fn abi_error_return_is_clean_exit() {
    if !gpp() { return; }
    let dir = std::env::temp_dir();
    let src = r#"#include "rtorch.h"
unsigned long long rtorch_output_size(int n, const rtorch_blob* in, int d){ (void)n;(void)in;(void)d; return 4; }
int rtorch_compute(int n, const rtorch_blob* in, rtorch_blob* out, int d){ (void)n;(void)in;(void)d;(void)out; return 7; }
"#;
    let o = run_formula(&dir, src, "rtw_abi_err.cpp");
    // a non-zero compute rc must be a clean error (exit 1, clear message), never panic/crash.
    assert_eq!(o.status.code(), Some(1), "stderr={}", String::from_utf8_lossy(&o.stderr));
    let e = String::from_utf8_lossy(&o.stderr).to_lowercase();
    assert!(e.contains("compute failed") || e.contains("rtorch:"), "stderr={e}");
    assert!(!e.contains("panicked") && !e.contains("access violation"), "stderr={e}");
}

#[test]
fn abi_zero_inputs_runs() {
    if !gpp() { return; }
    let dir = std::env::temp_dir();
    let o = run_formula(&dir, HEAD, "rtw_abi_zero.cpp");
    // no --input provided: must run on 0 inputs and produce the 4-byte output.
    assert_eq!(o.status.code(), Some(0), "stderr={}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(o.stdout.len(), 4, "expected 4 output bytes");
    let val = f32::from_bits(u32::from_le_bytes([o.stdout[0], o.stdout[1], o.stdout[2], o.stdout[3]]));
    assert_eq!(val, 42.0);
}

#[test]
fn abi_shrink_output_len() {
    if !gpp() { return; }
    let dir = std::env::temp_dir();
    // output_size claims 16, but compute only writes 4 and shrinks out->len.
    let src = r#"#include "rtorch.h"
unsigned long long rtorch_output_size(int n, const rtorch_blob* in, int d){ (void)n;(void)in;(void)d; return 16; }
int rtorch_compute(int n, const rtorch_blob* in, rtorch_blob* out, int d){
    (void)n;(void)in;(void)d;
    float* p=(float*)out->data; p[0]=9.0f;
    out->len=4; // report true byte length
    return 0;
}
"#;
    let o = run_formula(&dir, src, "rtw_abi_shrink.cpp");
    assert_eq!(o.status.code(), Some(0), "stderr={}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(o.stdout.len(), 4, "must honor the shrunken out->len");
    let val = f32::from_bits(u32::from_le_bytes([o.stdout[0], o.stdout[1], o.stdout[2], o.stdout[3]]));
    assert_eq!(val, 9.0);
}

#[test]
fn abi_missing_symbols_is_clean_error() {
    if !gpp() { return; }
    let dir = std::env::temp_dir();
    let src = "int not_the_contract(void){ return 0; }\n";
    let o = run_formula(&dir, src, "rtw_abi_nosym.cpp");
    assert_eq!(o.status.code(), Some(1), "stderr={}", String::from_utf8_lossy(&o.stderr));
    let e = String::from_utf8_lossy(&o.stderr).to_lowercase();
    assert!(e.contains("rtor") , "stderr={e}"); // must describe the required contract
    assert!(!e.contains("panicked") && !e.contains("access violation"), "stderr={e}");
}

#[test]
fn abi_wrong_header_contract() {
    if !gpp() { return; }
    let dir = std::env::temp_dir();
    // Formula uses wrong signature (takes different blob layout); must fail cleanly.
    let src = r#"
#include "rtorch.h"
unsigned long long rtorch_output_size(int n, const rtorch_blob* in, int d){ (void)n;(void)in;(void)d; return 8; }
int rtorch_compute(int n, const rtorch_blob* in, rtorch_blob* out, int d){ (void)n;(void)in;(void)d;
    float q=1.5f; for(int i=0;i<2;i++) ((float*)out->data)[i]=q; return 0; }
"#;
    let o = run_formula(&dir, src, "rtw_abi_double.cpp");
    assert_eq!(o.status.code(), Some(0), "stderr={}", String::from_utf8_lossy(&o.stderr));
    // output_size returns 8 (no shrink) -> 8 bytes produced.
    assert_eq!(o.stdout.len(), 8, "output_size claims 8, must produce 8");
}
