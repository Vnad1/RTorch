// RTorch resource locator — resolves runtime resources (engine DLL, SPIR-V
// kernels) WITHOUT hardcoded dev-machine absolute paths, regardless of the
// current working directory. Search order is exe-relative first, then env
// overrides, then cwd. Compiles into both the lib crate and the rtorch bin
// (it only depends on std).

use std::path::{Path, PathBuf};

/// Directory of the running executable (e.g. target/release).
pub fn exe_dir() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()?
        .parent()
        .map(|p| p.to_path_buf())
}

/// Candidate base dirs for a resource, walking up from the exe dir so that the
/// same layout works when the binary lives in target/release/deps (tests).
fn base_dirs() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(d) = exe_dir() {
        v.push(d.clone());
        if let Some(p) = d.parent() {
            v.push(p.to_path_buf());
        } // target/release
        if let Some(p) = d.parent().and_then(|p| p.parent()) {
            v.push(p.to_path_buf()); // target
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        v.push(cwd);
    }
    v
}

/// Find a kernel (`<name>.spv`) by searching kernel dirs and fallbacks.
pub fn find_kernel(name: &str) -> std::io::Result<PathBuf> {
    if let Ok(p) = std::env::var("RTORCH_KERNELS_DIR") {
        if !p.is_empty() {
            let q = PathBuf::from(&p).join(name);
            if q.exists() {
                return Ok(q);
            }
        }
    }
    let kname = format!("{name}.spv");
    for base in base_dirs() {
        for sub in ["kernels", "", "examples"] {
            let p = if sub.is_empty() {
                base.join(&kname)
            } else {
                base.join(sub).join(&kname)
            };
            if p.exists() {
                return Ok(p);
            }
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!(
            "kernel not found: {name}.spv (build with `cargo build --release`; set RTORCH_KERNELS_DIR to override)"
        ),
    ))
}

/// Read a kernel into bytes (panics-free: returns Err on any failure).
pub fn read_kernel(name: &str) -> std::io::Result<Vec<u8>> {
    std::fs::read(find_kernel(name)?)
}

/// Find the engine DLL (`rtorch_vk.dll`).
pub fn find_dll(name: &str) -> std::io::Result<PathBuf> {
    if let Ok(p) = std::env::var("RTORCH_DLL_DIR") {
        if !p.is_empty() {
            let q = PathBuf::from(&p).join(name);
            if q.exists() {
                return Ok(q);
            }
        }
    }
    for base in base_dirs() {
        let p = base.join(name);
        if p.exists() {
            return Ok(p);
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("{name} not found (build with `cargo build --release`)"),
    ))
}
