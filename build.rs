// RTorch build script — `cargo build --release` is the single formal entry point.
//
// It produces the runtime artifacts that used to be committed as binaries:
//   * <target>/<profile>/rtorch_vk.dll   — C++ Vulkan engine (src/vk_engine.cpp)
//   * <target>/<profile>/kernels/*.spv   — compiled GLSL compute kernels (examples/*.comp)
//
// External toolchain (auto-discovered, with clear errors):
//   * MinGW g++                    — required to build the engine DLL
//   * Vulkan SDK (vulkan-1.lib)    — required to link the engine DLL
//   * glslang (glslangValidator/glslang) — required to compile kernels .comp -> .spv
//
// Env controls:
//   * RTORCH_BUILD_ENGINE=0  skip engine + kernels entirely (CPU-only cargo build)
//   * RTORCH_BUILD_ENGINE=1  require toolchain; fail with a clear error if missing
//   * RTORCH_GXX=<path>      explicit g++ (else auto-discovered)
//   * VULKAN_SDK=<path>      explicit SDK root (else auto-discovered)
//   * RTORCH_GLSLANG=<path>  explicit glslang (else auto-discovered)
//
// Missing optional toolchain (glslang) is a warning, not a hard failure, so a
// CPU-only build still succeeds; the enclosing runtime reports the GPU issue.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/vk_engine.cpp");
    println!("cargo:rerun-if-changed=rtorch.h");
    println!("cargo:rerun-if-changed=Cargo.toml");

    let mode = env::var("RTORCH_BUILD_ENGINE").unwrap_or_default();
    if mode == "0" {
        println!("cargo:warning=RTORCH_BUILD_ENGINE=0: skipping engine + kernels (CPU-only build)");
        return;
    }

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let profile = profile_dir();
    let out_dll = profile.join("rtorch_vk.dll");

    let gxx = find_gxx();
    let sdk = find_vulkan_sdk();

    let can_build = gxx.is_some() && sdk.is_some();
    if !can_build {
        let msg = format!(
            "RTorch engine build needs MinGW g++ and a Vulkan SDK. \
             g++={} VULKAN_SDK={}",
            gxx.as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "NOT FOUND".into()),
            sdk.as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "NOT FOUND".into()),
        );
        if mode == "1" {
            panic!(
                "{msg}\n  Set RTORCH_BUILD_ENGINE=0 to skip the engine, or install g++ / Vulkan SDK."
            );
        }
        println!("cargo:warning={msg}");
        println!(
            "cargo:warning=Skipping engine DLL (CPU-only build). Set VULKAN_SDK / RTORCH_GXX to enable."
        );
        return;
    }

    let gxx = gxx.unwrap();
    let sdk = sdk.unwrap();
    let lib = sdk.join("Lib").join("vulkan-1.lib");
    let inc = sdk.join("Include");
    if !lib.exists() {
        println!("cargo:warning=Vulkan SDK missing {}", lib.display());
        if mode == "1" {
            panic!("Vulkan SDK missing vulkan-1.lib at {}", lib.display());
        }
        return;
    }

    // Build the engine DLL. g++ spawns internal subprocesses (collect2/ld), so
    // its own bin dir must be on PATH or it silently exits 1.
    let mut path = env::var("PATH").unwrap_or_default();
    if let Some(bin) = gxx.parent().and_then(|p| p.to_str()) {
        if !path.split(';').any(|p| p.eq_ignore_ascii_case(bin)) {
            path = format!("{};{}", bin, path);
        }
    }
    let st = Command::new(&gxx)
        .env("PATH", &path)
        .arg("-std=c++17")
        .arg("-shared")
        .arg("-O2")
        // Static-link the whole MinGW runtime (incl. libwinpthread) so the DLL
        // has NO dependency on the compiler's bin dir at load time — portable
        // across machines regardless of where g++ lives.
        .arg("-static")
        .arg("-static-libgcc")
        .arg("-static-libstdc++")
        .arg("-I")
        .arg(&inc)
        .arg(manifest.join("src").join("vk_engine.cpp"))
        .arg(&lib)
        .arg("-o")
        .arg(&out_dll)
        .status();
    match st {
        Ok(s) if s.success() => {
            println!("[build] rtorch_vk.dll -> {}", out_dll.display());
        }
        other => {
            let msg = format!(
                "compiling rtorch_vk.dll failed (g++ {other:?}): {} -> {}",
                gxx.display(),
                out_dll.display()
            );
            if mode == "1" {
                panic!("{msg}");
            }
            println!("cargo:warning={msg}");
            return;
        }
    }

    // Compile GLSL kernels .comp -> .spv into <profile>/kernels/.
    let glslang = find_glslang();
    let kernels_dir = profile.join("kernels");
    if let Some(gl) = glslang {
        let _ = fs::create_dir_all(&kernels_dir);
        let comps = list_comps(&manifest);
        for comp in comps {
            let spv = kernels_dir.join(
                comp.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .replace(".comp", ".spv"),
            );
            println!("cargo:rerun-if-changed={}", comp.display());
            // Only recompile when the source is newer than the cached .spv.
            let needs = !spv.exists() || mtime(&comp) > mtime(&spv);
            if !needs {
                continue;
            }
            let st = Command::new(&gl)
                .arg("-V")
                .arg(&comp)
                .arg("-o")
                .arg(&spv)
                .status();
            if let Ok(s) = st {
                if s.success() {
                    println!(
                        "[build] kernel {} -> {}",
                        comp.file_name().unwrap().to_string_lossy(),
                        spv.display()
                    );
                } else {
                    println!(
                        "cargo:warning=glslang failed on {} (skipped); GPU op will error at runtime",
                        comp.display()
                    );
                }
            }
        }
    } else {
        println!(
            "cargo:warning=glslang not found; kernels NOT compiled (GPU will error at runtime). Set RTORCH_GLSLANG."
        );
        if mode == "1" {
            // Kernels are required in strict mode; but engine DLL already built.
            println!(
                "cargo:warning=RTORCH_BUILD_ENGINE=1 but glslang missing; install glslang or unset to allow CPU-only."
            );
        }
    }
}

// profile dir = <target>/<profile> derived from OUT_DIR
// OUT_DIR = <target>/<profile>/build/<pkg>-<hash>/out  ->  up 3 parents
fn profile_dir() -> PathBuf {
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    out.parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| out.clone())
}

fn mtime(p: &Path) -> u64 {
    fs::metadata(p)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn list_comps(manifest: &Path) -> Vec<PathBuf> {
    let dir = manifest.join("examples");
    let mut out = Vec::new();
    if let Ok(rd) = fs::read_dir(&dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().map(|s| s == "comp").unwrap_or(false) {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

fn find_gxx() -> Option<PathBuf> {
    if let Ok(p) = env::var("RTORCH_GXX") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
        println!(
            "cargo:warning=RTORCH_GXX set but not found: {}",
            p.display()
        );
    }
    // Machine-agnostic discovery: scan PATH for g++.exe. No hardcoded install
    // paths — a machine that keeps g++ off PATH must set RTORCH_GXX (clear error).
    if let Ok(p) = env::var("PATH") {
        for d in p.split(';') {
            if d.is_empty() {
                continue;
            }
            let p = Path::new(d).join("g++.exe");
            if p.exists() {
                return Some(p);
            }
        }
    }
    None
}

fn find_vulkan_sdk() -> Option<PathBuf> {
    if let Ok(s) = env::var("VULKAN_SDK") {
        let p = PathBuf::from(s);
        if p.join("Include").join("vulkan").join("vulkan.h").exists() {
            return Some(p);
        }
    }
    // Search the standard SDK install root.
    let root = PathBuf::from("C:\\VulkanSDK");
    if let Ok(rd) = fs::read_dir(&root) {
        let mut versions: Vec<PathBuf> = Vec::new();
        for e in rd.flatten() {
            let p = e.path();
            let name = p
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            if name.starts_with("1.") && p.join("Include").join("vulkan").join("vulkan.h").exists()
            {
                versions.push(p);
            }
        }
        versions.sort();
        if let Some(v) = versions.pop() {
            return Some(v);
        }
    }
    None
}

fn find_glslang() -> Option<PathBuf> {
    if let Ok(p) = env::var("RTORCH_GLSLANG") {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    // Candidate lists: explicit SDK env, known roots, then newest SDK.
    let mut q: Vec<PathBuf> = Vec::new();
    if let Ok(s) = env::var("VULKAN_SDK") {
        let root = PathBuf::from(s);
        q.push(root.join("Bin").join("glslangValidator.exe"));
        q.push(root.join("Bin").join("glslang.exe"));
    }
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_default());
    q.push(manifest.join("tools").join("bin").join("glslang.exe"));
    q.push(
        manifest
            .join("tools")
            .join("bin")
            .join("glslangValidator.exe"),
    );
    // newest SDK root (so we prefer a freshly installed SDK regardless of VULKAN_SDK)
    let root = PathBuf::from("C:\\VulkanSDK");
    let mut vers: Vec<PathBuf> = Vec::new();
    if let Ok(rd) = fs::read_dir(&root) {
        for e in rd.flatten() {
            let p = e.path();
            if p.file_name()
                .map(|n| n.to_string_lossy().starts_with('1'))
                .unwrap_or(false)
            {
                vers.push(p);
            }
        }
    }
    vers.sort();
    for v in vers.into_iter().rev() {
        q.push(v.join("Bin").join("glslangValidator.exe"));
        q.push(v.join("Bin").join("glslang.exe"));
    }
    q.into_iter().find(|p| p.exists())
}
