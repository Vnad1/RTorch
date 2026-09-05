//! Formula pipeline — compile/load/execute a user formula DLL (the core of the
//! framework's ability to run `rtorch <formula.cpp|.dll>`).
//!
//! This is deliberately independent of the CLI (`main.rs`). The CLI only parses
//! arguments and does I/O; the actual compile/load/dispatch lives in the library
//! here (`rtorch::formula`), so a library consumer can run a formula without
//! going through the `rtorch` binary.
//!
//! Contract (see `rtorch.h`): a formula exports `rtorch_output_size` +
//! `rtorch_compute` (and optionally `rtorch_gpu_kernel` / `rtorch_gpu_groups`,
//! or the legacy `rtorch_main`). Given input blobs, we:
//!   1. compile the `.cpp` source on the fly (or load a pre-built `.dll`),
//!   2. load it with `LoadLibraryW`,
//!   3. resolve the protocol entry points,
//!   4. dispatch on the host (CPU) or via the C++ Vulkan engine (GPU),
//!   5. return the produced byte blob.
//!
//! Errors are reported with `crate::error::RtorchError` so they map to the
//! framework's stable exit codes.

use std::env;
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

#[cfg(windows)]
use std::os::windows::ffi::OsStrExt;

use crate::error::RtorchError;

// ---------------------------------------------------------------------------
// Windows FFI for loading a DLL + resolving symbols.
// ---------------------------------------------------------------------------

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryW(name: *const u16) -> *mut std::ffi::c_void;
    fn GetProcAddress(h: *mut std::ffi::c_void, name: *const u8) -> *mut std::ffi::c_void;
    fn FreeLibrary(h: *mut std::ffi::c_void) -> i32;
}

type RtorchMain = unsafe extern "C" fn(i32, *const *const u8) -> i32;
type OutputSizeFn = unsafe extern "C" fn(i32, *const Blob, i32) -> u64;
type ComputeFn = unsafe extern "C" fn(i32, *const Blob, *mut Blob, i32) -> i32;
type GpuKernelFn = unsafe extern "C" fn() -> *const std::ffi::c_char;
type GpuGroupsFn = unsafe extern "C" fn(*mut i32, *mut i32, *mut i32);

#[repr(C)]
pub struct Blob {
    data: *const std::ffi::c_void,
    len: usize,
}

// ---------------------------------------------------------------------------
// Public API.
// ---------------------------------------------------------------------------

/// Run a formula with the given input blobs and return the produced byte blob.
///
/// `formula` is either a `.cpp` source (compiled on the fly, needs `g++`) or a
/// pre-built `.dll` (loaded directly, no compiler). `refs` are extra source/object
/// files compiled alongside (only used for the `.cpp` path). `device` selects the
/// target: `0` = CPU host, `>0` = best accelerator (Vulkan compute).
pub fn run(
    formula: &Path,
    refs: &[&Path],
    inputs: &[Vec<u8>],
    device: i32,
) -> Result<Vec<u8>, RtorchError> {
    if !formula.exists() {
        return Err(RtorchError::io(format!(
            "formula not found: {}",
            formula.display()
        )));
    }

    // Determine the DLL to load and, when we compiled it, keep a handle on the
    // temp file to clean up afterwards.
    let (load_path, mut out_dll): (PathBuf, Option<PathBuf>) = if is_dll(formula) {
        // Pre-built DLL: resolve to an absolute path before LoadLibraryW (Windows'
        // default search does NOT include the cwd on Win8+, so a bare relative
        // filename like `delta.dll` can fail to load). Absolute path removes that.
        let dll_abs = std::fs::canonicalize(formula).unwrap_or_else(|_| formula.to_path_buf());
        (dll_abs, None)
    } else {
        let compiler = find_compiler().ok_or_else(|| {
            RtorchError::compile("no C++ compiler found (need MinGW g++; set RTORCH_GXX)")
        })?;
        // Ensure compiler runtime DLLs are on PATH before we LoadLibrary later.
        if let Some(dir) = compiler.parent() {
            let mut path = env::var("PATH").unwrap_or_default();
            let dstr = dir.display().to_string();
            if !path.split(';').any(|p| p.eq_ignore_ascii_case(&dstr)) {
                let newpath = format!("{};{}", dstr, path);
                unsafe {
                    let _ = env::set_var("PATH", newpath);
                }
            }
        }
        let dll = compile_formula(&compiler, formula, refs)?;
        (dll.clone(), Some(dll))
    };

    let res = execute_formula(&load_path, inputs, device);
    if let Some(dll) = out_dll.take() {
        let _ = std::fs::remove_file(dll);
    }
    res
}

fn is_dll(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("dll"))
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Compile a formula `.cpp` + refs into a shared DLL on the host.
// ---------------------------------------------------------------------------

pub fn find_compiler() -> Option<PathBuf> {
    if let Ok(p) = env::var("RTORCH_GXX") {
        let p = PathBuf::from(&p);
        if p.exists() {
            return Some(p);
        }
        eprintln!(
            "rtorch: warning: RTORCH_GXX set but not found: {}",
            p.display()
        );
    }
    // Machine-agnostic: `where g++` (no hardcoded install paths in code).
    for name in ["g++.exe", "gcc.exe"] {
        if let Ok(out) = Command::new("where").arg(name).output() {
            if out.status.success() {
                if let Ok(s) = String::from_utf8(out.stdout) {
                    if let Some(line) = s.lines().next() {
                        return Some(PathBuf::from(line));
                    }
                }
            }
        }
    }
    None
}

fn compile_formula(
    compiler: &Path,
    formula: &Path,
    refs: &[&Path],
) -> Result<PathBuf, RtorchError> {
    let out_dll = env::current_dir()
        .ok()
        .unwrap_or_else(|| env::temp_dir())
        .join(format!("rtorch_rt_{}.dll", std::process::id()));
    let _ = std::fs::remove_file(&out_dll);

    let mut cmd = Command::new(compiler);
    cmd.arg("-shared")
        .arg("-fPIC")
        .arg("-O3")
        .arg("-std=c++17")
        .arg("-march=native")
        .arg("-ffast-math")
        .arg("-funroll-loops")
        // Static-link the whole MinGW runtime (incl. libwinpthread) so the
        // formula DLL has NO dependency on the compiler's bin dir at load time.
        .arg("-static")
        .arg("-static-libgcc")
        .arg("-static-libstdc++")
        .arg(formula);
    // include formula dir + ref dirs so `#include "rtorch.h"` resolves.
    if let Some(dir) = formula.parent() {
        cmd.arg("-I").arg(dir);
    }
    for r in refs {
        let rp = Path::new(r);
        if rp.exists() {
            cmd.arg(rp);
            if let Some(dir) = rp.parent() {
                cmd.arg("-I").arg(dir);
            }
        } else {
            eprintln!("rtorch: warning: ref not found, skipping: {}", rp.display());
        }
    }
    cmd.arg("-o").arg(&out_dll);

    let st = cmd.status()?;
    if !st.success() {
        let _ = std::fs::remove_file(&out_dll);
        return Err(RtorchError::compile(
            "compilation of formula failed (see g++ output above)",
        ));
    }
    Ok(out_dll)
}

// ---------------------------------------------------------------------------
// Execute a compiled formula DLL (unified protocol, GPU path, or legacy).
// ---------------------------------------------------------------------------

fn execute_formula(dll: &Path, inputs: &[Vec<u8>], device: i32) -> Result<Vec<u8>, RtorchError> {
    let handle = load_dll(dll)?;

    let out_size = load_symbol::<OutputSizeFn>(handle, "rtorch_output_size");
    let compute = load_symbol::<ComputeFn>(handle, "rtorch_compute");
    let gpu_kernel = load_symbol::<GpuKernelFn>(handle, "rtorch_gpu_kernel");

    // GPU path: if a formula provides a GLSL kernel and we're on the accelerator,
    // compile it to SPIR-V and run it through the C++ Vulkan engine.
    if device > 0 {
        if let Some(gk) = gpu_kernel {
            return execute_gpu(handle, inputs, device, gk, out_size);
        }
    }

    if let (Some(out_size), Some(compute)) = (out_size, compute) {
        let in_blobs: Vec<Blob> = inputs
            .iter()
            .map(|b| Blob {
                data: b.as_ptr() as *const std::ffi::c_void,
                len: b.len(),
            })
            .collect();
        let n = in_blobs.len() as i32;
        let want = unsafe { out_size(n, in_blobs.as_ptr(), device) } as usize;
        let mut out_buf = vec![0u8; want];
        let mut out_blob = Blob {
            data: out_buf.as_ptr() as *const std::ffi::c_void,
            len: out_buf.len(),
        };

        eprintln!("[rtorch] device={device} inputs={n} output={want} bytes");

        let t0 = Instant::now();
        let rc = unsafe { compute(n, in_blobs.as_ptr(), &mut out_blob, device) };
        let elapsed = t0.elapsed();

        eprintln!(
            "[rtorch] compute rc={rc} elapsed={:.3} ms",
            elapsed.as_secs_f64() * 1000.0
        );
        if rc != 0 {
            unsafe { FreeLibrary(handle) };
            return Err(RtorchError::load(format!(
                "formula compute failed rc={rc}"
            )));
        }
        // The formula may set out->len to the true number of bytes written.
        let actual = out_blob.len.min(out_buf.len());
        let _ = device;
        unsafe { FreeLibrary(handle) };
        Ok(out_buf[..actual].to_vec())
    } else if let Some(main) = load_symbol::<RtorchMain>(handle, "rtorch_main") {
        let mut argv_repr: Vec<Vec<u8>> = Vec::new();
        argv_repr.push(CString::new("formula").unwrap().into_bytes_with_nul());
        let mut ptrs: Vec<*const u8> = argv_repr.iter().map(|v| v.as_ptr()).collect();
        ptrs.push(std::ptr::null());
        unsafe { main(ptrs.len() as i32 - 1, ptrs.as_ptr()) };
        unsafe { FreeLibrary(handle) };
        Ok(Vec::new())
    } else {
        unsafe { FreeLibrary(handle) };
        Err(RtorchError::load(
            "formula must implement rtorch_output_size + rtorch_compute (see rtorch.h) or legacy rtorch_main",
        ))
    }
}

fn load_dll(dll: &Path) -> Result<*mut std::ffi::c_void, RtorchError> {
    let wpath: Vec<u16> = dll
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let handle = unsafe { LoadLibraryW(wpath.as_ptr()) };
    if handle.is_null() {
        return Err(RtorchError::io(std::io::Error::last_os_error().to_string()));
    }
    Ok(handle)
}

fn load_symbol<T: Copy>(handle: *mut std::ffi::c_void, name: &str) -> Option<T> {
    let cname = CString::new(name).ok()?;
    let p = unsafe { GetProcAddress(handle, cname.as_ptr() as *const u8) };
    if p.is_null() {
        None
    } else if std::mem::size_of::<T>() != std::mem::size_of::<*mut std::ffi::c_void>() {
        None
    } else {
        Some(unsafe { std::mem::transmute_copy::<*mut std::ffi::c_void, T>(&p) })
    }
}

// ---------------------------------------------------------------------------
// GPU kernel path: compile GLSL -> SPIR-V, run through the Vulkan engine.
// ---------------------------------------------------------------------------

fn compile_glsl_to_spv(glsl: &str) -> std::io::Result<Vec<u8>> {
    let glslang = find_glslang().ok_or_else(|| {
        std::io::Error::other("glslangValidator not found (needed for GPU kernels)")
    })?;
    let dir = env::temp_dir();
    let comp = dir.join(format!("rtorch_k_{}.comp", std::process::id()));
    let spv = dir.join(format!("rtorch_k_{}.spv", std::process::id()));
    std::fs::write(&comp, glsl.as_bytes())?;
    let _ = std::fs::remove_file(&spv);
    let st = Command::new(&glslang)
        .arg("-V")
        .arg(&comp)
        .arg("-o")
        .arg(&spv)
        .status()?;
    if !st.success() {
        let _ = std::fs::remove_file(&comp);
        return Err(std::io::Error::other(
            "glslang compilation of GPU kernel failed (see glslang output)",
        ));
    }
    let bytes = std::fs::read(&spv)?;
    let _ = std::fs::remove_file(&comp);
    let _ = std::fs::remove_file(&spv);
    Ok(bytes)
}

fn find_glslang() -> Option<PathBuf> {
    if let Ok(p) = env::var("RTORCH_GLSLANG") {
        let p = PathBuf::from(&p);
        if p.exists() {
            return Some(p);
        }
    }
    // Candidate lists: explicit SDK env, known roots, then newest SDK.
    let mut q: Vec<PathBuf> = Vec::new();
    if let Ok(s) = env::var("VULKAN_SDK") {
        let root = PathBuf::from(&s);
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
    let root = PathBuf::from("C:\\VulkanSDK");
    let mut vers: Vec<PathBuf> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&root) {
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

fn execute_gpu(
    handle: *mut std::ffi::c_void,
    inputs: &[Vec<u8>],
    device: i32,
    gk: GpuKernelFn,
    out_size: Option<OutputSizeFn>,
) -> Result<Vec<u8>, RtorchError> {
    let glsl_ptr = unsafe { gk() };
    if glsl_ptr.is_null() {
        unsafe { FreeLibrary(handle) };
        return Err(RtorchError::compile("formula returned null GLSL kernel"));
    }
    let glsl = unsafe { std::ffi::CStr::from_ptr(glsl_ptr) }
        .to_string_lossy()
        .into_owned();

    // output length
    let in_blobs: Vec<Blob> = inputs
        .iter()
        .map(|b| Blob {
            data: b.as_ptr() as *const std::ffi::c_void,
            len: b.len(),
        })
        .collect();
    let n = in_blobs.len() as i32;
    let want = match out_size {
        Some(f) => (unsafe { f(n, in_blobs.as_ptr(), device) }) as usize,
        None => 0,
    };

    // workgroup counts: formula override, else default gx=ceil(out_elems/256)
    let mut gx: i32 = (((want / 4).max(1) + 255) / 256) as i32;
    let mut gy: i32 = 1;
    let mut gz: i32 = 1;
    if let Some(gf) = load_symbol::<GpuGroupsFn>(handle, "rtorch_gpu_groups") {
        unsafe { gf(&mut gx, &mut gy, &mut gz) };
    }

    let spv = compile_glsl_to_spv(&glsl)?;
    eprintln!(
        "[rtorch] gpu kernel compiled ({} bytes spv), groups=({gx},{gy},{gz}), out={want} B",
        spv.len()
    );
    let groups = [gx.max(1) as u32, gy.max(1) as u32, gz.max(1) as u32];

    // Persistent session: init Vulkan context once, then a warmup + repeated
    // dispatches so only real GPU time is measured.
    let input_sizes: Vec<usize> = inputs.iter().map(|b| b.len()).collect();
    let mut session = crate::vk::VkSession::init(&spv, &input_sizes, want)?;
    let mut out = vec![0u8; want];

    let reps = std::env::var("RTORCH_VK_REPS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(8);
    // warmup (covers first-launch/JIT/pipeline cache)
    session.dispatch(inputs, groups, &mut out)?;
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        if let Ok(ms) = session.dispatch(inputs, groups, &mut out) {
            if ms < best {
                best = ms;
            }
        }
    }
    eprintln!(
        "[rtorch] gpu dispatch avg(best) elapsed={best:.3} ms over {reps} reps",
    );
    unsafe { FreeLibrary(handle) };
    Ok(out)
}
