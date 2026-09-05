// RTorch — universal compute framework.
// Rust front-end. The unified formula interface is defined in rtorch.h:
//   any user formula implements rtorch_output_size + rtorch_compute.
// The framework compiles the formula (+ refs) into a DLL, feeds it input
// blobs (--input), allocates the output, times the compute, and writes the
// result to --output or stdout. CPU-only math runs on the host; the formula
// may use OpenCL (device=1) for accelerator work. CUDA-free.
//
// usage: rtorch <formula.cpp> with <refs...> [--input f] [--output f] [--device cpu|gpu]

use std::env;
use std::ffi::CString;
use std::path::{Path, PathBuf};

// Reuse the library's modules (single source of truth) instead of recompiling
// the same rtw/vk/loc/error sources into this bin — one copy in rtorch::*.
use rtorch::{error, loc, rtw, vk};

// A single process-wide trust gate (whitelist-first, in-process remember, then a
// temporary cmd.exe prompt). Shared by the `.rtw` load paths so a path is
// prompted at most once per process run.
fn trust_gate() -> &'static rtorch::trust::TrustGate {
    static GATE: std::sync::OnceLock<rtorch::trust::TrustGate> = std::sync::OnceLock::new();
    GATE.get_or_init(rtorch::trust::TrustGate::from_env)
}

/// Human-facing release version (date-based `YYYY.MM.DD.N`). Kept separate from
/// Cargo's semver `CARGO_PKG_VERSION` (which is a legal-semver marker).
fn rt_release_version() -> &'static str {
    "2026.09.06.1"
}

// Windows FFI for the OpenCL device-enumeration diagnostic (the formula pipeline
// FFI lives in rtorch::formula). Kept here because `report_devices` uses it.
#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryW(name: *const u16) -> *mut std::ffi::c_void;
    fn GetProcAddress(h: *mut std::ffi::c_void, name: *const u8) -> *mut std::ffi::c_void;
    fn FreeLibrary(h: *mut std::ffi::c_void) -> i32;
}

struct Opts {
    formula: String,
    refs: Vec<String>,
    inputs: Vec<PathBuf>,
    output: Option<PathBuf>,
    device: i32,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "--version" || a == "-V") {
        // Human-facing release version is the date-based YYYY.MM.DD.N (see README
        // "Versioning"). Cargo's CARGO_PKG_VERSION is a semver marker (2696.9.6)
        // because Cargo rejects the 4-part/leading-zero date form.
        println!("rtorch {}", rt_release_version());
        return;
    }
    if args.iter().any(|a| a == "--help" || a == "-h") {
        usage();
        return;
    }
    if args.iter().any(|a| a == "--vk-smoke") {
        vk_smoke();
        return;
    }
    // RTW subcommands: --dump <x.rtw>, --pack <formula.cpp> -o <x.rtw>,
    // or run a kernel/result container directly: rtorch <x.rtw> [--input ...]
    if let Some(sub) = args.get(1).map(|s| s.as_str()) {
        if sub == "--dump" || sub == "--pack" {
            let rc = rtw_subcommand(&args);
            std::process::exit(rc);
        }
        if sub.ends_with(".rtw") {
            let rc = run_rtw_file(&args);
            std::process::exit(rc);
        }
    }
    let opts = match parse_args(&args) {
        Some(o) => o,
        None => {
            usage();
            std::process::exit(2);
        }
    };

    let gpu_count = report_devices();

    // Resolve device: explicit flag wins, else auto (GPU if any).
    let device = if opts.device >= 0 {
        opts.device
    } else if gpu_count > 0 {
        1
    } else {
        0
    };

    // Explicit --device gpu with no GPU available must report clearly, not crash.
    if opts.device == 1 && gpu_count == 0 {
        eprintln!(
            "rtorch: GPU unavailable (no OpenCL/Vulkan compute device found); use --device cpu or check the Vulkan driver"
        );
        std::process::exit(1);
    }

    match run(&opts, device) {
        Ok(()) => {}
        Err(e) => {
            eprintln!("rtorch: {e}");
            std::process::exit(e.exit_code());
        }
    }
}

fn usage() {
    eprintln!("RTorch — universal compute framework (CUDA-free)");
    eprintln!(
        "usage: rtorch <formula.cpp|.dll> with <refs...> [--input <file>]... [--output <file>] [--device cpu|gpu]"
    );
    eprintln!("  <formula.cpp> : implements rtorch_output_size + rtorch_compute (see rtorch.h); compiled on the fly");
    eprintln!("  <formula.dll> : a pre-built formula DLL (e.g. delta.dll) loaded directly, no compiler needed");
    eprintln!("  --input       : raw byte blobs fed to the formula (repeatable)");
    eprintln!("  --output      : write result blob to file; default stdout");
}

// Hidden smoke test: run the Vulkan compute engine on vec_mul.spv to verify the
// backend end-to-end (device discovery, pipeline build, dispatch, readback).
#[cfg(windows)]
fn vk_smoke() {
    let mono = env::args().any(|a| a == "--vk-mono");
    let kernel = if mono { "mono" } else { "vec_mul" };
    let spv = match loc::read_kernel(kernel) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("vk-smoke: cannot read kernel {kernel}.spv: {e}");
            std::process::exit(1);
        }
    };
    let n: usize = 8;
    let a: Vec<u8> = (0..n).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let b: Vec<u8> = (0..n)
        .flat_map(|i| ((i as f32) * 2.0).to_le_bytes())
        .collect();
    let out_len = n * 4;
    let groups = [((n as u32) + 255) / 256, 1, 1];
    let inputs;
    if mono {
        inputs = Vec::new();
    } else {
        inputs = vec![a, b];
    }
    let in_sizes: Vec<usize> = inputs.iter().map(|b| b.len()).collect();
    let mut session = match vk::VkSession::init(&spv, &in_sizes, out_len) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("vk-smoke: vulkan init failed: {e}");
            std::process::exit(1);
        }
    };
    let mut out = vec![0u8; out_len];
    match session.dispatch(&inputs, groups, &mut out) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("vk-smoke: vulkan run failed: {e}");
            std::process::exit(1);
        }
    }
    {
        let vals: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        println!("vk-smoke: result -> {:?}", &vals[..n]);
        let ok = if mono {
            (0..n).all(|i| (vals[i] - (i as f32 + 1.0)).abs() < 1e-4)
        } else {
            (0..n).all(|i| (vals[i] - (i as f32) * (i as f32) * 2.0).abs() < 1e-4)
        };
        println!("vk-smoke: PASS={ok}");
        std::process::exit(if ok { 0 } else { 1 });
    }
}

#[cfg(not(windows))]
fn vk_smoke() {
    eprintln!("vk-smoke: Windows only");
}

fn parse_args(args: &[String]) -> Option<Opts> {
    let mut formula: Option<String> = None;
    let mut refs: Vec<String> = Vec::new();
    let mut inputs: Vec<PathBuf> = Vec::new();
    let mut output: Option<PathBuf> = None;
    let mut device = -1; // -1 = auto, 0 = cpu, 1 = gpu

    let mut after_with = false;
    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
            "--input" => {
                i += 1;
                inputs.push(PathBuf::from(args.get(i)?));
            }
            "--output" => {
                i += 1;
                output = Some(PathBuf::from(args.get(i)?));
            }
            "--device" => {
                i += 1;
                let d = args.get(i)?;
                if d.eq_ignore_ascii_case("gpu") {
                    device = 1;
                } else if d.eq_ignore_ascii_case("cpu") {
                    device = 0;
                }
            }
            "with" => after_with = true,
            _ => {
                if !after_with {
                    if formula.is_none() {
                        formula = Some(a.clone());
                    } else {
                        refs.push(a.clone());
                    }
                } else {
                    refs.push(a.clone());
                }
            }
        }
        i += 1;
    }

    Some(Opts {
        formula: formula?,
        refs,
        inputs,
        output,
        device,
    })
}

fn run(opts: &Opts, device: i32) -> error::Result<()> {
    // Read input blobs from disk (the CLI's job; the pipeline takes bytes).
    let mut input_bufs: Vec<Vec<u8>> = Vec::new();
    for f in &opts.inputs {
        let bytes = std::fs::read(f)?;
        input_bufs.push(bytes);
    }
    let refs: Vec<&Path> = opts.refs.iter().map(|r| Path::new(r.as_str())).collect();

    // Trust gate for a pre-built formula DLL (loadable code): whitelist first,
    // else in-process remembered, else a temporary cmd.exe prompt. A `.cpp`
    // formula is source compiled locally (trusted), so it is not prompted.
    let formula_path = Path::new(&opts.formula);
    if let Some(ext) = formula_path.extension().and_then(|e| e.to_str()) {
        if ext.eq_ignore_ascii_case("dll") {
            match trust_gate().check(formula_path, "formula DLL") {
                Ok(true) => {}
                Ok(false) => {
                    return Err(error::RtorchError::whitelist(format!(
                        "{}: formula DLL not trusted",
                        formula_path.display()
                    )));
                }
                Err(e) => {
                    return Err(error::RtorchError::whitelist(format!(
                        "trust check failed: {e}"
                    )));
                }
            }
        }
    }

    // Delegate the whole compile/load/dispatch pipeline to the library. This is
    // the single source of truth for running a formula; the CLI only does I/O.
    let bytes = rtorch::formula::run(formula_path, &refs, &input_bufs, device)?;

    if let Some(path) = &opts.output {
        std::fs::write(path, &bytes)?;
        eprintln!("[rtorch] wrote {} bytes -> {}", bytes.len(), path.display());
    } else {
        std::io::stdout().write_all(&bytes)?;
    }
    let _ = device;
    Ok(())
}

use std::io::Write;

// Tries the unified protocol (rtorch_output_size + rtorch_compute); falls back
// to legacy rtorch_main if the formula does not export the protocol functions.
// ---- OpenCL device enumeration ----
#[cfg(windows)]
const CL_DEVICE_NAME: u32 = 0x102B;
#[cfg(windows)]
const CL_DEVICE_TYPE: u32 = 0x1000;
#[cfg(windows)]
const CL_DEVICE_TYPE_GPU: u64 = (1 << 0);
#[cfg(windows)]
const CL_PLATFORM_NAME: u32 = 0x0902;

#[cfg(windows)]
type clGetPlatformIDs_t =
    unsafe extern "system" fn(u32, *mut *mut std::ffi::c_void, *mut u32) -> i32;
#[cfg(windows)]
type clGetDeviceIDs_t = unsafe extern "system" fn(
    *mut std::ffi::c_void,
    u64,
    u32,
    *mut *mut std::ffi::c_void,
    *mut u32,
) -> i32;
#[cfg(windows)]
type clGetInfo_t = unsafe extern "system" fn(
    *mut std::ffi::c_void,
    u32,
    usize,
    *mut std::ffi::c_void,
    *mut usize,
) -> i32;

#[cfg(windows)]
fn resolve_fn<T: Copy>(h: *mut std::ffi::c_void, name: &str) -> Option<T> {
    let cname = CString::new(name).ok()?;
    let p = unsafe { GetProcAddress(h, cname.as_ptr() as *const u8) };
    if p.is_null() {
        None
    } else if std::mem::size_of::<T>() != std::mem::size_of::<*mut std::ffi::c_void>() {
        None
    } else {
        Some(unsafe { std::mem::transmute_copy::<*mut std::ffi::c_void, T>(&p) })
    }
}

#[cfg(windows)]
fn report_devices() -> u32 {
    let w: Vec<u16> = b"OpenCL.dll\0"
        .iter()
        .map(|&b| b as u16)
        .chain(std::iter::once(0))
        .collect();
    let h = unsafe { LoadLibraryW(w.as_ptr()) };
    if h.is_null() {
        eprintln!("[rtorch] OpenCL: OpenCL.dll not found; host-only.");
        return 0;
    }
    let get_platforms: Option<clGetPlatformIDs_t> = resolve_fn(h, "clGetPlatformIDs");
    let get_devices: Option<clGetDeviceIDs_t> = resolve_fn(h, "clGetDeviceIDs");
    let get_info: Option<clGetInfo_t> = resolve_fn(h, "clGetDeviceInfo");

    if get_platforms.is_none() {
        eprintln!("[rtorch] OpenCL: clGetPlatformIDs unavailable; host-only.");
        unsafe { FreeLibrary(h) };
        return 0;
    }
    let mut nplat: u32 = 0;
    let status = unsafe { get_platforms.unwrap()(0, std::ptr::null_mut(), &mut nplat) };
    if status != 0 || nplat == 0 {
        eprintln!("[rtorch] OpenCL: no platform found (code {status}); host-only.");
        unsafe { FreeLibrary(h) };
        return 0;
    }
    let mut plats: Vec<*mut std::ffi::c_void> = vec![std::ptr::null_mut(); nplat as usize];
    unsafe { get_platforms.unwrap()(nplat, plats.as_mut_ptr(), &mut nplat) };

    let mut gpu_total: u32 = 0;
    for &p in plats.iter() {
        let mut ndev: u32 = 0;
        if let Some(f) = get_devices {
            unsafe { f(p, CL_DEVICE_TYPE_GPU, 0, std::ptr::null_mut(), &mut ndev) };
        }
        if ndev == 0 {
            continue;
        }
        let mut devs = vec![std::ptr::null_mut(); ndev as usize];
        if let Some(f) = get_devices {
            unsafe { f(p, CL_DEVICE_TYPE_GPU, ndev, devs.as_mut_ptr(), &mut ndev) };
        }
        for &d in devs.iter().take(ndev as usize) {
            let mut buf = [0u8; 256];
            let mut sz: usize = 0;
            if let Some(f) = get_info {
                unsafe {
                    f(
                        d,
                        CL_DEVICE_NAME,
                        buf.len(),
                        buf.as_mut_ptr().cast(),
                        &mut sz,
                    )
                };
            }
            let name = String::from_utf8_lossy(
                &buf[..sz.min(buf.len())]
                    .split(|&b| b == 0)
                    .next()
                    .unwrap_or(&[]),
            );
            eprintln!("[rtorch] OpenCL device: {name}");
            gpu_total += 1;
        }
    }
    eprintln!("[rtorch] OpenCL GPU devices discovered: {gpu_total}");
    unsafe { FreeLibrary(h) };
    gpu_total
}

#[cfg(not(windows))]
fn report_devices() -> u32 {
    eprintln!("[rtorch] OpenCL device enumeration is Windows-only in this preview.");
    0
}

// ---- RTW (.rtw) Workfile support ----

// `rtorch --dump <x.rtw>`: print a .rtw container's structure and data values.
// `rtorch --pack <formula.cpp> -o <x.rtw>`: package a formula source into a
// kind=kernel container. Values are shown as fp32 by default.
fn rtw_subcommand(args: &[String]) -> i32 {
    match args[1].as_str() {
        "--dump" => {
            if args.len() < 3 {
                eprintln!("usage: rtorch --dump <x.rtw>");
                return 2;
            }
            // Trust gate (whitelist -> remembered -> prompt) before dumping a .rtw.
            let dump_path = std::path::Path::new(&args[2]);
            match trust_gate().check(dump_path, "RTW container") {
                Ok(true) => {}
                Ok(false) => {
                    eprintln!("rtorch: {} refused (not trusted)", dump_path.display());
                    return 1;
                }
                Err(e) => {
                    eprintln!("rtorch: trust check failed: {e}");
                    return 1;
                }
            }
            let bytes = match rtw::read_file(dump_path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("rtorch: {e}");
                    return 1;
                }
            };
            let rtw = match rtw::decode(&bytes) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("rtorch: {e}");
                    return 1;
                }
            };
            println!("kind = {}", render_kind(rtw.kind));
            println!("dtype = {}", rtw::dtype_name(rtw.dtype));
            println!("shape = {:?}", rtw.shape);
            println!("count = {}", rtw.count());
            println!("bytes = {}", rtw.data.len());
            if rtw.kernel.is_some() {
                println!(
                    "kernel = embedded formula source ({} bytes)",
                    rtw.kernel.as_ref().unwrap().len()
                );
            }
            if rtw.kind == rtw::KIND_RESULT && rtw.dtype == rtw::DTYPE_FP32 {
                println!("data (first 12 fp32):");
                let n = rtw.data.len() / 4;
                for i in 0..n.min(12) {
                    let v = f32::from_le_bytes([
                        rtw.data[i * 4],
                        rtw.data[i * 4 + 1],
                        rtw.data[i * 4 + 2],
                        rtw.data[i * 4 + 3],
                    ]);
                    println!("  [{i}] = {v}");
                }
            }
            0
        }
        "--pack" => {
            // args: --pack <formula.cpp> -o <x.rtw> [dtype]
            let mut formula: Option<String> = None;
            let mut out: Option<String> = None;
            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "-o" => {
                        i += 1;
                        out = args.get(i).cloned();
                    }
                    _ => {
                        if formula.is_none() {
                            formula = Some(args[i].clone());
                        }
                    }
                }
                i += 1;
            }
            let (Some(f), Some(o)) = (formula, out) else {
                eprintln!("usage: rtorch --pack <formula.cpp> -o <x.rtw>");
                return 2;
            };
            let src = match std::fs::read(&f) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("rtorch: {e}");
                    return 1;
                }
            };
            let src_len = src.len();
            let rtw = rtw::Rtw {
                kind: rtw::KIND_KERNEL,
                dtype: rtw::DTYPE_BYTES,
                shape: vec![],
                data: vec![],
                kernel: Some(src),
                manifest: None,
            };
            match rtw::write_file(std::path::Path::new(&o), &rtw::encode(&rtw)) {
                Ok(_) => {
                    println!(
                        "[rtw] packed {f} -> {o} (kind=kernel, {} bytes)\n  run: rtorch {o} --input <data> [--device gpu]",
                        src_len
                    );
                    0
                }
                Err(e) => {
                    eprintln!("rtorch: {e}");
                    1
                }
            }
        }
        _ => {
            eprintln!("unknown subcommand");
            2
        }
    }
}

fn render_kind(k: u8) -> &'static str {
    match k {
        rtw::KIND_RESULT => "result",
        rtw::KIND_KERNEL => "kernel",
        rtw::KIND_MODEL => "model",
        rtw::KIND_MEMORY => "memory",
        _ => "unknown",
    }
}

// `rtorch <x.rtw> [--input f]... [--device cpu|gpu]`: run an embedded kernel
// container. Extracts the embedded formula source to a temp .cpp and routes to
// the standard compile/run pipeline.
fn run_rtw_file(args: &[String]) -> i32 {
    // Trust gate: whitelist first, else in-process remembered, else prompt via a
    // temporary cmd window. Rejects untrusted `.rtw` before it reaches decode.
    let rtw_path = std::path::Path::new(&args[1]);
    match trust_gate().check(rtw_path, "RTW container") {
        Ok(true) => {}
        Ok(false) => {
            eprintln!("rtorch: {} refused (not trusted)", rtw_path.display());
            return 1;
        }
        Err(e) => {
            eprintln!("rtorch: trust check failed: {e}");
            return 1;
        }
    }
    let bytes = match rtw::read_file(rtw_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("rtorch: {e}");
            return 1;
        }
    };
    let rtw = match rtw::decode(&bytes) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("rtorch: {e}");
            return 1;
        }
    };
    if rtw.kind != rtw::KIND_KERNEL {
        eprintln!(
            "rtorch: {} is kind=result (not runnable); use --dump to inspect",
            args[1]
        );
        return 2;
    }
    let Some(kernel_src) = &rtw.kernel else {
        eprintln!("rtorch: kernel container has no embedded source");
        return 2;
    };
    // write the embedded formula to a temp .cpp in the current dir (so
    // `#include "rtorch.h"` resolves against the project root), then run().
    let tmp = env::current_dir()
        .unwrap_or_else(|_| env::temp_dir())
        .join(format!("rtorch_rtw_{}.cpp", std::process::id()));
    if let Err(e) = std::fs::write(&tmp, kernel_src) {
        eprintln!("rtorch: write temp formula: {e}");
        return 1;
    }
    let mut run_args: Vec<String> = vec![String::from("rtorch"), tmp.to_string_lossy().to_string()];
    // copy through --input / --device / --output / with refs
    let mut i = 2;
    while i < args.len() {
        let a = args[i].clone();
        if a == "--input" || a == "--output" || a == "--device" {
            run_args.push(a);
            if i + 1 < args.len() {
                run_args.push(args[i + 1].clone());
                i += 1;
            }
        }
        i += 1;
    }
    let opts = match parse_args(&run_args) {
        Some(o) => o,
        None => {
            eprintln!("rtorch: bad args for container");
            return 2;
        }
    };
    let gpu_count = report_devices();
    let device = if opts.device >= 0 {
        opts.device
    } else if gpu_count > 0 {
        1
    } else {
        0
    };
    let rc = match run(&opts, device) {
        Ok(_) => 0,
        Err(e) => {
            eprintln!("rtorch: {e}");
            1
        }
    };
    let _ = std::fs::remove_file(&tmp);
    rc
}
