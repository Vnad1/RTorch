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
use std::process::Command;
use std::time::Instant;

#[cfg(windows)]
use std::os::windows::ffi::OsStrExt;

mod error;
mod loc;
mod rtw;
mod vk;

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
struct Blob {
    data: *const std::ffi::c_void,
    len: usize,
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
        println!("rtorch {}", env!("CARGO_PKG_VERSION"));
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
        "usage: rtorch <formula.cpp> with <refs...> [--input <file>]... [--output <file>] [--device cpu|gpu]"
    );
    eprintln!("  <formula.cpp> : implements rtorch_output_size + rtorch_compute (see rtorch.h)");
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

fn find_compiler() -> Option<PathBuf> {
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

fn run(opts: &Opts, device: i32) -> error::Result<()> {
    let compiler = find_compiler().ok_or_else(|| {
        error::RtorchError::compile("no C++ compiler found (need MinGW g++; set RTORCH_GXX)")
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

    let formula_path = Path::new(&opts.formula);
    if !formula_path.exists() {
        return Err(error::RtorchError::io(format!(
            "formula not found: {}",
            opts.formula
        )));
    }

    let out_dll = env::current_dir()
        .ok()
        .unwrap_or_else(|| env::temp_dir())
        .join(format!("rtorch_rt_{}.dll", std::process::id()));
    let _ = std::fs::remove_file(&out_dll);

    let mut cmd = Command::new(&compiler);
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
        .arg(formula_path);
    // include formula dir + ref dirs so `#include "rtorch.h"` resolves.
    if let Some(dir) = formula_path.parent() {
        cmd.arg("-I").arg(dir);
    }
    for r in &opts.refs {
        let rp = Path::new(r);
        if rp.exists() {
            cmd.arg(rp);
            if let Some(dir) = rp.parent() {
                cmd.arg("-I").arg(dir);
            }
        } else {
            eprintln!("rtorch: warning: ref not found, skipping: {r}");
        }
    }
    cmd.arg("-o").arg(&out_dll);

    let st = cmd.status()?;
    if !st.success() {
        let _ = std::fs::remove_file(&out_dll);
        return Err(error::RtorchError::compile(
            "compilation of formula failed (see g++ output above)",
        ));
    }

    // Read inputs.
    let mut input_bufs: Vec<Vec<u8>> = Vec::new();
    for f in &opts.inputs {
        let bytes = std::fs::read(f)?;
        input_bufs.push(bytes);
    }

    let bytes = execute_formula(&out_dll, &input_bufs, device)?;

    let _ = std::fs::remove_file(&out_dll);
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
fn execute_formula(dll: &Path, inputs: &[Vec<u8>], device: i32) -> error::Result<Vec<u8>> {
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
            return Err(error::RtorchError::load(format!(
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
        Err(error::RtorchError::load(
            "formula must implement rtorch_output_size + rtorch_compute (see rtorch.h) or legacy rtorch_main",
        ))
    }
}

fn load_dll(dll: &Path) -> std::io::Result<*mut std::ffi::c_void> {
    let wpath: Vec<u16> = dll
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let handle = unsafe { LoadLibraryW(wpath.as_ptr()) };
    if handle.is_null() {
        return Err(std::io::Error::last_os_error());
    }
    Ok(handle)
}

// Find a glslang compiler (glslangValidator.exe preferred, else glslang.exe;
// honest about the tool actually shipping in tools/bin and the Vulkan SDK).
fn find_glslang() -> Option<PathBuf> {
    if let Ok(p) = env::var("RTORCH_GLSLANG") {
        let p = Path::new(&p);
        if p.exists() {
            return Some(p.to_path_buf());
        }
        eprintln!(
            "rtorch: warning: RTORCH_GLSLANG set but not found: {}",
            p.display()
        );
    }
    let mut cands: Vec<PathBuf> = Vec::new();
    if let Ok(sdk) = env::var("VULKAN_SDK") {
        let d = Path::new(&sdk);
        cands.push(d.join("Bin").join("glslangValidator.exe"));
        cands.push(d.join("Bin").join("glslang.exe"));
    }
    // newest SDK under C:\VulkanSDK (independent of VULKAN_SDK)
    if let Ok(rd) = std::fs::read_dir("C:\\VulkanSDK") {
        let mut vers: Vec<PathBuf> = Vec::new();
        for e in rd.flatten() {
            let p = e.path();
            if p.file_name()
                .map(|n| n.to_string_lossy().starts_with('1'))
                .unwrap_or(false)
            {
                vers.push(p);
            }
        }
        vers.sort();
        for v in vers.into_iter().rev() {
            cands.push(v.join("Bin").join("glslangValidator.exe"));
            cands.push(v.join("Bin").join("glslang.exe"));
        }
    }
    // repo tools/bin (glslang.exe is what actually ships there)
    if let Ok(cwd) = env::current_dir() {
        cands.push(cwd.join("tools").join("bin").join("glslang.exe"));
        cands.push(cwd.join("tools").join("bin").join("glslangValidator.exe"));
    }
    // relative: tools/bin
    if let Some(exe) = loc::exe_dir() {
        cands.push(exe.join("..").join("tools").join("bin").join("glslang.exe"));
    }
    for c in cands {
        if c.exists() {
            return Some(c);
        }
    }
    None
}

// Compile a GLSL compute shader to SPIR-V via glslang. Returns SPIR-V bytes.
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

// Execute a formula's GPU kernel: fetch GLSL, compile to SPIR-V, run through the
// C++ Vulkan engine, and return the output blob.
fn execute_gpu(
    handle: *mut std::ffi::c_void,
    inputs: &[Vec<u8>],
    device: i32,
    gk: GpuKernelFn,
    out_size: Option<OutputSizeFn>,
) -> error::Result<Vec<u8>> {
    let glsl_ptr = unsafe { gk() };
    if glsl_ptr.is_null() {
        unsafe { FreeLibrary(handle) };
        return Err(error::RtorchError::compile(
            "formula returned null GLSL kernel",
        ));
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
    // dispatches so only real GPU time is measured (framework sets up the device,
    // pipeline and buffers a single time, like a real forward pass loop).
    let input_sizes: Vec<usize> = inputs.iter().map(|b| b.len()).collect();
    let mut session = vk::VkSession::init(&spv, &input_sizes, want)?;
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
    eprintln!("[rtorch] gpu dispatch avg(best) elapsed={best:.3} ms over {reps} reps",);
    unsafe { FreeLibrary(handle) };
    Ok(out)
}

fn load_symbol<T: Copy>(handle: *mut std::ffi::c_void, name: &str) -> Option<T> {
    let cname = CString::new(name).ok()?;
    let p = unsafe { GetProcAddress(handle, cname.as_ptr() as *const u8) };
    if p.is_null() {
        None
    } else {
        debug_assert_eq!(
            std::mem::size_of::<T>(),
            std::mem::size_of::<*mut std::ffi::c_void>()
        );
        Some(unsafe { std::mem::transmute_copy::<*mut std::ffi::c_void, T>(&p) })
    }
}

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
    } else {
        debug_assert_eq!(
            std::mem::size_of::<T>(),
            std::mem::size_of::<*mut std::ffi::c_void>()
        );
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
            let bytes = match rtw::read_file(std::path::Path::new(&args[2])) {
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
    let bytes = match rtw::read_file(std::path::Path::new(&args[1])) {
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
