// RTorch Vulkan engine binding — connects to the C++ engine (src/vk_engine.cpp,
// built as rtorch_vk.dll) which implements compute using the OFFICIAL Vulkan
// API. Two-phase: rtorch_vk_init (once) + rtorch_vk_dispatch (loop) so the
// expensive setup runs once and only real compute is measured per dispatch.
#![allow(dead_code)]

use std::ffi::c_void;

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryW(name: *const u16) -> *mut c_void;
    fn GetProcAddress(h: *mut c_void, name: *const u8) -> *mut c_void;
    fn FreeLibrary(h: *mut c_void) -> i32;
}

type InitFn = unsafe extern "C" fn(
    *const c_void, usize,        // spv, spv_len
    *const usize, i32,           // input_sizes[], num_inputs
    usize,                       // out_len
) -> i32;

type DispatchFn = unsafe extern "C" fn(
    i32,                         // ctx
    *const *const c_void,        // inputs[]
    u32, u32, u32,               // gx, gy, gz
    *mut c_void,                 // out
    *mut f64,                    // elapsed_ms
    i32,                         // reuse_input
) -> i32;

type DestroyFn = unsafe extern "C" fn(i32);

struct Engine {
    handle: *mut c_void,
    init: InitFn,
    dispatch: DispatchFn,
    destroy: DestroyFn,
}

fn load_module(name: &str) -> *mut c_void {
    let w: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
    unsafe { LoadLibraryW(w.as_ptr()) }
}

fn cx_exe_dir() -> Option<std::path::PathBuf> {
    std::env::current_exe().ok()?.parent().map(|p| p.to_path_buf())
}

fn find_engine() -> std::io::Result<*mut c_void> {
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    if let Some(d) = cx_exe_dir() {
        candidates.push(d.join("rtorch_vk.dll"));
        if let Some(p) = d.parent().and_then(|p| p.parent()) {
            candidates.push(p.join("rtorch_vk.dll"));
        }
    }
    candidates.push(std::path::PathBuf::from("rtorch_vk.dll"));
    for c in candidates {
        if c.exists() {
            return Ok(load_module(&c.to_string_lossy()));
        }
    }
    Err(std::io::Error::other("rtorch_vk.dll not found (build vk_engine.cpp) — see vk_engine.cpp header"))
}

fn sym<T: Copy>(h: *mut c_void, name: &str) -> Option<T> {
    let c = std::ffi::CString::new(name).ok()?;
    let p = unsafe { GetProcAddress(h, c.as_ptr() as *const u8) };
    if p.is_null() {
        None
    } else {
        debug_assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<*mut c_void>());
        Some(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&p) })
    }
}

fn ensure_path() {
    for dir in ["C:\\msys64\\ucrt64\\bin", "C:\\msys64\\mingw64\\bin"] {
        if std::path::Path::new(dir).exists() {
            let cur = std::env::var("PATH").unwrap_or_default();
            if !cur.split(';').any(|p| p.eq_ignore_ascii_case(dir)) {
                unsafe { let _ = std::env::set_var("PATH", format!("{};{}", dir, cur)); }
            }
        }
    }
}

pub struct VkSession {
    engine: Engine,
    ctx: i32,
    last_hash: u64,
}

// 64-bit FNV-1a over the concatenated input bytes (stable, collision-safe enough
// for reuse detection in the benchmark hot loop).
fn inputs_hash(inputs: &[Vec<u8>]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in inputs {
        for &x in b.iter() {
            h ^= x as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h ^= 0xff; // separator
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

impl VkSession {
    /// Build a persistent compute session: init the underlying Vulkan context
    /// once with the given kernel/spv and buffer layout.
    pub fn init(
        spv: &[u8],
        input_sizes: &[usize],
        out_len: usize,
    ) -> std::io::Result<VkSession> {
        ensure_path();
        let handle = find_engine()?;
        let init: Option<InitFn> = sym(handle, "rtorch_vk_init");
        let dispatch: Option<DispatchFn> = sym(handle, "rtorch_vk_dispatch");
        let destroy: Option<DestroyFn> = sym(handle, "rtorch_vk_destroy");
        let (init, dispatch, destroy) = match (init, dispatch, destroy) {
            (Some(a), Some(b), Some(c)) => (a, b, c),
            _ => {
                unsafe { FreeLibrary(handle) };
                return Err(std::io::Error::other("engine symbols not exported (old DLL?)"));
            }
        };
        let engine = Engine { handle, init, dispatch, destroy };
        let in_sizes: Vec<usize> = input_sizes.to_vec();
        let ctx = unsafe { init(spv.as_ptr() as *const c_void, spv.len(), in_sizes.as_ptr(), input_sizes.len() as i32, out_len) };
        if ctx < 0 {
            unsafe { FreeLibrary(handle) };
            return Err(std::io::Error::other(format!("rtorch_vk_init failed (ctx={ctx})")));
        }
        Ok(VkSession { engine, ctx, last_hash: 0 })
    }

    /// Dispatch one compute pass. `inputs` are the N input blobs (same lengths
    /// as given at init), `out` must be at least out_len bytes. Returns elapsed ms.
    pub fn dispatch(&mut self, inputs: &[Vec<u8>], groups: [u32; 3], out: &mut [u8]) -> std::io::Result<f64> {
        let h = inputs_hash(inputs);
        let reuse = if self.last_hash == h { 1 } else { 0 };
        self.last_hash = h;
        let in_ptrs: Vec<*const c_void> = inputs.iter().map(|b| b.as_ptr() as *const c_void).collect();
        let mut elapsed: f64 = 0.0;
        let rc = unsafe {
            (self.engine.dispatch)(
                self.ctx,
                in_ptrs.as_ptr(),
                groups[0], groups[1], groups[2],
                out.as_mut_ptr() as *mut c_void,
                &mut elapsed,
                reuse,
            )
        };
        if rc != 0 {
            return Err(std::io::Error::other(format!("rtorch_vk_dispatch failed rc={rc}")));
        }
        Ok(elapsed)
    }
}

impl Drop for VkSession {
    fn drop(&mut self) {
        unsafe { (self.engine.destroy)(self.ctx); }
        unsafe { FreeLibrary(self.engine.handle); }
    }
}

// ==========================================================================
// Device-resident model (NEW): tensors live on the GPU. One device owns a pool
// of device buffers + host staging, and many pipelines (one per kernel), each
// bound to specific input/output buffers. Ops chain device-side with no host
// copies in between — the real path to scale.
// ==========================================================================

type DevInitFn = unsafe extern "C" fn() -> i32;
type DevAllocFn = unsafe extern "C" fn(i32, usize) -> i32;
type DevUploadFn = unsafe extern "C" fn(i32, i32, *const c_void, usize);
type DevDownloadFn = unsafe extern "C" fn(i32, i32, *mut c_void, usize);
type DevFreeFn = unsafe extern "C" fn(i32, i32);
type DevPipeAddFn = unsafe extern "C" fn(i32, *const c_void, usize, *const i32, i32, i32, u32, u32, u32) -> i32;
type DevPipeBindFn = unsafe extern "C" fn(i32, i32, *const i32, i32, i32) -> i32;
type DevPipeRunFn = unsafe extern "C" fn(i32, i32, u32, u32, u32) -> i32;
type DevDestroyFn = unsafe extern "C" fn(i32);
type DevBeginFn = unsafe extern "C" fn(i32) -> i32;
type DevPipeRecordFn = unsafe extern "C" fn(i32, i32, u32, u32, u32) -> i32;
type DevSubmitFn = unsafe extern "C" fn(i32, i32) -> i32;

pub struct GpuDevice {
    handle: *mut c_void,
    dev: i32,
    init: DevInitFn,
    alloc: DevAllocFn,
    upload: DevUploadFn,
    download: DevDownloadFn,
    free: DevFreeFn,
    pipe_add: DevPipeAddFn,
    pipe_bind: DevPipeBindFn,
    pipe_run: DevPipeRunFn,
    destroy: DevDestroyFn,
    begin: DevBeginFn,
    pipe_record: DevPipeRecordFn,
    submit: DevSubmitFn,
}

impl GpuDevice {
    /// Create a persistent compute device (instance + device + queue + pools
    /// created once). Returns Err if the engine DLL / symbols are unavailable.
    pub fn new() -> std::io::Result<GpuDevice> {
        ensure_path();
        let handle = find_engine()?;
        let init: Option<DevInitFn> = sym(handle, "rtorch_vk_dev_init");
        let alloc: Option<DevAllocFn> = sym(handle, "rtorch_vk_alloc");
        let upload: Option<DevUploadFn> = sym(handle, "rtorch_vk_upload");
        let download: Option<DevDownloadFn> = sym(handle, "rtorch_vk_download");
        let free: Option<DevFreeFn> = sym(handle, "rtorch_vk_free");
        let pipe_add: Option<DevPipeAddFn> = sym(handle, "rtorch_vk_pipe_add");
        let pipe_bind: Option<DevPipeBindFn> = sym(handle, "rtorch_vk_pipe_bind");
        let pipe_run: Option<DevPipeRunFn> = sym(handle, "rtorch_vk_pipe_run");
        let begin: Option<DevBeginFn> = sym(handle, "rtorch_vk_dev_begin");
        let pipe_record: Option<DevPipeRecordFn> = sym(handle, "rtorch_vk_pipe_record");
        let submit: Option<DevSubmitFn> = sym(handle, "rtorch_vk_dev_submit");
        let destroy: Option<DevDestroyFn> = sym(handle, "rtorch_vk_dev_destroy");
        let (init, alloc, upload, download, free, pipe_add, pipe_bind, pipe_run, destroy, begin, pipe_record, submit) =
            match (init, alloc, upload, download, free, pipe_add, pipe_bind, pipe_run, destroy, begin, pipe_record, submit) {
                (Some(a), Some(b), Some(c), Some(d), Some(e), Some(g), Some(h), Some(f2), Some(i), Some(j), Some(k2), Some(l2)) => (a, b, c, d, e, g, h, f2, i, j, k2, l2),
                _ => {
                    unsafe { FreeLibrary(handle) };
                    return Err(std::io::Error::other("device-resident symbols not exported (rebuild vk_engine.cpp)"));
                }
            };
        let dev = unsafe { init() };
        if dev < 0 {
            unsafe { FreeLibrary(handle) };
            return Err(std::io::Error::other(format!("rtorch_vk_dev_init failed (dev={dev})")));
        }
        Ok(GpuDevice { handle, dev, init, alloc, upload, download, free, pipe_add, pipe_bind, pipe_run, destroy, begin, pipe_record, submit })
    }

    /// Allocate (or reuse) a device buffer of `size` bytes; returns a buffer id.
    pub fn alloc(&self, size: usize) -> i32 {
        unsafe { (self.alloc)(self.dev, size) }
    }

    /// Upload `data` into device buffer `buf`.
    pub fn upload(&self, buf: i32, data: &[u8]) {
        unsafe { (self.upload)(self.dev, buf, data.as_ptr() as *const c_void, data.len()) }
    }

    /// Download device buffer `buf` into `out` (at least len bytes).
    pub fn download(&self, buf: i32, out: &mut [u8]) {
        unsafe { (self.download)(self.dev, buf, out.as_mut_ptr() as *mut c_void, out.len()) }
    }

    /// Return a device buffer to the pool for reuse.
    pub fn free(&self, buf: i32) {
        unsafe { (self.free)(self.dev, buf) }
    }

    /// Add a pipeline for a kernel (SPIR-V) bound to given input buffers + one
    /// output buffer. Returns a pipeline id.
    pub fn pipe_add(&self, spv: &[u8], in_bufs: &[i32], out_buf: i32, groups: [u32; 3]) -> i32 {
        unsafe { (self.pipe_add)(self.dev, spv.as_ptr() as *const c_void, spv.len(), in_bufs.as_ptr(), in_bufs.len() as i32, out_buf, groups[0], groups[1], groups[2]) }
    }

    /// Re-point a pipeline's descriptor set at fresh input/output buffers.
    pub fn pipe_bind(&self, pipe: i32, in_bufs: &[i32], out_buf: i32) -> std::io::Result<()> {
        let rc = unsafe { (self.pipe_bind)(self.dev, pipe, in_bufs.as_ptr(), in_bufs.len() as i32, out_buf) };
        if rc != 0 {
            return Err(std::io::Error::other(format!("rtorch_vk_pipe_bind failed rc={rc}")));
        }
        Ok(())
    }

    /// Run the pipeline (dispatch + barrier). `groups` = workgroup counts.
    pub fn pipe_run(&self, pipe: i32, groups: [u32; 3]) -> std::io::Result<()> {
        let rc = unsafe { (self.pipe_run)(self.dev, pipe, groups[0], groups[1], groups[2]) };
        if rc != 0 {
            return Err(std::io::Error::other(format!("rtorch_vk_pipe_run failed rc={rc}")));
        }
        Ok(())
    }

    /// Begin a batched-recording pass on the shared record command buffer.
    pub fn dev_begin(&self) -> std::io::Result<()> {
        let rc = unsafe { (self.begin)(self.dev) };
        if rc != 0 { return Err(std::io::Error::other(format!("rtorch_vk_dev_begin failed rc={rc}"))); }
        Ok(())
    }

    /// Record a pipeline dispatch into the open batch (no per-dispatch barrier).
    pub fn dev_pipe_record(&self, pipe: i32, groups: [u32; 3]) -> std::io::Result<()> {
        let rc = unsafe { (self.pipe_record)(self.dev, pipe, groups[0], groups[1], groups[2]) };
        if rc != 0 { return Err(std::io::Error::other(format!("rtorch_vk_pipe_record failed rc={rc}"))); }
        Ok(())
    }

    /// Submit the recorded batch. `wait` drains the queue with one vkQueueWaitIdle.
    pub fn dev_submit(&self, wait: bool) -> std::io::Result<()> {
        let rc = unsafe { (self.submit)(self.dev, wait as i32) };
        if rc != 0 { return Err(std::io::Error::other(format!("rtorch_vk_dev_submit failed rc={rc}"))); }
        Ok(())
    }
}

impl Drop for GpuDevice {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.dev); }
        unsafe { FreeLibrary(self.handle); }
    }
}
