// RTorch device-resident GPU tensor layer — tensors live on the GPU.
// A GpuContext owns a persistent Vulkan device + a pool of device buffers +
// one pipeline per kernel. Ops (matmul / add / tanh / scale) allocate an
// output buffer, re-point the pipeline's descriptor set at the current
// input/output buffers, and dispatch — NO host copies in the chain. This is
// the real path to 500M throughput (weights + activations stay device-side).
//
// Current precision: f32. f64<->f32 conversion helpers are provided for the
// autograd Var wiring (phase 3).

use crate::loc;
use crate::vk::GpuDevice;
use std::rc::Rc;

fn f32_to_bytes(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn bytes_to_f32(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() }
fn u32_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

fn spv(name: &str) -> std::io::Result<Vec<u8>> {
    loc::read_kernel(name).map_err(|e| std::io::Error::new(std::io::ErrorKind::NotFound, format!("spv {name}: {e}")))
}

// Load a kernel, add a pipeline, and check the returned pipe id. Returns the
// pipe id or an io::Error (never panics) so a missing kernel / engine failure
// surfaces as a clean error instead of a panic.
fn pipe_kernel(dev: &GpuDevice, name: &str, in_bufs: &[i32], out: i32, groups: [u32; 3]) -> std::io::Result<i32> {
    let spv = spv(name)?;
    let id = dev.pipe_add(&spv, in_bufs, out, groups);
    if id < 0 {
        return Err(std::io::Error::other(format!("pipe_add {name} failed (id={id})")));
    }
    Ok(id)
}

/// Kernel registry — maps a kernel name to its pipeline id. New GPU kernels
/// (matmul/add/tanh/... plus future user kernels) are registered here instead of
/// being hardcoded as struct fields, so the Vulkan backend is extensible without
/// touching the runtime each time.
#[derive(Default)]
pub struct KernelRegistry {
    map: std::collections::HashMap<String, i32>,
}

impl KernelRegistry {
    pub fn insert(&mut self, name: &str, pipe: i32) { self.map.insert(name.to_string(), pipe); }
    pub fn get(&self, name: &str) -> Option<i32> { self.map.get(name).copied() }
    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.map.keys().cloned().collect();
        v.sort();
        v
    }
}

pub struct GpuContext {
    pub dev: Rc<GpuDevice>,
    pub kernels: KernelRegistry,
    batch: std::cell::Cell<bool>,
    free_pending: std::cell::RefCell<Vec<i32>>,
    // 持久 params buffer 缓存: (op_tag, bytes) -> buffer id. 值不变时复用(免重复 upload/sync).
    params_cache: std::cell::RefCell<std::collections::HashMap<(String, Vec<u8>), i32>>,
}

impl GpuContext {
    /// Create a persistent GPU context: device + pipelines for matmul/add/tanh/scale
    /// plus backward kernels.
    pub fn new() -> std::io::Result<GpuContext> {
        let dev = Rc::new(GpuDevice::new()?);
        // scratch buffers to anchor descriptor sets at pipeline-add time; the
        // real use re-points via pipe_bind. Each kernel's binding count:
        //  matmul/add/mmgrad_*: 3 in + 1 out; tanh/scale: 2 in + 1 out; tanhgrad: 2 in + 1 out.
        let scratch = dev.alloc(16);
        let s2 = dev.alloc(16);
        let s3 = dev.alloc(16);
        // Register every kernel in the registry (name -> pipeline id). Each
        // kernel's descriptor binding count is set at pipe_add time; binding
        // content is re-pointed per op via pipe_bind.
        let mut kernels = KernelRegistry::default();
        kernels.insert("gemm_tiled", pipe_kernel(&dev, "gemm_tiled", &[scratch, s2, s3], scratch, [1, 1, 1])?);
        kernels.insert("gemm_tile32", pipe_kernel(&dev, "gemm_tile32", &[scratch, s2, s3], scratch, [1, 1, 1])?);
        kernels.insert("add", pipe_kernel(&dev, "add", &[scratch, s2, s3], scratch, [1, 1, 1])?);
        kernels.insert("tanh", pipe_kernel(&dev, "tanh", &[scratch, s2], scratch, [1, 1, 1])?);
        kernels.insert("scale", pipe_kernel(&dev, "scale", &[scratch, s2], scratch, [1, 1, 1])?);
        kernels.insert("mmgrad_a", pipe_kernel(&dev, "mmgrad_a", &[scratch, s2, s3], scratch, [1, 1, 1])?);
        kernels.insert("mmgrad_b", pipe_kernel(&dev, "mmgrad_b", &[scratch, s2, s3], scratch, [1, 1, 1])?);
        kernels.insert("tanhgrad", pipe_kernel(&dev, "tanhgrad", &[scratch, s2, s3], scratch, [1, 1, 1])?);
        // gather: bind 0=E,1=ids,2=params,3=out
        kernels.insert("gather", pipe_kernel(&dev, "gather", &[scratch, s2, s3], scratch, [1, 1, 1])?);
        // softmax-CE: bind 0=logits,1=target,2=params,3=grad,4=loss
        kernels.insert("ce", pipe_kernel(&dev, "ce", &[scratch, s2, s3, scratch], scratch, [1, 1, 1])?);
        // row softmax: bind 0=A(B×V),1=P(B,V),2=C(B×V)
        kernels.insert("softmax", pipe_kernel(&dev, "softmax", &[scratch, s2, s3], scratch, [1, 1, 1])?);
        // adam: bind 0=param,1=grad,2=m,3=v,4=ps,5=out
        kernels.insert("adam", pipe_kernel(&dev, "adam", &[scratch, s2, s3, scratch, scratch], scratch, [1, 1, 1])?);
        // gather_scatter: bind 0=EGrad,1=ids,2=P,3=dc
        kernels.insert("gather_scatter", pipe_kernel(&dev, "gather_scatter", &[scratch, s2, s3], scratch, [1, 1, 1])?);
        // reduce: bind 0=dc,1=P,2=out
        kernels.insert("reduce", pipe_kernel(&dev, "reduce", &[scratch, s2], scratch, [1, 1, 1])?);
        Ok(GpuContext { dev, kernels, batch: std::cell::Cell::new(false), free_pending: std::cell::RefCell::new(Vec::new()), params_cache: std::cell::RefCell::new(std::collections::HashMap::new()) })
    }

    /// Look up a registered kernel's pipeline id (panic on a typo — programmer error).
    pub fn pipe(&self, name: &str) -> i32 {
        self.kernels.get(name).unwrap_or_else(|| panic!("kernel not registered: {name}"))
    }

    fn alloc_out(&self, n: usize) -> i32 { self.dev.alloc(n * 4) }

    fn run_op(&self, pipe: i32, in_bufs: &[i32], out_buf: i32, groups: [u32; 3]) {
        self.dev.pipe_bind(pipe, in_bufs, out_buf).expect("bind");
        if self.batch.get() {
            self.dev.dev_pipe_record(pipe, groups).expect("record");
        } else {
            self.dev.pipe_run(pipe, groups).expect("run");
        }
    }

    /// Open a batched recording pass: subsequent run_op calls record dispatches
    /// into ONE shared command buffer (no per-op submit/waitIdle). All recorded
    /// dispatches execute when `end_batch` submits once. Intermediate results are
    /// read back by the next dispatch via the inserted COMPUTE memory barriers.
    pub fn begin_batch(&self) {
        if self.batch.get() { panic!("batch already open"); }
        self.dev.dev_begin().expect("dev_begin");
        self.batch.set(true);
    }

    /// Close the batch and submit all recorded dispatches in one vkQueueSubmit +
    /// one waitIdle. After this, results are ready to read. Pending deferred frees
    /// are flushed only AFTER the submit (so no buffer id is reused while the
    /// recorded command buffer is still referencing it).
    pub fn end_batch(&self) {
        if !self.batch.get() { panic!("batch not open"); }
        self.batch.set(false);
        self.dev.dev_submit(true).expect("dev_submit");
        let pending: Vec<i32> = std::mem::take(&mut *self.free_pending.borrow_mut());
        for buf in pending { self.dev.free(buf); }
    }

    /// Free a device buffer. In batch (recording) mode the FREE IS DEFERRED until
    /// after the final submit, because the recorded command buffer still references
    /// the buffer id and a recycled id would corrupt the descriptor binding.
    pub fn ctx_free(&self, buf: i32) {
        if self.batch.get() {
            self.free_pending.borrow_mut().push(buf);
        } else {
            self.dev.free(buf);
        }
    }

    /// Get a persistent params buffer for (op_tag, bytes): reuse a cached buffer if
    /// the same bytes were already uploaded; else allocate + upload once and cache.
    /// This removes the per-op alloc/upload/sync of small params buffers.
    pub fn get_params(&self, tag: &str, bytes: &[u8]) -> i32 {
        let key = (tag.to_string(), bytes.to_vec());
        let mut cache = self.params_cache.borrow_mut();
        if let Some(&id) = cache.get(&key) { return id; }
        let id = self.dev.alloc(bytes.len().max(4));
        self.dev.upload(id, bytes);
        cache.insert(key, id);
        id
    }
}

pub struct GpuTensor {
    pub ctx: Rc<GpuContext>,
    pub buf: i32,
    pub r: usize,
    pub c: usize,
}

impl GpuTensor {
    /// Upload f32 data (r×c) to a fresh device buffer.
    pub fn from_data(ctx: Rc<GpuContext>, data: &[f32], r: usize, c: usize) -> GpuTensor {
        let n = r * c;
        assert_eq!(data.len(), n, "from_data len mismatch (r×c={n})");
        let bytes = f32_to_bytes(data);
        let buf = ctx.dev.alloc(bytes.len());
        ctx.dev.upload(buf, &bytes);
        GpuTensor { ctx, buf, r, c }
    }

    fn alloc(ctx: Rc<GpuContext>, r: usize, c: usize) -> GpuTensor {
        let buf = ctx.alloc_out(r * c);
        GpuTensor { ctx, buf, r, c }
    }

    pub fn zeros(ctx: Rc<GpuContext>, r: usize, c: usize) -> GpuTensor {
        Self::from_data(ctx, &vec![0.0; r * c], r, c)
    }

    /// Download to host f32 (r×c).
    pub fn to_vec(&self) -> Vec<f32> {
        let n = self.r * self.c;
        let mut out = vec![0u8; n * 4];
        self.ctx.dev.download(self.buf, &mut out);
        bytes_to_f32(&out)
    }
}

impl Drop for GpuTensor {
    fn drop(&mut self) { self.ctx.ctx_free(self.buf); }
}

// ---- ops ----

// out[r×c] = a[r×k] · b[k×c] — 默认走 tile32 (32×32 tile + 4×4 register block, 高占用).
pub fn matmul(a: &GpuTensor, b: &GpuTensor) -> GpuTensor {
    let (r, k, c) = (a.r, a.c, b.c);
    assert_eq!(a.c, b.r, "matmul dim mismatch a.c={} b.r={}", a.c, b.r);
    let out = GpuTensor::alloc(Rc::clone(&a.ctx), r, c);
    let params = u32_bytes(&[r as u32, c as u32, k as u32]);
    let pb = a.ctx.get_params("gemm_tile32", &params);
    let inb = [a.buf, b.buf, pb];
    a.ctx.run_op(a.ctx.pipe("gemm_tile32"), &inb, out.buf, [((c as u32) + 31) / 32, ((r as u32) + 31) / 32, 1]);
    out
}

// matmul 变体: 等价 tile32 (16 兼容已并入 matmul). 保留供对拍/回溯.
pub fn matmul32(a: &GpuTensor, b: &GpuTensor) -> GpuTensor { matmul(a, b) }

// out[r×c] = a[i] + b[idx] (等长 elementwise 或 1-row/1-col 广播), 全 GPU 无 host tile.
// 形状不相容报 ShapeError(不再静默回绕).
pub fn add(a: &GpuTensor, b: &GpuTensor) -> GpuTensor {
    let (Ar, Ac) = (a.r, a.c);
    let (Br, Bc) = (b.r, b.c);
    let (r, c) = crate::tensor::bcast_shape(Ar, Ac, Br, Bc).unwrap_or_else(|e| panic!("{e}"));
    let n = r * c;
    let out = GpuTensor::alloc(Rc::clone(&a.ctx), r, c);
    // 等长直接用原 buffer; 广播(1×c / r×1) 由 kernel 按 bidx 读取, 无需 host tile/上传.
    let params = u32_bytes(&[Ar as u32, Ac as u32, Br as u32, Bc as u32]);
    let mb = a.ctx.get_params("add", &params);
    a.ctx.run_op(a.ctx.pipe("add"), &[a.buf, b.buf, mb], out.buf, [((n as u32) + 255) / 256, 1, 1]);
    out
}

// 把 src(r0×c0) 复制平铺到 (r×c), 行复用(src 单行或整块复用).
fn tile_rows(src: &[f32], r0: usize, c0: usize, r: usize) -> Vec<f32> {
    let c = c0;
    let mut out = Vec::with_capacity(r * c);
    for _ in 0..r {
        out.extend_from_slice(&src[..c.min(src.len())]);
    }
    out
}

// out[r×c] = tanh(a[i])
pub fn tanh(a: &GpuTensor) -> GpuTensor {
    let n = a.r * a.c;
    let out = GpuTensor::alloc(Rc::clone(&a.ctx), a.r, a.c);
    let params = u32_bytes(&[n as u32]);
    let pb = a.ctx.get_params("tanh", &params);
    let inb = [a.buf, pb];
    a.ctx.run_op(a.ctx.pipe("tanh"), &inb, out.buf, [((n as u32) + 255) / 256, 1, 1]);
    out
}

// out[r×c] = s * a[i]
pub fn scale(s: f32, a: &GpuTensor) -> GpuTensor {
    let n = a.r * a.c;
    let out = GpuTensor::alloc(Rc::clone(&a.ctx), a.r, a.c);
    let params = u32_bytes(&[n as u32, s.to_bits()]);
    let pb = a.ctx.get_params("scale", &params);
    let inb = [a.buf, pb];
    a.ctx.run_op(a.ctx.pipe("scale"), &inb, out.buf, [((n as u32) + 255) / 256, 1, 1]);
    out
}

// gather: out[B×e] = E[ids[i]行]   (batch embedding lookup)
pub fn gather(emb: &GpuTensor, ids: &[usize], b: usize) -> GpuTensor {
    let e = emb.c;
    let ctx = Rc::clone(&emb.ctx);
    let out = GpuTensor::alloc(Rc::clone(&ctx), b, e);
    // ids -> u32 bytes buffer (每 op 内容变, 不缓存)
    let idb: Vec<u32> = ids.iter().map(|&x| x as u32).collect();
    let ib = ctx.dev.alloc(idb.len() * 4);
    ctx.dev.upload(ib, &bytes_from_u32(&idb));
    let params = u32_bytes(&[b as u32, e as u32]);
    let pb = ctx.get_params("gather", &params);
    let inb = [emb.buf, ib, pb];
    ctx.run_op(ctx.pipe("gather"), &inb, out.buf, [((b * e) as u32 + 255) / 256, 1, 1]);
    ctx.ctx_free(ib);
    out
}

fn bytes_from_u32(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

// softmax-CE: logits(B×V) with target(B). Grad(B×V)=softmax-onehot 全 GPU;
// loss 也全 GPU 写到 B 值 tensor(不下载 B×V grad). 返回 (loss_tensor B×1, grad).
// 调用方在无 read-back 段整链后 .to_vec() 读 loss(仅 B 值). 消除 0.4GB/s 下传 grad.
pub fn ce(logits: &GpuTensor, target: &[usize]) -> (GpuTensor, GpuTensor) {
    let (b, v) = (logits.r, logits.c);
    let ctx = Rc::clone(&logits.ctx);
    let grad = GpuTensor::alloc(Rc::clone(&ctx), b, v);    // binding 3 (in 槽)
    let loss_buf = ctx.dev.alloc(b * 4);                    // binding 4 (out)
    let idb: Vec<u32> = target.iter().map(|&x| x as u32).collect();
    let tb = ctx.dev.alloc(idb.len() * 4);
    ctx.dev.upload(tb, &bytes_from_u32(&idb));
    let params = u32_bytes(&[b as u32, v as u32]);
    let pb = ctx.get_params("ce", &params);
    let inb = [logits.buf, tb, pb, grad.buf];
    ctx.run_op(ctx.pipe("ce"), &inb, loss_buf, [b as u32, 1, 1]);
    ctx.ctx_free(tb);
    let loss_t = GpuTensor { ctx: Rc::clone(&ctx), buf: loss_buf, r: b, c: 1 };
    (loss_t, grad)
}

// ---- f64 <-> f32 conversion (for autograd Var wiring) ----
pub fn f64_to_f32(d: &[f64]) -> Vec<f32> { d.iter().map(|&x| x as f32).collect() }
pub fn f32_to_f64(d: &[f32]) -> Vec<f64> { d.iter().map(|&x| x as f64).collect() }

// Adam step, 全 GPU: param/grad/m/v 驻留设备, 一次 kernel 更新 m/v(原位) + 返回新 param(tensor).
// 输入 ps = [n, t, lr_bits, b1_bits, b2_bits, eps_bits]. m/v 为持久设备 buffer(调用方持有),
// kernel 就地更新其内容; p_out 为新 param 值(分配新 buffer, 供调用方 replace GVar.t).
pub fn adam_step(param: &GpuTensor, grad: &GpuTensor, m: &GpuTensor, v: &GpuTensor,
                 lr: f32, b1: f32, b2: f32, eps: f32, t: u32) -> GpuTensor {
    let n = param.r * param.c;
    assert_eq!(n, grad.r * grad.c, "adam grad len mismatch");
    assert_eq!(n, m.r * m.c, "adam m len mismatch");
    assert_eq!(n, v.r * v.c, "adam v len mismatch");
    let ctx = Rc::clone(&param.ctx);
    let out = GpuTensor::alloc(Rc::clone(&ctx), param.r, param.c);
    let ps = u32_bytes(&[n as u32, t, lr.to_bits(), b1.to_bits(), b2.to_bits(), eps.to_bits()]);
    let pb = ctx.get_params("adam", &ps);
    let inb = [param.buf, grad.buf, m.buf, v.buf, pb];
    ctx.run_op(ctx.pipe("adam"), &inb, out.buf, [((n as u32) + 255) / 256, 1, 1]);
    out
}

// row-wise softmax: out[B×V] = softmax over last dim. 全 GPU(不下传). Workgroup per row.
pub fn softmax(a: &GpuTensor) -> GpuTensor {
    let (b, v) = (a.r, a.c);
    let ctx = Rc::clone(&a.ctx);
    let out = GpuTensor::alloc(Rc::clone(&ctx), b, v);
    let params = u32_bytes(&[b as u32, v as u32]);
    let pb = ctx.get_params("softmax", &params);
    let inb = [a.buf, pb];
    ctx.run_op(ctx.pipe("softmax"), &inb, out.buf, [b as u32, 1, 1]);
    out
}

// Gather backward: scatter-add grad rows into embedding grad buffer (GPU, atomic).
// `dc` is B×e (grad wrt gather output). Returns a new EGrad tensor of emb shape (rows×e):
//   EGrad[ ids[row]*e + j ] += dc[row*e + j].  Uses atomicAdd on device (no host roundtrip).
pub fn gather_backward(emb: &GpuTensor, ids: &[usize], dc: &GpuTensor, b: usize) -> GpuTensor {
    let rows = emb.r; let e = emb.c;
    let ctx = Rc::clone(&emb.ctx);
    let out = GpuTensor::zeros(Rc::clone(&ctx), rows, e);
    let idb: Vec<u32> = ids.iter().map(|&x| x as u32).collect();
    let ib = ctx.dev.alloc(idb.len() * 4);
    ctx.dev.upload(ib, &bytes_from_u32(&idb));
    let params = u32_bytes(&[b as u32, e as u32]);
    let pb = ctx.get_params("gather_scatter", &params);
    let inb = [out.buf, ib, pb];
    ctx.run_op(ctx.pipe("gather_scatter"), &inb, dc.buf, [((b * e) as u32 + 255) / 256, 1, 1]);
    ctx.ctx_free(ib);
    out
}

// Broadcast backward reduce: fold grad dc(Or×Oc) to shape (dr×dc_dim), summing over
// broadcast (repeated) rows/cols. 全 GPU, 供 add-broadcast backward 使用(无 host).
pub fn reduce_sum(dc: &GpuTensor, dr: usize, dc_dim: usize) -> GpuTensor {
    if dc.r == dr && dc.c == dc_dim { return scale(1.0, dc); }
    let ctx = Rc::clone(&dc.ctx);
    let out = GpuTensor::alloc(Rc::clone(&ctx), dr, dc_dim);
    let params = u32_bytes(&[dc.r as u32, dc.c as u32, dr as u32, dc_dim as u32]);
    let pb = ctx.get_params("reduce", &params);
    let inb = [dc.buf, pb];
    ctx.run_op(ctx.pipe("reduce"), &inb, out.buf, [((dr * dc_dim) as u32 + 255) / 256, 1, 1]);
    out
}

// ---- backward ops (device-side) ----// dA = dC · Bᵀ  (a: m×k, b: k×n, dC: m×n) ; returns (dA, dB)
pub fn matmul_backward(a: &GpuTensor, b: &GpuTensor, dC: &GpuTensor) -> (GpuTensor, GpuTensor) {
    let (m, n, k) = (a.r, dC.c, a.c);
    assert_eq!(b.r, k); assert_eq!(b.c, n);
    let da = GpuTensor::alloc(Rc::clone(&a.ctx), m, k);
    let db = GpuTensor::alloc(Rc::clone(&a.ctx), k, n);
    let params = u32_bytes(&[m as u32, n as u32, k as u32]);
    let params = u32_bytes(&[m as u32, n as u32, k as u32]);
    let pb = a.ctx.get_params("mmgrad", &params);
    // dA
    let ina = [dC.buf, b.buf, pb];
    a.ctx.run_op(a.ctx.pipe("mmgrad_a"), &ina, da.buf, [((m as u32) + 15) / 16, ((k as u32) + 15) / 16, 1]);
    // dB
    let inb = [a.buf, dC.buf, pb];
    a.ctx.run_op(a.ctx.pipe("mmgrad_b"), &inb, db.buf, [((k as u32) + 15) / 16, ((n as u32) + 15) / 16, 1]);
    (da, db)
}

// dA = dC * (1 - a²), a = tanh(x) precomputed.
pub fn tanh_backward(dC: &GpuTensor, a: &GpuTensor) -> GpuTensor {
    let n = dC.r * dC.c;
    assert_eq!(n, a.r * a.c);
    let da = GpuTensor::alloc(Rc::clone(&dC.ctx), dC.r, dC.c);
    let params = u32_bytes(&[n as u32]);
    let pb = dC.ctx.get_params("tanhgrad", &params);
    let inb = [dC.buf, a.buf, pb];
    dC.ctx.run_op(dC.ctx.pipe("tanhgrad"), &inb, da.buf, [((n as u32) + 255) / 256, 1, 1]);
    da
}
