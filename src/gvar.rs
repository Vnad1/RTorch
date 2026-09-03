// RTorch device-resident autograd tensor (GVar) — the GPU mirror of autograd.rs
// Var. Forward and backward both run on the GPU via gpu_tensor. Values + grads
// live as device-resident GpuTensor; the autograd graph (parents) is recorded
// in Rust. Used by the Striker model layer to train at scale (500M path).
//
// Precision: f32 on device. Host-facing leaf/read uses f64 (converted).

use crate::gpu_tensor::{self, GpuContext, GpuTensor};
use std::cell::RefCell;
use std::rc::Rc;

pub struct GVarData {
    pub t: GpuTensor,
    pub grad: Option<GpuTensor>,
    pub parent: Option<Rc<GVarNode>>,
}
pub type GVar = Rc<RefCell<GVarData>>;

enum GVarNode {
    MatMul(GVar, GVar, usize, usize, usize), // a, b, m, k, n
    Add(GVar, GVar, usize, usize),           // a, b, r, c
    Tanh(GVar, usize, usize),
    Scale(GVar, f64, usize, usize),
    Gather(GVar, Vec<usize>, usize), // emb, ids, e  (batch lookup: out[B×e])
}

pub fn leaf(ctx: Rc<GpuContext>, data: Vec<f64>, r: usize, c: usize) -> GVar {
    let f = gpu_tensor::f64_to_f32(&data);
    let t = GpuTensor::from_data(ctx, &f, r, c);
    Rc::new(RefCell::new(GVarData {
        t,
        grad: None,
        parent: None,
    }))
}

fn wrap(ctx: Rc<GpuContext>, t: GpuTensor, p: GVarNode) -> GVar {
    Rc::new(RefCell::new(GVarData {
        t,
        grad: None,
        parent: Some(Rc::new(p)),
    }))
}

pub fn from_f32(ctx: Rc<GpuContext>, f: Vec<f32>, r: usize, c: usize) -> GVar {
    let t = GpuTensor::from_data(ctx, &f, r, c);
    Rc::new(RefCell::new(GVarData {
        t,
        grad: None,
        parent: None,
    }))
}

pub fn len(v: &GVar) -> usize {
    v.borrow().t.r * v.borrow().t.c
}

// ---- forward ops (device-side) ----
pub fn matmul(a: &GVar, b: &GVar) -> GVar {
    let (m, k, n) = (a.borrow().t.r, a.borrow().t.c, b.borrow().t.c);
    let t = gpu_tensor::matmul(&a.borrow().t, &b.borrow().t);
    let ctx = Rc::clone(&t.ctx);
    wrap(
        ctx,
        t,
        GVarNode::MatMul(Rc::clone(a), Rc::clone(b), m, k, n),
    )
}

pub fn add(a: &GVar, b: &GVar) -> GVar {
    let (r, c) = (
        a.borrow().t.r.max(b.borrow().t.r),
        a.borrow().t.c.max(b.borrow().t.c),
    );
    let t = gpu_tensor::add(&a.borrow().t, &b.borrow().t);
    let ctx = Rc::clone(&t.ctx);
    wrap(ctx, t, GVarNode::Add(Rc::clone(a), Rc::clone(b), r, c))
}

pub fn tanh(a: &GVar) -> GVar {
    let (r, c) = (a.borrow().t.r, a.borrow().t.c);
    let t = gpu_tensor::tanh(&a.borrow().t);
    let ctx = Rc::clone(&t.ctx);
    wrap(ctx, t, GVarNode::Tanh(Rc::clone(a), r, c))
}

pub fn scale(s: f64, a: &GVar) -> GVar {
    let (r, c) = (a.borrow().t.r, a.borrow().t.c);
    let t = gpu_tensor::scale(s as f32, &a.borrow().t);
    let ctx = Rc::clone(&t.ctx);
    wrap(ctx, t, GVarNode::Scale(Rc::clone(a), s, r, c))
}

// batch embedding gather: out[B×e] = emb 的 ids[i] 行. 一次整批查表(替代 onehot·matmul 的 O(V) 浪费).
pub fn gather(emb: &GVar, ids: &[usize], b: usize) -> GVar {
    let e = emb.borrow().t.c;
    let t = gpu_tensor::gather(&emb.borrow().t, ids, b);
    let ctx = Rc::clone(&t.ctx);
    wrap(ctx, t, GVarNode::Gather(Rc::clone(emb), ids.to_vec(), e))
}

/// Set (or overwrite) the output gradient (device-side). `g` must be r×c.
pub fn set_grad(out: &GVar, g: GpuTensor) {
    out.borrow_mut().grad = Some(g);
}

pub struct GVarContext {
    pub ctx: Rc<GpuContext>,
}

// ---- backward (device-side, topological) ----
fn add_grad(v: &GVar, g: GpuTensor) {
    let mut b = v.borrow_mut();
    let slot = &mut b.grad;
    if let Some(old) = slot.take() {
        let combined = gpu_tensor::add(&old, &g);
        *slot = Some(combined);
    } else {
        *slot = Some(g);
    }
}

fn topo(out: &GVar) -> Vec<GVar> {
    let mut order = Vec::new();
    let mut visited = std::collections::HashSet::new();
    fn dfs(v: &GVar, order: &mut Vec<GVar>, visited: &mut std::collections::HashSet<usize>) {
        let id = Rc::as_ptr(v) as usize;
        if !visited.insert(id) {
            return;
        }
        if let Some(p) = v.borrow().parent.clone() {
            match &*p {
                GVarNode::MatMul(a, b, ..) | GVarNode::Add(a, b, ..) => {
                    dfs(a, order, visited);
                    dfs(b, order, visited);
                }
                GVarNode::Tanh(a, ..) | GVarNode::Scale(a, ..) | GVarNode::Gather(a, _, _) => {
                    dfs(a, order, visited);
                }
            }
        }
        order.push(Rc::clone(v));
    }
    dfs(out, &mut order, &mut visited);
    order
}

/// Run backward from `out` (whose .grad must already be set). Fills .grad on all
/// reachable leaves/nodes.
pub fn backward(out: &GVar) {
    let order = topo(out);
    backprop(&order);
}

/// Backward over multiple roots (union graph), so shared backbone nodes accumulate
/// gradients from every head in a SINGLE pass (gvar::backward consumes grads and
/// only traverses one root's graph — insufficient for multi-head models).
pub fn backward_multi(outs: &[GVar]) {
    let mut order = Vec::new();
    let mut visited = std::collections::HashSet::new();
    for o in outs {
        dfs(o, &mut order, &mut visited);
    }
    backprop(&order);
}

fn backprop(order: &[GVar]) {
    for v in order.iter().rev() {
        let parent = v.borrow().parent.clone();
        let Some(p) = parent else { continue }; // leaf: keep its accumulated grad
        let dir = v.borrow_mut().grad.take();
        let Some(dc) = dir else { continue };
        match &*p {
            GVarNode::MatMul(a, b, m, k, n) => {
                let (da, db) = gpu_tensor::matmul_backward(&a.borrow().t, &b.borrow().t, &dc);
                add_grad(a, da);
                add_grad(b, db);
                let _ = (m, k, n);
            }
            GVarNode::Add(a, b, r, c) => {
                // Grab broadcast grad: reduce dc back to a/b's shape.
                let da = reduce_to(&dc, a.borrow().t.r, a.borrow().t.c);
                let db = reduce_to(&dc, b.borrow().t.r, b.borrow().t.c);
                add_grad(a, da);
                add_grad(b, db);
                let _ = (r, c);
            }
            GVarNode::Tanh(a, r, c) => {
                let da = gpu_tensor::tanh_backward(&dc, &v.borrow().t);
                add_grad(a, da);
                let _ = (r, c);
            }
            GVarNode::Scale(a, s, r, c) => {
                let da = gpu_tensor::scale(*s as f32, &dc);
                add_grad(a, da);
                let _ = (r, c);
            }
            GVarNode::Gather(e, ids, edim) => {
                // backward: scatter gradient rows back into the embedding (GPU, atomic, no host).
                let egrad = gpu_tensor::gather_backward(&e.borrow().t, ids, &dc, *edim);
                // egrad 是新 zeros+scatter 结果, 需要与 emb 已累积 grad 相加(若多次 gather 同 emb).
                add_grad(e, egrad);
            }
        }
    }
}

fn dfs(v: &GVar, order: &mut Vec<GVar>, visited: &mut std::collections::HashSet<usize>) {
    let id = Rc::as_ptr(v) as usize;
    if !visited.insert(id) {
        return;
    }
    if let Some(p) = v.borrow().parent.clone() {
        match &*p {
            GVarNode::MatMul(a, b, ..) | GVarNode::Add(a, b, ..) => {
                dfs(a, order, visited);
                dfs(b, order, visited);
            }
            GVarNode::Tanh(a, ..) | GVarNode::Scale(a, ..) | GVarNode::Gather(a, _, _) => {
                dfs(a, order, visited);
            }
        }
    }
    order.push(Rc::clone(v));
}

/// Download a value to host f64.
pub fn to_vec(v: &GVar) -> Vec<f64> {
    gpu_tensor::f32_to_f64(&v.borrow().t.to_vec())
}
/// Download a gradient to host f64 (panics if none).
pub fn grad_to_vec(v: &GVar) -> Vec<f64> {
    let g: Vec<f32> = v.borrow().grad.as_ref().expect("no grad").to_vec();
    gpu_tensor::f32_to_f64(&g)
}

// Gather backward 现走 GPU gather_backward (atomic scatter, 无 host 往返)。

// Add-backward for broadcast: reduce a gradient dc (r×c) to the shape (dr×dc_dim)
// of an operand, summing over broadcast (repeated) rows/cols on the GPU.
fn reduce_to(dc: &GpuTensor, dr: usize, dc_dim: usize) -> GpuTensor {
    if dc.r == dr && dc.c == dc_dim {
        return gpu_tensor::scale(1.0, dc);
    }
    gpu_tensor::reduce_sum(dc, dr, dc_dim)
}

/// Device-side SGD update: p.t = p.t - lr * p.grad. Grads consumed (taken+free).
pub fn sgd_step(params: &[GVar], lr: f64) {
    for p in params {
        let mut b = p.borrow_mut();
        if let Some(g) = b.grad.take() {
            let step = gpu_tensor::scale(-(lr as f32), &g);
            let new_t = gpu_tensor::add(&b.t, &step);
            let old = std::mem::replace(&mut b.t, new_t);
            drop(old); // free the old value buffer
        }
    }
}

/// Device-resident Adam: params + first/second moments keep device-resident.
/// Forward/backward on GPU; the optimizer step runs ONE GPU kernel that updates
/// m/v in place and returns the new param value — NO grad.to_vec() host download,
/// NO per-step H2D/D2H. Eliminates the ~0.4GB/s download bottleneck.
pub struct AdamG {
    pub lr: f64,
    pub b1: f64,
    pub b2: f64,
    pub eps: f64,
    m: std::collections::HashMap<usize, GpuTensor>,
    v: std::collections::HashMap<usize, GpuTensor>,
    t: u64,
}

impl AdamG {
    pub fn new(lr: f64) -> Self {
        AdamG {
            lr,
            b1: 0.9,
            b2: 0.999,
            eps: 1e-8,
            m: Default::default(),
            v: Default::default(),
            t: 0,
        }
    }
    pub fn step(&mut self, params: &[GVar]) {
        self.t += 1;
        let b1 = self.b1 as f32;
        let b2 = self.b2 as f32;
        let eps = self.eps as f32;
        for p in params {
            let id = Rc::as_ptr(p) as usize;
            let (pr, pc, ctxt) = {
                let b = p.borrow();
                let gv = match b.grad.as_ref() {
                    Some(g) => g,
                    None => continue,
                }; // 无 grad 参数跳过(如单跳样本不经状态更新层)
                (gv.r, gv.c, Rc::clone(&b.t.ctx))
            };
            // lazy init device m/v (zeros) per param on first step
            let m = self
                .m
                .entry(id)
                .or_insert_with(|| GpuTensor::zeros(Rc::clone(&ctxt), pr, pc));
            let v = self
                .v
                .entry(id)
                .or_insert_with(|| GpuTensor::zeros(Rc::clone(&ctxt), pr, pc));
            let new_t = {
                let b = p.borrow();
                gpu_tensor::adam_step(
                    &b.t,
                    b.grad.as_ref().unwrap(),
                    m,
                    v,
                    self.lr as f32,
                    b1,
                    b2,
                    eps,
                    self.t as u32,
                )
            };
            let mut bm = p.borrow_mut();
            let old = std::mem::replace(&mut bm.t, new_t);
            drop(old);
            bm.grad = None;
        }
    }

    // ==== 状态导出/导入 (V1 RTW 恢复续学): m/v 设备驻留, 导出时 to_vec 拉回, 导入时 zeros+回填. ====
    pub fn state(&self, params: &[GVar]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, u64) {
        let mut ms = Vec::new();
        let mut vs = Vec::new();
        for p in params {
            let id = Rc::as_ptr(p) as usize;
            let n = p.borrow().t.r * p.borrow().t.c;
            ms.push(
                self.m
                    .get(&id)
                    .map(|m| m.to_vec())
                    .unwrap_or_else(|| vec![0.0f32; n]),
            );
            vs.push(
                self.v
                    .get(&id)
                    .map(|v| v.to_vec())
                    .unwrap_or_else(|| vec![0.0f32; n]),
            );
        }
        (ms, vs, self.t)
    }
    pub fn load_state(&mut self, params: &[GVar], ms: &[Vec<f32>], vs: &[Vec<f32>], t: u64) {
        let mut nms = std::collections::HashMap::new();
        let mut nvs = std::collections::HashMap::new();
        for (i, p) in params.iter().enumerate() {
            let id = Rc::as_ptr(p) as usize;
            let (r, c, ctx) = {
                let b = p.borrow();
                (b.t.r, b.t.c, Rc::clone(&b.t.ctx))
            };
            let n = r * c;
            let mk = |src: &[f32]| {
                let mut z = vec![0.0f32; n];
                if !src.is_empty() {
                    let m = src.len().min(n);
                    z[..m].copy_from_slice(&src[..m]);
                }
                gpu_tensor::GpuTensor::from_data(Rc::clone(&ctx), &z, r, c)
            };
            nms.insert(id, mk(ms.get(i).map(|x| x.as_slice()).unwrap_or(&[])));
            nvs.insert(id, mk(vs.get(i).map(|x| x.as_slice()).unwrap_or(&[])));
        }
        self.m = nms;
        self.v = nvs;
        self.t = t;
    }
}
