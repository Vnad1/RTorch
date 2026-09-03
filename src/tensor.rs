// RTorch 张量库最小内核(tensor + 前向 ops + 梯度容器 + SGD)。
// MVP: 2D 张量(r×c, 1D 用 c=1), 前向 matmul/matvec/add/scal/tanh/softmax/onehot,
//   标量梯度容器(.grad 由模型层手动/后续 autograd 填充), SGD 优化器。
// 本版补 DType 形态 + Shape 抽象(shape/numel/reshape/view/transpose/strides),
// 让张量不再被"锁死 rows×cols", 为将来 [B,T,D]/[B,T,N,D] 铺路。
// 注意: 增量设计 —— r/c 保留为 2D 快速路径字段(既有 op/Striker 不受影响),
// dims 记录完整形状(r=dims[0], c=numel/dims[0]); dtype 是逻辑类型标签
// (CPU 参照存储为 f64, GPU 为 f32)。

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
    F16,
    BF16,
    I32,
}

impl DType {
    pub fn name(&self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::I32 => "i32",
        }
    }
    pub fn size(&self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F64 => 8,
            DType::F16 | DType::BF16 => 2,
        }
    }
}

/// A shape (dims) with row-major contiguous strides.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    pub dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self { Shape { dims } }
    pub fn dims(&self) -> &[usize] { &self.dims }
    pub fn rank(&self) -> usize { self.dims.len() }
    pub fn numel(&self) -> usize { self.dims.iter().product() }
    pub fn dim(&self, i: usize) -> usize { self.dims[i] }
    /// Row-major contiguous strides (last dim stride = 1).
    pub fn strides(&self) -> Vec<usize> {
        let mut s = vec![1; self.dims.len()];
        for i in (0..self.dims.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * self.dims[i + 1];
        }
        s
    }
}

#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<f64>,
    pub grad: Vec<f64>,
    pub r: usize,
    pub c: usize,
    pub dtype: DType,
    /// Full logical dims. r=c? r=dims[0], c=numel/dims[0]. For 2D = [r,c].
    pub dims: Vec<usize>,
}

pub fn sigmoid(z: f64) -> f64 { 1.0 / (1.0 + (-z).exp()) }

// Real broadcasting shape rule (right-aligned): a dim is compatible if equal or
// one of the pair is 1. Returns the broadcast output shape, or a ShapeError
// message on incompatibility. The old max+modulo path silently WROTE WRONG
// results for incompatible shapes — this turns that into an explicit error.
pub fn bcast_shape(r1: usize, c1: usize, r2: usize, c2: usize) -> Result<(usize, usize), String> {
    let r = if r1 == r2 { r1 } else if r1 == 1 { r2 } else if r2 == 1 { r1 } else {
        return Err(format!("shape: cannot broadcast {r1}×{c1} and {r2}×{c2} (row dims {r1} vs {r2} not equal or 1)"));
    };
    let c = if c1 == c2 { c1 } else if c1 == 1 { c2 } else if c2 == 1 { c1 } else {
        return Err(format!("shape: cannot broadcast {r1}×{c1} and {r2}×{c2} (col dims {c1} vs {c2} not equal or 1)"));
    };
    Ok((r, c))
}

fn mk(data: Vec<f64>, dims: Vec<usize>, dtype: DType) -> Tensor {
    let numel = dims.iter().product::<usize>();
    let r = dims.first().copied().unwrap_or(0);
    let c = if dims.is_empty() { 0 } else { numel / r };
    Tensor { grad: vec![0.0; data.len()], data, r, c, dtype, dims }
}

impl Tensor {
    pub fn zeros(r: usize, c: usize) -> Self { mk(vec![0.0; r * c], vec![r, c], DType::F64) }
    pub fn ones(r: usize, c: usize) -> Self { mk(vec![1.0; r * c], vec![r, c], DType::F64) }
    pub fn from_data(data: Vec<f64>, r: usize, c: usize) -> Self { mk(data, vec![r, c], DType::F64) }
    /// Build from an arbitrary shape + dtype (r/c derived as 2D projection).
    pub fn from_shape(data: Vec<f64>, dims: Vec<usize>, dtype: DType) -> Result<Self, String> {
        let numel: usize = dims.iter().product();
        if data.len() != numel {
            return Err(format!("shape: data len {} != numel {} for dims {:?}", data.len(), numel, dims));
        }
        Ok(mk(data, dims, dtype))
    }
    pub fn len(&self) -> usize { self.data.len() }
    pub fn numel(&self) -> usize { self.r * self.c }
    pub fn shape(&self) -> Shape { Shape::new(self.dims.clone()) }
    pub fn rank(&self) -> usize { self.dims.len() }
    pub fn zero_grad(&mut self) { for g in &mut self.grad { *g = 0.0; } }
    pub fn get(&self, i: usize, j: usize) -> f64 { self.data[i * self.c + j] }
    /// Return a copy with the given dims (same numel). Errors on size mismatch.
    pub fn reshape(&self, dims: Vec<usize>) -> Result<Tensor, String> {
        let numel: usize = dims.iter().product();
        if numel != self.numel() {
            return Err(format!("shape: reshape numel mismatch get {} want {} for {:?}", self.numel(), numel, dims));
        }
        let mut t = self.clone();
        let r = dims.first().copied().unwrap_or(0);
        let c = if dims.is_empty() { 0 } else { numel / r };
        t.r = r; t.c = c; t.dims = dims;
        Ok(t)
    }
    pub fn view(&self, dims: Vec<usize>) -> Result<Tensor, String> { self.reshape(dims) }
    /// 2D transpose (swap rows/cols, transpose the contiguous data).
    pub fn transpose(&self) -> Tensor {
        let (r, c) = (self.r, self.c);
        let mut d = vec![0.0; self.data.len()];
        for i in 0..r { for j in 0..c { d[j * r + i] = self.data[i * c + j]; } }
        let mut dims = self.dims.clone();
        if dims.len() >= 2 { dims.swap(0, 1); }
        mk(d, dims, self.dtype)
    }
    pub fn as_dtype(&self, dtype: DType) -> Tensor {
        let mut t = self.clone();
        t.dtype = dtype;
        t
    }
}

// out[r×c] = A[r×k] · B[k×c]
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let (r, k, c) = (a.r, a.c, b.c);
    if a.c != b.r {
        panic!("shape: matmul dim mismatch A {r}×{} vs B {}×{c}", a.c, b.r);
    }
    let mut out = Tensor::zeros(r, c);
    for i in 0..r {
        for j in 0..c {
            let mut acc = 0.0;
            for m in 0..k { acc += a.data[i * k + m] * b.data[m * c + j]; }
            out.data[i * c + j] = acc;
        }
    }
    out
}
// out[r×1] = A[r×k] · v[k×1]
pub fn matvec(a: &Tensor, v: &Tensor) -> Tensor {
    matmul(a, v)
}
// 逐元素相加(真广播): 形状相容才允许, 不相容直接报 ShapeError(不再静默回绕).
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let (r, c) = bcast_shape(a.r, a.c, b.r, b.c).unwrap_or_else(|e| panic!("{e}"));
    let mut out = Tensor::zeros(r, c);
    out.dtype = if a.dtype == b.dtype { a.dtype } else { panic!("dtype: add dtype mismatch {} vs {}", a.dtype.name(), b.dtype.name()) };
    for i in 0..out.r {
        for j in 0..out.c {
            let va = a.data[(i % a.r) * a.c + (j % a.c)];
            let vb = b.data[(i % b.r) * b.c + (j % b.c)];
            out.data[i * c + j] = va + vb;
        }
    }
    out
}
pub fn scal(s: f64, x: &Tensor) -> Tensor {
    let mut t = mk(x.data.iter().map(|v| s * v).collect(), x.dims.clone(), x.dtype);
    t.grad = vec![0.0; x.len()];
    t
}
pub fn tanh(x: &Tensor) -> Tensor {
    let mut t = mk(x.data.iter().map(|v| v.tanh()).collect(), x.dims.clone(), x.dtype);
    t.grad = vec![0.0; x.len()];
    t
}
// 行 softmax
pub fn softmax_row(x: &Tensor) -> Tensor {
    let mut out = Tensor::zeros(x.r, x.c);
    out.dtype = x.dtype;
    for i in 0..x.r {
        let start = i * x.c;
        let m = x.data[start..start + x.c].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut esum = 0.0;
        for j in 0..x.c { let e = (x.data[start + j] - m).exp(); out.data[start + j] = e; esum += e; }
        for j in 0..x.c { out.data[start + j] /= esum; }
    }
    out
}
// 1×n one-hot
pub fn onehot(tok: usize, n: usize) -> Tensor {
    let mut d = vec![0.0; n];
    if tok < n { d[tok] = 1.0; }
    mk(d, vec![1, n], DType::F64)
}
// 复制行(用于广播/构造)
pub fn tile_row(row: &Tensor, times: usize) -> Tensor {
    let mut out = Tensor::zeros(times, row.c);
    out.dtype = row.dtype;
    for i in 0..times { for j in 0..row.c { out.data[i * row.c + j] = row.data[j]; } }
    out
}
// SGD: p.data -= lr * p.grad
pub fn sgd_step(params: &mut [&mut Tensor], lr: f64) {
    for p in params {
        if p.grad.len() != p.data.len() {
            panic!("tensor: sgd grad len {} != param len {}", p.grad.len(), p.data.len());
        }
        for i in 0..p.len() { p.data[i] -= lr * p.grad[i]; }
    }
}
