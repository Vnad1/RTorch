// RTorch 张量库最小内核(tensor + 前向 ops + 梯度容器 + SGD)。
// MVP: 2D 张量(r×c, 1D 用 c=1), 前向 matmul/matvec/add/scal/tanh/softmax/onehot,
//   标量梯度容器(.grad 由模型层手动/后续 autograd 填充), SGD 优化器。
// Striker 模型框架用本 API 做计算。将来补 autograd 图 + GPU(现有 vk kernel)。

#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<f64>,
    pub grad: Vec<f64>,
    pub r: usize,
    pub c: usize,
}
pub fn sigmoid(z: f64) -> f64 { 1.0 / (1.0 + (-z).exp()) }

impl Tensor {
    pub fn zeros(r: usize, c: usize) -> Self { Tensor { data: vec![0.0; r * c], grad: vec![0.0; r * c], r, c } }
    pub fn ones(r: usize, c: usize) -> Self { Tensor { data: vec![1.0; r * c], grad: vec![0.0; r * c], r, c } }
    pub fn from_data(data: Vec<f64>, r: usize, c: usize) -> Self { let n = data.len(); Tensor { data, grad: vec![0.0; n], r, c } }
    pub fn len(&self) -> usize { self.data.len() }
    pub fn zero_grad(&mut self) { for g in &mut self.grad { *g = 0.0; } }
    pub fn get(&self, i: usize, j: usize) -> f64 { self.data[i * self.c + j] }
}

// out[r×c] = A[r×k] · B[k×c]
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let (r, k, c) = (a.r, a.c, b.c);
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
// 逐元素相加(同形广播: 若 b 是 1×1 or 同形)
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let mut out = Tensor::zeros(a.r.max(b.r), a.c.max(b.c));
    for i in 0..out.r {
        for j in 0..out.c {
            let va = a.data[(i % a.r) * a.c + (j % a.c)];
            let vb = b.data[(i % b.r) * b.c + (j % b.c)];
            out.data[i * out.c + j] = va + vb;
        }
    }
    out
}
pub fn scal(s: f64, x: &Tensor) -> Tensor { Tensor { data: x.data.iter().map(|v| s * v).collect(), grad: vec![0.0; x.len()], r: x.r, c: x.c } }
pub fn tanh(x: &Tensor) -> Tensor { Tensor { data: x.data.iter().map(|v| v.tanh()).collect(), grad: vec![0.0; x.len()], r: x.r, c: x.c } }
// 行 softmax
pub fn softmax_row(x: &Tensor) -> Tensor {
    let mut out = Tensor::zeros(x.r, x.c);
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
    Tensor { data: d, grad: vec![0.0; n], r: 1, c: n }
}
// 复制行(用于广播/构造)
pub fn tile_row(row: &Tensor, times: usize) -> Tensor {
    let mut out = Tensor::zeros(times, row.c);
    for i in 0..times { for j in 0..row.c { out.data[i * row.c + j] = row.data[j]; } }
    out
}
// SGD: p.data -= lr * p.grad
pub fn sgd_step(params: &mut [&mut Tensor], lr: f64) {
    for p in params {
        for i in 0..p.len() { p.data[i] -= lr * p.grad[i]; }
    }
}
