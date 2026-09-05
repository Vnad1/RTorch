// RTorch autograd — 最小自动微分张量(Var) + Adam 优化器。
// Var = Rc<RefCell<VarData>>, 前向 op 记录计算图(parents), backward 拓扑反向填充 .grad。
// 支持 matmul/add/scale/tanh/sigmoid/mul/gather/stack_rows, 行 softmax(softmax_row)。
// 交叉熵不是这里的 op: 标量损失由调用方(如 Striker 模型层)用 ce_loss 生成, 见 crate::train。
// 与 tensor.rs(纯数据容器)并存; GPU 接 tensor 后续(vk kernel 挂到 op)。

use std::cell::RefCell;
use std::rc::Rc;

pub struct VarData {
    pub data: Vec<f64>,
    pub grad: Vec<f64>,
    pub r: usize,
    pub c: usize,
    pub parent: Option<Rc<Node>>,
}
pub type Var = Rc<RefCell<VarData>>;

enum Node {
    MatMul(Var, Var, usize, usize, usize), // a, b, r, k, c  (a:r×k, b:k×c)
    Add(Var, Var, usize, usize),           // a, b, r, c
    Scale(Var, f64, usize, usize),
    Tanh(Var, usize, usize),
    Sigmoid(Var, usize, usize),
    Mul(Var, Var, usize, usize),   // elementwise a★b (广播)
    Gather(Var, usize, usize),     // e, idx, c  (查表: row idx of e(V×c) -> 1×c)
    Stack(Vec<Var>, usize, usize), // rows(b×c 行向量) -> B×c (批处理合并)
}

pub fn leaf(data: Vec<f64>, r: usize, c: usize) -> Var {
    Rc::new(RefCell::new(VarData {
        grad: vec![0.0; data.len()],
        data,
        r,
        c,
        parent: None,
    }))
}
pub fn zeros(r: usize, c: usize) -> Var {
    leaf(vec![0.0; r * c], r, c)
}
pub fn from_data(d: Vec<f64>, r: usize, c: usize) -> Var {
    leaf(d, r, c)
}
fn wrap(d: Vec<f64>, r: usize, c: usize, p: Node) -> Var {
    Rc::new(RefCell::new(VarData {
        grad: vec![0.0; d.len()],
        data: d,
        r,
        c,
        parent: Some(Rc::new(p)),
    }))
}
pub fn len(v: &Var) -> usize {
    v.borrow().data.len()
}

// out[i,j] = Σ_k a[i,k]*b[k,j]
pub fn matmul(a: &Var, b: &Var) -> Var {
    let (ar, ac, bc) = (a.borrow().r, a.borrow().c, b.borrow().c);
    let mut out = vec![0.0; ar * bc];
    for i in 0..ar {
        for j in 0..bc {
            let mut acc = 0.0;
            for k in 0..ac {
                acc += a.borrow().data[i * ac + k] * b.borrow().data[k * bc + j];
            }
            out[i * bc + j] = acc;
        }
    }
    wrap(
        out,
        ar,
        bc,
        Node::MatMul(Rc::clone(a), Rc::clone(b), ar, ac, bc),
    )
}
pub fn add(a: &Var, b: &Var) -> Var {
    let (ar, ac) = (a.borrow().r, a.borrow().c);
    let (br, bc) = (b.borrow().r, b.borrow().c);
    let (r, c) = crate::tensor::bcast_shape(ar, ac, br, bc).unwrap_or_else(|e| panic!("{e}"));
    let mut out = vec![0.0; r * c];
    for i in 0..r {
        for j in 0..c {
            let va = a.borrow().data[(i % ar) * ac + (j % ac)];
            let vb = b.borrow().data[(i % br) * bc + (j % bc)];
            out[i * c + j] = va + vb;
        }
    }
    wrap(out, r, c, Node::Add(Rc::clone(a), Rc::clone(b), r, c))
}
pub fn scale(s: f64, a: &Var) -> Var {
    let (r, c) = (a.borrow().r, a.borrow().c);
    let out = a.borrow().data.iter().map(|v| s * v).collect();
    wrap(out, r, c, Node::Scale(Rc::clone(a), s, r, c))
}
pub fn tanh(a: &Var) -> Var {
    let (r, c) = (a.borrow().r, a.borrow().c);
    let out = a.borrow().data.iter().map(|v| v.tanh()).collect();
    wrap(out, r, c, Node::Tanh(Rc::clone(a), r, c))
}
pub fn sigmoid(a: &Var) -> Var {
    let (r, c) = (a.borrow().r, a.borrow().c);
    let out = a
        .borrow()
        .data
        .iter()
        .map(|v| 1.0 / (1.0 + (-v).exp()))
        .collect();
    wrap(out, r, c, Node::Sigmoid(Rc::clone(a), r, c))
}
// 逐元素乘(真广播): out[i,j] = a[i%ar,j%ac] * b[i%br,j%bc]; 不相容报 ShapeError.
pub fn mul(a: &Var, b: &Var) -> Var {
    let (ar, ac) = (a.borrow().r, a.borrow().c);
    let (br, bc) = (b.borrow().r, b.borrow().c);
    let (r, c) = crate::tensor::bcast_shape(ar, ac, br, bc).unwrap_or_else(|e| panic!("{e}"));
    let mut out = vec![0.0; r * c];
    for i in 0..r {
        for j in 0..c {
            let va = a.borrow().data[(i % ar) * ac + (j % ac)];
            let vb = b.borrow().data[(i % br) * bc + (j % bc)];
            out[i * c + j] = va * vb;
        }
    }
    wrap(out, r, c, Node::Mul(Rc::clone(a), Rc::clone(b), r, c))
}
// 查表: out(1×c) = e 的第 idx 行(快: O(c), 替代 onehot·matmul 的 O(V·c))。越界报错(不再静默钳到末行).
pub fn gather(e: &Var, idx: usize) -> Var {
    let (v, c) = (e.borrow().r, e.borrow().c);
    if v == 0 {
        panic!("tensor: gather on empty embedding (0 rows)");
    }
    if idx >= v {
        panic!("tensor: gather index {idx} out of range for embedding size {v}");
    }
    let i = idx;
    let out = (0..c).map(|j| e.borrow().data[i * c + j]).collect();
    wrap(out, 1, c, Node::Gather(Rc::clone(e), i, c))
}
// 把多个行向量(每个 1×c)堆成 B×c 矩阵(批处理)。backward 把每行梯度分回各自 row。
pub fn stack_rows(rows: &[Var]) -> Var {
    let c = rows[0].borrow().c;
    let b = rows.len();
    let mut out = vec![0.0; b * c];
    for (i, r) in rows.iter().enumerate() {
        let d = r.borrow();
        for j in 0..c {
            out[i * c + j] = d.data[j];
        }
    }
    wrap(out, b, c, Node::Stack(rows.to_vec(), b, c))
}
// 行 softmax -> Var(1×n)
pub fn softmax_row(a: &Var) -> Vec<f64> {
    let ad = a.borrow();
    let n = ad.data.len();
    let m = ad.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut e: Vec<f64> = ad.data.iter().map(|x| (x - m).exp()).collect();
    let s: f64 = e.iter().sum();
    for p in &mut e {
        *p /= s;
    }
    e
}

// 拓扑反向: 从 output 反向沿 parents 填 grad。
pub fn backward(out: &Var) {
    // 置 output grad = 1(若标量损失已 set 则用). 这里约定 caller 已设 out.grad。
    let mut order: Vec<Var> = Vec::new();
    let mut visited = std::collections::HashSet::new();
    fn dfs(v: &Var, order: &mut Vec<Var>, visited: &mut std::collections::HashSet<usize>) {
        let id = Rc::as_ptr(v) as usize;
        if !visited.insert(id) {
            return;
        }
        if let Some(p) = v.borrow().parent.clone() {
            match &*p {
                Node::MatMul(a, b, ..) | Node::Add(a, b, ..) | Node::Mul(a, b, ..) => {
                    dfs(a, order, visited);
                    dfs(b, order, visited);
                }
                Node::Scale(a, ..)
                | Node::Tanh(a, ..)
                | Node::Sigmoid(a, ..)
                | Node::Gather(a, ..) => {
                    dfs(a, order, visited);
                }
                Node::Stack(rows, ..) => {
                    for r in rows {
                        dfs(r, order, visited);
                    }
                }
            }
        }
        order.push(Rc::clone(v));
    }
    dfs(out, &mut order, &mut visited);
    for v in order.iter().rev() {
        let (outg, parent) = {
            let b = v.borrow();
            (b.grad.clone(), b.parent.clone())
        };
        let Some(p) = parent else { continue };
        match &*p {
            Node::MatMul(a, b, r, k, c) => {
                let (ag, bg) = (a.borrow().grad.clone(), b.borrow().grad.clone());
                // dA[i,t] += Σ_j go[i,j] * B[t,j];  dB[t,j] += Σ_i A[i,t]*go[i,j]
                for i in 0..*r {
                    for t in 0..*k {
                        let mut s = 0.0;
                        for j in 0..*c {
                            s += outg[i * *c + j] * b.borrow().data[t * *c + j];
                        }
                        a.borrow_mut().grad[i * *k + t] += s;
                    }
                }
                for t in 0..*k {
                    for j in 0..*c {
                        let mut s = 0.0;
                        for i in 0..*r {
                            s += a.borrow().data[i * *k + t] * outg[i * *c + j];
                        }
                        b.borrow_mut().grad[t * *c + j] += s;
                    }
                }
                let _ = (ag, bg);
            }
            Node::Add(a, b, r, c) => {
                let (ar, ac) = (a.borrow().r, a.borrow().c);
                let (br, bc) = (b.borrow().r, b.borrow().c);
                for i in 0..*r {
                    for j in 0..*c {
                        let gv = outg[i * *c + j];
                        let ai = (i % ar) * ac + (j % ac);
                        let bi = (i % br) * bc + (j % bc);
                        a.borrow_mut().grad[ai] += gv;
                        b.borrow_mut().grad[bi] += gv;
                    }
                }
            }
            Node::Mul(a, b, r, c) => {
                for i in 0..*r {
                    for j in 0..*c {
                        let gv = outg[i * *c + j];
                        let ai = (i % a.borrow().r) * a.borrow().c + (j % a.borrow().c);
                        let bi = (i % b.borrow().r) * b.borrow().c + (j % b.borrow().c);
                        a.borrow_mut().grad[ai] += gv * b.borrow().data[bi];
                        b.borrow_mut().grad[bi] += gv * a.borrow().data[ai];
                    }
                }
            }
            Node::Scale(a, s, r, cc) => {
                for i in 0..*r {
                    for j in 0..*cc {
                        a.borrow_mut().grad[i * *cc + j] += s * outg[i * *cc + j];
                    }
                }
            }
            Node::Tanh(a, r, c) => {
                // d tanh(z)/dz = 1 - tanh(z)^2 = 1 - v^2, where v is the node's
                // OWN (output) value, not the input z stored in `a`.
                for i in 0..*r {
                    for j in 0..*c {
                        let x = v.borrow().data[i * *c + j];
                        a.borrow_mut().grad[i * *c + j] += outg[i * *c + j] * (1.0 - x * x);
                    }
                }
            }
            Node::Sigmoid(a, r, c) => {
                for i in 0..*r {
                    for j in 0..*c {
                        let x = a.borrow().data[i * *c + j];
                        let s = 1.0 / (1.0 + (-x).exp());
                        a.borrow_mut().grad[i * *c + j] += outg[i * *c + j] * s * (1.0 - s);
                    }
                }
            }
            Node::Gather(e, i, c) => {
                for j in 0..*c {
                    e.borrow_mut().grad[*i * *c + j] += outg[j];
                }
            }
            Node::Stack(rows, b, c) => {
                for i in 0..*b {
                    for j in 0..*c {
                        rows[i].borrow_mut().grad[j] += outg[i * *c + j];
                    }
                }
            }
        }
    }
}
// 清空所有可达节点梯度(从 output 沿图)
pub fn zero_grad(out: &Var) {
    let mut visited = std::collections::HashSet::new();
    fn dfs(v: &Var, visited: &mut std::collections::HashSet<usize>) {
        let id = Rc::as_ptr(v) as usize;
        if !visited.insert(id) {
            return;
        }
        for g in &mut v.borrow_mut().grad {
            *g = 0.0;
        }
        if let Some(p) = v.borrow().parent.clone() {
            match &*p {
                Node::MatMul(a, b, ..) | Node::Add(a, b, ..) | Node::Mul(a, b, ..) => {
                    dfs(a, visited);
                    dfs(b, visited);
                }
                Node::Scale(a, ..)
                | Node::Tanh(a, ..)
                | Node::Sigmoid(a, ..)
                | Node::Gather(a, ..) => {
                    dfs(a, visited);
                }
                Node::Stack(rows, ..) => {
                    for r in rows {
                        dfs(r, visited);
                    }
                }
            }
        }
    }
    dfs(out, &mut visited);
}

// ---- Adam 优化器 ----
pub struct Adam {
    pub lr: f64,
    pub b1: f64,
    pub b2: f64,
    pub eps: f64,
    m: std::collections::HashMap<usize, Vec<f64>>,
    v: std::collections::HashMap<usize, Vec<f64>>,
    t: u64,
}
impl Adam {
    pub fn new(lr: f64) -> Self {
        Adam {
            lr,
            b1: 0.9,
            b2: 0.999,
            eps: 1e-8,
            m: Default::default(),
            v: Default::default(),
            t: 0,
        }
    }
    // 更新一组可训练叶(Var 无权父), 用其 .grad
    pub fn step(&mut self, params: &[Var]) {
        self.t += 1;
        let b1 = self.b1;
        let b2 = self.b2;
        let bc1 = 1.0 - b1.powi(self.t as i32);
        let bc2 = 1.0 - b2.powi(self.t as i32);
        for p in params {
            let id = Rc::as_ptr(p) as usize;
            let g = p.borrow().grad.clone();
            let n = p.borrow().data.len();
            if g.len() != n {
                panic!("tensor: adam grad len {} != param len {}", g.len(), n);
            }
            let m = self.m.entry(id).or_insert_with(|| vec![0.0; g.len()]);
            let v = self.v.entry(id).or_insert_with(|| vec![0.0; g.len()]);
            for i in 0..g.len() {
                m[i] = b1 * m[i] + (1.0 - b1) * g[i];
                v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
                let mh = m[i] / bc1;
                let vh = v[i] / bc2;
                let upd = self.lr * mh / (vh.sqrt() + self.eps);
                p.borrow_mut().data[i] -= upd;
            }
        }
    }

    // ==== 状态导出/导入 (V1 RTW "保存→退出→恢复→继续学习"): m/v/t 按 params 顺序. ====
    // 导出: 对 params 逐个取 m/v(缺则全零), 存为 Vec<Vec<f64>>; t 单独返回.
    pub fn state(&self, params: &[Var]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, u64) {
        let mut ms = Vec::new();
        let mut vs = Vec::new();
        for p in params {
            let id = Rc::as_ptr(p) as usize;
            let n = p.borrow().data.len();
            ms.push(self.m.get(&id).cloned().unwrap_or_else(|| vec![0.0; n]));
            vs.push(self.v.get(&id).cloned().unwrap_or_else(|| vec![0.0; n]));
        }
        (ms, vs, self.t)
    }
    // 导入: 按 params 顺序把 m/v 重建进 HashMap(地址不同 key, 需顺序对应). t 恢复.
    pub fn load_state(&mut self, params: &[Var], ms: &[Vec<f64>], vs: &[Vec<f64>], t: u64) {
        let mut nms = std::collections::HashMap::new();
        let mut nvs = std::collections::HashMap::new();
        for (i, p) in params.iter().enumerate() {
            let id = Rc::as_ptr(p) as usize;
            let n = p.borrow().data.len();
            let mm = ms.get(i).map(|x| x.clone()).unwrap_or_else(|| vec![0.0; n]);
            let vv = vs.get(i).map(|x| x.clone()).unwrap_or_else(|| vec![0.0; n]);
            // 长度对齐
            let mm = if mm.len() == n {
                mm
            } else {
                let mut z = vec![0.0; n];
                z.copy_from_slice(&mm[..mm.len().min(n)]);
                z
            };
            let vv = if vv.len() == n {
                vv
            } else {
                let mut z = vec![0.0; n];
                z.copy_from_slice(&vv[..vv.len().min(n)]);
                z
            };
            nms.insert(id, mm);
            nvs.insert(id, vv);
        }
        self.m = nms;
        self.v = nvs;
        self.t = t;
    }
}
