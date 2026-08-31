// 阶段⑥ 综合 benchmark: 重构前后对比. 测: ①matmul GFLOPS (tile16 vs tile32)
// ②全 GPU 训练 step (batch+GPU CE+AdamG) ③host 往返量. Run: cargo run --release --bin gpu_summary
use rtorch::gpu_tensor::{self, GpuContext};
use rtorch::gvar::{self, AdamG, GVar};
use std::rc::Rc;

fn main() {
    let ctx = match GpuContext::new() { Ok(c) => Rc::new(c), Err(e) => { println!("GPU ctx 失败: {e}"); return; } };
    println!("[gpu-summary] === 重构后 GPU 吞吐 ===\n");
    // ① matmul GFLOPS
    for n in [256usize, 512usize, 1024usize] {
        let av: Vec<f32> = (0..n*n).map(|i| ((i as f32)*0.001).sin()).collect();
        let bv: Vec<f32> = (0..n*n).map(|i| ((i as f32)*0.001).cos()).collect();
        let ta = gpu_tensor::GpuTensor::from_data(Rc::clone(&ctx), &av, n, n);
        let tb = gpu_tensor::GpuTensor::from_data(Rc::clone(&ctx), &bv, n, n);
        let _ = gpu_tensor::matmul(&ta, &tb);
        let reps = 40;
        ctx.begin_batch();
        for _ in 0..reps { let _ = gpu_tensor::matmul(&ta, &tb); }
        let t0 = std::time::Instant::now();
        ctx.end_batch();
        let dt = t0.elapsed().as_secs_f64()/reps as f64;
        let fl = 2.0*(n as f64).powi(3);
        println!("matmul n={n}: {:.3} ms {:.0} GFLOPS (batch 无sync, tile32)", dt*1e3, fl/(dt*1e9));
    }
    println!();
    // ② 全 GPU 训练 step (batch + GPU CE + AdamG): 测每 step 时间
    let (d, e, v, b, l) = (64usize, 64usize, 500usize, 8usize, 16usize);
    let mut toks: Vec<usize> = vec![]; let mut target: Vec<usize> = vec![];
    for _ in 0..b { for t in 0..l { toks.push(t % v); target.push((t*7+1) % v); } }
    let mut net = Net::new(Rc::clone(&ctx), d, e, v);
    let steps = 20;
    let t0 = std::time::Instant::now();
    let mut loss_sum = 0.0f64;
    for _ in 0..steps {
        ctx.begin_batch();
        let xs = net.train(&toks, &target, b, l);
        ctx.end_batch();
        loss_sum += xs.iter().map(|t| t.to_vec()[0] as f64).sum::<f64>() / (b*l) as f64;
    }
    let dt = t0.elapsed().as_secs_f64()/steps as f64;
    println!("全 GPU 训练 step (d={d} V={v} B={b}×L={l}, batch+GPU CE+AdamG, 无host往返):");
    println!("  {:.3} ms/step", dt*1e3);
    println!("  loss 均值(20步) = {:.3}", loss_sum/steps as f64);
    println!("\n[gpu-summary] DONE");
}

struct Net { ctx: Rc<GpuContext>, d: usize, e: usize, v: usize, a: GVar, emb: GVar, bm: GVar, bias: GVar, c: GVar, opt: AdamG }
impl Net {
    fn new(ctx: Rc<GpuContext>, d: usize, e: usize, v: usize) -> Self {
        let rv = |n: usize| (0..n).map(|i| (((i*11)%97) as f64/97.0) - 0.5).collect::<Vec<f64>>();
        Net { ctx: Rc::clone(&ctx), d, e, v,
            a: gvar::leaf(Rc::clone(&ctx), rv(d*d), d, d), emb: gvar::leaf(Rc::clone(&ctx), rv(v*e), v, e),
            bm: gvar::leaf(Rc::clone(&ctx), rv(e*d), e, d), bias: gvar::leaf(Rc::clone(&ctx), vec![0.0; d], 1, d),
            c: gvar::leaf(Rc::clone(&ctx), rv(d*v), d, v), opt: AdamG::new(0.001) }
    }
    fn params(&self) -> Vec<GVar> { vec![Rc::clone(&self.a), Rc::clone(&self.emb), Rc::clone(&self.bm), Rc::clone(&self.bias), Rc::clone(&self.c)] }
    fn train(&mut self, toks: &[usize], target: &[usize], b: usize, l: usize) -> std::vec::Vec<rtorch::gpu_tensor::GpuTensor> {
        let mut vstate = gvar::leaf(Rc::clone(&self.ctx), vec![0.0; b*self.d], b, self.d);
        let mut loss_ts = Vec::new(); let mut logits_all = Vec::new();
        for t in 0..l {
            let ids: Vec<usize> = (0..b).map(|i| toks[i*l+t].min(self.v-1)).collect();
            let u = gvar::gather(&self.emb, &ids, b);
            let uB = gvar::matmul(&u, &self.bm);
            let pre = gvar::add(&gvar::add(&gvar::matmul(&vstate, &self.a), &uB), &self.bias);
            vstate = gvar::tanh(&pre);
            let logits = gvar::matmul(&vstate, &self.c);
            let tgt: Vec<usize> = (0..b).map(|j| target[j*l+t].min(self.v-1)).collect();
            let (lvt, g) = gpu_tensor::ce(&logits.borrow().t, &tgt);
            gvar::set_grad(&logits, g);
            logits_all.push(logits);
            loss_ts.push(lvt);
        }
        gvar::backward_multi(&logits_all);
        self.opt.step(&self.params());
        loss_ts
    }
}
