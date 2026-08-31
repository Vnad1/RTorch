// 阶段② 训练级验证 v2: batch 录制包裹整链 (前向+GPU CE+反向+AdamG) vs 逐op.
// 关键: CE 用 gpu_tensor::ce (loss 全 GPU, 不下载 B×V); loss 读延迟到 submit 后(调用方结束 batch).
// 同一模型/数据 K 步, 比较 loss 序列一致 + 计时. Run: cargo run --release --bin batch_train
use rtorch::gpu_tensor::{self, GpuContext, GpuTensor};
use rtorch::gvar::{self, AdamG, GVar};
use std::rc::Rc;

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
    // 单个 train_batch: 前向+CE+反向+Adam 全部 GPU. 返回每 step 的 loss GpuTensor(B×1),
    // 调用方在 batch 结束后 read(避免 read-before-execute; 非 batch 时立即可读).
    fn train_batch(&mut self, toks: &[usize], target: &[usize], b: usize, l: usize) -> Vec<GpuTensor> {
        let mut vstate = gvar::leaf(Rc::clone(&self.ctx), vec![0.0; b*self.d], b, self.d);
        let mut loss_ts = Vec::new();
        let mut logits_all = Vec::new();
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

fn sum_loss(loss_ts: &[GpuTensor]) -> f64 { loss_ts.iter().map(|t| t.to_vec()[0] as f64).sum::<f64>() }

fn main() {
    let ctx = match GpuContext::new() { Ok(c) => Rc::new(c), Err(e) => { println!("GPU ctx 失败: {e}"); return; } };
    println!("[batch-train] === batch vs 逐op 训练级验证 v2 ===\n");
    let (d, e, v, b, l) = (32usize, 32usize, 120usize, 8usize, 12usize);
    let mut toks = Vec::new(); let mut target = Vec::new();
    for _ in 0..b { for t in 0..l { toks.push(t % v); target.push((t*3+1) % v); } }

    // A) 逐op (loss 立即读)
    let mut netA = Net::new(Rc::clone(&ctx), d, e, v);
    let t0 = std::time::Instant::now();
    let mut lossesA = Vec::new();
    for _ in 0..10 { let xs = netA.train_batch(&toks, &target, b, l); lossesA.push(sum_loss(&xs) / (b*l) as f64); }
    let tA = t0.elapsed().as_secs_f64();
    // B) batch 包裹整链 (loss 在 end_batch 后读)
    let mut netB = Net::new(Rc::clone(&ctx), d, e, v);
    let t0 = std::time::Instant::now();
    let mut lossesB = Vec::new();
    for _ in 0..10 {
        ctx.begin_batch();
        let xs = netB.train_batch(&toks, &target, b, l);
        ctx.end_batch();
        lossesB.push(sum_loss(&xs) / (b*l) as f64);
    }
    let tB = t0.elapsed().as_secs_f64();

    let mut maxd = 0.0f64;
    for (i, (a, bb)) in lossesA.iter().zip(&lossesB).enumerate() {
        let dd = (a - bb).abs(); if dd > maxd { maxd = dd; }
        println!("  step {i}: A={a:.4}  B={bb:.4}  Δ={dd:.2e}");
    }
    println!("maxΔ(loss) = {maxd:.2e}  => {}", if maxd < 1e-4 { "PASS(语义一致)" } else { "FAIL" });
    println!("耗时(10步): 逐op={tA:.3}s  batch={tB:.3}s  提速 {:.2}×", tA/tB);
    let a_now = gvar::to_vec(&netA.a); let b_now = gvar::to_vec(&netB.a);
    let wmax = a_now.iter().zip(&b_now).map(|(x,y)| (x-y).abs()).fold(0.0f64, f64::max);
    println!("权重 A max|Δ| = {wmax:.3e}  => {}", if wmax < 1e-5 { "PASS" } else { "FAIL" });
    println!("\n[batch-train] DONE");
}
