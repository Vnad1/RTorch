// 阶段① 对拍: GPU Adam (AdamG + adam.comp) vs CPU 手写 Adam reference.
// 同 grad 序列步进 K 次, 比较 param 逐元素误差 + m/v. Run: cargo run --release --bin adam_check
use rtorch::gpu_tensor::GpuContext;
use rtorch::gvar::{self, AdamG, GVar};
use std::rc::Rc;

fn main() {
    let ctx = match GpuContext::new() { Ok(c) => Rc::new(c), Err(e) => { println!("GPU ctx 失败: {e}"); return; } };
    println!("[adam-check] === GPU Adam vs CPU reference ===\n");

    let (pr, pc) = (48usize, 32usize); // param shape (r×c)
    let n = pr * pc;
    let lr = 1e-3f64; let b1 = 0.9f64; let b2 = 0.999f64; let eps = 1e-8f64;

    // CPU reference state (m/v) + param, per step apply identical grad
    let mut p_cpu = vec![0.5f64; n];
    let mut m_cpu = vec![0.0f64; n];
    let mut v_cpu = vec![0.0f64; n];
    // GPU AdamG
    let gctx = Rc::clone(&ctx);
    let gp = gvar::leaf(Rc::clone(&gctx), p_cpu.clone(), pr, pc);
    let mut opt = AdamG::new(lr);

    let steps = 12;
    let mut max_err = 0.0f64;
    for t in 1..=steps {
        // 确定性 grad: 逐步变化的梯度
        let grad: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.01 + t as f64 * 0.1).sin()).collect();
        // CPU Adam (标准)
        let bc1 = 1.0 - b1.powi(t as i32);
        let bc2 = 1.0 - b2.powi(t as i32);
        for i in 0..n {
            m_cpu[i] = b1 * m_cpu[i] + (1.0 - b1) * grad[i];
            v_cpu[i] = b2 * v_cpu[i] + (1.0 - b2) * grad[i] * grad[i];
            let mh = m_cpu[i] / bc1; let vh = v_cpu[i] / bc2;
            p_cpu[i] -= lr * mh / (vh.sqrt() + eps);
        }
        // GPU AdamG: 设 grad(设备 tensor) → opt.step
        let gf: Vec<f32> = grad.iter().map(|&x| x as f32).collect();
        let ggt = rtorch::gpu_tensor::GpuTensor::from_data(Rc::clone(&gctx), &gf, pr, pc);
        gvar::set_grad(&gp, ggt);
        opt.step(&[gp.clone()]);
        // 比较
        let gp_now = gvar::to_vec(&gp);
        let mut e = 0.0f64;
        for i in 0..n { let d = (gp_now[i] - p_cpu[i]).abs(); if d > e { e = d; } }
        if e > max_err { max_err = e; }
        if t == steps {
            println!("step {t}: param max|GPU-CPU| = {e:.3e}  {}", if e < 1e-5 { "PASS" } else { "FAIL" });
            println!("  cpu[0..4]={:?}", &p_cpu[..4]);
            println!("  gpu[0..4]={:?}", &gp_now[..4]);
        }
    }
    println!("全程 max_err = {max_err:.3e}  => {}", if max_err < 1e-5 { "PASS" } else { "FAIL" });
    println!("\n[adam-check] DONE");
}
