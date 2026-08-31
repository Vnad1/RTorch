// 阶段② 对拍: batch 录制 (一次 submit) vs 逐 op submit+waitIdle 的 GPU 结果一致性.
// 相同 matmul→add→tanh→matmul 链, 两种跑法, 逐元素比较. Run: cargo run --release --bin batch_check
use rtorch::gpu_tensor::{self, GpuContext, GpuTensor};
use std::rc::Rc;

fn main() {
    let ctx = match GpuContext::new() { Ok(c) => Rc::new(c), Err(e) => { println!("GPU ctx 失败: {e}"); return; } };
    println!("[batch-check] === 逐op vs batch 结果一致性 ===\n");

    let (r, k, c) = (16usize, 32usize, 24usize);
    let a: Vec<f32> = (0..r * k).map(|i| ((i as f32) * 0.01).sin()).collect();
    let b: Vec<f32> = (0..k * c).map(|i| ((i as f32) * 0.02).cos()).collect();
    let bias: Vec<f32> = (0..r * c).map(|i| (i as f32) * 0.001).collect();
    let _ = ();
    // 用另一矩阵再链一次 (matmul2 out = c×1)
    let w: Vec<f32> = (0..c).map(|i| ((i as f32) * 0.03).cos()).collect();

    // --- 逐 op (基准) ---
    let ta = GpuTensor::from_data(Rc::clone(&ctx), &a, r, k);
    let tb = GpuTensor::from_data(Rc::clone(&ctx), &b, k, c);
    let tbias = GpuTensor::from_data(Rc::clone(&ctx), &bias, r, c);
    let tw = GpuTensor::from_data(Rc::clone(&ctx), &w, c, 1);
    let m1 = gpu_tensor::matmul(&ta, &tb);
    let ad = gpu_tensor::add(&m1, &tbias);
    let th = gpu_tensor::tanh(&ad);
    let m2 = gpu_tensor::matmul(&th, &tw);
    let base = m2.to_vec();

    // --- batch 模式 (一次 submit) ---
    let ta2 = GpuTensor::from_data(Rc::clone(&ctx), &a, r, k);
    let tb2 = GpuTensor::from_data(Rc::clone(&ctx), &b, k, c);
    let tbias2 = GpuTensor::from_data(Rc::clone(&ctx), &bias, r, c);
    let tw2 = GpuTensor::from_data(Rc::clone(&ctx), &w, c, 1);
    ctx.begin_batch();
    let m1 = gpu_tensor::matmul(&ta2, &tb2);
    let ad = gpu_tensor::add(&m1, &tbias2);
    let th = gpu_tensor::tanh(&ad);
    let m2 = gpu_tensor::matmul(&th, &tw2);
    ctx.end_batch();
    let batchy = m2.to_vec();

    let mut maxe = 0.0f32; let mut w = 0usize;
    for i in 0..base.len() { let d = (base[i] - batchy[i]).abs(); if d > maxe { maxe = d; w = i; } }
    println!("链 matmul→add→tanh→matmul: batch vs 逐op max|Δ| = {maxe:.3e}  => {}", if maxe < 1e-4 { "PASS" } else { "FAIL" });
    println!("  base[{}]={} batch[{}]={}", w, base[w], w, batchy[w]);
    println!("\n[batch-check] DONE");
}
