// 阶段④ 对拍: gemm_tile32 (32×32) vs gemm_tiled (16×16) vs CPU. 正确性 + 吞吐.
// Run: cargo run --release --bin gemm32_check
use rtorch::gpu_tensor::{self, GpuContext};
use std::rc::Rc;

fn cpu_matmul(a: &[f32], b: &[f32], r: usize, k: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0; r * c];
    for i in 0..r { for j in 0..c { let mut acc = 0.0; for m in 0..k { acc += a[i*k+m] * b[m*c+j]; } out[i*c+j] = acc; } }
    out
}
fn max_err(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x,y)| (x-y).abs()).fold(0.0, f32::max) }

fn main() {
    let ctx = match GpuContext::new() { Ok(c) => Rc::new(c), Err(e) => { println!("GPU ctx 失败: {e}"); return; } };
    println!("[gemm32-check] === tile32 vs tile16 vs CPU ===\n");
    // 正确性 (随机规模, 非 16/32 倍数, 测 OOB)
    for (r, k, c) in [(3usize,5usize,4usize), (16,16,16), (37,53,29), (64,64,64)] {
        let a: Vec<f32> = (0..r*k).map(|i| ((i as f32)*0.01).sin()).collect();
        let b: Vec<f32> = (0..k*c).map(|i| ((i as f32)*0.02).cos()).collect();
        let ta = gpu_tensor::GpuTensor::from_data(Rc::clone(&ctx), &a, r, k);
        let tb = gpu_tensor::GpuTensor::from_data(Rc::clone(&ctx), &b, k, c);
        let m16 = gpu_tensor::matmul(&ta, &tb).to_vec();
        let m32 = gpu_tensor::matmul32(&ta, &tb).to_vec();
        let refm = cpu_matmul(&a, &b, r, k, c);
        let e16 = max_err(&m16, &refm); let e32 = max_err(&m32, &refm);
        println!("{r}×{k}×{c}: tile16 err={e16:.2e}  tile32 err={e32:.2e}  {}", if e32 < 1e-2 { "PASS" } else { "FAIL" });
    }
    // 吞吐: n×n matmul, tile16 vs tile32 (batch, 无 sync)
    for n in [512usize, 1024usize] {
        let av: Vec<f32> = (0..n*n).map(|i| ((i as f32)*0.001).sin()).collect();
        let bv: Vec<f32> = (0..n*n).map(|i| ((i as f32)*0.001).cos()).collect();
        let ta = gpu_tensor::GpuTensor::from_data(Rc::clone(&ctx), &av, n, n);
        let tb = gpu_tensor::GpuTensor::from_data(Rc::clone(&ctx), &bv, n, n);
        // warm
        let _ = gpu_tensor::matmul(&ta, &tb); let _ = gpu_tensor::matmul32(&ta, &tb);
        // tile16
        let reps = 30;
        ctx.begin_batch();
        for _ in 0..reps { let _ = gpu_tensor::matmul(&ta, &tb); }
        let t0 = std::time::Instant::now();
        ctx.end_batch();
        let dt16 = t0.elapsed().as_secs_f64() / reps as f64;
        // tile32
        ctx.begin_batch();
        for _ in 0..reps { let _ = gpu_tensor::matmul32(&ta, &tb); }
        let t0 = std::time::Instant::now();
        ctx.end_batch();
        let dt32 = t0.elapsed().as_secs_f64() / reps as f64;
        let fl = 2.0*(n as f64).powi(3);
        println!("n={n}: tile16={:.3}ms {:.0}GF  tile32={:.3}ms {:.0}GF  提速 {:.2}×", dt16*1e3, fl/(dt16*1e9), dt32*1e3, fl/(dt32*1e9), dt16/dt32);
    }
    println!("\n[gemm32-check] DONE");
}
