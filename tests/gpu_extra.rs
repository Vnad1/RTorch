// GPU device-resident correctness beyond matmul/add/tanh:
//   1) end-to-end GVar + AdamG training convergence (loss must fall)
//   2) CPU/GPU numerical agreement for the ce, reduce, gather_scatter kernels.
// All skipped cleanly if the GPU/engine is unavailable.
use rtorch::gpu_tensor::{self, GpuContext, GpuTensor};
use rtorch::gvar::{self, AdamG};
use std::rc::Rc;

fn ctx() -> Option<Rc<GpuContext>> {
    match GpuContext::new() {
        Ok(c) => Some(Rc::new(c)),
        Err(e) => { eprintln!("GPU unavailable: {e}"); None }
    }
}

#[test]
fn gpu_training_converges() {
    let Some(ctx) = ctx() else { return };
    // Fit 2x with a tiny learnable linear map pred = x·w + b on the GPU.
    let x = gvar::leaf(ctx.clone(), vec![1.0, 2.0, 3.0, 4.0], 4, 1);
    let w = gvar::leaf(ctx.clone(), vec![0.0], 1, 1);
    let b = gvar::leaf(ctx.clone(), vec![0.0], 1, 1);
    let target = vec![2.0, 4.0, 6.0, 8.0];
    let mut opt = AdamG::new(0.05);
    let mut loss = f64::INFINITY;
    for _ in 0..300 {
        let pred = gvar::add(&gvar::matmul(&x, &w), &b);
        let p = gvar::to_vec(&pred);
        loss = p.iter().zip(&target).map(|(a, t)| (a - t) * (a - t)).sum::<f64>() / p.len() as f64;
        let grad: Vec<f32> = p.iter().zip(&target).map(|(a, t)| (2.0 * (a - t)) as f32).collect();
        let gt = GpuTensor::from_data(ctx.clone(), &grad, 4, 1);
        gvar::set_grad(&pred, gt);
        gvar::backward(&pred);
        opt.step(&[w.clone(), b.clone()]);
    }
    eprintln!("[gpu-train] final loss = {loss}");
    assert!(loss < 1e-2, "GPU training did not converge: loss {loss}");
    assert!((gvar::to_vec(&w)[0] - 2.0).abs() < 0.2, "W={} expected ~2", gvar::to_vec(&w)[0]);
}

#[test]
fn gpu_ce_matches_cpu_reference() {
    let Some(ctx) = ctx() else { return };
    let logits = vec![1.0f32, 2.0, 3.0, 0.5, -0.5, 1.0]; // 2x3
    let lg = GpuTensor::from_data(ctx.clone(), &logits, 2, 3);
    let (loss_t, grad) = gpu_tensor::ce(&lg, &[0, 2]);
    let loss = loss_t.to_vec();
    let gradf = grad.to_vec();

    // CPU reference: softmax over each row, CE, and grad = softmax - onehot.
    for row in 0..2 {
        let s = &logits[row * 3..row * 3 + 3];
        let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = s.iter().map(|x| (x - m).exp()).collect();
        let es: f32 = e.iter().sum();
        let sm: Vec<f32> = e.iter().map(|x| x / es).collect();
        let loss_ref = -sm[target_of(row)].ln();
        assert!((loss[row] - loss_ref).abs() < 1e-3, "ce loss row {row}: gpu {} cpu {}", loss[row], loss_ref);
        for j in 0..3 {
            let g_ref = sm[j] - if j == target_of(row) { 1.0 } else { 0.0 };
            assert!((gradf[row * 3 + j] - g_ref).abs() < 1e-3, "ce grad [{row},{j}] gpu {} cpu {}", gradf[row * 3 + j], g_ref);
        }
    }
    fn target_of(row: usize) -> usize { if row == 0 { 0 } else { 2 } }
}

#[test]
fn gpu_reduce_matches_cpu_sum() {
    let Some(ctx) = ctx() else { return };
    // reduce_sum 4x2 -> 1x2 (sum rows), a broadcast-add-backward reduction.
    let d: Vec<f32> = (0..8).map(|i| i as f32 + 0.5).collect(); // 4x2
    let dc = GpuTensor::from_data(ctx.clone(), &d, 4, 2);
    let r = gpu_tensor::reduce_sum(&dc, 1, 2);
    let rf = r.to_vec();
    for j in 0..2 {
        let s: f32 = (0..4).map(|i| d[i * 2 + j]).sum();
        assert!((rf[j] - s).abs() < 1e-3, "reduce[{}] gpu {} cpu {}", j, rf[j], s);
    }
}

#[test]
fn gpu_gather_backward_matches_cpu_scatter() {
    let Some(ctx) = ctx() else { return };
    // emb Vxe, ids B, dc Bxe -> scatter-add grad rows back into emb.
    let rows = 3; let e = 2;
    let emb = GpuTensor::from_data(ctx.clone(), &[0.0; 6], rows, e);
    let ids = [0usize, 2, 1];
    let dc = GpuTensor::from_data(ctx.clone(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, e);
    let eg = gpu_tensor::gather_backward(&emb, &ids, &dc, 3);
    let egf = eg.to_vec();
    // CPU reference
    let mut cpu = vec![0.0f32; rows * e];
    for i in 0..3 { for j in 0..e { cpu[ids[i] * e + j] += dc.to_vec()[i * e + j]; } }
    for k in 0..rows * e {
        assert!((egf[k] - cpu[k]).abs() < 1e-3, "gather_backward[{}] gpu {} cpu {}", k, egf[k], cpu[k]);
    }
}
