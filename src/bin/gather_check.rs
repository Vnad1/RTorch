// 阶段①b 对拍: GPU gather backward (gather_scatter atomc) vs CPU hand scatter.
// emb(V×e), ids(B), dc(B×e). 比较 emb.grad. Run: cargo run --release --bin gather_check
use rtorch::gpu_tensor::{self, GpuContext};
use rtorch::gvar::{self, GVar};
use std::rc::Rc;

fn main() {
    let ctx = match GpuContext::new() {
        Ok(c) => Rc::new(c),
        Err(e) => {
            println!("GPU ctx 失败: {e}");
            return;
        }
    };
    println!("[gather-check] === GPU gather backward vs CPU scatter ===\n");
    let (ve, e) = (24usize, 16usize);
    let b = 8usize;
    let emb_host: Vec<f64> = (0..ve * e).map(|i| ((i as f64) * 0.021).sin()).collect();
    let ids: Vec<usize> = vec![3, 7, 3, 12, 7, 20, 3, 5];
    let dc: Vec<f64> = (0..b * e).map(|i| ((i as f64) * 0.013).cos()).collect();

    // GPU: emb leaf + gather fwd + set grad on gather out + backward
    let gctx = Rc::clone(&ctx);
    let gemb = gvar::leaf(Rc::clone(&gctx), emb_host.clone(), ve, e);
    let gout = gvar::gather(&gemb, &ids, b);
    let gf: Vec<f32> = dc.iter().map(|&x| x as f32).collect();
    let gdc = gpu_tensor::GpuTensor::from_data(Rc::clone(&gctx), &gf, b, e);
    gvar::set_grad(&gout, gdc);
    gvar::backward(&gout);
    let gemb_grad = gvar::grad_to_vec(&gemb);

    // CPU reference scatter-add
    let mut refgrad = vec![0.0f64; ve * e];
    for (i, &id) in ids.iter().enumerate() {
        if id >= ve {
            continue;
        }
        for j in 0..e {
            refgrad[id * e + j] += dc[i * e + j];
        }
    }
    let mut maxe = 0.0f64;
    for i in 0..ve * e {
        let d = (gemb_grad[i] - refgrad[i]).abs();
        if d > maxe {
            maxe = d;
        }
    }
    println!(
        "emb.grad max|GPU-CPU| = {maxe:.3e}  => {}",
        if maxe < 1e-5 { "PASS" } else { "FAIL" }
    );
    // 只在 ids 对应行非零
    let mut crow = 0usize;
    for (i, &id) in ids.iter().enumerate() {
        if id == 3 {
            crow = i;
            break;
        }
    }
    println!(
        "row3 grad[..4]: GPU={:?} CPU={:?}",
        &gemb_grad[3 * e..3 * e + 4],
        &refgrad[3 * e..3 * e + 4]
    );
    println!("\n[gather-check] DONE");
}
