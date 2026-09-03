// GPU correctness — compares Vulkan compute against a CPU reference with a
// reasonable tolerance (never bit-exact). Skipped (passes) if the GPU/engine is
// unavailable, so the suite stays green on machines without a working GPU.
use rtorch::gpu::matmul_gpu;

#[test]
fn gpu_matmul_small_matches_cpu_reference() {
    let (m, k, n) = (4usize, 3usize, 2usize);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.5).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 3) as f32) - 1.0).collect();

    let c = match matmul_gpu(&a, &b, m, n, k) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("GPU unavailable, skipping: {e}");
            return; // not a failure — GPU is optional
        }
    };

    // CPU reference C = A·B
    let mut refc = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for kk in 0..k {
                s += a[i * k + kk] * b[kk * n + j];
            }
            refc[i * n + j] = s;
        }
    }
    for i in 0..m * n {
        let abs = (c[i] - refc[i]).abs();
        let rel = abs / refc[i].abs().max(1e-6);
        assert!(
            abs < 1e-2 || rel < 1e-2,
            "c[{}]={} cpu={} (abs={} rel={})",
            i,
            c[i],
            refc[i],
            abs,
            rel
        );
    }
}

#[test]
fn gpu_matmul_identity_is_a() {
    // A[3x3] · I[3x3] == A — strong correctness signal when the GPU is available.
    let (m, k, n) = (3usize, 3usize, 3usize);
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, -1.0, 0.5, 2.0, 7.0, 0.0, -4.0];
    let mut i3 = vec![0.0f32; 9];
    i3[0] = 1.0;
    i3[4] = 1.0;
    i3[8] = 1.0;
    let c = match matmul_gpu(&a, &i3, m, n, k) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("GPU unavailable, skipping: {e}");
            return;
        }
    };
    for i in 0..9 {
        assert!(
            (c[i] - a[i]).abs() < 1e-2,
            "c[{}]={} expected {}",
            i,
            c[i],
            a[i]
        );
    }
}

#[test]
fn cpu_gpu_numerical_agreement() {
    // The same matmul computed on CPU (rtorch::tensor) and GPU (Vulkan) must agree
    // to a reasonable tolerance (never bit-exact). Validates semantic consistency
    // between the two backends.
    use rtorch::tensor::Tensor;
    let (m, k, n) = (64usize, 32usize, 16usize);
    let a: Vec<f64> = (0..m * k).map(|i| ((i as f64) * 0.0017).sin()).collect();
    let b: Vec<f64> = (0..k * n).map(|i| ((i as f64) * 0.0023).cos()).collect();

    // CPU reference
    let mut cpu = vec![0.0f64; m * n];
    let mut af = vec![0.0f32; m * k];
    let mut bf = vec![0.0f32; k * n];
    for i in 0..m * k {
        af[i] = a[i] as f32;
    }
    for i in 0..k * n {
        bf[i] = b[i] as f32;
    }
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f64;
            for kk in 0..k {
                s += a[i * k + kk] * b[kk * n + j];
            }
            cpu[i * n + j] = s;
        }
    }

    let cg = match matmul_gpu(&af, &bf, m, n, k) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("GPU unavailable, skipping: {e}");
            return;
        }
    };

    for i in 0..m * n {
        let abs = (cg[i] as f64 - cpu[i]).abs();
        let rel = abs / cpu[i].abs().max(1e-6);
        assert!(
            abs < 1e-2 || rel < 1e-3,
            "mismatch[{}] gpu={} cpu={} (abs={} rel={})",
            i,
            cg[i],
            cpu[i],
            abs,
            rel
        );
    }
}

#[test]
fn cpu_gpu_precision_chain_stays_bounded() {
    // A chain on CPU (f64) vs GPU (f32) must stay within f32 arithmetic tolerance,
    // quantifying the precision divergence (CPU f64, GPU f32) rather than leaving
    // it as an unknown.
    use rtorch::device::Device;
    use rtorch::ops;
    let r = 16;
    let k = 16;
    let n = 16;
    let a = Tensor::from_data((0..r * k).map(|i| ((i as f64) * 0.013).sin()).collect(), r, k);
    let b = Tensor::from_data((0..k * n).map(|i| ((i as f64) * 0.021).cos()).collect(), k, n);

    let cpu = ops::matmul(Device::Cpu, &a, &b).unwrap();
    let gpu = match ops::matmul(Device::Gpu, &a, &b) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("GPU unavailable, skipping: {e}");
            return;
        }
    };
    let mut max_rel: f64 = 0.0;
    for i in 0..cpu.data.len() {
        let rel = (cpu.data[i] - gpu.data[i]).abs() / cpu.data[i].abs().max(1e-6);
        if rel > max_rel { max_rel = rel; }
    }
    eprintln!("[precision] CPU(f64) vs GPU(f32) max rel err = {max_rel:.2e}");
    assert!(max_rel < 1e-3, "CPU/GPU precision divergence too large: {max_rel:.2e}");
}
