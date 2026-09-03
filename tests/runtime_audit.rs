// Runtime Audit — silent-wrong + CPU/GPU numerical consistency hunt.
// Goal: find inputs where code "runs but gives a wrong answer" (not a crash),
// especially non-32-aligned shapes that tiled Vulkan kernels must guard.
use rtorch::gpu::matmul_gpu;
use rtorch::tensor::{self, Tensor};
use std::panic::{AssertUnwindSafe, catch_unwind};

fn cpu_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for kk in 0..k {
                s += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = s;
        }
    }
    c
}

fn check_agreement(m: usize, k: usize, n: usize) -> Result<(), String> {
    let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.023).cos()).collect();
    let cpu = cpu_matmul(&a, &b, m, k, n);
    let gpu = matmul_gpu(&a, &b, m, n, k).map_err(|e| format!("gpu err: {e}"))?;
    if gpu.len() != m * n {
        return Err(format!("len mismatch gpu={} expected {}", gpu.len(), m * n));
    }
    for i in 0..m * n {
        let g = gpu[i];
        let c = cpu[i];
        if g.is_nan() || g.is_infinite() {
            return Err(format!(
                "GPU NaN/Inf at [{i}]: gpu={g} cpu={c} (m={m} k={k} n={n})"
            ));
        }
        if c.is_nan() || c.is_infinite() {
            return Err(format!("CPU NaN/Inf at [{i}]"));
        }
        let abs = (g - c).abs();
        let rel = abs / c.abs().max(1e-6);
        if abs > 1e-1 && rel > 1e-2 {
            return Err(format!(
                "mismatch[{i}] gpu={g} cpu={c} (m={m} k={k} n={n} abs={abs} rel={rel})"
            ));
        }
    }
    Ok(())
}

#[test]
fn gpu_matches_cpu_non_aligned_shapes() {
    // Deliberately non-32-aligned / odd shapes: a bad tiled kernel would either
    // write out of bounds or silently produce wrong values here.
    let shapes = [
        (7, 5, 3),    // tiny odd
        (31, 17, 9),  // not multiple of 32
        (33, 33, 33), // just over a tile
        (65, 3, 47),
        (1, 1, 1),  // 1x1
        (1, 5, 3),  // 1 row
        (17, 1, 9), // k=1
        (200, 200, 64),
    ];
    let mut checked = 0;
    for &(m, k, n) in &shapes {
        if let Err(e) = check_agreement(m, k, n) {
            // GPU unavailable -> accept (skip), real mismatch -> fail.
            if e.contains("gpu err") {
                eprintln!("GPU unavailable (skip): {e}");
                continue;
            }
            panic!("{e}");
        }
        checked += 1;
    }
    eprintln!("[audit] agreed on {checked} non-aligned shapes");
}

#[test]
fn gpu_matmul_len_and_shape_are_correct() {
    let (m, k, n) = (3, 2, 4);
    let a: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..8).map(|i| i as f32 * 0.5).collect();
    match matmul_gpu(&a, &b, m, n, k) {
        Ok(c) => {
            assert_eq!(c.len(), m * n, "output length must be m*n");
        }
        Err(e) => {
            eprintln!("GPU unavailable (skip): {e}");
        }
    }
}

#[test]
fn matmul_bad_len_returns_error_not_silent() {
    // A wrong k (input length mismatch) must yield Err, not silently compute.
    let a = vec![1.0f32; 4];
    let b = vec![1.0f32; 4];
    let r = catch_unwind(AssertUnwindSafe(|| matmul_gpu(&a, &b, 2, 2, 2)));
    // either a clean Err (a.len != m*k) or a panic; must NOT silently succeed.
    match r {
        Ok(Ok(_)) => panic!("matmul_gpu with mismatched input len must not silently succeed"),
        Ok(Err(_)) => {}
        Err(_) => {}
    }
}

// ---- CPU fuzz: random shapes/ops, assert correct-or-clear-error (never silent wrong) ----

fn lcg(seed: &mut u64) -> u64 {
    // deterministic LCG for reproducible fuzz
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed
}

#[test]
fn fuzz_cpu_ops_never_silent_wrong() {
    let mut seed: u64 = 0x1234_5678_9abc_def0;
    for _ in 0..200 {
        let r = (lcg(&mut seed) % 6 + 1) as usize;
        let c = (lcg(&mut seed) % 6 + 1) as usize;
        let a = Tensor::from_data(
            (0..r * c)
                .map(|i| (lcg(&mut seed) % 100) as f64 / 7.0)
                .collect(),
            r,
            c,
        );
        let c2 = (lcg(&mut seed) % 6 + 1) as usize;
        let b = Tensor::from_data(
            (0..c * c2)
                .map(|i| (lcg(&mut seed) % 100) as f64 / 11.0)
                .collect(),
            c,
            c2,
        );
        // matmul must match reference
        let out = tensor::matmul(&a, &b);
        for i in 0..r {
            for j in 0..c2 {
                let mut s = 0.0;
                for k in 0..c {
                    s += a.data[i * c + k] * b.data[k * c2 + j];
                }
                assert!(
                    (out.data[i * c2 + j] - s).abs() < 1e-9,
                    "fuzz matmul mismatch"
                );
            }
        }
        // tanh matches std
        let t = tensor::tanh(&a);
        for i in 0..a.len() {
            assert!((t.data[i] - a.data[i].tanh()).abs() < 1e-12);
        }
        // softmax rows sum to 1
        let s = tensor::softmax_row(&a);
        for row in 0..r {
            let sum: f64 = (0..c).map(|j| s.data[row * c + j]).sum();
            assert!((sum - 1.0).abs() < 1e-9);
        }
    }
}

#[test]
fn boundary_inputs() {
    // NaN/Inf propagate honestly (not silent-wrong): tanh(NaN)=NaN.
    let nan = Tensor::from_data(vec![f64::NAN, 1.0], 2, 1);
    let t = tensor::tanh(&nan);
    assert!(t.data[0].is_nan());
    assert!(!t.data[1].is_nan());

    // Missing kernel -> clean Err (loc::read_kernel), not panic.
    let err = rtorch::loc::read_kernel("definitely_not_a_kernel");
    assert!(err.is_err());

    // 1x1 matmul is a valid identity-ish edge but must not be silent.
    let a = Tensor::from_data(vec![3.0], 1, 1);
    let b = Tensor::from_data(vec![4.0], 1, 1);
    let c = tensor::matmul(&a, &b);
    assert!((c.data[0] - 12.0).abs() < 1e-12);
}
