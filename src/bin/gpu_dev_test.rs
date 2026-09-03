// P1 validation: device-resident GPU tensor chain (matmul -> add -> tanh)
// matches the CPU reference with NO host copies in the chain, plus a throughput
// probe at scale. Run: cargo run --release --bin gpu_dev_test

use rtorch::gpu_tensor::{self, GpuContext, GpuTensor};

fn cpu_matmul(a: &[f32], b: &[f32], r: usize, k: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0; r * c];
    for i in 0..r {
        for j in 0..c {
            let mut acc = 0.0;
            for m in 0..k {
                acc += a[i * k + m] * b[m * c + j];
            }
            out[i * c + j] = acc;
        }
    }
    out
}

fn max_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn main() {
    println!("[gpu-dev-test] creating device context...");
    let ctx = match GpuContext::new() {
        Ok(c) => Rc::new(c),
        Err(e) => {
            eprintln!("context failed: {e}");
            std::process::exit(1);
        }
    };
    println!("[gpu-dev-test] device ok");

    // ---- tiny clean integer matmul to validate path ----
    {
        let (r, k, c) = (3usize, 3usize, 3usize);
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let ta = GpuTensor::from_data(Rc::clone(&ctx), &a, r, k);
        let tb = GpuTensor::from_data(Rc::clone(&ctx), &b, k, c);
        let x = gpu_tensor::matmul(&ta, &tb);
        let xh = x.to_vec();
        println!("[gpu-dev-test] tiny matmul gpu = {:?}", xh);
        println!(
            "[gpu-dev-test] tiny matmul cpu = {:?}",
            cpu_matmul(&a, &b, r, k, c)
        );
    }

    // ---- correctness: chain matmul -> add -> tanh ----
    let (r, k, c) = (64usize, 96usize, 32usize);
    let a: Vec<f32> = (0..r * k).map(|i| ((i as f32) * 0.01).sin()).collect();
    let b: Vec<f32> = (0..k * c).map(|i| ((i as f32) * 0.02).cos()).collect();
    let bias: Vec<f32> = (0..r * c).map(|i| (i as f32) * 0.001 - 0.05).collect();

    let ta = GpuTensor::from_data(Rc::clone(&ctx), &a, r, k);
    let tb = GpuTensor::from_data(Rc::clone(&ctx), &b, k, c);
    let tbias = GpuTensor::from_data(Rc::clone(&ctx), &bias, r, c);

    // device chain (all device-side; intermediates stay on GPU)
    let x = gpu_tensor::matmul(&ta, &tb);
    let x_host = x.to_vec();
    let mm = cpu_matmul(&a, &b, r, k, c);
    println!(
        "[gpu-dev-test] matmul alone: max_err={:.3e}",
        max_err(&x_host, &mm)
    );
    let nz: Vec<usize> = x_host
        .iter()
        .enumerate()
        .filter(|(_, v)| v.abs() < 1e-4)
        .map(|(i, _)| i)
        .collect();
    println!(
        "   #zero-ish={} first10={:?}",
        nz.len(),
        &nz[..nz.len().min(10)]
    );
    let mut worst = 0usize;
    let mut we = 0.0f32;
    for i in 0..x_host.len() {
        let e = (x_host[i] - mm[i]).abs();
        if e > we {
            we = e;
            worst = i;
        }
    }
    println!(
        "   worst idx={worst} (i={},j={}) gpu={} cpu={}",
        worst / c,
        worst % c,
        x_host[worst],
        mm[worst]
    );
    println!("   gpu[0..6] = {:?}", &x_host[..6.min(x_host.len())]);
    println!("   cpu[0..6] = {:?}", &mm[..6.min(mm.len())]);
    println!("   a[0..3] = {:?}", &a[..3]);
    println!("   b[0..3] = {:?}", &b[..3]);

    let x2 = gpu_tensor::add(&x, &tbias);
    let x2_host = x2.to_vec();
    let mut ab = vec![0.0; r * c];
    for i in 0..r * c {
        ab[i] = mm[i] + bias[i];
    }
    println!(
        "[gpu-dev-test] add alone: max_err={:.3e}",
        max_err(&x2_host, &ab)
    );

    let y = gpu_tensor::tanh(&x2);
    let y_host = y.to_vec();
    let mut refv = vec![0.0; r * c];
    for i in 0..r * c {
        refv[i] = (mm[i] + bias[i]).tanh();
    }
    let err = max_err(&y_host, &refv);
    println!(
        "[gpu-dev-test] chain matmul->add->tanh: max_err={:.3e}  {}",
        err,
        if err < 1e-3 { "PASS" } else { "FAIL" }
    );

    // ---- backward kernels: matmul + tanh ----
    {
        let (m, k, n) = (5usize, 4usize, 3usize);
        let av: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.37).sin()).collect();
        let bv: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.11).cos()).collect();
        let dcv: Vec<f32> = (0..m * n).map(|i| ((i as f32) * 0.07).sin()).collect();
        let ta = GpuTensor::from_data(Rc::clone(&ctx), &av, m, k);
        let tb = GpuTensor::from_data(Rc::clone(&ctx), &bv, k, n);
        let tdc = GpuTensor::from_data(Rc::clone(&ctx), &dcv, m, n);
        let (da, db) = gpu_tensor::matmul_backward(&ta, &tb, &tdc);
        let da_h = da.to_vec();
        let db_h = db.to_vec();
        // CPU refs
        let mut cda = vec![0.0f32; m * k];
        let mut cdb = vec![0.0f32; k * n];
        for i in 0..m {
            for t in 0..k {
                let mut s = 0.0;
                for j in 0..n {
                    s += dcv[i * n + j] * bv[t * n + j];
                }
                cda[i * k + t] = s;
            }
        }
        for t in 0..k {
            for j in 0..n {
                let mut s = 0.0;
                for i in 0..m {
                    s += av[i * k + t] * dcv[i * n + j];
                }
                cdb[t * n + j] = s;
            }
        }
        println!(
            "[gpu-dev-test] matmul_backward dA: max_err={:.3e}  {}",
            max_err(&da_h, &cda),
            if max_err(&da_h, &cda) < 1e-3 {
                "PASS"
            } else {
                "FAIL"
            }
        );
        println!(
            "[gpu-dev-test] matmul_backward dB: max_err={:.3e}  {}",
            max_err(&db_h, &cdb),
            if max_err(&db_h, &cdb) < 1e-3 {
                "PASS"
            } else {
                "FAIL"
            }
        );

        // tanh backward
        let xv: Vec<f32> = (0..16).map(|i| ((i as f32) * 0.31).sin()).collect();
        let a_tanh: Vec<f32> = xv.iter().map(|&x| x.tanh()).collect();
        let dcv2: Vec<f32> = (0..16).map(|i| ((i as f32) * 0.5).cos()).collect();
        let tx = GpuTensor::from_data(Rc::clone(&ctx), &xv, 16, 1);
        let ttanh = gpu_tensor::tanh(&tx); // a = tanh(x) on device
        let tdc2 = GpuTensor::from_data(Rc::clone(&ctx), &dcv2, 16, 1);
        let dtanh = gpu_tensor::tanh_backward(&tdc2, &ttanh);
        let dtanh_h = dtanh.to_vec();
        let ref_tanh: Vec<f32> = (0..16)
            .map(|i| dcv2[i] * (1.0 - a_tanh[i] * a_tanh[i]))
            .collect();
        println!(
            "[gpu-dev-test] tanh_backward: max_err={:.3e}  {}",
            max_err(&dtanh_h, &ref_tanh),
            if max_err(&dtanh_h, &ref_tanh) < 1e-3 {
                "PASS"
            } else {
                "FAIL"
            }
        );
    }

    // ---- t=1 analytic: dA[i,k]=(1-v1[i]^2)*v0[k]; dB[i,l]=(1-v1[i]^2)*u[l] ----
    {
        use rtorch::gvar;
        let d = 10usize;
        let v = 6usize;
        let a_host: Vec<f64> = (0..d * d).map(|i| ((i as f64) * 0.021).sin()).collect();
        let b_host: Vec<f64> = (0..d * v).map(|i| ((i as f64) * 0.017).cos()).collect();
        let v0: Vec<f64> = (0..d).map(|i| ((i as f64) * 0.05).sin()).collect();
        let u: Vec<f64> = {
            let mut m = vec![0.0; v];
            m[2] = 1.0;
            m
        };
        let gctx = Rc::new(GpuContext::new().expect("ctx"));
        let gA = gvar::leaf(Rc::clone(&gctx), a_host.clone(), d, d);
        let gB = gvar::leaf(Rc::clone(&gctx), b_host.clone(), d, v);
        let gv0 = gvar::leaf(Rc::clone(&gctx), v0.clone(), d, 1);
        let gu = gvar::leaf(Rc::clone(&gctx), u.clone(), v, 1);
        let z = gvar::add(&gvar::matmul(&gA, &gv0), &gvar::matmul(&gB, &gu));
        let v1 = gvar::tanh(&z);
        let ones = gpu_tensor::GpuTensor::from_data(Rc::clone(&gctx), &vec![1.0f32; d], d, 1);
        gvar::set_grad(&v1, ones);
        gvar::backward(&v1);
        let dA = gvar::grad_to_vec(&gA);
        let dB = gvar::grad_to_vec(&gB);
        // v1 = tanh(A v0 + B u)
        let mut v1r = vec![0.0f64; d];
        for i in 0..d {
            let mut s = 0.0;
            for k in 0..d {
                s += a_host[i * d + k] * v0[k];
            }
            for l in 0..v {
                s += b_host[i * v + l] * u[l];
            }
            v1r[i] = s.tanh();
        }
        let mut eda = 0.0f64;
        let mut edn = 0.0f64;
        for i in 0..d {
            for k in 0..d {
                let exp = (1.0 - v1r[i] * v1r[i]) * v0[k];
                let got = dA[i * d + k];
                eda = eda.max((got - exp).abs());
                edn = edn.max(exp.abs());
            }
        }
        println!(
            "[gpu-dev-test] t=1 analytic dA: max_err={:.3e} (scale {:.3e}) {}",
            eda,
            edn,
            if eda < 1e-2 * edn.max(1.0) {
                "PASS"
            } else {
                "FAIL"
            }
        );
    }

    // ---- P3/P4: StateNet cell via GVar (GPU) matches CPU autograd ----
    {
        use rtorch::gvar;
        let d = 24usize;
        let v = 12usize;
        let t = 4usize; // dim, vocab, steps
        // parameters (A, B) + sequence of onehot u and initial v0
        let a_host: Vec<f64> = (0..d * d).map(|i| ((i as f64) * 0.013).sin()).collect();
        let b_host: Vec<f64> = (0..d * v).map(|i| ((i as f64) * 0.019).cos()).collect();
        let v0_host: Vec<f64> = (0..d).map(|i| ((i as f64) * 0.07).sin()).collect();
        let u_host: Vec<Vec<f64>> = (0..t)
            .map(|ti| {
                let mut u = vec![0.0; v];
                u[(ti + 1) % v] = 1.0;
                u
            })
            .collect();

        // ---- GPU GVar ----
        let gctx = GpuContext::new().expect("gpu ctx");
        let gctx = Rc::new(gctx);
        let gA = gvar::leaf(Rc::clone(&gctx), a_host.clone(), d, d);
        let gB = gvar::leaf(Rc::clone(&gctx), b_host.clone(), d, v);
        let mut gv = gvar::leaf(Rc::clone(&gctx), v0_host.clone(), d, 1);
        let mut gvs = Vec::new();
        for ti in 0..t {
            let gu = gvar::leaf(Rc::clone(&gctx), u_host[ti].clone(), v, 1);
            let z = gvar::add(&gvar::matmul(&gA, &gv), &gvar::matmul(&gB, &gu));
            gv = gvar::tanh(&z);
            gvs.push(gv.clone());
        }
        let gloss = gvar::scale(1.0, &gv); // loss = 1*last, grad=ones flows back
        let gn = gloss.borrow().t.r * gloss.borrow().t.c;
        let ones_v = vec![1.0f32; gn];
        let ones = gpu_tensor::GpuTensor::from_data(Rc::clone(&gctx), &ones_v, gn, 1);
        gvar::set_grad(&gloss, ones);
        gvar::backward(&gloss);
        println!(
            "[gpu-dev-test] backward done: A grad={} B grad={}",
            gA.borrow().grad.is_some(),
            gB.borrow().grad.is_some()
        );
        let gA_grad = if gA.borrow().grad.is_some() {
            gvar::grad_to_vec(&gA)
        } else {
            vec![]
        };
        let gB_grad = if gB.borrow().grad.is_some() {
            gvar::grad_to_vec(&gB)
        } else {
            vec![]
        };
        let gA_max = gA_grad.iter().fold(0.0f64, |a, &v| a.max(v.abs()));

        // ---- manual BPTT reference (correct tanh grad: (1 - v^2)) ----
        let mut vseq: Vec<Vec<f64>> = vec![v0_host.clone()];
        for ti in 0..t {
            let vp = &vseq[ti];
            let mut z = vec![0.0; d];
            for i in 0..d {
                let mut s = 0.0;
                for k in 0..d {
                    s += a_host[i * d + k] * vp[k];
                }
                for l in 0..v {
                    s += b_host[i * v + l] * u_host[ti][l];
                }
                z[i] = s;
            }
            vseq.push(z.iter().map(|&x| x.tanh()).collect());
        }
        let mut refA = vec![0.0f64; d * d];
        let mut refB = vec![0.0f64; d * v];
        let mut g = vec![1.0f64; d];
        for ti in (0..t).rev() {
            let vt = &vseq[ti + 1];
            let mut gz = vec![0.0; d];
            for i in 0..d {
                gz[i] = g[i] * (1.0 - vt[i] * vt[i]);
            }
            let vp = &vseq[ti];
            for i in 0..d {
                for k in 0..d {
                    refA[i * d + k] += gz[i] * vp[k];
                }
            }
            for i in 0..d {
                for l in 0..v {
                    refB[i * v + l] += gz[i] * u_host[ti][l];
                }
            }
            let mut gnew = vec![0.0; d];
            for k in 0..d {
                let mut s = 0.0;
                for i in 0..d {
                    s += a_host[i * d + k] * gz[i];
                }
                gnew[k] = s;
            }
            g = gnew;
        }

        let eA = max_err(
            &gA_grad.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            &refA.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        );
        let eB = max_err(
            &gB_grad.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            &refB.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        );
        println!(
            "[gpu-dev-test] StateNet BPTT t={t} d={d} (manual ref): gradA max_err={:.3e} {} | gradB max_err={:.3e} {}",
            eA,
            if eA < 1e-2 { "PASS" } else { "FAIL" },
            eB,
            if eB < 1e-2 { "PASS" } else { "FAIL" }
        );

        // ---- CPU autograd regression check (tanh backward uses v^2 now) ----
        use rtorch::autograd as ca;
        let cA = ca::leaf(a_host.clone(), d, d);
        let cB = ca::leaf(b_host.clone(), d, v);
        let mut cv = ca::leaf(v0_host.clone(), d, 1);
        for ti in 0..t {
            let cu = ca::leaf(u_host[ti].clone(), v, 1);
            let z = ca::add(&ca::matmul(&cA, &cv), &ca::matmul(&cB, &cu));
            cv = ca::tanh(&z);
        }
        let cvn = cv.borrow().data.len();
        cv.borrow_mut().grad = vec![1.0; cvn];
        ca::backward(&cv);
        let cA_grad = cA.borrow().grad.clone();
        let cB_grad = cB.borrow().grad.clone();
        let ceA = max_err(
            &cA_grad.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            &refA.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        );
        let ceB = max_err(
            &cB_grad.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            &refB.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        );
        println!(
            "[gpu-dev-test] CPU autograd t={t}: gradA max_err={:.3e} {} | gradB max_err={:.3e} {}",
            ceA,
            if ceA < 1e-2 { "PASS" } else { "FAIL" },
            ceB,
            if ceB < 1e-2 { "PASS" } else { "FAIL" }
        );
    }

    // ---- throughput probe: N×N matmul device-side ----
    for n in [256usize, 512usize, 1024usize] {
        let av: Vec<f32> = (0..n * n).map(|i| ((i as f32) * 0.001).sin()).collect();
        let bv: Vec<f32> = (0..n * n).map(|i| ((i as f32) * 0.001).cos()).collect();
        let am = GpuTensor::from_data(Rc::clone(&ctx), &av, n, n);
        let bm = GpuTensor::from_data(Rc::clone(&ctx), &bv, n, n);
        // warmup
        let _ = gpu_tensor::matmul(&am, &bm);
        let t0 = std::time::Instant::now();
        let reps = 20;
        for _ in 0..reps {
            let m = gpu_tensor::matmul(&am, &bm);
            std::hint::black_box(&m);
        }
        let dt = t0.elapsed().as_secs_f64() / reps as f64;
        let flops = 2.0 * (n as f64).powi(3);
        let gflops = flops / (dt * 1e9);
        println!(
            "[gpu-dev-test] matmul n={n}: {:.3} ms -> {:.1} GFLOPS",
            dt * 1e3,
            gflops
        );
    }

    println!("[gpu-dev-test] DONE");
}

// Helper: wrap Rc not in std prelude 2021 (edition 2021 lacks prelude Rc? no, Rc is in prelude).
// Use explicit alias to avoid confusion.
use std::rc::Rc;
