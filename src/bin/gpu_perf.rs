// GPU 性能剖析 (gpu_perf) — 量化各类别耗时, 定位真正瓶颈(不猜).
// 测: ①单op固定开销(sync/submit/pipeline重绑) ②matmul纯kernel(tstamp隔离) ③matmul→add→tanh链
//     ④to_vec下载(B×V, softmax/loss路径) ⑤add广播(host往返)  vs  CPU对照.
// Run: RTORCH_VK_TSTAMP=1 cargo run --release --bin gpu_perf
use rtorch::gpu_tensor::{self, GpuContext, GpuTensor};
use std::rc::Rc;

fn main() {
    let ctx = match GpuContext::new() {
        Ok(c) => Rc::new(c),
        Err(e) => {
            println!("GPU ctx 失败: {e}");
            return;
        }
    };
    println!("[gpu-perf] === 分类性能剖析 ===\n");

    // ① 单 op 固定开销: 极小 matmul, 重复 N 次; 纯 kernel 时间极小 → 测到的是 sync/submit/重绑固定成本.
    {
        let n = 16usize;
        let a = vec![1.0f32; n * n];
        let b = vec![1.0f32; n * n];
        let ta = GpuTensor::from_data(Rc::clone(&ctx), &a, n, n);
        let tb = GpuTensor::from_data(Rc::clone(&ctx), &b, n, n);
        let _ = gpu_tensor::matmul(&ta, &tb); // warmup
        let reps = 200;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            let m = gpu_tensor::matmul(&ta, &tb);
            std::hint::black_box(&m);
        }
        let dt = t0.elapsed().as_secs_f64() / reps as f64;
        println!(
            "① 极小微 matmul n={n} x{reps}: {:.3} ms/op  (≈ 单 op 固定开销: sync+submit+重绑)",
            dt * 1e3
        );
    }

    // ② matmul 吞吐 (中/大矩阵, 含 sync 的端到端)
    for n in [256usize, 512usize, 1024usize] {
        let av: Vec<f32> = (0..n * n).map(|i| ((i as f32) * 0.001).sin()).collect();
        let bv: Vec<f32> = (0..n * n).map(|i| ((i as f32) * 0.001).cos()).collect();
        let ta = GpuTensor::from_data(Rc::clone(&ctx), &av, n, n);
        let tb = GpuTensor::from_data(Rc::clone(&ctx), &bv, n, n);
        // GPU 驻留 (不下载) 端到端循环
        let _ = gpu_tensor::matmul(&ta, &tb);
        let reps = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            let m = gpu_tensor::matmul(&ta, &tb);
            std::hint::black_box(&m);
        }
        let dt = t0.elapsed().as_secs_f64() / reps as f64;
        let flops = 2.0 * (n as f64).powi(3);
        println!(
            "② matmul n={n}: {:.3} ms/op -> {:.1} GFLOPS (含sync)",
            dt * 1e3,
            flops / (dt * 1e9)
        );
    }

    // ③ 链 matmul→add→tanh (3 op, 3 sync) vs 单独 matmul 1 op: 看链的额外开销.
    {
        let n = 512usize;
        let av: Vec<f32> = (0..n * n).map(|i| ((i as f32) * 0.001).sin()).collect();
        let bv: Vec<f32> = (0..n * n).map(|i| ((i as f32) * 0.001).cos()).collect();
        let bias: Vec<f32> = (0..n * n).map(|i| ((i as f32) * 0.0005).cos()).collect();
        let ta = GpuTensor::from_data(Rc::clone(&ctx), &av, n, n);
        let tb = GpuTensor::from_data(Rc::clone(&ctx), &bv, n, n);
        let tbias = GpuTensor::from_data(Rc::clone(&ctx), &bias, n, n);
        // 单独 matmul (1 op)
        let _ = gpu_tensor::matmul(&ta, &tb);
        let reps = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            let m = gpu_tensor::matmul(&ta, &tb);
            std::hint::black_box(&m);
        }
        let dt1 = t0.elapsed().as_secs_f64() / reps as f64;
        println!("③ 单 matmul n={n}: {:.3} ms/op", dt1 * 1e3);
        // 链 matmul→add→tanh (3 op)
        let _ = {
            let m = gpu_tensor::matmul(&ta, &tb);
            let x = gpu_tensor::add(&m, &tbias);
            gpu_tensor::tanh(&x);
        };
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            let m = gpu_tensor::matmul(&ta, &tb);
            let x = gpu_tensor::add(&m, &tbias);
            let y = gpu_tensor::tanh(&x);
            std::hint::black_box(&y);
        }
        let dt3 = t0.elapsed().as_secs_f64() / reps as f64;
        println!(
            "③ 链 matmul→add→tanh n={n}: {:.3} ms/op (3 op)  → 平均 {:.3} ms/op  vs 单op {:.3} ms",
            dt3 * 1e3,
            dt3 * 1e3 / 3.0,
            dt1 * 1e3
        );
    }

    // ④ to_vec 下载成本 (B×V, softmax/loss 路径): B=128 V=3401 (train25m 配置)
    {
        let b = 128usize;
        let v = 3401usize;
        let data: Vec<f32> = (0..b * v).map(|i| ((i as f32) * 0.001).sin()).collect();
        let t = GpuTensor::from_data(Rc::clone(&ctx), &data, b, v);
        let _ = t.to_vec(); // warmup
        let reps = 30;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            let h = t.to_vec();
            std::hint::black_box(h);
        }
        let dt = t0.elapsed().as_secs_f64() / reps as f64;
        let mb = (b * v * 4) as f64 / 1e6;
        println!(
            "④ to_vec 下载 B={b} V={v} ({mb:.0} MB): {:.3} ms/次  {:.1} GB/s",
            dt * 1e3,
            mb / dt / 1000.0
        );
    }

    // ⑤ add 广播成本 (host 往返: to_vec→host tile→重传): bias 1×n → B×n
    {
        let b = 128usize;
        let n = 512usize;
        let data: Vec<f32> = (0..b * n).map(|i| ((i as f32) * 0.001).cos()).collect();
        let bias: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.0005).sin()).collect();
        let ta = GpuTensor::from_data(Rc::clone(&ctx), &data, b, n);
        let tb = GpuTensor::from_data(Rc::clone(&ctx), &bias, 1, n);
        let _ = gpu_tensor::add(&ta, &tb);
        let reps = 30;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            let x = gpu_tensor::add(&ta, &tb);
            std::hint::black_box(&x);
        }
        let dt = t0.elapsed().as_secs_f64() / reps as f64;
        println!(
            "⑤ add 广播 bias(1×{n})→B={b}: {dt:.3} ms/次  (含 host tile+重传往返: 若有大量add广播是痛点)"
        );
    }

    // ⑦ softmax GPU (row-wise, B×V) correctness + perf vs CPU
    {
        let b = 128usize;
        let v = 3401usize;
        let data: Vec<f32> = (0..b * v).map(|i| ((i as f32) * 0.001).sin()).collect();
        let t = GpuTensor::from_data(Rc::clone(&ctx), &data, b, v);
        let s = gpu_tensor::softmax(&t);
        let sh = s.to_vec();
        // CPU 参考
        let mut maxe = 0.0f32;
        for r in 0..b {
            let row = &data[r * v..(r + 1) * v];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let ex: Vec<f32> = row.iter().map(|x| (x - m).exp()).collect();
            let ssum: f32 = ex.iter().sum();
            for c0 in 0..v {
                let refv = ex[c0] / ssum;
                maxe = maxe.max((sh[r * v + c0] - refv).abs());
            }
        }
        println!(
            "⑦ softmax GPU B={b} V={v}: max_err={maxe:.3e} {}",
            if maxe < 1e-3 { "PASS" } else { "FAIL" }
        );
        let _ = gpu_tensor::softmax(&t);
        let reps = 30;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            let s = gpu_tensor::softmax(&t);
            std::hint::black_box(&s);
        }
        let dt = t0.elapsed().as_secs_f64() / reps as f64;
        println!(
            "① softmax GPU 时间: {:.3} ms/次 (vs to_vec 下载 5.6 ms → GPU softmax 免下载)",
            dt * 1e3
        );
    }

    // ⑥ CPU 对照 matmul (同规模, 参考)
    {
        let n = 512usize;
        let a: Vec<f32> = (0..n * n).map(|i| ((i as f32) * 0.001).sin()).collect();
        let b: Vec<f32> = (0..n * n).map(|i| ((i as f32) * 0.001).cos()).collect();
        let t0 = std::time::Instant::now();
        let reps = 10;
        for _ in 0..reps {
            let mut out = vec![0.0f32; n * n];
            for i in 0..n {
                for j in 0..n {
                    let mut acc = 0.0;
                    for k in 0..n {
                        acc += a[i * n + k] * b[k * n + j];
                    }
                    out[i * n + j] = acc;
                }
            }
            std::hint::black_box(&out);
        }
        let dt = t0.elapsed().as_secs_f64() / reps as f64;
        println!(
            "⑥ CPU matmul n={n}: {:.1} ms/op -> {:.1} GFLOPS (对照)",
            dt * 1e3,
            2.0 * (n as f64).powi(3) / (dt * 1e9)
        );
    }

    println!("[gpu-perf] DONE (RTORCH_VK_TSTAMP=1 可隔离纯 kernel 时间)");
}
