// GPU 瓶颈隔离剖析 (gpu_prof2) — 分离三因, 不猜. 直接操作 GpuDevice(绕过高层 GpuTensor 封装).
//  A) 端到端单 op: bind+submit+waitIdle (现主路径)   B) batch: 一次 submit 打包 R 次 dispatch (一次 waitIdle)
//  C) host 往返: upload/download 带宽                 E) GEMM 纯 kernel 吞吐
// Run: cargo run --release --bin gpu_prof2
use rtorch::gpu_tensor::GpuContext;
use std::rc::Rc;

fn spv(name: &str) -> Vec<u8> {
    rtorch::loc::read_kernel(name).unwrap_or_else(|e| panic!("spv {name}: {e}"))
}

fn main() {
    let ctx = match GpuContext::new() {
        Ok(c) => Rc::new(c),
        Err(e) => {
            println!("GPU ctx 失败: {e}");
            return;
        }
    };
    let dev = &ctx.dev;
    println!("[gpu-prof2] === GPU 瓶颈隔离 ===\n");

    // ---- A vs B: matmul n=512, 单 op end-to-end vs batch ----
    for n in [256usize, 512usize, 1024usize] {
        let av: Vec<f32> = (0..n * n).map(|i| ((i as f32) * 0.001).sin()).collect();
        let bv: Vec<f32> = (0..n * n).map(|i| ((i as f32) * 0.001).cos()).collect();
        let abuf = dev.alloc(n * n * 4);
        let bbuf = dev.alloc(n * n * 4);
        let cbuf = dev.alloc(n * n * 4);
        dev.upload(abuf, &f32bytes(&av));
        dev.upload(bbuf, &f32bytes(&bv));
        let params = u32bytes(&[n as u32, n as u32, n as u32]);
        let pbuf = dev.alloc(params.len());
        dev.upload(pbuf, &params);
        let pipe = dev.pipe_add(
            &spv("gemm_tiled"),
            &[abuf, bbuf, pbuf],
            cbuf,
            [((n as u32) + 15) / 16, ((n as u32) + 15) / 16, 1],
        );
        if pipe < 0 {
            println!("pipe_add failed");
            return;
        }
        // warm
        dev.pipe_bind(pipe, &[abuf, bbuf, pbuf], cbuf).unwrap();
        dev.pipe_run(pipe, groups(n)).unwrap();

        // A) per-op end-to-end (bind + submit + waitIdle each), R reps
        let a = |reps: usize| -> f64 {
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                dev.pipe_bind(pipe, &[abuf, bbuf, pbuf], cbuf).unwrap();
                dev.pipe_run(pipe, groups(n)).unwrap();
            }
            t0.elapsed().as_secs_f64() / reps as f64
        };
        // B) batch: R dispatches in one submit (one waitIdle)
        let b = |reps: usize| -> f64 {
            let t0 = std::time::Instant::now();
            dev.dev_begin().unwrap();
            for _ in 0..reps {
                dev.pipe_bind(pipe, &[abuf, bbuf, pbuf], cbuf).unwrap();
                dev.dev_pipe_record(pipe, groups(n)).unwrap();
            }
            dev.dev_submit(true).unwrap();
            t0.elapsed().as_secs_f64() / reps as f64
        };
        let rA = 20;
        let tA = a(rA);
        let tB = b(200);
        let flops = 2.0 * (n as f64).powi(3);
        println!(
            "matmul n={n}: A(单op,含waitIdle)={:.3} ms  B(batch一次submit)={:.3} ms",
            tA * 1e3,
            tB * 1e3
        );
        println!(
            "   → A GFLOPS={:.0}  B GFLOPS={:.0}  |  batch 相对端到端提速 {:.0}×  (若B远小于A → sync是主因)",
            flops / (tA * 1e9),
            flops / (tB * 1e9),
            tA / tB
        );
        dev.free(abuf);
        dev.free(bbuf);
        dev.free(cbuf);
        dev.free(pbuf);
    }
    println!();

    // ---- C) host 往返: upload / download 带宽 ----
    for mb in [1usize, 4usize, 16usize] {
        let bytes = mb * 1024 * 1024;
        let data: Vec<u8> = (0..bytes).map(|i| (i & 0xFF) as u8).collect();
        let buf = dev.alloc(bytes);
        let t0 = std::time::Instant::now();
        for _ in 0..5 {
            dev.upload(buf, &data);
        }
        let tu = t0.elapsed().as_secs_f64() / 5.0;
        let t0 = std::time::Instant::now();
        let mut out = vec![0u8; bytes];
        for _ in 0..5 {
            dev.download(buf, &mut out);
        }
        let td = t0.elapsed().as_secs_f64() / 5.0;
        dev.free(buf);
        println!(
            "host 往返 {mb} MB: upload={:.3} ms ({:.1} GB/s)  download={:.3} ms ({:.1} GB/s)",
            tu * 1e3,
            mb as f64 / 1e3 / tu,
            td * 1e3,
            mb as f64 / 1e3 / td
        );
    }
    println!();

    // ---- E) GEMM 纯 kernel 吞吐 (n=1024, batch 无 sync) ----
    {
        let n = 1024usize;
        let av: Vec<f32> = (0..n * n).map(|i| ((i as f32) * 0.001).sin()).collect();
        let bv: Vec<f32> = (0..n * n).map(|i| ((i as f32) * 0.001).cos()).collect();
        let abuf = dev.alloc(n * n * 4);
        let bbuf = dev.alloc(n * n * 4);
        let cbuf = dev.alloc(n * n * 4);
        dev.upload(abuf, &f32bytes(&av));
        dev.upload(bbuf, &f32bytes(&bv));
        let params = u32bytes(&[n as u32, n as u32, n as u32]);
        let pbuf = dev.alloc(params.len());
        dev.upload(pbuf, &params);
        let pipe = dev.pipe_add(&spv("gemm_tiled"), &[abuf, bbuf, pbuf], cbuf, groups(n));
        dev.pipe_bind(pipe, &[abuf, bbuf, pbuf], cbuf).unwrap();
        let reps = 100;
        dev.dev_begin().unwrap();
        for _ in 0..reps {
            dev.dev_pipe_record(pipe, groups(n)).unwrap();
        }
        let t0 = std::time::Instant::now();
        dev.dev_submit(true).unwrap();
        let dt = t0.elapsed().as_secs_f64() / reps as f64;
        let flops = 2.0 * (n as f64).powi(3);
        println!(
            "GEMM 纯 kernel n={n} (batch{reps} 一次submit): {:.3} ms/op -> {:.0} GFLOPS",
            dt * 1e3,
            flops / (dt * 1e9)
        );
        dev.free(abuf);
        dev.free(bbuf);
        dev.free(cbuf);
        dev.free(pbuf);
    }
    println!("\n[gpu-prof2] DONE");
}

fn groups(n: usize) -> [u32; 3] {
    [((n as u32) + 15) / 16, ((n as u32) + 15) / 16, 1]
}
fn f32bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u32bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
