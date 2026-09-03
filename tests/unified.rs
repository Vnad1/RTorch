// CPU/GPU unified op layer — the SAME op call on Device::Cpu and Device::Gpu
// must agree (tolerance, not bit-exact). One signature, no two model paths.
use rtorch::device::Device;
use rtorch::ops;
use rtorch::tensor::Tensor;

#[test]
fn unified_matmul_cpu_gpu_agree() {
    let a = Tensor::from_data((0..16).map(|i| ((i as f64) * 0.13).sin()).collect(), 4, 4);
    let b = Tensor::from_data((0..16).map(|i| ((i as f64) * 0.21).cos()).collect(), 4, 4);
    let cpu = ops::matmul(Device::Cpu, &a, &b).unwrap();
    let gpu = match ops::matmul(Device::Gpu, &a, &b) {
        Ok(g) => g,
        Err(e) => { eprintln!("GPU unavailable, skipping: {e}"); return; }
    };
    assert_eq!((cpu.r, cpu.c), (gpu.r, gpu.c));
    for i in 0..cpu.data.len() {
        let abs = (cpu.data[i] - gpu.data[i]).abs();
        let rel = abs / cpu.data[i].abs().max(1e-6);
        assert!(abs < 1e-2 || rel < 1e-2, "matmul[{}] cpu={} gpu={}", i, cpu.data[i], gpu.data[i]);
    }
}

#[test]
fn unified_add_tanh_agree() {
    let a = Tensor::from_data((0..12).map(|i| (i as f64) * 0.3).collect(), 3, 4);
    let b = Tensor::from_data((0..12).map(|i| (i as f64) * 0.1).collect(), 3, 4);
    let cpu = ops::add(Device::Cpu, &a, &b).unwrap();
    let gpu = match ops::add(Device::Gpu, &a, &b) {
        Ok(g) => g,
        Err(e) => { eprintln!("GPU unavailable, skipping: {e}"); return; }
    };
    for i in 0..cpu.data.len() {
        assert!((cpu.data[i] - gpu.data[i]).abs() < 1e-2, "add[{}] cpu={} gpu={}", i, cpu.data[i], gpu.data[i]);
    }

    let tc = ops::tanh(Device::Cpu, &a).unwrap();
    let tg = match ops::tanh(Device::Gpu, &a) {
        Ok(g) => g,
        Err(e) => { eprintln!("GPU unavailable, skipping: {e}"); return; }
    };
    for i in 0..tc.data.len() {
        assert!((tc.data[i] - tg.data[i]).abs() < 1e-4, "tanh[{}] cpu={} gpu={}", i, tc.data[i], tg.data[i]);
    }
}
