// RTorch unified op layer — one set of op names that dispatch to the CPU or
// Vulkan GPU backend behind a single `Device` + `Tensor` signature. A model
// writer no longer maintains two compute paths. CPU computes directly (f64);
// GPU uploads to device, computes, and downloads back to a host Tensor (the
// first correctness-oriented unified surface — device-resident GVar stays the
// optimized path for scale).

use crate::device::Device;
use crate::gpu_tensor::{self, GpuContext};
use crate::tensor::{self, DType, Tensor};
use std::rc::Rc;

fn gpu_ctx() -> Result<Rc<GpuContext>, String> {
    GpuContext::new()
        .map(Rc::new)
        .map_err(|e| format!("vulkan: {e}"))
}

fn gpu_from_tensor(ctx: &Rc<GpuContext>, t: &Tensor) -> Result<gpu_tensor::GpuTensor, String> {
    let data: Vec<f32> = t.data.iter().map(|&x| x as f32).collect();
    Ok(gpu_tensor::GpuTensor::from_data(
        Rc::clone(ctx),
        &data,
        t.r,
        t.c,
    ))
}

fn gpu_to_tensor(ctx: &Rc<GpuContext>, g: &gpu_tensor::GpuTensor) -> Result<Tensor, String> {
    let res: Vec<f32> = g.to_vec();
    let data: Vec<f64> = res.iter().map(|&x| x as f64).collect();
    let dims = vec![g.r, g.c];
    Tensor::from_shape(data, dims, DType::F64).map_err(|e| e)
}

/// out[r×c] = a[r×k] · b[k×c], on the selected device. Returns a host Tensor.
pub fn matmul(device: Device, a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    match device {
        Device::Cpu => Ok(tensor::matmul(a, b)),
        Device::Gpu => {
            let ctx = gpu_ctx()?;
            let ga = gpu_from_tensor(&ctx, a)?;
            let gb = gpu_from_tensor(&ctx, b)?;
            let gc = gpu_tensor::matmul(&ga, &gb);
            gpu_to_tensor(&ctx, &gc)
        }
    }
}

/// out = a + b (elementwise, broadcast), on the selected device.
pub fn add(device: Device, a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    match device {
        Device::Cpu => Ok(tensor::add(a, b)),
        Device::Gpu => {
            let ctx = gpu_ctx()?;
            let ga = gpu_from_tensor(&ctx, a)?;
            let gb = gpu_from_tensor(&ctx, b)?;
            let gc = gpu_tensor::add(&ga, &gb);
            gpu_to_tensor(&ctx, &gc)
        }
    }
}

/// out = tanh(a), on the selected device.
pub fn tanh(device: Device, a: &Tensor) -> Result<Tensor, String> {
    match device {
        Device::Cpu => Ok(tensor::tanh(a)),
        Device::Gpu => {
            let ctx = gpu_ctx()?;
            let ga = gpu_from_tensor(&ctx, a)?;
            let gc = gpu_tensor::tanh(&ga);
            gpu_to_tensor(&ctx, &gc)
        }
    }
}

/// out = s * a, on the selected device.
pub fn scale(device: Device, s: f64, a: &Tensor) -> Result<Tensor, String> {
    match device {
        Device::Cpu => Ok(tensor::scal(s, a)),
        Device::Gpu => {
            let ctx = gpu_ctx()?;
            let ga = gpu_from_tensor(&ctx, a)?;
            let gc = gpu_tensor::scale(s as f32, &ga);
            gpu_to_tensor(&ctx, &gc)
        }
    }
}
