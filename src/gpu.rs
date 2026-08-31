// RTorch GPU matmul API — 把 GPU matmul 封装成可复用函数(供 Striker/Var 调用)。
// C = A·B (f32), 维度 M/N/K 经 params 缓冲传入 kernel(不用 spec constants)。
// 每次调用建 session(正确性优先, 性能后优化; 持久 session 是下一步)。

use crate::vk::VkSession;

fn f32_to_bytes(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn u32_to_bytes(v: &[u32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }
fn bytes_to_f32(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() }
fn spv_path() -> String { String::from("D:/AP/rtorch/examples/matmul_var.spv") }

// C[m×n] = A[m×k] · B[k×n] (f32), GPU 计算。返回 C 或 Err。
pub fn matmul_gpu(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Result<Vec<f32>, String> {
    assert_eq!(a.len(), m * k, "A 需 m×k");
    assert_eq!(b.len(), k * n, "B 需 k×n");
    let spv = std::fs::read(spv_path()).map_err(|e| format!("spv: {e}"))?;
    let params: Vec<u32> = vec![m as u32, n as u32, k as u32];
    let ab = f32_to_bytes(a);
    let bb = f32_to_bytes(b);
    let pb = u32_to_bytes(&params);
    let in_sizes = vec![ab.len(), bb.len(), pb.len()];
    let out_len = m * n * 4;
    let mut ses = VkSession::init(&spv, &in_sizes, out_len).map_err(|e| format!("init: {e}"))?;
    let mut out = vec![0u8; out_len];
    let groups: [u32; 3] = [((m as u32) + 15) / 16, ((n as u32) + 15) / 16, 1];
    ses.dispatch(&[ab, bb, pb], groups, &mut out).map_err(|e| format!("dispatch: {e}"))?;
    Ok(bytes_to_f32(&out))
}
