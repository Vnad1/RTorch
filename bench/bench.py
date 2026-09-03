#!/usr/bin/env python3
"""RTorch GPU throughput benchmark: RTorch (Vulkan, GLSL) vs PyTorch (cuBLAS).
Same op/size measured on the same GPU. Reports GFLOPS for each side.

Honest comparison: same problem size, same data, same device. PyTorch here is
used ONLY as a performance reference (a scale), never as RTorch's implementation.
"""
import subprocess, sys, time, struct
from pathlib import Path

import torch

M, K, N = 1024, 1024, 1024
RTORCH = Path(r"D:\AP\rtorch\target\release\rtorch.exe")
VAR = Path(r"D:\AP\rtorch\examples")

def make_inputs():
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randn(K, N, dtype=torch.float32)
    Al = A.reshape(-1).numpy().tobytes()
    Bl = B.reshape(-1).numpy().tobytes()
    a_path = VAR / "bench_A.bin"
    b_path = VAR / "bench_B.bin"
    a_path.write_bytes(Al)
    b_path.write_bytes(Bl)
    return A, B, a_path, b_path

def run_rtorch(a_path, b_path, reps=3):
    flops = 2.0 * M * K * N
    times = []
    ref = None
    # warmup + reps; recompile each run (framework cold) — measure dispatch time.
    for _ in range(reps):
        cmd = [str(RTORCH), str(VAR / "bench_matmul.cpp"),
               "--device", "gpu",
               "--input", str(a_path), "--input", str(b_path),
               "--output", str(VAR / "bench_out.bin")]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        last = None
        for line in r.stderr.splitlines():
            if "elapsed=" in line:
                last = float(line.split("elapsed=")[1].split()[0])
        if last is None:
            print(r.stderr); raise RuntimeError("no elapsed in rtorch output")
        times.append(last)
    out = (VAR / "bench_out.bin").read_bytes()
    vals = [struct.unpack_from("<f", out, i)[0] for i in range(0, len(out), 4)]
    return min(times), vals, flops

def ref_matmul(A, B, reps=20):
    A, B = A.cuda(), B.cuda()
    _ = A @ B  # warmup
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); C = A @ B; e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e))
    return best, C.cpu(), 2.0 * M * K * N

if __name__ == "__main__":
    A, B, a_path, b_path = make_inputs()
    print(f"problem: C[{M}x{N}] = A[{M}x{K}] @ B[{K}x{N}], fp32, {2*M*K*N/1e9:.2f} GFLOP")

    t_rt, vals_rt, flops = run_rtorch(a_path, b_path)
    t_pt, C, _ = ref_matmul(A, B)

    # correctness: compare RTorch output vs PyTorch (tolerance)
    Ct = C.reshape(-1)
    n = min(len(vals_rt), Ct.numel())
    err = max(abs(vals_rt[i] - Ct[i].item()) for i in range(n))
    print(f"\n  RTorch : {t_rt:.3f} ms  -> {flops/ (t_rt/1e3) /1e9:.1f} GFLOPS")
    print(f"  PyTorch: {t_pt:.3f} ms  -> {flops/ (t_pt/1e3) /1e9:.1f} GFLOPS (cuBLAS)")
    print(f"  RTorch speedup vs PyTorch: {t_pt/t_rt:.2f}x   correctness max|err|={err:.3e}")
