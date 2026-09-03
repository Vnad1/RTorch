# RTorch

**RTorch** is a universal compute framework — a general-purpose numerical engine that lets you describe a mathematical formula, a state transition, a custom kernel, or a training step, and have RTorch execute it on the **CPU** or a **Vulkan GPU**.

It is **not** a model library and **not** a Transformer/Deep-Learning framework. RTorch is the *compute layer*: it provides tensors, autograd, optimizers, a formula/kernel system, a device runtime, and a self-contained artifact format. A model framework (such as **Striker**) is built *on top of* RTorch, not inside it.

## What it provides

- **Tensor** — CPU host tensor (`rtorch::tensor`) with `Shape` / `DType` (F32/F64/F16/BF16/I32), `shape`/`numel`/`reshape`/`view`/`transpose`, and broadcasting.
- **Autograd** — differentiable graphs on CPU (`rtorch::autograd`) and on device-resident GPU (`rtorch::gvar`), with `backward`, gradient accumulation over shared parameters, and an Adam optimizer (`rtorch::autograd::Adam`, `rtorch::gvar::AdamG`).
- **Formula / Kernel system** — a C ABI (`rtorch.h`) so any `formula.cpp` implements `rtorch_output_size` + `rtorch_compute`; the framework compiles it and runs it on CPU or GPU. GPU paths can supply a GLSL compute kernel. See `rtorch.h`.
- **Device runtime** — `rtorch::device::Device` (Cpu/Gpu) and a unified op layer (`rtorch::ops`) so models do not need two implementations.
- **Vulkan / GPU** — a C++ Vulkan compute engine (`rtorch_vk.dll`) with a persistent/device-resident model and a `KernelRegistry`; no CUDA dependency (cross-vendor Vulkan Compute).
- **RTW (.rtw)** — a self-contained artifact format for results, kernels, models and memory (`rtw::`, `RTW.md`).

## Building

Requirements: a stable Rust toolchain, a C++ compiler (`g++` or `clang`), and (for GPU) the Vulkan SDK (headers + `vulkan-1.lib`).

```sh
git clone https://github.com/Vnad1/RTorch.git
cd RTorch
cargo build --release
```

`cargo build --release` is the single build entry point: it builds the Rust library and the `rtorch` command, and `build.rs` compiles the C++ Vulkan engine into `rtorch_vk.dll` and the GLSL kernels (`examples/*.comp`) into `target/release/kernels/*.spv`.

Environment overrides: `RTORCH_BUILD_ENGINE` (0 = CPU-only), `RTORCH_GXX`, `VULKAN_SDK`, `RTORCH_GLSLANG`.

## Usage (CLI)

Run a formula on the CPU:

```sh
rtorch examples/formula_verify.cpp --input examples/input_1000.bin --device cpu
```

Run a formula on the GPU (if the formula exports a GLSL kernel):

```sh
rtorch examples/formula_gpu.cpp --device gpu --input examples/input_1000.bin
```

`rtorch --help` and `rtorch --version` are available. Errors are reported with stable exit codes (2 = usage, 1 = runtime).

## Usage (library)

Add `rtorch` as a path dependency. Models use the unified op layer or the autograd API:

```rust
use rtorch::ops;                        // uniform CPU/GPU op dispatch
use rtorch::device::Device;
use rtorch::autograd::{self, Adam};     // CPU autograd + optimizer
use rtorch::gvar::{self, AdamG};        // device-resident GPU autograd
use rtorch::rtw;                        // .rtw container
```

See `tests/` for runnable examples (tensor, autograd, RTW, GPU, formula ABI, model cycle).

## Testing

```sh
cargo test --release
cargo clippy --release --all-targets
```

The test suite covers tensor correctness, autograd (including a central-difference numerical-gradient check), CPU/GPU numerical agreement, RTW round-trips, formula C-ABI edge cases, and GPU device-resident training convergence.

## Notes

- **Contributor:** [Vnad1](https://github.com/Vnad1/RTorch) is the sole contributor.
- **AI-generated code:** This repository contains AI-assisted source code. The code has been reviewed and corrected: unreasonable patterns were removed and defects (including silent broadcasting bugs, Vulkan API misuse, and resource/lifetime issues) were fixed and are covered by tests and sanitizer/validation checks.
- **License:** GNU LGPL-3.0 (see [LICENSE](./LICENSE)).

## License

This project is licensed under the **GNU Lesser General Public License, version 3.0 (LGPL-3.0)** — see [LICENSE](./LICENSE).

LGPL-3.0 applies to the RTorch library itself. Applications that **use** RTorch (e.g. a model framework built on it) may be under another license, provided they allow relinking/replacing the RTorch library portion (see the LGPL-3.0 terms). User **formulas** that only include the public interface header (`rtorch.h`) are not derivative works of the RTorch implementation and are not bound by this license (see `rtorch.h` header note and LGPL-3.0 §3).

## Related

- `RTW.md` — the `.rtw` artifact format specification.
- A model layer built on RTorch (see the Striker project) is kept separate.

---

[简体中文 README](./README.zh-CN.md)
