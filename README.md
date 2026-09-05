# RTorch

**RTorch** is a universal compute framework — a general-purpose numerical engine that lets you describe a mathematical formula, a state transition, a custom kernel, or a training step, and have RTorch execute it on the **CPU** or a **Vulkan GPU**.

It is **not** a model library and **not** a Transformer/Deep-Learning framework. RTorch is the *compute layer*: it provides tensors, autograd, optimizers, a formula/kernel system, a device runtime, and a self-contained artifact format. A model framework (such as **Striker**) is built *on top of* RTorch, not inside it.

**Where the boundary is:**

| layer | what it owns | examples |
|---|---|---|
| **Striker** (model framework) | intelligent semantics — Blocks, State, Fragments, Relations, Consolidation, Lifecycle, Routing | a model's "what to remember / how to reason" |
| **RTorch** (this project, compute layer) | Tensor, Autograd, Runtime, Formula/Kernel, RTW, CPU/GPU | build/run a formula, train a step, serialize an artifact |

RTorch does **not** understand Blocks/Fragments/Memory/Relations — it understands Tensor / Operator / Kernel / Artifact / Runtime. Build a model on top of it, not inside it.

## What it provides

- **Tensor** — CPU host tensor (`rtorch::tensor`) with `Shape` / `DType` (F32/F64/F16/BF16/I32), `shape`/`numel`/`reshape`/`view`/`transpose`, and broadcasting.
- **Autograd** — differentiable graphs on CPU (`rtorch::autograd`) and on device-resident GPU (`rtorch::gvar`), with `backward`, gradient accumulation over shared parameters, and an Adam optimizer (`rtorch::autograd::Adam`, `rtorch::gvar::AdamG`).
- **Formula / Kernel system** — a C ABI (`rtorch.h`) so any `formula.cpp` implements `rtorch_output_size` + `rtorch_compute`; the framework compiles it and runs it on CPU or GPU. GPU paths can supply a GLSL compute kernel. See `rtorch.h`.
  - **Forward-only by design (scheme D2(a))**: a formula / `model.dll` carries the **forward pass** only (inference, one compute step), and is **stateless** (no parameter blob, no backward). **Training is NOT run through the formula** — the model framework (Striker) owns forward + backward + optimizer using RTorch's autograd (`rtorch::autograd` / `rtorch::gvar`) and `Adam` / `AdamG`. See `forward_spec.md`.
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

`cargo build --release` is the single build entry point: it builds the Rust library and the `rtorch` command, and `build.rs` compiles the C++ Vulkan engine into `rtorch_vk.dll` and the GLSL kernels (`kernels/*.comp`) into `target/release/kernels/*.spv`.

Input blobs (`--input`) are user data files (raw float32/etc.); `rtorch` reads them as raw bytes and passes the blob to the formula.

Environment overrides: `RTORCH_BUILD_ENGINE` (0 = CPU-only), `RTORCH_GXX`, `VULKAN_SDK`, `RTORCH_GLSLANG`.

## Usage (CLI)

`rtorch` runs a formula: a `.cpp` file is compiled on the fly, a `.dll` is loaded
directly. The formula implements `rtorch_output_size` + `rtorch_compute` (see
`rtorch.h`), so the framework compiles/loads it, feeds it input blobs, allocates
the output, times the run, and surfaces the result.

### Synopsis

```
rtorch <formula.cpp|.dll> with <refs...> [--input <file>]... [--output <file>] [--device cpu|gpu]
```

| argument | meaning |
|---|---|
| `<formula.cpp>` | a formula source; compiled on the fly (needs `g++`) |
| `<formula.dll>` | a pre-built formula DLL (e.g. `delta.dll`); loaded directly, no compiler |
| `with <refs...>` | extra source/object files to compile alongside (optional) |
| `--input <file>` | a raw byte blob fed to the formula; repeatable |
| `--output <file>` | write the result blob to a file; default stdout |
| `--device cpu\|gpu` | `cpu` = host, `gpu` = best accelerator (Vulkan compute) |

### Run a formula source on the CPU

```sh
rtorch examples/formula_verify.cpp --input examples/input_1000.bin --device cpu
```

### Run a formula on the GPU (if it exports a GLSL kernel)

```sh
rtorch examples/formula_gpu.cpp --device gpu --input examples/input_1000.bin
```

### Use a pre-built formula DLL (no compiler, no on-the-fly compile)

Distribute an already-compiled formula like `delta.dll` and reference it directly:

```sh
rtorch delta.dll --input trig_input.bin --device cpu
```

`rtorch.exe` `LoadLibrary`s the DLL and calls `rtorch_output_size` /
`rtorch_compute` (and `rtorch_gpu_kernel` on GPU). This is the fastest way to ship
a formula to a machine without a compiler.

### Trig formula example (`examples/formula_trig.cpp` / `delta.dll`)

`formula_trig.cpp` reads a blob of float32 values `x[i]` and writes a triple
`[sin(x), cos(x), tan(x)]` per element (3 floats out for each 1 float in).

```sh
# compile + run the source (needs g++)
rtorch examples/formula_trig.cpp --input trig_input.bin --output trig_out.bin --device cpu

# load a pre-built DLL (no g++)
rtorch delta.dll --input trig_input.bin --output trig_out.bin --device cpu

# GPU (Vulkan) via the DLL's embedded GLSL kernel
rtorch delta.dll --input trig_input.bin --output trig_out.bin --device gpu
```

`trig_input.bin` is a raw little-endian float32 array, e.g. the angles
`0, π/6, π/4, π/3, π/2`. Output is `3×n` float32 values. CPU and GPU agree
bit-for-bit (verified).

### Package a formula into a self-contained `.rtw` container

`--pack` bundles a formula source into `RTW.md`'s runtime container; running the
`.rtw` executes the embedded formula (no source file needed afterwards):

```sh
rtorch --pack examples/formula_verify.cpp -o my_formula.rtw
rtorch my_formula.rtw --input examples/input_1000.bin --device cpu
```

`--dump <file.rtw>` prints the container's structure.

### Input / output blobs

`--input` files are raw bytes passed to the formula as a `rtorch_blob`
(`{ data, len }`). There is no encoding wrapper for the CLI path — the formula
decides the layout (e.g. float32 data). `--output` writes the exact byte blob the
formula produced; default is stdout.

### Other flags

```sh
rtorch --version
rtorch --help
```

Errors use stable exit codes: **2** = usage error, **1** = runtime/I-O error.
Diagnostics go to stderr so stdout carries only formula output.

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

> **Tooling limits (honest):** Miri does not run on this crate — it reports
> `running 0 tests` on every target (nightly Gnu + `panic="abort"` + a `build.rs`
> that shells out to g++/Vulkan), so it checks nothing and must not be treated as
> a UB-clean result. AddressSanitizer via `-Zsanitizer=address` is also
> unavailable on this Gnu nightly. Memory safety is instead established by
> static review of each `unsafe` site (the FFI symbol loader enforces a size
> check before `transmute_copy`) plus the runtime dynamic edge tests. See
> `tests/TOOLING_NOTES.md`.

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
