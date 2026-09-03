# RTorch

**RTorch** 是一个通用计算框架——一个通用数值引擎,让你描述一条数学公式、一次状态转移、一个自定义内核,或一个训练步骤,然后由 RTorch 在 **CPU** 或 **Vulkan GPU** 上执行。

它**不是**模型库,也**不是** Transformer/深度学习框架。RTorch 属于*计算层*:提供张量、自动微分、优化器、公式/内核系统、设备运行时,以及一种自包含的产品格式。一个模型框架(例如 **Striker**)是在 RTorch *之上*构建的,而不是在它内部。

## 它提供什么

- **张量** — CPU 宿主张量(`rtorch::tensor`),带 `Shape` / `DType`(F32/F64/F16/BF16/I32)、`shape`/`numel`/`reshape`/`view`/`transpose`,以及广播。
- **自动微分** — CPU 上的可微计算图(`rtorch::autograd`)与设备驻留 GPU 上的可微计算图(`rtorch::gvar`),支持 `backward`、共享参数上的梯度累积,以及 Adam 优化器(`rtorch::autograd::Adam`、`rtorch::gvar::AdamG`)。
- **公式/内核系统** — 一个 C ABI(`rtorch.h`),使任意 `formula.cpp` 实现 `rtorch_output_size` + `rtorch_compute`;框架负责编译并在 CPU 或 GPU 上运行。GPU 路径可提供 GLSL 计算内核。见 `rtorch.h`。
- **设备运行时** — `rtorch::device::Device`(Cpu/Gpu)与统一操作层(`rtorch::ops`),使模型不需要维护两份实现。
- **Vulkan / GPU** — 一个 C++ Vulkan 计算引擎(`rtorch_vk.dll`),采用持久化 / 设备驻留模型与 `KernelRegistry`;无 CUDA 依赖(跨厂商 Vulkan Compute)。
- **RTW(.rtw)** — 一种自包含的产品格式,用于结果、内核、模型与记忆(`rtw::`,见 `RTW.md`)。

## 构建

要求:稳定的 Rust 工具链、一个 C++ 编译器(`g++` 或 `clang`),以及(用于 GPU)Vulkan SDK(头文件 + `vulkan-1.lib`)。

```sh
git clone https://github.com/Vnad1/RTorch.git
cd RTorch
cargo build --release
```

`cargo build --release` 是唯一构建入口:它构建 Rust 库与 `rtorch` 命令,并由 `build.rs` 把 C++ Vulkan 引擎编译成 `rtorch_vk.dll`,把 GLSL 内核(`kernels/*.comp`)编译到 `target/release/kernels/*.spv`。

`--input` 输入文件是用户数据文件(原始 float32 等);`rtorch` 按原始字节读取并把 blob 传给公式。

环境变量覆盖:`RTORCH_BUILD_ENGINE`(0 = 仅 CPU)、`RTORCH_GXX`、`VULKAN_SDK`、`RTORCH_GLSLANG`。

## 使用方法(CLI)

在 CPU 上运行一个公式:

```sh
rtorch examples/formula_verify.cpp --input examples/input_1000.bin --device cpu
```

在 GPU 上运行一个公式(若公式导出 GLSL 内核):

```sh
rtorch examples/formula_gpu.cpp --device gpu --input examples/input_1000.bin
```

可用 `rtorch --help` 与 `rtorch --version`。错误以稳定的退出码报告(2 = 用法错误,1 = 运行时错误)。

## 使用方法(库)

将 `rtorch` 添加为路径依赖。模型使用统一操作层或自动微分 API:

```rust
use rtorch::ops;                        // 统一 CPU/GPU 操作分发
use rtorch::device::Device;
use rtorch::autograd::{self, Adam};     // CPU 自动微分 + 优化器
use rtorch::gvar::{self, AdamG};        // 设备驻留 GPU 自动微分
use rtorch::rtw;                        // .rtw 容器
```

可运行示例见 `tests/`(张量、自动微分、RTW、GPU、公式 ABI、模型周期)。

## 测试

```sh
cargo test --release
cargo clippy --release --all-targets
```

测试套件覆盖:张量正确性、自动微分(含中心差分数值梯度校验)、CPU/GPU 数值一致性、RTW 往返、公式 C-ABI 边界用例,以及 GPU 设备驻留训练收敛。

## 说明

- **贡献者:** [Vnad1](https://github.com/Vnad1/RTorch) 是唯一贡献者。
- **AI 生成代码:** 本仓库包含 AI 辅助生成的源代码。代码已经过审查与修正:移除了不合理的模式,并修复了缺陷(包括静默广播错误、Vulkan API 误用、资源/生命周期问题),且这些由测试与 sanitizer/校验检查覆盖。

## 相关

- `RTW.md` — `.rtw` 产品格式规范。
- 构建在 RTorch 之上的模型层(见 Striker 项目)与之保持分离。

## 许可证

本项目采用 **GNU LGPL,version 3.0(LGPL-3.0)** —— 见 [LICENSE](./LICENSE)。

LGPL-3.0 仅适用于 RTorch 库本身。**使用** RTorch 的应用(例如基于它构建的模型框架)可以采用其他许可,前提是允许对 RTorch 库部分**重链接/替换**(见 LGPL-3.0 条款)。用户**公式**若只包含公开接口头(`rtorch.h`),则不构成 RTorch 实现的衍生作品,不受本许可约束(详见 `rtorch.h` 顶部说明与 LGPL-3.0 §3)。

---

[English](./README.md)
