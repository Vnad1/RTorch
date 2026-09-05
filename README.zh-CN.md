# RTorch

**RTorch** 是一个通用计算框架——一个通用数值引擎,让你描述一条数学公式、一次状态转移、一个自定义内核,或一个训练步骤,然后由 RTorch 在 **CPU** 或 **Vulkan GPU** 上执行。

它**不是**模型库,也**不是** Transformer/深度学习框架。RTorch 属于*计算层*:提供张量、自动微分、优化器、公式/内核系统、设备运行时,以及一种自包含的产品格式。一个模型框架(例如 **Striker**)是在 RTorch *之上*构建的,而不是在它内部。

## 版本号设定

RTorch 使用**日期版本号**: `YYYY.MM.DD.N`。

- `2026.09.06.1` = **2026 年 9 月 6 日**发布的**第 1 个**版本。
- 末尾 `.N` 是**当日序号**——同一天发布第 2 个版本记为 `.2`,依此类推。
- **进入新的一天,序号重置为 `.1`**(不会累积):例如 `2026.09.07.1` 是 9 月 7 日的第 1 版,与 9 月 6 日发了多少个无关。

这个版本号记录的是**何时发布**以及**当日第几版**,与 RTorch 的格式版本、依赖组件的版本相互独立。

**边界在哪:**

| 层 | 拥有什么 | 例子 |
|---|---|---|
| **Striker**(模型框架) | 智能语义——Block、State、Fragment、Relation、Consolidation、Lifecycle、Routing | 一个模型"该记什么/如何推理" |
| **RTorch**(本项目,计算层) | Tensor、Autograd、Runtime、Formula/Kernel、RTW、CPU/GPU | 构建/运行一条公式、训练一步、序列化一件产物 |

RTorch **不理解** Block/Fragment/Memory/Relation——它只理解 Tensor / Operator / Kernel / Artifact / Runtime。模型请在它之上构建,而不是在它内部。

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

`rtorch` 运行公式:传 `.cpp` 会现场编译,传 `.dll` 则直接加载。公式实现
`rtorch_output_size` + `rtorch_compute`(见 `rtorch.h`),框架负责编译/加载、喂入
输入 blob、分配输出、计时并给出结果。

### 语法

```
rtorch <formula.cpp|.dll> with <refs...> [--input <file>]... [--output <file>] [--device cpu|gpu]
```

| 参数 | 含义 |
|---|---|
| `<formula.cpp>` | 公式源码;现场编译(需 `g++`) |
| `<formula.dll>` | 预编译的公式 DLL(如 `delta.dll`);直接加载,无需编译器 |
| `with <refs...>` | 附加的源/目标文件,一起编译(可选) |
| `--input <file>` | 喂给公式的原始字节 blob;可多次 |
| `--output <file>` | 把结果 blob 写到文件;默认 stdout |
| `--device cpu\|gpu` | `cpu`=宿主,`gpu`=最佳加速器(Vulkan compute) |

### CPU 上运行公式源码

```sh
rtorch examples/formula_verify.cpp --input examples/input_1000.bin --device cpu
```

### GPU 上运行(若公式导出 GLSL 内核)

```sh
rtorch examples/formula_gpu.cpp --device gpu --input examples/input_1000.bin
```

### 使用预编译的公式 DLL(无需编译器、无需现场编译)

分发一个已编译的公式(如 `delta.dll`)并直接引用:

```sh
rtorch delta.dll --input trig_input.bin --device cpu
```

`rtorch.exe` 会 `LoadLibrary` 该 DLL 并调用 `rtorch_output_size` /
`rtorch_compute`(GPU 下还有 `rtorch_gpu_kernel`)。这是把公式分发到无编译器机器的
最快方式。

### 三角函数公式示例(`examples/formula_trig.cpp` / `delta.dll`)

`formula_trig.cpp` 读取 float32 数组 `x[i]`,每元素写一个三元组
`[sin(x), cos(x), tan(x)]`(每 1 个输入 float 输出 3 个 float)。

```sh
# 编译+运行源码(需 g++)
rtorch examples/formula_trig.cpp --input trig_input.bin --output trig_out.bin --device cpu

# 加载预编译 DLL(无需 g++)
rtorch delta.dll --input trig_input.bin --output trig_out.bin --device cpu

# 用 DLL 内嵌的 GLSL 内核跑 GPU(Vulkan)
rtorch delta.dll --input trig_input.bin --output trig_out.bin --device gpu
```

`trig_input.bin` 是原始小端 float32 数组,例如角度 `0, π/6, π/4, π/3, π/2`。
输出为 `3×n` 个 float32。CPU 与 GPU 逐位一致(已被验证)。

### 把公式打包成自包含的 `.rtw` 容器

`--pack` 把公式源码打包进 `RTW.md` 描述的运行时容器;运行该 `.rtw` 即执行内嵌
公式(之后不再需要源码文件):

```sh
rtorch --pack examples/formula_verify.cpp -o my_formula.rtw
rtorch my_formula.rtw --input examples/input_1000.bin --device cpu
```

`--dump <file.rtw>` 打印容器结构。

### 输入/输出 blob

`--input` 文件是原始字节,作为 `rtorch_blob`(`{ data, len }`)传给公式。CLI 路径
不做编码包装——公式自己决定布局(例如 float32 数据)。`--output` 写公式产出的精确
字节 blob;默认是 stdout。

### 其它参数

```sh
rtorch --version
rtorch --help
```

错误用稳定退出码:**2** = 用法错误,**1** = 运行时/IO 错误。诊断信息走 stderr,stdout
只承载公式输出。

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

> **工具限制(诚实说明):** Miri 无法在本 crate 上运行——它对每个目标都报告
> `running 0 tests`(nightly Gnu + `panic="abort"` + 会调用 g++/Vulkan 的
> `build.rs`),等于什么都没检查,不能当作"无 UB"的依据。`-Zsanitizer=address`
> 的 AddressSanitizer 在这个 Gnu nightly 上也不可用。内存安全改为依靠逐处
> `unsafe` 的静态审查(FFI 符号加载器在 `transmute_copy` 前有尺寸校验)加上
> 运行时动态边界测试。详见 `tests/TOOLING_NOTES.md`。

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
