# RTorch — 通用计算框架（Compute Framework）

> **这是计算架构，不是模型。**
>
> RTorch = 通用计算层（Rust 张量库 + C++ Vulkan 引擎 + 统一公式协议 + `.rtw` 容器）。
> 它不定义任何模型/记忆/思考语义。模型层是 **Striker**（仿脑记忆与思考系统），
> Striker 依赖 RTorch 做全部底层计算——就像 Transformer 依赖 PyTorch。

## 定位一句话

**任何人写一条公式（计算内核），经框架验证或计算；GPU 越强，算得越狠。**

不绑定特定任务、不绑定特定模型、**不绑定 CUDA 生态**（GPU 走跨厂商 Vulkan Compute，零 CUDA 依赖）、
**禁用现代深度学习栈**（无 Transformer/Mamba/PyTorch），研究定位在古典、非现代技术路线。

## 分层（谁是计算，谁不是）

```
┌─────────────────────────────────────────────┐
│  Striker（模型框架：记忆/思考/状态，另见 STRIKER.md）│
│    —— 模型层，依赖 RTorch 做计算              │
└──────────────────┬──────────────────────────┘
                   │  调用
┌──────────────────▼──────────────────────────┐
│  RTorch（本仓库 = 计算架构）                  │
│   用户公式(.cpp/.GLSL) → rtorch.exe(Rust 前端)│
│     → rtorch_vk.dll(C++ Vulkan 引擎) → GPU   │
│   或 → Rust lib(tensor/autograd/GVar) → CPU   │
└─────────────────────────────────────────────┘
```

## 三块计算能力

### 1. 统一公式接口协议（`rtorch.h`）
用户公式只实现两个函数，框架管编译/输入/输出/设备/计时：
- `rtorch_output_size(n_in, in, device)` → 输出字节数
- `rtorch_compute(n_in, in, out, device)` → 计算写入 `out`
- 可选 GPU：`rtorch_gpu_kernel()`（GLSL 源码）、`rtorch_gpu_groups(gx,gy,gz)`

CLI：`rtorch <公式.cpp> with <引用...> [--input f]... [--output f] [--device cpu|gpu]`
公式经 g++ 编成 DLL，`LoadLibrary` 执行；GPU 时框架调 `glslangValidator` 编 GLSL→SPIR-V。

### 2. Rust 张量库（`src/lib.rs` 暴露）
Striker 模型层直接调用的计算原语，模块：
- `tensor` — 2D 张量 + matmul/加/缩放/softmax 等前向 ops + SGD
- `autograd` — CPU 自动微分图（Var：matmul/add/scale/tanh/sigmoid/softmax_row + backward + Adam 优化器）
- `gpu_tensor` — **设备驻留** GPU 张量（`GpuContext` 持 Vulkan 设备 + 每 kernel 一 pipeline，ops 全 GPU 免宿主拷贝；含 `ce` softmax-CE 内核）
- `gvar` — GPU 设备驻留自动微分（`GVar`=Rc<RefCell<GVarData>>，前向/反向全 GPU，`backward_multi` 多根联合反向，`AdamG` 设备驻留优化器）
- `vk` / `gpu` — Vulkan 设备（`GpuDevice`）与 GPU matmul 入口
- `rtw` — `.rtw` 容器编解码

### 3. C++ Vulkan 计算引擎（`rtorch_vk.dll`）
Rust 只 `LoadLibrary` 调 C ABI；引擎用官方 `vulkan.h` + 标准 API + 静态链接 `vulkan-1.lib`。
两阶段：`rtorch_vk_init`（一次性建 instance/device/pipeline/descriptor/device-local 缓冲）→
`rtorch_vk_dispatch`（循环 dispatch + staging 传输入/读回）→ `rtorch_vk_destroy`。
设备驻留 + 输入复用后，1024³ fp32 matmul 达到 **3436 GFLOPS**（纯 fp32 实践极限 ≈6500，
与 cuBLAS 纯 fp32 差距 ≈2.5×）。

## 进目录前先读

| 文档 | 归属 | 内容 |
|---|---|---|
| `CONCLUSION.md` | **计算层** | RTorch 研发结论：统一协议/CPU-GPU 路径/性能基准（14.3→3436 GFLOPS）/精度测试/教训 |
| `RTW.md` | **计算层** | `.rtw` 自包含计算容器格式规范（magic+头+净荷，跨精度 fp32/fp16/fp8/fp4） |
| `rtorch.h` | 计算层 | 统一公式 C ABI 头（框架+用户同源） |
| `STRIKER.md` | **模型层** | Striker 理念（记忆/取舍/遗忘/状态-权重双轨）—— 不是 RTorch 计算架构 |
| `RTORCH_MODEL.md` | **模型层** | RTorch 之上的仿脑状态模型实验 —— 属 Striker 模型研究，非计算框架主旨 |

> 模型层文档放在本目录是因为历史上与 RTorch 一起演化，但它们不是"RTorch 的计算内容"。
> 计算架构只看前四行。

## 快速开始

```bash
# 公式 → GPU 运行（max_abs_err=0.0 验证）
rtorch examples/formula_gpu.cpp with examples/helper.cpp --device gpu \
       --input mata.bin --output out.bin

# 公式打包成 .rtw 自包含容器
rtorch --pack examples/formula_square.cpp -o square.rtw

# 跑 .rtw 内嵌公式
rtorch square.rtw --device gpu --input x.bin --output y.bin
```

库端（Striker 用）：
```rust
use rtorch::gvar::{self, AdamG};        // GPU 设备驻留 autograd
use rtorch::gpu_tensor::{self, GpuContext};
// 见 striker_framework/src/striker_v12.rs 的实际用法
```

## 研发立场

研究者。结论基于实测 + 已归档证据，区分"已验证/待验证"。从不把"比 PyTorch 慢"归因成
"语言慢"——三层拆解证明是**厂商 BLAS/micro-kernel 的优化深度**（+ GPU tensor-core）而非语言问题。
详见 `CONCLUSION.md`。

## 构建

```bash
cargo build --release
# 运行前把 C:\msys64\ucrt64\bin 前置进 PATH（g++ 编译 + libwinpthread 运行时依赖）
```
