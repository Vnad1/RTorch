# RTW — RTorch Workfile 格式规范 (v1)

RTorch 的自包含计算容器，替代"裸字节 blob / 仅 `.cpp` 源"，让**一条公式或一份数据打包成单文件**，可分发、可版本化、跨平台——不依赖任何运行时（非 PyTorch 的 pickle/.pt 绑定）。

## 设计目标

1. **自包含**：单一 `.rtw` 文件，任何语言（Rust/C++/Python）读它的二进制头即可解析，不依赖 torch/运行时。
2. **跨精度**：dtype 覆盖 fp32/fp16/fp8(E4M3)/fp4(E2M1)/int32/bytes——`.pt` 只有 fp32/fp16/fp64。
3. **跨平台**：小端字节序，Windows/Linux 一致；运行统一走 `rtorch x.rtw`。
4. **通用**：不绑深度学习，任何"公式/计算结果"都可打包。

## 二进制布局（小端）

```
偏移    字段         类型        说明
0       magic       char[4]     "RTW1"
4       version     u16         =1
6       kind        u8          0=result  1=kernel
7       dtype       u8          0=fp32 1=fp16 2=fp8 3=fp4 4=int32 5=bytes
8       rank        u32         shape 维度数
12      shape[]     u32[rank]   各维大小
12+4*rank count     u64         = data 字节数 / dtype 字节宽
...     data        byte[...]   连续数据净荷（小端原生值）
...     flags       u8          0x01 = has_kernel
...     kernel      u64 len + bytes   内嵌公式源码(kind=kernel)，可选
...     end_magic   char[5]     "RTEND"（完整性检测）
```

- `dtype_width`：fp32/int32=4，fp16=2，fp8/fp4=1。
- `kind=kernel`：`data` 为空，`kernel` 字段放**完整公式 `.cpp` 源码**（含 `rtorch_output_size`/`rtorch_compute`/可选 `rtorch_gpu_kernel`），运行时 `rtorch` 提取、写临时 `.cpp`、走既有编译→调度管线。
- `kind=result`：`data` 存计算结果，`shape`/`dtype` 描述；`kernel` 缺省。
- `kind=model`（规划，v1 未修）：内嵌推理内核 + 权重块 + 可选结构。

## CLI

```
rtorch --pack <formula.cpp> -o <x.rtw>          # 公式打包成 kernel 容器
rtorch --dump <x.rtw>                            # 打印 kind/dtype/shape/kernel
rtorch <x.rtw> --input <data> [--device gpu]     # 直接运行内嵌公式
```

## 示例

```
> rtorch --pack examples\formula_square.cpp -o square.rtw
[rtw] packed ... -> square.rtw (kind=kernel, 896 bytes)

> rtorch --dump square.rtw
kind = kernel | dtype = bytes | kernel = embedded formula source (896 bytes)

> rtorch square.rtw --device gpu --input mata.bin --output sq.bin
[rtorch] gpu kernel compiled (1248 bytes spv)
[rtorch] gpu dispatch avg(best) elapsed=0.446 ms
```

经验证：封装的平方公式 GPU 结果 `max_abs_err=0.0`（1M 元素），正确。

## 与 PyTorch `.pt` 的区别（对外口径）

| | `.pt`（PyTorch） | `.rtw`（RTorch） |
|---|---|---|
| 容器 | Python pickle 流 | 自包含二进制（magic+头+净荷） |
| 读取依赖 | 需 torch/Python 运行时 | 任何语言直接解析，零依赖 |
| dtype | fp32/16/64 | fp32/16/fp8/fp4/int32/bytes |
| 定位 | 模型权重 | 通用计算容器（公式/结果/权重） |
| 生态 | 绑 PyTorch | 不绑生态、跨厂商 GPU |
