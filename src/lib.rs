// RTorch — universal compute framework (library).
// 分层: RTorch = 计算框架(类 PyTorch), Striker 依赖本库做计算。
// 本 lib 提供张量库 API(tensor + matmul/递推 + 梯度容器 + 优化器), 供 Striker 模型框架调用。
// MVP: 2D tensor + 前向 ops + 手动梯度容器 + SGD。autograd 图后补(先手动反向, Striker 赋 .grad)。

pub mod autograd;
pub mod device;
pub mod error;
pub mod gpu;
pub mod gpu_tensor;
pub mod gvar;
pub mod loc;
pub mod ops;
pub mod rtw;
pub mod tensor;
pub mod vk;

pub use tensor::{DType, Shape, Tensor};
