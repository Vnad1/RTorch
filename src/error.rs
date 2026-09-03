// RTorch unified error type — the single error surface for every
// user-triggerable path. The CLI maps these to a message + a stable exit code,
// and the library returns them instead of `panic!/expect!` on bad input or
// missing resources. Compiles into both the lib crate and the rtorch bin
// (it only depends on std).

use std::fmt;

// error.rs is compiled into both the lib (public) and the rtorch bin (as a
// private `mod`), so unused-in-older-target constructor helpers can warn about
// dead code. These are public API factories; keep them.
#![allow(dead_code)]

#[derive(Debug)]
pub enum RtorchError {
    /// The formula/GLSL/kernel source failed to compile.
    CompileError(String),
    /// A DLL or symbol failed to load/resolve.
    LoadError(String),
    /// A Vulkan call returned a non-success result.
    VulkanError(String),
    /// A compute kernel could not be found/loaded/run.
    KernelError(String),
    /// A tensor operation was given invalid dimensions/shapes.
    TensorError(String),
    /// Shape/strides are inconsistent.
    ShapeError(String),
    /// An unsupported or mismatched dtype was requested.
    DTypeError(String),
    /// A model-level error.
    ModelError(String),
    /// A `.rtw` container encode/decode failure.
    RTWError(String),
    /// General CLI/runtime error (missing file, bad args, etc.).
    Io(String),
}

impl fmt::Display for RtorchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (tag, msg) = match self {
            RtorchError::CompileError(m) => ("compile", m),
            RtorchError::LoadError(m) => ("load", m),
            RtorchError::VulkanError(m) => ("vulkan", m),
            RtorchError::KernelError(m) => ("kernel", m),
            RtorchError::TensorError(m) => ("tensor", m),
            RtorchError::ShapeError(m) => ("shape", m),
            RtorchError::DTypeError(m) => ("dtype", m),
            RtorchError::ModelError(m) => ("model", m),
            RtorchError::RTWError(m) => ("rtw", m),
            RtorchError::Io(m) => ("io", m),
        };
        write!(f, "{tag}: {msg}")
    }
}

impl std::error::Error for RtorchError {}

pub type Result<T> = std::result::Result<T, RtorchError>;

impl From<&str> for RtorchError {
    fn from(s: &str) -> Self { RtorchError::Io(s.to_string()) }
}
impl From<String> for RtorchError {
    fn from(s: String) -> Self { RtorchError::Io(s) }
}
impl From<std::io::Error> for RtorchError {
    fn from(e: std::io::Error) -> Self { RtorchError::Io(e.to_string()) }
}

impl RtorchError {
    pub fn compile(m: impl Into<String>) -> Self { RtorchError::CompileError(m.into()) }
    pub fn load(m: impl Into<String>) -> Self { RtorchError::LoadError(m.into()) }
    pub fn vulkan(m: impl Into<String>) -> Self { RtorchError::VulkanError(m.into()) }
    pub fn kernel(m: impl Into<String>) -> Self { RtorchError::KernelError(m.into()) }
    pub fn tensor(m: impl Into<String>) -> Self { RtorchError::TensorError(m.into()) }
    pub fn shape(m: impl Into<String>) -> Self { RtorchError::ShapeError(m.into()) }
    pub fn dtype(m: impl Into<String>) -> Self { RtorchError::DTypeError(m.into()) }
    pub fn model(m: impl Into<String>) -> Self { RtorchError::ModelError(m.into()) }
    pub fn rtw(m: impl Into<String>) -> Self { RtorchError::RTWError(m.into()) }
    pub fn io(m: impl Into<String>) -> Self { RtorchError::Io(m.into()) }

    /// Stable exit code for a CLI invocation: 0 handled by caller; 1 = runtime
    /// error; 2 = usage/argument error.
    pub fn exit_code(&self) -> i32 {
        match self {
            RtorchError::CompileError(_) | RtorchError::LoadError(_)
            | RtorchError::VulkanError(_) | RtorchError::KernelError(_)
            | RtorchError::TensorError(_) | RtorchError::ShapeError(_)
            | RtorchError::DTypeError(_) | RtorchError::ModelError(_)
            | RtorchError::RTWError(_) | RtorchError::Io(_) => 1,
        }
    }
}
