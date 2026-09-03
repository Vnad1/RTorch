// RTorch device abstraction — the backend selector for the unified op layer.
// A model writes ONE op call against a `Device`; the framework routes to the
// CPU (host reference) or Vulkan GPU backend. Device-resident GPU autograd
// (GVar) remains the optimized path; this layer is the uniform, host-facing
// surface (correctness/reference/mixed).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Gpu,
}

impl Device {
    pub fn name(&self) -> &'static str {
        match self { Device::Cpu => "cpu", Device::Gpu => "gpu" }
    }
    /// Parse a --device string (case-insensitive). Unknown -> None.
    pub fn from_cli(s: &str) -> Option<Device> {
        if s.eq_ignore_ascii_case("cpu") { Some(Device::Cpu) }
        else if s.eq_ignore_ascii_case("gpu") { Some(Device::Gpu) }
        else { None }
    }
}
