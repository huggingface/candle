mod device;
mod storage;

pub mod error;
pub mod wgpu_functions;

#[cfg(feature = "wgpu_debug")]
pub mod debug_info;

pub use device::WgpuDevice;
pub use storage::WgpuStorage;

#[cfg(feature = "wgpu_debug")]
pub use debug_info::Measurements;
#[cfg(feature = "wgpu_debug")]
pub use debug_info::MInfo;