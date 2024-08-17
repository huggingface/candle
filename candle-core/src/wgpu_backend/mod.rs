mod device;
mod storage;

pub mod error;
pub mod wgpu_functions;
pub mod cache;
pub mod util;

#[cfg(feature = "wgpu_debug")]
pub mod debug_info;

pub use device::WgpuDevice;
pub use storage::WgpuStorage;
pub use device::MatmulAlgorithm;

pub use storage::create_wgpu_storage;
pub use storage::create_wgpu_storage_init;

#[cfg(feature = "wgpu_debug")]
pub use debug_info::Measurements;
#[cfg(feature = "wgpu_debug")]
pub use debug_info::MInfo;