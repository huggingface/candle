mod device;
mod storage;

pub mod cache;
pub mod error;
pub mod util;
pub mod wgpu_functions;
pub mod queue_buffer;

#[cfg(feature = "wgpu_debug")]
pub mod debug_info;

pub use wgpu_functions::matmul::MatmulAlgorithm;
pub use wgpu_functions::matmul::QuantizedMatmulAlgorithm;
pub use device::WgpuDevice;
pub use storage::WgpuStorage;


#[cfg(feature = "wgpu_debug_serialize")]
pub use device::DebugPipelineRecording;
#[cfg(not(feature = "wgpu_debug_serialize"))]
pub(crate) use device::DebugPipelineRecording;

#[cfg(feature = "wgpu_debug")]
pub use debug_info::MInfo;
#[cfg(feature = "wgpu_debug")]
pub use debug_info::Measurements;
