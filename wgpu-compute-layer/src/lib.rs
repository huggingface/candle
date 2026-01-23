mod device;
mod storage;
pub mod shader_loader;

pub mod cache;
pub mod error;
pub mod util;
pub mod wgpu_functions;
pub mod queue_buffer;

pub use error::Error;
pub use error::Result;

extern crate wgpu_compute_layer_macro;
pub use wgpu_compute_layer_macro::create_loader;


#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
#[cfg_attr(any(feature = "wgpu_debug_serialize", feature = "wgpu_debug"),
    derive(serde::Serialize, serde::Deserialize)
)]
pub enum DType {
    F32,
    U32,
    U8,
    I64,
    F64,
    F16,
}
pub const DTYPE_COUNT: u16 = 6;

impl DType {
    pub fn get_index(&self) -> u16 {
        match self {
            DType::F32 => 0,
            DType::U32 => 1,
            DType::U8 => 2,
            DType::I64 => 3,
            DType::F64 => 4,
            DType::F16 => 5,
        }
    }

    pub fn from_index(index: u16) -> Self {
        match index {
            0 => DType::F32,
            1 => DType::U32,
            2 => DType::U8,
            3 => DType::I64,
            4 => DType::F64,
            5 => DType::F16,
            _ => {
                todo!()
            }
        }
    }

    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::U32 => 4,
            DType::U8 => 1,
            DType::I64 => 8,
            DType::F64 => 8,
            DType::F16 => 2,
        }
    }
}

pub trait EntryPoint {
    fn get_entry_point(&self) -> &'static str;
}


#[cfg(feature = "wgpu_debug")]
pub mod debug_info;

pub use device::WgpuDevice;
pub use storage::WgpuStorage;
pub use device::WgpuDeviceConfig;
pub use device::WgpuBackends;
pub use device::DebugPipelineRecording;

pub use shader_loader::ShaderLoader;
pub use shader_loader::LoaderIndex;
pub use shader_loader::PipelineIndex;
pub use shader_loader::ShaderIndex;

#[cfg(feature = "wgpu_debug")]
pub use debug_info::MInfo;
#[cfg(feature = "wgpu_debug")]
pub use debug_info::Measurements;
