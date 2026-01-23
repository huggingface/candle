mod device;
mod storage;

pub mod error;
pub mod wgpu_functions;

pub use device::WgpuDevice;
pub use storage::WgpuStorage;
pub use wgpu_functions::matmul::MatmulAlgorithm;
pub use wgpu_functions::matmul::QuantizedMatmulAlgorithm;

impl From<candle_wgpu_kernels::DType> for crate::DType{
    fn from(value: candle_wgpu_kernels::DType) -> Self {
        match value {
            candle_wgpu_kernels::DType::F32 => crate::DType::F32,
            candle_wgpu_kernels::DType::U32 => crate::DType::U32,
            candle_wgpu_kernels::DType::U8 => crate::DType::U8,
            candle_wgpu_kernels::DType::I64 => crate::DType::I64,
            candle_wgpu_kernels::DType::F64 => crate::DType::F64,
            candle_wgpu_kernels::DType::F16 => crate::DType::F16,
        }
    }
}

impl From<crate::DType> for candle_wgpu_kernels::DType {
    fn from(val: crate::DType) -> Self {
        match val {
            crate::DType::F32 => candle_wgpu_kernels::DType::F32,
            crate::DType::U32 => candle_wgpu_kernels::DType::U32,
            crate::DType::U8 => candle_wgpu_kernels::DType::U8,
            crate::DType::I64 => candle_wgpu_kernels::DType::I64,
            crate::DType::F64 => candle_wgpu_kernels::DType::F64,
            crate::DType::F16 => candle_wgpu_kernels::DType::F16,
            _ => panic!("{val:?} is not supported in candle_wgpu_kernels"),
        }
    }
}

impl From<wgpu_compute_layer::Error> for crate::Error {
    fn from(value: wgpu_compute_layer::Error) -> Self {
        crate::Error::Wgpu(crate::WgpuError::Message(value.to_string()))
    }
}
