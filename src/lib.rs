mod cpu_backend;
#[cfg(feature = "cuda")]
mod cuda_backend;
mod device;
mod dtype;
mod error;
mod op;
mod shape;
mod storage;
mod strided_index;
mod tensor;

pub use cpu_backend::CpuStorage;
pub use device::{Device, DeviceLocation};
pub use dtype::{DType, WithDType};
pub use error::{Error, Result};
pub use shape::Shape;
pub use storage::Storage;
use strided_index::StridedIndex;
pub use tensor::{Tensor, TensorId};

#[cfg(feature = "cuda")]
pub use cuda_backend::{CudaDevice, CudaError, CudaStorage};
