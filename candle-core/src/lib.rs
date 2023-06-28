mod backprop;
mod cpu_backend;
#[cfg(feature = "cuda")]
mod cuda_backend;
mod device;
pub mod display;
mod dtype;
mod dummy_cuda_backend;
mod error;
mod layout;
mod npy;
mod op;
mod shape;
mod storage;
mod strided_index;
mod tensor;
pub mod utils;

pub use cpu_backend::CpuStorage;
pub use device::{Device, DeviceLocation};
pub use dtype::{DType, WithDType};
pub use error::{Error, Result};
pub use layout::Layout;
pub use shape::Shape;
pub use storage::Storage;
use strided_index::StridedIndex;
pub use tensor::{Tensor, TensorId};

#[cfg(feature = "cuda")]
pub use cuda_backend::{CudaDevice, CudaError, CudaStorage};

#[cfg(not(feature = "cuda"))]
pub use dummy_cuda_backend::{CudaDevice, CudaError, CudaStorage};
