//! ML framework for Rust
//!
//! ```rust
//! use candle::{Tensor, DType, Device};
//! # use candle::Error;
//! # fn main() -> Result<(), Error>{
//!
//! let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
//! let b = Tensor::zeros((3, 4), DType::F32, &Device::Cpu)?;
//!
//! let c = a.matmul(&b)?;
//! # Ok(())}
//! ```
//!
//! ## Features
//!
//! - Simple syntax (looks and like PyTorch)
//! - CPU and Cuda backends (and M1 support)
//! - Enable serverless (CPU) small and fast deployments
//! - Model training
//! - Distributed computing (NCCL).
//! - Models out of the box (Llama, Whisper, Falcon, ...)
//!
//! ## FAQ
//!
//! - Why Candle?
//!
//! Candle stems from the need to reduce binary size in order to *enable serverless*
//! possible by making the whole engine smaller than PyTorch very large library volume
//!
//! And simply *removing Python* from production workloads.
//! Python can really add overhead in more complex workflows and the [GIL](https://www.backblaze.com/blog/the-python-gil-past-present-and-future/) is a notorious source of headaches.
//!
//! Rust is cool, and a lot of the HF ecosystem already has Rust crates [safetensors](https://github.com/huggingface/safetensors) and [tokenizers](https://github.com/huggingface/tokenizers)

mod backprop;
mod conv;
mod cpu_backend;
#[cfg(feature = "cuda")]
mod cuda_backend;
mod device;
pub mod display;
mod dtype;
mod dummy_cuda_backend;
mod error;
mod indexer;
mod layout;
mod npy;
mod op;
pub mod safetensors;
mod shape;
mod storage;
mod strided_index;
mod tensor;
pub mod utils;

pub use cpu_backend::CpuStorage;
pub use device::{Device, DeviceLocation};
pub use dtype::{DType, WithDType};
pub use error::{Error, Result};
pub use indexer::IndexOp;
pub use layout::Layout;
pub use shape::{Shape, D};
pub use storage::Storage;
use strided_index::StridedIndex;
pub use tensor::{Tensor, TensorId};

#[cfg(feature = "cuda")]
pub use cuda_backend::{CudaDevice, CudaError, CudaStorage};

#[cfg(not(feature = "cuda"))]
pub use dummy_cuda_backend::{CudaDevice, CudaError, CudaStorage};
