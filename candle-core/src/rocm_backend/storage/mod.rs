//! Storage module for ROCm backend
//!
//! Organized into focused submodules:
//! - `struct_impl`: RocmStorage struct and basic methods
//! - `slice`: RocmStorageSlice enum
//! - `conversions`: Type conversion operations
//! - `operations`: Tensor operations (affine, reduce, cmp, etc.)
//! - `advanced`: Advanced operations (conv2d, pooling, matmul)
//! - `backend_trait`: BackendStorage trait implementation

mod advanced;
mod backend_trait;
mod conversions;
mod operations;
mod slice;
mod struct_impl;

pub use slice::RocmStorageSlice;
pub use struct_impl::RocmStorage;
