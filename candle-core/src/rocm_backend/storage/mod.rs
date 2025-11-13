//! ROCm storage module - organized by functionality
//! Created by: TEAM-488 (Phase 1 - Device integration)
//! Modified by: TEAM-491-496 (Kernel work and backend implementation)
//! Module split and CUDA parity verified by: TEAM-497d
//!
//! Organized into focused submodules:
//! - `struct_impl`: RocmStorage struct and basic methods
//! - `slice`: RocmStorageSlice enum
//! - `conversions`: Type conversion operations
//! - `operations`: Tensor operations (affine, reduce, cmp, etc.)
//! - `advanced`: Advanced operations (conv2d, pooling, matmul)
//! - `indexing`: Indexing operations (gather, scatter, index_select, upsample)
//! - `backend_trait`: BackendStorage trait implementation

mod advanced;
mod backend_trait;
mod conversions;
mod indexing;
mod operations;
mod slice;
mod struct_impl;

pub use slice::RocmStorageSlice;
pub use struct_impl::RocmStorage;
