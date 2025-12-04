//! Selects the vector implementation based on the `cuda-pinned-memory` feature flag.
//! This keeps the conditional import centralized.
#[cfg(not(feature = "cuda-pinned-memory"))]
pub use std::vec::{self as storage_vec, Vec};

#[cfg(feature = "cuda-pinned-memory")]
pub use allocator_api2::vec::{self as storage_vec, Vec};

#[cfg(feature = "cuda-pinned-memory")]
#[macro_export]
macro_rules! storage_vec {
    ($($tt:tt)*) => {
        allocator_api2::vec![$($tt)*]
    };
}

#[cfg(not(feature = "cuda-pinned-memory"))]
#[macro_export]
macro_rules! storage_vec {
    ($($tt:tt)*) => {
        std::vec![$($tt)*]
    };
}

/// Alias to the vector type used by the CPU backend storage.
pub type StorageVec<T> = Vec<T>;

/// Convert an owned standard vector into the backend storage vector.
#[cfg(feature = "cuda-pinned-memory")]
pub fn to_storage_vec<T>(v: std::vec::Vec<T>) -> StorageVec<T> {
    v.into_iter().collect()
}

#[cfg(not(feature = "cuda-pinned-memory"))]
pub fn to_storage_vec<T>(v: std::vec::Vec<T>) -> StorageVec<T> {
    v
}

