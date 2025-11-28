//! CUDA pinned memory allocator support.
//!
//! This module is only available when both `cuda-pinned-memory` and `cuda` features are enabled.
//! The `cuda-pinned-memory` feature automatically enables `cuda` (see Cargo.toml), so checking
//! for `cuda-pinned-memory` is sufficient.

#![cfg(feature = "cuda-pinned-memory")]

use cudarc::driver::PinnedHostSlice;

/// A wrapper around PinnedHostSlice that provides Vec-like functionality.
///
/// This type allows us to use CUDA pinned memory as if it were a regular Vec,
/// while ensuring proper deallocation through the PinnedHostSlice's Drop implementation.
pub struct PinnedVec<T> {
    pinned: PinnedHostSlice<T>,
    len: usize,
}

impl<T> PinnedVec<T> {
    /// Create a new PinnedVec from a PinnedHostSlice.
    ///
    /// The length must not exceed the capacity of the pinned slice.
    pub fn new(pinned: PinnedHostSlice<T>, len: usize) -> Self {
        Self { pinned, len }
    }

    /// Get the length of the vector.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Get the capacity of the vector.
    pub fn capacity(&self) -> usize {
        self.pinned.len()
    }
}

impl<T: cudarc::driver::ValidAsZeroBits> PinnedVec<T> {
    /// Get a slice of the data.
    pub fn as_slice(&self) -> &[T] {
        self.pinned
            .as_slice()
            .expect("pinned slice")
            .get(..self.len)
            .unwrap_or(&[])
    }

    /// Get a mutable slice of the data.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.pinned
            .as_mut_slice()
            .expect("pinned slice")
            .get_mut(..self.len)
            .unwrap_or(&mut [])
    }

    /// Set the length of the vector.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the first `new_len` elements are initialized.
    pub unsafe fn set_len(&mut self, new_len: usize) {
        assert!(new_len <= self.capacity());
        self.len = new_len;
    }

    /// Get a reference to the underlying PinnedHostSlice.
    pub fn pinned_slice(&self) -> &PinnedHostSlice<T> {
        &self.pinned
    }

    /// Get a mutable reference to the underlying PinnedHostSlice.
    pub fn pinned_slice_mut(&mut self) -> &mut PinnedHostSlice<T> {
        &mut self.pinned
    }

    /// Convert into the underlying PinnedHostSlice.
    pub fn into_pinned_slice(self) -> PinnedHostSlice<T> {
        self.pinned
    }
}

impl<T> Clone for PinnedVec<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        // For cloning, we'd need to allocate a new pinned buffer
        // This is complex, so for now we'll just panic
        // In practice, we might want to implement this differently
        panic!("PinnedVec::clone is not yet implemented")
    }
}

impl<T> std::fmt::Debug for PinnedVec<T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedVec")
            .field("len", &self.len)
            .field("capacity", &self.capacity())
            .finish()
    }
}

