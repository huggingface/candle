//! The dtype-tagged device buffer a [`super::RocmStorage`] holds.
//!
//! Split out of `mod.rs`, which is far over the workspace 400-line cap.

use std::ffi::c_void;

use float8::F8E4M3;
use half::{bf16, f16};

use super::alloc::SendSyncDeviceMemory;
use crate::DType;

pub enum RocmStorageSlice {
    U8(SendSyncDeviceMemory<u8>),
    U32(SendSyncDeviceMemory<u32>),
    I16(SendSyncDeviceMemory<i16>),
    I32(SendSyncDeviceMemory<i32>),
    I64(SendSyncDeviceMemory<i64>),
    BF16(SendSyncDeviceMemory<bf16>),
    F16(SendSyncDeviceMemory<f16>),
    F32(SendSyncDeviceMemory<f32>),
    F64(SendSyncDeviceMemory<f64>),
    F8E4M3(SendSyncDeviceMemory<F8E4M3>),
}

/// F8E4M3 carries its own type rather than sharing `u8`'s, so that the generic
/// `f` in the `Map*` traits resolves to the `*_f8_e4m3` kernels instead of
/// silently to the `*_u8` ones. The byte-view shortcuts here and in `device.rs`
/// still assume it is exactly one byte wide.
const _: () = assert!(std::mem::size_of::<F8E4M3>() == 1);

impl std::fmt::Debug for RocmStorageSlice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RocmStorageSlice::U8(m) => write!(f, "U8({} bytes)", m.size()),
            RocmStorageSlice::U32(m) => write!(f, "U32({} bytes)", m.size()),
            RocmStorageSlice::I16(m) => write!(f, "I16({} bytes)", m.size()),
            RocmStorageSlice::I32(m) => write!(f, "I32({} bytes)", m.size()),
            RocmStorageSlice::I64(m) => write!(f, "I64({} bytes)", m.size()),
            RocmStorageSlice::BF16(m) => write!(f, "BF16({} bytes)", m.size()),
            RocmStorageSlice::F16(m) => write!(f, "F16({} bytes)", m.size()),
            RocmStorageSlice::F32(m) => write!(f, "F32({} bytes)", m.size()),
            RocmStorageSlice::F64(m) => write!(f, "F64({} bytes)", m.size()),
            RocmStorageSlice::F8E4M3(m) => write!(f, "F8E4M3({} bytes)", m.size()),
        }
    }
}

impl RocmStorageSlice {
    pub fn dtype(&self) -> DType {
        match self {
            RocmStorageSlice::U8(_) => DType::U8,
            RocmStorageSlice::U32(_) => DType::U32,
            RocmStorageSlice::I16(_) => DType::I16,
            RocmStorageSlice::I32(_) => DType::I32,
            RocmStorageSlice::I64(_) => DType::I64,
            RocmStorageSlice::BF16(_) => DType::BF16,
            RocmStorageSlice::F16(_) => DType::F16,
            RocmStorageSlice::F32(_) => DType::F32,
            RocmStorageSlice::F64(_) => DType::F64,
            RocmStorageSlice::F8E4M3(_) => DType::F8E4M3,
        }
    }

    /// Base device pointer, for reading.
    ///
    /// Deliberately `*const`: this used to hand a `*mut` out of a `&self`, which
    /// let any caller obtain a writable device pointer through a shared
    /// reference without writing `unsafe`. Writers take [`Self::as_mut_ptr`],
    /// which needs the `&mut` that says so.
    pub fn as_ptr(&self) -> *const c_void {
        match self {
            RocmStorageSlice::U8(m) => m.as_ptr(),
            RocmStorageSlice::U32(m) => m.as_ptr(),
            RocmStorageSlice::I16(m) => m.as_ptr(),
            RocmStorageSlice::I32(m) => m.as_ptr(),
            RocmStorageSlice::I64(m) => m.as_ptr(),
            RocmStorageSlice::BF16(m) => m.as_ptr(),
            RocmStorageSlice::F16(m) => m.as_ptr(),
            RocmStorageSlice::F32(m) => m.as_ptr(),
            RocmStorageSlice::F64(m) => m.as_ptr(),
            RocmStorageSlice::F8E4M3(m) => m.as_ptr(),
        }
    }

    /// Base device pointer, for writing.
    pub fn as_mut_ptr(&mut self) -> *mut c_void {
        self.as_ptr() as *mut c_void
    }

    pub(super) fn elem_size(&self) -> usize {
        match self {
            RocmStorageSlice::U8(_) | RocmStorageSlice::F8E4M3(_) => 1,
            RocmStorageSlice::I16(_) | RocmStorageSlice::BF16(_) | RocmStorageSlice::F16(_) => 2,
            RocmStorageSlice::U32(_) | RocmStorageSlice::I32(_) | RocmStorageSlice::F32(_) => 4,
            RocmStorageSlice::I64(_) | RocmStorageSlice::F64(_) => 8,
        }
    }

    /// Device pointer to element `offset`, for reading.
    ///
    /// # Safety
    /// `offset` must be an element index within the allocation — this scales it
    /// by [`Self::elem_size`] and offsets the base pointer, so an out-of-range
    /// index yields a pointer outside the buffer and any kernel reading through
    /// it faults or reads another tensor's memory. `offset` must be an *element*
    /// count, not a byte count.
    pub(super) unsafe fn offset_ptr(&self, offset: usize) -> *const c_void {
        self.as_ptr().add(offset * self.elem_size())
    }

    /// Device pointer to element `offset`, for writing.
    ///
    /// # Safety
    /// As [`Self::offset_ptr`].
    pub(super) unsafe fn offset_mut_ptr(&mut self, offset: usize) -> *mut c_void {
        let elem_size = self.elem_size();
        self.as_mut_ptr().add(offset * elem_size)
    }

    /// Number of elements the underlying allocation holds.
    pub(super) fn count(&self) -> usize {
        match self {
            RocmStorageSlice::U8(m) => m.count(),
            RocmStorageSlice::U32(m) => m.count(),
            RocmStorageSlice::I16(m) => m.count(),
            RocmStorageSlice::I32(m) => m.count(),
            RocmStorageSlice::I64(m) => m.count(),
            RocmStorageSlice::BF16(m) => m.count(),
            RocmStorageSlice::F16(m) => m.count(),
            RocmStorageSlice::F32(m) => m.count(),
            RocmStorageSlice::F64(m) => m.count(),
            // One byte per element, so the u8 count is the element count.
            RocmStorageSlice::F8E4M3(m) => m.count(),
        }
    }
}
