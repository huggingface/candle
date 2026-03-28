//! Implementation of Backend traits for ROCm (AMD GPU) device
//!
use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, WithDType};
use half::{bf16, f16};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

mod device;
mod error;
pub use device::{DeviceId, RocmDevice};
pub use error::{RocmError, WrapErr};

use rocm_rs::hip::{kernel::AsKernelArg, Device, DeviceMemory, Dim3, Function, Module, Stream};

/// Helper enum for handling optional device pointers for non-contiguous layouts
pub enum SlicePtrOrNull<T> {
    Ptr(DeviceMemory<T>),
    Null,
}

impl<T> SlicePtrOrNull<T> {
    /// Convert to kernel argument
    pub fn as_kernel_arg(&self) -> *mut std::ffi::c_void {
        match self {
            SlicePtrOrNull::Ptr(mem) => mem.as_ptr(),
            SlicePtrOrNull::Null => std::ptr::null_mut(),
        }
    }
}

impl SlicePtrOrNull<usize> {
    pub fn params_from_layout(dev: &RocmDevice, l: &Layout) -> Result<Self> {
        let ds = if l.is_contiguous() {
            SlicePtrOrNull::Null
        } else {
            // Concat dims and strides into a single buffer
            let mut dims_strides = l.dims().to_vec();
            dims_strides.extend_from_slice(l.stride());
            SlicePtrOrNull::Ptr(dev.clone_htod(&dims_strides)?)
        };
        Ok(ds)
    }
}

/// Storage enum for different data types on ROCm device
#[derive(Debug)]
pub enum RocmStorageSlice {
    U8(DeviceMemory<u8>),
    U32(DeviceMemory<u32>),
    I16(DeviceMemory<i16>),
    I32(DeviceMemory<i32>),
    I64(DeviceMemory<i64>),
    BF16(DeviceMemory<bf16>),
    F16(DeviceMemory<f16>),
    F32(DeviceMemory<f32>),
    F64(DeviceMemory<f64>),
    // Note: F8E4M3, F6E2M3, F6E3M2, F4, F8E8M0 are not supported on ROCm in the same way
    // For unsupported types, we can either use CPU fallback or store as raw bytes
    F8E4M3(DeviceMemory<u8>), // Placeholder - needs proper handling
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
}

/// Generate kernel name based on operation and dtype
pub fn kernel_name<T: WithDType>(root: &str) -> String {
    let dtype = T::DTYPE.as_str();
    format!("{root}_{dtype}")
}

/// Helper trait for mapping unary operations
pub trait Map1 {
    fn f<T: AsKernelArg>(
        &self,
        s: &DeviceMemory<T>,
        dev: &RocmDevice,
        layout: &Layout,
    ) -> Result<DeviceMemory<T>>;
}

/// Helper trait for mapping binary operations
pub trait Map2 {
    fn f<T: AsKernelArg>(
        &self,
        s1: &DeviceMemory<T>,
        s2: &DeviceMemory<T>,
        dev: &RocmDevice,
        l1: &Layout,
        l2: &Layout,
    ) -> Result<DeviceMemory<T>>;
}

/// Simple clone operation
struct Clone;
impl Map1 for Clone {
    fn f<T: AsKernelArg>(
        &self,
        s: &DeviceMemory<T>,
        dev: &RocmDevice,
        _: &Layout,
    ) -> Result<DeviceMemory<T>> {
        dev.memcpy_dtod(s, s)
    }
}

/// Affine transformation: out = a * x + b
struct Affine(f64, f64);
impl Map1 for Affine {
    fn f<T: AsKernelArg + WithDType>(
        &self,
        src: &DeviceMemory<T>,
        dev: &RocmDevice,
        layout: &Layout,
    ) -> Result<DeviceMemory<T>> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el = shape.elem_count();

        // Get kernel function
        let func = dev.get_or_load_kernel(&kernel_name::<T>("affine"))?;

        // Allocate output
        let mut out = DeviceMemory::<T>::new(el)?;

        // Setup launch config
        let block_size = 256;
        let grid_dim = Dim3::new_1d(((el as u32) + block_size - 1) / block_size);
        let block_dim = Dim3::new_1d(block_size);

        // Get layout params
        let ds = SlicePtrOrNull::params_from_layout(dev, layout)?;

        // Prepare kernel arguments
        let el_u32 = el as u32;
        let dims_len = dims.len() as u32;
        let a_val = T::from_f64(self.0);
        let b_val = T::from_f64(self.1);

        // Pack arguments according to kernel signature:
        // affine(el, ndims, dims_strides, src, out, a, b)
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &el_u32 as *const _ as *mut _,
            &dims_len as *const _ as *mut _,
            ds.as_kernel_arg(),
            src.as_kernel_arg(),
            out.as_kernel_arg(),
            &a_val as *const _ as *mut _,
            &b_val as *const _ as *mut _,
        ];

        // Launch kernel
        func.launch(grid_dim, block_dim, 0, Some(&dev.stream), &mut args)?;

        Ok(out)
    }
}

/// ELU activation: x < 0 ? alpha * (exp(x) - 1) : x
struct Elu(f64);
impl Map1 for Elu {
    fn f<T: AsKernelArg + WithDType>(
        &self,
        src: &DeviceMemory<T>,
        dev: &RocmDevice,
        layout: &Layout,
    ) -> Result<DeviceMemory<T>> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el = shape.elem_count();

        let func = dev.get_or_load_kernel(&kernel_name::<T>("uelu"))?;
        let mut out = DeviceMemory::<T>::new(el)?;

        let block_size = 256;
        let grid_dim = Dim3::new_1d(((el as u32) + block_size - 1) / block_size);
        let block_dim = Dim3::new_1d(block_size);

        let ds = SlicePtrOrNull::params_from_layout(dev, layout)?;

        let el_u32 = el as u32;
        let dims_len = dims.len() as u32;
        let alpha_val = T::from_f64(self.0);

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &el_u32 as *const _ as *mut _,
            &dims_len as *const _ as *mut _,
            ds.as_kernel_arg(),
            &alpha_val as *const _ as *mut _,
            src.as_kernel_arg(),
            out.as_kernel_arg(),
        ];

        func.launch(grid_dim, block_dim, 0, Some(&dev.stream), &mut args)?;

        Ok(out)
    }
}

/// Powf operation: x^e
struct Powf(f64);
impl Map1 for Powf {
    fn f<T: AsKernelArg + WithDType>(
        &self,
        src: &DeviceMemory<T>,
        dev: &RocmDevice,
        layout: &Layout,
    ) -> Result<DeviceMemory<T>> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el = shape.elem_count();

        let func = dev.get_or_load_kernel(&kernel_name::<T>("upowf"))?;
        let mut out = DeviceMemory::<T>::new(el)?;

        let block_size = 256;
        let grid_dim = Dim3::new_1d(((el as u32) + block_size - 1) / block_size);
        let block_dim = Dim3::new_1d(block_size);

        let ds = SlicePtrOrNull::params_from_layout(dev, layout)?;

        let el_u32 = el as u32;
        let dims_len = dims.len() as u32;
        let exp_val = T::from_f64(self.0);

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &el_u32 as *const _ as *mut _,
            &dims_len as *const _ as *mut _,
            ds.as_kernel_arg(),
            &exp_val as *const _ as *mut _,
            src.as_kernel_arg(),
            out.as_kernel_arg(),
        ];

        func.launch(grid_dim, block_dim, 0, Some(&dev.stream), &mut args)?;

        Ok(out)
    }
}

/// Storage implementation for ROCm
#[derive(Debug)]
pub struct RocmStorage {
    pub slice: RocmStorageSlice,
    pub device: RocmDevice,
}
