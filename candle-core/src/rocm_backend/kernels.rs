// candle-core/src/rocm_backend/kernels.rs
// TEAM-492: Direct kernel loading from rocm-rs with Candle CUDA parity
// Matches Candle's CUDA kernel calling convention EXACTLY

use crate::rocm_backend::{Result, RocmError};
use crate::{DType, Layout};
use rocm_rs::hip::{DeviceMemory, Dim3, Function, Module};
use std::ffi::c_void;
use std::sync::Once;

static INIT: Once = Once::new();
static mut ROCM_RS_MODULE: Option<Module> = None;

/// Initialize rocm-rs kernel module
pub fn init_kernels() -> Result<()> {
    INIT.call_once(|| {
        match rocm_rs::rocarray::kernels::init() {
            Ok(_) => {}
            Err(e) => {
                eprintln!("Failed to initialize rocm-rs kernels: {:?}", e);
            }
        }
    });
    Ok(())
}

/// Get a kernel function from rocm-rs module
pub fn get_kernel(name: &str) -> Result<Function> {
    init_kernels()?;
    rocm_rs::rocarray::kernels::get_function(name)
        .map_err(|e| RocmError::KernelError(format!("Failed to get kernel '{}': {:?}", name, e)))
}

/// SlicePtrOrNull - matches Candle CUDA pattern
/// Either a pointer to dims/strides info, or null for contiguous tensors
pub enum SlicePtrOrNull {
    Ptr(DeviceMemory<usize>),
    Null,
}

impl SlicePtrOrNull {
    /// Create from layout - matches Candle's params_from_layout
    pub fn from_layout(device: &rocm_rs::hip::Device, layout: &Layout) -> Result<Self> {
        if layout.is_contiguous() {
            Ok(SlicePtrOrNull::Null)
        } else {
            // Concatenate dims and strides like Candle does
            let mut info = Vec::with_capacity(layout.shape().rank() * 2);
            info.extend_from_slice(layout.dims());
            info.extend_from_slice(layout.stride());
            
            let device_info = device
                .htod_copy(info)
                .map_err(|e| RocmError::KernelError(format!("Failed to copy layout info: {:?}", e)))?;
            Ok(SlicePtrOrNull::Ptr(device_info))
        }
    }

    /// Get pointer for kernel arg (null or actual pointer)
    pub fn as_ptr(&self) -> *const c_void {
        match self {
            SlicePtrOrNull::Ptr(mem) => mem.as_ptr() as *const c_void,
            SlicePtrOrNull::Null => std::ptr::null(),
        }
    }
}

/// Launch config - matches Candle's LaunchConfig::for_num_elems
#[inline]
pub fn launch_config_for_num_elems(num_elems: u32) -> (Dim3, Dim3) {
    let block_size = 256u32;
    let grid_size = (num_elems + block_size - 1) / block_size;
    (Dim3::new(grid_size, 1, 1), Dim3::new(block_size, 1, 1))
}

/// Launch unary operation kernel - MATCHES Candle CUDA signature
/// Signature: (numel, num_dims, info, inp, out)
pub fn launch_unary<T>(
    kernel_name: &str,
    device: &rocm_rs::hip::Device,
    src: &DeviceMemory<T>,
    layout: &Layout,
) -> Result<DeviceMemory<T>> {
    let func = get_kernel(kernel_name)?;
    let shape = layout.shape();
    let el = shape.elem_count();
    let (grid, block) = launch_config_for_num_elems(el as u32);
    
    let ds = SlicePtrOrNull::from_layout(device, layout)?;
    let src_offset = &src.slice(layout.start_offset()..);
    
    // Allocate output
    let out = device
        .alloc::<T>(el)
        .map_err(|e| RocmError::OutOfMemory { requested: el * std::mem::size_of::<T>() })?;
    
    // Build args: (numel, num_dims, info, inp, out)
    let mut args = [
        &(el as usize) as *const usize as *mut c_void,
        &shape.rank() as *const usize as *mut c_void,
        ds.as_ptr() as *mut c_void,
        src_offset.as_ptr() as *mut c_void,
        out.as_ptr() as *mut c_void,
    ];
    
    unsafe {
        func.launch(grid, block, 0, None, &mut args)
            .map_err(|e| RocmError::KernelError(format!("Unary kernel launch failed: {:?}", e)))?;
    }
    
    Ok(out)
}

/// Launch affine operation kernel - MATCHES Candle CUDA signature
/// Signature: (numel, num_dims, info, inp, out, mul, add)
pub fn launch_affine<T>(
    kernel_name: &str,
    device: &rocm_rs::hip::Device,
    src: &DeviceMemory<T>,
    layout: &Layout,
    mul: T,
    add: T,
) -> Result<DeviceMemory<T>>
where
    T: Copy,
{
    let func = get_kernel(kernel_name)?;
    let shape = layout.shape();
    let el = shape.elem_count();
    let (grid, block) = launch_config_for_num_elems(el as u32);
    
    let ds = SlicePtrOrNull::from_layout(device, layout)?;
    let src_offset = &src.slice(layout.start_offset()..);
    
    let out = device
        .alloc::<T>(el)
        .map_err(|e| RocmError::OutOfMemory { requested: el * std::mem::size_of::<T>() })?;
    
    // Build args: (numel, num_dims, info, inp, out, mul, add)
    let mut args = [
        &(el as usize) as *const usize as *mut c_void,
        &shape.rank() as *const usize as *mut c_void,
        ds.as_ptr() as *mut c_void,
        src_offset.as_ptr() as *mut c_void,
        out.as_ptr() as *mut c_void,
        &mul as *const T as *mut c_void,
        &add as *const T as *mut c_void,
    ];
    
    unsafe {
        func.launch(grid, block, 0, None, &mut args)
            .map_err(|e| RocmError::KernelError(format!("Affine kernel launch failed: {:?}", e)))?;
    }
    
    Ok(out)
}

/// Launch ternary (where) operation kernel - MATCHES Candle CUDA signature
/// Signature: (numel, num_dims, info, ids, t, f, out)
pub fn launch_ternary<C, T>(
    kernel_name: &str,
    device: &rocm_rs::hip::Device,
    cond: &DeviceMemory<C>,
    cond_layout: &Layout,
    true_vals: &DeviceMemory<T>,
    true_layout: &Layout,
    false_vals: &DeviceMemory<T>,
    false_layout: &Layout,
) -> Result<DeviceMemory<T>> {
    let func = get_kernel(kernel_name)?;
    let shape = cond_layout.shape();
    let el = shape.elem_count();
    let (grid, block) = launch_config_for_num_elems(el as u32);
    
    // Candle's ternary uses SEPARATE strides for cond, true_vals, false_vals!
    // info layout: [dims, cond_strides, true_strides, false_strides]
    let mut info = Vec::with_capacity(shape.rank() * 4);
    info.extend_from_slice(cond_layout.dims());
    info.extend_from_slice(cond_layout.stride());
    info.extend_from_slice(true_layout.stride());
    info.extend_from_slice(false_layout.stride());
    
    let device_info = device
        .htod_copy(info)
        .map_err(|e| RocmError::KernelError(format!("Failed to copy ternary layout info: {:?}", e)))?;
    
    let cond_offset = &cond.slice(cond_layout.start_offset()..);
    let true_offset = &true_vals.slice(true_layout.start_offset()..);
    let false_offset = &false_vals.slice(false_layout.start_offset()..);
    
    let out = device
        .alloc::<T>(el)
        .map_err(|e| RocmError::OutOfMemory { requested: el * std::mem::size_of::<T>() })?;
    
    // Build args: (numel, num_dims, info, ids, t, f, out)
    let mut args = [
        &(el as usize) as *const usize as *mut c_void,
        &shape.rank() as *const usize as *mut c_void,
        device_info.as_ptr() as *mut c_void,
        cond_offset.as_ptr() as *mut c_void,
        true_offset.as_ptr() as *mut c_void,
        false_offset.as_ptr() as *mut c_void,
        out.as_ptr() as *mut c_void,
    ];
    
    unsafe {
        func.launch(grid, block, 0, None, &mut args)
            .map_err(|e| RocmError::KernelError(format!("Ternary kernel launch failed: {:?}", e)))?;
    }
    
    Ok(out)
}

/// Launch cast operation kernel - MATCHES Candle CUDA signature
/// Signature: (numel, num_dims, info, inp, out)
/// NOTE: Input and output types are DIFFERENT!
pub fn launch_cast<I, O>(
    kernel_name: &str,
    device: &rocm_rs::hip::Device,
    src: &DeviceMemory<I>,
    layout: &Layout,
) -> Result<DeviceMemory<O>> {
    let func = get_kernel(kernel_name)?;
    let shape = layout.shape();
    let el = shape.elem_count();
    let (grid, block) = launch_config_for_num_elems(el as u32);
    
    let ds = SlicePtrOrNull::from_layout(device, layout)?;
    let src_offset = &src.slice(layout.start_offset()..);
    
    // Allocate output with OUTPUT type
    let out = device
        .alloc::<O>(el)
        .map_err(|e| RocmError::OutOfMemory { requested: el * std::mem::size_of::<O>() })?;
    
    // Build args: (numel, num_dims, info, inp, out)
    let mut args = [
        &(el as usize) as *const usize as *mut c_void,
        &shape.rank() as *const usize as *mut c_void,
        ds.as_ptr() as *mut c_void,
        src_offset.as_ptr() as *mut c_void,
        out.as_ptr() as *mut c_void,
    ];
    
    unsafe {
        func.launch(grid, block, 0, None, &mut args)
            .map_err(|e| RocmError::KernelError(format!("Cast kernel launch failed: {:?}", e)))?;
    }
    
    Ok(out)
}

/// Launch binary operation kernel - MATCHES Candle CUDA signature
/// Signature: (numel, num_dims, info, lhs, rhs, out)
/// TEAM-494: Added for binary operations (add, sub, mul, div)
pub fn launch_binary<T>(
    kernel_name: &str,
    device: &rocm_rs::hip::Device,
    lhs: &DeviceMemory<T>,
    lhs_layout: &Layout,
    rhs: &DeviceMemory<T>,
    rhs_layout: &Layout,
) -> Result<DeviceMemory<T>> {
    let func = get_kernel(kernel_name)?;
    let shape = lhs_layout.shape();
    let el = shape.elem_count();
    let (grid, block) = launch_config_for_num_elems(el as u32);
    
    // Binary ops use SEPARATE strides for lhs and rhs
    // info layout: [dims, lhs_strides, rhs_strides]
    let mut info = Vec::with_capacity(shape.rank() * 3);
    info.extend_from_slice(lhs_layout.dims());
    info.extend_from_slice(lhs_layout.stride());
    info.extend_from_slice(rhs_layout.stride());
    
    let device_info = device
        .htod_copy(info)
        .map_err(|e| RocmError::KernelError(format!("Failed to copy binary layout info: {:?}", e)))?;
    
    let lhs_offset = &lhs.slice(lhs_layout.start_offset()..);
    let rhs_offset = &rhs.slice(rhs_layout.start_offset()..);
    
    let out = device
        .alloc::<T>(el)
        .map_err(|e| RocmError::OutOfMemory { requested: el * std::mem::size_of::<T>() })?;
    
    // Build args: (numel, num_dims, info, lhs, rhs, out)
    let mut args = [
        &(el as usize) as *const usize as *mut c_void,
        &shape.rank() as *const usize as *mut c_void,
        device_info.as_ptr() as *mut c_void,
        lhs_offset.as_ptr() as *mut c_void,
        rhs_offset.as_ptr() as *mut c_void,
        out.as_ptr() as *mut c_void,
    ];
    
    unsafe {
        func.launch(grid, block, 0, None, &mut args)
            .map_err(|e| RocmError::KernelError(format!("Binary kernel launch failed: {:?}", e)))?;
    }
    
    Ok(out)
}

/// Launch reduce operation kernel - MATCHES Candle CUDA signature
/// Signature: (numel, num_dims, info, inp, out)
/// TEAM-494: Added for reduce operations (sum, min, max)
pub fn launch_reduce<T>(
    kernel_name: &str,
    device: &rocm_rs::hip::Device,
    src: &DeviceMemory<T>,
    layout: &Layout,
    sum_dims: &[usize],
) -> Result<DeviceMemory<T>>
where
    T: Copy + Default,
{
    let func = get_kernel(kernel_name)?;
    let shape = layout.shape();
    
    // Calculate output shape (reduced dimensions)
    let mut out_dims = shape.dims().to_vec();
    for &dim in sum_dims.iter().rev() {
        out_dims.remove(dim);
    }
    let out_el = if out_dims.is_empty() { 1 } else { out_dims.iter().product() };
    
    let el = shape.elem_count();
    let (grid, block) = launch_config_for_num_elems(el as u32);
    
    let ds = SlicePtrOrNull::from_layout(device, layout)?;
    let src_offset = &src.slice(layout.start_offset()..);
    
    // Allocate output with reduced size
    let out = device
        .alloc::<T>(out_el)
        .map_err(|e| RocmError::OutOfMemory { requested: out_el * std::mem::size_of::<T>() })?;
    
    // Build args: (numel, num_dims, info, inp, out)
    let mut args = [
        &(el as usize) as *const usize as *mut c_void,
        &shape.rank() as *const usize as *mut c_void,
        ds.as_ptr() as *mut c_void,
        src_offset.as_ptr() as *mut c_void,
        out.as_ptr() as *mut c_void,
    ];
    
    unsafe {
        func.launch(grid, block, 0, None, &mut args)
            .map_err(|e| RocmError::KernelError(format!("Reduce kernel launch failed: {:?}", e)))?;
    }
    
    Ok(out)
}
