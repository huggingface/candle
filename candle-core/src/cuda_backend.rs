use crate::{CpuStorage, DType, Layout, Shape};
use candle_kernels as kernels;
use cudarc::cublas::{Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::driver::{CudaFunction, CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};
use half::{bf16, f16};
use std::sync::Arc;

/// cudarc related errors
#[derive(thiserror::Error, Debug)]
pub enum CudaError {
    #[error(transparent)]
    Cuda(#[from] cudarc::driver::DriverError),

    #[error(transparent)]
    Compiler(#[from] cudarc::nvrtc::CompileError),

    #[error(transparent)]
    Cublas(#[from] cudarc::cublas::result::CublasError),

    #[error("{op} only supports contiguous tensors")]
    RequiresContiguous { op: &'static str },

    #[error("missing kernel '{module_name}'")]
    MissingKernel { module_name: String },

    #[error("internal error '{0}'")]
    InternalError(&'static str),

    #[error("internal error '{0}'")]
    WrappedError(Box<dyn std::error::Error>),

    #[error("matmul is only supported for contiguous tensors lstride: {lhs_stride:?} rstride: {rhs_stride:?} mnk: {mnk:?}")]
    MatMulNonContiguous {
        lhs_stride: Vec<usize>,
        rhs_stride: Vec<usize>,
        mnk: (usize, usize, usize),
    },

    #[error("{msg}, expected: {expected:?}, got: {got:?}")]
    UnexpectedDType {
        msg: &'static str,
        expected: DType,
        got: DType,
    },

    #[error("{cuda} when loading {module_name}")]
    Load {
        cuda: cudarc::driver::DriverError,
        module_name: String,
    },
}

type Result<T> = std::result::Result<T, CudaError>;

/// Unique identifier for cuda devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct DeviceId(usize);

impl DeviceId {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

#[derive(Debug, Clone)]
pub struct CudaDevice {
    id: DeviceId,
    device: Arc<cudarc::driver::CudaDevice>,
    #[allow(dead_code)]
    blas: Arc<cudarc::cublas::CudaBlas>,
}

impl std::ops::Deref for CudaDevice {
    type Target = Arc<cudarc::driver::CudaDevice>;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl CudaDevice {
    pub(crate) fn new(ordinal: usize) -> Result<Self> {
        let device = cudarc::driver::CudaDevice::new(ordinal)?;
        let blas = cudarc::cublas::CudaBlas::new(device.clone())?;
        Ok(Self {
            id: DeviceId::new(),
            device,
            blas: Arc::new(blas),
        })
    }

    pub(crate) fn same_id(&self, rhs: &Self) -> bool {
        self.id == rhs.id
    }

    pub(crate) fn ordinal(&self) -> usize {
        self.device.ordinal()
    }

    pub(crate) fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U32 => {
                let data = self.alloc_zeros::<u32>(elem_count)?;
                CudaStorageSlice::U32(data)
            }
            DType::BF16 => {
                let data = self.alloc_zeros::<bf16>(elem_count)?;
                CudaStorageSlice::BF16(data)
            }
            DType::F16 => {
                let data = self.alloc_zeros::<f16>(elem_count)?;
                CudaStorageSlice::F16(data)
            }
            DType::F32 => {
                let data = self.alloc_zeros::<f32>(elem_count)?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let data = self.alloc_zeros::<f64>(elem_count)?;
                CudaStorageSlice::F64(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    pub(crate) fn const_impl(&self, v: f64, shape: &Shape, dtype: DType) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(elem_count as u32);
        let slice = match dtype {
            DType::U32 => {
                // SAFETY: Set later by running the fill kernel.
                let data = unsafe { self.alloc::<u32>(elem_count) }?;
                let func = self.get_or_load_func("fill_u32", kernels::FILL)?;
                let params = (&data, v as u32, elem_count);
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::U32(data)
            }
            DType::BF16 => {
                // SAFETY: Set later by running the fill kernel.
                let data = unsafe { self.alloc::<bf16>(elem_count) }?;
                let func = self.get_or_load_func("fill_bf16", kernels::FILL)?;
                let params = (&data, bf16::from_f64(v), elem_count);
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::BF16(data)
            }
            DType::F16 => {
                // SAFETY: Set later by running the fill kernel.
                let data = unsafe { self.alloc::<f16>(elem_count) }?;
                let func = self.get_or_load_func("fill_f16", kernels::FILL)?;
                let params = (&data, f16::from_f64(v), elem_count);
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::F16(data)
            }
            DType::F32 => {
                // SAFETY: Set later by running the fill kernel.
                let data = unsafe { self.alloc::<f32>(elem_count) }?;
                let func = self.get_or_load_func("fill_f32", kernels::FILL)?;
                let params = (&data, v as f32, elem_count);
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                // SAFETY: Set later by running the fill kernel.
                let data = unsafe { self.alloc::<f64>(elem_count) }?;
                let func = self.get_or_load_func("fill_f64", kernels::FILL)?;
                let params = (&data, v, elem_count);
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::F64(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    pub(crate) fn ones_impl(&self, shape: &Shape, dtype: DType) -> Result<CudaStorage> {
        self.const_impl(1., shape, dtype)
    }

    pub(crate) fn cuda_from_cpu_storage(&self, storage: &CpuStorage) -> Result<CudaStorage> {
        let slice = match storage {
            CpuStorage::U32(storage) => {
                let data = self.htod_sync_copy(storage)?;
                CudaStorageSlice::U32(data)
            }
            CpuStorage::BF16(storage) => {
                let data = self.htod_sync_copy(storage)?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorage::F16(storage) => {
                let data = self.htod_sync_copy(storage)?;
                CudaStorageSlice::F16(data)
            }
            CpuStorage::F32(storage) => {
                let data = self.htod_sync_copy(storage)?;
                CudaStorageSlice::F32(data)
            }
            CpuStorage::F64(storage) => {
                let data = self.htod_sync_copy(storage)?;
                CudaStorageSlice::F64(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn get_or_load_func(&self, module_name: &str, ptx: &'static str) -> Result<CudaFunction> {
        if !self.has_func(module_name, module_name) {
            // Leaking the string here is a bit sad but we need a &'static str and this is only
            // done once per kernel name.
            let static_module_name = Box::leak(module_name.to_string().into_boxed_str());
            self.load_ptx(ptx.into(), module_name, &[static_module_name])
                .map_err(|cuda| CudaError::Load {
                    cuda,
                    module_name: module_name.to_string(),
                })?;
        }
        self.get_func(module_name, module_name)
            // Clippy recommends this `ok_or` rather than `ok_or_else` so hopefully the compiler is
            // able to only build the error value if needed.
            .ok_or(CudaError::MissingKernel {
                module_name: module_name.to_string(),
            })
    }
}

#[derive(Debug)]
enum CudaStorageSlice {
    U32(CudaSlice<u32>),
    BF16(CudaSlice<bf16>),
    F16(CudaSlice<f16>),
    F32(CudaSlice<f32>),
    F64(CudaSlice<f64>),
}

fn slice_src_and_dst<'a, T>(
    src: &'a CudaSlice<T>,
    src_offset: usize,
    dst: &'a mut CudaSlice<T>,
    dst_offset: usize,
) -> (
    cudarc::driver::CudaView<'a, T>,
    cudarc::driver::CudaViewMut<'a, T>,
) {
    let to_copy = dst
        .len()
        .saturating_sub(dst_offset)
        .min(src.len().saturating_sub(src_offset));
    let src = src.slice(src_offset..src_offset + to_copy);
    let dst = dst.slice_mut(dst_offset..dst_offset + to_copy);
    (src, dst)
}

#[derive(Debug)]
pub struct CudaStorage {
    slice: CudaStorageSlice,
    device: CudaDevice,
}

fn gemm_config<T>(
    alpha: T,
    beta: T,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_l: &Layout,
    rhs_l: &Layout,
) -> Result<StridedBatchedConfig<T>> {
    // https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm
    use cudarc::cublas::sys::cublasOperation_t;

    let lhs_stride = lhs_l.stride();
    let rhs_stride = rhs_l.stride();
    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];
    // The a tensor has dims batching, k, n (rhs)
    let (lda, transa) = if rhs_m1 == 1 && rhs_m2 == n {
        (n as i32, cublasOperation_t::CUBLAS_OP_N)
    } else if rhs_m1 == k && rhs_m2 == 1 {
        (k as i32, cublasOperation_t::CUBLAS_OP_T)
    } else {
        Err(CudaError::MatMulNonContiguous {
            lhs_stride: lhs_stride.to_vec(),
            rhs_stride: rhs_stride.to_vec(),
            mnk: (m, n, k),
        })?
    };
    // The b tensor has dims batching, m, k (lhs)
    let (ldb, transb) = if lhs_m1 == 1 && lhs_m2 == k {
        (k as i32, cublasOperation_t::CUBLAS_OP_N)
    } else if lhs_m1 == m && lhs_m2 == 1 {
        (m as i32, cublasOperation_t::CUBLAS_OP_T)
    } else {
        Err(CudaError::MatMulNonContiguous {
            lhs_stride: lhs_stride.to_vec(),
            rhs_stride: rhs_stride.to_vec(),
            mnk: (m, n, k),
        })?
    };
    // The setup below was copied from:
    // https://github.com/lebedov/scikit-cuda/blob/7e7300474286019c917a6c8a4bca59405c64fbce/tests/test_cublas.py#L531
    let gemm = GemmConfig {
        alpha,
        beta,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        lda,
        ldb,
        ldc: n as i32,
        transa,
        transb,
    };
    Ok(StridedBatchedConfig {
        batch_size: b as i32,
        gemm,
        stride_a: (n * k) as i64,
        stride_b: (m * k) as i64,
        stride_c: (m * n) as i64,
    })
}

impl CudaStorage {
    pub fn try_clone(&self) -> Result<Self> {
        let slice = match &self.slice {
            CudaStorageSlice::U32(slice) => CudaStorageSlice::U32(slice.try_clone()?),
            CudaStorageSlice::BF16(slice) => CudaStorageSlice::BF16(slice.try_clone()?),
            CudaStorageSlice::F16(slice) => CudaStorageSlice::F16(slice.try_clone()?),
            CudaStorageSlice::F32(slice) => CudaStorageSlice::F32(slice.try_clone()?),
            CudaStorageSlice::F64(slice) => CudaStorageSlice::F64(slice.try_clone()?),
        };
        let device = self.device.clone();
        Ok(Self { slice, device })
    }

    pub fn dtype(&self) -> DType {
        match self.slice {
            CudaStorageSlice::U32(_) => DType::U32,
            CudaStorageSlice::BF16(_) => DType::BF16,
            CudaStorageSlice::F16(_) => DType::F16,
            CudaStorageSlice::F32(_) => DType::F32,
            CudaStorageSlice::F64(_) => DType::F64,
        }
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    pub(crate) fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        use cudarc::driver::DevicePtr;
        let shape = layout.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(el as u32);
        let dev = self.device();
        let ds = dev.htod_copy([dims, layout.stride()].concat())?;
        let start_o = layout.start_offset();
        let inp = match &self.slice {
            CudaStorageSlice::U32(inp) => inp.slice(start_o..).device_ptr(),
            CudaStorageSlice::BF16(inp) => inp.slice(start_o..).device_ptr(),
            CudaStorageSlice::F16(inp) => inp.slice(start_o..).device_ptr(),
            CudaStorageSlice::F32(inp) => inp.slice(start_o..).device_ptr(),
            CudaStorageSlice::F64(inp) => inp.slice(start_o..).device_ptr(),
        };
        let kernel_name = format!("cast_{}_{}", self.dtype().as_str(), dtype.as_str());
        let func = dev.get_or_load_func(&kernel_name, kernels::CAST)?;
        let slice = match dtype {
            DType::U32 => {
                let out = unsafe { dev.alloc::<u32>(el) }?;
                let params = (el, dims.len(), &ds, *inp, &out);
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::U32(out)
            }
            DType::BF16 => {
                let out = unsafe { dev.alloc::<bf16>(el) }?;
                let params = (el, dims.len(), &ds, *inp, &out);
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::BF16(out)
            }
            DType::F16 => {
                let out = unsafe { dev.alloc::<f16>(el) }?;
                let params = (el, dims.len(), &ds, *inp, &out);
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::F16(out)
            }
            DType::F32 => {
                let out = unsafe { dev.alloc::<f32>(el) }?;
                let params = (el, dims.len(), &ds, *inp, &out);
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::F32(out)
            }
            DType::F64 => {
                let out = unsafe { dev.alloc::<f64>(el) }?;
                let params = (el, dims.len(), &ds, *inp, &out);
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::F64(out)
            }
        };
        Ok(Self {
            slice,
            device: dev.clone(),
        })
    }

    pub(crate) fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el_count = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(el_count as u32);
        let dev = self.device();
        let ds = dev.htod_copy([dims, layout.stride()].concat())?;
        let slice = match &self.slice {
            CudaStorageSlice::U32(arg) => {
                let arg = &arg.slice(layout.start_offset()..);
                let func = dev.get_or_load_func("affine_u32", kernels::AFFINE)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.alloc::<u32>(el_count) }?;
                let params = (el_count, dims.len(), &ds, arg, &out, mul as u32, add as u32);
                // SAFETY: ffi.
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::U32(out)
            }
            CudaStorageSlice::BF16(arg) => {
                let arg = &arg.slice(layout.start_offset()..);
                let func = dev.get_or_load_func("affine_bf16", kernels::AFFINE)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.alloc::<bf16>(el_count) }?;
                let params = (
                    el_count,
                    dims.len(),
                    &ds,
                    arg,
                    &out,
                    bf16::from_f64(mul),
                    bf16::from_f64(add),
                );
                // SAFETY: ffi.
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::BF16(out)
            }
            CudaStorageSlice::F16(arg) => {
                let arg = &arg.slice(layout.start_offset()..);
                let func = dev.get_or_load_func("affine_f16", kernels::AFFINE)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.alloc::<f16>(el_count) }?;
                let params = (
                    el_count,
                    dims.len(),
                    &ds,
                    arg,
                    &out,
                    f16::from_f64(mul),
                    f16::from_f64(add),
                );
                // SAFETY: ffi.
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::F16(out)
            }
            CudaStorageSlice::F32(arg) => {
                let arg = &arg.slice(layout.start_offset()..);
                let func = dev.get_or_load_func("affine_f32", kernels::AFFINE)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.alloc::<f32>(el_count) }?;
                let params = (el_count, dims.len(), &ds, arg, &out, mul as f32, add as f32);
                // SAFETY: ffi.
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::F32(out)
            }
            CudaStorageSlice::F64(arg) => {
                let arg = &arg.slice(layout.start_offset()..);
                let func = dev.get_or_load_func("affine_f64", kernels::AFFINE)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.alloc::<f64>(el_count) }?;
                let params = (el_count, dims.len(), &ds, arg, &out, mul, add);
                // SAFETY: ffi.
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::F64(out)
            }
        };
        let device = dev.clone();
        Ok(Self { slice, device })
    }

    pub(crate) fn sum(&self, layout: &Layout, sum_dims: &[usize]) -> Result<Self> {
        let shape = layout.shape();
        let src_dims = shape.dims();
        let el = shape.elem_count();
        let mut dst_el = el;
        for &sum_dim in sum_dims.iter() {
            dst_el /= src_dims[sum_dim];
        }
        let mut sum_dims = sum_dims.to_vec();
        // Sort the sum_dims as they have to be processed from left to right when converting the
        // indexes.
        sum_dims.sort();
        let sum_dims_l: Vec<usize> = sum_dims.iter().map(|&d| src_dims[d]).collect();
        let sum_dims_s: Vec<usize> = sum_dims
            .iter()
            .map(|&d| src_dims[d + 1..].iter().product::<usize>())
            .collect();
        let cfg = LaunchConfig::for_num_elems(el as u32);
        let dev = self.device();
        let ds = dev.htod_copy([src_dims, layout.stride(), &sum_dims_l, &sum_dims_s].concat())?;
        let slice = match &self.slice {
            CudaStorageSlice::U32(arg) => {
                let arg = &arg.slice(layout.start_offset()..);
                let func = dev.get_or_load_func("sum_u32", kernels::REDUCE)?;
                let out = dev.alloc_zeros::<u32>(dst_el)?;
                let params = (el, src_dims.len(), sum_dims.len(), &ds, arg, &out);
                // SAFETY: ffi.
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::U32(out)
            }
            CudaStorageSlice::BF16(arg) => {
                let arg = &arg.slice(layout.start_offset()..);
                let func = dev.get_or_load_func("sum_bf16", kernels::REDUCE)?;
                let out = dev.alloc_zeros::<bf16>(dst_el)?;
                let params = (el, src_dims.len(), sum_dims.len(), &ds, arg, &out);
                // SAFETY: ffi.
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::BF16(out)
            }
            CudaStorageSlice::F16(arg) => {
                let arg = &arg.slice(layout.start_offset()..);
                let func = dev.get_or_load_func("sum_f16", kernels::REDUCE)?;
                let out = dev.alloc_zeros::<f16>(dst_el)?;
                let params = (el, src_dims.len(), sum_dims.len(), &ds, arg, &out);
                // SAFETY: ffi.
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::F16(out)
            }
            CudaStorageSlice::F32(arg) => {
                let arg = &arg.slice(layout.start_offset()..);
                let func = dev.get_or_load_func("sum_f32", kernels::REDUCE)?;
                let out = dev.alloc_zeros::<f32>(dst_el)?;
                let params = (el, src_dims.len(), sum_dims.len(), &ds, arg, &out);
                // SAFETY: ffi.
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::F32(out)
            }
            CudaStorageSlice::F64(arg) => {
                let arg = &arg.slice(layout.start_offset()..);
                let func = dev.get_or_load_func("sum_f64", kernels::REDUCE)?;
                let out = dev.alloc_zeros::<f64>(dst_el)?;
                let params = (el, src_dims.len(), sum_dims.len(), &ds, arg, &out);
                // SAFETY: ffi.
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::F64(out)
            }
        };
        let device = dev.clone();
        Ok(Self { slice, device })
    }

    pub(crate) fn divide_by_sum_over_dim(&mut self, _: &Shape, _: usize) -> Result<()> {
        Err(CudaError::InternalError(
            "TODO: implement divide_by_sum_over_dim",
        ))
    }

    pub(crate) fn unary_impl<U: crate::op::UnaryOp>(&self, layout: &Layout) -> Result<Self> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el_count = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(el_count as u32);
        let dev = &self.device;
        let ds = dev.htod_copy([dims, layout.stride()].concat())?;
        let slice = match &self.slice {
            CudaStorageSlice::U32(_arg) => {
                todo!("No unary kernels for u32");
            }
            CudaStorageSlice::BF16(arg) => {
                let arg = &arg.slice(layout.start_offset()..);
                let func = dev.get_or_load_func(U::KERNEL_BF16, kernels::UNARY)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.alloc::<bf16>(el_count) }?;
                let params = (el_count, dims.len(), &ds, arg, &out);
                // SAFETY: ffi.
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::BF16(out)
            }
            CudaStorageSlice::F16(arg) => {
                let arg = &arg.slice(layout.start_offset()..);
                let func = dev.get_or_load_func(U::KERNEL_F16, kernels::UNARY)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.alloc::<f16>(el_count) }?;
                let params = (el_count, dims.len(), &ds, arg, &out);
                // SAFETY: ffi.
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::F16(out)
            }
            CudaStorageSlice::F32(arg) => {
                let arg = &arg.slice(layout.start_offset()..);
                let func = dev.get_or_load_func(U::KERNEL_F32, kernels::UNARY)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.alloc::<f32>(el_count) }?;
                let params = (el_count, dims.len(), &ds, arg, &out);
                // SAFETY: ffi.
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::F32(out)
            }
            CudaStorageSlice::F64(arg) => {
                let arg = &arg.slice(layout.start_offset()..);
                let func = dev.get_or_load_func(U::KERNEL_F64, kernels::UNARY)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.alloc::<f64>(el_count) }?;
                let params = (el_count, dims.len(), &ds, arg, &out);
                // SAFETY: ffi.
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::F64(out)
            }
        };
        let device = dev.clone();
        Ok(Self { slice, device })
    }

    pub(crate) fn binary_impl<B: crate::op::BinaryOp>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let shape = lhs_l.shape();
        let dims = shape.dims();
        let elem_count = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(elem_count as u32);
        let dev = self.device();
        let dims_and_strides = dev.htod_copy([dims, lhs_l.stride(), rhs_l.stride()].concat())?;
        let slice = match (&self.slice, &rhs.slice) {
            (CudaStorageSlice::BF16(lhs), CudaStorageSlice::BF16(rhs)) => {
                let lhs = &lhs.slice(lhs_l.start_offset()..);
                let rhs = &rhs.slice(rhs_l.start_offset()..);
                let func = dev.get_or_load_func(B::KERNEL_BF16, kernels::BINARY)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.alloc::<bf16>(elem_count) }?;
                let params = (elem_count, dims.len(), &dims_and_strides, lhs, rhs, &out);
                // SAFETY: ffi
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::BF16(out)
            }
            (CudaStorageSlice::F16(lhs), CudaStorageSlice::F16(rhs)) => {
                let lhs = &lhs.slice(lhs_l.start_offset()..);
                let rhs = &rhs.slice(rhs_l.start_offset()..);
                let func = dev.get_or_load_func(B::KERNEL_F16, kernels::BINARY)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.alloc::<f16>(elem_count) }?;
                let params = (elem_count, dims.len(), &dims_and_strides, lhs, rhs, &out);
                // SAFETY: ffi
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::F16(out)
            }
            (CudaStorageSlice::F32(lhs), CudaStorageSlice::F32(rhs)) => {
                let lhs = &lhs.slice(lhs_l.start_offset()..);
                let rhs = &rhs.slice(rhs_l.start_offset()..);
                let func = dev.get_or_load_func(B::KERNEL_F32, kernels::BINARY)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.alloc::<f32>(elem_count) }?;
                let params = (elem_count, dims.len(), &dims_and_strides, lhs, rhs, &out);
                // SAFETY: ffi
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::F32(out)
            }
            (CudaStorageSlice::F64(lhs), CudaStorageSlice::F64(rhs)) => {
                let lhs = &lhs.slice(lhs_l.start_offset()..);
                let rhs = &rhs.slice(rhs_l.start_offset()..);
                // SAFETY: Set later by running the kernel.
                let func = dev.get_or_load_func(B::KERNEL_F64, kernels::BINARY)?;
                let out = unsafe { dev.alloc::<f64>(elem_count) }?;
                let params = (elem_count, dims.len(), &dims_and_strides, lhs, rhs, &out);
                // SAFETY: ffi
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::F64(out)
            }
            (CudaStorageSlice::U32(lhs), CudaStorageSlice::U32(rhs)) => {
                let lhs = &lhs.slice(lhs_l.start_offset()..);
                let rhs = &rhs.slice(rhs_l.start_offset()..);
                // SAFETY: Set later by running the kernel.
                let func = dev.get_or_load_func(B::KERNEL_U32, kernels::BINARY)?;
                let out = unsafe { dev.alloc::<u32>(elem_count) }?;
                let params = (elem_count, dims.len(), &dims_and_strides, lhs, rhs, &out);
                // SAFETY: ffi
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::U32(out)
            }
            // The dtypes should have been checked at this point so this is an internal error.
            _ => return Err(CudaError::InternalError("dtype mismatch in binary op")),
        };
        let device = dev.clone();
        Ok(Self { slice, device })
    }

    pub(crate) fn to_cpu_storage(&self) -> Result<CpuStorage> {
        match &self.slice {
            CudaStorageSlice::U32(slice) => {
                let dev = slice.device();
                let cpu_storage = dev.dtoh_sync_copy(slice)?;
                Ok(CpuStorage::U32(cpu_storage))
            }
            CudaStorageSlice::BF16(slice) => {
                let dev = slice.device();
                let cpu_storage = dev.dtoh_sync_copy(slice)?;
                Ok(CpuStorage::BF16(cpu_storage))
            }
            CudaStorageSlice::F16(slice) => {
                let dev = slice.device();
                let cpu_storage = dev.dtoh_sync_copy(slice)?;
                Ok(CpuStorage::F16(cpu_storage))
            }
            CudaStorageSlice::F32(slice) => {
                let dev = slice.device();
                let cpu_storage = dev.dtoh_sync_copy(slice)?;
                Ok(CpuStorage::F32(cpu_storage))
            }
            CudaStorageSlice::F64(slice) => {
                let dev = slice.device();
                let cpu_storage = dev.dtoh_sync_copy(slice)?;
                Ok(CpuStorage::F64(cpu_storage))
            }
        }
    }

    pub(crate) fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        layout_t: &Layout,
        f: &Self,
        layout_f: &Layout,
    ) -> Result<Self> {
        let ids = match &self.slice {
            CudaStorageSlice::U32(slice) => &slice.slice(layout.start_offset()..),
            _ => Err(CudaError::UnexpectedDType {
                msg: "where conditions should be u32",
                expected: DType::U32,
                got: self.dtype(),
            })?,
        };
        let shape = layout.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(el as u32);
        let dev = self.device();
        let ds =
            dev.htod_copy([dims, layout.stride(), layout_t.stride(), layout_f.stride()].concat())?;
        let slice = match (&t.slice, &f.slice) {
            (CudaStorageSlice::BF16(t), CudaStorageSlice::BF16(f)) => {
                let t = &t.slice(layout_t.start_offset()..);
                let f = &f.slice(layout_f.start_offset()..);
                let func = dev.get_or_load_func("where_bf16", kernels::TERNARY)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.alloc::<bf16>(el) }?;
                let params = (el, dims.len(), &ds, ids, t, f, &out);
                // SAFETY: ffi
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::BF16(out)
            }
            (CudaStorageSlice::F16(t), CudaStorageSlice::F16(f)) => {
                let t = &t.slice(layout_t.start_offset()..);
                let f = &f.slice(layout_f.start_offset()..);
                let func = dev.get_or_load_func("where_f16", kernels::TERNARY)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.alloc::<f16>(el) }?;
                let params = (el, dims.len(), &ds, ids, t, f, &out);
                // SAFETY: ffi
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::F16(out)
            }
            (CudaStorageSlice::F32(t), CudaStorageSlice::F32(f)) => {
                let t = &t.slice(layout_t.start_offset()..);
                let f = &f.slice(layout_f.start_offset()..);
                let func = dev.get_or_load_func("where_f32", kernels::TERNARY)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.alloc::<f32>(el) }?;
                let params = (el, dims.len(), &ds, ids, t, f, &out);
                // SAFETY: ffi
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::F32(out)
            }
            (CudaStorageSlice::F64(t), CudaStorageSlice::F64(f)) => {
                let t = &t.slice(layout_t.start_offset()..);
                let f = &f.slice(layout_f.start_offset()..);
                // SAFETY: Set later by running the kernel.
                let func = dev.get_or_load_func("where_f64", kernels::TERNARY)?;
                let out = unsafe { dev.alloc::<f64>(el) }?;
                let params = (el, dims.len(), &ds, ids, t, f, &out);
                // SAFETY: ffi
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::F64(out)
            }
            (CudaStorageSlice::U32(t), CudaStorageSlice::U32(f)) => {
                let t = &t.slice(layout_t.start_offset()..);
                let f = &f.slice(layout_f.start_offset()..);
                // SAFETY: Set later by running the kernel.
                let func = dev.get_or_load_func("where_u32", kernels::TERNARY)?;
                let out = unsafe { dev.alloc::<u32>(el) }?;
                let params = (el, dims.len(), &ds, ids, t, f, &out);
                // SAFETY: ffi
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::U32(out)
            }
            // The dtypes should have been checked at this point so this is an internal error.
            _ => Err(CudaError::InternalError("dtype mismatch in binary op"))?,
        };
        let device = dev.clone();
        Ok(Self { slice, device })
    }

    pub(crate) fn embedding(&self, layout: &Layout, rhs: &Self, rhs_l: &Layout) -> Result<Self> {
        let ids = match &self.slice {
            CudaStorageSlice::U32(slice) => &slice.slice(layout.start_offset()..),
            _ => Err(CudaError::UnexpectedDType {
                msg: "embedding ids should be u32",
                expected: DType::U32,
                got: self.dtype(),
            })?,
        };
        let shape = layout.shape();
        let (v_size, h_size) = rhs_l
            .shape()
            .r2()
            .map_err(|e| CudaError::WrappedError(Box::new(e)))?;
        let dims = shape.dims();
        let el = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(el as u32);
        let dev = self.device();
        let ds = dev.htod_copy([dims, layout.stride()].concat())?;
        let slice = match &rhs.slice {
            // The kernels below assume that rhs is contiguous.
            CudaStorageSlice::U32(arg) => {
                let arg = &arg.slice(rhs_l.start_offset()..);
                let func = dev.get_or_load_func("emb_u32", kernels::EMBEDDINGS)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.alloc::<u32>(el * h_size) }?;
                let params = (el, dims.len(), &ds, ids, arg, &out, h_size, v_size);
                // SAFETY: ffi.
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::U32(out)
            }
            CudaStorageSlice::BF16(arg) => {
                let arg = &arg.slice(rhs_l.start_offset()..);
                let func = dev.get_or_load_func("emb_bf16", kernels::EMBEDDINGS)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.alloc::<bf16>(el * h_size) }?;
                let params = (el, dims.len(), &ds, ids, arg, &out, h_size, v_size);
                // SAFETY: ffi.
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::BF16(out)
            }
            CudaStorageSlice::F16(arg) => {
                let arg = &arg.slice(rhs_l.start_offset()..);
                let func = dev.get_or_load_func("emb_f16", kernels::EMBEDDINGS)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.alloc::<f16>(el * h_size) }?;
                let params = (el, dims.len(), &ds, ids, arg, &out, h_size, v_size);
                // SAFETY: ffi.
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::F16(out)
            }
            CudaStorageSlice::F32(arg) => {
                let arg = &arg.slice(rhs_l.start_offset()..);
                let func = dev.get_or_load_func("emb_f32", kernels::EMBEDDINGS)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.alloc::<f32>(el * h_size) }?;
                let params = (el, dims.len(), &ds, ids, arg, &out, h_size, v_size);
                // SAFETY: ffi.
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::F32(out)
            }
            CudaStorageSlice::F64(arg) => {
                let arg = &arg.slice(rhs_l.start_offset()..);
                let func = dev.get_or_load_func("emb_f64", kernels::EMBEDDINGS)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.alloc::<f64>(el * h_size) }?;
                let params = (el, dims.len(), &ds, ids, arg, &out, h_size, v_size);
                // SAFETY: ffi.
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::F64(out)
            }
        };
        let device = dev.clone();
        Ok(Self { slice, device })
    }

    pub(crate) fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let elem_count = b * m * n;
        let dev = &self.device;
        let slice = match (&self.slice, &rhs.slice) {
            (CudaStorageSlice::BF16(_lhs), CudaStorageSlice::BF16(_rhs)) => {
                todo!("bf16")
            }
            (CudaStorageSlice::F16(lhs), CudaStorageSlice::F16(rhs)) => {
                let lhs = &lhs.slice(lhs_l.start_offset()..);
                let rhs = &rhs.slice(rhs_l.start_offset()..);
                let cfg = gemm_config(f16::ONE, f16::ZERO, (b, m, n, k), lhs_l, rhs_l)?;
                let mut out = unsafe { dev.alloc::<f16>(elem_count) }?;
                unsafe {
                    self.device
                        .blas
                        .gemm_strided_batched(cfg, rhs, lhs, &mut out)
                }?;
                CudaStorageSlice::F16(out)
            }
            (CudaStorageSlice::F32(lhs), CudaStorageSlice::F32(rhs)) => {
                let lhs = &lhs.slice(lhs_l.start_offset()..);
                let rhs = &rhs.slice(rhs_l.start_offset()..);
                let cfg = gemm_config(1., 0., (b, m, n, k), lhs_l, rhs_l)?;
                let mut out = unsafe { dev.alloc::<f32>(elem_count) }?;
                unsafe {
                    self.device
                        .blas
                        .gemm_strided_batched(cfg, rhs, lhs, &mut out)
                }?;
                CudaStorageSlice::F32(out)
            }
            (CudaStorageSlice::F64(lhs), CudaStorageSlice::F64(rhs)) => {
                let lhs = &lhs.slice(lhs_l.start_offset()..);
                let rhs = &rhs.slice(rhs_l.start_offset()..);
                let cfg = gemm_config(1., 0., (b, m, n, k), lhs_l, rhs_l)?;
                let mut out = unsafe { dev.alloc::<f64>(elem_count) }?;
                unsafe {
                    self.device
                        .blas
                        .gemm_strided_batched(cfg, rhs, lhs, &mut out)
                }?;
                CudaStorageSlice::F64(out)
            }
            _ => return Err(CudaError::InternalError("dtype mismatch in matmul op")),
        };
        let device = dev.clone();
        Ok(Self { slice, device })
    }

    pub(crate) fn copy_strided_src(
        &self,
        dst: &mut Self,
        dst_offset: usize,
        src_l: &Layout,
    ) -> Result<()> {
        let dims = src_shape.dims();
        let el_count = src_shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(el_count as u32);
        let dev = &self.device;
        let ds = dev.htod_copy([dims, src_stride].concat())?;
        match (&self.slice, &mut dst.slice) {
            (CudaStorageSlice::BF16(src), CudaStorageSlice::BF16(dst)) => {
                let (src, mut dst) = slice_src_and_dst(src, src_offset, dst, dst_offset);
                if src_shape.is_contiguous(src_stride) {
                    dev.dtod_copy(&src, &mut dst)?
                } else {
                    let func = dev.get_or_load_func("ucopy_bf16", kernels::UNARY)?;
                    // SAFETY: Set later by running the kernel.
                    let params = (el_count, dims.len(), &ds, &src, &mut dst);
                    // SAFETY: ffi.
                    unsafe { func.launch(cfg, params) }?
                }
            }
            (CudaStorageSlice::F16(src), CudaStorageSlice::F16(dst)) => {
                let (src, mut dst) = slice_src_and_dst(src, src_offset, dst, dst_offset);
                if src_shape.is_contiguous(src_stride) {
                    dev.dtod_copy(&src, &mut dst)?
                } else {
                    let func = dev.get_or_load_func("ucopy_f16", kernels::UNARY)?;
                    // SAFETY: Set later by running the kernel.
                    let params = (el_count, dims.len(), &ds, &src, &mut dst);
                    // SAFETY: ffi.
                    unsafe { func.launch(cfg, params) }?
                }
            }
            (CudaStorageSlice::F32(src), CudaStorageSlice::F32(dst)) => {
                let (src, mut dst) = slice_src_and_dst(src, src_offset, dst, dst_offset);
                if src_shape.is_contiguous(src_stride) {
                    dev.dtod_copy(&src, &mut dst)?
                } else {
                    let func = dev.get_or_load_func("ucopy_f32", kernels::UNARY)?;
                    // SAFETY: Set later by running the kernel.
                    let params = (el_count, dims.len(), &ds, &src, &mut dst);
                    // SAFETY: ffi.
                    unsafe { func.launch(cfg, params) }?
                }
            }
            (CudaStorageSlice::U32(src), CudaStorageSlice::U32(dst)) => {
                let (src, mut dst) = slice_src_and_dst(src, src_offset, dst, dst_offset);
                if src_shape.is_contiguous(src_stride) {
                    dev.dtod_copy(&src, &mut dst)?
                } else {
                    let func = dev.get_or_load_func("ucopy_u32", kernels::UNARY)?;
                    // SAFETY: Set later by running the kernel.
                    let params = (el_count, dims.len(), &ds, &src, &mut dst);
                    // SAFETY: ffi.
                    unsafe { func.launch(cfg, params) }?
                }
            }
            (CudaStorageSlice::F64(src), CudaStorageSlice::F64(dst)) => {
                let (src, mut dst) = slice_src_and_dst(src, src_offset, dst, dst_offset);
                if src_shape.is_contiguous(src_stride) {
                    dev.dtod_copy(&src, &mut dst)?
                } else {
                    let func = dev.get_or_load_func("ucopy_64", kernels::UNARY)?;
                    // SAFETY: Set later by running the kernel.
                    let params = (el_count, dims.len(), &ds, &src, &mut dst);
                    // SAFETY: ffi.
                    unsafe { func.launch(cfg, params) }?;
                }
            }
            _ => {
                return Err(CudaError::InternalError(
                    "dtype mismatch in copy_strided op",
                ))
            }
        }
        Ok(())
    }
}
