use crate::{CpuStorage, DType, Shape};
use candle_kernels as kernels;
use cudarc::cublas::{Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::driver::{CudaFunction, CudaSlice, LaunchAsync, LaunchConfig};
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
    MissingKernel { module_name: &'static str },

    #[error("internal error '{0}'")]
    InternalError(&'static str),

    #[error("{msg}, expected: {expected:?}, got: {got:?}")]
    UnexpectedDType {
        msg: &'static str,
        expected: DType,
        got: DType,
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

    fn get_or_load_func(
        &self,
        module_name: &'static str,
        ptx: &'static str,
    ) -> Result<CudaFunction> {
        if !self.has_func(module_name, module_name) {
            self.load_ptx(ptx.into(), module_name, &[module_name])?;
        }
        self.get_func(module_name, module_name)
            // Clippy recommends this `ok_or` rather than `ok_or_else` so hopefully the compiler is
            // able to only build the error value if needed.
            .ok_or(CudaError::MissingKernel { module_name })
    }
}

#[derive(Debug)]
enum CudaStorageSlice {
    U32(CudaSlice<u32>),
    F32(CudaSlice<f32>),
    F64(CudaSlice<f64>),
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
    _lhs_stride: &[usize],
    _rhs_stride: &[usize],
) -> StridedBatchedConfig<T> {
    // TODO: Handle lhs_stride and rhs_stride.
    // https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm
    use cudarc::cublas::sys::cublasOperation_t;
    // The setup below was copied from:
    // https://github.com/lebedov/scikit-cuda/blob/7e7300474286019c917a6c8a4bca59405c64fbce/tests/test_cublas.py#L531
    let gemm = GemmConfig {
        alpha,
        beta,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        lda: n as i32,
        ldb: k as i32,
        ldc: n as i32,
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
    };
    StridedBatchedConfig {
        batch_size: b as i32,
        gemm,
        stride_a: (m * k) as i64,
        stride_b: (n * k) as i64,
        stride_c: (m * n) as i64,
    }
}

impl CudaStorage {
    pub fn try_clone(&self) -> Result<Self> {
        let slice = match &self.slice {
            CudaStorageSlice::U32(slice) => CudaStorageSlice::U32(slice.try_clone()?),
            CudaStorageSlice::F32(slice) => CudaStorageSlice::F32(slice.try_clone()?),
            CudaStorageSlice::F64(slice) => CudaStorageSlice::F64(slice.try_clone()?),
        };
        let device = self.device.clone();
        Ok(Self { slice, device })
    }

    pub fn dtype(&self) -> DType {
        match self.slice {
            CudaStorageSlice::U32(_) => DType::U32,
            CudaStorageSlice::F32(_) => DType::F32,
            CudaStorageSlice::F64(_) => DType::F64,
        }
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    pub(crate) fn to_dtype(&self, _: &Shape, _: &[usize], _: DType) -> Result<Self> {
        Err(CudaError::InternalError("TODO: implement to_dtype"))
    }

    pub(crate) fn affine_impl(
        &self,
        shape: &Shape,
        stride: &[usize],
        mul: f64,
        add: f64,
    ) -> Result<Self> {
        let dims = shape.dims();
        let el_count = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(el_count as u32);
        let dev = self.device();
        let ds = dev.htod_copy([dims, stride].concat())?;
        let slice = match &self.slice {
            CudaStorageSlice::U32(arg) => {
                let func = dev.get_or_load_func("affine_u32", kernels::AFFINE)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.alloc::<u32>(el_count) }?;
                let params = (el_count, dims.len(), &ds, arg, &out, mul as u32, add as u32);
                // SAFETY: ffi.
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::U32(out)
            }
            CudaStorageSlice::F32(arg) => {
                let func = dev.get_or_load_func("affine_f32", kernels::AFFINE)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.alloc::<f32>(el_count) }?;
                let params = (el_count, dims.len(), &ds, arg, &out, mul as f32, add as f32);
                // SAFETY: ffi.
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::F32(out)
            }
            CudaStorageSlice::F64(arg) => {
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

    pub(crate) fn sum(&self, shape: &Shape, stride: &[usize], sum_dims: &[usize]) -> Result<Self> {
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
        let ds = dev.htod_copy([src_dims, stride, &sum_dims_l, &sum_dims_s].concat())?;
        let slice = match &self.slice {
            CudaStorageSlice::U32(arg) => {
                let func = dev.get_or_load_func("sum_u32", kernels::REDUCE)?;
                let out = dev.alloc_zeros::<u32>(dst_el)?;
                let params = (el, src_dims.len(), sum_dims.len(), &ds, arg, &out);
                // SAFETY: ffi.
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::U32(out)
            }
            CudaStorageSlice::F32(arg) => {
                let func = dev.get_or_load_func("sum_f32", kernels::REDUCE)?;
                let out = dev.alloc_zeros::<f32>(dst_el)?;
                let params = (el, src_dims.len(), sum_dims.len(), &ds, arg, &out);
                // SAFETY: ffi.
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::F32(out)
            }
            CudaStorageSlice::F64(arg) => {
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

    pub(crate) fn unary_impl<U: crate::op::UnaryOp>(
        &self,
        shape: &Shape,
        stride: &[usize],
    ) -> Result<Self> {
        let dims = shape.dims();
        let el_count = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(el_count as u32);
        let dev = &self.device;
        let ds = dev.htod_copy([dims, stride].concat())?;
        let slice = match &self.slice {
            CudaStorageSlice::U32(_arg) => {
                todo!("No unary kernels for u32");
            }
            CudaStorageSlice::F32(arg) => {
                let func = dev.get_or_load_func(U::KERNEL_F32, kernels::UNARY)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.alloc::<f32>(el_count) }?;
                let params = (el_count, dims.len(), &ds, arg, &out);
                // SAFETY: ffi.
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::F32(out)
            }
            CudaStorageSlice::F64(arg) => {
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
        shape: &Shape,
        lhs_stride: &[usize],
        rhs_stride: &[usize],
    ) -> Result<Self> {
        let dims = shape.dims();
        let elem_count = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(elem_count as u32);
        let dev = self.device();
        let dims_and_strides = dev.htod_copy([dims, lhs_stride, rhs_stride].concat())?;
        let slice = match (&self.slice, &rhs.slice) {
            (CudaStorageSlice::F32(lhs), CudaStorageSlice::F32(rhs)) => {
                let func = dev.get_or_load_func(B::KERNEL_F32, kernels::BINARY)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.alloc::<f32>(elem_count) }?;
                let params = (elem_count, dims.len(), &dims_and_strides, lhs, rhs, &out);
                // SAFETY: ffi
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::F32(out)
            }
            (CudaStorageSlice::F64(lhs), CudaStorageSlice::F64(rhs)) => {
                // SAFETY: Set later by running the kernel.
                let func = dev.get_or_load_func(B::KERNEL_F64, kernels::BINARY)?;
                let out = unsafe { dev.alloc::<f64>(elem_count) }?;
                let params = (elem_count, dims.len(), &dims_and_strides, lhs, rhs, &out);
                // SAFETY: ffi
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::F64(out)
            }
            (CudaStorageSlice::U32(lhs), CudaStorageSlice::U32(rhs)) => {
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
        shape: &Shape,
        stride: &[usize],
        t: &Self,
        stride_t: &[usize],
        f: &Self,
        stride_f: &[usize],
    ) -> Result<Self> {
        let ids = match &self.slice {
            CudaStorageSlice::U32(slice) => slice,
            _ => Err(CudaError::UnexpectedDType {
                msg: "embedding ids should be u32",
                expected: DType::U32,
                got: self.dtype(),
            })?,
        };
        let dims = shape.dims();
        let el = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(el as u32);
        let dev = self.device();
        let ds = dev.htod_copy([dims, stride, stride_t, stride_f].concat())?;
        let slice = match (&t.slice, &f.slice) {
            (CudaStorageSlice::F32(t), CudaStorageSlice::F32(f)) => {
                let func = dev.get_or_load_func("where_f32", kernels::BINARY)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.alloc::<f32>(el) }?;
                let params = (el, dims.len(), &ds, ids, t, f, &out);
                // SAFETY: ffi
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::F32(out)
            }
            (CudaStorageSlice::F64(t), CudaStorageSlice::F64(f)) => {
                // SAFETY: Set later by running the kernel.
                let func = dev.get_or_load_func("where_f64", kernels::BINARY)?;
                let out = unsafe { dev.alloc::<f64>(el) }?;
                let params = (el, dims.len(), &ds, ids, t, f, &out);
                // SAFETY: ffi
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::F64(out)
            }
            (CudaStorageSlice::U32(t), CudaStorageSlice::U32(f)) => {
                // SAFETY: Set later by running the kernel.
                let func = dev.get_or_load_func("where_u32", kernels::BINARY)?;
                let out = unsafe { dev.alloc::<u32>(el) }?;
                let params = (el, dims.len(), &ds, ids, t, f, &out);
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

    pub(crate) fn embedding_impl(
        &self,
        shape: &Shape,
        stride: &[usize],
        rhs: &Self,
        h_size: usize, // hidden size
        v_size: usize, // vocab size
    ) -> Result<Self> {
        let ids = match &self.slice {
            CudaStorageSlice::U32(slice) => slice,
            _ => Err(CudaError::UnexpectedDType {
                msg: "embedding ids should be u32",
                expected: DType::U32,
                got: self.dtype(),
            })?,
        };
        let dims = shape.dims();
        let el = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(el as u32);
        let dev = self.device();
        let ds = dev.htod_copy([dims, stride].concat())?;
        let slice = match &rhs.slice {
            // The kernels below assume that rhs is contiguous.
            CudaStorageSlice::U32(arg) => {
                let func = dev.get_or_load_func("emb_f16", kernels::EMBEDDINGS)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.alloc::<u32>(el * h_size) }?;
                let params = (el, dims.len(), &ds, ids, arg, &out, h_size, v_size);
                // SAFETY: ffi.
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::U32(out)
            }
            CudaStorageSlice::F32(arg) => {
                let func = dev.get_or_load_func("emb_f32", kernels::EMBEDDINGS)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.alloc::<f32>(el * h_size) }?;
                let params = (el, dims.len(), &ds, ids, arg, &out, h_size, v_size);
                // SAFETY: ffi.
                unsafe { func.launch(cfg, params) }?;
                CudaStorageSlice::F32(out)
            }
            CudaStorageSlice::F64(arg) => {
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

    pub(crate) fn matmul_impl(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_stride: &[usize],
        rhs_stride: &[usize],
    ) -> Result<Self> {
        let elem_count = b * m * n;
        let dev = &self.device;
        let slice = match (&self.slice, &rhs.slice) {
            (CudaStorageSlice::F32(lhs), CudaStorageSlice::F32(rhs)) => {
                let cfg = gemm_config(1., 0., (b, m, n, k), lhs_stride, rhs_stride);
                let mut out = unsafe { dev.alloc::<f32>(elem_count) }?;
                unsafe {
                    self.device
                        .blas
                        .gemm_strided_batched(cfg, rhs, lhs, &mut out)
                }?;
                CudaStorageSlice::F32(out)
            }
            (CudaStorageSlice::F64(lhs), CudaStorageSlice::F64(rhs)) => {
                let cfg = gemm_config(1., 0., (b, m, n, k), lhs_stride, rhs_stride);
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
        src_shape: &Shape,
        src_stride: &[usize],
        src_offset: usize,
    ) -> Result<()> {
        if src_shape.rank() != src_stride.len() {
            panic!("incoherent shape and strides {src_shape:?} {src_stride:?}")
        }
        let dims = src_shape.dims();
        let el_count = src_shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(el_count as u32);
        let dev = &self.device;
        let ds = dev.htod_copy([dims, src_stride].concat())?;
        match (&self.slice, &mut dst.slice) {
            (CudaStorageSlice::F32(src), CudaStorageSlice::F32(dst)) => {
                let src = src.slice(src_offset..);
                let mut dst = dst.slice_mut(dst_offset..);
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
            (CudaStorageSlice::F64(src), CudaStorageSlice::F64(dst)) => {
                let src = src.slice(src_offset..);
                let mut dst = dst.slice_mut(dst_offset..);
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
