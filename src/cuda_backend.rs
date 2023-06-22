use crate::{CpuStorage, DType, Shape};
use candle_kernels as kernels;
use cudarc::driver::{CudaFunction, CudaSlice, LaunchAsync, LaunchConfig};

/// cudarc related errors
#[derive(thiserror::Error, Debug)]
pub enum CudaError {
    #[error(transparent)]
    Cuda(#[from] cudarc::driver::DriverError),

    #[error(transparent)]
    Compiler(#[from] cudarc::nvrtc::CompileError),

    #[error("{op} only supports contiguous tensors")]
    RequiresContiguous { op: &'static str },

    #[error("missing kernel '{module_name}'")]
    MissingKernel { module_name: &'static str },

    #[error("internal error '{0}'")]
    InternalError(&'static str),
}

type Result<T> = std::result::Result<T, CudaError>;

#[derive(Debug, Clone)]
pub struct CudaDevice(std::sync::Arc<cudarc::driver::CudaDevice>);

impl CudaDevice {
    pub(crate) fn new(ordinal: usize) -> Result<Self> {
        let device = cudarc::driver::CudaDevice::new(ordinal)?;
        Ok(Self(device))
    }

    pub(crate) fn ordinal(&self) -> usize {
        self.0.ordinal()
    }

    pub(crate) fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        match dtype {
            DType::F32 => {
                let data = self.0.alloc_zeros::<f32>(elem_count)?;
                Ok(CudaStorage::F32(data))
            }
            DType::F64 => {
                let data = self.0.alloc_zeros::<f64>(elem_count)?;
                Ok(CudaStorage::F64(data))
            }
        }
    }

    pub(crate) fn const_impl(&self, v: f64, shape: &Shape, dtype: DType) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(elem_count as u32);
        let dev = &self.0;
        match dtype {
            DType::F32 => {
                // SAFETY: Set later by running the fill kernel.
                let data = unsafe { dev.alloc::<f32>(elem_count) }?;
                let func = self.get_or_load_func("fill_f32", kernels::FILL)?;
                let params = (&data, v as f32, elem_count);
                unsafe { func.launch(cfg, params) }?;
                Ok(CudaStorage::F32(data))
            }
            DType::F64 => {
                // SAFETY: Set later by running the fill kernel.
                let data = unsafe { dev.alloc::<f64>(elem_count) }?;
                let func = self.get_or_load_func("fill_f64", kernels::FILL)?;
                let params = (&data, v, elem_count);
                unsafe { func.launch(cfg, params) }?;
                Ok(CudaStorage::F64(data))
            }
        }
    }

    pub(crate) fn ones_impl(&self, shape: &Shape, dtype: DType) -> Result<CudaStorage> {
        self.const_impl(1., shape, dtype)
    }

    pub(crate) fn cuda_from_cpu_storage(&self, storage: &CpuStorage) -> Result<CudaStorage> {
        match storage {
            CpuStorage::F32(storage) => {
                let data = self.0.htod_sync_copy(storage)?;
                Ok(CudaStorage::F32(data))
            }
            CpuStorage::F64(storage) => {
                let data = self.0.htod_sync_copy(storage)?;
                Ok(CudaStorage::F64(data))
            }
        }
    }

    fn get_or_load_func(
        &self,
        module_name: &'static str,
        ptx: &'static str,
    ) -> Result<CudaFunction> {
        let dev = &self.0;
        if !dev.has_func(module_name, module_name) {
            dev.load_ptx(ptx.into(), module_name, &[module_name])?;
        }
        dev.get_func(module_name, module_name)
            // Clippy recommends this `ok_or` rather than `ok_or_else` so hopefully the compiler is
            // able to only build the error value if needed.
            .ok_or(CudaError::MissingKernel { module_name })
    }
}

#[derive(Debug, Clone)]
pub enum CudaStorage {
    F32(CudaSlice<f32>),
    F64(CudaSlice<f64>),
}

impl CudaStorage {
    pub fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
        }
    }

    pub fn device(&self) -> CudaDevice {
        match self {
            Self::F32(slice) => CudaDevice(slice.device()),
            Self::F64(slice) => CudaDevice(slice.device()),
        }
    }

    pub(crate) fn affine_impl(
        &self,
        shape: &Shape,
        stride: &[usize],
        mul: f64,
        add: f64,
    ) -> Result<Self> {
        if !shape.is_contiguous(stride) {
            return Err(CudaError::RequiresContiguous { op: "affine" });
        }

        let elem_count = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(elem_count as u32);
        let dev = self.device();
        match self {
            Self::F32(arg) => {
                let func = dev.get_or_load_func("affine_f32", kernels::AFFINE)?;
                // SAFETY: if this function returns Ok(..), the kernel has been applied
                // and has set the initially unset memory.
                let out = unsafe { dev.0.alloc::<f32>(elem_count) }?;
                let params = (elem_count, arg, &out, mul as f32, add as f32);
                // SAFETY: well, well, well...
                unsafe { func.launch(cfg, params) }?;
                Ok(Self::F32(out))
            }
            Self::F64(arg) => {
                let func = dev.get_or_load_func("affine_f64", kernels::AFFINE)?;
                // SAFETY: if this function returns Ok(..), the kernel has been applied
                // and has set the initially unset memory.
                let out = unsafe { dev.0.alloc::<f64>(elem_count) }?;
                let params = (elem_count, arg, &out, mul, add);
                // SAFETY: well, well, well...
                unsafe { func.launch(cfg, params) }?;
                Ok(Self::F64(out))
            }
        }
    }

    pub(crate) fn binary_impl<B: crate::storage::BinaryOp>(
        &self,
        rhs: &Self,
        shape: &Shape,
        lhs_stride: &[usize],
        rhs_stride: &[usize],
    ) -> Result<Self> {
        let elem_count = shape.elem_count();
        let dims = shape.dims();
        let cfg = LaunchConfig::for_num_elems(elem_count as u32);
        let dev = self.device();
        let dims_and_strides = [dims, lhs_stride, rhs_stride].concat();
        match (self, rhs) {
            (Self::F32(lhs), Self::F32(rhs)) => {
                let func = dev.get_or_load_func(B::KERNEL_F32, kernels::BINARY)?;
                // SAFETY: Set later by running the kernel.
                let out = unsafe { dev.0.alloc::<f32>(elem_count) }?;
                let dims_and_strides = dev.0.htod_copy(dims_and_strides)?;
                let params = (elem_count, dims.len(), &dims_and_strides, lhs, rhs, &out);
                // SAFETY: ffi
                unsafe { func.launch(cfg, params) }?;
                Ok(Self::F32(out))
            }
            (Self::F64(lhs), Self::F64(rhs)) => {
                // SAFETY: Set later by running the kernel.
                let func = dev.get_or_load_func(B::KERNEL_F64, kernels::BINARY)?;
                let out = unsafe { dev.0.alloc::<f64>(elem_count) }?;
                let dims_and_strides = dev.0.htod_copy(dims_and_strides)?;
                let params = (elem_count, dims.len(), &dims_and_strides, lhs, rhs, &out);
                // SAFETY: ffi
                unsafe { func.launch(cfg, params) }?;
                Ok(Self::F64(out))
            }
            // The dtypes should have been checked at this point so this is an internal error.
            _ => Err(CudaError::InternalError("dtype mismatch in binary op")),
        }
    }

    pub(crate) fn to_cpu_storage(&self) -> Result<CpuStorage> {
        match self {
            Self::F32(slice) => {
                let dev = slice.device();
                let cpu_storage = dev.dtoh_sync_copy(slice)?;
                Ok(CpuStorage::F32(cpu_storage))
            }
            Self::F64(slice) => {
                let dev = slice.device();
                let cpu_storage = dev.dtoh_sync_copy(slice)?;
                Ok(CpuStorage::F64(cpu_storage))
            }
        }
    }
}
