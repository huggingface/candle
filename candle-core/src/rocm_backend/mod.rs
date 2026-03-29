use crate::backend::BackendStorage;
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result};
use half::{bf16, f16};
use rocm_rs::hip::{bindings, DeviceMemory};

mod device;
mod error;
pub use device::{DeviceId, RocmDevice};
pub use error::{RocmError, WrapErr};

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
    F8E4M3(DeviceMemory<u8>),
}

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
}

pub struct RocmStorage {
    pub slice: RocmStorageSlice,
    pub device: RocmDevice,
}

impl std::fmt::Debug for RocmStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RocmStorage {{ slice: {:?}, device: {:?} }}",
            self.slice, self.device
        )
    }
}

impl BackendStorage for RocmStorage {
    type Device = RocmDevice;

    fn try_clone(&self, layout: &Layout) -> Result<Self> {
        let device = self.device.clone();
        let elem_count = layout.shape().elem_count();
        let slice = match &self.slice {
            RocmStorageSlice::U8(s) => {
                let mut dst = device.alloc::<u8>(elem_count)?;
                dst.copy_from_device(s)?;
                RocmStorageSlice::U8(dst)
            }
            RocmStorageSlice::U32(s) => {
                let mut dst = device.alloc::<u32>(elem_count)?;
                dst.copy_from_device(s)?;
                RocmStorageSlice::U32(dst)
            }
            RocmStorageSlice::I16(s) => {
                let mut dst = device.alloc::<i16>(elem_count)?;
                dst.copy_from_device(s)?;
                RocmStorageSlice::I16(dst)
            }
            RocmStorageSlice::I32(s) => {
                let mut dst = device.alloc::<i32>(elem_count)?;
                dst.copy_from_device(s)?;
                RocmStorageSlice::I32(dst)
            }
            RocmStorageSlice::I64(s) => {
                let mut dst = device.alloc::<i64>(elem_count)?;
                dst.copy_from_device(s)?;
                RocmStorageSlice::I64(dst)
            }
            RocmStorageSlice::BF16(s) => {
                let mut dst = device.alloc::<bf16>(elem_count)?;
                dst.copy_from_device(s)?;
                RocmStorageSlice::BF16(dst)
            }
            RocmStorageSlice::F16(s) => {
                let mut dst = device.alloc::<f16>(elem_count)?;
                dst.copy_from_device(s)?;
                RocmStorageSlice::F16(dst)
            }
            RocmStorageSlice::F32(s) => {
                let mut dst = device.alloc::<f32>(elem_count)?;
                dst.copy_from_device(s)?;
                RocmStorageSlice::F32(dst)
            }
            RocmStorageSlice::F64(s) => {
                let mut dst = device.alloc::<f64>(elem_count)?;
                dst.copy_from_device(s)?;
                RocmStorageSlice::F64(dst)
            }
            RocmStorageSlice::F8E4M3(s) => {
                let mut dst = device.alloc::<u8>(elem_count)?;
                dst.copy_from_device(s)?;
                RocmStorageSlice::F8E4M3(dst)
            }
        };
        Ok(Self { slice, device })
    }

    fn dtype(&self) -> DType {
        self.slice.dtype()
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        match &self.slice {
            RocmStorageSlice::U8(s) => Ok(CpuStorage::U8(self.device.clone_dtoh(s)?.into())),
            RocmStorageSlice::U32(s) => Ok(CpuStorage::U32(self.device.clone_dtoh(s)?.into())),
            RocmStorageSlice::I16(s) => Ok(CpuStorage::I16(self.device.clone_dtoh(s)?.into())),
            RocmStorageSlice::I32(s) => Ok(CpuStorage::I32(self.device.clone_dtoh(s)?.into())),
            RocmStorageSlice::I64(s) => Ok(CpuStorage::I64(self.device.clone_dtoh(s)?.into())),
            RocmStorageSlice::BF16(s) => Ok(CpuStorage::BF16(self.device.clone_dtoh(s)?.into())),
            RocmStorageSlice::F16(s) => Ok(CpuStorage::F16(self.device.clone_dtoh(s)?.into())),
            RocmStorageSlice::F32(s) => Ok(CpuStorage::F32(self.device.clone_dtoh(s)?.into())),
            RocmStorageSlice::F64(s) => Ok(CpuStorage::F64(self.device.clone_dtoh(s)?.into())),
            RocmStorageSlice::F8E4M3(s) => {
                let bytes = self.device.clone_dtoh(s)?;
                let v: Vec<float8::F8E4M3> =
                    bytes.into_iter().map(float8::F8E4M3::from_bits).collect();
                Ok(CpuStorage::F8E4M3(v.into()))
            }
        }
    }

    fn affine(&self, _l: &Layout, _a: f64, _b: f64) -> Result<Self> {
        Err(crate::Error::Msg(
            "affine not yet implemented for ROCm".to_string(),
        ))
    }

    fn powf(&self, _l: &Layout, _e: f64) -> Result<Self> {
        Err(crate::Error::Msg(
            "powf not yet implemented for ROCm".to_string(),
        ))
    }

    fn elu(&self, _l: &Layout, _alpha: f64) -> Result<Self> {
        Err(crate::Error::Msg(
            "elu not yet implemented for ROCm".to_string(),
        ))
    }

    fn reduce_op(&self, _op: ReduceOp, _l: &Layout, _dims: &[usize]) -> Result<Self> {
        Err(crate::Error::Msg(
            "reduce_op not yet implemented for ROCm".to_string(),
        ))
    }

    fn cmp(&self, _op: CmpOp, _rhs: &Self, _l1: &Layout, _l2: &Layout) -> Result<Self> {
        Err(crate::Error::Msg(
            "cmp not yet implemented for ROCm".to_string(),
        ))
    }

    fn to_dtype(&self, _l: &Layout, _dtype: DType) -> Result<Self> {
        Err(crate::Error::Msg(
            "to_dtype not yet implemented for ROCm".to_string(),
        ))
    }

    fn unary_impl<B: UnaryOpT>(&self, _l: &Layout) -> Result<Self> {
        Err(crate::Error::Msg(
            "unary_impl not yet implemented for ROCm".to_string(),
        ))
    }

    fn binary_impl<B: BinaryOpT>(&self, _rhs: &Self, _l1: &Layout, _l2: &Layout) -> Result<Self> {
        Err(crate::Error::Msg(
            "binary_impl not yet implemented for ROCm".to_string(),
        ))
    }

    fn where_cond(
        &self,
        _l: &Layout,
        _a: &Self,
        _la: &Layout,
        _b: &Self,
        _lb: &Layout,
    ) -> Result<Self> {
        Err(crate::Error::Msg(
            "where_cond not yet implemented for ROCm".to_string(),
        ))
    }

    fn conv1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kl: &Layout,
        _params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        Err(crate::Error::Msg(
            "conv1d not yet implemented for ROCm".to_string(),
        ))
    }

    fn conv_transpose1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kl: &Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        Err(crate::Error::Msg(
            "conv_transpose1d not yet implemented for ROCm".to_string(),
        ))
    }

    fn conv2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kl: &Layout,
        _params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        Err(crate::Error::Msg(
            "conv2d not yet implemented for ROCm".to_string(),
        ))
    }

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kl: &Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        Err(crate::Error::Msg(
            "conv_transpose2d not yet implemented for ROCm".to_string(),
        ))
    }

    fn avg_pool2d(&self, _l: &Layout, _k: (usize, usize), _s: (usize, usize)) -> Result<Self> {
        Err(crate::Error::Msg(
            "avg_pool2d not yet implemented for ROCm".to_string(),
        ))
    }

    fn max_pool2d(&self, _l: &Layout, _k: (usize, usize), _s: (usize, usize)) -> Result<Self> {
        Err(crate::Error::Msg(
            "max_pool2d not yet implemented for ROCm".to_string(),
        ))
    }

    fn upsample_nearest1d(&self, _l: &Layout, _sz: usize) -> Result<Self> {
        Err(crate::Error::Msg(
            "upsample_nearest1d not yet implemented for ROCm".to_string(),
        ))
    }

    fn upsample_nearest2d(&self, _l: &Layout, _w: usize, _h: usize) -> Result<Self> {
        Err(crate::Error::Msg(
            "upsample_nearest2d not yet implemented for ROCm".to_string(),
        ))
    }

    fn upsample_bilinear2d(
        &self,
        _l: &Layout,
        _w: usize,
        _h: usize,
        _align: bool,
        _fh: Option<f64>,
        _fv: Option<f64>,
    ) -> Result<Self> {
        Err(crate::Error::Msg(
            "upsample_bilinear2d not yet implemented for ROCm".to_string(),
        ))
    }

    fn gather(&self, _l: &Layout, _idx: &Self, _il: &Layout, _dim: usize) -> Result<Self> {
        Err(crate::Error::Msg(
            "gather not yet implemented for ROCm".to_string(),
        ))
    }

    fn scatter_set(
        &mut self,
        _l: &Layout,
        _val: &Self,
        _vl: &Layout,
        _idx: &Self,
        _il: &Layout,
        _dim: usize,
    ) -> Result<()> {
        Err(crate::Error::Msg(
            "scatter_set not yet implemented for ROCm".to_string(),
        ))
    }

    fn scatter_add_set(
        &mut self,
        _l: &Layout,
        _val: &Self,
        _vl: &Layout,
        _idx: &Self,
        _il: &Layout,
        _dim: usize,
    ) -> Result<()> {
        Err(crate::Error::Msg(
            "scatter_add_set not yet implemented for ROCm".to_string(),
        ))
    }

    fn index_select(&self, _idx: &Self, _il: &Layout, _sl: &Layout, _dim: usize) -> Result<Self> {
        Err(crate::Error::Msg(
            "index_select not yet implemented for ROCm".to_string(),
        ))
    }

    fn index_add(
        &self,
        _l: &Layout,
        _idx: &Self,
        _il: &Layout,
        _val: &Self,
        _vl: &Layout,
        _dim: usize,
    ) -> Result<Self> {
        Err(crate::Error::Msg(
            "index_add not yet implemented for ROCm".to_string(),
        ))
    }

    fn matmul(
        &self,
        _rhs: &Self,
        _bmnk: (usize, usize, usize, usize),
        _l1: &Layout,
        _l2: &Layout,
    ) -> Result<Self> {
        Err(crate::Error::Msg(
            "matmul not yet implemented for ROCm".to_string(),
        ))
    }

    fn copy_strided_src(&self, _dst: &mut Self, _offset: usize, _l: &Layout) -> Result<()> {
        Err(crate::Error::Msg(
            "copy_strided_src not yet implemented for ROCm".to_string(),
        ))
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_s1: usize,
        dst_s1: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Result<()> {
        if d1 == 0 || d2 == 0 {
            return Ok(());
        }
        let (src_ptr, dst_ptr, el_size) = match (&self.slice, &mut dst.slice) {
            (RocmStorageSlice::U8(s), RocmStorageSlice::U8(d)) => (s.as_ptr(), d.as_ptr(), 1usize),
            (RocmStorageSlice::U32(s), RocmStorageSlice::U32(d)) => (s.as_ptr(), d.as_ptr(), 4),
            (RocmStorageSlice::I16(s), RocmStorageSlice::I16(d)) => (s.as_ptr(), d.as_ptr(), 2),
            (RocmStorageSlice::I32(s), RocmStorageSlice::I32(d)) => (s.as_ptr(), d.as_ptr(), 4),
            (RocmStorageSlice::I64(s), RocmStorageSlice::I64(d)) => (s.as_ptr(), d.as_ptr(), 8),
            (RocmStorageSlice::BF16(s), RocmStorageSlice::BF16(d)) => (s.as_ptr(), d.as_ptr(), 2),
            (RocmStorageSlice::F16(s), RocmStorageSlice::F16(d)) => (s.as_ptr(), d.as_ptr(), 2),
            (RocmStorageSlice::F32(s), RocmStorageSlice::F32(d)) => (s.as_ptr(), d.as_ptr(), 4),
            (RocmStorageSlice::F64(s), RocmStorageSlice::F64(d)) => (s.as_ptr(), d.as_ptr(), 8),
            (RocmStorageSlice::F8E4M3(s), RocmStorageSlice::F8E4M3(d)) => {
                (s.as_ptr(), d.as_ptr(), 1)
            }
            _ => crate::bail!("dtype mismatch in copy2d"),
        };
        let src_ptr = unsafe { src_ptr.add(src_o * el_size) };
        let dst_ptr = unsafe { dst_ptr.add(dst_o * el_size) };
        let width = d2 * el_size;
        let spitch = src_s1 * el_size;
        let dpitch = dst_s1 * el_size;
        let result = unsafe {
            bindings::hipMemcpy2D(
                dst_ptr,
                dpitch,
                src_ptr,
                spitch,
                width,
                d1,
                bindings::hipMemcpyKind_hipMemcpyDeviceToDevice,
            )
        };
        if result != bindings::hipError_t_hipSuccess {
            crate::bail!("hipMemcpy2D failed with error {}", result);
        }
        Ok(())
    }

    fn const_set(&mut self, _val: crate::scalar::Scalar, _l: &Layout) -> Result<()> {
        Err(crate::Error::Msg(
            "const_set not yet implemented for ROCm".to_string(),
        ))
    }
}
