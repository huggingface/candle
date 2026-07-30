//! Implementation of Backend traits for ROCm device
//!
use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result};
pub use candle_rocm_kernels as kernels;
use half::{bf16, f16};
pub use rocm_rs;
use rocm_rs::hip::bindings;

mod alloc;
mod device;
mod device_backend;
mod error;
mod gemm;
mod launch;
#[cfg(feature = "miopen")]
mod miopen;
mod ops_conv;
mod ops_conv_direct;
mod ops_elementwise;
mod ops_indexing;
mod ops_pool;
mod ops_reduce;
mod ops_scalar;
mod params;
mod rng;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_alloc;
#[cfg(test)]
mod tests_conv;
#[cfg(test)]
mod tests_copy;
#[cfg(test)]
mod tests_indexing;
#[cfg(test)]
mod tests_rng;
#[cfg(test)]
mod tests_sort;
mod wrappers;
pub use alloc::SendSyncDeviceMemory;
pub use device::{DeviceId, RocmDevice};
pub use error::{RocmError, WrapErr};
pub use launch::{
    kernel_name, launch_config, launch_config_for, launch_config_layout, try_kernel_name,
};
pub use params::{params_from_vec, ParamBuffer};
pub mod utils;
pub use utils::{Map1, Map1Any, Map2, Map2Any, Map2InPlace, Map3, S};

use launch::launch_kernel;
use ops_elementwise::{CloneBuffer, Cmp, WhereCond};
use ops_indexing::{Gather, IndexAdd, IndexSelect, Scatter, ScatterKind};
use ops_reduce::FastReduce;
pub(crate) use ops_scalar::Affine;
use ops_scalar::{Elu, Powf};
use params::{dims_and_strides, dims_and_strides_pair, ParamCache};

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
    F8E4M3(SendSyncDeviceMemory<u8>),
}

/// `RocmStorageSlice::F8E4M3` stores its payload as `u8`, so every byte-view
/// shortcut in this backend (and in `device.rs`) is only correct while F8E4M3 is
/// exactly one byte wide.
const _: () = assert!(std::mem::size_of::<float8::F8E4M3>() == 1);

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

    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
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

    fn elem_size(&self) -> usize {
        match self {
            RocmStorageSlice::U8(_) | RocmStorageSlice::F8E4M3(_) => 1,
            RocmStorageSlice::I16(_) | RocmStorageSlice::BF16(_) | RocmStorageSlice::F16(_) => 2,
            RocmStorageSlice::U32(_) | RocmStorageSlice::I32(_) | RocmStorageSlice::F32(_) => 4,
            RocmStorageSlice::I64(_) | RocmStorageSlice::F64(_) => 8,
        }
    }

    unsafe fn offset_ptr(&self, offset: usize) -> *mut std::ffi::c_void {
        self.as_ptr().add(offset * self.elem_size())
    }

    /// Number of elements the underlying allocation holds.
    fn count(&self) -> usize {
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

pub struct RocmStorage {
    pub slice: RocmStorageSlice,
    pub device: RocmDevice,
}

macro_rules! cast_launch {
    ($dev:expr, $grid:expr, $block:expr, $el:expr, $dims_len:expr, $ds_ptr:expr, $src_ptr:expr, $src_dtype:expr, $rust_type:ty, $variant:ident) => {{
        let out = $dev.alloc::<$rust_type>($el)?;
        let out_ptr = out.as_ptr() as *mut std::ffi::c_void;
        let func_name = format!("cast_{}_{}", $src_dtype.as_str(), stringify!($rust_type));
        unsafe {
            launch_kernel(
                &$dev,
                &kernels::CAST,
                &func_name,
                $grid,
                $block,
                &mut [
                    &$el as *const usize as *mut std::ffi::c_void,
                    &$dims_len as *const usize as *mut std::ffi::c_void,
                    (&$ds_ptr) as *const *const usize as *mut std::ffi::c_void,
                    (&$src_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                ],
            )?;
        }
        RocmStorageSlice::$variant(out)
    }};
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
        let slice = match &self.slice {
            // `Map1` refuses F8E4M3 (it shares the u8 storage and would resolve
            // to the u8 kernels), but a raw buffer copy is dtype agnostic.
            RocmStorageSlice::F8E4M3(s) => {
                RocmStorageSlice::F8E4M3(CloneBuffer.f(s, &device, layout)?)
            }
            slice => CloneBuffer.map(slice, &device, layout)?,
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
            RocmStorageSlice::U8(s) => Ok(CpuStorage::U8(self.device.clone_dtoh(s)?)),
            RocmStorageSlice::U32(s) => Ok(CpuStorage::U32(self.device.clone_dtoh(s)?)),
            RocmStorageSlice::I16(s) => Ok(CpuStorage::I16(self.device.clone_dtoh(s)?)),
            RocmStorageSlice::I32(s) => Ok(CpuStorage::I32(self.device.clone_dtoh(s)?)),
            RocmStorageSlice::I64(s) => Ok(CpuStorage::I64(self.device.clone_dtoh(s)?)),
            RocmStorageSlice::BF16(s) => Ok(CpuStorage::BF16(self.device.clone_dtoh(s)?)),
            RocmStorageSlice::F16(s) => Ok(CpuStorage::F16(self.device.clone_dtoh(s)?)),
            RocmStorageSlice::F32(s) => Ok(CpuStorage::F32(self.device.clone_dtoh(s)?)),
            RocmStorageSlice::F64(s) => Ok(CpuStorage::F64(self.device.clone_dtoh(s)?)),
            RocmStorageSlice::F8E4M3(s) => {
                let bytes = self.device.clone_dtoh(s)?;
                let v: Vec<float8::F8E4M3> =
                    bytes.into_iter().map(float8::F8E4M3::from_bits).collect();
                Ok(CpuStorage::F8E4M3(v))
            }
        }
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let device = self.device.clone();
        let slice = Affine(mul, add).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        let device = self.device.clone();
        let slice = Powf(e).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        let device = self.device.clone();
        let slice = Elu(alpha).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn reduce_op(&self, op: ReduceOp, l: &Layout, sum_dims: &[usize]) -> Result<Self> {
        let device = self.device.clone();
        let slice = FastReduce(sum_dims, op).map(&self.slice, &device, l)?;
        Ok(Self { slice, device })
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, l1: &Layout, l2: &Layout) -> Result<Self> {
        let device = self.device.clone();
        let slice = Cmp(op).map(&self.slice, l1, &rhs.slice, l2, &device)?;
        Ok(Self { slice, device })
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let dev = self.device.clone();

        let ds = dims_and_strides(&dev, layout, 1)?;
        let start_o = layout.start_offset();
        let src_ptr = unsafe { self.slice.offset_ptr(start_o) };

        let (grid, block) = launch_config_layout(&dev, el, ds.is_null());
        let ds_ptr: *const usize = ds.as_ptr();

        let src_dtype = self.slice.dtype();
        let slice = match dtype {
            DType::U8 => cast_launch!(
                dev,
                grid,
                block,
                el,
                dims.len(),
                ds_ptr,
                src_ptr,
                src_dtype,
                u8,
                U8
            ),
            DType::U32 => cast_launch!(
                dev,
                grid,
                block,
                el,
                dims.len(),
                ds_ptr,
                src_ptr,
                src_dtype,
                u32,
                U32
            ),
            DType::I64 => cast_launch!(
                dev,
                grid,
                block,
                el,
                dims.len(),
                ds_ptr,
                src_ptr,
                src_dtype,
                i64,
                I64
            ),
            DType::BF16 => cast_launch!(
                dev,
                grid,
                block,
                el,
                dims.len(),
                ds_ptr,
                src_ptr,
                src_dtype,
                bf16,
                BF16
            ),
            DType::F16 => cast_launch!(
                dev,
                grid,
                block,
                el,
                dims.len(),
                ds_ptr,
                src_ptr,
                src_dtype,
                f16,
                F16
            ),
            DType::F32 => cast_launch!(
                dev,
                grid,
                block,
                el,
                dims.len(),
                ds_ptr,
                src_ptr,
                src_dtype,
                f32,
                F32
            ),
            DType::F64 => cast_launch!(
                dev,
                grid,
                block,
                el,
                dims.len(),
                ds_ptr,
                src_ptr,
                src_dtype,
                f64,
                F64
            ),
            // Still real: the shared `cast.cu` declares no `CAST_OP` with an
            // int16_t/int32_t source or destination (the only i32 entries are
            // the fp8 pair below), so there is no kernel to launch.
            DType::I16 | DType::I32 => {
                return Err(crate::Error::Msg(
                    "i16/i32 dtypes are not supported for to_dtype on ROCm".to_string(),
                ))
            }
            // `cast.cu` does ship the f8e4m3 casts, but they are named
            // `cast_*_f8_e4m3` while `DType::as_str` yields `f8e4m3` and
            // `cast_launch!` derives the destination suffix from the Rust type
            // (F8E4M3 is stored as `u8` here). Wiring that up is a separate
            // change; the other four dtypes have no kernels at all.
            DType::F8E4M3 | DType::F4 | DType::F6E2M3 | DType::F6E3M2 | DType::F8E8M0 => {
                return Err(crate::Error::Msg(format!(
                    "{:?} dtype is not supported for to_dtype on ROCm",
                    dtype
                )))
            }
        };

        Ok(Self { slice, device: dev })
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let device = self.device.clone();
        let slice = B::V.map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn binary_impl<B: BinaryOpT>(&self, rhs: &Self, l1: &Layout, l2: &Layout) -> Result<Self> {
        let device = self.device.clone();
        let slice = B::V.map(&self.slice, l1, &rhs.slice, l2, &device)?;
        Ok(Self { slice, device })
    }

    fn where_cond(&self, l: &Layout, t: &Self, lt: &Layout, f: &Self, lf: &Layout) -> Result<Self> {
        let device = self.device.clone();
        let slice = WhereCond(self, l).map(&t.slice, lt, &f.slice, lf, &device)?;
        Ok(Self { slice, device })
    }

    #[cfg(not(feature = "miopen"))]
    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        ops_conv::conv1d(self, l, kernel, kernel_l, params)
    }

    #[cfg(feature = "miopen")]
    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        miopen::conv1d(self, l, kernel, kernel_l, params)
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        ops_conv::conv_transpose1d(self, l, kernel, kernel_l, params)
    }

    #[cfg(not(feature = "miopen"))]
    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        ops_conv::conv2d(self, l, kernel, kernel_l, params)
    }

    #[cfg(feature = "miopen")]
    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        miopen::conv2d(self, l, kernel, kernel_l, params)
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        ops_conv::conv_transpose2d(self, l, kernel, kernel_l, params)
    }

    fn avg_pool2d(&self, l: &Layout, k: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        let device = self.device.clone();
        let slice = ops_pool::Pool2D {
            w_k: k.0,
            h_k: k.1,
            w_stride: stride.0,
            h_stride: stride.1,
            op: ops_pool::PoolOp::Avg,
        }
        .map(&self.slice, &device, l)?;
        Ok(Self { slice, device })
    }

    fn max_pool2d(&self, l: &Layout, k: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        let device = self.device.clone();
        let slice = ops_pool::Pool2D {
            w_k: k.0,
            h_k: k.1,
            w_stride: stride.0,
            h_stride: stride.1,
            op: ops_pool::PoolOp::Max,
        }
        .map(&self.slice, &device, l)?;
        Ok(Self { slice, device })
    }

    /// `conv.cu` ships no 1-D upsample kernel; cuda_backend bails here too.
    fn upsample_nearest1d(&self, _l: &Layout, _sz: usize) -> Result<Self> {
        crate::bail!("upsample-nearest1d is not supported on rocm")
    }

    fn upsample_nearest2d(&self, l: &Layout, out_w: usize, out_h: usize) -> Result<Self> {
        let device = self.device.clone();
        let slice = ops_pool::UpsampleNearest2D(out_w, out_h).map(&self.slice, &device, l)?;
        Ok(Self { slice, device })
    }

    fn upsample_bilinear2d(
        &self,
        l: &Layout,
        out_h: usize,
        out_w: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Self> {
        let device = self.device.clone();
        let slice = ops_pool::UpsampleBilinear2D {
            out_w,
            out_h,
            align_corners,
            scale_h_factor: scale_h,
            scale_w_factor: scale_w,
        }
        .map(&self.slice, &device, l)?;
        Ok(Self { slice, device })
    }

    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        let device = self.device.clone();
        let slice = Gather(ids, ids_l, dim).map(&self.slice, &device, l)?;
        Ok(Self { slice, device })
    }

    fn scatter_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let device = self.device.clone();
        Scatter::new(ids, ids_l, dim, ScatterKind::Set).map(
            &mut self.slice,
            l,
            &src.slice,
            src_l,
            &device,
        )
    }

    fn scatter_add_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let device = self.device.clone();
        Scatter::new(ids, ids_l, dim, ScatterKind::Add).map(
            &mut self.slice,
            l,
            &src.slice,
            src_l,
            &device,
        )
    }

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        let device = self.device.clone();
        let slice = IndexSelect(ids, ids_l, dim).map(&self.slice, &device, l)?;
        Ok(Self { slice, device })
    }

    fn index_add(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        let device = self.device.clone();
        let mut acc = unsafe { device.alloc_uninit(l.shape(), self.dtype())? };
        self.copy_strided_src(&mut acc, 0, l)?;
        // DELIBERATE DIVERGENCE FROM cuda_backend.
        //
        // `acc` was just allocated and `copy_strided_src` filled it from element
        // 0, so it is contiguous with no start offset regardless of what `l`
        // describes. cuda_backend passes `l` itself here, which re-applies
        // `l.start_offset()` to a buffer that does not carry one, so
        // `index_add` on a narrowed tensor accumulates into the wrong rows there
        // (`[[1,2,3],[14,5,26]]` instead of `[[11,2,23],[34,5,46]]`).
        //
        // The CUDA path is the outlier: metal_backend does not apply this layout
        // at all, and `Tensor::scatter` builds `Layout::contiguous(shape)` for
        // the same situation. Pinned by
        // `index_add_handles_a_source_with_a_start_offset`; worth fixing upstream.
        let acc_l = Layout::contiguous(l.shape());
        IndexAdd(ids, ids_l, dim).map(&mut acc.slice, &acc_l, &src.slice, src_l, &device)?;
        Ok(acc)
    }

    fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        gemm::matmul(self, rhs, (b, m, n, k), lhs_l, rhs_l)
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        let src_shape = src_l.shape();
        let dims = src_shape.dims();
        let el_count = src_shape.elem_count();
        if el_count == 0 {
            return Ok(());
        }
        if self.dtype() != dst.dtype() {
            crate::bail!("dtype mismatch in copy_strided_src");
        }
        self.device.bind()?;

        // Callers over-request: `Tensor::cat` and the autograd accumulators size
        // the copy from the *source shape*, which can run past the end of either
        // allocation once start offsets are applied. cuda_backend clamps for the
        // same reason (`slice_src_and_dst`); without it this memcpy walks off the
        // end of a device buffer.
        let dst_avail = dst.slice.count().saturating_sub(dst_offset);

        if src_l.is_contiguous() {
            let src_avail = self.slice.count().saturating_sub(src_l.start_offset());
            let to_copy = el_count.min(src_avail).min(dst_avail);
            if to_copy == 0 {
                return Ok(());
            }
            let el_size = self.slice.elem_size();
            let src_ptr = unsafe { self.slice.offset_ptr(src_l.start_offset()) };
            let dst_ptr = unsafe { dst.slice.offset_ptr(dst_offset) };
            // Stream-ordered: the copy is sequenced against the kernels that
            // produced `src` and consume `dst`, so the host never has to wait.
            let result = unsafe {
                bindings::hipMemcpyAsync(
                    dst_ptr,
                    src_ptr,
                    to_copy * el_size,
                    bindings::hipMemcpyKind_hipMemcpyDeviceToDevice,
                    self.device.stream().as_raw(),
                )
            };
            if result != bindings::hipError_t_hipSuccess {
                crate::bail!("hipMemcpyAsync failed with error {}", result);
            }
            return Ok(());
        }

        // Same destination clamp. The source side is deliberately *not* clamped
        // here: a broadcast layout carries stride 0 and legitimately reads far
        // fewer elements than it writes.
        let el_count = el_count.min(dst_avail);
        if el_count == 0 {
            return Ok(());
        }
        // This branch is only reached for a non-contiguous source.
        let (grid, block) = launch_config_layout(&self.device, el_count, false);
        let ds = dims_and_strides(&self.device, src_l, 1)?;

        macro_rules! copy_strided {
            ($variant:ident, $suffix:expr, $ty:ty) => {{
                let (src_mem, dst_mem) = match (&self.slice, &mut dst.slice) {
                    (RocmStorageSlice::$variant(s), RocmStorageSlice::$variant(d)) => (s, d),
                    _ => crate::bail!("dtype mismatch in copy_strided_src"),
                };
                let func_name = format!("ucopy_{}", $suffix);
                let (src_ptr, dst_ptr) = unsafe {
                    (
                        src_mem.ptr_at(src_l.start_offset()),
                        dst_mem.ptr_at(dst_offset),
                    )
                };
                let ds_ptr: *const usize = ds.as_ptr();
                unsafe {
                    launch_kernel(
                        &self.device,
                        &kernels::UNARY,
                        &func_name,
                        grid,
                        block,
                        &mut [
                            &el_count as *const usize as *mut std::ffi::c_void,
                            &dims.len() as *const usize as *mut std::ffi::c_void,
                            (&ds_ptr) as *const *const usize as *mut std::ffi::c_void,
                            (&src_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&dst_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                        ],
                    )?;
                }
            }};
        }

        match &self.slice {
            RocmStorageSlice::U8(_) => copy_strided!(U8, "u8", u8),
            RocmStorageSlice::U32(_) => copy_strided!(U32, "u32", u32),
            RocmStorageSlice::I16(_) => copy_strided!(I16, "i16", i16),
            RocmStorageSlice::I32(_) => copy_strided!(I32, "i32", i32),
            RocmStorageSlice::I64(_) => copy_strided!(I64, "i64", i64),
            RocmStorageSlice::BF16(_) => copy_strided!(BF16, "bf16", bf16),
            RocmStorageSlice::F16(_) => copy_strided!(F16, "f16", f16),
            RocmStorageSlice::F32(_) => copy_strided!(F32, "f32", f32),
            RocmStorageSlice::F64(_) => copy_strided!(F64, "f64", f64),
            // `unary.cu` gates `ucopy_f8_e4m3` on `__CUDA_ARCH__ >= 890` while the
            // ROCm module is compiled at 800, so that symbol is not in the
            // binary. F8E4M3 is exactly one byte and its payload is already held
            // as `u8`, so `ucopy_u8` moves the identical bytes — this is the same
            // reasoning `try_clone` uses for its raw buffer copy.
            RocmStorageSlice::F8E4M3(_) => copy_strided!(F8E4M3, "u8", u8),
        }

        Ok(())
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
        self.device.bind()?;
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
        // Stream-ordered, like the contiguous path in `copy_strided_src`.
        let result = unsafe {
            bindings::hipMemcpy2DAsync(
                dst_ptr,
                dpitch,
                src_ptr,
                spitch,
                width,
                d1,
                bindings::hipMemcpyKind_hipMemcpyDeviceToDevice,
                self.device.stream().as_raw(),
            )
        };
        if result != bindings::hipError_t_hipSuccess {
            crate::bail!("hipMemcpy2DAsync failed with error {}", result);
        }
        Ok(())
    }

    fn const_set(&mut self, val: crate::scalar::Scalar, layout: &Layout) -> Result<()> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el_count = shape.elem_count();
        if el_count == 0 {
            return Ok(());
        }

        let ds = dims_and_strides(&self.device, layout, 1)?;
        let (grid, block) = launch_config_layout(&self.device, el_count, ds.is_null());

        macro_rules! const_set {
            ($variant:ident, $suffix:expr, $ty:ty, $val:expr) => {{
                let mem = match &mut self.slice {
                    RocmStorageSlice::$variant(m) => m,
                    _ => crate::bail!("dtype mismatch in const_set"),
                };
                let func_name = format!("const_set_{}", $suffix);
                let out_ptr = unsafe { mem.ptr_at(layout.start_offset()) };
                let scalar_val: $ty = $val;
                let ds_ptr: *const usize = ds.as_ptr();
                unsafe {
                    launch_kernel(
                        &self.device,
                        &kernels::FILL,
                        &func_name,
                        grid,
                        block,
                        &mut [
                            &el_count as *const usize as *mut std::ffi::c_void,
                            &dims.len() as *const usize as *mut std::ffi::c_void,
                            (&ds_ptr) as *const *const usize as *mut std::ffi::c_void,
                            &scalar_val as *const $ty as *mut std::ffi::c_void,
                            (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                        ],
                    )?;
                }
            }};
        }

        match (&mut self.slice, val) {
            (RocmStorageSlice::U8(_), crate::scalar::Scalar::U8(v)) => const_set!(U8, "u8", u8, v),
            (RocmStorageSlice::U32(_), crate::scalar::Scalar::U32(v)) => {
                const_set!(U32, "u32", u32, v)
            }
            (RocmStorageSlice::I64(_), crate::scalar::Scalar::I64(v)) => {
                const_set!(I64, "i64", i64, v)
            }
            (RocmStorageSlice::F32(_), crate::scalar::Scalar::F32(v)) => {
                const_set!(F32, "f32", f32, v)
            }
            (RocmStorageSlice::F64(_), crate::scalar::Scalar::F64(v)) => {
                const_set!(F64, "f64", f64, v)
            }
            (RocmStorageSlice::BF16(_), crate::scalar::Scalar::BF16(v)) => {
                const_set!(BF16, "bf16", bf16, v)
            }
            (RocmStorageSlice::F16(_), crate::scalar::Scalar::F16(v)) => {
                const_set!(F16, "f16", f16, v)
            }
            (RocmStorageSlice::I16(_), crate::scalar::Scalar::I16(v)) => {
                const_set!(I16, "i16", i16, v)
            }
            (RocmStorageSlice::I32(_), crate::scalar::Scalar::I32(v)) => {
                const_set!(I32, "i32", i32, v)
            }
            // `RocmStorageSlice::F8E4M3` keeps its payload as bytes, and F8E4M3
            // is exactly one byte, so `ptr_at` on the u8 buffer still lands on
            // element `start_offset`.
            (RocmStorageSlice::F8E4M3(_), crate::scalar::Scalar::F8E4M3(v)) => {
                const_set!(F8E4M3, "f8_e4m3", float8::F8E4M3, v)
            }
            _ => crate::bail!("dtype mismatch in const_set"),
        }

        Ok(())
    }
}
