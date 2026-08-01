//! Implementation of Backend traits for ROCm device
//!
//! The backend is split by concern rather than by dtype: this file holds
//! [`RocmStorage`] and its `BackendStorage` impl, which is dispatch and little
//! else — every method delegates to the module that owns that operation.
//!
//! | module | what it owns |
//! |---|---|
//! | [`alloc`] | the caching device allocator and [`SendSyncDeviceMemory`] |
//! | [`slice`] | [`RocmStorageSlice`], the dtype-tagged buffer |
//! | [`device`] / [`device_backend`] | the device, its stream and `BackendDevice` |
//! | [`launch`] | kernel names, launch geometry, the launch itself |
//! | [`params`] | the cached `[dims, strides…]` buffers kernels take |
//! | [`utils`] | the `Map*` dtype-dispatch traits ops are written against |
//! | `ops_*` | one family of operations each |
use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result};
pub use candle_rocm_kernels as kernels;
pub use rocm_rs;

mod alloc;
mod device;
mod device_backend;
mod error;
mod gemm;
mod gemm_precision;
mod launch;
#[cfg(feature = "miopen")]
mod miopen;
mod ops_cast;
mod ops_conv;
mod ops_conv_direct;
mod ops_copy;
mod ops_elementwise;
mod ops_indexing;
mod ops_pool;
mod ops_reduce;
mod ops_scalar;
mod params;
mod rng;
mod slice;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_alloc;
#[cfg(test)]
mod tests_conv;
#[cfg(test)]
mod tests_copy;
#[cfg(test)]
mod tests_custom_kernel;
#[cfg(test)]
mod tests_device;
#[cfg(test)]
mod tests_f8e4m3;
#[cfg(test)]
mod tests_gemm;
#[cfg(test)]
mod tests_indexing;
#[cfg(test)]
mod tests_rng;
#[cfg(test)]
mod tests_sort;
#[cfg(all(test, feature = "ug"))]
mod tests_ug;
mod wrappers;
pub use alloc::SendSyncDeviceMemory;
pub use device::{DeviceId, RocmDevice};
pub use error::{RocmError, WrapErr};
pub use gemm_precision::{
    gemm_reduced_precision_bf16, gemm_reduced_precision_f16, gemm_reduced_precision_f32,
    set_gemm_reduced_precision_bf16, set_gemm_reduced_precision_f16,
    set_gemm_reduced_precision_f32,
};
pub use launch::{
    kernel_name, launch_config, launch_config_for, launch_config_layout, try_kernel_name,
};
pub use params::{params_from_vec, ParamBuffer};
pub use slice::RocmStorageSlice;
pub use wrappers::RocmBlas;
pub mod utils;
pub use utils::{Map1, Map1Any, Map2, Map2Any, Map2InPlace, Map3, S};

pub(crate) use error::rocm_error;

use ops_elementwise::{CloneBuffer, Cmp, WhereCond};
use ops_indexing::{Gather, IndexAdd, IndexSelect, Scatter, ScatterKind};
use ops_reduce::FastReduce;
pub(crate) use ops_scalar::Affine;
use ops_scalar::{Elu, Powf};
use params::ParamCache;

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
        let slice = CloneBuffer.map(&self.slice, &device, layout)?;
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
            RocmStorageSlice::F8E4M3(s) => Ok(CpuStorage::F8E4M3(self.device.clone_dtoh(s)?)),
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
        ops_cast::to_dtype(self, layout, dtype)
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
        ops_copy::copy_strided_src(self, dst, dst_offset, src_l)
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
        ops_copy::copy2d(self, dst, d1, d2, src_s1, dst_s1, src_o, dst_o)
    }

    fn const_set(&mut self, val: crate::scalar::Scalar, layout: &Layout) -> Result<()> {
        ops_copy::const_set(self, val, layout)
    }
}
