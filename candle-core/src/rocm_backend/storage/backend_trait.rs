//! BackendStorage trait implementation for RocmStorage
//! Created by: TEAM-488 (Phase 1 - Device integration)
//! Kernel operations: TEAM-494 (unary, binary, reduce, cmp, where)
//! MIOpen operations: TEAM-495 (conv, pooling)
//! rocBLAS operations: TEAM-496 (matmul)
//! Indexing operations: TEAM-497 (gather, scatter, index_select, index_add)
//! CUDA parity verified by: TEAM-497

use super::RocmStorage;
use crate::rocm_backend::{ops, RocmDevice};
use crate::Result;

impl crate::backend::BackendStorage for RocmStorage {
    type Device = RocmDevice;

    fn try_clone(&self, layout: &crate::Layout) -> Result<Self> {
        let slice = ops::Clone.map(&self.slice, self.device(), layout)?;
        let device = self.device.clone();
        Ok(Self { slice, device })
    }

    fn dtype(&self) -> crate::DType {
        self.dtype()
    }

    fn device(&self) -> &RocmDevice {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<crate::CpuStorage> {
        self.to_cpu_storage_impl()
    }

    fn to_dtype(&self, layout: &crate::Layout, dtype: crate::DType) -> Result<Self> {
        self.to_dtype_impl(layout, dtype)
    }

    // Created by: TEAM-494 | CUDA parity verified by: TEAM-497 (cuda_backend/mod.rs:1472-1476)
    fn affine(&self, layout: &crate::Layout, mul: f64, add: f64) -> Result<Self> {
        self.affine_impl(layout, mul, add)
    }

    // Created by: TEAM-494 | CUDA parity verified by: TEAM-497 (cuda_backend/mod.rs:1478-1482)
    fn powf(&self, layout: &crate::Layout, e: f64) -> Result<Self> {
        self.powf_impl(layout, e)
    }

    // Created by: TEAM-494 | CUDA parity verified by: TEAM-497 (cuda_backend/mod.rs:1484-1488)
    fn elu(&self, layout: &crate::Layout, alpha: f64) -> Result<Self> {
        self.elu_impl(layout, alpha)
    }

    // Created by: TEAM-494 | CUDA parity verified by: TEAM-497 (cuda_backend/mod.rs:1490-1494)
    fn reduce_op(
        &self,
        op: crate::op::ReduceOp,
        layout: &crate::Layout,
        sum_dims: &[usize],
    ) -> Result<Self> {
        self.reduce_op_impl(op, layout, sum_dims)
    }

    // Created by: TEAM-494 | CUDA parity verified by: TEAM-497 (cuda_backend/mod.rs:1496-1500)
    fn cmp(
        &self,
        op: crate::op::CmpOp,
        rhs: &Self,
        lhs_l: &crate::Layout,
        rhs_l: &crate::Layout,
    ) -> Result<Self> {
        self.cmp_impl(op, rhs, lhs_l, rhs_l)
    }

    // Created by: TEAM-494 | CUDA parity verified by: TEAM-497 (cuda_backend/mod.rs:1502-1506)
    fn unary_impl<B: crate::op::UnaryOpT>(&self, layout: &crate::Layout) -> Result<Self> {
        self.unary_impl::<B>(layout)
    }

    // Created by: TEAM-494 | CUDA parity verified by: TEAM-497 (cuda_backend/mod.rs:1508-1517)
    fn binary_impl<B: crate::op::BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &crate::Layout,
        rhs_l: &crate::Layout,
    ) -> Result<Self> {
        self.binary_impl::<B>(rhs, lhs_l, rhs_l)
    }

    // Created by: TEAM-494 | CUDA parity verified by: TEAM-497 (cuda_backend/mod.rs:1556-1566)
    fn where_cond(
        &self,
        layout: &crate::Layout,
        t: &Self,
        layout_t: &crate::Layout,
        f: &Self,
        layout_f: &crate::Layout,
    ) -> Result<Self> {
        self.where_cond_impl(layout, t, layout_t, f, layout_f)
    }

    // Created by: TEAM-497 | CUDA parity verified by: TEAM-497 (cuda_backend/mod.rs:1621-1684)
    fn conv1d(
        &self,
        l: &crate::Layout,
        kernel: &Self,
        kernel_l: &crate::Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        self.conv1d_impl(l, kernel, kernel_l, params)
    }

    // Created by: TEAM-497 | CUDA parity verified by: TEAM-497 (cuda_backend/mod.rs:1686-1743)
    fn conv_transpose1d(
        &self,
        l: &crate::Layout,
        kernel: &Self,
        kernel_l: &crate::Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        self.conv_transpose1d_impl(l, kernel, kernel_l, params)
    }

    // Created by: TEAM-495 | CUDA parity verified by: TEAM-497 (cuda_backend/mod.rs:1801-1863)
    fn conv2d(
        &self,
        inp_l: &crate::Layout,
        kernel: &Self,
        kernel_l: &crate::Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        self.conv2d_impl(inp_l, kernel, kernel_l, params)
    }

    // Created by: TEAM-497 | CUDA parity verified by: TEAM-497 (cuda_backend/mod.rs:1866-1876)
    fn conv_transpose2d(
        &self,
        l: &crate::Layout,
        kernel: &Self,
        kernel_l: &crate::Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        self.conv_transpose2d_impl(l, kernel, kernel_l, params)
    }

    // Created by: TEAM-495 | CUDA parity verified by: TEAM-497 (cuda_backend/mod.rs:1879-1890)
    fn avg_pool2d(
        &self,
        layout: &crate::Layout,
        k: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        self.avg_pool2d_impl(layout, k, stride)
    }

    // Created by: TEAM-495 | CUDA parity verified by: TEAM-497 (cuda_backend/mod.rs:1892-1903)
    fn max_pool2d(
        &self,
        layout: &crate::Layout,
        k: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        self.max_pool2d_impl(layout, k, stride)
    }

    // TEAM-497: Not supported (CUDA: cuda_backend/mod.rs:1905-1907)
    fn upsample_nearest1d(&self, _: &crate::Layout, _: usize) -> Result<Self> {
        crate::bail!("upsample-nearest1d is not supported on rocm")
    }

    // Created by: TEAM-497 | CUDA parity verified by: TEAM-497 (cuda_backend/mod.rs:1909-1913)
    fn upsample_nearest2d(
        &self,
        layout: &crate::Layout,
        out_w: usize,
        out_h: usize,
    ) -> Result<Self> {
        self.upsample_nearest2d_impl(layout, out_w, out_h)
    }

    // Created by: TEAM-497 | CUDA parity verified by: TEAM-497 (cuda_backend/mod.rs:1920-1924)
    fn gather(
        &self,
        layout: &crate::Layout,
        ids: &Self,
        ids_l: &crate::Layout,
        dim: usize,
    ) -> Result<Self> {
        self.gather_impl(layout, ids, ids_l, dim)
    }

    // Created by: TEAM-497 | CUDA parity verified by: TEAM-497 (cuda_backend/mod.rs:1925-1936)
    fn scatter_set(
        &mut self,
        layout: &crate::Layout,
        ids: &Self,
        ids_l: &crate::Layout,
        src: &Self,
        src_l: &crate::Layout,
        dim: usize,
    ) -> Result<()> {
        self.scatter_set_impl(layout, ids, ids_l, src, src_l, dim)
    }

    // Created by: TEAM-497 | CUDA parity verified by: TEAM-497 (cuda_backend/mod.rs:1937-1948)
    fn scatter_add_set(
        &mut self,
        layout: &crate::Layout,
        ids: &Self,
        ids_l: &crate::Layout,
        src: &Self,
        src_l: &crate::Layout,
        dim: usize,
    ) -> Result<()> {
        self.scatter_add_set_impl(layout, ids, ids_l, src, src_l, dim)
    }

    // Created by: TEAM-497 | CUDA parity verified by: TEAM-497 (cuda_backend/mod.rs:1915-1919)
    fn index_select(
        &self,
        ids: &Self,
        layout: &crate::Layout,
        ids_l: &crate::Layout,
        dim: usize,
    ) -> Result<Self> {
        self.index_select_impl(ids, layout, ids_l, dim)
    }

    // Created by: TEAM-497 | CUDA parity verified by: TEAM-497 (cuda_backend/mod.rs:1949-1962)
    fn index_add(
        &self,
        layout: &crate::Layout,
        ids: &Self,
        ids_l: &crate::Layout,
        src: &Self,
        src_l: &crate::Layout,
        dim: usize,
    ) -> Result<Self> {
        self.index_add_impl(layout, ids, ids_l, src, src_l, dim)
    }

    // Created by: TEAM-496 | CUDA parity verified by: TEAM-497 (cuda_backend/mod.rs:1965-2019)
    fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &crate::Layout,
        rhs_l: &crate::Layout,
    ) -> Result<Self> {
        self.matmul_impl(rhs, (b, m, n, k), lhs_l, rhs_l)
    }

    // Created by: TEAM-495 | CUDA parity verified by: TEAM-497 (cuda_backend/mod.rs:2021-2066)
    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Result<()> {
        self.copy2d_impl(dst, d1, d2, src_s, dst_s, src_o, dst_o)
    }

    // Created by: TEAM-497 | CUDA parity verified by: TEAM-497 (cuda_backend/mod.rs:2068-end)
    fn copy_strided_src(
        &self,
        dst: &mut Self,
        dst_offset: usize,
        src_l: &crate::Layout,
    ) -> Result<()> {
        self.copy_strided_src_impl(dst, dst_offset, src_l)
    }
}
