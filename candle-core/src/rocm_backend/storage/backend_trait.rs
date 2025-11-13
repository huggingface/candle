//! BackendStorage trait implementation for RocmStorage

use crate::rocm_backend::{ops, RocmDevice};
use crate::Result;
use super::RocmStorage;

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

    fn affine(&self, layout: &crate::Layout, mul: f64, add: f64) -> Result<Self> {
        self.affine_impl(layout, mul, add)
    }

    fn powf(&self, layout: &crate::Layout, e: f64) -> Result<Self> {
        self.powf_impl(layout, e)
    }

    fn elu(&self, layout: &crate::Layout, alpha: f64) -> Result<Self> {
        self.elu_impl(layout, alpha)
    }

    fn reduce_op(
        &self,
        op: crate::op::ReduceOp,
        layout: &crate::Layout,
        sum_dims: &[usize],
    ) -> Result<Self> {
        self.reduce_op_impl(op, layout, sum_dims)
    }

    fn cmp(
        &self,
        op: crate::op::CmpOp,
        rhs: &Self,
        lhs_l: &crate::Layout,
        rhs_l: &crate::Layout,
    ) -> Result<Self> {
        self.cmp_impl(op, rhs, lhs_l, rhs_l)
    }

    fn unary_impl<B: crate::op::UnaryOpT>(&self, layout: &crate::Layout) -> Result<Self> {
        self.unary_impl::<B>(layout)
    }

    fn binary_impl<B: crate::op::BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &crate::Layout,
        rhs_l: &crate::Layout,
    ) -> Result<Self> {
        self.binary_impl::<B>(rhs, lhs_l, rhs_l)
    }

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

    fn conv1d(
        &self,
        _l: &crate::Layout,
        _kernel: &Self,
        _kernel_l: &crate::Layout,
        _params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        unimplemented!("conv1d")
    }

    fn conv_transpose1d(
        &self,
        _l: &crate::Layout,
        _kernel: &Self,
        _kernel_l: &crate::Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        unimplemented!("conv_transpose1d")
    }

    fn conv2d(
        &self,
        inp_l: &crate::Layout,
        kernel: &Self,
        kernel_l: &crate::Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        self.conv2d_impl(inp_l, kernel, kernel_l, params)
    }

    fn conv_transpose2d(
        &self,
        _l: &crate::Layout,
        _kernel: &Self,
        _kernel_l: &crate::Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        unimplemented!("conv_transpose2d")
    }

    fn avg_pool2d(
        &self,
        layout: &crate::Layout,
        k: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        self.avg_pool2d_impl(layout, k, stride)
    }

    fn max_pool2d(
        &self,
        layout: &crate::Layout,
        k: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        self.max_pool2d_impl(layout, k, stride)
    }

    fn upsample_nearest1d(&self, _: &crate::Layout, _: usize) -> Result<Self> {
        unimplemented!("upsample_nearest1d")
    }

    fn upsample_nearest2d(&self, _: &crate::Layout, _: usize, _: usize) -> Result<Self> {
        unimplemented!("upsample_nearest2d")
    }

    fn gather(&self, _: &crate::Layout, _: &Self, _: &crate::Layout, _: usize) -> Result<Self> {
        unimplemented!("gather")
    }

    fn scatter_set(
        &mut self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: usize,
    ) -> Result<()> {
        unimplemented!("scatter_set")
    }

    fn scatter_add_set(
        &mut self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: usize,
    ) -> Result<()> {
        unimplemented!("scatter_add_set")
    }

    fn index_select(
        &self,
        _: &Self,
        _: &crate::Layout,
        _: &crate::Layout,
        _: usize,
    ) -> Result<Self> {
        unimplemented!("index_select")
    }

    fn index_add(
        &self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: usize,
    ) -> Result<Self> {
        unimplemented!("index_add")
    }

    fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &crate::Layout,
        rhs_l: &crate::Layout,
    ) -> Result<Self> {
        self.matmul_impl(rhs, (b, m, n, k), lhs_l, rhs_l)
    }

    fn copy2d(
        &self,
        _: &mut Self,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
    ) -> Result<()> {
        unimplemented!("copy2d")
    }

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &crate::Layout) -> Result<()> {
        unimplemented!("copy_strided_src")
    }
}
