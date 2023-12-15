use crate::op::{BinaryOpPub, CmpOp, ReduceOp, UnaryOpPub};
use crate::{CpuStorage, DType, Layout, Result, Shape};
use std::{fmt::Debug, hash::Hash};

pub trait BackendStorage: 'static + Sized + Send + Sync + Debug {
    type Device: BackendDevice<Storage = Self>;

    fn try_clone(&self, layout: &Layout) -> Result<Self>;

    fn dtype(&self) -> DType;

    fn device(&self) -> Self::Device;

    // Maybe this should return a Cow instead so that no copy is done on the cpu case.
    fn to_cpu_storage(&self) -> Result<CpuStorage>;

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self>;

    fn powf(&self, layout: &Layout, e: f64) -> Result<Self>;

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self>;

    fn reduce(&self, op: ReduceOp, layout: &Layout, sum_dims: &[usize]) -> Result<Self>;

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self>;

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self>;

    fn unary_impl<U: UnaryOpPub>(&self, layout: &Layout) -> Result<Self>;

    fn binary_impl<B: BinaryOpPub>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self>;

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self>;

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self>;

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self>;

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self>;

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self>;

    fn avg_pool2d(&self, l: &Layout, k: (usize, usize), stride: (usize, usize)) -> Result<Self>;

    fn max_pool2d(&self, l: &Layout, k: (usize, usize), stride: (usize, usize)) -> Result<Self>;

    fn upsample_nearest1d(&self, l: &Layout, out_sz: usize) -> Result<Self>;

    fn upsample_nearest2d(&self, l: &Layout, out_w: usize, out_h: usize) -> Result<Self>;

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self>;

    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self>;

    fn scatter_add(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self>;

    fn index_add(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self>;

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self>;

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()>;
}

pub trait BackendDevice: 'static + Sized + Send + Sync + Clone + Debug {
    type Storage: BackendStorage<Device = Self>;
    type Location: BackendLocation;

    fn location(&self) -> Self::Location;

    fn same_device(&self, rhs: &Self) -> bool;

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage>;

    fn ones_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage>;

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage>;

    fn rand_uniform_f64(
        &self,
        shape: &Shape,
        dtype: DType,
        lo: f64,
        up: f64,
    ) -> Result<Self::Storage>;

    fn rand_normal_f64(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<Self::Storage>;

    fn set_seed(&self, seed: u64) -> Result<()>;
}

pub trait BackendLocation: 'static + Send + Sync + Clone + Eq + Hash + Debug {}
impl<T: 'static + Send + Sync + Clone + Eq + Hash + Debug> BackendLocation for T {}
