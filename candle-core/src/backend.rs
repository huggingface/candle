use crate::{CpuStorage, DType, Layout, Result, Shape};

trait BackendStorage: Sized {
    type Device: BackendDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self>;

    fn dtype(&self) -> DType;

    fn device(&self) -> &Self::Device;

    fn to_cpu_storage(&self) -> Result<CpuStorage>;

    fn affine(&self, _: &Layout, _: f64, _: f64) -> Result<Self>;

    fn elu(&self, _: &Layout, _: f64) -> Result<Self>;

    fn sum(&self, _: &Layout, _: &[usize]) -> Result<Self>;

    fn divide_by_sum_over_dim(&mut self, _: &Shape, _: usize) -> Result<()>;

    fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self>;

    fn unary_impl<B: crate::op::UnaryOp>(&self, _: &Layout) -> Result<Self>;

    fn binary_impl<B: crate::op::BinaryOp>(&self, _: &Self, _: &Layout, _: &Layout)
        -> Result<Self>;

    fn where_cond(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self>;

    fn conv1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConv1D,
    ) -> Result<Self>;

    fn embedding(&self, _: &Layout, _: &Self, _: &Layout) -> Result<Self>;

    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self>;

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()>;
}

trait BackendDevice: Sized {
    type Storage: BackendStorage;

    fn new(_: usize) -> Result<Self>;

    fn same_device(&self, _: &Self) -> bool;

    fn zeros_impl(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage>;

    fn ones_impl(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage>;

    fn storage_from_cpu_storage(&self, _: &CpuStorage) -> Result<Self::Storage>;

    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage>;

    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage>;
}
