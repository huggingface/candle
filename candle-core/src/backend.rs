//! Traits to Define Backend Behavior
//!
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape};

pub trait BackendStorage: Sized {
    type Device: BackendDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self>;

    fn dtype(&self) -> DType;

    fn device(&self) -> &Self::Device;

    // Maybe this should return a Cow instead so that no copy is done on the cpu case.
    fn to_cpu_storage(&self) -> Result<CpuStorage>;

    fn affine(&self, _: &Layout, _: f64, _: f64) -> Result<Self>;

    fn powf(&self, _: &Layout, _: f64) -> Result<Self>;

    fn elu(&self, _: &Layout, _: f64) -> Result<Self>;

    fn reduce_op(&self, _: ReduceOp, _: &Layout, _: &[usize]) -> Result<Self>;

    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self>;

    fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self>;

    fn unary_impl<B: UnaryOpT>(&self, _: &Layout) -> Result<Self>;

    fn binary_impl<B: BinaryOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> Result<Self>;

    fn where_cond(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self>;

    fn conv1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConv1D,
    ) -> Result<Self>;

    fn conv_transpose1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self>;

    fn conv2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConv2D,
    ) -> Result<Self>;

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self>;

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self>;
    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self>;
    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self>;
    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self>;
    fn upsample_bilinear2d(
        &self,
        _: &Layout,
        _: usize,
        _: usize,
        _: bool,
        _: Option<f64>,
        _: Option<f64>,
    ) -> Result<Self>;

    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self>;

    fn scatter_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()>;

    fn scatter_add_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()>;

    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self>;
    fn index_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self>;

    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self>;

    /// `y = lhs @ rhs + bias`: the matrix product (same shapes as
    /// `matmul`, `bmnk = (bsz, m, n, k)`) with the bias vector (one
    /// value per column of the product, `n` elements) added to every
    /// row.
    ///
    /// The default implementation is the unfused `matmul` followed by
    /// the element-wise add; backends with a fused kernel (the Vulkan
    /// cooperative-matrix affine) override it.
    #[allow(clippy::too_many_arguments)]
    fn affine_matmul(
        &self,
        rhs: &Self,
        bias: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
        bias_l: &Layout,
    ) -> Result<Self> {
        let (bsz, m, n, _) = bmnk;
        let out = self.matmul(rhs, bmnk, lhs_l, rhs_l)?;
        let out_l = if bsz == 1 {
            crate::Layout::contiguous_with_offset(&[m, n], 0)
        } else {
            crate::Layout::contiguous_with_offset(&[bsz, m, n], 0)
        };
        out.binary_impl::<crate::op::Add>(bias, &out_l, bias_l)
    }

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()>;

    #[allow(clippy::too_many_arguments)]
    // Similar to cudaMemcpy2D, though values are in elements and not in bytes.
    fn copy2d(
        &self,
        _: &mut Self,
        _d1: usize,
        _d2: usize,
        _src_stride1: usize,
        _dst_stride1: usize,
        _src_offset: usize,
        _dst_offset: usize,
    ) -> Result<()>;

    fn const_set(&mut self, _: crate::scalar::Scalar, _: &Layout) -> Result<()>;
}

pub trait BackendDevice: Sized + std::fmt::Debug + Clone {
    type Storage: BackendStorage;

    // TODO: Make the usize generic and part of a generic DeviceLocation.
    fn new(_: usize) -> Result<Self>;

    fn location(&self) -> crate::DeviceLocation;

    fn same_device(&self, _: &Self) -> bool;

    fn zeros_impl(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage>;

    /// # Safety
    /// This function is unsafe as it doesn't initialize the underlying data store.
    /// The caller should ensure that the data is properly initialized as early as possible
    /// after this call.
    unsafe fn alloc_uninit(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage>;

    fn storage_from_slice<T: crate::WithDType>(&self, _: &[T]) -> Result<Self::Storage>;

    fn storage_from_cpu_storage(&self, _: &CpuStorage) -> Result<Self::Storage>;

    fn storage_from_cpu_storage_owned(&self, _: CpuStorage) -> Result<Self::Storage>;

    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage>;

    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage>;

    fn set_seed(&self, _: u64) -> Result<()>;
    fn get_current_seed(&self) -> Result<u64>;

    /// Synchronize should block until all the operations on the device are completed.
    fn synchronize(&self) -> Result<()>;
}
