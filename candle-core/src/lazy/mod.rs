use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape};

#[derive(Debug, Clone)]
pub struct LazyStorage {
    shape: Shape,
    dtype: DType,
}

impl BackendStorage for LazyStorage {
    type Device = LazyDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        Ok(self.clone())
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &LazyDevice
    }

    // Maybe this should return a Cow instead so that no copy is done on the cpu case.
    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        todo!()
    }

    fn affine(&self, _: &Layout, _: f64, _: f64) -> Result<Self> {
        todo!()
    }

    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        todo!()
    }

    fn elu(&self, _: &Layout, _: f64) -> Result<Self> {
        todo!()
    }

    fn reduce_op(&self, _: ReduceOp, _: &Layout, _: &[usize]) -> Result<Self> {
        todo!()
    }

    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        todo!()
    }

    fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self> {
        todo!()
    }

    fn unary_impl<B: UnaryOpT>(&self, _: &Layout) -> Result<Self> {
        todo!()
    }

    fn binary_impl<B: BinaryOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        todo!()
    }

    fn where_cond(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
        todo!()
    }

    fn conv1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        todo!()
    }

    fn conv_transpose1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        todo!()
    }

    fn conv2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        todo!()
    }

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        todo!()
    }

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        todo!()
    }
    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        todo!()
    }
    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        todo!()
    }
    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        todo!()
    }

    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        todo!()
    }

    fn scatter_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()> {
        todo!()
    }

    fn scatter_add_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()> {
        todo!()
    }

    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self> {
        todo!()
    }
    fn index_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self> {
        todo!()
    }

    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self> {
        todo!()
    }

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()> {
        todo!()
    }

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
    ) -> Result<()> {
        todo!()
    }

    fn const_set(&mut self, _: crate::scalar::Scalar, _: &Layout) -> Result<()> {
        todo!()
    }
}


#[derive(Debug, Clone)]
pub struct LazyDevice;


impl BackendDevice for LazyDevice {
    type Storage = LazyStorage;

    fn new(_ordinal: usize) -> Result<Self> {
        Ok(LazyDevice)
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Lazy
    }

    fn same_device(&self, _rhs: &Self) -> bool {
        true
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        Ok(LazyStorage { shape: shape.clone(), dtype })
    }

    fn zeros_impl(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        todo!()
    }

    fn storage_from_slice<T: crate::WithDType>(&self, _s: &[T]) -> Result<Self::Storage> {
        todo!()
    }

    fn storage_from_cpu_storage(&self, _storage: &CpuStorage) -> Result<Self::Storage> {
        todo!()
    }

    fn storage_from_cpu_storage_owned(&self, _storage: CpuStorage) -> Result<Self::Storage> {
        todo!()
    }

    fn rand_uniform(
        &self,
        _shape: &Shape,
        _dtype: DType,
        _min: f64,
        _max: f64,
    ) -> Result<Self::Storage> {
        todo!()
    }

    fn rand_normal(
        &self,
        _shape: &Shape,
        _dtype: DType,
        _mean: f64,
        _stddev: f64,
    ) -> Result<Self::Storage> {
        todo!()
    }

    fn set_seed(&self, _seed: u64) -> Result<()> {
        todo!()
    }

    fn get_current_seed(&self) -> Result<u64> {
        todo!()
    }

    fn synchronize(&self) -> Result<()> {
        todo!()
    }
}
