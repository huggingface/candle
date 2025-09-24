use crate::{
    BackendDevice, BackendStorage, CpuStorage, DType, DeviceLocation, FloatDType, Layout, NdArray,
    Result, Shape, WithDType,
};

#[derive(Clone, Debug)]
pub struct Lazy<B: BackendStorage> {
    backend: B,
    device: LazyDevice<B>,
}

impl<B: BackendStorage> Lazy<B> {
    pub fn new(backend: B) -> Self {
        let device = LazyDevice::new(backend.device().clone());
        Self { backend, device }
    }
}

impl<B: BackendStorage> From<B> for Lazy<B> {
    fn from(backend: B) -> Self {
        Self::new(backend)
    }
}

#[derive(Clone, Debug)]
pub struct LazyDevice<B: BackendStorage> {
    device: B::Device,
}

impl<B: BackendStorage> LazyDevice<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: BackendStorage> AsRef<B::Device> for LazyDevice<B> {
    fn as_ref(&self) -> &B::Device {
        &self.device
    }
}

impl<B: BackendStorage> BackendDevice<Lazy<B>> for LazyDevice<B> {
    fn new(ordinal: usize) -> Result<Self> {
        B::Device::new(ordinal).map(|device| Self { device })
    }

    fn location(&self) -> DeviceLocation {
        self.device.location()
    }

    fn is_cpu(&self) -> bool {
        self.device.is_cpu()
    }

    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Lazy<B>> {
        self.device.zeros(shape, dtype).map(Into::into)
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Lazy<B>> {
        self.device.alloc_uninit(shape, dtype).map(Into::into)
    }

    fn storage_from_slice<T: WithDType>(&self, data: &[T]) -> Result<Lazy<B>> {
        self.device.storage_from_slice(data).map(Into::into)
    }

    fn storage_from_cpu_storage(&self, cpu_storage: &CpuStorage) -> Result<Lazy<B>> {
        self.device
            .storage_from_cpu_storage(cpu_storage)
            .map(Into::into)
    }

    fn storage_from_cpu_storage_owned(&self, cpu_storage: CpuStorage) -> Result<Lazy<B>> {
        self.device
            .storage_from_cpu_storage_owned(cpu_storage)
            .map(Into::into)
    }

    fn storage<A: NdArray>(&self, array: A) -> Result<Lazy<B>> {
        self.device.storage(array).map(Into::into)
    }

    fn storage_owned<S: WithDType>(&self, data: Vec<S>) -> Result<Lazy<B>> {
        self.device.storage_owned(data).map(Into::into)
    }

    fn rand_uniform<T: FloatDType>(
        &self,
        shape: &Shape,
        dtype: DType,
        low: T,
        high: T,
    ) -> Result<Lazy<B>> {
        self.device
            .rand_uniform(shape, dtype, low, high)
            .map(Into::into)
    }

    fn rand_normal<T: FloatDType>(
        &self,
        shape: &Shape,
        dtype: DType,
        low: T,
        high: T,
    ) -> Result<Lazy<B>> {
        self.device
            .rand_normal(shape, dtype, low, high)
            .map(Into::into)
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        self.device.set_seed(seed)
    }

    fn synchronize(&self) -> Result<()> {
        self.device.synchronize()
    }
}

impl<B: BackendStorage> BackendStorage for Lazy<B> {
    type Device = LazyDevice<B>;

    fn try_clone(&self, layout: &Layout) -> Result<Self> {
        self.backend.try_clone(layout).map(Into::into)
    }

    fn dtype(&self) -> DType {
        self.backend.dtype()
    }

    fn device(&self) -> LazyDevice<B> {
        LazyDevice::<B>::new(self.backend.device().clone())
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        self.backend.to_cpu_storage()
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

    fn reduce_op(&self, _: crate::op::ReduceOp, _: &Layout, _: &[usize]) -> Result<Self> {
        todo!()
    }

    fn cmp(&self, _: crate::op::CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        todo!()
    }

    fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self> {
        todo!()
    }

    fn unary_impl<UnOp: crate::op::UnaryOpT>(&self, _: &Layout) -> Result<Self> {
        todo!()
    }

    fn binary_impl<BinOp: crate::op::BinaryOpT>(
        &self,
        _: &Self,
        _: &Layout,
        _: &Layout,
    ) -> Result<Self> {
        todo!()
    }

    fn where_cond(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
    ) -> crate::Result<Self> {
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

    fn apply_op1(&self, _l: &Layout, _c: &dyn crate::CustomOp1<Self>) -> Result<(Self, Shape)> {
        todo!()
    }

    fn apply_op2(
        &self,
        _l1: &Layout,
        _t2: &Self,
        _l2: &Layout,
        _c: &dyn crate::CustomOp2<Self>,
    ) -> Result<(Self, Shape)> {
        todo!()
    }

    fn apply_op3(
        &self,
        _l1: &Layout,
        _t2: &Self,
        _l2: &Layout,
        _t3: &Self,
        _l3: &Layout,
        _c: &dyn crate::CustomOp3<Self>,
    ) -> Result<(Self, Shape)> {
        todo!()
    }

    fn inplace_op1(&mut self, _l: &Layout, _c: &dyn crate::InplaceOp1) -> Result<()> {
        todo!()
    }

    fn inplace_op2(
        &mut self,
        _l1: &Layout,
        _t2: &Self,
        _l2: &Layout,
        _c: &dyn crate::InplaceOp2,
    ) -> Result<()> {
        todo!()
    }

    fn inplace_op3(
        &mut self,
        _l1: &Layout,
        _t2: &Self,
        _l2: &Layout,
        _t3: &Self,
        _l3: &Layout,
        _c: &dyn crate::InplaceOp3,
    ) -> Result<()> {
        todo!()
    }
}
