use crate::{
    op::{BackpropOp, Op},
    BackendDevice, BackendStorage, CpuStorage, DType, DeviceLocation, FloatDType, Layout, NdArray,
    Result, Shape, WithDType,
};
use core::ops::Deref;
use std::sync::{Arc, RwLock};

#[derive(Clone, Debug)]
pub struct Bwd<B: BackendStorage> {
    backend: B,
    op: BackpropOp<Bwd<B>>,
    is_variable: bool,
}

impl<B: BackendStorage> Bwd<B> {
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            op: BackpropOp::none(),
            is_variable: false,
        }
    }
}

impl<B: BackendStorage> From<B> for Bwd<B> {
    fn from(backend: B) -> Self {
        Self::new(backend)
    }
}

impl<B: BackendStorage> AsRef<B> for Bwd<B> {
    fn as_ref(&self) -> &B {
        &self.backend
    }
}

#[derive(Clone, Debug)]
pub struct BwdDevice<B: BackendStorage> {
    device: B::Device,
}

impl<B: BackendStorage> From<&B::Device> for BwdDevice<B> {
    fn from(device: &B::Device) -> Self {
        Self {
            device: device.clone(),
        }
    }
}

impl<B: BackendStorage> BwdDevice<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: BackendStorage> AsRef<B::Device> for BwdDevice<B> {
    fn as_ref(&self) -> &B::Device {
        &self.device
    }
}

impl<B: BackendStorage> BackendDevice<Bwd<B>> for BwdDevice<B> {
    fn new(ordinal: usize) -> Result<Self> {
        B::Device::new(ordinal).map(|device| Self { device })
    }

    fn location(&self) -> DeviceLocation {
        self.device.location()
    }

    fn is_cpu(&self) -> bool {
        self.device.is_cpu()
    }

    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Bwd<B>> {
        self.device.zeros(shape, dtype).map(Into::into)
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Bwd<B>> {
        self.device.alloc_uninit(shape, dtype).map(Into::into)
    }

    fn storage_from_slice<T: WithDType>(&self, data: &[T]) -> Result<Bwd<B>> {
        self.device.storage_from_slice(data).map(Into::into)
    }

    fn storage_from_cpu_storage(&self, cpu_storage: &CpuStorage) -> Result<Bwd<B>> {
        self.device
            .storage_from_cpu_storage(cpu_storage)
            .map(Into::into)
    }

    fn storage_from_cpu_storage_owned(&self, cpu_storage: CpuStorage) -> Result<Bwd<B>> {
        self.device
            .storage_from_cpu_storage_owned(cpu_storage)
            .map(Into::into)
    }

    fn storage<A: NdArray>(&self, array: A) -> Result<Bwd<B>> {
        self.device.storage(array).map(Into::into)
    }

    fn storage_owned<S: WithDType>(&self, data: Vec<S>) -> Result<Bwd<B>> {
        self.device.storage_owned(data).map(Into::into)
    }

    fn rand_uniform<T: FloatDType>(
        &self,
        shape: &Shape,
        dtype: DType,
        low: T,
        high: T,
    ) -> Result<Bwd<B>> {
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
    ) -> Result<Bwd<B>> {
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

impl<B: BackendStorage> BackendStorage for Bwd<B> {
    type Device = BwdDevice<B>;
    type Storage = B;

    fn backprop_op(&self) -> Option<Op<Self>> {
        self.op.deref().clone()
    }

    fn is_variable(&self) -> bool {
        self.is_variable
    }

    fn to_tensor(
        storage: Arc<RwLock<Self>>,
        layout: Layout,
        op: crate::op::BackpropOp<Self>,
        is_variable: bool,
    ) -> crate::Tensor<Self> {
        let (backend, device) = {
            let guard = storage.read().unwrap();
            (guard.backend.clone(), guard.device())
        };
        let backprop = Bwd {
            backend,
            op,
            is_variable,
        };
        crate::Tensor::create(Arc::new(RwLock::new(backprop)), layout, is_variable)
    }

    fn try_clone(&self, layout: &Layout) -> Result<Self> {
        self.backend.try_clone(layout).map(Into::into)
    }

    fn dtype(&self) -> DType {
        self.backend.dtype()
    }

    fn device(&self) -> BwdDevice<B> {
        BwdDevice::<B>::new(self.backend.device())
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        self.backend.to_cpu_storage()
    }

    fn affine(&self, layout: &Layout, a: f64, b: f64) -> Result<Self> {
        self.backend.affine(layout, a, b).map(Into::into)
    }

    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        self.backend.powf(layout, e).map(Into::into)
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        self.backend.elu(layout, alpha).map(Into::into)
    }

    fn reduce_op(
        &self,
        op: crate::op::ReduceOp,
        layout: &Layout,
        reduce_dims: &[usize],
    ) -> Result<Self> {
        self.backend
            .reduce_op(op, layout, reduce_dims)
            .map(Into::into)
    }

    fn cmp(
        &self,
        op: crate::op::CmpOp,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        self.backend
            .cmp(op, rhs.as_ref(), lhs_l, rhs_l)
            .map(Into::into)
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        self.backend.to_dtype(layout, dtype).map(Into::into)
    }

    fn unary_impl<U: crate::op::UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        self.backend.unary_impl::<U>(layout).map(Into::into)
    }

    fn binary_impl<BinOp: crate::op::BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        self.backend
            .binary_impl::<BinOp>(rhs.as_ref(), lhs_l, rhs_l)
            .map(Into::into)
    }

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        self.backend
            .where_cond(layout, t.as_ref(), t_l, f.as_ref(), f_l)
            .map(Into::into)
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        self.backend
            .conv1d(l, kernel.as_ref(), kernel_l, params)
            .map(Into::into)
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        self.backend
            .conv_transpose1d(l, kernel.as_ref(), kernel_l, params)
            .map(Into::into)
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        self.backend
            .conv2d(l, kernel.as_ref(), kernel_l, params)
            .map(Into::into)
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        self.backend
            .conv_transpose2d(l, kernel.as_ref(), kernel_l, params)
            .map(Into::into)
    }

    fn avg_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        self.backend
            .avg_pool2d(layout, kernel_size, stride)
            .map(Into::into)
    }

    fn max_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        self.backend
            .max_pool2d(layout, kernel_size, stride)
            .map(Into::into)
    }

    fn upsample_nearest1d(&self, layout: &Layout, sz: usize) -> Result<Self> {
        self.backend.upsample_nearest1d(layout, sz).map(Into::into)
    }

    fn upsample_nearest2d(&self, layout: &Layout, h: usize, w: usize) -> Result<Self> {
        self.backend
            .upsample_nearest2d(layout, h, w)
            .map(Into::into)
    }

    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        self.backend
            .gather(l, ids.as_ref(), ids_l, dim)
            .map(Into::into)
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
        self.backend
            .scatter_set(l, ids.as_ref(), ids_l, src.as_ref(), src_l, dim)
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
        self.backend
            .scatter_add_set(l, ids.as_ref(), ids_l, src.as_ref(), src_l, dim)
    }

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        self.backend
            .index_select(ids.as_ref(), l, ids_l, dim)
            .map(Into::into)
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
        self.backend
            .index_add(l, ids.as_ref(), ids_l, src.as_ref(), src_l, dim)
            .map(Into::into)
    }

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        self.backend
            .matmul(rhs.as_ref(), bmnk, lhs_l, rhs_l)
            .map(Into::into)
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        self.backend
            .copy_strided_src(&mut dst.backend, dst_offset, src_l)
    }

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
        self.backend
            .copy2d(&mut dst.backend, d1, d2, src_s, dst_s, src_o, dst_o)
    }

    fn const_set(&mut self, s: crate::scalar::Scalar, l: &Layout) -> Result<()> {
        self.backend.const_set(s, l)
    }

    fn apply_op1(
        &self,
        _l: &Layout,
        c: &dyn crate::CustomOp1<Self::Storage>,
    ) -> Result<(Self, Shape)> {
        todo!("apply_op1: {}", c.name());
    }

    fn apply_op2(
        &self,
        _l1: &Layout,
        _t2: &Self,
        _l2: &Layout,
        c: &dyn crate::CustomOp2<Self::Storage>,
    ) -> Result<(Self, Shape)> {
        todo!("apply_op2: {}", c.name());
    }

    fn apply_op3(
        &self,
        _l1: &Layout,
        _t2: &Self,
        _l2: &Layout,
        _t3: &Self,
        _l3: &Layout,
        c: &dyn crate::CustomOp3<Self::Storage>,
    ) -> Result<(Self, Shape)> {
        todo!("apply_op3: {}", c.name());
    }

    fn inplace_op1(&mut self, l: &Layout, c: &dyn crate::InplaceOp1) -> Result<()> {
        self.backend.inplace_op1(l, c)
    }

    fn inplace_op2(
        &mut self,
        l1: &Layout,
        t2: &Self,
        l2: &Layout,
        c: &dyn crate::InplaceOp2,
    ) -> Result<()> {
        self.backend.inplace_op2(l1, t2.as_ref(), l2, c)
    }

    fn inplace_op3(
        &mut self,
        l1: &Layout,
        t2: &Self,
        l2: &Layout,
        t3: &Self,
        l3: &Layout,
        c: &dyn crate::InplaceOp3,
    ) -> Result<()> {
        self.backend
            .inplace_op3(l1, t2.as_ref(), l2, t3.as_ref(), l3, c)
    }
}
