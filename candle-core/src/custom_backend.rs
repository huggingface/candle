use crate::{
    backend::{BackendDevice, BackendLocation, BackendStorage},
    bail, conv, op, CpuStorage, DType, Layout, Result, Shape,
};
use std::{
    any::{Any, TypeId},
    fmt::{self, Debug},
    hash::{Hash, Hasher},
};

pub trait AnyStorage: Any + Send + Sync + Debug {
    fn as_any(self: Box<Self>) -> Box<dyn Any>;
    fn as_any_ref(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn try_clone(&self, layout: &Layout) -> Result<Box<dyn AnyStorage>>;

    fn dtype(&self) -> DType;

    fn device(&self) -> Box<dyn AnyDevice>;

    fn to_cpu_storage(&self) -> Result<CpuStorage>;

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Box<dyn AnyStorage>>;

    fn powf(&self, layout: &Layout, e: f64) -> Result<Box<dyn AnyStorage>>;

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Box<dyn AnyStorage>>;

    fn reduce(
        &self,
        op: op::ReduceOp,
        layout: &Layout,
        sum_dims: &[usize],
    ) -> Result<Box<dyn AnyStorage>>;

    fn cmp(
        &self,
        op: op::CmpOp,
        rhs: &dyn AnyStorage,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Box<dyn AnyStorage>>;

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Box<dyn AnyStorage>>;

    fn unary_impl(&self, layout: &Layout, op: &dyn op::UnaryOpDyn) -> Result<Box<dyn AnyStorage>>;

    fn binary_impl(
        &self,
        rhs: &dyn AnyStorage,
        lhs_l: &Layout,
        rhs_l: &Layout,
        op: &dyn op::BinaryOpDyn,
    ) -> Result<Box<dyn AnyStorage>>;

    fn where_cond(
        &self,
        layout: &Layout,
        t: &dyn AnyStorage,
        t_l: &Layout,
        f: &dyn AnyStorage,
        f_l: &Layout,
    ) -> Result<Box<dyn AnyStorage>>;

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &dyn AnyStorage,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Box<dyn AnyStorage>>;

    fn conv_transpose1d(
        &self,
        _: &Layout,
        _: &dyn AnyStorage,
        _: &Layout,
        _: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Box<dyn AnyStorage>>;

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &dyn AnyStorage,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Box<dyn AnyStorage>>;

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &dyn AnyStorage,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Box<dyn AnyStorage>>;

    fn avg_pool2d(
        &self,
        l: &Layout,
        k: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Box<dyn AnyStorage>>;

    fn max_pool2d(
        &self,
        l: &Layout,
        k: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Box<dyn AnyStorage>>;

    fn upsample_nearest1d(&self, _: &Layout, _out_sz: usize) -> Result<Box<dyn AnyStorage>>;

    fn upsample_nearest2d(
        &self,
        l: &Layout,
        out_w: usize,
        out_h: usize,
    ) -> Result<Box<dyn AnyStorage>>;

    fn index_select(
        &self,
        ids: &dyn AnyStorage,
        l: &Layout,
        ids_l: &Layout,
        dim: usize,
    ) -> Result<Box<dyn AnyStorage>>;

    fn gather(
        &self,
        l: &Layout,
        ids: &dyn AnyStorage,
        ids_l: &Layout,
        dim: usize,
    ) -> Result<Box<dyn AnyStorage>>;

    fn scatter_add(
        &self,
        l: &Layout,
        ids: &dyn AnyStorage,
        ids_l: &Layout,
        src: &dyn AnyStorage,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Box<dyn AnyStorage>>;
    fn index_add(
        &self,
        l: &Layout,
        ids: &dyn AnyStorage,
        ids_l: &Layout,
        src: &dyn AnyStorage,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Box<dyn AnyStorage>>;
    fn matmul(
        &self,
        rhs: &dyn AnyStorage,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Box<dyn AnyStorage>>;

    fn copy_strided_src(
        &self,
        dst: &mut dyn AnyStorage,
        dst_offset: usize,
        src_l: &Layout,
    ) -> Result<()>;
}

pub trait AnyDevice: Any + Send + Sync + Debug {
    fn as_any(self: Box<Self>) -> Box<dyn Any>;
    fn as_any_ref(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn clone(&self) -> Box<dyn AnyDevice>;

    fn location(&self) -> Box<dyn AnyLocation>;

    fn same_device(&self, rhs: &dyn AnyDevice) -> bool;

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn AnyStorage>>;

    fn ones_impl(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn AnyStorage>>;

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Box<dyn AnyStorage>>;

    fn rand_uniform_f64(
        &self,
        shape: &Shape,
        dtype: DType,
        lo: f64,
        up: f64,
    ) -> Result<Box<dyn AnyStorage>>;

    fn rand_normal_f64(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<Box<dyn AnyStorage>>;

    fn set_seed(&self, seed: u64) -> Result<()>;
}

pub trait AnyLocation: Any + Send + Sync + Debug {
    fn as_any_ref(&self) -> &dyn Any;

    fn clone(&self) -> Box<dyn AnyLocation>;
    fn eq(&self, other: &dyn AnyLocation) -> bool;
    fn hash(&self, state: &mut dyn Hasher);
}

impl<L: BackendLocation> AnyLocation for L {
    fn as_any_ref(&self) -> &dyn Any {
        self
    }
    fn clone(&self) -> Box<dyn AnyLocation> {
        Box::new(self.clone())
    }
    fn eq(&self, other: &dyn AnyLocation) -> bool {
        match other.as_any_ref().downcast_ref::<L>() {
            Some(other) => self == other,
            None => false,
        }
    }
    fn hash(&self, state: &mut dyn Hasher) {
        self.hash(&mut DynHasher(state))
    }
}

struct DynHasher<'a>(&'a mut dyn Hasher);
impl<'a> Hasher for DynHasher<'a> {
    fn finish(&self) -> u64 {
        self.0.finish()
    }
    fn write(&mut self, bytes: &[u8]) {
        self.0.write(bytes)
    }
}

impl dyn AnyStorage {
    pub fn downcast<S: BackendStorage>(self: Box<Self>) -> core::result::Result<Box<S>, Box<Self>> {
        if self.type_id() == TypeId::of::<S>() {
            Ok(self
                .downcast()
                .expect("type ids are equal, but downcast failed"))
        } else {
            Err(self)
        }
    }
    pub fn downcast_ref<S: BackendStorage>(&self) -> Option<&S> {
        self.as_any_ref().downcast_ref()
    }
    pub fn downcast_mut<S: BackendStorage>(&mut self) -> Option<&mut S> {
        self.as_any_mut().downcast_mut()
    }

    fn downcast_ref_or_error<S: BackendStorage>(&self, lhs: &S, op: &'static str) -> Result<&S> {
        match self.downcast_ref::<S>() {
            Some(rhs) => Ok(rhs),
            None => bail!(
                "custom device mismatch in {}, lhs: {:?}, rhs: {:?}",
                op,
                BackendDevice::location(&S::device(lhs)),
                self.device().location()
            ),
        }
    }
    fn downcast_mut_or_error<S: BackendStorage>(
        &mut self,
        lhs: &S,
        op: &'static str,
    ) -> Result<&mut S> {
        let location = self.device().location();
        match self.downcast_mut::<S>() {
            Some(rhs) => Ok(rhs),
            None => bail!(
                "custom device mismatch in {}, lhs: {:?}, rhs: {:?}",
                op,
                BackendDevice::location(&S::device(lhs)),
                location
            ),
        }
    }
}

impl dyn AnyDevice {
    pub fn downcast<D: BackendDevice>(self: Box<Self>) -> core::result::Result<Box<D>, Box<Self>> {
        if self.type_id() == TypeId::of::<D>() {
            Ok(self
                .downcast()
                .expect("type ids are equal, but downcast failed"))
        } else {
            Err(self)
        }
    }
    pub fn downcast_ref<D: BackendDevice>(&self) -> Option<&D> {
        self.as_any_ref().downcast_ref()
    }
    pub fn downcast_mut<D: BackendDevice>(&mut self) -> Option<&mut D> {
        self.as_any_mut().downcast_mut()
    }
}

impl<S: BackendStorage> op::UnaryVisitor<Result<S>> for (&S, &Layout) {
    fn visit<B: op::UnaryOpPub>(self, _op: B) -> Result<S> {
        S::unary_impl::<B>(self.0, self.1)
    }
}
impl<S: BackendStorage> op::BinaryVisitor<Result<S>> for (&S, &S, &Layout, &Layout) {
    fn visit<B: op::BinaryOpPub>(self, _op: B) -> Result<S> {
        S::binary_impl::<B>(self.0, self.1, self.2, self.3)
    }
}

impl<S: BackendStorage> AnyStorage for S {
    fn as_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
    fn as_any_ref(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn try_clone(&self, layout: &Layout) -> Result<Box<dyn AnyStorage>> {
        Ok(Box::new(S::try_clone(self, layout)?))
    }

    fn dtype(&self) -> DType {
        S::dtype(self)
    }

    fn device(&self) -> Box<dyn AnyDevice> {
        Box::new(S::device(self))
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        S::to_cpu_storage(self)
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Box<dyn AnyStorage>> {
        Ok(Box::new(S::affine(self, layout, mul, add)?))
    }

    fn powf(&self, layout: &Layout, e: f64) -> Result<Box<dyn AnyStorage>> {
        Ok(Box::new(S::powf(self, layout, e)?))
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Box<dyn AnyStorage>> {
        Ok(Box::new(S::elu(self, layout, alpha)?))
    }

    fn reduce(
        &self,
        op: op::ReduceOp,
        layout: &Layout,
        sum_dims: &[usize],
    ) -> Result<Box<dyn AnyStorage>> {
        Ok(Box::new(S::reduce(self, op, layout, sum_dims)?))
    }

    fn cmp(
        &self,
        op: op::CmpOp,
        rhs: &dyn AnyStorage,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Box<dyn AnyStorage>> {
        Ok(Box::new(S::cmp(
            self,
            op,
            rhs.downcast_ref_or_error(self, "cmp")?,
            lhs_l,
            rhs_l,
        )?))
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Box<dyn AnyStorage>> {
        Ok(Box::new(S::to_dtype(self, layout, dtype)?))
    }

    fn unary_impl(&self, layout: &Layout, op: &dyn op::UnaryOpDyn) -> Result<Box<dyn AnyStorage>> {
        Ok(Box::new(op.visit((self, layout))?))
    }

    fn binary_impl(
        &self,
        rhs: &dyn AnyStorage,
        lhs_l: &Layout,
        rhs_l: &Layout,
        op: &dyn op::BinaryOpDyn,
    ) -> Result<Box<dyn AnyStorage>> {
        Ok(Box::new(op.visit((
            self,
            rhs.downcast_ref_or_error(self, op.name())?,
            lhs_l,
            rhs_l,
        ))?))
    }

    fn where_cond(
        &self,
        layout: &Layout,
        t: &dyn AnyStorage,
        t_l: &Layout,
        f: &dyn AnyStorage,
        f_l: &Layout,
    ) -> Result<Box<dyn AnyStorage>> {
        Ok(Box::new(S::where_cond(
            self,
            layout,
            t.downcast_ref_or_error(self, "where_cond")?,
            t_l,
            f.downcast_ref_or_error(self, "where_cond")?,
            f_l,
        )?))
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &dyn AnyStorage,
        kernel_l: &Layout,
        params: &conv::ParamsConv1D,
    ) -> Result<Box<dyn AnyStorage>> {
        Ok(Box::new(S::conv1d(
            self,
            l,
            kernel.downcast_ref_or_error(self, "conv1d")?,
            kernel_l,
            params,
        )?))
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &dyn AnyStorage,
        kernel_l: &Layout,
        params: &conv::ParamsConvTranspose1D,
    ) -> Result<Box<dyn AnyStorage>> {
        Ok(Box::new(S::conv_transpose1d(
            self,
            l,
            kernel.downcast_ref_or_error(self, "conv1d")?,
            kernel_l,
            params,
        )?))
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &dyn AnyStorage,
        kernel_l: &Layout,
        params: &conv::ParamsConv2D,
    ) -> Result<Box<dyn AnyStorage>> {
        Ok(Box::new(S::conv2d(
            self,
            l,
            kernel.downcast_ref_or_error(self, "conv1d")?,
            kernel_l,
            params,
        )?))
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &dyn AnyStorage,
        kernel_l: &Layout,
        params: &conv::ParamsConvTranspose2D,
    ) -> Result<Box<dyn AnyStorage>> {
        Ok(Box::new(S::conv_transpose2d(
            self,
            l,
            kernel.downcast_ref_or_error(self, "conv1d")?,
            kernel_l,
            params,
        )?))
    }

    fn avg_pool2d(
        &self,
        l: &Layout,
        k: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Box<dyn AnyStorage>> {
        Ok(Box::new(S::avg_pool2d(self, l, k, stride)?))
    }

    fn max_pool2d(
        &self,
        l: &Layout,
        k: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Box<dyn AnyStorage>> {
        Ok(Box::new(S::max_pool2d(self, l, k, stride)?))
    }

    fn upsample_nearest1d(&self, l: &Layout, out_sz: usize) -> Result<Box<dyn AnyStorage>> {
        Ok(Box::new(S::upsample_nearest1d(self, l, out_sz)?))
    }

    fn upsample_nearest2d(
        &self,
        l: &Layout,
        out_w: usize,
        out_h: usize,
    ) -> Result<Box<dyn AnyStorage>> {
        Ok(Box::new(S::upsample_nearest2d(self, l, out_w, out_h)?))
    }

    fn index_select(
        &self,
        ids: &dyn AnyStorage,
        l: &Layout,
        ids_l: &Layout,
        dim: usize,
    ) -> Result<Box<dyn AnyStorage>> {
        Ok(Box::new(S::index_select(
            self,
            ids.downcast_ref_or_error(self, "index_select")?,
            l,
            ids_l,
            dim,
        )?))
    }

    fn gather(
        &self,
        l: &Layout,
        ids: &dyn AnyStorage,
        ids_l: &Layout,
        dim: usize,
    ) -> Result<Box<dyn AnyStorage>> {
        Ok(Box::new(S::gather(
            self,
            l,
            ids.downcast_ref_or_error(self, "gather")?,
            ids_l,
            dim,
        )?))
    }

    fn scatter_add(
        &self,
        l: &Layout,
        ids: &dyn AnyStorage,
        ids_l: &Layout,
        src: &dyn AnyStorage,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Box<dyn AnyStorage>> {
        Ok(Box::new(S::scatter_add(
            self,
            l,
            ids.downcast_ref_or_error(self, "scatter_add")?,
            ids_l,
            src.downcast_ref_or_error(self, "scatter_add")?,
            src_l,
            dim,
        )?))
    }

    fn index_add(
        &self,
        l: &Layout,
        ids: &dyn AnyStorage,
        ids_l: &Layout,
        src: &dyn AnyStorage,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Box<dyn AnyStorage>> {
        Ok(Box::new(S::index_add(
            self,
            l,
            ids.downcast_ref_or_error(self, "index_add")?,
            ids_l,
            src.downcast_ref_or_error(self, "index_add")?,
            src_l,
            dim,
        )?))
    }

    fn matmul(
        &self,
        rhs: &dyn AnyStorage,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Box<dyn AnyStorage>> {
        Ok(Box::new(S::matmul(
            self,
            rhs.downcast_ref_or_error(self, "matmul")?,
            bmnk,
            lhs_l,
            rhs_l,
        )?))
    }

    fn copy_strided_src(
        &self,
        dst: &mut dyn AnyStorage,
        dst_offset: usize,
        src_l: &Layout,
    ) -> Result<()> {
        S::copy_strided_src(
            self,
            dst.downcast_mut_or_error(self, "copy_strided_src")?,
            dst_offset,
            src_l,
        )
    }
}

impl<D: BackendDevice> AnyDevice for D {
    fn as_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
    fn as_any_ref(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn clone(&self) -> Box<dyn AnyDevice> {
        Box::new(self.clone())
    }

    fn location(&self) -> Box<dyn AnyLocation> {
        Box::new(D::location(self))
    }

    fn same_device(&self, rhs: &dyn AnyDevice) -> bool {
        match rhs.downcast_ref::<D>() {
            Some(rhs) => D::same_device(self, rhs),
            None => false,
        }
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn AnyStorage>> {
        Ok(Box::new(D::zeros_impl(self, shape, dtype)?))
    }

    fn ones_impl(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn AnyStorage>> {
        Ok(Box::new(D::ones_impl(self, shape, dtype)?))
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Box<dyn AnyStorage>> {
        Ok(Box::new(D::storage_from_cpu_storage(self, storage)?))
    }

    fn rand_uniform_f64(
        &self,
        shape: &Shape,
        dtype: DType,
        lo: f64,
        up: f64,
    ) -> Result<Box<dyn AnyStorage>> {
        Ok(Box::new(D::rand_uniform_f64(self, shape, dtype, lo, up)?))
    }

    fn rand_normal_f64(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<Box<dyn AnyStorage>> {
        Ok(Box::new(D::rand_normal_f64(self, shape, dtype, mean, std)?))
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        D::set_seed(self, seed)
    }
}

pub struct CustomStorage(Box<dyn AnyStorage>);

impl Debug for CustomStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl CustomStorage {
    pub fn new<S: BackendStorage>(storage: S) -> Self {
        Self(Box::new(storage))
    }

    pub fn downcast<S: BackendStorage>(self) -> Option<S> {
        Some(*self.0.downcast::<S>().ok()?)
    }
    pub fn downcast_ref<S: BackendStorage>(&self) -> Option<&S> {
        self.0.downcast_ref::<S>()
    }
    pub fn downcast_mut<S: BackendStorage>(&mut self) -> Option<&mut S> {
        self.0.downcast_mut::<S>()
    }
}

impl BackendStorage for CustomStorage {
    type Device = CustomDevice;

    fn try_clone(&self, layout: &Layout) -> Result<Self> {
        Ok(Self(self.0.try_clone(layout)?))
    }

    fn dtype(&self) -> DType {
        self.0.dtype()
    }

    fn device(&self) -> Self::Device {
        CustomDevice(self.0.device())
    }

    // Maybe this should return a Cow instead so that no copy is done on the cpu case.
    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        self.0.to_cpu_storage()
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        Ok(Self(self.0.affine(layout, mul, add)?))
    }

    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        Ok(Self(self.0.powf(layout, e)?))
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        Ok(Self(self.0.elu(layout, alpha)?))
    }

    fn reduce(&self, op: op::ReduceOp, layout: &Layout, sum_dims: &[usize]) -> Result<Self> {
        Ok(Self(self.0.reduce(op, layout, sum_dims)?))
    }

    fn cmp(&self, op: op::CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        Ok(Self(self.0.cmp(op, rhs.0.as_ref(), lhs_l, rhs_l)?))
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        Ok(Self(self.0.to_dtype(layout, dtype)?))
    }

    fn unary_impl<U: op::UnaryOpPub>(&self, layout: &Layout) -> Result<Self> {
        Ok(Self(
            self.0.unary_impl(layout, &U::V as &dyn op::UnaryOpDyn)?,
        ))
    }

    fn binary_impl<B: op::BinaryOpPub>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        Ok(Self(self.0.binary_impl(
            rhs.0.as_ref(),
            lhs_l,
            rhs_l,
            &B::V as &dyn op::BinaryOpDyn,
        )?))
    }

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        Ok(Self(self.0.where_cond(
            layout,
            t.0.as_ref(),
            t_l,
            f.0.as_ref(),
            f_l,
        )?))
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        Ok(Self(self.0.conv1d(
            l,
            kernel.0.as_ref(),
            kernel_l,
            params,
        )?))
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        Ok(Self(self.0.conv_transpose1d(
            l,
            kernel.0.as_ref(),
            kernel_l,
            params,
        )?))
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        Ok(Self(self.0.conv2d(
            l,
            kernel.0.as_ref(),
            kernel_l,
            params,
        )?))
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        Ok(Self(self.0.conv_transpose2d(
            l,
            kernel.0.as_ref(),
            kernel_l,
            params,
        )?))
    }

    fn avg_pool2d(&self, l: &Layout, k: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        Ok(Self(self.0.avg_pool2d(l, k, stride)?))
    }

    fn max_pool2d(&self, l: &Layout, k: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        Ok(Self(self.0.max_pool2d(l, k, stride)?))
    }

    fn upsample_nearest1d(&self, l: &Layout, out_sz: usize) -> Result<Self> {
        Ok(Self(self.0.upsample_nearest1d(l, out_sz)?))
    }

    fn upsample_nearest2d(&self, l: &Layout, out_w: usize, out_h: usize) -> Result<Self> {
        Ok(Self(self.0.upsample_nearest2d(l, out_w, out_h)?))
    }

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        Ok(Self(self.0.index_select(ids.0.as_ref(), l, ids_l, dim)?))
    }

    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        Ok(Self(self.0.gather(l, ids.0.as_ref(), ids_l, dim)?))
    }

    fn scatter_add(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        Ok(Self(self.0.scatter_add(
            l,
            ids.0.as_ref(),
            ids_l,
            src.0.as_ref(),
            src_l,
            dim,
        )?))
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
        Ok(Self(self.0.index_add(
            l,
            ids.0.as_ref(),
            ids_l,
            src.0.as_ref(),
            src_l,
            dim,
        )?))
    }

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        Ok(Self(self.0.matmul(rhs.0.as_ref(), bmnk, lhs_l, rhs_l)?))
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        self.0.copy_strided_src(dst.0.as_mut(), dst_offset, src_l)
    }
}

#[derive(Debug)]
pub struct CustomDevice(Box<dyn AnyDevice>);

impl Clone for CustomDevice {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl CustomDevice {
    pub fn new<D: BackendDevice>(device: D) -> Self {
        Self(Box::new(device))
    }

    pub fn downcast<D: BackendDevice>(self) -> Option<D> {
        Some(*self.0.downcast::<D>().ok()?)
    }
    pub fn downcast_ref<D: BackendDevice>(&self) -> Option<&D> {
        self.0.downcast_ref::<D>()
    }
    pub fn downcast_mut<D: BackendDevice>(&mut self) -> Option<&mut D> {
        self.0.downcast_mut::<D>()
    }
}

impl BackendDevice for CustomDevice {
    type Storage = CustomStorage;
    type Location = CustomLocation;

    fn location(&self) -> Self::Location {
        CustomLocation(self.0.location())
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.0.same_device(rhs.0.as_ref())
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        Ok(CustomStorage(self.0.zeros_impl(shape, dtype)?))
    }

    fn ones_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        Ok(CustomStorage(self.0.ones_impl(shape, dtype)?))
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        Ok(CustomStorage(self.0.storage_from_cpu_storage(storage)?))
    }

    fn rand_uniform_f64(
        &self,
        shape: &Shape,
        dtype: DType,
        lo: f64,
        up: f64,
    ) -> Result<Self::Storage> {
        Ok(CustomStorage(
            self.0.rand_uniform_f64(shape, dtype, lo, up)?,
        ))
    }

    fn rand_normal_f64(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<Self::Storage> {
        Ok(CustomStorage(
            self.0.rand_normal_f64(shape, dtype, mean, std)?,
        ))
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        self.0.set_seed(seed)
    }
}

#[derive(Debug)]
pub struct CustomLocation(Box<dyn AnyLocation>);

impl Clone for CustomLocation {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl PartialEq for CustomLocation {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(other.0.as_ref())
    }
}
impl Eq for CustomLocation {}

impl Hash for CustomLocation {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}
