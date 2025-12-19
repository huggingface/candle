use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape};

#[derive(Debug, Clone)]
pub struct LazyStorage {
    shape: Shape,
    dtype: DType,
    graph: Graph
}

impl LazyStorage {
    fn new(shape: Shape, dtype: DType) -> LazyStorage {
        LazyStorage {
            shape,
            dtype,
            graph: Default::default()
        }
    }

    fn graph_mut(&mut self) -> &mut Graph {
        &mut self.graph
    }
}


// Very much WIP.
// We want to be able to run "deferred eager" before we start working on actual lazy optimizations.
#[derive(Debug, Clone)]
enum LazyOp {
    ToCpu,
    Affine(Layout, f64, f64),
    Powf(Layout, f64),
    Elu(Layout, f64),
    Reduce(ReduceOp, Layout, Vec<usize>),
    Cmp(CmpOp, LazyStorage, Layout, Layout),
    ToDType(Layout, DType),
    Unary(Layout, &'static str),
    Binary(Layout, LazyStorage, Layout, &'static str),
    WhereCond(Layout, LazyStorage, Layout, LazyStorage, Layout),
    Conv1D(Layout, LazyStorage, Layout, crate::conv::ParamsConv1D),
    ConvTranspose1D(Layout, LazyStorage, Layout, crate::conv::ParamsConvTranspose1D),
    Conv2D(Layout, LazyStorage, Layout, crate::conv::ParamsConv2D),
    ConvTranspose2D(Layout, LazyStorage, Layout, crate::conv::ParamsConvTranspose2D),
    AvgPool2D(Layout, (usize, usize), (usize, usize)),
    MaxPool2D(Layout, (usize, usize), (usize, usize)),
    UpsampleNearest1D(Layout, usize),
    UpsampleNearest2D(Layout, usize, usize),
    Gather(Layout, LazyStorage, Layout, usize),
    ScatterSet(Layout, LazyStorage, Layout, LazyStorage, Layout, usize),
    ScatterAddSet(Layout, LazyStorage, Layout, LazyStorage, Layout, usize),
    IndexSelect(Layout, LazyStorage, Layout, usize),
    IndexAdd(Layout, LazyStorage, Layout, LazyStorage, Layout, usize),
    Matmul(LazyStorage, (usize, usize, usize, usize), Layout, Layout),
    CopyStridedSrc(LazyStorage, usize, Layout),
    Copy2D(LazyStorage, usize, usize, usize, usize, usize, usize),
    ConstSet(crate::scalar::Scalar, Layout)
}

#[derive(Debug, Clone, Default)]
struct Graph {
    operations: Vec<LazyOp>
}

impl Graph {
    fn push_node(&mut self, op: LazyOp) {
        self.operations.push(op)
    }
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

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        let mut next = self.clone();
        next.graph_mut().push_node(LazyOp::ToCpu);
        todo!()
    }

    fn affine(&self, l: &Layout, mul: f64, add: f64) -> Result<Self> {
        let op = LazyOp::Affine(l.clone(), mul, add);
        let mut next = self.clone();
        next.graph_mut().push_node(op);
        Ok(next)
    }

    fn powf(&self, l: &Layout, pow: f64) -> Result<Self> {
        let op = LazyOp::Powf(l.clone(), pow);
        let mut next = self.clone();
        next.graph_mut().push_node(op);
        Ok(next)
    }

    fn elu(&self, l: &Layout, alpha: f64) -> Result<Self> {
        let op = LazyOp::Elu(l.clone(), alpha);
        let mut next = self.clone();
        next.graph_mut().push_node(op);
        Ok(next)
    }

    fn reduce_op(&self, reduce: ReduceOp, l: &Layout, dims: &[usize]) -> Result<Self> {
        let op = LazyOp::Reduce(reduce, l.clone(), dims.to_vec());
        let mut next = self.clone();
        next.graph_mut().push_node(op);
        Ok(next)
    }

    fn cmp(&self, cmp_op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        let op = LazyOp::Cmp(cmp_op, rhs.clone(), lhs_l.clone(), rhs_l.clone());
        let mut next = self.clone();
        next.graph_mut().push_node(op);
        Ok(next)
    }

    fn to_dtype(&self, l: &Layout, dtype: DType) -> Result<Self> {
        let op = LazyOp::ToDType(l.clone(), dtype);
        let mut next = self.clone();
        next.graph_mut().push_node(op);
        Ok(next)
    }

    fn unary_impl<B: UnaryOpT>(&self, l: &Layout) -> Result<Self> {
        let op = LazyOp::Unary(l.clone(), B::KERNEL);
        let mut next = self.clone();
        next.graph_mut().push_node(op);
        Ok(next)
    }

    fn binary_impl<B: BinaryOpT>(&self, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        let op = LazyOp::Binary(lhs_l.clone(), rhs.clone(), rhs_l.clone(), B::KERNEL);
        let mut next = self.clone();
        next.graph_mut().push_node(op);
        Ok(next)
    }

    fn where_cond(&self, l: &Layout, t: &Self, t_l: &Layout, f: &Self, f_l: &Layout) -> Result<Self> {
        let op = LazyOp::WhereCond(l.clone(), t.clone(), t_l.clone(), f.clone(), f_l.clone());
        let mut next = self.clone();
        next.graph_mut().push_node(op);
        Ok(next)
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        let op = LazyOp::Conv1D(l.clone(), kernel.clone(), kernel_l.clone(), params.clone());
        let mut next = self.clone();
        next.graph_mut().push_node(op);
        Ok(next)
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        let op = LazyOp::ConvTranspose1D(l.clone(), kernel.clone(), kernel_l.clone(), params.clone());
        let mut next = self.clone();
        next.graph_mut().push_node(op);
        Ok(next)
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        let op = LazyOp::Conv2D(l.clone(), kernel.clone(), kernel_l.clone(), params.clone());
        let mut next = self.clone();
        next.graph_mut().push_node(op);
        Ok(next)
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        let op = LazyOp::ConvTranspose2D(l.clone(), kernel.clone(), kernel_l.clone(), params.clone());
        let mut next = self.clone();
        next.graph_mut().push_node(op);
        Ok(next)
    }

    fn avg_pool2d(&self, l: &Layout, (w_k, h_k): (usize, usize), (w_stride, h_stride): (usize, usize)) -> Result<Self> {
        let op = LazyOp::AvgPool2D(l.clone(), (w_k, h_k), (w_stride, h_stride));
        let mut next = self.clone();
        next.graph_mut().push_node(op);
        Ok(next)
    }
    fn max_pool2d(&self, l: &Layout, (w_k, h_k): (usize, usize), (w_stride, h_stride): (usize, usize)) -> Result<Self> {
        let op = LazyOp::MaxPool2D(l.clone(), (w_k, h_k), (w_stride, h_stride));
        let mut next = self.clone();
        next.graph_mut().push_node(op);
        Ok(next)
    }

    fn upsample_nearest1d(&self, l: &Layout, sz: usize) -> Result<Self> {
        let op = LazyOp::UpsampleNearest1D(l.clone(), sz);
        let mut next = self.clone();
        next.graph_mut().push_node(op);
        Ok(next)
    }

    fn upsample_nearest2d(&self, l: &Layout, out_w: usize, out_h: usize) -> Result<Self> {
        let op = LazyOp::UpsampleNearest2D(l.clone(), out_w, out_h);
        let mut next = self.clone();
        next.graph_mut().push_node(op);
        Ok(next)
    }

    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        let op = LazyOp::Gather(l.clone(), ids.clone(), ids_l.clone(), dim);
        let mut next = self.clone();
        next.graph_mut().push_node(op);
        Ok(next)
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
        let op = LazyOp::ScatterSet(l.clone(), ids.clone(), ids_l.clone(), src.clone(), src_l.clone(), dim);
        self.graph_mut().push_node(op);
        Ok(())
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
        let op = LazyOp::ScatterAddSet(l.clone(), ids.clone(), ids_l.clone(), src.clone(), src_l.clone(), dim);
        self.graph_mut().push_node(op);
        Ok(())
    }

    fn index_select(&self, ids: &Self, src_l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        let op = LazyOp::IndexSelect(src_l.clone(), ids.clone(), ids_l.clone(), dim);
        let mut next = self.clone();
        next.graph_mut().push_node(op);
        Ok(next)
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
        let op = LazyOp::IndexAdd(l.clone(), ids.clone(), ids_l.clone(), src.clone(), src_l.clone(), dim);
        let mut next = self.clone();
        next.graph_mut().push_node(op);
        Ok(next)
    }

    fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let op = LazyOp::Matmul(rhs.clone(), (b, m, n, k), lhs_l.clone(), rhs_l.clone());
        let mut next = self.clone();
        next.graph_mut().push_node(op);
        Ok(next)
    }

    fn copy_strided_src(&self, rhs: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        let op = LazyOp::CopyStridedSrc(self.clone(), dst_offset, src_l.clone());
        rhs.graph_mut().push_node(op);
        Ok(())
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
        let op = LazyOp::Copy2D(self.clone(), d1, d2, src_s, dst_s, src_o, dst_o);
        dst.graph_mut().push_node(op);
        Ok(())
    }

    fn const_set(&mut self, s: crate::scalar::Scalar, l: &Layout) -> Result<()> {
        let op = LazyOp::ConstSet(s, l.clone());
        self.graph_mut().push_node(op);
        Ok(())
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
        Ok(LazyStorage::new(shape.clone(), dtype))
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let mut storage = unsafe { self.alloc_uninit(shape, dtype)? };
        storage.const_set(crate::scalar::Scalar::zero(dtype), &Layout::contiguous(shape.clone()))?;
        Ok(storage)
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
