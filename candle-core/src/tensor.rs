use crate::{op::Op, storage::Storage, DType, Device, Error, Layout, Result, Shape};
use std::sync::Arc;

/// Unique identifier for tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorId(usize);

impl TensorId {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

pub struct Tensor_ {
    id: TensorId,
    storage: Arc<Storage>,
    layout: Layout,
    op: Option<Op>,
    is_variable: bool,
}

impl AsRef<Tensor> for Tensor {
    fn as_ref(&self) -> &Tensor {
        self
    }
}

// Tensors are refcounted so that cloning is cheap when building the op graph.
// Storages are also refcounted independently so that its possible to avoid
// copying the storage for operations that only modify the shape or stride.
#[derive(Clone)]
pub struct Tensor(Arc<Tensor_>);

impl std::ops::Deref for Tensor {
    type Target = Tensor_;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

macro_rules! unary_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name(&self) -> Result<Self> {
            let shape = self.shape();
            let storage = self
                .storage
                .unary_impl::<crate::op::$op_name>(self.layout())?;
            let op = if self.track_op() {
                Some(Op::$op_name(self.clone()))
            } else {
                None
            };
            Ok(from_storage(storage, shape.clone(), op, false))
        }
    };
}

macro_rules! binary_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name(&self, rhs: &Self) -> Result<Self> {
            let shape = self.same_shape_binary_op(rhs, stringify!($fn_name))?;
            let storage = self.storage.binary_impl::<crate::op::$op_name>(
                &rhs.storage,
                self.layout(),
                rhs.layout(),
            )?;
            let op = if self.track_op() || rhs.track_op() {
                Some(Op::$op_name(self.clone(), rhs.clone()))
            } else {
                None
            };
            Ok(from_storage(storage, shape.clone(), op, false))
        }
    };
}

macro_rules! broadcast_binary_op {
    ($fn_name:ident, $inner_fn_name:ident) => {
        pub fn $fn_name(&self, rhs: &Self) -> Result<Self> {
            let lhs = self;
            let shape = lhs.broadcast_shape_binary_op(rhs, stringify!($fn_name))?;
            let l_broadcast = shape != *lhs.shape();
            let r_broadcast = shape != *rhs.shape();
            match (l_broadcast, r_broadcast) {
                (true, true) => lhs
                    .broadcast_as(&shape)?
                    .$inner_fn_name(&rhs.broadcast_as(&shape)?),
                (false, true) => lhs.$inner_fn_name(&rhs.broadcast_as(&shape)?),
                (true, false) => lhs.broadcast_as(&shape)?.$inner_fn_name(rhs),
                (false, false) => lhs.$inner_fn_name(rhs),
            }
        }
    };
}

/// Creates a fresh tensor structure based on a storage and a shape, this uses contiguous strides.
fn from_storage<S: Into<Shape>>(
    storage: Storage,
    shape: S,
    op: Option<Op>,
    is_variable: bool,
) -> Tensor {
    let tensor_ = Tensor_ {
        id: TensorId::new(),
        storage: Arc::new(storage),
        layout: Layout::contiguous(shape),
        op,
        is_variable,
    };
    Tensor(Arc::new(tensor_))
}

impl Tensor {
    // TODO: Maybe this should be a broadcast rather than actually creating the full tensor.
    fn ones_impl<S: Into<Shape>>(
        shape: S,
        dtype: DType,
        device: &Device,
        is_variable: bool,
    ) -> Result<Self> {
        let shape = shape.into();
        let storage = device.ones(&shape, dtype)?;
        Ok(from_storage(storage, shape, None, is_variable))
    }

    pub fn ones<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        Self::ones_impl(shape, dtype, device, false)
    }

    pub fn ones_var<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        Self::ones_impl(shape, dtype, device, true)
    }

    pub fn ones_like(&self) -> Result<Self> {
        Tensor::ones(self.shape(), self.dtype(), &self.device())
    }

    // TODO: Maybe this should be a broadcast rather than actually creating the full tensor.
    fn zeros_impl<S: Into<Shape>>(
        shape: S,
        dtype: DType,
        device: &Device,
        is_variable: bool,
    ) -> Result<Self> {
        let shape = shape.into();
        let storage = device.zeros(&shape, dtype)?;
        Ok(from_storage(storage, shape, None, is_variable))
    }

    pub fn zeros<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        Self::zeros_impl(shape, dtype, device, false)
    }

    pub fn zeros_var<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        Self::zeros_impl(shape, dtype, device, true)
    }

    pub fn zeros_like(&self) -> Result<Self> {
        Tensor::zeros(self.shape(), self.dtype(), &self.device())
    }

    pub fn new_impl<A: crate::device::NdArray>(
        array: A,
        shape: Shape,
        device: &Device,
        is_variable: bool,
    ) -> Result<Self> {
        let n: usize = shape.elem_count();
        let buffer_size: usize = array.shape()?.elem_count();
        if buffer_size != n {
            return Err(Error::ShapeMismatch { buffer_size, shape });
        }
        let storage = device.storage(array)?;
        Ok(from_storage(storage, shape, None, is_variable))
    }

    pub fn new<A: crate::device::NdArray>(array: A, device: &Device) -> Result<Self> {
        let shape = array.shape()?;
        Self::new_impl(array, shape, device, false)
    }

    pub fn var<A: crate::device::NdArray>(array: A, device: &Device) -> Result<Self> {
        let shape = array.shape()?;
        Self::new_impl(array, shape, device, true)
    }

    pub fn from_vec_impl<S: Into<Shape>, D: crate::WithDType>(
        data: Vec<D>,
        shape: S,
        device: &Device,
        is_variable: bool,
    ) -> Result<Self> {
        let shape = shape.into();
        let buffer_size = data.len();
        if buffer_size != shape.elem_count() {
            return Err(Error::ShapeMismatch { buffer_size, shape });
        }
        let storage = device.storage_owned(data)?;
        Ok(from_storage(storage, shape, None, is_variable))
    }

    pub fn from_vec<S: Into<Shape>, D: crate::WithDType>(
        data: Vec<D>,
        shape: S,
        device: &Device,
    ) -> Result<Self> {
        Self::from_vec_impl(data, shape, device, false)
    }

    pub fn var_from_vec<S: Into<Shape>, D: crate::WithDType>(
        data: Vec<D>,
        shape: S,
        device: &Device,
    ) -> Result<Self> {
        Self::from_vec_impl(data, shape, device, true)
    }

    pub fn from_slice<S: Into<Shape>, D: crate::WithDType>(
        array: &[D],
        shape: S,
        device: &Device,
    ) -> Result<Self> {
        Self::new_impl(array, shape.into(), device, false)
    }

    pub fn var_from_slice<S: Into<Shape>, D: crate::WithDType>(
        array: &[D],
        shape: S,
        device: &Device,
    ) -> Result<Self> {
        Self::new_impl(array, shape.into(), device, true)
    }

    pub(crate) fn broadcast_shape_binary_op<'a>(
        &'a self,
        rhs: &'a Self,
        op: &'static str,
    ) -> Result<Shape> {
        let lhs = self;
        let lhs_dims = lhs.shape().dims();
        let rhs_dims = rhs.shape().dims();
        let lhs_ndims = lhs_dims.len();
        let rhs_ndims = rhs_dims.len();
        let bcast_ndims = usize::max(lhs_ndims, rhs_ndims);
        let mut bcast_dims = vec![0; bcast_ndims];
        for (idx, bcast_value) in bcast_dims.iter_mut().enumerate() {
            let rev_idx = bcast_ndims - idx;
            let l_value = if lhs_ndims < rev_idx {
                1
            } else {
                lhs_dims[lhs_ndims - rev_idx]
            };
            let r_value = if rhs_ndims < rev_idx {
                1
            } else {
                rhs_dims[rhs_ndims - rev_idx]
            };
            *bcast_value = if l_value == r_value {
                l_value
            } else if l_value == 1 {
                r_value
            } else if r_value == 1 {
                l_value
            } else {
                Err(Error::ShapeMismatchBinaryOp {
                    lhs: self.shape().clone(),
                    rhs: rhs.shape().clone(),
                    op,
                })?
            }
        }
        Ok(Shape::from(bcast_dims))
    }

    pub(crate) fn same_shape_binary_op(&self, rhs: &Self, op: &'static str) -> Result<&Shape> {
        let lhs = self.shape();
        let rhs = rhs.shape();
        if lhs != rhs {
            Err(Error::ShapeMismatchBinaryOp {
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                op,
            })
        } else {
            Ok(lhs)
        }
    }

    /// Returns true if the computation graph should track this op, that is if it is
    /// a variable or if it has some variable as dependencies.
    pub(crate) fn track_op(&self) -> bool {
        self.is_variable || self.op.is_some()
    }

    // TODO: Also make an inplace version or a pre-allocated? This could be tricky
    // if this can create cycles in the compute graph.
    binary_op!(add, Add);
    binary_op!(mul, Mul);
    binary_op!(sub, Sub);
    binary_op!(div, Div);
    broadcast_binary_op!(broadcast_add, add);
    broadcast_binary_op!(broadcast_mul, mul);
    broadcast_binary_op!(broadcast_sub, sub);
    broadcast_binary_op!(broadcast_div, div);

    unary_op!(neg, Neg);
    unary_op!(exp, Exp);
    unary_op!(log, Log);
    unary_op!(sin, Sin);
    unary_op!(cos, Cos);
    unary_op!(abs, Abs);
    unary_op!(sqr, Sqr);
    unary_op!(sqrt, Sqrt);
    unary_op!(gelu, Gelu);
    unary_op!(relu, Relu);

    pub fn to_scalar<S: crate::WithDType>(&self) -> Result<S> {
        if self.rank() != 0 {
            return Err(Error::UnexpectedNumberOfDims {
                expected: 0,
                got: self.rank(),
                shape: self.shape().clone(),
            });
        }
        let from_cpu_storage = |cpu_storage: &crate::CpuStorage| {
            let data = S::cpu_storage_as_slice(cpu_storage)?;
            Ok::<_, Error>(data[0])
        };
        match self.storage.as_ref() {
            Storage::Cpu(cpu_storage) => from_cpu_storage(cpu_storage),
            Storage::Cuda(storage) => from_cpu_storage(&storage.to_cpu_storage()?),
        }
    }

    pub fn affine(&self, mul: f64, add: f64) -> Result<Self> {
        let storage = self.storage.affine(self.layout(), mul, add)?;
        let op = if self.track_op() {
            Some(Op::Affine {
                arg: self.clone(),
                mul,
                add,
            })
        } else {
            None
        };
        Ok(from_storage(storage, self.shape(), op, false))
    }

    /// Returns a new tensor that is a narrowed version of the input, the dimension `dim`
    /// ranges from `start` to `start + length`.
    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Self> {
        let op = if self.track_op() {
            Some(Op::Narrow(self.clone(), dim, start, length))
        } else {
            None
        };
        let tensor_ = Tensor_ {
            id: TensorId::new(),
            storage: self.storage.clone(),
            layout: self.layout().narrow(dim, start, length)?,
            op,
            is_variable: false,
        };
        Ok(Tensor(Arc::new(tensor_)))
    }

    pub fn softmax(&self, dim: usize) -> Result<Self> {
        // TODO: unify the two branches.
        if self.device().is_cuda() {
            // We do not have a cuda kernel for divide_by_sum_over_dim so split
            // the operation.
            let exp = self.exp()?;
            let sum_exp = exp.sum(&[dim])?;
            exp.broadcast_div(&sum_exp)
        } else {
            let shape = self.shape();
            let mut storage = self.storage.unary_impl::<crate::op::Exp>(self.layout())?;
            // The resulting storage is contiguous.
            storage.divide_by_sum_over_dim(shape, dim)?;
            let op = if self.track_op() {
                Some(Op::Softmax(self.clone(), dim))
            } else {
                None
            };
            Ok(from_storage(storage, shape.clone(), op, false))
        }
    }

    pub fn sum(&self, sum_dims: &[usize]) -> Result<Self> {
        let storage = self.storage.sum(self.layout(), sum_dims)?;
        let op = if self.track_op() {
            Some(Op::Sum(self.clone(), sum_dims.to_vec()))
        } else {
            None
        };
        let mut dims = self.dims().to_vec();
        for &sum_dim in sum_dims.iter() {
            dims[sum_dim] = 1
        }
        Ok(from_storage(storage, dims, op, false))
    }

    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        let a_dims = self.shape().dims();
        let b_dims = rhs.shape().dims();

        let dim = a_dims.len();

        if dim < 2 || b_dims.len() != dim {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape().clone(),
                rhs: rhs.shape().clone(),
                op: "matmul",
            });
        }

        let m = a_dims[dim - 2];
        let k = a_dims[dim - 1];
        let k2 = b_dims[dim - 2];
        let n = b_dims[dim - 1];
        if k != k2 {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape().clone(),
                rhs: rhs.shape().clone(),
                op: "matmul",
            });
        }

        let c_shape = Shape::from(&a_dims[..dim - 2]).extend(&[m, n]);
        let batching: usize = a_dims[..dim - 2].iter().product();

        let storage = self.storage.matmul(
            &rhs.storage,
            (batching, m, n, k),
            self.layout(),
            rhs.layout(),
        )?;
        let op = if self.track_op() || rhs.track_op() {
            Some(Op::Matmul(self.clone(), rhs.clone()))
        } else {
            None
        };
        Ok(from_storage(storage, c_shape, op, false))
    }

    pub fn where_cond(&self, on_true: &Self, on_false: &Self) -> Result<Self> {
        let _shap = self.same_shape_binary_op(on_true, "where_cond")?;
        let shape = self.same_shape_binary_op(on_false, "where_cond")?;
        let storage = self.storage.where_cond(
            self.layout(),
            &on_true.storage,
            on_true.layout(),
            &on_false.storage,
            on_false.layout(),
        )?;
        let op = if self.track_op() || on_true.track_op() || on_false.track_op() {
            Some(Op::WhereCond(
                self.clone(),
                on_true.clone(),
                on_false.clone(),
            ))
        } else {
            None
        };
        Ok(from_storage(storage, shape, op, false))
    }

    pub fn embedding(ids: &Self, rhs: &Self) -> Result<Self> {
        if !rhs.is_contiguous() {
            return Err(Error::RequiresContiguous { op: "embedding" });
        } else if rhs.rank() != 2 || ids.rank() != 1 {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: ids.shape().clone(),
                rhs: rhs.shape().clone(),
                op: "embedding",
            });
        }
        let ids_shape = ids.shape();
        let seq_len = ids_shape.r1()?;
        let (_, hidden_size) = rhs.shape().r2()?;
        let storage = ids
            .storage
            .embedding(ids.layout(), &rhs.storage, rhs.layout())?;
        let shape: Shape = (seq_len, hidden_size).into();
        let op = if ids.track_op() || rhs.track_op() {
            Some(Op::Embedding(ids.clone(), rhs.clone()))
        } else {
            None
        };
        Ok(from_storage(storage, shape, op, false))
    }

    pub(crate) fn strided_index(&self) -> crate::StridedIndex {
        self.layout.strided_index()
    }

    /// Returns data from the underlying storage, this does not take the strides
    /// into account so the size of the resulting buffer might be larger than the
    /// tensor number of elements.
    pub fn storage_data<S: crate::WithDType>(&self) -> Result<std::borrow::Cow<[S]>> {
        match self.storage.as_ref() {
            Storage::Cpu(cpu_storage) => {
                let slice = S::cpu_storage_as_slice(cpu_storage)?;
                Ok(std::borrow::Cow::Borrowed(slice))
            }
            Storage::Cuda(slice) => {
                let cpu_storage = slice.to_cpu_storage()?;
                let storage_data = S::cpu_storage_data(cpu_storage)?;
                Ok(std::borrow::Cow::Owned(storage_data))
            }
        }
    }

    pub fn to_vec1<S: crate::WithDType>(&self) -> Result<Vec<S>> {
        if self.rank() != 1 {
            return Err(Error::UnexpectedNumberOfDims {
                expected: 1,
                got: self.rank(),
                shape: self.shape().clone(),
            });
        }
        match self.storage.as_ref() {
            Storage::Cpu(cpu_storage) => {
                let data = S::cpu_storage_as_slice(cpu_storage)?;
                Ok(self.strided_index().map(|i| data[i]).collect())
            }
            Storage::Cuda(slice) => {
                // TODO: Would it be possible to only fetch the necessary data?
                let cpu_storage = slice.to_cpu_storage()?;
                let data = S::cpu_storage_as_slice(&cpu_storage)?;
                Ok(self.strided_index().map(|i| data[i]).collect())
            }
        }
    }

    pub fn to_vec2<S: crate::WithDType>(&self) -> Result<Vec<Vec<S>>> {
        let (dim1, dim2) = self.shape().r2()?;
        let from_cpu_storage = |cpu_storage: &crate::CpuStorage| {
            let data = S::cpu_storage_as_slice(cpu_storage)?;
            let mut rows = vec![];
            let mut src_index = self.strided_index();
            for _idx_row in 0..dim1 {
                let row = (0..dim2).map(|_| data[src_index.next().unwrap()]).collect();
                rows.push(row)
            }
            assert!(src_index.next().is_none());
            Ok(rows)
        };
        match self.storage.as_ref() {
            Storage::Cpu(storage) => from_cpu_storage(storage),
            Storage::Cuda(storage) => from_cpu_storage(&storage.to_cpu_storage()?),
        }
    }

    pub fn to_vec3<S: crate::WithDType>(&self) -> Result<Vec<Vec<Vec<S>>>> {
        let (dim1, dim2, dim3) = self.shape().r3()?;
        let from_cpu_storage = |cpu_storage: &crate::CpuStorage| {
            let data = S::cpu_storage_as_slice(cpu_storage)?;
            let mut top_rows = vec![];
            let mut src_index = self.strided_index();
            for _idx in 0..dim1 {
                let mut rows = vec![];
                for _jdx in 0..dim2 {
                    let row = (0..dim3).map(|_| data[src_index.next().unwrap()]).collect();
                    rows.push(row)
                }
                top_rows.push(rows);
            }
            assert!(src_index.next().is_none());
            Ok(top_rows)
        };
        match self.storage.as_ref() {
            Storage::Cpu(storage) => from_cpu_storage(storage),
            Storage::Cuda(storage) => from_cpu_storage(&storage.to_cpu_storage()?),
        }
    }

    pub fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    pub fn device(&self) -> Device {
        self.storage.device()
    }

    pub fn shape(&self) -> &Shape {
        self.layout().shape()
    }

    pub fn dims(&self) -> &[usize] {
        self.shape().dims()
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    // TODO: Rename to `stride` once the PR that introduced the layout has been merged.
    pub fn stride_tmp(&self) -> &[usize] {
        self.layout.stride()
    }

    pub fn rank(&self) -> usize {
        self.shape().rank()
    }

    pub fn elem_count(&self) -> usize {
        self.shape().elem_count()
    }

    pub fn id(&self) -> TensorId {
        self.id
    }

    pub fn is_variable(&self) -> bool {
        self.is_variable
    }

    pub(crate) fn op(&self) -> &Option<Op> {
        &self.op
    }

    pub fn sum_all(&self) -> Result<Tensor> {
        let dims: Vec<_> = (0..self.rank()).collect();
        self.sum(&dims)
    }

    pub fn flatten(&self, start_dim: Option<usize>, end_dim: Option<usize>) -> Result<Tensor> {
        if self.rank() == 0 {
            self.reshape(1)
        } else {
            let start_dim = start_dim.unwrap_or(0);
            let end_dim = end_dim.unwrap_or_else(|| self.rank() - 1);
            if start_dim < end_dim {
                let dims = self.dims();
                let mut dst_dims = dims[..start_dim].to_vec();
                dst_dims.push(dims[start_dim..end_dim + 1].iter().product::<usize>());
                if end_dim + 1 < dims.len() {
                    dst_dims.extend(&dims[end_dim + 1..]);
                }
                self.reshape(dst_dims)
            } else {
                Ok(self.clone())
            }
        }
    }

    pub fn flatten_all(&self) -> Result<Tensor> {
        self.flatten(None, None)
    }

    pub fn get(&self, i: usize) -> Result<Tensor> {
        let dims = self.dims();
        if dims.is_empty() {
            Ok(self.clone())
        } else {
            self.narrow(0, i, 1)?.reshape(&dims[1..])
        }
    }

    /// Returns a tensor that is a transposed version of the input, the two last dimensions of the
    /// input are swapped.
    pub fn t(&self) -> Result<Tensor> {
        let rank = self.rank();
        if rank < 2 {
            return Err(Error::UnexpectedNumberOfDims {
                expected: 2,
                got: rank,
                shape: self.shape().clone(),
            });
        }
        self.transpose(rank - 2, rank - 1)
    }

    /// Returns a tensor that is a transposed version of the input, the given dimensions are
    /// swapped.
    pub fn transpose(&self, dim1: usize, dim2: usize) -> Result<Tensor> {
        let op = if self.track_op() {
            Some(Op::Transpose(self.clone(), dim1, dim2))
        } else {
            None
        };
        let tensor_ = Tensor_ {
            id: TensorId::new(),
            storage: self.storage.clone(),
            layout: self.layout.transpose(dim1, dim2)?,
            op,
            is_variable: false,
        };
        Ok(Tensor(Arc::new(tensor_)))
    }

    /// Returns true if the data is stored in a C contiguous (aka row major) way.
    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    /// Returns true if the data is stored in a Fortran contiguous (aka column major) way.
    pub fn is_fortran_contiguous(&self) -> bool {
        self.layout.is_fortran_contiguous()
    }

    /// Compared to clone, this copies the actual storage but may fail because of running out of
    /// memory.
    pub fn copy(&self) -> Result<Tensor> {
        let tensor_ = Tensor_ {
            id: TensorId::new(),
            storage: Arc::new(self.storage.try_clone()?),
            layout: self.layout.clone(),
            op: None, // TODO
            is_variable: false,
        };
        Ok(Tensor(Arc::new(tensor_)))
    }

    /// Returns a new tensor detached from the current graph, gradient are not propagated through
    /// this new node.
    pub fn detach(&self) -> Result<Tensor> {
        let tensor_ = Tensor_ {
            id: TensorId::new(),
            storage: self.storage.clone(),
            layout: self.layout.clone(),
            op: None,
            is_variable: false,
        };
        Ok(Tensor(Arc::new(tensor_)))
    }

    /// If the target device is the same as the tensor device, only a shallow copy is performed.
    pub fn to_device(&self, device: &Device) -> Result<Tensor> {
        if self.device().same_id(device) {
            Ok(self.clone())
        } else {
            let storage = match (self.storage.as_ref(), device) {
                (Storage::Cpu(storage), Device::Cuda(cuda)) => {
                    Storage::Cuda(cuda.cuda_from_cpu_storage(storage)?)
                }
                (Storage::Cuda(storage), Device::Cpu) => Storage::Cpu(storage.to_cpu_storage()?),
                (Storage::Cuda(storage), Device::Cuda(cuda)) => {
                    // TODO: Avoid passing through the cpu storage here, especially if the gpu ids
                    // are the same.
                    let cpu_storage = storage.to_cpu_storage()?;
                    Storage::Cuda(cuda.cuda_from_cpu_storage(&cpu_storage)?)
                }
                (Storage::Cpu(storage), Device::Cpu) => Storage::Cpu(storage.clone()),
            };
            let op = if self.track_op() {
                Some(Op::ToDevice(self.clone()))
            } else {
                None
            };
            let tensor_ = Tensor_ {
                id: TensorId::new(),
                storage: Arc::new(storage),
                layout: self.layout.clone(),
                op,
                is_variable: false,
            };
            Ok(Tensor(Arc::new(tensor_)))
        }
    }

    /// Returns a new tensor duplicating data from the original tensor. New dimensions are inserted
    /// on the left.
    pub fn broadcast_left<S: Into<Shape>>(&self, left_shape: S) -> Result<Self> {
        let left_shape = left_shape.into();
        let mut dims = left_shape.into_dims();
        dims.extend(self.dims());
        self.broadcast_as(dims)
    }

    pub fn broadcast_as<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let op = if self.track_op() {
            Some(Op::Broadcast(self.clone()))
        } else {
            None
        };
        let tensor_ = Tensor_ {
            id: TensorId::new(),
            storage: self.storage.clone(),
            layout: self.layout.broadcast_as(shape)?,
            op,
            is_variable: false,
        };
        Ok(Tensor(Arc::new(tensor_)))
    }

    /// An alias for broadcast_as.
    pub fn expand<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        self.broadcast_as(shape)
    }

    pub fn to_dtype(&self, dtype: DType) -> Result<Self> {
        if self.dtype() == dtype {
            Ok(self.clone())
        } else {
            let shape = self.shape();
            let storage = self.storage.to_dtype(self.layout(), dtype)?;
            let op = if self.track_op() {
                Some(Op::ToDType(self.clone()))
            } else {
                None
            };
            Ok(from_storage(storage, shape.clone(), op, false))
        }
    }

    pub fn contiguous(&self) -> Result<Tensor> {
        if self.is_contiguous() {
            Ok(self.clone())
        } else {
            let shape = self.shape();
            let mut storage = self.device().zeros(shape, self.dtype())?;
            self.storage
                .copy_strided_src(&mut storage, 0, self.layout())?;
            Ok(from_storage(
                storage,
                shape.clone(),
                None, // TODO
                false,
            ))
        }
    }

    // TODO: Do we want to allow target shape using -1 on some dimensions?
    /// Reshape returns a tensor with the target shape provided that the number of elements of the
    /// original tensor is the same.
    /// If the input tensor is contiguous, this is a view on the original data. Otherwise this uses
    /// a new storage and copies the data over, the returned tensor is always contiguous.
    pub fn reshape<S: Into<Shape>>(&self, shape: S) -> Result<Tensor> {
        let shape = shape.into();
        if shape.elem_count() != self.elem_count() {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape().clone(),
                rhs: shape,
                op: "reshape",
            });
        }
        let op = if self.track_op() {
            Some(Op::Reshape(self.clone()))
        } else {
            None
        };
        if self.is_contiguous() {
            let tensor_ = Tensor_ {
                id: TensorId::new(),
                storage: self.storage.clone(),
                layout: Layout::contiguous_with_offset(shape, self.layout.start_offset()),
                op,
                is_variable: false,
            };
            Ok(Tensor(Arc::new(tensor_)))
        } else {
            let mut storage = self.device().zeros(&shape, self.dtype())?;
            self.storage
                .copy_strided_src(&mut storage, 0, self.layout())?;
            Ok(from_storage(storage, shape, op, false))
        }
    }

    pub fn squeeze(&self, index: usize) -> Result<Self> {
        // The PyTorch semantics are to return the same tensor if the target dimension
        // does not have a size of 1.
        let dims = self.dims();
        if dims[index] == 1 {
            let mut dims = dims.to_vec();
            dims.remove(index);
            self.reshape(dims)
        } else {
            Ok(self.clone())
        }
    }

    pub fn unsqueeze(&self, index: usize) -> Result<Self> {
        let mut dims = self.dims().to_vec();
        dims.insert(index, 1);
        self.reshape(dims)
    }

    pub fn stack<A: AsRef<Tensor>>(args: &[A], dim: usize) -> Result<Self> {
        if args.is_empty() {
            return Err(Error::OpRequiresAtLeastOneTensor { op: "stack" });
        }
        let args = args
            .iter()
            .map(|t| t.as_ref().unsqueeze(dim))
            .collect::<Result<Vec<_>>>()?;
        Self::cat(&args, dim)
    }

    pub fn cat<A: AsRef<Tensor>>(args: &[A], dim: usize) -> Result<Self> {
        if args.is_empty() {
            return Err(Error::OpRequiresAtLeastOneTensor { op: "cat" });
        }
        let arg0 = args[0].as_ref();
        if args.len() == 1 {
            return Ok(arg0.clone());
        }
        let rank = arg0.rank();
        if dim >= rank {
            return Err(Error::UnexpectedNumberOfDims {
                expected: (dim + 1),
                got: rank,
                shape: arg0.shape().clone(),
            });
        }
        if dim == 0 {
            Self::cat0(args)
        } else {
            // TODO: Avoid these transpositions and have an implementation that works
            // for dim != 0...
            let args: Vec<Tensor> = args
                .iter()
                .map(|a| a.as_ref().transpose(0, dim))
                .collect::<Result<Vec<_>>>()?;
            let cat = Self::cat0(&args)?;
            cat.transpose(0, dim)
        }
    }

    pub fn cat0<A: AsRef<Tensor>>(args: &[A]) -> Result<Self> {
        if args.is_empty() {
            return Err(Error::OpRequiresAtLeastOneTensor { op: "cat" });
        }
        let arg0 = args[0].as_ref();
        if args.len() == 1 {
            return Ok(arg0.clone());
        }
        let rank = arg0.rank();
        let device = arg0.device();
        let dtype = arg0.dtype();
        let first_dims = arg0.shape().dims();
        let mut cat_dims = first_dims.to_vec();
        cat_dims[0] = 0;
        let mut offsets = vec![0usize];
        for (arg_idx, arg) in args.iter().enumerate() {
            let arg = arg.as_ref();
            if arg.dtype() != dtype {
                // TODO: Improve the error message.
                return Err(Error::DTypeMismatchBinaryOp {
                    lhs: dtype,
                    rhs: arg.dtype(),
                    op: "cat",
                });
            }
            if arg.device().location() != device.location() {
                // TODO: Improve the error message.
                return Err(Error::DeviceMismatchBinaryOp {
                    lhs: device.location(),
                    rhs: arg.device().location(),
                    op: "cat",
                });
            }
            let mut mismatch = arg.rank() != rank;
            for (dim_idx, (v1, v2)) in arg0
                .shape()
                .dims()
                .iter()
                .zip(arg.shape().dims().iter())
                .enumerate()
            {
                if dim_idx == 0 {
                    cat_dims[0] += v2;
                }
                if dim_idx != 0 && v1 != v2 {
                    // TODO: It would probably be good to have a nicer error message here, i.e.
                    // mention the problematic dimension and the values.
                    mismatch = true;
                }
            }
            if mismatch {
                return Err(Error::ShapeMismatchCat {
                    dim: 0, // TODO: not the appropriate error message
                    first_shape: arg0.shape().clone(),
                    n: arg_idx + 1,
                    nth_shape: arg.shape().clone(),
                });
            }
            let next_offset = offsets.last().unwrap() + arg.elem_count();
            offsets.push(next_offset);
        }
        let shape = Shape::from(cat_dims);
        let op = if args.iter().any(|arg| arg.as_ref().track_op()) {
            let args: Vec<Tensor> = args.iter().map(|arg| arg.as_ref().clone()).collect();
            Some(Op::Cat(args, 0))
        } else {
            None
        };
        let mut storage = device.zeros(&shape, dtype)?;
        for (arg, &offset) in args.iter().zip(offsets.iter()) {
            let arg = arg.as_ref();
            arg.storage
                .copy_strided_src(&mut storage, offset, arg.layout())?;
        }
        Ok(from_storage(storage, shape, op, false))
    }
}

macro_rules! bin_trait {
    ($trait:ident, $fn1:ident, $mul:expr, $add:expr) => {
        impl<B: std::borrow::Borrow<Tensor>> std::ops::$trait<B> for Tensor {
            type Output = Result<Tensor>;

            fn $fn1(self, rhs: B) -> Self::Output {
                Tensor::$fn1(&self, rhs.borrow())
            }
        }

        impl<B: std::borrow::Borrow<Tensor>> std::ops::$trait<B> for &Tensor {
            type Output = Result<Tensor>;

            fn $fn1(self, rhs: B) -> Self::Output {
                Tensor::$fn1(&self, rhs.borrow())
            }
        }

        impl<B: std::borrow::Borrow<Tensor>> std::ops::$trait<Result<B>> for Tensor {
            type Output = Result<Tensor>;

            fn $fn1(self, rhs: Result<B>) -> Self::Output {
                Tensor::$fn1(&self, rhs?.borrow())
            }
        }

        impl<B: std::borrow::Borrow<Tensor>> std::ops::$trait<Result<B>> for &Tensor {
            type Output = Result<Tensor>;

            fn $fn1(self, rhs: Result<B>) -> Self::Output {
                Tensor::$fn1(&self, rhs?.borrow())
            }
        }

        impl std::ops::$trait<f64> for Tensor {
            type Output = Result<Tensor>;

            fn $fn1(self, rhs: f64) -> Self::Output {
                self.affine($mul(rhs), $add(rhs))
            }
        }

        impl std::ops::$trait<f64> for &Tensor {
            type Output = Result<Tensor>;

            fn $fn1(self, rhs: f64) -> Self::Output {
                self.affine($mul(rhs), $add(rhs))
            }
        }
    };
}

bin_trait!(Add, add, |_| 1., |v| v);
bin_trait!(Sub, sub, |_| 1., |v: f64| -v);
bin_trait!(Mul, mul, |v| v, |_| 0.);
bin_trait!(Div, div, |v| 1. / v, |_| 0.);
