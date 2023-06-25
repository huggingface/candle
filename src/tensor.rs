use crate::{op::Op, storage::Storage, DType, Device, Error, Result, Shape};
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
    shape: Shape,
    // The strides are given in number of elements and not in bytes.
    stride: Vec<usize>,
    op: Option<Op>,
    is_variable: bool,
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

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:?}, {:?}]", &self.shape().dims(), self.device())
    }
}

macro_rules! unary_op {
    ($fn_name:ident, $op_name:ident) => {
        pub fn $fn_name(&self) -> Result<Self> {
            let shape = self.shape();
            let storage = self
                .storage
                .unary_impl::<crate::op::$op_name>(self.shape(), self.stride())?;
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
                shape,
                self.stride(),
                rhs.stride(),
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
    ($fn_name:ident, $impl_name:ident, $op_name:ident) => {
        pub fn $fn_name(&self, rhs: &Self) -> Result<Self> {
            let shape = self.broadcast_shape_binary_op(rhs, stringify!($fn_name))?;
            let storage = self.storage.binary_impl::<crate::op::$impl_name>(
                &rhs.storage,
                shape,
                self.stride(),
                rhs.stride(),
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

/// Creates a fresh tensor structure based on a storage and a shape, this uses contiguous strides.
fn from_storage<S: Into<Shape>>(
    storage: Storage,
    shape: S,
    op: Option<Op>,
    is_variable: bool,
) -> Tensor {
    let shape = shape.into();
    let stride = shape.stride_contiguous();
    let tensor_ = Tensor_ {
        id: TensorId::new(),
        storage: Arc::new(storage),
        shape,
        stride,
        op,
        is_variable,
    };
    Tensor(Arc::new(tensor_))
}

impl Tensor {
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
    ) -> Result<&'a Shape> {
        let lhs = self;
        let lhs_dims = lhs.shape().dims();
        let rhs_dims = rhs.shape().dims();
        if lhs_dims.strip_suffix(rhs_dims).is_some() {
            Ok(self.shape())
        } else if rhs_dims.strip_suffix(lhs_dims).is_some() {
            Ok(rhs.shape())
        } else {
            Err(Error::ShapeMismatchBinaryOp {
                lhs: self.shape().clone(),
                rhs: rhs.shape().clone(),
                op,
            })
        }
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
    broadcast_binary_op!(broadcast_add, Add, BroadcastAdd);
    broadcast_binary_op!(broadcast_mul, Mul, BroadcastMul);
    broadcast_binary_op!(broadcast_sub, Sub, BroadcastSub);
    broadcast_binary_op!(broadcast_div, Div, BroadcastDiv);

    unary_op!(neg, Neg);
    unary_op!(exp, Exp);
    unary_op!(log, Log);
    unary_op!(sin, Sin);
    unary_op!(cos, Cos);
    unary_op!(abs, Abs);
    unary_op!(sqr, Sqr);
    unary_op!(sqrt, Sqrt);
    unary_op!(gelu, Gelu);
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
        let shape = self.shape();
        let storage = self.storage.affine_impl(shape, self.stride(), mul, add)?;
        let op = if self.track_op() {
            Some(Op::Affine {
                arg: self.clone(),
                mul,
                add,
            })
        } else {
            None
        };
        Ok(from_storage(storage, shape.clone(), op, false))
    }

    /// Returns a new tensor that is a narrowed version of the input, the dimension `dim`
    /// ranges from `start` to `start + length`.
    // TODO: Once we've refactored the shape and strides, make this return a view of the same data
    // rather than copying.
    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Self> {
        let dims = self.shape().dims();
        if dim >= dims.len() {
            return Err(Error::UnexpectedNumberOfDims {
                expected: dim + 1,
                got: dims.len(),
                shape: self.shape().clone(),
            });
        }
        if start + length > dims[dim] {
            todo!("add a proper error: out of bounds for narrow {dim} {start} {length} {dims:?}")
        }
        let mut dims = dims.to_vec();
        dims[dim] = length;
        let adjusted_shape = Shape::from(dims);
        let mut storage = self.device().zeros(&adjusted_shape, self.dtype())?;
        self.storage.copy_strided_src(
            &mut storage,
            /* dst_offset= */ 0,
            &adjusted_shape,
            &self.stride,
            /* src_offest= */ self.stride[dim] * start,
        )?;
        let op = if self.track_op() {
            Some(Op::Narrow(self.clone(), dim, start, length))
        } else {
            None
        };
        Ok(from_storage(storage, adjusted_shape, op, false))
    }

    pub fn softmax(&self, dim: usize) -> Result<Self> {
        let shape = self.shape();
        let mut storage = self
            .storage
            .unary_impl::<crate::op::Exp>(shape, self.stride())?;
        // The resulting storage is contiguous.
        storage.divide_by_sum_over_dim(shape, dim)?;
        let op = if self.track_op() {
            Some(Op::Softmax(self.clone(), dim))
        } else {
            None
        };
        Ok(from_storage(storage, shape.clone(), op, false))
    }

    pub fn sum(&self, sum_dims: &[usize]) -> Result<Self> {
        let storage = self.storage.sum(self.shape(), &self.stride, sum_dims)?;
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
        if let crate::DeviceLocation::Cuda { .. } = self.device().location() {
            if !self.is_contiguous() || !rhs.is_contiguous() {
                // It looks like the cublas implementation of XgemmStridedBatched only supports
                // non-standard strides on the batch dimension.
                return Err(Error::RequiresContiguous {
                    op: "matmul-cublas",
                });
            }
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

        let storage = self.storage.matmul_impl(
            &rhs.storage,
            (batching, m, n, k),
            self.stride(),
            rhs.stride(),
        )?;
        let op = if self.track_op() || rhs.track_op() {
            Some(Op::Matmul(self.clone(), rhs.clone()))
        } else {
            None
        };
        Ok(from_storage(storage, c_shape, op, false))
    }

    pub fn embedding(ids: &Self, rhs: &Self) -> Result<Self> {
        if !rhs.is_contiguous() {
            return Err(Error::RequiresContiguous { op: "embedding" });
        } else if rhs.shape().rank() != 2 || ids.shape().rank() != 1 {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: ids.shape.clone(),
                rhs: rhs.shape.clone(),
                op: "embedding",
            });
        }
        let seq_len = ids.shape().r1()?;
        let (vocab_size, hidden_size) = rhs.shape().r2()?;
        let storage = ids
            .storage
            .embedding_impl(&rhs.storage, hidden_size, vocab_size)?;
        let shape: Shape = (seq_len, hidden_size).into();
        let op = if ids.track_op() || rhs.track_op() {
            Some(Op::Embedding(ids.clone(), rhs.clone()))
        } else {
            None
        };
        Ok(from_storage(storage, shape, op, false))
    }

    pub(crate) fn strided_index(&self) -> crate::StridedIndex {
        crate::StridedIndex::new(self.dims(), self.stride())
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
        &self.shape
    }

    pub fn dims(&self) -> &[usize] {
        self.shape().dims()
    }

    pub fn stride(&self) -> &[usize] {
        &self.stride
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
        let rank = self.rank();
        if rank <= dim1 || rank <= dim2 {
            return Err(Error::UnexpectedNumberOfDims {
                expected: usize::max(dim1, dim2),
                got: rank,
                shape: self.shape().clone(),
            });
        }
        let mut stride = self.stride().to_vec();
        let mut dims = self.shape().dims().to_vec();
        dims.swap(dim1, dim2);
        stride.swap(dim1, dim2);
        let op = if self.track_op() {
            Some(Op::Transpose(self.clone(), dim1, dim2))
        } else {
            None
        };
        let tensor_ = Tensor_ {
            id: TensorId::new(),
            storage: self.storage.clone(),
            shape: Shape::from(dims),
            stride,
            op,
            is_variable: false,
        };
        Ok(Tensor(Arc::new(tensor_)))
    }

    pub fn is_contiguous(&self) -> bool {
        self.shape.is_contiguous(&self.stride)
    }

    /// Compared to clone, this copies the actual storage but may fail because of running out of
    /// memory.
    pub fn copy(&self) -> Result<Tensor> {
        let tensor_ = Tensor_ {
            id: TensorId::new(),
            storage: Arc::new(self.storage.try_clone()?),
            shape: self.shape.clone(),
            stride: self.stride.clone(),
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
            shape: self.shape.clone(),
            stride: self.stride.clone(),
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
                shape: self.shape.clone(),
                stride: self.stride.clone(),
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
        dims.extend(self.shape.dims());
        self.broadcast_as(dims)
    }

    pub fn broadcast_as<S: Into<Shape>>(&self, shape: S) -> Result<Self> {
        let op = if self.track_op() {
            Some(Op::Broadcast(self.clone()))
        } else {
            None
        };
        let shape = shape.into();
        if shape.rank() < self.rank() {
            return Err(Error::BroadcastIncompatibleShapes {
                src_shape: self.shape().clone(),
                dst_shape: shape,
            });
        }
        let added_dims = shape.rank() - self.rank();
        let mut stride = vec![0; added_dims];
        for (&dst_dim, (&src_dim, &src_stride)) in shape.dims()[added_dims..]
            .iter()
            .zip(self.dims().iter().zip(self.stride()))
        {
            let s = if dst_dim == src_dim {
                src_stride
            } else if src_dim != 1 {
                return Err(Error::BroadcastIncompatibleShapes {
                    src_shape: self.shape().clone(),
                    dst_shape: shape,
                });
            } else {
                0
            };
            stride.push(s)
        }
        let tensor_ = Tensor_ {
            id: TensorId::new(),
            storage: self.storage.clone(),
            shape,
            stride,
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
        let shape = self.shape();
        let storage = self.storage.to_dtype(shape, self.stride(), dtype)?;
        let op = if self.track_op() {
            Some(Op::ToDType(self.clone()))
        } else {
            None
        };
        Ok(from_storage(storage, shape.clone(), op, false))
    }

    pub fn contiguous(&self) -> Result<Tensor> {
        if self.is_contiguous() {
            Ok(self.clone())
        } else {
            let shape = self.shape();
            let mut storage = self.device().zeros(shape, self.dtype())?;
            self.storage
                .copy_strided_src(&mut storage, 0, &self.shape, &self.stride, 0)?;
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
            let stride = shape.stride_contiguous();
            let tensor_ = Tensor_ {
                id: TensorId::new(),
                storage: self.storage.clone(),
                shape,
                stride,
                op,
                is_variable: false,
            };
            Ok(Tensor(Arc::new(tensor_)))
        } else {
            let mut storage = self.device().zeros(&shape, self.dtype())?;
            self.storage
                .copy_strided_src(&mut storage, 0, &self.shape, &self.stride, 0)?;
            Ok(from_storage(storage, shape, op, false))
        }
    }

    pub fn cat(args: &[&Self], dim: usize) -> Result<Self> {
        if args.is_empty() {
            return Err(Error::OpRequiresAtLeastOneTensor { op: "cat" });
        }
        let rank = args[0].rank();
        if dim >= rank {
            return Err(Error::UnexpectedNumberOfDims {
                expected: (dim + 1),
                got: rank,
                shape: args[0].shape().clone(),
            });
        }
        let device = args[0].device();
        let dtype = args[0].dtype();
        let first_dims = args[0].shape().dims();
        let mut cat_dims = first_dims.to_vec();
        cat_dims[dim] = 0;
        let mut offsets = vec![0usize];
        for (arg_idx, arg) in args.iter().enumerate() {
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
            for (dim_idx, (v1, v2)) in args[0]
                .shape()
                .dims()
                .iter()
                .zip(arg.shape().dims().iter())
                .enumerate()
            {
                if dim == dim_idx {
                    cat_dims[dim] += v2;
                }
                if dim != dim_idx && v1 != v2 {
                    // TODO: It would probably be good to have a nicer error message here, i.e.
                    // mention the problematic dimension and the values.
                    mismatch = true;
                }
            }
            if mismatch {
                return Err(Error::ShapeMismatchCat {
                    dim,
                    first_shape: args[0].shape().clone(),
                    n: arg_idx + 1,
                    nth_shape: arg.shape().clone(),
                });
            }
            let next_offset = offsets.last().unwrap() + arg.elem_count();
            offsets.push(next_offset);
        }
        let shape = Shape::from(cat_dims);
        let op = if args.iter().any(|arg| arg.track_op()) {
            let args: Vec<Tensor> = args.iter().map(|&arg| arg.clone()).collect();
            Some(Op::Cat(args, dim))
        } else {
            None
        };
        let mut storage = device.zeros(&shape, dtype)?;
        for (arg, &offset) in args.iter().zip(offsets.iter()) {
            arg.storage
                .copy_strided_src(&mut storage, offset, &arg.shape, &arg.stride, 0)?
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
