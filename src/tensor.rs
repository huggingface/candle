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

#[allow(dead_code)]
pub struct Tensor_ {
    id: TensorId,
    storage: Storage,
    shape: Shape,
    // The strides are given in number of elements and not in bytes.
    stride: Vec<usize>,
    op: Option<Op>,
}

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
    ($fn_name:ident, $op_name:ident, $impl_name:ident) => {
        pub fn $fn_name(&self) -> Result<Self> {
            let shape = self.shape();
            let storage = self.storage.$impl_name(self.shape(), self.stride())?;
            let tensor_ = Tensor_ {
                id: TensorId::new(),
                storage,
                shape: shape.clone(),
                stride: shape.stride_contiguous(),
                op: Some(Op::$op_name(self.clone())),
            };
            Ok(Self(Arc::new(tensor_)))
        }
    };
}

macro_rules! binary_op {
    ($fn_name:ident, $op_name:ident, $impl_name:ident) => {
        pub fn $fn_name(&self, rhs: &Self) -> Result<Self> {
            let shape = self.same_shape_binary_op(rhs, stringify!($fn_name))?;
            let storage =
                self.storage
                    .$impl_name(&rhs.storage, shape, self.stride(), rhs.stride())?;
            let tensor_ = Tensor_ {
                id: TensorId::new(),
                storage,
                shape: shape.clone(),
                stride: shape.stride_contiguous(),
                op: Some(Op::$op_name(self.clone(), rhs.clone())),
            };
            Ok(Self(Arc::new(tensor_)))
        }
    };
}

impl Tensor {
    pub fn zeros<S: Into<Shape>>(shape: S, dtype: DType, device: Device) -> Self {
        let shape = shape.into();
        let storage = device.zeros(&shape, dtype);
        let stride = shape.stride_contiguous();
        let tensor_ = Tensor_ {
            id: TensorId::new(),
            storage,
            shape,
            stride,
            op: None,
        };
        Self(Arc::new(tensor_))
    }

    pub fn new<A: crate::device::NdArray>(array: A, device: Device) -> Result<Self> {
        let shape = array.shape()?;
        let storage = device.tensor(array);
        let stride = shape.stride_contiguous();
        let tensor_ = Tensor_ {
            id: TensorId::new(),
            storage,
            shape,
            stride,
            op: None,
        };
        Ok(Self(Arc::new(tensor_)))
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

    // TODO: Also make an inplace version or a pre-allocated? This could be tricky
    // if this can create cycles in the compute graph.
    binary_op!(add, Add, add_impl);
    binary_op!(mul, Mul, mul_impl);

    unary_op!(sqr, Sqr, sqr_impl);
    unary_op!(sqrt, Sqrt, sqrt_impl);
    pub fn to_scalar<S: crate::WithDType>(&self) -> Result<S> {
        if self.rank() != 0 {
            return Err(Error::UnexpectedNumberOfDims {
                expected: 0,
                got: self.rank(),
                shape: self.shape().clone(),
            });
        }
        match &self.storage {
            Storage::Cpu(cpu_storage) => {
                let data = S::cpu_storage_as_slice(cpu_storage)?;
                Ok(data[0])
            }
        }
    }

    pub(crate) fn strided_index(&self) -> crate::storage::StridedIndex {
        crate::storage::StridedIndex::new(self.dims(), self.stride())
    }

    pub fn to_vec1<S: crate::WithDType>(&self) -> Result<Vec<S>> {
        if self.rank() != 1 {
            return Err(Error::UnexpectedNumberOfDims {
                expected: 1,
                got: self.rank(),
                shape: self.shape().clone(),
            });
        }
        match &self.storage {
            Storage::Cpu(cpu_storage) => {
                let data = S::cpu_storage_as_slice(cpu_storage)?;
                Ok(self.strided_index().map(|i| data[i]).collect())
            }
        }
    }

    pub fn to_vec2<S: crate::WithDType>(&self) -> Result<Vec<Vec<S>>> {
        let (dim1, dim2) = self.shape().r2()?;
        match &self.storage {
            Storage::Cpu(cpu_storage) => {
                let data = S::cpu_storage_as_slice(cpu_storage)?;
                let mut rows = vec![];
                let mut src_index = self.strided_index();
                for _idx_row in 0..dim1 {
                    let row = (0..dim2).map(|_| data[src_index.next().unwrap()]).collect();
                    rows.push(row)
                }
                assert!(src_index.next().is_none());
                Ok(rows)
            }
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
}
