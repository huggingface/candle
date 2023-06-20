use crate::{op::Op, storage::Storage, DType, Device, Error, Result, Shape};
use std::sync::Arc;

#[allow(dead_code)]
pub struct Tensor_ {
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

impl Tensor {
    pub fn zeros<S: Into<Shape>>(shape: S, dtype: DType, device: Device) -> Self {
        let shape = shape.into();
        let storage = device.zeros(&shape, dtype);
        let stride = shape.stride_contiguous();
        let tensor_ = Tensor_ {
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
    pub fn add(&self, rhs: &Self) -> Result<Self> {
        let shape = self.same_shape_binary_op(rhs, "add")?;
        let storage = self
            .storage
            .add_impl(&rhs.storage, shape, self.stride(), rhs.stride())?;
        let tensor_ = Tensor_ {
            storage,
            shape: shape.clone(),
            stride: shape.stride_contiguous(),
            op: Some(Op::Add(self.clone(), rhs.clone())),
        };
        Ok(Self(Arc::new(tensor_)))
    }

    pub fn mul(&self, rhs: &Self) -> Result<Self> {
        let shape = self.same_shape_binary_op(rhs, "mul")?;
        let storage = self
            .storage
            .mul_impl(&rhs.storage, shape, self.stride(), rhs.stride())?;
        let tensor_ = Tensor_ {
            storage,
            shape: shape.clone(),
            stride: shape.stride_contiguous(),
            op: Some(Op::Mul(self.clone(), rhs.clone())),
        };
        Ok(Self(Arc::new(tensor_)))
    }

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

    pub fn to_vec1<S: crate::WithDType>(&self) -> Result<Vec<S>> {
        // TODO: properly use the strides here.
        todo!()
    }

    pub fn to_vec2<S: crate::WithDType>(&self) -> Result<Vec<Vec<S>>> {
        // TODO: properly use the strides here.
        todo!()
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
}
