use crate::{op::Op, shape, storage::Storage, DType, Device, Result};
use std::sync::Arc;

#[allow(dead_code)]
pub(crate) struct Tensor_ {
    storage: Storage,
    shape: shape::Shape,
    stride: Vec<usize>,
    op: Option<Op>,
}

pub struct Tensor(Arc<Tensor_>);

impl Tensor {
    pub fn zeros<S: Into<shape::Shape>>(shape: S, dtype: DType, device: Device) -> Self {
        let shape = shape.into();
        let storage = device.zeros(&shape, dtype);
        let rank = shape.rank();
        let tensor_ = Tensor_ {
            storage,
            shape,
            stride: vec![1; rank],
            op: None,
        };
        Self(Arc::new(tensor_))
    }

    pub fn new<A: crate::device::NdArray>(array: A, device: Device) -> Result<Self> {
        let shape = array.shape()?;
        let storage = device.tensor(array);
        let rank = shape.rank();
        let tensor_ = Tensor_ {
            storage,
            shape,
            stride: vec![1; rank],
            op: None,
        };
        Ok(Self(Arc::new(tensor_)))
    }

    pub fn dtype(&self) -> DType {
        self.0.storage.dtype()
    }

    pub fn device(&self) -> Device {
        self.0.storage.device()
    }

    pub fn shape(&self) -> &shape::Shape {
        &self.0.shape
    }

    pub fn dims(&self) -> &[usize] {
        self.shape().dims()
    }

    pub fn stride(&self) -> &[usize] {
        &self.0.stride
    }

    pub fn rank(&self) -> usize {
        self.shape().rank()
    }

    pub fn elem_count(&self) -> usize {
        self.shape().elem_count()
    }
}
