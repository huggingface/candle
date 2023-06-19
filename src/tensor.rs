use crate::{op::Op, storage::Storage, DType, Device};
use std::sync::Arc;

#[allow(dead_code)]
pub(crate) struct Tensor_ {
    storage: Storage,
    shape: Vec<usize>,
    stride: Vec<usize>,
    op: Option<Op>,
}

pub struct Tensor(Arc<Tensor_>);

impl Tensor {
    pub fn zeros(shape: &[usize], dtype: DType, device: Device) -> Self {
        let storage = device.zeros(shape, dtype);
        let tensor_ = Tensor_ {
            storage,
            shape: shape.to_vec(),
            stride: vec![1; shape.len()],
            op: None,
        };
        Tensor(Arc::new(tensor_))
    }

    pub fn dtype(&self) -> DType {
        self.0.storage.dtype()
    }

    pub fn device(&self) -> Device {
        self.0.storage.device()
    }

    pub fn shape(&self) -> &[usize] {
        &self.0.shape
    }

    pub fn stride(&self) -> &[usize] {
        &self.0.stride
    }

    pub fn rank(&self) -> usize {
        self.0.shape.len()
    }

    pub fn elem_count(&self) -> usize {
        self.0.shape.iter().product()
    }
}
