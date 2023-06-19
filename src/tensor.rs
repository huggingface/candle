use crate::{op::Op, storage::Storage, DType, Device, Error, Result};
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

    pub fn shape1(&self) -> Result<usize> {
        let shape = self.shape();
        if shape.len() == 1 {
            Ok(shape[0])
        } else {
            Err(Error::UnexpectedNumberOfDims {
                expected: 1,
                got: shape.len(),
                shape: shape.to_vec(),
            })
        }
    }

    pub fn shape2(&self) -> Result<(usize, usize)> {
        let shape = self.shape();
        if shape.len() == 2 {
            Ok((shape[0], shape[1]))
        } else {
            Err(Error::UnexpectedNumberOfDims {
                expected: 2,
                got: shape.len(),
                shape: shape.to_vec(),
            })
        }
    }

    pub fn shape3(&self) -> Result<(usize, usize, usize)> {
        let shape = self.shape();
        if shape.len() == 3 {
            Ok((shape[0], shape[1], shape[2]))
        } else {
            Err(Error::UnexpectedNumberOfDims {
                expected: 3,
                got: shape.len(),
                shape: shape.to_vec(),
            })
        }
    }

    pub fn shape4(&self) -> Result<(usize, usize, usize, usize)> {
        let shape = self.shape();
        if shape.len() == 4 {
            Ok((shape[0], shape[1], shape[2], shape[4]))
        } else {
            Err(Error::UnexpectedNumberOfDims {
                expected: 4,
                got: shape.len(),
                shape: shape.to_vec(),
            })
        }
    }
}
