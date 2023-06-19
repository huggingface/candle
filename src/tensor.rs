use crate::{DType, Device, Storage};

pub struct Tensor {
    storage: Storage,
    shape: Vec<usize>,
    stride: Vec<usize>,
}

impl Tensor {
    pub fn zeros(shape: &[usize], dtype: DType, device: Device) -> Self {
        let storage = device.zeros(shape, dtype);
        Self {
            storage,
            shape: shape.to_vec(),
            stride: vec![1; shape.len()],
        }
    }

    pub fn device(&self) -> Device {
        match self.storage {
            Storage::Cpu { .. } => Device::Cpu,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn stride(&self) -> &[usize] {
        &self.stride
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn elem_count(&self) -> usize {
        self.shape.iter().product()
    }
}
