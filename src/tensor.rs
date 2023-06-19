#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
}

impl DType {
    pub fn size_in_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

pub struct Tensor {
    storage: Storage,
    shape: Vec<usize>,
    stride: Vec<usize>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
}

#[allow(dead_code)]
enum Storage {
    Cpu { dtype: DType, buffer: Vec<u8> },
}

impl Tensor {
    pub fn zeros(shape: &[usize], dtype: DType, device: Device) -> Self {
        let storage = match device {
            Device::Cpu => {
                let elem_count: usize = shape.iter().product();
                let buffer = vec![0; elem_count * dtype.size_in_bytes()];
                Storage::Cpu { dtype, buffer }
            }
        };
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
