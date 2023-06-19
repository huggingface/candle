use crate::{DType, Storage};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
}

impl Device {
    pub(crate) fn zeros(&self, shape: &[usize], dtype: DType) -> Storage {
        match self {
            Device::Cpu => {
                let elem_count: usize = shape.iter().product();
                let buffer = vec![0; elem_count * dtype.size_in_bytes()];
                Storage::Cpu { dtype, buffer }
            }
        }
    }
}
