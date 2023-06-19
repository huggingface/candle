use crate::{
    storage::{CpuStorage, Storage},
    DType,
};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
}

impl Device {
    pub(crate) fn zeros(&self, shape: &[usize], dtype: DType) -> Storage {
        match self {
            Device::Cpu => {
                let elem_count: usize = shape.iter().product();
                let storage = match dtype {
                    DType::F32 => {
                        let data = vec![0f32; elem_count];
                        CpuStorage::F32(data)
                    }
                    DType::F64 => {
                        let data = vec![0f64; elem_count];
                        CpuStorage::F64(data)
                    }
                };
                Storage::Cpu(storage)
            }
        }
    }
}
