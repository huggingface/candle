use crate::{DType, Device, Error, Result, Shape};

// TODO: Think about whether we would be better off with a dtype and
// a buffer as an owned slice of bytes.
#[derive(Debug, Clone)]
pub enum CpuStorage {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl CpuStorage {
    pub(crate) fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum Storage {
    Cpu(CpuStorage),
}

impl Storage {
    pub fn device(&self) -> Device {
        match self {
            Self::Cpu(_) => Device::Cpu,
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::Cpu(storage) => storage.dtype(),
        }
    }

    pub(crate) fn same_device(&self, rhs: &Self, op: &'static str) -> Result<()> {
        let lhs = self.device();
        let rhs = rhs.device();
        if lhs != rhs {
            Err(Error::DeviceMismatchBinaryOp { lhs, rhs, op })
        } else {
            Ok(())
        }
    }

    pub(crate) fn same_dtype(&self, rhs: &Self, op: &'static str) -> Result<()> {
        let lhs = self.dtype();
        let rhs = rhs.dtype();
        if lhs != rhs {
            Err(Error::DTypeMismatchBinaryOp { lhs, rhs, op })
        } else {
            Ok(())
        }
    }

    pub(crate) fn add_impl(
        &self,
        rhs: &Self,
        shape: &Shape,
        _lhs_stride: &[usize],
        _rhs_stride: &[usize],
    ) -> Result<Self> {
        self.same_device(rhs, "add")?;
        self.same_dtype(rhs, "add")?;
        // The ggml implementation has different paths based on whether the rhs is contiguous
        // or not, for now we only consider the general case but we should benchmark and do the
        // same if it helps.
        // https://github.com/ggerganov/llama.cpp/blob/aacdbd40562684665b6f7b8ba6695b7a2088bbb0/ggml.c#L7895
        match (self, rhs) {
            (Storage::Cpu(lhs), Storage::Cpu(rhs)) => match (lhs, rhs) {
                (CpuStorage::F32(_), CpuStorage::F32(_)) => {
                    let elem_count = shape.elem_count();
                    let data = vec![0f32; elem_count];
                    // TODO: properly fill data with the sum
                    Ok(Storage::Cpu(CpuStorage::F32(data)))
                }
                (CpuStorage::F64(_), CpuStorage::F64(_)) => {
                    let elem_count = shape.elem_count();
                    let data = vec![0f64; elem_count];
                    // TODO: properly fill data with the sum
                    Ok(Storage::Cpu(CpuStorage::F64(data)))
                }
                _ => {
                    // This should be covered by the dtype check above.
                    Err(Error::DTypeMismatchBinaryOp {
                        lhs: lhs.dtype(),
                        rhs: rhs.dtype(),
                        op: "add",
                    })
                }
            },
        }
    }

    pub(crate) fn mul_impl(
        &self,
        rhs: &Self,
        _shape: &Shape,
        _lhs_stride: &[usize],
        _rhs_stride: &[usize],
    ) -> Result<Self> {
        self.same_device(rhs, "mul")?;
        self.same_dtype(rhs, "mul")?;
        todo!()
    }
}
