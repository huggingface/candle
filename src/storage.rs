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

pub(crate) struct StridedIndex<'a> {
    next_storage_index: Option<usize>,
    multi_index: Vec<usize>,
    dims: &'a [usize],
    stride: &'a [usize],
}

impl<'a> StridedIndex<'a> {
    pub(crate) fn new(dims: &'a [usize], stride: &'a [usize]) -> Self {
        let elem_count: usize = dims.iter().product();
        let next_storage_index = if elem_count == 0 {
            None
        } else {
            // This applies to the scalar case.
            Some(0)
        };
        StridedIndex {
            next_storage_index,
            multi_index: vec![0; dims.len()],
            dims,
            stride,
        }
    }
}

impl<'a> Iterator for StridedIndex<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let storage_index = match self.next_storage_index {
            None => return None,
            Some(storage_index) => storage_index,
        };
        let mut updated = false;
        for (multi_i, max_i) in self.multi_index.iter_mut().zip(self.dims.iter()).rev() {
            let next_i = *multi_i + 1;
            if next_i < *max_i {
                *multi_i = next_i;
                updated = true;
                break;
            } else {
                *multi_i = 0
            }
        }
        self.next_storage_index = if updated {
            let next_storage_index = self
                .multi_index
                .iter()
                .zip(self.stride.iter())
                .map(|(&x, &y)| x * y)
                .sum();
            Some(next_storage_index)
        } else {
            None
        };
        Some(storage_index)
    }
}

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

    // TODO: Support broadcasting?
    pub(crate) fn add_impl(
        &self,
        rhs: &Self,
        shape: &Shape,
        lhs_stride: &[usize],
        rhs_stride: &[usize],
    ) -> Result<Self> {
        self.same_device(rhs, "add")?;
        self.same_dtype(rhs, "add")?;
        // The ggml implementation has different paths based on whether the rhs is contiguous
        // or not, for now we only consider the general case but we should benchmark and do the
        // same if it helps.
        // https://github.com/ggerganov/llama.cpp/blob/aacdbd40562684665b6f7b8ba6695b7a2088bbb0/ggml.c#L7895
        match (self, rhs) {
            (Storage::Cpu(lhs), Storage::Cpu(rhs)) => match (lhs, rhs) {
                (CpuStorage::F32(lhs), CpuStorage::F32(rhs)) => {
                    let lhs_index = StridedIndex::new(shape.dims(), lhs_stride);
                    let rhs_index = StridedIndex::new(shape.dims(), rhs_stride);
                    let data = lhs_index
                        .zip(rhs_index)
                        .map(|(lhs_i, rhs_i)| lhs[lhs_i] + rhs[rhs_i])
                        .collect();
                    Ok(Storage::Cpu(CpuStorage::F32(data)))
                }
                (CpuStorage::F64(lhs), CpuStorage::F64(rhs)) => {
                    let lhs_index = StridedIndex::new(shape.dims(), lhs_stride);
                    let rhs_index = StridedIndex::new(shape.dims(), rhs_stride);
                    let data = lhs_index
                        .zip(rhs_index)
                        .map(|(lhs_i, rhs_i)| lhs[lhs_i] + rhs[rhs_i])
                        .collect();
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

    // TODO: Support broadcasting?
    pub(crate) fn mul_impl(
        &self,
        rhs: &Self,
        shape: &Shape,
        lhs_stride: &[usize],
        rhs_stride: &[usize],
    ) -> Result<Self> {
        self.same_device(rhs, "mul")?;
        self.same_dtype(rhs, "mul")?;
        // TODO: share this code with the add implementation, using a macro or a trait?
        match (self, rhs) {
            (Storage::Cpu(lhs), Storage::Cpu(rhs)) => match (lhs, rhs) {
                (CpuStorage::F32(lhs), CpuStorage::F32(rhs)) => {
                    let lhs_index = StridedIndex::new(shape.dims(), lhs_stride);
                    let rhs_index = StridedIndex::new(shape.dims(), rhs_stride);
                    let data = lhs_index
                        .zip(rhs_index)
                        .map(|(lhs_i, rhs_i)| lhs[lhs_i] * rhs[rhs_i])
                        .collect();
                    Ok(Storage::Cpu(CpuStorage::F32(data)))
                }
                (CpuStorage::F64(lhs), CpuStorage::F64(rhs)) => {
                    let lhs_index = StridedIndex::new(shape.dims(), lhs_stride);
                    let rhs_index = StridedIndex::new(shape.dims(), rhs_stride);
                    let data = lhs_index
                        .zip(rhs_index)
                        .map(|(lhs_i, rhs_i)| lhs[lhs_i] * rhs[rhs_i])
                        .collect();
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
}
