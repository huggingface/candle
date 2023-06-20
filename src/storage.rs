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

#[derive(Debug)]
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

trait UnaryOp {
    const NAME: &'static str;
    fn f32(v1: f32) -> f32;
    fn f64(v1: f64) -> f64;
}

trait BinaryOp {
    const NAME: &'static str;
    fn f32(v1: f32, v2: f32) -> f32;
    fn f64(v1: f64, v2: f64) -> f64;
}

struct Add;
struct Mul;
struct Sqr;
struct Sqrt;

impl BinaryOp for Add {
    const NAME: &'static str = "add";
    fn f32(v1: f32, v2: f32) -> f32 {
        v1 + v2
    }
    fn f64(v1: f64, v2: f64) -> f64 {
        v1 + v2
    }
}

impl BinaryOp for Mul {
    const NAME: &'static str = "mul";
    fn f32(v1: f32, v2: f32) -> f32 {
        v1 * v2
    }
    fn f64(v1: f64, v2: f64) -> f64 {
        v1 * v2
    }
}

impl UnaryOp for Sqr {
    const NAME: &'static str = "sqr";
    fn f32(v1: f32) -> f32 {
        v1 * v1
    }
    fn f64(v1: f64) -> f64 {
        v1 * v1
    }
}

impl UnaryOp for Sqrt {
    const NAME: &'static str = "sqrt";
    fn f32(v1: f32) -> f32 {
        v1.sqrt()
    }
    fn f64(v1: f64) -> f64 {
        v1.sqrt()
    }
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

    fn unary_impl<B: UnaryOp>(&self, shape: &Shape, stride: &[usize]) -> Result<Self> {
        // TODO: Different code path for the contiguous case?
        match self {
            Storage::Cpu(storage) => match storage {
                CpuStorage::F32(storage) => {
                    let index = StridedIndex::new(shape.dims(), stride);
                    let data = index.map(|i| B::f32(storage[i])).collect();
                    Ok(Storage::Cpu(CpuStorage::F32(data)))
                }
                CpuStorage::F64(storage) => {
                    let index = StridedIndex::new(shape.dims(), stride);
                    let data = index.map(|i| B::f64(storage[i])).collect();
                    Ok(Storage::Cpu(CpuStorage::F64(data)))
                }
            },
        }
    }

    // TODO: Support broadcasting?
    fn binary_impl<B: BinaryOp>(
        &self,
        rhs: &Self,
        shape: &Shape,
        lhs_stride: &[usize],
        rhs_stride: &[usize],
    ) -> Result<Self> {
        self.same_device(rhs, B::NAME)?;
        self.same_dtype(rhs, B::NAME)?;
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
                        .map(|(lhs_i, rhs_i)| B::f32(lhs[lhs_i], rhs[rhs_i]))
                        .collect();
                    Ok(Storage::Cpu(CpuStorage::F32(data)))
                }
                (CpuStorage::F64(lhs), CpuStorage::F64(rhs)) => {
                    let lhs_index = StridedIndex::new(shape.dims(), lhs_stride);
                    let rhs_index = StridedIndex::new(shape.dims(), rhs_stride);
                    let data = lhs_index
                        .zip(rhs_index)
                        .map(|(lhs_i, rhs_i)| B::f64(lhs[lhs_i], rhs[rhs_i]))
                        .collect();
                    Ok(Storage::Cpu(CpuStorage::F64(data)))
                }
                _ => {
                    // This should be covered by the dtype check above.
                    Err(Error::DTypeMismatchBinaryOp {
                        lhs: lhs.dtype(),
                        rhs: rhs.dtype(),
                        op: B::NAME,
                    })
                }
            },
        }
    }

    pub(crate) fn add_impl(
        &self,
        rhs: &Self,
        shape: &Shape,
        lhs_stride: &[usize],
        rhs_stride: &[usize],
    ) -> Result<Self> {
        self.binary_impl::<Add>(rhs, shape, lhs_stride, rhs_stride)
    }

    pub(crate) fn mul_impl(
        &self,
        rhs: &Self,
        shape: &Shape,
        lhs_stride: &[usize],
        rhs_stride: &[usize],
    ) -> Result<Self> {
        self.binary_impl::<Mul>(rhs, shape, lhs_stride, rhs_stride)
    }

    pub(crate) fn sqr_impl(&self, shape: &Shape, stride: &[usize]) -> Result<Self> {
        self.unary_impl::<Sqr>(shape, stride)
    }

    pub(crate) fn sqrt_impl(&self, shape: &Shape, stride: &[usize]) -> Result<Self> {
        self.unary_impl::<Sqrt>(shape, stride)
    }
}
