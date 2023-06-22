use crate::{op, CpuStorage, CudaStorage, DType, Device, Error, Result, Shape};

// We do not want to implement Clone on Storage as cloning may fail because of
// out of memory. Instead try_clone should be used.
#[derive(Debug)]
pub enum Storage {
    Cpu(CpuStorage),
    Cuda(CudaStorage),
}

// <<<<<<< HEAD
// =======
// pub(crate) trait UnaryOp {
//     const NAME: &'static str;
//     fn f32(v1: f32) -> f32;
//     fn f64(v1: f64) -> f64;
// }
//
// pub(crate) trait BinaryOp {
//     const NAME: &'static str;
//     fn u32(v1: u32, v2: u32) -> u32;
//     fn f32(v1: f32, v2: f32) -> f32;
//     fn f64(v1: f64, v2: f64) -> f64;
// }
//
// struct Add;
// struct Div;
// struct Mul;
// struct Sub;
// struct Neg;
// struct Sqr;
// struct Sqrt;
//
// macro_rules! binary_op {
//     ($ty:ty, $name:literal, $fn:expr) => {
//         impl BinaryOp for $ty {
//             const NAME: &'static str = $name;
//             fn u32(v1: u32, v2: u32) -> u32 {
//                 $fn(v1, v2)
//             }
//             fn f32(v1: f32, v2: f32) -> f32 {
//                 $fn(v1, v2)
//             }
//             fn f64(v1: f64, v2: f64) -> f64 {
//                 $fn(v1, v2)
//             }
//         }
//     };
// }
//
// binary_op!(Add, "add", |v1, v2| v1 + v2);
// binary_op!(Sub, "sub", |v1, v2| v1 - v2);
// binary_op!(Mul, "mul", |v1, v2| v1 * v2);
// binary_op!(Div, "div", |v1, v2| v1 / v2);
//
// impl UnaryOp for Neg {
//     const NAME: &'static str = "neg";
//     fn f32(v1: f32) -> f32 {
//         -v1
//     }
//     fn f64(v1: f64) -> f64 {
//         -v1
//     }
// }
//
// impl UnaryOp for Sqr {
//     const NAME: &'static str = "sqr";
//     fn f32(v1: f32) -> f32 {
//         v1 * v1
//     }
//     fn f64(v1: f64) -> f64 {
//         v1 * v1
//     }
// }
//
// impl UnaryOp for Sqrt {
//     const NAME: &'static str = "sqrt";
//     fn f32(v1: f32) -> f32 {
//         v1.sqrt()
//     }
//     fn f64(v1: f64) -> f64 {
//         v1.sqrt()
//     }
// }
//
// >>>>>>> cf96abc ([WIP] First draft of what GPT2 might look like.)
impl Storage {
    pub fn try_clone(&self) -> Result<Self> {
        match self {
            Self::Cpu(storage) => Ok(Self::Cpu(storage.clone())),
            Self::Cuda(storage) => {
                let storage = storage.try_clone()?;
                Ok(Self::Cuda(storage))
            }
        }
    }

    pub fn device(&self) -> Device {
        match self {
            Self::Cpu(_) => Device::Cpu,
            Self::Cuda(storage) => Device::Cuda(storage.device().clone()),
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::Cpu(storage) => storage.dtype(),
            Self::Cuda(storage) => storage.dtype(),
        }
    }

    pub(crate) fn same_device(&self, rhs: &Self, op: &'static str) -> Result<()> {
        let lhs = self.device().location();
        let rhs = rhs.device().location();
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

    pub(crate) fn affine_impl(
        &self,
        shape: &Shape,
        stride: &[usize],
        mul: f64,
        add: f64,
    ) -> Result<Self> {
        // TODO: Different code path for the contiguous case?
        match self {
            Storage::Cpu(storage) => {
                let storage = storage.affine_impl(shape, stride, mul, add)?;
                Ok(Self::Cpu(storage))
            }
            Self::Cuda(storage) => {
                let storage = storage.affine_impl(shape, stride, mul, add)?;
                Ok(Self::Cuda(storage))
            }
        }
    }

    pub(crate) fn unary_impl<B: op::UnaryOp>(
        &self,
        shape: &Shape,
        stride: &[usize],
    ) -> Result<Self> {
        // TODO: Different code path for the contiguous case?
        match self {
            Storage::Cpu(storage) => {
                let storage = storage.unary_impl::<B>(shape, stride)?;
                Ok(Self::Cpu(storage))
            }
            Self::Cuda(storage) => {
                let storage = storage.unary_impl::<B>(shape, stride)?;
                Ok(Self::Cuda(storage))
            }
        }
    }

    // TODO: Support broadcasting?
    pub(crate) fn binary_impl<B: op::BinaryOp>(
        &self,
        rhs: &Self,
        shape: &Shape,
        lhs_stride: &[usize],
        rhs_stride: &[usize],
    ) -> Result<Self> {
        self.same_device(rhs, B::NAME)?;
        self.same_dtype(rhs, B::NAME)?;
        match (self, rhs) {
            (Storage::Cpu(lhs), Storage::Cpu(rhs)) => {
                let storage = lhs.binary_impl::<B>(rhs, shape, lhs_stride, rhs_stride)?;
                Ok(Self::Cpu(storage))
            }
            (Self::Cuda(lhs), Self::Cuda(rhs)) => {
                let storage = lhs.binary_impl::<B>(rhs, shape, lhs_stride, rhs_stride)?;
                Ok(Self::Cuda(storage))
            }
            (lhs, rhs) => {
                // Should not happen because of the same device check above but we're defensive
                // anyway.
                Err(Error::DeviceMismatchBinaryOp {
                    lhs: lhs.device().location(),
                    rhs: rhs.device().location(),
                    op: B::NAME,
                })
            }
        }
    }

    pub(crate) fn embedding_impl(
        &self,
        rhs: &Self,
        hidden_size: usize,
        vocab_size: usize,
    ) -> Result<Self> {
        self.same_device(rhs, "embedding")?;
        self.same_dtype(rhs, "embedding")?;
        match (self, rhs) {
            (Storage::Cpu(lhs), Storage::Cpu(rhs)) => {
                let storage = lhs.embedding_impl(rhs, hidden_size, vocab_size)?;
                Ok(Self::Cpu(storage))
            }
            (Self::Cuda(lhs), Self::Cuda(rhs)) => {
                let storage = lhs.embedding_impl(rhs, hidden_size, vocab_size)?;
                Ok(Self::Cuda(storage))
            }
            (lhs, rhs) => Err(Error::DeviceMismatchBinaryOp {
                lhs: lhs.device().location(),
                rhs: rhs.device().location(),
                op: "embedding",
            }),
        }
    }

    pub(crate) fn matmul_impl(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_stride: &[usize],
        rhs_stride: &[usize],
    ) -> Result<Self> {
        self.same_device(rhs, "matmul")?;
        self.same_dtype(rhs, "matmul")?;
        match (self, rhs) {
            (Self::Cpu(lhs), Self::Cpu(rhs)) => {
                let storage = lhs.matmul_impl(rhs, bmnk, lhs_stride, rhs_stride)?;
                Ok(Self::Cpu(storage))
            }
            (Self::Cuda(lhs), Self::Cuda(rhs)) => {
                let storage = lhs.matmul_impl(rhs, bmnk, lhs_stride, rhs_stride)?;
                Ok(Self::Cuda(storage))
            }
            (lhs, rhs) => Err(Error::DeviceMismatchBinaryOp {
                lhs: lhs.device().location(),
                rhs: rhs.device().location(),
                op: "matmul",
            }),
        }
    }

    // self, the source can be strided whereas dst is contiguous.
    pub(crate) fn copy_strided_src(
        &self,
        dst: &mut Self,
        src_shape: &Shape,
        src_stride: &[usize],
        dst_offset: usize,
    ) -> Result<()> {
        match (self, dst) {
            (Self::Cpu(src), Self::Cpu(dst)) => {
                src.copy_strided_src(dst, src_shape, src_stride, dst_offset)
            }
            (Self::Cuda(src), Self::Cuda(dst)) => {
                Ok(src.copy_strided_src(dst, src_shape, src_stride, dst_offset)?)
            }
            (lhs, rhs) => Err(Error::DeviceMismatchBinaryOp {
                lhs: lhs.device().location(),
                rhs: rhs.device().location(),
                op: "copy",
            }),
        }
    }
}
