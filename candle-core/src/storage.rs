use crate::backend::BackendStorage;
use crate::{op, CpuStorage, CudaStorage, DType, Device, Error, Layout, Result, Shape};

// We do not want to implement Clone on Storage as cloning may fail because of
// out of memory. Instead try_clone should be used.
#[derive(Debug)]
pub enum Storage {
    Cpu(CpuStorage),
    Cuda(CudaStorage),
}

impl Storage {
    pub fn try_clone(&self, layout: &Layout) -> Result<Self> {
        match self {
            Self::Cpu(storage) => Ok(Self::Cpu(storage.clone())),
            Self::Cuda(storage) => {
                let storage = storage.try_clone(layout)?;
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

    pub(crate) fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => {
                let storage = storage.affine(layout, mul, add)?;
                Ok(Self::Cpu(storage))
            }
            Self::Cuda(storage) => {
                let storage = storage.affine(layout, mul, add)?;
                Ok(Self::Cuda(storage))
            }
        }
    }

    pub(crate) fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => {
                let storage = storage.elu(layout, alpha)?;
                Ok(Self::Cpu(storage))
            }
            Self::Cuda(storage) => {
                let storage = storage.elu(layout, alpha)?;
                Ok(Self::Cuda(storage))
            }
        }
    }

    pub(crate) fn sum(&self, layout: &Layout, s: &[usize]) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => {
                let storage = storage.sum(layout, s)?;
                Ok(Self::Cpu(storage))
            }
            Self::Cuda(storage) => {
                let storage = storage.sum(layout, s)?;
                Ok(Self::Cuda(storage))
            }
        }
    }

    // This assumes a contiguous layout and no offset.
    pub(crate) fn divide_by_sum_over_dim(&mut self, shape: &Shape, dim: usize) -> Result<()> {
        match self {
            Storage::Cpu(storage) => storage.divide_by_sum_over_dim(shape, dim)?,
            Self::Cuda(storage) => storage.divide_by_sum_over_dim(shape, dim)?,
        }
        Ok(())
    }

    pub(crate) fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => {
                let storage = storage.to_dtype(layout, dtype)?;
                Ok(Self::Cpu(storage))
            }
            Self::Cuda(storage) => {
                let storage = storage.to_dtype(layout, dtype)?;
                Ok(Self::Cuda(storage))
            }
        }
    }

    pub(crate) fn unary_impl<B: op::UnaryOp>(&self, layout: &Layout) -> Result<Self> {
        // TODO: Different code path for the contiguous case?
        match self {
            Storage::Cpu(storage) => {
                let storage = storage.unary_impl::<B>(layout)?;
                Ok(Self::Cpu(storage))
            }
            Self::Cuda(storage) => {
                let storage = storage.unary_impl::<B>(layout)?;
                Ok(Self::Cuda(storage))
            }
        }
    }

    pub(crate) fn binary_impl<B: op::BinaryOp>(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        self.same_device(rhs, B::NAME)?;
        self.same_dtype(rhs, B::NAME)?;
        match (self, rhs) {
            (Storage::Cpu(lhs), Storage::Cpu(rhs)) => {
                let storage = lhs.binary_impl::<B>(rhs, lhs_layout, rhs_layout)?;
                Ok(Self::Cpu(storage))
            }
            (Self::Cuda(lhs), Self::Cuda(rhs)) => {
                let storage = lhs.binary_impl::<B>(rhs, lhs_layout, rhs_layout)?;
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

    pub(crate) fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        self.same_device(kernel, "conv1d")?;
        self.same_dtype(kernel, "conv1d")?;
        match (self, &kernel) {
            (Storage::Cpu(inp), Storage::Cpu(kernel)) => {
                let s = inp.conv1d(l, kernel, kernel_l, params)?;
                Ok(Self::Cpu(s))
            }
            (Storage::Cuda(inp), Storage::Cuda(kernel)) => {
                let s = inp.conv1d(l, kernel, kernel_l, params)?;
                Ok(Self::Cuda(s))
            }
            (lhs, rhs) => Err(Error::DeviceMismatchBinaryOp {
                lhs: lhs.device().location(),
                rhs: rhs.device().location(),
                op: "conv1d",
            }),
        }
    }

    pub(crate) fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        layout_t: &Layout,
        f: &Self,
        layout_f: &Layout,
    ) -> Result<Self> {
        self.same_device(t, "where")?;
        self.same_device(f, "where")?;
        t.same_dtype(f, "where")?;
        match (self, t, f) {
            (Storage::Cpu(cond), Storage::Cpu(t), Storage::Cpu(f)) => {
                let storage = cond.where_cond(layout, t, layout_t, f, layout_f)?;
                Ok(Self::Cpu(storage))
            }
            (Self::Cuda(cond), Self::Cuda(t), Self::Cuda(f)) => {
                let storage = cond.where_cond(layout, t, layout_t, f, layout_f)?;
                Ok(Self::Cuda(storage))
            }
            (_, lhs, rhs) => Err(Error::DeviceMismatchBinaryOp {
                lhs: lhs.device().location(),
                rhs: rhs.device().location(),
                op: "where",
            }),
        }
    }

    pub(crate) fn embedding(&self, layout: &Layout, rhs: &Self, rhs_l: &Layout) -> Result<Self> {
        self.same_device(rhs, "embedding")?;
        match (self, rhs) {
            (Storage::Cpu(lhs), Storage::Cpu(rhs)) => {
                let storage = lhs.embedding(layout, rhs, rhs_l)?;
                Ok(Self::Cpu(storage))
            }
            (Self::Cuda(lhs), Self::Cuda(rhs)) => {
                let storage = lhs.embedding(layout, rhs, rhs_l)?;
                Ok(Self::Cuda(storage))
            }
            (lhs, rhs) => Err(Error::DeviceMismatchBinaryOp {
                lhs: lhs.device().location(),
                rhs: rhs.device().location(),
                op: "embedding",
            }),
        }
    }

    pub(crate) fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        self.same_device(rhs, "matmul")?;
        self.same_dtype(rhs, "matmul")?;
        match (self, rhs) {
            (Self::Cpu(lhs), Self::Cpu(rhs)) => {
                let storage = lhs.matmul(rhs, bmnk, lhs_layout, rhs_layout)?;
                Ok(Self::Cpu(storage))
            }
            (Self::Cuda(lhs), Self::Cuda(rhs)) => {
                let storage = lhs.matmul(rhs, bmnk, lhs_layout, rhs_layout)?;
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
        dst_offset: usize,
        src_l: &Layout,
    ) -> Result<()> {
        match (self, dst) {
            (Self::Cpu(src), Self::Cpu(dst)) => src.copy_strided_src(dst, dst_offset, src_l),
            (Self::Cuda(src), Self::Cuda(dst)) => Ok(src.copy_strided_src(dst, dst_offset, src_l)?),
            (lhs, rhs) => Err(Error::DeviceMismatchBinaryOp {
                lhs: lhs.device().location(),
                rhs: rhs.device().location(),
                op: "copy",
            }),
        }
    }
}
