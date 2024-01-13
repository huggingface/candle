use crate::backend::{BackendDevice, BackendStorage};
use crate::custom_backend::CustomStorage;
use crate::op::{self, CmpOp, CustomOp1, CustomOp2, CustomOp3, ReduceOp};
use crate::{CpuStorage, CudaStorage, DType, Device, Error, Layout, MetalStorage, Result, Shape};

// We do not want to implement Clone on Storage as cloning may fail because of
// out of memory. Instead try_clone should be used.
#[derive(Debug)]
pub enum Storage {
    Cpu(CpuStorage),
    Cuda(CudaStorage),
    Metal(MetalStorage),
    Custom(CustomStorage),
}

impl Storage {
    pub(crate) fn same_device(&self, rhs: &Self, op: &'static str) -> Result<()> {
        let lhs = self.device().location();
        let rhs = rhs.device().location();
        if lhs != rhs {
            Err(Error::DeviceMismatchBinaryOp { lhs, rhs, op }.bt())
        } else {
            Ok(())
        }
    }

    pub(crate) fn same_dtype(&self, rhs: &Self, op: &'static str) -> Result<()> {
        let lhs = self.dtype();
        let rhs = rhs.dtype();
        if lhs != rhs {
            Err(Error::DTypeMismatchBinaryOp { lhs, rhs, op }.bt())
        } else {
            Ok(())
        }
    }
}

impl BackendStorage for Storage {
    type Device = Device;

    fn try_clone(&self, layout: &Layout) -> Result<Self> {
        match self {
            Self::Cpu(storage) => Ok(Self::Cpu(storage.clone())),
            Self::Cuda(storage) => {
                let storage = storage.try_clone(layout)?;
                Ok(Self::Cuda(storage))
            }
            Self::Metal(storage) => {
                let storage = storage.try_clone(layout)?;
                Ok(Self::Metal(storage))
            }
            Self::Custom(storage) => {
                let storage = storage.try_clone(layout)?;
                Ok(Self::Custom(storage))
            }
        }
    }

    fn device(&self) -> Device {
        match self {
            Self::Cpu(_) => Device::Cpu,
            Self::Cuda(storage) => Device::Cuda(storage.device().clone()),
            Self::Metal(storage) => Device::Metal(storage.device().clone()),
            Self::Custom(storage) => Device::Custom(storage.device().clone()),
        }
    }

    fn dtype(&self) -> DType {
        match self {
            Self::Cpu(storage) => storage.dtype(),
            Self::Cuda(storage) => storage.dtype(),
            Self::Metal(storage) => storage.dtype(),
            Self::Custom(storage) => storage.dtype(),
        }
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        Ok(match self {
            Storage::Cpu(storage) => storage.clone(),
            Self::Cuda(storage) => storage.to_cpu_storage()?,
            Self::Metal(storage) => storage.to_cpu_storage()?,
            Self::Custom(storage) => storage.to_cpu_storage()?,
        })
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => {
                let storage = storage.affine(layout, mul, add)?;
                Ok(Self::Cpu(storage))
            }
            Self::Cuda(storage) => {
                let storage = storage.affine(layout, mul, add)?;
                Ok(Self::Cuda(storage))
            }
            Self::Metal(storage) => {
                let storage = storage.affine(layout, mul, add)?;
                Ok(Self::Metal(storage))
            }
            Self::Custom(storage) => {
                let storage = storage.affine(layout, mul, add)?;
                Ok(Self::Custom(storage))
            }
        }
    }

    fn powf(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => {
                let storage = storage.powf(layout, alpha)?;
                Ok(Self::Cpu(storage))
            }
            Self::Cuda(storage) => {
                let storage = storage.powf(layout, alpha)?;
                Ok(Self::Cuda(storage))
            }
            Self::Metal(storage) => {
                let storage = storage.powf(layout, alpha)?;
                Ok(Self::Metal(storage))
            }
            Self::Custom(storage) => {
                let storage = storage.powf(layout, alpha)?;
                Ok(Self::Custom(storage))
            }
        }
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => {
                let storage = storage.elu(layout, alpha)?;
                Ok(Self::Cpu(storage))
            }
            Self::Cuda(storage) => {
                let storage = storage.elu(layout, alpha)?;
                Ok(Self::Cuda(storage))
            }
            Self::Metal(storage) => {
                let storage = storage.elu(layout, alpha)?;
                Ok(Self::Metal(storage))
            }
            Self::Custom(storage) => {
                let storage = storage.elu(layout, alpha)?;
                Ok(Self::Custom(storage))
            }
        }
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_layout: &Layout, rhs_layout: &Layout) -> Result<Self> {
        self.same_device(rhs, "cmp")?;
        self.same_dtype(rhs, "cmp")?;
        match (self, rhs) {
            (Storage::Cpu(lhs), Storage::Cpu(rhs)) => {
                let storage = lhs.cmp(op, rhs, lhs_layout, rhs_layout)?;
                Ok(Self::Cpu(storage))
            }
            (Self::Cuda(lhs), Self::Cuda(rhs)) => {
                let storage = lhs.cmp(op, rhs, lhs_layout, rhs_layout)?;
                Ok(Self::Cuda(storage))
            }
            (Self::Metal(lhs), Self::Metal(rhs)) => {
                let storage = lhs.cmp(op, rhs, lhs_layout, rhs_layout)?;
                Ok(Self::Metal(storage))
            }
            (Self::Custom(lhs), Self::Custom(rhs)) => {
                let storage = lhs.cmp(op, rhs, lhs_layout, rhs_layout)?;
                Ok(Self::Custom(storage))
            }
            (lhs, rhs) => {
                // Should not happen because of the same device check above but we're defensive
                // anyway.
                Err(Error::DeviceMismatchBinaryOp {
                    lhs: lhs.device().location(),
                    rhs: rhs.device().location(),
                    op: "cmp",
                }
                .bt())
            }
        }
    }

    fn reduce(&self, op: ReduceOp, layout: &Layout, s: &[usize]) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => {
                let storage = storage.reduce(op, layout, s)?;
                Ok(Self::Cpu(storage))
            }
            Self::Cuda(storage) => {
                let storage = storage.reduce(op, layout, s)?;
                Ok(Self::Cuda(storage))
            }
            Self::Metal(storage) => {
                let storage = storage.reduce(op, layout, s)?;
                Ok(Self::Metal(storage))
            }
            Self::Custom(storage) => {
                let storage = storage.reduce(op, layout, s)?;
                Ok(Self::Custom(storage))
            }
        }
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => {
                let storage = storage.to_dtype(layout, dtype)?;
                Ok(Self::Cpu(storage))
            }
            Self::Cuda(storage) => {
                let storage = storage.to_dtype(layout, dtype)?;
                Ok(Self::Cuda(storage))
            }
            Self::Metal(storage) => {
                let storage = storage.to_dtype(layout, dtype)?;
                Ok(Self::Metal(storage))
            }
            Self::Custom(storage) => {
                let storage = storage.to_dtype(layout, dtype)?;
                Ok(Self::Custom(storage))
            }
        }
    }

    fn unary_impl<B: op::UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => {
                let storage = storage.unary_impl::<B>(layout)?;
                Ok(Self::Cpu(storage))
            }
            Self::Cuda(storage) => {
                let storage = storage.unary_impl::<B>(layout)?;
                Ok(Self::Cuda(storage))
            }
            Self::Metal(storage) => {
                let storage = storage.unary_impl::<B>(layout)?;
                Ok(Self::Metal(storage))
            }
            Self::Custom(storage) => {
                let storage = storage.unary_impl::<B>(layout)?;
                Ok(Self::Custom(storage))
            }
        }
    }

    fn binary_impl<B: op::BinaryOpT>(
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
            (Self::Metal(lhs), Self::Metal(rhs)) => {
                let storage = lhs.binary_impl::<B>(rhs, lhs_layout, rhs_layout)?;
                Ok(Self::Metal(storage))
            }
            (Self::Custom(lhs), Self::Custom(rhs)) => {
                let storage = lhs.binary_impl::<B>(rhs, lhs_layout, rhs_layout)?;
                Ok(Self::Custom(storage))
            }
            (lhs, rhs) => {
                // Should not happen because of the same device check above but we're defensive
                // anyway.
                Err(Error::DeviceMismatchBinaryOp {
                    lhs: lhs.device().location(),
                    rhs: rhs.device().location(),
                    op: B::NAME,
                }
                .bt())
            }
        }
    }

    fn conv1d(
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
            (Storage::Metal(inp), Storage::Metal(kernel)) => {
                let s = inp.conv1d(l, kernel, kernel_l, params)?;
                Ok(Self::Metal(s))
            }
            (Storage::Custom(inp), Storage::Custom(kernel)) => {
                let s = inp.conv1d(l, kernel, kernel_l, params)?;
                Ok(Self::Custom(s))
            }
            (lhs, rhs) => Err(Error::DeviceMismatchBinaryOp {
                lhs: lhs.device().location(),
                rhs: rhs.device().location(),
                op: "conv1d",
            }
            .bt()),
        }
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        self.same_device(kernel, "conv-transpose1d")?;
        self.same_dtype(kernel, "conv-transpose1d")?;
        match (self, &kernel) {
            (Storage::Cpu(inp), Storage::Cpu(kernel)) => {
                let s = inp.conv_transpose1d(l, kernel, kernel_l, params)?;
                Ok(Self::Cpu(s))
            }
            (Storage::Cuda(inp), Storage::Cuda(kernel)) => {
                let s = inp.conv_transpose1d(l, kernel, kernel_l, params)?;
                Ok(Self::Cuda(s))
            }
            (Storage::Metal(inp), Storage::Metal(kernel)) => {
                let s = inp.conv_transpose1d(l, kernel, kernel_l, params)?;
                Ok(Self::Metal(s))
            }
            (Storage::Custom(inp), Storage::Custom(kernel)) => {
                let s = inp.conv_transpose1d(l, kernel, kernel_l, params)?;
                Ok(Self::Custom(s))
            }
            (lhs, rhs) => Err(Error::DeviceMismatchBinaryOp {
                lhs: lhs.device().location(),
                rhs: rhs.device().location(),
                op: "conv-transpose1d",
            }
            .bt()),
        }
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        self.same_device(kernel, "conv2d")?;
        self.same_dtype(kernel, "conv2d")?;
        match (self, &kernel) {
            (Storage::Cpu(inp), Storage::Cpu(kernel)) => {
                let s = inp.conv2d(l, kernel, kernel_l, params)?;
                Ok(Self::Cpu(s))
            }
            (Storage::Cuda(inp), Storage::Cuda(kernel)) => {
                let s = inp.conv2d(l, kernel, kernel_l, params)?;
                Ok(Self::Cuda(s))
            }
            (Storage::Metal(inp), Storage::Metal(kernel)) => {
                let s = inp.conv2d(l, kernel, kernel_l, params)?;
                Ok(Self::Metal(s))
            }
            (Storage::Custom(inp), Storage::Custom(kernel)) => {
                let s = inp.conv2d(l, kernel, kernel_l, params)?;
                Ok(Self::Custom(s))
            }
            (lhs, rhs) => Err(Error::DeviceMismatchBinaryOp {
                lhs: lhs.device().location(),
                rhs: rhs.device().location(),
                op: "conv2d",
            }
            .bt()),
        }
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        self.same_device(kernel, "conv_transpose2d")?;
        self.same_dtype(kernel, "conv_transpose2d")?;
        match (self, &kernel) {
            (Storage::Cpu(inp), Storage::Cpu(kernel)) => {
                let s = inp.conv_transpose2d(l, kernel, kernel_l, params)?;
                Ok(Self::Cpu(s))
            }
            (Storage::Cuda(inp), Storage::Cuda(kernel)) => {
                let s = inp.conv_transpose2d(l, kernel, kernel_l, params)?;
                Ok(Self::Cuda(s))
            }
            (Storage::Metal(inp), Storage::Metal(kernel)) => {
                let s = inp.conv_transpose2d(l, kernel, kernel_l, params)?;
                Ok(Self::Metal(s))
            }
            (Storage::Custom(inp), Storage::Custom(kernel)) => {
                let s = inp.conv_transpose2d(l, kernel, kernel_l, params)?;
                Ok(Self::Custom(s))
            }
            (lhs, rhs) => Err(Error::DeviceMismatchBinaryOp {
                lhs: lhs.device().location(),
                rhs: rhs.device().location(),
                op: "conv_transpose2d",
            }
            .bt()),
        }
    }

    fn avg_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => {
                let storage = storage.avg_pool2d(layout, kernel_size, stride)?;
                Ok(Self::Cpu(storage))
            }
            Self::Cuda(storage) => {
                let storage = storage.avg_pool2d(layout, kernel_size, stride)?;
                Ok(Self::Cuda(storage))
            }
            Self::Metal(storage) => {
                let storage = storage.avg_pool2d(layout, kernel_size, stride)?;
                Ok(Self::Metal(storage))
            }
            Self::Custom(storage) => {
                let storage = storage.avg_pool2d(layout, kernel_size, stride)?;
                Ok(Self::Custom(storage))
            }
        }
    }

    fn max_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => {
                let storage = storage.max_pool2d(layout, kernel_size, stride)?;
                Ok(Self::Cpu(storage))
            }
            Self::Cuda(storage) => {
                let storage = storage.max_pool2d(layout, kernel_size, stride)?;
                Ok(Self::Cuda(storage))
            }
            Self::Metal(storage) => {
                let storage = storage.max_pool2d(layout, kernel_size, stride)?;
                Ok(Self::Metal(storage))
            }
            Self::Custom(storage) => {
                let storage = storage.max_pool2d(layout, kernel_size, stride)?;
                Ok(Self::Custom(storage))
            }
        }
    }

    fn upsample_nearest1d(&self, layout: &Layout, sz: usize) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => {
                let storage = storage.upsample_nearest1d(layout, sz)?;
                Ok(Self::Cpu(storage))
            }
            Self::Cuda(storage) => {
                let storage = storage.upsample_nearest1d(layout, sz)?;
                Ok(Self::Cuda(storage))
            }
            Self::Metal(storage) => {
                let storage = storage.upsample_nearest1d(layout, sz)?;
                Ok(Self::Metal(storage))
            }
            Self::Custom(storage) => {
                let storage = storage.upsample_nearest1d(layout, sz)?;
                Ok(Self::Custom(storage))
            }
        }
    }

    fn upsample_nearest2d(&self, layout: &Layout, h: usize, w: usize) -> Result<Self> {
        match self {
            Storage::Cpu(storage) => {
                let storage = storage.upsample_nearest2d(layout, h, w)?;
                Ok(Self::Cpu(storage))
            }
            Self::Cuda(storage) => {
                let storage = storage.upsample_nearest2d(layout, h, w)?;
                Ok(Self::Cuda(storage))
            }
            Self::Metal(storage) => {
                let storage = storage.upsample_nearest2d(layout, h, w)?;
                Ok(Self::Metal(storage))
            }
            Self::Custom(storage) => {
                let storage = storage.upsample_nearest2d(layout, h, w)?;
                Ok(Self::Custom(storage))
            }
        }
    }

    fn where_cond(
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
            (Self::Metal(cond), Self::Metal(t), Self::Metal(f)) => {
                let storage = cond.where_cond(layout, t, layout_t, f, layout_f)?;
                Ok(Self::Metal(storage))
            }
            (Self::Custom(cond), Self::Custom(t), Self::Custom(f)) => {
                let storage = cond.where_cond(layout, t, layout_t, f, layout_f)?;
                Ok(Self::Custom(storage))
            }
            (_, lhs, rhs) => Err(Error::DeviceMismatchBinaryOp {
                lhs: lhs.device().location(),
                rhs: rhs.device().location(),
                op: "where",
            }
            .bt()),
        }
    }

    fn gather(&self, l: &Layout, indexes: &Self, indexes_l: &Layout, d: usize) -> Result<Self> {
        self.same_device(indexes, "index-add")?;
        match (self, indexes) {
            (Self::Cpu(s), Self::Cpu(indexes)) => {
                let storage = s.gather(l, indexes, indexes_l, d)?;
                Ok(Self::Cpu(storage))
            }
            (Self::Cuda(s), Self::Cuda(indexes)) => {
                let storage = s.gather(l, indexes, indexes_l, d)?;
                Ok(Self::Cuda(storage))
            }
            (Self::Metal(s), Self::Metal(indexes)) => {
                let storage = s.gather(l, indexes, indexes_l, d)?;
                Ok(Self::Metal(storage))
            }
            (Self::Custom(s), Self::Custom(indexes)) => {
                let storage = s.gather(l, indexes, indexes_l, d)?;
                Ok(Self::Custom(storage))
            }
            _ => unreachable!(),
        }
    }

    fn scatter_add(
        &self,
        l: &Layout,
        indexes: &Self,
        indexes_l: &Layout,
        source: &Self,
        source_l: &Layout,
        d: usize,
    ) -> Result<Self> {
        self.same_device(indexes, "scatter-add")?;
        self.same_device(source, "scatter-add")?;
        match (self, indexes, source) {
            (Self::Cpu(s), Self::Cpu(indexes), Self::Cpu(source)) => {
                let storage = s.scatter_add(l, indexes, indexes_l, source, source_l, d)?;
                Ok(Self::Cpu(storage))
            }
            (Self::Cuda(s), Self::Cuda(indexes), Self::Cuda(source)) => {
                let storage = s.scatter_add(l, indexes, indexes_l, source, source_l, d)?;
                Ok(Self::Cuda(storage))
            }
            (Self::Metal(s), Self::Metal(indexes), Self::Metal(source)) => {
                let storage = s.scatter_add(l, indexes, indexes_l, source, source_l, d)?;
                Ok(Self::Metal(storage))
            }
            (Self::Custom(s), Self::Custom(indexes), Self::Custom(source)) => {
                let storage = s.scatter_add(l, indexes, indexes_l, source, source_l, d)?;
                Ok(Self::Custom(storage))
            }
            _ => unreachable!(),
        }
    }

    fn index_add(
        &self,
        l: &Layout,
        indexes: &Self,
        indexes_l: &Layout,
        source: &Self,
        source_l: &Layout,
        d: usize,
    ) -> Result<Self> {
        self.same_device(indexes, "index-add")?;
        self.same_device(source, "index-add")?;
        match (self, indexes, source) {
            (Self::Cpu(s), Self::Cpu(indexes), Self::Cpu(source)) => {
                let storage = s.index_add(l, indexes, indexes_l, source, source_l, d)?;
                Ok(Self::Cpu(storage))
            }
            (Self::Cuda(s), Self::Cuda(indexes), Self::Cuda(source)) => {
                let storage = s.index_add(l, indexes, indexes_l, source, source_l, d)?;
                Ok(Self::Cuda(storage))
            }
            (Self::Metal(s), Self::Metal(indexes), Self::Metal(source)) => {
                let storage = s.index_add(l, indexes, indexes_l, source, source_l, d)?;
                Ok(Self::Metal(storage))
            }
            (Self::Custom(s), Self::Custom(indexes), Self::Custom(source)) => {
                let storage = s.index_add(l, indexes, indexes_l, source, source_l, d)?;
                Ok(Self::Custom(storage))
            }
            _ => unreachable!(),
        }
    }

    fn index_select(&self, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout, d: usize) -> Result<Self> {
        self.same_device(rhs, "index-select")?;
        match (self, rhs) {
            (Self::Cpu(lhs), Self::Cpu(rhs)) => {
                let storage = lhs.index_select(rhs, lhs_l, rhs_l, d)?;
                Ok(Self::Cpu(storage))
            }
            (Self::Cuda(lhs), Self::Cuda(rhs)) => {
                let storage = lhs.index_select(rhs, lhs_l, rhs_l, d)?;
                Ok(Self::Cuda(storage))
            }
            (Self::Metal(lhs), Self::Metal(rhs)) => {
                let storage = lhs.index_select(rhs, lhs_l, rhs_l, d)?;
                Ok(Self::Metal(storage))
            }
            (Self::Custom(lhs), Self::Custom(rhs)) => {
                let storage = lhs.index_select(rhs, lhs_l, rhs_l, d)?;
                Ok(Self::Custom(storage))
            }
            (lhs, rhs) => Err(Error::DeviceMismatchBinaryOp {
                lhs: lhs.device().location(),
                rhs: rhs.device().location(),
                op: "index-select",
            }
            .bt()),
        }
    }

    fn matmul(
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
            (Self::Metal(lhs), Self::Metal(rhs)) => {
                let storage = lhs.matmul(rhs, bmnk, lhs_layout, rhs_layout)?;
                Ok(Self::Metal(storage))
            }
            (Self::Custom(lhs), Self::Custom(rhs)) => {
                let storage = lhs.matmul(rhs, bmnk, lhs_layout, rhs_layout)?;
                Ok(Self::Custom(storage))
            }
            (lhs, rhs) => Err(Error::DeviceMismatchBinaryOp {
                lhs: lhs.device().location(),
                rhs: rhs.device().location(),
                op: "matmul",
            }
            .bt()),
        }
    }

    // self, the source can be strided whereas dst is contiguous.
    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        match (self, dst) {
            (Self::Cpu(src), Self::Cpu(dst)) => src.copy_strided_src(dst, dst_offset, src_l),
            (Self::Cuda(src), Self::Cuda(dst)) => Ok(src.copy_strided_src(dst, dst_offset, src_l)?),
            (Self::Metal(src), Self::Metal(dst)) => {
                Ok(src.copy_strided_src(dst, dst_offset, src_l)?)
            }
            (Self::Custom(src), Self::Custom(dst)) => {
                Ok(src.copy_strided_src(dst, dst_offset, src_l)?)
            }
            (lhs, rhs) => Err(Error::DeviceMismatchBinaryOp {
                lhs: lhs.device().location(),
                rhs: rhs.device().location(),
                op: "copy",
            }
            .bt()),
        }
    }
}

impl Storage {
    pub(crate) fn apply_op1(&self, l: &Layout, c: &dyn CustomOp1) -> Result<(Self, Shape)> {
        match self {
            Self::Cpu(storage) => {
                let (storage, shape) = c.cpu_fwd(storage, l)?;
                Ok((Self::Cpu(storage), shape))
            }
            Self::Cuda(storage) => {
                let (storage, shape) = c.cuda_fwd(storage, l)?;
                Ok((Self::Cuda(storage), shape))
            }
            Self::Metal(storage) => {
                let (storage, shape) = c.metal_fwd(storage, l)?;
                Ok((Self::Metal(storage), shape))
            }
            Self::Custom(storage) => {
                let (storage, shape) = c.custom_fwd(storage, l)?;
                Ok((Self::Custom(storage), shape))
            }
        }
    }

    pub(crate) fn apply_op2(
        &self,
        l1: &Layout,
        t2: &Self,
        l2: &Layout,
        c: &dyn CustomOp2,
    ) -> Result<(Self, Shape)> {
        self.same_device(t2, c.name())?;
        match (self, t2) {
            (Self::Cpu(s1), Self::Cpu(s2)) => {
                let (s, shape) = c.cpu_fwd(s1, l1, s2, l2)?;
                Ok((Self::Cpu(s), shape))
            }
            (Self::Cuda(s1), Self::Cuda(s2)) => {
                let (s, shape) = c.cuda_fwd(s1, l1, s2, l2)?;
                Ok((Self::Cuda(s), shape))
            }
            (Self::Metal(s1), Self::Metal(s2)) => {
                let (s, shape) = c.metal_fwd(s1, l1, s2, l2)?;
                Ok((Self::Metal(s), shape))
            }
            (Self::Custom(s1), Self::Custom(s2)) => {
                let (s, shape) = c.custom_fwd(s1, l1, s2, l2)?;
                Ok((Self::Custom(s), shape))
            }
            _ => unreachable!(),
        }
    }

    pub(crate) fn apply_op3(
        &self,
        l1: &Layout,
        t2: &Self,
        l2: &Layout,
        t3: &Self,
        l3: &Layout,
        c: &dyn CustomOp3,
    ) -> Result<(Self, Shape)> {
        self.same_device(t2, c.name())?;
        self.same_device(t3, c.name())?;
        match (self, t2, t3) {
            (Self::Cpu(s1), Self::Cpu(s2), Self::Cpu(s3)) => {
                let (s, shape) = c.cpu_fwd(s1, l1, s2, l2, s3, l3)?;
                Ok((Self::Cpu(s), shape))
            }
            (Self::Cuda(s1), Self::Cuda(s2), Self::Cuda(s3)) => {
                let (s, shape) = c.cuda_fwd(s1, l1, s2, l2, s3, l3)?;
                Ok((Self::Cuda(s), shape))
            }
            (Self::Metal(s1), Self::Metal(s2), Self::Metal(s3)) => {
                let (s, shape) = c.metal_fwd(s1, l1, s2, l2, s3, l3)?;
                Ok((Self::Metal(s), shape))
            }
            (Self::Custom(s1), Self::Custom(s2), Self::Custom(s3)) => {
                let (s, shape) = c.custom_fwd(s1, l1, s2, l2, s3, l3)?;
                Ok((Self::Custom(s), shape))
            }
            _ => unreachable!(),
        }
    }
}
