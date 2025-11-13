//! RocmStorage implementation for ROCm backend
//!
//! This module contains the RocmStorage struct and its BackendStorage trait implementation.
//! All operations delegate to specialized modules (ops, miopen, rocblas).
//!
//! Created by: TEAM-496

use crate::rocm_backend::{
    kernels, miopen, ops, rocblas, utils, RocmDevice, RocmError, RocmStorageSlice,
};
use crate::{Result, WithDType};
use rocm_rs::hip::DeviceMemory;

// Type alias for convenience (matches CUDA backend pattern)
type S = RocmStorageSlice;

// ============================================================================
// RocmStorage - matches CUDA backend pattern
// ============================================================================

/// ROCm storage - wraps storage slice + device
/// MATCHES: cuda_backend/mod.rs CudaStorage (line 1132)
#[derive(Debug)]
pub struct RocmStorage {
    pub(crate) slice: RocmStorageSlice,
    pub(crate) device: RocmDevice,
}

impl RocmStorage {
    pub fn new(slice: RocmStorageSlice, device: RocmDevice) -> Self {
        Self { slice, device }
    }

    pub fn device(&self) -> &RocmDevice {
        &self.device
    }

    pub fn dtype(&self) -> crate::DType {
        match &self.slice {
            S::U8(_) => crate::DType::U8,
            S::U32(_) => crate::DType::U32,
            S::I64(_) => crate::DType::I64,
            S::BF16(_) => crate::DType::BF16,
            S::F16(_) => crate::DType::F16,
            S::F32(_) => crate::DType::F32,
            S::F64(_) => crate::DType::F64,
            S::F8E4M3(_) => crate::DType::F8E4M3,
        }
    }

    /// Helper method for pooling operations
    /// Delegates to miopen::pool2d
    fn pool2d(
        &self,
        layout: &crate::Layout,
        k: (usize, usize),
        stride: (usize, usize),
        mode: rocm_rs::miopen::ffi::miopenPoolingMode_t,
    ) -> Result<Self> {
        miopen::pool2d(self, layout, k, stride, mode)
    }
}

// ============================================================================
// BackendStorage Implementation
// ============================================================================

impl crate::backend::BackendStorage for RocmStorage {
    type Device = RocmDevice;

    fn try_clone(&self, layout: &crate::Layout) -> Result<Self> {
        let slice = ops::Clone.map(&self.slice, self.device(), layout)?;
        let device = self.device.clone();
        Ok(Self { slice, device })
    }

    fn dtype(&self) -> crate::DType {
        self.dtype()
    }

    fn device(&self) -> &RocmDevice {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<crate::CpuStorage> {
        let cpu_storage = match &self.slice {
            S::U8(slice) => {
                let data = slice.copy_to_host()?;
                crate::CpuStorage::U8(data)
            }
            S::U32(slice) => {
                let data = slice.copy_to_host()?;
                crate::CpuStorage::U32(data)
            }
            S::I64(slice) => {
                let data = slice.copy_to_host()?;
                crate::CpuStorage::I64(data)
            }
            S::BF16(slice) => {
                let data = slice.copy_to_host()?;
                crate::CpuStorage::BF16(data)
            }
            S::F16(slice) => {
                let data = slice.copy_to_host()?;
                crate::CpuStorage::F16(data)
            }
            S::F32(slice) => {
                let data = slice.copy_to_host()?;
                crate::CpuStorage::F32(data)
            }
            S::F64(slice) => {
                let data = slice.copy_to_host()?;
                crate::CpuStorage::F64(data)
            }
            S::F8E4M3(slice) => {
                let data = slice.copy_to_host()?;
                crate::CpuStorage::F8E4M3(data)
            }
        };
        Ok(cpu_storage)
    }

    fn to_dtype(&self, layout: &crate::Layout, dtype: crate::DType) -> Result<Self> {
        use crate::DType;
        let shape = layout.shape();
        let el = shape.elem_count();
        let dev = self.device.hip_device();

        let kernel_name = format!("cast_{}_{}", self.dtype().as_str(), dtype.as_str());

        let slice = match (&self.slice, dtype) {
            (S::U8(src), DType::U8) => S::U8(kernels::launch_cast(&kernel_name, dev, src, layout)?),
            (S::U8(src), DType::U32) => {
                S::U32(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::U8(src), DType::I64) => {
                S::I64(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::U8(src), DType::BF16) => {
                S::BF16(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::U8(src), DType::F16) => {
                S::F16(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::U8(src), DType::F32) => {
                S::F32(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::U8(src), DType::F64) => {
                S::F64(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::U8(src), DType::F8E4M3) => {
                S::F8E4M3(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }

            (S::U32(src), DType::U8) => {
                S::U8(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::U32(src), DType::U32) => {
                S::U32(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::U32(src), DType::I64) => {
                S::I64(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::U32(src), DType::BF16) => {
                S::BF16(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::U32(src), DType::F16) => {
                S::F16(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::U32(src), DType::F32) => {
                S::F32(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::U32(src), DType::F64) => {
                S::F64(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::U32(src), DType::F8E4M3) => {
                S::F8E4M3(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }

            (S::I64(src), DType::U8) => {
                S::U8(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::I64(src), DType::U32) => {
                S::U32(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::I64(src), DType::I64) => {
                S::I64(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::I64(src), DType::BF16) => {
                S::BF16(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::I64(src), DType::F16) => {
                S::F16(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::I64(src), DType::F32) => {
                S::F32(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::I64(src), DType::F64) => {
                S::F64(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::I64(src), DType::F8E4M3) => {
                S::F8E4M3(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }

            (S::BF16(src), DType::U8) => {
                S::U8(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::BF16(src), DType::U32) => {
                S::U32(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::BF16(src), DType::I64) => {
                S::I64(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::BF16(src), DType::BF16) => {
                S::BF16(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::BF16(src), DType::F16) => {
                S::F16(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::BF16(src), DType::F32) => {
                S::F32(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::BF16(src), DType::F64) => {
                S::F64(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::BF16(src), DType::F8E4M3) => {
                S::F8E4M3(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }

            (S::F16(src), DType::U8) => {
                S::U8(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::F16(src), DType::U32) => {
                S::U32(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::F16(src), DType::I64) => {
                S::I64(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::F16(src), DType::BF16) => {
                S::BF16(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::F16(src), DType::F16) => {
                S::F16(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::F16(src), DType::F32) => {
                S::F32(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::F16(src), DType::F64) => {
                S::F64(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::F16(src), DType::F8E4M3) => {
                S::F8E4M3(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }

            (S::F32(src), DType::U8) => {
                S::U8(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::F32(src), DType::U32) => {
                S::U32(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::F32(src), DType::I64) => {
                S::I64(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::F32(src), DType::BF16) => {
                S::BF16(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::F32(src), DType::F16) => {
                S::F16(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::F32(src), DType::F32) => {
                S::F32(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::F32(src), DType::F64) => {
                S::F64(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::F32(src), DType::F8E4M3) => {
                S::F8E4M3(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }

            (S::F64(src), DType::U8) => {
                S::U8(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::F64(src), DType::U32) => {
                S::U32(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::F64(src), DType::I64) => {
                S::I64(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::F64(src), DType::BF16) => {
                S::BF16(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::F64(src), DType::F16) => {
                S::F16(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::F64(src), DType::F32) => {
                S::F32(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::F64(src), DType::F64) => {
                S::F64(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::F64(src), DType::F8E4M3) => {
                S::F8E4M3(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }

            (S::F8E4M3(src), DType::U8) => {
                S::U8(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::F8E4M3(src), DType::U32) => {
                S::U32(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::F8E4M3(src), DType::I64) => {
                S::I64(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::F8E4M3(src), DType::BF16) => {
                S::BF16(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::F8E4M3(src), DType::F16) => {
                S::F16(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::F8E4M3(src), DType::F32) => {
                S::F32(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::F8E4M3(src), DType::F64) => {
                S::F64(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
            (S::F8E4M3(src), DType::F8E4M3) => {
                S::F8E4M3(kernels::launch_cast(&kernel_name, dev, src, layout)?)
            }
        };

        Ok(Self { slice, device: self.device.clone() })
    }

    fn affine(&self, layout: &crate::Layout, mul: f64, add: f64) -> Result<Self> {
        let device = self.device().clone();
        let slice = ops::Affine(mul, add).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn powf(&self, layout: &crate::Layout, e: f64) -> Result<Self> {
        let device = self.device().clone();
        let slice = ops::Powf(e).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn elu(&self, layout: &crate::Layout, alpha: f64) -> Result<Self> {
        let device = self.device().clone();
        let slice = ops::Elu(alpha).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn reduce_op(
        &self,
        op: crate::op::ReduceOp,
        layout: &crate::Layout,
        sum_dims: &[usize],
    ) -> Result<Self> {
        // Matches cuda_backend/mod.rs:1490 - FastReduce pattern
        use crate::op::ReduceOp;

        let device = self.device().clone();
        let sum_dims = sum_dims.to_vec();

        let slice = match op {
            ReduceOp::Sum => ops::ReduceSum { sum_dims }.map(&self.slice, &device, layout)?,
            ReduceOp::Min => ops::ReduceMin { sum_dims }.map(&self.slice, &device, layout)?,
            ReduceOp::Max => ops::ReduceMax { sum_dims }.map(&self.slice, &device, layout)?,
            ReduceOp::ArgMin | ReduceOp::ArgMax => {
                // ArgMin/ArgMax need special handling - return indices (U32)
                return Err(RocmError::InternalError(
                    "ArgMin/ArgMax not yet implemented for ROCm - need index-returning kernels",
                )
                .into());
            }
        };

        Ok(Self { slice, device })
    }

    fn cmp(
        &self,
        op: crate::op::CmpOp,
        rhs: &Self,
        lhs_l: &crate::Layout,
        rhs_l: &crate::Layout,
    ) -> Result<Self> {
        // Matches cuda_backend/mod.rs:1495 - Cmp pattern
        use crate::op::CmpOp;

        let device = self.device().clone();

        let slice = match op {
            CmpOp::Eq => ops::CmpEq.map(&self.slice, lhs_l, &rhs.slice, rhs_l, &device)?,
            CmpOp::Ne => ops::CmpNe.map(&self.slice, lhs_l, &rhs.slice, rhs_l, &device)?,
            CmpOp::Lt => ops::CmpLt.map(&self.slice, lhs_l, &rhs.slice, rhs_l, &device)?,
            CmpOp::Le => ops::CmpLe.map(&self.slice, lhs_l, &rhs.slice, rhs_l, &device)?,
            CmpOp::Gt => ops::CmpGt.map(&self.slice, lhs_l, &rhs.slice, rhs_l, &device)?,
            CmpOp::Ge => ops::CmpGe.map(&self.slice, lhs_l, &rhs.slice, rhs_l, &device)?,
        };

        Ok(Self { slice, device })
    }

    fn unary_impl<B: crate::op::UnaryOpT>(&self, layout: &crate::Layout) -> Result<Self> {
        // Matches cuda_backend/mod.rs:1502 - UnaryOpT dispatch
        let device = self.device().clone();
        let slice = ops::UnaryOp::<B>::new().map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn binary_impl<B: crate::op::BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &crate::Layout,
        rhs_l: &crate::Layout,
    ) -> Result<Self> {
        // Matches cuda_backend/mod.rs:1508 - BinaryOpT dispatch
        use crate::op::{Add, Div, Mul, Sub};

        let device = self.device().clone();

        // Dispatch based on BinaryOpT type using type_name as a workaround
        // This is safe because we're matching on the concrete types
        let type_name = std::any::type_name::<B>();

        let slice = if type_name.contains("::Add") {
            ops::BinaryAdd.map(&self.slice, lhs_l, &rhs.slice, rhs_l, &device)?
        } else if type_name.contains("::Sub") {
            ops::BinarySub.map(&self.slice, lhs_l, &rhs.slice, rhs_l, &device)?
        } else if type_name.contains("::Mul") {
            ops::BinaryMul.map(&self.slice, lhs_l, &rhs.slice, rhs_l, &device)?
        } else if type_name.contains("::Div") {
            ops::BinaryDiv.map(&self.slice, lhs_l, &rhs.slice, rhs_l, &device)?
        } else {
            // For Maximum/Minimum, we need to implement them separately
            return Err(RocmError::InternalError(&format!(
                "Binary operation {} not yet implemented for ROCm",
                type_name
            ))
            .into());
        };

        Ok(Self { slice, device })
    }

    fn where_cond(
        &self,
        layout: &crate::Layout,
        t: &Self,
        layout_t: &crate::Layout,
        f: &Self,
        layout_f: &crate::Layout,
    ) -> Result<Self> {
        use crate::DType;
        let dev = self.device.hip_device();

        // Determine kernel name from condition type and value type
        let cond_type = match &self.slice {
            S::U8(_) => "u8",
            S::U32(_) => "u32",
            S::I64(_) => "i64",
            _ => {
                return Err(
                    RocmError::InternalError("where_cond: condition must be u8/u32/i64").into()
                )
            }
        };

        let val_type = t.dtype().as_str();
        let kernel_name = format!("where_{}_{}", cond_type, val_type);

        let slice = match (&self.slice, &t.slice, &f.slice) {
            (S::U8(cond), S::F32(t_val), S::F32(f_val)) => S::F32(kernels::launch_ternary(
                &kernel_name,
                dev,
                cond,
                layout,
                t_val,
                layout_t,
                f_val,
                layout_f,
            )?),
            (S::U8(cond), S::F64(t_val), S::F64(f_val)) => S::F64(kernels::launch_ternary(
                &kernel_name,
                dev,
                cond,
                layout,
                t_val,
                layout_t,
                f_val,
                layout_f,
            )?),
            (S::U8(cond), S::F16(t_val), S::F16(f_val)) => S::F16(kernels::launch_ternary(
                &kernel_name,
                dev,
                cond,
                layout,
                t_val,
                layout_t,
                f_val,
                layout_f,
            )?),
            (S::U8(cond), S::BF16(t_val), S::BF16(f_val)) => S::BF16(kernels::launch_ternary(
                &kernel_name,
                dev,
                cond,
                layout,
                t_val,
                layout_t,
                f_val,
                layout_f,
            )?),
            (S::U8(cond), S::U8(t_val), S::U8(f_val)) => S::U8(kernels::launch_ternary(
                &kernel_name,
                dev,
                cond,
                layout,
                t_val,
                layout_t,
                f_val,
                layout_f,
            )?),
            (S::U8(cond), S::U32(t_val), S::U32(f_val)) => S::U32(kernels::launch_ternary(
                &kernel_name,
                dev,
                cond,
                layout,
                t_val,
                layout_t,
                f_val,
                layout_f,
            )?),
            (S::U8(cond), S::I64(t_val), S::I64(f_val)) => S::I64(kernels::launch_ternary(
                &kernel_name,
                dev,
                cond,
                layout,
                t_val,
                layout_t,
                f_val,
                layout_f,
            )?),
            (S::U8(cond), S::F8E4M3(t_val), S::F8E4M3(f_val)) => {
                S::F8E4M3(kernels::launch_ternary(
                    &kernel_name,
                    dev,
                    cond,
                    layout,
                    t_val,
                    layout_t,
                    f_val,
                    layout_f,
                )?)
            }
            _ => {
                return Err(
                    RocmError::InternalError("where_cond: unsupported type combination").into()
                )
            }
        };

        Ok(Self { slice, device: self.device.clone() })
    }

    // Advanced operations - unimplemented for now
    fn conv1d(
        &self,
        _l: &crate::Layout,
        _kernel: &Self,
        _kernel_l: &crate::Layout,
        _params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        unimplemented!("conv1d - need MIOpen integration")
    }

    fn conv_transpose1d(
        &self,
        _l: &crate::Layout,
        _kernel: &Self,
        _kernel_l: &crate::Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        unimplemented!("conv_transpose1d - need MIOpen integration")
    }

    fn conv2d(
        &self,
        inp_l: &crate::Layout,
        kernel: &Self,
        kernel_l: &crate::Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        // Matches cuda_backend/mod.rs:1801 - MIOpen convolution
        miopen::conv2d(self, inp_l, kernel, kernel_l, params)
    }

    fn conv_transpose2d(
        &self,
        _l: &crate::Layout,
        _kernel: &Self,
        _kernel_l: &crate::Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        unimplemented!("conv_transpose2d - need MIOpen integration")
    }

    fn avg_pool2d(
        &self,
        layout: &crate::Layout,
        k: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        // Matches cuda_backend/mod.rs:1879 - MIOpen pooling
        self.pool2d(
            layout,
            k,
            stride,
            rocm_rs::miopen::ffi::miopenPoolingMode_t_miopenPoolingAverage,
        )
    }

    fn max_pool2d(
        &self,
        layout: &crate::Layout,
        k: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        // Matches cuda_backend/mod.rs:1892 - MIOpen pooling
        self.pool2d(layout, k, stride, rocm_rs::miopen::ffi::miopenPoolingMode_t_miopenPoolingMax)
    }

    fn upsample_nearest1d(&self, _: &crate::Layout, _: usize) -> Result<Self> {
        unimplemented!("upsample_nearest1d")
    }

    fn upsample_nearest2d(&self, _: &crate::Layout, _: usize, _: usize) -> Result<Self> {
        unimplemented!("upsample_nearest2d")
    }

    fn gather(&self, _: &crate::Layout, _: &Self, _: &crate::Layout, _: usize) -> Result<Self> {
        unimplemented!("gather - need custom kernels")
    }

    fn scatter_set(
        &mut self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: usize,
    ) -> Result<()> {
        unimplemented!("scatter_set - need custom kernels")
    }

    fn scatter_add_set(
        &mut self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: usize,
    ) -> Result<()> {
        unimplemented!("scatter_add_set - need custom kernels")
    }

    fn index_select(
        &self,
        _: &Self,
        _: &crate::Layout,
        _: &crate::Layout,
        _: usize,
    ) -> Result<Self> {
        unimplemented!("index_select - need custom kernels")
    }

    fn index_add(
        &self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: usize,
    ) -> Result<Self> {
        unimplemented!("index_add - need custom kernels")
    }

    fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &crate::Layout,
        rhs_l: &crate::Layout,
    ) -> Result<Self> {
        // Matches cuda_backend/mod.rs:1965 - rocBLAS GEMM
        rocblas::matmul(self, rhs, (b, m, n, k), lhs_l, rhs_l)
    }

    fn copy2d(
        &self,
        _: &mut Self,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
    ) -> Result<()> {
        unimplemented!("copy2d")
    }

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &crate::Layout) -> Result<()> {
        unimplemented!("copy_strided_src")
    }
}
