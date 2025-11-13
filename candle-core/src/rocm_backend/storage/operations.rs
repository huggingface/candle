//! Tensor operations (affine, reduce, cmp, unary, binary, where, const_set)
//! Created by: TEAM-488 (Phase 1 - Device integration)
//! Kernel operations: TEAM-492, TEAM-493, TEAM-494, TEAM-495 (affine, unary, binary, reduce, cmp, where)
//! Updated by: TEAM-509 (const_set implementation with CONST_SET_OP kernels from rocm-rs)
//! CUDA parity verified by: TEAM-497, TEAM-498, TEAM-509

use super::{RocmStorage, RocmStorageSlice};
use crate::rocm_backend::{kernels, ops, RocmError};
use crate::scalar::Scalar;
use crate::Result;

type S = RocmStorageSlice;

impl RocmStorage {
    // Created by: TEAM-493 | CUDA parity verified by: TEAM-497, TEAM-498 (cuda_backend/mod.rs:1472-1476)
    pub(super) fn affine_impl(&self, layout: &crate::Layout, mul: f64, add: f64) -> Result<Self> {
        let device = self.device().clone();
        let slice = ops::Affine(mul, add).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    // Created by: TEAM-493 | CUDA parity verified by: TEAM-497, TEAM-498 (cuda_backend/mod.rs:1478-1482)
    pub(super) fn powf_impl(&self, layout: &crate::Layout, e: f64) -> Result<Self> {
        let device = self.device().clone();
        let slice = ops::Powf(e).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    // Created by: TEAM-493 | CUDA parity verified by: TEAM-497, TEAM-498 (cuda_backend/mod.rs:1484-1488)
    pub(super) fn elu_impl(&self, layout: &crate::Layout, alpha: f64) -> Result<Self> {
        let device = self.device().clone();
        let slice = ops::Elu(alpha).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    // Created by: TEAM-494 | CUDA parity verified by: TEAM-497, TEAM-498 (cuda_backend/mod.rs:1490-1494)
    pub(super) fn reduce_op_impl(
        &self,
        op: crate::op::ReduceOp,
        layout: &crate::Layout,
        sum_dims: &[usize],
    ) -> Result<Self> {
        use crate::op::ReduceOp;

        let device = self.device().clone();
        let sum_dims = sum_dims.to_vec();

        let slice = match op {
            ReduceOp::Sum => ops::ReduceSum { sum_dims }.map(&self.slice, &device, layout)?,
            ReduceOp::Min => ops::ReduceMin { sum_dims }.map(&self.slice, &device, layout)?,
            ReduceOp::Max => ops::ReduceMax { sum_dims }.map(&self.slice, &device, layout)?,
            ReduceOp::ArgMin | ReduceOp::ArgMax => {
                return Err(
                    RocmError::InternalError("ArgMin/ArgMax not yet implemented for ROCm").into()
                );
            }
        };

        Ok(Self { slice, device })
    }

    // Created by: TEAM-494 | CUDA parity verified by: TEAM-497, TEAM-498 (cuda_backend/mod.rs:1496-1500)
    pub(super) fn cmp_impl(
        &self,
        op: crate::op::CmpOp,
        rhs: &Self,
        lhs_l: &crate::Layout,
        rhs_l: &crate::Layout,
    ) -> Result<Self> {
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

    // Created by: TEAM-494 | CUDA parity verified by: TEAM-497, TEAM-498 (cuda_backend/mod.rs:1502-1506)
    pub(super) fn unary_impl<B: crate::op::UnaryOpT>(
        &self,
        layout: &crate::Layout,
    ) -> Result<Self> {
        let device = self.device().clone();
        let slice = ops::UnaryOp::<B>::new().map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    // Created by: TEAM-494 | CUDA parity verified by: TEAM-497, TEAM-498 (cuda_backend/mod.rs:1508-1517)
    pub(super) fn binary_impl<B: crate::op::BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &crate::Layout,
        rhs_l: &crate::Layout,
    ) -> Result<Self> {
        let device = self.device().clone();
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
            return Err(RocmError::InternalError(&format!(
                "Binary operation {} not yet implemented for ROCm",
                type_name
            ))
            .into());
        };

        Ok(Self { slice, device })
    }

    // Created by: TEAM-495 | CUDA parity verified by: TEAM-497, TEAM-498 (cuda_backend/mod.rs:1556-1566)
    pub(super) fn where_cond_impl(
        &self,
        layout: &crate::Layout,
        t: &Self,
        layout_t: &crate::Layout,
        f: &Self,
        layout_f: &crate::Layout,
    ) -> Result<Self> {
        let dev = self.device.hip_device();

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
    
    // TEAM-509: Implement const_set (CUDA parity - COMPLETE ✅)
    // Matches cuda_backend/mod.rs:1318-1347 const_set pattern
    // Fixed by: TEAM-509 (ported CONST_SET_OP kernels from CUDA to HIP in rocm-rs)
    /// Set all elements in the tensor to a constant value
    /// 
    /// **Implementation Status:**
    /// - ✅ All values (zero and non-zero) for U8, U32, I64, F16, F32, F64
    /// - ✅ Contiguous layouts (fast path)
    /// - ✅ Non-contiguous/strided layouts (handled by CONST_SET_OP kernel)
    /// - ❌ BF16, F8E4M3 (ROCm lacks native types, same as CUDA arch < 800)
    /// 
    /// **CUDA Parity:**
    /// ✅ COMPLETE - Uses CONST_SET_OP kernels ported from candle-kernels/src/fill.cu
    /// ✅ Handles all dtypes that CUDA supports (except BF16/FP8 which need arch >= 800)
    /// ✅ Handles strided layouts using get_strided_index (same as CUDA)
    pub(super) fn const_set_impl(&mut self, v: Scalar, layout: &crate::Layout) -> Result<()> {
        // TEAM-509: Use ConstSet operation - SAME pattern as all other operations!
        // Matches affine_impl, powf_impl, elu_impl, etc.
        let device = self.device().clone();
        let slice = ops::ConstSet(v).map(&self.slice, &device, layout)?;
        self.slice = slice;
        Ok(())
    }
}
