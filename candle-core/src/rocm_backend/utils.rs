// candle-core/src/rocm_backend/utils.rs
// TEAM-493: ROCm utility traits - EXACT parity with CUDA backend
// Matches candle-core/src/cuda_backend/utils.rs

use crate::{Layout, Result, WithDType};
use rocm_rs::hip::DeviceMemory;

use super::{RocmDevice, RocmError};

pub type S = super::RocmStorageSlice;

/// Map1 trait - unary operations on storage slices
/// MATCHES: cuda_backend/utils.rs Map1
pub trait Map1 {
    fn f<T: WithDType>(
        &self,
        src: &DeviceMemory<T>,
        dev: &RocmDevice,
        layout: &Layout,
    ) -> Result<DeviceMemory<T>>;

    fn map(&self, s: &S, d: &RocmDevice, l: &Layout) -> Result<S> {
        let out = match s {
            S::U8(s) => S::U8(self.f(s, d, l)?),
            S::U32(s) => S::U32(self.f(s, d, l)?),
            S::I64(s) => S::I64(self.f(s, d, l)?),
            S::BF16(s) => S::BF16(self.f(s, d, l)?),
            S::F16(s) => S::F16(self.f(s, d, l)?),
            S::F32(s) => S::F32(self.f(s, d, l)?),
            S::F64(s) => S::F64(self.f(s, d, l)?),
            S::F8E4M3(s) => S::F8E4M3(self.f(s, d, l)?),
        };
        Ok(out)
    }
}

/// Map2 trait - binary operations on storage slices
/// MATCHES: cuda_backend/utils.rs Map2
pub trait Map2 {
    fn f<T: WithDType>(
        &self,
        src1: &DeviceMemory<T>,
        layout1: &Layout,
        src2: &DeviceMemory<T>,
        layout2: &Layout,
        dev: &RocmDevice,
    ) -> Result<DeviceMemory<T>>;

    fn map(&self, s1: &S, l1: &Layout, s2: &S, l2: &Layout, d: &RocmDevice) -> Result<S> {
        let out = match (s1, s2) {
            (S::U8(s1), S::U8(s2)) => S::U8(self.f(s1, l1, s2, l2, d)?),
            (S::U32(s1), S::U32(s2)) => S::U32(self.f(s1, l1, s2, l2, d)?),
            (S::I64(s1), S::I64(s2)) => S::I64(self.f(s1, l1, s2, l2, d)?),
            (S::BF16(s1), S::BF16(s2)) => S::BF16(self.f(s1, l1, s2, l2, d)?),
            (S::F16(s1), S::F16(s2)) => S::F16(self.f(s1, l1, s2, l2, d)?),
            (S::F32(s1), S::F32(s2)) => S::F32(self.f(s1, l1, s2, l2, d)?),
            (S::F64(s1), S::F64(s2)) => S::F64(self.f(s1, l1, s2, l2, d)?),
            (S::F8E4M3(s1), S::F8E4M3(s2)) => S::F8E4M3(self.f(s1, l1, s2, l2, d)?),
            _ => Err(RocmError::InternalError("dtype mismatch in binary op"))?,
        };
        Ok(out)
    }
}

/// Map3 trait - ternary operations on storage slices
/// MATCHES: cuda_backend/utils.rs Map3
pub trait Map3 {
    #[allow(clippy::too_many_arguments)]
    fn f<T: WithDType>(
        &self,
        src1: &DeviceMemory<T>,
        layout1: &Layout,
        src2: &DeviceMemory<T>,
        layout2: &Layout,
        src3: &DeviceMemory<T>,
        layout3: &Layout,
        dev: &RocmDevice,
    ) -> Result<DeviceMemory<T>>;

    #[allow(clippy::too_many_arguments)]
    fn map(
        &self,
        s1: &S,
        l1: &Layout,
        s2: &S,
        l2: &Layout,
        s3: &S,
        l3: &Layout,
        d: &RocmDevice,
    ) -> Result<S> {
        let out = match (s1, s2, s3) {
            (S::U8(s1), S::U8(s2), S::U8(s3)) => S::U8(self.f(s1, l1, s2, l2, s3, l3, d)?),
            (S::U32(s1), S::U32(s2), S::U32(s3)) => S::U32(self.f(s1, l1, s2, l2, s3, l3, d)?),
            (S::I64(s1), S::I64(s2), S::I64(s3)) => S::I64(self.f(s1, l1, s2, l2, s3, l3, d)?),
            (S::BF16(s1), S::BF16(s2), S::BF16(s3)) => S::BF16(self.f(s1, l1, s2, l2, s3, l3, d)?),
            (S::F16(s1), S::F16(s2), S::F16(s3)) => S::F16(self.f(s1, l1, s2, l2, s3, l3, d)?),
            (S::F32(s1), S::F32(s2), S::F32(s3)) => S::F32(self.f(s1, l1, s2, l2, s3, l3, d)?),
            (S::F64(s1), S::F64(s2), S::F64(s3)) => S::F64(self.f(s1, l1, s2, l2, s3, l3, d)?),
            (S::F8E4M3(s1), S::F8E4M3(s2), S::F8E4M3(s3)) => {
                S::F8E4M3(self.f(s1, l1, s2, l2, s3, l3, d)?)
            }
            _ => Err(RocmError::InternalError("dtype mismatch in ternary op"))?,
        };
        Ok(out)
    }
}

/// Map2InPlace trait - in-place binary operations
/// MATCHES: cuda_backend/utils.rs Map2InPlace
pub trait Map2InPlace {
    fn f<T: WithDType>(
        &self,
        dst: &mut DeviceMemory<T>,
        dst_l: &Layout,
        src: &DeviceMemory<T>,
        src_l: &Layout,
        dev: &RocmDevice,
    ) -> Result<()>;

    fn map(
        &self,
        dst: &mut S,
        dst_l: &Layout,
        src: &S,
        src_l: &Layout,
        d: &RocmDevice,
    ) -> Result<()> {
        match (dst, src) {
            (S::U8(dst), S::U8(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::U32(dst), S::U32(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::I64(dst), S::I64(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::BF16(dst), S::BF16(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::F16(dst), S::F16(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::F32(dst), S::F32(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::F64(dst), S::F64(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::F8E4M3(dst), S::F8E4M3(src)) => self.f(dst, dst_l, src, src_l, d),
            _ => Err(RocmError::InternalError("dtype mismatch in binary op"))?,
        }
    }
}

/// Map1Any trait - unary operations that can change dtype
/// MATCHES: cuda_backend/utils.rs Map1Any
pub trait Map1Any {
    fn f<T: WithDType, W: Fn(DeviceMemory<T>) -> S>(
        &self,
        src: &DeviceMemory<T>,
        dev: &RocmDevice,
        layout: &Layout,
        wrap: W,
    ) -> Result<S>;

    fn map(&self, s: &S, d: &RocmDevice, l: &Layout) -> Result<S> {
        let out = match s {
            S::U8(s) => self.f(s, d, l, S::U8)?,
            S::U32(s) => self.f(s, d, l, S::U32)?,
            S::I64(s) => self.f(s, d, l, S::I64)?,
            S::BF16(s) => self.f(s, d, l, S::BF16)?,
            S::F16(s) => self.f(s, d, l, S::F16)?,
            S::F32(s) => self.f(s, d, l, S::F32)?,
            S::F64(s) => self.f(s, d, l, S::F64)?,
            S::F8E4M3(s) => self.f(s, d, l, S::F8E4M3)?,
        };
        Ok(out)
    }
}

/// Map2Any trait - binary operations that can change dtype
/// MATCHES: cuda_backend/utils.rs Map2Any
pub trait Map2Any {
    fn f<T: WithDType>(
        &self,
        src1: &DeviceMemory<T>,
        layout1: &Layout,
        src2: &DeviceMemory<T>,
        layout2: &Layout,
        dev: &RocmDevice,
    ) -> Result<S>;

    fn map(&self, s1: &S, l1: &Layout, s2: &S, l2: &Layout, d: &RocmDevice) -> Result<S> {
        let out = match (s1, s2) {
            (S::U8(s1), S::U8(s2)) => self.f(s1, l1, s2, l2, d)?,
            (S::U32(s1), S::U32(s2)) => self.f(s1, l1, s2, l2, d)?,
            (S::I64(s1), S::I64(s2)) => self.f(s1, l1, s2, l2, d)?,
            (S::BF16(s1), S::BF16(s2)) => self.f(s1, l1, s2, l2, d)?,
            (S::F16(s1), S::F16(s2)) => self.f(s1, l1, s2, l2, d)?,
            (S::F32(s1), S::F32(s2)) => self.f(s1, l1, s2, l2, d)?,
            (S::F64(s1), S::F64(s2)) => self.f(s1, l1, s2, l2, d)?,
            (S::F8E4M3(s1), S::F8E4M3(s2)) => self.f(s1, l1, s2, l2, d)?,
            _ => Err(RocmError::InternalError("dtype mismatch in binary op"))?,
        };
        Ok(out)
    }
}
