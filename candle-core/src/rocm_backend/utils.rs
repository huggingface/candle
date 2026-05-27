//! Helper functions to plug ROCm kernels in candle.
use crate::{Layout, Result};

use super::wrappers::SendSyncDeviceMemory;
use super::{RocmDevice, RocmStorageSlice};

pub type S = RocmStorageSlice;

/// Trait for applying unary operations to ROCm storage.
///
/// Implement this trait for your custom operation and use the `map` method
/// to apply it to any storage type.
pub trait Map1 {
    fn f<T: Copy + Send + Sync + 'static>(
        &self,
        src: &SendSyncDeviceMemory<T>,
        dev: &RocmDevice,
        layout: &Layout,
    ) -> Result<SendSyncDeviceMemory<T>>;

    fn map(&self, s: &S, d: &RocmDevice, l: &Layout) -> Result<S> {
        let out = match s {
            S::U8(s) => S::U8(self.f(s, d, l)?),
            S::U32(s) => S::U32(self.f(s, d, l)?),
            S::I16(s) => S::I16(self.f(s, d, l)?),
            S::I32(s) => S::I32(self.f(s, d, l)?),
            S::I64(s) => S::I64(self.f(s, d, l)?),
            S::BF16(s) => S::BF16(self.f(s, d, l)?),
            S::F16(s) => S::F16(self.f(s, d, l)?),
            S::F32(s) => S::F32(self.f(s, d, l)?),
            S::F64(s) => S::F64(self.f(s, d, l)?),
            S::F8E4M3(_) => crate::bail!("Map1 does not support F8E4M3 for ROCm"),
        };
        Ok(out)
    }
}

/// Trait for applying binary operations to ROCm storage.
pub trait Map2 {
    fn f<T: Copy + Send + Sync + 'static>(
        &self,
        lhs: &SendSyncDeviceMemory<T>,
        lhs_l: &Layout,
        rhs: &SendSyncDeviceMemory<T>,
        rhs_l: &Layout,
        dev: &RocmDevice,
    ) -> Result<SendSyncDeviceMemory<T>>;

    fn map(&self, s1: &S, l1: &Layout, s2: &S, l2: &Layout, d: &RocmDevice) -> Result<S> {
        let out = match (s1, s2) {
            (S::U8(a), S::U8(b)) => S::U8(self.f(a, l1, b, l2, d)?),
            (S::U32(a), S::U32(b)) => S::U32(self.f(a, l1, b, l2, d)?),
            (S::I16(a), S::I16(b)) => S::I16(self.f(a, l1, b, l2, d)?),
            (S::I32(a), S::I32(b)) => S::I32(self.f(a, l1, b, l2, d)?),
            (S::I64(a), S::I64(b)) => S::I64(self.f(a, l1, b, l2, d)?),
            (S::BF16(a), S::BF16(b)) => S::BF16(self.f(a, l1, b, l2, d)?),
            (S::F16(a), S::F16(b)) => S::F16(self.f(a, l1, b, l2, d)?),
            (S::F32(a), S::F32(b)) => S::F32(self.f(a, l1, b, l2, d)?),
            (S::F64(a), S::F64(b)) => S::F64(self.f(a, l1, b, l2, d)?),
            _ => crate::bail!("dtype mismatch in binary op"),
        };
        Ok(out)
    }
}

/// Trait for applying ternary operations to ROCm storage.
pub trait Map3 {
    fn f<T: Copy + Send + Sync + 'static>(
        &self,
        s1: &SendSyncDeviceMemory<T>,
        l1: &Layout,
        s2: &SendSyncDeviceMemory<T>,
        l2: &Layout,
        s3: &SendSyncDeviceMemory<T>,
        l3: &Layout,
        dev: &RocmDevice,
    ) -> Result<SendSyncDeviceMemory<T>>;

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
            (S::U8(a), S::U8(b), S::U8(c)) => S::U8(self.f(a, l1, b, l2, c, l3, d)?),
            (S::U32(a), S::U32(b), S::U32(c)) => S::U32(self.f(a, l1, b, l2, c, l3, d)?),
            (S::I16(a), S::I16(b), S::I16(c)) => S::I16(self.f(a, l1, b, l2, c, l3, d)?),
            (S::I32(a), S::I32(b), S::I32(c)) => S::I32(self.f(a, l1, b, l2, c, l3, d)?),
            (S::I64(a), S::I64(b), S::I64(c)) => S::I64(self.f(a, l1, b, l2, c, l3, d)?),
            (S::BF16(a), S::BF16(b), S::BF16(c)) => S::BF16(self.f(a, l1, b, l2, c, l3, d)?),
            (S::F16(a), S::F16(b), S::F16(c)) => S::F16(self.f(a, l1, b, l2, c, l3, d)?),
            (S::F32(a), S::F32(b), S::F32(c)) => S::F32(self.f(a, l1, b, l2, c, l3, d)?),
            (S::F64(a), S::F64(b), S::F64(c)) => S::F64(self.f(a, l1, b, l2, c, l3, d)?),
            _ => crate::bail!("dtype mismatch in ternary op"),
        };
        Ok(out)
    }
}
