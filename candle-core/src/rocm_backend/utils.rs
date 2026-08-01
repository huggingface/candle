//! The dtype-dispatch traits every ROCm op is written against.
//!
//! An implementor writes one generic `f` over `SendSyncDeviceMemory<T>` and gets
//! `map`, which walks the [`RocmStorageSlice`] variants for it. Launch geometry
//! and the launch itself live in [`super::launch`]; the `[dims, strides]`
//! buffers in [`super::params`].

use crate::{Layout, Result};

use super::alloc::SendSyncDeviceMemory;
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
            S::F8E4M3(s) => S::F8E4M3(self.f(s, d, l)?),
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
            (S::F8E4M3(a), S::F8E4M3(b)) => S::F8E4M3(self.f(a, l1, b, l2, d)?),
            _ => crate::bail!("dtype mismatch in binary op"),
        };
        Ok(out)
    }
}

/// Trait for applying binary operations whose output dtype differs from the
/// input dtype (comparisons yield `u8`).
pub trait Map2Any {
    fn f<T: Copy + Send + Sync + 'static>(
        &self,
        lhs: &SendSyncDeviceMemory<T>,
        lhs_l: &Layout,
        rhs: &SendSyncDeviceMemory<T>,
        rhs_l: &Layout,
        dev: &RocmDevice,
    ) -> Result<S>;

    fn map(&self, s1: &S, l1: &Layout, s2: &S, l2: &Layout, d: &RocmDevice) -> Result<S> {
        let out = match (s1, s2) {
            (S::U8(a), S::U8(b)) => self.f(a, l1, b, l2, d)?,
            (S::U32(a), S::U32(b)) => self.f(a, l1, b, l2, d)?,
            (S::I16(a), S::I16(b)) => self.f(a, l1, b, l2, d)?,
            (S::I32(a), S::I32(b)) => self.f(a, l1, b, l2, d)?,
            (S::I64(a), S::I64(b)) => self.f(a, l1, b, l2, d)?,
            (S::BF16(a), S::BF16(b)) => self.f(a, l1, b, l2, d)?,
            (S::F16(a), S::F16(b)) => self.f(a, l1, b, l2, d)?,
            (S::F32(a), S::F32(b)) => self.f(a, l1, b, l2, d)?,
            (S::F64(a), S::F64(b)) => self.f(a, l1, b, l2, d)?,
            (S::F8E4M3(a), S::F8E4M3(b)) => self.f(a, l1, b, l2, d)?,
            _ => crate::bail!("dtype mismatch in binary op"),
        };
        Ok(out)
    }
}

/// Trait for applying unary operations whose output dtype differs from the
/// input dtype (argmin/argmax yield `u32`).
///
/// `wrap` re-wraps a same-dtype result, so implementations that keep the input
/// dtype stay generic while the ones that do not can build any [`S`] variant.
pub trait Map1Any {
    fn f<T: Copy + Send + Sync + 'static, W: Fn(SendSyncDeviceMemory<T>) -> S>(
        &self,
        src: &SendSyncDeviceMemory<T>,
        dev: &RocmDevice,
        layout: &Layout,
        wrap: W,
    ) -> Result<S>;

    fn map(&self, s: &S, d: &RocmDevice, l: &Layout) -> Result<S> {
        let out = match s {
            S::U8(s) => self.f(s, d, l, S::U8)?,
            S::U32(s) => self.f(s, d, l, S::U32)?,
            S::I16(s) => self.f(s, d, l, S::I16)?,
            S::I32(s) => self.f(s, d, l, S::I32)?,
            S::I64(s) => self.f(s, d, l, S::I64)?,
            S::BF16(s) => self.f(s, d, l, S::BF16)?,
            S::F16(s) => self.f(s, d, l, S::F16)?,
            S::F32(s) => self.f(s, d, l, S::F32)?,
            S::F64(s) => self.f(s, d, l, S::F64)?,
            // Not a dispatch limitation: upstream leaves every `SUM_OP`/`FAST_OP`
            // fp8 instantiation in `reduce.cu` commented out, so at no
            // `__CUDA_ARCH__` is there a kernel for `f` to launch. The error
            // says so, and points at the one workaround that exists.
            S::F8E4M3(_) => crate::bail!(
                "reduce/argmin/argmax are not available for F8E4M3 on ROCm: \
                 candle-kernels/src/reduce.cu instantiates no fp8 kernels. \
                 Cast to f32 first."
            ),
        };
        Ok(out)
    }
}

/// Trait for binary operations that write into an existing destination.
pub trait Map2InPlace {
    fn f<T: Copy + Send + Sync + 'static>(
        &self,
        dst: &mut SendSyncDeviceMemory<T>,
        dst_l: &Layout,
        src: &SendSyncDeviceMemory<T>,
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
            (S::I16(dst), S::I16(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::I32(dst), S::I32(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::I64(dst), S::I64(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::BF16(dst), S::BF16(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::F16(dst), S::F16(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::F32(dst), S::F32(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::F64(dst), S::F64(src)) => self.f(dst, dst_l, src, src_l, d),
            (S::F8E4M3(dst), S::F8E4M3(src)) => self.f(dst, dst_l, src, src_l, d),
            _ => crate::bail!("dtype mismatch in binary op"),
        }
    }
}

/// Trait for applying ternary operations to ROCm storage.
pub trait Map3 {
    #[allow(clippy::too_many_arguments)]
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
            (S::U8(a), S::U8(b), S::U8(c)) => S::U8(self.f(a, l1, b, l2, c, l3, d)?),
            (S::U32(a), S::U32(b), S::U32(c)) => S::U32(self.f(a, l1, b, l2, c, l3, d)?),
            (S::I16(a), S::I16(b), S::I16(c)) => S::I16(self.f(a, l1, b, l2, c, l3, d)?),
            (S::I32(a), S::I32(b), S::I32(c)) => S::I32(self.f(a, l1, b, l2, c, l3, d)?),
            (S::I64(a), S::I64(b), S::I64(c)) => S::I64(self.f(a, l1, b, l2, c, l3, d)?),
            (S::BF16(a), S::BF16(b), S::BF16(c)) => S::BF16(self.f(a, l1, b, l2, c, l3, d)?),
            (S::F16(a), S::F16(b), S::F16(c)) => S::F16(self.f(a, l1, b, l2, c, l3, d)?),
            (S::F32(a), S::F32(b), S::F32(c)) => S::F32(self.f(a, l1, b, l2, c, l3, d)?),
            (S::F64(a), S::F64(b), S::F64(c)) => S::F64(self.f(a, l1, b, l2, c, l3, d)?),
            (S::F8E4M3(a), S::F8E4M3(b), S::F8E4M3(c)) => {
                S::F8E4M3(self.f(a, l1, b, l2, c, l3, d)?)
            }
            _ => crate::bail!("dtype mismatch in ternary op"),
        };
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{kernel_name, try_kernel_name};
    use half::{bf16, f16};

    #[test]
    fn kernel_name_suffixes() {
        assert_eq!(try_kernel_name::<f32>("ucopy").unwrap(), "ucopy_f32");
        assert_eq!(try_kernel_name::<f64>("ucopy").unwrap(), "ucopy_f64");
        assert_eq!(try_kernel_name::<u8>("ucopy").unwrap(), "ucopy_u8");
        assert_eq!(try_kernel_name::<u32>("ucopy").unwrap(), "ucopy_u32");
        assert_eq!(try_kernel_name::<i16>("ucopy").unwrap(), "ucopy_i16");
        assert_eq!(try_kernel_name::<i32>("ucopy").unwrap(), "ucopy_i32");
        assert_eq!(try_kernel_name::<i64>("ucopy").unwrap(), "ucopy_i64");
        assert_eq!(try_kernel_name::<f16>("ucopy").unwrap(), "ucopy_f16");
        // `half::bf16`'s type name also contains "f16".
        assert_eq!(try_kernel_name::<bf16>("ucopy").unwrap(), "ucopy_bf16");
    }

    /// `try_clone` hands the copy back to a caller that keeps using the
    /// *original* layout, so absolute element offsets have to survive.
    #[test]
    fn try_clone_preserves_start_offset() -> crate::Result<()> {
        use crate::backend::{BackendDevice, BackendStorage};
        use crate::rocm_backend::RocmDevice;
        use crate::{CpuStorage, Layout};

        let dev = match RocmDevice::new(0) {
            Ok(dev) => dev,
            // No ROCm device on this machine.
            Err(_) => return Ok(()),
        };
        let src = dev.storage_from_slice(&[0f32, 1., 2., 3., 4., 5.])?;
        let cloned = src.try_clone(&Layout::contiguous_with_offset(3, 3))?;
        match cloned.to_cpu_storage()? {
            CpuStorage::F32(v) => assert_eq!(v, vec![0., 1., 2., 3., 4., 5.]),
            other => crate::bail!("unexpected dtype {:?}", other.dtype()),
        }
        Ok(())
    }

    #[test]
    fn unsupported_dtype_errors_instead_of_panicking() {
        assert!(try_kernel_name::<crate::dummy_dtype::F8E8M0>("ucopy").is_err());
        // The infallible variant yields a name no module defines, so the
        // failure surfaces as "kernel not found" rather than an abort.
        assert_eq!(
            kernel_name::<crate::dummy_dtype::F8E8M0>("ucopy"),
            "ucopy_unsupported_dtype"
        );
    }

    /// F8E4M3 carries its own type so the generic `f` reaches the fp8 kernels
    /// rather than the u8 ones it would resolve to if the storage were shared.
    #[test]
    fn f8e4m3_resolves_to_the_fp8_kernels() {
        assert_eq!(
            try_kernel_name::<float8::F8E4M3>("ucopy").unwrap(),
            "ucopy_f8_e4m3"
        );
        assert_eq!(
            try_kernel_name::<float8::F8E4M3>("badd").unwrap(),
            "badd_f8_e4m3"
        );
        // unary.cu and ternary.cu spell the same dtype differently.
        assert_eq!(
            try_kernel_name::<float8::F8E4M3>("uneg").unwrap(),
            "uneg_fp8_e4m3"
        );
        assert_eq!(
            try_kernel_name::<float8::F8E4M3>("where_u8").unwrap(),
            "where_u8_fp8_e4m3"
        );
    }
}
