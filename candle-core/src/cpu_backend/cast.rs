use super::{unary_map, CpuStorage};
use crate::backend::BackendStorage;
use crate::{DType, Error, Layout, Result};
use float8::F8E4M3;
use half::{bf16, f16};

// Keep the existing FP8 conversion paths explicit, including the direct f64 path and
// bit-preserving identity conversion, rather than using a common float pivot.
macro_rules! cast_element {
    ($v:ident, F8E4M3, F8E4M3, $dst:ty) => {
        $v
    };
    ($v:ident, F64, F8E4M3, $dst:ty) => {
        F8E4M3::from_f64($v)
    };
    ($v:ident, $src:ident, F8E4M3, $dst:ty) => {
        F8E4M3::from_f32(num_traits::AsPrimitive::<f32>::as_($v))
    };
    ($v:ident, F8E4M3, F64, $dst:ty) => {
        $v.to_f64()
    };
    ($v:ident, F8E4M3, $dst_variant:ident, $dst:ty) => {
        num_traits::AsPrimitive::<$dst>::as_($v.to_f32())
    };
    ($v:ident, $src:ident, $dst_variant:ident, $dst:ty) => {
        num_traits::AsPrimitive::<$dst>::as_($v)
    };
}

macro_rules! cast_storage {
    ($storage:ident, $layout:ident, $dtype:ident; [$(($variant:ident, $ty:ty)),+ $(,)?]) => {
        cast_storage!(@sources $storage, $layout, $dtype;
            [$(($variant, $ty)),+]; [$($variant),+])
    };
    (@sources $storage:ident, $layout:ident, $dtype:ident;
        $types:tt; [$($src:ident),+]) => {
        match ($storage, $dtype) {
            // Match the destination first, preserving the error reported when
            // both source and destination are unsupported packed formats.
            (_, DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0) => {
                Err(Error::UnsupportedDTypeForOp($dtype, "to_dtype").bt())
            }
            $((CpuStorage::$src(values), _) => {
                cast_storage!(@destinations values, $layout, $dtype; $src; $types)
            })+
            (CpuStorage::F6E2M3(_)
                | CpuStorage::F6E3M2(_)
                | CpuStorage::F4(_)
                | CpuStorage::F8E8M0(_), _) => {
                Err(Error::UnsupportedDTypeForOp($storage.dtype(), "to_dtype").bt())
            }
        }
    };
    (@destinations $values:ident, $layout:ident, $dtype:ident;
        $src:ident; [$(($dst_variant:ident, $dst:ty)),+]) => {
        match $dtype {
            $(DType::$dst_variant => {
                let data = unary_map($values, $layout, |v| {
                    cast_element!(v, $src, $dst_variant, $dst)
                });
                Ok(CpuStorage::$dst_variant(data))
            })+
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                Err(Error::UnsupportedDTypeForOp($dtype, "to_dtype").bt())
            }
        }
    };
}

pub(super) fn to_dtype(storage: &CpuStorage, layout: &Layout, dtype: DType) -> Result<CpuStorage> {
    // One O(T) list generates the source and destination dispatches for T types.
    // There are still T^2 instantiated conversion pairs; converting N elements
    // still takes O(N) time. All layouts retain the existing unary_map paths.
    cast_storage!(storage, layout, dtype; [
        (U8, u8),
        (U32, u32),
        (I16, i16),
        (I32, i32),
        (I64, i64),
        (BF16, bf16),
        (F16, f16),
        (F32, f32),
        (F64, f64),
        (F8E4M3, F8E4M3),
    ])
}
