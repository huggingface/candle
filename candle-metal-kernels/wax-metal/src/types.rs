//! wax to air type mapping.

// TODO: Fairly certain most of these mappings could be improved.

use crate::air_dialect::types as t;
use pliron::context::Context;
use pliron::r#type::TypeHandle;
use wax_ir::{ScalarType, TileElementType, TileType, Type};

/// Scalar type to LLVM IR storage type name.
///
/// There is no native `f42m1fn` in LLVM. The storage is `i8`.
pub fn ll_scalar(s: &ScalarType) -> &'static str {
    match s {
        ScalarType::I1 => "i1",
        ScalarType::I4 | ScalarType::I8 => "i8",
        ScalarType::I16 => "i16",
        ScalarType::I32 => "i32",
        ScalarType::I64 => "i64",
        ScalarType::F16 => "half",
        ScalarType::BF16 => "bfloat",
        ScalarType::F32 | ScalarType::TF32 => "float",
        ScalarType::F64 => "double",
        ScalarType::F8E4M3FN
        | ScalarType::F8E5M2
        | ScalarType::F8E8M0FNU
        | ScalarType::F4E2M1FN => "i8",
    }
}

/// Is the scalar a floating-point type.
pub fn is_float(s: &ScalarType) -> bool {
    matches!(
        s,
        ScalarType::F16
            | ScalarType::BF16
            | ScalarType::F32
            | ScalarType::TF32
            | ScalarType::F64
            | ScalarType::F8E4M3FN
            | ScalarType::F8E5M2
            | ScalarType::F8E8M0FNU
            | ScalarType::F4E2M1FN
    )
}

/// Byte width of a scalar type.
pub fn scalar_byte_width(s: &ScalarType) -> u32 {
    match s {
        ScalarType::I1
        | ScalarType::I4
        | ScalarType::I8
        | ScalarType::F8E4M3FN
        | ScalarType::F8E5M2
        | ScalarType::F8E8M0FNU
        | ScalarType::F4E2M1FN => 1,
        ScalarType::I16 | ScalarType::F16 | ScalarType::BF16 => 2,
        ScalarType::I32 | ScalarType::F32 | ScalarType::TF32 => 4,
        ScalarType::I64 | ScalarType::F64 => 8,
    }
}

pub fn param_needs_load(ty: &Type) -> bool {
    match ty {
        Type::Scalar(_) => true,
        // Only scalar-element tiles need a load. Pointer-element tiles are device pointers.
        Type::Tile(TileType {
            shape,
            element_type: TileElementType::Scalar(_),
        }) => shape.is_empty(),
        _ => false,
    }
}

pub(crate) fn param_elem(ty: &Type) -> Option<ScalarType> {
    Some(match ty {
        Type::Pointer(p) => p.pointee.clone(),
        Type::TensorView(tv) => tv.element_type.clone(),
        Type::PartitionView(pv) => pv.tensor_view.element_type.clone(),
        Type::Tile(TileType {
            element_type: TileElementType::Pointer(p),
            ..
        }) => p.pointee.clone(),
        other => return scalar_of(other),
    })
}

pub(crate) fn tile_elem_count(ty: &Type) -> Option<i32> {
    match ty {
        Type::Tile(t) => i32::try_from(t.shape.iter().copied().product::<i64>()).ok(),
        _ => None,
    }
}

pub(crate) fn float_elem(ctx: &Context, s: &ScalarType) -> Option<(TypeHandle, &'static str)> {
    match s {
        ScalarType::F32 => Some((t::f32_ty(ctx), "f32")),
        ScalarType::F16 => Some((t::f16_ty(ctx), "f16")),
        ScalarType::BF16 => Some((t::bf16_ty(ctx), "bf16")),
        _ => None,
    }
}

/// The air type a wax scalar lowers to.
/// AIR has no `i4`, `tf32`, fp8/fp4, or `double`.
pub(crate) fn air_scalar(ctx: &Context, s: &ScalarType) -> Option<TypeHandle> {
    Some(match s {
        ScalarType::I1 => t::i1_ty(ctx),
        ScalarType::I8 => t::i8_ty(ctx),
        ScalarType::I16 => t::i16_ty(ctx),
        ScalarType::I32 => t::i32_ty(ctx),
        ScalarType::I64 => t::i64_ty(ctx),
        ScalarType::F16 => t::f16_ty(ctx),
        ScalarType::BF16 => t::bf16_ty(ctx),
        ScalarType::F32 => t::f32_ty(ctx),
        _ => return None,
    })
}

/// The air type a wax scalar lowers to as a memory element.
/// Same as `air_scalar` except no `i1`, because bools are represented as a byte in memory.
pub(crate) fn mem_elem_ty(ctx: &Context, s: &ScalarType) -> Option<TypeHandle> {
    match s {
        ScalarType::I1 => None,
        other => air_scalar(ctx, other),
    }
}

/// The scalar a type is made of. `None` when it has none.
pub(crate) fn scalar_of(ty: &Type) -> Option<ScalarType> {
    Some(match ty {
        Type::Scalar(s) => s.clone(),
        Type::Tile(TileType {
            element_type: TileElementType::Scalar(s),
            ..
        }) => s.clone(),
        Type::TensorView(tv) => tv.element_type.clone(),
        Type::PartitionView(pv) => pv.tensor_view.element_type.clone(),
        _ => return None,
    })
}

pub(crate) fn atomic_scope(compiler_scope: i64) -> Option<i32> {
    Some(match compiler_scope {
        0 => 1, // TileBlock -> threadgroup
        1 => 2, // Device
        2 => 4, // System
        _ => return None,
    })
}

/// The AIR atomic suffix for an element.
pub(crate) fn atomic_air_ty(elem_ll: &str) -> Option<&'static str> {
    Some(match elem_ll {
        "float" => "f32",
        "half" => "f16",
        "bfloat" => "bf16",
        "i32" => "i32",
        _ => return None,
    })
}

/// AIR intrinsic for an atomic read-modify-write, if it exists for the provided parameters.
///
/// `mode` is [`wax_ir::attr::AtomicRMWMode`]. `Xchg` is not supported.
pub(crate) fn atomic_rmw_intrinsic(mode: i64, elem_ll: &str) -> Option<(String, &'static str)> {
    // Device memory by construction; see the cas arm in `elementwise.rs`.
    let mem_space = "global";
    let air_ty = atomic_air_ty(elem_ll)?;
    let is_float = matches!(elem_ll, "float" | "half" | "bfloat");
    let op_part = if is_float {
        match (mode, air_ty) {
            // `AddF` on f32
            // TODO: Double check that there are no other atomic intrinsics for f32, or that
            // can/should be supported somehow.
            (4, "f32") => "add",
            _ => return None,
        }
    } else {
        match mode {
            0 => "and.s",
            1 => "or.s",
            2 => "xor.s",
            3 => "add.s",
            5 => "max.s",
            6 => "min.s",
            7 => "max.u",
            8 => "min.u",
            _ => return None,
        }
    };
    Some((
        format!("@air.atomic.{mem_space}.{op_part}.{air_ty}"),
        air_ty,
    ))
}

pub(crate) fn atomic_ty(ctx: &Context, air_ty: &str) -> Option<TypeHandle> {
    Some(match air_ty {
        "f32" => t::f32_ty(ctx),
        "f16" => t::f16_ty(ctx),
        "bf16" => t::bf16_ty(ctx),
        "i32" => t::i32_ty(ctx),
        "i64" => t::i64_ty(ctx),
        _ => return None,
    })
}

/// Returns (nr, nc) in 8x8 tiles for any MxN f16/bf16/f32/tf32 tile where
/// M and N are positive multiples of 8. None for all other types.
pub fn sgmma_dims(ty: &Type) -> Option<(usize, usize)> {
    let (shape, elem_scalar) = match ty {
        Type::Tile(TileType {
            shape,
            element_type: TileElementType::Scalar(s),
        }) => (shape, s),
        _ => return None,
    };
    if shape.len() != 2 || shape[0] <= 0 || shape[0] % 8 != 0 || shape[1] <= 0 || shape[1] % 8 != 0
    {
        return None;
    }
    match elem_scalar.clone() {
        ScalarType::F16 | ScalarType::BF16 | ScalarType::F32 | ScalarType::TF32 => {
            Some((shape[0] as usize / 8, shape[1] as usize / 8))
        }
        _ => None,
    }
}

// TODO: Migrate all unroll related concepts out of `types.rs`. `metadata.rs` or `utils.rs`.

// TODO: Simplify to `Option<u32>` and move the zero check to caller.
pub(crate) fn loop_unroll_override() -> Option<Option<u32>> {
    let v = std::env::var_os("WAX_METAL_UNROLL")?;
    let n = v.to_str().and_then(|s| s.parse::<u32>().ok())?;
    Some((n != 0).then_some(n))
}

/// LLVM unrolling defaults.
const UNROLL_PARTIAL_THRESHOLD: usize = 150;
const UNROLL_BE_INSNS: usize = 2;
const UNROLL_RUNTIME_COUNT: u32 = 8;

/// LLVM's unroll factor approach.
///
/// `(LoopSize - BEInsns) * Count + BEInsns`.
/// Our `body_insts` already excludes the terminator, aka `LoopSize - BEInsns`.
///
/// Approach is simple. Start at `UNROLL_RUNTIME_COUNT` and halve until budget fits.
/// `count <= 1` implies no unrolling aka `None`.
pub(crate) fn loop_unroll_factor(body_straight: bool, body_insts: usize) -> Option<u32> {
    if !body_straight || body_insts == 0 {
        return None;
    }
    let budget = UNROLL_PARTIAL_THRESHOLD - UNROLL_BE_INSNS;
    let mut count = UNROLL_RUNTIME_COUNT;
    while count > 1 && body_insts * count as usize > budget {
        count >>= 1;
    }
    (count > 1).then_some(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_atomic_scope_is_rejected() {
        assert_eq!(atomic_scope(0), Some(1), "TileBlock -> threadgroup");
        assert_eq!(atomic_scope(1), Some(2), "Device");
        assert_eq!(atomic_scope(2), Some(4), "System");
        for bad in [-1, 3, 7, i64::MAX] {
            assert_eq!(
                atomic_scope(bad),
                None,
                "scope {bad} must not fabricate device"
            );
        }
    }

    #[test]
    fn atomic_air_ty_rejects_unsupported_types() {
        assert_eq!(atomic_air_ty("float"), Some("f32"));
        assert_eq!(atomic_air_ty("i32"), Some("i32"));
        assert_eq!(atomic_air_ty("i64"), None);
        assert_eq!(atomic_air_ty("i1"), None);
        assert_eq!(atomic_air_ty("i8"), None);
        assert_eq!(atomic_air_ty("i16"), None);
        assert_eq!(atomic_air_ty("double"), None);
    }

    #[test]
    fn missing_atomic_mode_is_rejected() {
        // Mode 3 is `add.s`.
        assert!(atomic_rmw_intrinsic(3, "i32").is_some());
        assert!(atomic_rmw_intrinsic(99, "i32").is_none());
    }
}
