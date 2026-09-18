//! CUDA Tile IR -> LLVM IR type mapping.

use wax_ir::{ScalarType, TileElementType, TileType, Type};

/// Lower a scalar type to an LLVM IR **storage** type name.
///
/// This is the type used for GEP, load, and store instructions (memory ops).
/// For `External` types, delegates to the storage type.
/// Use [`ll_compute_scalar`] for arithmetic / in-register contexts.
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
        ScalarType::External(t) => {
            let st = t.storage_type();
            ll_scalar(&st)
        }
    }
}

/// Lower a scalar type to its LLVM IR **compute** (in-register) type name.
///
/// For built-in types this is identical to [`ll_scalar`].  For `External`
/// types this returns the compute type (e.g. "half" for fp8E4M3FN stored as
/// i8 but computed in f16).  Use this for arithmetic instructions on loaded
/// tile values.
pub fn ll_compute_scalar(s: &ScalarType) -> &'static str {
    if let ScalarType::External(t) = s {
        let ct = t.compute_type();
        return ll_scalar(&ct);
    }
    ll_scalar(s)
}

/// True if the scalar type is a floating-point type.
pub fn is_float(s: &ScalarType) -> bool {
    match s {
        ScalarType::F16
        | ScalarType::BF16
        | ScalarType::F32
        | ScalarType::TF32
        | ScalarType::F64
        | ScalarType::F8E4M3FN
        | ScalarType::F8E5M2
        | ScalarType::F8E8M0FNU
        | ScalarType::F4E2M1FN => true,
        ScalarType::External(t) => t.is_float(),
        _ => false,
    }
}

/// Byte width of a scalar type (min 1).
///
/// For `External` types, returns the storage-type byte width (memory footprint).
pub fn scalar_byte_width(s: &ScalarType) -> u32 {
    match s {
        ScalarType::I1 | ScalarType::I4 | ScalarType::I8 => 1,
        ScalarType::F8E4M3FN
        | ScalarType::F8E5M2
        | ScalarType::F8E8M0FNU
        | ScalarType::F4E2M1FN => 1,
        ScalarType::I16 | ScalarType::F16 | ScalarType::BF16 => 2,
        ScalarType::I32 | ScalarType::F32 | ScalarType::TF32 => 4,
        ScalarType::I64 | ScalarType::F64 => 8,
        ScalarType::External(t) => t.byte_width(),
    }
}

/// Lower a cutile `Type` to an LLVM IR type string.
///
/// `TensorView`, `PartitionView`, and related view types are lowered to their
/// underlying device pointer; shape/stride information lives only in the type
/// and is consulted at the use site (e.g. `LoadViewTko`).
pub fn ll_type(ty: &Type) -> String {
    match ty {
        Type::Scalar(s) => ll_scalar(s).to_string(),
        Type::Pointer(p) => format!("{} addrspace(1)*", ll_scalar(&p.pointee)),
        Type::Tile(TileType {
            shape,
            element_type,
        }) => {
            let elem = match element_type {
                // In-register type: use compute type for External (e.g. "half" for fp8 stored as i8).
                TileElementType::Scalar(s) => ll_compute_scalar(s).to_string(),
                TileElementType::Pointer(p) => format!("{} addrspace(1)*", ll_scalar(&p.pointee)),
            };
            if shape.is_empty() {
                // Scalar tile - a single register value.
                elem
            } else {
                let n: i64 = shape.iter().product();
                // 8x8 tiles become SGMMA <64 x float> registers.
                // Use is_sgmma_tile so External types (e.g. fp8 computing in f16) also map here.
                if is_sgmma_tile(&Type::Tile(TileType {
                    shape: shape.clone(),
                    element_type: element_type.clone(),
                })) {
                    return "<64 x float>".to_string();
                }
                format!("[{n} x {elem}]")
            }
        }
        // View types: underlying device pointer (shape/stride accessed via type metadata).
        Type::TensorView(tv) => format!("{} addrspace(1)*", ll_scalar(&tv.element_type)),
        Type::PartitionView(pv) => {
            format!("{} addrspace(1)*", ll_scalar(&pv.tensor_view.element_type))
        }
        Type::GatherScatterView(gsv) => {
            format!("{} addrspace(1)*", ll_scalar(&gsv.tensor_view.element_type))
        }
        Type::StridedView(sv) => {
            format!("{} addrspace(1)*", ll_scalar(&sv.tensor_view.element_type))
        }
        // Token: a synchronisation handle - we represent it as i1 (unused at runtime).
        Type::Token => "i1".to_string(),
        Type::Func(_) => "void ()*".to_string(),
    }
}

/// Lower a type as it appears in the AIR function signature.
///
/// - Pointer types -> `{elem} addrspace(1)*` (device buffer)
/// - View types -> `{elem} addrspace(1)*` (device buffer)
/// - Scalar types -> `{ty} addrspace(2)*` (passed as const reference in Metal ABI)
/// - Tile<[], T> -> `{T} addrspace(2)*` (scalar register passed by const ref)
pub fn ll_param_type(ty: &Type) -> String {
    match ty {
        Type::Scalar(s) => format!("{} addrspace(2)*", ll_scalar(s)),
        // Scalar-element empty-shape tile: passed by const-ref in addrspace(2) (e.g. stride i32).
        Type::Tile(TileType {
            shape,
            element_type: TileElementType::Scalar(s),
        }) if shape.is_empty() => {
            format!("{} addrspace(2)*", ll_scalar(s))
        }
        // Pointer-element empty-shape tile: a device buffer pointer (addrspace 1).
        Type::Tile(TileType {
            shape,
            element_type: TileElementType::Pointer(p),
        }) if shape.is_empty() => {
            format!("{} addrspace(1)*", ll_scalar(&p.pointee))
        }
        Type::Pointer(p) => format!("{} addrspace(1)*", ll_scalar(&p.pointee)),
        Type::TensorView(tv) => format!("{} addrspace(1)*", ll_scalar(&tv.element_type)),
        Type::PartitionView(pv) => {
            format!("{} addrspace(1)*", ll_scalar(&pv.tensor_view.element_type))
        }
        Type::GatherScatterView(gsv) => {
            format!("{} addrspace(1)*", ll_scalar(&gsv.tensor_view.element_type))
        }
        Type::StridedView(sv) => {
            format!("{} addrspace(1)*", ll_scalar(&sv.tensor_view.element_type))
        }
        _ => ll_type(ty),
    }
}

/// Whether a parameter needs to be loaded on entry (scalar ABI: passed as
/// `addrspace(2)*`, so we emit a load at the start of the function body).
pub fn param_needs_load(ty: &Type) -> bool {
    match ty {
        Type::Scalar(_) => true,
        // Only scalar-element tiles need a load; pointer-element tiles are device pointers.
        Type::Tile(TileType {
            shape,
            element_type: TileElementType::Scalar(_),
        }) => shape.is_empty(),
        _ => false,
    }
}

/// Returns (nr, nc) in 8x8 sub-tile units for any MxN f16/bf16/f32/tf32 tile where
/// M and N are positive multiples of 8. Returns None for all other types.
///
/// Also accepts `External` scalar types whose `compute_type()` is F16, BF16, F32, or TF32
/// (e.g. fp8 types that compute in f16).
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
    let compute = match elem_scalar {
        ScalarType::External(t) => t.compute_type(),
        other => other.clone(),
    };
    match compute {
        ScalarType::F16 | ScalarType::BF16 | ScalarType::F32 | ScalarType::TF32 => {
            Some((shape[0] as usize / 8, shape[1] as usize / 8))
        }
        _ => None,
    }
}

/// Whether a type maps to one or more SGMMA registers (any MxN f32/tf32 tile where
/// M and N are positive multiples of 8).
pub fn is_sgmma_tile(ty: &Type) -> bool {
    sgmma_dims(ty).is_some()
}

/// 64-bit hex representation of an f32 value for LLVM IR float literals.
pub fn float_hex(v: f32) -> String {
    // LLVM uses 64-bit hex for all float constants in text IR.
    format!("0x{:016X}", (v as f64).to_bits())
}

/// LLVM IR half-precision hex literal (0xH prefix, 4 hex digits).
pub fn half_hex(v: f32) -> String {
    format!("0xH{:04X}", f32_to_f16_bits(v))
}

/// LLVM IR bfloat16 hex literal (0xR prefix, 4 hex digits).
pub fn bfloat_hex(v: f32) -> String {
    format!("0xR{:04X}", f32_to_bf16_bits(v))
}

pub(crate) fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 31) as u16) << 15;
    let exp32 = ((bits >> 23) & 0xff) as i32;
    let mant32 = bits & 0x7f_ffff;
    if exp32 == 0xff {
        return if mant32 != 0 {
            sign | 0x7e00
        } else {
            sign | 0x7c00
        };
    }
    let exp16 = exp32 - 127 + 15;
    if exp16 >= 31 {
        return sign | 0x7c00;
    }
    if exp16 <= 0 {
        if exp16 < -10 {
            return sign;
        }
        let mant = (0x80_0000 | mant32) >> (1 - exp16) as u32;
        return sign | mant as u16;
    }
    sign | ((exp16 as u16) << 10) | ((mant32 >> 13) as u16)
}

pub(crate) fn f32_to_bf16_bits(v: f32) -> u16 {
    if v.is_nan() {
        return 0x7fc0;
    }
    let bits = v.to_bits();
    let lsb = (bits >> 16) & 1;
    ((bits.wrapping_add(0x7fff + lsb)) >> 16) as u16
}
