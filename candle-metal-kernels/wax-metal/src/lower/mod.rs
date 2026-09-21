//! The lowering: wax IR -> AIR dialect -> LLVM bitcode -> `.metallib`, in process.
//!
//! The route is the pliron dialect conversion in `crate::wax_to_air`: the entry
//! function's `wax` ops are rewritten in place into `air` ops, and `to_bitcode`
//! emits the bitcode the AIR reader wants. Nothing external is invoked, so this
//! works where `xcrun` does not exist (iOS, sandboxed apps).

/// The bitcode emitter and the AIR-dialect-to-bitcode bridge.
pub mod to_bitcode;
/// The tile-level IR, re-exported from the crate the frontend shares.

// AIR intrinsic names, shared with the module `declare` strings. Constants
// prevent declaration/call drift.
pub(crate) const AIR_WG_BARRIER: &str = "@air.wg.barrier";
pub(crate) const AIR_SIMD_SHUFFLE_XOR_F32: &str = "@air.simd_shuffle_xor.f32";

use std::collections::HashSet;

use wax_ir::{DenseElements, ScalarType, TileElementType, TileType, Type};

use wax_ir::Module as TileModule;

use crate::LowerError;
use crate::profile::TargetProfile;
use crate::types::{
    is_float, ll_scalar, param_needs_load, scalar_byte_width, sgmma_dims,
};

/// The structured value of a `DenseElements` attribute: what the bitcode emitter
/// records and the conversion reads for its constant pool.
#[derive(Clone, PartialEq, Debug)]
pub(crate) enum ImmVal {
    Int(i64),
    /// A float constant as its raw bit pattern, at the operand type's width. This is
    /// what bitcode's float constant record stores, and it is the only form that can
    /// carry f16/bf16 specials (infinities) without going through an f32.
    FloatBits(u64),
    /// A vector of float constants by raw bits, e.g. a `<64 x float>` SGMMA tile.
    VecFloatBits(Vec<u64>),
    Undef,
    Zero,
}

// Constant collection

// Public entrypoint

/// Lower `module`'s single entry function all the way to `.metallib` bytes in process:
/// LLVM bitcode written here, wrapped in a container written here. Nothing external is
/// invoked, so this works where `xcrun` does not exist (iOS, sandboxed apps).
///
/// Errors if the module holds more than one entry - a container per kernel is what the
/// callers build.
pub fn lower_to_metallib(module: &TileModule) -> Result<Vec<u8>, LowerError> {
    lower_to_metallib_for(module, &TargetProfile::host())
}

/// [`lower_to_metallib`] for an explicit target - cross-compiling a kernel for another
/// Apple platform, or for an older deployment target whose Metal runtime would reject
/// this build's AIR version.
pub fn lower_to_metallib_for(
    module: &TileModule,
    profile: &TargetProfile,
) -> Result<Vec<u8>, LowerError> {
    if module.functions.len() != 1 {
        return Err(LowerError::MalformedIr(format!(
            "lower_to_metallib: expected exactly one entry function, found {}",
            module.functions.len()
        )));
    }
    // The tile-level staging decisions, made here on the dialect: the conversion reads
    // the annotations to stage a cross-loop reuse, or to refuse it.
    let mut analyses = wax_ir::passes::AnalysisManager::default();
    let func = module.functions[0];
    let _ = wax_ir::passes::cross_loop_reuse::annotate_with(&mut analyses, &module.ctx, func);
    let args = kernel_args_for_module(module, func);
    crate::wax_to_air::lower_to_metallib_via_conversion(module, &args, &[], profile)
        .map_err(LowerError::Unimplemented)
}

fn buffer_elem_facts(ty: &Type) -> (u32, &'static str) {
    match ty {
        Type::Pointer(p) => (scalar_byte_width(&p.pointee), ll_scalar(&p.pointee)),
        Type::TensorView(tv) => (
            scalar_byte_width(&tv.element_type),
            ll_scalar(&tv.element_type),
        ),
        Type::PartitionView(pv) => (
            scalar_byte_width(&pv.tensor_view.element_type),
            ll_scalar(&pv.tensor_view.element_type),
        ),
        Type::Tile(TileType {
            element_type: TileElementType::Pointer(p),
            ..
        }) => (scalar_byte_width(&p.pointee), ll_scalar(&p.pointee)),
        _ => (4, "float"),
    }
}

pub(crate) fn kernel_args_for_signature(
    ptr_param_types: &[&Type],
    const_slots: &HashSet<usize>,
    n_stride_args: usize,
    n_shape_args: usize,
) -> Vec<crate::bitcode::metadata::KernelArg> {
    use crate::bitcode::metadata::{Access, KernelArg};
    let mut args: Vec<KernelArg> = Vec::new();
    for (slot, ty) in ptr_param_types.iter().enumerate() {
        let (elem_size, type_name) = buffer_elem_facts(ty);
        args.push(KernelArg::Buffer {
            index: slot as u32,
            address_space: if const_slots.contains(&slot) { 2 } else { 1 },
            // Buffer params are emitted `air.read_write` unless they bind in
            // constant space, which is read-only by construction.
            access: if const_slots.contains(&slot) {
                Access::Read
            } else {
                Access::ReadWrite
            },
            elem_size,
            elem_align: elem_size,
            type_name: type_name.to_string(),
            name: format!("param{slot}"),
        });
    }
    let stride_base_slot = ptr_param_types.len();
    for i in 0..n_stride_args {
        args.push(KernelArg::ScalarBuffer {
            index: (stride_base_slot + i) as u32,
            byte_size: 4,
            type_name: "uint".to_string(),
            name: format!("stride{i}"),
        });
    }
    let dim_base_slot = stride_base_slot + n_stride_args;
    for i in 0..n_shape_args {
        args.push(KernelArg::ScalarBuffer {
            index: (dim_base_slot + i) as u32,
            byte_size: 4,
            type_name: "uint".to_string(),
            name: format!("dim{i}"),
        });
    }
    for (kind, name) in [
        ("air.thread_position_in_threadgroup", "_tp"),
        ("air.threadgroup_position_in_grid", "_tg"),
    ] {
        args.push(KernelArg::Builtin {
            kind: kind.to_string(),
            type_name: "uint3".to_string(),
            name: name.to_string(),
        });
    }
    args
}

/// The kernel-argument metadata the bitcode writer embeds, read off the entry
/// function of the `wax`-dialect module: the buffer params (constant-space or
/// device), the promoted dynamic stride and extent scalars, and the two position
/// builtins. Everything it reads is signature-level - no pass, no materialisation.
pub fn kernel_args_for_module(
    module: &TileModule,
    func: pliron::context::Ptr<pliron::operation::Operation>,
) -> Vec<crate::bitcode::metadata::KernelArg> {
    use wax_ir::dialect::attr_mirror::WaxAttr;
    use wax_ir::source::WaxSource;
    let ctx = &module.ctx;
    let const_slots: HashSet<usize> = match "const_param_slots".try_into() {
        Ok(key) => match func.deref(ctx).attributes.get::<WaxAttr>(&key) {
            Some(a) => match &a.0 {
                wax_ir::attr::Attribute::DenseI32Array(v) => {
                    v.iter().map(|&x| x as usize).collect()
                }
                _ => HashSet::new(),
            },
            None => HashSet::new(),
        },
        Err(_) => HashSet::new(),
    };
    let entry = crate::wax_to_air::entry_block(ctx, func)
        .expect("entry function has an entry block");
    let params: Vec<Type> = module
        .block_args(entry)
        .into_iter()
        .map(|(_, t)| t)
        .filter(|ty| !param_needs_load(ty))
        .collect();
    let ptr_param_types: Vec<&Type> = params.iter().collect();
    kernel_args_for_signature(
        &ptr_param_types,
        &const_slots,
        crate::wax_to_air::promoted_dyn_stride_args(ctx, func, entry).len(),
        crate::wax_to_air::promoted_extent_args(ctx, func, entry).len(),
    )
}

fn half_source_f32(data: &[u8], is_f16: bool) -> f32 {
    if data.len() >= 4 {
        return f32::from_le_bytes(data[..4].try_into().unwrap());
    }
    if data.len() < 2 {
        return 0.0;
    }
    let bits = u16::from_le_bytes(data[..2].try_into().unwrap());
    if !is_f16 {
        return f32::from_bits((bits as u32) << 16);
    }
    let sign = ((bits >> 15) as u32) << 31;
    let exp = (((bits >> 10) & 0x1f) as u32)
        .saturating_add(127)
        .saturating_sub(15)
        << 23;
    let mant = ((bits & 0x3ff) as u32) << 13;
    f32::from_bits(sign | exp | mant)
}

pub(crate) fn ptr_elem_type_and_as(ty: &Type) -> (String, u32) {
    match ty {
        Type::Pointer(p) => (ll_scalar(&p.pointee).to_string(), 1),
        Type::TensorView(tv) => (ll_scalar(&tv.element_type).to_string(), 1),
        Type::PartitionView(pv) => (ll_scalar(&pv.tensor_view.element_type).to_string(), 1),
        _ => ("float".to_string(), 1),
    }
}

pub(crate) fn dense_elements_value(de: &DenseElements, ty: &Type) -> crate::lower::ImmVal {

    if matches!(sgmma_dims(ty), Some((1, 1))) {
        if de.data.is_empty() || de.data.iter().all(|&b| b == 0) {
            return ImmVal::Zero;
        }
        let bits = (0..64usize)
            .map(|i| {
                let off = i * 4;
                let v = if off + 4 <= de.data.len() {
                    f32::from_le_bytes(de.data[off..off + 4].try_into().unwrap())
                } else {
                    0.0f32
                };
                v.to_bits() as u64
            })
            .collect();
        return ImmVal::VecFloatBits(bits);
    }

    let scalar = match ty {
        Type::Scalar(s) => s.clone(),
        Type::Tile(TileType {
            element_type: TileElementType::Scalar(s),
            ..
        }) => s.clone(),
        _ => return ImmVal::Undef,
    };
    let scalar_is_float = is_float(&scalar);
    if de.data.is_empty() {
        return if scalar_is_float {
            ImmVal::FloatBits(0)
        } else {
            ImmVal::Int(0)
        };
    }
    match scalar {
        ScalarType::F32 | ScalarType::TF32 if de.data.len() >= 4 => {
            let v = f32::from_le_bytes(de.data[..4].try_into().unwrap());
            ImmVal::FloatBits(v.to_bits() as u64)
        }
        ScalarType::F64 if de.data.len() >= 8 => {
            let v = f64::from_le_bytes(de.data[..8].try_into().unwrap());
            ImmVal::FloatBits(v.to_bits())
        }
        // Data may be 2 bytes (a native half pattern) or 4 (an f32 source value).
        ScalarType::F16 => {
            let v = half_source_f32(&de.data, true);
            ImmVal::FloatBits(crate::types::f32_to_f16_bits(v) as u64)
        }
        ScalarType::BF16 => {
            let v = half_source_f32(&de.data, false);
            ImmVal::FloatBits(crate::types::f32_to_bf16_bits(v) as u64)
        }
        ScalarType::I32 if de.data.len() >= 4 => {
            ImmVal::Int(i32::from_le_bytes(de.data[..4].try_into().unwrap()) as i64)
        }
        ScalarType::I64 if de.data.len() >= 8 => {
            ImmVal::Int(i64::from_le_bytes(de.data[..8].try_into().unwrap()))
        }
        ScalarType::I16 if de.data.len() >= 2 => {
            ImmVal::Int(i16::from_le_bytes(de.data[..2].try_into().unwrap()) as i64)
        }
        ScalarType::I8 if !de.data.is_empty() => ImmVal::Int(de.data[0] as i8 as i64),
        _ if scalar_is_float => ImmVal::FloatBits(0),
        _ => ImmVal::Int(0),
    }
}
