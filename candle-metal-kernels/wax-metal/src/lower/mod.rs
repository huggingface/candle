//! wax IR -> AIR dialect -> LLVM bitcode -> `.metallib`.

// Bitcode emitter and AIR dialect to bitcode bridge.
pub mod to_bitcode;
use crate::LowerError;
use crate::profile::TargetProfile;
use crate::types::{is_float, ll_scalar, param_needs_load, scalar_byte_width, sgmma_dims};
use crate::wax_to_air::EntryUnit;
use half::{bf16, f16};
use std::collections::HashMap;
use wax_ir::Module as WaxModule;
use wax_ir::{DenseElements, ScalarType, TileElementType, TileType, Type};

// AIR intrinsic names, shared with the module `declare` strings.
pub(crate) const AIR_WG_BARRIER: &str = "@air.wg.barrier";
pub(crate) const AIR_SIMD_SHUFFLE_XOR_F32: &str = "@air.simd_shuffle_xor.f32";

/// `DenseElements` attribute as structured value.
/// Used by the bitcode emitter for records and the conversion reads for its constant pool.
// TODO: Change
#[derive(Clone, PartialEq)]
pub(crate) enum ImmVal {
    Int(i64),
    FloatBits(u64),
    VecFloatBits(Vec<u64>),
    Undef,
    Zero,
}

// Not derived because of the float raw bits.
impl std::fmt::Debug for ImmVal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Int(v) => write!(f, "Int({v})"),
            Self::FloatBits(b) => write!(f, "FloatBits({b:#x})"),
            Self::VecFloatBits(b) => {
                let hex: Vec<String> = b.iter().map(|v| format!("{v:#x}")).collect();
                write!(f, "VecFloatBits([{}])", hex.join(", "))
            }
            Self::Undef => write!(f, "Undef"),
            Self::Zero => write!(f, "Zero"),
        }
    }
}

/// Lower a single entry [`WaxModule`] function all the way to `.metallib` bytes.
///
/// Errors if the module holds more than one entry.
// TODO: `module.entries()` is pub. The default behaviour of `lower_to_metallib` should be to
// compile all declared entry functions.
pub fn lower_to_metallib(module: &mut WaxModule) -> Result<Vec<u8>, LowerError> {
    lower_to_metallib_for(module, &TargetProfile::host())
}

/// Max amount of full optimization loops. Avoids two passes that undo each other spinning.
const OPT_MAX_ITERS: usize = 8;

/// Run optimization passes on an entry function.
fn optimize(ctx: &mut pliron::context::Context, func: wax_ir::OpId) -> Result<(), LowerError> {
    use wax_ir::passes::{AnalysisManager, Fixpoint, Pass as _, Passes};
    let mut opts = Passes::default();
    opts.add_pass(wax_ir::passes::cross_loop_reuse::CrossLoopReusePass::default());
    let mut analyses = AnalysisManager::default();
    Fixpoint::new(opts, OPT_MAX_ITERS)
        .run(func, ctx, &mut analyses)
        .map(|_| ())
        .map_err(|e| LowerError::MalformedIr(format!("pass pipeline: {e:?}")))
}

/// [`lower_to_metallib`] with an explicit target. Allows cross-compilation.
pub fn lower_to_metallib_for(
    module: &mut WaxModule,
    profile: &TargetProfile,
) -> Result<Vec<u8>, LowerError> {
    if module.entries().len() != 1 {
        return Err(LowerError::MalformedIr(format!(
            "lower_to_metallib: expected exactly one entry function, found {}",
            module.entries().len()
        )));
    }
    let func = module.entries()[0];
    optimize(&mut module.ctx, func)?;
    let args = kernel_args_for_module(module, func)
        .ok_or_else(|| LowerError::MalformedIr("entry function has no entry block".into()))?;
    crate::wax_to_air::lower_to_metallib_via_conversion(module, &args, profile)
        .map_err(LowerError::Unimplemented)
}

/// [`lower_to_metallib_for`] for a module with multiple entry functions.
/// Each entry is a named kernel in the same metallib.
pub fn lower_to_metallib_entries(
    module: &mut WaxModule,
    entries: &[EntryUnit],
    profile: &TargetProfile,
) -> Result<Vec<u8>, LowerError> {
    for e in entries {
        optimize(&mut module.ctx, e.func)?;
    }
    crate::wax_to_air::lower_to_metallib_multi(module, entries, profile)
        .map_err(LowerError::Unimplemented)
}

/// Wax IR scalar of a buffer parameter.
// TODO: Move to wax-ir
pub(crate) fn buffer_elem(ty: &Type) -> Option<&wax_ir::types::ScalarType> {
    Some(match ty {
        Type::Pointer(p) => &p.pointee,
        Type::TensorView(tv) => &tv.element_type,
        Type::PartitionView(pv) => &pv.tensor_view.element_type,
        Type::Tile(t) => match &t.element_type {
            TileElementType::Pointer(p) => &p.pointee,
            TileElementType::Scalar(s) => s,
        },
        _ => return None,
    })
}

pub(crate) fn kernel_args_for_signature(
    ptr_param_types: &[&Type],
    n_stride_args: usize,
    n_shape_args: usize,
) -> Vec<crate::bitcode::metadata::KernelArg> {
    use crate::bitcode::metadata::{Access, KernelArg};
    let mut args: Vec<KernelArg> = Vec::new();
    for (slot, ty) in ptr_param_types.iter().enumerate() {
        // Defaulting to (4, float) is not guaranteed to be correct, but the kernel has already
        // been verified, converted, and built, so shouldn't be an issue.
        // TODO: Change to `.expect` or return `Result`.
        let (elem_size, type_name) = buffer_elem(ty)
            .map(|s| (scalar_byte_width(s), ll_scalar(s)))
            .unwrap_or((4, "float"));
        args.push(KernelArg::Buffer {
            index: slot as u32,
            // TODO: Consider removing `address_space` since constant space has already been
            // mapped via `const_param_slots`.
            address_space: 1,
            access: Access::ReadWrite,
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

/// Extract kernel arguments from module.
pub fn kernel_args_for_module(
    module: &WaxModule,
    func: pliron::context::Ptr<pliron::operation::Operation>,
) -> Option<Vec<crate::bitcode::metadata::KernelArg>> {
    let ctx = &module.ctx;
    let entry = crate::wax_to_air::entry_block(ctx, func)?;
    let params: Vec<Type> = wax_ir::build::block_arg_types_in(ctx, entry)?
        .into_iter()
        .filter(|ty| !param_needs_load(ty))
        .collect();
    let ptr_param_types: Vec<&Type> = params.iter().collect();
    Some(kernel_args_for_signature(
        &ptr_param_types,
        crate::wax_to_air::promoted_dyn_stride_args(ctx, func, entry).len(),
        crate::wax_to_air::promoted_extent_args(ctx, func, entry).len(),
    ))
}

// TODO: change to something like
// enum DynArg {
//     Extent { param: usize, dim: usize },
//     Stride { param: usize, dim: usize },   // or Stride(Unknown)
// }
/// Tensor dimension/stride to infer from types, to be added to the function signature.
/// Coordinates into the function signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromotedScalar {
    /// Tensor parameter index to this extent.
    pub param: usize,
    /// Tensor dimension.
    pub dim: usize,
}

/// Extract dynamic dimensions/strides from the types in the module IR.
pub fn promoted_scalars(module: &WaxModule) -> Vec<PromotedScalar> {
    let Some(&func) = module.entries().first() else {
        return Vec::new();
    };
    promoted_scalars_for(module, func)
}

/// The same, for a named function.
pub fn promoted_scalars_for(module: &WaxModule, func: wax_ir::OpId) -> Vec<PromotedScalar> {
    use pliron::r#type::Typed;
    let ctx = &module.ctx;
    let Some(entry) = crate::wax_to_air::entry_block(ctx, func) else {
        return Vec::new();
    };
    let n_args = entry.deref(ctx).get_num_arguments();
    // Get argument index
    let arg_index = |v: pliron::value::Value| -> Option<usize> {
        (0..n_args).find(|&i| entry.deref(ctx).get_argument(i) == v)
    };

    // Map each dynamic dimension/stride to the (`tensor view base`, `dim`) it is the extent of.
    let mut desc: HashMap<pliron::value::Value, PromotedScalar> = HashMap::new();
    crate::wax_to_air::for_each_block(ctx, func, &mut |ops| {
        for &op in ops {
            if crate::wax_to_air::try_opcode_of(ctx, op)
                != Some(wax_ir::opcode::Opcode::MakeTensorView)
            {
                continue;
            }
            let (n_base, n_dyn_shape, _) = crate::wax_to_air::wax_operand_segments(ctx, op);
            if n_dyn_shape == 0 {
                continue;
            }

            let operands: Vec<_> = op.deref(ctx).operands().collect();
            let Some(&base) = operands.first() else {
                continue;
            };
            let Some(param) = arg_index(base) else {
                continue;
            };

            let shape: Vec<i64> = match wax_ir::dialect::types::from_pliron(
                ctx,
                op.deref(ctx).get_result(0).get_type(ctx),
            ) {
                Some(wax_ir::types::Type::TensorView(tv)) => tv.shape.clone(),
                _ => continue,
            };
            let dyn_dims: Vec<usize> = shape
                .iter()
                .enumerate()
                .filter(|(_, e)| **e == wax_ir::DYNAMIC)
                .map(|(i, _)| i)
                .collect();
            // The i-th dynamic shape operand is the extent of the i-th dynamic shape
            for (i, &sv) in operands
                .iter()
                .skip(n_base)
                .take(n_dyn_shape.min(operands.len().saturating_sub(n_base)))
                .enumerate()
            {
                let ba = crate::wax_to_air::trace_through_assume(ctx, sv);

                if let Some(&dim) = dyn_dims.get(i) {
                    desc.entry(ba).or_insert(PromotedScalar { param, dim });
                }
            }
        }
    });

    // Binding order equals kernel signature order. Dynamic strides, then extents.
    let mut out = Vec::new();
    for v in crate::wax_to_air::promoted_dyn_stride_args(ctx, func, entry)
        .into_iter()
        .chain(crate::wax_to_air::promoted_extent_args(ctx, func, entry))
    {
        if let Some(&d) = desc.get(&v) {
            out.push(d);
        } else {
            // Could not infer. Must be provided by caller.
            out.push(PromotedScalar {
                param: usize::MAX,
                dim: usize::MAX,
            });
        }
    }
    out
}

/// Amount of values this module requires to be bound at runtime (after buffers).
pub fn promoted_scalar_count(module: &WaxModule) -> usize {
    promoted_scalars(module).len()
}

// TODO: Move these helpers out?

/// The leading `N` bytes, or `None` when the blob is short.
fn le<const N: usize>(d: &[u8]) -> Option<[u8; N]> {
    d.first_chunk::<N>().copied()
}

/// The AIR element name an atomic's pointer operand addresses.
// TODO: Should we converge all of these with traits and dyn dispatch?
pub(crate) fn ptr_elem_ll(ty: &Type) -> Option<&'static str> {
    buffer_elem(ty).map(ll_scalar)
}

/// Extract structured value from `DenseElements` attribute.
pub(crate) fn dense_elements_value(de: &DenseElements, ty: &Type) -> crate::lower::ImmVal {
    if matches!(sgmma_dims(ty), Some((1, 1))) {
        if de.data.is_empty() || de.data.iter().all(|&b| b == 0) {
            return ImmVal::Zero;
        }
        // TODO: This seems off.
        let bits = (0..64usize)
            .map(|i| {
                let off = i * 4;
                let v = de
                    .data
                    .get(off..)
                    .and_then(le::<4>)
                    .map_or(0.0f32, f32::from_le_bytes);
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
    // `None` from an arm means the blob is short for its declared width. Too few bytes, or a
    // scalar with no arm, is a malformed constant: refuse rather than decode it as zero.
    let d = de.data.as_slice();
    let decoded = match scalar {
        ScalarType::F32 | ScalarType::TF32 => {
            le::<4>(d).map(|w| ImmVal::FloatBits(f32::from_le_bytes(w).to_bits() as u64))
        }
        ScalarType::F64 => le::<8>(d).map(|w| ImmVal::FloatBits(f64::from_le_bytes(w).to_bits())),
        // 2 bytes: store directly. 4 bytes: convert f32 -> f16/bf16.
        // TODO: Support f64->f16/bf16 as well?
        ScalarType::F16 => match d.len() {
            2..=3 => le::<2>(d).map(|w| ImmVal::FloatBits(u16::from_le_bytes(w) as u64)),
            _ => le::<4>(d)
                .map(|w| ImmVal::FloatBits(f16::from_f32(f32::from_le_bytes(w)).to_bits() as u64)),
        },
        ScalarType::BF16 => match d.len() {
            2..=3 => le::<2>(d).map(|w| ImmVal::FloatBits(u16::from_le_bytes(w) as u64)),
            _ => le::<4>(d)
                .map(|w| ImmVal::FloatBits(bf16::from_f32(f32::from_le_bytes(w)).to_bits() as u64)),
        },
        ScalarType::I32 => le::<4>(d).map(|w| ImmVal::Int(i32::from_le_bytes(w) as i64)),
        ScalarType::I64 => le::<8>(d).map(|w| ImmVal::Int(i64::from_le_bytes(w))),
        ScalarType::I16 => le::<2>(d).map(|w| ImmVal::Int(i16::from_le_bytes(w) as i64)),
        ScalarType::I8 => d.first().map(|&b| ImmVal::Int(b as i8 as i64)),
        _ => None,
    };
    decoded.unwrap_or(ImmVal::Undef)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wax_ir::types::ScalarType as S;

    fn de(elem: S, data: Vec<u8>) -> DenseElements {
        DenseElements {
            element_type: Type::Scalar(elem.clone()),
            shape: vec![],
            data,
        }
    }

    #[test]
    fn invalid_dense_elements_value_call_is_undef() {
        // i32 needs 4 bytes. 2 is invalid.
        let v = dense_elements_value(&de(S::I32, vec![1, 0]), &Type::Scalar(S::I32));
        assert_eq!(v, ImmVal::Undef);
        let v = dense_elements_value(&de(S::F32, vec![0, 0, 128]), &Type::Scalar(S::F32));
        assert_eq!(v, ImmVal::Undef);
        // Half needs 2 bytes.
        let v = dense_elements_value(&de(S::F16, vec![1]), &Type::Scalar(S::F16));
        assert_eq!(v, ImmVal::Undef);
        let v = dense_elements_value(&de(S::BF16, vec![1]), &Type::Scalar(S::BF16));
        assert_eq!(v, ImmVal::Undef);
    }

    #[test]
    fn dense_elements_value_half_type_supports_f32_source_blob() {
        for (elem, src, bits) in [
            (S::F16, 1.5f32, 0x3E00u64),
            (S::F16, 3.0517578e-5, 0x0200),
            (S::F16, 5.9604645e-8, 0x0001),
            (S::F16, 2.9802322e-8, 0x0000),
            (S::F16, 65520.0, 0x7C00),
            (S::BF16, 1.5, 0x3FC0),
            (S::BF16, 3.14159265, 0x4049),
            (S::BF16, 1.1754944e-38, 0x0080),
        ] {
            let ty = Type::Scalar(elem.clone());
            assert_eq!(
                dense_elements_value(&de(elem.clone(), src.to_le_bytes().to_vec()), &ty),
                ImmVal::FloatBits(bits),
                "{elem:?} {src:e}"
            );
        }
    }

    #[test]
    fn dense_elements_value_decodes_correctly() {
        assert_eq!(
            dense_elements_value(
                &de(S::I32, 7i32.to_le_bytes().to_vec()),
                &Type::Scalar(S::I32)
            ),
            ImmVal::Int(7)
        );
        assert_eq!(
            dense_elements_value(
                &de(S::F32, 1.5f32.to_le_bytes().to_vec()),
                &Type::Scalar(S::F32)
            ),
            ImmVal::FloatBits(1.5f32.to_bits() as u64)
        );
        assert_eq!(
            dense_elements_value(
                &de(S::BF16, 1.5f32.to_le_bytes().to_vec()),
                &Type::Scalar(S::BF16)
            ),
            ImmVal::FloatBits(0x3fc0)
        );
        assert_eq!(
            dense_elements_value(
                &de(S::F16, 0x7E00u16.to_le_bytes().to_vec()),
                &Type::Scalar(S::F16)
            ),
            ImmVal::FloatBits(0x7e00)
        );
    }
}
