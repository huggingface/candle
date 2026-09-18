//! Main CUDA Tile IR -> AIR lowering pass.

#[cfg(feature = "pliron-air")]
pub mod cfg;
#[cfg(not(feature = "pliron-air"))]
pub(crate) mod cfg;
/// The tile-level IR, re-exported from the crate the frontend shares.
pub use wax_ir::{build, builder, ir, source};
mod sgmma_set;
#[cfg(feature = "pliron-air")]
pub mod to_bitcode;
#[cfg(not(feature = "pliron-air"))]
pub(crate) mod to_bitcode;
mod uniformity;
mod verify;
mod walk;

// AIR intrinsic names - used in both `declare` strings at module init and at
// call sites throughout the lowerer.  Constants prevent declaration/call drift.
const AIR_WG_BARRIER: &str = "@air.wg.barrier";
const AIR_SIMD_SHUFFLE_XOR_F32: &str = "@air.simd_shuffle_xor.f32";
const AIR_SIMD_SHUFFLE_UP_F32: &str = "@air.simd_shuffle_up.f32";
const AIR_SIMD_SHUFFLE_DOWN_F32: &str = "@air.simd_shuffle_down.f32";
const AIR_FAST_EXP_F32: &str = "@air.fast_exp.f32";
const AIR_FAST_FMAX_F32: &str = "@air.fast_fmax.f32";
const AIR_FAST_ATAN2_F32: &str = "@air.fast_atan2.f32";
const AIR_SIMD_SUM_F32: &str = "@air.simd_sum.f32";
const AIR_SIMD_MAX_F32: &str = "@air.simd_max.f32";
const AIR_SIMD_MIN_F32: &str = "@air.simd_min.f32";
const AIR_SIMD_PRODUCT_F32: &str = "@air.simd_product.f32";
/// Precise (non-fast-math) unary math, as `air.<name>.<suffix>`. Every one of these
/// was verified to resolve *and compute the right value* on the GPU for f32 and f16;
/// Apple's frontend promotes f16 to f32 for them, but the f16 forms exist and work.
const PRECISE_MATH: [&str; 7] = ["fabs", "sqrt", "log", "log2", "exp2", "sin", "cos"];
use std::collections::HashSet;

use wax_ir::Opcode;
use wax_ir::{Attribute, DenseElements, ScalarType, TileElementType, TileType, Type};

use self::ir::{OpRef, ValRef, WaxIr};
use wax_ir::Module as TileModule;

use crate::LowerError;
use crate::profile::TargetProfile;
use crate::types::{
    bfloat_hex, float_hex, half_hex, is_float, ll_param_type, ll_scalar, param_needs_load,
    scalar_byte_width, sgmma_dims,
};

// Public entrypoint

/// Lower all entry functions in `module` to a single AIR text string, for the
/// `metal-as` path, targeting the platform this build runs on.
pub fn lower_to_air(module: &TileModule) -> Result<String, LowerError> {
    lower_to_air_for(module, &TargetProfile::host())
}

/// [`lower_to_air`] for an explicit target - a different platform, or an older
/// deployment target than this build's.
pub fn lower_to_air_for(
    module: &TileModule,
    profile: &TargetProfile,
) -> Result<String, LowerError> {
    Ok(lower_frontend_module(module, false, profile)?.0)
}

/// Lower `module`'s single entry function all the way to `.metallib` bytes in process:
/// LLVM bitcode written here, wrapped in a container written here. Nothing external is
/// invoked, so this works where `xcrun` does not exist (iOS, sandboxed apps).
///
/// Errors if the module holds more than one entry - a container per kernel is what the
/// callers build - or if the kernel uses a construct the bitcode writer cannot encode,
/// in which case the text path via [`lower_to_air`] still handles it.
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
    let (_, mut libs) = lower_frontend_module(module, true, profile)?;
    if libs.len() != 1 {
        return Err(LowerError::MalformedIr(format!(
            "lower_to_metallib: expected exactly one entry function, found {}",
            libs.len()
        )));
    }
    Ok(libs.remove(0).1)
}

/// Every entry function of a frontend-built module, materialised and lowered.
fn lower_frontend_module(
    module: &TileModule,
    want_metallib: bool,
    profile: &TargetProfile,
) -> Result<(String, Vec<Metallib>), LowerError> {
    // Tile-level passes run HERE, on the dialect, before it is flattened: they need the
    // structure a materialised view has already lost the ability to change. Their
    // conclusions ride along as attributes, which materialisation carries verbatim.
    // One manager for the whole module, so every tile-level pass added later shares the
    // analyses rather than recomputing them. It buys no hits today - the cache is keyed
    // (analysis, operation) and each function is asked exactly once - but it is where that
    // reuse will live, and hoisting it later would mean touching every pass at once.
    let mut analyses = wax_ir::passes::AnalysisManager::default();
    let irs: Vec<WaxIr> = module
        .functions
        .iter()
        .map(|&f| {
            // A failure here is an analysis error, not a lowering decision: fall through
            // unannotated rather than abort, which is what the pass not firing already means.
            let _ = wax_ir::passes::cross_loop_reuse::annotate_with(&mut analyses, &module.ctx, f);
            ir::from_source(module, f)
        })
        .collect();
    lower_tile_irs(&irs, want_metallib, profile, false)
}

/// Lower a metallib from wax IR directly.
///
/// The entrypoint a frontend that emits wax IR itself would use.
pub fn lower_to_metallib_from_wax_ir(
    ir: &WaxIr,
    profile: &TargetProfile,
) -> Result<Vec<u8>, LowerError> {
    let (_, mut libs) = lower_tile_irs(std::slice::from_ref(ir), true, profile, false)?;
    if libs.len() != 1 {
        return Err(LowerError::MalformedIr(format!(
            "lower_to_metallib_from_wax_ir: expected exactly one entry function, found {}",
            libs.len()
        )));
    }
    Ok(libs.remove(0).1)
}

/// One kernel's self-built container, keyed by entry-point name.
type Metallib = (String, Vec<u8>);

/// The lowering proper, from wax IR alone. Nothing below this point knows what produced
/// it - a `cutile_ir::Module`, the `wax` dialect, or a frontend building it directly.
fn lower_tile_irs(
    irs: &[WaxIr],
    want_metallib: bool,
    profile: &TargetProfile,
    air_dialect: bool,
) -> Result<(String, Vec<Metallib>), LowerError> {
    let mut out = String::new();
    let mut metallibs: Vec<Metallib> = Vec::new();

    out.push_str(&format!(
        "target datalayout = \"{}\"\n",
        profile.datalayout()
    ));
    out.push_str(&format!("target triple = \"{}\"\n\n", profile.triple()));

    // Mark barrier and SIMD intrinsics as `convergent` so the Metal LLVM
    // backend cannot legally reorder or eliminate them.  Without this flag,
    // the optimizer treats them as side-effect-free and may hoist barriers out
    // of loops or eliminate them entirely, breaking cross-SIMD synchronization.
    out.push_str(&format!("declare void {AIR_WG_BARRIER}(i32, i32) #1\n"));
    // Precise math. `@air.*` is what Apple's frontend emits; the `@llvm.*` spellings
    // that used to be here work on every toolchain measured but are not the ABI.
    // f32 and f16 only: bf16 computes in f32 and truncates back (see `intrin_ty`).
    for (ty, suf) in [("float", "f32"), ("half", "f16")] {
        for f in PRECISE_MATH {
            out.push_str(&format!("declare {ty} @air.{f}.{suf}({ty}) #1\n"));
        }
    }
    out.push_str(&format!("declare float {AIR_FAST_EXP_F32}(float) #1\n"));
    out.push_str(&format!(
        "declare float {AIR_FAST_FMAX_F32}(float, float) #1\n"
    ));
    out.push_str(&format!(
        "declare float {AIR_FAST_ATAN2_F32}(float, float) #1\n"
    ));
    // mask is i16 - passing i32 assembles but produces wrong runtime behavior
    out.push_str(&format!(
        "declare float {AIR_SIMD_SHUFFLE_XOR_F32}(float, i16) #1\n"
    ));
    out.push_str(&format!(
        "declare float {AIR_SIMD_SHUFFLE_UP_F32}(float, i16) #1\n"
    ));
    out.push_str(&format!(
        "declare float {AIR_SIMD_SHUFFLE_DOWN_F32}(float, i16) #1\n"
    ));
    out.push_str(&format!("declare float {AIR_SIMD_SUM_F32}(float) #1\n"));
    out.push_str(&format!("declare float {AIR_SIMD_MAX_F32}(float) #1\n"));
    out.push_str(&format!("declare float {AIR_SIMD_MIN_F32}(float) #1\n"));
    out.push_str(&format!("declare float {AIR_SIMD_PRODUCT_F32}(float) #1\n"));
    // Half/bfloat AIR SIMD + fast-math intrinsics (elementwise, reduce, scan, shuffle).
    for (ty, suf) in [("half", "f16"), ("bfloat", "bf16")] {
        out.push_str(&format!("declare {ty} @air.fast_exp.{suf}({ty}) #1\n"));
        out.push_str(&format!(
            "declare {ty} @air.fast_fmax.{suf}({ty}, {ty}) #1\n"
        ));
        out.push_str(&format!(
            "declare {ty} @air.fast_atan2.{suf}({ty}, {ty}) #1\n"
        ));
        out.push_str(&format!(
            "declare {ty} @air.simd_shuffle_xor.{suf}({ty}, i16) #1\n"
        ));
        out.push_str(&format!(
            "declare {ty} @air.simd_shuffle_up.{suf}({ty}, i16) #1\n"
        ));
        out.push_str(&format!(
            "declare {ty} @air.simd_shuffle_down.{suf}({ty}, i16) #1\n"
        ));
        out.push_str(&format!("declare {ty} @air.simd_sum.{suf}({ty}) #1\n"));
        out.push_str(&format!("declare {ty} @air.simd_max.{suf}({ty}) #1\n"));
        out.push_str(&format!("declare {ty} @air.simd_min.{suf}({ty}) #1\n"));
        out.push_str(&format!("declare {ty} @air.simd_product.{suf}({ty}) #1\n"));
    }
    out.push_str(
        "declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p1f32(\
        float addrspace(1)*, <2 x i64>, <2 x i64>, <2 x i64>)\n",
    );
    out.push_str(
        "declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p3f32(\
        float addrspace(3)*, <2 x i64>, <2 x i64>, <2 x i64>)\n",
    );
    out.push_str(
        "declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate\
        .v64f32.v64f32.v64f32.v64f32(<64 x float>, <64 x float>, <64 x float>)\n",
    );
    out.push_str(
        "declare void @air.simdgroup_matrix_8x8_store.v64f32.p1f32(\
        <64 x float>, float addrspace(1)*, <2 x i64>, <2 x i64>, <2 x i64>)\n",
    );
    out.push_str(
        "declare <64 x half> @air.simdgroup_matrix_8x8_load.v64f16.p1f16(\
        half addrspace(1)*, <2 x i64>, <2 x i64>, <2 x i64>)\n",
    );
    out.push_str(
        "declare <64 x bfloat> @air.simdgroup_matrix_8x8_load.v64bf16.p1bf16(\
        bfloat addrspace(1)*, <2 x i64>, <2 x i64>, <2 x i64>)\n",
    );
    out.push_str(
        "declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate\
        .v64f32.v64f16.v64f16.v64f32(<64 x half>, <64 x half>, <64 x float>)\n",
    );
    out.push_str(
        "declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate\
        .v64f32.v64bf16.v64bf16.v64f32(<64 x bfloat>, <64 x bfloat>, <64 x float>)\n",
    );
    // Same-type stores only: Metal has no `simdgroup_store(simdgroup_float8x8,
    // half *)` overload, so an f32 accumulator is fptrunc'ed per fragment first.
    out.push_str(
        "declare void @air.simdgroup_matrix_8x8_store.v64f16.p1f16(\
        <64 x half>, half addrspace(1)*, <2 x i64>, <2 x i64>, <2 x i64>)\n",
    );
    out.push_str(
        "declare void @air.simdgroup_matrix_8x8_store.v64bf16.p1bf16(\
        <64 x bfloat>, bfloat addrspace(1)*, <2 x i64>, <2 x i64>, <2 x i64>)\n",
    );
    out.push_str("declare <64 x float> @air.simdgroup_matrix_8x8_init_diag.v64f32.f32(float)\n");
    // AGX predicated memory. `edgecheck(pos, lo, hi)` returns a 4-bit element mask (bit k
    // set iff `lo <= pos+k < hi`); the masked load reads ZERO on masked-off lanes and
    // SUPPRESSES their access entirely, so a bounds-masked read needs neither an address
    // clamp to avoid faulting nor a `select` to supply the pad value. Only integer element
    // types exist - floats bitcast through. See `tests/agx_emask_probe.rs`.
    out.push_str("declare i16 @llvm.agx3.edgecheck(i32, i32, i32) #2\n");
    out.push_str(
        "declare <2 x i32> @llvm.agx3.load.with.emask.global.v2i32(\
        i32 addrspace(1)*, i16, i16, i16) #3\n",
    );
    out.push_str(
        "declare <2 x i16> @llvm.agx3.load.with.emask.global.v2i16(\
        i16 addrspace(1)*, i16, i16, i16) #3\n",
    );
    out.push_str(
        "declare void @llvm.agx3.store.with.emask.global.v2i32(\
        i32 addrspace(1)*, <2 x i32>, i16, i16, i16) #4\n",
    );
    out.push_str(
        "declare void @llvm.agx3.store.with.emask.global.v2i16(\
        i16 addrspace(1)*, <2 x i16>, i16, i16, i16) #4\n",
    );
    out.push('\n');

    let mut meta_nodes: Vec<String> = Vec::new();
    let mut kernel_meta_refs: Vec<usize> = Vec::new();

    for ir in irs {
        let sink = want_metallib.then_some(&mut metallibs);
        let r = lower_kernel(ir, &mut meta_nodes, &mut out, sink, profile, air_dialect)?;
        kernel_meta_refs.push(r);
    }

    // Module flags, in the order Apple's frontend emits them. Behaviour 1 = error,
    // 2 = warning, 7 = max.
    let (sdk_major, sdk_minor) = profile.sdk_version();
    let mut flag_refs: Vec<usize> = Vec::new();
    let mut flag = |nodes: &mut Vec<String>, node: String| {
        flag_refs.push(nodes.len());
        nodes.push(node);
    };
    flag(
        &mut meta_nodes,
        format!("!{{i32 2, !\"SDK Version\", [2 x i32] [i32 {sdk_major}, i32 {sdk_minor}]}}"),
    );
    flag(
        &mut meta_nodes,
        "!{i32 1, !\"wchar_size\", i32 4}".to_string(),
    );
    flag(
        &mut meta_nodes,
        "!{i32 7, !\"frame-pointer\", i32 2}".to_string(),
    );
    for (key, val) in profile.limits.flags() {
        flag(
            &mut meta_nodes,
            format!("!{{i32 7, !\"{key}\", i32 {val}}}"),
        );
    }

    let (air0, air1, air2) = profile.air_version;
    let ver_node = meta_nodes.len();
    meta_nodes.push(format!("!{{i32 {air0}, i32 {air1}, i32 {air2}}}"));
    let (lang0, lang1, lang2) = profile.language_version;
    let lang_node = meta_nodes.len();
    meta_nodes.push(format!(
        "!{{!\"Metal\", i32 {lang0}, i32 {lang1}, i32 {lang2}}}"
    ));
    let ident_node = meta_nodes.len();
    meta_nodes.push("!{!\"wax-metal\"}".into());

    out.push('\n');
    for (i, node) in meta_nodes.iter().enumerate() {
        out.push_str(&format!("!{i} = {node}\n"));
    }

    out.push('\n');
    let kerns: Vec<String> = kernel_meta_refs.iter().map(|i| format!("!{i}")).collect();
    out.push_str(&format!("!air.kernel = !{{{}}}\n", kerns.join(", ")));
    let flags: Vec<String> = flag_refs.iter().map(|i| format!("!{i}")).collect();
    out.push_str(&format!("!llvm.module.flags = !{{{}}}\n", flags.join(", ")));
    out.push_str(&format!("!air.version = !{{!{ver_node}}}\n"));
    out.push_str(&format!("!air.language_version = !{{!{lang_node}}}\n"));
    out.push_str(&format!("!llvm.ident = !{{!{ident_node}}}\n"));

    // #0: kernel function - must be convergent (SIMT execution model)
    // #1: GPU intrinsics (barriers, SIMD ops, math) - convergent + willreturn
    out.push_str("\nattributes #0 = { convergent nounwind }\n");
    out.push_str("attributes #1 = { convergent nounwind willreturn }\n");
    out.push_str("attributes #2 = { nounwind readnone speculatable }\n");
    out.push_str("attributes #3 = { nounwind readonly }\n");
    out.push_str("attributes #4 = { nounwind writeonly }\n");

    Ok((out, metallibs))
}

// Per-kernel lowering

fn lower_kernel(
    ir: &WaxIr,
    meta_nodes: &mut Vec<String>,
    out: &mut String,
    metallib_sink: Option<&mut Vec<Metallib>>,
    profile: &TargetProfile,
    air_dialect: bool,
) -> Result<usize, LowerError> {
    let func_op = ir.op(ir.root.expect("wax IR has a root op"));

    let raw_kernel_name = func_op
        .attributes
        .iter()
        .find(|(k, _)| k == "sym_name")
        .and_then(|(_, v)| {
            if let Attribute::String(s) = v {
                Some(s.as_str())
            } else {
                None
            }
        })
        .unwrap_or("kernel");
    // The cutile-ir compiler appends _entry to the IR function name; strip it so
    // the Metal kernel lookup name matches the user-visible function name.
    let kernel_name = raw_kernel_name
        .strip_suffix("_entry")
        .unwrap_or(raw_kernel_name);

    // Buffer-param slots (index among pointer params) to bind in constant address
    // space (addrspace 2) instead of device (addrspace 1). See
    // KernelCompiler::const_params.
    let const_slots: HashSet<usize> = func_op
        .attributes
        .iter()
        .find(|(k, _)| k == "const_param_slots")
        .and_then(|(_, v)| match v {
            Attribute::DenseI32Array(a) => Some(a.iter().map(|&x| x as usize).collect()),
            _ => None,
        })
        .unwrap_or_default();

    // Rows of `tg_store`/`tg_load` scratch this kernel needs, if it asked for a
    // specific size. See KernelCompiler::tg_scratch_rows.
    let tg_scratch_rows: Option<i32> = func_op
        .attributes
        .iter()
        .find(|(k, _)| k == "tg_scratch_rows")
        .and_then(|(_, v)| match v {
            Attribute::DenseI32Array(a) => a.first().copied(),
            _ => None,
        });

    let entry_block_id = ir.entry_block();
    let params: Vec<(ValRef, Type)> = ir.block(entry_block_id).args.clone();

    // Only device-buffer pointer params appear in the AIR signature.
    // Scalar tile params (strides, shapes) are handled separately below.
    let ptr_params: Vec<(usize, &ValRef, &Type)> = params
        .iter()
        .enumerate()
        .filter(|(_, (_, ty))| !param_needs_load(ty))
        .map(|(orig_idx, (v, ty))| (orig_idx, v, ty))
        .collect();

    // Pre-scan: find tile<i32> block args used as dynamic stride values in MakeTensorView ops.
    let dyn_stride_args = find_dyn_stride_block_args(ir, entry_block_id);
    // Runtime extents, for padded stores only - see `find_dyn_shape_block_args`. Placed
    // AFTER the stride args so a kernel that does not opt in keeps its exact signature.
    let dyn_shape_args = find_dyn_shape_block_args(ir, entry_block_id);

    // Signature element types drive the `!air.kernel` function-type metadata
    // (the function body/signature text itself is produced by the CFG backend).
    let mut sig_types: Vec<String> = Vec::new();
    // Structured mirror of the argument metadata, for the bitcode path.
    let mut air_args: Vec<crate::bitcode::metadata::KernelArg> = Vec::new();
    for (slot, (_, _, ty)) in ptr_params.iter().enumerate() {
        let t = ll_param_type(ty);
        // Constant-space buffer params bind in addrspace(2) (like stride args).
        let t = if const_slots.contains(&slot) {
            t.replace("addrspace(1)", "addrspace(2)")
        } else {
            t
        };
        sig_types.push(t);
    }
    // Each dynamic stride block arg becomes an i32 constant-buffer (addrspace 2) param.
    for _ in dyn_stride_args.iter() {
        sig_types.push("i32 addrspace(2)*".into());
    }
    for _ in dyn_shape_args.iter() {
        sig_types.push("i32 addrspace(2)*".into());
    }
    let tp_idx = ptr_params.len() + dyn_stride_args.len() + dyn_shape_args.len();
    let tg_idx = tp_idx + 1;
    sig_types.push("<3 x i32>".into());
    sig_types.push("<3 x i32>".into());

    // The typed CFG backend is now the sole lowering path: every real kernel
    // routes through it. A `None` means the kernel uses a construct the backend
    // cannot lower - which, for the current kernel set, only happens for the
    // genuinely-invalid M<8 SGMMA-accumulator GEMV (an 8x8 simdgroup fragment
    // can't be loaded from a 1-row tile). Reject it with a precise error instead
    // of emitting broken AIR.
    // `!llvm.loop` metadata referenced in the function body is numbered from the
    // current end of the module metadata table; the nodes are appended below.
    let meta_base = meta_nodes.len();
    let lowered = match walk::try_lower_via_cfg(
        ir,
        kernel_name,
        entry_block_id,
        &params,
        &dyn_stride_args,
        &dyn_shape_args,
        &walk::KernelOptions {
            const_slots: &const_slots,
            tg_scratch_rows,
            meta_base,
        },
    ) {
        Some(l) => l,
        None => {
            return Err(LowerError::MalformedIr(format!(
                "kernel @{kernel_name}: the CFG backend could not lower this kernel - \
                 it uses an unsupported or invalid construct (e.g. an M<8 SGMMA \
                 accumulator, which cannot be loaded as an 8x8 simdgroup fragment)"
            )));
        }
    };
    let fn_out = format!("; cfg-lowered\n{}\n", lowered.text);
    let tg_globals = &lowered.globals;

    // Verify SSA well-formedness before the text ever reaches metal-as, so
    // representation-mismatch defects surface as precise errors naming the
    // offending value and block rather than a cryptic assembler line number.
    verify::verify_function(&fn_out)?;

    // Append the loop metadata nodes at exactly [meta_base, meta_base + n) so
    // their positional index matches the `!llvm.loop !N` references in fn_out.
    debug_assert_eq!(meta_nodes.len(), meta_base);
    meta_nodes.extend(lowered.loop_meta.iter().cloned());

    // Emit threadgroup globals before the function body.
    for g in tg_globals {
        out.push_str(g);
        out.push('\n');
    }
    out.push_str(&fn_out);

    // AIR metadata
    let mut param_meta_refs: Vec<usize> = Vec::new();

    // Only emit metadata for device-buffer pointer params.
    for (slot, (_, _, ty)) in ptr_params.iter().enumerate() {
        let node_idx = meta_nodes.len();
        let access = "!\"air.read_write\"";
        let (elem_size, type_name): (u32, &str) = match ty {
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
        };
        let param_name = format!("param{slot}");
        // Same facts, structured, for the bitcode path's metadata.
        air_args.push(crate::bitcode::metadata::KernelArg::Buffer {
            index: slot as u32,
            address_space: if const_slots.contains(&slot) { 2 } else { 1 },
            access: if const_slots.contains(&slot) {
                crate::bitcode::metadata::Access::Read
            } else if access.contains("read_write") {
                crate::bitcode::metadata::Access::ReadWrite
            } else if access.contains("write") {
                crate::bitcode::metadata::Access::Write
            } else {
                crate::bitcode::metadata::Access::Read
            },
            elem_size,
            elem_align: elem_size,
            type_name: type_name.to_string(),
            name: param_name.clone(),
        });
        if const_slots.contains(&slot) {
            // Constant-space buffer: addrspace 2, read-only (mirrors stride args).
            meta_nodes.push(format!(
                "!{{i32 {slot}, !\"air.buffer\", \
                 !\"air.location_index\", i32 {slot}, i32 1, \
                 !\"air.read\", !\"air.address_space\", i32 2, \
                 !\"air.arg_type_size\", i32 {elem_size}, \
                 !\"air.arg_type_align_size\", i32 {elem_size}, \
                 !\"air.arg_type_name\", !\"{type_name}\", \
                 !\"air.arg_name\", !\"{param_name}\"}}"
            ));
        } else {
            meta_nodes.push(format!(
                "!{{i32 {slot}, !\"air.buffer\", !\"air.location_index\", i32 {slot}, i32 1, \
                 {access}, !\"air.address_space\", i32 1, \
                 !\"air.arg_type_size\", i32 {elem_size}, \
                 !\"air.arg_type_align_size\", i32 {elem_size}, \
                 !\"air.arg_type_name\", !\"{type_name}\", \
                 !\"air.arg_name\", !\"{param_name}\"}}"
            ));
        }
        param_meta_refs.push(node_idx);
    }

    // Stride scalar args: constant buffer (addrspace 2), read-only, i32.
    let stride_base_slot = ptr_params.len();
    for (i, _) in dyn_stride_args.iter().enumerate() {
        let param_slot = stride_base_slot + i;
        air_args.push(crate::bitcode::metadata::KernelArg::ScalarBuffer {
            index: param_slot as u32,
            byte_size: 4,
            type_name: "uint".to_string(),
            name: format!("stride{i}"),
        });
        let node_idx = meta_nodes.len();
        meta_nodes.push(format!(
            "!{{i32 {param_slot}, !\"air.buffer\", !\"air.buffer_size\", i32 4, \
             !\"air.location_index\", i32 {param_slot}, i32 1, \
             !\"air.read\", !\"air.address_space\", i32 2, \
             !\"air.arg_type_size\", i32 4, \
             !\"air.arg_type_align_size\", i32 4, \
             !\"air.arg_type_name\", !\"uint\", \
             !\"air.arg_name\", !\"stride_arg_{i}\"}}"
        ));
        param_meta_refs.push(node_idx);
    }

    // Extent scalar args, same shape as the stride ones, numbered after them.
    let dim_base_slot = stride_base_slot + dyn_stride_args.len();
    for (i, _) in dyn_shape_args.iter().enumerate() {
        let param_slot = dim_base_slot + i;
        air_args.push(crate::bitcode::metadata::KernelArg::ScalarBuffer {
            index: param_slot as u32,
            byte_size: 4,
            type_name: "uint".to_string(),
            name: format!("dim{i}"),
        });
        let node_idx = meta_nodes.len();
        meta_nodes.push(format!(
            "!{{i32 {param_slot}, !\"air.buffer\", !\"air.buffer_size\", i32 4, \
             !\"air.location_index\", i32 {param_slot}, i32 1, \
             !\"air.read\", !\"air.address_space\", i32 2, \
             !\"air.arg_type_size\", i32 4, \
             !\"air.arg_type_align_size\", i32 4, \
             !\"air.arg_type_name\", !\"uint\", \
             !\"air.arg_name\", !\"dim_arg_{i}\"}}"
        ));
        param_meta_refs.push(node_idx);
    }

    for (kind, name) in [
        ("air.thread_position_in_threadgroup", "_tp"),
        ("air.threadgroup_position_in_grid", "_tg"),
    ] {
        air_args.push(crate::bitcode::metadata::KernelArg::Builtin {
            kind: kind.to_string(),
            type_name: "uint3".to_string(),
            name: name.to_string(),
        });
    }

    let tp_meta = meta_nodes.len();
    meta_nodes.push(format!(
        "!{{i32 {tp_idx}, !\"air.thread_position_in_threadgroup\", \
         !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"_tp\"}}"
    ));
    param_meta_refs.push(tp_meta);

    let tg_meta = meta_nodes.len();
    meta_nodes.push(format!(
        "!{{i32 {tg_idx}, !\"air.threadgroup_position_in_grid\", \
         !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"_tg\"}}"
    ));
    param_meta_refs.push(tg_meta);

    let fn_type_str = format!("void ({})*", sig_types.join(", "));
    let empty_idx = meta_nodes.len();
    meta_nodes.push("!{}".into());

    let param_list_idx = meta_nodes.len();
    let refs: Vec<String> = param_meta_refs.iter().map(|i| format!("!{i}")).collect();
    meta_nodes.push(format!("!{{{}}}", refs.join(", ")));

    let kernel_node_idx = meta_nodes.len();
    meta_nodes.push(format!(
        "!{{{fn_type_str} @{kernel_name}, !{empty_idx}, !{param_list_idx}}}"
    ));

    // The same kernel, encoded as bitcode and wrapped in a container - no assembler.
    // Built here rather than where the CFG is lowered because it needs the finished
    // argument list: without those descriptors Metal cannot bind the kernel's buffers
    // and its compiler service dies rather than reporting an error.
    if metallib_sink.is_some() || bitcode_dump_enabled() {
        let lib = if air_dialect {
            #[cfg(feature = "pliron-air")]
            {
                crate::air_dialect::to_bitcode::metallib_from_air(
                    kernel_name,
                    &lowered.func,
                    &air_args,
                    &lowered.tg_globals,
                    profile,
                )
            }
            #[cfg(not(feature = "pliron-air"))]
            unreachable!()
        } else {
            to_bitcode::metallib_from_cfg(
                kernel_name,
                &lowered.func,
                &air_args,
                &lowered.tg_globals,
                profile,
            )
        }
        .map_err(|e| {
            LowerError::Unimplemented(format!(
                "kernel @{kernel_name}: the bitcode writer cannot encode this yet: {e}"
            ))
        });
        if bitcode_dump_enabled() {
            dump_bitcode_artifacts(kernel_name, &lowered, &air_args, &lib, profile);
        }
        if let Some(sink) = metallib_sink {
            sink.push((kernel_name.to_string(), lib?));
        }
    }

    Ok(kernel_node_idx)
}

/// `METAL_TILE_BITCODE_EMIT` writes each kernel's self-built bitcode and container to
/// the temp dir as `mtb_<kernel>.bc` / `mtb_<kernel>.metallib`, and reports what the
/// writer could not encode. Diagnostic only: the text path still produces what is
/// actually dispatched, so the two can be compared kernel by kernel.
fn bitcode_dump_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("METAL_TILE_BITCODE_EMIT").is_some())
}

fn dump_bitcode_artifacts(
    kernel_name: &str,
    lowered: &walk::Lowered,
    air_args: &[crate::bitcode::metadata::KernelArg],
    lib: &Result<Vec<u8>, LowerError>,
    profile: &TargetProfile,
) {
    let dir = std::env::temp_dir();
    match lib {
        Ok(bytes) => {
            let path = dir.join(format!("mtb_{kernel_name}.metallib"));
            let _ = std::fs::write(&path, bytes);
            if let Ok(bc) = to_bitcode::bitcode_from_cfg(
                kernel_name,
                &lowered.func,
                air_args,
                &lowered.tg_globals,
                profile,
            ) {
                let _ = std::fs::write(dir.join(format!("mtb_{kernel_name}.bc")), &bc);
            }
            eprintln!(
                "[bitcode-emit] OK   {kernel_name}  {} bytes -> {}",
                bytes.len(),
                path.display()
            );
        }
        Err(e) => eprintln!("[bitcode-emit] MISS {kernel_name}  {e}"),
    }
}

// Constant collection

/// Extract the scalar element type from any Type - tile, scalar, or view.
/// Anything carrying tile attributes. Only the materialised `OpData` implements it now,
/// but the readers stay generic: they care about the attribute list, not who owns it.
pub(crate) trait HasAttrs {
    fn attrs(&self) -> &[(String, Attribute)];
    fn res_types(&self) -> &[Type];
}

impl HasAttrs for ir::OpData {
    fn attrs(&self) -> &[(String, Attribute)] {
        &self.attributes
    }
    fn res_types(&self) -> &[Type] {
        &self.result_types
    }
}

fn scalar_of(ty: &Type) -> ScalarType {
    match ty {
        Type::Scalar(s) => s.clone(),
        Type::Tile(TileType {
            element_type: TileElementType::Scalar(s),
            ..
        }) => s.clone(),
        Type::TensorView(tv) => tv.element_type.clone(),
        Type::PartitionView(pv) => pv.tensor_view.element_type.clone(),
        _ => ScalarType::F32,
    }
}

/// [`lower_to_metallib_for`], with the body emitted from the `air` pliron dialect instead
/// of straight from the CFG.
///
/// The `tile` half of this pair is gone: the frontend emits the `wax` dialect, so there is
/// no second tile source left to compare against. Only the emitter is still a choice.
#[cfg(feature = "pliron-air")]
pub fn lower_to_metallib_via_air(
    module: &TileModule,
    profile: &TargetProfile,
) -> Result<Vec<u8>, LowerError> {
    let irs: Vec<WaxIr> = module
        .functions
        .iter()
        .map(|&f| ir::from_source(module, f))
        .collect();
    let (_, mut libs) = lower_tile_irs(&irs, true, profile, true)?;
    if libs.len() != 1 {
        return Err(LowerError::MalformedIr(format!(
            "lower_to_metallib_via_air: expected exactly one entry function, found {}",
            libs.len()
        )));
    }
    Ok(libs.remove(0).1)
}

fn attr_i32(op: &impl HasAttrs, key: &str) -> Option<i64> {
    op.attrs()
        .iter()
        .find(|(k, _)| k == key)
        .and_then(|(_, v)| {
            if let Attribute::Integer(i, _) = v {
                Some(*i)
            } else {
                None
            }
        })
}

fn constant_literal(op: &impl HasAttrs) -> String {
    let de = op
        .attrs()
        .iter()
        .find(|(k, _)| k == "value")
        .and_then(|(_, v)| {
            if let Attribute::DenseElements(de) = v {
                Some(de)
            } else {
                None
            }
        });
    let ty = op.res_types().first();
    match (de, ty) {
        (Some(de), Some(ty)) => dense_elements_literal(de, ty),
        (None, Some(Type::Scalar(ScalarType::I32))) => "0".into(),
        _ => "undef".into(),
    }
}
/// The f32 a 2-byte or 4-byte half constant denotes.
///
/// SHARED by the text and value forms deliberately. Two copies of this widening can drift,
/// and a reconstruction of this file once reintroduced exactly that duplication.
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

fn dense_elements_literal(de: &DenseElements, ty: &Type) -> String {
    // Single 8x8 SGMMA tile -> <64 x float> constant.
    // Multi-SGMMA tiles (nr*nc > 1) are handled lazily in emit_for, not here.
    if matches!(sgmma_dims(ty), Some((1, 1))) {
        if de.data.is_empty() || de.data.iter().all(|&b| b == 0) {
            return "zeroinitializer".into();
        }
        let mut elems = Vec::with_capacity(64);
        for i in 0..64usize {
            let off = i * 4;
            let v = if off + 4 <= de.data.len() {
                f32::from_le_bytes(de.data[off..off + 4].try_into().unwrap())
            } else {
                0.0f32
            };
            elems.push(format!("float {}", float_hex(v)));
        }
        return format!("<{}>", elems.join(", "));
    }

    let scalar = match ty {
        Type::Scalar(s) => s.clone(),
        Type::Tile(TileType {
            element_type: TileElementType::Scalar(s),
            ..
        }) => s.clone(),
        _ => return "undef".into(),
    };
    let scalar_is_float = is_float(&scalar);
    if de.data.is_empty() {
        return if scalar_is_float { "0.0" } else { "0" }.into();
    }
    match scalar {
        ScalarType::F32 | ScalarType::TF32 if de.data.len() >= 4 => {
            let v = f32::from_le_bytes(de.data[..4].try_into().unwrap());
            float_hex(v)
        }
        ScalarType::F64 if de.data.len() >= 8 => {
            let v = f64::from_le_bytes(de.data[..8].try_into().unwrap());
            format!("0x{:016X}", v.to_bits())
        }
        // Data may be 2 bytes (a native half pattern) or 4 (an f32 source value).
        ScalarType::F16 => half_hex(half_source_f32(&de.data, true)),
        ScalarType::BF16 => bfloat_hex(half_source_f32(&de.data, false)),
        ScalarType::I32 if de.data.len() >= 4 => {
            let v = i32::from_le_bytes(de.data[..4].try_into().unwrap());
            format!("{v}")
        }
        ScalarType::I64 if de.data.len() >= 8 => {
            let v = i64::from_le_bytes(de.data[..8].try_into().unwrap());
            format!("{v}")
        }
        ScalarType::I16 if de.data.len() >= 2 => {
            let v = i16::from_le_bytes(de.data[..2].try_into().unwrap());
            format!("{v}")
        }
        ScalarType::I8 if !de.data.is_empty() => format!("{}", de.data[0] as i8),
        _ if scalar_is_float => "0.0".into(),
        _ => "0".into(),
    }
}
/// Detect the 2-pass load pattern: two For ops in `block` (the first appearing
/// before the second) that both load from the same PartitionView and share the
/// same loop upper bound constant value.
///
/// The loops need not be adjacent — other ops may appear between them.
///
/// Returns: (first_for_opid, second_for_opid, partition_view_value, num_chunks)
/// The two-pass plans a pass already decided on, read back off the IR.
///
/// The DECISION - which loop pairs qualify, and whether the span fits in registers - is
/// `passes::cross_loop_reuse`'s, made on the dialect. All that is left here is looking up
/// WHICH view, which cannot travel in an attribute because attributes cannot name SSA
/// values.
/// Thread-private register budget for two-pass staging, in bytes per thread.
///
/// Apple's number, and the reason it lives here rather than in the IR: the IR pass records
/// how many bytes a staging would hold live, and each backend decides whether its register
/// file can absorb them.
pub(crate) const PRIV_BUDGET_BYTES: i64 = 128;

pub(crate) fn two_pass_plans(ir: &WaxIr, block: ir::BlockRef) -> Vec<TwoPassPlan> {
    use wax_ir::passes::cross_loop_reuse::{
        ATTR_CACHE_BYTES, ATTR_CACHE_CONSUMER, ATTR_CACHE_ID, ATTR_CACHE_PRODUCER,
    };

    let fors: Vec<OpRef> = ir
        .block(block)
        .ops
        .iter()
        .copied()
        .filter(|&o| ir.op(o).opcode == Opcode::For)
        .collect();

    let mut plans = Vec::new();
    for (idx, &staging) in fors.iter().enumerate() {
        let Some(num_chunks) = attr_i32(ir.op(staging), ATTR_CACHE_PRODUCER) else {
            continue;
        };
        // The pass computes the COST; the budget is ours, because it is a property of this
        // target's register file. Staging more than this is a pessimisation rather than a
        // smaller win: the second pass is already served from L2 (the data was just read in
        // pass one), so the extra stores in pass one cost more than they save. MEASURED at
        // 32 f32 chunks; do not raise without re-measuring.
        if attr_i32(ir.op(staging), ATTR_CACHE_BYTES).is_none_or(|b| b > PRIV_BUDGET_BYTES) {
            continue;
        }
        // The partner is the one the PASS paired this loop with, named by a shared id. Taking
        // "the next loop marked as consuming" instead would be a different question, and with
        // three or more loops it can answer differently than the pass did.
        let Some(id) = attr_i32(ir.op(staging), ATTR_CACHE_ID) else {
            continue;
        };
        let Some(&consuming) = fors[idx + 1..].iter().find(|&&o| {
            attr_i32(ir.op(o), ATTR_CACHE_CONSUMER).is_some()
                && attr_i32(ir.op(o), ATTR_CACHE_ID) == Some(id)
        }) else {
            continue;
        };
        let read_later = load_view_tko_partition_views(ir, ir.op(consuming));
        if let Some(view) = load_view_tko_partition_views(ir, ir.op(staging))
            .into_iter()
            .find(|v| read_later.contains(v))
        {
            plans.push(TwoPassPlan {
                staging,
                consuming,
                view,
                num_chunks: num_chunks as i32,
            });
        }
    }
    plans
}

/// One staging/consuming loop pair, as the lowering needs it.
#[derive(Clone, Copy)]
pub(crate) struct TwoPassPlan {
    pub staging: OpRef,
    pub consuming: OpRef,
    pub view: ValRef,
    pub num_chunks: i32,
}

/// Pre-scan the entry block for MakeTensorView ops that carry dynamic stride operands.
/// Returns the block arg Values that carry runtime strides, ordered by arg_index.
/// Entry block args carrying a tensor's runtime EXTENT, for the views that need one.
///
/// The mirror of `find_dyn_stride_block_args` on the other operand range of
/// `MakeTensorView`: dynamic dims come first, dynamic strides after. Strides have always
/// been promoted to kernel parameters; dims never were, so a bounds check against one
/// resolved to the constant 0 that `setup_params` gives an unpromoted scalar - which
/// silently masks away every access.
///
/// Promotion is GATED, and deliberately so: lifting every dynamic dim would add scalar
/// arguments to every kernel with a mutable tensor and break every existing dispatch. A
/// dim is promoted only where something consumes it - a partition view that is STORED
/// through and DECLARES padding. `partition_full_mut` declares `padding::None`, so a
/// kernel opts in by asking for it, and one that does not keeps the signature it had.
/// Tensor views behind a padded FRAGMENT LOAD - an A or B operand of an `mmaf`.
///
/// A ragged GEMM whose M is a RUNTIME value (candle's prefill: M is the prompt length, and
/// a const generic would monomorphise per prompt and pay the DSL compile on every new one)
/// cannot bound its A tile from a declared shape. Without a bound the last block reads up
/// to BM-1 rows past the tensor, which is why that kernel clamped its tile origin instead -
/// the whole reason `tensor_offset_ptr` existed.
fn collect_padded_mma_operands(ir: &WaxIr, want: &mut HashSet<ValRef>) {
    for op in ir.ops.iter() {
        if op.opcode != Opcode::MmaF {
            continue;
        }
        // Operands 0 and 1 only: the accumulator is not loaded from memory.
        for &v in op.operands.iter().take(2) {
            let Some(ld) = ir.defining_op(v) else {
                continue;
            };
            let ld = ir.op(ld);
            if ld.opcode != Opcode::LoadViewTko {
                continue;
            }
            let Some(&pv) = ld.operands.first() else {
                continue;
            };
            if !matches!(
                ir.value_type(pv),
                Type::PartitionView(p) if p.padding_value.is_some()
            ) {
                continue;
            }
            if let Some(def) = ir.defining_op(pv)
                && ir.op(def).opcode == Opcode::MakePartitionView
                && let Some(&tv) = ir.op(def).operands.first()
            {
                want.insert(tv);
            }
        }
    }
}

fn find_dyn_shape_block_args(ir: &WaxIr, block_id: ir::BlockRef) -> Vec<ValRef> {
    // Tensor views behind a padded store destination.
    let mut want: HashSet<ValRef> = HashSet::new();
    for op in ir.ops.iter() {
        if op.opcode != Opcode::StoreViewTko {
            continue;
        }
        let Some(&pv) = op.operands.get(1) else {
            continue;
        };
        let padded = matches!(
            ir.value_type(pv),
            Type::PartitionView(p) if p.padding_value.is_some()
        );
        if !padded {
            continue;
        }
        let Some(def) = ir.defining_op(pv) else {
            continue;
        };
        let pv_op = ir.op(def);
        if pv_op.opcode == Opcode::MakePartitionView
            && let Some(&tv) = pv_op.operands.first()
        {
            want.insert(tv);
        }
    }
    // GATED ON THE KERNEL ALREADY MASKING ITS OUTPUT STORE. Extending bounds to every
    // padded load would be far too implicit: `Tensor::partition` declares `padding::Zero`
    // BY DEFAULT, so nearly every view is "padded", and promoting on that alone silently
    // adds parameters to any kernel that reads a runtime-shaped tensor - an ABI change that
    // shows up only as a wrong answer at dispatch. Two private copies of `gemm` returned
    // zeros before this gate was narrowed.
    //
    // A kernel that masks its STORE has already opted into the padded regime and pays the
    // extents at dispatch, so extending them to its loads costs it nothing new.
    if !want.is_empty() {
        collect_padded_mma_operands(ir, &mut want);
    }

    if want.is_empty() {
        return Vec::new();
    }

    let block = ir.block(block_id);
    let mut seen: HashSet<ValRef> = HashSet::new();
    let mut ordered: Vec<ValRef> = Vec::new();
    for &op_id in &block.ops {
        let op = ir.op(op_id);
        if op.opcode != Opcode::MakeTensorView {
            continue;
        }
        let produces_wanted = op.results.iter().flatten().any(|r| want.contains(r));
        if !produces_wanted {
            continue;
        }
        let (n_base, n_dyn_shape, _) = parse_operand_seg_sizes(op);
        if n_dyn_shape == 0 {
            continue;
        }
        for &sv in &op.operands[n_base..(n_base + n_dyn_shape).min(op.operands.len())] {
            let ba = trace_to_block_arg(ir, sv);
            if seen.insert(ba) {
                ordered.push(ba);
            }
        }
    }
    ordered.sort_by_key(|&v| match ir.producer(v) {
        Some(ir::Producer::BlockArg { arg_index, .. }) => arg_index,
        _ => u32::MAX,
    });
    ordered
}

fn find_dyn_stride_block_args(ir: &WaxIr, block_id: ir::BlockRef) -> Vec<ValRef> {
    let block = ir.block(block_id);
    let mut seen: HashSet<ValRef> = HashSet::new();
    let mut ordered: Vec<ValRef> = Vec::new();

    for &op_id in &block.ops {
        let op = ir.op(op_id);
        if op.opcode != Opcode::MakeTensorView {
            continue;
        }
        let (n_base, n_dyn_shape, n_dyn_stride) = parse_operand_seg_sizes(op);
        if n_dyn_stride == 0 {
            continue;
        }
        let start = n_base + n_dyn_shape;
        for &sv in &op.operands[start..(start + n_dyn_stride).min(op.operands.len())] {
            let ba = trace_to_block_arg(ir, sv);
            if seen.insert(ba) {
                ordered.push(ba);
            }
        }
    }

    // Sort by the block-arg's position in the parameter list.
    ordered.sort_by_key(|&v| match ir.producer(v) {
        Some(ir::Producer::BlockArg { arg_index, .. }) => arg_index,
        _ => u32::MAX,
    });
    ordered
}
/// Parse `operandSegmentSizes` from a MakeTensorView op: returns (n_base, n_dyn_shape, n_dyn_stride).
fn parse_operand_seg_sizes(op: &impl HasAttrs) -> (usize, usize, usize) {
    for (name, attr) in op.attrs() {
        if name == "operandSegmentSizes" {
            match attr {
                Attribute::DenseI32Array(arr) => {
                    let get = |i: usize| arr.get(i).copied().unwrap_or(0) as usize;
                    return (get(0), get(1), get(2));
                }
                Attribute::Array(arr) => {
                    let get = |i: usize| match arr.get(i) {
                        Some(Attribute::Integer(v, _)) => *v as usize,
                        _ => 0,
                    };
                    return (get(0), get(1), get(2));
                }
                _ => {}
            }
        }
    }
    (1, 0, 0)
}
fn ptr_elem_type_and_as(ty: &Type) -> (String, u32) {
    match ty {
        Type::Pointer(p) => (ll_scalar(&p.pointee).to_string(), 1),
        Type::TensorView(tv) => (ll_scalar(&tv.element_type).to_string(), 1),
        Type::PartitionView(pv) => (ll_scalar(&pv.tensor_view.element_type).to_string(), 1),
        _ => ("float".to_string(), 1),
    }
}

/// Collect all PartitionView values loaded by LoadViewTko ops at the top level of a For body.
fn load_view_tko_partition_views(ir: &WaxIr, for_op: &ir::OpData) -> Vec<ValRef> {
    let mut views = Vec::new();
    if for_op.regions.is_empty() {
        return views;
    }
    let body_region = ir.region(for_op.regions[0]);
    for &block_id in &body_region.blocks {
        let block = ir.block(block_id);
        for &op_id in &block.ops {
            let op = ir.op(op_id);
            if op.opcode == Opcode::LoadViewTko && !op.operands.is_empty() {
                let pv = op.operands[0];
                let pv_ty = ir.value_type(pv);
                if matches!(pv_ty, Type::PartitionView(_)) {
                    // Skip SGMMA tiles (8×8 matrix tiles) — only promote scalar per-thread loads.
                    let is_sgmma = op.res_types().first().and_then(sgmma_dims).is_some();
                    if !is_sgmma {
                        views.push(pv);
                    }
                }
            }
        }
    }
    views
}
/// Trace a value back through Assume (AssumeBounded) ops to the block arg underneath.
pub(crate) fn trace_to_block_arg(ir: &WaxIr, v: ValRef) -> ValRef {
    match ir.defining_op(v) {
        Some(op) => {
            let op_data = ir.op(op);
            if op_data.opcode == Opcode::Assume
                && let Some(&src) = op_data.operands.first()
            {
                return src;
            }
            v
        }
        None => v,
    }
}

/// The structured form of [`constant_literal`].
///
/// The text emitter reads the literal and the bitcode emitter reads this, so the two must
/// agree on every constant - `const_value_tests` pins exactly that.
pub(crate) fn constant_value(op: &impl HasAttrs) -> crate::lower::cfg::ImmVal {
    let de = op
        .attrs()
        .iter()
        .find(|(k, _)| k == "value")
        .and_then(|(_, v)| {
            if let Attribute::DenseElements(de) = v {
                Some(de)
            } else {
                None
            }
        });
    match (de, op.res_types().first()) {
        (Some(de), Some(ty)) => dense_elements_value(de, ty),
        _ => crate::lower::cfg::ImmVal::Undef,
    }
}

/// The structured value matching [`dense_elements_literal`]'s text.
///
/// Deliberately a separate pass rather than threaded through that function's many
/// return points: it shares the byte-decoding helpers below, and anything it does not
/// recognise yields `ImmVal::Text`, which the bitcode emitter reports instead of
/// silently encoding the wrong constant.
fn dense_elements_value(de: &DenseElements, ty: &Type) -> crate::lower::cfg::ImmVal {
    use crate::lower::cfg::ImmVal;
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

#[cfg(test)]
mod const_value_tests {
    use super::*;
    use crate::lower::cfg::ImmVal;

    fn de(bytes: &[u8]) -> DenseElements {
        DenseElements {
            element_type: Type::Scalar(ScalarType::I8),
            shape: vec![bytes.len() as i64],
            data: bytes.to_vec(),
        }
    }

    /// `dense_elements_value` must agree with `dense_elements_literal`: the text
    /// emitter reads one and the bitcode emitter the other, so a disagreement means
    /// the two backends compute different things from the same DSL constant.
    #[test]
    fn structured_value_agrees_with_text() {
        let f32t = Type::Scalar(ScalarType::F32);
        let one = 1.0f32;
        assert_eq!(
            dense_elements_value(&de(&one.to_le_bytes()), &f32t),
            ImmVal::FloatBits(one.to_bits() as u64)
        );
        assert_eq!(
            dense_elements_literal(&de(&one.to_le_bytes()), &f32t),
            float_hex(one)
        );

        let i32t = Type::Scalar(ScalarType::I32);
        assert_eq!(
            dense_elements_value(&de(&(-7i32).to_le_bytes()), &i32t),
            ImmVal::Int(-7)
        );
        assert_eq!(
            dense_elements_literal(&de(&(-7i32).to_le_bytes()), &i32t),
            "-7"
        );

        // Empty data is zero in both forms.
        assert_eq!(dense_elements_value(&de(&[]), &i32t), ImmVal::Int(0));
        assert_eq!(dense_elements_literal(&de(&[]), &i32t), "0");
        assert_eq!(dense_elements_value(&de(&[]), &f32t), ImmVal::FloatBits(0));
        assert_eq!(dense_elements_literal(&de(&[]), &f32t), "0.0");
    }

    /// The arms the reconstruction of `dense_elements_value` was NOT pinned by.
    ///
    /// After that function had to be rebuilt from its text-form twin, only f32/i32/f16/bf16
    /// were covered. These are the rest - so a future rebuild has the whole shape pinned,
    /// not most of it.
    #[test]
    fn the_remaining_arms_agree_too() {
        // 8x8 f32 tile: the SGMMA fragment arm, which returns a vector rather than a scalar.
        let sgmma = Type::Tile(TileType {
            shape: vec![8, 8],
            element_type: TileElementType::Scalar(ScalarType::F32),
        });
        assert_eq!(dense_elements_value(&de(&[]), &sgmma), ImmVal::Zero);
        assert_eq!(dense_elements_literal(&de(&[]), &sgmma), "zeroinitializer");
        assert_eq!(
            dense_elements_value(&de(&[0u8; 256]), &sgmma),
            ImmVal::Zero,
            "an all-zero fragment is zeroinitializer, not 64 zeros"
        );

        let mut bytes = vec![0u8; 256];
        bytes[..4].copy_from_slice(&2.5f32.to_le_bytes());
        let ImmVal::VecFloatBits(v) = dense_elements_value(&de(&bytes), &sgmma) else {
            panic!("a non-zero fragment must carry 64 floats");
        };
        assert_eq!(v.len(), 64);
        assert_eq!(v[0], 2.5f32.to_bits() as u64);
        assert!(v[1..].iter().all(|&b| b == 0));

        // Widths the text form spells differently.
        let f64t = Type::Scalar(ScalarType::F64);
        let x = 1.0f64 / 3.0;
        assert_eq!(
            dense_elements_value(&de(&x.to_le_bytes()), &f64t),
            ImmVal::FloatBits(x.to_bits())
        );
        assert_eq!(
            dense_elements_literal(&de(&x.to_le_bytes()), &f64t),
            format!("0x{:016X}", x.to_bits())
        );

        for (ty, bytes, want) in [
            (ScalarType::I64, (-9i64).to_le_bytes().to_vec(), -9i64),
            (ScalarType::I16, (-3i16).to_le_bytes().to_vec(), -3i64),
            (ScalarType::I8, (-1i8).to_le_bytes().to_vec(), -1i64),
        ] {
            let t = Type::Scalar(ty.clone());
            assert_eq!(
                dense_elements_value(&de(&bytes), &t),
                ImmVal::Int(want),
                "{ty:?} value form"
            );
            assert_eq!(
                dense_elements_literal(&de(&bytes), &t),
                format!("{want}"),
                "{ty:?} text form"
            );
        }
    }

    /// f16/bf16 constants - native 2-byte patterns and values widened from an f32
    /// source - must carry the same bits the text form prints, since a mismatch here
    /// would put a different constant in the bitcode than in the AIR text.
    #[test]
    fn half_constants_match_their_text_form() {
        let f16t = Type::Scalar(ScalarType::F16);
        let bf16t = Type::Scalar(ScalarType::BF16);
        for bytes in [
            1.5f32.to_le_bytes().to_vec(),
            (-0.125f32).to_le_bytes().to_vec(),
            0.1f32.to_le_bytes().to_vec(), // not representable: exercises rounding
            f32::INFINITY.to_le_bytes().to_vec(),
            0x3C00u16.to_le_bytes().to_vec(), // native half pattern
        ] {
            for ty in [&f16t, &bf16t] {
                let text = dense_elements_literal(&de(&bytes), ty);
                let ImmVal::FloatBits(bits) = dense_elements_value(&de(&bytes), ty) else {
                    panic!("{ty:?} constant did not carry structured bits: {text}");
                };
                let from_text = u64::from_str_radix(&text[3..], 16).expect("0xH/0xR literal");
                assert_eq!(bits, from_text, "{ty:?}: text {text} vs value {bits:#x}");
            }
        }
        assert_eq!(
            dense_elements_value(&de(&0x3C00u16.to_le_bytes()), &f16t),
            ImmVal::FloatBits(0x3C00)
        );
    }
}
