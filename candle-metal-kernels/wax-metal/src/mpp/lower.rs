//! MPP kernel lowering.

use super::abi::mpp_abi;
use super::air::mpp_air;
use super::config::MppGemm;
use crate::air_dialect::to_bitcode::{AirFunc, body_from_air};
use crate::lower::to_bitcode::{BitcodeModuleAssembler, Callee, Unsupported};
use crate::profile::TargetProfile;
use pliron::builtin::ops::FuncOp;
use pliron::context::Context;

/// Lower MPP GEMM as an air kernel.
pub fn lower_mpp_gemm_air(
    cfg: &MppGemm,
    profile: &TargetProfile,
) -> Result<crate::LoweredKernel, crate::LowerError> {
    let abi = mpp_abi(cfg, profile);
    let mut ctx = Context::new();
    let func = mpp_air(cfg, &mut ctx);
    let air = air_func(func, &ctx);
    let mut bm = BitcodeModuleAssembler::new(profile);
    for c in abi.callees.iter().cloned() {
        bm.declare_callee(c);
    }
    let body = body_from_air(
        &ctx,
        &air,
        crate::mpp::air::ENTRY_PARAMS,
        &|n| bm.callee_idx(n),
        &|_| None,
    )
    .map_err(unsupported)?;
    bm.add_kernel_pinning(
        cfg.entry.as_str(),
        abi.entry_params.clone(),
        &abi.args,
        body,
        abi.pinned_types.clone(),
    );
    let bc = bm.finish().map_err(unsupported)?;
    Ok(lowered(cfg, &bc, &[&cfg.entry], None, profile))
}

/// Lower MPP Split-K GEMM as an air kernel.
pub fn lower_mpp_gemm_splitk(
    cfg: &MppGemm,
    reduce_module: &mut wax_ir::Module,
    reduce_func: wax_ir::OpId,
    reduce_entry: &str,
    reduce_launch: crate::Launch,
    profile: &TargetProfile,
) -> Result<crate::LoweredKernel, crate::LowerError> {
    let mut abi = mpp_abi(cfg, profile);
    let air_p = {
        let ctx = &mut reduce_module.ctx;
        let func = mpp_air(cfg, ctx);
        air_func(func, ctx)
    };
    // Read kernel arguments metadata off original function signature before in-place conversion.
    let args_r =
        crate::lower::kernel_args_for_module(reduce_module, reduce_func).ok_or_else(|| {
            crate::LowerError::MalformedIr("split-k reduce has no entry block".into())
        })?;
    let ck = crate::wax_to_air::convert_kernel(reduce_module, reduce_func)
        .map_err(crate::LowerError::Unimplemented)?;
    let (air_r, param_r, callees_r, globals_r) =
        (ck.air, ck.param_types, ck.callees, ck.tg_globals);

    // Merge function prototypes from both gemm and reduce into one list.
    let mut union: Vec<Callee> = std::mem::take(&mut abi.callees);
    for (n, r, p) in &callees_r {
        if !union.iter().any(|c| &c.name == n) {
            union.push(Callee::new(n, r.clone(), p.clone()));
        }
    }
    let mut bm = BitcodeModuleAssembler::new(profile);
    for c in union {
        bm.declare_callee(c);
    }
    for (name, ty, count) in globals_r {
        bm.declare_global(name, ty, count);
    }
    let body_p = body_from_air(
        &reduce_module.ctx,
        &air_p,
        crate::mpp::air::ENTRY_PARAMS,
        &|n| bm.callee_idx(n),
        &|n| bm.global_idx(n),
    )
    .map_err(unsupported)?;
    let body_r = body_from_air(
        &reduce_module.ctx,
        &air_r,
        param_r.len(),
        &|n| bm.callee_idx(n),
        &|n| bm.global_idx(n),
    )
    .map_err(unsupported)?;

    bm.add_kernel_pinning(
        cfg.entry.as_str(),
        abi.entry_params.clone(),
        &abi.args,
        body_p,
        abi.pinned_types.clone(),
    );
    bm.add_kernel(reduce_entry, param_r.clone(), &args_r, body_r);
    let bc = bm.finish().map_err(unsupported)?;

    Ok(lowered(
        cfg,
        &bc,
        &[&cfg.entry, reduce_entry],
        Some(crate::launch::SecondaryDispatch {
            entry: reduce_entry.to_string(),
            launch: reduce_launch,
            scalar_args: Vec::new(),
        }),
        profile,
    ))
}

fn unsupported(e: Unsupported) -> crate::LowerError {
    crate::LowerError::Unimplemented(e.to_string())
}

fn lowered(
    cfg: &MppGemm,
    bc: &[u8],
    entries: &[&str],
    second: Option<crate::launch::SecondaryDispatch>,
    profile: &TargetProfile,
) -> crate::LoweredKernel {
    let fns: Vec<_> = entries
        .iter()
        .map(|e| crate::metallib::Function::kernel(*e))
        .collect();
    crate::LoweredKernel {
        metallib: crate::metallib::wrap(bc, &fns, profile),
        entry: cfg.entry.clone(),
        launch: cfg.launch(),
        scratch_bytes: cfg.partial_elems() * 4,
        scalar_args: Vec::new(),
        second,
    }
}

fn air_func(func: FuncOp, ctx: &Context) -> AirFunc {
    AirFunc {
        blocks: vec![func.get_entry_block(ctx)],
        func,
    }
}
