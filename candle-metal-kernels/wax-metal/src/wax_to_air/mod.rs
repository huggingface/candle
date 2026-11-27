//! `wax` to `air` as a pliron [`DialectConversion`].

// TODO: A lot of what is in `wax_to_air` needs to be refactored. For example most concepts that are
// collections of `Value` or keyed by `Value` (like `HashSet<Value>` or `HashMap<Value, _>`),
// which is a lot, should actually be pliron `Analysis`es and/or `Pass`es that store their results
// in the IR. See `run_conversion`.

use crate::air_dialect::ops as air;
use crate::air_dialect::to_bitcode::{AirFunc, body_from_air};
use crate::air_dialect::type_interfaces::bitcode_ty;
use crate::air_dialect::types as t;
use crate::bitcode::Ty;
use crate::lower::to_bitcode::{BitcodeModuleAssembler, Callee};
use crate::wax_to_air::wax_operand_ty;
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{
    DialectConversion, DialectConversionRewriter, OperandsInfo,
};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::TypeHandle;
use pliron::value::{DefiningEntity, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use wax_ir::Opcode;
use wax_ir::dialect::source::opcode_of;
use wax_ir::dialect::types::from_pliron as wax_type_of;
use wax_ir::types::{ScalarType, TileElementType, Type as WaxType};

mod access;
mod broadcast;
mod cache;
mod control;
mod elementwise;
mod fragments;
mod helpers;
pub(crate) mod keys;
mod lanes;
mod prepare;
mod reduce;

pub(crate) use access::*;
pub(crate) use broadcast::*;
pub(crate) use cache::*;
pub(crate) use control::*;
pub(crate) use elementwise::*;
pub(crate) use fragments::*;
pub use helpers::air_type_for;
pub(crate) use helpers::*;
pub(crate) use lanes::*;
pub use prepare::entry_block;
pub(crate) use prepare::*;
pub(crate) use reduce::*;

#[derive(Default)]
pub(crate) struct Addr {
    /// Access bounds. See `collect_access_bounds`.
    bounds: HashMap<Ptr<Operation>, Vec<(usize, Bound)>>,
    /// A fragment access's dynamic stride values.
    frag_strides: HashMap<Ptr<Operation>, Vec<Option<Value>>>,
}

/// The fragment classification and the fragment grids it registers.
#[derive(Default)]
pub(crate) struct Frags {
    /// Fragment grids, keyed by the value that stands for them. See `grid_of`.
    grids: HashMap<Value, Vec<Value>>,
    /// Values that must live in matrix-fragment registers.
    fragments: HashSet<Value>,
}

/// The reduce/scan/threadgroup-memory state.
#[derive(Default)]
pub(crate) struct ReduceMem {
    /// Values that differ across a SIMD group's lanes.
    divergent: HashSet<Value>,
    /// Column reduce op -> pre-conversion source.
    reduce_sources: HashMap<Ptr<Operation>, Value>,
    /// Threadgroup globals.
    pub tg_globals: Vec<(String, Ty, u32)>,
    /// Threadgroup scratch element count.
    pub tg_scratch_floats: u32,
    /// Batches of `[32, 32]` column reduces.
    batches: Vec<ReduceBatch>,
    of_member: HashMap<Ptr<Operation>, (BatchId, Slot)>,
    /// Batch result keyed by op.
    batch_results: HashMap<Ptr<Operation>, Value>,
    /// Results that are logically `[rows]` but materialised `[rows, 32]`. A dim-1 reduce
    /// leaves its total in all 32 lanes of the group that held the row, so broadcasting back
    /// is a forward, not a reshape.
    pub row_replicated: HashSet<Value>,
}

/// Loop and branch state.
/// Position parameters, built loops, back edges and deferred body cleanups.
#[derive(Default)]
pub(crate) struct Control {
    /// `%_tp`, the thread position.
    pub tp: Option<Value>,
    /// `%_tg`, the threadgroup position.
    pub tg: Option<Value>,
    /// One thread per element from largest tile conversion decomposed over `tp`.
    /// An exception is matrix fragments requiring threads >= SIMD_WIDTH, see [`run_conversion`].
    pub threads: usize,
    /// Loops this conversion built, as (body, latch).
    pub loops: Vec<(Ptr<BasicBlock>, Ptr<BasicBlock>)>,
    /// Back-edge branches to be applied. Finished in `finish_pending_back_edges`.
    pending_back_edges: Vec<PendingBackEdge>,
    /// Bodies with fragment placeholders to strip after conversion.
    body_cleanups: Vec<BodyCleanup>,
}

/// The lane-vector fusion state.
/// Runs planned before conversion and run membership tracking.
#[derive(Default)]
pub(crate) struct LaneVecs {
    runs: Vec<LaneRun>,
    of_member: HashMap<Ptr<Operation>, (RunId, Slot)>,
}

/// The cross-loop staging state.
/// Plans from the `cross_loop_reuse` pass, and each member's plan.
// TODO:
#[derive(Default)]
pub(crate) struct Staging {
    /// Member load op -> (plan, is_loop_member). The `cross_loop_reuse` pass
    /// pairs two loops, the second reading the first's loads from a thread-private cache.
    cache_members: HashMap<Ptr<Operation>, (usize, bool)>,
    /// Loop op -> (plan, is_staging_loop), so `rewrite_for` can create the cache at
    /// the right position and record the IV the members index it with.
    cache_fors: HashMap<Ptr<Operation>, (usize, bool)>,
    cache_plans: Vec<CachePlan>,
}

// TODO: Store `Context` and/or `DialectConversionRewriter` in `WaxToAir`?
/// The [`DialectConversion`] strategy.
#[derive(Default)]
pub struct WaxToAir {
    /// Ops this conversion rewrote.
    pub converted: usize,
    /// Ops with no rewrite yet. A work list, not a verdict.
    pub unconverted: Unconverted,
    pub(crate) addr: Addr,
    pub(crate) frags: Frags,
    pub(crate) reduce: ReduceMem,
    pub(crate) control: Control,
    pub(crate) lanes: LaneVecs,
    pub(crate) staging: Staging,
    pub(crate) cur_code: Option<Opcode>,
}

/// What the conversion could not handle, by opcode. Empty means the function converted whole.
#[derive(Default, Debug)]
pub struct Unconverted(pub BTreeMap<String, usize>);

impl Unconverted {
    fn note(&mut self, op: Opcode) {
        *self.0.entry(format!("{op:?}")).or_insert(0) += 1;
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}
impl std::fmt::Display for Unconverted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;
        for (k, v) in &self.0 {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, "{k} x{v}")?;
            first = false;
        }
        Ok(())
    }
}

/// A loop back edge.
///
/// The latch block, [`Self::back`], already exists with `iv_next` emitted into it.
/// The terminator edge back to the header is pending.
struct PendingBackEdge {
    back: Ptr<BasicBlock>,
    hdr: Ptr<BasicBlock>,
    /// `iv + step`, already emitted in the back block.
    iv_next: Value,
    /// The yield's iteration values, in order.
    updates: Vec<Value>,
    /// `None` for a scalar iteration value, else its fragment grid.
    iter_value_frags: Vec<Option<(usize, usize)>>,
}

fn has_rewrite(code: Opcode) -> bool {
    matches!(
        code,
        Opcode::AbsF
            | Opcode::AddF
            | Opcode::AddI
            | Opcode::AndI
            | Opcode::Assert
            | Opcode::Assume
            | Opcode::Atan2
            | Opcode::AtomicCAS
            | Opcode::AtomicRMW
            | Opcode::Bitcast
            | Opcode::Broadcast
            | Opcode::CmpF
            | Opcode::CmpI
            | Opcode::Constant
            | Opcode::Cos
            | Opcode::DivF
            | Opcode::DivI
            | Opcode::Exp
            | Opcode::Exp2
            | Opcode::ExtI
            | Opcode::FToF
            | Opcode::FToI
            | Opcode::Fma
            | Opcode::For
            | Opcode::GetTensorShape
            | Opcode::GetTileBlockId
            | Opcode::IToF
            | Opcode::If
            | Opcode::IntToPtr
            | Opcode::Iota
            | Opcode::LoadPtrTko
            | Opcode::LoadViewTko
            | Opcode::Log
            | Opcode::Log2
            | Opcode::MakePartitionView
            | Opcode::MakeStridedView
            | Opcode::MakeTensorView
            | Opcode::MakeToken
            | Opcode::MaxF
            | Opcode::MaxI
            | Opcode::MinF
            | Opcode::MinI
            | Opcode::MmaF
            | Opcode::MulF
            | Opcode::MulI
            | Opcode::NegF
            | Opcode::NegI
            | Opcode::Offset
            | Opcode::OrI
            | Opcode::Reduce
            | Opcode::RemI
            | Opcode::Reshape
            | Opcode::Return
            | Opcode::Rsqrt
            | Opcode::Scan
            | Opcode::Select
            | Opcode::ShLI
            | Opcode::ShRI
            | Opcode::Sin
            | Opcode::Sqrt
            | Opcode::StoreViewTko
            | Opcode::SubF
            | Opcode::SubI
            | Opcode::TruncI
            | Opcode::XOrI
    )
}

impl DialectConversion for WaxToAir {
    fn can_convert_op(&self, ctx: &Context, op: Ptr<Operation>) -> bool {
        // Claim every wax op but the function itself. Ops with no rewrite are recorded in
        // `unconverted` rather than silently left in place. Ops this conversion inserted are
        // not wax ops and are not claimed again.
        matches!(opcode_of(ctx, op), Some(c) if c != Opcode::Entry && has_rewrite(c))
    }

    fn can_convert_type(&self, ctx: &Context, ty: TypeHandle) -> bool {
        wax_type_of(ctx, ty)
            .and_then(|w| air_type_for(ctx, &w))
            .is_some()
    }

    /// A type that is not a wax type is already an air one and passes through. A wax type
    /// with no air lowering is rejected.
    fn convert_type(&mut self, ctx: &mut Context, ty: TypeHandle) -> Result<TypeHandle> {
        let Some(w) = wax_type_of(ctx, ty) else {
            return Ok(ty);
        };
        match air_type_for(ctx, &w) {
            Some(air) => Ok(air),
            None => pliron::verify_err!(
                pliron::location::Location::Unknown,
                "no air type for wax type `{w:?}`"
            ),
        }
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        // TODO: wax op's `loc()` is dropped here. We should thread it into AIR debug metadata.
        let Some(code) = opcode_of(ctx, op) else {
            return Ok(());
        };
        // A reshape that only adds or drops 1's does not move elements, and can be considered a
        // noop / identity.
        let unit_axis_reshape = code == Opcode::Reshape && {
            let strip_ones = |s: Vec<_>| -> Vec<_> { s.into_iter().filter(|&d| d != 1).collect() };
            let shape_of = |v| match wax_operand_ty(ctx, operands_info, v) {
                Some(WaxType::Tile(tt)) => Some(strip_ones(tt.shape)),
                _ => None,
            };
            let o = op.deref(ctx);
            let a = o.operands().next();
            let b = (o.get_num_results() == 1).then(|| o.get_result(0));
            a.zip(b)
                .and_then(|(a, b)| Some(shape_of(a)? == shape_of(b)?))
                .unwrap_or(false)
        };
        // If operands are registered as using the fragment grid, then we also ensure that the
        // op code is actually one that interact with it.
        // TODO: This check should at the very least not be defined here. Additionally it seems
        // like something that can be proved as unreachable state.
        if !matches!(
            code,
            Opcode::MmaF | Opcode::StoreViewTko | Opcode::For | Opcode::Reduce | Opcode::FToF
        ) && !unit_axis_reshape
            && op
                .deref(ctx)
                .operands()
                .any(|v| self.frags.grids.contains_key(&v))
        {
            return self.refuse(code, "consumes a fragment grid");
        }
        self.cur_code = Some(code);
        match code {
            Opcode::Return => self.rewrite_return(ctx, rewriter, op),
            Opcode::LoadViewTko => self.rewrite_load(ctx, rewriter, op, operands_info),
            Opcode::StoreViewTko => self.rewrite_store(ctx, rewriter, op, operands_info),
            // Simply forwarded. Views and assume is not an air op.
            Opcode::MakeTensorView
            | Opcode::MakePartitionView
            | Opcode::MakeStridedView
            | Opcode::Reshape
            | Opcode::Assume => self.fold_view(ctx, rewriter, op, code),
            Opcode::Broadcast => self.rewrite_broadcast(ctx, rewriter, op, operands_info),
            // Ordering tokens are not handled explicitly. There is, for example, no DMA to deal
            // with in metal, so they are not explicitly needed.
            // TODO: Investigate using token presence to infer sync like barriers, fences, etc.
            Opcode::MakeToken => self.erase_to_undef(ctx, rewriter, op),
            Opcode::Constant => self.rewrite_constant(ctx, rewriter, op),
            Opcode::AddF
            | Opcode::SubF
            | Opcode::MulF
            | Opcode::DivF
            | Opcode::AddI
            | Opcode::SubI
            | Opcode::MulI
            | Opcode::DivI
            | Opcode::RemI
            | Opcode::AndI
            | Opcode::OrI
            | Opcode::XOrI
            | Opcode::ShLI
            | Opcode::ShRI => self.rewrite_binary(ctx, rewriter, op, code),
            Opcode::GetTileBlockId => self.rewrite_tile_block_id(ctx, rewriter, op),
            Opcode::Iota => self.rewrite_iota(ctx, rewriter, op),
            Opcode::If => self.rewrite_if(ctx, rewriter, op),
            Opcode::Reduce => self.rewrite_reduce(ctx, rewriter, op, operands_info),
            Opcode::Scan => self.rewrite_scan(ctx, rewriter, op, operands_info),
            Opcode::MmaF => self.rewrite_mma_f(ctx, rewriter, op, operands_info),
            Opcode::IntToPtr | Opcode::Offset | Opcode::LoadPtrTko => {
                self.rewrite_ptr_op(ctx, rewriter, op, code)
            }
            Opcode::For => self.rewrite_for(ctx, rewriter, op, operands_info),
            // `assert` carries no code: the conversion emits nothing for it.
            Opcode::Assert => self.erase_to_undef(ctx, rewriter, op),
            Opcode::CmpF => self.rewrite_cmp_float(ctx, rewriter, op),
            Opcode::MinF | Opcode::MaxI | Opcode::MinI => {
                self.rewrite_minmax_select(ctx, rewriter, op, code)
            }
            Opcode::Fma => self.rewrite_fma(ctx, rewriter, op),
            Opcode::NegI => self.rewrite_neg_int(ctx, rewriter, op),
            Opcode::NegF => self.rewrite_neg_f(ctx, rewriter, op),
            Opcode::Atan2 => self.rewrite_atan2(ctx, rewriter, op),
            Opcode::GetTensorShape => {
                self.rewrite_get_tensor_shape(ctx, rewriter, op, operands_info)
            }
            Opcode::AtomicRMW => self.rewrite_atomic_rmw(ctx, rewriter, op, operands_info),
            Opcode::AtomicCAS => self.rewrite_atomic_cas(ctx, rewriter, op, operands_info),
            Opcode::Rsqrt => self.rewrite_rsqrt(ctx, rewriter, op),
            Opcode::FToF
            | Opcode::ExtI
            | Opcode::TruncI
            | Opcode::IToF
            | Opcode::FToI
            | Opcode::Bitcast => self.rewrite_cast(ctx, rewriter, op, code, operands_info),
            // Unary float intrinsics.
            // TODO: Prefer no magic strings.
            Opcode::Exp => self.rewrite_float_call(ctx, rewriter, op, code, "@air.fast_exp", 1),
            Opcode::Log => self.rewrite_float_call(ctx, rewriter, op, code, "@air.log", 1),
            Opcode::Log2 => self.rewrite_float_call(ctx, rewriter, op, code, "@air.log2", 1),
            Opcode::Exp2 => self.rewrite_float_call(ctx, rewriter, op, code, "@air.exp2", 1),
            Opcode::Sin => self.rewrite_float_call(ctx, rewriter, op, code, "@air.sin", 1),
            Opcode::Cos => self.rewrite_float_call(ctx, rewriter, op, code, "@air.cos", 1),
            Opcode::Sqrt => self.rewrite_float_call(ctx, rewriter, op, code, "@air.sqrt", 1),
            Opcode::AbsF => self.rewrite_float_call(ctx, rewriter, op, code, "@air.fabs", 1),
            Opcode::MaxF => self.rewrite_float_call(ctx, rewriter, op, code, "@air.fast_fmax", 2),
            Opcode::CmpI => self.rewrite_cmp_int(ctx, rewriter, op),
            Opcode::Select => self.rewrite_select(ctx, rewriter, op),
            other => self.refuse(other, "supposedly has_rewrite, but has no rewrite arm"),
        }
    }
}

impl ReduceMem {
    pub(crate) fn has_global(&self, name: &str) -> bool {
        self.tg_globals.iter().any(|(n, _, _)| n == name)
    }

    pub(crate) fn declare_global(&mut self, name: &str, ty: Ty, count: u32) {
        if !self.has_global(name) {
            self.tg_globals.push((name.to_string(), ty, count));
        }
    }
}

impl WaxToAir {
    /// `bail`, as the tail of an arm that is giving up.
    pub(crate) fn refuse(&mut self, code: Opcode, reason: &str) -> Result<()> {
        self.bail(code, reason);
        Ok(())
    }
    pub(crate) fn refuse_same_width_fpcast(&mut self) {
        if let Some(c) = self.cur_code {
            self.bail(c, "no cast between same width floats");
        }
    }
    /// Record an op as unconverted, with a reason.
    pub(crate) fn bail(&mut self, code: Opcode, reason: &str) {
        self.unconverted.note(code);
        if std::env::var_os("WAX_CONVERSION_WHY").is_some() {
            eprintln!("[wax-to-air] {code:?}: {reason}");
        }
    }
    /// Replace every result with `air.undef` of the same type.
    pub(crate) fn erase_to_undef(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
    ) -> Result<()> {
        let res_tys: Vec<TypeHandle> = op_result_types(ctx, op);
        let mut vals = Vec::with_capacity(res_tys.len());
        for ty in res_tys {
            let ty = undef_type(ctx, ty);
            let u = air::UndefOp::new(ctx, ty);
            vals.push(emit_op(ctx, rewriter, u));
        }
        rewriter.replace_operation_with_values(ctx, op, vals);
        self.converted += 1;
        Ok(())
    }
    /// Replace `op` with `replacements`, padding the remaining results with per-type `undef`s.
    pub(crate) fn finish_replacement(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        res_tys: &[TypeHandle],
        mut replacements: Vec<Value>,
    ) {
        for &ty in res_tys.iter().skip(replacements.len()) {
            let ty = undef_type(ctx, ty);
            let u = air::UndefOp::new(ctx, ty);
            replacements.push(emit_op(ctx, rewriter, u));
        }
        rewriter.replace_operation_with_values(ctx, op, replacements);
        self.converted += 1;
    }
}
/// Convert `func`, erroring if any wax op is left without a rewrite.
///
/// [`run_conversion`] allows incomplete conversion, returning a list of ops that were unconverted.
/// Here an incomplete conversion is an error.
pub fn convert(ctx: &mut Context, func: Ptr<Operation>) -> Result<usize> {
    let (c, left) = run_conversion(ctx, func)?;
    if !left.is_empty() {
        let loc = func.deref(ctx).loc().clone();
        return pliron::verify_err!(loc, "conversion incomplete, no rewrite for: {left}");
    }
    Ok(c.converted)
}
/// Run the conversion. Planning passes, the driver, finish passes.
///
/// Returns the converter state and the potential list of unconverted ops.
pub fn run_conversion(ctx: &mut Context, func: Ptr<Operation>) -> Result<(WaxToAir, Unconverted)> {
    #[cfg(debug_assertions)]
    assert_verifies(ctx, func, "wax");

    let mut c = WaxToAir::default();
    if let Some(entry) = entry_block(ctx, func) {
        // TODO: Store analysis results in IR instead of in `WaxToAir`.

        // Run all analyses first
        let mut analyses = pliron::pass::AnalysisManager::default();
        c.frags.fragments = analyses
            .get_analysis::<wax_ir::passes::fragment_set::FragmentValues>(func, ctx)?
            .values
            .clone();
        // A matrix fragment is an 8x8 tile spread across the 32 lanes of one SIMD group.
        // Thus a kernel using matrix fragments needs at least a SIMD group worth of threads.
        if !c.frags.fragments.is_empty() {
            c.control.threads = c.control.threads.max(crate::gemm::SIMD_WIDTH as usize);
        }
        c.reduce.divergent = analyses
            .get_analysis::<wax_ir::passes::uniformity::DivergentValues>(func, ctx)?
            .values
            .clone();
        c.reduce.reduce_sources = analyses
            .get_analysis::<crate::wax_to_air::reduce::ColumnReduceSources>(func, ctx)?
            .sources
            .clone();
        c.reduce.tg_scratch_floats = analyses
            .get_analysis::<crate::wax_to_air::reduce::TgScratchFloats>(func, ctx)?
            .floats;
        let (tp, tg, promoted) = add_implicit_params(ctx, func, entry);
        c.control.tp = Some(tp);
        c.control.tg = Some(tg);

        c.addr.bounds = collect_access_bounds(ctx, func, &promoted);
        c.collect_lane_runs(ctx, func);
        c.addr.frag_strides = collect_frag_strides(ctx, func);
        c.collect_reduce_batches(ctx, func);
        c.collect_cache_plans(ctx, func);
    }
    pliron::irbuild::dialect_conversion::apply_dialect_conversion(ctx, &mut c, func)?;
    // Finish pending back edges and body cleanup.
    c.finish_pending_back_edges(ctx);
    c.finish_body_cleanups(ctx);

    // Now that its all air ops we can decide the unroll factor.
    apply_loop_unroll(ctx, &c.control.loops);
    // Ensure pool values reach their uses.
    hoist_pool_values(ctx, func);

    #[cfg(debug_assertions)]
    {
        assert_definitions_precede_uses(ctx, func);
        assert_verifies(ctx, func, "air");
    }

    // Report residual wax ops.
    Ok((c, residual_wax_ops(ctx, func)))
}

/// The function passes pliron's own verifier.
#[cfg(debug_assertions)]
fn assert_verifies(ctx: &Context, func: Ptr<Operation>, dialect: &str) {
    if let Err(e) = pliron::operation::verify_operation(func, ctx) {
        panic!("{dialect} function did not pass verification: {e:?}");
    }
}

/// Ensure all operand definitions are reached before the operand itself.
#[cfg(debug_assertions)]
fn assert_definitions_precede_uses(ctx: &Context, func: Ptr<Operation>) {
    use pliron::linked_list::{ContainsLinkedList, LinkedList};
    let Some(r) = func.deref(ctx).regions().next() else {
        return;
    };
    let mut seen: Vec<Ptr<Operation>> = Vec::new();
    let mut failures = Vec::new();
    for b in r.deref(ctx).iter(ctx).collect::<Vec<_>>() {
        for op in b.deref(ctx).iter(ctx).collect::<Vec<_>>() {
            let forwards = op.deref(ctx).successors().next().is_some();
            for v in op.deref(ctx).operands().collect::<Vec<_>>() {
                let DefiningEntity::Op(d) = v.defining_entity() else {
                    continue;
                };
                let reason = if d.deref(ctx).get_container().is_none() {
                    "is detached"
                } else if !forwards && !seen.contains(&d) {
                    "is defined later"
                } else {
                    continue;
                };
                failures.push(format!(
                    "operand of {} {reason}: {}",
                    Operation::get_op_dyn(op, ctx).get_opid(),
                    Operation::get_op_dyn(d, ctx).get_opid(),
                ));
            }
            seen.push(op);
        }
    }
    assert!(
        failures.is_empty(),
        "{} unreachable definitions after conversion:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Lower one kernel through conversion, wax to air to metallib.
pub fn lower_to_metallib_via_conversion(
    module: &wax_ir::Module,
    args: &[crate::bitcode::metadata::KernelArg],
    profile: &crate::profile::TargetProfile,
) -> std::result::Result<(Vec<u8>, usize), String> {
    use crate::lower::to_bitcode::Callee;

    // Clone fresh module since conversion rewrites in-place.
    let root = *module
        .entries()
        .first()
        .ok_or("module has no entry function")?;
    let kernel_name = kernel_symbol(module, root).ok_or("function has no symbol name")?;
    let mut m = wax_ir::Module::new(&kernel_name);
    let func = m
        .clone_function_from(module, root, &kernel_name)
        .map_err(|e| format!("clone entry function: {e}"))?;
    m.mark_entry(func);

    let ConvertedKernel {
        air,
        param_types,
        callees,
        tg_globals,
        threads,
    } = convert_kernel(&mut m, func)?;

    // Debug hook. The converted function in air text, for when a diff is structural.
    if let Some(path) = std::env::var_os("WAX_CONVERSION_PRINT") {
        use pliron::printable::Printable;
        let text = func.deref(&m.ctx).disp(&m.ctx).to_string();
        let _ = std::fs::write(path, text);
    }

    let callee_tys: Vec<Callee> = callees
        .iter()
        .map(|(n, r, p)| Callee::new(n, r.clone(), p.clone()))
        .collect();

    let mut bm = crate::lower::to_bitcode::BitcodeModuleAssembler::new(profile);
    for c in callee_tys {
        bm.declare_callee(c);
    }
    for (n, t, c) in tg_globals {
        bm.declare_global(n, t, c);
    }
    let body = body_from_air(
        &m.ctx,
        &air,
        param_types.len(),
        &|n| bm.callee_idx(n),
        &|n| bm.global_idx(n),
    )
    .map_err(|e| format!("bitcode body: {e}"))?;
    bm.add_kernel(
        &kernel_name,
        param_types,
        args,
        body,
        (threads > 0).then_some(threads),
    );
    let bc = bm.finish().map_err(|e| format!("bitcode: {e}"))?;

    Ok((
        crate::metallib::wrap(
            &bc,
            &[crate::metallib::Function::kernel(&kernel_name)],
            profile,
        ),
        threads,
    ))
}

pub(crate) struct ConvertedKernel {
    pub(crate) air: AirFunc,
    pub(crate) param_types: Vec<Ty>,
    pub(crate) callees: Vec<(String, Ty, Vec<Ty>)>,
    pub(crate) tg_globals: Vec<(String, Ty, u32)>,
    /// See [`Control::threads`].
    pub(crate) threads: usize,
}

/// Convert one function in place and hand back [`ConvertedKernel`].
pub(crate) fn convert_kernel(
    module: &mut wax_ir::Module,
    func: wax_ir::OpId,
) -> std::result::Result<ConvertedKernel, String> {
    let (c, left) = run_conversion(&mut module.ctx, func).map_err(|e| e.to_string())?;
    if !left.is_empty() {
        return Err(format!("conversion incomplete, no rewrite for: {left}"));
    }
    let air = as_air_func(&module.ctx, func).ok_or("converted function has no blocks")?;
    let param_types = air_param_types(&module.ctx, func).ok_or("cannot read parameter types")?;
    let callees = air_callees(&module.ctx, func).ok_or("a callee signature has no bitcode type")?;
    Ok(ConvertedKernel {
        air,
        param_types,
        callees,
        tg_globals: c.reduce.tg_globals,
        threads: c.control.threads,
    })
}

/// Move pool values to the front of the entry block and drop unused ones.
///
/// A pool value (constant, undef, null, or global reference) is not an instruction, so
/// it is placed wherever the insertion point happens to be during emission. The bitcode
/// emitter requires def before use.
fn hoist_pool_values(ctx: &mut Context, func: Ptr<Operation>) {
    use pliron::linked_list::{ContainsLinkedList, LinkedList};

    let Some(r) = func.deref(ctx).regions().next() else {
        return;
    };
    let blocks: Vec<_> = r.deref(ctx).iter(ctx).collect();
    let Some(entry) = blocks.first().copied() else {
        return;
    };
    let mut used: Vec<Ptr<Operation>> = Vec::new();
    let mut resident: Vec<Ptr<Operation>> = Vec::new();

    // Collect pool values.
    for b in &blocks {
        for op in b.deref(ctx).iter(ctx).collect::<Vec<_>>() {
            for v in op.deref(ctx).operands().collect::<Vec<_>>() {
                if let DefiningEntity::Op(d) = v.defining_entity()
                    && is_const_pool_value(ctx, d)
                    && !used.contains(&d)
                {
                    used.push(d);
                }
            }
            if is_const_pool_value(ctx, op) {
                resident.push(op);
            }
        }
    }

    // Reversed traversal since each insert goes in front of the last.
    for d in used.iter().rev() {
        if d.deref(ctx).get_container().is_some() {
            d.unlink(ctx);
        }
        d.insert_at_front(entry, ctx);
    }
    // Drop unused.
    for d in resident {
        if d.deref(ctx).get_result(0).uses(ctx).is_empty() {
            Operation::erase(d, ctx);
        }
    }
}

fn as_air_func(ctx: &Context, func: Ptr<Operation>) -> Option<AirFunc> {
    use pliron::linked_list::ContainsLinkedList;
    use pliron::op::Op;
    let r = func.deref(ctx).regions().next()?;
    let blocks: Vec<_> = r.deref(ctx).iter(ctx).collect();
    let obj = pliron::builtin::ops::FuncOp::wrap_operation(func);
    let f = obj.downcast::<pliron::builtin::ops::FuncOp>()?;
    Some(AirFunc { func: f, blocks })
}

fn air_param_types(ctx: &Context, func: Ptr<Operation>) -> Option<Vec<Ty>> {
    use pliron::r#type::Typed;
    let entry = entry_block(ctx, func)?;
    let b = entry.deref(ctx);
    (0..b.get_num_arguments())
        .map(|i| bitcode_ty(ctx, b.get_argument(i).get_type(ctx)))
        .collect()
}

fn air_callees(ctx: &Context, func: Ptr<Operation>) -> Option<Vec<(String, Ty, Vec<Ty>)>> {
    let mut out: Vec<(String, Ty, Vec<Ty>)> = Vec::new();
    let mut bad = false;
    let mut walk_blocks = |ops: &[Ptr<Operation>]| {
        for &op in ops {
            let Some(name) = callee_name(ctx, op) else {
                continue;
            };
            if out.iter().any(|(n, _, _)| *n == name) {
                continue;
            }
            use pliron::r#type::Typed;
            let o = op.deref(ctx);
            let ret = if o.get_num_results() > 0 {
                bitcode_ty(ctx, o.get_result(0).get_type(ctx))
            } else {
                Some(Ty::Void)
            };
            let ptys: Option<Vec<Ty>> = o
                .operands()
                .map(|v| bitcode_ty(ctx, v.get_type(ctx)))
                .collect();
            drop(o);
            match (ret, ptys) {
                (Some(ret), Some(ptys)) => out.push((name, ret, ptys)),
                _ => bad = true,
            }
        }
    };
    for_each_block(ctx, func, &mut walk_blocks);
    (!bad).then_some(out)
}

fn callee_name(ctx: &Context, op: Ptr<Operation>) -> Option<String> {
    let call = air::CallIntrinsicOp::wrap_operation(op).downcast::<air::CallIntrinsicOp>()?;
    call.get_attr_llvm_intrinsic_name(ctx)
        .map(|s| s.clone().into())
}

/// The `sym_name` attribute of a wax function, if any.
fn symbol_of(m: &wax_ir::Module, func: Ptr<Operation>) -> Option<String> {
    m.op_attrs(func).into_iter().find_map(|(k, v)| match v {
        wax_ir::Attribute::String(s) if k == "sym_name" => Some(s),
        _ => None,
    })
}

/// The entry function's `sym_name` sans the `_entry` suffix.
pub fn kernel_symbol(module: &wax_ir::Module, func: Ptr<Operation>) -> Option<String> {
    let raw = symbol_of(module, func)?;
    Some(raw.strip_suffix("_entry").unwrap_or(&raw).to_string())
}

/// An entry function of a module.
pub struct EntryUnit {
    pub func: wax_ir::OpId,
    pub kernel_symbol: String,
    /// `air.kernel` argument list.
    pub args: Vec<crate::bitcode::metadata::KernelArg>,
}

/// [`lower_to_metallib_via_conversion`] with several entry kernels.
pub fn lower_to_metallib_multi(
    module: &wax_ir::Module,
    units: &[EntryUnit],
    profile: &crate::profile::TargetProfile,
) -> std::result::Result<(Vec<u8>, Vec<usize>), String> {
    if units.is_empty() {
        return Err("no entry kernels to lower".to_string());
    }

    // New module that all functions will be cloned into.
    let mut m = wax_ir::Module::new(&units[0].kernel_symbol);

    let mut thread_domains: Vec<usize> = Vec::new();
    let mut collected_globals: Vec<Vec<(String, Ty, u32)>> = Vec::new();
    let mut airs: Vec<AirFunc> = Vec::new();
    let mut param_lists: Vec<Vec<Ty>> = Vec::new();
    let mut callee_lists: Vec<Vec<(String, Ty, Vec<Ty>)>> = Vec::new();
    for u in units.iter() {
        // Clone function into the new module since conversion rewrites in-place.
        let f = m
            .clone_function_from(module, u.func, &u.kernel_symbol)
            .map_err(|e| format!("{}: clone: {e}", u.kernel_symbol))?;
        m.mark_entry(f);

        let ConvertedKernel {
            air,
            param_types,
            callees,
            tg_globals,
            threads,
        } = convert_kernel(&mut m, f).map_err(|e| format!("{}: {e}", u.kernel_symbol))?;

        thread_domains.push(threads);
        collected_globals.push(tg_globals);
        airs.push(air);
        param_lists.push(param_types);
        callee_lists.push(callees);
    }

    let mut bm = BitcodeModuleAssembler::new(profile);
    callee_lists.into_iter().flatten().for_each(|(n, r, p)| {
        let _ = bm.declare_callee(Callee::new(n, r, p));
    });
    collected_globals
        .into_iter()
        .flatten()
        .for_each(|(n, t, c)| {
            bm.declare_global(n, t, c);
        });
    for (i, u) in units.iter().enumerate() {
        let body = body_from_air(
            &m.ctx,
            &airs[i],
            param_lists[i].len(),
            &|n| bm.callee_idx(n),
            &|n| bm.global_idx(n),
        )
        .map_err(|e| format!("bitcode body for {}: {e}", u.kernel_symbol))?;
        bm.add_kernel(
            u.kernel_symbol.as_str(),
            param_lists[i].clone(),
            u.args.as_slice(),
            body,
            (thread_domains[i] > 0).then_some(thread_domains[i]),
        );
    }

    let bc = bm.finish().map_err(|e| format!("bitcode: {e}"))?;

    Ok((
        crate::metallib::wrap(
            &bc,
            &units
                .iter()
                .map(|u| crate::metallib::Function::kernel(&u.kernel_symbol))
                .collect::<Vec<_>>(),
            profile,
        ),
        thread_domains,
    ))
}

pub fn converted_param_types(module: &wax_ir::Module) -> Option<Vec<Ty>> {
    let root = *module.entries().first()?;
    let mut m = wax_ir::Module::new("probe");
    let func = m.clone_function_from(module, root, "probe").ok()?;
    m.mark_entry(func);
    // Function signature is not dependant on complete conversion, so we ignore unconverted.
    let _ = run_conversion(&mut m.ctx, func).ok()?;
    air_param_types(&m.ctx, func)
}

fn undef_type(ctx: &Context, ty: TypeHandle) -> TypeHandle {
    air_ty_for(ctx, ty).unwrap_or_else(|| t::i32_ty(ctx))
}
