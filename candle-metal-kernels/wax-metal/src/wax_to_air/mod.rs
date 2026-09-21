//! `wax` dialect -> `air` dialect, as a pliron [`DialectConversion`].
//!
//! The lowering route: `lower_to_metallib` runs the entry function through this
//! conversion and emits the `.metallib` from the converted `air` ops. The spec that
//! built it was BYTE PARITY against the CFG route that preceded it - for every kernel
//! the corpus can compile, this route's `.metallib` had to be byte-identical to the
//! CFG route's, or the route had to refuse the kernel with a named opcode. That parity
//! held across the whole 167-case corpus when the CFG route was retired, and the
//! corpus (`tests/wax_to_air_corpus*.rs`) now enforces that the route keeps converting
//! every one of them. The history of the gaps this module has closed, in order, is
//! `CONVERSION_WORKLOG.md` - the rules in it are load-bearing.
//!
//! ## Shape of the conversion
//!
//! `WaxToAir` implements `DialectConversion`; `convert_reporting` runs the pre-conversion
//! planning passes, drives the conversion, runs the finish passes, and returns the
//! converter with the side decisions the bitcode step needs (globals, scratch size, loop
//! table). Two structural rules govern everything below:
//!
//! - **The driver rewrites ops in ITS order, not block order** - a body op commonly
//!   rewrites before the loop that owns it. Anything that must see a SET of sibling ops
//!   (a lane-vector run, a reduce batch, a cross-loop staging pair) is planned on the
//!   PRE-conversion function, and the emission is position-independent: emit at the
//!   canonical op's position via an explicit `set_insertion_point`, answer the later
//!   members from a recorded map.
//! - **Values the driver rewires are not the values the analyses saw.** Pre-conversion
//!   analyses record their source values up front (`reduce_sources`, `fragments`, the
//!   staging IVs), and type-reading arms read the pre-conversion history, never the live
//!   type.
//!
//! The planning passes, each mirroring a `walk.rs` decision check for check: `bounds`
//! (per-access masking), `fragments` / `divergent` (the shared analyses), `lane_runs`
//! (128-bit fused loads), `reduce_batches` (shared scratch round trips), `cache_plans`
//! (cross-loop-reuse staging). Emission arms: loads/stores, fragments + MmaF, the four
//! reduce classes, `for`/`if`/`break`, scans, constants, the two-arg float calls, atomics.
//!
//! ## Origins (the POC this grew out of)
//!
//! The file started as a mechanism prototype answering one question: does pliron's
//! conversion framework work against our two dialects at all. What it established still
//! holds: the type mapping is a pure function of the wax type (see [`air_type_for`], NOT
//! what `types::ll_type` says); one-to-many rewrites work (`wax.load_view` becomes
//! `air.gep` + `air.load`); and result arity is checked, not silent - every `*_tko` op
//! yields a token the air side has to bind to an explicit `air.undef`.

use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{
    DialectConversion, DialectConversionRewriter, OperandsInfo,
};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;

use crate::air_dialect::ops as air;
use crate::ir_builder::LlType;
use wax_ir::Opcode;
use wax_ir::dialect::types::from_pliron as wax_type_of;
use wax_ir::types::{ScalarType, TileElementType, Type as WaxType};
use pliron::r#type::TypeHandle;

mod access;
mod cache;
mod control;
mod elementwise;
mod fragments;
mod helpers;
mod lanes;
mod planning;
mod reduce;

pub(crate) use access::*;
pub(crate) use cache::*;
pub(crate) use control::*;
pub(crate) use elementwise::*;
pub(crate) use fragments::*;
pub(crate) use helpers::*;
pub(crate) use lanes::*;
pub(crate) use planning::*;
pub(crate) use reduce::*;

/// Publicly reachable through the module: the type mapping the bitcode step and the
/// tests read.
pub use helpers::air_type_for;

#[derive(Default)]
pub(crate) struct Addr {
    /// Per-access bounds, collected before conversion. See `collect_access_bounds`.
    bounds: std::collections::HashMap<Ptr<Operation>, Vec<(usize, Bound)>>,
    /// A fragment access's DYNAMIC stride values, resolved BEFORE conversion (the stride
    /// operand's defining `MakeTensorView` is gone by the time the access rewrites): access
    /// op -> per-dim loaded value (`None` where the stride is static). Mirrors
    /// `walk.rs::resolve_stride`, whose `find_stride_val_for_pv` the values come from.
    frag_strides: std::collections::HashMap<Ptr<Operation>, Vec<Option<pliron::value::Value>>>,
}

/// The fragment classification and the fragment grids it registers.
#[derive(Default)]
pub(crate) struct Frags {
    /// Fragment grids, keyed by the value that stands for them. See `grid_of`.
    grids: std::collections::HashMap<pliron::value::Value, Vec<pliron::value::Value>>,
    /// Values that must live in matrix-fragment registers, from the SHARED analysis. The
    /// classification decides which reduce lowering is correct, so guessing it routes a
    /// fragment to the scalar intrinsic and reads one lane's two elements as if they were
    /// the whole row.
    fragments: std::collections::HashSet<pliron::value::Value>,
}

/// The reduce/scan/threadgroup-memory state: the analyses a reduce lowering asks, the
/// globals it declares, and the reduce batches it planned.
#[derive(Default)]
pub(crate) struct ReduceMem {
    /// Values that are NOT per-lane, from the shared uniformity analysis run on the
    /// PRE-conversion function. A column reduce whose source is SG-uniform skips the
    /// redistribution round trip, exactly as `walk.rs` does.
    divergent: std::collections::HashSet<pliron::value::Value>,
    /// Column-reduce op -> its source value as it was BEFORE the driver rewired the
    /// operand, the value the divergence analysis ran on.
    reduce_sources:
        std::collections::HashMap<Ptr<Operation>, pliron::value::Value>,
    /// A column reduce was emitted: the scratch is live and the next writer must wait
    /// for the last readers - `walk.rs`'s `tg_scratch_dirty`.
    tg_scratch_dirty: bool,
    /// A hierarchical reduce's two globals have been requested (`walk.rs`'s
    /// `tg_reduce_allocated` - one type per kernel).
    tg_reduce_allocated: bool,
    /// A hierarchical scan's partials global has been requested (`walk.rs`'s
    /// `tg_scan_allocated`).
    tg_scan_allocated: bool,
    /// The module globals this conversion declared, in encounter order - the same order
    /// `walk.rs`'s `tg_globals` fills, which is the order the scaffolding emits them in.
    pub tg_globals: Vec<(String, LlType, u32)>,
    /// Its element count: the kernel's requested rows times the padded pitch, else the
    /// default. `walk.rs` computes the same value; the two must stay in sync or the
    /// globals the two paths declare diverge.
    pub tg_scratch_floats: u32,
    /// Column-reduce batches planned before conversion, by `collect_reduce_batches`:
    /// member op -> (slot, record). `walk.rs`'s `batch_column_reduces` shares one set of
    /// barriers across a run of adjacent `[32, 32]` reduces; the pre-pass exists for the
    /// same reason the lane-vector one does - the driver reaches the members in its own
    /// order, so the run must be decided on the pre-conversion function.
    reduce_batches: std::collections::HashMap<Ptr<Operation>, (usize, usize)>,
    reduce_batch_records: Vec<ReduceBatch>,
    /// A batch the FIRST processed member already emitted: member op -> its result.
    reduce_batch_results: std::collections::HashMap<Ptr<Operation>, pliron::value::Value>,
}

/// The loop and branch state: the implicit position parameters, the loops built, the
/// back edges and body cleanups deferred to after conversion.
#[derive(Default)]
pub(crate) struct Control {
    /// `%_tp`, the thread position, once the signature rewrite has introduced it.
    pub tp: Option<pliron::value::Value>,
    /// `%_tg`, the threadgroup position.
    pub tg: Option<pliron::value::Value>,
    /// Loops this conversion built, as (body, latch). The unroll hint cannot be decided
    /// while `wax.for` is rewritten - the factor is a function of the LOWERED instruction
    /// count, and the body is still wax ops at that point - so the decision is deferred to
    /// `apply_loop_unroll` once conversion has finished.
    pub loops: Vec<(
        Ptr<pliron::basic_block::BasicBlock>,
        Ptr<pliron::basic_block::BasicBlock>,
    )>,
    /// Exit blocks of carry-free loops, innermost last, for `wax.break`.
    pub loop_exits: Vec<Ptr<pliron::basic_block::BasicBlock>>,
    /// Back edges that could not be terminated at `for`-rewrite time.
    ///
    /// A MULTI-fragment carry's update is the `mma`'s result, and the driver rewrites the
    /// `for` before the body ops, so the update's grid does not exist yet. The branch is
    /// built in `finish_pending_latches` once the grids do.
    pending_latches: Vec<PendingLatch>,
    /// Bodies with fragment placeholders to strip after conversion.
    body_cleanups: Vec<BodyCleanup>,
}

/// The lane-vector fusion state: the runs planned before conversion and what the
/// already-emitted run head recorded for the later members.
#[derive(Default)]
pub(crate) struct LaneVecs {
    /// A lane-vector run the FIRST member already emitted: member op -> (slot, the
    /// `extractelement` value). `walk.rs` fuses W contiguous per-lane scalar loads into one
    /// wide transaction and answers the later members from its extracts; the map is how the
    /// later members' rewrites find their slot once the first one is erased.
    lane_vectors: std::collections::HashMap<Ptr<Operation>, (usize, pliron::value::Value)>,
    /// Runs planned before conversion, by `collect_lane_runs`: member op -> (slot, record).
    lane_runs: std::collections::HashMap<Ptr<Operation>, (usize, usize)>,
    lane_records: Vec<LaneRun>,
    /// Store runs whose wide store is already emitted, by record index. The load runs
    /// answer later members through `lane_vectors`; a store member only needs its token
    /// bound, so a plain done-flag suffices.
    lane_store_done: std::collections::HashSet<usize>,
    /// A store run's member values, captured WHEN EACH MEMBER IS PROCESSED: record ->
    /// slot -> the converted value. A processed member's op is erased, so its operand
    /// can no longer be read - the capture is the only copy. A member not yet processed
    /// is still in the block, so its operand is read live at emission.
    lane_store_values: std::collections::HashMap<usize, std::collections::HashMap<usize, pliron::value::Value>>,
}

/// The cross-loop staging state, `walk.rs`'s `smem_cache`: the plans paired by the
/// `cross_loop_reuse` pass and the members' slots in them.
#[derive(Default)]
pub(crate) struct Staging {
    /// Member load op -> (record, is a STAGING-loop member). The `cross_loop_reuse` pass
    /// pairs two loops; the second reads the first's loads from a thread-private cache
    /// instead of global memory. Planned pre-conversion like the lane runs and the reduce
    /// batches - the driver reaches the members in its own order, and the alloca and the
    /// IV wiring must hold up to it.
    cache_members: std::collections::HashMap<Ptr<Operation>, (usize, bool)>,
    /// Loop op -> (record, is the staging loop), so `rewrite_for` can create the cache at
    /// the right position and record the IV the members index it with.
    cache_fors: std::collections::HashMap<Ptr<Operation>, (usize, bool)>,
    cache_plans: Vec<CachePlan>,
}

/// Rewrites the op-local part of the `wax` dialect into `air`.
#[derive(Default)]
pub struct WaxToAir {
    /// Ops this conversion rewrote.
    pub converted: usize,
    /// Ops it has no rewrite for yet. The work list, produced by running it.
    pub unconverted: Unconverted,
    pub(crate) addr: Addr,
    pub(crate) frags: Frags,
    pub(crate) reduce: ReduceMem,
    pub(crate) control: Control,
    pub(crate) lanes: LaneVecs,
    pub(crate) staging: Staging,
}
/// What the conversion could not handle, by opcode. Empty means the function converted whole.
#[derive(Default, Debug)]
pub struct Unconverted(pub std::collections::BTreeMap<String, usize>);

/// A loop back edge whose branch was deferred until the fragment grids exist.
struct PendingLatch {
    back: Ptr<pliron::basic_block::BasicBlock>,
    hdr: Ptr<pliron::basic_block::BasicBlock>,
    /// `iv + step`, already emitted in the back block.
    iv_next: pliron::value::Value,
    /// The yield's carry values, in carry order.
    updates: Vec<pliron::value::Value>,
    /// `None` for a scalar carry, else the carry's fragment grid.
    carry_frags: Vec<Option<(usize, usize)>>,
}
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
impl DialectConversion for WaxToAir {
    fn can_convert_op(&self, ctx: &Context, op: Ptr<Operation>) -> bool {
        // Claim every wax op but the function itself. Ops with no rewrite yet are recorded in
        // `unconverted` rather than silently left in place, so the work list comes from running
        // it. Ops this conversion inserted are not wax ops and are not claimed again.
        matches!(try_opcode_of(ctx, op), Some(c) if c != Opcode::Entry)
    }

    fn can_convert_type(&self, ctx: &Context, ty: TypeHandle) -> bool {
        wax_type_of(ctx, ty)
            .as_ref()
            .and_then(air_type_for)
            .is_some()
    }

    fn convert_type(&mut self, ctx: &mut Context, ty: TypeHandle) -> Result<TypeHandle> {
        let Some(ll) = wax_type_of(ctx, ty).as_ref().and_then(air_type_for) else {
            return Ok(ty);
        };
        Ok(crate::air_dialect::types::to_pliron(ctx, &ll))
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let _ = operands_info;
        let Some(code) = try_opcode_of(ctx, op) else {
            return Ok(());
        };
        let _operands_info = operands_info;
        // A value standing for a fragment GRID may only be consumed by an arm that knows to
        // look the grid up. Any other arm would silently operate on fragment 0 and drop the
        // rest, which is a wrong answer rather than a missing feature - so refuse instead.
        // `For` is admitted because it carries grids explicitly: inits and updates are
        // resolved through the table in its own arm, and a missing grid is a bail there.
        // `Reduce` the same way: its fragment arm reads the grid and bails without one.
        // `FToF` narrows or widens every register of the grid (`walk.rs::cast_op` does the
        // same) and re-binds the result to a grid.
        if !matches!(
            code,
            Opcode::MmaF | Opcode::StoreViewTko | Opcode::For | Opcode::Reduce | Opcode::FToF
        )
            && op
                .deref(ctx)
                .operands()
                .any(|v| self.frags.grids.contains_key(&v))
        {
            self.bail(code, "consumes a fragment grid");
            return Ok(());
        }
        match code {
            Opcode::Return => self.rewrite_return(ctx, rewriter, op),
            Opcode::LoadViewTko => self.rewrite_access(ctx, rewriter, op, _operands_info, false),
            Opcode::StoreViewTko => self.rewrite_access(ctx, rewriter, op, _operands_info, true),
            // Identity forwarders. A view is not an air op: it resolves to the pointer it was
            // built from, and its shape and strides live in the type, where the address
            // arithmetic reads them. `assume` is the same shape - it carries a fact in an
            // attribute and forwards its operand - and the CFG path handles all of these
            // together, in one arm, for exactly this reason.
            Opcode::MakeTensorView
            | Opcode::MakePartitionView
            | Opcode::MakeStridedView
            | Opcode::Broadcast
            | Opcode::Reshape
            | Opcode::Assume => self.fold_view(ctx, rewriter, op),
            // Ordering tokens have no AGX counterpart - there is no DMA engine to order
            // against - so they become `undef`, which is what the CFG path binds them to.
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
            Opcode::Reduce => self.rewrite_reduce(ctx, rewriter, op, _operands_info),
            Opcode::Scan => self.rewrite_scan(ctx, rewriter, op, _operands_info),
            Opcode::Break => self.rewrite_break(ctx, rewriter, op),
            Opcode::MmaF => self.rewrite_mma_f(ctx, rewriter, op, _operands_info),
            Opcode::IntToPtr | Opcode::Offset | Opcode::LoadPtrTko => {
                self.rewrite_ptr_op(ctx, rewriter, op, code)
            }
            Opcode::For => self.rewrite_for(ctx, rewriter, op, _operands_info),
            // `assert` carries no code: the CFG path emits nothing for it.
            Opcode::Assert => self.erase_to_undef(ctx, rewriter, op),
            Opcode::CmpF => self.rewrite_cmp_float(ctx, rewriter, op),
            Opcode::MinF | Opcode::MaxI | Opcode::MinI => {
                self.rewrite_minmax_select(ctx, rewriter, op, code)
            }
            Opcode::Fma => self.rewrite_fma(ctx, rewriter, op),
            Opcode::NegI => self.rewrite_neg_int(ctx, rewriter, op),
            Opcode::NegF => self.rewrite_neg_f(ctx, rewriter, op),
            Opcode::Atan2 => self.rewrite_atan2(ctx, rewriter, op),
            Opcode::GetTensorShape => self.rewrite_get_tensor_shape(ctx, rewriter, op, _operands_info),
            Opcode::AtomicRMW => self.rewrite_atomic_rmw(ctx, rewriter, op, _operands_info),
            Opcode::AtomicCAS => self.rewrite_atomic_cas(ctx, rewriter, op, _operands_info),
            Opcode::Rsqrt => self.rewrite_rsqrt(ctx, rewriter, op),
            Opcode::FToF
            | Opcode::ExtI
            | Opcode::TruncI
            | Opcode::IToF
            | Opcode::FToI
            | Opcode::Bitcast => self.rewrite_cast(ctx, rewriter, op, code, _operands_info),
            // Unary float intrinsics. `fast_` variants where the CFG path uses them.
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
            other => {
                self.unconverted.note(other);
                Ok(())
            }
        }
    }
}
impl WaxToAir {
    /// Record an op as unconverted, with a reason.
    ///
    /// The reason is for the corpus driver: an opcode name alone says a kernel did not
    /// convert, not which restriction stopped it, and with several bail sites per arm that
    /// is the difference between a work list and a guess.
    pub(crate) fn bail(&mut self, code: Opcode, why: &str) {
        self.unconverted.note(code);
        if std::env::var_os("WAX_CONVERSION_WHY").is_some() {
            eprintln!("[conv-why] {code:?}: {why}");
        }
    }
    /// The air type for a type read off a wax op.
    ///
    /// Every air op this conversion builds must be given a CONVERTED type. `convert_type`
    /// handles block arguments and records operand history, but a type read from an existing
    /// op's result is still the wax one, and handing that to an air op leaves a wax type in the
    /// module for the emitter to trip over.
    pub(crate) fn air_ty(&self, ctx: &mut Context, ty: TypeHandle) -> Option<TypeHandle> {
        let ll = wax_type_of(ctx, ty).as_ref().and_then(air_type_for)?;
        Some(crate::air_dialect::types::to_pliron(ctx, &ll))
    }
    /// The air type to give an `undef` standing in for a wax result.
    ///
    /// A token has no air type - there is nothing to order against on AGX - and the CFG path
    /// binds it to `undef i32`. Anything else keeps its converted type.
    pub(crate) fn undef_type(&self, ctx: &mut Context, ty: TypeHandle) -> TypeHandle {
        match wax_type_of(ctx, ty).as_ref().and_then(air_type_for) {
            Some(ll) => crate::air_dialect::types::to_pliron(ctx, &ll),
            None => crate::air_dialect::types::to_pliron(ctx, &LlType::I32),
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
            let ty = self.undef_type(ctx, ty);
            let u = air::UndefOp::new(ctx, ty);
            rewriter.insert_operation(ctx, u.get_operation());
            vals.push(u.get_operation().deref(ctx).get_result(0));
        }
        rewriter.replace_operation_with_values(ctx, op, vals);
        self.converted += 1;
        Ok(())
    }
    /// Replace `op` with `replacements`, padding the remaining results with per-type
    /// `undef`s - AGX's ordering tokens have nothing to order against - and counting
    /// the op converted.
    pub(crate) fn finish_replacement(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        res_tys: &[TypeHandle],
        mut replacements: Vec<pliron::value::Value>,
    ) {
        for &ty in res_tys.iter().skip(replacements.len()) {
            let ty = self.undef_type(ctx, ty);
            let u = air::UndefOp::new(ctx, ty);
            rewriter.insert_operation(ctx, u.get_operation());
            replacements.push(u.get_operation().deref(ctx).get_result(0));
        }
        rewriter.replace_operation_with_values(ctx, op, replacements);
        self.converted += 1;
    }
}
/// Apply the conversion to `func`, returning how many ops were rewritten.
pub fn convert(ctx: &mut Context, func: Ptr<Operation>) -> Result<usize> {
    Ok(convert_reporting(ctx, func)?.0.converted)
}
/// Apply the conversion, returning the converter itself - whose `unconverted` is the work
/// list, every opcode with no rewrite yet and how often it appeared; an empty one means
/// the function converted whole. The converter is returned because the bitcode step needs
/// its side decisions (the threadgroup scratch a column reduce declared, its size).
pub fn convert_reporting(
    ctx: &mut Context,
    func: Ptr<Operation>,
) -> Result<(WaxToAir, Unconverted)> {
    let mut c = WaxToAir::default();
    // Constant-space buffer parameters, before the driver converts any block argument.
    // `air_type_for` maps a pointer tile to a DEVICE pointer, because the type alone cannot
    // say otherwise; which slots bind in addrspace(2) is a compile option carried on the
    // function as `const_param_slots`. Retyping the arguments up front means every gep and
    // load downstream inherits the right address space, as they do on the CFG path.
    // The implicit parameters must exist before any op that reads them is rewritten: the lane
    // term reads `%_tp`.
    if let Some(entry) = entry_block(ctx, func) {
        // BEFORE the parameter rewrites. Those insert `air.load` and `air.const_bits` into
        // the entry block, and `DialectView` reads opcodes through `opcode_of`, which panics
        // on an op that is not wax. The classification is unaffected by the parameter
        // rewrite, so the ordering costs nothing.
        c.frags.fragments = wax_ir::passes::fragment_set::fragment_values(
            &wax_ir::passes::dialect_view::DialectView { ctx },
            func,
        );
        // With `fragments`, and for the same reason: the parameter rewrites below insert
        // `air` ops into the entry block, and `DialectView` reads opcodes through
        // `opcode_of`, which panics on an op that is not wax. The uniformity answer and
        // the column-reduce sources are also read off the PRE-conversion function: the
        // analysis is seeded from wax ops, and the operand it saw is not the value the
        // driver hands the reduce arm later.
        c.reduce.divergent = wax_ir::passes::uniformity::divergent_values(
            &wax_ir::passes::dialect_view::DialectView { ctx },
            func,
        );
        c.reduce.reduce_sources = record_column_reduce_sources(ctx, func);
        c.reduce.tg_scratch_floats = tg_scratch_floats_of(ctx, func);
        let (tp, tg, promoted) = add_implicit_params(ctx, func, entry);
        c.control.tp = Some(tp);
        c.control.tg = Some(tg);
        retype_const_params(ctx, func, entry);
        // After the extents are promoted, so a dynamic bound records the LOADED value - and
        // only where one exists. See `collect_access_bounds`.
        c.addr.bounds = collect_access_bounds(ctx, func, &promoted);
        // The lane-vector runs, with the bounds already decided: a masked run cannot fuse.
        c.collect_lane_runs(ctx, func);
        // The fragment accesses' dynamic strides, from the same pre-conversion walk.
        c.addr.frag_strides = collect_frag_strides(ctx, func);
        // The shared column-reduce batches, the same way: the driver's order decides who
        // emits, the pre-conversion run decides what is emitted.
        c.collect_reduce_batches(ctx, func);
        // The cross-loop staging plans, last of the pre-passes: a lane-vector member is not
        // cached, so the run collection comes first.
        c.collect_cache_plans(ctx, func);
    }
    pliron::irbuild::dialect_conversion::apply_dialect_conversion(ctx, &mut c, func)?;
    // Multi-fragment loop back edges wait for the body's mma to register its grid.
    c.finish_pending_latches(ctx);
    // Fragment placeholders in loop bodies are conversion scaffolding; the CFG path's body
    // blocks have no arguments at all.
    c.finish_body_cleanups(ctx);
    // Now that the bodies are air ops, the unroll factor is decidable.
    apply_loop_unroll(ctx, &c.control.loops);
    // Report what is actually LEFT, not what was noted along the way. An arm records a note
    // when it declines an op, but the driver's order is its own: a `wax.yield` is commonly
    // reached before the `wax.for` that owns it, noted as unhandled, and then erased by that
    // `for` moments later. Reporting the notes rejects kernels that converted whole.
    Ok((c, residual_wax_ops(ctx, func)))
}
/// Lower one kernel through the conversion: wax dialect -> air dialect -> metallib.
///
/// The experimental route. `lower_to_metallib` remains the shipping one and is untouched; this
/// exists to be compared against it, for output and for cost.
///
/// Fails rather than falls back. A kernel the conversion cannot handle whole is reported with
/// the opcodes it choked on, because silently producing a different lowering is exactly what a
/// comparison must not do.
pub fn lower_to_metallib_via_conversion(
    module: &wax_ir::Module,
    args: &[crate::bitcode::metadata::KernelArg],
    tg_globals: &[(String, LlType, u32)],
    profile: &crate::profile::TargetProfile,
) -> std::result::Result<Vec<u8>, String> {
    // The conversion rewrites in place, so build a fresh module from the caller's rather than
    // mutating it: the CFG path must be able to lower the same kernel afterwards. `Module`
    // implements `WaxSource`, so this is the same builder the frontend uses.
    let root = *module
        .functions
        .first()
        .ok_or("module has no entry function")?;
    let raw_kernel_name = symbol_of(module, root).ok_or("function has no symbol name")?;
    // The frontend names the entry `<kernel>_entry`; the exported symbol drops the suffix,
    // as `lower_kernel` does.
    let kernel_name = raw_kernel_name
        .strip_suffix("_entry")
        .unwrap_or(&raw_kernel_name)
        .to_string();
    let params: Vec<_> = {
        use wax_ir::source::WaxSource;
        let r = module.regions(root);
        let first = r.first().copied().ok_or("entry has no region")?;
        let b = module.region_blocks(first);
        let e = b.first().copied().ok_or("entry has no block")?;
        module.block_args(e).into_iter().map(|(_, t)| t).collect()
    };
    let mut m = wax_ir::from_source::module_from_source(&kernel_name, module, root, &params)
        .map_err(|e| e.to_string())?;
    let func = *m.functions.first().ok_or("built module has no function")?;

    let (c, left) = convert_reporting(&mut m.ctx, func).map_err(|e| e.to_string())?;
    // The threadgroup globals a reduce declared, in the encounter order `walk.rs`
    // declares the identical ones. An empty list is the common case: most kernels
    // touch no threadgroup memory at all.
    let tg_globals: Vec<(String, LlType, u32)> = if !tg_globals.is_empty() {
        tg_globals.to_vec()
    } else {
        c.reduce.tg_globals
    };
    if !left.is_empty() {
        return Err(format!("conversion incomplete, no rewrite for: {left}"));
    }

    let air = as_air_func(&m.ctx, func).ok_or("converted function has no blocks")?;

    // Debug hook: the converted function in air text, for when a `DIFF` is structural.
    if let Some(path) = std::env::var_os("WAX_CONVERSION_PRINT") {
        use pliron::printable::Printable;
        let text = func.deref(&m.ctx).disp(&m.ctx).to_string();
        let _ = std::fs::write(path, text);
    }

    let param_types = air_param_types(&m.ctx, func).ok_or("cannot read parameter types")?;
    let callees = air_callees(&m.ctx, func);
    let num_params = param_types.len();

    let bc = crate::lower::to_bitcode::bitcode_with_body_parts(
        &kernel_name,
        &callees,
        &param_types,
        args,
        &tg_globals,
        profile,
        &|callee_index, global_index| {
            crate::air_dialect::to_bitcode::body_from_air(
                &m.ctx,
                &air,
                num_params,
                callee_index,
                global_index,
            )
        },
    )
    .map_err(|e| format!("bitcode: {e}"))?;

    Ok(crate::metallib::wrap(
        &bc,
        &[crate::metallib::Function::kernel(&kernel_name)],
        profile,
    ))
}
/// Wrap a converted function as the `air` emitter expects it.
fn as_air_func(
    ctx: &Context,
    func: Ptr<Operation>,
) -> Option<crate::air_dialect::to_bitcode::AirFunc> {
    use pliron::linked_list::ContainsLinkedList;
    use pliron::op::Op;
    let r = func.deref(ctx).regions().next()?;
    let blocks: Vec<_> = r.deref(ctx).iter(ctx).collect();
    let obj = pliron::builtin::ops::FuncOp::wrap_operation(func);
    let f = obj.downcast::<pliron::builtin::ops::FuncOp>()?;
    Some(crate::air_dialect::to_bitcode::AirFunc { func: f, blocks })
}
/// The converted function's parameter types, from its entry block arguments.
fn air_param_types(ctx: &Context, func: Ptr<Operation>) -> Option<Vec<LlType>> {
    use pliron::r#type::Typed;
    let entry = entry_block(ctx, func)?;
    let b = entry.deref(ctx);
    (0..b.get_num_arguments())
        .map(|i| crate::air_dialect::types::from_pliron(ctx, b.get_argument(i).get_type(ctx)))
        .collect()
}
/// Every callee the converted function references, with its signature.
///
/// A CFG records these as it emits calls; an `air` module has to be scanned for them. Order is
/// declaration order in the bitcode module, and call records reference callees by that order,
/// so it must be stable - hence the de-duplication preserving first appearance.
fn air_callees(ctx: &Context, func: Ptr<Operation>) -> Vec<(String, LlType, Vec<LlType>)> {
    let mut out: Vec<(String, LlType, Vec<LlType>)> = Vec::new();
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
            let ret = (o.get_num_results() > 0)
                .then(|| crate::air_dialect::types::from_pliron(ctx, o.get_result(0).get_type(ctx)))
                .flatten()
                .unwrap_or(LlType::Void);
            let ptys: Vec<LlType> = o
                .operands()
                .filter_map(|v| crate::air_dialect::types::from_pliron(ctx, v.get_type(ctx)))
                .collect();
            drop(o);
            out.push((name, ret, ptys));
        }
    };
    for_each_block(ctx, func, &mut walk_blocks);
    out
}
/// The callee symbol of an `air.call`, or `None` for any other op.
fn callee_name(ctx: &Context, op: Ptr<Operation>) -> Option<String> {
    let is_call = Operation::get_op_dyn(op, ctx)
        .op_ref()
        .get_opid()
        .to_string()
        == "air.call";
    if !is_call {
        return None;
    }
    op.deref(ctx)
        .attributes
        .0
        .iter()
        .find(|(k, _)| k.to_string() == crate::air_dialect::ops::ATTR_KEY_AIR_CALLEE)
        .and_then(|(_, a)| a.downcast_ref::<pliron::builtin::attributes::StringAttr>())
        .map(|s| s.clone().into())
}
/// The `sym_name` attribute of a function in the wax dialect.
fn symbol_of(m: &wax_ir::Module, func: Ptr<Operation>) -> Option<String> {
    m.op_attrs(func).into_iter().find_map(|(k, v)| match v {
        wax_ir::Attribute::String(s) if k == "sym_name" => Some(s),
        _ => None,
    })
}
/// The AIR parameter list the conversion produces for a module's entry function.
///
/// Exposed because the signature is decided by the parameter rewrite alone - the buffer
/// pointers, the promoted runtime extents, then `%_tp`/`%_tg` - and is therefore checkable
/// even for a kernel whose body does not convert whole. It is an ABI: a missing or extra
/// parameter is a wrong answer at dispatch, not a compile error.
pub fn converted_param_types(module: &wax_ir::Module) -> Option<Vec<LlType>> {
    use wax_ir::source::WaxSource;
    let root = *module.functions.first()?;
    let params: Vec<_> = {
        let r = module.regions(root);
        let b = module.region_blocks(*r.first()?);
        module
            .block_args(*b.first()?)
            .into_iter()
            .map(|(_, t)| t)
            .collect()
    };
    let mut m =
        wax_ir::from_source::module_from_source("probe", module, root, &params).ok()?;
    let func = *m.functions.first()?;
    // The residual work list is ignored on purpose: the signature does not depend on whether
    // the body converted.
    let _ = convert_reporting(&mut m.ctx, func).ok()?;
    air_param_types(&m.ctx, func)
}

