//! Translate a tile-level kernel into the typed CFG builder.
//!
//! `try_lower_via_cfg` takes a whole function through [`cfg::FuncBuilder`]. `None` means
//! the kernel uses a construct the backend cannot lower, and the caller turns that into a
//! precise error - there is no fallback emitter.
//!
//! The input is a [`WaxIr`], not a `cutile_ir::Module`: materialising first means any
//! [`super::source::WaxSource`] can drive the SAME emission decisions, so a second
//! frontend costs nothing here.

use std::collections::HashMap;

use wax_ir::Opcode;
use wax_ir::{Attribute, ScalarType, TileElementType, TileType, Type};

use super::ir::{BlockRef, OpData, OpRef, ValRef, WaxIr};
use super::{constant_literal, constant_value, scalar_of};
use crate::ir_builder::LlType;
use crate::lower::cfg::{
    BlockId as CfgBlockId, FuncBuilder, Operand, Terminator, ValueId as CfgValueId,
};
use crate::types::param_needs_load;

/// With `METAL_TILE_BITCODE_PROBE` set, report whether this CFG can be encoded as
/// bitcode yet. Diagnostic only: the text path is unaffected either way. Lets coverage
/// be measured over every kernel a real build produces, rather than guessed at from the
/// shape of the source.
fn bitcode_coverage_probe(kernel_name: &str, f: &super::cfg::FuncBuilder) {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var("METAL_TILE_BITCODE_PROBE").is_ok()) {
        return;
    }
    // Every callee resolves, so the probe measures instruction coverage rather than
    // tripping over declaration bookkeeping the real path will supply.
    match super::to_bitcode::body_from_cfg(f, &|_| Some(0), &|_| Some(0)) {
        Ok(body) => eprintln!(
            "[bitcode-probe] OK   {kernel_name}  ({} insts, {} blocks)",
            body.insts.len(),
            body.num_blocks
        ),
        Err(e) => eprintln!("[bitcode-probe] MISS {kernel_name}  {e}"),
    }
}

/// Per-kernel lowering choices, grouped so the entry point keeps a readable signature.
pub(crate) struct KernelOptions<'a> {
    /// Buffer-parameter slots to bind in constant address space.
    pub const_slots: &'a std::collections::HashSet<usize>,
    /// Rows of `tg_store`/`tg_load` scratch this kernel needs, if it asked.
    pub tg_scratch_rows: Option<i32>,
    /// Index the kernel's first `!llvm.loop` metadata node takes.
    pub meta_base: usize,
}

/// Try to translate `entry_block` of a kernel into typed-CFG AIR text. Returns
/// `Some((function_text, tg_globals))` - a full `define ... { ... }` plus any
/// threadgroup-memory global declarations to emit before it - if fully handled,
/// or `None` if the kernel cannot be lowered (the caller reports it as an error).
pub(crate) fn try_lower_via_cfg(
    ir: &WaxIr,
    kernel_name: &str,
    entry_block_id: BlockRef,
    params: &[(ValRef, Type)],
    dyn_stride_args: &[ValRef],
    dyn_shape_args: &[ValRef],
    opts: &KernelOptions<'_>,
) -> Option<Lowered> {
    let KernelOptions {
        const_slots,
        tg_scratch_rows,
        meta_base,
    } = *opts;
    let mut w = Walker {
        ir,
        sgmma_vals: super::sgmma_set::sgmma_values(ir),
        divergent: super::uniformity::divergent_values(ir),
        cur_block: None,
        fused_loads: std::collections::HashSet::new(),
        batched_reduces: HashMap::new(),
        tg_scratch_dirty: false,
        promoted_dims: std::collections::HashSet::new(),
        f: FuncBuilder::new(kernel_name),
        kernel_name: kernel_name.to_string(),
        vmap: HashMap::new(),
        tp: Operand::i32(0),
        tg: Operand::i32(0),
        globals: Vec::new(),
        tg_reduce_allocated: false,
        tg_scan_allocated: false,
        tg_scratch_allocated: false,
        tg_globals: Vec::new(),
        tg_scratch_floats: tg_scratch_rows
            .map(|r| r * tg_row_pitch(32))
            .unwrap_or(TG_SCRATCH_FLOATS_DEFAULT),
        label_ctr: 0,
        smem_cache: HashMap::new(),
        smem_mode: HashMap::new(),
        loop_iv_stack: Vec::new(),
        loop_exit_stack: Vec::new(),
    };
    // Params (and the dynamic-stride prologue loads) emit into the entry block.
    w.f.append_block("entry");
    if w.setup_params(params, dyn_stride_args, dyn_shape_args, const_slots)
        .is_none()
    {
        cfg_bail(kernel_name, "setup_params");
        return None;
    }
    w.translate_block(entry_block_id)?;
    bitcode_coverage_probe(kernel_name, &w.f);
    let (text, loop_meta) = match w.f.serialize(meta_base) {
        Ok(t) => t,
        Err(e) => {
            cfg_bail(kernel_name, &format!("serialize: {e:?}"));
            return None;
        }
    };
    Some(Lowered {
        text,
        globals: w.globals,
        loop_meta,
        func: w.f,
        tg_globals: w.tg_globals,
    })
}

/// What one lowered kernel yields: the AIR text the assembler path consumes, plus the
/// CFG itself so the in-process bitcode writer can encode the same function without
/// re-deriving anything from the text.
pub(crate) struct Lowered {
    pub text: String,
    /// Module-level declarations (threadgroup globals, intrinsic prototypes) as text.
    pub globals: Vec<String>,
    /// `!llvm.loop` nodes the body references, numbered from `meta_base`.
    pub loop_meta: Vec<String>,
    pub func: FuncBuilder,
    /// Threadgroup globals as (name, element type, element count); count 1 is a scalar.
    pub tg_globals: Vec<(String, LlType, u32)>,
}

/// Choose an unroll factor for a counted loop with a straight-line body of
/// `body_insts` instructions, or `None` to leave it rolled.
///
/// Rolled reduction/elementwise loops (e.g. GEMV's dot product) are bound by loop
/// overhead and by too few global loads in flight to hide memory latency. Forcing
/// several body copies per iteration issues independent loads and cuts the
/// branch/induction overhead - measured ~7% on 4096x4096 GEMV. It is a pure
/// hint: the driver unrolls a semantically
/// identical loop, so results stay bit-for-bit unchanged (no reassociation).
///
/// Only small straight-line bodies qualify. Nested control flow and large bodies
/// (e.g. SGMMA GEMM inner loops) are left alone to avoid code-size blowup and
/// register pressure. The factor is sized so the unrolled body stays around ~160
/// instructions, capped at 8x (the empirical sweet spot on Apple GPUs).
fn loop_unroll_factor(body_straight: bool, body_insts: usize) -> Option<u32> {
    // METAL_TILE_UNROLL overrides the policy for tuning: `0` disables unrolling,
    // any other value forces that factor on every straight-line counted loop
    // (including the large SGMMA bodies this normally skips).
    if let Some(v) = std::env::var_os("METAL_TILE_UNROLL")
        && let Some(n) = v.to_str().and_then(|s| s.parse::<u32>().ok())
    {
        if !body_straight || body_insts == 0 {
            return None;
        }
        return if n == 0 { None } else { Some(n) };
    }
    if !body_straight || body_insts == 0 || body_insts > 32 {
        return None;
    }
    Some((160 / body_insts).clamp(2, 8) as u32)
}

/// Log a CFG-backend bail (kernel + reason) when `CFG_AUDIT` is set.
fn cfg_bail(kernel: &str, reason: &str) {
    if std::env::var_os("CFG_AUDIT").is_some() {
        eprintln!("CFG_BAIL kernel={kernel} reason={reason}");
    }
}

struct Walker<'a> {
    ir: &'a WaxIr,
    /// Values that must be represented as SGMMA fragment registers. Computed from what
    /// the MMA ops actually touch - see `sgmma_set` for why shape alone will not do.
    sgmma_vals: std::collections::HashSet<ValRef>,
    /// Values that differ across the lanes of a SIMD group. Everything else every lane
    /// holds identically - see `uniformity` for why this is a fixpoint and not a peephole.
    divergent: std::collections::HashSet<ValRef>,
    /// The block being translated, so a load can see its siblings (see `fuse_lane_vector`).
    cur_block: Option<BlockRef>,
    /// Column reduces already answered by a batched lowering; reaching one binds the
    /// result that batch produced instead of lowering it again.
    batched_reduces: HashMap<OpRef, Operand>,
    /// Something has already written the shared threadgroup scratch, so the next writer
    /// has to wait for those readers before reusing it.
    tg_scratch_dirty: bool,
    /// Block args whose runtime extent reached the signature; see `find_shape_val_for_pv`.
    promoted_dims: std::collections::HashSet<ValRef>,
    /// Loads already answered by a fused wide load; reaching one is a no-op.
    fused_loads: std::collections::HashSet<OpRef>,
    f: FuncBuilder,
    kernel_name: String,
    /// Representation registry: each IR value maps to one-or-more CFG operands -
    /// a single operand for scalars/1x1 SGMMA, or an nrxnc row-major fragment grid
    /// for a multi-fragment SGMMA tile. This is the single source of truth that
    /// keeps every op (and loop-carry phi) agreeing on a value's representation.
    vmap: HashMap<ValRef, Vec<Operand>>,
    tp: Operand, // %_tp : <3 x i32>
    tg: Operand, // %_tg : <3 x i32>
    /// Module-level threadgroup global declarations (addrspace(3)) to emit before the function.
    globals: Vec<String>,
    /// Guard: the cross-SIMD reduce globals are emitted at most once per kernel.
    tg_reduce_allocated: bool,
    /// Guard: the cross-SIMD scan partials global is emitted at most once.
    tg_scan_allocated: bool,
    /// Guard: the user threadgroup-scratch global is emitted at most once.
    tg_scratch_allocated: bool,
    /// Structured mirror of the threadgroup globals pushed as text into `globals`, so
    /// the bitcode path can declare them instead of re-parsing the declarations.
    /// `count == 1` means a scalar rather than an array.
    tg_globals: Vec<(String, LlType, u32)>,
    /// Size of `@_tg_user_scratch` in floats for THIS kernel (see
    /// `KernelCompiler::tg_scratch_rows`).
    tg_scratch_floats: i32,
    /// Monotonic counter for unique block-label suffixes.
    label_ctr: u32,
    /// Two-pass smem cache per partition view (view.0 -> (alloca, num_chunks)):
    /// a thread-private `[num_chunks x float]` populated in the staging loop and
    /// reused in the consuming loop instead of re-reading global memory.
    smem_cache: HashMap<u32, (CfgValueId, i32, LlType)>,
    /// Which staging phase a cached view is currently in.
    smem_mode: HashMap<u32, StageMode>,
    /// Induction variables of the enclosing For loops (innermost last) - the cache
    /// slot index within a staging/consuming loop.
    loop_iv_stack: Vec<Operand>,
    /// Exit block of each enclosing For (innermost last) + whether that loop
    /// carries values. `Break` jumps to the top exit; break-with-carries bails.
    loop_exit_stack: Vec<(CfgBlockId, bool)>,
}

#[derive(Clone, Copy, PartialEq)]
enum StageMode {
    /// First loop: load from global AND write the cache.
    Staging,
    /// Second loop: read the cache instead of global.
    Consuming,
}

impl Walker<'_> {
    fn setup_params(
        &mut self,
        params: &[(ValRef, Type)],
        dyn_stride_args: &[ValRef],
        dyn_shape_args: &[ValRef],
        const_slots: &std::collections::HashSet<usize>,
    ) -> Option<()> {
        let mut slot = 0usize;
        for (v, ty) in params {
            if param_needs_load(ty) {
                // Scalar tile params (strides/shapes) are not in the signature;
                // the legacy path resolves them to the constant 0.
                self.vmap.insert(*v, vec![Operand::i32(0)]);
            } else {
                let scalar = scalar_lltype(&param_elem(ty))?;
                // Constant-space buffers (marked via KernelCompiler::const_params)
                // bind in addrspace(2); gep/load inherit it, so scalar loads become
                // cheap cached reads. Everything else is a device pointer.
                let ptr_ty = if const_slots.contains(&slot) {
                    LlType::const_ptr(scalar)
                } else {
                    LlType::device_ptr(scalar)
                };
                let id = self.f.param(&format!("%_p{slot}"), ptr_ty);
                self.vmap.insert(*v, vec![Operand::Val(id)]);
                slot += 1;
            }
        }
        // Dynamic stride values come in as i32 constant-buffer (addrspace 2) args,
        // in signature order after the device pointers. Load each at the prologue
        // and bind its block-arg ValRef so stride resolution can reference it.
        for (i, &sv) in dyn_stride_args.iter().enumerate() {
            let arg = self
                .f
                .param(&format!("%_stride_arg_{i}"), LlType::const_ptr(LlType::I32));
            let loaded = self.f.load(Operand::Val(arg), 4);
            self.vmap.insert(sv, vec![Operand::Val(loaded)]);
        }
        // Runtime extents, bound the same way as strides and numbered after them. Only
        // present when a padded store asked for one (see `find_dyn_shape_block_args`).
        self.promoted_dims = dyn_shape_args.iter().copied().collect();
        for (i, &sv) in dyn_shape_args.iter().enumerate() {
            let arg = self
                .f
                .param(&format!("%_dim_arg_{i}"), LlType::const_ptr(LlType::I32));
            let loaded = self.f.load(Operand::Val(arg), 4);
            self.vmap.insert(sv, vec![Operand::Val(loaded)]);
        }
        let vec3 = LlType::vec(3, LlType::I32);
        self.tp = Operand::Val(self.f.param("%_tp", vec3.clone()));
        self.tg = Operand::Val(self.f.param("%_tg", vec3));
        Some(())
    }

    /// Resolve a scalar/single-fragment value to one operand. Returns None for a
    /// multi-fragment grid, so scalar op handlers safely bail on grids.
    fn resolve(&self, v: ValRef) -> Option<Operand> {
        let g = self.vmap.get(&v)?;
        (g.len() == 1).then(|| g[0].clone())
    }

    /// Resolve a value to its full fragment grid (1 element for scalars).
    fn resolve_grid(&self, v: ValRef) -> Option<Vec<Operand>> {
        self.vmap.get(&v).cloned()
    }

    fn result(&self, op_id: OpRef, ri: u32) -> Option<ValRef> {
        *self.ir.op(op_id).results.get(ri as usize)?
    }

    fn bind(&mut self, op_id: OpRef, ri: u32, val: Operand) {
        if let Some(v) = self.result(op_id, ri) {
            self.vmap.insert(v, vec![val]);
        }
    }

    fn bind_grid(&mut self, op_id: OpRef, ri: u32, vals: Vec<Operand>) {
        if let Some(v) = self.result(op_id, ri) {
            self.vmap.insert(v, vals);
        }
    }

    fn translate_block(&mut self, block_id: BlockRef) -> Option<()> {
        // Two-pass caching: read the plan a tile-level pass already decided on. Nothing
        // here judges whether caching is worth it - see `passes::cross_loop_reuse`.
        let plans = super::two_pass_plans(self.ir, block_id);
        let outer_block = self.cur_block.replace(block_id);

        let ops: Vec<OpRef> = self.ir.block(block_id).ops.clone();
        for op_id in ops {
            // Before a staging loop: allocate the thread-private cache and mark
            // the view as staging (its loads will also write the cache).
            if let Some(p) = plans.iter().find(|p| p.staging == op_id) {
                let (view, nc) = (p.view, p.num_chunks);
                let elem = self.pv_elem(view)?;
                let elem_ty = mem_elem_lltype(&elem)?;
                let a = align_of(&elem_ty);
                let cache = self.f.alloca_array(elem_ty.clone(), nc as u32, a)?;
                self.smem_cache.insert(view.0, (cache, nc, elem_ty));
                self.smem_mode.insert(view.0, StageMode::Staging);
            }

            let op = self.ir.op(op_id).clone();
            if self.translate_op(op_id, &op).is_none() {
                cfg_bail(&self.kernel_name, &format!("opcode {:?}", op.opcode));
                return None;
            }

            // After the staging loop the cache is populated: switch to consuming.
            if let Some(p) = plans.iter().find(|p| p.staging == op_id) {
                self.smem_mode.insert(p.view.0, StageMode::Consuming);
            }
            // After the consuming loop, the cache is done.
            if let Some(p) = plans.iter().find(|p| p.consuming == op_id) {
                self.smem_mode.remove(&p.view.0);
            }
        }
        self.cur_block = outer_block;
        Some(())
    }

    fn translate_op(&mut self, op_id: OpRef, op: &OpData) -> Option<()> {
        match op.opcode {
            // No-ops / dead-value producers.
            Opcode::Assert => {}
            Opcode::MakeToken => self.bind(op_id, 0, Operand::undef(LlType::I32)),

            // Constants -> inline immediates. A single 8x8 SGMMA constant is a
            // `<64 x float>` (usually zeroinitializer); a non-SGMMA tile constant
            // is a scalar splat in the SIMD model (each thread holds one element),
            // matching the legacy emitter which takes just the first element.
            // Multi-fragment SGMMA constants are deferred to a later cut.
            Opcode::Constant => {
                let rt = op.result_types.first()?;
                let frag = op
                    .results
                    .first()
                    .and_then(|r| *r)
                    .and_then(|v| self.sgmma_of(v));
                let _ = rt;
                match frag {
                    Some((1, 1)) => self.bind(
                        op_id,
                        0,
                        Operand::constant(constant_literal(op), vec64f(), constant_value(op)),
                    ),
                    Some((nr, nc)) => {
                        // Multi-fragment SGMMA constant - only the zero splat
                        // (accumulator init) is supported; each fragment is a
                        // zeroinitialized <64 x float>.
                        if !constant_is_zero(op) {
                            return None;
                        }
                        let frags = vec![Operand::zeroinit(vec64f()); nr * nc];
                        self.bind_grid(op_id, 0, frags);
                    }
                    None => {
                        let ty = scalar_lltype(&scalar_of(rt))?;
                        self.bind(
                            op_id,
                            0,
                            Operand::constant(constant_literal(op), ty, constant_value(op)),
                        );
                    }
                }
            }

            // Identity forwarders: result aliases operand 0.
            //
            // Broadcast/Reshape are pass-throughs in the scalar-tile SIMD model
            // (each thread already holds its element / the reduced scalar), exactly
            // as the legacy emitter treats them.
            //
            // `MakeStridedView` belongs here now: strided is the ONLY view addressing,
            // so the marker selects nothing and a view is just its base pointer. It is
            // still meaningful to another producer, and `unwrap_strided_view` below keeps
            // DYNAMIC stride tracing working through it.
            //
            // `MakeGatherScatterView` deliberately does NOT belong in this list, though
            // it has the same shape: it carries a tile of gather offsets, so forwarding
            // operand 0 discards them and yields the base pointer, silently reading the
            // wrong elements. Nothing here emits it (a gather is spelled `int_to_ptr` +
            // `addptr_tile` + `load_ptr_tko`), so it falls to the unhandled arm and fails
            // the lowering by name. A module from another producer - a cuTile-authored
            // kernel, or one round-tripped through the CUDA path - can legitimately
            // contain it, and must not compile to a wrong address just because the op
            // looked like a view.
            Opcode::Assume
            | Opcode::MakeTensorView
            | Opcode::MakePartitionView
            | Opcode::MakeStridedView
            | Opcode::Broadcast
            | Opcode::Reshape => {
                let src = self.resolve_grid(*op.operands.first()?)?;
                self.bind_grid(op_id, 0, src);
            }

            // Element-granular view/pointer shift: `result = base + offset`. A
            // tensor or partition view lowers to just its base pointer (see the
            // identity forwarders above), so shifting one is a single GEP in the
            // view's element type.
            // `int_to_ptr`: reinterpret an integer as a base pointer. This is upstream's
            // own route to a base (its `load_tensor` helper does exactly this), and the one
            // a GATHER needs - `addptr_tile` then adds a per-lane offset to it and
            // `load_ptr_tko` reads through the result. Together those three are the pure
            // upstream spelling of what a bespoke gather op used to be needed for.
            Opcode::IntToPtr => {
                let src = self.resolve(*op.operands.first()?)?;
                let elem = mem_elem_lltype(&param_elem(op.result_types.first()?))?;
                let p = self.f.cast("inttoptr", src, LlType::device_ptr(elem));
                self.bind(op_id, 0, Operand::Val(p));
            }
            // `load_ptr_tko`: one address per lane, so a plain load off the thread's own
            // pointer register - the thread-per-element model, no affine lane term.
            Opcode::LoadPtrTko => {
                let ptr = self.resolve(*op.operands.first()?)?;
                let elem = mem_elem_lltype(&param_elem(op.result_types.first()?))?;
                let v = self.f.load(ptr, align_of(&elem));
                self.bind(op_id, 0, Operand::Val(v));
                self.bind(op_id, 1, Operand::undef(LlType::I32)); // token
            }
            Opcode::Offset => {
                let base = self.resolve(*op.operands.first()?)?;
                let off = self.resolve(*op.operands.get(1)?)?;
                let elem = mem_elem_lltype(&param_elem(op.result_types.first()?))?;
                let r = self.f.gep(elem, base, off);
                self.bind(op_id, 0, Operand::Val(r));
            }

            // Thread/block indices.
            Opcode::GetTileBlockId => {
                for axis in 0..3u32 {
                    let idx = self
                        .f
                        .extractelement(self.tg.clone(), Operand::i32(axis as i64));
                    self.bind(op_id, axis, Operand::Val(idx));
                }
            }

            // Element index WITHIN the result tile, so it uses the same
            // extent-derived lane term as a load of that tile (see `lane_term`).
            // It used to be the raw `%_tp.x`, which for a tile narrower than the
            // threadgroup ran past the tile and silently corrupted tail masks -
            // and disagreed with the addresses the tile's own loads computed.
            Opcode::Iota => {
                let n = op
                    .result_types
                    .first()
                    .and_then(tile_elem_count)
                    .unwrap_or(1);
                let lane = match self.lane_term(n) {
                    Some(l) => l,
                    None => Operand::i32(0),
                };
                self.bind(op_id, 0, lane);
            }

            // Integer bounds arithmetic (feeds only Assert; kept faithful).
            Opcode::AddI => self.int_binop(op_id, op, IntOp::Add)?,
            Opcode::SubI => self.int_binop(op_id, op, IntOp::Sub)?,
            Opcode::MulI => self.int_binop(op_id, op, IntOp::Mul)?,
            Opcode::DivI => self.int_binop(op_id, op, IntOp::SDiv)?,
            Opcode::RemI => self.int_binop(op_id, op, IntOp::SRem)?,
            Opcode::AndI => self.int_binop(op_id, op, IntOp::And)?,
            Opcode::OrI => self.int_binop(op_id, op, IntOp::Or)?,
            Opcode::XOrI => self.int_binop(op_id, op, IntOp::Xor)?,
            Opcode::ShLI => self.int_binop(op_id, op, IntOp::Shl)?,
            // `signedness` is 1 for signed, 0 for unsigned (see `signedness_attr`).
            // A missing attribute means signed, which is the DSL's default element
            // signedness and the conservative choice: sign-filling preserves the value
            // of a negative operand, zero-filling silently turns it positive.
            Opcode::ShRI => {
                let signed = super::attr_i32(op, "signedness").unwrap_or(1) != 0;
                let kind = if signed { IntOp::AShr } else { IntOp::LShr };
                self.int_binop(op_id, op, kind)?
            }

            // Tensor/index-space shape queries: each result axis is a static
            // dim as an i32 (`add i32 dim, 0`); GetNumTileBlocks is 1 per axis
            // (mirrors legacy). Dynamic tensor dims bail to legacy.
            Opcode::GetTensorShape => self.shape_query(op_id, op, false)?,
            Opcode::GetIndexSpaceShape => self.shape_query(op_id, op, true)?,
            Opcode::GetNumTileBlocks => {
                for axis in 0..op.result_types.len() as u32 {
                    self.bind(op_id, axis, Operand::i32(1));
                }
            }
            Opcode::CmpI => {
                let a = self.resolve(op.operands[0])?;
                let b = self.resolve(op.operands[1])?;
                if !self.is_i32(&a) || !self.is_i32(&b) {
                    return None;
                }
                let signed = super::attr_i32(op, "signedness") != Some(0);
                let pred = match super::attr_i32(op, "comparison_predicate").unwrap_or(0) {
                    0 => "eq",
                    1 => "ne",
                    2 => {
                        if signed {
                            "slt"
                        } else {
                            "ult"
                        }
                    }
                    3 => {
                        if signed {
                            "sle"
                        } else {
                            "ule"
                        }
                    }
                    4 => {
                        if signed {
                            "sgt"
                        } else {
                            "ugt"
                        }
                    }
                    5 => {
                        if signed {
                            "sge"
                        } else {
                            "uge"
                        }
                    }
                    _ => "eq",
                };
                let r = self.f.icmp(pred, a, b);
                self.bind(op_id, 0, Operand::Val(r));
            }

            // Single-SIMD reduction (n_elems <= 32, f32). The hierarchical
            // cross-SIMD path (n_elems > 32) needs threadgroup globals, barriers,
            // and branches - deferred to a later cut with walker control flow.
            Opcode::Reduce => self.reduce(op_id, op)?,

            // Single-SIMD inclusive prefix scan (n_elems <= 32, f32).
            Opcode::Scan => self.scan(op_id, op)?,

            // Counted loop with scalar carries.
            Opcode::For => self.translate_for(op_id, op)?,

            // if/else producing merged (phi) values.
            Opcode::If => self.translate_if(op_id, op)?,

            // Early loop exit -> branch to the innermost loop's exit block.
            // Break-with-carries isn't supported (the exit phis expect the
            // header carry values, which a mid-body break wouldn't provide).
            Opcode::Break => {
                let (exit, has_carry) = *self.loop_exit_stack.last()?;
                if has_carry {
                    return None;
                }
                self.f.set_terminator(Terminator::Br(exit));
            }

            // Single 8x8 SGMMA matrix-multiply-accumulate (f32).
            Opcode::MmaF => self.mma_f(op_id, op)?,

            // Memory.
            Opcode::LoadViewTko => self.load_view(op_id, op)?,
            Opcode::StoreViewTko => self.store_view(op_id, op)?,

            // Atomics.
            Opcode::AtomicRMW => self.atomic_rmw(op_id, op)?,
            Opcode::AtomicCAS => self.atomic_cas(op_id, op)?,

            // Extract/replace one owned f32 of an 8x8 SGMMA register
            // (sgmma_elem0/1 -> extractelement, sgmma_set_elem0/1 -> insertelement).
            Opcode::Extract => self.sgmma_extract(op_id, op)?,
            Opcode::Pack => self.sgmma_set_elem(op_id, op)?,

            // Math.
            Opcode::AddF => self.float_binop(op_id, op, "fadd")?,
            Opcode::SubF => self.float_binop(op_id, op, "fsub")?,
            Opcode::MulF => self.float_binop(op_id, op, "fmul")?,
            Opcode::DivF => self.float_binop(op_id, op, "fdiv")?,
            Opcode::Atan2 => {
                let (cty, suf, rty) = self.intrin_ty(op.operands[0])?;
                let y = self.resolve(op.operands[0])?;
                let x = self.resolve(op.operands[1])?;
                let y = self.fpext_if(y, &rty, &cty);
                let x = self.fpext_if(x, &rty, &cty);
                let r =
                    self.f
                        .call_fast(cty.clone(), format!("@air.fast_atan2.{suf}"), vec![y, x])?;
                let r = self.fptrunc_if(r, &cty, &rty);
                self.bind(op_id, 0, Operand::Val(r));
            }

            // Unary transcendentals (elementwise, scalar per thread; f32/f16/bf16).
            Opcode::Exp => self.unary_call(op_id, op, "@air.fast_exp")?,
            // `@air.*`, not `@llvm.*`: the AIR intrinsics are the ABI Apple's own
            // frontend emits for precise math. The `@llvm.*` spellings this used to
            // emit are accepted and correct on every toolchain measured, but they are
            // not the documented interface, so they are one toolchain revision away
            // from being a silent-wrong-answer bug.
            Opcode::Log => self.unary_call(op_id, op, "@air.log")?,
            Opcode::Log2 => self.unary_call(op_id, op, "@air.log2")?,
            Opcode::Exp2 => self.unary_call(op_id, op, "@air.exp2")?,
            Opcode::Sin => self.unary_call(op_id, op, "@air.sin")?,
            Opcode::Cos => self.unary_call(op_id, op, "@air.cos")?,
            Opcode::Sqrt => self.unary_call(op_id, op, "@air.sqrt")?,
            Opcode::AbsF => self.unary_call(op_id, op, "@air.fabs")?,

            // rsqrt(x) = 1 / sqrt(x).
            Opcode::Rsqrt => {
                let (cty, suf, rty) = self.intrin_ty(op.operands[0])?;
                let a = self.resolve(op.operands[0])?;
                let a = self.fpext_if(a, &rty, &cty);
                // `air.rsqrt.f32` exists and is what Apple emits for
                // `precise::rsqrt`, but only in f32; 1/sqrt keeps one shape for all
                // three element types.
                let sq = self
                    .f
                    .call(cty.clone(), format!("@air.sqrt.{suf}"), vec![a])?;
                let one = FuncBuilder::float_one(&cty);
                let r = self.f.fdiv(one, Operand::Val(sq));
                let r = self.fptrunc_if(r, &cty, &rty);
                self.bind(op_id, 0, Operand::Val(r));
            }

            // max(a,b): Metal's fast-math max. nan/ftz attrs are ignored.
            Opcode::MaxF => {
                let (cty, suf, rty) = self.intrin_ty(op.operands[0])?;
                let a = self.resolve(op.operands[0])?;
                let b = self.resolve(op.operands[1])?;
                let a = self.fpext_if(a, &rty, &cty);
                let b = self.fpext_if(b, &rty, &cty);
                let r = self
                    .f
                    .call(cty.clone(), format!("@air.fast_fmax.{suf}"), vec![a, b])?;
                let r = self.fptrunc_if(r, &cty, &rty);
                self.bind(op_id, 0, Operand::Val(r));
            }
            // min(a,b) via fcmp + select (type-inferred from the operands).
            Opcode::MinF => {
                self.float_ty(op.operands[0])?; // gate: float element only
                let a = self.resolve(op.operands[0])?;
                let b = self.resolve(op.operands[1])?;
                let cmp = self.f.fcmp("olt", a.clone(), b.clone());
                let r = self.f.select(Operand::Val(cmp), a, b);
                self.bind(op_id, 0, Operand::Val(r));
            }
            // fused multiply-add a*b+c (fmul + fadd; type-inferred from operands).
            Opcode::Fma => {
                self.float_ty(op.operands[0])?; // gate: float element only
                let a = self.resolve(op.operands[0])?;
                let b = self.resolve(op.operands[1])?;
                let c = self.resolve(op.operands[2])?;
                let m = self.f.fmul(a, b);
                let r = self.f.fadd(Operand::Val(m), c);
                self.bind(op_id, 0, Operand::Val(r));
            }

            // Float negation: `fneg`, preserving scalar-or-vector type.
            Opcode::NegF => {
                self.float_ty(op.operands[0])?; // gate: float element
                let a = self.resolve(op.operands[0])?;
                let r = self.f.fneg(a);
                self.bind(op_id, 0, Operand::Val(r));
            }
            // Integer negation: `0 - x`.
            Opcode::NegI => {
                let a = self.resolve(op.operands[0])?;
                if !self.is_int(&a) {
                    return None;
                }
                let r = self.f.sub(Operand::i32(0), a);
                self.bind(op_id, 0, Operand::Val(r));
            }
            // Float compare -> i1 (or <N x i1> for tiles); ordered predicates.
            Opcode::CmpF => {
                self.float_ty(op.operands[0])?;
                let a = self.resolve(op.operands[0])?;
                let b = self.resolve(op.operands[1])?;
                let pred = match super::attr_i32(op, "comparison_predicate").unwrap_or(0) {
                    0 => "oeq",
                    1 => "one",
                    2 => "olt",
                    3 => "ole",
                    4 => "ogt",
                    5 => "oge",
                    _ => "oeq",
                };
                let r = self.f.fcmp(pred, a, b);
                self.bind(op_id, 0, Operand::Val(r));
            }
            // select(cond, t, f); cond is i1 or a <N x i1> vector mask.
            Opcode::Select => {
                let cond = self.resolve(op.operands[0])?;
                let t = self.resolve(op.operands[1])?;
                let f = self.resolve(op.operands[2])?;
                let r = self.f.select(cond, t, f);
                self.bind(op_id, 0, Operand::Val(r));
            }
            // Integer max/min via icmp + select (signed).
            Opcode::MaxI | Opcode::MinI => {
                let a = self.resolve(op.operands[0])?;
                let b = self.resolve(op.operands[1])?;
                if !self.is_i32(&a) || !self.is_i32(&b) {
                    return None;
                }
                let pred = if matches!(op.opcode, Opcode::MaxI) {
                    "sgt"
                } else {
                    "slt"
                };
                let cmp = self.f.icmp(pred, a.clone(), b.clone());
                let r = self.f.select(Operand::Val(cmp), a, b);
                self.bind(op_id, 0, Operand::Val(r));
            }

            // Scalar numeric conversions (int<->float, widen/narrow). The LLVM
            // instruction is inferred from source/dest types (signed), matching
            // the legacy emitter.
            Opcode::IToF
            | Opcode::FToI
            | Opcode::FToF
            | Opcode::ExtI
            | Opcode::TruncI
            | Opcode::Bitcast => self.cast_op(op_id, op)?,

            Opcode::Return => {
                self.f.set_terminator(Terminator::Ret);
            }

            // Anything else: bail to the legacy emitter.
            _ => return None,
        }
        Some(())
    }

    fn int_binop(&mut self, op_id: OpRef, op: &OpData, kind: IntOp) -> Option<()> {
        let a = self.resolve(op.operands[0])?;
        let b = self.resolve(op.operands[1])?;
        // Bitwise/logical ops also apply to i1 (bool `&&`/`||` lower to AndI/OrI);
        // arithmetic ops are i32-only.
        let ok = match kind {
            IntOp::And | IntOp::Or | IntOp::Xor => self.is_int(&a) && self.is_int(&b),
            // Shifts are defined on the integer element types, not on i1.
            IntOp::Shl | IntOp::LShr | IntOp::AShr => self.is_i32(&a) && self.is_i32(&b),
            _ => self.is_i32(&a) && self.is_i32(&b),
        };
        if !ok {
            return None;
        }
        let r = match kind {
            IntOp::Add => self.f.add(a, b),
            IntOp::Sub => self.f.sub(a, b),
            IntOp::Mul => self.f.mul(a, b),
            IntOp::SDiv => self.f.sdiv(a, b),
            IntOp::SRem => self.f.srem(a, b),
            IntOp::And => self.f.and(a, b),
            IntOp::Or => self.f.or(a, b),
            IntOp::Xor => self.f.xor(a, b),
            IntOp::Shl => self.f.shl(a, b),
            IntOp::LShr => self.f.lshr(a, b),
            IntOp::AShr => self.f.ashr(a, b),
        };
        self.bind(op_id, 0, Operand::Val(r));
        Some(())
    }

    /// Shape query: bind each result axis to its static dim as an i32 constant.
    /// `index_space` reads a PartitionView's tile_shape (always static);
    /// otherwise a TensorView's shape (bails to legacy on a dynamic axis).
    fn shape_query(&mut self, op_id: OpRef, op: &OpData, index_space: bool) -> Option<()> {
        let ty = self.ir.value_type(op.operands[0]);
        for ri in 0..op.result_types.len() as u32 {
            let dim: i64 = if index_space {
                match ty {
                    Type::PartitionView(pv) => *pv.tile_shape.get(ri as usize)? as i64,
                    _ => return None,
                }
            } else {
                match ty {
                    Type::TensorView(tv) => *tv.shape.get(ri as usize)?,
                    _ => return None,
                }
            };
            if dim == wax_ir::DYNAMIC {
                return None; // dynamic tensor-shape query - stays on legacy
            }
            self.bind(op_id, ri, Operand::i32(dim));
        }
        Some(())
    }

    /// Unary float intrinsic `dst = call {ty} {base}.{suf}({ty} src)`, where the
    /// element type (f32/f16/bf16) picks the suffix and LLVM type. `base` is the
    /// intrinsic name without the type suffix, e.g. `@air.fast_exp` / `@air.log`.
    fn unary_call(&mut self, op_id: OpRef, op: &OpData, base: &str) -> Option<()> {
        let (cty, suf, rty) = self.intrin_ty(op.operands[0])?;
        let a = self.resolve(op.operands[0])?;
        let a = self.fpext_if(a, &rty, &cty);
        let r = self.f.call(cty.clone(), format!("{base}.{suf}"), vec![a])?;
        let r = self.fptrunc_if(r, &cty, &rty);
        self.bind(op_id, 0, Operand::Val(r));
        Some(())
    }

    /// LLVM float type + AIR intrinsic suffix for a float value's element type.
    /// None for non-float types, so callers bail (-> error) on unsupported ops.
    fn float_ty(&self, v: ValRef) -> Option<(LlType, &'static str)> {
        float_elem(&scalar_of(self.ir.value_type(v)))
    }

    /// `(compute_ty, compute_suffix, result_ty)` for a SIMD/fast-math intrinsic
    /// op. f32/f16 compute natively; bf16 computes in f32 (no bf16 intrinsics on
    /// Apple GPUs) and the result is truncated back to bf16.
    fn intrin_ty(&self, v: ValRef) -> Option<(LlType, &'static str, LlType)> {
        let scalar = scalar_of(self.ir.value_type(v));
        let (rty, _) = float_elem(&scalar)?;
        let (cty, csuf) = match scalar {
            ScalarType::BF16 => (LlType::Float, "f32"),
            other => float_elem(&other)?,
        };
        Some((cty, csuf, rty))
    }

    /// `fpext v` from `from` to `to` if they differ (bf16 -> f32); else identity.
    fn fpext_if(&mut self, v: Operand, from: &LlType, to: &LlType) -> Operand {
        if from == to {
            v
        } else {
            Operand::Val(self.f.cast("fpext", v, to.clone()))
        }
    }

    /// `fptrunc v` from `from` to `to` if they differ (f32 -> bf16); else identity.
    fn fptrunc_if(&mut self, v: CfgValueId, from: &LlType, to: &LlType) -> CfgValueId {
        if from == to {
            v
        } else {
            self.f.cast("fptrunc", Operand::Val(v), to.clone())
        }
    }

    /// Scalar numeric conversion. Infers the LLVM cast from source/dest scalar
    /// types (signed), like the legacy emitter; an identity cast just forwards.
    /// The LLVM cast for a float->float conversion, or `None` if either side is not a
    /// float (fragments are always float, so nothing else can reach the caller).
    fn float_cast_inst(from: &LlType, to: &LlType) -> Option<&'static str> {
        if !from.is_float() || !to.is_float() {
            return None;
        }
        Some(if lltype_bits(from) > lltype_bits(to) {
            "fptrunc"
        } else {
            "fpext"
        })
    }

    fn cast_op(&mut self, op_id: OpRef, op: &OpData) -> Option<()> {
        let from = scalar_lltype(&scalar_of(self.ir.value_type(op.operands[0])))?;
        let to = scalar_lltype(&scalar_of(op.result_types.first()?))?;

        // An SGMMA fragment is a GRID of `<64 x elem>` registers, so `resolve` - which
        // only answers for single-operand values - returns None for it and the whole
        // kernel used to bail. Cast each fragment instead: LLVM's float casts are
        // elementwise on vectors, so one instruction per fragment does it.
        //
        // Without this, narrowing an f32 accumulator to an f16/bf16 output could only be
        // said with a dedicated narrowing STORE, because `convert_tile` could not be
        // lowered on a fragment at all.
        if self.sgmma_of(op.operands[0]).is_some() {
            let grid = self.resolve_grid(op.operands[0])?;
            if from == to {
                self.bind_grid(op_id, 0, grid);
                return Some(());
            }
            let inst = Self::float_cast_inst(&from, &to)?;
            let vec_to = LlType::vec(64, to.clone());
            let out: Vec<Operand> = grid
                .into_iter()
                .map(|frag| Operand::Val(self.f.cast(inst, frag, vec_to.clone())))
                .collect();
            self.bind_grid(op_id, 0, out);
            return Some(());
        }

        let src = self.resolve(op.operands[0])?;
        if from == to {
            self.bind(op_id, 0, src); // identity
            return Some(());
        }
        let inst = match (from.is_float(), to.is_float()) {
            (true, true) => {
                if lltype_bits(&from) > lltype_bits(&to) {
                    "fptrunc"
                } else {
                    "fpext"
                }
            }
            (true, false) => "fptosi",
            (false, true) => "sitofp",
            (false, false) => {
                if lltype_bits(&from) < lltype_bits(&to) {
                    // Widen: sign-extend signed sources, zero-extend unsigned
                    // (ExtI carries signedness; 0 = unsigned). Matters for u8
                    // masks / u32 ids whose high bit is set.
                    if super::attr_i32(op, "signedness") == Some(0) {
                        "zext"
                    } else {
                        "sext"
                    }
                } else {
                    "trunc"
                }
            }
        };
        let r = self.f.cast(inst, src, to.clone());
        self.bind(op_id, 0, Operand::Val(r));
        Some(())
    }

    fn float_binop(&mut self, op_id: OpRef, op: &OpData, inst: &'static str) -> Option<()> {
        // Vector-element tile (V128<E>) or scalar: the operands normally already carry
        // the right type - `<width x elem>` from the wide load, or the compute type of
        // an External dtype (e.g. f16 for fp8) - so the typed builders infer it.
        let a = self.resolve(op.operands[0])?;
        let b = self.resolve(op.operands[1])?;
        // Mixed widths do occur: the f16/bf16 SDPA kernels scale a half tile by an f32
        // scalar. Promote to the wider type, compute there, and narrow back to the
        // result type - the same shape the intrinsic and reduce paths use. Without this
        // the builder's type assertion fires and the kernel cannot lower at all.
        let (ta, tb) = (self.f.type_of(&a), self.f.type_of(&b));
        let (a, b, cty) = if ta == tb {
            (a, b, ta)
        } else {
            let cty = wider_float(&ta, &tb)?;
            let a = self.fpext_if(a, &ta, &cty);
            let b = self.fpext_if(b, &tb, &cty);
            (a, b, cty)
        };
        let r = match inst {
            "fadd" => self.f.fadd(a, b),
            "fmul" => self.f.fmul(a, b),
            "fsub" => self.f.fsub(a, b),
            "fdiv" => self.f.fdiv(a, b),
            // Only the four callers above exist; bail rather than emit a Raw the
            // bitcode writer cannot encode.
            other => {
                cfg_bail(&self.kernel_name, &format!("float binop {other}"));
                return None;
            }
        };
        let rty = self.binop_result_ty(op).unwrap_or_else(|| cty.clone());
        let r = self.fptrunc_if(r, &cty, &rty);
        self.bind(op_id, 0, Operand::Val(r));
        Some(())
    }

    /// The LLVM type this float binop's result should have: `<width x elem>` for a
    /// vector-element tile, else the result scalar's type. `None` when the result type
    /// has no direct LLVM spelling (e.g. an External dtype computed in a wider type),
    /// in which case the compute type stands.
    fn binop_result_ty(&self, op: &OpData) -> Option<LlType> {
        let s = scalar_of(op.result_types.first()?);
        match s {
            ScalarType::External(_) => None,
            _ => scalar_lltype(&s),
        }
    }

    /// Single-SIMD reduction: `dst = @air.simd_{sum,max,min,product}.{suf}(src)`
    /// for f32/f16/bf16. Bails (-> error) for other scalar types; SGMMA tiles use
    /// the row-reduce path. n_elems > 32 goes through the hierarchical path.
    fn reduce(&mut self, op_id: OpRef, op: &OpData) -> Option<()> {
        let src_ty = self.ir.value_type(op.operands[0]);

        // A COLUMN reduce (dim 0) of a threadgroup-spanning `[rows, 32]` tile. This is
        // tested before the SGMMA classifier below because that one claims any 2-D tile
        // whose extents are both multiples of 8 - `[32, 32]` included - and would route
        // this to a per-ROW reduce instead. An SGMMA row reduce is inherently dim 1, so
        // dim 0 is unambiguous.
        let column_reduce = matches!(
            src_ty,
            Type::Tile(tt) if tt.shape.len() == 2 && tt.shape[1] == 32 && tt.shape[0] > 1
        ) && super::attr_i32(op, "dim") == Some(0);

        // A ROW reduce (dim 1) of a threadgroup-spanning `[rows, 32]` tile. Each row IS one
        // SIMD group, so the plain SIMD intrinsic already gives every lane of row s the
        // reduction over row s - no hierarchical step, and none wanted. Without this the
        // `n_elems > 32` test below routes it to `reduce_hierarchical`, which collapses the
        // WHOLE TILE across the threadgroup: the right answer for a full-tile reduce and
        // silently wrong for a per-row one. It went unnoticed because at `[1, 32]` - the
        // only shape any kernel used - the two coincide.
        let row_reduce = matches!(
            src_ty,
            Type::Tile(tt) if tt.shape.len() == 2 && tt.shape[1] == 32 && tt.shape[0] > 1
        ) && super::attr_i32(op, "dim") == Some(1);

        if !column_reduce && self.sgmma_of(op.operands[0]).is_some() {
            return self.sgmma_row_reduce(op_id, op); // per-row reduce of an SGMMA tile
        }
        let (cty, suf, rty) = self.intrin_ty(op.operands[0])?;
        let n_elems: i64 = match src_ty {
            Type::Tile(tt) => tt.shape.iter().product(),
            _ => 1,
        };
        let kind = self.reduce_intrinsic(op);
        let src = self.resolve(op.operands[0])?;
        let src = self.fpext_if(src, &rty, &cty); // bf16 reduces in f32

        // One result per LANE, replicated across SIMD groups - which is exactly the
        // `[32]` result tile's own layout. This is the shape the SDPA merge needs, and
        // it is what `tg_store` + `tg_load_transposed` + `reduce` spelled by hand.
        if column_reduce {
            let rows = match src_ty {
                Type::Tile(tt) => tt.shape[0] as i32,
                _ => return None,
            };
            if let Some(v) = self.batched_reduces.remove(&op_id) {
                self.bind(op_id, 0, v);
                return Some(());
            }
            if self.batch_column_reduces(op_id, rows).is_some() {
                return Some(());
            }
            return self.reduce_columns(op_id, src, kind, cty, suf, rty, rows);
        }

        if n_elems > 32 && !row_reduce {
            return self.reduce_hierarchical(op_id, src, kind, cty, suf, rty);
        }
        let intrinsic = simd_intrinsic(kind, suf);
        let r = self.f.call_fast(cty.clone(), intrinsic, vec![src])?;
        let r = self.fptrunc_if(r, &cty, &rty);
        self.bind(op_id, 0, Operand::Val(r));
        Some(())
    }

    /// Per-row reduction of an 8x8 SGMMA tile (`reduce_max/min/sum` over dim 1).
    /// Each lane owns elements 0 and 1 of its row; combine those within-lane, then
    /// across the 4 lanes holding the same row via shuffle_xor masks 1 and 8.
    /// Mirrors legacy `emit_sgmma_row_reduce`. Result is one scalar per lane.
    fn sgmma_row_reduce(&mut self, op_id: OpRef, op: &OpData) -> Option<()> {
        let src = op.operands[0];
        let (_, nc) = self.sgmma_of(src)?;
        let intrinsic = self.reduce_intrinsic(op);
        let grid = self.resolve_grid(src)?; // nrxnc row-major; row 0 = indices 0..nc

        // Within-lane: combine both owned elements of each row-0 sub-tile, then
        // accumulate across the nc sub-tiles.
        // SGMMA fragments are always f32 registers.
        let (ft, fs) = (LlType::Float, "f32");
        let mut partial: Option<Operand> = None;
        for c in 0..nc {
            let tile = grid.get(c)?.clone();
            let e0 = self.f.extractelement(tile.clone(), Operand::i32(0));
            let e1 = self.f.extractelement(tile, Operand::i32(1));
            let r01 = self.scan_binop(Operand::Val(e0), Operand::Val(e1), intrinsic, &ft, fs)?;
            partial = Some(match partial {
                None => Operand::Val(r01),
                Some(p) => {
                    Operand::Val(self.scan_binop(p, Operand::Val(r01), intrinsic, &ft, fs)?)
                }
            });
        }
        let partial = partial?;

        // Cross-lane: combine the 4 SIMD lanes holding the same row (xor 1 and 8).
        let rx1 = self.f.call(
            LlType::Float,
            "@air.simd_shuffle_xor.f32",
            vec![partial.clone(), Operand::i16(1)],
        )?;
        let ra1 = self.scan_binop(partial, Operand::Val(rx1), intrinsic, &ft, fs)?;
        let rx8 = self.f.call(
            LlType::Float,
            "@air.simd_shuffle_xor.f32",
            vec![Operand::Val(ra1), Operand::i16(8)],
        )?;
        let res = self.scan_binop(Operand::Val(ra1), Operand::Val(rx8), intrinsic, &ft, fs)?;
        self.bind(op_id, 0, Operand::Val(res));
        Some(())
    }

    /// Hierarchical cross-SIMD f32 reduction (n_elems > 32), mirroring the legacy
    /// emitter: within-SIMD reduce, then two barrier-separated threadgroup-memory
    /// phases, then a broadcast load. Emits a diamond of CFG blocks (no phis -
    /// values cross blocks through threadgroup memory). Leaves the final block
    /// current so subsequent ops continue after it.
    fn reduce_hierarchical(
        &mut self,
        op_id: OpRef,
        src: Operand,
        kind: &str,
        ty: LlType,
        suf: &str,
        rty: LlType,
    ) -> Option<()> {
        let lls = ty.ll_str();
        let al = align_of(&ty);
        // Threadgroup globals are typed by the reduce element (one type per kernel;
        // the fixed names assume a kernel doesn't mix f32/f16/bf16 reductions).
        if !self.tg_reduce_allocated {
            self.globals.push(format!(
                "@_tg_reduce_partial = internal unnamed_addr addrspace(3) global [32 x {lls}] undef, align {al}"
            ));
            self.globals.push(format!(
                "@_tg_reduce_result  = internal unnamed_addr addrspace(3) global {lls} undef, align {al}"
            ));
            self.tg_globals
                .push(("@_tg_reduce_partial".into(), ty.clone(), 32));
            self.tg_globals
                .push(("@_tg_reduce_result".into(), ty.clone(), 1));
            self.tg_reduce_allocated = true;
        }
        let n = self.label_ctr;
        self.label_ctr += 1;
        let identity = reduce_identity(kind, suf);
        let intrinsic = simd_intrinsic(kind, suf);
        let ty2 = ty.clone();
        let result_ptr = move || Operand::global("@_tg_reduce_result", shared_ptr(ty2.clone()));

        // Current block: within-SIMD reduce + simd group/lane indices.
        let s1 = self.f.call_fast(ty.clone(), intrinsic.clone(), vec![src])?;
        let tid = self.f.extractelement(self.tp.clone(), Operand::i32(0));
        let grp = self.f.udiv(Operand::Val(tid), Operand::i32(32));
        let lane = self.f.urem(Operand::Val(tid), Operand::i32(32));

        // Pre-create every block so terminators can reference them freely.
        let init = self.f.block(&format!("_rs_init_{n}"));
        let skip_init = self.f.block(&format!("_rs_skip_init_{n}"));
        let after_init = self.f.block(&format!("_rs_after_init_{n}"));
        let write = self.f.block(&format!("_rs_write_{n}"));
        let skip_write = self.f.block(&format!("_rs_skip_write_{n}"));
        let after_write = self.f.block(&format!("_rs_after_write_{n}"));
        let final_b = self.f.block(&format!("_rs_final_{n}"));
        let write_res = self.f.block(&format!("_rs_write_res_{n}"));
        let skip_res = self.f.block(&format!("_rs_skip_res_{n}"));
        let after_res = self.f.block(&format!("_rs_after_res_{n}"));
        let skip_final = self.f.block(&format!("_rs_skip_final_{n}"));
        let after_final = self.f.block(&format!("_rs_after_final_{n}"));

        // Phase 0: SIMD group 0 initialises all 32 partial slots to identity.
        let is_grp0_a = self.f.icmp("eq", Operand::Val(grp), Operand::i32(0));
        self.f.set_terminator(Terminator::CondBr {
            cond: Operand::Val(is_grp0_a),
            then_: init,
            else_: skip_init,
        });
        self.f.switch_to(init);
        let init_gep = self.partial_gep(Operand::Val(lane), ty.clone())?;
        self.f.store(
            Operand::float_bits(identity.0, ty.clone(), identity.1),
            Operand::Val(init_gep),
            al,
        );
        self.f.set_terminator(Terminator::Br(after_init));
        self.f.switch_to(skip_init);
        self.f.set_terminator(Terminator::Br(after_init));

        // Phase 1: lane 0 of each SIMD group writes its partial result.
        self.f.switch_to(after_init);
        self.barrier();
        let is_lane0_a = self.f.icmp("eq", Operand::Val(lane), Operand::i32(0));
        self.f.set_terminator(Terminator::CondBr {
            cond: Operand::Val(is_lane0_a),
            then_: write,
            else_: skip_write,
        });
        self.f.switch_to(write);
        let write_gep = self.partial_gep(Operand::Val(grp), ty.clone())?;
        self.f.store(Operand::Val(s1), Operand::Val(write_gep), al);
        self.f.set_terminator(Terminator::Br(after_write));
        self.f.switch_to(skip_write);
        self.f.set_terminator(Terminator::Br(after_write));

        // Phase 2: SIMD group 0 reduces the partials; lane 0 writes the result.
        self.f.switch_to(after_write);
        self.barrier();
        let is_grp0_b = self.f.icmp("eq", Operand::Val(grp), Operand::i32(0));
        self.f.set_terminator(Terminator::CondBr {
            cond: Operand::Val(is_grp0_b),
            then_: final_b,
            else_: skip_final,
        });
        self.f.switch_to(final_b);
        let load_gep = self.partial_gep(Operand::Val(lane), ty.clone())?;
        let partial_i = self.f.load(Operand::Val(load_gep), al);
        let total = self
            .f
            .call_fast(ty.clone(), intrinsic, vec![Operand::Val(partial_i)])?;
        let is_lane0_b = self.f.icmp("eq", Operand::Val(lane), Operand::i32(0));
        self.f.set_terminator(Terminator::CondBr {
            cond: Operand::Val(is_lane0_b),
            then_: write_res,
            else_: skip_res,
        });
        self.f.switch_to(write_res);
        self.f.store(Operand::Val(total), result_ptr(), al);
        self.f.set_terminator(Terminator::Br(after_res));
        self.f.switch_to(skip_res);
        self.f.set_terminator(Terminator::Br(after_res));
        self.f.switch_to(after_res);
        self.f.set_terminator(Terminator::Br(after_final));
        self.f.switch_to(skip_final);
        self.f.set_terminator(Terminator::Br(after_final));

        // All threads read the broadcast result.
        self.f.switch_to(after_final);
        self.barrier();
        let result = self.f.load(result_ptr(), al);
        let result = self.fptrunc_if(result, &ty, &rty); // f32 -> bf16 if promoted
        self.bind(op_id, 0, Operand::Val(result));
        Some(())
    }

    /// Cross-SIMD COLUMN reduce of a `[rows, 32]` tile: `out[c] = combine over r of
    /// src[r][c]`, one value per lane, replicated across SIMD groups.
    ///
    /// Each thread writes its own element to scratch, then every thread walks its
    /// column. The row pitch is PADDED TO AN ODD value (`tg_row_pitch`): threadgroup
    /// memory is banked every 4 bytes over 32 banks, so at a pitch of 32 a column walk
    /// puts all 32 lanes of a group on one bank and serializes 32 ways. That cost
    /// dominated the SDPA merge before it was padded.
    ///
    /// This is the primitive `tg_store` + `tg_load_transposed` + `reduce` expressed by
    /// hand; having it here means the kernel can say `reduce_sum(t, 0)` instead, which
    /// is upstream cuTile's own spelling and lowers on any backend.
    /// Wait for readers of the shared scratch before overwriting it.
    ///
    /// Every column reduce ends by READING scratch, and the next one starts by WRITING
    /// it, so back-to-back reduces race unless separated. The first writer in a kernel
    /// has nobody to wait for, which is why this is tracked rather than unconditional.
    fn tg_scratch_war_barrier(&mut self) {
        if self.tg_scratch_dirty {
            self.barrier();
        }
        self.tg_scratch_dirty = true;
    }

    /// The `rows`, if `op` is a column reduce (`dim = 0` over a `[rows, 32]` tile).
    fn column_reduce_rows(&self, op: &OpData) -> Option<i32> {
        if op.opcode != Opcode::Reduce {
            return None;
        }
        let src = *op.operands.first()?;
        let rows = match self.ir.value_type(src) {
            Type::Tile(tt) if tt.shape.len() == 2 && tt.shape[1] == 32 && tt.shape[0] > 1 => {
                tt.shape[0] as i32
            }
            _ => return None,
        };
        (super::attr_i32(op, "dim") == Some(0)).then_some(rows)
    }

    /// Lower a run of adjacent independent column reduces together, sharing one set of
    /// barriers instead of paying three each.
    ///
    /// Each column reduce is store / barrier / transposed-read-and-reduce / barrier /
    /// redistribute / barrier / read. Those barriers are threadgroup-wide, so N of them
    /// back to back cost 3N even though the reduces touch disjoint scratch and do not
    /// depend on each other. Giving each member its own scratch region lets all N share
    /// three barriers - which is precisely what the hand-rolled `tg_store` merge in the
    /// decode SDPA does by writing all four components before its single barrier.
    ///
    /// Only members whose source is ALREADY lowered can join: a later sibling's source is
    /// computed by ops this forward walk has not reached yet, and hoisting the reduce
    /// above them would reorder the program. In the kernels this targets the sources are
    /// all computed before the first reduce, so the run is found intact.
    fn batch_column_reduces(&mut self, op_id: OpRef, rows: i32) -> Option<()> {
        if rows != 32 {
            return None; // only the transposed fast path has barriers worth sharing
        }
        let block = self.cur_block?;
        let ops = &self.ir.block(block).ops;
        let start = ops.iter().position(|&o| o == op_id)?;

        // A maximal run of consecutive column reduces, same shape, sources in hand, none
        // of them lane-invariant (those skip the redistribution and are cheap already).
        let mut members: Vec<(OpRef, Operand, LlType, &'static str, String, LlType)> = Vec::new();
        for &o in &ops[start..] {
            let op = self.ir.op(o);
            if self.column_reduce_rows(op) != Some(rows) {
                break;
            }
            let src_v = *op.operands.first()?;
            if self.is_sg_uniform(src_v) {
                break;
            }
            let Some(src) = self.resolve(src_v) else {
                break;
            };
            let Some((cty, suf, rty)) = self.intrin_ty(src_v) else {
                break;
            };
            let kind = self.reduce_intrinsic(self.ir.op(o));
            members.push((o, src, cty, kind, suf.to_string(), rty));
        }
        if members.len() < 2 {
            return None;
        }
        let pitch = tg_row_pitch(32) as i64;
        let cap = self.tg_scratch_floats as i64;
        // Phase 1 needs a full [group][lane] block per member to transpose through.
        let per_member = rows as i64 * pitch;
        let chunk = (cap / per_member).max(1) as usize;
        // The shared redistribution needs one float per (member, group) - see below.
        if members.len() as i64 * 32 > cap {
            return None;
        }

        self.ensure_tg_scratch();
        self.tg_scratch_war_barrier();
        let tid = self.f.extractelement(self.tp.clone(), Operand::i32(0));
        let grp = self.f.udiv(Operand::Val(tid), Operand::i32(32));
        let lane = self.f.and(Operand::Val(tid), Operand::i32(31));

        // Phases 1 and 2, in as few passes as the scratch allows. Each member writes
        // [row = its SIMD group][col = its lane], then reads the TRANSPOSE so the
        // cross-group reduction becomes a within-group one - the SIMD reduce intrinsic,
        // log depth, in registers.
        let mut per_group = Vec::with_capacity(members.len());
        for (ci, sub) in members.chunks(chunk).enumerate() {
            if ci > 0 {
                self.barrier(); // previous pass must finish reading before we overwrite
            }
            for (j, (_, src, cty, _, _, _)) in sub.iter().enumerate() {
                let region = Operand::i32(j as i64 * per_member);
                let base = self.f.mul(Operand::Val(grp), Operand::i32(pitch));
                let flat = self.f.add(Operand::Val(base), Operand::Val(lane));
                let flat = self.f.add(Operand::Val(flat), region);
                let gep = self.tg_scratch_ptr(Operand::Val(flat), "batched reduce store");
                self.f.store(src.clone(), Operand::Val(gep), align_of(cty));
            }
            self.barrier();
            for (j, (_, _, cty, kind, suf, _)) in sub.iter().enumerate() {
                let region = Operand::i32(j as i64 * per_member);
                let row_off = self.f.mul(Operand::Val(lane), Operand::i32(pitch));
                let flat_t = self.f.add(Operand::Val(row_off), Operand::Val(grp));
                let flat_t = self.f.add(Operand::Val(flat_t), region);
                let gep_t = self.tg_scratch_ptr(Operand::Val(flat_t), "batched reduce transposed");
                let v = self.f.load(Operand::Val(gep_t), align_of(cty));
                let intrinsic = simd_intrinsic(kind, suf);
                per_group.push(
                    self.f
                        .call_fast(cty.clone(), intrinsic, vec![Operand::Val(v)])?,
                );
            }
        }

        // One redistribution for every member, COMPACT: the value each group holds is
        // SG-uniform, so it needs one float per (member, group), not the full row per
        // group the per-member spelling wrote. That is what lets a single round trip
        // serve all members no matter how few fit in phase 1 - which is the difference
        // between three barriers and three per member. It overlays the phase-1 blocks,
        // safe because the barrier above has retired their reads.
        //
        // Every lane of a group stores the same value to the same slot. That is a race
        // only in the sense that the winner is unspecified; all writers agree on the
        // value, so the result does not depend on which one lands.
        self.barrier();
        for (i, (_, _, cty, _, _, _)) in members.iter().enumerate() {
            let slot = self.f.add(Operand::Val(grp), Operand::i32(i as i64 * 32));
            let gep_w = self.tg_scratch_ptr(Operand::Val(slot), "batched reduce redistribute");
            self.f.store(
                Operand::Val(per_group[i]),
                Operand::Val(gep_w),
                align_of(cty),
            );
        }
        self.barrier();
        for (i, (o, _, cty, _, _, rty)) in members.iter().enumerate() {
            let slot = self.f.add(Operand::Val(lane), Operand::i32(i as i64 * 32));
            let gep_r = self.tg_scratch_ptr(Operand::Val(slot), "batched reduce broadcast");
            let out = self.f.load(Operand::Val(gep_r), align_of(cty));
            let out = self.fptrunc_if(out, cty, rty);
            if *o == op_id {
                self.bind(op_id, 0, Operand::Val(out));
            } else {
                self.batched_reduces.insert(*o, Operand::Val(out));
            }
        }
        Some(())
    }

    fn reduce_columns(
        &mut self,
        op_id: OpRef,
        src: Operand,
        kind: &str,
        ty: LlType,
        suf: &str,
        rty: LlType,
        rows: i32,
    ) -> Option<()> {
        self.ensure_tg_scratch();
        self.tg_scratch_war_barrier();
        let al = align_of(&ty);
        let pitch = tg_row_pitch(32) as i64;

        let tid = self.f.extractelement(self.tp.clone(), Operand::i32(0));
        let grp = self.f.udiv(Operand::Val(tid), Operand::i32(32));
        let lane = self.f.and(Operand::Val(tid), Operand::i32(31));

        // Write this thread's element at [row = its SIMD group][col = its lane].
        let base = self.f.mul(Operand::Val(grp), Operand::i32(pitch));
        let flat = self.f.add(Operand::Val(base), Operand::Val(lane));
        let gep = self.tg_scratch_ptr(Operand::Val(flat), "reduce_columns store");
        self.f.store(src, Operand::Val(gep), al);
        self.barrier();

        // With one row per SIMD group the transpose lines up: read the element at
        // [lane][sg] instead of [sg][lane] and the cross-group reduction becomes a
        // WITHIN-group one, which is the SIMD reduce intrinsic - log depth, in
        // registers. Walking the column with `rows` serial threadgroup loads instead
        // measured ~35% slower than the hand-rolled spelling that does this.
        //
        // That leaves the value per SIMD GROUP (group `c` holds column `c`), while a
        // `[C]` result tile is per LANE, so a second transposed round trip through the
        // same scratch redistributes it. Every lane of a group holds the same value at
        // that point, so writing each to its own slot avoids a guard.
        //
        // The redistribution below is skipped entirely when the source is lane-invariant
        // (see `uniformity`); the branch that does it is in the `rows == 32` path.
        // Remaining uses that analysis could still pay for, none of them here:
        //   - a `broadcast_scalar` of an SG-uniform value is a no-op, not a splat;
        //   - a branch on an SG-uniform condition needs no divergence handling;
        //   - a store guarded by `lane == 0` of an SG-uniform value can become one
        //     uniform store with no guard block at all;
        //   - a barrier whose only purpose is publishing an SG-uniform value to its own
        //     group is redundant.
        if rows == 32 {
            let row_off = self.f.mul(Operand::Val(lane), Operand::i32(pitch));
            let flat_t = self.f.add(Operand::Val(row_off), Operand::Val(grp));
            let gep_t = self.tg_scratch_ptr(Operand::Val(flat_t), "reduce_columns transposed");
            let v = self.f.load(Operand::Val(gep_t), al);
            let intrinsic = simd_intrinsic(kind, suf);
            let per_group = self
                .f
                .call_fast(ty.clone(), intrinsic, vec![Operand::Val(v)])?;

            // When the source is constant along the lane dim, every column reduces to the
            // same answer, so the value this group just produced IS the answer for every
            // lane - the round trip below would redistribute a value to itself. This is
            // the decode SDPA's m/l merge exactly: one per-group scalar broadcast across
            // lanes, which is the shape any cross-group reduction of a per-group quantity
            // takes.
            // Source constant along the lane dim => every column reduces alike.
            if self
                .ir
                .op(op_id)
                .operands
                .first()
                .is_some_and(|&src| self.is_sg_uniform(src))
            {
                let out = self.fptrunc_if(per_group, &ty, &rty);
                self.bind(op_id, 0, Operand::Val(out));
                return Some(());
            }

            self.barrier();
            let gep_w = self.tg_scratch_ptr(Operand::Val(flat), "reduce_columns redistribute");
            self.f
                .store(Operand::Val(per_group), Operand::Val(gep_w), al);
            self.barrier();
            let flat_r = self.f.mul(Operand::Val(lane), Operand::i32(pitch));
            let gep_r = self.tg_scratch_ptr(Operand::Val(flat_r), "reduce_columns broadcast");
            let out = self.f.load(Operand::Val(gep_r), al);
            let out = self.fptrunc_if(out, &ty, &rty);
            self.bind(op_id, 0, Operand::Val(out));
            return Some(());
        }

        // Fallback for a row count that does not match the SIMD width: walk the column.
        let mut acc: Option<CfgValueId> = None;
        for r in 0..rows {
            let flat_r = self
                .f
                .add(Operand::i32(r as i64 * pitch), Operand::Val(lane));
            let gep_r = self.tg_scratch_ptr(Operand::Val(flat_r), "reduce_columns load");
            let v = self.f.load(Operand::Val(gep_r), al);
            acc = Some(match acc {
                None => v,
                Some(a) => self.scan_binop(Operand::Val(a), Operand::Val(v), kind, &ty, suf)?,
            });
        }
        // Everyone has read the column before any later write reuses the scratch.
        self.barrier();
        let out = self.fptrunc_if(acc?, &ty, &rty);
        self.bind(op_id, 0, Operand::Val(out));
        Some(())
    }

    /// True when every lane of a SIMD group holds the same value for `v`.
    fn is_sg_uniform(&self, v: ValRef) -> bool {
        !self.divergent.contains(&v)
    }

    /// Fragment dimensions of `v`, or `None` if it is not an SGMMA fragment.
    ///
    /// Both halves matter: the value has to be one the MMA ops actually touch, AND its
    /// type has to map onto 8x8 registers. Shape alone used to decide this, which
    /// claimed any `[8k, 8m]` tile - including threadgroup-spanning ones that never go
    /// near an MMA.
    /// Per-carry register types for a value crossing a loop or `if` boundary: `nr*nc`
    /// fragment registers for a real SGMMA fragment, otherwise ONE per-thread scalar.
    ///
    /// Classified by PROVENANCE, not shape. The shape test - "rank 2 and both extents
    /// multiples of 8" - calls any threadgroup-spanning `[SG, 32]` tile a fragment grid,
    /// so a plain accumulator carried across a loop was rejected: its single init could
    /// not match the invented fragment count, and the whole kernel failed to lower. That
    /// is the same shape-vs-provenance confusion `sgmma_of` exists to settle.
    fn carry_frag_tys(&self, v: ValRef) -> Option<Vec<LlType>> {
        let ty = self.ir.value_type(v);
        match self.sgmma_of(v) {
            Some((nr, nc)) => {
                if scalar_of(ty) != ScalarType::F32 {
                    return None; // mixed-precision SGMMA carry -> not this cut
                }
                Some(vec![vec64f(); nr * nc])
            }
            None => Some(vec![scalar_lltype(&scalar_of(ty))?]),
        }
    }

    fn sgmma_of(&self, v: ValRef) -> Option<(usize, usize)> {
        if !self.sgmma_vals.contains(&v) {
            return None;
        }
        crate::types::sgmma_dims(self.ir.value_type(v))
    }

    /// `getelementptr` into the threadgroup partials array, returning a
    /// `{elem} addrspace(3)*`.
    fn partial_gep(&mut self, idx: Operand, elem: LlType) -> Option<crate::lower::cfg::ValueId> {
        self.f.gep_global("@_tg_reduce_partial", elem, 32, 3, idx)
    }

    fn barrier(&mut self) {
        // mem_flags 2 (threadgroup), scope 1 - the same constants the text form used.
        self.f.call(
            LlType::Void,
            "@air.wg.barrier",
            vec![Operand::i32(2), Operand::i32(1)],
        );
    }

    /// User threadgroup scratch: a fixed `[TG_SCRATCH_FLOATS x float]
    /// addrspace(3)` array, declared once per kernel (only when tg_store/tg_load are
    /// used). Sized for the single-dispatch SDPA merge at BN=32 (4 tiles x 32 SIMD
    /// groups = 128 rows) and within Metal's 32 KB threadgroup limit.
    fn ensure_tg_scratch(&mut self) {
        if !self.tg_scratch_allocated {
            let n = self.tg_scratch_floats;
            self.globals.push(format!(
                "@_tg_user_scratch = internal unnamed_addr addrspace(3) global [{n} x float] undef, align 4"
            ));
            self.tg_globals
                .push(("@_tg_user_scratch".into(), LlType::Float, n as u32));
            self.tg_scratch_allocated = true;
        }
    }

    /// `&@_tg_user_scratch[flat]` as a `float addrspace(3)*`.
    fn tg_scratch_ptr(&mut self, flat: Operand, what: &str) -> crate::lower::cfg::ValueId {
        let n = self.tg_scratch_floats;
        self.f
            .gep_global("@_tg_user_scratch", LlType::Float, n as u32, 3, flat)
            .unwrap_or_else(|| panic!("{what}"))
    }

    /// Counted loop `for iv in start..end step step`. Builds header/body/back/exit
    /// blocks; each carry becomes a group of header phis (one per fragment - a
    /// single phi for scalars/1x1 SGMMA, nrxnc for a multi-fragment SGMMA
    /// accumulator) wired to its inits (entry edge) and the body's yielded updates
    /// (back edge). The fragment count flows from one source (`carry_frag_tys`) via
    /// `carry_phis`/`wire_carries`, so header and body cannot disagree. Leaves the
    /// exit block current.
    fn translate_for(&mut self, op_id: OpRef, op: &OpData) -> Option<()> {
        let n_carry = op.result_types.len();
        if op.operands.len() < 3 + n_carry {
            return None;
        }
        let iv_ty = scalar_lltype(&scalar_of(self.ir.value_type(op.operands[0])))?;
        if iv_ty != LlType::I32 {
            return None; // cut: i32 induction variable
        }
        let start = self.resolve(op.operands[0])?;
        let end = self.resolve(op.operands[1])?;
        let step = self.resolve(op.operands[2])?;
        // Per-carry fragment types (1 element for scalars/1x1 SGMMA, nr*nc for
        // multi-fragment). Bails on unsupported scalar/mixed-precision carries.
        let mut carry_frag_tys: Vec<Vec<LlType>> = Vec::with_capacity(n_carry);
        for i in 0..n_carry {
            carry_frag_tys.push(self.carry_frag_tys(op.operands[3 + i])?);
        }
        let inits: Vec<Vec<Operand>> = (0..n_carry)
            .map(|i| self.resolve_grid(op.operands[3 + i]))
            .collect::<Option<_>>()?;
        for i in 0..n_carry {
            if inits[i].len() != carry_frag_tys[i].len() {
                return None; // init fragment count disagrees with the carry type
            }
        }

        let body_region = *op.regions.first()?;
        let body_block_id = self.ir.region(body_region).blocks.first().copied()?;
        let body_args: Vec<ValRef> = self
            .ir
            .block(body_block_id)
            .args
            .iter()
            .map(|(v, _)| *v)
            .collect();
        if body_args.len() != 1 + n_carry {
            return None;
        }

        let n = self.label_ctr;
        self.label_ctr += 1;
        let pre = self.f.current_block()?;
        let hdr = self.f.block(&format!("_for{n}_hdr"));
        let body = self.f.block(&format!("_for{n}_body"));
        let back = self.f.block(&format!("_for{n}_back"));
        let exit = self.f.block(&format!("_for{n}_exit"));

        // Pre-header falls into the loop header.
        self.f.set_terminator(Terminator::Br(hdr));

        // Header: induction phi + per-carry fragment phis, then the condition.
        self.f.switch_to(hdr);
        let iv_phi = self.f.phi(iv_ty);
        let carry_phis: Vec<Vec<CfgValueId>> = self.f.carry_phis(&carry_frag_tys);
        self.vmap.insert(body_args[0], vec![Operand::Val(iv_phi)]);
        for (i, &arg) in body_args.iter().enumerate().skip(1) {
            let grid: Vec<Operand> = carry_phis[i - 1].iter().map(|&p| Operand::Val(p)).collect();
            self.vmap.insert(arg, grid);
        }
        let cond = self.f.icmp("slt", Operand::Val(iv_phi), end);
        self.f.set_terminator(Terminator::CondBr {
            cond: Operand::Val(cond),
            then_: body,
            else_: exit,
        });

        // Body: translate the region up to its Yield/Continue, capturing updates.
        // Expose the induction variable so cached loads can index their slot, and
        // the exit block so a nested `break` can jump straight to it.
        self.f.switch_to(body);
        self.loop_iv_stack.push(Operand::Val(iv_phi));
        self.loop_exit_stack.push((exit, n_carry > 0));
        let updates = self.translate_for_body(body_block_id)?;
        self.loop_exit_stack.pop();
        self.loop_iv_stack.pop();
        if updates.len() != n_carry {
            return None;
        }
        // A straight-line body ends in the same block it started; nested control
        // flow (an `if`, a `break`) leaves us in some other block. Only the former
        // is a safe, worthwhile unroll target.
        let body_straight = self.f.current_block() == Some(body);
        let body_insts = self.f.block_inst_count(body);
        // Whatever block the body ended in falls into the back-edge block - unless
        // it already branched away (e.g. an unconditional `break` to the exit).
        self.f.set_terminator_if_open(Terminator::Br(back));

        // Back-edge: advance the induction variable and loop.
        self.f.switch_to(back);
        let iv_next = self.f.add(Operand::Val(iv_phi), step);
        self.f.set_terminator(Terminator::Br(hdr));
        if let Some(count) = loop_unroll_factor(body_straight, body_insts) {
            self.f.set_loop_unroll(back, count);
        }

        // Wire the phis now that both incoming values exist as handles. The carry
        // helper asserts fragment counts agree across phis / inits / updates.
        self.f.phi_incoming(iv_phi, start, pre);
        self.f.phi_incoming(iv_phi, Operand::Val(iv_next), back);
        self.f
            .wire_carries(&carry_phis, pre, &inits, back, &updates);

        // Exit: each For result is its carry phi group's exit value.
        self.f.switch_to(exit);
        for (i, phis) in carry_phis.iter().enumerate() {
            let grid: Vec<Operand> = phis.iter().map(|&p| Operand::Val(p)).collect();
            self.bind_grid(op_id, i as u32, grid);
        }
        Some(())
    }

    /// if/else with a phi merge. Builds then/else/merge blocks; each region is
    /// translated up to its Yield, and each If result becomes a phi in the merge
    /// block merging the then/else yielded values (per fragment). Leaves merge
    /// current. Requires an else region when there are results.
    fn translate_if(&mut self, op_id: OpRef, op: &OpData) -> Option<()> {
        let cond = self.resolve(*op.operands.first()?)?;
        let then_block = self
            .ir
            .region(*op.regions.first()?)
            .blocks
            .first()
            .copied()?;
        let else_block = op
            .regions
            .get(1)
            .and_then(|&r| self.ir.region(r).blocks.first().copied());
        let n_results = op.result_types.len();
        let mut result_frag_tys: Vec<Vec<LlType>> = Vec::with_capacity(n_results);
        for i in 0..n_results {
            // Prefer the RESULT VALUE, so the classification is by provenance; fall back
            // to the type only when the result has no value to trace.
            result_frag_tys.push(match op.results.get(i).and_then(|r| *r) {
                Some(v) => self.carry_frag_tys(v)?,
                None => carry_fragment_types(op.result_types.get(i)?)?,
            });
        }
        if n_results > 0 && else_block.is_none() {
            return None; // a value-producing If needs both arms
        }

        let n = self.label_ctr;
        self.label_ctr += 1;
        let then_b = self.f.block(&format!("_if{n}_then"));
        let else_b = self.f.block(&format!("_if{n}_else"));
        let merge_b = self.f.block(&format!("_if{n}_merge"));
        self.f.set_terminator(Terminator::CondBr {
            cond,
            then_: then_b,
            else_: else_b,
        });

        // Then arm. An arm may branch away early (e.g. a `break`), leaving its
        // block already terminated - fall through to merge only if it's still open.
        self.f.switch_to(then_b);
        let then_vals = self.translate_for_body(then_block)?;
        let then_broke = self.f.is_terminated();
        let then_end = self.f.current_block()?;
        self.f.set_terminator_if_open(Terminator::Br(merge_b));

        // Else arm (empty yields if there's no else region and no results).
        self.f.switch_to(else_b);
        let else_vals = match else_block {
            Some(eb) => self.translate_for_body(eb)?,
            None => Vec::new(),
        };
        let else_broke = self.f.is_terminated();
        let else_end = self.f.current_block()?;
        self.f.set_terminator_if_open(Terminator::Br(merge_b));

        if n_results > 0 && (then_vals.len() != n_results || else_vals.len() != n_results) {
            return None;
        }
        // A value-producing If whose arm breaks can't feed the merge phis.
        if n_results > 0 && (then_broke || else_broke) {
            return None;
        }

        // Merge: one phi per result fragment.
        self.f.switch_to(merge_b);
        for i in 0..n_results {
            let fts = &result_frag_tys[i];
            if then_vals[i].len() != fts.len() || else_vals[i].len() != fts.len() {
                return None;
            }
            let grid: Vec<Operand> = fts
                .iter()
                .enumerate()
                .map(|(f, ft)| {
                    let phi = self.f.phi(ft.clone());
                    self.f.phi_incoming(phi, then_vals[i][f].clone(), then_end);
                    self.f.phi_incoming(phi, else_vals[i][f].clone(), else_end);
                    Operand::Val(phi)
                })
                .collect();
            self.bind_grid(op_id, i as u32, grid);
        }
        Some(())
    }

    /// Translate a loop body block, returning the fragment grids of its
    /// Yield/Continue operands (the updated carry values).
    fn translate_for_body(&mut self, body_block_id: BlockRef) -> Option<Vec<Vec<Operand>>> {
        // The loop body is walked here rather than through `translate_block`, so it has
        // to publish the current block itself - `fuse_lane_vector` looks for a load's
        // siblings in it, and most such loads are inside a loop.
        let outer_block = self.cur_block.replace(body_block_id);
        let out = self.translate_for_body_inner(body_block_id);
        self.cur_block = outer_block;
        out
    }

    fn translate_for_body_inner(&mut self, body_block_id: BlockRef) -> Option<Vec<Vec<Operand>>> {
        let ops: Vec<OpRef> = self.ir.block(body_block_id).ops.clone();
        for op_id in ops {
            let op = self.ir.op(op_id).clone();
            if matches!(op.opcode, Opcode::Yield | Opcode::Continue) {
                return op.operands.iter().map(|v| self.resolve_grid(*v)).collect();
            }
            self.translate_op(op_id, &op)?;
        }
        Some(Vec::new())
    }

    /// Single-SIMD inclusive prefix scan (n_elems <= 32, f32) - a Kogge-Stone
    /// shuffle scan. Bails on SGMMA tiles, non-f32, or n_elems > 32 (which needs
    /// the 4-phase hierarchical path).
    fn scan(&mut self, op_id: OpRef, op: &OpData) -> Option<()> {
        let src_ty = self.ir.value_type(op.operands[0]);
        if self.sgmma_of(op.operands[0]).is_some() {
            return None;
        }
        // Compute type: bf16 scans run in f32 (no bf16 shuffle intrinsic).
        let (ty, suf, rty) = self.intrin_ty(op.operands[0])?;
        let n_elems: i64 = match src_ty {
            Type::Tile(tt) => tt.shape.iter().product(),
            _ => 1,
        };
        let intrinsic = self.reduce_intrinsic(op); // same binop-from-region logic
        let reverse = op
            .attributes
            .iter()
            .find(|(k, _)| k == "reverse")
            .map(|(_, v)| matches!(v, Attribute::Bool(true)))
            .unwrap_or(false);
        let src = self.resolve(op.operands[0])?;
        let src = self.fpext_if(src, &rty, &ty);

        let tid = self.f.extractelement(self.tp.clone(), Operand::i32(0));
        let simd_lane = self.f.urem(Operand::Val(tid), Operand::i32(32));
        let shuffle = simd_intrinsic(
            if reverse {
                "@air.simd_shuffle_down.f32"
            } else {
                "@air.simd_shuffle_up.f32"
            },
            suf,
        );

        let simd_n = n_elems.min(32);
        let mut steps = Vec::new();
        let mut d = 1i64;
        while d < simd_n {
            steps.push(d);
            d <<= 1;
        }
        if steps.is_empty() {
            // D == 1: the scan is the identity on the single element.
            let zero = Operand::float_bits("0.0", ty.clone(), 0);
            let r = self.f.fadd(zero, src);
            let r = self.fptrunc_if(r, &ty, &rty);
            self.bind(op_id, 0, Operand::Val(r));
            return Some(());
        }

        let mut cur = src;
        for &delta in &steps {
            let shuf = self.f.call_fast(
                ty.clone(),
                shuffle.clone(),
                vec![cur.clone(), Operand::i16(delta)],
            )?;
            // Lanes without a neighbor in the shuffle direction keep their value.
            let cond = if reverse {
                let limit = (simd_n - delta) as i32;
                self.f
                    .icmp("uge", Operand::Val(simd_lane), Operand::i32(limit as i64))
            } else {
                self.f
                    .icmp("ult", Operand::Val(simd_lane), Operand::i32(delta))
            };
            let combined = self.scan_binop(Operand::Val(shuf), cur.clone(), intrinsic, &ty, suf)?;
            let next = self
                .f
                .select(Operand::Val(cond), cur.clone(), Operand::Val(combined));
            cur = Operand::Val(next);
        }
        // Single SIMD group: the within-SIMD scan is the full result.
        if n_elems <= 32 {
            let cur = match cur {
                Operand::Val(v) => Operand::Val(self.fptrunc_if(v, &ty, &rty)),
                other => other,
            };
            self.bind(op_id, 0, cur);
            return Some(());
        }
        // Cross-SIMD (n_elems > 32): 4-phase hierarchical scan.
        self.scan_hierarchical(
            op_id, cur, tid, simd_lane, &shuffle, reverse, intrinsic, n_elems, ty, suf, rty,
        )
    }

    /// Phases 2-4 of the hierarchical scan: the boundary lane of each SIMD group
    /// writes its group total to @_tg_scan_partials; group 0 prefix-scans the
    /// group totals; then each thread combines the exclusive group prefix with its
    /// within-SIMD result. Leaves the final block current.
    #[allow(clippy::too_many_arguments)]
    fn scan_hierarchical(
        &mut self,
        op_id: OpRef,
        cur: Operand,
        tid: CfgValueId,
        simd_lane: CfgValueId,
        shuffle: &str,
        reverse: bool,
        intrinsic: &str,
        n_elems: i64,
        ty: LlType,
        suf: &str,
        rty: LlType,
    ) -> Option<()> {
        let lls = ty.ll_str();
        let al = align_of(&ty);
        if !self.tg_scan_allocated {
            self.globals.push(format!(
                "@_tg_scan_partials = internal unnamed_addr addrspace(3) global [32 x {lls}] undef, align {al}"
            ));
            self.tg_globals
                .push(("@_tg_scan_partials".into(), ty.clone(), 32));
            self.tg_scan_allocated = true;
        }
        let identity = reduce_identity(intrinsic, suf);
        let n_groups = (n_elems / 32) as i32;
        let n = self.label_ctr;
        self.label_ctr += 1;
        let simd_grp = self.f.udiv(Operand::Val(tid), Operand::i32(32));

        let wt = self.f.block(&format!("_sc{n}_wt"));
        let skip_wt = self.f.block(&format!("_sc{n}_skip_wt"));
        let after_wt = self.f.block(&format!("_sc{n}_after_wt"));
        let gs = self.f.block(&format!("_sc{n}_gs"));
        let skip_gs = self.f.block(&format!("_sc{n}_skip_gs"));
        let after_gs = self.f.block(&format!("_sc{n}_after_gs"));

        // Phase 2: boundary lane (31 forward / 0 reverse) writes its group total.
        let boundary = if reverse { 0 } else { 31 };
        let is_bnd = self
            .f
            .icmp("eq", Operand::Val(simd_lane), Operand::i32(boundary));
        self.f.set_terminator(Terminator::CondBr {
            cond: Operand::Val(is_bnd),
            then_: wt,
            else_: skip_wt,
        });
        self.f.switch_to(wt);
        let wt_gep = self.scan_partial_gep(Operand::Val(simd_grp), ty.clone());
        self.f.store(cur.clone(), Operand::Val(wt_gep), al);
        self.f.set_terminator(Terminator::Br(after_wt));
        self.f.switch_to(skip_wt);
        self.f.set_terminator(Terminator::Br(after_wt));

        // Phase 3: SIMD group 0 prefix-scans the 32 group totals.
        self.f.switch_to(after_wt);
        self.barrier();
        let is_grp0 = self.f.icmp("eq", Operand::Val(simd_grp), Operand::i32(0));
        self.f.set_terminator(Terminator::CondBr {
            cond: Operand::Val(is_grp0),
            then_: gs,
            else_: skip_gs,
        });
        self.f.switch_to(gs);
        let gt_gep = self.scan_partial_gep(Operand::Val(simd_lane), ty.clone());
        let gt_raw = self.f.load(Operand::Val(gt_gep), al);
        let oob = self.f.icmp(
            "uge",
            Operand::Val(simd_lane),
            Operand::i32(n_groups as i64),
        );
        let gt_val = self.f.select(
            Operand::Val(oob),
            Operand::float_bits(identity.0, ty.clone(), identity.1),
            Operand::Val(gt_raw),
        );
        let mut gv = Operand::Val(gt_val);
        for gd in [1i64, 2, 4, 8, 16] {
            let gshuf =
                self.f
                    .call_fast(ty.clone(), shuffle, vec![gv.clone(), Operand::i16(gd)])?;
            let gc = if reverse {
                let lim = (32 - gd) as i32;
                self.f
                    .icmp("uge", Operand::Val(simd_lane), Operand::i32(lim as i64))
            } else {
                self.f
                    .icmp("ult", Operand::Val(simd_lane), Operand::i32(gd))
            };
            let gcomb = self.scan_binop(Operand::Val(gshuf), gv.clone(), intrinsic, &ty, suf)?;
            let gn = self
                .f
                .select(Operand::Val(gc), gv.clone(), Operand::Val(gcomb));
            gv = Operand::Val(gn);
        }
        let wb_gep = self.scan_partial_gep(Operand::Val(simd_lane), ty.clone());
        self.f.store(gv, Operand::Val(wb_gep), al);
        self.f.set_terminator(Terminator::Br(after_gs));
        self.f.switch_to(skip_gs);
        self.f.set_terminator(Terminator::Br(after_gs));

        // Phase 4: exclusive group prefix combined with the within-SIMD result.
        self.f.switch_to(after_gs);
        self.barrier();
        let edge_grp: i32 = if reverse { n_groups - 1 } else { 0 };
        let is_edge = self
            .f
            .icmp("eq", Operand::Val(simd_grp), Operand::i32(edge_grp as i64));
        let adj_raw = if reverse {
            self.f.add(Operand::Val(simd_grp), Operand::i32(1))
        } else {
            self.f.sub(Operand::Val(simd_grp), Operand::i32(1))
        };
        let adj_safe = self.f.select(
            Operand::Val(is_edge),
            Operand::i32(0),
            Operand::Val(adj_raw),
        );
        let adj_gep = self.scan_partial_gep(Operand::Val(adj_safe), ty.clone());
        let adj_loaded = self.f.load(Operand::Val(adj_gep), al);
        let group_prefix = self.f.select(
            Operand::Val(is_edge),
            Operand::float_bits(identity.0, ty.clone(), identity.1),
            Operand::Val(adj_loaded),
        );
        let result = self.scan_binop(Operand::Val(group_prefix), cur, intrinsic, &ty, suf)?;
        let result = self.fptrunc_if(result, &ty, &rty); // f32 -> bf16 if promoted
        self.bind(op_id, 0, Operand::Val(result));
        Some(())
    }

    /// GEP into the threadgroup scan-partials array, returning `{elem} addrspace(3)*`.
    fn scan_partial_gep(&mut self, idx: Operand, elem: LlType) -> CfgValueId {
        self.f
            .gep_global("@_tg_scan_partials", elem, 32, 3, idx)
            .expect("scan_partial_gep produces a result")
    }

    /// The reduce/scan combine step (`a (+) b`) for the given kind + element type.
    fn scan_binop(
        &mut self,
        a: Operand,
        b: Operand,
        kind: &str,
        ty: &LlType,
        suf: &str,
    ) -> Option<CfgValueId> {
        Some(match kind {
            s if s == super::AIR_SIMD_PRODUCT_F32 => self.f.fmul(a, b),
            s if s == super::AIR_SIMD_MAX_F32 => {
                self.f
                    .call_fast(ty.clone(), format!("@air.fast_fmax.{suf}"), vec![a, b])?
            }
            s if s == super::AIR_SIMD_MIN_F32 => {
                let c = self.f.fcmp("olt", a.clone(), b.clone());
                self.f.select(Operand::Val(c), a, b)
            }
            _ => self.f.fadd(a, b), // sum
        })
    }

    /// Pick the SIMD reduction intrinsic from the reduction body's binary op.
    fn reduce_intrinsic(&self, op: &OpData) -> &'static str {
        let Some(&region_id) = op.regions.first() else {
            return super::AIR_SIMD_SUM_F32;
        };
        let body = self.ir.block(self.ir.region(region_id).blocks[0]);
        for &bop in &body.ops {
            match self.ir.op(bop).opcode {
                Opcode::MaxF => return super::AIR_SIMD_MAX_F32,
                Opcode::MinF => return super::AIR_SIMD_MIN_F32,
                Opcode::MulF => return super::AIR_SIMD_PRODUCT_F32,
                Opcode::AddF => return super::AIR_SIMD_SUM_F32,
                _ => {}
            }
        }
        super::AIR_SIMD_SUM_F32
    }

    /// Constant integer value of `v`, if it is one.
    fn const_int(&self, v: ValRef) -> Option<i64> {
        let op = self.ir.op(self.ir.defining_op(v)?);
        if op.opcode != Opcode::Constant {
            return None;
        }
        match super::constant_value(op) {
            crate::lower::cfg::ImmVal::Int(i) => Some(i),
            crate::lower::cfg::ImmVal::Zero => Some(0),
            _ => None,
        }
    }

    /// See through view-forming ops to the value they ultimately rest on, so two
    /// partition views built by separate `load_tile` calls over the same tensor compare
    /// equal.
    fn view_root(&self, mut v: ValRef) -> ValRef {
        for _ in 0..8 {
            let Some(op_id) = self.ir.defining_op(v) else {
                return v;
            };
            let op = self.ir.op(op_id);
            if !matches!(
                op.opcode,
                Opcode::MakePartitionView | Opcode::MakeStridedView | Opcode::MakeTensorView
            ) {
                return v;
            }
            match op.operands.first() {
                Some(&inner) => v = inner,
                None => return v,
            }
        }
        v
    }

    /// Do two index operands denote the same value? Either the same SSA name, or two
    /// constants of equal value - `load_tile` mints a fresh constant per call.
    fn same_index(&self, a: ValRef, b: ValRef) -> bool {
        a == b || matches!((self.const_int(a), self.const_int(b)), (Some(x), Some(y)) if x == y)
    }

    /// Fuse W adjacent scalar loads that each read one element of the SAME contiguous
    /// per-lane run into a single `<W x elem>` load.
    ///
    /// A kernel that wants lane L to hold `x[L*W .. L*W+W]` writes W loads whose indices
    /// differ only in the innermost dim, over a view whose lane stride is W and whose
    /// innermost stride is 1. That is what upstream's vocabulary can say - it has no
    /// vector element type - and taken literally it is 4x to 9x slower than one wide
    /// load, because for a fixed `j` consecutive LANES are W elements apart: the access
    /// is strided across the warp instead of coalesced. Neither LLVM nor Metal fuses
    /// these (measured: the emitted AIR keeps W scalar loads at every dtype).
    ///
    /// So do it here. One `<W x elem>` load per lane is a single coalesced transaction
    /// and each original load becomes an `extractelement` - which is exactly what the
    /// `V128<E>` element type existed to express.
    ///
    /// Matching is STRUCTURAL, on the IR, not on resolved operands: the siblings sit
    /// later in the block and have not been lowered yet when the first one is reached,
    /// so `resolve` would answer `None` for every one of them.
    /// The W sibling ops of a contiguous per-lane run, ordered by their innermost
    /// index, or `None` if this op does not start/belong to a complete one.
    ///
    /// `view_at` is where the partition view sits in the operand list (0 for a load,
    /// 1 for a store, whose operand 0 is the tile) and `idx_at` where the indices begin.
    fn lane_run_group(
        &self,
        op_id: OpRef,
        op: &OpData,
        view_at: usize,
        idx_at: usize,
    ) -> Option<(usize, Vec<OpRef>)> {
        let pv = *op.operands.get(view_at)?;
        let (tile_shape, strides) = match self.ir.value_type(pv) {
            Type::PartitionView(pvt) => (pvt.tile_shape.clone(), pvt.tensor_view.strides.clone()),
            _ => return None,
        };
        let rank = tile_shape.len();
        // Innermost dim must be the per-lane run: extent 1, unit stride. The lane's own
        // dim then has to step by exactly W for the run to be contiguous and disjoint.
        if rank < 2 || *tile_shape.last()? != 1 || *strides.last()? != 1 {
            return None;
        }
        let lane_dim = (0..rank - 1).rev().find(|&d| tile_shape[d] > 1)?;
        let w = strides[lane_dim];
        if !(2..=8).contains(&w) {
            return None;
        }
        let w = w as usize;
        let last_idx = idx_at + rank - 1;
        let opcode = op.opcode;
        let want_ty = self.ir.value_type(pv).clone();
        let want_root = self.view_root(pv);

        let block = self.cur_block?;
        let ops: Vec<OpRef> = self.ir.block(block).ops.clone();
        let mut group: Vec<Option<OpRef>> = vec![None; w];
        for &cand in &ops {
            let c = self.ir.op(cand);
            if c.opcode != opcode || c.operands.len() != op.operands.len() {
                continue;
            }
            let cpv = match c.operands.get(view_at) {
                Some(&v) => v,
                None => continue,
            };
            if *self.ir.value_type(cpv) != want_ty || self.view_root(cpv) != want_root {
                continue;
            }
            if !(idx_at..last_idx).all(|d| self.same_index(c.operands[d], op.operands[d])) {
                continue;
            }
            let Some(j) = self.const_int(c.operands[last_idx]) else {
                continue;
            };
            if (0..w as i64).contains(&j) && group[j as usize].is_none() {
                group[j as usize] = Some(cand);
            }
        }
        let group: Vec<OpRef> = group.into_iter().collect::<Option<Vec<_>>>()?;
        if !group.contains(&op_id) {
            return None;
        }
        Some((w, group))
    }

    /// Mirror of [`Self::fuse_lane_vector`] for stores: W stores covering one contiguous
    /// per-lane run become one `<W x elem>` store, built by `insertelement`.
    ///
    /// Emitted at the LAST member in block order, because that is the first point at
    /// which every stored value is guaranteed to have been lowered - a value is defined
    /// before the store that uses it, but store `j`'s value may be computed after store
    /// `0`. The earlier members emit nothing.
    fn fuse_lane_vector_store(&mut self, op_id: OpRef, op: &OpData) -> Option<()> {
        let (w, group) = self.lane_run_group(op_id, op, 1, 2)?;
        let block = self.cur_block?;
        let ops = &self.ir.block(block).ops;
        let pos = |o: OpRef| ops.iter().position(|&x| x == o).unwrap_or(usize::MAX);
        if group.iter().any(|&m| pos(m) > pos(op_id)) {
            return Some(()); // a later member emits the whole run
        }

        let pv = op.operands[1];
        let elem = mem_elem_lltype(&self.pv_elem(pv)?)?;
        let vec_ty = LlType::vec(w as u32, elem.clone());
        let mut vec = Operand::undef(vec_ty.clone());
        for (j, &member) in group.iter().enumerate() {
            let val = self.resolve(self.ir.op(member).operands[0])?;
            vec = Operand::Val(self.f.insertelement(vec, val, Operand::i32(j as i64)));
        }

        // Address the run from its element-0 member, exactly as the load path does.
        let first = self.ir.op(group[0]).clone();
        let ptr = self.resolve(pv)?;
        let idx_ops = self.resolve_index_ops(&first, pv, 2)?;
        let base = self.strided_elem_gep(ptr, pv, &idx_ops, elem)?;
        let vptr = self
            .f
            .cast("bitcast", base, LlType::device_ptr(vec_ty.clone()));
        self.f.store(vec, Operand::Val(vptr), align_of(&vec_ty));
        for &member in &group {
            self.bind(member, 0, Operand::undef(LlType::I32));
        }
        Some(())
    }

    fn fuse_lane_vector(&mut self, op_id: OpRef, op: &OpData) -> Option<()> {
        let pv = *op.operands.first()?;
        let (tile_shape, strides) = match self.ir.value_type(pv) {
            Type::PartitionView(pvt) => (pvt.tile_shape.clone(), pvt.tensor_view.strides.clone()),
            _ => return None,
        };
        let rank = tile_shape.len();
        // Innermost dim must be the per-lane run: extent 1, unit stride. The lane's own
        // dim then has to step by exactly W for the run to be contiguous and disjoint.
        if rank < 2 || *tile_shape.last()? != 1 || *strides.last()? != 1 {
            return None;
        }
        let lane_dim = (0..rank - 1).rev().find(|&d| tile_shape[d] > 1)?;
        let w = strides[lane_dim];
        if !(2..=8).contains(&w) {
            return None;
        }
        let w = w as usize;
        // This load must be element 0 of its run; a later member is reached through
        // `fused_loads` instead.
        if self.const_int(*op.operands.get(rank)?) != Some(0) {
            return None;
        }

        let want_ty = self.ir.value_type(pv).clone();
        let want_root = self.view_root(pv);
        let block = self.cur_block?;
        let ops: Vec<OpRef> = self.ir.block(block).ops.clone();
        let here = ops.iter().position(|&o| o == op_id)?;

        let mut group: Vec<Option<OpRef>> = vec![None; w];
        group[0] = Some(op_id);
        for &cand in &ops[here + 1..] {
            let c = self.ir.op(cand);
            if c.opcode != Opcode::LoadViewTko || c.operands.len() != op.operands.len() {
                continue;
            }
            let cpv = match c.operands.first() {
                Some(&v) => v,
                None => continue,
            };
            if *self.ir.value_type(cpv) != want_ty || self.view_root(cpv) != want_root {
                continue;
            }
            if !(1..rank).all(|d| self.same_index(c.operands[d], op.operands[d])) {
                continue;
            }
            let Some(j) = self.const_int(c.operands[rank]) else {
                continue;
            };
            if (1..w as i64).contains(&j) && group[j as usize].is_none() {
                group[j as usize] = Some(cand);
            }
            if group.iter().all(Option::is_some) {
                break;
            }
        }
        let group: Vec<OpRef> = group.into_iter().collect::<Option<Vec<_>>>()?;

        // One wide load at the lane's run, then an extract per original load.
        let elem = mem_elem_lltype(&self.pv_elem(pv)?)?;
        let vec_ty = LlType::vec(w as u32, elem.clone());
        let ptr = self.resolve(pv)?;
        let idx_ops = self.resolve_index_ops(op, pv, 1)?;
        let base = self.strided_elem_gep(ptr, pv, &idx_ops, elem)?;
        let vptr = self
            .f
            .cast("bitcast", base, LlType::device_ptr(vec_ty.clone()));
        let wide = self.f.load(Operand::Val(vptr), align_of(&vec_ty));
        for (j, member) in group.into_iter().enumerate() {
            let e = self
                .f
                .extractelement(Operand::Val(wide), Operand::i32(j as i64));
            self.bind(member, 0, Operand::Val(e));
            self.bind(member, 1, Operand::undef(LlType::I32));
            self.fused_loads.insert(member);
        }
        Some(())
    }

    /// Simple scalar f32 partition-view load: `dst = base[block*n_elems + lane]`.
    fn load_view(&mut self, op_id: OpRef, op: &OpData) -> Option<()> {
        // Already answered by a sibling's fused wide load.
        if self.fused_loads.contains(&op_id) {
            return Some(());
        }
        // W contiguous per-lane loads -> one `<W x elem>` load. Tried before the scalar
        // path because it subsumes it when it applies.
        if self.fuse_lane_vector(op_id, op).is_some() {
            return Some(());
        }
        // Single 8x8 SGMMA tile -> simdgroup_matrix load. Multi-fragment (nr*nc>1)
        // is deferred to the registry cut.
        if let Some((nr, nc)) = op
            .results
            .first()
            .and_then(|r| *r)
            .and_then(|v| self.sgmma_of(v))
        {
            // A padded view over a genuinely ragged extent cannot go through the load
            // intrinsic - it has no mask operand - so spell the fragment out instead.
            if self.sgmma_load_masked(op_id, op, nr, nc).is_some() {
                return Some(());
            }
            return if (nr, nc) == (1, 1) {
                self.sgmma_load_single(op_id, op)
            } else {
                self.sgmma_load_multi(op_id, op, nr, nc)
            };
        }
        let pv = op.operands[0];

        // Two-pass smem cache: in the consuming loop, read the thread-private
        // cache slot for this iteration instead of re-reading global memory.
        let cached = self.smem_cache.get(&pv.0).cloned();
        let mode = self.smem_mode.get(&pv.0).copied();
        if let (Some((cache, nc, elem_ty)), Some(StageMode::Consuming)) = (&cached, mode) {
            let iv = self.loop_iv_stack.last().cloned()?;
            let cgep = self.cache_gep(*cache, *nc, iv, elem_ty.clone());
            let val = self.f.load(Operand::Val(cgep), align_of(elem_ty));
            self.bind(op_id, 0, Operand::Val(val));
            self.bind(op_id, 1, Operand::undef(LlType::I32));
            return Some(());
        }

        // External (fp8) scalar tile: load the storage byte and convert to the
        // compute type (f16). Same flat-index convention as the f32 path.
        if let Some(elem @ ScalarType::External(_)) = self.pv_elem(pv) {
            return self.scalar_ext_load(op_id, op, pv, &elem);
        }

        // Normal global load. Framework convention: operands[1] is the FLATTENED
        // tile index (the source linearizes multi-dim indices itself, e.g.
        // `load([row*num_chunks + j, 0])`); consecutive tiles are n_elems apart in
        // memory. So flat = block_idx * n_elems + lane. Element type is native
        // f32/f16/bf16, or native integer (index/mask tensors).
        let elem_ty = mem_elem_lltype(&self.pv_elem(pv)?)?;
        let ptr = self.resolve(pv)?;
        let mut pad: Option<(Operand, Operand)> = None;
        let gep = {
            // Strided is the only view addressing: `addr = sum_d index[d]*tile_shape[d]
            // *stride[d] + lane*stride_innermost`, which is what upstream's
            // `index: [i32; N]` means. The old flat form (`block_idx*n_elems + lane`)
            // required callers to pre-flatten the coordinate themselves and is gone.
            let idx_ops = self.resolve_index_ops(op, pv, 1)?;
            // A view that declares `padding` is telling us its last tile may hang off the
            // end of the tensor. Clamp the offset so the out-of-range lanes address
            // element 0 rather than faulting; the value they read is replaced below.
            match self
                .bounds_predicate(pv, &idx_ops)
                .zip(self.padding_constant(pv, &elem_ty))
            {
                Some((pred, pad_const)) => {
                    let off = self.strided_elem_offset(pv, &idx_ops)?;
                    let safe = self.f.select(pred.clone(), off, Operand::i32(0));
                    pad = Some((pred, pad_const));
                    Operand::Val(self.f.gep(elem_ty.clone(), ptr, Operand::Val(safe)))
                }
                None => self.strided_elem_gep(ptr, pv, &idx_ops, elem_ty.clone())?,
            }
        };
        let val = self.f.load(gep, align_of(&elem_ty));
        // Out-of-range lanes read element 0 above; give them the pad value instead.
        let val = match pad {
            Some((pred, pad_const)) => self.f.select(pred, Operand::Val(val), pad_const),
            None => val,
        };
        self.bind(op_id, 0, Operand::Val(val));
        // result 1 is the load token - unused downstream.
        self.bind(op_id, 1, Operand::undef(LlType::I32));

        // In the staging loop, also write the loaded value into the cache slot.
        if let (Some((cache, nc, elem_ty)), Some(StageMode::Staging)) = (&cached, mode) {
            let iv = self.loop_iv_stack.last().cloned()?;
            let cgep = self.cache_gep(*cache, *nc, iv, elem_ty.clone());
            self.f
                .store(Operand::Val(val), Operand::Val(cgep), align_of(elem_ty));
        }
        Some(())
    }

    /// GEP into a thread-private `[nc x {elem}]` cache at slot `iv`, returning a
    /// `{elem}*` (addrspace 0).
    fn cache_gep(&mut self, cache: CfgValueId, nc: i32, iv: Operand, elem: LlType) -> CfgValueId {
        self.f
            .gep_array_value(Operand::Val(cache), elem, nc as u32, 0, iv)
            .expect("cache_gep produces a result")
    }

    /// Simple scalar partition-view store (native f32/f16/bf16).
    fn store_view(&mut self, op_id: OpRef, op: &OpData) -> Option<()> {
        // W contiguous per-lane stores -> one `<W x elem>` store.
        if self.sgmma_of(op.operands[0]).is_none()
            && self.fuse_lane_vector_store(op_id, op).is_some()
        {
            return Some(());
        }
        if let Some((nr, nc)) = self.sgmma_of(op.operands[0]) {
            // A view that declares `padding` gets the masked spelling; the intrinsic
            // has no mask operand of its own.
            if let Some(()) = self.sgmma_store_masked(op_id, op, nr, nc) {
                return Some(());
            }
            return if (nr, nc) == (1, 1) {
                self.sgmma_store_single(op_id, op)
            } else {
                self.sgmma_store_multi(op_id, op, nr, nc)
            };
        }
        let pv = op.operands[1];
        // External (fp8) scalar tile: convert compute value -> storage byte.
        if let Some(elem @ ScalarType::External(_)) = self.pv_elem(pv) {
            return self.scalar_ext_store(op_id, op, pv, &elem);
        }
        let elem_ty = mem_elem_lltype(&self.pv_elem(pv)?)?;
        let tile = self.resolve(op.operands[0])?;
        let ptr = self.resolve(pv)?;
        // Strided addressing only; see the load path above.
        let idx_ops = self.resolve_index_ops(op, pv, 2)?;
        // A store cannot be clamped the way a padded load can - a clamped address would
        // overwrite a live element - so the out-of-range lanes have to be branched around.
        //
        // INERT TODAY: a `&mut Tensor` kernel parameter reaches the backend with a DYNAMIC
        // extent even when its shape is a literal, so `bounds_predicate` never has a bound
        // to compare against and this never fires. The gap is in the compiler front end,
        // not here; see `tail_lanes_do_not_store`, which is ignored for the same reason.
        let guard = self.bounds_predicate(pv, &idx_ops);
        let skip_b = if let Some(pred) = guard {
            let n = self.label_ctr;
            self.label_ctr += 1;
            let do_b = self.f.block(&format!("_pad{n}_store"));
            let skip_b = self.f.block(&format!("_pad{n}_skip"));
            self.f.set_terminator(Terminator::CondBr {
                cond: pred,
                then_: do_b,
                else_: skip_b,
            });
            self.f.switch_to(do_b);
            Some(skip_b)
        } else {
            None
        };
        let gep = self.strided_elem_gep(ptr, pv, &idx_ops, elem_ty.clone())?;
        self.f.store(tile, gep, align_of(&elem_ty));
        if let Some(skip_b) = skip_b {
            self.f.set_terminator_if_open(Terminator::Br(skip_b));
            self.f.switch_to(skip_b);
        }
        // result 0 is the store token - unused downstream.
        self.bind(op_id, 0, Operand::undef(LlType::I32));
        Some(())
    }

    /// Scalar fp8 (External) load: `half = fp8_to_f16(buf_i8[block*n + lane])`.
    fn scalar_ext_load(
        &mut self,
        op_id: OpRef,
        op: &OpData,
        pv: ValRef,
        elem: &ScalarType,
    ) -> Option<()> {
        let (storage, compute) = external_types(elem)?;
        if !matches!((storage, compute), (ScalarType::F8E4M3FN, ScalarType::F16)) {
            return None;
        }
        let n_elems = self.partition_elem_count(pv)?;
        let ptr = self.resolve(pv)?;
        let block_idx = self.resolve(op.operands[1])?;
        if !self.is_i32(&block_idx) {
            return None;
        }
        let gep = self.elem_gep_ty(ptr, block_idx, n_elems, LlType::I8);
        let raw = self.f.load(gep, 1);
        let half = self.fp8e4m3fn_to_half(Operand::Val(raw));
        self.bind(op_id, 0, half);
        self.bind(op_id, 1, Operand::undef(LlType::I32)); // token
        Some(())
    }

    /// Scalar fp8 (External) store: `buf_i8[block*n + lane] = f16_to_fp8(val)`.
    fn scalar_ext_store(
        &mut self,
        op_id: OpRef,
        op: &OpData,
        pv: ValRef,
        elem: &ScalarType,
    ) -> Option<()> {
        let (storage, compute) = external_types(elem)?;
        if !matches!((storage, compute), (ScalarType::F8E4M3FN, ScalarType::F16)) {
            return None;
        }
        let val = self.resolve(op.operands[0])?;
        let n_elems = self.partition_elem_count(pv)?;
        let ptr = self.resolve(pv)?;
        let block_idx = self.resolve(op.operands[2])?;
        if !self.is_i32(&block_idx) {
            return None;
        }
        let byte = self.half_to_fp8e4m3fn(val);
        let gep = self.elem_gep_ty(ptr, block_idx, n_elems, LlType::I8);
        self.f.store(byte, gep, 1);
        self.bind(op_id, 0, Operand::undef(LlType::I32)); // token
        Some(())
    }

    /// Atomic read-modify-write: `old = @air.atomic.{space}.{op}.{ty}(ptr, val, ...)`.
    fn atomic_rmw(&mut self, op_id: OpRef, op: &OpData) -> Option<()> {
        let ptr = self.resolve(op.operands[0])?;
        let val = self.resolve(op.operands[1])?;
        let (elem_ll, asn) = super::ptr_elem_type_and_as(self.ir.value_type(op.operands[0]));
        let mode = super::attr_i32(op, "mode").unwrap_or(3);
        let scope = atomic_scope(super::attr_i32(op, "memory_scope").unwrap_or(1));
        let (intrinsic, air_ty) = atomic_rmw_intrinsic(mode, &elem_ll, asn);
        let ll = atomic_ll_elem(air_ty);
        self.add_global(format!(
            "declare {ll} {intrinsic}({ll} addrspace({asn})* nocapture, {ll}, i32, i32, i1) #2"
        ));
        let r = self.f.call(
            atomic_lltype(air_ty),
            intrinsic,
            vec![
                ptr,
                val,
                Operand::i32(0),
                Operand::i32(scope as i64),
                Operand::i1(true),
            ],
        )?;
        self.bind(op_id, 0, Operand::Val(r));
        self.bind(op_id, 1, Operand::undef(LlType::I32)); // token
        Some(())
    }

    /// Atomic compare-and-swap via `@air.atomic.{space}.cmpxchg.weak.{ty}`. The
    /// expected value lives in a stack slot the intrinsic reads/updates.
    fn atomic_cas(&mut self, op_id: OpRef, op: &OpData) -> Option<()> {
        let ptr = self.resolve(op.operands[0])?;
        let cmp = self.resolve(op.operands[1])?;
        let new_val = self.resolve(op.operands[2])?;
        let (elem_ll, asn) = super::ptr_elem_type_and_as(self.ir.value_type(op.operands[0]));
        let scope = atomic_scope(super::attr_i32(op, "memory_scope").unwrap_or(1));
        let mem_space = if asn == 3 { "local" } else { "global" };
        let air_ty = atomic_air_ty(&elem_ll);
        let ll = atomic_ll_elem(air_ty);
        let intrinsic = format!("@air.atomic.{mem_space}.cmpxchg.weak.{air_ty}");
        self.add_global(format!(
            "declare {ll} {intrinsic}({ll} addrspace({asn})* nocapture, \
             {ll}* nocapture, {ll}, i32, i32, i32, i1) #2"
        ));
        let elem_lt = atomic_lltype(air_ty);
        let exp = self.f.alloca_scalar(elem_lt.clone(), 4)?;
        self.f.store(cmp, Operand::Val(exp), 4);
        let r = self.f.call(
            elem_lt,
            intrinsic.clone(),
            vec![
                ptr,
                Operand::Val(exp),
                new_val,
                Operand::i32(0),
                Operand::i32(0),
                Operand::i32(scope as i64),
                Operand::i1(true),
            ],
        )?;
        self.bind(op_id, 0, Operand::Val(r));
        self.bind(op_id, 1, Operand::undef(LlType::I32)); // token
        Some(())
    }

    /// Emit a module-level `declare` (deduplicated) before the function.
    fn add_global(&mut self, decl: String) {
        if !self.globals.contains(&decl) {
            self.globals.push(decl);
        }
    }

    /// `sgmma_elem{0,1}`: extract owned float slot N of a `<64 x float>` register.
    /// Only the SGMMA-register form is supported; other Extract uses bail to legacy.
    fn sgmma_extract(&mut self, op_id: OpRef, op: &OpData) -> Option<()> {
        self.sgmma_of(op.operands[0])?;
        let slot = super::attr_i32(op, "sgmma_slot").unwrap_or(0);
        let tile = self.resolve(op.operands[0])?;
        let e = self.f.extractelement(tile, Operand::i32(slot));
        self.bind(op_id, 0, Operand::Val(e));
        Some(())
    }

    /// `sgmma_set_elem{0,1}`: replace owned float slot N of a `<64 x float>`
    /// register (insertelement), returning the new register.
    fn sgmma_set_elem(&mut self, op_id: OpRef, op: &OpData) -> Option<()> {
        if op.operands.len() < 2
            || super::attr_i32(op, "sgmma_slot").is_none()
            || self.sgmma_of(op.operands[0]).is_none()
        {
            return None;
        }
        let slot = super::attr_i32(op, "sgmma_slot").unwrap_or(0);
        let tile = self.resolve(op.operands[0])?;
        let val = self.resolve(op.operands[1])?;
        let r = self.f.insertelement(tile, val, Operand::i32(slot));
        self.bind(op_id, 0, Operand::Val(r));
        Some(())
    }

    /// A bounds-masked SGMMA fragment LOAD, for a view that declares `padding`.
    ///
    /// `@air.simdgroup_matrix_8x8_load` has no mask operand, so a ragged edge cannot be
    /// predicated through the intrinsic. The backend does not need it to be: it knows the
    /// lane distribution, so it loads the two elements each lane owns and assembles the
    /// fragment with `insertelement`.
    ///
    /// That an assembled fragment is a legitimate MMA operand is not an assumption - it is
    /// what `sgmma_load_external` has always done for fp8, and
    /// `tests/gemm_fp8.rs::fp8_fragment_assembly_matches_the_load_intrinsic` pins it
    /// bit-exactly against this very intrinsic.
    ///
    /// Out-of-bounds lanes take the pad value, and their ADDRESS is clamped back inside
    /// the tensor: unlike a masked store, a masked load still performs the load, so
    /// forming the out-of-range address at all would fault.
    ///
    /// Single and multi-fragment are the same code - a single fragment is nr = nc = 1 with
    /// tile_h = tile_w = 8.
    fn sgmma_load_masked(&mut self, op_id: OpRef, op: &OpData, nr: usize, nc: usize) -> Option<()> {
        // Decide before emitting: the lane decomposition below is not free, and leaving it
        // behind on an access that needs no mask costs the enclosing loop its unroll hint.
        let pv = op.operands[0];
        let bounds = self.dim_bounds(pv, 2);
        if bounds.is_empty() {
            return None;
        }
        let elem = self.pv_elem(pv)?;
        let (_, _, _, frag_scalar) = sgmma_scalar_info(&elem)?;
        let pad = self.padding_constant(pv, &frag_scalar)?;
        let (tile_h, tile_w) = self.sgmma_tile_dims(pv)?;
        let (row_s, row_d) = self.resolve_stride(pv, 0, 8);
        let (col_s, col_d) = self.resolve_stride(pv, 1, 1);
        let ptr = self.resolve(pv)?;
        let n_bi = op.operands.len().saturating_sub(2);
        let bis: Vec<Operand> = (0..n_bi)
            .map(|i| self.resolve(op.operands[1 + i]))
            .collect::<Option<_>>()?;

        // The lane distribution, the same one `sgmma_load_external` and
        // `sgmma_store_masked` walk: x = (lane>>3)&1, y = lane>>4, loc = lane&7, and the
        // lane owns (row, col) and (row, col+1) at vector indices 0 and 1.
        let lane = Operand::Val(self.f.extractelement(self.tp.clone(), Operand::i32(0)));
        let x1 = self.f.lshr(lane.clone(), Operand::i32(3));
        let x = self.f.and(Operand::Val(x1), Operand::i32(1));
        let y = self.f.lshr(lane.clone(), Operand::i32(4));
        let loc = self.f.and(lane, Operand::i32(7));
        let rloc = self.f.lshr(Operand::Val(loc), Operand::i32(1));
        let y4 = self.f.shl(Operand::Val(y), Operand::i32(2));
        let row = self.f.add(Operand::Val(rloc), Operand::Val(y4));
        let lmod2 = self.f.and(Operand::Val(loc), Operand::i32(1));
        let clo = self.f.shl(Operand::Val(lmod2), Operand::i32(1));
        let x4 = self.f.shl(Operand::Val(x), Operand::i32(2));
        let col = self.f.add(Operand::Val(clo), Operand::Val(x4));

        let rs = self.stride_i32(row_s, &row_d);
        let cs = self.stride_i32(col_s, &col_d);
        // The AGX predicated load is an INTEGER op - floats bitcast through it - and it
        // fetches the lane's two owned elements as ONE contiguous vector, so it applies
        // only when the column stride is a unit. Anything else keeps the portable
        // clamp-and-select spelling below.
        let (int_ty, int_suf, elem_bytes) = match frag_scalar {
            LlType::Float => (LlType::I32, "i32", 4i64),
            LlType::Half | LlType::BFloat => (LlType::I16, "i16", 2i64),
            _ => return None,
        };
        let predicated = col_s == 1 && col_d.is_none();
        let ext_r = bounds.iter().find(|(d, _)| *d == 0).map(|(_, e)| e.clone());
        let ext_c = bounds.iter().find(|(d, _)| *d == 1).map(|(_, e)| e.clone());
        // The block's own origin, added once rather than per sub-tile.
        let org_r = bis
            .first()
            .cloned()
            .map(|b| Operand::Val(self.f.mul(b, Operand::i32(tile_h))));
        let org_c = bis
            .get(1)
            .cloned()
            .map(|b| Operand::Val(self.f.mul(b, Operand::i32(tile_w))));

        let frag_ty = LlType::vec(64, frag_scalar.clone());
        let align = align_of(&frag_scalar);
        let mut frags = Vec::with_capacity(nr * nc);
        for r in 0..nr {
            for c in 0..nc {
                // Absolute row of this lane's elements: block origin + sub-tile + lane.
                let mut ar = Operand::Val(row);
                if r > 0 {
                    ar = Operand::Val(self.f.add(ar, Operand::i32(r as i64 * 8)));
                }
                if let Some(o) = org_r.clone() {
                    ar = Operand::Val(self.f.add(ar, o));
                }
                let mut ac0 = Operand::Val(col);
                if c > 0 {
                    ac0 = Operand::Val(self.f.add(ac0, Operand::i32(c as i64 * 8)));
                }
                if let Some(o) = org_c.clone() {
                    ac0 = Operand::Val(self.f.add(ac0, o));
                }
                // Row half of the address and the predicate, shared by both elements.
                let frag = if predicated {
                    // AGX predicated load: masked-off lanes read ZERO and their access is
                    // SUPPRESSED, so this needs no address clamp to avoid faulting and no
                    // `select` to supply the pad value - and the lane's two adjacent elements
                    // are ONE vector access instead of two scalar ones. That is the whole
                    // difference between a mask costing ~12 scalar ops and costing ~2.
                    let roff = self.f.mul(ar.clone(), rs.clone());
                    let coff = self.f.mul(ac0.clone(), cs.clone());
                    let off = self.f.add(Operand::Val(roff), Operand::Val(coff));
                    let gep = self
                        .f
                        .gep(frag_scalar.clone(), ptr.clone(), Operand::Val(off));

                    // `edgecheck(pos, lo, hi)` bounds ONE axis and hands back a 4-bit element
                    // mask; we own 2 elements, so keep the low 2 bits. The row axis has no
                    // such instruction, so it selects the mask to 0 - which is what Apple's
                    // own async-copy code does for its second axis.
                    let mut mask = match &ext_c {
                        Some(e) => {
                            let m = self.f.call(
                                LlType::I16,
                                "@llvm.agx3.edgecheck",
                                vec![ac0.clone(), Operand::i32(0), e.clone()],
                            )?;
                            Operand::Val(self.f.and(Operand::Val(m), Operand::i16(3)))
                        }
                        None => Operand::i16(3),
                    };
                    if let Some(e) = ext_r.as_ref() {
                        let rok = Operand::Val(self.f.icmp("slt", ar.clone(), e.clone()));
                        mask = Operand::Val(self.f.select(rok, mask, Operand::i16(0)));
                    }

                    let iptr = self
                        .f
                        .cast("bitcast", gep, LlType::device_ptr(int_ty.clone()));
                    let vi = self.f.call(
                        LlType::vec(2, int_ty.clone()),
                        format!("@llvm.agx3.load.with.emask.global.v2{int_suf}"),
                        vec![
                            Operand::Val(iptr),
                            mask,
                            Operand::i16(3),
                            Operand::i16(elem_bytes),
                        ],
                    )?;
                    let vf = self.f.cast(
                        "bitcast",
                        Operand::Val(vi),
                        LlType::vec(2, frag_scalar.clone()),
                    );
                    let mut frag_p = Operand::undef(frag_ty.clone());
                    for j in 0..2i64 {
                        let v = self.f.extractelement(Operand::Val(vf), Operand::i32(j));
                        frag_p = Operand::Val(self.f.insertelement(
                            frag_p,
                            Operand::Val(v),
                            Operand::i32(j),
                        ));
                    }
                    frag_p
                } else {
                    // Portable spelling for a non-unit column stride: clamp the
                    // address so an out-of-range lane cannot fault, load, then select the
                    // pad value. This is what the predicated path above replaces.
                    let rok = ext_r
                        .as_ref()
                        .map(|e| Operand::Val(self.f.icmp("slt", ar.clone(), e.clone())));
                    let ar_addr = self.clamp_below(ar.clone(), ext_r.as_ref());
                    let roff = self.f.mul(ar_addr, rs.clone());
                    let mut f2 = Operand::undef(frag_ty.clone());
                    for j in 0..2i64 {
                        let ac = if j == 0 {
                            ac0.clone()
                        } else {
                            Operand::Val(self.f.add(ac0.clone(), Operand::i32(j)))
                        };
                        let cok = ext_c
                            .as_ref()
                            .map(|e| Operand::Val(self.f.icmp("slt", ac.clone(), e.clone())));
                        let ac_addr = self.clamp_below(ac, ext_c.as_ref());
                        let coff = self.f.mul(ac_addr, cs.clone());
                        let off = self.f.add(Operand::Val(roff), Operand::Val(coff));
                        let gep = self
                            .f
                            .gep(frag_scalar.clone(), ptr.clone(), Operand::Val(off));
                        let v = self.f.load(Operand::Val(gep), align);
                        let pred = match (rok.clone(), cok) {
                            (Some(a), Some(b)) => Some(Operand::Val(self.f.and(a, b))),
                            (Some(a), None) => Some(a),
                            (None, Some(b)) => Some(b),
                            (None, None) => None,
                        };
                        let v = match pred {
                            Some(p) => Operand::Val(self.f.select(p, Operand::Val(v), pad.clone())),
                            None => Operand::Val(v),
                        };
                        f2 = Operand::Val(self.f.insertelement(f2, v, Operand::i32(j)));
                    }
                    f2
                };
                frags.push(frag);
            }
        }
        if nr * nc == 1 {
            self.bind(op_id, 0, frags.pop()?);
        } else {
            self.bind_grid(op_id, 0, frags);
        }
        self.bind(op_id, 1, Operand::undef(LlType::I32)); // token
        Some(())
    }

    /// `min(v, ext - 1)`, for keeping an out-of-range coordinate from forming an address
    /// that faults. `None` bound means the coordinate is already in range by construction.
    fn clamp_below(&mut self, v: Operand, ext: Option<&Operand>) -> Operand {
        let Some(e) = ext else { return v };
        let last = self.f.sub(e.clone(), Operand::i32(1));
        let lt = self.f.icmp("slt", v.clone(), e.clone());
        Operand::Val(self.f.select(Operand::Val(lt), v, Operand::Val(last)))
    }

    fn sgmma_load_single(&mut self, op_id: OpRef, op: &OpData) -> Option<()> {
        let pv = op.operands[0];
        let elem = self.pv_elem(pv)?;
        // External element types (e.g. fp8) have no native SGMMA load intrinsic;
        // use the manual lane-distribution load instead.
        if matches!(elem, ScalarType::External(_)) {
            return self.sgmma_load_external(op_id, op, pv, &elem);
        }
        let (_, load_suf, _store_suf, frag_scalar) = sgmma_scalar_info(&elem)?;
        let frag_ty = LlType::vec(64, frag_scalar.clone());
        let (row_s, row_d) = self.resolve_stride(pv, 0, 8);
        let (col_s, col_d) = self.resolve_stride(pv, 1, 1);
        let ptr = self.resolve(pv)?;
        // operands: [pv, bi0, .., token] -> block indices exclude pv and token.
        let n_bi = op.operands.len().saturating_sub(2);
        let bis: Vec<Operand> = (0..n_bi)
            .map(|i| self.resolve(op.operands[1 + i]))
            .collect::<Option<_>>()?;
        let gep = self.sgmma_gep_dyn(ptr, &bis, row_s, &row_d, col_s, &col_d, frag_scalar)?;
        let vec0 = self.build_i64x2(row_s, row_d.clone(), 8, None);
        let vec1 = self.build_i64x2(col_s, col_d, row_s, row_d);
        let r = self.f.call(
            frag_ty,
            format!("@air.simdgroup_matrix_8x8_load.{load_suf}"),
            vec![gep, vec0, vec1, sgmma_zero_strides()],
        )?;
        self.bind(op_id, 0, Operand::Val(r));
        self.bind(op_id, 1, Operand::undef(LlType::I32)); // token
        Some(())
    }

    /// Single 8x8 SGMMA tile store (static or DYNAMIC strides). The fragment is
    /// always f32; the destination buffer may be f32/f16/bf16 (down-converting).
    /// A bounds-masked 8x8 fragment store, for a view that declares `padding`.
    ///
    /// `@air.simdgroup_matrix_8x8_store` has no mask operand, so a ragged edge cannot be
    /// predicated at fragment granularity - which is why the ragged GEMM clamps its origin
    /// and lets overlapping tiles rewrite the same values instead. The backend does not
    /// have that limitation: it holds the `<64 x float>` and knows the lane distribution,
    /// so it can spell the same store out as the two scalar stores each lane actually
    /// owns, each with its own bounds check.
    ///
    /// This is slower than the intrinsic and is only reached for a view that opted in, so
    /// the cost lands on the edge tile and nowhere else.
    fn sgmma_store_masked(
        &mut self,
        op_id: OpRef,
        op: &OpData,
        nr: usize,
        nc: usize,
    ) -> Option<()> {
        let pv = op.operands[1];
        let bounds = self.dim_bounds(pv, 2);
        if bounds.is_empty() {
            return None;
        }
        let elem = self.pv_elem(pv)?;
        let (_, _, _, frag_scalar) = sgmma_scalar_info(&elem)?;
        let mem_ty = mem_elem_lltype(&elem)?;
        let frags: Vec<Operand> = if nr * nc == 1 {
            vec![self.resolve(op.operands[0])?]
        } else {
            self.resolve_grid(op.operands[0])?
        };
        if frags.len() != nr * nc {
            return None;
        }
        let (tile_h, tile_w) = self.sgmma_tile_dims(pv)?;
        let (row_s, row_d) = self.resolve_stride(pv, 0, 8);
        let (col_s, col_d) = self.resolve_stride(pv, 1, 1);
        let ptr = self.resolve(pv)?;
        // operands: [tile, pv, bi0, .., token].
        let n_bi = op.operands.len().saturating_sub(3);
        let bis: Vec<Operand> = (0..n_bi)
            .map(|i| self.resolve(op.operands[2 + i]))
            .collect::<Option<_>>()?;

        // The lane distribution an 8x8 fragment uses, the same one `sgmma_load_external`
        // walks: x = (lane>>3)&1, y = lane>>4, loc = lane&7, and each lane owns the two
        // elements at (row, col) and (row, col+1), held at vector indices 0 and 1.
        let lane = Operand::Val(self.f.extractelement(self.tp.clone(), Operand::i32(0)));
        let x1 = self.f.lshr(lane.clone(), Operand::i32(3));
        let x = self.f.and(Operand::Val(x1), Operand::i32(1));
        let y = self.f.lshr(lane.clone(), Operand::i32(4));
        let loc = self.f.and(lane, Operand::i32(7));
        let rloc = self.f.lshr(Operand::Val(loc), Operand::i32(1));
        let y4 = self.f.shl(Operand::Val(y), Operand::i32(2));
        let row = self.f.add(Operand::Val(rloc), Operand::Val(y4));
        let lmod2 = self.f.and(Operand::Val(loc), Operand::i32(1));
        let clo = self.f.shl(Operand::Val(lmod2), Operand::i32(1));
        let x4 = self.f.shl(Operand::Val(x), Operand::i32(2));
        let col = self.f.add(Operand::Val(clo), Operand::Val(x4));

        let rs = self.stride_i32(row_s, &row_d);
        let cs = self.stride_i32(col_s, &col_d);
        // Same predication as the load: one masked store replaces two predicated basic
        // blocks per element, and the lane's two owned elements go out as one vector.
        let (int_ty, int_suf, elem_bytes) = match frag_scalar {
            LlType::Float => (LlType::I32, "i32", 4i64),
            LlType::Half | LlType::BFloat => (LlType::I16, "i16", 2i64),
            _ => return None,
        };
        let predicated = col_s == 1 && col_d.is_none();
        let ext_r = bounds.iter().find(|(d, _)| *d == 0).map(|(_, e)| e.clone());
        let ext_c = bounds.iter().find(|(d, _)| *d == 1).map(|(_, e)| e.clone());
        // The block's own origin, added once rather than per sub-tile. Addressing is done
        // in ABSOLUTE coordinates off the raw pointer - the same way `sgmma_load_masked`
        // does it - rather than folding the block offset into a base GEP and then adding
        // within-tile coordinates, which is easy to double-count.
        let org_r = bis
            .first()
            .cloned()
            .map(|b| Operand::Val(self.f.mul(b, Operand::i32(tile_h))));
        let org_c = bis
            .get(1)
            .cloned()
            .map(|b| Operand::Val(self.f.mul(b, Operand::i32(tile_w))));

        for (n, frag) in frags.into_iter().enumerate() {
            let (r, c) = (n / nc, n % nc);
            let frag = self.sgmma_frag_to(frag, &frag_scalar);
            let mut ar = Operand::Val(row);
            if r > 0 {
                ar = Operand::Val(self.f.add(ar, Operand::i32(r as i64 * 8)));
            }
            if let Some(o) = org_r.clone() {
                ar = Operand::Val(self.f.add(ar, o));
            }
            let mut ac0 = Operand::Val(col);
            if c > 0 {
                ac0 = Operand::Val(self.f.add(ac0, Operand::i32(c as i64 * 8)));
            }
            if let Some(o) = org_c.clone() {
                ac0 = Operand::Val(self.f.add(ac0, o));
            }
            if predicated {
                let mut mask = match &ext_c {
                    Some(e) => {
                        let m = self.f.call(
                            LlType::I16,
                            "@llvm.agx3.edgecheck",
                            vec![ac0.clone(), Operand::i32(0), e.clone()],
                        )?;
                        Operand::Val(self.f.and(Operand::Val(m), Operand::i16(3)))
                    }
                    None => Operand::i16(3),
                };
                if let Some(e) = ext_r.as_ref() {
                    let rok = Operand::Val(self.f.icmp("slt", ar.clone(), e.clone()));
                    mask = Operand::Val(self.f.select(rok, mask, Operand::i16(0)));
                }
                // Pack the lane's two owned elements into one vector value.
                let mut pair = Operand::undef(LlType::vec(2, frag_scalar.clone()));
                for j in 0..2i64 {
                    let v = self.f.extractelement(frag.clone(), Operand::i32(j));
                    pair =
                        Operand::Val(self.f.insertelement(pair, Operand::Val(v), Operand::i32(j)));
                }
                let pair = self.f.cast("bitcast", pair, LlType::vec(2, int_ty.clone()));
                let roff = self.f.mul(ar.clone(), rs.clone());
                let coff = self.f.mul(ac0.clone(), cs.clone());
                let off = self.f.add(Operand::Val(roff), Operand::Val(coff));
                let gep = self.f.gep(mem_ty.clone(), ptr.clone(), Operand::Val(off));
                let iptr = self.f.cast(
                    "bitcast",
                    Operand::Val(gep),
                    LlType::device_ptr(int_ty.clone()),
                );
                self.f.call(
                    LlType::Void,
                    format!("@llvm.agx3.store.with.emask.global.v2{int_suf}"),
                    vec![
                        Operand::Val(iptr),
                        Operand::Val(pair),
                        mask,
                        Operand::i16(3),
                        Operand::i16(elem_bytes),
                    ],
                );
                continue;
            }
            // Portable spelling for a non-unit column stride: one predicated basic-block
            // pair per owned element. This is what the masked store above replaces.
            let rok = ext_r
                .as_ref()
                .map(|e| Operand::Val(self.f.icmp("slt", ar.clone(), e.clone())));
            let roff = self.f.mul(ar.clone(), rs.clone());
            for j in 0..2i64 {
                let ac = if j == 0 {
                    ac0.clone()
                } else {
                    Operand::Val(self.f.add(ac0.clone(), Operand::i32(j)))
                };
                let cok = ext_c
                    .as_ref()
                    .map(|e| Operand::Val(self.f.icmp("slt", ac.clone(), e.clone())));
                let pred = match (rok.clone(), cok) {
                    (Some(a), Some(b)) => Operand::Val(self.f.and(a, b)),
                    (Some(a), None) => a,
                    (None, Some(b)) => b,
                    (None, None) => return None,
                };
                let v = self.f.extractelement(frag.clone(), Operand::i32(j));
                let nlab = self.label_ctr;
                self.label_ctr += 1;
                let do_b = self.f.block(&format!("_frag{nlab}_store"));
                let skip_b = self.f.block(&format!("_frag{nlab}_skip"));
                self.f.set_terminator(Terminator::CondBr {
                    cond: pred,
                    then_: do_b,
                    else_: skip_b,
                });
                self.f.switch_to(do_b);
                let coff = self.f.mul(ac, cs.clone());
                let flat = self.f.add(Operand::Val(roff), Operand::Val(coff));
                let gep = self.f.gep(mem_ty.clone(), ptr.clone(), Operand::Val(flat));
                self.f
                    .store(Operand::Val(v), Operand::Val(gep), align_of(&mem_ty));
                self.f.set_terminator_if_open(Terminator::Br(skip_b));
                self.f.switch_to(skip_b);
            }
        }
        self.bind(op_id, 0, Operand::undef(LlType::I32)); // token
        Some(())
    }

    fn sgmma_store_single(&mut self, op_id: OpRef, op: &OpData) -> Option<()> {
        let tile = self.resolve(op.operands[0])?;
        let pv = op.operands[1];
        let elem = self.pv_elem(pv)?;
        let (_, _load_suf, store_suf, frag_scalar) = sgmma_scalar_info(&elem)?;
        let (row_s, row_d) = self.resolve_stride(pv, 0, 8);
        let (col_s, col_d) = self.resolve_stride(pv, 1, 1);
        let ptr = self.resolve(pv)?;
        // operands: [tile, pv, bi0, .., token].
        let n_bi = op.operands.len().saturating_sub(3);
        let bis: Vec<Operand> = (0..n_bi)
            .map(|i| self.resolve(op.operands[2 + i]))
            .collect::<Option<_>>()?;
        let gep =
            self.sgmma_gep_dyn(ptr, &bis, row_s, &row_d, col_s, &col_d, frag_scalar.clone())?;
        let vec0 = self.build_i64x2(row_s, row_d.clone(), 8, None);
        let vec1 = self.build_i64x2(col_s, col_d, row_s, row_d);
        let tile = self.sgmma_frag_to(tile, &frag_scalar);
        // Void call - `call` returns None (no result); do not use `?`.
        self.f.call(
            LlType::Void,
            format!("@air.simdgroup_matrix_8x8_store.{store_suf}"),
            vec![tile, gep, vec0, vec1, sgmma_zero_strides()],
        );
        self.bind(op_id, 0, Operand::undef(LlType::I32)); // token
        Some(())
    }

    /// Resolve stride for `dim` of a partition view: `(static, None)`, or
    /// `(default, Some(runtime_register))` for a DYNAMIC stride (traced to the
    /// loaded stride-arg operand).
    fn resolve_stride(&self, pv: ValRef, dim: usize, default: i64) -> (i64, Option<Operand>) {
        let s = match self.ir.value_type(pv) {
            Type::PartitionView(pvt) => pvt.tensor_view.strides.get(dim).copied(),
            Type::TensorView(tv) => tv.strides.get(dim).copied(),
            _ => None,
        }
        .unwrap_or(default);
        if s != wax_ir::DYNAMIC {
            return (s, None);
        }
        if let Some(sv) = self.find_stride_val_for_pv(pv, dim)
            && let Some(reg) = self.resolve(sv)
        {
            return (default, Some(reg));
        }
        (default, None)
    }

    /// See through any `MakeStridedView` wrappers to the view underneath.
    ///
    /// The wrapper only annotates the addressing mode; every structural query about the
    /// view - element type, tile shape, and in particular the DYNAMIC stride operands
    /// traced below - has to reach the `MakePartitionView` it wraps. Without this, a
    /// wrapped view silently falls back to a stride of 1, which is a wrong address rather
    /// than a failed compile.
    fn unwrap_strided_view(&self, pv: ValRef) -> ValRef {
        let mut cur = pv;
        for _ in 0..8 {
            let Some(op_id) = self.ir.defining_op(cur) else {
                return cur;
            };
            if self.ir.op(op_id).opcode != Opcode::MakeStridedView {
                return cur;
            }
            match self.ir.op(op_id).operands.first() {
                Some(&inner) => cur = inner,
                None => return cur,
            }
        }
        cur
    }

    /// The SSA value carrying `pv`'s runtime extent along `dim`, when its type records
    /// that dim as DYNAMIC.
    ///
    /// The mirror of `find_stride_val_for_pv` on the other operand range: `MakeTensorView`
    /// carries its dynamic dims first and its dynamic strides after. This only resolves to
    /// something useful when the dim was promoted to a kernel parameter, which happens
    /// exactly for the padded stores that ask for it - otherwise `setup_params` leaves it
    /// as the constant 0 and the caller must not build a bound from it.
    fn find_shape_val_for_pv(&self, pv: ValRef, dim: usize) -> Option<ValRef> {
        let pv = self.unwrap_strided_view(pv);
        let tv = {
            let pv_op = self.ir.op(self.ir.defining_op(pv)?);
            if pv_op.opcode != Opcode::MakePartitionView {
                return None;
            }
            *pv_op.operands.first()?
        };
        let tv_op_id = self.ir.defining_op(tv)?;
        let tv_op = self.ir.op(tv_op_id);
        if tv_op.opcode != Opcode::MakeTensorView {
            return None;
        }
        let tv_shape = match self.ir.value_type(tv) {
            Type::TensorView(t) => &t.shape,
            _ => return None,
        };
        if tv_shape.get(dim).copied() != Some(wax_ir::DYNAMIC) {
            return None;
        }
        let dyn_idx = tv_shape[..=dim]
            .iter()
            .filter(|&&e| e == wax_ir::DYNAMIC)
            .count()
            - 1;
        let (n_base, n_dyn_shape, _) = super::parse_operand_seg_sizes(tv_op);
        if dyn_idx >= n_dyn_shape {
            return None;
        }
        let v = tv_op.operands.get(n_base + dyn_idx).copied()?;
        // The promoted set holds BLOCK ARGS, and the operand may sit behind the wrapper's
        // `assume` chain, so trace before asking - and hand back the traced value, which
        // is the one `setup_params` bound. Only usable if it really was promoted: an
        // unpromoted scalar param resolves to the constant 0, and a bound of 0 masks away
        // every access.
        let ba = super::trace_to_block_arg(self.ir, v);
        self.promoted_dims.contains(&ba).then_some(ba)
    }

    fn find_stride_val_for_pv(&self, pv: ValRef, dim: usize) -> Option<ValRef> {
        let pv = self.unwrap_strided_view(pv);
        let tv = {
            let pv_op = self.ir.op(self.ir.defining_op(pv)?);
            if pv_op.opcode != Opcode::MakePartitionView {
                return None;
            }
            *pv_op.operands.first()?
        };
        let tv_op_id = self.ir.defining_op(tv)?;
        let tv_op = self.ir.op(tv_op_id);
        if tv_op.opcode != Opcode::MakeTensorView {
            return None;
        }
        let tv_strides = match self.ir.value_type(tv) {
            Type::TensorView(t) => &t.strides,
            _ => return None,
        };
        let up_to = dim.min(tv_strides.len().saturating_sub(1));
        let dyn_count = tv_strides[..=up_to]
            .iter()
            .filter(|&&s| s == wax_ir::DYNAMIC)
            .count();
        if dyn_count == 0 {
            return None;
        }
        let dyn_idx = dyn_count - 1;
        let (n_base, n_dyn_shape, _) = super::parse_operand_seg_sizes(tv_op);
        tv_op.operands.get(n_base + n_dyn_shape + dyn_idx).copied()
    }

    /// Build a `<2 x i64>` stride vector `<e0, e1>` where each element is a static
    /// literal or a runtime i32 stride zero-extended to i64 (via insertelement).
    fn build_i64x2(
        &mut self,
        s0: i64,
        d0: Option<Operand>,
        s1: i64,
        d1: Option<Operand>,
    ) -> Operand {
        let vec2 = LlType::vec(2, LlType::I64);
        if d0.is_none() && d1.is_none() {
            return Operand::vec_int(format!("<i64 {s0}, i64 {s1}>"), vec2, vec![s0, s1]);
        }
        let e0 = self.stride_elem_i64(s0, d0);
        let e1 = self.stride_elem_i64(s1, d1);
        let t0 = self
            .f
            .insertelement(Operand::undef(vec2), e0, Operand::i32(0));
        let t1 = self.f.insertelement(Operand::Val(t0), e1, Operand::i32(1));
        Operand::Val(t1)
    }

    /// An i64 stride element: static literal, or a runtime i32 register zext'd.
    fn stride_elem_i64(&mut self, s: i64, d: Option<Operand>) -> Operand {
        match d {
            Some(reg) => Operand::Val(self.f.cast("zext", reg, LlType::I64)),
            None => Operand::i64(s),
        }
    }

    /// `base * stride * scale` in i32, with a runtime or static stride.
    fn mul_stride(
        &mut self,
        base: Operand,
        stride_s: i64,
        stride_d: &Option<Operand>,
        scale: i64,
    ) -> Operand {
        match stride_d {
            Some(reg) => {
                let m = self.f.mul(base, reg.clone());
                if scale == 1 {
                    Operand::Val(m)
                } else {
                    Operand::Val(self.f.mul(Operand::Val(m), Operand::i32(scale)))
                }
            }
            None => Operand::Val(self.f.mul(base, Operand::i32(stride_s * scale))),
        }
    }

    /// Base pointer for a single 8x8 SGMMA tile at block index `[bi0, bi1]`
    /// (row, col), scaling each index by its stride x 8. No offset for 0 indices.
    #[allow(clippy::too_many_arguments)]
    fn sgmma_gep_dyn(
        &mut self,
        ptr: Operand,
        bis: &[Operand],
        row_s: i64,
        row_d: &Option<Operand>,
        col_s: i64,
        col_d: &Option<Operand>,
        elem: LlType,
    ) -> Option<Operand> {
        if bis.is_empty() {
            return Some(ptr);
        }
        for b in bis {
            if !self.is_i32(b) {
                return None;
            }
        }
        let off = if bis.len() == 1 {
            self.mul_stride(bis[0].clone(), col_s, col_d, 8)
        } else {
            let o0 = self.mul_stride(bis[0].clone(), row_s, row_d, 8);
            let o1 = self.mul_stride(bis[1].clone(), col_s, col_d, 8);
            Operand::Val(self.f.add(o0, o1))
        };
        Some(Operand::Val(self.f.gep(elem, ptr, off)))
    }

    /// Manual SGMMA load for External element types (e.g. fp8). The native load
    /// intrinsic only supports f16/bf16/f32 pointers, so we use the known SGMMA
    /// lane distribution to compute exactly which two matrix elements each thread
    /// owns, load their storage bytes, convert to the compute type, widen to f32,
    /// and insert into a `<64 x float>` register at slots 0 and 1. Mirrors the
    /// legacy `emit_sgmma_load_external`. Only fp8 E4M3FN (storage i8, compute
    /// f16) is supported; anything else bails to legacy.
    fn sgmma_load_external(
        &mut self,
        op_id: OpRef,
        op: &OpData,
        pv: ValRef,
        elem: &ScalarType,
    ) -> Option<()> {
        // Only F8E4M3FN (i8 storage -> f16 compute) is handled here.
        let (storage, compute) = match elem {
            ScalarType::External(t) => (t.storage_type(), t.compute_type()),
            _ => return None,
        };
        if !matches!(
            (&storage, &compute),
            (ScalarType::F8E4M3FN, ScalarType::F16)
        ) {
            return None;
        }

        let (row_s, row_d) = self.resolve_stride(pv, 0, 8);
        let (col_s, col_d) = self.resolve_stride(pv, 1, 1);
        let ptr = self.resolve(pv)?;
        // Base pointer to tile [0,0] using the same block-index logic as the
        // intrinsic path, but in i8 storage units.
        let n_bi = op.operands.len().saturating_sub(2);
        let bis: Vec<Operand> = (0..n_bi)
            .map(|i| self.resolve(op.operands[1 + i]))
            .collect::<Option<_>>()?;
        let base = self.sgmma_gep_dyn(ptr, &bis, row_s, &row_d, col_s, &col_d, LlType::I8)?;

        // Per-thread (row, col_base) from the SGMMA lane distribution.
        //   x = (lane>>3)&1, y = lane>>4, loc = lane&7
        //   row = (loc>>1) + y*4,  col_base = (loc&1)*2 + x*4
        let lane = Operand::Val(self.f.extractelement(self.tp.clone(), Operand::i32(0)));
        let x1 = self.f.lshr(lane.clone(), Operand::i32(3));
        let x = self.f.and(Operand::Val(x1), Operand::i32(1));
        let y = self.f.lshr(lane.clone(), Operand::i32(4));
        let loc = self.f.and(lane, Operand::i32(7));
        let rloc = self.f.lshr(Operand::Val(loc), Operand::i32(1));
        let y4 = self.f.shl(Operand::Val(y), Operand::i32(2));
        let row = self.f.add(Operand::Val(rloc), Operand::Val(y4));
        let lmod2 = self.f.and(Operand::Val(loc), Operand::i32(1));
        let clo = self.f.shl(Operand::Val(lmod2), Operand::i32(1));
        let x4 = self.f.shl(Operand::Val(x), Operand::i32(2));
        let col = self.f.add(Operand::Val(clo), Operand::Val(x4));

        // Flat element indices for the two owned elements (col_base, col_base+1).
        let rs = self.stride_i32(row_s, &row_d);
        let cs = self.stride_i32(col_s, &col_d);
        let rscl = self.f.mul(Operand::Val(row), rs);
        let col_scl = self.f.mul(Operand::Val(col), cs.clone());
        let fi0 = self.f.add(Operand::Val(rscl), Operand::Val(col_scl));
        let fi1 = self.f.add(Operand::Val(fi0), cs);

        let gep0 = self.f.gep(LlType::I8, base.clone(), Operand::Val(fi0));
        let gep1 = self.f.gep(LlType::I8, base, Operand::Val(fi1));
        let raw0 = self.f.load(Operand::Val(gep0), 1);
        let raw1 = self.f.load(Operand::Val(gep1), 1);

        // fp8 i8 -> f16 -> f32.
        let h0 = self.fp8e4m3fn_to_half(Operand::Val(raw0));
        let h1 = self.fp8e4m3fn_to_half(Operand::Val(raw1));
        let fv0 = self.f.cast("fpext", h0, LlType::Float);
        let fv1 = self.f.cast("fpext", h1, LlType::Float);

        // Build the <64 x float> register with our two owned elements at 0 and 1.
        let mat0 =
            self.f
                .insertelement(Operand::undef(vec64f()), Operand::Val(fv0), Operand::i32(0));
        let mat1 = self
            .f
            .insertelement(Operand::Val(mat0), Operand::Val(fv1), Operand::i32(1));
        self.bind(op_id, 0, Operand::Val(mat1));
        self.bind(op_id, 1, Operand::undef(LlType::I32)); // token
        Some(())
    }

    /// fp8 E4M3FN (i8) -> f16 (half) via bit manipulation. Mirrors legacy
    /// `emit_fp8e4m3fn_to_f16`.
    ///
    /// NORMAL VALUES ONLY. Exact zero is special-cased, but a SUBNORMAL (E=0, M!=0) is NOT:
    /// this builds `(E + 8) << 10` and ORs the mantissa in, which supplies the implicit
    /// leading 1 a subnormal does not have, so 0x01 decodes ~4.5x too large rather than to
    /// zero. It yields a plausible finite number, so it will not announce itself.
    /// `tests/ext_fp8.rs::fp8_out_of_domain_behaviour_is_pinned` records the exact mapping.
    fn fp8e4m3fn_to_half(&mut self, src: Operand) -> Operand {
        let i16 = LlType::I16;
        let imm = |n: i64| Operand::i16(n);
        let v = self.f.cast("zext", src, i16.clone());
        let sign = self.f.and(Operand::Val(v), imm(128));
        let sign_f16 = self.f.shl(Operand::Val(sign), imm(8));
        let exp_fp8 = self.f.lshr(Operand::Val(v), imm(3));
        let exp_mask = self.f.and(Operand::Val(exp_fp8), imm(15));
        let exp_adj = self.f.add(Operand::Val(exp_mask), imm(8));
        let exp_f16 = self.f.shl(Operand::Val(exp_adj), imm(10));
        let mant = self.f.and(Operand::Val(v), imm(7));
        let mant_f16 = self.f.shl(Operand::Val(mant), imm(7));
        let f16_ns = self.f.or(Operand::Val(exp_f16), Operand::Val(mant_f16));
        let f16_bits = self.f.or(Operand::Val(sign_f16), Operand::Val(f16_ns));
        let abs = self.f.and(Operand::Val(v), imm(127));
        let is_zero = self.f.icmp("eq", Operand::Val(abs), imm(0));
        let fin = self
            .f
            .select(Operand::Val(is_zero), imm(0), Operand::Val(f16_bits));
        Operand::Val(self.f.cast("bitcast", Operand::Val(fin), LlType::Half))
    }

    /// f16 (half) -> fp8 E4M3FN (i8) via bit manipulation. Mirrors legacy
    /// `emit_f16_to_fp8e4m3fn`. Truncates the mantissa - it does not round.
    ///
    /// The exponent clamp is NOT saturation to max finite: the mantissa is preserved
    /// across it, so 256 doubles back to 256 rather than to 448, and 240 doubles to 0x7F -
    /// which E4M3**FN** spends on NaN instead of infinity. Pinned in
    /// `tests/ext_fp8.rs::fp8_out_of_domain_behaviour_is_pinned`.
    fn half_to_fp8e4m3fn(&mut self, src: Operand) -> Operand {
        let imm = |n: i64| Operand::i16(n);
        let bits = self.f.cast("bitcast", src, LlType::I16);
        let sign = self.f.and(Operand::Val(bits), imm(32768));
        let sign_fp8 = self.f.lshr(Operand::Val(sign), imm(8));
        let exp_sh = self.f.lshr(Operand::Val(bits), imm(10));
        let exp_m = self.f.and(Operand::Val(exp_sh), imm(31));
        let exp_raw = self.f.sub(Operand::Val(exp_m), imm(8));
        let is_neg = self.f.icmp("slt", Operand::Val(exp_raw), imm(0));
        let is_ovf = self.f.icmp("sgt", Operand::Val(exp_raw), imm(15));
        let clamp0 = self
            .f
            .select(Operand::Val(is_neg), imm(0), Operand::Val(exp_raw));
        let clamp15 = self
            .f
            .select(Operand::Val(is_ovf), imm(15), Operand::Val(clamp0));
        let exp_fp8 = self.f.shl(Operand::Val(clamp15), imm(3));
        let mant_sh = self.f.lshr(Operand::Val(bits), imm(7));
        let mant_fp8 = self.f.and(Operand::Val(mant_sh), imm(7));
        let body = self.f.or(Operand::Val(exp_fp8), Operand::Val(mant_fp8));
        let full = self.f.or(Operand::Val(sign_fp8), Operand::Val(body));
        Operand::Val(self.f.trunc(Operand::Val(full), LlType::I8))
    }

    /// A stride as an i32 operand: runtime register or static literal.
    fn stride_i32(&self, s: i64, d: &Option<Operand>) -> Operand {
        match d {
            Some(reg) => reg.clone(),
            None => Operand::i32(s),
        }
    }

    /// Multi-fragment (nrxnc) f32 SGMMA tile load -> a row-major grid of loads.
    fn sgmma_load_multi(&mut self, op_id: OpRef, op: &OpData, nr: usize, nc: usize) -> Option<()> {
        let pv = op.operands[0];
        let elem = self.pv_elem(pv)?;
        let (_, load_suf, _store_suf, frag_scalar) = sgmma_scalar_info(&elem)?;
        let frag_ty = LlType::vec(64, frag_scalar.clone());
        let (row_s, row_d) = self.resolve_stride(pv, 0, 8);
        let (col_s, col_d) = self.resolve_stride(pv, 1, 1);
        let (tile_h, tile_w) = self.sgmma_tile_dims(pv)?;
        let ptr = self.resolve(pv)?;
        let n_bi = op.operands.len().saturating_sub(2);
        let bis: Vec<Operand> = (0..n_bi)
            .map(|i| self.resolve(op.operands[1 + i]))
            .collect::<Option<_>>()?;
        let base = self.sgmma_base_gep(
            ptr,
            &bis,
            row_s,
            &row_d,
            col_s,
            &col_d,
            tile_h,
            tile_w,
            frag_scalar.clone(),
        )?;
        // Stride vectors are the same for every sub-tile (built once).
        let vec0 = self.build_i64x2(row_s, row_d.clone(), 8, None);
        let vec1 = self.build_i64x2(col_s, col_d.clone(), row_s, row_d.clone());
        let mut frags = Vec::with_capacity(nr * nc);
        for r in 0..nr {
            for c in 0..nc {
                let sub = self.sub_tile_gep(
                    base.clone(),
                    r,
                    c,
                    row_s,
                    &row_d,
                    col_s,
                    &col_d,
                    frag_scalar.clone(),
                );
                let f = self.f.call(
                    frag_ty.clone(),
                    format!("@air.simdgroup_matrix_8x8_load.{load_suf}"),
                    vec![sub, vec0.clone(), vec1.clone(), sgmma_zero_strides()],
                )?;
                frags.push(Operand::Val(f));
            }
        }
        self.bind_grid(op_id, 0, frags);
        self.bind(op_id, 1, Operand::undef(LlType::I32)); // token
        Some(())
    }

    /// Multi-fragment (nrxnc) f32 SGMMA tile store.
    fn sgmma_store_multi(&mut self, op_id: OpRef, op: &OpData, nr: usize, nc: usize) -> Option<()> {
        let frags = self.resolve_grid(op.operands[0])?;
        if frags.len() != nr * nc {
            return None;
        }
        let pv = op.operands[1];
        let elem = self.pv_elem(pv)?;
        let (_, _load_suf, store_suf, frag_scalar) = sgmma_scalar_info(&elem)?;
        let (row_s, row_d) = self.resolve_stride(pv, 0, 8);
        let (col_s, col_d) = self.resolve_stride(pv, 1, 1);
        let (tile_h, tile_w) = self.sgmma_tile_dims(pv)?;
        let ptr = self.resolve(pv)?;
        let n_bi = op.operands.len().saturating_sub(3);
        let bis: Vec<Operand> = (0..n_bi)
            .map(|i| self.resolve(op.operands[2 + i]))
            .collect::<Option<_>>()?;
        let base = self.sgmma_base_gep(
            ptr,
            &bis,
            row_s,
            &row_d,
            col_s,
            &col_d,
            tile_h,
            tile_w,
            frag_scalar.clone(),
        )?;
        let vec0 = self.build_i64x2(row_s, row_d.clone(), 8, None);
        let vec1 = self.build_i64x2(col_s, col_d.clone(), row_s, row_d.clone());
        for r in 0..nr {
            for c in 0..nc {
                let sub = self.sub_tile_gep(
                    base.clone(),
                    r,
                    c,
                    row_s,
                    &row_d,
                    col_s,
                    &col_d,
                    frag_scalar.clone(),
                );
                let frag = self.sgmma_frag_to(frags[r * nc + c].clone(), &frag_scalar);
                self.f.call(
                    LlType::Void,
                    format!("@air.simdgroup_matrix_8x8_store.{store_suf}"),
                    vec![frag, sub, vec0.clone(), vec1.clone(), sgmma_zero_strides()],
                );
            }
        }
        self.bind(op_id, 0, Operand::undef(LlType::I32)); // token
        Some(())
    }

    /// Convert one SGMMA fragment to `<64 x elem>` if it is not already.
    ///
    /// A `mmaf` accumulator is always `<64 x float>`, but the store intrinsic is
    /// typed by the destination element - Metal has no float-fragment-to-half
    /// store - so writing f16/bf16 output needs a per-fragment `fptrunc`. The
    /// widening direction is here for symmetry (an f16 fragment stored to an f32
    /// buffer); identity when the types already agree, which is the common case
    /// and emits nothing.
    fn sgmma_frag_to(&mut self, frag: Operand, elem: &LlType) -> Operand {
        let want = LlType::vec(64, elem.clone());
        let have = self.f.type_of(&frag);
        if have == want || !elem.is_float() {
            return frag;
        }
        let (LlType::Vec(_, have_elem), LlType::Vec(_, want_elem)) = (&have, &want) else {
            return frag;
        };
        if !have_elem.is_float() {
            return frag;
        }
        let inst = if lltype_bits(have_elem) > lltype_bits(want_elem) {
            "fptrunc"
        } else {
            "fpext"
        };
        Operand::Val(self.f.cast(inst, frag, want))
    }

    /// Tile (height, width) of a PartitionView, for multi-fragment block-index scaling.
    fn sgmma_tile_dims(&self, pv: ValRef) -> Option<(i64, i64)> {
        match self.ir.value_type(pv) {
            Type::PartitionView(pvt) => Some((
                *pvt.tile_shape.first()? as i64,
                *pvt.tile_shape.get(1)? as i64,
            )),
            _ => None,
        }
    }

    /// Base pointer for a multi-fragment tile: block index scaled by the full tile
    /// dim (`bi0*row*tile_h + bi1*col*tile_w`). No offset for zero indices.
    #[allow(clippy::too_many_arguments)]
    fn sgmma_base_gep(
        &mut self,
        ptr: Operand,
        bis: &[Operand],
        row_s: i64,
        row_d: &Option<Operand>,
        col_s: i64,
        col_d: &Option<Operand>,
        tile_h: i64,
        tile_w: i64,
        elem: LlType,
    ) -> Option<Operand> {
        if bis.is_empty() {
            return Some(ptr);
        }
        for b in bis {
            if !self.is_i32(b) {
                return None;
            }
        }
        // Block index scales by the full tile dim (stride x tile_h/tile_w).
        let off = if bis.len() == 1 {
            self.mul_stride(bis[0].clone(), row_s, row_d, tile_h)
        } else {
            let o0 = self.mul_stride(bis[0].clone(), row_s, row_d, tile_h);
            let o1 = self.mul_stride(bis[1].clone(), col_s, col_d, tile_w);
            Operand::Val(self.f.add(o0, o1))
        };
        Some(Operand::Val(self.f.gep(elem, ptr, off)))
    }

    /// GEP for sub-tile (r, c) within a multi-fragment tile: `(r*row + c*col)*8`.
    #[allow(clippy::too_many_arguments)]
    fn sub_tile_gep(
        &mut self,
        base: Operand,
        r: usize,
        c: usize,
        row_s: i64,
        row_d: &Option<Operand>,
        col_s: i64,
        col_d: &Option<Operand>,
        elem: LlType,
    ) -> Operand {
        // Sub-tile (r,c) is r/c 8x8 tiles into the grid: off = (r*8)*row + (c*8)*col.
        if row_d.is_none() && col_d.is_none() {
            let off = (r as i64 * row_s + c as i64 * col_s) * 8;
            if off == 0 {
                return base;
            }
            return Operand::Val(self.f.gep(elem, base, Operand::i32(off)));
        }
        let o_r = self.mul_stride(Operand::i32(r as i64 * 8), row_s, row_d, 1);
        let o_c = self.mul_stride(Operand::i32(c as i64 * 8), col_s, col_d, 1);
        let off = self.f.add(o_r, o_c);
        Operand::Val(self.f.gep(elem, base, Operand::Val(off)))
    }

    /// SGMMA multiply-accumulate, single (1x1) or multi-fragment. A/B may be
    /// f32/f16/bf16 (same precision); the accumulator C and result are f32. The
    /// mac intrinsic and A/B vector type are chosen from A's element type.
    fn mma_f(&mut self, op_id: OpRef, op: &OpData) -> Option<()> {
        let a_elem = self.sgmma_elem(op.operands[0])?;
        let b_elem = self.sgmma_elem(op.operands[1])?;
        let c_elem = self.sgmma_elem(op.operands[2])?;
        // Normalize each operand to its register class. External types (fp8) are
        // loaded via lane-distribution into a widened `<64 x float>` register, so
        // they MAC as f32 - same as a native f32 tile.
        let a_cls = mma_reg_class(&a_elem);
        let b_cls = mma_reg_class(&b_elem);
        let c_cls = mma_reg_class(&c_elem);
        if c_cls != MmaClass::F32 || a_cls != b_cls {
            return None; // accumulator must be f32; A and B must share precision
        }
        let (_, mac_suf) = match a_cls {
            MmaClass::F32 => ("<64 x float>", "v64f32.v64f32.v64f32.v64f32"),
            MmaClass::F16 => ("<64 x half>", "v64f32.v64f16.v64f16.v64f32"),
            MmaClass::BF16 => ("<64 x bfloat>", "v64f32.v64bf16.v64bf16.v64f32"),
        };
        let (nr_a, nc_a) = crate::types::sgmma_dims(self.ir.value_type(op.operands[0]))?;
        let (_, nc_b) = crate::types::sgmma_dims(self.ir.value_type(op.operands[1]))?;
        let a = self.resolve_grid(op.operands[0])?;
        let b = self.resolve_grid(op.operands[1])?;
        let c = self.resolve_grid(op.operands[2])?;
        // Fragment grids: A is nr_axnc_a, B is nc_axnc_b, C is nr_axnc_b (row-major).
        let (nr, nc, ks) = (nr_a, nc_b, nc_a);
        if a.len() != nr_a * nc_a || b.len() != nc_a * nc_b || c.len() != nr * nc {
            return None;
        }
        let mut result = Vec::with_capacity(nr * nc);
        for r in 0..nr {
            for col in 0..nc {
                // C[r,col] = sum_k SGMMA(A[r,k], B[k,col], acc)
                let mut acc = c[r * nc + col].clone();
                for k in 0..ks {
                    let a_frag = a[r * nc_a + k].clone();
                    let b_frag = b[k * nc_b + col].clone();
                    let res = self.f.call(
                        vec64f(),
                        format!("@air.simdgroup_matrix_8x8_multiply_accumulate.{mac_suf}"),
                        vec![a_frag, b_frag, acc],
                    )?;
                    acc = Operand::Val(res);
                }
                result.push(acc);
            }
        }
        self.bind_grid(op_id, 0, result);
        Some(())
    }

    /// SGMMA tile element type (F32/F16/BF16 ...) of a value, or None if it isn't an
    /// SGMMA-shaped tile. Used for mma operands (which are tiles).
    fn sgmma_elem(&self, v: ValRef) -> Option<ScalarType> {
        let ty = self.ir.value_type(v);
        crate::types::sgmma_dims(ty).map(|_| scalar_of(ty))
    }

    /// Element type of a PartitionView (for SGMMA load/store), or None otherwise.
    fn pv_elem(&self, pv: ValRef) -> Option<ScalarType> {
        match self.ir.value_type(pv) {
            Type::PartitionView(pvt) => Some(pvt.tensor_view.element_type.clone()),
            _ => None,
        }
    }

    /// As `elem_gep`, but stepping the pointer in units of `elem` (e.g. `i8` for
    /// fp8 storage). `flat = block_idx * n_elems + lane`, with the lane term
    /// inferred from the tile extent - see [`Self::lane_term`].
    fn elem_gep_ty(
        &mut self,
        ptr: Operand,
        block_idx: Operand,
        n_elems: i32,
        elem: LlType,
    ) -> Operand {
        let base = self.f.mul(block_idx, Operand::i32(n_elems as i64));
        let flat = match self.lane_term(n_elems) {
            Some(lane) => Operand::Val(self.f.add(Operand::Val(base), lane)),
            None => Operand::Val(base),
        };
        Operand::Val(self.f.gep(elem, ptr, flat))
    }

    /// The affine lane term for a tile of `n_elems` elements: which thread of
    /// the threadgroup holds tile element 0, 1, ... .
    ///
    /// The tile model has no vocabulary for *who holds this element*, so the
    /// binding has to come from somewhere; it comes from the tile extent, which
    /// is static:
    ///
    /// - extent 1: no lane term at all, i.e. a kernel-uniform load. A scalar
    ///   parameter is one element in memory, not one copy per lane.
    /// - extent 2..=32 (power of two): `tp.x & (extent-1)`. A tile that size is
    ///   SIMD-group-resident, which is what the SIMD reductions already assume -
    ///   `reduce_sum` on a `[1,32]` tile lowers to `air.simd_sum`, meaningful
    ///   only if those 32 elements are one SIMD group. The mask is what lets a
    ///   threadgroup hold several independent tile-processors, so `load([i])`
    ///   means tile `i` in every SIMD group with no `- simd_group_id` fix-up.
    /// - extent > 32: plain `tp.x` - the cuTile tile-block-per-tile case, where
    ///   one tile spans the threadgroup and the thread position already IS the
    ///   element index.
    ///
    /// THE >32 CUTOFF is the one part not derived from semantics: masking a wide tile is
    /// a no-op when the extent equals the threadgroup width, which the compiler cannot see
    /// (dispatch size is a runtime argument), so the mask would be dead work.
    ///
    /// SCOPE, easy to get wrong: this affects only kernels addressing through
    /// `elem_gep_ty`/`strided_elem_gep`. GEMM does NOT - `load_view` takes the SGMMA
    /// fragment path and the MPP emitter never touches the thread position.
    ///
    /// ACCEPTED LIMITATION: a 33..threadgroup-width tile in a WIDER threadgroup needs
    /// masking and will not get it; no kernel does this, and such a tile could not use the
    /// SIMD reductions anyway. Non-power-of-two extents keep unmasked `tp.x` over a `urem`.
    fn lane_term(&mut self, n_elems: i32) -> Option<Operand> {
        if n_elems <= 1 {
            return None;
        }
        let tp = self.tp.clone();
        let x = self.f.extractelement(tp, Operand::i32(0));
        if n_elems > 32 || !(n_elems as u32).is_power_of_two() {
            return Some(Operand::Val(x));
        }
        Some(Operand::Val(
            self.f
                .and(Operand::Val(x), Operand::i32(n_elems as i64 - 1)),
        ))
    }

    /// Element count of a PartitionView tile regardless of element type.
    fn partition_elem_count(&self, pv: ValRef) -> Option<i32> {
        match self.ir.value_type(pv) {
            Type::PartitionView(pvt) => Some(pvt.tile_shape.iter().product()),
            _ => None,
        }
    }

    /// Resolve the `rank` per-dim index operands of a view op starting at operand
    /// `start` (load: view at 0, index at 1..; store: tile,view at 0,1, index at 2..).
    fn resolve_index_ops(&mut self, op: &OpData, pv: ValRef, start: usize) -> Option<Vec<Operand>> {
        let rank = match self.ir.value_type(pv) {
            Type::PartitionView(pvt) => pvt.tile_shape.len(),
            _ => return None,
        };
        let mut ops = Vec::with_capacity(rank);
        for d in 0..rank {
            let o = self.resolve(*op.operands.get(start + d)?)?;
            if !self.is_i32(&o) {
                return None;
            }
            ops.push(o);
        }
        Some(ops)
    }

    /// Stride-aware element GEP: `addr = sum_d index[d]*tile_shape[d]*stride[d] +
    /// lane*stride_innermost`, using the view's per-dim strides. Unlike
    /// `elem_gep_ty` (flat `block_idx*n_elems`), this addresses rows whose stride
    /// is not a multiple of the tile width.
    fn strided_elem_gep(
        &mut self,
        ptr: Operand,
        pv: ValRef,
        index_ops: &[Operand],
        elem: LlType,
    ) -> Option<Operand> {
        let off = self.strided_elem_offset(pv, index_ops)?;
        Some(Operand::Val(self.f.gep(elem, ptr, off)))
    }

    /// Per-dim absolute coordinates of this lane's element: `index[d]*tile[d] + c_d`,
    /// where `c_d` is the lane's own coordinate inside the tile. Same decomposition as
    /// `strided_elem_offset` - that one folds the coordinates straight into an address,
    /// this one keeps them so they can be compared against the tensor extent.
    fn elem_coords(&mut self, pv: ValRef, index_ops: &[Operand]) -> Option<Vec<Operand>> {
        let tile_shape: Vec<i32> = match self.ir.value_type(pv) {
            Type::PartitionView(pvt) => pvt.tile_shape.clone(),
            _ => return None,
        };
        let rank = index_ops.len();
        let n_elems = self.partition_elem_count(pv)?;
        let mut coords: Vec<Option<Operand>> = vec![None; rank];
        if let Some(lane) = self.lane_term(n_elems) {
            let mut running = lane;
            let mut remaining: i64 = (0..rank)
                .map(|d| *tile_shape.get(d).unwrap_or(&1) as i64)
                .product();
            for d in (0..rank).rev() {
                let extent = *tile_shape.get(d).unwrap_or(&1) as i64;
                if extent <= 1 {
                    continue; // c_d is always 0
                }
                let coord = if remaining == extent {
                    running.clone()
                } else {
                    Operand::Val(self.f.urem(running.clone(), Operand::i32(extent)))
                };
                coords[d] = Some(coord);
                remaining /= extent;
                if remaining > 1 {
                    running = Operand::Val(self.f.udiv(running, Operand::i32(extent)));
                }
            }
        }
        let mut out = Vec::with_capacity(rank);
        for (d, idx) in index_ops.iter().enumerate() {
            let tile_d = *tile_shape.get(d).unwrap_or(&1) as i64;
            let base = if tile_d == 1 {
                idx.clone()
            } else {
                Operand::Val(self.f.mul(idx.clone(), Operand::i32(tile_d)))
            };
            out.push(match coords[d].clone() {
                None => base,
                Some(c) => Operand::Val(self.f.add(base, c)),
            });
        }
        Some(out)
    }

    /// Per-dim bounds for a padded view: `(dim, extent)` for each dim worth checking.
    ///
    /// Masking is opt-in through the view's `padding` mode, exactly as upstream models it:
    /// a view declared `padding::None` promises every access is in bounds and stays a bare
    /// load/store. Two kinds of bound survive that gate. A STATIC extent is only worth
    /// checking where the tiling is actually ragged - our kernels routinely declare shapes
    /// that are not the addressing domain (`bcast_elem` indexes dim 0 of a `[1, K]` tensor
    /// far past extent 1; `tcolsum_reduce` reads a `[1, BN*32]` tile out of 32 columns),
    /// and masking those to zero is silent corruption. A DYNAMIC one is the tensor's real
    /// extent passed in as a kernel argument - the usual case for an output, whose declared
    /// shape is its TILE shape and not the tensor's.
    fn dim_bounds(&mut self, pv: ValRef, rank: usize) -> Vec<(usize, Operand)> {
        let (shape, padding, tile_shape) = match self.ir.value_type(pv) {
            Type::PartitionView(pvt) => (
                pvt.tensor_view.shape.clone(),
                pvt.padding_value,
                pvt.tile_shape.clone(),
            ),
            _ => return Vec::new(),
        };
        if padding.is_none() {
            return Vec::new();
        }
        let mut out = Vec::new();
        for d in 0..rank {
            let Some(&ext) = shape.get(d) else { continue };
            if ext == wax_ir::DYNAMIC {
                if let Some(v) = self.find_shape_val_for_pv(pv, d)
                    && let Some(op) = self.resolve(v)
                {
                    out.push((d, op));
                }
                continue;
            }
            let t = *tile_shape.get(d).unwrap_or(&1) as i64;
            if ext > 0 && t > 0 && ext > t && ext % t != 0 {
                out.push((d, Operand::i32(ext)));
            }
        }
        out
    }

    /// True for lanes whose element lies inside the tensor, or `None` when this access
    /// needs no mask.
    fn bounds_predicate(&mut self, pv: ValRef, index_ops: &[Operand]) -> Option<Operand> {
        // Decide BEFORE emitting anything: `elem_coords` materializes the lane
        // decomposition, and leaving that behind on an access that needs no mask is not
        // free - dead arithmetic in a loop body costs the unroll heuristic its
        // straight-line body and the loop its `!llvm.loop` hint.
        let bounds = self.dim_bounds(pv, index_ops.len());
        if bounds.is_empty() {
            return None;
        }
        let coords = self.elem_coords(pv, index_ops)?;
        let mut pred: Option<Operand> = None;
        for (d, ext) in bounds {
            let ok = Operand::Val(self.f.icmp("slt", coords.get(d)?.clone(), ext));
            pred = Some(match pred {
                None => ok,
                Some(p) => Operand::Val(self.f.and(p, ok)),
            });
        }
        pred
    }

    /// The scalar this view pads out-of-bounds reads with, as LLVM constant text plus the
    /// raw bit pattern at the element width - the two emitters need different encodings,
    /// the same split `reduce_identity` makes.
    fn padding_constant(&self, pv: ValRef, elem: &LlType) -> Option<Operand> {
        use wax_ir::PaddingValue;
        let pad = match self.ir.value_type(pv) {
            Type::PartitionView(pvt) => pvt.padding_value?,
            _ => return None,
        };
        let (text, bits): (&str, u64) = match (elem, pad) {
            (LlType::Float, PaddingValue::Zero) => ("0.0", 0),
            (LlType::Float, PaddingValue::NegZero) => ("-0.0", 0x8000_0000),
            (LlType::Float, PaddingValue::Nan) => ("0x7FF8000000000000", 0x7FC0_0000),
            (LlType::Float, PaddingValue::PosInf) => ("0x7FF0000000000000", 0x7F80_0000),
            (LlType::Float, PaddingValue::NegInf) => ("0xFFF0000000000000", 0xFF80_0000),
            (LlType::Half, PaddingValue::Zero) => ("0xH0000", 0),
            (LlType::Half, PaddingValue::NegZero) => ("0xH8000", 0x8000),
            (LlType::Half, PaddingValue::Nan) => ("0xH7E00", 0x7E00),
            (LlType::Half, PaddingValue::PosInf) => ("0xH7C00", 0x7C00),
            (LlType::Half, PaddingValue::NegInf) => ("0xHFC00", 0xFC00),
            (LlType::BFloat, PaddingValue::Zero) => ("0xR0000", 0),
            (LlType::BFloat, PaddingValue::NegZero) => ("0xR8000", 0x8000),
            (LlType::BFloat, PaddingValue::Nan) => ("0xR7FC0", 0x7FC0),
            (LlType::BFloat, PaddingValue::PosInf) => ("0xR7F80", 0x7F80),
            (LlType::BFloat, PaddingValue::NegInf) => ("0xRFF80", 0xFF80),
            // Integer tiles (index/mask tensors) only have a zero to pad with.
            (LlType::I32, PaddingValue::Zero) => return Some(Operand::i32(0)),
            _ => return None,
        };
        Some(Operand::float_bits(text, elem.clone(), bits))
    }

    /// The element offset `strided_elem_gep` indexes by, before the GEP. Split out so
    /// the padding path can clamp it: an out-of-bounds lane must not form the address at
    /// all, or the load faults.
    fn strided_elem_offset(&mut self, pv: ValRef, index_ops: &[Operand]) -> Option<Operand> {
        let tile_shape: Vec<i32> = match self.ir.value_type(pv) {
            Type::PartitionView(pvt) => pvt.tile_shape.clone(),
            _ => return None,
        };
        let rank = index_ops.len();
        // The lane is a FLAT row-major index into the tile; decompose it into per-dim
        // tile coordinates and pay each one its own stride:
        //
        //     c_k = lane % d_k,  c_{k-1} = (lane / d_k) % d_{k-1},  ...
        //     offset = sum_j c_j * stride[j]
        //
        // Multiplying the whole lane by the innermost stride - what this used to do -
        // is only correct when the tile covers a CONTIGUOUS block, i.e. when every
        // outer tile extent is 1. It silently read the wrong elements for any tile
        // spanning rows that are farther apart than the tile is wide, which is what a
        // threadgroup-spanning `[rows, 32]` accumulator needs.
        //
        // Extent-1 dims contribute nothing and are skipped, so the common `[1, B]`
        // tile still emits exactly `lane * stride[inner]` and nothing changes for it.
        let n_elems = self.partition_elem_count(pv)?;
        let mut acc = Operand::i32(0);
        if let Some(lane) = self.lane_term(n_elems) {
            let mut running = lane;
            // Elements still to be split off, innermost-first.
            let mut remaining: i64 = (0..rank)
                .map(|d| *tile_shape.get(d).unwrap_or(&1) as i64)
                .product();
            for d in (0..rank).rev() {
                let extent = *tile_shape.get(d).unwrap_or(&1) as i64;
                if extent <= 1 {
                    continue; // c_d is always 0
                }
                // The modulo is redundant once nothing outer is left to strip.
                let coord = if remaining == extent {
                    running.clone()
                } else {
                    Operand::Val(self.f.urem(running.clone(), Operand::i32(extent)))
                };
                let (s, sd) = self.resolve_stride(pv, d, 1);
                let term = self.mul_stride(coord, s, &sd, 1);
                acc = if matches!(acc, Operand::Imm { .. }) {
                    term
                } else {
                    Operand::Val(self.f.add(acc, term))
                };
                remaining /= extent;
                if remaining > 1 {
                    running = Operand::Val(self.f.udiv(running, Operand::i32(extent)));
                }
            }
        }
        for (d, idx) in index_ops.iter().enumerate() {
            let (s, sd) = self.resolve_stride(pv, d, 1);
            let tile_d = *tile_shape.get(d).unwrap_or(&1) as i64;
            let term = self.mul_stride(idx.clone(), s, &sd, tile_d);
            acc = Operand::Val(self.f.add(acc, term));
        }
        Some(acc)
    }

    fn is_i32(&self, o: &Operand) -> bool {
        matches!(o, Operand::Imm { ty, .. } if *ty == LlType::I32)
            || matches!(o, Operand::Val(id) if self.f.value_type(*id) == LlType::I32)
    }

    /// True for an integer operand (i1 or i32) - bool logic uses i1.
    fn is_int(&self, o: &Operand) -> bool {
        let ty = match o {
            Operand::Imm { ty, .. } => ty.clone(),
            Operand::Val(id) => self.f.value_type(*id),
        };
        ty == LlType::I1 || ty == LlType::I32
    }
}

enum IntOp {
    Add,
    Sub,
    Mul,
    SDiv,
    SRem,
    And,
    Or,
    Xor,
    Shl,
    /// `>>` on an unsigned element: logical, zero-filling.
    LShr,
    /// `>>` on a signed element: arithmetic, sign-filling.
    AShr,
}

/// The register class an SGMMA MAC operand lowers to. fp8 (and other External
/// types) load into a widened `<64 x float>` register, so they MAC as f32.
#[derive(PartialEq, Eq, Clone, Copy)]
enum MmaClass {
    F32,
    F16,
    BF16,
}

/// Register class for an SGMMA tile's element type, or None if unsupported.
fn mma_reg_class(elem: &ScalarType) -> MmaClass {
    match elem {
        ScalarType::F16 => MmaClass::F16,
        ScalarType::BF16 => MmaClass::BF16,
        _ => MmaClass::F32, // f32, tf32, and External (fp8, widened to f32)
    }
}

/// The `<64 x float>` type of a single 8x8 SGMMA register.
/// The trailing `<2 x i64> zeroinitializer` every SGMMA load/store takes.
fn sgmma_zero_strides() -> Operand {
    Operand::zeroinit(LlType::vec(2, LlType::I64))
}

/// The wider of two float types, or `None` if either is not a float (a genuine type
/// error the caller reports rather than papering over).
fn wider_float(a: &LlType, b: &LlType) -> Option<LlType> {
    let rank = |t: &LlType| match t {
        LlType::Half | LlType::BFloat => Some(1),
        LlType::Float => Some(2),
        LlType::Double => Some(3),
        _ => None,
    };
    let (ra, rb) = (rank(a)?, rank(b)?);
    Some(if ra >= rb { a.clone() } else { b.clone() })
}

fn vec64f() -> LlType {
    LlType::vec(64, LlType::Float)
}

/// Retarget an `@air.simd_*.f32` kind token to a typed intrinsic name for `suf`
/// (`"f32"`/`"f16"`/`"bf16"`), e.g. `@air.simd_sum.f32` + `f16` -> `@air.simd_sum.f16`.
fn simd_intrinsic(kind: &str, suf: &str) -> String {
    let base = kind.strip_suffix(".f32").unwrap_or(kind);
    format!("{base}.{suf}")
}

/// Identity literal for a reduce/scan combine op (fills unused lanes/slots),
/// typed per element: sum->0, product->1, max->-inf, min->+inf. +/-inf uses the
/// type-specific hex (double for f32, `0xH` for half, `0xR` for bfloat).
/// Identity value for a cross-SIMD reduction, as both LLVM constant text and the raw
/// bit pattern at the reduced type's width (the text emitter needs the former, the
/// bitcode emitter the latter). Note the text for f32 is LLVM's *double* hex form
/// while the bits are the 32-bit pattern - they are different encodings of the same
/// value, which is exactly why the structured form is carried separately.
fn reduce_identity(kind: &str, suf: &str) -> (&'static str, u64) {
    let is_max = kind == super::AIR_SIMD_MAX_F32;
    let is_min = kind == super::AIR_SIMD_MIN_F32;
    let is_prod = kind == super::AIR_SIMD_PRODUCT_F32;
    match suf {
        "f16" if is_max => ("0xHFC00", 0xFC00),
        "f16" if is_min => ("0xH7C00", 0x7C00),
        "bf16" if is_max => ("0xRFF80", 0xFF80),
        "bf16" if is_min => ("0xR7F80", 0x7F80),
        _ if is_max => ("0xFFF0000000000000", 0xFF80_0000), // -inf
        _ if is_min => ("0x7FF0000000000000", 0x7F80_0000), // +inf
        _ if is_prod => ("1.0", 0x3F80_0000),
        _ => ("0.0", 0), // sum
    }
}

/// Element scalar type of a device-pointer kernel param. Unlike `scalar_of`,
/// this handles `Type::Pointer` and pointer tiles (raw tensor args), so a bf16
/// tensor param is typed `bfloat*` rather than defaulting to `float*`. External
/// dtypes (e.g. fp8) resolve to their STORAGE scalar so the pointer is typed by
/// its in-memory representation (`i8*`), not the compute type.
fn param_elem(ty: &Type) -> ScalarType {
    let e = match ty {
        Type::Pointer(p) => p.pointee.clone(),
        Type::TensorView(tv) => tv.element_type.clone(),
        Type::PartitionView(pv) => pv.tensor_view.element_type.clone(),
        Type::Tile(TileType {
            element_type: TileElementType::Pointer(p),
            ..
        }) => p.pointee.clone(),
        other => scalar_of(other),
    };
    match e {
        ScalarType::External(t) => t.storage_type(),
        other => other,
    }
}

/// SGMMA per-element info for an f32/f16/bf16 tile:
/// `(elem_ll, load_intrinsic_suffix, store_intrinsic_suffix, fragment_scalar)`.
/// Loads produce and stores consume a `<64 x {elem}>` fragment - Metal has no
/// mixed store (`simdgroup_store(simdgroup_float8x8, half*)` does not exist), so
/// an f32 accumulator is `fptrunc`ed per fragment before an f16/bf16 store.
/// None for other types.
fn sgmma_scalar_info(
    elem: &ScalarType,
) -> Option<(&'static str, &'static str, &'static str, LlType)> {
    match elem {
        ScalarType::F32 => Some(("float", "v64f32.p1f32", "v64f32.p1f32", LlType::Float)),
        ScalarType::F16 => Some(("half", "v64f16.p1f16", "v64f16.p1f16", LlType::Half)),
        ScalarType::BF16 => Some(("bfloat", "v64bf16.p1bf16", "v64bf16.p1bf16", LlType::BFloat)),
        _ => None,
    }
}

/// LLVM float type + AIR intrinsic type-suffix for a float scalar. None for
/// non-float types. This is the single mapping that makes the elementwise /
/// reduce / scan ops polymorphic over f32 / f16 / bf16.
/// Element count of a tile type, for the extent-derived lane term.
fn tile_elem_count(ty: &Type) -> Option<i32> {
    match ty {
        Type::Tile(t) => i32::try_from(t.shape.iter().copied().product::<i64>()).ok(),
        _ => None,
    }
}

fn float_elem(s: &ScalarType) -> Option<(LlType, &'static str)> {
    match s {
        ScalarType::F32 => Some((LlType::Float, "f32")),
        ScalarType::F16 => Some((LlType::Half, "f16")),
        ScalarType::BF16 => Some((LlType::BFloat, "bf16")),
        _ => None,
    }
}

/// LLVM type for a tensor element that is loaded/stored to memory verbatim -
/// native float (f32/f16/bf16) or native integer (i8/i16/i32/i64, covering
/// u8/u16/u32/u64 index/mask tensors). Returns None for types that
/// need a conversion on load (External/fp8), which take a dedicated path, and
/// for i1 (bool tensors are byte-backed and arrive as i8).
fn mem_elem_lltype(s: &ScalarType) -> Option<LlType> {
    match s {
        ScalarType::F32 => Some(LlType::Float),
        ScalarType::F16 => Some(LlType::Half),
        ScalarType::BF16 => Some(LlType::BFloat),
        ScalarType::I8 => Some(LlType::I8),
        ScalarType::I16 => Some(LlType::I16),
        ScalarType::I32 => Some(LlType::I32),
        ScalarType::I64 => Some(LlType::I64),
        _ => None,
    }
}

/// Storage + compute scalar types of an External dtype; None if not External.
fn external_types(elem: &ScalarType) -> Option<(ScalarType, ScalarType)> {
    match elem {
        ScalarType::External(t) => Some((t.storage_type(), t.compute_type())),
        _ => None,
    }
}

/// AIR atomic memory scope from the compiler's scope attr (0=TileBlock->2,
/// 2=System->4, else Device->2). Mirrors the legacy emitter.
fn atomic_scope(compiler_scope: i64) -> i32 {
    match compiler_scope {
        0 => 1, // TileBlock -> threadgroup
        2 => 4, // System
        _ => 2, // Device (default)
    }
}

/// AIR type suffix for an atomic element LLVM type.
fn atomic_air_ty(elem_ll: &str) -> &'static str {
    match elem_ll {
        "float" => "f32",
        "half" => "f16",
        "bfloat" => "bf16",
        "i64" => "i64",
        _ => "i32",
    }
}

/// AIR atomic RMW intrinsic name + air-type suffix: `@air.atomic.{space}.{op}.{ty}`.
fn atomic_rmw_intrinsic(mode: i64, elem_ll: &str, asn: u32) -> (String, &'static str) {
    let mem_space = if asn == 3 { "local" } else { "global" };
    let is_float = matches!(elem_ll, "float" | "half" | "bfloat");
    let air_ty = atomic_air_ty(elem_ll);
    let op_part = match mode {
        0 => "and.s",
        1 => "or.s",
        2 => "xor.s",
        3 => "add.s", // integer signed add
        4 => "add",   // float add (no sign)
        5 if !is_float => "max.s",
        6 if !is_float => "min.s",
        7 => "max.u",
        8 => "min.u",
        _ => "add.s",
    };
    (
        format!("@air.atomic.{mem_space}.{op_part}.{air_ty}"),
        air_ty,
    )
}

/// LLVM element type string for an atomic air-type suffix.
fn atomic_ll_elem(air_ty: &str) -> &'static str {
    match air_ty {
        "f32" => "float",
        "f16" => "half",
        "bf16" => "bfloat",
        "i64" => "i64",
        _ => "i32",
    }
}

/// LlType for an atomic air-type suffix (for the raw result's recorded type).
fn atomic_lltype(air_ty: &str) -> LlType {
    match air_ty {
        "f32" => LlType::Float,
        "f16" => LlType::Half,
        "bf16" => LlType::BFloat,
        "i64" => LlType::I64,
        _ => LlType::I32,
    }
}

/// Bit width of a scalar LLVM type (for choosing widen vs narrow casts).
fn lltype_bits(t: &LlType) -> u32 {
    match t {
        LlType::I1 => 1,
        LlType::I8 => 8,
        LlType::I16 | LlType::Half | LlType::BFloat => 16,
        LlType::I32 | LlType::Float => 32,
        LlType::I64 | LlType::Double => 64,
        _ => 0,
    }
}

/// Fragment LLVM types of a loop carry: one `<64 x float>` per fragment for an
/// f32 SGMMA accumulator (nr*nc of them), or a single scalar type otherwise.
/// `None` for mixed-precision SGMMA or unsupported scalars.
fn carry_fragment_types(ty: &Type) -> Option<Vec<LlType>> {
    match crate::types::sgmma_dims(ty) {
        Some((nr, nc)) => {
            if scalar_of(ty) != ScalarType::F32 {
                return None; // mixed-precision SGMMA carry -> not this cut
            }
            Some(vec![vec64f(); nr * nc])
        }
        None => Some(vec![scalar_lltype(&scalar_of(ty))?]),
    }
}

/// True if a Constant op's value is entirely zero bytes (a zero splat).
fn constant_is_zero(op: &OpData) -> bool {
    op.attributes
        .iter()
        .find(|(k, _)| k == "value")
        .is_some_and(|(_, v)| match v {
            Attribute::DenseElements(de) => de.data.iter().all(|&b| b == 0),
            _ => false,
        })
}

/// Element count (row width) of a tile type, for threadgroup-scratch indexing.
/// Element pitch between consecutive `@_tg_user_scratch` rows holding `width`-wide
/// tiles: one float of padding past the row.
///
/// The padding exists for `tg_load_transposed`. Apple threadgroup memory is banked
/// every 4 bytes (32 banks), so with an unpadded pitch of 32 a column read
/// (`lane*32 + slot`) puts all 32 lanes on the same bank and serializes 32 ways -
/// measured at ~0.2 us per such load with a 1024-thread threadgroup, which
/// dominated the SDPA merge. An odd pitch maps lane `L` to bank `(L*33 + slot) % 32
/// = (L + slot) % 32`, so all 32 lanes hit distinct banks. Row stores stay
/// conflict-free either way (consecutive lanes, consecutive addresses).
const fn tg_row_pitch(width: i32) -> i32 {
    width + 1
}

/// Default size of `@_tg_user_scratch` in floats when a kernel does not request one:
/// 128 rows (4 tiles x 32 SIMD groups) at the padded pitch. Kernels that touch fewer
/// rows should call `KernelCompiler::tg_scratch_rows` - threadgroup memory is a
/// per-core budget and an oversized declaration costs co-residency with
/// concurrently dispatched kernels.
const TG_SCRATCH_FLOATS_DEFAULT: i32 = 128 * tg_row_pitch(32);

/// Natural memory alignment (bytes) of a scalar LLVM type.
fn align_of(ty: &LlType) -> u32 {
    match ty {
        LlType::Half | LlType::BFloat | LlType::I16 => 2,
        LlType::I64 | LlType::Double => 8,
        LlType::I8 | LlType::I1 => 1,
        // A `<N x T>` vector's natural alignment is N*align(T) (16 for <4 x float>).
        LlType::Vec(n, e) => *n * align_of(e),
        _ => 4, // f32 / i32
    }
}

/// A `T addrspace(3)*` (threadgroup memory) pointer type.
fn shared_ptr(pointee: LlType) -> LlType {
    LlType::Ptr {
        pointee: Box::new(pointee),
        addrspace: 3,
    }
}

/// Map a cutile scalar type to the CFG builder's LLVM type (cut-1 subset).
fn scalar_lltype(s: &ScalarType) -> Option<LlType> {
    Some(match s {
        ScalarType::I1 => LlType::I1,
        ScalarType::I8 => LlType::I8,
        ScalarType::I16 => LlType::I16,
        ScalarType::I32 => LlType::I32,
        ScalarType::I64 => LlType::I64,
        ScalarType::F16 => LlType::Half,
        ScalarType::BF16 => LlType::BFloat,
        ScalarType::F32 => LlType::Float,
        ScalarType::F64 => LlType::Double,
        // Sub-byte float storage types are backed by an i8 in memory.
        ScalarType::F8E4M3FN
        | ScalarType::F8E5M2
        | ScalarType::F8E8M0FNU
        | ScalarType::F4E2M1FN => LlType::I8,
        _ => return None,
    })
}

/// Element scalar of a tile-or-scalar constant type (for `scalar_lltype`).
#[allow(dead_code)]
fn tile_scalar(t: &TileElementType) -> Option<&ScalarType> {
    match t {
        TileElementType::Scalar(s) => Some(s),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::loop_unroll_factor;

    #[test]
    fn tiny_straight_line_body_unrolls_by_eight() {
        // GEMV's dot-product body is ~20 instructions -> 160/20 = 8.
        assert_eq!(loop_unroll_factor(true, 20), Some(8));
        assert_eq!(loop_unroll_factor(true, 4), Some(8)); // clamp upper
    }

    #[test]
    fn medium_body_scales_factor_down() {
        assert_eq!(loop_unroll_factor(true, 30), Some(5)); // 160/30 = 5
        assert_eq!(loop_unroll_factor(true, 32), Some(5)); // at the size cap
    }

    #[test]
    fn large_or_branchy_body_stays_rolled() {
        assert_eq!(loop_unroll_factor(true, 33), None); // over size cap
        assert_eq!(loop_unroll_factor(false, 8), None); // nested control flow
        assert_eq!(loop_unroll_factor(true, 0), None); // empty
    }
}
