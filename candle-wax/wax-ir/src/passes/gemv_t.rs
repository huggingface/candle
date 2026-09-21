//! GEMV_T, GEMM with `M = 1`: `C[1, N] = A[1, K] @ B[K, N]`.
//!
//! GEMM is shape-generic on `M`, so this pass is only applicable when `M` is provided (JIT).
//!
//! If the GEMM matches `A` is viewed as `[K, lanes]`, the accumulator is re-tiled to `[SG, lanes]`,
//! the `mma_f` is swapped with per-lane `mul`+`add` and a group `reduce`. All in wax IR.
//!
//! [`GemvTPass`] rewrites in place. Splitting K is a separate pass ([`super::split_k`]).

use super::gemm_analysis::{GemmAnalysis, GemmPattern};
pub use super::launch::{Launch, Tile, TileBlock};
use super::utils::{
    assume_bounded, binf, bounded, const_i32, const_scalar, convert, emit_at, i32, load_view,
    make_token, partition_view, store_view, tensor_view, terminator, tile, tile_block_id,
};
use crate::attr::{Attribute, FloatBits};
use crate::build::{BlockId, OpId, Value};
use crate::builder::{OpBuilder, build_block_in};
use crate::location::Location;
use crate::opcode::Opcode;
use crate::types::{PaddingValue, ScalarType, TensorViewType, Type};
use pliron::context::Context;
use pliron::irbuild::IRStatus;
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::match_rewrite::{
    MatchRewrite, MatchRewriter, RewriterOrder, apply_match_rewrite,
};
use pliron::irbuild::rewriter::Rewriter;
use pliron::pass::{AnalysisManager, Pass, PassResult};

/// The specialisation's inputs. None is in the IR.
pub struct Facts {
    /// Pass fires only for `M = 1`.
    pub m: i64,
    pub n: i64,
    /// SIMD-group count chosen by caller. Not derived as `wax-ir` does not see the backend.
    pub sg: i32,
    /// SIMD width. Lanes in lockstep / columns per tile. Also not derived for same reason.
    pub lanes: i64,
}

/// The GEMV realisation as a [`Pass`]. It rewrites the recognised GEMM in place rather than
/// as a fresh module. Parameter list and caller bindings are untouched.
///
/// Single-pass only. Split-K is a later pass ([`super::split_k`]) based on the same stamp
/// this one carries over.
pub struct GemvTPass {
    facts: Facts,
}

impl GemvTPass {
    pub fn new(facts: Facts) -> Self {
        GemvTPass { facts }
    }
}

impl Pass for GemvTPass {
    fn name(&self) -> &str {
        "wax-gemv"
    }

    fn run(
        &mut self,
        op: OpId,
        ctx: &mut Context,
        analyses: &mut AnalysisManager,
    ) -> pliron::result::Result<PassResult> {
        let pattern = analyses
            .get_analysis::<GemmAnalysis>(op, ctx)?
            .pattern
            .clone();
        let Ok(p) = pattern else {
            return Ok(PassResult::default());
        };
        if !applies(&p, &self.facts) {
            return Ok(PassResult::default());
        }
        // The three buffers, in signature order. The by-value scalars between them are unused.
        let buffers = buffer_args(ctx, op);
        let [out, x, y] = buffers[..] else {
            return Ok(PassResult::default());
        };
        // The store's token is the end of the chain
        if p.store.deref(ctx).get_num_results() > 0 && p.store.deref(ctx).get_result(0).is_used(ctx)
        {
            return Ok(PassResult::default());
        }

        let mut rw = GemvRewrite {
            p,
            facts: &self.facts,
            out,
            x,
            y,
            rewrote: false,
        };
        apply_match_rewrite(ctx, &mut rw, RewriterOrder::default(), op)?;
        if !rw.rewrote {
            return Ok(PassResult::default());
        }

        // Launch geometry stamped on the IR
        launch_for(&self.facts, Layout::of(&rw.p, &self.facts)).stamp(ctx, op);

        // Report change.
        let mut res = PassResult::default();
        res.ir_changed = IRStatus::Changed;
        Ok(res)
    }
}

/// Whether the realisation applies.
fn applies(p: &GemmPattern, facts: &Facts) -> bool {
    if facts.m != 1 || facts.sg < 1 || facts.lanes < 1 {
        return false;
    }
    // The wax gemv_t kernel tiles N in lane-wide chunks. `N < lanes` does not apply.
    if facts.n < facts.lanes {
        return false;
    }
    let k = p.k();
    if k <= 0 || k % (facts.sg as i64) != 0 {
        return false;
    }
    // TODO: Supported types can be extended.
    let ok = |t: &ScalarType| matches!(t, ScalarType::F32 | ScalarType::F16 | ScalarType::BF16);
    ok(&p.elem_t) && ok(&p.dst_t)
}

/// One tile block per output-column block (`lanes` wide under `Nn`, `sg` under `Nt`).
fn launch_for(facts: &Facts, layout: Layout) -> Launch {
    let n = facts.n;
    let per_block = match layout {
        Layout::Nn => facts.lanes,
        Layout::Nt => facts.sg as i64,
    };
    Launch {
        block: TileBlock {
            x: (n.div_euclid(per_block) + (n % per_block != 0) as i64) as usize,
            y: 1,
            z: 1,
        },
        tile: Tile(vec![facts.sg as usize, facts.lanes as usize]),
    }
}

/// The buffer parameters, in signature order.
/// Note that a rank-0 scalar tile is a by value extent or stride, not a buffer, and is excluded.
fn buffer_args(ctx: &Context, func: OpId) -> Vec<Value> {
    use pliron::linked_list::ContainsLinkedList;
    use pliron::r#type::Typed;
    let Some(region) = func.deref(ctx).regions().next() else {
        return Vec::new();
    };
    let Some(entry) = region.deref(ctx).iter(ctx).next() else {
        return Vec::new();
    };
    entry
        .deref(ctx)
        .arguments()
        .filter(|v| {
            matches!(
                crate::dialect::types::from_pliron(ctx, v.get_type(ctx)),
                Some(Type::Tile(t))
                    if t.shape.is_empty()
                        && matches!(t.element_type, crate::types::TileElementType::Pointer(_))
            )
        })
        .collect()
}

/// Matches the recognised GEMM by identity and replaces it with a GEMV_T.
struct GemvRewrite<'a> {
    p: GemmPattern,
    facts: &'a Facts,
    out: Value,
    x: Value,
    y: Value,
    rewrote: bool,
}

impl MatchRewrite for GemvRewrite<'_> {
    fn r#match(&mut self, _ctx: &Context, op: OpId) -> bool {
        op == self.p.for_op && !self.rewrote
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rw: &mut MatchRewriter,
        _op: OpId,
    ) -> pliron::result::Result<()> {
        self.realise(ctx, rw)?;
        self.rewrote = true;
        Ok(())
    }
}

impl GemvRewrite<'_> {
    /// Emit the realisation around the recognised loop, then erase what it replaced.
    fn realise(&mut self, ctx: &mut Context, rw: &mut MatchRewriter) -> pliron::result::Result<()> {
        let p = &self.p;
        let e = p.elem_t.clone();
        let oe = p.dst_t.clone();
        let sg = self.facts.sg;
        let n = self.facts.n;
        let k = p.k();
        let lanes = self.facts.lanes;
        let wide = [sg as i64, lanes];
        let acc = tile(&wide, ScalarType::F32);
        let l = Layout::of(p, self.facts);
        let trips = l.trips(k, sg, lanes);

        // Prologue
        let out_tv = l.out_tv(p, &oe, n, lanes);
        let out_view = tensor_view(ctx, rw, self.out, &out_tv)?;
        let x_tv = l.x_tv(&e, k, n, lanes);
        let x_view = tensor_view(ctx, rw, self.x, &x_tv)?;
        let y_tv = l.y_tv(p, &e, k, n);
        let y_view = tensor_view(ctx, rw, self.y, &y_tv)?;

        let (tok_out, tok_x, tok_y) = (
            make_token(ctx, rw)?,
            make_token(ctx, rw)?,
            make_token(ctx, rw)?,
        );
        let bid = tile_block_id(ctx, rw)?;
        let blk = assume_bounded(ctx, rw, bid[0], 0, None)?;

        let pv_x = partition_view(
            ctx,
            rw,
            x_view,
            &[sg, lanes as i32],
            &x_tv,
            Some(PaddingValue::Zero),
        )?;
        let pv_y = partition_view(
            ctx,
            rw,
            y_view,
            &[sg, lanes as i32],
            &y_tv,
            Some(PaddingValue::Zero),
        )?;
        let pv_out = partition_view(
            ctx,
            rw,
            out_view,
            &l.out_tile(sg, lanes),
            &out_tv,
            Some(PaddingValue::Zero),
        )?;

        let lo = const_i32(ctx, rw, 0)?;
        let hi = const_i32(ctx, rw, trips)?;
        let step = const_i32(ctx, rw, 1)?;
        let acc0 = const_scalar(ctx, rw, &acc, ScalarType::F32, [0, 0, 0, 0])?;

        // Loop body (detached, placed correctly later)
        let (body, body_args) = build_block_in(ctx, &[i32(), acc.clone()]);
        let (iv, carry) = (body_args[0], body_args[1]);
        let after_prologue = rw.get_insertion_point();
        rw.set_insertion_point_to_block_end(body);

        let row = emit_at(
            ctx,
            rw,
            Opcode::Assume,
            &[iv],
            &[i32()],
            &[("predicate", bounded(0, Some(trips - 1)))],
        )?[0];
        let row0 = const_i32(ctx, rw, 0)?;
        let load =
            |ctx: &mut Context,
             rw: &mut MatchRewriter,
             pv: Value,
             i0: Value,
             i1: Value,
             t: Value| { load_view(ctx, rw, pv, i0, i1, t, &tile(&wide, e.clone())) };

        // Layout decides how we load
        // `Nn`: loop down K, block along N.
        // `Nt`: block along N, loop down K.
        let (a_raw, b_raw) = match l {
            Layout::Nn => (
                load(ctx, rw, pv_x, row, row0, tok_x)?,
                load(ctx, rw, pv_y, row, blk, tok_y)?,
            ),
            Layout::Nt => (
                load(ctx, rw, pv_x, blk, row, tok_x)?,
                load(ctx, rw, pv_y, blk, row, tok_y)?,
            ),
        };
        // Accumulate in f32 regardless. Loaded values are widened as they arrive.
        let (a, b) = if e == ScalarType::F32 {
            (a_raw, b_raw)
        } else {
            (
                convert(ctx, rw, a_raw, &acc)?,
                convert(ctx, rw, b_raw, &acc)?,
            )
        };
        let prod = binf(ctx, rw, Opcode::MulF, a, b)?;
        let sum = binf(ctx, rw, Opcode::AddF, carry, prod)?;
        terminator(ctx, rw, Opcode::Continue, &[sum])?;

        // Loop with body from above. Placed in `for` op.
        rw.set_insertion_point(after_prologue);
        let (new_for, for_res) = OpBuilder::new(Opcode::For, Location::Unknown)
            .operands([lo, hi, step, acc0])
            .results(std::iter::once(acc.clone()))
            .region(vec![body])
            .build_in(ctx)?;
        rw.append_operation(ctx, new_for);
        // Apply pre-existing split-K stamp to new for loop.
        carry_split_k_stamp(ctx, p.for_op, new_for);
        // Combine partials across reduced axis
        let scalar_f32 = tile(&[], ScalarType::F32);
        let (red_body, ra) = build_block_in(ctx, &[scalar_f32.clone(), scalar_f32.clone()]);
        let after_for = rw.get_insertion_point();
        rw.set_insertion_point_to_block_end(red_body);
        let comb = binf(ctx, rw, Opcode::AddF, ra[0], ra[1])?;
        terminator(ctx, rw, Opcode::Yield, &[comb])?;
        rw.set_insertion_point(after_for);

        let (red_op, red) = OpBuilder::new(Opcode::Reduce, Location::Unknown)
            .operands([for_res[0]])
            .results(std::iter::once(tile(
                &[l.red_extent(sg, lanes)],
                ScalarType::F32,
            )))
            .attr("dim", Attribute::i32(l.reduce_dim()))
            .attr(
                "identities",
                Attribute::Array(vec![Attribute::Float(
                    FloatBits::new(0.0),
                    Type::Scalar(ScalarType::F32),
                )]),
            )
            .region(vec![red_body])
            .build_in(ctx)?;
        rw.append_operation(ctx, red_op);
        let out_tile = l.out_tile(sg, lanes);
        let out_shape = [out_tile[0] as i64, out_tile[1] as i64];
        // `Nn` reshapes its `[lanes]` row to one element per lane.
        // `Nt` broadcasts its `[sg]` column across the lanes.
        // If the backend's lane-axis reduce leaves its results replicated then the `Nt` broadcast
        // can be compiled away completely.
        let rowtile = emit_at(
            ctx,
            rw,
            match l {
                Layout::Nn => Opcode::Reshape,
                Layout::Nt => Opcode::Broadcast,
            },
            &[red[0]],
            &[tile(&out_shape, ScalarType::F32)],
            &[],
        )?[0];
        let value = if oe != ScalarType::F32 {
            emit_at(
                ctx,
                rw,
                Opcode::FToF,
                &[rowtile],
                &[tile(&out_shape, oe.clone())],
                &[],
            )?[0]
        } else {
            rowtile
        };
        let zero = const_i32(ctx, rw, 0)?;
        let (s0, s1) = match l {
            Layout::Nn => (zero, blk),
            Layout::Nt => (blk, zero),
        };
        store_view(ctx, rw, value, pv_out, s0, s1, tok_out)?;

        // Erase old IR
        let old_res = p.for_op.deref(ctx).get_result(0);
        let epilogue: Vec<OpId> = old_res
            .uses(ctx)
            .iter()
            .map(|u| u.user_op())
            .filter(|&u| u != p.store)
            .collect();
        rw.erase_operation(ctx, p.store);
        for op in epilogue {
            rw.erase_operation(ctx, op);
        }
        let block = p.for_op.deref(ctx).get_parent_block();
        rw.erase_operation(ctx, p.for_op);
        erase_dead_prologue(ctx, rw, block);
        Ok(())
    }
}

/// Copy the split-K reduction stamp from the old loop to the new one.
fn carry_split_k_stamp(ctx: &Context, from: OpId, to: OpId) {
    use super::gemm_analysis::{ATTR_SPLIT_REDUCTION, ATTR_SPLIT_REDUCTION_K};
    use crate::dialect::attr_mirror::WaxAttr;
    for key in [&ATTR_SPLIT_REDUCTION, &ATTR_SPLIT_REDUCTION_K] {
        let got = from
            .deref(ctx)
            .attributes
            .get::<WaxAttr>(key)
            .map(|a| a.0.clone());
        if let Some(v) = got {
            crate::builder::set_op_attr_in(ctx, to, key, v);
        }
    }
}

/// Erase old GEMM prologue.
fn erase_dead_prologue(ctx: &mut Context, rw: &mut MatchRewriter, block: Option<BlockId>) {
    use pliron::linked_list::ContainsLinkedList;
    let Some(block) = block else { return };
    let pure = |c: Opcode| {
        matches!(
            c,
            Opcode::MakeTensorView
                | Opcode::MakePartitionView
                | Opcode::MakeStridedView
                | Opcode::MakeToken
                | Opcode::GetTileBlockId
                | Opcode::Assume
                | Opcode::Constant
        )
    };
    // Recursively remove ops. Removing a view frees its' token and the const that fed it.
    loop {
        let ops: Vec<OpId> = block.deref(ctx).iter(ctx).collect();
        let mut erased = 0;
        for op in ops {
            if !pure(crate::dialect::source::opcode_of(ctx, op)) {
                continue;
            }
            let (has_region, n) = {
                let o = op.deref(ctx);
                (o.regions().next().is_some(), o.get_num_results())
            };
            if has_region {
                continue;
            }
            if (0..n).any(|i| op.deref(ctx).get_result(i).is_used(ctx)) {
                continue;
            }
            rw.erase_operation(ctx, op);
            erased += 1;
        }
        if erased == 0 {
            return;
        }
    }
}

/// Indicates wether `C = A @ B` or `C = A @ B.t()`
///
/// The effect of the layout is which axis of the `[SG, lanes]` tile the lanes walk.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Layout {
    /// B is `[K, N]` row-major. Lanes walk N, the loop walks K and sums partials down tile rows.
    Nn,
    /// B is `[N, K]` row-major. `A @ B.t()`. Lanes walk K, so that B is contiguous per lane.
    /// One tile row to one output column. Partials are sum reduced across lanes.
    Nt,
}

impl Layout {
    fn of(p: &GemmPattern, facts: &Facts) -> Layout {
        if p.trans_b && applies_nt(p, facts) {
            Layout::Nt
        } else {
            Layout::Nn
        }
    }

    /// Loop trips. `Nn` splits K across the tile's rows, `Nt` across its lanes.
    fn trips(self, k: i64, sg: i32, lanes: i64) -> i64 {
        match self {
            Layout::Nn => k / sg as i64,
            Layout::Nt => k / lanes,
        }
    }

    fn reduce_dim(self) -> i64 {
        match self {
            Layout::Nn => 0,
            Layout::Nt => 1,
        }
    }

    /// Elements the reduce leaves. One per output column the tile block owns.
    fn red_extent(self, sg: i32, lanes: i64) -> i64 {
        match self {
            Layout::Nn => lanes,
            Layout::Nt => sg as i64,
        }
    }

    /// Destination partition tile.
    fn out_tile(self, sg: i32, lanes: i64) -> [i32; 2] {
        match self {
            Layout::Nn => [1, lanes as i32],
            Layout::Nt => [sg, lanes as i32],
        }
    }

    /// The destination view. M = 1 aka one row.
    /// `Nn` is row-major `[1, n]`.
    /// `Nt`'s shape stems from what the reduce needs.
    fn out_tv(self, p: &GemmPattern, oe: &ScalarType, n: i64, lanes: i64) -> TensorViewType {
        match self {
            Layout::Nn => TensorViewType {
                element_type: oe.clone(),
                shape: vec![1, n],
                strides: vec![p.ldc, 1],
            },
            // We use `[N, lanes]` with a lane stride = 0. This means all lanes of a tile
            // point to the same output column (of that group).
            // Note that a backend must ensure that this shape is handled correctly.
            // It may be beneficial to materialise the `[sg]` result as `[sg, lanes]`, signifying
            // that the reduce has placed the sum in all lanes of the group. A `[sg, 1]` tile
            // could produce silently incorrect values unless other measures have been taken
            // during lowering.
            Layout::Nt => TensorViewType {
                element_type: oe.clone(),
                shape: vec![n, lanes],
                strides: vec![1, 0],
            },
        }
    }

    /// A view. Replicates the one K-vector across the tile's other axis with a zero stride.
    /// Layout decides the axis.
    fn x_tv(self, e: &ScalarType, k: i64, n: i64, lanes: i64) -> TensorViewType {
        match self {
            Layout::Nn => TensorViewType {
                element_type: e.clone(),
                shape: vec![k, lanes],
                strides: vec![1, 0],
            },
            Layout::Nt => TensorViewType {
                element_type: e.clone(),
                shape: vec![n, k],
                strides: vec![0, 1],
            },
        }
    }

    /// B view.
    /// `Nt` describes the shape as `[N, K]` (which is the actual shape), instead
    /// of `[K, N]`. Same bytes, but lane axis stride becomes 1.
    fn y_tv(self, p: &GemmPattern, e: &ScalarType, k: i64, n: i64) -> TensorViewType {
        match self {
            Layout::Nn => TensorViewType {
                element_type: e.clone(),
                shape: vec![k, n],
                strides: if p.trans_b {
                    vec![1, p.ldb]
                } else {
                    vec![p.ldb, 1]
                },
            },
            Layout::Nt => TensorViewType {
                element_type: e.clone(),
                shape: vec![n, k],
                strides: vec![p.ldb, 1],
            },
        }
    }
}

/// Whether the transposed spelling applies.
fn applies_nt(p: &GemmPattern, facts: &Facts) -> bool {
    let k = p.k();
    k > 0 && facts.lanes > 0 && k % facts.lanes == 0 && facts.sg >= 1
}
