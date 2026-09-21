//! Rewrite an existing K loop into `splits` independent slices.

use super::Partials;
use crate::build::{OpId, Value};
use crate::builder::set_op_attr_in;
use crate::dialect::source::{const_int_of, defining_op, opcode_of, ops_in, value_type};
use crate::opcode::Opcode;
use crate::passes::gemm_analysis::stamped_loops;
use crate::passes::utils::{assume_bounded, bin, bounded, const_i32, dens_i32, entry_block};
use crate::types::{PartitionViewType, ScalarType, TensorViewType, Type};
use pliron::context::Context;
use pliron::irbuild::IRStatus;
use pliron::irbuild::inserter::{Inserter, OpInsertionPoint};
use pliron::irbuild::match_rewrite::MatchRewriter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::pass::{AnalysisManager, Pass, PassResult};
use pliron::value::DefiningEntity;

/// Pass that splits a stamped reduction into `splits` independent work.
pub struct SplitKPass {
    splits: i32,
}

impl SplitKPass {
    pub fn new(splits: i32) -> Self {
        SplitKPass { splits }
    }
}

impl Pass for SplitKPass {
    fn name(&self) -> &str {
        "wax-split-k"
    }

    fn run(
        &mut self,
        op: OpId,
        ctx: &mut Context,
        _analyses: &mut AnalysisManager,
    ) -> pliron::result::Result<PassResult> {
        if self.splits < 2 {
            return Ok(PassResult::default());
        }
        // Only a realised kernel can use split-k via a wax `Pass`.
        // The split-k `build` uses rows/cols as static extents in the reduce kernel, which is
        // impossible for a shape generic kernel like the default GEMM. It must be done JIT when
        // the GEMM shape is known.
        // Shape generic kernels will not have launch geometry as an attribute.
        if crate::passes::launch::Launch::of(ctx, op).is_none() {
            return Ok(PassResult::default());
        }
        let Some(plan) = plan(ctx, op, self.splits) else {
            return Ok(PassResult::default());
        };
        let mut rw = MatchRewriter::default();
        // Apply split. Stamp partials definition.
        split(ctx, &mut rw, &plan)?.stamp(ctx, op);
        // Reshape launch geometry for split-k.
        if let Some(l) = crate::passes::launch::Launch::of(ctx, op) {
            crate::passes::launch::Launch {
                block: crate::passes::launch::TileBlock {
                    z: self.splits as usize,
                    ..l.block
                },
                ..l
            }
            .stamp(ctx, op);
        }
        let mut res = PassResult::default();
        res.ir_changed = IRStatus::Changed;
        Ok(res)
    }
}

/// Everything [`split`] needs.
///
/// Contains `splits`, anchors into IR for rewrites, and snapshots of previous state.
struct Plan {
    splits: i32,

    // Anchors.
    /// Loop upper bound. `split` sets value to `trips / splits`.
    hi_op: OpId,
    /// The `assume` bound on the loop's induction variable, shrinks to one split.
    iv_assume: OpId,
    /// `wax.tile_block_id`.
    bid: OpId,
    /// Where to emit split index.
    after: OpId,
    /// The destination's views. Both retyped to f32 scratch.
    out_view: Value,
    pv_out: Value,
    /// The store op, and the old row index.
    store: OpId,
    old_row: Value,
    /// Narrow epilogue, which must be moved to the reduce.
    narrow: Option<OpId>,

    // Snapshots.
    /// Original loop upper bound.
    trips: i64,
    /// Original destination views.
    out_tv: TensorViewType,
    pv_ty: PartitionViewType,
}

impl Plan {
    fn rows(&self) -> i64 {
        self.out_tv.shape[0]
    }
    fn cols(&self) -> i64 {
        self.out_tv.shape[1]
    }
    /// Destination row stride.
    fn ldc(&self) -> i64 {
        self.out_tv.strides[0]
    }
    fn out_elem(&self) -> &ScalarType {
        &self.out_tv.element_type
    }
    /// Destination tile rows.
    fn tile_rows(&self) -> i64 {
        self.rows() / self.pv_ty.tile_shape[0] as i64
    }
    /// Check if store row index is a const zero, so the add can be skipped.
    fn old_row_is_zero(&self, ctx: &Context) -> bool {
        defining_op(self.old_row).and_then(|d| const_int_of(ctx, d)) == Some(0)
    }
}

fn operand(ctx: &Context, op: OpId, i: usize) -> Option<Value> {
    op.deref(ctx).operands().nth(i)
}

/// Plan the rewrite.
fn plan(ctx: &Context, func: OpId, splits: i32) -> Option<Plan> {
    let loops = stamped_loops(ctx, func);
    let [for_op] = loops[..] else {
        return None;
    };
    let entry = entry_block(ctx, func)?;
    let top = ops_in(ctx, entry);

    // The iteration count.
    let hi_val = operand(ctx, for_op, 1)?;
    let hi_op = defining_op(hi_val)?;
    let trips = const_int_of(ctx, hi_op)?;
    // `trips % splits == 0` must hold or we would miss part of the reduction.
    if trips <= 0 || trips % splits as i64 != 0 {
        return None;
    }

    // Get the store
    let stores: Vec<OpId> = top
        .iter()
        .copied()
        .filter(|&o| opcode_of(ctx, o) == Opcode::StoreViewTko)
        .collect();
    // Which must be the only store.
    let [store] = stores[..] else {
        return None;
    };
    let pv_out = operand(ctx, store, 1)?;
    let pv_def = defining_op(pv_out)?;
    if opcode_of(ctx, pv_def) != Opcode::MakePartitionView {
        return None;
    }
    let out_view = operand(ctx, pv_def, 0)?;
    let out_def = defining_op(out_view)?;
    if opcode_of(ctx, out_def) != Opcode::MakeTensorView {
        return None;
    }
    let base = operand(ctx, out_def, 0)?;
    // This pass only supports properly defined views. A view built from a computed pointer is not
    // not supported.
    if !matches!(base.defining_entity(), DefiningEntity::Block(_)) {
        return None;
    }
    let Type::TensorView(out_tv) = value_type(ctx, out_view)? else {
        return None;
    };
    let Type::PartitionView(pv_ty) = value_type(ctx, pv_out)? else {
        return None;
    };
    if out_tv.shape.len() != 2 || out_tv.strides.len() != 2 || out_tv.strides[1] != 1 {
        return None;
    }
    let &t = pv_ty.tile_shape.first()?;

    // Validate shapes
    let rows = out_tv.shape[0];
    if t <= 0 || rows <= 0 || rows % t as i64 != 0 {
        return None;
    }

    // The partials are always f32.
    // If required we narrow to the output type.
    let value = operand(ctx, store, 0)?;
    let is_f32 = |v: Value| {
        matches!(value_type(ctx, v), Some(Type::Tile(t))
            if matches!(t.element_type, crate::types::TileElementType::Scalar(ScalarType::F32)))
    };
    let narrow = match defining_op(value) {
        Some(d) if opcode_of(ctx, d) == Opcode::FToF => {
            let src = operand(ctx, d, 0)?;
            if !is_f32(src) {
                return None;
            }
            Some(d)
        }
        _ if is_f32(value) => None,
        _ => return None,
    };

    // Tile block id maps nicely as (x, y, z) => (m, n, k). In other words z is the split index.
    let bid = top
        .iter()
        .copied()
        .find(|&o| opcode_of(ctx, o) == Opcode::GetTileBlockId)?;
    let bid_x = bid.deref(ctx).get_result(0);
    let after = top
        .iter()
        .copied()
        .find(|&o| opcode_of(ctx, o) == Opcode::Assume && operand(ctx, o, 0) == Some(bid_x))
        .unwrap_or(bid);

    let body = {
        use pliron::linked_list::ContainsLinkedList;
        let r = for_op.deref(ctx).regions().next()?;
        r.deref(ctx).iter(ctx).next()?
    };
    let iv = body.deref(ctx).get_argument(0);
    let iv_assume = ops_in(ctx, body)
        .into_iter()
        .find(|&o| opcode_of(ctx, o) == Opcode::Assume && operand(ctx, o, 0) == Some(iv))?;

    let old_row = operand(ctx, store, 2)?;

    Some(Plan {
        splits,
        hi_op,
        iv_assume,
        bid,
        after,
        out_view,
        pv_out,
        store,
        old_row,
        narrow,
        trips,
        out_tv,
        pv_ty,
    })
}

/// Perform the split.
fn split(ctx: &mut Context, rw: &mut MatchRewriter, p: &Plan) -> pliron::result::Result<Partials> {
    let f32 = ScalarType::F32;
    let per_split = p.trips / p.splits as i64;

    // Change the destination view of the producer to be the f32 scratch.
    let out_tv = TensorViewType {
        element_type: f32.clone(),
        shape: vec![p.splits as i64 * p.rows(), p.cols()],
        strides: vec![p.cols(), 1],
    };
    retype(ctx, p.out_view, &Type::TensorView(out_tv.clone()));
    retype(
        ctx,
        p.pv_out,
        &Type::PartitionView(PartitionViewType {
            tensor_view: out_tv,
            ..p.pv_ty.clone()
        }),
    );

    // Set Assume::Bounded([0, splits]) attribute on block id z.
    rw.set_insertion_point(OpInsertionPoint::AfterOperation(p.after));
    let bid_z = p.bid.deref(ctx).get_result(2);
    let spl = assume_bounded(ctx, rw, bid_z, 0, Some(p.splits as i64 - 1))?;

    // Update loop to use `per_split`.
    set_op_attr_in(ctx, p.hi_op, &crate::build::ATTR_VALUE, dens_i32(per_split));
    // Set Assume::Bounded([0, per_split]) attribute on the loop IV.
    set_op_attr_in(
        ctx,
        p.iv_assume,
        &crate::build::ATTR_PREDICATE,
        bounded(0, Some(per_split - 1)),
    );

    // Row
    let iv_a = p.iv_assume.deref(ctx).get_result(0);
    rw.set_insertion_point(OpInsertionPoint::AfterOperation(p.iv_assume));
    // Calculate this split's K row offset.
    let stride = const_i32(ctx, rw, per_split)?;
    let off = bin(ctx, rw, Opcode::MulI, spl, stride)?;
    let row = bin(ctx, rw, Opcode::AddI, iv_a, off)?;
    let row_op = defining_op(row).expect("just emitted");
    iv_a.replace_some_uses_with(ctx, |_, u| u.user_op() != row_op, &row);

    // Add narrowing if needed
    if let Some(n) = p.narrow {
        let res = n.deref(ctx).get_result(0);
        let src = operand(ctx, n, 0).expect("the convert has a source");
        res.replace_all_uses_with(ctx, &src);
        rw.erase_operation(ctx, n);
    }

    // Calculate plane-strided row address
    let plane = plane_row(ctx, rw, p, spl)?;
    p.old_row
        .replace_some_uses_with(ctx, |_, u| u.user_op() == p.store, &plane);
    if let Some(d) = defining_op(p.old_row)
        && !p.old_row.is_used(ctx)
        && opcode_of(ctx, d) == Opcode::Constant
    {
        rw.erase_operation(ctx, d);
    }

    Ok(Partials {
        out_elem: p.out_elem().clone(),
        rows: p.rows(),
        cols: p.cols(),
        ldc: p.ldc(),
        splits: p.splits,
    })
}

/// Calculate plane-strided row address: `spl * tile_rows + old_row`.
fn plane_row(
    ctx: &mut Context,
    rw: &mut MatchRewriter,
    p: &Plan,
    spl: Value,
) -> pliron::result::Result<Value> {
    rw.set_insertion_point(OpInsertionPoint::BeforeOperation(p.store));
    // No mul if there is only 1 row (GEMV).
    let tile_rows = p.tile_rows();
    let scaled = if tile_rows == 1 {
        spl
    } else {
        let k = const_i32(ctx, rw, tile_rows)?;
        bin(ctx, rw, Opcode::MulI, spl, k)?
    };
    // No add if old row are zeros.
    if p.old_row_is_zero(ctx) {
        return Ok(scaled);
    }
    bin(ctx, rw, Opcode::AddI, scaled, p.old_row)
}

/// Give `v` a new wax type.
fn retype(ctx: &mut Context, v: Value, t: &Type) {
    let h = crate::dialect::types::to_pliron(ctx, t).expect("a value cannot be a function type");
    v.set_type(ctx, h);
}
