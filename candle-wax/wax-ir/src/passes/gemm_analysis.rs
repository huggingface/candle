//! Recognising a blocked GEMM: an `mma_f` accumulated over a counted loop.

use crate::attr::Attribute;
use crate::opcode::Opcode;
use crate::types::Type;
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::irbuild::IRStatus;
use pliron::operation::Operation;
use pliron::pass::{Analysis, AnalysisManager, Pass, PassResult};
use pliron::value::Value;
use std::result::Result;

// TODO: Tidy up utils

fn opcode(ctx: &Context, op: Ptr<Operation>) -> Opcode {
    crate::dialect::source::opcode_of(ctx, op)
}

fn operands_of(ctx: &Context, op: Ptr<Operation>) -> Vec<Value> {
    op.deref(ctx).operands().collect()
}

fn result_types(ctx: &Context, op: Ptr<Operation>) -> Vec<Type> {
    use pliron::r#type::Typed;
    let o = op.deref(ctx);
    (0..o.get_num_results())
        .filter_map(|i| crate::dialect::types::from_pliron(ctx, o.get_result(i).get_type(ctx)))
        .collect()
}

fn attributes(ctx: &Context, op: Ptr<Operation>) -> Vec<(String, Attribute)> {
    use crate::dialect::attr_mirror::WaxAttr;
    op.deref(ctx)
        .attributes
        .0
        .iter()
        .filter_map(|(k, v)| {
            v.downcast_ref::<WaxAttr>()
                .map(|a| (k.to_string(), a.0.clone()))
        })
        .collect()
}

fn regions(ctx: &Context, op: Ptr<Operation>) -> Vec<Ptr<pliron::region::Region>> {
    op.deref(ctx).regions().collect()
}

fn op_result(ctx: &Context, op: Ptr<Operation>, i: u32) -> Option<Value> {
    let o = op.deref(ctx);
    ((i as usize) < o.get_num_results()).then(|| o.get_result(i as usize))
}

fn value_type(ctx: &Context, v: Value) -> Type {
    use pliron::r#type::Typed;
    crate::dialect::types::from_pliron(ctx, v.get_type(ctx)).unwrap_or(Type::Token)
}

fn region_blocks(ctx: &Context, r: Ptr<pliron::region::Region>) -> Vec<Ptr<BasicBlock>> {
    use pliron::linked_list::ContainsLinkedList;
    r.deref(ctx).iter(ctx).collect()
}

fn block_ops(ctx: &Context, b: Ptr<BasicBlock>) -> Vec<Ptr<Operation>> {
    use pliron::linked_list::ContainsLinkedList;
    b.deref(ctx).iter(ctx).collect()
}

fn block_args(ctx: &Context, b: Ptr<BasicBlock>) -> Vec<(Value, Type)> {
    let blk = b.deref(ctx);
    blk.arguments().map(|v| (v, value_type(ctx, v))).collect()
}

/// Ops between the GEMM loop and the final output. A closed set for now.
// TODO: model the epilogue more generically.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Epilogue {
    None,
    /// The f32 accumulator narrowed to the destination type, the mixed-precision GEMM case.
    Convert(crate::types::ScalarType),
}

/// `C = A @ B`, with everything a backend needs to rewrite to an optimised version.
#[derive(Debug, Clone)]
pub struct GemmPattern {
    pub for_op: Ptr<Operation>,
    pub mma: Ptr<Operation>,
    // A
    pub a_load: Ptr<Operation>,
    // B
    pub b_load: Ptr<Operation>,
    // C
    pub store: Ptr<Operation>,
    // Accumulator tile: `[BM, BN]`.
    pub bm: i64,
    pub bn: i64,
    // Reduction blocking. A is `[BM, BK]`, B is `[BK, BN]`.
    pub bk: i64,
    // Loop trip count. `k == trips * bk`.
    pub trips: i64,
    // Row strides of A, B and C, in elements.
    pub lda: i64,
    pub ldb: i64,
    pub ldc: i64,
    // Element type of A and B.
    pub elem_t: crate::types::ScalarType,
    // Element type of the accumulator.
    pub accum_t: crate::types::ScalarType,
    // Element type of C.
    pub dst_t: crate::types::ScalarType,
    // Indicates B is `[N, K]` and `C = A @ B.t()`. Typical `Linear` weight layout.
    pub trans_b: bool,
    pub epilogue: Epilogue,
}

impl GemmPattern {
    // The reduction extent
    pub fn k(&self) -> i64 {
        self.trips * self.bk
    }
}

// Ops that have no effect on the output.
fn is_noise(op: Opcode) -> bool {
    matches!(op, Opcode::Constant | Opcode::Assume | Opcode::MakeToken)
}

fn tile_shape(ctx: &Context, v: Value) -> Option<Vec<i64>> {
    match value_type(ctx, v) {
        Type::Tile(t) => Some(t.shape.clone()),
        _ => None,
    }
}

fn tile_scalar(ctx: &Context, v: Value) -> Option<crate::types::ScalarType> {
    match value_type(ctx, v) {
        Type::Tile(t) => match t.element_type {
            crate::types::TileElementType::Scalar(s) => Some(s),
            _ => None,
        },
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// Rows are contiguous. `stride = [ld, 1]`.
    RowMajor,
    /// Columns are contiguous. `stride = [1, ld]`. The operand is the transpose of what its shape
    /// says.
    Transposed,
}

/// `(layout, leading dimension, element type)` of the view an access reads through.
fn view_facts(ctx: &Context, view: Value) -> Option<(Layout, i64, crate::types::ScalarType)> {
    let Type::PartitionView(pv) = value_type(ctx, view) else {
        return None;
    };
    let st = &pv.tensor_view.strides;
    if st.len() != 2 {
        return None;
    }
    let elem = pv.tensor_view.element_type;
    match (st[0], st[1]) {
        (ld, 1) => Some((Layout::RowMajor, ld, elem)),
        (1, ld) => Some((Layout::Transposed, ld, elem)),
        // Neither axis is contiguous. Strided view.
        _ => None,
    }
}

/// An integer constant produced by `op`, if it is one.
fn const_i(ctx: &Context, v: Value, ops: &[(Ptr<Operation>, Opcode)]) -> Option<i64> {
    let def = ops
        .iter()
        .find(|(o, c)| *c == Opcode::Constant && op_result(ctx, *o, 0) == Some(v))?;
    let attrs = attributes(ctx, def.0);
    let (_, a) = attrs.iter().find(|(k, _)| k == "value")?;
    match a {
        crate::attr::Attribute::DenseElements(de) => {
            let b = de.data.get(..4)?;
            Some(i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64)
        }
        crate::attr::Attribute::Integer(i, _) => Some(*i),
        _ => None,
    }
}

/// Whether `v` is an all zero constant.
fn is_zero_const(ctx: &Context, v: Value, ops: &[(Ptr<Operation>, Opcode)]) -> bool {
    let Some(def) = ops
        .iter()
        .find(|(o, c)| *c == Opcode::Constant && op_result(ctx, *o, 0) == Some(v))
    else {
        return false;
    };
    attributes(ctx, def.0).iter().any(|(k, a)| {
        k == "value"
            && match a {
                crate::attr::Attribute::DenseElements(de) => de.data.iter().all(|&b| b == 0),
                _ => false,
            }
    })
}

/// Match the function at `root` against the blocked GEMM shape.
pub fn match_gemm(ctx: &Context, root: Ptr<Operation>) -> Result<GemmPattern, &'static str> {
    // The kernel is one entry block. Control flow around the GEMM is not yet supported.
    let region = *regions(ctx, root).first().ok_or("function has no region")?;
    let blocks = region_blocks(ctx, region);
    if blocks.len() != 1 {
        return Err("function is not a single entry block");
    }
    let top: Vec<(Ptr<Operation>, Opcode)> = block_ops(ctx, blocks[0])
        .into_iter()
        .map(|o| (o, opcode(ctx, o)))
        .collect();

    // Single loop. The only significant computation.
    let fors: Vec<Ptr<Operation>> = top
        .iter()
        .filter(|(_, c)| *c == Opcode::For)
        .map(|(o, _)| *o)
        .collect();
    if fors.len() != 1 {
        return Err("not exactly one loop at the top level");
    }
    let for_op = fors[0];

    // `(lo, hi, step, init)`. A single iter value.
    let operands = operands_of(ctx, for_op);
    if operands.len() != 4 || result_types(ctx, for_op).len() != 1 {
        return Err("loop is not (lo, hi, step, init) with one carry");
    }
    let (lo, hi, step, init) = (operands[0], operands[1], operands[2], operands[3]);
    let lo_v = const_i(ctx, lo, &top).ok_or("loop lower bound is not a constant")?;
    let step_v = const_i(ctx, step, &top).ok_or("loop step is not a constant")?;
    if lo_v != 0 || step_v != 1 {
        return Err("loop does not start at 0 with step 1");
    }
    let trips = const_i(ctx, hi, &top).ok_or("loop upper bound is not a constant")?;
    if trips <= 0 {
        return Err("loop trip count is not a positive constant");
    }
    // Accumulator starts at zero. Non-zero start is a different computation.
    if !is_zero_const(ctx, init, &top) {
        return Err("accumulator does not start at zero");
    }
    let acc_shape = tile_shape(ctx, init).ok_or("accumulator is not a tile")?;
    if acc_shape.len() != 2 {
        return Err("accumulator is not a 2D tile");
    }
    let (bm, bn) = (acc_shape[0], acc_shape[1]);
    let accum_t = tile_scalar(ctx, init).ok_or("accumulator has no scalar element")?;

    let body_region = *regions(ctx, for_op)
        .first()
        .ok_or("loop has no body region")?;
    let body_blocks = region_blocks(ctx, body_region);
    if body_blocks.len() != 1 {
        return Err("loop body is not a single block");
    }
    let body = body_blocks[0];
    let args = block_args(ctx, body);
    if args.len() != 2 {
        return Err("loop body does not take (iv, carry)");
    }
    let carry = args[1].0;

    // Significant ops of the fn body, in order. Two loads, one mma, terminator.
    // TODO: Extend pattern.
    let body_ops: Vec<(Ptr<Operation>, Opcode)> = block_ops(ctx, body)
        .into_iter()
        .map(|o| (o, opcode(ctx, o)))
        .filter(|(_, c)| !is_noise(*c))
        .collect();
    if body_ops.len() != 4 {
        return Err("loop body is not exactly two loads, an mma and a terminator");
    }
    let (a_load, b_load, mma, term) = (body_ops[0], body_ops[1], body_ops[2], body_ops[3]);
    if a_load.1 != Opcode::LoadViewTko
        || b_load.1 != Opcode::LoadViewTko
        || mma.1 != Opcode::MmaF
        || !matches!(term.1, Opcode::Yield | Opcode::Continue)
    {
        return Err("loop body ops are not load, load, mma, terminator");
    }

    // `mma(a, b, carry)`
    let mma_ops = operands_of(ctx, mma.0);
    if mma_ops.len() != 3
        || Some(mma_ops[0]) != op_result(ctx, a_load.0, 0)
        || Some(mma_ops[1]) != op_result(ctx, b_load.0, 0)
        || mma_ops[2] != carry
    {
        return Err("mma does not consume (a, b, carry)");
    }
    let mma_res = op_result(ctx, mma.0, 0).ok_or("mma result is unused")?;
    if operands_of(ctx, term.0).first().copied() != Some(mma_res) {
        return Err("loop does not yield the mma result");
    }

    // `A: [BM, BK], B: [BK, BN]`. Shared BK is a prerequisite for this being a matmul.
    let a_shape = tile_shape(
        ctx,
        op_result(ctx, a_load.0, 0).ok_or("A load has no result")?,
    )
    .ok_or("A is not a tile")?;
    let b_shape = tile_shape(
        ctx,
        op_result(ctx, b_load.0, 0).ok_or("B load has no result")?,
    )
    .ok_or("B is not a tile")?;
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err("A or B is not a 2D tile");
    }
    if a_shape[0] != bm || b_shape[1] != bn || a_shape[1] != b_shape[0] {
        return Err("tile shapes do not compose as a matmul");
    }
    let bk = a_shape[1];

    let (a_layout, lda, elem_t) = view_facts(
        ctx,
        *operands_of(ctx, a_load.0)
            .first()
            .ok_or("A load has no view")?,
    )
    .ok_or("A view has no leading dimension")?;
    let (b_layout, ldb, b_elem) = view_facts(
        ctx,
        *operands_of(ctx, b_load.0)
            .first()
            .ok_or("B load has no view")?,
    )
    .ok_or("B view has no leading dimension")?;

    // Currently we only support transposed B. Cases with transposed A or C are rejected.
    // TODO: Support more cases
    if a_layout != Layout::RowMajor {
        return Err("A is transposed");
    }
    if elem_t != b_elem {
        return Err("A and B have different element types");
    }

    // The result must be consumed by exactly one store. A GEMM substitution would drop any
    // additional store.
    let for_res = op_result(ctx, for_op, 0).ok_or("loop result is unused")?;
    let consumers: Vec<(Ptr<Operation>, Opcode)> = top
        .iter()
        .filter(|(o, _)| operands_of(ctx, *o).contains(&for_res))
        .copied()
        .collect();
    if consumers.len() != 1 {
        return Err("the product has more than one consumer");
    }

    // Extract store op and epilogue
    let (store, epilogue) = match consumers[0] {
        (op, Opcode::StoreViewTko) => (op, Epilogue::None),
        (cast, Opcode::FToF) => {
            let cast_res = op_result(ctx, cast, 0).ok_or("the convert after the loop is unused")?;
            let after: Vec<(Ptr<Operation>, Opcode)> = top
                .iter()
                .filter(|(o, _)| operands_of(ctx, *o).contains(&cast_res))
                .copied()
                .collect();
            if after.len() != 1 || after[0].1 != Opcode::StoreViewTko {
                return Err("the convert after the loop is not consumed by one store");
            }
            let to = tile_scalar(ctx, cast_res).ok_or("the convert has no scalar element")?;
            (after[0].0, Epilogue::Convert(to))
        }
        _ => return Err("the product is consumed by something other than a store or a convert"),
    };
    // `store_view(value, view, indices.., token)`.
    let (c_layout, ldc, dst_t) = view_facts(
        ctx,
        *operands_of(ctx, store).get(1).ok_or("store has no view")?,
    )
    .ok_or("C view has no leading dimension")?;
    if c_layout != Layout::RowMajor {
        return Err("C is transposed, which no backend expresses");
    }

    Ok(GemmPattern {
        for_op,
        mma: mma.0,
        a_load: a_load.0,
        b_load: b_load.0,
        store,
        bm,
        bn,
        bk,
        trips,
        lda,
        ldb,
        ldc,
        elem_t,
        accum_t,
        dst_t,
        trans_b: b_layout == Layout::Transposed,
        epilogue,
    })
}

/// GEMM analysis result.
pub struct GemmAnalysis {
    // Result lets us describe why a pattern was not recognized.
    pub pattern: Result<GemmPattern, &'static str>,
}

impl Analysis for GemmAnalysis {
    fn name(&self) -> &str {
        "wax-gemm"
    }

    fn compute(
        op: Ptr<Operation>,
        ctx: &Context,
        _analyses: &mut AnalysisManager,
    ) -> pliron::result::Result<Self> {
        Ok(GemmAnalysis {
            pattern: match_gemm(ctx, op),
        })
    }
}

pliron::dict_key!(
    /// Set on a `wax.for` whose reduction may be split and summed across its extent.
    ///
    /// Does not care whether the loop came from a classic blocked GEMM or something completely
    /// different.
    ATTR_SPLIT_REDUCTION,
    "wax_split_reduction"
);

pliron::dict_key!(
    /// The reduction's extent, set alongside [`ATTR_SPLIT_REDUCTION`]. `K` for a GEMM.
    ATTR_SPLIT_REDUCTION_K,
    "wax_split_reduction_k"
);

/// Marks a recognised GEMM's K loop as a splittable reduction.
///
/// Reads [`GemmAnalysis`] and stamps the loop with the detected K.
/// Does not decide if the split should happen, that is left to the backend.
#[derive(Default)]
pub struct GemmMarkPass {
    /// Loops marked. `0` means fn was not a recognised GEMM.
    pub marked: usize,
}

impl Pass for GemmMarkPass {
    fn name(&self) -> &str {
        "wax-gemm-mark"
    }

    fn run(
        &mut self,
        op: Ptr<Operation>,
        ctx: &mut Context,
        analyses: &mut AnalysisManager,
    ) -> pliron::result::Result<PassResult> {
        self.marked = 0;
        let pattern = analyses
            .get_analysis::<GemmAnalysis>(op, ctx)?
            .pattern
            .clone();
        let Ok(p) = pattern else {
            // Not a GEMM. `IRStatus::Unchanged` implies every analysis is preserved.
            return Ok(PassResult::default());
        };

        crate::builder::set_op_attr_in(ctx, p.for_op, &ATTR_SPLIT_REDUCTION, Attribute::Bool(true));
        crate::builder::set_op_attr_in(
            ctx,
            p.for_op,
            &ATTR_SPLIT_REDUCTION_K,
            Attribute::i32(p.k()),
        );
        self.marked = 1;

        let mut res = PassResult::default();
        res.ir_changed = IRStatus::Changed;
        // Setting an attribute is harmless. Preserve GEMM analysis.
        res.set_preserved::<GemmAnalysis>();
        Ok(res)
    }
}

/// All loops in `func` carrying [`ATTR_SPLIT_REDUCTION`].
pub fn stamped_loops(ctx: &Context, func: Ptr<Operation>) -> Vec<Ptr<Operation>> {
    use crate::dialect::attr_mirror::WaxAttr;
    use pliron::graph::walkers::{IRNode, WalkConfig, uninterruptible::immutable::walk_op};

    let mut state: (pliron::identifier::Identifier, Vec<Ptr<Operation>>) =
        (ATTR_SPLIT_REDUCTION.clone(), Vec::new());
    walk_op(
        ctx,
        &mut state,
        &WalkConfig::default(),
        func,
        |ctx, (key, out), node| {
            if let IRNode::Operation(op) = node
                && opcode(ctx, op) == Opcode::For
                && op.deref(ctx).attributes.get::<WaxAttr>(key).is_some()
            {
                out.push(op);
            }
        },
    );
    state.1
}
