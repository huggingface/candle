//! Operations of the `wax` dialect.

use super::attrs::{BinaryAttr, CastAttr, CmpKindAttr, UnaryAttr};
use pliron::basic_block::BasicBlock;
use pliron::builtin::attributes::IntegerAttr;
use pliron::builtin::op_interfaces::{
    IsTerminatorInterface, NResultsInterface, OneOpdInterface, OneResultInterface,
    SingleBlockRegionInterface,
};
use pliron::builtin::op_interfaces::{
    RegionBranchOpInterface, RegionBranchPoint, RegionEdge, RegionSuccessor,
};
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::derive::pliron_op;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::region::Region;
use pliron::r#type::TypeHandle;
use pliron::utils::apint::APInt;
use pliron::value::Value;
use std::result::Result;

pliron::dict_key!(
    /// Key for the operator/predicate an op carries.
    ATTR_KEY_WAX_OP,
    "wax_op"
);
pliron::dict_key!(
    /// Key for a cmp predicate code.
    ATTR_KEY_WAX_PRED,
    "wax_pred"
);
pliron::dict_key!(
    /// Key for a dimension index (reduce, scan, shape queries, etc).
    ATTR_KEY_WAX_DIM,
    "wax_dim"
);

macro_rules! op_new {
    ($ty:ident, $($opd:ident),*) => {
        impl $ty {
            pub fn new(ctx: &mut Context, res_ty: TypeHandle, $($opd: Value),*) -> Self {
                $ty {
                    op: Operation::new(
                        ctx,
                        Self::get_concrete_op_info(),
                        vec![res_ty],
                        vec![$($opd),*],
                        vec![],
                        0,
                    ),
                }
            }
        }
    };
}

// Uniform families.

/// Binary arithmetic. Which op is defined by [`BinaryAttr`].
#[pliron_op(
    name = "wax.binary",
    interfaces = [OneResultInterface],
    operands = (lhs, rhs),
    format,
    verifier = "succ"
)]
pub struct BinaryOp;
op_new!(BinaryOp, lhs, rhs);

/// Unary arithmetic and transcendentals. Which op is defined by [`UnaryAttr`].
#[pliron_op(
    name = "wax.unary",
    interfaces = [OneResultInterface, OneOpdInterface],
    operands = (val),
    format,
    verifier = "succ"
)]
pub struct UnaryOp;
op_new!(UnaryOp, val);

/// Casting. The result type is defined by [`CastAttr`].
#[pliron_op(
    name = "wax.cast",
    interfaces = [OneResultInterface, OneOpdInterface],
    operands = (val),
    format,
    verifier = "succ"
)]
pub struct CastOp;
op_new!(CastOp, val);

/// Comparison. Wether this is between floats or ints is defined by [`CmpKindAttr`].
/// The predicate is stored by [`ATTR_KEY_WAX_PRED`], with the same mapping as `cutile_ir`,
/// so that we don't have to keep our own mapping in sync.
#[pliron_op(
    name = "wax.cmp",
    interfaces = [OneResultInterface],
    operands = (lhs, rhs),
    format,
    verifier = "succ"
)]
pub struct CmpOp;
op_new!(CmpOp, lhs, rhs);

/// Fused multiply-add.
#[pliron_op(
    name = "wax.fma",
    interfaces = [OneResultInterface],
    operands = (a, b, c),
    format,
    verifier = "succ"
)]
pub struct FmaOp;
op_new!(FmaOp, a, b, c);

#[pliron_op(
    name = "wax.select",
    interfaces = [OneResultInterface],
    operands = (cond, t, f),
    format,
    verifier = "succ"
)]
pub struct SelectOp;
op_new!(SelectOp, cond, t, f);

// Tile-shaped ops.
//

/// Constant value. Can be a tile or a scalar.
/// We use the same mapping as `cutile_ir` to avoid having to keep our own approach in sync.
#[pliron_op(name = "wax.constant", interfaces = [OneResultInterface], format, verifier = "succ")]
pub struct ConstantOp;

impl ConstantOp {
    pub fn new(ctx: &mut Context, res_ty: TypeHandle) -> Self {
        ConstantOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![res_ty],
                vec![],
                vec![],
                0,
            ),
        }
    }
}

/// Broadcasting a tile to a wider shape, defined by the result type.
#[pliron_op(
    name = "wax.broadcast",
    interfaces = [OneResultInterface, OneOpdInterface],
    operands = (val),
    format,
    verifier = "succ"
)]
pub struct BroadcastOp;
op_new!(BroadcastOp, val);

/// Reshape a tile. Element count unchanged.
#[pliron_op(
    name = "wax.reshape",
    interfaces = [OneResultInterface, OneOpdInterface],
    operands = (val),
    format,
    verifier = "succ"
)]
pub struct ReshapeOp;
op_new!(ReshapeOp, val);

/// Extract one element of a tile.
#[pliron_op(
    name = "wax.extract",
    interfaces = [OneResultInterface],
    operands = (val, idx),
    format,
    verifier = "succ"
)]
pub struct ExtractOp;

impl ExtractOp {
    /// Note that ExtractOp is variadic in the sense that for it to extract from a
    /// multidimensional tile, it carries one index per dimension, and a rank 0 tile has none.
    /// Assuming exact amount of operands can be a footgun.
    pub fn new_with(ctx: &mut Context, res_ty: TypeHandle, opds: Vec<Value>) -> Self {
        ExtractOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![res_ty],
                opds,
                vec![],
                0,
            ),
        }
    }
}

/// Applies an offset to the view origin.
#[pliron_op(
    name = "wax.offset",
    interfaces = [OneResultInterface],
    operands = (view, off),
    format,
    verifier = "succ"
)]
pub struct OffsetOp;
op_new!(OffsetOp, view, off);

/// Floating-point matrix multiply-accumulate (cutile `mmaf`).
#[pliron_op(
    name = "wax.mma_f",
    interfaces = [OneResultInterface],
    operands = (a, b, acc),
    format,
    verifier = "succ"
)]
pub struct MmaFOp;
op_new!(MmaFOp, a, b, acc);

// Views and memory.

/// Make a tensor view from a pointer.
#[pliron_op(
    name = "wax.make_tensor_view",
    interfaces = [OneResultInterface],
    operands = (ptr, token),
    format,
    verifier = "succ"
)]
pub struct MakeTensorViewOp;
op_new!(MakeTensorViewOp, ptr, token);

/// Partition a tensor view into tiles. Tile shape defined by result type.
#[pliron_op(
    name = "wax.make_partition_view",
    interfaces = [OneResultInterface, OneOpdInterface],
    operands = (view),
    format,
    verifier = "succ"
)]
pub struct MakePartitionViewOp;
op_new!(MakePartitionViewOp, view);

/// An ordering token.
#[pliron_op(name = "wax.make_token", interfaces = [OneResultInterface], format, verifier = "succ")]
pub struct MakeTokenOp;

/// Load a tile through a view.
///
/// Produces two results: the tile itself and an async token a later wait consumes.
#[pliron_op(
    name = "wax.load_view",
    operands = (view, indices),
    format,
    verifier = "succ"
)]
pub struct LoadViewOp;

/// Store a tile through a view, producing an async token a later wait consumes.
#[pliron_op(
    name = "wax.store_view",
    operands = (val, view, rest),
    format,
    verifier = "succ"
)]
pub struct StoreViewOp;

/// A range/divisibility assumption the frontend proved.
///
/// Assume forwards a refined value so that the assumption is carried to downstream ops.
#[pliron_op(
    name = "wax.assume",
    interfaces = [OneResultInterface, OneOpdInterface],
    operands = (val),
    format,
    verifier = "succ"
)]
pub struct AssumeOp;
op_new!(AssumeOp, val);

/// A runtime assertion.
#[pliron_op(
    name = "wax.assert",
    interfaces = [NResultsInterface<0>],
    operands = (cond),
    format,
    verifier = "succ"
)]
pub struct AssertOp;

impl AssertOp {
    pub fn new(ctx: &mut Context, cond: Value) -> Self {
        AssertOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                vec![cond],
                vec![],
                0,
            ),
        }
    }
}

// Simple operations that are just "operands in -> results out", with everything else riding in
// their attributes.
macro_rules! simple_ops {
    ($(($op:ident, $name:literal, $code:ident, $doc:literal)),+ $(,)?) => {
        $(
            #[doc = $doc]
            #[pliron_op(name = $name, operands = (args), format, verifier = "succ")]
            pub struct $op;

            impl $op {
                /// `regions` denotes the how many regions are in the source op.
                pub fn new(
                    ctx: &mut Context,
                    res_tys: Vec<TypeHandle>,
                    opds: Vec<Value>,
                    regions: usize,
                ) -> Self {
                    $op {
                        op: Operation::new(ctx, Self::get_concrete_op_info(), res_tys, opds, vec![], regions),
                    }
                }
            }
        )+

        /// The `wax` op for a simple opcode, or `None` if there is none.
        pub fn simple_from_opcode(
            ctx: &mut Context,
            code: crate::Opcode,
            res_tys: Vec<TypeHandle>,
            opds: Vec<Value>,
            regions: usize,
        ) -> Option<Ptr<Operation>> {
            match code {
                $(crate::Opcode::$code => {
                    Some($op::new(ctx, res_tys, opds, regions).get_operation())
                })+
                _ => None,
            }
        }

        /// The opcode a simple `wax` op came from.
        pub fn simple_to_opcode(
            ctx: &Context,
            op: Ptr<Operation>,
        ) -> Option<crate::Opcode> {
            $(if Operation::is_op::<$op>(op, ctx) {
                return Some(crate::Opcode::$code);
            })+
            None
        }
    };
}

simple_ops!(
    (
        ReduceOp,
        "wax.reduce",
        Reduce,
        "Reduce a tile along the dimension named in the carried attributes."
    ),
    (
        BlockIdOp,
        "wax.tile_block_id",
        GetTileBlockId,
        "This tile block's index in the grid."
    ),
    (
        NumBlocksOp,
        "wax.num_tile_blocks",
        GetNumTileBlocks,
        "The grid's tile-block count."
    ),
    (
        TensorShapeOp,
        "wax.tensor_shape",
        GetTensorShape,
        "The shape of a tensor."
    ),
    (
        IndexSpaceShapeOp,
        "wax.index_space_shape",
        GetIndexSpaceShape,
        "The index space's extent along a dimension."
    ),
    (
        IntToPtrOp,
        "wax.int_to_ptr",
        IntToPtr,
        "Reinterpret an integer tile as a pointer tile. Same as cutile's own route to a base \
         pointer (see `load_tensor`), and the one a gather needs."
    ),
    (
        LoadPtrTkoOp,
        "wax.load_ptr_tko",
        LoadPtrTko,
        "Load through a tile of pointers - one address per lane."
    ),
    (
        IotaOp,
        "wax.iota",
        Iota,
        "Lane index along the dimension named in the carried attributes."
    ),
    (
        AtomicRmwOp,
        "wax.atomic_rmw",
        AtomicRMW,
        "Read-modify-write against a view."
    ),
    (
        AtomicCasOp,
        "wax.atomic_cas",
        AtomicCAS,
        "Compare-and-swap against a view."
    ),
    (
        ScanOp,
        "wax.scan",
        Scan,
        "Prefix scan along the dimension named in the carried attributes."
    ),
    (
        PackOp,
        "wax.pack",
        Pack,
        "Pack narrower elements into a wider one."
    ),
    (
        MakeGatherScatterViewOp,
        "wax.make_gather_scatter_view",
        MakeGatherScatterView,
        "A view indexed by a tensor of indices."
    ),
    (
        MakeStridedViewOp,
        "wax.make_strided_view",
        MakeStridedView,
        "A view with explicit per-dimension strides."
    ),
);

// Structured control flow. Carries regions.

/// A for-loop.
/// The region captures the induction variable, the block's arguments, and [`YieldOp`] ends it.
#[pliron_op(
    name = "wax.for",
    interfaces = [SingleBlockRegionInterface],
    operands = (lo, hi, step, inits),
    format,
    verifier = "succ"
)]
pub struct ForOp;

/// If statement. Contains two regions: then and else.
#[pliron_op(
    name = "wax.if",
    operands = (cond),
    format,
    verifier = "succ"
)]
pub struct IfOp;

/// Ends a region, yielding results.
#[pliron_op(
    name = "wax.yield",
    interfaces = [IsTerminatorInterface, NResultsInterface<0>],
    operands = (vals),
    format,
    verifier = "succ"
)]
pub struct YieldOp;

/// Skip to next iteration, carrying loop values forward.
#[pliron_op(
    name = "wax.continue",
    interfaces = [IsTerminatorInterface, NResultsInterface<0>],
    operands = (vals),
    format,
    verifier = "succ"
)]
pub struct ContinueOp;

/// Return operation
#[pliron_op(
    name = "wax.return",
    interfaces = [IsTerminatorInterface, NResultsInterface<0>],
    format,
    verifier = "succ"
)]
pub struct ReturnOp;

pub fn set_binary(op: Ptr<Operation>, ctx: &Context, a: BinaryAttr) {
    op.deref_mut(ctx).attributes.set(ATTR_KEY_WAX_OP.clone(), a);
}
pub fn set_unary(op: Ptr<Operation>, ctx: &Context, a: UnaryAttr) {
    op.deref_mut(ctx).attributes.set(ATTR_KEY_WAX_OP.clone(), a);
}
pub fn set_cast(op: Ptr<Operation>, ctx: &Context, a: CastAttr) {
    op.deref_mut(ctx).attributes.set(ATTR_KEY_WAX_OP.clone(), a);
}

/// Set the compare kind and and its `cutile_ir` predicate code.
pub fn set_cmp(op: Ptr<Operation>, ctx: &mut Context, kind: CmpKindAttr, pred: i64) {
    op.deref_mut(ctx)
        .attributes
        .set(ATTR_KEY_WAX_OP.clone(), kind);
    let ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let attr = IntegerAttr::new(
        ty,
        APInt::from_u64(pred as u64, core::num::NonZero::new(32).unwrap()),
    );
    op.deref_mut(ctx)
        .attributes
        .set(ATTR_KEY_WAX_PRED.clone(), attr);
}

/// Set dimension index. Used by reduce/scan/iota and shape queries.
pub fn set_dim(op: Ptr<Operation>, ctx: &mut Context, dim: i64) {
    let ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let attr = IntegerAttr::new(
        ty,
        APInt::from_u64(dim as u64, core::num::NonZero::new(32).unwrap()),
    );
    op.deref_mut(ctx)
        .attributes
        .set(ATTR_KEY_WAX_DIM.clone(), attr);
}

/// The bounds of a loop, as `(lo, hi, step)`.
pub struct Bounds {
    pub lo: Value,
    pub hi: Value,
    pub step: Value,
}

impl ForOp {
    /// A for loop with `carried.len()` values. The body region gets one block
    /// with arguments `(induction_var, carried...)`.
    pub fn new(
        ctx: &mut Context,
        results: Vec<TypeHandle>,
        bounds: Bounds,
        inits: Vec<Value>,
        iv_ty: TypeHandle,
        carried: Vec<TypeHandle>,
    ) -> Self {
        let Bounds { lo, hi, step } = bounds;
        let mut operands = vec![lo, hi, step];
        operands.extend(inits);
        let op = ForOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                results,
                operands,
                vec![],
                1,
            ),
        };
        let region = op.get_operation().deref_mut(ctx).get_region(0);
        let mut args = vec![iv_ty];
        args.extend(carried);
        let body = pliron::basic_block::BasicBlock::new(ctx, Some(pliron::ident!("body")), args);
        body.insert_at_front(region, ctx);
        op
    }

    pub fn body(&self, ctx: &Context) -> Ptr<pliron::basic_block::BasicBlock> {
        let region = self.get_operation().deref(ctx).get_region(0);
        region
            .deref(ctx)
            .get_entry_block()
            .expect("for body is empty")
    }
}

impl IfOp {
    /// If statements always have the `then` region, while the `else` region is optional.
    pub fn new(ctx: &mut Context, results: Vec<TypeHandle>, cond: Value, with_else: bool) -> Self {
        let n = if with_else { 2 } else { 1 };
        let op = IfOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                results,
                vec![cond],
                vec![],
                n,
            ),
        };
        for i in 0..n {
            let region = op.get_operation().deref_mut(ctx).get_region(i);
            let label = if i == 0 {
                pliron::ident!("then")
            } else {
                pliron::ident!("else")
            };
            let b = pliron::basic_block::BasicBlock::new(ctx, Some(label), vec![]);
            b.insert_at_front(region, ctx);
        }
        op
    }

    pub fn arm(&self, ctx: &Context, idx: usize) -> Ptr<pliron::basic_block::BasicBlock> {
        let region = self.get_operation().deref(ctx).get_region(idx);
        region
            .deref(ctx)
            .get_entry_block()
            .expect("if arm is empty")
    }
}

impl YieldOp {
    pub fn new(ctx: &mut Context, vals: Vec<Value>) -> Self {
        YieldOp {
            op: Operation::new(ctx, Self::get_concrete_op_info(), vec![], vals, vec![], 0),
        }
    }
}

impl ContinueOp {
    pub fn new(ctx: &mut Context, vals: Vec<Value>) -> Self {
        ContinueOp {
            op: Operation::new(ctx, Self::get_concrete_op_info(), vec![], vals, vec![], 0),
        }
    }
}

impl ReturnOp {
    pub fn new(ctx: &mut Context) -> Self {
        ReturnOp {
            op: Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0),
        }
    }
}

impl MakeTokenOp {
    pub fn new(ctx: &mut Context, res_ty: TypeHandle) -> Self {
        MakeTokenOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![res_ty],
                vec![],
                vec![],
                0,
            ),
        }
    }
}

impl LoadViewOp {
    /// `LoadViewOp` is variadic wrt `indices.len()`, as this indicates the rank of the view.
    /// `res_tys` is `[tile, token]`.
    pub fn new(
        ctx: &mut Context,
        res_tys: Vec<TypeHandle>,
        view: Value,
        indices: Vec<Value>,
    ) -> Self {
        let mut opds = vec![view];
        opds.extend(indices);
        LoadViewOp {
            op: Operation::new(ctx, Self::get_concrete_op_info(), res_tys, opds, vec![], 0),
        }
    }
}

impl StoreViewOp {
    pub fn new(
        ctx: &mut Context,
        res_tys: Vec<TypeHandle>,
        val: Value,
        view: Value,
        rest: Vec<Value>,
    ) -> Self {
        let mut opds = vec![val, view];
        opds.extend(rest);
        StoreViewOp {
            op: Operation::new(ctx, Self::get_concrete_op_info(), res_tys, opds, vec![], 0),
        }
    }
}

impl MakeTensorViewOp {
    pub fn new_with(ctx: &mut Context, res_ty: TypeHandle, opds: Vec<Value>) -> Self {
        MakeTensorViewOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![res_ty],
                opds,
                vec![],
                0,
            ),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum WaxVerifyErr {
    #[error("wax.for: {0}")]
    For(&'static str),
    #[error("wax.if needs a condition and one or two regions, each with an entry block")]
    If,
}

// Verifying structured control flow (`wax.for` and `wax.if`).

impl ForOp {
    fn body_shape(&self, ctx: &Context) -> Result<(Ptr<BasicBlock>, Ptr<Operation>), WaxVerifyErr> {
        let op = self.get_operation();
        let (n_regions, n_res, n_opds) = {
            let o = op.deref(ctx);
            (o.num_regions(), o.get_num_results(), o.get_num_operands())
        };
        // Single region
        if n_regions != 1 {
            return Err(WaxVerifyErr::For("Entry block must have a single region."));
        }
        // `lo`, `hi`, `step` + 1 per iter value
        if n_opds != 3 + n_res {
            return Err(WaxVerifyErr::For("Entry block has incorrect operands."));
        }
        // Induction variable + 1 per iter value
        let Some(entry) = op.deref(ctx).get_region(0).deref(ctx).get_entry_block() else {
            return Err(WaxVerifyErr::For("Missing body."));
        };
        if entry.deref(ctx).get_num_arguments() != 1 + n_res {
            return Err(WaxVerifyErr::For("Body has incorrect arguments."));
        }
        // Terminator must have same amount of operands as there are results.
        let Some(term) = entry.deref(ctx).get_terminator(ctx) else {
            return Err(WaxVerifyErr::For("Missing terminator."));
        };
        if term.deref(ctx).get_num_operands() != n_res {
            return Err(WaxVerifyErr::For("Terminator has incorrect operands."));
        }
        Ok((entry, term))
    }
}

#[pliron::derive::op_interface_impl]
impl RegionBranchOpInterface for ForOp {
    fn verify_region_edges_layout(&self, ctx: &Context) -> pliron::result::Result<()> {
        if let Err(err) = self.body_shape(ctx) {
            return pliron::verify_err!(self.loc(ctx), err);
        }
        Ok(())
    }

    fn region_edges(&self, ctx: &Context) -> Vec<RegionEdge> {
        let Ok((entry, term)) = self.body_shape(ctx) else {
            return Vec::new();
        };
        let op = self.get_operation();
        let n = op.deref(ctx).get_num_results();
        let body = op.deref(ctx).get_region(0);
        let init = |i: usize| op.deref(ctx).get_operand_as_use(3 + i);
        let carried = |i: usize| entry.deref(ctx).get_argument(1 + i);
        let result = |i: usize| op.deref(ctx).get_result(i);
        let yielded = |i: usize| term.deref(ctx).get_operand_as_use(i);
        let (lo, hi, step) = {
            let o = op.deref(ctx);
            (
                o.get_operand_as_use(0),
                o.get_operand_as_use(1),
                o.get_operand_as_use(2),
            )
        };
        vec![
            // Entering loop.
            RegionEdge {
                from: RegionBranchPoint::Parent,
                to: RegionSuccessor::Region(body),
                flows: (0..n).map(|i| (init(i), carried(i))).collect(),
                reads: vec![lo, hi, step],
            },
            // Skip the loop. Inits are the results.
            RegionEdge {
                from: RegionBranchPoint::Parent,
                to: RegionSuccessor::After,
                flows: (0..n).map(|i| (init(i), result(i))).collect(),
                reads: vec![lo, hi, step],
            },
            // The back edge of the loop.
            RegionEdge {
                from: RegionBranchPoint::Terminator(term),
                to: RegionSuccessor::Region(body),
                flows: (0..n).map(|i| (yielded(i), carried(i))).collect(),
                reads: vec![hi, step],
            },
            // Exiting loop.
            RegionEdge {
                from: RegionBranchPoint::Terminator(term),
                to: RegionSuccessor::After,
                flows: (0..n).map(|i| (yielded(i), result(i))).collect(),
                reads: vec![hi],
            },
        ]
    }
}

type Arms = Vec<(Ptr<Region>, Option<Ptr<Operation>>)>;

impl IfOp {
    fn arms(&self, ctx: &Context) -> Result<Arms, WaxVerifyErr> {
        let op = self.get_operation();
        let (n_regions, n_res, n_opds) = {
            let o = op.deref(ctx);
            (o.num_regions(), o.get_num_results(), o.get_num_operands())
        };
        // TODO: Tighten. Report specific error like we do for `ForOp`.
        if n_opds == 0 || !(1..=2).contains(&n_regions) {
            return Err(WaxVerifyErr::If);
        }
        Ok((0..n_regions)
            .filter_map(|i| {
                let r = op.deref(ctx).get_region(i);
                let entry = r.deref(ctx).get_entry_block()?;
                let term = entry
                    .deref(ctx)
                    .get_terminator(ctx)
                    .filter(|t| t.deref(ctx).get_num_operands() == n_res);
                Some((r, term))
            })
            .collect())
    }
}

#[pliron::derive::op_interface_impl]
impl RegionBranchOpInterface for IfOp {
    fn verify_region_edges_layout(&self, ctx: &Context) -> pliron::result::Result<()> {
        if let Err(err) = self.arms(ctx) {
            return pliron::verify_err!(self.loc(ctx), err);
        }
        Ok(())
    }

    fn region_edges(&self, ctx: &Context) -> Vec<RegionEdge> {
        let Ok(arms) = self.arms(ctx) else {
            return Vec::new();
        };
        let op = self.get_operation();
        let n = op.deref(ctx).get_num_results();
        let cond = op.deref(ctx).get_operand_as_use(0);
        let result = |i: usize| op.deref(ctx).get_result(i);
        let mut edges = Vec::with_capacity(2 * arms.len() + 1);
        for &(region, term) in &arms {
            // Entering arm.
            edges.push(RegionEdge {
                from: RegionBranchPoint::Parent,
                to: RegionSuccessor::Region(region),
                flows: Vec::new(),
                reads: vec![cond],
            });
            if let Some(term) = term {
                // Arm produces value
                edges.push(RegionEdge {
                    from: RegionBranchPoint::Terminator(term),
                    to: RegionSuccessor::After,
                    flows: (0..n)
                        .map(|i| (term.deref(ctx).get_operand_as_use(i), result(i)))
                        .collect(),
                    reads: Vec::new(),
                });
            }
        }
        if arms.len() < 2 {
            // Must still be possible to skip if-statement even if there is no "else" arm.
            edges.push(RegionEdge {
                from: RegionBranchPoint::Parent,
                to: RegionSuccessor::After,
                flows: Vec::new(),
                reads: vec![cond],
            });
        }
        edges
    }
}
