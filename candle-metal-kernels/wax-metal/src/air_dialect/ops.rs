//! Ops of the `air` dialect: the emitter's instruction set as pliron ops.
//!
//! Mirrors `cfg.rs`'s `InstKind` and `Terminator` one-for-one, with two deliberate
//! differences:
//!
//! 1. NO PHI OP. pliron has block arguments and no phi anywhere - not in `builtin`, not
//!    in its LLVM dialect - so a phi becomes a block argument and the branch carries the
//!    incoming value as a successor operand. This is MLIR's model and it is strictly more
//!    canonical (no phi ordering to agree on, and an incoming value is an ordinary use),
//!    but it means the bitcode emitter has to rebuild phi records. See `to_bitcode`.
//! 2. The operator of a bin/cast/compare is a typed attribute, not a `&'static str`, so
//!    the emitter's match is exhaustive.
//!
//! `InstKind::Raw` has NO op here on purpose: it is an LLVM-text escape hatch with no
//! typed content, so it cannot cross into a typed IR. A function containing one is
//! reported as unsupported rather than half-translated - the same policy `to_bitcode`
//! already applies to `ImmVal::Text`.

use pliron::builtin::attributes::{IntegerAttr, StringAttr, TypeAttr};
use pliron::builtin::op_interfaces::{
    BranchOpInterface, IsTerminatorInterface, NResultsInterface, NSuccsInterface, OneOpdInterface,
    OneResultInterface, OperandSegmentInterface,
};
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::derive::{op_interface_impl, pliron_op};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::TypeHandle;
use pliron::utils::apint::APInt;
use pliron::value::Value;

use super::attrs::{BinOpAttr, CastOpAttr, FCmpPredAttr, ICmpPredAttr};

/// Key under which an op stores its operator/predicate attribute.
/// Underscored, not dotted: a pliron `Identifier` rejects a `.`.
pub static ATTR_KEY_AIR_OP: &str = "air_op";
/// Key for a load/store/alloca alignment in bytes.
pub static ATTR_KEY_AIR_ALIGN: &str = "air_align";
/// Key for the element type a GEP or alloca steps over.
pub static ATTR_KEY_AIR_ELEM: &str = "air_elem";
/// Key for a scalar constant's raw bits.
pub static ATTR_KEY_AIR_VALUE: &str = "air_value";
/// Key marking a call as `call fast`.
pub static ATTR_KEY_AIR_FAST: &str = "air_fast";
/// Key for an array length (gep_array, alloca).
pub static ATTR_KEY_AIR_COUNT: &str = "air_count";
/// Key for a loop latch's `!llvm.loop` unroll count.
pub static ATTR_KEY_AIR_UNROLL: &str = "air_unroll";
/// Key for a call's callee symbol, and for a global reference's symbol.
pub static ATTR_KEY_AIR_CALLEE: &str = "air_callee";

macro_rules! simple_new {
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

/// `add`/`fmul`/... - operator in the [`BinOpAttr`] under [`ATTR_KEY_AIR_OP`].
#[pliron_op(
    name = "air.bin",
    interfaces = [OneResultInterface],
    operands = (lhs, rhs),
    format,
    verifier = "succ"
)]
pub struct BinOp;
simple_new!(BinOp, lhs, rhs);

impl BinOp {
    pub fn set_op(&self, ctx: &Context, op: BinOpAttr) {
        self.get_operation()
            .deref_mut(ctx)
            .attributes
            .set(ATTR_KEY_AIR_OP.try_into().unwrap(), op);
    }
}

/// A conversion; the target type is the result type.
#[pliron_op(
    name = "air.cast",
    interfaces = [OneResultInterface, OneOpdInterface],
    operands = (src),
    format,
    verifier = "succ"
)]
pub struct CastOp;
simple_new!(CastOp, src);

impl CastOp {
    pub fn set_op(&self, ctx: &Context, op: CastOpAttr) {
        self.get_operation()
            .deref_mut(ctx)
            .attributes
            .set(ATTR_KEY_AIR_OP.try_into().unwrap(), op);
    }
}

/// Integer compare; result is i1.
#[pliron_op(name = "air.icmp", interfaces = [OneResultInterface], operands = (lhs, rhs),
    format,
    verifier = "succ"
)]
pub struct ICmpOp;
simple_new!(ICmpOp, lhs, rhs);

impl ICmpOp {
    pub fn set_pred(&self, ctx: &Context, p: ICmpPredAttr) {
        self.get_operation()
            .deref_mut(ctx)
            .attributes
            .set(ATTR_KEY_AIR_OP.try_into().unwrap(), p);
    }
}

/// Float compare; result is i1.
#[pliron_op(name = "air.fcmp", interfaces = [OneResultInterface], operands = (lhs, rhs),
    format,
    verifier = "succ"
)]
pub struct FCmpOp;
simple_new!(FCmpOp, lhs, rhs);

impl FCmpOp {
    pub fn set_pred(&self, ctx: &Context, p: FCmpPredAttr) {
        self.get_operation()
            .deref_mut(ctx)
            .attributes
            .set(ATTR_KEY_AIR_OP.try_into().unwrap(), p);
    }
}

#[pliron_op(
    name = "air.select",
    interfaces = [OneResultInterface],
    operands = (cond, t, f),
    format,
    verifier = "succ"
)]
pub struct SelectOp;
simple_new!(SelectOp, cond, t, f);

#[pliron_op(
    name = "air.load",
    interfaces = [OneResultInterface, OneOpdInterface],
    operands = (address),
    format,
    verifier = "succ"
)]
pub struct LoadOp;
simple_new!(LoadOp, address);

/// No result.
#[pliron_op(
    name = "air.store",
    interfaces = [NResultsInterface<0>],
    operands = (value, address),
    format,
    verifier = "succ"
)]
pub struct StoreOp;

impl StoreOp {
    pub fn new(ctx: &mut Context, value: Value, address: Value) -> Self {
        StoreOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                vec![value, address],
                vec![],
                0,
            ),
        }
    }
}

/// `getelementptr elem, ptr, idx` - element type in [`ATTR_KEY_AIR_ELEM`].
#[pliron_op(
    name = "air.gep",
    interfaces = [OneResultInterface],
    operands = (address, idx),
    format,
    verifier = "succ"
)]
pub struct GepOp;
simple_new!(GepOp, address, idx);

#[pliron_op(
    name = "air.extract_elem",
    interfaces = [OneResultInterface],
    operands = (vector, idx),
    format,
    verifier = "succ"
)]
pub struct ExtractElemOp;
simple_new!(ExtractElemOp, vector, idx);

#[pliron_op(
    name = "air.insert_elem",
    interfaces = [OneResultInterface],
    operands = (vector, elem, idx),
    format,
    verifier = "succ"
)]
pub struct InsertElemOp;
simple_new!(InsertElemOp, vector, elem, idx);

/// A thread-private stack slot. Element type in [`ATTR_KEY_AIR_ELEM`]; an array slot
/// carries its length as the single operand, a scalar slot has none.
#[pliron_op(name = "air.alloca", interfaces = [OneResultInterface],
    format,
    verifier = "succ"
)]
pub struct AllocaOp;

impl AllocaOp {
    pub fn new(ctx: &mut Context, res_ty: TypeHandle, count: Option<Value>) -> Self {
        AllocaOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![res_ty],
                count.into_iter().collect(),
                vec![],
                0,
            ),
        }
    }
}

/// A call. Variadic operands; callee symbol in [`ATTR_KEY_AIR_CALLEE`]. A void call
/// declares no result.
#[pliron_op(name = "air.call", format, verifier = "succ")]
pub struct CallOp;

impl CallOp {
    pub fn new(
        ctx: &mut Context,
        callee: &str,
        res_ty: Option<TypeHandle>,
        args: Vec<Value>,
    ) -> Self {
        let op = CallOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                res_ty.into_iter().collect(),
                args,
                vec![],
                0,
            ),
        };
        // A STRING, not an `Identifier`: AIR intrinsic symbols are dotted and carry a
        // leading `@` (`@air.simdgroup_matrix_8x8_load.v64f32.p1f32`), both of which a
        // pliron `Identifier` rejects.
        op.get_operation().deref_mut(ctx).attributes.set(
            ATTR_KEY_AIR_CALLEE.try_into().unwrap(),
            StringAttr::new(callee.to_string()),
        );
        op
    }
}

// Terminators. Successor operands carry what `cfg.rs` spells as phi incomings.

#[pliron_op(
    name = "air.br",
    interfaces = [IsTerminatorInterface, NResultsInterface<0>, NSuccsInterface<1>],
    operands = (dest_opds),
    format,
    verifier = "succ"
)]
pub struct BrOp;

#[op_interface_impl]
impl BranchOpInterface for BrOp {
    fn successor_operands(&self, ctx: &Context, succ_idx: usize) -> Vec<Value> {
        assert!(succ_idx == 0, "air.br has one successor");
        self.get_operation().deref(ctx).operands().collect()
    }
    fn add_successor_operand(&self, ctx: &mut Context, succ_idx: usize, operand: Value) -> usize {
        assert!(succ_idx == 0, "air.br has one successor");
        Operation::push_operand(self.get_operation(), ctx, operand)
    }
    fn remove_successor_operand(
        &self,
        ctx: &mut Context,
        succ_idx: usize,
        opd_idx: usize,
    ) -> Value {
        assert!(succ_idx == 0, "air.br has one successor");
        Operation::remove_operand(self.get_operation(), ctx, opd_idx)
    }
}

impl BrOp {
    pub fn new(
        ctx: &mut Context,
        dest: Ptr<pliron::basic_block::BasicBlock>,
        opds: Vec<Value>,
    ) -> Self {
        BrOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                opds,
                vec![dest],
                0,
            ),
        }
    }
}

#[pliron_op(
    name = "air.cond_br",
    interfaces = [
        IsTerminatorInterface,
        NResultsInterface<0>,
        NSuccsInterface<2>,
        OperandSegmentInterface
    ],
    operands = (condition, true_dest_opds, false_dest_opds),
    format,
    verifier = "succ"
)]
pub struct CondBrOp;

impl CondBrOp {
    pub fn new(
        ctx: &mut Context,
        condition: Value,
        true_dest: Ptr<pliron::basic_block::BasicBlock>,
        true_dest_opds: Vec<Value>,
        false_dest: Ptr<pliron::basic_block::BasicBlock>,
        false_dest_opds: Vec<Value>,
    ) -> Self {
        let (operands, segment_sizes) =
            Self::compute_segment_sizes(vec![vec![condition], true_dest_opds, false_dest_opds]);
        let op = CondBrOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                operands,
                vec![true_dest, false_dest],
                0,
            ),
        };
        op.set_operand_segment_sizes(ctx, segment_sizes);
        op
    }
}

#[op_interface_impl]
impl BranchOpInterface for CondBrOp {
    // Segment 0 is the condition, so successor i is segment i + 1.
    fn successor_operands(&self, ctx: &Context, succ_idx: usize) -> Vec<Value> {
        assert!(succ_idx < 2, "air.cond_br has two successors");
        self.get_segment(ctx, succ_idx + 1)
    }
    fn add_successor_operand(&self, ctx: &mut Context, succ_idx: usize, operand: Value) -> usize {
        self.push_to_segment(ctx, succ_idx + 1, operand)
    }
    fn remove_successor_operand(
        &self,
        ctx: &mut Context,
        succ_idx: usize,
        opd_idx: usize,
    ) -> Value {
        self.remove_from_segment(ctx, succ_idx + 1, opd_idx)
    }
}

#[pliron_op(
    name = "air.ret",
    interfaces = [IsTerminatorInterface, NResultsInterface<0>],
    format,
    verifier = "succ"
)]
pub struct RetOp;

impl RetOp {
    pub fn new(ctx: &mut Context) -> Self {
        RetOp {
            op: Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0),
        }
    }
}

#[pliron_op(
    name = "air.unreachable",
    interfaces = [IsTerminatorInterface, NResultsInterface<0>],
    format,
    verifier = "succ"
)]
pub struct UnreachableOp;

impl UnreachableOp {
    pub fn new(ctx: &mut Context) -> Self {
        UnreachableOp {
            op: Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0),
        }
    }
}

/// Attach an alignment to a load/store/alloca.
pub fn set_align(op: Ptr<Operation>, ctx: &mut Context, align: u32) {
    let ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let attr = IntegerAttr::new(
        ty,
        APInt::from_u64(align as u64, core::num::NonZero::new(32).unwrap()),
    );
    op.deref_mut(ctx)
        .attributes
        .set(ATTR_KEY_AIR_ALIGN.try_into().unwrap(), attr);
}

/// Attach the element type a GEP or alloca steps over.
pub fn set_elem_ty(op: Ptr<Operation>, ctx: &Context, ty: TypeHandle) {
    op.deref_mut(ctx)
        .attributes
        .set(ATTR_KEY_AIR_ELEM.try_into().unwrap(), TypeAttr::new(ty));
}

// Constants. `cfg.rs` carries these inline on the consuming operand; here they are SSA
// values, which is also what bitcode wants (a constant-pool entry, not an instruction).
// The text form of `ImmVal::Text` has no equivalent on purpose - that failure class is
// gone rather than represented.

/// A scalar constant given as its RAW BITS at the result type's width. Covers both
/// `ImmVal::Int` (integer result type) and `ImmVal::FloatBits` (float result type); the
/// result type says how to read them, so one op serves both.
#[pliron_op(name = "air.const_bits", interfaces = [OneResultInterface], format, verifier = "succ")]
pub struct ConstBitsOp;

impl ConstBitsOp {
    pub fn new(ctx: &mut Context, res_ty: TypeHandle, bits: u64) -> Self {
        let op = ConstBitsOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![res_ty],
                vec![],
                vec![],
                0,
            ),
        };
        let ty = IntegerType::get(ctx, 64, Signedness::Signless);
        let attr = IntegerAttr::new(
            ty,
            APInt::from_u64(bits, core::num::NonZero::new(64).unwrap()),
        );
        op.get_operation()
            .deref_mut(ctx)
            .attributes
            .set(ATTR_KEY_AIR_VALUE.try_into().unwrap(), attr);
        op
    }
}

/// `undef`.
#[pliron_op(name = "air.undef", interfaces = [OneResultInterface], format, verifier = "succ")]
pub struct UndefOp;

impl UndefOp {
    pub fn new(ctx: &mut Context, res_ty: TypeHandle) -> Self {
        UndefOp {
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

/// `zeroinitializer`.
#[pliron_op(name = "air.zero", interfaces = [OneResultInterface], format, verifier = "succ")]
pub struct ZeroOp;

impl ZeroOp {
    pub fn new(ctx: &mut Context, res_ty: TypeHandle) -> Self {
        ZeroOp {
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

/// A reference to a module-scope symbol, e.g. threadgroup scratch.
#[pliron_op(name = "air.global_ref", interfaces = [OneResultInterface], format, verifier = "succ")]
pub struct GlobalRefOp;

impl GlobalRefOp {
    pub fn new(ctx: &mut Context, res_ty: TypeHandle, sym: &str) -> Self {
        let op = GlobalRefOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![res_ty],
                vec![],
                vec![],
                0,
            ),
        };
        op.get_operation().deref_mut(ctx).attributes.set(
            ATTR_KEY_AIR_CALLEE.try_into().unwrap(),
            StringAttr::new(sym.to_string()),
        );
        op
    }
}

/// Mark a call as `call fast` (AIR's `air.fast_*` intrinsics are called this way).
pub fn set_fast(op: Ptr<Operation>, ctx: &mut Context) {
    let ty = IntegerType::get(ctx, 1, Signedness::Signless);
    let attr = IntegerAttr::new(ty, APInt::from_u64(1, core::num::NonZero::new(1).unwrap()));
    op.deref_mut(ctx)
        .attributes
        .set(ATTR_KEY_AIR_FAST.try_into().unwrap(), attr);
}

/// A dense vector constant, elements as raw bits. `ImmVal::VecInt` / `VecFloatBits`.
#[pliron_op(name = "air.const_vec", interfaces = [OneResultInterface], format, verifier = "succ")]
pub struct ConstVecOp;

impl ConstVecOp {
    pub fn new(ctx: &mut Context, res_ty: TypeHandle, bits: Vec<u64>) -> Self {
        let op = ConstVecOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![res_ty],
                vec![],
                vec![],
                0,
            ),
        };
        op.get_operation().deref_mut(ctx).attributes.set(
            ATTR_KEY_AIR_VALUE.try_into().unwrap(),
            super::attrs::DenseBitsAttr(bits),
        );
        op
    }
}

/// Indexing a module-scope array, e.g. threadgroup scratch. Distinct from [`GepOp`]
/// because it encodes as `inbounds` with a LEADING ZERO index stepping through the array
/// type - collapsing the two loses that and emits the wrong record.
/// Element type in [`ATTR_KEY_AIR_ELEM`], array length in [`ATTR_KEY_AIR_COUNT`].
#[pliron_op(
    name = "air.gep_array",
    interfaces = [OneResultInterface],
    operands = (address, idx),
    format,
    verifier = "succ"
)]
pub struct GepArrayOp;
simple_new!(GepArrayOp, address, idx);

/// Array length for [`GepArrayOp`] and [`AllocaOp`].
pub fn set_count(op: Ptr<Operation>, ctx: &mut Context, count: u32) {
    let ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let attr = IntegerAttr::new(
        ty,
        APInt::from_u64(count as u64, core::num::NonZero::new(32).unwrap()),
    );
    op.deref_mut(ctx)
        .attributes
        .set(ATTR_KEY_AIR_COUNT.try_into().unwrap(), attr);
}

/// Attach an unroll count to a loop latch's terminator. A pure hint; see
/// `crate::lower::walk`'s note on why the count must be explicit.
pub fn set_unroll(op: Ptr<Operation>, ctx: &mut Context, count: u32) {
    let ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let attr = IntegerAttr::new(
        ty,
        APInt::from_u64(count as u64, core::num::NonZero::new(32).unwrap()),
    );
    op.deref_mut(ctx)
        .attributes
        .set(ATTR_KEY_AIR_UNROLL.try_into().unwrap(), attr);
}
