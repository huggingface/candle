//! Bridge: a `cfg.rs` [`FuncBuilder`] into the `air` dialect.
//!
//! The one structural change is PHIS BECOMING BLOCK ARGUMENTS. `cfg.rs` stores per block a
//! list of phis, each with `(incoming, predecessor)` pairs. pliron has no phi: a block
//! declares arguments and each branch supplies them positionally. So for a block `T` with
//! phis `p0..pn`, every predecessor `B` passes `[incoming_from_B(p0), ..]` - and phi index
//! `i` becomes block argument `i`, which is what lets the emitter rebuild phi records in
//! the same order `to_bitcode` produces today.
//!
//! Immediates become constant OPS rather than inline operands. That matches bitcode, where
//! a constant is a pool entry rather than an instruction, and it removes `ImmVal::Text` as
//! a representable state.

use std::collections::HashMap;

use pliron::basic_block::BasicBlock;
use pliron::builtin::op_interfaces::OneRegionInterface;
use pliron::builtin::ops::FuncOp;
use pliron::builtin::types::FunctionType;
use pliron::context::{Context, Ptr};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::value::Value;

use super::attrs::{BinOpAttr, CastOpAttr, FCmpPredAttr, ICmpPredAttr};
use super::ops::*;
use super::types::to_pliron;
use crate::lower::cfg::{BlockId, FuncBuilder, GepBase, ImmVal, InstKind, Operand, Terminator};

/// Why a function could not be expressed in the `air` dialect.
#[derive(Debug, PartialEq)]
pub enum BridgeErr {
    /// The `InstKind::Raw` escape hatch. No production path calls `FuncBuilder::raw`, but
    /// `cfg.rs` can still represent it, so it is refused rather than guessed at.
    RawInstruction(String),
    /// An immediate that only ever existed as LLVM constant text.
    TextImmediate(String),
    /// An operator spelling the typed attributes do not cover.
    UnknownOperator(String),
    /// A phi with no incoming registered for one of its block's predecessors.
    MissingIncoming { block: usize, phi: usize },
}

impl std::fmt::Display for BridgeErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BridgeErr::RawInstruction(t) => write!(f, "raw instruction `{t}`"),
            BridgeErr::TextImmediate(t) => write!(f, "text-only immediate `{t}`"),
            BridgeErr::UnknownOperator(o) => write!(f, "unknown operator `{o}`"),
            BridgeErr::MissingIncoming { block, phi } => {
                write!(
                    f,
                    "phi {phi} of block {block} has no incoming for a predecessor"
                )
            }
        }
    }
}

/// The `air` form of one function.
pub struct AirFunc {
    pub func: FuncOp,
    /// cfg block index -> pliron block.
    pub blocks: Vec<Ptr<BasicBlock>>,
}

/// Translate `f` into a `builtin.func` whose body is `air` ops. `name` is passed in
/// because `FuncBuilder` does not carry the mangled symbol.
pub fn build(ctx: &mut Context, f: &FuncBuilder, name: &str) -> Result<AirFunc, BridgeErr> {
    let param_tys: Vec<_> = f.param_types().iter().map(|t| to_pliron(ctx, t)).collect();
    let fn_ty = FunctionType::get(ctx, param_tys, vec![]);
    let func = FuncOp::new(ctx, name.try_into().expect("kernel symbol"), fn_ty);
    let region = func.get_region(ctx);

    // All blocks first: a branch can target a block defined later.
    let mut blocks = Vec::with_capacity(f.num_blocks());
    for bi in 0..f.num_blocks() {
        let arg_tys: Vec<_> = f
            .block_phis(BlockId(bi as u32))
            .iter()
            .map(|p| to_pliron(ctx, &f.value_type(*p)))
            .collect();
        let b = if bi == 0 {
            // An entry block has no predecessors, so it cannot carry a phi.
            debug_assert!(arg_tys.is_empty(), "entry block has phis");
            func.get_entry_block(ctx)
        } else {
            let b = BasicBlock::new(ctx, Some(format!("bb{bi}").try_into().unwrap()), arg_tys);
            b.insert_at_back(region, ctx);
            b
        };
        blocks.push(b);
    }

    // cfg ValueId -> pliron Value: parameters, then phis as block args, then results.
    let mut vals: HashMap<u32, Value> = HashMap::new();
    for i in 0..f.num_params() {
        vals.insert(f.param_id(i).0, blocks[0].deref(ctx).get_argument(i));
    }
    for (bi, blk) in blocks.iter().enumerate() {
        for (pi, phi) in f.block_phis(BlockId(bi as u32)).iter().enumerate() {
            vals.insert(phi.0, blk.deref(ctx).get_argument(pi));
        }
    }

    for bi in 0..f.num_blocks() {
        let b = BlockId(bi as u32);
        let blk = blocks[bi];
        for (res, kind) in f.block_insts(b) {
            let kind = kind.clone();
            translate_inst(ctx, f, res.map(|r| r.0), &kind, blk, &mut vals)?;
        }
        translate_term(ctx, f, bi, &blocks, blk, &mut vals)?;
    }
    Ok(AirFunc { func, blocks })
}

/// Materialise an operand as an SSA value, creating a constant op for an immediate.
fn operand(
    ctx: &mut Context,
    op: &Operand,
    blk: Ptr<BasicBlock>,
    vals: &mut HashMap<u32, Value>,
) -> Result<Value, BridgeErr> {
    match op {
        Operand::Val(v) => Ok(*vals
            .get(&v.0)
            .unwrap_or_else(|| panic!("unresolved value %v{}", v.0))),
        Operand::Imm { text, ty, val } => {
            let res_ty = to_pliron(ctx, ty);
            let o: Ptr<Operation> = match val {
                ImmVal::Int(i) => ConstBitsOp::new(ctx, res_ty, *i as u64).get_operation(),
                ImmVal::FloatBits(b) => ConstBitsOp::new(ctx, res_ty, *b).get_operation(),
                ImmVal::Float(x) => {
                    ConstBitsOp::new(ctx, res_ty, x.to_bits() as u64).get_operation()
                }
                ImmVal::Undef => UndefOp::new(ctx, res_ty).get_operation(),
                ImmVal::Zero => ZeroOp::new(ctx, res_ty).get_operation(),
                ImmVal::Global(s) => GlobalRefOp::new(ctx, res_ty, s).get_operation(),
                ImmVal::VecInt(v) => {
                    ConstVecOp::new(ctx, res_ty, v.iter().map(|x| *x as u64).collect())
                        .get_operation()
                }
                ImmVal::VecFloatBits(b) => ConstVecOp::new(ctx, res_ty, b.clone()).get_operation(),
                ImmVal::Text => return Err(BridgeErr::TextImmediate(text.clone())),
            };
            o.insert_at_back(blk, ctx);
            Ok(o.deref(ctx).get_result(0))
        }
    }
}

fn translate_inst(
    ctx: &mut Context,
    f: &FuncBuilder,
    res: Option<u32>,
    kind: &InstKind,
    blk: Ptr<BasicBlock>,
    vals: &mut HashMap<u32, Value>,
) -> Result<(), BridgeErr> {
    // The result type of an op comes from the value table, except where the instruction
    // names it (a cast) or it is fixed (a compare yields i1).
    let res_ty = |ctx: &mut Context| -> Option<pliron::r#type::TypeHandle> {
        res.map(|r| to_pliron(ctx, &f.value_type(crate::lower::cfg::ValueId(r))))
    };

    let o: Ptr<Operation> = match kind {
        InstKind::Bin { op, lhs, rhs } => {
            let a = operand(ctx, lhs, blk, vals)?;
            let b = operand(ctx, rhs, blk, vals)?;
            let ty = res_ty(ctx).expect("bin has a result");
            let attr =
                BinOpAttr::from_llvm(op).ok_or(BridgeErr::UnknownOperator(op.to_string()))?;
            let i = BinOp::new(ctx, ty, a, b);
            i.set_op(ctx, attr);
            i.get_operation()
        }
        InstKind::Cast { op, src, to } => {
            let s = operand(ctx, src, blk, vals)?;
            let ty = to_pliron(ctx, to);
            let attr =
                CastOpAttr::from_llvm(op).ok_or(BridgeErr::UnknownOperator(op.to_string()))?;
            let i = CastOp::new(ctx, ty, s);
            i.set_op(ctx, attr);
            i.get_operation()
        }
        InstKind::Icmp { pred, lhs, rhs } => {
            let a = operand(ctx, lhs, blk, vals)?;
            let b = operand(ctx, rhs, blk, vals)?;
            let ty = to_pliron(ctx, &crate::ir_builder::LlType::I1);
            let attr = ICmpPredAttr::from_llvm(pred)
                .ok_or(BridgeErr::UnknownOperator(pred.to_string()))?;
            let i = ICmpOp::new(ctx, ty, a, b);
            i.set_pred(ctx, attr);
            i.get_operation()
        }
        InstKind::Fcmp { pred, lhs, rhs } => {
            let a = operand(ctx, lhs, blk, vals)?;
            let b = operand(ctx, rhs, blk, vals)?;
            let ty = to_pliron(ctx, &crate::ir_builder::LlType::I1);
            let attr = FCmpPredAttr::from_llvm(pred)
                .ok_or(BridgeErr::UnknownOperator(pred.to_string()))?;
            let i = FCmpOp::new(ctx, ty, a, b);
            i.set_pred(ctx, attr);
            i.get_operation()
        }
        InstKind::Select { cond, t, f: fv } => {
            let c = operand(ctx, cond, blk, vals)?;
            let a = operand(ctx, t, blk, vals)?;
            let b = operand(ctx, fv, blk, vals)?;
            let ty = res_ty(ctx).expect("select has a result");
            SelectOp::new(ctx, ty, c, a, b).get_operation()
        }
        InstKind::Load { ptr, align } => {
            let p = operand(ctx, ptr, blk, vals)?;
            let ty = res_ty(ctx).expect("load has a result");
            let i = LoadOp::new(ctx, ty, p);
            let o = i.get_operation();
            set_align(o, ctx, *align);
            o
        }
        InstKind::Store { val, ptr, align } => {
            let v = operand(ctx, val, blk, vals)?;
            let p = operand(ctx, ptr, blk, vals)?;
            let o = StoreOp::new(ctx, v, p).get_operation();
            set_align(o, ctx, *align);
            o
        }
        InstKind::Gep { elem, ptr, idx } => {
            let p = operand(ctx, ptr, blk, vals)?;
            let i = operand(ctx, idx, blk, vals)?;
            let ty = res_ty(ctx).expect("gep has a result");
            let elem_ty = to_pliron(ctx, elem);
            let o = GepOp::new(ctx, ty, p, i).get_operation();
            set_elem_ty(o, ctx, elem_ty);
            o
        }
        InstKind::ExtractElem { vec, idx } => {
            let v = operand(ctx, vec, blk, vals)?;
            let i = operand(ctx, idx, blk, vals)?;
            let ty = res_ty(ctx).expect("extractelement has a result");
            ExtractElemOp::new(ctx, ty, v, i).get_operation()
        }
        InstKind::InsertElem { vec, elem, idx } => {
            let v = operand(ctx, vec, blk, vals)?;
            let e = operand(ctx, elem, blk, vals)?;
            let i = operand(ctx, idx, blk, vals)?;
            let ty = res_ty(ctx).expect("insertelement has a result");
            InsertElemOp::new(ctx, ty, v, e, i).get_operation()
        }
        InstKind::GepArray {
            base,
            elem,
            count,
            addrspace: _,
            idx,
        } => {
            let ty = res_ty(ctx).expect("gep has a result");
            let elem_ty = to_pliron(ctx, elem);
            // A symbol base becomes an explicit `air.global_ref` so the op has a uniform
            // pointer operand; the emitter turns it back into a `Val::Global`.
            let p = match base {
                GepBase::Value(v) => operand(ctx, v, blk, vals)?,
                GepBase::Global(sym) => {
                    let g = GlobalRefOp::new(ctx, ty, sym).get_operation();
                    g.insert_at_back(blk, ctx);
                    g.deref(ctx).get_result(0)
                }
            };
            let i = operand(ctx, idx, blk, vals)?;
            let o = GepArrayOp::new(ctx, ty, p, i).get_operation();
            set_elem_ty(o, ctx, elem_ty);
            set_count(o, ctx, *count);
            o
        }
        InstKind::Alloca { elem, count, align } => {
            let ty = res_ty(ctx).expect("alloca has a result");
            let elem_ty = to_pliron(ctx, elem);
            // No count OPERAND: the bitcode form puts the length in the TYPE and always
            // passes a literal 1, so materialising one here would intern a stray constant
            // and shift the pool.
            let o = AllocaOp::new(ctx, ty, None).get_operation();
            set_elem_ty(o, ctx, elem_ty);
            set_align(o, ctx, *align);
            if let Some(c) = count {
                set_count(o, ctx, *c);
            }
            o
        }
        InstKind::Call { callee, args, fast } => {
            let mut vs = Vec::with_capacity(args.len());
            for a in args {
                vs.push(operand(ctx, a, blk, vals)?);
            }
            let ty = res_ty(ctx);
            let o = CallOp::new(ctx, callee, ty, vs).get_operation();
            if *fast {
                set_fast(o, ctx);
            }
            o
        }
        InstKind::Raw { template, .. } => {
            return Err(BridgeErr::RawInstruction(template.clone()));
        }
    };

    o.insert_at_back(blk, ctx);
    if let Some(r) = res {
        let v = o.deref(ctx).get_result(0);
        vals.insert(r, v);
    }
    Ok(())
}

fn translate_term(
    ctx: &mut Context,
    f: &FuncBuilder,
    bi: usize,
    blocks: &[Ptr<BasicBlock>],
    blk: Ptr<BasicBlock>,
    vals: &mut HashMap<u32, Value>,
) -> Result<(), BridgeErr> {
    let term = f
        .block_term(BlockId(bi as u32))
        .expect("block has no terminator")
        .clone();
    let o: Ptr<Operation> = match term {
        Terminator::Ret => RetOp::new(ctx).get_operation(),
        Terminator::Unreachable => UnreachableOp::new(ctx).get_operation(),
        Terminator::Br(t) => {
            let opds = edge_operands(ctx, f, bi, t.0 as usize, blk, vals)?;
            BrOp::new(ctx, blocks[t.0 as usize], opds).get_operation()
        }
        Terminator::CondBr { cond, then_, else_ } => {
            let c = operand(ctx, &cond, blk, vals)?;
            let t_opds = edge_operands(ctx, f, bi, then_.0 as usize, blk, vals)?;
            let e_opds = edge_operands(ctx, f, bi, else_.0 as usize, blk, vals)?;
            CondBrOp::new(
                ctx,
                c,
                blocks[then_.0 as usize],
                t_opds,
                blocks[else_.0 as usize],
                e_opds,
            )
            .get_operation()
        }
    };
    o.insert_at_back(blk, ctx);
    // The unroll hint rides on the terminator, as it does in the bitcode body.
    if let Some(count) = f.block_loop_unroll(BlockId(bi as u32)) {
        set_unroll(o, ctx, count);
    }
    Ok(())
}

/// What this edge forwards: for each phi of the target, the incoming registered for THIS
/// predecessor, in phi order.
fn edge_operands(
    ctx: &mut Context,
    f: &FuncBuilder,
    from: usize,
    to: usize,
    blk: Ptr<BasicBlock>,
    vals: &mut HashMap<u32, Value>,
) -> Result<Vec<Value>, BridgeErr> {
    let mut out = Vec::new();
    for (pi, phi) in f.block_phis(BlockId(to as u32)).iter().enumerate() {
        let incs = f.phi_incomings(*phi);
        let inc = incs
            .iter()
            .find(|(_, pred)| pred.0 as usize == from)
            .ok_or(BridgeErr::MissingIncoming { block: to, phi: pi })?;
        out.push(operand(ctx, &inc.0, blk, vals)?);
    }
    Ok(out)
}
