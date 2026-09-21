//! `air` to bitcode.
//!
//! Each type describes how to emit itself as bitcode.

pub mod impls;

use super::ops::{MD_LLVM_LOOP, *};
use super::types::is_float;
use crate::bitcode::func::pred;
use crate::bitcode::func::{Body, Val};
use crate::lower::to_bitcode::Unsupported;
use pliron::attribute::Attribute;
use pliron::basic_block::BasicBlock;
use pliron::builtin::op_interfaces::BranchOpInterface;
use pliron::builtin::ops::FuncOp;
use pliron::common_traits::Named;
use pliron::context::{Context, Ptr};
use pliron::linked_list::ContainsLinkedList;
use pliron::operation::Operation;
use pliron::printable::Printable;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::{DefiningEntity, Value};
use pliron_llvm::attributes::AggregateAttr;
use std::cell::RefCell;
use std::collections::HashMap;

/// Context, entry function's block, and converted type cache.
pub(crate) struct Cx<'a> {
    ctx: &'a Context,
    entry: Option<Ptr<BasicBlock>>,
    types: RefCell<HashMap<TypeHandle, crate::bitcode::Ty>>,
}

impl<'a> Cx<'a> {
    fn new(ctx: &'a Context, entry: Option<Ptr<BasicBlock>>) -> Self {
        Cx {
            ctx,
            entry,
            types: RefCell::new(HashMap::new()),
        }
    }

    /// Check if provided block is the entry function block.
    fn is_entry(&self, block: Ptr<BasicBlock>) -> bool {
        self.entry == Some(block)
    }

    /// Bitcode type of a pliron type handle, from cache where possible.
    pub(crate) fn ty(&self, h: TypeHandle) -> crate::bitcode::Ty {
        if let Some(t) = self.types.borrow().get(&h) {
            return t.clone();
        }
        let t = super::type_interfaces::bitcode_ty(self.ctx, h);
        self.types.borrow_mut().insert(h, t.clone());
        t
    }
}

/// Let's us transiently reference `Context` from `Cx`.
impl<'a> std::ops::Deref for Cx<'a> {
    type Target = Context;
    fn deref(&self) -> &Context {
        self.ctx
    }
}

/// Get unroll count from the `!llvm.loop` metadata attached to a terminator, if present.
///
/// The unroll metadata is stored in IR (see [`crate::air_dialect::ops::set_unroll`]).
/// The writer builds from the count extracted here.
fn get_unroll(
    op: Ptr<Operation>,
    ctx: &Context,
    table: &pliron_llvm::metadata::MdTableAttr,
) -> Result<Option<u32>, Unsupported> {
    use pliron_llvm::metadata::{MdNodeAttr, MdOperandAttr, get_attachments};
    let Some(id) = get_attachments(ctx, op).and_then(|a| a.get(MD_LLVM_LOOP)) else {
        return Ok(None);
    };
    let bad = || Unsupported::Opcode("unrecognised !llvm.loop metadata".to_string());
    let node = table.get(id).ok_or_else(bad)?;
    let MdNodeAttr::Tuple { distinct, operands } = node;
    if !*distinct {
        return Err(bad());
    }
    // `{ self, llvm.loop.unroll.enable, llvm.loop.unroll.count }`, in that order.
    let [
        MdOperandAttr::Node(self_id),
        MdOperandAttr::Node(_enable),
        MdOperandAttr::Node(count),
    ] = operands[..]
    else {
        return Err(bad());
    };
    if self_id != id {
        return Err(bad());
    }
    let MdNodeAttr::Tuple { operands, .. } = table.get(count).ok_or_else(bad)?;
    let [MdOperandAttr::String(key), MdOperandAttr::Constant(value)] = &operands[..] else {
        return Err(bad());
    };
    if key != "llvm.loop.unroll.count" {
        return Err(bad());
    }
    let n = (&**value as &dyn pliron::attribute::Attribute)
        .downcast_ref::<pliron::builtin::attributes::IntegerAttr>()
        .ok_or_else(bad)?
        .value()
        .to_u64();
    Ok(Some(n as u32))
}

/// Get LLVM integer predicate code from pliron-llvm's representation.
pub fn icmp_code(p: pliron_llvm::attributes::ICmpPredicateAttr) -> u32 {
    // TODO: `pred::` could be a repr(C) integer enum. Same with `fmcp_code` below.
    use pliron_llvm::attributes::ICmpPredicateAttr as P;
    match p {
        P::EQ => pred::ICMP_EQ,
        P::NE => pred::ICMP_NE,
        P::UGT => pred::ICMP_UGT,
        P::UGE => pred::ICMP_UGE,
        P::ULT => pred::ICMP_ULT,
        P::ULE => pred::ICMP_ULE,
        P::SGT => pred::ICMP_SGT,
        P::SGE => pred::ICMP_SGE,
        P::SLT => pred::ICMP_SLT,
        P::SLE => pred::ICMP_SLE,
    }
}

/// Get LLVM float predicate code from pliron-llvm's representation.
pub fn fcmp_code(p: pliron_llvm::attributes::FCmpPredicateAttr) -> u32 {
    use pliron_llvm::attributes::FCmpPredicateAttr as P;
    match p {
        P::False => pred::FCMP_FALSE,
        P::OEQ => pred::FCMP_OEQ,
        P::OGT => pred::FCMP_OGT,
        P::OGE => pred::FCMP_OGE,
        P::OLT => pred::FCMP_OLT,
        P::OLE => pred::FCMP_OLE,
        P::ONE => pred::FCMP_ONE,
        P::ORD => pred::FCMP_ORD,
        P::UNO => pred::FCMP_UNO,
        P::UEQ => pred::FCMP_UEQ,
        P::UGT => pred::FCMP_UGT,
        P::UGE => pred::FCMP_UGE,
        P::ULT => pred::FCMP_ULT,
        P::ULE => pred::FCMP_ULE,
        P::UNE => pred::FCMP_UNE,
        P::True => pred::FCMP_TRUE,
    }
}

/// The raw bits of a scalar constant attribute.
///
/// The bitcode writer stores all scalar constants as bits and lets the type specify how to read
/// the value properly.
fn attr_bits(a: &dyn pliron::attribute::Attribute) -> u64 {
    if let Some(i) = a.downcast_ref::<pliron::builtin::attributes::IntegerAttr>() {
        // `IntegerAttr` stores the value at its type's width. We have to sign extend correctly
        // since the bitcode writer takes `i64`s.
        let v = i.value();
        let (bits, bw) = (v.to_u64(), v.bw());
        return match bw {
            1..64 if (bits >> (bw - 1)) & 1 == 1 => bits | (!0u64 << bw),
            _ => bits,
        };
    }
    if let Some(f) =
        pliron::attribute::attr_cast::<dyn pliron::builtin::attr_interfaces::FloatAttr>(a)
    {
        return f.get_inner().to_bits() as u64;
    }
    // TODO: Error
    panic!("constant attribute is neither an integer nor a float");
}

/// The intrinsic a `llvm.call_intrinsic` names.
pub(crate) fn callee_of(call: &CallIntrinsicOp, ctx: &Context) -> String {
    // TODO: Error
    call.get_attr_llvm_intrinsic_name(ctx)
        .expect("call has no intrinsic name")
        .clone()
        .into()
}

/// The module-scope symbol an `air.global_ref` names.
pub(crate) fn global_sym(op: Ptr<Operation>, ctx: &Context) -> String {
    // TODO: Error
    Operation::get_op::<GlobalRefOp>(op, ctx)
        .expect("not a global reference")
        .get_attr_air_sym(ctx)
        .expect("op has no symbol")
        .clone()
        .into()
}

/// Blocks of the function, in order.
fn blocks_of(air: &AirFunc) -> &[Ptr<BasicBlock>] {
    &air.blocks
}

pub fn body_from_air(
    ctx: &Context,
    air: &AirFunc,
    num_params: usize,
    callee_index: &dyn Fn(&str) -> Option<u32>,
    global_index: &dyn Fn(&str) -> Option<u32>,
) -> Result<Body, Unsupported> {
    let entry = {
        use pliron::builtin::op_interfaces::OneRegionInterface;
        use pliron::graph::ControlFlowGraph;
        air.func.get_region(ctx).entry_node(ctx)
    };
    let cx = Cx::new(ctx, entry);
    let ctx = &cx;
    let blocks = blocks_of(air);
    let mut body = Body::new(num_params as u32);
    body.num_blocks = blocks.len() as u32;

    // Module metadata table
    let md_table = {
        use pliron::op::Op;
        pliron_llvm::metadata::find_metadata_table(ctx, air.func.get_operation())
    };

    // Map from pliron's value definition to bitcode value (location + id).
    let mut value_map: HashMap<Value, Val> = HashMap::new();
    // Block argument (block id, block argument id) to phi reference map. Phi's are wired when
    // terminator is emitted.
    type BlockArgumentId = (usize, usize);
    let mut phis: HashMap<BlockArgumentId, crate::bitcode::func::PhiRef> = HashMap::new();
    // Pending values. Stored as (successor, arg index, pending value, predecessor) in ascending
    // predecessor order. Used when wiring up phis/values.
    let mut pending: Vec<(usize, usize, Value, u32)> = Vec::new();

    // Process all blocks.
    for (bi, b) in blocks.iter().enumerate() {
        if bi > 0 {
            // Gather deferred phis..
            for (ai, arg) in b.deref(ctx).arguments().enumerate() {
                let ty = ctx.ty(arg.get_type(ctx));
                let (val, slot) = body.push_phi(ty);
                value_map.insert(arg, val);
                phis.insert((bi, ai), slot);
            }
        }
        for op in ops_of(*b, ctx) {
            // Translate pliron op to bitcode value.
            let produced = translate(
                ctx,
                op,
                &value_map,
                &mut body,
                num_params,
                blocks,
                &Symbols {
                    callee: callee_index,
                    global: global_index,
                },
            )?;
            {
                // Update value map
                let o = op.deref(ctx);
                if let (Some(val), 1) = (produced, o.get_num_results()) {
                    value_map.insert(o.get_result(0), val);
                }
            }
            // Apply unroll hint if available.
            if let Some(t) = md_table.as_ref()
                && let Some(c) = get_unroll(op, ctx, t)?
            {
                body.loop_hints.push((body.len() as u32 - 1, c));
            }
            // Gather pending values.
            let succs: Vec<_> = op.deref(ctx).successors().collect();
            for (si, s) in succs.iter().enumerate() {
                let Some(ti) = blocks.iter().position(|x| x == s) else {
                    continue;
                };
                for (ai, v) in successor_operands(op, ctx, si)?.into_iter().enumerate() {
                    pending.push((ti, ai, v, bi as u32));
                }
            }
        }
    }
    // Wire up deferred phis and pending values.
    for (ti, ai, v, pred) in pending {
        let slot = *phis.get(&(ti, ai)).ok_or_else(|| {
            Unsupported::Opcode(format!(
                "block {ti} has no argument {ai} for a predecessor to forward to"
            ))
        })?;
        let val = value_to_val(ctx, v, &value_map, &mut body, num_params, global_index)?;
        body.add_phi_incoming(slot, val, pred);
    }

    Ok(body)
}

fn ops_of(b: Ptr<BasicBlock>, ctx: &Context) -> Vec<Ptr<Operation>> {
    b.deref(ctx).iter(ctx).collect()
}

/// The values the terminator `term` forwards to it's `idx`th successor.
fn successor_operands(
    term: Ptr<Operation>,
    ctx: &Context,
    idx: usize,
) -> Result<Vec<Value>, Unsupported> {
    let obj = Operation::get_op_dyn(term, ctx);
    pliron::op::op_cast::<dyn BranchOpInterface>(obj.as_ref())
        .map(|br| br.successor_operands(ctx, idx))
        .ok_or_else(|| Unsupported::Opcode("terminator is not a branch".into()))
}

/// Handle const value bitcode encoding.
pub(crate) fn const_val(
    ctx: &Cx<'_>,
    op: Ptr<Operation>,
    v: Value,
    body: &mut Body,
    global_index: &dyn Fn(&str) -> Option<u32>,
) -> Result<Val, Unsupported> {
    let ty = ctx.ty(v.get_type(ctx));
    if let Some(c) = Operation::get_op::<ConstantOp>(op, ctx) {
        // TODO: Error
        let value = c
            .get_attr_llvm_constant_value(ctx)
            .expect("constant has no value");
        let value: &dyn Attribute = &**value;
        if let Some(agg) = value.downcast_ref::<AggregateAttr>() {
            let bits: Vec<u64> = agg.elements().iter().map(|e| attr_bits(&**e)).collect();
            return Ok(Val::LocalConst(
                body.consts
                    .intern(crate::bitcode::Const::DataArray(ty, bits)),
            ));
        }
        let bits = attr_bits(value);
        if is_float(ctx, v.get_type(ctx)) {
            return Ok(body.fconst(ty, bits));
        } else {
            return Ok(body.iconst(ty, bits as i64));
        }
    }
    if Operation::is_op::<ZeroOp>(op, ctx) {
        return Ok(Val::LocalConst(
            body.consts.intern(crate::bitcode::Const::Null(ty)),
        ));
    }
    if Operation::is_op::<UndefOp>(op, ctx) {
        return Ok(Val::LocalConst(
            body.consts.intern(crate::bitcode::Const::Undef(ty)),
        ));
    }
    // Globals are module values, not constants.
    let name = global_sym(op, ctx);
    global_index(&name)
        .map(Val::Global)
        .ok_or(Unsupported::UnknownCallee(name))
}

/// Convert pliron `Value` to bitcode `Val`. Definitions already mapped in `value_map`.
fn value_to_val(
    ctx: &Cx<'_>,
    v: Value,
    value_map: &HashMap<Value, Val>,
    body: &mut Body,
    num_params: usize,
    global_index: &dyn Fn(&str) -> Option<u32>,
) -> Result<Val, Unsupported> {
    let _ = (body, global_index);
    // Handle function parameters (entry block arguments).
    if let DefiningEntity::Block(b) = v.defining_entity() {
        let idx = v.find_index(ctx);
        if ctx.is_entry(b) && idx < num_params {
            return Ok(Val::Arg(idx as u32));
        }
    }
    // Otherwise get from map.
    value_map.get(&v).cloned().ok_or_else(|| {
        // Extract unresolved value info for error.
        let (op, loc) = match v.defining_entity() {
            DefiningEntity::Op(d) => (
                Operation::get_op_dyn(d, ctx.ctx).get_opid().to_string(),
                Some(d.deref(ctx.ctx).loc().disp(ctx.ctx).to_string()),
            ),
            DefiningEntity::Block(b) => (
                format!(
                    "block argument of {}",
                    b.deref(ctx.ctx).unique_name(ctx.ctx)
                ),
                None,
            ),
        };
        Unsupported::UnresolvedValue { op, loc }
    })
}

/// Module symbol tables for resolving instructions.
pub(crate) struct Symbols<'a> {
    pub(crate) callee: &'a dyn Fn(&str) -> Option<u32>,
    pub(crate) global: &'a dyn Fn(&str) -> Option<u32>,
}

/// How to write [`Op`](pliron::op::Op) as bitcode.
#[pliron::derive::op_interface]
pub trait ToBitcode {
    fn to_bitcode(&self, lx: &mut LowerCx<'_, '_>) -> Result<Option<Val>, Unsupported>;

    fn verify(_op: &dyn pliron::op::Op, _ctx: &Context) -> pliron::result::Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

/// The context needed for lowering an op.
pub struct LowerCx<'a, 'c> {
    pub(crate) cx: &'a Cx<'c>,
    pub(crate) value_map: &'a HashMap<Value, Val>,
    pub(crate) body: &'a mut Body,
    pub(crate) num_params: usize,
    pub(crate) blocks: &'a [Ptr<BasicBlock>],
    pub(crate) syms: &'a Symbols<'a>,
    /// The op being lowered, so an impl can reference its operands, results, etc.
    pub(crate) op: Ptr<Operation>,
}

impl LowerCx<'_, '_> {
    pub(crate) fn const_val(&mut self) -> Result<Val, Unsupported> {
        let res = self.op.deref(self.cx.ctx).get_result(0);
        const_val(self.cx, self.op, res, self.body, self.syms.global)
    }

    pub(crate) fn ctx(&self) -> &Context {
        self.cx.ctx
    }

    pub(crate) fn ty(&self, h: TypeHandle) -> crate::bitcode::Ty {
        self.cx.ty(h)
    }

    pub(crate) fn res_ty(&self) -> crate::bitcode::Ty {
        let h = self.op.deref(self.ctx()).get_result(0).get_type(self.ctx());
        self.ty(h)
    }

    pub(crate) fn opd(&mut self, i: usize) -> Result<Val, Unsupported> {
        let v = self.op.deref(self.cx.ctx).get_operand(i);
        value_to_val(
            self.cx,
            v,
            self.value_map,
            self.body,
            self.num_params,
            self.syms.global,
        )
    }

    pub(crate) fn block_idx(&self, b: Ptr<BasicBlock>) -> u32 {
        // TODO: Error
        self.blocks
            .iter()
            .position(|x| *x == b)
            .expect("unknown block") as u32
    }
}

/// Convert an op to bitcode.
fn translate(
    ctx: &Cx<'_>,
    op: Ptr<Operation>,
    value_map: &HashMap<Value, Val>,
    body: &mut Body,
    num_params: usize,
    blocks: &[Ptr<BasicBlock>],
    syms: &Symbols<'_>,
) -> Result<Option<Val>, Unsupported> {
    let obj = Operation::get_op_dyn(op, ctx.ctx);
    let Some(lowering) = pliron::op::op_cast::<dyn ToBitcode>(obj.as_ref()) else {
        return Err(Unsupported::UnresolvedValue {
            op: format!("no bitcode for `{}`", obj.get_opid()),
            loc: Some(op.deref(ctx.ctx).loc().disp(ctx.ctx).to_string()),
        });
    };
    let mut lx = LowerCx {
        cx: ctx,
        value_map,
        body,
        num_params,
        blocks,
        syms,
        op,
    };
    lowering.to_bitcode(&mut lx)
}

/// The `air` form of one function.
// TODO: Verify if this is actually valuable / needed.
pub struct AirFunc {
    pub func: FuncOp,
    pub blocks: Vec<Ptr<BasicBlock>>,
}
