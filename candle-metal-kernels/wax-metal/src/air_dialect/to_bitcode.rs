//! `air` dialect -> [`bitcode::func::Body`], the mirror of [`crate::lower::to_bitcode`].
//!
//! Two things decide whether the output is byte-identical to the `cfg.rs` path:
//!
//! 1. VALUE NUMBERING. Block arguments are numbered first per block, then non-constant op
//!    results - matching `body_from_cfg`, which numbers phis before instructions.
//! 2. CONSTANTS ARE NOT INSTRUCTIONS. The bridge materialises immediates as ops, but
//!    bitcode wants constant-pool entries. They are interned when a consumer references
//!    them, exactly as `operand_to_val` does. Numbering them as instructions instead would
//!    shift every subsequent value number - a total, not a local, failure.
//!
//! Phi records are rebuilt from each predecessor's successor operands, in ascending
//! predecessor-block order. `tests/pliron_air_bridge.rs` pins that this reproduces the
//! lowerer's registration order.

use std::cell::RefCell;
use std::collections::HashMap;

use pliron::basic_block::BasicBlock;
use pliron::builtin::op_interfaces::BranchOpInterface;
use pliron::context::{Context, Ptr};
use pliron::linked_list::ContainsLinkedList;
use pliron::operation::Operation;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::{DefiningEntity, Value};

use super::attrs::{BinOpAttr, CastOpAttr, FCmpPredAttr, ICmpPredAttr};
use super::from_cfg::AirFunc;
use super::ops::*;
use super::types::{from_pliron, is_float};
use crate::bitcode::func::{Body, Inst, Val};
use crate::lower::to_bitcode::{Unsupported, binop_of, cast_of, fcmp_of, icmp_of, ty_of};

/// The context, plus the types already converted out of it.
///
/// MEASURED on the `tile` side first, where rebuilding types per access was 49% of the
/// whole read. Same defect here: a type was reconstructed at every instruction instead of
/// once per DISTINCT type. pliron uniques its types, so the handle is a stable key.
///
/// Keyed by handle, which is only sound because the cache cannot outlive the `Context` it
/// was built from - handles are arena indices and a different `Context` reuses them. That
/// is why this is a borrowed struct and not a thread-local.
struct Cx<'a> {
    ctx: &'a Context,
    types: RefCell<HashMap<TypeHandle, crate::bitcode::Ty>>,
}

impl<'a> Cx<'a> {
    fn new(ctx: &'a Context) -> Self {
        Cx {
            ctx,
            types: RefCell::new(HashMap::new()),
        }
    }

    /// The bitcode type for a pliron type handle, from cache where possible.
    fn ty(&self, h: TypeHandle) -> crate::bitcode::Ty {
        if let Some(t) = self.types.borrow().get(&h) {
            return t.clone();
        }
        let t = ty_of(&from_pliron(self.ctx, h).expect("air type has no LlType"));
        self.types.borrow_mut().insert(h, t.clone());
        t
    }
}

/// So every existing `ctx` use inside this file keeps working unchanged.
impl<'a> std::ops::Deref for Cx<'a> {
    type Target = Context;
    fn deref(&self) -> &Context {
        self.ctx
    }
}

fn is_float_ty(h: TypeHandle, ctx: &Context) -> bool {
    is_float(ctx, h)
}

/// Read a typed operator/predicate attribute off an op.
fn get_op_attr<T: pliron::attribute::Attribute + Clone>(op: Ptr<Operation>, ctx: &Context) -> T {
    op.deref(ctx)
        .attributes
        .get::<T>(&ATTR_KEY_AIR_OP.try_into().unwrap())
        .expect("op has no operator attribute")
        .clone()
}

fn get_align(op: Ptr<Operation>, ctx: &Context) -> u32 {
    op.deref(ctx)
        .attributes
        .get::<pliron::builtin::attributes::IntegerAttr>(&ATTR_KEY_AIR_ALIGN.try_into().unwrap())
        .map(|a| a.value().to_u64() as u32)
        .unwrap_or(0)
}

fn const_bits(op: Ptr<Operation>, ctx: &Context) -> u64 {
    op.deref(ctx)
        .attributes
        .get::<pliron::builtin::attributes::IntegerAttr>(&ATTR_KEY_AIR_VALUE.try_into().unwrap())
        .map(|a| a.value().to_u64())
        .expect("const op has no value")
}

fn elem_ty(op: Ptr<Operation>, ctx: &Cx<'_>) -> crate::bitcode::Ty {
    let h = op
        .deref(ctx)
        .attributes
        .get::<pliron::builtin::attributes::TypeAttr>(&ATTR_KEY_AIR_ELEM.try_into().unwrap())
        .expect("op has no element type")
        .clone();
    ctx.ty(h.get_type(ctx))
}

fn get_count_opt(op: Ptr<Operation>, ctx: &Context) -> Option<u32> {
    op.deref(ctx)
        .attributes
        .get::<pliron::builtin::attributes::IntegerAttr>(&ATTR_KEY_AIR_COUNT.try_into().unwrap())
        .map(|a| a.value().to_u64() as u32)
}

fn get_count(op: Ptr<Operation>, ctx: &Context) -> u32 {
    get_count_opt(op, ctx).expect("op has no count")
}

fn get_unroll(op: Ptr<Operation>, ctx: &Context) -> Option<u32> {
    op.deref(ctx)
        .attributes
        .get::<pliron::builtin::attributes::IntegerAttr>(&ATTR_KEY_AIR_UNROLL.try_into().unwrap())
        .map(|a| a.value().to_u64() as u32)
}

fn has_fast(op: Ptr<Operation>, ctx: &Context) -> bool {
    op.deref(ctx)
        .attributes
        .get::<pliron::builtin::attributes::IntegerAttr>(&ATTR_KEY_AIR_FAST.try_into().unwrap())
        .is_some()
}

fn global_sym(op: Ptr<Operation>, ctx: &Context) -> String {
    op.deref(ctx)
        .attributes
        .get::<pliron::builtin::attributes::StringAttr>(&ATTR_KEY_AIR_CALLEE.try_into().unwrap())
        .map(|a| String::from(a.clone()))
        .expect("op has no symbol")
}

/// The function's entry block is the first one.
fn is_entry(b: Ptr<BasicBlock>, ctx: &Context) -> bool {
    let _ = ctx;
    ENTRY.with(|e| e.get() == Some(b))
}

thread_local! {
    static ENTRY: std::cell::Cell<Option<Ptr<BasicBlock>>> = const { std::cell::Cell::new(None) };
}

/// True for the ops that are constant-pool entries rather than instructions.
fn is_const_op(op: Ptr<Operation>, ctx: &Context) -> bool {
    Operation::is_op::<ConstBitsOp>(op, ctx)
        || Operation::is_op::<ConstVecOp>(op, ctx)
        || Operation::is_op::<UndefOp>(op, ctx)
        || Operation::is_op::<ZeroOp>(op, ctx)
        || Operation::is_op::<GlobalRefOp>(op, ctx)
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
    // Shadowed deliberately: `Cx` derefs to `Context`, so every use below reads unchanged
    // while type conversions now go through the cache.
    let cx = Cx::new(ctx);
    let ctx = &cx;
    let blocks = blocks_of(air);
    ENTRY.with(|e| e.set(blocks.first().copied()));
    let mut body = Body::new(num_params as u32);
    body.num_blocks = blocks.len() as u32;

    // Pass 1: number block arguments then non-constant results, per block. A back edge can
    // reference a value defined further down, so this has to precede emission.
    let mut num_of: HashMap<u64, u32> = HashMap::new();
    let mut n = 0u32;
    for (bi, b) in blocks.iter().enumerate() {
        if bi > 0 {
            for arg in b.deref(ctx).arguments() {
                num_of.insert(key(&arg), n);
                n += 1;
            }
        }
        for op in ops_of(*b, ctx) {
            if is_const_op(op, ctx) {
                continue;
            }
            let o = op.deref(ctx);
            if o.get_num_results() == 1 {
                num_of.insert(key(&o.get_result(0)), n);
                n += 1;
            }
        }
    }

    // Pass 2: emit.
    for (bi, b) in blocks.iter().enumerate() {
        if bi > 0 {
            for (ai, arg) in b.deref(ctx).arguments().enumerate() {
                let ty = ctx.ty(arg.get_type(ctx));
                let mut incoming = Vec::new();
                // Ascending predecessor-block order; see the module note.
                for (pi, p) in blocks.iter().enumerate() {
                    let Some(term) = p.deref(ctx).get_terminator(ctx) else {
                        continue;
                    };
                    let succs: Vec<_> = term.deref(ctx).successors().collect();
                    for (si, s) in succs.iter().enumerate() {
                        if *s != *b {
                            continue;
                        }
                        let fwd = successor_operands(term, ctx, si)?;
                        let v = fwd.get(ai).copied().ok_or_else(|| {
                            Unsupported::Opcode(format!(
                                "block {bi} arg {ai}: predecessor {pi} forwards too few values"
                            ))
                        })?;
                        incoming.push((
                            value_to_val(ctx, v, &num_of, &mut body, num_params, global_index)?,
                            pi as u32,
                        ));
                    }
                }
                body.insts.push(Inst::Phi { ty, incoming });
            }
        }
        for op in ops_of(*b, ctx) {
            if is_const_op(op, ctx) {
                continue;
            }
            if let Some(inst) = translate(
                ctx,
                op,
                &num_of,
                &mut body,
                num_params,
                blocks,
                &Symbols {
                    callee: callee_index,
                    global: global_index,
                },
            )? {
                let is_term = op.deref(ctx).get_num_successors() > 0
                    || Operation::is_op::<RetOp>(op, ctx)
                    || Operation::is_op::<UnreachableOp>(op, ctx);
                body.insts.push(inst);
                if is_term && let Some(c) = get_unroll(op, ctx) {
                    body.loop_hints.push((body.insts.len() as u32 - 1, c));
                }
            }
        }
    }
    Ok(body)
}

/// A stable key for a value: pliron's `Value` is `Copy` and hashable, but the numbering map
/// is keyed on the uid so a value looked up through a different handle still matches.
fn key(v: &Value) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    v.hash(&mut h);
    h.finish()
}

fn ops_of(b: Ptr<BasicBlock>, ctx: &Context) -> Vec<Ptr<Operation>> {
    b.deref(ctx).iter(ctx).collect()
}

fn successor_operands(
    term: Ptr<Operation>,
    ctx: &Context,
    idx: usize,
) -> Result<Vec<Value>, Unsupported> {
    if let Some(br) = Operation::get_op::<BrOp>(term, ctx) {
        Ok(br.successor_operands(ctx, idx))
    } else if let Some(cbr) = Operation::get_op::<CondBrOp>(term, ctx) {
        Ok(cbr.successor_operands(ctx, idx))
    } else {
        Err(Unsupported::Opcode("terminator is not a branch".into()))
    }
}

/// Resolve a value to a bitcode operand, interning a constant op's result into the pool.
fn value_to_val(
    ctx: &Cx<'_>,
    v: Value,
    num_of: &HashMap<u64, u32>,
    body: &mut Body,
    num_params: usize,
    global_index: &dyn Fn(&str) -> Option<u32>,
) -> Result<Val, Unsupported> {
    if let DefiningEntity::Op(op) = v.defining_entity()
        && is_const_op(op, ctx)
    {
        let ty = ctx.ty(v.get_type(ctx));
        if Operation::is_op::<ConstBitsOp>(op, ctx) {
            let bits = const_bits(op, ctx);
            // An integer type takes the integer form, a float type the raw-bits form -
            // the same split `operand_to_val` makes on `ImmVal::Int` vs `FloatBits`.
            return Ok(if is_float_ty(v.get_type(ctx), ctx) {
                body.fconst(ty, bits)
            } else {
                body.iconst(ty, bits as i64)
            });
        }
        if Operation::is_op::<ConstVecOp>(op, ctx) {
            let bits = op
                .deref(ctx)
                .attributes
                .get::<super::attrs::DenseBitsAttr>(&ATTR_KEY_AIR_VALUE.try_into().unwrap())
                .expect("const vec has no value")
                .0
                .clone();
            return Ok(Val::LocalConst(
                body.consts
                    .intern(crate::bitcode::Const::DataArray(ty, bits)),
            ));
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
        // A global reference is a module value, not a constant.
        let name = global_sym(op, ctx);
        return global_index(&name)
            .map(Val::Global)
            .ok_or(Unsupported::UnknownCallee(name));
    }
    // An entry-block argument is a function parameter.
    if let DefiningEntity::Block(b) = v.defining_entity() {
        let idx = v.find_index(ctx);
        if is_entry(b, ctx) && idx < num_params {
            return Ok(Val::Arg(idx as u32));
        }
    }
    num_of
        .get(&key(&v))
        .map(|n| Val::Inst(*n))
        .ok_or_else(|| Unsupported::Opcode("unresolved value".into()))
}

/// The module symbol tables an instruction may need to resolve against.
struct Symbols<'a> {
    callee: &'a dyn Fn(&str) -> Option<u32>,
    global: &'a dyn Fn(&str) -> Option<u32>,
}

fn translate(
    ctx: &Cx<'_>,
    op: Ptr<Operation>,
    num_of: &HashMap<u64, u32>,
    body: &mut Body,
    num_params: usize,
    blocks: &[Ptr<BasicBlock>],
    syms: &Symbols<'_>,
) -> Result<Option<Inst>, Unsupported> {
    let (callee_index, global_index) = (syms.callee, syms.global);
    let opd = |i: usize, body: &mut Body| -> Result<Val, Unsupported> {
        let v = op.deref(ctx).get_operand(i);
        value_to_val(ctx, v, num_of, body, num_params, global_index)
    };
    let block_idx = |b: Ptr<BasicBlock>| -> u32 {
        blocks.iter().position(|x| *x == b).expect("unknown block") as u32
    };

    if Operation::is_op::<BinOp>(op, ctx) {
        let a = opd(0, body)?;
        let b = opd(1, body)?;
        let attr: BinOpAttr = get_op_attr(op, ctx);
        return Ok(Some(Inst::BinOp {
            op: binop_of(attr.as_str()).ok_or(Unsupported::Opcode(attr.as_str().into()))?,
            lhs: a,
            rhs: b,
        }));
    }
    if Operation::is_op::<CastOp>(op, ctx) {
        let v = opd(0, body)?;
        let attr: CastOpAttr = get_op_attr(op, ctx);
        return Ok(Some(Inst::Cast {
            op: cast_of(attr.as_str()).ok_or(Unsupported::Opcode(attr.as_str().into()))?,
            val: v,
            to: ctx.ty(op.deref(ctx).get_result(0).get_type(ctx)),
        }));
    }
    if Operation::is_op::<ICmpOp>(op, ctx) {
        let a = opd(0, body)?;
        let b = opd(1, body)?;
        let attr: ICmpPredAttr = get_op_attr(op, ctx);
        return Ok(Some(Inst::Cmp {
            pred: icmp_of(attr.as_str()).ok_or(Unsupported::Opcode(attr.as_str().into()))?,
            lhs: a,
            rhs: b,
        }));
    }
    if Operation::is_op::<FCmpOp>(op, ctx) {
        let a = opd(0, body)?;
        let b = opd(1, body)?;
        let attr: FCmpPredAttr = get_op_attr(op, ctx);
        return Ok(Some(Inst::Cmp {
            pred: fcmp_of(attr.as_str()).ok_or(Unsupported::Opcode(attr.as_str().into()))?,
            lhs: a,
            rhs: b,
        }));
    }
    if Operation::is_op::<SelectOp>(op, ctx) {
        let c = opd(0, body)?;
        let t = opd(1, body)?;
        let f = opd(2, body)?;
        return Ok(Some(Inst::Select { cond: c, t, f }));
    }
    if Operation::is_op::<LoadOp>(op, ctx) {
        let p = opd(0, body)?;
        return Ok(Some(Inst::Load {
            ptr: p,
            ty: ctx.ty(op.deref(ctx).get_result(0).get_type(ctx)),
            align: get_align(op, ctx),
        }));
    }
    if Operation::is_op::<StoreOp>(op, ctx) {
        let v = opd(0, body)?;
        let p = opd(1, body)?;
        return Ok(Some(Inst::Store {
            ptr: p,
            val: v,
            align: get_align(op, ctx),
        }));
    }
    if Operation::is_op::<ExtractElemOp>(op, ctx) {
        let v = opd(0, body)?;
        let i = opd(1, body)?;
        return Ok(Some(Inst::ExtractElt { vec: v, idx: i }));
    }
    if Operation::is_op::<InsertElemOp>(op, ctx) {
        let v = opd(0, body)?;
        let e = opd(1, body)?;
        let i = opd(2, body)?;
        return Ok(Some(Inst::InsertElt {
            vec: v,
            elem: e,
            idx: i,
        }));
    }
    if Operation::is_op::<GepOp>(op, ctx) {
        let p = opd(0, body)?;
        let i = opd(1, body)?;
        return Ok(Some(Inst::Gep {
            inbounds: false,
            elem: elem_ty(op, ctx),
            ptr: p,
            indices: vec![i],
        }));
    }
    if Operation::is_op::<GepArrayOp>(op, ctx) {
        let p = opd(0, body)?;
        // The leading zero steps through the array type itself, as the text form's `i32 0`.
        let zero = body.iconst(crate::bitcode::Ty::Int(32), 0);
        let i = opd(1, body)?;
        return Ok(Some(Inst::Gep {
            inbounds: true,
            elem: crate::bitcode::Ty::Array(get_count(op, ctx) as u64, Box::new(elem_ty(op, ctx))),
            ptr: p,
            indices: vec![zero, i],
        }));
    }
    if Operation::is_op::<AllocaOp>(op, ctx) {
        let n = body.iconst(crate::bitcode::Ty::Int(32), 1);
        let elem = elem_ty(op, ctx);
        let ty = match get_count_opt(op, ctx) {
            Some(c) => crate::bitcode::Ty::Array(c as u64, Box::new(elem)),
            None => elem,
        };
        return Ok(Some(Inst::Alloca {
            ty,
            count: n,
            align: get_align(op, ctx),
        }));
    }
    if Operation::is_op::<CallOp>(op, ctx) {
        let n_args = op.deref(ctx).get_num_operands();
        let mut args = Vec::with_capacity(n_args);
        let mut params = Vec::with_capacity(n_args);
        for i in 0..n_args {
            params.push(ctx.ty(op.deref(ctx).get_operand(i).get_type(ctx)));
            args.push(opd(i, body)?);
        }
        let returns_value = op.deref(ctx).get_num_results() == 1;
        let ret = if returns_value {
            ctx.ty(op.deref(ctx).get_result(0).get_type(ctx))
        } else {
            crate::bitcode::Ty::Void
        };
        let name = global_sym(op, ctx);
        let idx = callee_index(&name).ok_or(Unsupported::UnknownCallee(name))?;
        return Ok(Some(Inst::Call {
            fnty: crate::bitcode::Ty::Func {
                ret: Box::new(ret),
                params,
                vararg: false,
            },
            callee: Val::Func(idx),
            args,
            returns_value,
            fast: has_fast(op, ctx),
        }));
    }
    if Operation::is_op::<RetOp>(op, ctx) {
        return Ok(Some(Inst::Ret(None)));
    }
    if Operation::is_op::<UnreachableOp>(op, ctx) {
        return Ok(Some(Inst::Unreachable));
    }
    if Operation::is_op::<BrOp>(op, ctx) {
        let succ = op
            .deref(ctx)
            .successors()
            .next()
            .expect("br has a successor");
        return Ok(Some(Inst::Br(block_idx(succ))));
    }
    if Operation::is_op::<CondBrOp>(op, ctx) {
        let c = opd(0, body)?;
        let succs: Vec<_> = op.deref(ctx).successors().collect();
        return Ok(Some(Inst::CondBr {
            cond: c,
            then_bb: block_idx(succs[0]),
            else_bb: block_idx(succs[1]),
        }));
    }
    Err(Unsupported::Opcode(
        "no bitcode translation for this air op".into(),
    ))
}

/// A full `.metallib` for one kernel, with the body emitted from the `air` dialect.
///
/// Shares `bitcode_with_body`'s module scaffolding with the `cfg` path, so the two can
/// only differ in the function body - which is exactly what the parity test measures.
pub fn metallib_from_air(
    kernel_name: &str,
    f: &crate::lower::cfg::FuncBuilder,
    args: &[crate::bitcode::metadata::KernelArg],
    tg_globals: &[(String, crate::ir_builder::LlType, u32)],
    profile: &crate::profile::TargetProfile,
) -> Result<Vec<u8>, Unsupported> {
    let bc = crate::lower::to_bitcode::bitcode_with_body(
        kernel_name,
        f,
        args,
        tg_globals,
        profile,
        &|f, callee_index, global_index| {
            let ctx = &mut Context::new();
            let air = super::from_cfg::build(ctx, f, kernel_name)
                .map_err(|e| Unsupported::Opcode(e.to_string()))?;
            body_from_air(ctx, &air, f.num_params(), callee_index, global_index)
        },
    )?;
    Ok(crate::metallib::wrap(
        &bc,
        &[crate::metallib::Function::kernel(kernel_name)],
        profile,
    ))
}
