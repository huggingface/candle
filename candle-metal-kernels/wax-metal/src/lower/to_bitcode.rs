//! Translates a lowered [`FuncBuilder`] CFG into a [`bitcode::func::Body`].
//!
//! One CFG feeds both backends (text emitter prints it, this encodes it), so they cannot
//! drift in structure - only in rendering, which the per-kernel GPU comparison checks.
//!
//! Reconciles three mismatches: a `FUNCTION_BLOCK` has no block records, so blocks are
//! flattened as phis/instructions/terminator with `num_blocks` recorded separately; the
//! CFG numbers values by `ValueId` while bitcode numbers by position among value-producing
//! instructions, so the map is built in a first pass for back-edge phis; and only a
//! structured [`ImmVal`] can be encoded, so a leftover `ImmVal::Text` is reported, never
//! guessed. `InstKind::Raw` is likewise reported with its template.

use std::collections::HashMap;

use super::cfg::{BlockId, FuncBuilder, GepBase, ImmVal, InstKind, Operand, Terminator, ValueId};
use crate::bitcode::Ty;
use crate::bitcode::func::{Body, Inst, Val, binop, cast, pred};
use crate::ir_builder::LlType;
use crate::profile::TargetProfile;

/// Why a CFG could not be encoded. Carries enough detail to act on.
#[derive(Debug, Clone, PartialEq)]
pub enum Unsupported {
    /// An `InstKind::Raw` escape hatch, with its template.
    RawInstruction(String),
    /// An immediate that only exists as LLVM constant text.
    TextImmediate(String),
    /// A binary/cast/compare spelling with no mapping yet.
    Opcode(String),
    /// A call to a function the module has not declared.
    UnknownCallee(String),
    /// A type with no bitcode equivalent.
    Type(String),
}

impl std::fmt::Display for Unsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Unsupported::RawInstruction(t) => write!(f, "raw instruction `{t}`"),
            Unsupported::TextImmediate(t) => write!(f, "text-only immediate `{t}`"),
            Unsupported::Opcode(o) => write!(f, "unmapped opcode `{o}`"),
            Unsupported::UnknownCallee(c) => write!(f, "undeclared callee `{c}`"),
            Unsupported::Type(t) => write!(f, "unmapped type `{t}`"),
        }
    }
}

/// Map an `LlType` to the bitcode type model.
pub fn ty_of(t: &LlType) -> Ty {
    match t {
        LlType::I1 => Ty::Int(1),
        LlType::I8 => Ty::Int(8),
        LlType::I16 => Ty::Int(16),
        LlType::I32 => Ty::Int(32),
        LlType::I64 => Ty::Int(64),
        LlType::Half => Ty::Half,
        LlType::BFloat => Ty::BFloat,
        LlType::Float => Ty::Float,
        LlType::Double => Ty::Double,
        LlType::Void => Ty::Void,
        LlType::Vec(n, e) => Ty::vec(*n, ty_of(e)),
        LlType::Ptr { pointee, addrspace } => Ty::ptr(ty_of(pointee), *addrspace),
    }
}

/// Bitcode binary opcode for a CFG spelling. Float variants share the integer codes -
/// the reader picks the floating-point operation from the operand type.
/// Float binops reuse the integer opcodes, but not the ones the mnemonics suggest:
/// `fdiv` shares `sdiv`'s code and `frem` shares `srem`'s. The reader rejects a float
/// operand under `udiv`/`urem` outright ("Invalid record"), so getting this wrong makes
/// every kernel that divides floats unloadable.
pub(crate) fn binop_of(op: &str) -> Option<u32> {
    Some(match op {
        "add" | "fadd" => binop::ADD,
        "sub" | "fsub" => binop::SUB,
        "mul" | "fmul" => binop::MUL,
        "udiv" => binop::UDIV,
        "sdiv" | "fdiv" => binop::SDIV,
        "urem" => binop::UREM,
        "srem" | "frem" => binop::SREM,
        "shl" => binop::SHL,
        "lshr" => binop::LSHR,
        "ashr" => binop::ASHR,
        "and" => binop::AND,
        "or" => binop::OR,
        "xor" => binop::XOR,
        _ => return None,
    })
}

pub(crate) fn cast_of(op: &str) -> Option<u32> {
    Some(match op {
        "trunc" => cast::TRUNC,
        "zext" => cast::ZEXT,
        "sext" => cast::SEXT,
        "fptoui" => cast::FPTOUI,
        "fptosi" => cast::FPTOSI,
        "uitofp" => cast::UITOFP,
        "sitofp" => cast::SITOFP,
        "fptrunc" => cast::FPTRUNC,
        "fpext" => cast::FPEXT,
        "ptrtoint" => cast::PTRTOINT,
        "inttoptr" => cast::INTTOPTR,
        "bitcast" => cast::BITCAST,
        "addrspacecast" => cast::ADDRSPACECAST,
        _ => return None,
    })
}

pub(crate) fn icmp_of(p: &str) -> Option<u32> {
    Some(match p {
        "eq" => pred::ICMP_EQ,
        "ne" => pred::ICMP_NE,
        "ugt" => pred::ICMP_UGT,
        "uge" => pred::ICMP_UGE,
        "ult" => pred::ICMP_ULT,
        "ule" => pred::ICMP_ULE,
        "sgt" => pred::ICMP_SGT,
        "sge" => pred::ICMP_SGE,
        "slt" => pred::ICMP_SLT,
        "sle" => pred::ICMP_SLE,
        _ => return None,
    })
}

pub(crate) fn fcmp_of(p: &str) -> Option<u32> {
    Some(match p {
        "oeq" => pred::FCMP_OEQ,
        "ogt" => pred::FCMP_OGT,
        "oge" => pred::FCMP_OGE,
        "olt" => pred::FCMP_OLT,
        "ole" => pred::FCMP_OLE,
        "one" => pred::FCMP_ONE,
        "uno" => pred::FCMP_UNO,
        _ => return None,
    })
}

/// Encode `f`'s body. `callee_index` resolves a function name to its module
/// declaration order, which is how bitcode refers to call targets.
pub fn body_from_cfg(
    f: &FuncBuilder,
    callee_index: &dyn Fn(&str) -> Option<u32>,
    global_index: &dyn Fn(&str) -> Option<u32>,
) -> Result<Body, Unsupported> {
    let mut body = Body::new(f.num_params() as u32);
    body.num_blocks = f.num_blocks() as u32;

    // Pass 1: assign a result number to every value-producing instruction, in the same
    // flattened order pass 2 emits. Phis need this up front because a loop back edge
    // can reference a value defined further down.
    let mut result_of: HashMap<u32, u32> = HashMap::new();
    let mut n = 0u32;
    for b in 0..f.num_blocks() {
        let bid = BlockId(b as u32);
        for phi in f.block_phis(bid) {
            result_of.insert(phi.0, n);
            n += 1;
        }
        for (res, _) in f.block_insts(bid) {
            if let Some(v) = res {
                result_of.insert(v.0, n);
                n += 1;
            }
        }
    }

    // Pass 2: emit.
    for b in 0..f.num_blocks() {
        let bid = BlockId(b as u32);
        for phi in f.block_phis(bid) {
            let ty = ty_of(&f.value_type(phi));
            let mut incoming = Vec::new();
            for (op, pred_b) in f.phi_incomings(phi) {
                let v = operand_to_val(f, &op, &result_of, &mut body, global_index)?;
                incoming.push((v, pred_b.0));
            }
            body.insts.push(Inst::Phi { ty, incoming });
        }
        for (res, kind) in f.block_insts(bid) {
            let inst = translate(
                f,
                res,
                kind,
                &result_of,
                &mut body,
                callee_index,
                global_index,
            )?;
            body.insts.push(inst);
        }
        match f.block_term(bid) {
            Some(Terminator::Br(t)) => body.insts.push(Inst::Br(t.0)),
            Some(Terminator::CondBr { cond, then_, else_ }) => {
                let c = operand_to_val(f, cond, &result_of, &mut body, global_index)?;
                body.insts.push(Inst::CondBr {
                    cond: c,
                    then_bb: then_.0,
                    else_bb: else_.0,
                });
            }
            Some(Terminator::Ret) => body.insts.push(Inst::Ret(None)),
            Some(Terminator::Unreachable) => body.insts.push(Inst::Unreachable),
            // A block with no terminator cannot happen in verified output, but treat it
            // as unreachable rather than silently dropping the block boundary.
            None => body.insts.push(Inst::Unreachable),
        }
        // The unroll hint rides on the terminator just pushed.
        if let Some(count) = f.block_loop_unroll(bid) {
            body.loop_hints.push((body.insts.len() as u32 - 1, count));
        }
    }
    Ok(body)
}

/// Resolve a CFG operand to a bitcode value, interning immediates into the body's
/// constant pool as needed.
fn operand_to_val(
    f: &FuncBuilder,
    op: &Operand,
    result_of: &HashMap<u32, u32>,
    body: &mut Body,
    global_index: &dyn Fn(&str) -> Option<u32>,
) -> Result<Val, Unsupported> {
    match op {
        Operand::Val(v) => {
            if let Some(i) = f.param_index(*v) {
                return Ok(Val::Arg(i as u32));
            }
            result_of
                .get(&v.0)
                .map(|n| Val::Inst(*n))
                .ok_or_else(|| Unsupported::Opcode(format!("unresolved value %v{}", v.0)))
        }
        Operand::Imm { text, ty, val } => {
            let t = ty_of(ty);
            match val {
                ImmVal::Int(i) => Ok(body.iconst(t, *i)),
                ImmVal::FloatBits(bits) => Ok(body.fconst(t, *bits)),
                ImmVal::Float(v) => Ok(body.fconst(t, v.to_bits() as u64)),
                ImmVal::Zero => Ok(Val::LocalConst(
                    body.consts.intern(crate::bitcode::Const::Null(t)),
                )),
                ImmVal::Undef => Ok(Val::LocalConst(
                    body.consts.intern(crate::bitcode::Const::Undef(t)),
                )),
                // Dense vectors go through LLVM's ConstantDataSequential form, which
                // stores raw element bits.
                ImmVal::VecInt(vals) => Ok(Val::LocalConst(body.consts.intern(
                    crate::bitcode::Const::DataArray(t, vals.iter().map(|v| *v as u64).collect()),
                ))),
                ImmVal::VecFloatBits(bits) => Ok(Val::LocalConst(
                    body.consts
                        .intern(crate::bitcode::Const::DataArray(t, bits.clone())),
                )),
                // A symbol operand is a module value, not a constant.
                ImmVal::Global(name) => global_index(name)
                    .map(Val::Global)
                    .ok_or_else(|| Unsupported::UnknownCallee(name.clone())),
                // Text has no structured value to encode. Reported, not guessed: a
                // wrong constant is far worse than a clear refusal.
                ImmVal::Text => Err(Unsupported::TextImmediate(text.clone())),
            }
        }
    }
}

fn translate(
    f: &FuncBuilder,
    result: Option<ValueId>,
    kind: &InstKind,
    result_of: &HashMap<u32, u32>,
    body: &mut Body,
    callee_index: &dyn Fn(&str) -> Option<u32>,
    global_index: &dyn Fn(&str) -> Option<u32>,
) -> Result<Inst, Unsupported> {
    Ok(match kind {
        InstKind::Bin { op, lhs, rhs } => Inst::BinOp {
            op: binop_of(op).ok_or_else(|| Unsupported::Opcode(op.to_string()))?,
            lhs: operand_to_val(f, lhs, result_of, body, global_index)?,
            rhs: operand_to_val(f, rhs, result_of, body, global_index)?,
        },
        InstKind::Cast { op, src, to } => Inst::Cast {
            op: cast_of(op).ok_or_else(|| Unsupported::Opcode(op.to_string()))?,
            val: operand_to_val(f, src, result_of, body, global_index)?,
            to: ty_of(to),
        },
        InstKind::Icmp { pred, lhs, rhs } => Inst::Cmp {
            pred: icmp_of(pred).ok_or_else(|| Unsupported::Opcode(pred.to_string()))?,
            lhs: operand_to_val(f, lhs, result_of, body, global_index)?,
            rhs: operand_to_val(f, rhs, result_of, body, global_index)?,
        },
        InstKind::Fcmp { pred, lhs, rhs } => Inst::Cmp {
            pred: fcmp_of(pred).ok_or_else(|| Unsupported::Opcode(pred.to_string()))?,
            lhs: operand_to_val(f, lhs, result_of, body, global_index)?,
            rhs: operand_to_val(f, rhs, result_of, body, global_index)?,
        },
        InstKind::Select { cond, t, f: fv } => Inst::Select {
            cond: operand_to_val(f, cond, result_of, body, global_index)?,
            t: operand_to_val(f, t, result_of, body, global_index)?,
            f: operand_to_val(f, fv, result_of, body, global_index)?,
        },
        InstKind::Load { ptr, align } => {
            let pty = f.type_of(ptr);
            let elem = match &pty {
                LlType::Ptr { pointee, .. } => ty_of(pointee),
                other => return Err(Unsupported::Type(other.ll_str())),
            };
            Inst::Load {
                ptr: operand_to_val(f, ptr, result_of, body, global_index)?,
                ty: elem,
                align: *align,
            }
        }
        InstKind::Store { val, ptr, align } => Inst::Store {
            ptr: operand_to_val(f, ptr, result_of, body, global_index)?,
            val: operand_to_val(f, val, result_of, body, global_index)?,
            align: *align,
        },
        InstKind::Gep { elem, ptr, idx } => Inst::Gep {
            inbounds: false,
            elem: ty_of(elem),
            ptr: operand_to_val(f, ptr, result_of, body, global_index)?,
            indices: vec![operand_to_val(f, idx, result_of, body, global_index)?],
        },
        InstKind::ExtractElem { vec, idx } => Inst::ExtractElt {
            vec: operand_to_val(f, vec, result_of, body, global_index)?,
            idx: operand_to_val(f, idx, result_of, body, global_index)?,
        },
        InstKind::InsertElem { vec, elem, idx } => Inst::InsertElt {
            vec: operand_to_val(f, vec, result_of, body, global_index)?,
            elem: operand_to_val(f, elem, result_of, body, global_index)?,
            idx: operand_to_val(f, idx, result_of, body, global_index)?,
        },
        InstKind::GepArray {
            base,
            elem,
            count,
            idx,
            ..
        } => {
            let ptr = match base {
                GepBase::Global(g) => Val::Global(
                    global_index(g).ok_or_else(|| Unsupported::UnknownCallee(g.clone()))?,
                ),
                GepBase::Value(v) => operand_to_val(f, v, result_of, body, global_index)?,
            };
            // The leading zero index steps through the array type itself, exactly as
            // the text form's `i32 0` does.
            let zero = body.iconst(Ty::Int(32), 0);
            let i = operand_to_val(f, idx, result_of, body, global_index)?;
            Inst::Gep {
                inbounds: true,
                elem: Ty::Array(*count as u64, Box::new(ty_of(elem))),
                ptr,
                indices: vec![zero, i],
            }
        }
        InstKind::Alloca { elem, count, align } => {
            let n = body.iconst(Ty::Int(32), 1);
            let ty = match count {
                Some(n) => Ty::Array(*n as u64, Box::new(ty_of(elem))),
                None => ty_of(elem),
            };
            Inst::Alloca {
                ty,
                count: n,
                align: *align,
            }
        }
        InstKind::Call { callee, args, fast } => {
            let idx =
                callee_index(callee).ok_or_else(|| Unsupported::UnknownCallee(callee.clone()))?;
            let mut vals = Vec::with_capacity(args.len());
            for a in args {
                vals.push(operand_to_val(f, a, result_of, body, global_index)?);
            }
            // The callee's signature is recoverable from the CFG itself: the result's
            // type is the return type (void when there is no result), and each
            // argument operand knows its own type. That avoids threading declared
            // signatures through, and it cannot disagree with the text emitter, which
            // prints those same types.
            let ret = result.map_or(Ty::Void, |r| ty_of(&f.value_type(r)));
            let params: Vec<Ty> = args.iter().map(|a| ty_of(&f.type_of(a))).collect();
            Inst::Call {
                fnty: Ty::Func {
                    ret: Box::new(ret),
                    params,
                    vararg: false,
                },
                callee: Val::Func(idx),
                args: vals,
                returns_value: result.is_some(),
                fast: *fast,
            }
        }
        InstKind::Raw { template, .. } => {
            return Err(Unsupported::RawInstruction(template.clone()));
        }
    })
}

/// Assemble a complete `.metallib` for `kernel_name` from a finished CFG.
///
/// Declares one prototype per call target (signatures derived from the call sites
/// themselves, so they cannot disagree), then the entry point, then encodes the body
/// and the `!air.kernel` metadata describing `args`.
///
/// Returns `Err` with the specific reason when the CFG uses something the bitcode path
/// does not cover yet - never a partially-formed module.
pub(crate) fn metallib_from_cfg(
    kernel_name: &str,
    f: &FuncBuilder,
    args: &[crate::bitcode::metadata::KernelArg],
    tg_globals: &[(String, LlType, u32)],
    profile: &TargetProfile,
) -> Result<Vec<u8>, Unsupported> {
    let bc = bitcode_from_cfg(kernel_name, f, args, tg_globals, profile)?;
    Ok(crate::metallib::wrap(
        &bc,
        &[crate::metallib::Function::kernel(kernel_name)],
        profile,
    ))
}

/// The LLVM bitcode for one kernel, before the container wraps it. Split out so the
/// bitcode can be dumped and read with `llvm-dis` when diagnosing an encoding bug.
pub(crate) fn bitcode_from_cfg(
    kernel_name: &str,
    f: &FuncBuilder,
    args: &[crate::bitcode::metadata::KernelArg],
    tg_globals: &[(String, LlType, u32)],
    profile: &TargetProfile,
) -> Result<Vec<u8>, Unsupported> {
    bitcode_with_body(kernel_name, f, args, tg_globals, profile, &|f, c, g| {
        body_from_cfg(f, c, g)
    })
}

/// Builds one function body: `(func, callee_index, global_index)`. The only part of a
/// module that differs between the `cfg` and `air` emitters.
pub type MakeBody<'a> = dyn Fn(
        &FuncBuilder,
        &dyn Fn(&str) -> Option<u32>,
        &dyn Fn(&str) -> Option<u32>,
    ) -> Result<crate::bitcode::func::Body, Unsupported>
    + 'a;

/// The module scaffolding both emitters share - globals, callee prototypes, the kernel
/// declaration, loop metadata and the container. `make_body` supplies the function body,
/// which is the ONLY part that differs between the `cfg` and `air` paths. Factored so a
/// second emitter cannot drift on the surrounding module structure.
pub fn bitcode_with_body(
    kernel_name: &str,
    f: &FuncBuilder,
    args: &[crate::bitcode::metadata::KernelArg],
    tg_globals: &[(String, LlType, u32)],
    profile: &TargetProfile,
    make_body: &MakeBody,
) -> Result<Vec<u8>, Unsupported> {
    use crate::bitcode::Module;
    use crate::bitcode::metadata::KernelMetadata;

    let mut m = Module::new(profile.triple(), profile.datalayout());
    let ia = m.intrinsic_attrs();
    let ka = m.kernel_attrs();

    // Globals come first in the module value table, so they must be declared before
    // anything that could shift their ids.
    let mut global_slot = std::collections::HashMap::new();
    for (name, elem, count) in tg_globals {
        // CFG symbols carry the text form's leading `@`; the symbol table stores the
        // bare name.
        let slot = m.threadgroup_global(name.trim_start_matches('@'), ty_of(elem), *count as u64);
        global_slot.insert(name.clone(), slot);
    }

    // Prototypes first: their declaration order is what call records refer to.
    let callees = f.callees();
    let mut callee_slot = std::collections::HashMap::new();
    for (i, (name, ret, params)) in callees.iter().enumerate() {
        let fty = Ty::Func {
            ret: Box::new(ty_of(ret)),
            params: params.iter().map(ty_of).collect(),
            vararg: false,
        };
        m.declare(name.trim_start_matches('@'), fty, true, ia);
        callee_slot.insert(name.clone(), i as u32);
    }

    let entry_slot = callees.len();
    let kty = Ty::Func {
        ret: Box::new(Ty::Void),
        params: f.param_types().iter().map(ty_of).collect(),
        vararg: false,
    };
    m.declare(kernel_name, kty.clone(), false, ka);
    m.types.intern(&Ty::Int(64));

    let mut body = make_body(f, &|n| callee_slot.get(n).copied(), &|n| {
        global_slot.get(n).copied()
    })?;

    // `!llvm.loop` unroll hints. The loop id is self-referential, so it is reserved
    // before its operand list exists. The count is an i32 module constant, which is
    // safe to intern here: `const_base` only counts globals and functions, both of
    // which are already declared.
    let const_base = m.const_base();
    if !body.loop_hints.is_empty() {
        let kind = m.md_kinds.intern("llvm.loop");
        let enable = m.md.string("llvm.loop.unroll.enable");
        let enable_node = m.md.tuple(&[enable]);
        let count_key = m.md.string("llvm.loop.unroll.count");
        let hints = std::mem::take(&mut body.loop_hints);
        for (idx, count) in hints {
            m.types.intern(&Ty::Int(32));
            let cid = m
                .consts
                .intern(crate::bitcode::Const::Int(Ty::Int(32), count as i64));
            let cval = m.md.value(Ty::Int(32), const_base + cid);
            let count_node = m.md.tuple(&[count_key, cval]);
            let loop_id = m.md.distinct_node();
            m.md.set_node(loop_id, &[loop_id, enable_node, count_node]);
            body.attach(idx, kind, loop_id);
        }
    }
    m.set_body(entry_slot, body);

    let km = KernelMetadata {
        entry_value_id: m.func_value_id(entry_slot as u32),
        entry_fn_ptr_ty: Ty::ptr(kty, 0),
        args,
        air_version: profile.air_version,
        language_version: profile.language_version,
        limits: profile.limits,
        producer: "wax-metal",
    };
    let mut md = std::mem::take(&mut m.md);
    let mut consts = std::mem::take(&mut m.consts);
    let mut types = std::mem::take(&mut m.types);
    km.build(&mut md, &mut consts, &mut types, const_base);
    types.intern(&km.entry_fn_ptr_ty);
    m.md = md;
    m.consts = consts;
    m.types = types;

    Ok(m.to_bitcode())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_builder::LlType;
    use crate::lower::cfg::FuncBuilder;

    /// Every instruction shape that was converted from `InstKind::Raw` must translate.
    /// These are the ones that took the kernel coverage from 0/38 to 38/38, so a
    /// regression here silently drops kernels back to the text-only path.
    #[test]
    fn converted_shapes_all_translate() {
        let mut f = FuncBuilder::new("k");
        let p = f.param(
            "out",
            LlType::Ptr {
                pointee: Box::new(LlType::Float),
                addrspace: 1,
            },
        );
        let tid = f.param("tid", LlType::I32);
        f.append_block("entry");

        // integer division / remainder (were raw)
        let q = f.sdiv(tid, Operand::i32(4));
        let r = f.urem(tid, Operand::i32(32));
        let _ = f.udiv(tid, Operand::i32(32));
        let _ = f.srem(tid, Operand::i32(3));
        // float arithmetic (fsub/fdiv were raw)
        let fv = f.cast("sitofp", q, LlType::Float);
        let d = f.fsub(fv, FuncBuilder::float_one(&LlType::Float));
        let _ = f.fdiv(FuncBuilder::float_one(&LlType::Float), d);
        // casts (were raw)
        let back = f.cast("fptosi", d, LlType::I32);
        // intrinsic calls, plain and fast-math (were raw)
        let _ = f
            .call(LlType::Float, "@air.fast_exp.f32", vec![d.into()])
            .unwrap();
        let _ = f
            .call_fast(
                LlType::Float,
                "@air.simd_sum.f32".to_string(),
                vec![d.into()],
            )
            .unwrap();
        f.call(
            LlType::Void,
            "@air.wg.barrier",
            vec![Operand::i32(2), Operand::i32(1)],
        );
        // array GEPs: into a threadgroup global, and through an alloca (both were raw)
        let slot = f
            .gep_global("@_tg_user_scratch", LlType::Float, 1056, 3, r)
            .unwrap();
        let _ = f.load(slot, 4);
        let stack = f.alloca_array(LlType::Float, 8, 4).unwrap();
        let cell = f.gep_array_value(stack, LlType::Float, 8, 0, back).unwrap();
        f.store(d, cell, 4);
        let addr = f.gep(LlType::Float, p, back);
        f.store(d, addr, 4);
        f.set_terminator(crate::lower::cfg::Terminator::Ret);

        match body_from_cfg(&f, &|_| Some(0), &|_| Some(0)) {
            Ok(body) => assert!(
                body.insts.len() >= 15,
                "expected every instruction to translate, got {}",
                body.insts.len()
            ),
            Err(e) => panic!("translation regressed: {e}"),
        }
    }

    /// The escape hatch must still be reported rather than silently encoded.
    #[test]
    fn raw_instructions_are_reported() {
        let mut f = FuncBuilder::new("k");
        f.append_block("entry");
        f.raw(None, "some.exotic.thing", vec![]);
        f.set_terminator(crate::lower::cfg::Terminator::Ret);
        match body_from_cfg(&f, &|_| Some(0), &|_| Some(0)) {
            Err(Unsupported::RawInstruction(t)) => assert_eq!(t, "some.exotic.thing"),
            Err(other) => panic!("wrong error: {other}"),
            Ok(_) => panic!("a raw instruction must not silently translate"),
        }
    }
}
