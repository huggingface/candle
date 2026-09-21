//! The LLVM bitcode writer's kernel-level scaffolding: the function record, the
//! parameter and global declaration, and the container a ready instruction body plugs
//! into. The conversion route supplies the body by encoding the `air` dialect in
//! `crate::air_dialect::to_bitcode`.

use crate::bitcode::Ty;
use crate::bitcode::func::{binop, cast, pred};
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
        _ => return None,
    })
}

/// The BITCODE unary-op codes (`llvm::bitc::UnOpcodes`, where `UNOP_FNEG = 0`),
/// emitted as `INST_UNOP`.
pub(crate) fn unop_of(op: &str) -> Option<u32> {
    Some(match op {
        "fneg" => 0,
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
/// The module scaffolding around one function body: globals, callee prototypes, the kernel
/// declaration and the container. The scaffolding needs exactly two facts about the function
/// that the body does not carry - which callees to declare prototypes for, and the parameter
/// types - which the conversion reads off the converted `air` ops and signature.
pub fn bitcode_with_body_parts(
    kernel_name: &str,
    callees: &[(String, LlType, Vec<LlType>)],
    param_types: &[LlType],
    args: &[crate::bitcode::metadata::KernelArg],
    tg_globals: &[(String, LlType, u32)],
    profile: &TargetProfile,
    make_body: &dyn Fn(
        &dyn Fn(&str) -> Option<u32>,
        &dyn Fn(&str) -> Option<u32>,
    ) -> Result<crate::bitcode::func::Body, Unsupported>,
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
    let callees = callees.to_vec();
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
        params: param_types.iter().map(ty_of).collect(),
        vararg: false,
    };
    m.declare(kernel_name, kty.clone(), false, ka);
    m.types.intern(&Ty::Int(64));

    let mut body = make_body(&|n| callee_slot.get(n).copied(), &|n| {
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
