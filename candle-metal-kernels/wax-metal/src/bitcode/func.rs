//! Function block. Instruction bodies.
//!
//! Relative value numbering: Operands are stored as `current_id - operand_id`, so backward
//! references encode as small positive numbers under VBR, and the operand's type is
//! already known to the reader. Most instructions do not carry type operands.
//!
//! The value table (in order): module values (globals, functions, module constants),
//! the function's arguments, function-local constants, then one entry per value-producing
//! instruction in program order. Store, br and ret void do not advance.
//!
//! Encodings were read out of `metal-as` output with `llvm-bcanalyzer`.

use super::bitstream::BitWriter;
use super::metadata::{MdId, MetadataBuilder};
use super::{Const, ConstPool, Ty, TypeTable};

const BLOCK_FUNCTION: u32 = 12;
const BLOCK_METADATA_ATTACHMENT: u32 = 16;
/// `METADATA_ATTACHMENT` inside a METADATA_ATTACHMENT_BLOCK. Determined empirically by
/// emitting one record per candidate code and seeing which one `llvm-bcanalyzer` named
/// ATTACHMENT. It is not 2, which is what the metadata block uses for VALUE.
const METADATA_ATTACHMENT_CODE: u32 = 11;

const FUNC_CODE_DECLAREBLOCKS: u32 = 1;
const FUNC_CODE_INST_BINOP: u32 = 2;
const FUNC_CODE_INST_CAST: u32 = 3;
const FUNC_CODE_INST_SELECT: u32 = 5;
const FUNC_CODE_INST_EXTRACTELT: u32 = 6;
const FUNC_CODE_INST_INSERTELT: u32 = 7;
const FUNC_CODE_INST_RET: u32 = 10;
const FUNC_CODE_INST_BR: u32 = 11;
const FUNC_CODE_INST_UNREACHABLE: u32 = 15;
const FUNC_CODE_INST_PHI: u32 = 16;
const FUNC_CODE_INST_ALLOCA: u32 = 19;
const FUNC_CODE_INST_LOAD: u32 = 20;
const FUNC_CODE_INST_CMP2: u32 = 28;
const FUNC_CODE_INST_CALL: u32 = 34;
const FUNC_CODE_INST_GEP: u32 = 43;
const FUNC_CODE_INST_STORE: u32 = 44;
const FUNC_CODE_INST_UNOP: u32 = 56;

/// Binary opcodes as the bitcode encodes them (`llvm::bitc::BinaryOpcodes`), not to be confused
/// with `Instruction::BinaryOps`. Float operations reuse the same codes. A reader picks the
/// floating-point variant when the operands are floats, so `ADD` on floats is `fadd` and `UDIV`
/// on floats is `fdiv`, etc.
pub mod binop {
    pub const ADD: u32 = 0;
    pub const SUB: u32 = 1;
    pub const MUL: u32 = 2;
    pub const UDIV: u32 = 3;
    pub const SDIV: u32 = 4;
    pub const UREM: u32 = 5;
    pub const SREM: u32 = 6;
    pub const SHL: u32 = 7;
    pub const LSHR: u32 = 8;
    pub const ASHR: u32 = 9;
    pub const AND: u32 = 10;
    pub const OR: u32 = 11;
    pub const XOR: u32 = 12;
}

/// `llvm::Instruction::CastOps`.
pub mod cast {
    pub const TRUNC: u32 = 0;
    pub const ZEXT: u32 = 1;
    pub const SEXT: u32 = 2;
    pub const FPTOUI: u32 = 3;
    pub const FPTOSI: u32 = 4;
    pub const UITOFP: u32 = 5;
    pub const SITOFP: u32 = 6;
    pub const FPTRUNC: u32 = 7;
    pub const FPEXT: u32 = 8;
    pub const PTRTOINT: u32 = 9;
    pub const INTTOPTR: u32 = 10;
    pub const BITCAST: u32 = 11;
    pub const ADDRSPACECAST: u32 = 12;
}

/// Predicates for `INST_CMP2` (`llvm::CmpInst::Predicate`).
pub mod pred {
    pub const FCMP_OEQ: u32 = 1;
    pub const FCMP_OGT: u32 = 2;
    pub const FCMP_OGE: u32 = 3;
    pub const FCMP_OLT: u32 = 4;
    pub const FCMP_OLE: u32 = 5;
    pub const FCMP_ONE: u32 = 6;
    pub const FCMP_UNO: u32 = 14;
    pub const ICMP_EQ: u32 = 32;
    pub const ICMP_NE: u32 = 33;
    pub const ICMP_UGT: u32 = 34;
    pub const ICMP_UGE: u32 = 35;
    pub const ICMP_ULT: u32 = 36;
    pub const ICMP_ULE: u32 = 37;
    pub const ICMP_SGT: u32 = 38;
    pub const ICMP_SGE: u32 = 39;
    pub const ICMP_SLT: u32 = 40;
    pub const ICMP_SLE: u32 = 41;
}

/// Where a value comes from. Resolved to an absolute id when the body is written.
#[derive(Clone, Debug)]
pub enum Val {
    /// A module-scope variable, by declaration order. Globals come first in the
    /// module value table, so their ids are their indices.
    Global(u32),
    /// A module-level function, by declaration order.
    Func(u32),
    /// A module-level constant, by index in the module constant pool.
    ModConst(u32),
    /// A function-local constant, by index in the body's own pool.
    LocalConst(u32),
    /// A function argument, by position.
    Arg(u32),
    /// The result of a previous value-producing instruction, by result number.
    Inst(u32),
}

/// An instruction. Result numbering is implicit.
#[derive(Clone, Debug)]
pub enum Inst {
    BinOp {
        op: u32,
        lhs: Val,
        rhs: Val,
    },
    Cast {
        op: u32,
        val: Val,
        to: Ty,
    },
    UnOp {
        op: u32,
        val: Val,
    },
    Cmp {
        pred: u32,
        lhs: Val,
        rhs: Val,
    },
    Select {
        cond: Val,
        t: Val,
        f: Val,
    },
    ExtractElt {
        vec: Val,
        idx: Val,
    },
    InsertElt {
        vec: Val,
        elem: Val,
        idx: Val,
    },
    Gep {
        inbounds: bool,
        elem: Ty,
        ptr: Val,
        indices: Vec<Val>,
    },
    Load {
        ptr: Val,
        ty: Ty,
        align: u32,
    },
    Store {
        ptr: Val,
        val: Val,
        align: u32,
    },
    /// A direct call. `fnty` must be the callee's function type. `fast` sets the
    /// full fast-math bundle, matching `call fast` in the text form.
    Call {
        fnty: Ty,
        callee: Val,
        args: Vec<Val>,
        returns_value: bool,
        fast: bool,
    },
    /// `alloca [n x elem], align a`.
    Alloca {
        ty: Ty,
        count: Val,
        align: u32,
    },
    Phi {
        ty: Ty,
        incoming: Vec<(Val, u32)>,
    },
    Br(u32),
    CondBr {
        cond: Val,
        then_bb: u32,
        else_bb: u32,
    },
    Ret(Option<Val>),
    Unreachable,
}

impl Inst {
    /// Whether this instruction contributes an entry to the value table.
    fn produces_value(&self) -> bool {
        match self {
            Inst::Store { .. }
            | Inst::Br(_)
            | Inst::CondBr { .. }
            | Inst::Ret(_)
            | Inst::Unreachable => false,
            Inst::Call { returns_value, .. } => *returns_value,
            _ => true,
        }
    }
}

/// A function definition's body. Instructions are held in one flat list. `num_blocks` records
/// how many basic blocks they form. Branch targets are block indices.
#[derive(Default)]
pub struct Body {
    pub insts: Vec<Inst>,
    pub num_blocks: u32,
    pub consts: ConstPool,
    pub num_args: u32,
    /// (instruction index, metadata kind id, node). Used for `!llvm.loop` on a latch
    /// branch, which is how the unroll count reaches the backend.
    attachments: Vec<(u32, u32, MdId)>,
    /// (instruction index, unroll count) for terminators carrying an unroll hint. Recorded
    /// while translating, since only then are instruction indices known. Turned into
    /// `!llvm.loop` attachments once the module's metadata exists.
    pub loop_hints: Vec<(u32, u32)>,
}

impl Body {
    pub fn new(num_args: u32) -> Self {
        Self {
            insts: Vec::new(),
            num_blocks: 1,
            consts: ConstPool::default(),
            num_args,
            attachments: Vec::new(),
            loop_hints: Vec::new(),
        }
    }

    /// Every type this body's records name. The writer looks each one up in the
    /// module type table and panics if it is absent, so the module interns these before
    /// writing rather than relying on the signature happening to mention them all.
    pub fn referenced_types(&self) -> Vec<Ty> {
        let mut out = vec![Ty::Int(32)]; // alloca's size operand type
        for i in &self.insts {
            match i {
                Inst::Cast { to, .. } => out.push(to.clone()),
                Inst::Gep { elem, .. } => out.push(elem.clone()),
                Inst::Load { ty, .. } => out.push(ty.clone()),
                Inst::Call { fnty, .. } => out.push(fnty.clone()),
                Inst::Alloca { ty, .. } => out.push(ty.clone()),
                Inst::Phi { ty, .. } => out.push(ty.clone()),
                _ => {}
            }
        }
        out.extend(self.consts.types());
        out
    }

    /// Attach metadata to the instruction at `idx` in program order.
    pub fn attach(&mut self, idx: u32, kind: u32, node: MdId) {
        self.attachments.push((idx, kind, node));
    }

    /// Attach metadata to the instruction most recently pushed.
    ///
    /// The index counts every instruction in program order, not just the value-producing
    /// ones. A terminator is a valid target, which is exactly what `!llvm.loop` needs.
    pub fn attach_to_last(&mut self, kind: u32, node: MdId) {
        let idx = self.insts.len().saturating_sub(1) as u32;
        self.attachments.push((idx, kind, node));
    }

    pub fn push(&mut self, i: Inst) -> Val {
        let produces = i.produces_value();
        let n = self.insts.iter().filter(|x| x.produces_value()).count() as u32;
        self.insts.push(i);
        // Non-producing instructions have no result. Callers should ignore this.
        Val::Inst(if produces { n } else { u32::MAX })
    }

    /// A function-local integer constant.
    pub fn iconst(&mut self, ty: Ty, v: i64) -> Val {
        Val::LocalConst(self.consts.intern(Const::Int(ty, v)))
    }

    /// A function-local float constant, given by its raw bits at `ty`'s width.
    pub fn fconst(&mut self, ty: Ty, bits: u64) -> Val {
        Val::LocalConst(self.consts.intern(Const::Float(ty, bits)))
    }

    /// Absolute value id for `v`.
    ///
    /// `num_globals` is the count of module globals, `const_base` the id where module
    /// constants start, and `func_base` the id of the function's first argument.
    fn abs(&self, v: &Val, num_globals: u32, const_base: u32, func_base: u32) -> u64 {
        match v {
            Val::Global(i) => *i as u64,
            // Functions sit after the globals.
            Val::Func(i) => (num_globals + *i) as u64,
            Val::ModConst(i) => (const_base + i) as u64,
            Val::Arg(i) => (func_base + i) as u64,
            Val::LocalConst(i) => (func_base + self.num_args + i) as u64,
            Val::Inst(i) => (func_base + self.num_args + self.consts.len() as u32 + i) as u64,
        }
    }

    pub(super) fn write(
        &self,
        w: &mut BitWriter,
        types: &TypeTable,
        md: &MetadataBuilder,
        num_globals: u32,
        const_base: u32,
        func_base: u32,
    ) {
        w.enter_block(BLOCK_FUNCTION, 4);
        w.record(FUNC_CODE_DECLAREBLOCKS, &[self.num_blocks as u64]);
        self.consts.write(w, types);

        // The id the next value-producing instruction will take.
        let mut cur = func_base + self.num_args + self.consts.len() as u32;
        let rel = |v: &Val, cur: u32| -> u64 {
            let a = self.abs(v, num_globals, const_base, func_base);
            (cur as i64 - a as i64) as u64
        };
        let ty_of = |t: &Ty| types.get(t).expect("instruction type interned") as u64;

        for inst in &self.insts {
            match inst {
                Inst::BinOp { op, lhs, rhs } => {
                    w.record(
                        FUNC_CODE_INST_BINOP,
                        &[rel(lhs, cur), rel(rhs, cur), *op as u64],
                    );
                }
                Inst::Cast { op, val, to } => {
                    w.record(FUNC_CODE_INST_CAST, &[rel(val, cur), ty_of(to), *op as u64]);
                }
                Inst::UnOp { op, val } => {
                    // `INST_UNOP` decodes [opval, (ty), opcode]. The value comes first, the
                    // code last, the opposite of `INST_BINOP`'s trailing op order.
                    w.record(FUNC_CODE_INST_UNOP, &[rel(val, cur), *op as u64]);
                }
                Inst::Cmp { pred, lhs, rhs } => {
                    w.record(
                        FUNC_CODE_INST_CMP2,
                        &[rel(lhs, cur), rel(rhs, cur), *pred as u64],
                    );
                }
                Inst::Select { cond, t, f } => {
                    // Operand order is (true, false, cond).
                    w.record(
                        FUNC_CODE_INST_SELECT,
                        &[rel(t, cur), rel(f, cur), rel(cond, cur)],
                    );
                }
                Inst::ExtractElt { vec, idx } => {
                    w.record(FUNC_CODE_INST_EXTRACTELT, &[rel(vec, cur), rel(idx, cur)]);
                }
                Inst::InsertElt { vec, elem, idx } => {
                    w.record(
                        FUNC_CODE_INST_INSERTELT,
                        &[rel(vec, cur), rel(elem, cur), rel(idx, cur)],
                    );
                }
                Inst::Gep {
                    inbounds,
                    elem,
                    ptr,
                    indices,
                } => {
                    let mut ops = vec![*inbounds as u64, ty_of(elem), rel(ptr, cur)];
                    for i in indices {
                        ops.push(rel(i, cur));
                    }
                    w.record(FUNC_CODE_INST_GEP, &ops);
                }
                Inst::Load { ptr, ty, align } => {
                    // Alignment is stored as log2(align) + 1.
                    w.record(
                        FUNC_CODE_INST_LOAD,
                        &[
                            rel(ptr, cur),
                            ty_of(ty),
                            align.trailing_zeros() as u64 + 1,
                            0,
                        ],
                    );
                }
                Inst::Store { ptr, val, align } => {
                    w.record(
                        FUNC_CODE_INST_STORE,
                        &[
                            rel(ptr, cur),
                            rel(val, cur),
                            align.trailing_zeros() as u64 + 1,
                            0,
                        ],
                    );
                }
                Inst::Call {
                    fnty,
                    callee,
                    args,
                    fast,
                    ..
                } => {
                    // 1<<15 marks an explicit function type. 1<<17 says a fast-math operand
                    // follows. 254 is the flag bundle `fast` expands to.
                    let flags = (1u64 << 15) | if *fast { 1 << 17 } else { 0 };
                    let mut ops = vec![0, flags];
                    if *fast {
                        ops.push(254);
                    }
                    ops.push(ty_of(fnty));
                    ops.push(rel(callee, cur));
                    for a in args {
                        ops.push(rel(a, cur));
                    }
                    w.record(FUNC_CODE_INST_CALL, &ops);
                }
                Inst::Alloca { ty, count, align } => {
                    // The size operand is an absolute value id here, unlike every other
                    // instruction's relative operands. Bit 6 of the align field flags that
                    // the allocated type is given explicitly.
                    w.record(
                        FUNC_CODE_INST_ALLOCA,
                        &[
                            ty_of(ty),
                            ty_of(&Ty::Int(32)),
                            self.abs(count, num_globals, const_base, func_base),
                            (align.trailing_zeros() as u64 + 1) | (1 << 6),
                        ],
                    );
                }
                Inst::Phi { ty, incoming } => {
                    let mut ops = vec![ty_of(ty)];
                    for (v, bb) in incoming {
                        // Phi operands use a signed VBR relative id: incoming values
                        // may be defined later (loop back edges).
                        let a = self.abs(v, num_globals, const_base, func_base) as i64;
                        let d = cur as i64 - a;
                        ops.push(if d < 0 {
                            ((-d as u64) << 1) | 1
                        } else {
                            (d as u64) << 1
                        });
                        ops.push(*bb as u64);
                    }
                    w.record(FUNC_CODE_INST_PHI, &ops);
                }
                Inst::Br(bb) => w.record(FUNC_CODE_INST_BR, &[*bb as u64]),
                Inst::CondBr {
                    cond,
                    then_bb,
                    else_bb,
                } => {
                    w.record(
                        FUNC_CODE_INST_BR,
                        &[*then_bb as u64, *else_bb as u64, rel(cond, cur)],
                    );
                }
                Inst::Ret(None) => w.record(FUNC_CODE_INST_RET, &[]),
                Inst::Ret(Some(v)) => w.record(FUNC_CODE_INST_RET, &[rel(v, cur)]),
                Inst::Unreachable => w.record(FUNC_CODE_INST_UNREACHABLE, &[]),
            }
            if inst.produces_value() {
                cur += 1;
            }
        }

        if !self.attachments.is_empty() {
            w.enter_block(BLOCK_METADATA_ATTACHMENT, 3);
            for (inst_idx, kind, node) in &self.attachments {
                w.record(
                    METADATA_ATTACHMENT_CODE,
                    &[*inst_idx as u64, *kind as u64, md.resolve(*node) as u64],
                );
            }
            w.end_block();
        }
        w.end_block();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_value_producing_instructions_advance_numbering() {
        let mut b = Body::new(1);
        let a = b.push(Inst::Cast {
            op: cast::SEXT,
            val: Val::Arg(0),
            to: Ty::Int(64),
        });
        assert!(matches!(a, Val::Inst(0)));
        b.push(Inst::Store {
            ptr: Val::Arg(0),
            val: Val::Arg(0),
            align: 4,
        });
        let c = b.push(Inst::Cast {
            op: cast::SEXT,
            val: Val::Arg(0),
            to: Ty::Int(64),
        });
        assert!(
            matches!(c, Val::Inst(1)),
            "store must not consume a result number"
        );
    }

    /// Operand ids are relative to the instruction's own id, so a use of the
    /// immediately preceding result encodes as 1.
    #[test]
    fn relative_ids_are_distance_backwards() {
        let b = Body::new(0);
        // func_base 10, no args, no local consts: instruction 0 has id 10.
        assert_eq!(b.abs(&Val::Inst(0), 0, 5, 10), 10);
        assert_eq!(b.abs(&Val::Inst(1), 0, 5, 10), 11);
        assert_eq!(b.abs(&Val::ModConst(2), 0, 5, 10), 7);
        assert_eq!(b.abs(&Val::Func(3), 0, 5, 10), 3);
        // With two globals present every function id shifts up by two.
        assert_eq!(b.abs(&Val::Global(1), 2, 7, 12), 1);
        assert_eq!(b.abs(&Val::Func(3), 2, 7, 12), 5);
    }
}
