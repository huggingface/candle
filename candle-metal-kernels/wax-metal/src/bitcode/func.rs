//! Function block. Instruction bodies.
//!
//! Uses relative value numbering. Operands are stored as `current_id - operand_id`. Backward refs
//! are small positive numbers (VBR).
//!
//! The value table:
//! * Module values (globals, functions, module constants).
//! * Function arguments.
//! * Function local constants.
//! * One entry per value producing instruction in program order.
//!   (Store, br,and ret void do not advance)

use super::bitstream::BitWriter;
use super::metadata::{MdId, MetadataBuilder};
use super::{BLOCK_FUNCTION, BLOCK_METADATA_ATTACHMENT};
use super::{Const, ConstPool, Ty, TypeTable};

/// Metadata attachment record code, inside a metadata attachment block.
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

/// `llvm::bitc::BinaryOpcodes`. Not to be confused with `llvm::Instruction::BinaryOps`.
/// Float ops reuse the same codes. User picks fp variant `add` -> `fadd`, `udiv` -> `fdiv` etc.
pub(crate) mod binop {
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

/// `llvm::bitc::UnOpcodes`, emitted as `INST_UNOP`.
pub(crate) mod unop {
    pub const FNEG: u32 = 0;
}

/// `llvm::Instruction::CastOpcodes`. Not do be confused with `llvm::Instruction::CastOps`.
pub(crate) mod cast {
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
    #[allow(unused)]
    pub const ADDRSPACECAST: u32 = 12;
}

/// Predicates for `INST_CMP2` (`llvm::CmpInst::Predicate`).
/// Float codes 0->15, integer 32->41.
pub(crate) mod pred {
    pub const FCMP_FALSE: u32 = 0;
    pub const FCMP_OEQ: u32 = 1;
    pub const FCMP_OGT: u32 = 2;
    pub const FCMP_OGE: u32 = 3;
    pub const FCMP_OLT: u32 = 4;
    pub const FCMP_OLE: u32 = 5;
    pub const FCMP_ONE: u32 = 6;
    pub const FCMP_ORD: u32 = 7;
    pub const FCMP_UNO: u32 = 8;
    pub const FCMP_UEQ: u32 = 9;
    pub const FCMP_UGT: u32 = 10;
    pub const FCMP_UGE: u32 = 11;
    pub const FCMP_ULT: u32 = 12;
    pub const FCMP_ULE: u32 = 13;
    pub const FCMP_UNE: u32 = 14;
    pub const FCMP_TRUE: u32 = 15;
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

/// Where a value exists. Resolved to absolute id during emission.
#[derive(Clone, Debug)]
pub enum Val {
    /// Global variable.
    Global(u32),
    /// Module function.
    Func(u32),
    /// Module constant. Indexes the module constant pool.
    ModConst(u32),
    /// Function local constant. Indexes the body's own pool.
    LocalConst(u32),
    /// Function argument, by position.
    Arg(u32),
    /// Result of previous value producing instruction.
    Inst(u32),
}

/// Call flags
/// Function type is explicit.
const CALL_EXPLICIT_FNTY: u64 = 1 << 15;
/// Indicates fast math operand bundle follows.
const CALL_FASTMATH_PRESENT: u64 = 1 << 17;
/// fast math operand bundle id.
const CALL_FAST_BUNDLE: u64 = 254;

/// Allocated type is explicit.
const ALLOCA_EXPLICIT_TY: u64 = 1 << 6;

/// Alignment is stored as `log2(align) + 1`.
fn align_field(align: u32) -> u64 {
    align.trailing_zeros() as u64 + 1
}

/// An instruction. Result numbering is implicit.
#[derive(Clone, Debug)]
pub(crate) enum Inst {
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
    /// A direct call. `fnty` must be the callee's function type. `fast` sets fast math.
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
    /// The record code and operands this instruction writes.
    ///
    /// LLVM numbers operands relative to the instruction, but alloca's size operands use absolute,
    /// ref `rel` vs `abs` usage below.
    pub(super) fn record(&self, e: &IdEnv) -> (u32, Vec<u64>) {
        match self {
            Inst::BinOp { op, lhs, rhs } => (
                FUNC_CODE_INST_BINOP,
                vec![e.rel(lhs), e.rel(rhs), *op as u64],
            ),
            Inst::Cast { op, val, to } => {
                (FUNC_CODE_INST_CAST, vec![e.rel(val), e.ty(to), *op as u64])
            }
            Inst::UnOp { op, val } => (FUNC_CODE_INST_UNOP, vec![e.rel(val), *op as u64]),
            Inst::Cmp { pred, lhs, rhs } => (
                FUNC_CODE_INST_CMP2,
                vec![e.rel(lhs), e.rel(rhs), *pred as u64],
            ),
            // True, false, THEN the condition.
            Inst::Select { cond, t, f } => {
                (FUNC_CODE_INST_SELECT, vec![e.rel(t), e.rel(f), e.rel(cond)])
            }
            Inst::ExtractElt { vec, idx } => {
                (FUNC_CODE_INST_EXTRACTELT, vec![e.rel(vec), e.rel(idx)])
            }
            Inst::InsertElt { vec, elem, idx } => (
                FUNC_CODE_INST_INSERTELT,
                vec![e.rel(vec), e.rel(elem), e.rel(idx)],
            ),
            Inst::Gep {
                inbounds,
                elem,
                ptr,
                indices,
            } => {
                let mut ops = vec![*inbounds as u64, e.ty(elem), e.rel(ptr)];
                ops.extend(indices.iter().map(|i| e.rel(i)));
                (FUNC_CODE_INST_GEP, ops)
            }
            Inst::Load { ptr, ty, align } => (
                FUNC_CODE_INST_LOAD,
                vec![e.rel(ptr), e.ty(ty), align_field(*align), 0],
            ),
            Inst::Store { ptr, val, align } => (
                FUNC_CODE_INST_STORE,
                vec![e.rel(ptr), e.rel(val), align_field(*align), 0],
            ),
            Inst::Call {
                fnty,
                callee,
                args,
                fast,
                ..
            } => {
                let flags = CALL_EXPLICIT_FNTY | if *fast { CALL_FASTMATH_PRESENT } else { 0 };
                let mut ops = vec![0, flags];
                if *fast {
                    ops.push(CALL_FAST_BUNDLE);
                }
                ops.push(e.ty(fnty));
                ops.push(e.rel(callee));
                ops.extend(args.iter().map(|a| e.rel(a)));
                (FUNC_CODE_INST_CALL, ops)
            }
            // Alloca size operand is absolute
            Inst::Alloca { ty, count, align } => (
                FUNC_CODE_INST_ALLOCA,
                vec![
                    e.ty(ty),
                    e.ty(&Ty::Int(32)),
                    e.layout.abs(count),
                    align_field(*align) | ALLOCA_EXPLICIT_TY,
                ],
            ),
            // Signed VBR. Back edge's incoming value is defined after the phi.
            Inst::Phi { ty, incoming } => {
                let mut ops = vec![e.ty(ty)];
                for (v, bb) in incoming {
                    ops.push(e.rel_signed(v));
                    ops.push(*bb as u64);
                }
                (FUNC_CODE_INST_PHI, ops)
            }
            Inst::Br(bb) => (FUNC_CODE_INST_BR, vec![*bb as u64]),
            Inst::CondBr {
                cond,
                then_bb,
                else_bb,
            } => (
                FUNC_CODE_INST_BR,
                vec![*then_bb as u64, *else_bb as u64, e.rel(cond)],
            ),
            Inst::Ret(v) => (
                FUNC_CODE_INST_RET,
                v.as_ref().map(|v| vec![e.rel(v)]).unwrap_or_default(),
            ),
            Inst::Unreachable => (FUNC_CODE_INST_UNREACHABLE, Vec::new()),
        }
    }

    /// Whether this instruction ends a basic block. The body is a flat list, so this is
    /// the only block structure it has.
    fn is_terminator(&self) -> bool {
        matches!(
            self,
            Inst::Br(_) | Inst::CondBr { .. } | Inst::Ret(_) | Inst::Unreachable
        )
    }

    /// Whether this instruction contributes an entry to the value table.
    fn produces_value(&self) -> bool {
        match self {
            Inst::Store { .. }
            | Inst::Br(_)
            | Inst::CondBr { .. }
            | Inst::Ret(_)
            | Inst::Unreachable => false,
            // A call may or may not produce a value.
            Inst::Call { returns_value, .. } => *returns_value,
            Inst::BinOp { .. }
            | Inst::Cast { .. }
            | Inst::UnOp { .. }
            | Inst::Cmp { .. }
            | Inst::Select { .. }
            | Inst::ExtractElt { .. }
            | Inst::InsertElt { .. }
            | Inst::Gep { .. }
            | Inst::Load { .. }
            | Inst::Alloca { .. }
            | Inst::Phi { .. } => true,
        }
    }
}

/// Where each kind of value sits in the value table.
#[derive(Clone, Copy)]
pub struct ValueLayout {
    pub num_globals: u32,
    pub const_base: u32,
    pub func_base: u32,
    pub num_args: u32,
    pub num_consts: u32,
}

impl ValueLayout {
    pub fn abs(&self, v: &Val) -> u64 {
        match v {
            Val::Global(i) => *i as u64,
            Val::Func(i) => (self.num_globals + *i) as u64,
            Val::ModConst(i) => (self.const_base + i) as u64,
            Val::Arg(i) => (self.func_base + i) as u64,
            Val::LocalConst(i) => (self.func_base + self.num_args + i) as u64,
            Val::Inst(i) => (self.func_base + self.num_args + self.num_consts + i) as u64,
        }
    }

    /// The first value producing instruction id.
    pub fn first_inst_id(&self) -> u32 {
        self.func_base + self.num_args + self.num_consts
    }
}

/// The information required to to correctly record `Val`/`Ty`.
/// Layout, type table, and pending instruction id.
pub struct IdEnv<'a> {
    pub layout: ValueLayout,
    pub types: &'a TypeTable,
    pub cur: u32,
}

impl IdEnv<'_> {
    /// Relative position id. `cur - abs(v)`.
    pub fn rel(&self, v: &Val) -> u64 {
        (self.cur as i64 - self.layout.abs(v) as i64) as u64
    }

    /// Phi needs signed VBR relative id, as position can be negative (back edge phi).
    pub fn rel_signed(&self, v: &Val) -> u64 {
        let d = self.cur as i64 - self.layout.abs(v) as i64;
        if d < 0 {
            ((-d as u64) << 1) | 1
        } else {
            (d as u64) << 1
        }
    }

    pub fn ty(&self, t: &Ty) -> u64 {
        self.types.get(t).expect("interned: Module::check_types") as u64
    }
}

/// Function definition body.
#[derive(Default)]
pub struct Body {
    insts: Vec<Inst>,
    /// The next value producing instruction id.
    next_id: u32,
    pub num_blocks: u32,
    pub consts: ConstPool,
    pub num_args: u32,
    /// `(inst idx, metadata kind id, node)`. Used for encoding `!llvm.loop` unroll hints.
    attachments: Vec<(u32, u32, MdId)>,
    /// `(inst idx, unroll count)` for terminators with unroll hints.
    /// Turned into `attachments` once the module's metadata exists.
    pub loop_hints: Vec<(u32, u32)>,
}

/// A phi to be wired up. Created by [`Body::push_phi`].
#[derive(Clone, Copy, Debug)]
pub struct PhiRef(usize);

impl Body {
    pub fn new(num_args: u32) -> Self {
        Self {
            insts: Vec::new(),
            next_id: 0,
            num_blocks: 1,
            consts: ConstPool::default(),
            num_args,
            attachments: Vec::new(),
            loop_hints: Vec::new(),
        }
    }

    /// Current instruction count.
    pub fn len(&self) -> usize {
        self.insts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.insts.is_empty()
    }

    /// Validate the body.
    ///
    /// Checks that phis are wired correctly, and that terminators match blocks.
    pub(crate) fn validate(&self) -> Result<(), String> {
        // Edges into each block.
        let mut preds = vec![0usize; self.num_blocks as usize];
        for i in &self.insts {
            match i {
                Inst::Br(bb) => {
                    if let Some(p) = preds.get_mut(*bb as usize) {
                        *p += 1;
                    }
                }
                Inst::CondBr {
                    then_bb, else_bb, ..
                } => {
                    for bb in [then_bb, else_bb] {
                        if let Some(p) = preds.get_mut(*bb as usize) {
                            *p += 1;
                        }
                    }
                }
                _ => {}
            }
        }

        let mut block = 0usize;
        let mut past_phis = false;
        let mut terminators = 0usize;
        for (idx, inst) in self.insts.iter().enumerate() {
            match inst {
                Inst::Phi { incoming, .. } => {
                    if past_phis {
                        return Err(format!(
                            "instruction {idx}: phi follows a non-phi in block {block}"
                        ));
                    }
                    let predecessors = preds.get(block).copied().unwrap_or(0);
                    if incoming.len() != predecessors {
                        return Err(format!(
                            "instruction {idx}: phi in block {block} has {} incoming for \
                            {predecessors} predecessors",
                            incoming.len()
                        ));
                    }
                }
                _ => past_phis = true,
            }
            if inst.is_terminator() {
                terminators += 1;
                block += 1;
                past_phis = false;
            }
        }
        if terminators != self.num_blocks as usize {
            return Err(format!(
                "{terminators} terminators for {} blocks: block is not terminated or a terminator \
                 is mid-block",
                self.num_blocks
            ));
        }
        Ok(())
    }

    /// All types needed in the module type table to write the body.
    /// A type missing from the table is an error during writing, so we intern all.
    pub(super) fn referenced_types(&self) -> Vec<Ty> {
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

    /// Attach metadata to the instruction at `idx`.
    pub fn attach(&mut self, idx: u32, kind: u32, node: MdId) {
        self.attachments.push((idx, kind, node));
    }

    /// Attach metadata to the most recent instruction.
    pub fn attach_to_last(&mut self, kind: u32, node: MdId) {
        let idx = self.insts.len().saturating_sub(1) as u32;
        self.attachments.push((idx, kind, node));
    }

    pub(crate) fn push(&mut self, i: Inst) -> Option<Val> {
        let produces = i.produces_value();
        self.insts.push(i);
        produces.then(|| self.next_val())
    }

    /// Mint the id of the instruction just pushed.
    fn next_val(&mut self) -> Val {
        let v = Val::Inst(self.next_id);
        self.next_id += 1;
        v
    }

    /// Phi with no incoming values yet, and a handle to wire it up later.
    pub fn push_phi(&mut self, ty: Ty) -> (Val, PhiRef) {
        let at = self.insts.len();
        self.insts.push(Inst::Phi {
            ty,
            incoming: Vec::new(),
        });
        (self.next_val(), PhiRef(at))
    }

    /// Wire one predecessor edge into an existing phi.
    pub fn add_phi_incoming(&mut self, p: PhiRef, v: Val, from_bb: u32) {
        match &mut self.insts[p.0] {
            Inst::Phi { incoming, .. } => incoming.push((v, from_bb)),
            // `PhiRef`'s field is private and only `push_phi` mints one, immediately
            // after pushing the phi it indexes. Instructions are never removed.
            other => panic!("PhiRef does not point at a phi: {other:?}"),
        }
    }

    /// A function local integer constant.
    pub fn iconst(&mut self, ty: Ty, v: i64) -> Val {
        Val::LocalConst(self.consts.intern(Const::Int(ty, v)))
    }

    /// A function local float constant, as raw bits at `ty`'s width.
    pub fn fconst(&mut self, ty: Ty, bits: u64) -> Val {
        Val::LocalConst(self.consts.intern(Const::Float(ty, bits)))
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

        let layout = ValueLayout {
            num_globals,
            const_base,
            func_base,
            num_args: self.num_args,
            num_consts: self.consts.len() as u32,
        };
        let mut e = IdEnv {
            layout,
            types,
            cur: layout.first_inst_id(),
        };

        for inst in &self.insts {
            let (code, ops) = inst.record(&e);
            w.record(code, &ops);
            if inst.produces_value() {
                e.cur += 1;
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
    fn body_validation_works_as_expected() {
        let good = || {
            let mut b = Body::new(0);
            b.num_blocks = 2;
            b.push(Inst::Br(1));
            b.push(Inst::Ret(None));
            b
        };
        assert!(good().validate().is_ok());

        // A block with no terminator.
        let mut b = Body::new(0);
        b.num_blocks = 2;
        b.push(Inst::Ret(None));
        let e = b.validate().expect_err("one terminator for two blocks");
        assert!(e.contains("terminator"), "{e}");

        // A phi after a non-phi in the same block.
        let mut b = Body::new(0);
        b.num_blocks = 2;
        b.push(Inst::Br(1));
        let (_, slot) = b.push_phi(Ty::Int(32));
        b.add_phi_incoming(slot, Val::Arg(0), 0);
        b.push(Inst::Ret(None));
        assert!(b.validate().is_ok(), "a phi at a block start is fine");

        let mut b = Body::new(0);
        b.num_blocks = 2;
        b.push(Inst::Br(1));
        b.push(Inst::Cast {
            op: cast::SEXT,
            val: Val::Arg(0),
            to: Ty::Int(64),
        });
        let (_, slot) = b.push_phi(Ty::Int(32));
        b.add_phi_incoming(slot, Val::Arg(0), 0);
        b.push(Inst::Ret(None));
        let e = b.validate().expect_err("phi must not follow non-phi");
        assert!(e.contains("follows a non-phi"), "{e}");

        // A phi wired for fewer edges than reach its block.
        let mut b = Body::new(0);
        b.num_blocks = 2;
        b.push(Inst::Br(1));
        let (_, _slot) = b.push_phi(Ty::Int(32));
        b.push(Inst::Ret(None));
        let e = b.validate().expect_err("phi has no incoming");
        assert!(e.contains("incoming"), "{e}");
    }

    #[test]
    fn only_value_producing_instructions_advance_numbering() {
        let mut b = Body::new(1);
        let a = b.push(Inst::Cast {
            op: cast::SEXT,
            val: Val::Arg(0),
            to: Ty::Int(64),
        });
        assert!(matches!(a, Some(Val::Inst(0))));
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
            matches!(c, Some(Val::Inst(1))),
            "store must not consume a result number"
        );
    }

    #[test]
    fn value_layout_ids_are_calculated_correctly() {
        let l = |num_globals, const_base, func_base| ValueLayout {
            num_globals,
            const_base,
            func_base,
            num_args: 0,
            num_consts: 0,
        };
        assert_eq!(l(0, 5, 10).abs(&Val::Inst(0)), 10);
        assert_eq!(l(0, 5, 10).abs(&Val::Inst(1)), 11);
        assert_eq!(l(0, 5, 10).abs(&Val::ModConst(2)), 7);
        assert_eq!(l(0, 5, 10).abs(&Val::Func(3)), 3);
        // Two globals -> function id shifts by two.
        assert_eq!(l(2, 7, 12).abs(&Val::Global(1)), 1);
        assert_eq!(l(2, 7, 12).abs(&Val::Func(3)), 5);
    }

    fn table() -> TypeTable {
        let mut t = TypeTable::default();
        t.intern(&Ty::Int(32));
        t.intern(&Ty::Float);
        t.intern(&Ty::ptr(Ty::Float, 1));
        t
    }

    /// Distinct bases.
    fn env(types: &TypeTable) -> IdEnv<'_> {
        IdEnv {
            layout: ValueLayout {
                num_globals: 2,
                const_base: 5,
                func_base: 10,
                num_args: 3,
                num_consts: 4,
            },
            types,
            cur: 20,
        }
    }

    /// The operands alone, for the cases whose code is not the point.
    fn ops(i: Inst, e: &IdEnv) -> Vec<u64> {
        i.record(e).1
    }

    #[test]
    fn abs_places_each_value_kind_in_its_own_range() {
        let t = table();
        let e = env(&t);
        assert_eq!(e.layout.abs(&Val::Global(1)), 1);
        assert_eq!(e.layout.abs(&Val::Func(1)), 3, "functions follow globals");
        assert_eq!(e.layout.abs(&Val::ModConst(1)), 6);
        assert_eq!(e.layout.abs(&Val::Arg(1)), 11);
        assert_eq!(e.layout.abs(&Val::LocalConst(1)), 14, "after the args");
        assert_eq!(e.layout.abs(&Val::Inst(1)), 18, "after the local consts");
        assert_eq!(e.layout.first_inst_id(), 17);
    }

    #[test]
    fn unop_and_binop_record_correctly() {
        let t = table();
        let e = env(&t);
        let (a, b) = (Val::Arg(0), Val::Arg(1));
        assert_eq!(
            Inst::BinOp {
                op: 7,
                lhs: a.clone(),
                rhs: b.clone()
            }
            .record(&e),
            (FUNC_CODE_INST_BINOP, vec![e.rel(&a), e.rel(&b), 7])
        );
        assert_eq!(
            Inst::UnOp {
                op: 3,
                val: a.clone()
            }
            .record(&e),
            (FUNC_CODE_INST_UNOP, vec![e.rel(&a), 3])
        );
    }

    #[test]
    fn select_records_correctly() {
        let t = table();
        let e = env(&t);
        let (c, x, y) = (Val::Arg(0), Val::Arg(1), Val::Arg(2));
        assert_eq!(
            ops(
                Inst::Select {
                    cond: c.clone(),
                    t: x.clone(),
                    f: y.clone()
                },
                &e
            ),
            vec![e.rel(&x), e.rel(&y), e.rel(&c)]
        );
    }

    #[test]
    fn load_and_store_encode_align_as_log2_plus_one() {
        let t = table();
        let e = env(&t);
        assert_eq!(
            ops(
                Inst::Load {
                    ptr: Val::Arg(0),
                    ty: Ty::Float,
                    align: 4
                },
                &e
            )[2],
            3
        );
        assert_eq!(
            ops(
                Inst::Store {
                    ptr: Val::Arg(0),
                    val: Val::Arg(1),
                    align: 8
                },
                &e
            )[2],
            4
        );
    }

    #[test]
    fn alloca_records_absolute_size_id_and_explicit_type_flag() {
        let t = table();
        let e = env(&t);
        let count = Val::LocalConst(1);
        let got = ops(
            Inst::Alloca {
                ty: Ty::Float,
                count: count.clone(),
                align: 4,
            },
            &e,
        );
        assert_eq!(got[2], e.layout.abs(&count), "the size operand is absolute");
        assert_ne!(got[2], e.rel(&count), "and differs from the relative form");
        assert_eq!(got[3], 3 | ALLOCA_EXPLICIT_TY);
    }

    #[test]
    fn call_records_explicit_type_and_fast_bundle_correctly() {
        let mut t = table();
        let fnty = Ty::Func {
            ret: Box::new(Ty::Void),
            params: vec![Ty::Float],
            vararg: false,
        };
        t.intern(&fnty);
        let e = env(&t);
        let call = |fast| Inst::Call {
            fnty: fnty.clone(),
            callee: Val::Func(0),
            args: vec![Val::Arg(0)],
            returns_value: false,
            fast,
        };
        let plain = ops(call(false), &e);
        let fast = ops(call(true), &e);
        assert_eq!(plain[1], CALL_EXPLICIT_FNTY);
        assert_eq!(fast[1], CALL_EXPLICIT_FNTY | CALL_FASTMATH_PRESENT);
        assert_eq!(fast[2], CALL_FAST_BUNDLE);
        assert_eq!(fast.len(), plain.len() + 1, "the bundle is the only extra");
    }

    #[test]
    fn phi_uses_signed_vbr_for_back_edge_support() {
        let t = table();
        let e = env(&t);
        let (fwd, back) = (Val::Arg(0), Val::Inst(50));
        assert!(
            e.layout.abs(&back) > e.cur as u64,
            "the fixture must actually be a back edge"
        );
        assert_eq!(e.rel_signed(&fwd) & 1, 0, "forward is even");
        assert_eq!(e.rel_signed(&back) & 1, 1, "backward sets the sign bit");
        let got = ops(
            Inst::Phi {
                ty: Ty::Float,
                incoming: vec![(fwd.clone(), 1), (back.clone(), 2)],
            },
            &e,
        );
        assert_eq!(
            got,
            vec![
                e.ty(&Ty::Float),
                e.rel_signed(&fwd),
                1,
                e.rel_signed(&back),
                2
            ]
        );
    }

    #[test]
    fn br_and_cond_br_records_correctly() {
        let t = table();
        let e = env(&t);
        let (uncond, cond) = (
            Inst::Br(3).record(&e),
            Inst::CondBr {
                cond: Val::Arg(0),
                then_bb: 3,
                else_bb: 7,
            }
            .record(&e),
        );
        assert_eq!(uncond, (FUNC_CODE_INST_BR, vec![3]));
        assert_eq!(cond, (FUNC_CODE_INST_BR, vec![3, 7, e.rel(&Val::Arg(0))]));
    }

    #[test]
    fn gep_records_correctly() {
        let t = table();
        let e = env(&t);
        let got = ops(
            Inst::Gep {
                inbounds: true,
                elem: Ty::Float,
                ptr: Val::Arg(0),
                indices: vec![Val::Arg(1), Val::Arg(2)],
            },
            &e,
        );
        assert_eq!(got[0], 1);
        assert_eq!(got[1], e.ty(&Ty::Float));
        assert_eq!(got.len(), 5, "inbounds, elem ty, ptr, two indices");
    }
}
