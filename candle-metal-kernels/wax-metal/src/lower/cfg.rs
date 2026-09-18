//! Typed control-flow-graph builder for AIR - the sole lowering path.
//!
//! A function is typed basic blocks whose instructions reference values by [`ValueId`]
//! handles, never by predicted names. A phi's back-edge incoming is a handle to a value
//! produced later in the body, so there is no name to predict and no second classifier to
//! disagree with. Names are assigned in a final serialization pass, and a structural
//! verifier checks SSA dominance before any text is produced.
//!
//! This replaced a string emitter that encoded SSA identity in name-mangling conventions
//! several code paths had to agree on; when they disagreed the dangling reference surfaced
//! only as an assembler line number.
#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use crate::LowerError;
use crate::ir_builder::LlType;

// Handles

/// Handle to an SSA value (function parameter or instruction result).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ValueId(pub u32);

/// Handle to a basic block.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BlockId(pub u32);

// Operands

/// An instruction operand: either a defined SSA value or an inline literal.
#[derive(Clone, Debug)]
pub enum Operand {
    Val(ValueId),
    Imm {
        text: String,
        ty: LlType,
        /// The same constant in structured form. The text emitter uses `text`; the
        /// bitcode emitter needs a value it can encode, and cannot parse LLVM
        /// constant syntax back out of a string. Sites that are still only
        /// expressible as text carry [`ImmVal::Text`], which the bitcode path
        /// reports rather than mis-encoding.
        val: ImmVal,
    },
}

/// What an array-indexing GEP starts from: a module-scope symbol, or a pointer value
/// such as an `alloca` result.
#[derive(Clone, Debug)]
pub enum GepBase {
    Global(String),
    Value(Operand),
}

/// Structured form of an immediate operand.
#[derive(Clone, PartialEq, Debug)]
pub enum ImmVal {
    Int(i64),
    Float(f32),
    /// A float constant as its raw bit pattern, at the operand type's width. This is
    /// what bitcode's float constant record stores, and it is the only form that can
    /// carry f16/bf16 specials (infinities) without going through an f32.
    FloatBits(u64),
    /// A vector of integers, e.g. an SGMMA `<i64 128, i64 8>` stride pair.
    VecInt(Vec<i64>),
    /// A vector of float constants by raw bits, e.g. a `<64 x float>` SGMMA tile.
    VecFloatBits(Vec<u64>),
    Undef,
    Zero,
    /// A reference to a module-scope symbol, e.g. `@_tg_reduce_result`.
    Global(String),
    /// Only available as LLVM constant text (e.g. a `<i64 128, i64 8>` vector).
    Text,
}

impl Operand {
    /// An immediate whose only form is LLVM constant text. Prefer a typed
    /// constructor: this one is invisible to the bitcode emitter.
    pub fn imm(text: impl Into<String>, ty: LlType) -> Self {
        Operand::Imm {
            text: text.into(),
            ty,
            val: ImmVal::Text,
        }
    }
    /// An immediate carrying both its text and its structured value.
    fn typed(text: impl Into<String>, ty: LlType, val: ImmVal) -> Self {
        Operand::Imm {
            text: text.into(),
            ty,
            val,
        }
    }
    pub fn i32(v: i64) -> Self {
        Operand::typed(v.to_string(), LlType::I32, ImmVal::Int(v))
    }
    pub fn i16(v: i64) -> Self {
        Operand::typed(v.to_string(), LlType::I16, ImmVal::Int(v))
    }
    pub fn i64(v: i64) -> Self {
        Operand::typed(v.to_string(), LlType::I64, ImmVal::Int(v))
    }
    pub fn i1(b: bool) -> Self {
        Operand::typed((b as i32).to_string(), LlType::I1, ImmVal::Int(b as i64))
    }
    pub fn undef(ty: LlType) -> Self {
        Operand::typed("undef", ty, ImmVal::Undef)
    }
    pub fn zeroinit(ty: LlType) -> Self {
        Operand::typed("zeroinitializer", ty, ImmVal::Zero)
    }
    /// f32 constant in LLVM's 64-bit hex form.
    pub fn f32(v: f32) -> Self {
        Operand::typed(crate::types::float_hex(v), LlType::Float, ImmVal::Float(v))
    }
    /// A float constant given by its raw bits at `ty`'s width.
    pub fn float_bits(text: impl Into<String>, ty: LlType, bits: u64) -> Self {
        Operand::typed(text, ty, ImmVal::FloatBits(bits))
    }
    /// An integer vector constant.
    pub fn vec_int(text: impl Into<String>, ty: LlType, vals: Vec<i64>) -> Self {
        Operand::typed(text, ty, ImmVal::VecInt(vals))
    }
    /// An immediate with an explicit structured value, for callers that compute the
    /// text and the value together.
    pub fn constant(text: impl Into<String>, ty: LlType, val: ImmVal) -> Self {
        Operand::typed(text, ty, val)
    }
    /// A module-scope symbol reference, e.g. a threadgroup global.
    pub fn global(name: impl Into<String>, ty: LlType) -> Self {
        let name = name.into();
        Operand::typed(name.clone(), ty, ImmVal::Global(name))
    }

    /// The structured value, if this is an immediate.
    pub fn imm_val(&self) -> Option<&ImmVal> {
        match self {
            Operand::Imm { val, .. } => Some(val),
            Operand::Val(_) => None,
        }
    }
}

impl From<ValueId> for Operand {
    fn from(v: ValueId) -> Self {
        Operand::Val(v)
    }
}

// Instructions & terminators

#[derive(Clone, Debug)]
pub(crate) enum InstKind {
    Bin {
        op: &'static str,
        lhs: Operand,
        rhs: Operand,
    },
    Cast {
        op: &'static str,
        src: Operand,
        to: LlType,
    },
    Icmp {
        pred: &'static str,
        lhs: Operand,
        rhs: Operand,
    },
    Fcmp {
        pred: &'static str,
        lhs: Operand,
        rhs: Operand,
    },
    Select {
        cond: Operand,
        t: Operand,
        f: Operand,
    },
    Load {
        ptr: Operand,
        align: u32,
    },
    Store {
        val: Operand,
        ptr: Operand,
        align: u32,
    },
    Gep {
        elem: LlType,
        ptr: Operand,
        idx: Operand,
    },
    ExtractElem {
        vec: Operand,
        idx: Operand,
    },
    InsertElem {
        vec: Operand,
        elem: Operand,
        idx: Operand,
    },
    /// `getelementptr inbounds [count x elem], [count x elem] addrspace(A)* @global,
    /// i32 0, i32 idx` - indexing a module-scope array such as threadgroup scratch.
    /// Distinct from `Gep` because it needs the leading zero index that steps through
    /// the array type, and because its base is a symbol rather than a value.
    GepArray {
        base: GepBase,
        elem: LlType,
        count: u32,
        addrspace: u32,
        idx: Operand,
    },
    /// A thread-private stack slot: `alloca [count x elem]`, or `alloca elem` when
    /// `count` is `None`. The two are distinct types, and the atomics' expected-value
    /// slot needs the scalar one.
    Alloca {
        elem: LlType,
        count: Option<u32>,
        align: u32,
    },
    Call {
        callee: String,
        args: Vec<Operand>,
        /// Emit `call fast`, i.e. with the full fast-math bundle. AIR's `air.fast_*`
        /// intrinsics are called this way.
        fast: bool,
    },
    /// Escape hatch for AIR text without a dedicated typed variant (threadgroup
    /// globals, `alloca`, exotic intrinsics). `template` is the instruction body
    /// (no `%dst = ` prefix, no newline); `$0`, `$1`, ... are replaced by the
    /// names of `operands`, which still participate in SSA verification.
    Raw {
        template: String,
        operands: Vec<Operand>,
    },
}

#[derive(Clone, Debug)]
struct Inst {
    result: Option<ValueId>,
    kind: InstKind,
    /// Program-order sequence number within the function (for same-block
    /// def-before-use checks).
    seq: u32,
}

#[derive(Clone, Debug)]
pub(crate) struct Phi {
    result: ValueId,
    ty: LlType,
    incomings: Vec<(Operand, BlockId)>,
}

/// How a basic block ends.
#[derive(Clone, Debug)]
pub enum Terminator {
    Br(BlockId),
    CondBr {
        cond: Operand,
        then_: BlockId,
        else_: BlockId,
    },
    Ret,
    Unreachable,
}

struct BlockData {
    label: String,
    phis: Vec<Phi>,
    insts: Vec<Inst>,
    term: Option<Terminator>,
    /// If set, attach `!llvm.loop` metadata forcing this many unroll copies to
    /// the block's terminator. Set on a loop latch (back-edge) branch.
    loop_unroll: Option<u32>,
}

// Function builder

pub struct FuncBuilder {
    name: String,
    params: Vec<ValueId>,
    blocks: Vec<BlockData>,
    cur: Option<BlockId>,

    // Value tables, indexed by ValueId.0.
    value_types: Vec<LlType>,
    value_names: Vec<Option<String>>, // Some for params (fixed); None => assigned at serialize.
    value_def_block: Vec<Option<BlockId>>, // None => parameter (dominates everything).
    value_def_seq: Vec<u32>,

    /// Locate a phi by its result handle: ValueId -> (block index, phi index).
    phi_loc: HashMap<u32, (usize, usize)>,
    seq: u32,
}

impl FuncBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        FuncBuilder {
            name: name.into(),
            params: Vec::new(),
            blocks: Vec::new(),
            cur: None,
            value_types: Vec::new(),
            value_names: Vec::new(),
            value_def_block: Vec::new(),
            value_def_seq: Vec::new(),
            phi_loc: HashMap::new(),
            seq: 0,
        }
    }

    fn alloc_value(
        &mut self,
        ty: LlType,
        name: Option<String>,
        def_block: Option<BlockId>,
    ) -> ValueId {
        let id = ValueId(self.value_types.len() as u32);
        let seq = self.seq;
        self.seq += 1;
        self.value_types.push(ty);
        self.value_names.push(name);
        self.value_def_block.push(def_block);
        self.value_def_seq.push(seq);
        id
    }

    /// Declare a function parameter with a fixed AIR name (e.g. `"%_p0"`).
    pub fn param(&mut self, name: &str, ty: LlType) -> ValueId {
        let id = self.alloc_value(ty, Some(name.to_string()), None);
        self.params.push(id);
        id
    }

    /// Create a block and return its handle (does not switch to it).
    pub fn block(&mut self, label: &str) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        self.blocks.push(BlockData {
            label: label.to_string(),
            phis: Vec::new(),
            insts: Vec::new(),
            term: None,
            loop_unroll: None,
        });
        id
    }

    /// Make `b` the current block for subsequent emission.
    pub fn switch_to(&mut self, b: BlockId) {
        self.cur = Some(b);
    }

    /// Force `count` unroll copies of the loop whose latch is `b`, by attaching
    /// `!llvm.loop` unroll metadata to `b`'s terminator. Purely a performance
    /// hint: the driver still unrolls a semantically identical loop, so numerics
    /// (including floating-point summation order) are unchanged.
    pub fn set_loop_unroll(&mut self, b: BlockId, count: u32) {
        self.blocks[b.0 as usize].loop_unroll = Some(count);
    }

    /// Number of non-phi instructions currently in `b` - used to size unroll
    /// hints so a large loop body is not blown up.
    pub fn block_inst_count(&self, b: BlockId) -> usize {
        self.blocks[b.0 as usize].insts.len()
    }

    /// Create a block and immediately switch to it.
    pub fn append_block(&mut self, label: &str) -> BlockId {
        let b = self.block(label);
        self.switch_to(b);
        b
    }

    fn cur_block(&self) -> BlockId {
        self.cur
            .expect("no current block; call switch_to/append_block first")
    }

    /// The LLVM type of a value handle.
    /// Types of the function's parameters, in signature order.
    pub(crate) fn param_types(&self) -> Vec<LlType> {
        self.params
            .iter()
            .map(|p| self.value_types[p.0 as usize].clone())
            .collect()
    }

    /// Distinct call targets with the signature each call site implies: the result's
    /// type is the return type, the operand types are the parameters. Derived from the
    /// CFG so a declaration can never disagree with the call that uses it.
    pub(crate) fn callees(&self) -> Vec<(String, LlType, Vec<LlType>)> {
        let mut out: Vec<(String, LlType, Vec<LlType>)> = Vec::new();
        for b in &self.blocks {
            for i in &b.insts {
                if let InstKind::Call { callee, args, .. } = &i.kind {
                    if out.iter().any(|(n, _, _)| n == callee) {
                        continue;
                    }
                    let ret = i
                        .result
                        .map(|r| self.value_types[r.0 as usize].clone())
                        .unwrap_or(LlType::Void);
                    let ps = args.iter().map(|a| self.type_of(a)).collect();
                    out.push((callee.clone(), ret, ps));
                }
            }
        }
        out
    }

    /// Read-only views used by the bitcode translator (`lower::to_bitcode`), which
    /// needs to walk the finished CFG rather than build one.
    pub fn num_params(&self) -> usize {
        self.params.len()
    }

    pub(crate) fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Position of `v` among the function's parameters, if it is one.
    /// The `i`th parameter's value id. `param_index` is the reverse; the bitcode emitter
    /// only needs that direction, but a translation that rebuilds a parameter list needs
    /// this one.
    pub(crate) fn param_id(&self, i: usize) -> ValueId {
        self.params[i]
    }

    pub(crate) fn param_index(&self, v: ValueId) -> Option<usize> {
        self.params.iter().position(|p| *p == v)
    }

    /// Result ids of `b`'s phi nodes, in order.
    pub fn block_phis(&self, b: BlockId) -> Vec<ValueId> {
        self.blocks[b.0 as usize]
            .phis
            .iter()
            .map(|p| p.result)
            .collect()
    }

    /// Incoming (value, predecessor) pairs of a phi.
    pub fn phi_incomings(&self, phi: ValueId) -> Vec<(Operand, BlockId)> {
        for b in &self.blocks {
            for p in &b.phis {
                if p.result == phi {
                    return p.incomings.clone();
                }
            }
        }
        Vec::new()
    }

    /// `(result, kind)` for each instruction in `b`, in order.
    pub(crate) fn block_insts(&self, b: BlockId) -> Vec<(Option<ValueId>, &InstKind)> {
        self.blocks[b.0 as usize]
            .insts
            .iter()
            .map(|i| (i.result, &i.kind))
            .collect()
    }

    /// Unroll hint on `b`'s terminator, if any.
    pub fn block_loop_unroll(&self, b: BlockId) -> Option<u32> {
        self.blocks[b.0 as usize].loop_unroll
    }

    pub(crate) fn block_term(&self, b: BlockId) -> Option<&Terminator> {
        self.blocks[b.0 as usize].term.as_ref()
    }

    pub fn value_type(&self, v: ValueId) -> LlType {
        self.value_types[v.0 as usize].clone()
    }

    /// The current block, if one is selected.
    pub fn current_block(&self) -> Option<BlockId> {
        self.cur
    }

    fn op_ty(&self, o: &Operand) -> LlType {
        match o {
            Operand::Val(v) => self.value_types[v.0 as usize].clone(),
            Operand::Imm { ty, .. } => ty.clone(),
        }
    }

    /// Public accessor for an operand's LLVM type (scalar or `<N x T>` vector).
    pub fn type_of(&self, o: &Operand) -> LlType {
        self.op_ty(o)
    }

    /// Emit `fneg {ty} $0`, preserving the operand's (possibly vector) type.
    pub fn fneg(&mut self, a: impl Into<Operand>) -> ValueId {
        let a = a.into();
        let ty = self.op_ty(&a);
        self.push_inst(
            Some(ty.clone()),
            InstKind::Raw {
                template: format!("fneg {} $0", ty.ll_str()),
                operands: vec![a],
            },
        )
        .unwrap()
    }

    fn push_inst(&mut self, result_ty: Option<LlType>, kind: InstKind) -> Option<ValueId> {
        let b = self.cur_block();
        let seq = self.seq;
        self.seq += 1;
        let result = result_ty.map(|ty| {
            // alloc_value bumps seq again; keep the inst's seq stable by using `seq`.
            let id = ValueId(self.value_types.len() as u32);
            self.value_types.push(ty);
            self.value_names.push(None);
            self.value_def_block.push(Some(b));
            self.value_def_seq.push(seq);
            id
        });
        self.blocks[b.0 as usize]
            .insts
            .push(Inst { result, kind, seq });
        result
    }

    // Emission helpers

    fn bin(
        &mut self,
        op: &'static str,
        lhs: impl Into<Operand>,
        rhs: impl Into<Operand>,
    ) -> ValueId {
        let (lhs, rhs) = (lhs.into(), rhs.into());
        let ty = self.op_ty(&lhs);
        assert_eq!(ty, self.op_ty(&rhs), "{op}: operand type mismatch");
        self.push_inst(Some(ty), InstKind::Bin { op, lhs, rhs })
            .unwrap()
    }
    pub fn add(&mut self, a: impl Into<Operand>, b: impl Into<Operand>) -> ValueId {
        self.bin("add", a, b)
    }
    pub fn sub(&mut self, a: impl Into<Operand>, b: impl Into<Operand>) -> ValueId {
        self.bin("sub", a, b)
    }
    pub fn mul(&mut self, a: impl Into<Operand>, b: impl Into<Operand>) -> ValueId {
        self.bin("mul", a, b)
    }
    pub fn and(&mut self, a: impl Into<Operand>, b: impl Into<Operand>) -> ValueId {
        self.bin("and", a, b)
    }
    pub fn or(&mut self, a: impl Into<Operand>, b: impl Into<Operand>) -> ValueId {
        self.bin("or", a, b)
    }
    pub fn xor(&mut self, a: impl Into<Operand>, b: impl Into<Operand>) -> ValueId {
        self.bin("xor", a, b)
    }
    pub fn shl(&mut self, a: impl Into<Operand>, b: impl Into<Operand>) -> ValueId {
        self.bin("shl", a, b)
    }
    pub fn fsub(&mut self, a: impl Into<Operand>, b: impl Into<Operand>) -> ValueId {
        self.bin("fsub", a, b)
    }
    pub fn fdiv(&mut self, a: impl Into<Operand>, b: impl Into<Operand>) -> ValueId {
        self.bin("fdiv", a, b)
    }
    pub fn sdiv(&mut self, a: impl Into<Operand>, b: impl Into<Operand>) -> ValueId {
        self.bin("sdiv", a, b)
    }
    pub fn udiv(&mut self, a: impl Into<Operand>, b: impl Into<Operand>) -> ValueId {
        self.bin("udiv", a, b)
    }
    pub fn srem(&mut self, a: impl Into<Operand>, b: impl Into<Operand>) -> ValueId {
        self.bin("srem", a, b)
    }
    pub fn urem(&mut self, a: impl Into<Operand>, b: impl Into<Operand>) -> ValueId {
        self.bin("urem", a, b)
    }
    pub fn lshr(&mut self, a: impl Into<Operand>, b: impl Into<Operand>) -> ValueId {
        self.bin("lshr", a, b)
    }
    pub fn ashr(&mut self, a: impl Into<Operand>, b: impl Into<Operand>) -> ValueId {
        self.bin("ashr", a, b)
    }
    pub fn fadd(&mut self, a: impl Into<Operand>, b: impl Into<Operand>) -> ValueId {
        self.bin("fadd", a, b)
    }
    pub fn fmul(&mut self, a: impl Into<Operand>, b: impl Into<Operand>) -> ValueId {
        self.bin("fmul", a, b)
    }

    pub fn icmp(
        &mut self,
        pred: &'static str,
        a: impl Into<Operand>,
        b: impl Into<Operand>,
    ) -> ValueId {
        let (lhs, rhs) = (a.into(), b.into());
        let ty = self.cmp_result_ty(&lhs);
        self.push_inst(Some(ty), InstKind::Icmp { pred, lhs, rhs })
            .unwrap()
    }
    pub fn fcmp(
        &mut self,
        pred: &'static str,
        a: impl Into<Operand>,
        b: impl Into<Operand>,
    ) -> ValueId {
        let (lhs, rhs) = (a.into(), b.into());
        let ty = self.cmp_result_ty(&lhs);
        self.push_inst(Some(ty), InstKind::Fcmp { pred, lhs, rhs })
            .unwrap()
    }

    /// Result type of a compare: `i1` for scalars, `<N x i1>` for an `<N x T>`
    /// operand (so a vector-mask select prints the correct condition type).
    fn cmp_result_ty(&self, operand: &Operand) -> LlType {
        match self.op_ty(operand) {
            LlType::Vec(n, _) => LlType::Vec(n, Box::new(LlType::I1)),
            _ => LlType::I1,
        }
    }

    pub fn cast(&mut self, op: &'static str, src: impl Into<Operand>, to: LlType) -> ValueId {
        let src = src.into();
        self.push_inst(Some(to.clone()), InstKind::Cast { op, src, to })
            .unwrap()
    }
    pub fn sext(&mut self, src: impl Into<Operand>, to: LlType) -> ValueId {
        self.cast("sext", src, to)
    }
    pub fn trunc(&mut self, src: impl Into<Operand>, to: LlType) -> ValueId {
        self.cast("trunc", src, to)
    }

    pub fn select(
        &mut self,
        cond: impl Into<Operand>,
        t: impl Into<Operand>,
        f: impl Into<Operand>,
    ) -> ValueId {
        let (cond, t, f) = (cond.into(), t.into(), f.into());
        let ty = self.op_ty(&t);
        self.push_inst(Some(ty), InstKind::Select { cond, t, f })
            .unwrap()
    }

    pub fn load(&mut self, ptr: impl Into<Operand>, align: u32) -> ValueId {
        let ptr = ptr.into();
        let pointee = self.op_ty(&ptr).pointee().clone();
        self.push_inst(Some(pointee), InstKind::Load { ptr, align })
            .unwrap()
    }
    pub fn store(&mut self, val: impl Into<Operand>, ptr: impl Into<Operand>, align: u32) {
        let (val, ptr) = (val.into(), ptr.into());
        self.push_inst(None, InstKind::Store { val, ptr, align });
    }
    pub fn gep(
        &mut self,
        elem: LlType,
        ptr: impl Into<Operand>,
        idx: impl Into<Operand>,
    ) -> ValueId {
        let ptr = ptr.into();
        let ty = self.op_ty(&ptr);
        self.push_inst(
            Some(ty),
            InstKind::Gep {
                elem,
                ptr,
                idx: idx.into(),
            },
        )
        .unwrap()
    }

    pub fn extractelement(&mut self, vec: impl Into<Operand>, idx: impl Into<Operand>) -> ValueId {
        let vec = vec.into();
        let elem = self.op_ty(&vec).elem().clone();
        self.push_inst(
            Some(elem),
            InstKind::ExtractElem {
                vec,
                idx: idx.into(),
            },
        )
        .unwrap()
    }
    pub fn insertelement(
        &mut self,
        vec: impl Into<Operand>,
        elem: impl Into<Operand>,
        idx: impl Into<Operand>,
    ) -> ValueId {
        let vec = vec.into();
        let ty = self.op_ty(&vec);
        self.push_inst(
            Some(ty),
            InstKind::InsertElem {
                vec,
                elem: elem.into(),
                idx: idx.into(),
            },
        )
        .unwrap()
    }

    /// Emit a call. `ret == LlType::Void` produces a result-less `call void`.
    /// Index a module-scope array global, yielding an `elem addrspace(A)*`.
    pub fn gep_global(
        &mut self,
        global: impl Into<String>,
        elem: LlType,
        count: u32,
        addrspace: u32,
        idx: impl Into<Operand>,
    ) -> Option<ValueId> {
        self.gep_array(GepBase::Global(global.into()), elem, count, addrspace, idx)
    }

    /// Index an array through a pointer value, e.g. an `alloca` result.
    pub fn gep_array_value(
        &mut self,
        ptr: impl Into<Operand>,
        elem: LlType,
        count: u32,
        addrspace: u32,
        idx: impl Into<Operand>,
    ) -> Option<ValueId> {
        self.gep_array(GepBase::Value(ptr.into()), elem, count, addrspace, idx)
    }

    fn gep_array(
        &mut self,
        base: GepBase,
        elem: LlType,
        count: u32,
        addrspace: u32,
        idx: impl Into<Operand>,
    ) -> Option<ValueId> {
        let ptr = LlType::Ptr {
            pointee: Box::new(elem.clone()),
            addrspace,
        };
        self.push_inst(
            Some(ptr),
            InstKind::GepArray {
                base,
                elem,
                count,
                addrspace,
                idx: idx.into(),
            },
        )
    }

    /// `alloca [count x elem]`, yielding an `elem addrspace(0)*`.
    pub fn alloca_array(&mut self, elem: LlType, count: u32, align: u32) -> Option<ValueId> {
        self.alloca(elem, Some(count), align)
    }

    /// `alloca elem`, yielding an `elem addrspace(0)*`.
    pub fn alloca_scalar(&mut self, elem: LlType, align: u32) -> Option<ValueId> {
        self.alloca(elem, None, align)
    }

    fn alloca(&mut self, elem: LlType, count: Option<u32>, align: u32) -> Option<ValueId> {
        let ptr = LlType::Ptr {
            pointee: Box::new(elem.clone()),
            addrspace: 0,
        };
        self.push_inst(Some(ptr), InstKind::Alloca { elem, count, align })
    }

    /// A float constant `1.0` at `ty`, carrying the right bit pattern for its width.
    pub fn float_one(ty: &LlType) -> Operand {
        let bits: u64 = match ty {
            LlType::Half => 0x3C00,
            LlType::BFloat => 0x3F80,
            LlType::Double => 0x3FF0_0000_0000_0000,
            _ => 0x3F80_0000,
        };
        Operand::float_bits("1.0", ty.clone(), bits)
    }

    /// `call fast` - the fast-math form used for AIR's `air.fast_*` intrinsics and
    /// the SIMD reductions.
    pub fn call_fast(
        &mut self,
        ret: LlType,
        callee: impl Into<String>,
        args: Vec<Operand>,
    ) -> Option<ValueId> {
        let kind = InstKind::Call {
            callee: callee.into(),
            args,
            fast: true,
        };
        if ret == LlType::Void {
            self.push_inst(None, kind)
        } else {
            self.push_inst(Some(ret), kind)
        }
    }

    pub fn call(
        &mut self,
        ret: LlType,
        callee: impl Into<String>,
        args: Vec<Operand>,
    ) -> Option<ValueId> {
        let kind = InstKind::Call {
            callee: callee.into(),
            args,
            fast: false,
        };
        if ret == LlType::Void {
            self.push_inst(None, kind)
        } else {
            self.push_inst(Some(ret), kind)
        }
    }

    /// Emit an instruction from a raw template. `$i` placeholders are replaced by
    /// `operands[i]`'s assigned name; pass `Some(ty)` for a result-producing
    /// instruction (which gets a `%dst = ` prefix) or `None` for a void one.
    pub fn raw(
        &mut self,
        result_ty: Option<LlType>,
        template: impl Into<String>,
        operands: Vec<Operand>,
    ) -> Option<ValueId> {
        self.push_inst(
            result_ty,
            InstKind::Raw {
                template: template.into(),
                operands,
            },
        )
    }

    // Phis

    /// Add a phi to the current block and return its value handle. Incomings are
    /// filled later via [`Self::phi_incoming`], so back-edge values need not
    /// exist yet - the key to eliminating name prediction.
    pub fn phi(&mut self, ty: LlType) -> ValueId {
        let b = self.cur_block();
        assert!(
            self.blocks[b.0 as usize].insts.is_empty(),
            "phi must be added before non-phi instructions in a block"
        );
        let seq = self.seq;
        self.seq += 1;
        let id = ValueId(self.value_types.len() as u32);
        self.value_types.push(ty.clone());
        self.value_names.push(None);
        self.value_def_block.push(Some(b));
        self.value_def_seq.push(seq);
        let phi_idx = self.blocks[b.0 as usize].phis.len();
        self.blocks[b.0 as usize].phis.push(Phi {
            result: id,
            ty,
            incomings: Vec::new(),
        });
        self.phi_loc.insert(id.0, (b.0 as usize, phi_idx));
        id
    }

    /// Add one `[ value, %pred ]` incoming to a previously created phi.
    pub fn phi_incoming(&mut self, phi: ValueId, val: impl Into<Operand>, pred: BlockId) {
        let (bi, pi) = *self
            .phi_loc
            .get(&phi.0)
            .expect("phi_incoming: not a phi handle");
        self.blocks[bi].phis[pi].incomings.push((val.into(), pred));
    }

    // Loop carries
    //
    // A carried value is a `Vec<ValueId>`: one handle for a scalar, or one per
    // fragment for an SGMMA grid. These helpers make the fragment count of each
    // carry flow from a single source, so the loop header and body cannot end up
    // with differently-shaped representations - the exact defect (`%v83` dangling
    // against `%vs83_0_*`) that motivated this rewrite becomes impossible.

    /// Create the header phis for a set of loop carries. `fragment_types[i]` lists
    /// the LLVM types of carry `i`'s fragments; returns the matching phi handles.
    /// Must be called while the header block is current.
    pub fn carry_phis(&mut self, fragment_types: &[Vec<LlType>]) -> Vec<Vec<ValueId>> {
        fragment_types
            .iter()
            .map(|frags| frags.iter().map(|t| self.phi(t.clone())).collect())
            .collect()
    }

    /// Wire the entry and back edges of carries created by [`Self::carry_phis`].
    /// Panics if any carry's fragment count differs across the header phis, the
    /// entry-edge inits, or the back-edge updates - the invariant the legacy
    /// emitter silently violated.
    pub fn wire_carries(
        &mut self,
        phis: &[Vec<ValueId>],
        entry: BlockId,
        inits: &[Vec<Operand>],
        back: BlockId,
        updates: &[Vec<Operand>],
    ) {
        assert_eq!(
            phis.len(),
            inits.len(),
            "carry count mismatch: phis vs inits"
        );
        assert_eq!(
            phis.len(),
            updates.len(),
            "carry count mismatch: phis vs updates"
        );
        for i in 0..phis.len() {
            assert_eq!(
                phis[i].len(),
                inits[i].len(),
                "carry {i}: fragment count mismatch between header phis and entry inits"
            );
            assert_eq!(
                phis[i].len(),
                updates[i].len(),
                "carry {i}: fragment count mismatch between header phis and back-edge updates"
            );
            for f in 0..phis[i].len() {
                self.phi_incoming(phis[i][f], inits[i][f].clone(), entry);
                self.phi_incoming(phis[i][f], updates[i][f].clone(), back);
            }
        }
    }

    pub fn set_terminator(&mut self, term: Terminator) {
        let b = self.cur_block();
        self.blocks[b.0 as usize].term = Some(term);
    }

    /// Set the current block's terminator only if it has none yet. Used for
    /// fall-through edges after a body that may have already branched away
    /// (e.g. a `break` that terminated the block with a jump to the loop exit).
    pub fn set_terminator_if_open(&mut self, term: Terminator) {
        let b = self.cur_block();
        if self.blocks[b.0 as usize].term.is_none() {
            self.blocks[b.0 as usize].term = Some(term);
        }
    }

    /// Whether the current block already has a terminator.
    pub fn is_terminated(&self) -> bool {
        self.current_block()
            .is_some_and(|b| self.blocks[b.0 as usize].term.is_some())
    }

    // Verify

    /// Structural SSA verification: every block terminates, phi incomings match
    /// predecessors exactly, and every value use is dominated by its definition.
    pub fn verify(&self) -> Result<(), LowerError> {
        let n = self.blocks.len();
        if n == 0 {
            return Err(self.err("<none>", "function has no blocks"));
        }

        // 1. Every block must have a terminator.
        for b in &self.blocks {
            if b.term.is_none() {
                return Err(self.err(&b.label, "block has no terminator"));
            }
        }

        // 2. CFG edges + reachability from entry (block 0).
        let succs: Vec<Vec<usize>> = self
            .blocks
            .iter()
            .map(|b| match b.term.as_ref().unwrap() {
                Terminator::Br(t) => vec![t.0 as usize],
                Terminator::CondBr { then_, else_, .. } => vec![then_.0 as usize, else_.0 as usize],
                Terminator::Ret | Terminator::Unreachable => vec![],
            })
            .collect();

        let mut reachable = vec![false; n];
        let mut stack = vec![0usize];
        reachable[0] = true;
        while let Some(x) = stack.pop() {
            for &s in &succs[x] {
                if !reachable[s] {
                    reachable[s] = true;
                    stack.push(s);
                }
            }
        }

        // 3. Predecessors (over reachable blocks only).
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (x, ss) in succs.iter().enumerate() {
            if !reachable[x] {
                continue;
            }
            for &s in ss {
                preds[s].push(x);
            }
        }

        // 4. Iterative dominators over reachable blocks.
        let doms = self.compute_dominators(n, &preds, &reachable);
        let dominates = |a: Option<BlockId>, b: usize| -> bool {
            match a {
                None => true, // parameter - defined before entry, dominates all.
                Some(ab) => doms[b].contains(&(ab.0 as usize)),
            }
        };

        // 5. Phi incomings must equal the predecessor set exactly.
        for (bi, blk) in self.blocks.iter().enumerate() {
            if !reachable[bi] {
                continue;
            }
            let pred_set: HashSet<usize> = preds[bi].iter().copied().collect();
            for phi in &blk.phis {
                let mut seen: HashSet<usize> = HashSet::new();
                for (_, pred) in &phi.incomings {
                    seen.insert(pred.0 as usize);
                }
                if seen != pred_set {
                    return Err(self.err(
                        &blk.label,
                        &format!(
                            "phi {} incomings {:?} do not match block predecessors {:?}",
                            self.debug_name(phi.result),
                            self.labels_of(&seen),
                            self.labels_of(&pred_set),
                        ),
                    ));
                }
                // Each incoming value's def must dominate the predecessor edge.
                for (op, pred) in &phi.incomings {
                    if let Operand::Val(v) = op {
                        let db = self.value_def_block[v.0 as usize];
                        if reachable[pred.0 as usize] && !dominates(db, pred.0 as usize) {
                            return Err(self.err(
                                &blk.label,
                                &format!(
                                    "phi {} incoming {} from %{} is not dominated by its definition",
                                    self.debug_name(phi.result),
                                    self.debug_name(*v),
                                    self.blocks[pred.0 as usize].label,
                                ),
                            ));
                        }
                    }
                }
            }
        }

        // 6. Non-phi value uses must be dominated by their definitions.
        for (bi, blk) in self.blocks.iter().enumerate() {
            if !reachable[bi] {
                continue;
            }
            for inst in &blk.insts {
                for op in inst_operands(&inst.kind) {
                    if let Operand::Val(v) = op {
                        self.check_use(*v, bi, Some(inst.seq), &dominates)?;
                    }
                }
            }
            if let Some(term) = &blk.term
                && let Terminator::CondBr {
                    cond: Operand::Val(v),
                    ..
                } = term
            {
                self.check_use(*v, bi, None, &dominates)?;
            }
        }

        Ok(())
    }

    fn check_use(
        &self,
        v: ValueId,
        use_block: usize,
        use_seq: Option<u32>,
        dominates: &dyn Fn(Option<BlockId>, usize) -> bool,
    ) -> Result<(), LowerError> {
        let db = self.value_def_block[v.0 as usize];
        if !dominates(db, use_block) {
            return Err(self.err(
                &self.blocks[use_block].label,
                &format!(
                    "use of {} is not dominated by its definition",
                    self.debug_name(v)
                ),
            ));
        }
        // Same-block: definition must precede use in program order.
        if let (Some(dbb), Some(us)) = (db, use_seq)
            && dbb.0 as usize == use_block
            && self.value_def_seq[v.0 as usize] >= us
        {
            return Err(self.err(
                &self.blocks[use_block].label,
                &format!("use of {} precedes its definition", self.debug_name(v)),
            ));
        }
        Ok(())
    }

    fn compute_dominators(
        &self,
        n: usize,
        preds: &[Vec<usize>],
        reachable: &[bool],
    ) -> Vec<HashSet<usize>> {
        let all: HashSet<usize> = (0..n).filter(|&i| reachable[i]).collect();
        let mut dom: Vec<HashSet<usize>> = (0..n)
            .map(|i| {
                if i == 0 {
                    let mut s = HashSet::new();
                    s.insert(0);
                    s
                } else if reachable[i] {
                    all.clone()
                } else {
                    HashSet::new()
                }
            })
            .collect();

        let mut changed = true;
        while changed {
            changed = false;
            for b in 1..n {
                if !reachable[b] {
                    continue;
                }
                let mut new: Option<HashSet<usize>> = None;
                for &p in &preds[b] {
                    new = Some(match new {
                        None => dom[p].clone(),
                        Some(acc) => acc.intersection(&dom[p]).copied().collect(),
                    });
                }
                let mut new = new.unwrap_or_default();
                new.insert(b);
                if new != dom[b] {
                    dom[b] = new;
                    changed = true;
                }
            }
        }
        dom
    }

    fn labels_of(&self, set: &HashSet<usize>) -> Vec<String> {
        let mut v: Vec<String> = set.iter().map(|&i| self.blocks[i].label.clone()).collect();
        v.sort();
        v
    }

    fn debug_name(&self, v: ValueId) -> String {
        self.value_names[v.0 as usize]
            .clone()
            .unwrap_or_else(|| format!("%v{}", v.0))
    }

    fn err(&self, block: &str, msg: &str) -> LowerError {
        LowerError::MalformedIr(format!("{msg} (kernel @{}, block %{block})", self.name))
    }

    // Serialize

    /// Verify then render the function to AIR text (`define ... { ... }`).
    /// Serialize the function to AIR text. `meta_base` is the index of the next
    /// free module-level metadata node; any `!llvm.loop` nodes this function needs
    /// are numbered from there and returned so the caller can append them to the
    /// module metadata table (their positional index must match the references
    /// emitted here).
    pub fn serialize(&mut self, meta_base: usize) -> Result<(String, Vec<String>), LowerError> {
        self.verify()?;
        self.assign_names();

        // Pre-allocate loop-metadata node ids for every block carrying an unroll
        // hint, so the `!llvm.loop !N` reference on the terminator matches the
        // node table appended by the caller. Each loop needs three nodes:
        //   !L   = distinct !{!L, !E, !C}   (the loop id, self-referential)
        //   !E   = !{!"llvm.loop.unroll.enable"}
        //   !C   = !{!"llvm.loop.unroll.count", i32 <count>}
        let mut loop_meta: Vec<String> = Vec::new();
        let mut loop_id_of: Vec<Option<usize>> = vec![None; self.blocks.len()];
        for (bi, blk) in self.blocks.iter().enumerate() {
            if let Some(count) = blk.loop_unroll {
                let l = meta_base + loop_meta.len();
                let e = l + 1;
                let c = l + 2;
                loop_meta.push(format!("distinct !{{!{l}, !{e}, !{c}}}"));
                loop_meta.push("!{!\"llvm.loop.unroll.enable\"}".into());
                loop_meta.push(format!("!{{!\"llvm.loop.unroll.count\", i32 {count}}}"));
                loop_id_of[bi] = Some(l);
            }
        }

        let mut out = String::new();
        let params: Vec<String> = self
            .params
            .iter()
            .map(|p| {
                format!(
                    "{} {}",
                    self.value_types[p.0 as usize].ll_str(),
                    self.name_of(*p)
                )
            })
            .collect();
        let _ = writeln!(
            out,
            "define void @{}({}) #0 {{",
            self.name,
            params.join(", ")
        );

        for (bi, blk) in self.blocks.iter().enumerate() {
            let _ = writeln!(out, "{}:", blk.label);
            for phi in &blk.phis {
                let incs: Vec<String> = phi
                    .incomings
                    .iter()
                    .map(|(op, pred)| {
                        format!(
                            "[ {}, %{} ]",
                            self.operand_name(op),
                            self.blocks[pred.0 as usize].label
                        )
                    })
                    .collect();
                let _ = writeln!(
                    out,
                    "    {} = phi {} {}",
                    self.name_of(phi.result),
                    phi.ty.ll_str(),
                    incs.join(", ")
                );
            }
            for inst in &blk.insts {
                let _ = writeln!(out, "    {}", self.render_inst(inst));
            }
            let term = self.render_term(blk.term.as_ref().unwrap());
            match loop_id_of[bi] {
                Some(l) => {
                    let _ = writeln!(out, "    {term}, !llvm.loop !{l}");
                }
                None => {
                    let _ = writeln!(out, "    {term}");
                }
            }
        }

        let _ = writeln!(out, "}}");
        Ok((out, loop_meta))
    }

    fn assign_names(&mut self) {
        let mut counter = 0u32;
        for i in 0..self.value_names.len() {
            if self.value_names[i].is_none() {
                self.value_names[i] = Some(format!("%v{counter}"));
                counter += 1;
            }
        }
    }

    fn name_of(&self, v: ValueId) -> String {
        self.value_names[v.0 as usize]
            .clone()
            .unwrap_or_else(|| format!("%v{}", v.0))
    }

    fn operand_name(&self, o: &Operand) -> String {
        match o {
            Operand::Val(v) => self.name_of(*v),
            Operand::Imm { text, .. } => text.clone(),
        }
    }
    fn operand_typed(&self, o: &Operand) -> String {
        format!("{} {}", self.op_ty(o).ll_str(), self.operand_name(o))
    }

    fn render_inst(&self, inst: &Inst) -> String {
        let dst = inst
            .result
            .map(|r| format!("{} = ", self.name_of(r)))
            .unwrap_or_default();
        match &inst.kind {
            InstKind::Bin { op, lhs, rhs } => format!(
                "{dst}{op} {} {}, {}",
                self.op_ty(lhs).ll_str(),
                self.operand_name(lhs),
                self.operand_name(rhs)
            ),
            InstKind::Cast { op, src, to } => {
                format!("{dst}{op} {} to {}", self.operand_typed(src), to.ll_str())
            }
            InstKind::Icmp { pred, lhs, rhs } => format!(
                "{dst}icmp {pred} {} {}, {}",
                self.op_ty(lhs).ll_str(),
                self.operand_name(lhs),
                self.operand_name(rhs)
            ),
            InstKind::Fcmp { pred, lhs, rhs } => format!(
                "{dst}fcmp {pred} {} {}, {}",
                self.op_ty(lhs).ll_str(),
                self.operand_name(lhs),
                self.operand_name(rhs)
            ),
            InstKind::Select { cond, t, f } => format!(
                "{dst}select {}, {}, {}",
                self.operand_typed(cond),
                self.operand_typed(t),
                self.operand_typed(f)
            ),
            InstKind::Load { ptr, align } => format!(
                "{dst}load {}, {}, align {align}",
                self.op_ty(ptr).pointee().ll_str(),
                self.operand_typed(ptr)
            ),
            InstKind::Store { val, ptr, align } => format!(
                "store {}, {}, align {align}",
                self.operand_typed(val),
                self.operand_typed(ptr)
            ),
            InstKind::Gep { elem, ptr, idx } => format!(
                "{dst}getelementptr {}, {}, {}",
                elem.ll_str(),
                self.operand_typed(ptr),
                self.operand_typed(idx)
            ),
            InstKind::ExtractElem { vec, idx } => format!(
                "{dst}extractelement {}, {}",
                self.operand_typed(vec),
                self.operand_typed(idx)
            ),
            InstKind::InsertElem { vec, elem, idx } => format!(
                "{dst}insertelement {}, {}, {}",
                self.operand_typed(vec),
                self.operand_typed(elem),
                self.operand_typed(idx)
            ),
            InstKind::GepArray {
                base,
                elem,
                count,
                addrspace,
                idx,
            } => {
                let t = format!("[{count} x {}]", elem.ll_str());
                // addrspace(0) is spelled by omission, matching what the raw templates
                // emitted for alloca-based bases.
                let asp = if *addrspace == 0 {
                    String::new()
                } else {
                    format!(" addrspace({addrspace})")
                };
                let b = match base {
                    GepBase::Global(g) => g.clone(),
                    GepBase::Value(v) => self.operand_name(v),
                };
                format!(
                    "{dst}getelementptr inbounds {t}, {t}{asp}* {b}, i32 0, i32 {}",
                    self.operand_name(idx)
                )
            }
            InstKind::Alloca { elem, count, align } => {
                let t = match count {
                    Some(n) => format!("[{n} x {}]", elem.ll_str()),
                    None => elem.ll_str(),
                };
                format!("{dst}alloca {t}, align {align}")
            }
            InstKind::Call { callee, args, fast } => {
                let ret_ty = inst
                    .result
                    .map(|r| self.value_types[r.0 as usize].ll_str())
                    .unwrap_or_else(|| "void".into());
                let a: Vec<String> = args.iter().map(|o| self.operand_typed(o)).collect();
                let fm = if *fast { "fast " } else { "" };
                format!("{dst}call {fm}{ret_ty} {callee}({})", a.join(", "))
            }
            InstKind::Raw { template, operands } => {
                let mut body = template.clone();
                // Replace highest index first so `$1` does not clobber `$10`.
                for (i, o) in operands.iter().enumerate().rev() {
                    body = body.replace(&format!("${i}"), &self.operand_name(o));
                }
                format!("{dst}{body}")
            }
        }
    }

    fn render_term(&self, t: &Terminator) -> String {
        match t {
            Terminator::Br(b) => format!("br label %{}", self.blocks[b.0 as usize].label),
            Terminator::CondBr { cond, then_, else_ } => format!(
                "br i1 {}, label %{}, label %{}",
                self.operand_name(cond),
                self.blocks[then_.0 as usize].label,
                self.blocks[else_.0 as usize].label
            ),
            Terminator::Ret => "ret void".to_string(),
            Terminator::Unreachable => "unreachable".to_string(),
        }
    }
}

/// The value operands referenced by an instruction (for use-checking).
fn inst_operands(kind: &InstKind) -> Vec<&Operand> {
    match kind {
        InstKind::Bin { lhs, rhs, .. }
        | InstKind::Icmp { lhs, rhs, .. }
        | InstKind::Fcmp { lhs, rhs, .. } => vec![lhs, rhs],
        InstKind::Cast { src, .. } => vec![src],
        InstKind::Select { cond, t, f } => vec![cond, t, f],
        InstKind::Load { ptr, .. } => vec![ptr],
        InstKind::Store { val, ptr, .. } => vec![val, ptr],
        InstKind::Gep { ptr, idx, .. } => vec![ptr, idx],
        InstKind::ExtractElem { vec, idx } => vec![vec, idx],
        InstKind::InsertElem { vec, elem, idx } => vec![vec, elem, idx],
        InstKind::GepArray { base, idx, .. } => match base {
            GepBase::Global(_) => vec![idx],
            GepBase::Value(p) => vec![p, idx],
        },
        // An alloca's size is a compile-time constant, so it has no operands.
        InstKind::Alloca { .. } => vec![],
        InstKind::Call { args, .. } => args.iter().collect(),
        InstKind::Raw { operands, .. } => operands.iter().collect(),
    }
}

#[cfg(test)]
mod tests {
    /// Every immediate must carry a structured value that agrees with its text, since
    /// the two emitters read different fields: divergence would show up as a text
    /// module and a bitcode module that compute different things.
    #[test]
    fn immediates_carry_matching_structured_values() {
        use super::{ImmVal, LlType, Operand};
        assert_eq!(Operand::i32(-5).imm_val(), Some(&ImmVal::Int(-5)));
        assert_eq!(Operand::i16(3).imm_val(), Some(&ImmVal::Int(3)));
        assert_eq!(Operand::i64(1 << 40).imm_val(), Some(&ImmVal::Int(1 << 40)));
        assert_eq!(Operand::i1(true).imm_val(), Some(&ImmVal::Int(1)));
        assert_eq!(
            Operand::undef(LlType::Float).imm_val(),
            Some(&ImmVal::Undef)
        );
        assert_eq!(
            Operand::zeroinit(LlType::Float).imm_val(),
            Some(&ImmVal::Zero)
        );
        assert_eq!(
            Operand::float_bits("0xH7C00", LlType::Half, 0x7C00).imm_val(),
            Some(&ImmVal::FloatBits(0x7C00))
        );
        assert_eq!(
            Operand::vec_int("<i64 1, i64 2>", LlType::vec(2, LlType::I64), vec![1, 2]).imm_val(),
            Some(&ImmVal::VecInt(vec![1, 2]))
        );
        assert_eq!(
            Operand::global("@g", LlType::I32).imm_val(),
            Some(&ImmVal::Global("@g".to_string()))
        );
        // The untyped escape hatch is explicitly marked so the bitcode emitter can
        // report it instead of silently emitting something wrong.
        assert_eq!(
            Operand::imm("<i64 128, i64 8>", LlType::I64).imm_val(),
            Some(&ImmVal::Text)
        );
        // Text is preserved for the text emitter regardless.
        match Operand::i32(-5) {
            Operand::Imm { text, .. } => assert_eq!(text, "-5"),
            _ => panic!("expected an immediate"),
        }
    }

    use super::*;
    use crate::ir_builder::LlType;
    use crate::lower::verify::verify_function;

    /// Build a scalar-accumulator reduction loop - the shape the legacy emitter
    /// mis-lowered for GEMV. Here the carry is a single handle, so the header and
    /// body cannot disagree about its representation.
    fn build_sum_loop() -> FuncBuilder {
        let mut f = FuncBuilder::new("sum_loop");
        let p = f.param("%_p0", LlType::device_ptr(LlType::Float));

        let entry = f.append_block("entry");
        let hdr = f.block("hdr");
        let body = f.block("body");
        let back = f.block("back");
        let exit = f.block("exit");

        f.switch_to(entry);
        f.set_terminator(Terminator::Br(hdr));

        f.switch_to(hdr);
        let iv = f.phi(LlType::I32);
        let acc = f.phi(LlType::Float);
        let cond = f.icmp("slt", iv, Operand::i32(16));
        f.set_terminator(Terminator::CondBr {
            cond: cond.into(),
            then_: body,
            else_: exit,
        });

        f.switch_to(body);
        let ptr = f.gep(LlType::Float, p, iv);
        let x = f.load(ptr, 4);
        let acc_next = f.fadd(acc, x);
        f.set_terminator(Terminator::Br(back));

        f.switch_to(back);
        let iv_next = f.add(iv, Operand::i32(1));
        f.set_terminator(Terminator::Br(hdr));

        // Back-edge incomings reference values defined in the body - as handles,
        // with no textual name prediction.
        f.phi_incoming(iv, Operand::i32(0), entry);
        f.phi_incoming(iv, iv_next, back);
        f.phi_incoming(acc, Operand::f32(0.0), entry);
        f.phi_incoming(acc, acc_next, back);

        f.switch_to(exit);
        f.store(acc, p, 4);
        f.set_terminator(Terminator::Ret);

        f
    }

    #[test]
    fn valid_loop_verifies() {
        let f = build_sum_loop();
        assert!(f.verify().is_ok(), "{:?}", f.verify());
    }

    #[test]
    fn serialized_loop_passes_text_verifier() {
        // Cross-check: the typed serializer's output satisfies the independent
        // Phase-0 textual SSA verifier.
        let mut f = build_sum_loop();
        let text = f.serialize(0).expect("serialize").0;
        verify_function(&text).unwrap_or_else(|e| panic!("text verifier rejected:\n{text}\n{e}"));
        assert!(text.contains("phi i32"));
        assert!(text.contains("phi float"));
    }

    #[test]
    fn detects_use_not_dominated() {
        // A value defined in one arm of a branch, used in the other arm.
        let mut f = FuncBuilder::new("bad_dom");
        let entry = f.append_block("entry");
        let a = f.block("a");
        let b = f.block("b");
        let merge = f.block("merge");

        f.switch_to(entry);
        let c = f.icmp("slt", Operand::i32(0), Operand::i32(1));
        f.set_terminator(Terminator::CondBr {
            cond: c.into(),
            then_: a,
            else_: b,
        });

        f.switch_to(a);
        let only_in_a = f.add(Operand::i32(1), Operand::i32(2));
        f.set_terminator(Terminator::Br(merge));

        f.switch_to(b);
        // Use a value from block `a` - not dominating block `b`.
        let _ = f.add(only_in_a, Operand::i32(3));
        f.set_terminator(Terminator::Br(merge));

        f.switch_to(merge);
        f.set_terminator(Terminator::Ret);

        let err = f.verify().unwrap_err();
        assert!(err.to_string().contains("not dominated"), "{err}");
    }

    #[test]
    fn detects_phi_incoming_mismatch() {
        let mut f = FuncBuilder::new("bad_phi");
        let entry = f.append_block("entry");
        let hdr = f.block("hdr");
        f.switch_to(entry);
        f.set_terminator(Terminator::Br(hdr));
        f.switch_to(hdr);
        let p = f.phi(LlType::I32);
        // Missing the back-edge incoming from hdr itself.
        f.phi_incoming(p, Operand::i32(0), entry);
        f.set_terminator(Terminator::Br(hdr));
        let err = f.verify().unwrap_err();
        assert!(
            err.to_string().contains("do not match block predecessors"),
            "{err}"
        );
    }

    #[test]
    fn detects_missing_terminator() {
        let mut f = FuncBuilder::new("no_term");
        f.append_block("entry");
        let err = f.verify().unwrap_err();
        assert!(err.to_string().contains("no terminator"), "{err}");
    }

    /// Assemble `header_extra + body` with xcrun metal-as. Returns Ok(()) on
    /// success or if xcrun is unavailable; Err(stderr) if metal-as rejects it.
    fn metal_as(tag: &str, header_extra: &str, body: &str) -> Result<(), String> {
        let p = crate::profile::TargetProfile::host();
        let module = format!(
            "target datalayout = \"{}\"\ntarget triple = \"{}\"\n{header_extra}\n{body}",
            p.datalayout(),
            p.triple()
        );
        let tmp = std::env::temp_dir();
        let ll = tmp.join(format!("{tag}.ll"));
        let air = tmp.join(format!("{tag}.air"));
        std::fs::write(&ll, &module).unwrap();
        let out = std::process::Command::new("xcrun")
            .args([
                "metal-as",
                ll.to_str().unwrap(),
                "-o",
                air.to_str().unwrap(),
            ])
            .output();
        match out {
            Err(_) => Ok(()), // xcrun unavailable - skip.
            Ok(o) if o.status.success() => Ok(()),
            Ok(o) => Err(format!(
                "metal-as rejected:\n{module}\nstderr:\n{}",
                String::from_utf8_lossy(&o.stderr)
            )),
        }
    }

    #[test]
    fn metal_as_assembles_serialized_loop() {
        let mut f = build_sum_loop();
        let body = f.serialize(0).expect("serialize").0;
        metal_as("cfg_sum_loop", "", &body).unwrap();
    }

    /// The GEMV failure at its root: an M<8 accumulator that SGMMA fragments into
    /// a grid, carried across a loop. In the legacy emitter the header classified
    /// this carry as scalar while the body used the fragment form, producing a
    /// dangling `%v83`. Here the carry is N phi handles wired to N body results,
    /// with the count coming from one place - a mismatch is structurally
    /// impossible. Proves the typed model fixes the bug that motivated the rewrite.
    #[test]
    fn sgmma_grid_carry_loop_verifies_and_assembles() {
        const NC: usize = 4; // 1x4 fragment grid - the M=1, BN=32 accumulator.
        let v64 = LlType::vec(64, LlType::Float);
        let dptr = LlType::device_ptr(LlType::Float);
        let stride = || {
            Operand::vec_int(
                "<i64 128, i64 8>",
                LlType::vec(2, LlType::I64),
                vec![128, 8],
            )
        };
        let zero2 = || Operand::zeroinit(LlType::vec(2, LlType::I64));

        let mut f = FuncBuilder::new("sgmma_gemv");
        let pz = f.param("%_p0", dptr.clone());
        let pa = f.param("%_p1", dptr.clone());
        let pb = f.param("%_p2", dptr.clone());

        let entry = f.append_block("entry");
        let hdr = f.block("hdr");
        let body = f.block("body");
        let back = f.block("back");
        let exit = f.block("exit");

        // entry: initialise NC accumulator fragments to the zero diagonal.
        f.switch_to(entry);
        let inits: Vec<ValueId> = (0..NC)
            .map(|_| {
                f.call(
                    v64.clone(),
                    "@air.simdgroup_matrix_8x8_init_diag.v64f32.f32",
                    vec![Operand::f32(0.0)],
                )
                .unwrap()
            })
            .collect();
        f.set_terminator(Terminator::Br(hdr));

        // hdr: carries = [iv] ++ NC accumulator fragments, all via carry_phis so
        // their fragment counts come from one description.
        f.switch_to(hdr);
        let mut carry_types: Vec<Vec<LlType>> = vec![vec![LlType::I32]];
        carry_types.extend((0..NC).map(|_| vec![v64.clone()]));
        let phis = f.carry_phis(&carry_types);
        let iv = phis[0][0];
        let acc: Vec<ValueId> = (0..NC).map(|k| phis[1 + k][0]).collect();
        let cond = f.icmp("slt", iv, Operand::i32(16));
        f.set_terminator(Terminator::CondBr {
            cond: cond.into(),
            then_: body,
            else_: exit,
        });

        // body: load A, load NC B fragments, NC multiply-accumulates.
        f.switch_to(body);
        let a = f
            .call(
                v64.clone(),
                "@air.simdgroup_matrix_8x8_load.v64f32.p1f32",
                vec![pa.into(), stride(), stride(), zero2()],
            )
            .unwrap();
        let updates: Vec<ValueId> = (0..NC)
            .map(|k| {
                let b_k = f
                    .call(
                        v64.clone(),
                        "@air.simdgroup_matrix_8x8_load.v64f32.p1f32",
                        vec![pb.into(), stride(), stride(), zero2()],
                    )
                    .unwrap();
                f.call(
                    v64.clone(),
                    "@air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32",
                    vec![a.into(), b_k.into(), acc[k].into()],
                )
                .unwrap()
            })
            .collect();
        f.set_terminator(Terminator::Br(back));

        // back: increment.
        f.switch_to(back);
        let iv_next = f.add(iv, Operand::i32(1));
        f.set_terminator(Terminator::Br(hdr));

        // Wire carries through the helper: fragment counts must agree across
        // header phis, entry inits, and back-edge updates or it panics.
        let mut init_ops: Vec<Vec<Operand>> = vec![vec![Operand::i32(0)]];
        init_ops.extend(inits.iter().map(|&v| vec![v.into()]));
        let mut update_ops: Vec<Vec<Operand>> = vec![vec![iv_next.into()]];
        update_ops.extend(updates.iter().map(|&v| vec![v.into()]));
        f.wire_carries(&phis, entry, &init_ops, back, &update_ops);

        // exit: store NC fragments.
        f.switch_to(exit);
        for a_k in &acc {
            f.call(
                LlType::Void,
                "@air.simdgroup_matrix_8x8_store.v64f32.p1f32",
                vec![(*a_k).into(), pz.into(), stride(), stride(), zero2()],
            );
        }
        f.set_terminator(Terminator::Ret);

        let text = f.serialize(0).expect("serialize").0;
        verify_function(&text).unwrap_or_else(|e| panic!("text verifier rejected:\n{text}\n{e}"));
        assert_eq!(
            text.matches("phi <64 x float>").count(),
            NC,
            "expected {NC} distinct fragment-carry phis:\n{text}"
        );

        let decls = "\
declare <64 x float> @air.simdgroup_matrix_8x8_init_diag.v64f32.f32(float)\n\
declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p1f32(float addrspace(1)*, <2 x i64>, <2 x i64>, <2 x i64>)\n\
declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float>, <64 x float>, <64 x float>)\n\
declare void @air.simdgroup_matrix_8x8_store.v64f32.p1f32(<64 x float>, float addrspace(1)*, <2 x i64>, <2 x i64>, <2 x i64>)";
        metal_as("cfg_sgmma_gemv", decls, &text).unwrap();
    }

    #[test]
    #[should_panic(expected = "fragment count mismatch")]
    fn wire_carries_rejects_fragment_count_mismatch() {
        // The exact GEMV defect: header says 1 fragment, body produces 4. The
        // helper turns that into a loud panic instead of a dangling reference.
        let v64 = LlType::vec(64, LlType::Float);
        let mut f = FuncBuilder::new("mismatch");
        let entry = f.append_block("entry");
        let hdr = f.block("hdr");
        let back = f.block("back");
        f.switch_to(entry);
        f.set_terminator(Terminator::Br(hdr));
        f.switch_to(hdr);
        let phis = f.carry_phis(&[vec![v64.clone()]]); // header: 1 fragment
        f.set_terminator(Terminator::Br(back));
        f.switch_to(back);
        f.set_terminator(Terminator::Br(hdr));
        // Back-edge claims 4 fragments - mismatch.
        let init = vec![vec![Operand::undef(v64.clone())]];
        let update = vec![vec![
            Operand::undef(v64.clone()),
            Operand::undef(v64.clone()),
            Operand::undef(v64.clone()),
            Operand::undef(v64.clone()),
        ]];
        f.wire_carries(&phis, entry, &init, back, &update);
    }

    #[test]
    fn raw_instruction_participates_in_verification() {
        // A Raw instruction's value operands are still SSA-checked: using a value
        // from a non-dominating block must be caught.
        let mut f = FuncBuilder::new("raw_dom");
        let entry = f.append_block("entry");
        let a = f.block("a");
        let b = f.block("b");
        let merge = f.block("merge");

        f.switch_to(entry);
        let c = f.icmp("slt", Operand::i32(0), Operand::i32(1));
        f.set_terminator(Terminator::CondBr {
            cond: c.into(),
            then_: a,
            else_: b,
        });

        f.switch_to(a);
        let only_in_a = f.add(Operand::i32(1), Operand::i32(2));
        f.set_terminator(Terminator::Br(merge));

        f.switch_to(b);
        f.raw(Some(LlType::I32), "add i32 $0, 7", vec![only_in_a.into()]);
        f.set_terminator(Terminator::Br(merge));

        f.switch_to(merge);
        f.set_terminator(Terminator::Ret);

        let err = f.verify().unwrap_err();
        assert!(err.to_string().contains("not dominated"), "{err}");
    }

    #[test]
    fn raw_instruction_renders_and_assembles() {
        let mut f = FuncBuilder::new("raw_ok");
        let p = f.param("%_p0", LlType::device_ptr(LlType::I32));
        f.append_block("entry");
        let x = f.load(p, 4);
        let y = f.raw(
            Some(LlType::I32),
            "add i32 $0, $1",
            vec![x.into(), Operand::i32(5)],
        );
        f.store(y.unwrap(), p, 4);
        f.set_terminator(Terminator::Ret);
        let text = f.serialize(0).expect("serialize").0;
        assert!(text.contains("add i32 %v"), "raw not rendered:\n{text}");
        verify_function(&text).unwrap_or_else(|e| panic!("{text}\n{e}"));
        metal_as("cfg_raw_ok", "", &text).unwrap();
    }
}
