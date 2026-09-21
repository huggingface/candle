//! `ToBitcode` for every op an `air` function can contain.

use super::{LowerCx, ToBitcode, callee_of, fcmp_code, icmp_code};
use crate::air_dialect::op_interfaces::{AirArrayLen, AirElemType};
use crate::air_dialect::ops::{AllocaOp, CastOp, GepArrayOp, GepOp, LoadOp, StoreOp};
use crate::bitcode::Ty;
use crate::bitcode::func::{Inst, Val};
use crate::bitcode::func::{binop, cast, unop};
use crate::lower::to_bitcode::Unsupported;
use pliron::derive::op_interface_impl;
use pliron::r#type::Typed;
use pliron_llvm::op_interfaces::AlignableOpInterface;
use pliron_llvm::ops as llvm;

macro_rules! binops {
    ($($op:ident => $code:expr),+ $(,)?) => {$(
        #[op_interface_impl]
        impl ToBitcode for llvm::$op {
            fn to_bitcode(&self, lx: &mut LowerCx<'_, '_>) -> Result<Option<Val>, Unsupported> {
                let lhs = lx.opd(0)?;
                let rhs = lx.opd(1)?;
                Ok(lx.body.push(Inst::BinOp { op: $code, lhs, rhs }))
            }
        }
    )+};
}

binops! {
    AddOp => binop::ADD,   FAddOp => binop::ADD,
    SubOp => binop::SUB,   FSubOp => binop::SUB,
    MulOp => binop::MUL,   FMulOp => binop::MUL,
    UDivOp => binop::UDIV, SDivOp => binop::SDIV, FDivOp => binop::SDIV,
    URemOp => binop::UREM, SRemOp => binop::SREM, FRemOp => binop::SREM,
    ShlOp => binop::SHL,   LShrOp => binop::LSHR, AShrOp => binop::ASHR,
    AndOp => binop::AND,   OrOp => binop::OR,     XorOp => binop::XOR,
}

macro_rules! casts {
    ($($op:ident => $code:expr),+ $(,)?) => {$(
        #[op_interface_impl]
        impl ToBitcode for llvm::$op {
            fn to_bitcode(&self, lx: &mut LowerCx<'_, '_>) -> Result<Option<Val>, Unsupported> {
                let val = lx.opd(0)?;
                Ok(lx.body.push(Inst::Cast { op: $code, val, to: lx.res_ty() }))
            }
        }
    )+};
}

casts! {
    TruncOp => cast::TRUNC,     ZExtOp => cast::ZEXT,     SExtOp => cast::SEXT,
    FPTruncOp => cast::FPTRUNC, FPExtOp => cast::FPEXT,
    FPToSIOp => cast::FPTOSI,   FPToUIOp => cast::FPTOUI,
    SIToFPOp => cast::SITOFP,   UIToFPOp => cast::UITOFP,
    BitcastOp => cast::BITCAST,
}

#[op_interface_impl]
impl ToBitcode for CastOp {
    fn to_bitcode(&self, lx: &mut LowerCx<'_, '_>) -> Result<Option<Val>, Unsupported> {
        // TODO: Error
        let kind = *self
            .get_attr_air_op(lx.ctx())
            .expect("air.cast has no operator");
        let op = match kind {
            crate::air_dialect::attrs::CastOpAttr::IntToPtr => cast::INTTOPTR,
            crate::air_dialect::attrs::CastOpAttr::PtrToInt => cast::PTRTOINT,
            _ => return Err(Unsupported::Opcode("air.cast is pointer-only".to_string())),
        };
        let val = lx.opd(0)?;
        Ok(lx.body.push(Inst::Cast {
            op,
            val,
            to: lx.res_ty(),
        }))
    }
}

#[op_interface_impl]
impl ToBitcode for llvm::FNegOp {
    fn to_bitcode(&self, lx: &mut LowerCx<'_, '_>) -> Result<Option<Val>, Unsupported> {
        let val = lx.opd(0)?;
        Ok(lx.body.push(Inst::UnOp {
            op: unop::FNEG,
            val,
        }))
    }
}

#[op_interface_impl]
impl ToBitcode for llvm::ICmpOp {
    fn to_bitcode(&self, lx: &mut LowerCx<'_, '_>) -> Result<Option<Val>, Unsupported> {
        let pred = icmp_code(self.predicate(lx.ctx()));
        let lhs = lx.opd(0)?;
        let rhs = lx.opd(1)?;
        Ok(lx.body.push(Inst::Cmp { pred, lhs, rhs }))
    }
}

#[op_interface_impl]
impl ToBitcode for llvm::FCmpOp {
    fn to_bitcode(&self, lx: &mut LowerCx<'_, '_>) -> Result<Option<Val>, Unsupported> {
        let pred = fcmp_code(self.predicate(lx.ctx()));
        let lhs = lx.opd(0)?;
        let rhs = lx.opd(1)?;
        Ok(lx.body.push(Inst::Cmp { pred, lhs, rhs }))
    }
}

#[op_interface_impl]
impl ToBitcode for llvm::SelectOp {
    fn to_bitcode(&self, lx: &mut LowerCx<'_, '_>) -> Result<Option<Val>, Unsupported> {
        let cond = lx.opd(0)?;
        let t = lx.opd(1)?;
        let f = lx.opd(2)?;
        Ok(lx.body.push(Inst::Select { cond, t, f }))
    }
}

#[op_interface_impl]
impl ToBitcode for llvm::ExtractElementOp {
    fn to_bitcode(&self, lx: &mut LowerCx<'_, '_>) -> Result<Option<Val>, Unsupported> {
        let vec = lx.opd(0)?;
        let idx = lx.opd(1)?;
        Ok(lx.body.push(Inst::ExtractElt { vec, idx }))
    }
}

#[op_interface_impl]
impl ToBitcode for llvm::InsertElementOp {
    fn to_bitcode(&self, lx: &mut LowerCx<'_, '_>) -> Result<Option<Val>, Unsupported> {
        let vec = lx.opd(0)?;
        let elem = lx.opd(1)?;
        let idx = lx.opd(2)?;
        Ok(lx.body.push(Inst::InsertElt { vec, elem, idx }))
    }
}

#[op_interface_impl]
impl ToBitcode for LoadOp {
    fn to_bitcode(&self, lx: &mut LowerCx<'_, '_>) -> Result<Option<Val>, Unsupported> {
        // TODO: Error
        let align = self.alignment(lx.ctx()).expect("op has no alignment");
        let ty = lx.res_ty();
        let ptr = lx.opd(0)?;
        Ok(lx.body.push(Inst::Load { ptr, ty, align }))
    }
}

#[op_interface_impl]
impl ToBitcode for StoreOp {
    fn to_bitcode(&self, lx: &mut LowerCx<'_, '_>) -> Result<Option<Val>, Unsupported> {
        // TODO: Error
        let align = self.alignment(lx.ctx()).expect("op has no alignment");
        let val = lx.opd(0)?;
        let ptr = lx.opd(1)?;
        Ok(lx.body.push(Inst::Store { ptr, val, align }))
    }
}

#[op_interface_impl]
impl ToBitcode for GepOp {
    fn to_bitcode(&self, lx: &mut LowerCx<'_, '_>) -> Result<Option<Val>, Unsupported> {
        let elem = lx.ty(self.elem_type(lx.ctx()));
        let ptr = lx.opd(0)?;
        let idx = lx.opd(1)?;
        Ok(lx.body.push(Inst::Gep {
            inbounds: false,
            elem,
            ptr,
            indices: vec![idx],
        }))
    }
}

#[op_interface_impl]
impl ToBitcode for GepArrayOp {
    fn to_bitcode(&self, lx: &mut LowerCx<'_, '_>) -> Result<Option<Val>, Unsupported> {
        // TODO: Error
        let n = self.array_len(lx.ctx()).expect("gep_array has no length");
        let elem = lx.ty(self.elem_type(lx.ctx()));
        let ptr = lx.opd(0)?;
        let zero = lx.body.iconst(Ty::Int(32), 0);
        let idx = lx.opd(1)?;
        Ok(lx.body.push(Inst::Gep {
            inbounds: true,
            elem: Ty::Array(n, Box::new(elem)),
            ptr,
            indices: vec![zero, idx],
        }))
    }
}

#[op_interface_impl]
impl ToBitcode for AllocaOp {
    fn to_bitcode(&self, lx: &mut LowerCx<'_, '_>) -> Result<Option<Val>, Unsupported> {
        // TODO: Error
        let align = self.alignment(lx.ctx()).expect("op has no alignment");
        let elem = lx.ty(self.elem_type(lx.ctx()));
        // One operand means a runtime length. Otherwise it is the static array length, if any.
        if lx.op.deref(lx.ctx()).get_num_operands() == 1 {
            let count = lx.opd(0)?;
            return Ok(lx.body.push(Inst::Alloca {
                ty: elem,
                count,
                align,
            }));
        }
        let ty = match self.array_len(lx.ctx()) {
            Some(n) => Ty::Array(n, Box::new(elem)),
            None => elem,
        };
        let count = lx.body.iconst(Ty::Int(32), 1);
        Ok(lx.body.push(Inst::Alloca { ty, count, align }))
    }
}

#[op_interface_impl]
impl ToBitcode for llvm::CallIntrinsicOp {
    fn to_bitcode(&self, lx: &mut LowerCx<'_, '_>) -> Result<Option<Val>, Unsupported> {
        let n_args = lx.op.deref(lx.ctx()).get_num_operands();
        let mut args = Vec::with_capacity(n_args);
        let mut params = Vec::with_capacity(n_args);
        for i in 0..n_args {
            let h = lx.op.deref(lx.ctx()).get_operand(i).get_type(lx.ctx());
            params.push(lx.ty(h));
            args.push(lx.opd(i)?);
        }
        // LLVM declares this op always has a result. That is not always the case for us, but we
        // can declare that it returns `Ty::Void` to get around the constraint.
        let ret = lx.res_ty();
        let returns_value = ret != Ty::Void;
        let fast = self
            .get_attr_llvm_intrinsic_fastmath_flags(lx.ctx())
            .is_some_and(|f| *f != pliron_llvm::attributes::FastmathFlagsAttr::default());
        let name = callee_of(self, lx.ctx());
        let idx = (lx.syms.callee)(&name).ok_or(Unsupported::UnknownCallee(name))?;
        Ok(lx.body.push(Inst::Call {
            fnty: Ty::Func {
                ret: Box::new(ret),
                params,
                vararg: false,
            },
            callee: Val::Func(idx),
            args,
            returns_value,
            fast,
        }))
    }
}

#[op_interface_impl]
impl ToBitcode for llvm::ReturnOp {
    fn to_bitcode(&self, lx: &mut LowerCx<'_, '_>) -> Result<Option<Val>, Unsupported> {
        Ok(lx.body.push(Inst::Ret(None)))
    }
}

#[op_interface_impl]
impl ToBitcode for llvm::UnreachableOp {
    fn to_bitcode(&self, lx: &mut LowerCx<'_, '_>) -> Result<Option<Val>, Unsupported> {
        Ok(lx.body.push(Inst::Unreachable))
    }
}

#[op_interface_impl]
impl ToBitcode for llvm::BrOp {
    fn to_bitcode(&self, lx: &mut LowerCx<'_, '_>) -> Result<Option<Val>, Unsupported> {
        // TODO: Error
        let succ = lx
            .op
            .deref(lx.ctx())
            .successors()
            .next()
            .expect("br has a successor");
        Ok(lx.body.push(Inst::Br(lx.block_idx(succ))))
    }
}

#[op_interface_impl]
impl ToBitcode for llvm::CondBrOp {
    fn to_bitcode(&self, lx: &mut LowerCx<'_, '_>) -> Result<Option<Val>, Unsupported> {
        let cond = lx.opd(0)?;
        let succs: Vec<_> = lx.op.deref(lx.ctx()).successors().collect();
        Ok(lx.body.push(Inst::CondBr {
            cond,
            then_bb: lx.block_idx(succs[0]),
            else_bb: lx.block_idx(succs[1]),
        }))
    }
}

macro_rules! const_like {
    ($($op:ty),+ $(,)?) => {$(
        #[op_interface_impl]
        impl ToBitcode for $op {
            fn to_bitcode(&self, lx: &mut LowerCx<'_, '_>) -> Result<Option<Val>, Unsupported> {
                lx.const_val().map(Some)
            }
        }
    )+};
}

const_like!(
    llvm::ConstantOp,
    llvm::UndefOp,
    llvm::ZeroOp,
    crate::air_dialect::ops::GlobalRefOp,
);
