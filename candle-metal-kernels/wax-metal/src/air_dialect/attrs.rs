//! Attributes of the `air` dialect.

use pliron::derive::pliron_attr;

/// Binary arithmetic and bitwise operators, as in LLVM.
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum BinKind {
    Add,
    Sub,
    Mul,
    SDiv,
    UDiv,
    SRem,
    URem,
    Shl,
    LShr,
    AShr,
    And,
    Or,
    Xor,
    FAdd,
    FSub,
    FMul,
    FDiv,
    FRem,
}

/// Conversions. `Bitcast` is included because AIR uses it for SGMMA fragment reshapes.
#[pliron_attr(name = "air.cast_op", verifier = "succ", format)]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum CastOpAttr {
    Trunc,
    ZExt,
    SExt,
    FPTrunc,
    FPExt,
    FPToSI,
    FPToUI,
    SIToFP,
    UIToFP,
    Bitcast,
    IntToPtr,
    PtrToInt,
}
