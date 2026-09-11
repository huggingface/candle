//! Attributes of the `wax` dialect.
use pliron::derive::pliron_attr;

/// Binary ops
#[pliron_attr(name = "wax.binary_op", verifier = "succ", format)]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum BinaryAttr {
    AddF,
    AddI,
    SubF,
    SubI,
    MulF,
    MulI,
    DivF,
    DivI,
    RemI,
    AndI,
    OrI,
    XOrI,
    ShLI,
    ShRI,
    MaxF,
    MaxI,
    MinF,
    MinI,
    Atan2,
}

/// Unary ops
#[pliron_attr(name = "wax.unary_op", verifier = "succ", format)]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum UnaryAttr {
    AbsF,
    NegF,
    NegI,
    Sin,
    Cos,
    Exp,
    Exp2,
    Log,
    Log2,
    Sqrt,
    Rsqrt,
}

/// Casting. Note: `Bitcast` reinterprets.
#[pliron_attr(name = "wax.cast_op", verifier = "succ", format)]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum CastAttr {
    Bitcast,
    ExtI,
    TruncI,
    FToF,
    FToI,
    IToF,
}

/// Whether a comparison is between floats or integers.
#[pliron_attr(name = "wax.cmp_kind", verifier = "succ", format)]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum CmpKindAttr {
    Float,
    Int,
}

/// Impl from/to between a enum and `cutile_ir`'s opcodes.
macro_rules! from_opcode {
    ($ty:ident, $($op:ident),+ $(,)?) => {
        impl $ty {
            pub fn from_opcode(op: crate::Opcode) -> Option<Self> {
                match op {
                    $(crate::Opcode::$op => Some($ty::$op),)+
                    _ => None,
                }
            }
            pub fn to_opcode(self) -> crate::Opcode {
                match self {
                    $($ty::$op => crate::Opcode::$op,)+
                }
            }
        }
    };
}

from_opcode!(
    BinaryAttr, AddF, AddI, SubF, SubI, MulF, MulI, DivF, DivI, RemI, AndI, OrI, XOrI, ShLI, ShRI,
    MaxF, MaxI, MinF, MinI, Atan2,
);
from_opcode!(
    UnaryAttr, AbsF, NegF, NegI, Sin, Cos, Exp, Exp2, Log, Log2, Sqrt, Rsqrt,
);
from_opcode!(CastAttr, Bitcast, ExtI, TruncI, FToF, FToI, IToF);
