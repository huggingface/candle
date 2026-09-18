//! Attributes of the `air` dialect.
//!
//! The operator of an arithmetic/cast/compare op is an ENUM here, where `cfg.rs` carries
//! a `&'static str`. Same information, but the emitter's match becomes exhaustive: a new
//! operator cannot reach the bitcode writer as an unrecognised string.

use pliron::derive::pliron_attr;

/// Binary arithmetic and bitwise operators, spelled as in LLVM.
#[pliron_attr(name = "air.bin_op", verifier = "succ", format)]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum BinOpAttr {
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

#[pliron_attr(name = "air.icmp_pred", verifier = "succ", format)]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum ICmpPredAttr {
    Eq,
    Ne,
    Slt,
    Sle,
    Sgt,
    Sge,
    Ult,
    Ule,
    Ugt,
    Uge,
}

#[pliron_attr(name = "air.fcmp_pred", verifier = "succ", format)]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum FCmpPredAttr {
    Oeq,
    One,
    Olt,
    Ole,
    Ogt,
    Oge,
    Ueq,
    Une,
    Ult,
    Ule,
    Ugt,
    Uge,
    Ord,
    Uno,
}

macro_rules! parse_attr {
    ($ty:ident, $fname:ident, $($text:literal => $variant:ident),+ $(,)?) => {
        impl $ty {
            /// Parse the LLVM spelling `cfg.rs` stores as a `&'static str`.
            pub fn $fname(s: &str) -> Option<Self> {
                match s {
                    $($text => Some($ty::$variant),)+
                    _ => None,
                }
            }
            /// The LLVM spelling, for the text emitter and for error messages.
            pub fn as_str(self) -> &'static str {
                match self {
                    $($ty::$variant => $text,)+
                }
            }
        }
    };
}

parse_attr!(BinOpAttr, from_llvm,
    "add" => Add, "sub" => Sub, "mul" => Mul, "sdiv" => SDiv, "udiv" => UDiv,
    "srem" => SRem, "urem" => URem, "shl" => Shl, "lshr" => LShr, "ashr" => AShr,
    "and" => And, "or" => Or, "xor" => Xor, "fadd" => FAdd, "fsub" => FSub,
    "fmul" => FMul, "fdiv" => FDiv, "frem" => FRem,
);
parse_attr!(CastOpAttr, from_llvm,
    "trunc" => Trunc, "zext" => ZExt, "sext" => SExt, "fptrunc" => FPTrunc,
    "fpext" => FPExt, "fptosi" => FPToSI, "fptoui" => FPToUI, "sitofp" => SIToFP,
    "uitofp" => UIToFP, "bitcast" => Bitcast, "inttoptr" => IntToPtr,
    "ptrtoint" => PtrToInt,
);
parse_attr!(ICmpPredAttr, from_llvm,
    "eq" => Eq, "ne" => Ne, "slt" => Slt, "sle" => Sle, "sgt" => Sgt, "sge" => Sge,
    "ult" => Ult, "ule" => Ule, "ugt" => Ugt, "uge" => Uge,
);
parse_attr!(FCmpPredAttr, from_llvm,
    "oeq" => Oeq, "one" => One, "olt" => Olt, "ole" => Ole, "ogt" => Ogt, "oge" => Oge,
    "ueq" => Ueq, "une" => Une, "ult" => Ult, "ule" => Ule, "ugt" => Ugt, "uge" => Uge,
    "ord" => Ord, "uno" => Uno,
);

/// The raw element bits of a dense vector constant, e.g. an SGMMA `<i64 128, i64 8>`
/// stride pair. Stored as bits so integer and float vectors share one form, matching
/// LLVM's `ConstantDataSequential`.
#[pliron_attr(
    name = "air.dense_bits",
    verifier = "succ",
    format = "`[` vec($0, CharSpace(`,`)) `]`"
)]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct DenseBitsAttr(pub Vec<u64>);
