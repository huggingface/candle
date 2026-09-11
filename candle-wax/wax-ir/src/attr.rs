/*
 * Derived from cutile-ir:
 *   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 *   SPDX-License-Identifier: Apache-2.0
 */
//! Tile IR attribute types.
//!
//! Mirrors the Cuda Tile dialect attributes defined in `AttrDefs.td`.
//!
//! Wax additions:
//! `FloatBits` and friends have been added so that `Eq`/`Hash` can be derived which makes
//! the IR compatible with pliron.
//! `Attribute::bytecode_tag` - extracts the bytecode tag of an attribute.
//!
//! The attributes defined here are verified against the CudaTile MLIR spec via the `bytecode_tags` tests.

use super::types::{ScalarType, Type};

// ---------------------------------------------------------------------------
// Enum attributes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Signedness {
    Unsigned = 0,
    Signed = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum IntegerOverflow {
    None = 0,
    NoSignedWrap = 1,
    NoUnsignedWrap = 2,
    NoWrap = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RoundingMode {
    NearestEven = 0,
    Zero = 1,
    NegativeInf = 2,
    PositiveInf = 3,
    Approx = 4,
    Full = 5,
    NearestIntToZero = 6,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ComparisonOrdering {
    Unordered = 0,
    Ordered = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ComparisonPredicate {
    Equal = 0,
    NotEqual = 1,
    LessThan = 2,
    LessThanOrEqual = 3,
    GreaterThan = 4,
    GreaterThanOrEqual = 5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AtomicRMWMode {
    And = 0,
    Or = 1,
    Xor = 2,
    Add = 3,
    AddF = 4,
    Max = 5,
    Min = 6,
    UMax = 7,
    UMin = 8,
    Xchg = 9,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MemoryScope {
    TileBlock = 0,
    Device = 1,
    System = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MemoryOrdering {
    Weak = 0,
    Relaxed = 1,
    Acquire = 2,
    Release = 3,
    AcqRel = 4,
}

// ---------------------------------------------------------------------------
// Structured attributes
// ---------------------------------------------------------------------------

/// `div_by` assumption predicate.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DivBy {
    pub divisor: u64,
    pub every: Option<i64>,
    pub along: Option<i64>,
}

/// `same_elements` assumption predicate.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SameElements {
    pub values: Vec<i64>,
}

/// `bounded` assumption predicate.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Bounded {
    pub lb: Option<i64>,
    pub ub: Option<i64>,
}

/// A double stored as its bits.
///
/// So `Attribute` can derive `Eq` and `Hash`, which is needed by pliron.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FloatBits(u64);

impl FloatBits {
    pub fn new(v: f64) -> Self {
        FloatBits(v.to_bits())
    }
    pub fn get(self) -> f64 {
        f64::from_bits(self.0)
    }
    pub fn bits(self) -> u64 {
        self.0
    }
}

/// Per-architecture optimization hints dictionary.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OptimizationHints {
    pub entries: Vec<(String, Vec<(String, Attribute)>)>,
}

// ---------------------------------------------------------------------------
// Unified attribute enum
// ---------------------------------------------------------------------------

/// An IR attribute value.
///
/// `Integer` and `Float` carry their scalar type, matching MLIR's
/// `IntegerAttr` (value + IntegerType) and `FloatAttr` (value + FloatType).
///
/// `Eq` and `Hash` hold across every variant, which is what lets an attribute be stored in
/// pliron directly instead of through a mirrored copy.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Attribute {
    Integer(i64, Type),
    Float(FloatBits, Type),
    Bool(bool),
    Type(Type),
    String(String),
    Array(Vec<Attribute>),
    DenseElements(DenseElements),
    /// Dense array of i32 values, matching MLIR's `DenseI32ArrayAttr`; the
    /// encoding of op attributes declared with that type (`permutation`).
    ///
    /// Not for `operandSegmentSizes`: the bytecode writer reads the operand
    /// group sizes only from an [`Attribute::Array`] of [`Attribute::Integer`]s
    /// (see `operand_segment_sizes` in `bytecode/op_writer.rs`), and the
    /// attribute itself is never serialized — it only drives how the
    /// operands are grouped. A `DenseI32Array` there is silently treated as
    /// "one operand per group".
    DenseI32Array(Vec<i32>),
    DivBy(DivBy),
    SameElements(SameElements),
    Dictionary(Vec<(String, Attribute)>),
    OptimizationHints(OptimizationHints),
    Bounded(Bounded),
}

impl Attribute {
    pub fn bytecode_tag(&self) -> Option<u8> {
        Some(match self {
            Self::Integer(..) => 1,
            Self::Float(..) => 2,
            Self::Bool(..) => 3,
            Self::Type(..) => 4,
            Self::String(..) => 5,
            Self::Array(..) => 6,
            Self::DenseElements(..) => 7,
            Self::DivBy(..) => 8,
            Self::SameElements(..) => 9,
            Self::Dictionary(..) => 10,
            Self::OptimizationHints(..) => 11,
            Self::Bounded(..) => 12,
            // MLIR builtin has no tag.
            Self::DenseI32Array(..) => return None,
        })
    }
    /// Create a typed integer attribute. Shorthand for `Attribute::Integer(v, Type::Scalar(ty))`.
    pub fn int(v: i64, ty: ScalarType) -> Self {
        Attribute::Integer(v, Type::Scalar(ty))
    }
    /// Create an i32 integer attribute (the most common case for enum-valued attrs).
    pub fn i32(v: i64) -> Self {
        Attribute::Integer(v, Type::Scalar(ScalarType::I32))
    }
    /// Create a typed float attribute. Shorthand for `Attribute::Float(v, Type::Scalar(ty))`.
    pub fn float(v: f64, ty: ScalarType) -> Self {
        Attribute::Float(FloatBits::new(v), Type::Scalar(ty))
    }
}

/// Dense element data for constant tensors.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DenseElements {
    pub element_type: Type,
    pub shape: Vec<i64>,
    pub data: Vec<u8>,
}
