/*
 * Derived from cutile-ir:
 *   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 *   SPDX-License-Identifier: Apache-2.0
 */
//! Tile IR type system.
//!
//! Mirrors the CUDA Tile dialect types defined in `Types.td`.
//!
//! Wax additions:
//! `bytecode_tag` - extract the tag of a type as bytecode.
//!
//! We verify the definitions here by checking against dialect spec (`tests/bytecode_tags.rs`).

/// Marker for a dimension or stride whose size is not statically known.
/// Matches MLIR's `ShapedType::kDynamic` (`i64::MIN`); printed as `?` in text.
pub const DYNAMIC: i64 = i64::MIN;

// ---------------------------------------------------------------------------
// Scalar types
// ---------------------------------------------------------------------------

/// Scalar (element) types — integers, floats, and user-defined external types.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ScalarType {
    I1,
    I4,
    I8,
    I16,
    I32,
    I64,
    F16,
    BF16,
    F32,
    TF32,
    F64,
    F8E4M3FN,
    F8E5M2,
    F8E8M0FNU,
    F4E2M1FN,
}

impl ScalarType {
    /// Byte width of this scalar type (I1 rounds up to 1 byte).
    pub fn byte_width(&self) -> usize {
        match self {
            Self::I1
            | Self::I4
            | Self::I8
            | Self::F8E4M3FN
            | Self::F8E5M2
            | Self::F8E8M0FNU
            | Self::F4E2M1FN => 1,
            Self::I16 | Self::F16 | Self::BF16 => 2,
            Self::I32 | Self::F32 | Self::TF32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }

    /// This type's frozen bytecode tag.
    ///
    /// Values from cuda-tile spec:
    /// `cuda-tile/include/cuda_tile/Dialect/CudaTile/IR/BytecodeTypeOpcodes.td` (the "Explicit
    /// Type Tag Assignments", frozen for backward compatibility).
    pub fn bytecode_tag(&self) -> u8 {
        match self {
            Self::I1 => 0,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 3,
            Self::I64 => 4,
            Self::F16 => 5,
            Self::BF16 => 6,
            Self::F32 => 7,
            Self::TF32 => 8,
            Self::F64 => 9,
            Self::F8E4M3FN => 10,
            Self::F8E5M2 => 11,
            Self::F8E8M0FNU => 18,
            Self::F4E2M1FN => 19,
            Self::I4 => 22,
        }
    }

    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            Self::I1 | Self::I4 | Self::I8 | Self::I16 | Self::I32 | Self::I64
        )
    }

    pub fn is_float(&self) -> bool {
        !self.is_integer()
    }
}

// ---------------------------------------------------------------------------
// Compound types
// ---------------------------------------------------------------------------

/// An element type that can appear inside a Tile (scalar or pointer).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TileElementType {
    Scalar(ScalarType),
    Pointer(Box<PointerType>),
}

/// Pointer to a scalar value in global device memory.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PointerType {
    pub pointee: ScalarType,
}

/// A statically-shaped tile of elements.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TileType {
    pub shape: Vec<i64>,
    pub element_type: TileElementType,
}

/// A reference to a tensor in global memory with shape and strides.
/// Use [`DYNAMIC`] for dimensions/strides that are not statically known.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorViewType {
    pub element_type: ScalarType,
    pub shape: Vec<i64>,
    pub strides: Vec<i64>,
}

/// A view into a tensor where tiles are laid out in a grid pattern.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PartitionViewType {
    pub tile_shape: Vec<i32>,
    pub tensor_view: TensorViewType,
    pub dim_map: Vec<i32>,
    pub padding_value: Option<PaddingValue>,
}

/// A view that uses a 1D tensor of sparse indices along one tensor dimension.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GatherScatterViewType {
    pub tile_shape: Vec<i32>,
    pub tensor_view: TensorViewType,
    pub sparse_dim: i32,
    pub padding_value: Option<PaddingValue>,
}

/// A view that traverses a tensor with configurable per-dimension strides.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StridedViewType {
    pub tile_shape: Vec<i32>,
    pub traversal_strides: Vec<i32>,
    pub tensor_view: TensorViewType,
    pub dim_map: Vec<i32>,
    pub padding_value: Option<PaddingValue>,
}

/// Padding value for out-of-bounds accesses in a partition view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PaddingValue {
    Zero = 0,
    NegZero = 1,
    Nan = 2,
    PosInf = 3,
    NegInf = 4,
}

/// Function signature type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FuncType {
    pub inputs: Vec<Type>,
    pub results: Vec<Type>,
}

/// The unified type enum — every value in the IR has one of these types.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Scalar(ScalarType),
    Pointer(PointerType),
    Tile(TileType),
    TensorView(TensorViewType),
    PartitionView(PartitionViewType),
    GatherScatterView(GatherScatterViewType),
    StridedView(StridedViewType),
    Func(FuncType),
    Token,
}

impl From<ScalarType> for Type {
    fn from(s: ScalarType) -> Self {
        Self::Scalar(s)
    }
}

impl From<PointerType> for Type {
    fn from(p: PointerType) -> Self {
        Self::Pointer(p)
    }
}

impl From<TileType> for Type {
    fn from(t: TileType) -> Self {
        Self::Tile(t)
    }
}

impl From<TensorViewType> for Type {
    fn from(t: TensorViewType) -> Self {
        Self::TensorView(t)
    }
}

impl From<PartitionViewType> for Type {
    fn from(p: PartitionViewType) -> Self {
        Self::PartitionView(p)
    }
}

impl From<GatherScatterViewType> for Type {
    fn from(g: GatherScatterViewType) -> Self {
        Self::GatherScatterView(g)
    }
}

impl From<StridedViewType> for Type {
    fn from(s: StridedViewType) -> Self {
        Self::StridedView(s)
    }
}

impl From<FuncType> for Type {
    fn from(f: FuncType) -> Self {
        Self::Func(f)
    }
}

impl Type {
    /// This type's frozen bytecode tag.
    pub fn bytecode_tag(&self) -> u8 {
        match self {
            Self::Scalar(s) => s.bytecode_tag(),
            Self::Pointer(_) => 12,
            Self::Tile(_) => 13,
            Self::TensorView(_) => 14,
            Self::PartitionView(_) => 15,
            Self::Func(_) => 16,
            Self::Token => 17,
            Self::GatherScatterView(_) => 20,
            Self::StridedView(_) => 21,
        }
    }
}
