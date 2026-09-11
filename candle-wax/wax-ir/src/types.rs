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
//! `ScalarType::External` - escape hatch for types not defined by Cuda Tile.
//! `bytecode_tag` - extract the tag of a type as bytecode.
//!
//! We verify the definitions here by checking against dialect spec (`tests/bytecode_tags.rs`).

use std::sync::Arc;

/// Marker for a dimension or stride whose size is not statically known.
/// Matches MLIR's `ShapedType::kDynamic` (`i64::MIN`); printed as `?` in text.
pub const DYNAMIC: i64 = i64::MIN;

// ---------------------------------------------------------------------------
// External scalar type trait
// ---------------------------------------------------------------------------

/// Trait for user-defined scalar types that live outside this crate.
///
/// Implement this to extend the IR with custom dtypes (e.g. fp8, microscaling
/// formats, custom quantized integers) without modifying this crate.
///
/// The implementation must supply storage and compute scalar types drawn from the built-in
/// [`ScalarType`] variants. That is the whole contract: a backend sees only those built-in
/// variants, so an external dtype needs no backend support of its own.
pub trait ScalarT: std::fmt::Debug + Send + Sync {
    /// Unique name for this type (used for display and equality).
    fn name(&self) -> &str;
    /// How values of this type are stored in memory (must be a built-in variant).
    fn storage_type(&self) -> ScalarType;
    /// How values of this type are represented in registers during computation.
    fn compute_type(&self) -> ScalarType;
    /// Size of one element in memory, in bytes.
    fn byte_width(&self) -> u32;
    /// Whether this type represents a floating-point value.
    fn is_float(&self) -> bool;
}

// ---------------------------------------------------------------------------
// Scalar types
// ---------------------------------------------------------------------------

/// Scalar (element) types — integers, floats, and user-defined external types.
#[derive(Debug, Clone)]
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
    /// Escape hatch for dtypes defined outside this crate.
    External(Arc<dyn ScalarT>),
}

impl PartialEq for ScalarType {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::External(a), Self::External(b)) => a.name() == b.name(),
            _ => std::mem::discriminant(self) == std::mem::discriminant(other),
        }
    }
}

impl Eq for ScalarType {}

impl std::hash::Hash for ScalarType {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        if let Self::External(t) = self {
            t.name().hash(state);
        }
    }
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
            Self::External(t) => t.byte_width() as usize,
        }
    }

    /// This type's FROZEN bytecode tag, or `None` for [`ScalarType::External`], which is
    /// ours and has no upstream encoding.
    ///
    /// Values are the spec's, from
    /// `cuda-tile/include/cuda_tile/Dialect/CudaTile/IR/BytecodeTypeOpcodes.td`
    /// ("Explicit Type Tag Assignments - FROZEN for backward compatibility"), NOT copied
    /// from anyone's implementation. `cutile-ir` has an equivalent `type_tag` returning a
    /// `TypeTag` from its `bytecode` module; we vendored `ir/` and not `bytecode/`, so that
    /// function could not come with it. Any cutile export needs these, so they live here
    /// with a test that pins them.
    pub fn bytecode_tag(&self) -> Option<u8> {
        Some(match self {
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
            Self::External(_) => return None,
        })
    }

    pub fn is_integer(&self) -> bool {
        match self {
            Self::I1 | Self::I4 | Self::I8 | Self::I16 | Self::I32 | Self::I64 => true,
            Self::External(t) => !t.is_float(),
            _ => false,
        }
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

// ---------------------------------------------------------------------------
// Type parser — equivalent of melior::ir::Type::parse(ctx, s)
// ---------------------------------------------------------------------------

impl Type {
    /// This type's FROZEN bytecode tag. A [`Type::Scalar`] defers to
    /// [`ScalarType::bytecode_tag`], so `None` means an external dtype with no upstream
    /// encoding. See that method for provenance.
    pub fn bytecode_tag(&self) -> Option<u8> {
        Some(match self {
            Self::Scalar(s) => return s.bytecode_tag(),
            Self::Pointer(_) => 12,
            Self::Tile(_) => 13,
            Self::TensorView(_) => 14,
            Self::PartitionView(_) => 15,
            Self::Func(_) => 16,
            Self::Token => 17,
            Self::GatherScatterView(_) => 20,
            Self::StridedView(_) => 21,
        })
    }

    /// Parse a CUDA Tile MLIR type string into a `Type`.
    ///
    /// This is the tile-ir equivalent of `melior::ir::Type::parse(ctx, s)`.
    /// It handles the full grammar of CUDA Tile dialect types:
    ///
    /// - `!cuda_tile.tile<[shape x]elem>` where elem is scalar or `!cuda_tile.ptr<scalar>`
    /// - `!cuda_tile.tensor_view<[shape x]scalar, strides=[strides]>`
    /// - `!cuda_tile.partition_view<tile=(dims), [padding_value=X,] tensor_view<...>[, dim_map=[...]]>`
    /// - `!cuda_tile.token`
    /// - Bare scalar names (e.g. `f32`, `i32`)
    ///
    /// Returns `None` if the string doesn't match any known type.
    pub fn parse(s: &str) -> Option<Type> {
        let s = s.trim();
        // Accept both `!cuda_tile.token` and shorthand `token`.
        if s == "!cuda_tile.token" || s == "token" {
            return Some(Type::Token);
        }
        // Prefixed forms.
        if let Some(inner) = strip_prefix_suffix(s, "!cuda_tile.tile<", ">") {
            return parse_tile(inner);
        }
        if let Some(inner) = strip_prefix_suffix(s, "!cuda_tile.tensor_view<", ">") {
            return parse_tensor_view(inner);
        }
        if let Some(inner) = strip_prefix_suffix(s, "!cuda_tile.partition_view<", ">") {
            return parse_partition_view(inner);
        }
        if let Some(inner) = strip_prefix_suffix(s, "!cuda_tile.gather_scatter_view<", ">") {
            return parse_gather_scatter_view(inner);
        }
        if let Some(inner) = strip_prefix_suffix(s, "!cuda_tile.strided_view<", ">") {
            return parse_strided_view(inner);
        }
        if let Some(inner) = strip_prefix_suffix(s, "!cuda_tile.ptr<", ">") {
            let pointee = parse_scalar(inner)?;
            return Some(Type::Pointer(PointerType { pointee }));
        }
        // Shorthand forms (no `!cuda_tile.` prefix).
        if let Some(inner) = strip_prefix_suffix(s, "tile<", ">") {
            return parse_tile(inner);
        }
        if let Some(inner) = strip_prefix_suffix(s, "tensor_view<", ">") {
            return parse_tensor_view(inner);
        }
        if let Some(inner) = strip_prefix_suffix(s, "partition_view<", ">") {
            return parse_partition_view(inner);
        }
        if let Some(inner) = strip_prefix_suffix(s, "gather_scatter_view<", ">") {
            return parse_gather_scatter_view(inner);
        }
        if let Some(inner) = strip_prefix_suffix(s, "strided_view<", ">") {
            return parse_strided_view(inner);
        }
        if let Some(inner) = strip_prefix_suffix(s, "ptr<", ">") {
            let pointee = parse_scalar(inner)?;
            return Some(Type::Pointer(PointerType { pointee }));
        }
        // Bare scalar name.
        parse_scalar(s).map(Type::Scalar)
    }
}

impl ScalarType {
    /// Parse a scalar type name string.
    pub fn parse(s: &str) -> Option<ScalarType> {
        parse_scalar(s)
    }
}

/// Strip a known prefix and the matching closing `>`, handling nested `<>`.
fn strip_prefix_suffix<'a>(s: &'a str, prefix: &str, _suffix: &str) -> Option<&'a str> {
    if !s.starts_with(prefix) {
        return None;
    }
    let after_prefix = &s[prefix.len()..];
    // Find the matching closing '>' by counting nesting depth.
    let mut depth = 1;
    for (i, c) in after_prefix.char_indices() {
        match c {
            '<' => depth += 1,
            '>' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&after_prefix[..i]);
                }
            }
            _ => {}
        }
    }
    // No matching close — try without nesting (just strip last char if it's '>').
    after_prefix.strip_suffix('>')
}

fn parse_scalar(s: &str) -> Option<ScalarType> {
    match s.trim() {
        "i1" => Some(ScalarType::I1),
        "i4" => Some(ScalarType::I4),
        "i8" => Some(ScalarType::I8),
        "i16" => Some(ScalarType::I16),
        "i32" => Some(ScalarType::I32),
        "i64" => Some(ScalarType::I64),
        "f16" => Some(ScalarType::F16),
        "bf16" => Some(ScalarType::BF16),
        "f32" => Some(ScalarType::F32),
        "tf32" => Some(ScalarType::TF32),
        "f64" => Some(ScalarType::F64),
        "f8E4M3FN" | "f8e4m3fn" => Some(ScalarType::F8E4M3FN),
        "f8E5M2" | "f8e5m2" => Some(ScalarType::F8E5M2),
        "f8E8M0FNU" | "f8e8m0fnu" => Some(ScalarType::F8E8M0FNU),
        "f4E2M1FN" | "f4e2m1fn" => Some(ScalarType::F4E2M1FN),
        // Rust-facing names used by the compiler.
        "bool" => Some(ScalarType::I1),
        _ => None,
    }
}

/// Parse a dim string: integer or `?` for DYNAMIC.
fn parse_dim(s: &str) -> i64 {
    let s = s.trim();
    if s == "?" {
        DYNAMIC
    } else {
        s.parse::<i64>().unwrap_or(DYNAMIC)
    }
}

/// Parse `[shape x]elem` where shape is `dim[xdim]*` and elem is a scalar or `!cuda_tile.ptr<scalar>`.
fn parse_tile(inner: &str) -> Option<Type> {
    // Check for pointer element: `!cuda_tile.ptr<scalar>` or `ptr<scalar>`
    if let Some(ptr_start) = inner.find("ptr<") {
        let before = inner[..ptr_start].trim().trim_end_matches("!cuda_tile.");
        let shape: Vec<i64> = if before.is_empty() {
            vec![]
        } else {
            before
                .trim_end_matches('x')
                .split('x')
                .map(parse_dim)
                .collect()
        };
        let ptr_inner_start = ptr_start + "ptr<".len();
        let ptr_inner_end = inner[ptr_inner_start..].find('>')?;
        let pointee = parse_scalar(&inner[ptr_inner_start..ptr_inner_start + ptr_inner_end])?;
        return Some(Type::Tile(TileType {
            shape,
            element_type: TileElementType::Pointer(Box::new(PointerType { pointee })),
        }));
    }

    // Split on 'x' to get shape dims and trailing element type.
    // e.g. "128xf32" → ["128", "f32"], "f32" → ["f32"], "4x128xf32" → ["4", "128", "f32"]
    let parts: Vec<&str> = inner.split('x').collect();
    if parts.is_empty() {
        return None;
    }
    let elem_str = parts.last()?;
    let scalar = parse_scalar(elem_str)?;
    let shape: Vec<i64> = parts[..parts.len() - 1]
        .iter()
        .map(|p| parse_dim(p))
        .collect();
    Some(Type::Tile(TileType {
        shape,
        element_type: TileElementType::Scalar(scalar),
    }))
}

/// Parse `[shape x]scalar, strides=[strides]`.
fn parse_tensor_view(inner: &str) -> Option<Type> {
    // Split at ", strides=" to separate shape+elem from strides.
    let (shape_elem, strides_part) = if let Some(pos) = inner.find(", strides=") {
        (&inner[..pos], Some(&inner[pos + ", strides=".len()..]))
    } else if let Some(pos) = inner.find(",strides=") {
        (&inner[..pos], Some(&inner[pos + ",strides=".len()..]))
    } else {
        (inner, None)
    };

    let parts: Vec<&str> = shape_elem.split('x').collect();
    if parts.is_empty() {
        return None;
    }
    let elem_str = parts.last()?;
    let scalar = parse_scalar(elem_str)?;
    let shape: Vec<i64> = parts[..parts.len() - 1]
        .iter()
        .map(|p| parse_dim(p))
        .collect();

    let strides = if let Some(sp) = strides_part {
        let sp = sp.trim_start_matches('[').trim_end_matches(']');
        sp.split(',').map(parse_dim).collect()
    } else {
        vec![DYNAMIC; shape.len()]
    };

    Some(Type::TensorView(TensorViewType {
        element_type: scalar,
        shape,
        strides,
    }))
}

/// Parse `tile=(dims), [padding_value=X,] tensor_view<...>[, dim_map=[...]]`.
fn parse_partition_view(inner: &str) -> Option<Type> {
    // Extract tile shape from "tile=(d1xd2)" or "tile=(d1, d2)" or "tile=(d1)".
    let tile_start = inner.find("tile=(")?;
    let dims_start = tile_start + "tile=(".len();
    let dims_end = inner[dims_start..].find(')')? + dims_start;
    let dims_str = &inner[dims_start..dims_end];
    let tile_shape: Vec<i32> = if dims_str.contains('x') {
        dims_str
            .split('x')
            .map(|s| s.trim().parse::<i32>().unwrap_or(1))
            .collect()
    } else {
        dims_str
            .split(',')
            .map(|s| s.trim().parse::<i32>().unwrap_or(1))
            .collect()
    };

    // Extract padding_value if present.
    let padding_value = if inner.contains("padding_value") {
        if inner.contains("zero") && !inner.contains("neg_zero") {
            Some(PaddingValue::Zero)
        } else if inner.contains("neg_zero") {
            Some(PaddingValue::NegZero)
        } else if inner.contains("nan") {
            Some(PaddingValue::Nan)
        } else if inner.contains("pos_inf") {
            Some(PaddingValue::PosInf)
        } else if inner.contains("neg_inf") {
            Some(PaddingValue::NegInf)
        } else {
            None
        }
    } else {
        None
    };

    // Extract the tensor_view — may be "tensor_view=!cuda_tile.tensor_view<...>"
    // or just "tensor_view<...>" or "!cuda_tile.tensor_view<...>".
    let tv_search = inner;
    let tv_prefix_start = tv_search
        .find("tensor_view=!cuda_tile.tensor_view<")
        .map(|p| (p, "tensor_view=!cuda_tile.tensor_view<"))
        .or_else(|| {
            tv_search
                .find("!cuda_tile.tensor_view<")
                .map(|p| (p, "!cuda_tile.tensor_view<"))
        })
        .or_else(|| {
            // "tensor_view<" without the "tensor_view=" or "!cuda_tile." prefix
            // but NOT matching the "tile=" prefix that also appears
            let remaining = &tv_search[dims_end + 1..]; // after the tile=(...) part
            remaining
                .find("tensor_view<")
                .map(|p| (p + dims_end + 1, "tensor_view<"))
        })?;
    let (tv_pos, tv_prefix) = tv_prefix_start;
    let tv_inner_start = tv_pos + tv_prefix.len();

    // Find matching closing '>' for the tensor_view.
    let mut depth = 1;
    let mut tv_inner_end = tv_inner_start;
    for (i, c) in inner[tv_inner_start..].char_indices() {
        match c {
            '<' => depth += 1,
            '>' => {
                depth -= 1;
                if depth == 0 {
                    tv_inner_end = tv_inner_start + i;
                    break;
                }
            }
            _ => {}
        }
    }
    let tv_inner = &inner[tv_inner_start..tv_inner_end];
    let tv_type = parse_tensor_view(tv_inner)?;
    let Type::TensorView(tv) = tv_type else {
        return None;
    };

    // Extract dim_map if present.
    let dim_map = if let Some(dm_start) = inner.find("dim_map=[") {
        let dm_inner_start = dm_start + "dim_map=[".len();
        let dm_end = inner[dm_inner_start..].find(']')? + dm_inner_start;
        inner[dm_inner_start..dm_end]
            .split(',')
            .map(|s| s.trim().parse::<i32>().unwrap_or(0))
            .collect()
    } else {
        (0..tile_shape.len() as i32).collect()
    };

    Some(Type::PartitionView(PartitionViewType {
        tile_shape,
        tensor_view: tv,
        dim_map,
        padding_value,
    }))
}

fn parse_view_tile_shape(inner: &str) -> Option<(Vec<i32>, usize)> {
    let tile_start = inner.find("tile=(")?;
    let dims_start = tile_start + "tile=(".len();
    let dims_end = inner[dims_start..].find(')')? + dims_start;
    let dims_str = &inner[dims_start..dims_end];
    let tile_shape: Vec<i32> = if dims_str.contains('x') {
        dims_str
            .split('x')
            .map(|s| s.trim().parse::<i32>().unwrap_or(1))
            .collect()
    } else {
        dims_str
            .split(',')
            .map(|s| s.trim().parse::<i32>().unwrap_or(1))
            .collect()
    };
    Some((tile_shape, dims_end))
}

fn parse_padding_value(inner: &str) -> Option<PaddingValue> {
    if inner.contains("padding_value") {
        if inner.contains("zero") && !inner.contains("neg_zero") {
            Some(PaddingValue::Zero)
        } else if inner.contains("neg_zero") {
            Some(PaddingValue::NegZero)
        } else if inner.contains("nan") {
            Some(PaddingValue::Nan)
        } else if inner.contains("pos_inf") {
            Some(PaddingValue::PosInf)
        } else if inner.contains("neg_inf") {
            Some(PaddingValue::NegInf)
        } else {
            None
        }
    } else {
        None
    }
}

fn parse_nested_tensor_view(inner: &str, after_pos: usize) -> Option<TensorViewType> {
    let tv_search = inner;
    let tv_prefix_start = tv_search
        .find("tensor_view=!cuda_tile.tensor_view<")
        .map(|p| (p, "tensor_view=!cuda_tile.tensor_view<"))
        .or_else(|| {
            tv_search
                .find("!cuda_tile.tensor_view<")
                .map(|p| (p, "!cuda_tile.tensor_view<"))
        })
        .or_else(|| {
            let remaining = &tv_search[after_pos..];
            remaining
                .find("tensor_view<")
                .map(|p| (p + after_pos, "tensor_view<"))
        })?;
    let (tv_pos, tv_prefix) = tv_prefix_start;
    let tv_inner_start = tv_pos + tv_prefix.len();

    let mut depth = 1;
    let mut tv_inner_end = tv_inner_start;
    for (i, c) in inner[tv_inner_start..].char_indices() {
        match c {
            '<' => depth += 1,
            '>' => {
                depth -= 1;
                if depth == 0 {
                    tv_inner_end = tv_inner_start + i;
                    break;
                }
            }
            _ => {}
        }
    }
    let tv_inner = &inner[tv_inner_start..tv_inner_end];
    let tv_type = parse_tensor_view(tv_inner)?;
    let Type::TensorView(tv) = tv_type else {
        return None;
    };
    Some(tv)
}

fn parse_i32_list_after(inner: &str, prefix: &str) -> Option<Vec<i32>> {
    let start = inner.find(prefix)? + prefix.len();
    let end = inner[start..].find(']')? + start;
    Some(
        inner[start..end]
            .split(',')
            .map(|s| s.trim().parse::<i32>().unwrap_or(0))
            .collect(),
    )
}

fn parse_gather_scatter_view(inner: &str) -> Option<Type> {
    let (tile_shape, dims_end) = parse_view_tile_shape(inner)?;
    let padding_value = parse_padding_value(inner);
    let tensor_view = parse_nested_tensor_view(inner, dims_end + 1)?;
    let sparse_dim = if let Some(start) = inner.find("sparse_dim=") {
        let value_start = start + "sparse_dim=".len();
        let value = inner[value_start..]
            .split([',', '>'])
            .next()
            .unwrap_or("0")
            .trim();
        value.parse::<i32>().unwrap_or(0)
    } else {
        0
    };

    Some(Type::GatherScatterView(GatherScatterViewType {
        tile_shape,
        tensor_view,
        sparse_dim,
        padding_value,
    }))
}

fn parse_strided_view(inner: &str) -> Option<Type> {
    let (tile_shape, dims_end) = parse_view_tile_shape(inner)?;
    let padding_value = parse_padding_value(inner);
    let traversal_strides = parse_i32_list_after(inner, "traversal_strides=[")
        .unwrap_or_else(|| vec![1; tile_shape.len()]);
    let tensor_view = parse_nested_tensor_view(inner, dims_end + 1)?;
    let dim_map = parse_i32_list_after(inner, "dim_map=[")
        .unwrap_or_else(|| (0..tile_shape.len() as i32).collect());

    Some(Type::StridedView(StridedViewType {
        tile_shape,
        traversal_strides,
        tensor_view,
        dim_map,
        padding_value,
    }))
}
