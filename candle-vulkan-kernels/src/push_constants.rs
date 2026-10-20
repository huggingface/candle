//! Push-constant parameter blocks for compute kernels.
//!
//! Kernels whose parameters are a handful of scalars receive them through
//! the pipeline's push constant block instead of a dedicated storage
//! buffer. This removes a per-dispatch parameter buffer upload (the
//! dominant `Buffer::from_iter` cost in profiling) and lets the parameter
//! mix carry the host types verbatim: `usize` as `uint64_t`, `f64` as
//! `double`.
//!
//! Every struct below must byte-for-byte match the `layout(push_constant)`
//! block of the corresponding Slang shader (std430 layout).

use bytemuck::AnyBitPattern;

/// `to_dtype` shader: `layout(push_constant) uint64_t g_Count;`
#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern)]
pub struct ToDtypeParams {
    /// Number of elements to convert.
    pub count: u64,
}

/// `affine` shader: `Params { uint64_t total; double mul; double add; }`.
#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern)]
pub struct AffineParams {
    /// Number of elements.
    pub total: u64,
    /// Multiplicand, full `f64` precision.
    pub mul: f64,
    /// Addend, full `f64` precision.
    pub add: f64,
}

/// `unary` shader: `Params { uint64_t count; double extra; }`.
#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern)]
pub struct UnaryParams {
    /// Number of elements.
    pub count: u64,
    /// Op-specific extra parameter (`pow` exponent, `elu` alpha),
    /// full `f64` precision.
    pub extra: f64,
}

/// `const_set` shader: `Params { uint64_t count; float value; }`.
#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern)]
pub struct ConstSetParams {
    /// Number of elements to fill.
    pub count: u64,
    /// Value to write.
    pub value: f32,
}

/// `rms_norm`/`layer_norm` shaders:
/// `Params { uint64_t n_rows; uint64_t n; double eps; }`.
#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern)]
pub struct NormParams {
    /// Number of rows.
    pub n_rows: u64,
    /// Row length (the normalized dim).
    pub n: u64,
    /// Epsilon, full `f64` precision.
    pub eps: f64,
}

/// `upsample` shader:
/// `Params { uint64_t planes; uint64_t d0..d3; double scale_h; double scale_w;
/// uint64_t align; }`. The `dN` fields hold the src/dst extents; the
/// scale fields are only used by the bilinear entry point.
#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern)]
pub struct UpsampleParams {
    /// Number of planes (batch x channels).
    pub planes: u64,
    /// First src/dst extent (nearest1d: src_sz).
    pub d0: u64,
    /// Second extent (nearest1d: dst_sz).
    pub d1: u64,
    /// Third extent (2d: dst_h).
    pub d2: u64,
    /// Fourth extent (2d: dst_w).
    pub d3: u64,
    /// Vertical scale factor (bilinear only), full `f64` precision.
    pub scale_h: f64,
    /// Horizontal scale factor (bilinear only), full `f64` precision.
    pub scale_w: f64,
    /// 1 if `align_corners` (bilinear only).
    pub align: u64,
}
