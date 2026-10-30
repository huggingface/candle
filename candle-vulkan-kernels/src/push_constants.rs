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

/// `const_fill_bytes` shader: `Params { uint64_t count; uint32_t pattern; }`.
#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern)]
pub struct ConstFillBytesParams {
    /// Number of bytes to fill.
    pub count: u64,
    /// 4-byte pattern stamped byte-wise.
    pub pattern: u32,
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

/// `kv_row_copy` shader: layout `Params { uint64_t heads; uint64_t head_dim;
/// uint64_t row; uint64_t stride; }`.
#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern)]
pub struct KvRowCopyParams {
    /// Number of KV heads.
    pub heads: u64,
    /// Dimension of each head.
    pub head_dim: u64,
    /// Row index the new k/v rows are written to.
    pub row: u64,
    /// Head stride of the pre-allocated caches (in elements).
    pub stride: u64,
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

/// `decode_attention` shader:
/// `Params { uint64_t num_heads; uint64_t num_kv_heads; uint64_t head_dim;
/// uint64_t kv_len; uint64_t kv_stride; float scale; }`.
#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern)]
pub struct DecodeAttentionParams {
    /// Number of query heads.
    pub num_heads: u64,
    /// Number of key/value heads.
    pub num_kv_heads: u64,
    /// Head dimension.
    pub head_dim: u64,
    /// Number of cached key/value positions.
    pub kv_len: u64,
    /// Element stride between consecutive kv heads (`>= kv_len *
    /// head_dim`; equal for a contiguous cache).
    pub kv_stride: u64,
    /// Attention scale (`1 / sqrt(head_dim)`).
    pub scale: f32,
}

/// `zero_fill` shader: `Params { uint64_t count; }`.
#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern)]
pub struct ZeroFillParams {
    /// Number of elements to zero.
    pub count: u64,
}

/// `dequant` shader: `Params { uint64_t num_blocks; }`.
#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern)]
pub struct DequantParams {
    /// Number of quantized blocks to dequantize.
    pub num_blocks: u64,
}

/// `sigmoid` shader: `Params { uint64_t count; }`.
#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern)]
pub struct SigmoidParams {
    /// Number of elements.
    pub count: u64,
}

/// `softmax` shader: `Params { uint64_t n_rows; uint64_t n; }`.
#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern)]
pub struct SoftmaxParams {
    /// Number of rows.
    pub n_rows: u64,
    /// Row length.
    pub n: u64,
}

/// `copy2d` shader:
/// `Params { uint64_t d1; uint64_t d2; uint64_t src_stride;
/// uint64_t dst_stride; uint64_t src_offset; uint64_t dst_offset; }`.
#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern)]
pub struct Copy2dParams {
    /// Number of rows.
    pub d1: u64,
    /// Row length (elements).
    pub d2: u64,
    /// Source row stride.
    pub src_stride: u64,
    /// Destination row stride.
    pub dst_stride: u64,
    /// Source first-element offset.
    pub src_offset: u64,
    /// Destination first-element offset.
    pub dst_offset: u64,
}

/// `pool2d` shader:
/// `Params { uint64_t planes; uint64_t h; uint64_t w; uint64_t k_h;
/// uint64_t k_w; uint64_t s_h; uint64_t s_w; uint64_t h_out;
/// uint64_t w_out; }`.
#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern)]
pub struct Pool2dParams {
    /// Number of planes (batch x channels).
    pub planes: u64,
    /// Input height.
    pub h: u64,
    /// Input width.
    pub w: u64,
    /// Kernel height.
    pub k_h: u64,
    /// Kernel width.
    pub k_w: u64,
    /// Stride height.
    pub s_h: u64,
    /// Stride width.
    pub s_w: u64,
    /// Output height.
    pub h_out: u64,
    /// Output width.
    pub w_out: u64,
}

/// `conv` shader:
/// `Params { uint64_t b; uint64_t c_in; uint64_t c_out; uint64_t k1;
/// uint64_t k2; uint64_t i1; uint64_t i2; uint64_t o1; uint64_t o2;
/// uint64_t padding; uint64_t stride; uint64_t dilation; }`.
#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern)]
pub struct ConvParams {
    /// Batch size.
    pub b: u64,
    /// Input channels.
    pub c_in: u64,
    /// Output channels.
    pub c_out: u64,
    /// Kernel extent 1 (kernel size for 1D, k_h for 2D).
    pub k1: u64,
    /// Kernel extent 2 (k_w for 2D, 0 for 1D).
    pub k2: u64,
    /// Input extent 1 (length for 1D, height for 2D).
    pub i1: u64,
    /// Input extent 2 (width for 2D, 0 for 1D).
    pub i2: u64,
    /// Output extent 1.
    pub o1: u64,
    /// Output extent 2.
    pub o2: u64,
    /// Padding.
    pub padding: u64,
    /// Stride.
    pub stride: u64,
    /// Dilation.
    pub dilation: u64,
}

/// `scatter` shader:
/// `Params { uint64_t dst_dim_len; uint64_t dst_right_len;
/// uint64_t ids_dim_len; uint64_t ids_right_len; uint64_t ids_left_len; }`.
#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern)]
pub struct ScatterParams {
    /// Length of the dst scatter dim.
    pub dst_dim_len: u64,
    /// Elements right of the scatter dim in dst.
    pub dst_right_len: u64,
    /// Length of the ids index dim.
    pub ids_dim_len: u64,
    /// Elements right of the index dim in ids.
    pub ids_right_len: u64,
    /// Elements left of the index dim in ids.
    pub ids_left_len: u64,
}

/// `rope` shader: `Params { uint64_t bh; uint64_t t; uint64_t d; }`.
#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern)]
pub struct RopeParams {
    /// Batch x heads.
    pub bh: u64,
    /// Sequence length.
    pub t: u64,
    /// Head dimension.
    pub d: u64,
}

/// `rotary_emb_int` shader:
/// `Params { uint64_t b; uint64_t h; uint64_t t; uint64_t d;
/// uint64_t unbatched; }`.
#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern)]
pub struct RotaryEmbIntParams {
    /// Batch size.
    pub b: u64,
    /// Number of heads.
    pub h: u64,
    /// Sequence length.
    pub t: u64,
    /// Head dimension.
    pub d: u64,
    /// 1 for the unbatched layout.
    pub unbatched: u64,
}

/// `index_add` shader:
/// `Params { uint64_t count; uint64_t post_dim; uint64_t nsrc;
/// uint64_t max_idx; }`.
#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern)]
pub struct IndexAddParams {
    /// Number of output elements.
    pub count: u64,
    /// Post-dim length.
    pub post_dim: u64,
    /// Number of sources.
    pub nsrc: u64,
    /// Maximum index.
    pub max_idx: u64,
}

/// `gather_idx` shader:
/// `Params { uint64_t count; uint64_t block_len; uint64_t emb_h;
/// uint64_t dst_offset; }`. The `main` entry reads only
/// `count`/`block_len`/`emb_h`; the gather-rows/index-select entries also
/// read `dst_offset`.
#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern)]
pub struct GatherIdxParams {
    /// Total number of output elements.
    pub count: u64,
    /// Block length.
    pub block_len: u64,
    /// Embedding row stride (output offset for the `main` entry).
    pub emb_h: u64,
    /// Destination offset.
    pub dst_offset: u64,
}

/// `reduce_ops` shader: `Params { uint64_t rows; uint64_t cols; }`.
#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern)]
pub struct ReduceParams {
    /// Number of rows.
    pub rows: u64,
    /// Number of columns.
    pub cols: u64,
}

/// `cmp` shader:
/// `Params { uint64_t count; uint64_t lhs_ndim; uint64_t rhs_ndim;
/// uint64_t lhs_dims[4]; uint64_t rhs_dims[4]; }`.
#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern)]
pub struct CmpParams {
    /// Number of output elements.
    pub count: u64,
    /// LHS ndim.
    pub lhs_ndim: u64,
    /// RHS ndim.
    pub rhs_ndim: u64,
    /// LHS dims (rightmost first).
    pub lhs_dims: [u64; 4],
    /// RHS dims (rightmost first).
    pub rhs_dims: [u64; 4],
}

/// `where` shader:
/// `Params { uint64_t count; uint64_t ndim; uint64_t pred_dims[4];
/// uint64_t true_dims[4]; uint64_t false_dims[4]; }`.
#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern)]
pub struct WhereParams {
    /// Number of output elements.
    pub count: u64,
    /// Number of dims.
    pub ndim: u64,
    /// Pred dims (rightmost first).
    pub pred_dims: [u64; 4],
    /// True dims (rightmost first).
    pub true_dims: [u64; 4],
    /// False dims (rightmost first).
    pub false_dims: [u64; 4],
}

/// `binary` shader:
/// `Params { uint64_t count; uint64_t lhs_ndim; uint64_t rhs_ndim;
/// uint64_t lhs_dims[8]; uint64_t rhs_dims[8]; uint64_t lhs_data_ndim;
/// uint64_t lhs_data_dims[8]; }`.
#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern)]
pub struct BinaryParams {
    /// Number of output elements.
    pub count: u64,
    /// LHS ndim.
    pub lhs_ndim: u64,
    /// RHS ndim.
    pub rhs_ndim: u64,
    /// LHS dims (rightmost first).
    pub lhs_dims: [u64; 8],
    /// RHS dims (rightmost first).
    pub rhs_dims: [u64; 8],
    /// LHS data ndim (storage dims after broadcasting).
    pub lhs_data_ndim: u64,
    /// LHS data dims (rightmost first).
    pub lhs_data_dims: [u64; 8],
}

/// `gemm` (tiled) shader:
/// `Params { uint64_t batch; uint64_t m; uint64_t n; uint64_t k;
/// uint64_t rhs_transposed; uint64_t sk; uint64_t sn; }`.
#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern)]
pub struct GemmParams {
    /// Number of batches.
    pub batch: u64,
    /// Output rows.
    pub m: u64,
    /// Output columns.
    pub n: u64,
    /// Reduction extent.
    pub k: u64,
    /// 1 if the RHS is stored transposed.
    pub rhs_transposed: u64,
    /// LHS stride over k.
    pub sk: u64,
    /// LHS stride over n (0 for the contiguous layout).
    pub sn: u64,
}

/// fused quant GEMM shaders: `Params { uint64_t m; uint64_t n; uint64_t k; }`.
#[repr(C)]
#[derive(Clone, Copy, AnyBitPattern)]
pub struct FusedQuantGemmParams {
    /// Output rows.
    pub m: u64,
    /// Output columns.
    pub n: u64,
    /// Reduction extent.
    pub k: u64,
}
