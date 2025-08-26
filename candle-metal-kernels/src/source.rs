pub const AFFINE: &str = include_str!("kernels/affine.metal");
pub const BINARY: &str = include_str!("kernels/binary.metal");
pub const CAST: &str = include_str!("kernels/cast.metal");
pub const CONV: &str = include_str!("kernels/conv.metal");
pub const FILL: &str = include_str!("kernels/fill.metal");
pub const INDEXING: &str = include_str!("kernels/indexing.metal");
pub const MLX_GEMM: &str = include_str!("kernels/mlx_gemm.metal");
pub const MLX_SORT: &str = include_str!("kernels/mlx_sort.metal");
pub const QUANTIZED: &str = include_str!("kernels/quantized.metal");
pub const RANDOM: &str = include_str!("kernels/random.metal");
pub const REDUCE: &str = include_str!("kernels/reduce.metal");
pub const SORT: &str = include_str!("kernels/sort.metal");
pub const TERNARY: &str = include_str!("kernels/ternary.metal");
pub const UNARY: &str = include_str!("kernels/unary.metal");
pub const SDPA: &str = include_str!("kernels/scaled_dot_product_attention.metal");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    Affine,
    Binary,
    Cast,
    Conv,
    Fill,
    Gemm,
    Indexing,
    MlxSort,
    Quantized,
    Random,
    Reduce,
    Sort,
    Ternary,
    Unary,
    Sdpa,
}
