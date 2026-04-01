// ROCm/HIP kernel source files
// These are embedded as strings and compiled at runtime

pub const BINARY: &str = include_str!("kernels/binary.hip");
pub const UNARY: &str = include_str!("kernels/unary.hip");
pub const AFFINE: &str = include_str!("kernels/affine.hip");
pub const FILL: &str = include_str!("kernels/fill.hip");
pub const CAST: &str = include_str!("kernels/cast.hip");
pub const CMP: &str = include_str!("kernels/cmp.hip");
pub const REDUCE: &str = include_str!("kernels/reduce.hip");
pub const TERNARY: &str = include_str!("kernels/ternary.hip");
pub const INDEXING: &str = include_str!("kernels/indexing.hip");
pub const CONV: &str = include_str!("kernels/conv.hip");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    Binary,
    Unary,
    Affine,
    Fill,
    Cast,
    Cmp,
    Reduce,
    Ternary,
    Indexing,
    Conv,
}

impl Source {
    pub fn code(&self) -> &'static str {
        match self {
            Source::Binary => BINARY,
            Source::Unary => UNARY,
            Source::Affine => AFFINE,
            Source::Fill => FILL,
            Source::Cast => CAST,
            Source::Cmp => CMP,
            Source::Reduce => REDUCE,
            Source::Ternary => TERNARY,
            Source::Indexing => INDEXING,
            Source::Conv => CONV,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Source::Binary => "binary",
            Source::Unary => "unary",
            Source::Affine => "affine",
            Source::Fill => "fill",
            Source::Cast => "cast",
            Source::Cmp => "cmp",
            Source::Reduce => "reduce",
            Source::Ternary => "ternary",
            Source::Indexing => "indexing",
            Source::Conv => "conv",
        }
    }
}
