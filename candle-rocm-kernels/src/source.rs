// ROCm/HIP kernel source files
// These are embedded as strings and compiled at runtime

pub const BINARY: &str = include_str!("kernels/binary.hip");
pub const UNARY: &str = include_str!("kernels/unary.hip");
pub const AFFINE: &str = include_str!("kernels/affine.hip");
pub const FILL: &str = include_str!("kernels/fill.hip");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    Binary,
    Unary,
    Affine,
    Fill,
}

impl Source {
    pub fn code(&self) -> &'static str {
        match self {
            Source::Binary => BINARY,
            Source::Unary => UNARY,
            Source::Affine => AFFINE,
            Source::Fill => FILL,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Source::Binary => "binary",
            Source::Unary => "unary",
            Source::Affine => "affine",
            Source::Fill => "fill",
        }
    }
}
