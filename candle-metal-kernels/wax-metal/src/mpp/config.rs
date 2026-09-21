//! MPP GEMM configuration.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MppDtype {
    BF16,
    F16,
    F32,
}

impl MppDtype {
    // TODO: Repeated logic
    pub(super) fn ll(self) -> &'static str {
        match self {
            MppDtype::BF16 => "bfloat",
            MppDtype::F16 => "half",
            MppDtype::F32 => "float",
        }
    }

    /// MPP symbol suffix (`..._run_cooperative_dv_b16_dv_b16_b16`).
    pub(super) fn suffix(self) -> &'static str {
        match self {
            MppDtype::BF16 => "b16",
            MppDtype::F16 => "f16",
            MppDtype::F32 => "f32",
        }
    }

    /// Element type tag the `__tensorops_impl_*` entry points take.
    pub(super) fn tag(self) -> i32 {
        match self {
            MppDtype::BF16 => 0x90000010u32 as i32,
            MppDtype::F16 => 0x10000010,
            MppDtype::F32 => 0x10000020,
        }
    }

    pub fn size(self) -> u32 {
        match self {
            MppDtype::BF16 | MppDtype::F16 => 2,
            MppDtype::F32 => 4,
        }
    }
}

/// MPP GEMM. `C[M,N] = A[M,K] @ B[K,N]`.
/// Row-major. One `BM x BN` output tile per threadgroup.
#[derive(Debug, Clone)]
pub struct MppGemm {
    pub entry: String,
    pub dtype: MppDtype,
    // Shape
    pub m: i32,
    pub n: i32,
    pub k: i32,
    /// Strides.
    pub lda: i32,
    pub ldb: i32,
    pub ldc: i32,
    pub bm: i32,
    pub bn: i32,
    pub out_dtype: MppDtype,
    /// Split-K factor.
    pub splits: i32,
    pub trans_b: bool,
    // TODO: Sweep. Currently hardcoded to 1.
    pub simdgroups: i32,
}

impl MppGemm {
    /// Grid this kernel is to be dispatched with.
    /// `(x, y, z)`. x indexes N tiles and y indexes M tiles.
    pub fn grid(&self) -> (usize, usize, usize) {
        (
            (self.n as usize).div_ceil(self.bn as usize),
            (self.m as usize).div_ceil(self.bm as usize),
            self.splits.max(1) as usize,
        )
    }

    /// Split-K f32 scratch buffer size in elements. 0 if no splits.
    pub fn partial_elems(&self) -> usize {
        if self.splits <= 1 {
            0
        } else {
            self.splits as usize * self.m as usize * self.n as usize
        }
    }

    /// Threads per threadgroup.
    pub fn threadgroup(&self) -> (usize, usize, usize) {
        (self.simdgroups as usize * 32, 1, 1)
    }

    /// Launch geometry for this config.
    pub fn launch(&self) -> crate::Launch {
        crate::Launch::new(self.grid(), self.threadgroup())
    }
}
