//! Launch geometry as a compiler output.
//!
//! A kernel's launch geometry is part of its contract, and it differs per BACKEND
//! for the same mathematical operation: an SGMMA GEMM wants a
//! `(M/BM, N/BN)` threadgroup grid, while the MPP cooperative-tensor path wants
//! `(N/BN, M/BM, splits)` with a 32-wide threadgroup, and split-K additionally
//! needs an `[S,M,N]` f32 scratch buffer that a reduce pass sums.
//!
//! Handing back only the `.metallib` and letting each caller compute its own grid
//! means a backend swap silently invalidates every caller - and it fails as WRONG
//! NUMBERS rather than as an error, which is the worst available failure mode. So
//! the thing that chooses the backend returns [`LoweredKernel`], which cannot be
//! obtained without the launch that goes with it.

/// How to dispatch one pass. `grid` counts THREADGROUPS, not threads - the same
/// convention as `metal_core::DispatchConfig`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Launch {
    pub grid: (usize, usize, usize),
    pub threadgroup: (usize, usize, usize),
}

impl Launch {
    pub fn new(grid: (usize, usize, usize), threadgroup: (usize, usize, usize)) -> Self {
        Self { grid, threadgroup }
    }
}

/// A compiled kernel bundled with everything the caller needs to run it.
#[derive(Debug, Clone)]
pub struct LoweredKernel {
    /// The container, ready for `newLibraryWithData`.
    pub metallib: Vec<u8>,
    /// Entry point name inside `metallib`.
    pub entry: String,
    pub launch: Launch,
    /// Device scratch this backend requires, in bytes; 0 when it writes its output
    /// buffer directly. The caller allocates and binds it as the FIRST buffer in
    /// place of the destination, then runs a reduce pass over it.
    ///
    /// This exists because the requirement is a property of the BACKEND CHOICE, not
    /// of the problem: the same GEMM needs no scratch through SGMMA and needs
    /// `S*M*N*4` bytes through split-K MPP. A caller cannot predict it from the
    /// shape, so the compiler has to say.
    pub scratch_bytes: usize,
    /// Runtime scalars to bind AFTER the buffers, in order.
    ///
    /// A kernel that masks a store against its output's extent needs that extent, and a
    /// MUTABLE tensor arg cannot carry it statically - the entry wrapper deliberately
    /// erases a mutable view's shape to `-1`, because a mutable arg is declared TILE-shaped
    /// by convention (`gemm` takes `z: &mut Tensor<E,{[BM,BN]}>` while the tensor is
    /// `[M,N]`). So the extent is promoted to a kernel parameter and travels here, next to
    /// the metallib that decided it needed one. Empty for every kernel that does not.
    pub scalar_args: Vec<i32>,
}

impl LoweredKernel {
    /// `scalar_args` in the `(ptr, len)` form `dispatch_buffers` / `encode_to` take. The
    /// pointers borrow from `self`, so the kernel must outlive the dispatch.
    pub fn scalars(&self) -> Vec<(*const std::ffi::c_void, usize)> {
        self.scalar_args
            .iter()
            .map(|v| (v as *const i32 as *const std::ffi::c_void, 4))
            .collect()
    }

    /// True when this kernel writes f32 partials to scratch rather than the
    /// destination, so the caller owes a reduce pass.
    pub fn needs_reduce(&self) -> bool {
        self.scratch_bytes > 0
    }
}
