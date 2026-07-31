//! The `gemm_reduced_precision_*` knobs, mirroring `cuda_backend`'s.
//!
//! Split out of [`super::gemm`] to keep that file under the workspace 400-line
//! cap. Only the f16 knob does anything on ROCm; the other two are stored so
//! that portable code can express the intent, and each says so in its docs.

// Default for the reduced precision setting is false, matching `cuda_backend`
// and pytorch. https://github.com/pytorch/pytorch/issues/123157
static MM_F16_REDUCED_PRECISION: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static MM_BF16_REDUCED_PRECISION: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static MM_F32_REDUCED_PRECISION: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Whether reduced precision accumulation is allowed for f32 GEMMs.
///
/// # No-op on ROCm
/// The counterpart of CUDA's `CUBLAS_COMPUTE_32F_FAST_TF32` is rocBLAS' xf32
/// math mode, which exists only on the CDNA parts (gfx94x) — there is no xf32
/// datatype at all in the rocBLAS ABI, only `rocblas_xf32_xdl_math_op`, and it
/// is a *handle*-wide mode rather than a per-call argument. Toggling it per
/// matmul would mean mutating the device's shared rocBLAS handle under a lock on
/// every call, for a mode this hardware does not implement. The flag is stored
/// and readable so portable code can express the intent, but the f32 path does
/// not consult it.
pub fn gemm_reduced_precision_f32() -> bool {
    MM_F32_REDUCED_PRECISION.load(std::sync::atomic::Ordering::Relaxed)
}

/// See [`gemm_reduced_precision_f32`] — stored, but not consulted on ROCm.
pub fn set_gemm_reduced_precision_f32(b: bool) {
    MM_F32_REDUCED_PRECISION.store(b, std::sync::atomic::Ordering::Relaxed)
}

/// Whether reduced precision accumulation is allowed for f16 GEMMs.
///
/// This one is live: it selects an f16 rocBLAS compute type, the analogue of
/// `CUBLAS_COMPUTE_16F`, in place of the f32 compute type the backend defaults
/// to. Accumulating in f16 is faster and materially less accurate; the default
/// of `false` is what makes ROCm's f16 matmul match CUDA's out of the box.
pub fn gemm_reduced_precision_f16() -> bool {
    MM_F16_REDUCED_PRECISION.load(std::sync::atomic::Ordering::Relaxed)
}

/// See [`gemm_reduced_precision_f16`].
pub fn set_gemm_reduced_precision_f16(b: bool) {
    MM_F16_REDUCED_PRECISION.store(b, std::sync::atomic::Ordering::Relaxed)
}

/// Whether reduced precision accumulation is allowed for bf16 GEMMs.
///
/// # No-op on ROCm
/// CUDA selects `CUBLAS_COMPUTE_32F_FAST_16BF` here. rocBLAS has no bf16 compute
/// type — `rocblas_datatype_bf16_r` is an operand type only, and passing it as
/// `compute_type` is rejected — so the bf16 path always accumulates in f32. The
/// flag is stored and readable so portable code can express the intent, but it
/// changes nothing.
pub fn gemm_reduced_precision_bf16() -> bool {
    MM_BF16_REDUCED_PRECISION.load(std::sync::atomic::Ordering::Relaxed)
}

/// See [`gemm_reduced_precision_bf16`] — stored, but not consulted on ROCm.
pub fn set_gemm_reduced_precision_bf16(b: bool) {
    MM_BF16_REDUCED_PRECISION.store(b, std::sync::atomic::Ordering::Relaxed)
}
