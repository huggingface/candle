static MOE_KERNEL_HFMA2: &str = "hfma2";

#[cfg(has_wmma)]
fn select_wmma_for_dtype(dtype: i32) -> bool {
    if dtype == DTYPE_BF16 {
        cfg!(has_wmma_bf16)
    } else {
        cfg!(has_wmma_f16) || cfg!(has_wmma)
    }
}

#[cfg(not(has_wmma))]
pub fn select_moe_kernel(_m: usize, _n: usize, _k: usize, _dtype: i32) -> &'static str {
    MOE_KERNEL_HFMA2
}

#[cfg(has_wmma)]
pub fn select_moe_kernel(_m: usize, _n: usize, _k: usize, dtype: i32) -> &'static str {
    if select_wmma_for_dtype(dtype) {
        return MOE_KERNEL_WMMA;
    }

    MOE_KERNEL_HFMA2
}

#[cfg(any(has_bf16, allow_legacy_bf16))]
pub const HAS_BF16_SUPPORT: bool = true;
#[cfg(not(any(has_bf16, allow_legacy_bf16)))]
pub const HAS_BF16_SUPPORT: bool = false;

// ── Exported capability constants (replaces ir_caps::export_capabilities!) ──
pub const HAS_HALF2_NATIVE: bool = cfg!(has_half2_native);
pub const HAS_BF16: bool = cfg!(any(has_bf16, allow_legacy_bf16));
pub const HAS_FP8: bool = cfg!(any(has_fp8, allow_legacy_fp8));
pub const HAS_WMMA: bool = cfg!(has_wmma);
pub const HAS_WMMA_F16: bool = cfg!(has_wmma_f16);
pub const HAS_WMMA_BF16: bool = cfg!(has_wmma_bf16);
pub const HAS_TENSOR_CORES: bool = cfg!(has_tensor_cores);
pub const HAS_F16_ARITHMETIC: bool = cfg!(has_f16_arithmetic);
pub const ALLOW_LEGACY_BF16: bool = cfg!(allow_legacy_bf16);
pub const ALLOW_LEGACY_FP8: bool = cfg!(allow_legacy_fp8);
