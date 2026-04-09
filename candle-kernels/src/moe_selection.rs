pub const MOE_BACKEND_HFMA2: &str = "hfma2";
pub const MOE_BACKEND_WMMA: &str = "wmma";

#[cfg(has_wmma)]
const DTYPE_BF16: i32 = 1;

#[cfg(has_wmma)]
fn supports_wmma_for_dtype(dtype: i32) -> bool {
    if dtype == DTYPE_BF16 {
        cfg!(has_wmma_bf16)
    } else {
        cfg!(has_wmma_f16) || cfg!(has_wmma)
    }
}

pub fn select_moe_backend(_m: usize, _n: usize, _k: usize, _dtype: i32) -> &'static str {
    #[cfg(has_wmma)]
    if supports_wmma_for_dtype(_dtype) {
        return MOE_BACKEND_WMMA;
    }

    MOE_BACKEND_HFMA2
}
