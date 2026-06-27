use core::ffi::{c_int, c_void};

extern "C" {
    pub(crate) fn run_fp8_block_gemm_f32(
        x: *const f32,
        w: *const c_void,
        scale: *const f32,
        y: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
        block_size: c_int,
        scale_cols: c_int,
    );
}
