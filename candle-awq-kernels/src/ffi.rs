use core::ffi::c_int;

extern "C" {
    pub(crate) fn run_awq_gemm_f32(
        x: *const f32,
        qweight: *const i32,
        qzeros: *const i32,
        scales: *const f32,
        y: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
        group_size: c_int,
        n_packed_out: c_int,
    );
}
