use core::ffi::c_int;

extern "C" {
    pub(crate) fn run_gptq_gemm_f32(
        x: *const f32,
        qweight: *const i32,
        qzeros: *const i32,
        scales: *const f32,
        g_idx: *const i32,
        y: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
        bits: c_int,
        pack_factor: c_int,
        n_groups_out: c_int,
    );

    /// Tensor-core (WMMA `mma.sync`) variant, 4-bit only: `bits`/`pack_factor` are fixed at 4/8
    /// in the kernel itself, so they are not parameters here.
    pub(crate) fn run_gptq_gemm_tc_f32(
        x: *const f32,
        qweight: *const i32,
        qzeros: *const i32,
        scales: *const f32,
        g_idx: *const i32,
        y: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
        n_groups_out: c_int,
    );

    /// Drive the vendored Marlin FP16xINT4 tensor-core GEMM (see `kernels/marlin/`).
    /// `a`/`c`/`s` are fp16 (`half`) device pointers, `b`/`workspace` are int32. `workspace` must
    /// hold at least `n / 128 * max_par` zero-initialized entries. Returns 0 on success, 1 for a
    /// problem-shape error, 2 for an unsupported kernel-shape combination.
    pub(crate) fn run_marlin_gemm(
        a: *const core::ffi::c_void,
        b: *const core::ffi::c_void,
        c: *mut core::ffi::c_void,
        s: *const core::ffi::c_void,
        prob_m: c_int,
        prob_n: c_int,
        prob_k: c_int,
        workspace: *mut core::ffi::c_void,
        groupsize: c_int,
        dev: c_int,
        max_par: c_int,
    ) -> c_int;
}
