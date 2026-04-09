use crate::moe_selection::{select_moe_backend, MOE_BACKEND_WMMA};
use core::ffi::c_void;
//const DTYPE_F16: i32 = 0;
const DTYPE_BF16: i32 = 1;

#[allow(dead_code)]
extern "C" {
    // for unquntized models
    pub fn moe_gemm_wmma(
        input: *const c_void,         // device pointer [size_m, size_k]
        weights: *const c_void,       // device pointer [num_experts, size_n, size_k]
        sorted_token_ids: *const i32, // device pointer [size_m]
        expert_ids: *const i32,       // host array [size_m] (expert id per sorted token)
        topk_weights: *const f32,
        output: *mut c_void,      // device pointer [size_m, size_n]
        expert_counts: *mut i32,  // pre-allocated buffer [num_experts]
        expert_offsets: *mut i32, // pre-allocated buffer [num_experts + 1]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        dtype: i32, // 0=float16, 1=bf16 (for input/output)
        is_prefill: bool,
        stream: i64,
    );

    pub fn moe_gemm_hfma2(
        input: *const c_void,
        weights: *const c_void,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32,
        output: *mut c_void,
        expert_counts: *mut i32,
        expert_offsets: *mut i32,
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        dtype: i32,
        is_prefill: bool,
        stream: i64,
    );
}

pub unsafe fn dispatch_moe_gemm(
    input: *const c_void,
    weights: *const c_void,
    sorted_token_ids: *const i32,
    expert_ids: *const i32,
    topk_weights: *const f32,
    output: *mut c_void,
    expert_counts: *mut i32,
    expert_offsets: *mut i32,
    num_experts: i32,
    topk: i32,
    size_m: i32,
    size_n: i32,
    size_k: i32,
    dtype: i32,
    is_prefill: bool,
    stream: i64,
) {
    let backend = select_moe_backend(size_m as usize, size_n as usize, size_k as usize, dtype);
    if backend == MOE_BACKEND_WMMA {
        moe_gemm_wmma(
            input,
            weights,
            sorted_token_ids,
            expert_ids,
            topk_weights,
            output,
            expert_counts,
            expert_offsets,
            num_experts,
            topk,
            size_m,
            size_n,
            size_k,
            dtype,
            is_prefill,
            stream,
        );
    } else {
        if dtype == DTYPE_BF16 && !cfg!(has_bf16) && !cfg!(allow_legacy_bf16) {
            panic!("BF16 MoE requires has_bf16 or allow_legacy_bf16. Set ALLOW_LEGACY=bf16 or ALLOW_LEGACY=all.");
        }

        moe_gemm_hfma2(
            input,
            weights,
            sorted_token_ids,
            expert_ids,
            topk_weights,
            output,
            expert_counts,
            expert_offsets,
            num_experts,
            topk,
            size_m,
            size_n,
            size_k,
            dtype,
            is_prefill,
            stream,
        );
    }
}

pub unsafe fn dispatch_moe(
    input: *const c_void,
    weights: *const c_void,
    sorted_token_ids: *const i32,
    expert_ids: *const i32,
    topk_weights: *const f32,
    output: *mut c_void,
    expert_counts: *mut i32,
    expert_offsets: *mut i32,
    num_experts: i32,
    topk: i32,
    size_m: i32,
    size_n: i32,
    size_k: i32,
    dtype: i32,
    is_prefill: bool,
    stream: i64,
) {
    dispatch_moe_gemm(
        input,
        weights,
        sorted_token_ids,
        expert_ids,
        topk_weights,
        output,
        expert_counts,
        expert_offsets,
        num_experts,
        topk,
        size_m,
        size_n,
        size_k,
        dtype,
        is_prefill,
        stream,
    )
}

extern "C" {
    pub fn moe_gemm_gguf(
        input: *const f32,      // input [size_m, size_k]
        weights: *const c_void, // weights [num_experts, size_n, size_k]
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32, // device ptr or nullptr
        output: *mut c_void,      // float output [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        gguf_dtype: i32, // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5  (for weights)
        stream: i64,
    );

    pub fn moe_gemm_gguf_prefill(
        input: *const c_void, // input [size_m, size_k]
        weights: *const u8,   // weights [num_experts, size_n, size_k]
        sorted_token_ids: *const i32,
        expert_ids: *const i32,   //must be host ptr
        topk_weights: *const f32, // device ptr or nullptr
        output: *mut c_void,      // float output [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        input_dtype: i32, // 0=f16, 1=bf16 (for inputs)
        gguf_dtype: i32,  //Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5  (for weights)
        stream: i64,
    );
}
