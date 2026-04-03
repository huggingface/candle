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
#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DeviceRepr, ValidAsZeroBits};
    use half::f16;

    #[derive(Copy, Clone, Debug, PartialEq)]
    #[repr(transparent)]
    struct F16(f16);
    unsafe impl DeviceRepr for F16 {}
    unsafe impl ValidAsZeroBits for F16 {}

    #[test]
    fn test_moe_gemm_dispatch_f16() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();

        let num_experts = 2;
        let topk = 1;
        let size_m = 4;
        let size_n = 32;
        let size_k = 32;
        let dtype = 0; // f16

        // Inputs [size_m, size_k]
        let input_host = vec![F16(f16::from_f32(0.5)); size_m as usize * size_k as usize];
        let input_dev = stream.clone_htod(&input_host).expect("H2D Copy failed");

        // Weights [num_experts, size_n, size_k]
        let weights_host =
            vec![F16(f16::from_f32(0.1)); num_experts as usize * size_n as usize * size_k as usize];
        let weights_dev = stream.clone_htod(&weights_host).expect("H2D Copy failed");

        // Sorted Token IDs [size_m]
        let sorted_token_ids_host = vec![0, 1, 2, 3];
        let sorted_token_ids_dev = stream
            .clone_htod(&sorted_token_ids_host)
            .expect("H2D Copy failed");

        // Expert IDs [size_m * topk]
        let expert_ids_host = vec![0, 0, 1, 1];
        let expert_ids_dev = stream
            .clone_htod(&expert_ids_host)
            .expect("H2D Copy failed");

        // Output [size_m, size_n]
        let output_dev: CudaSlice<F16> = stream
            .alloc_zeros(size_m as usize * size_n as usize)
            .expect("Alloc failed");

        // Pre-allocated buffers
        let expert_counts_dev: CudaSlice<i32> = stream
            .alloc_zeros(num_experts as usize)
            .expect("Alloc failed");
        let expert_offsets_dev: CudaSlice<i32> = stream
            .alloc_zeros(num_experts as usize + 1)
            .expect("Alloc failed");

        unsafe {
            dispatch_moe_gemm(
                input_dev.device_ptr(&stream).0 as *const _,
                weights_dev.device_ptr(&stream).0 as *const _,
                sorted_token_ids_dev.device_ptr(&stream).0 as *const _,
                expert_ids_dev.device_ptr(&stream).0 as *const _,
                core::ptr::null(), // topk_weights
                output_dev.device_ptr(&stream).0 as *mut _,
                expert_counts_dev.device_ptr(&stream).0 as *mut _,
                expert_offsets_dev.device_ptr(&stream).0 as *mut _,
                num_experts,
                topk,
                size_m,
                size_n,
                size_k,
                dtype,
                true,                              // is_prefill
                stream.cu_stream() as *mut _ as _, // stream handle
            );
        }

        let output_host = stream.clone_dtoh(&output_dev).expect("D2H Copy failed");
        println!(
            "MoE Output sample: {:?}",
            &output_host[..8]
                .iter()
                .map(|x| x.0.to_f32())
                .collect::<Vec<_>>()
        );
        // Basic sanity check: output should be non-zero
        assert!(output_host.iter().any(|x| x.0.to_f32() != 0.0f32));
    }

    #[derive(Copy, Clone, Debug, PartialEq)]
    #[repr(transparent)]
    struct BF16(half::bf16);
    unsafe impl DeviceRepr for BF16 {}
    unsafe impl ValidAsZeroBits for BF16 {}

    /// Test MoE BF16 via hfma2 fallback on CC 6.1 (no WMMA, no native BF16).
    /// This exercises the ALLOW_LEGACY_BF16 path in the hfma2 kernel.
    #[test]
    fn test_moe_bf16_hfma2() {
        let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
        let stream = ctx.default_stream();

        let num_experts = 2i32;
        let topk = 1i32;
        let size_m = 4i32;
        let size_n = 32i32;
        let size_k = 32i32;
        let dtype = 1; // bf16

        // Inputs [size_m, size_k]
        let input_host = vec![BF16(half::bf16::from_f32(0.5)); size_m as usize * size_k as usize];
        let input_dev = stream.clone_htod(&input_host).expect("H2D Copy failed");

        // Weights [num_experts, size_n, size_k]
        let weights_host = vec![
            BF16(half::bf16::from_f32(0.1));
            num_experts as usize * size_n as usize * size_k as usize
        ];
        let weights_dev = stream.clone_htod(&weights_host).expect("H2D Copy failed");

        // Sorted Token IDs [size_m]
        let sorted_token_ids_host = vec![0i32, 1, 2, 3];
        let sorted_token_ids_dev = stream
            .clone_htod(&sorted_token_ids_host)
            .expect("H2D Copy failed");

        // Expert IDs [size_m]
        let expert_ids_host = vec![0i32, 0, 1, 1];
        let expert_ids_dev = stream
            .clone_htod(&expert_ids_host)
            .expect("H2D Copy failed");

        // Output [size_m, size_n]
        let output_dev: CudaSlice<BF16> = stream
            .alloc_zeros(size_m as usize * size_n as usize)
            .expect("Alloc failed");

        // Pre-allocated buffers
        let expert_counts_dev: CudaSlice<i32> = stream
            .alloc_zeros(num_experts as usize)
            .expect("Alloc failed");
        let expert_offsets_dev: CudaSlice<i32> = stream
            .alloc_zeros(num_experts as usize + 1)
            .expect("Alloc failed");

        unsafe {
            dispatch_moe_gemm(
                input_dev.device_ptr(&stream).0 as *const _,
                weights_dev.device_ptr(&stream).0 as *const _,
                sorted_token_ids_dev.device_ptr(&stream).0 as *const _,
                expert_ids_dev.device_ptr(&stream).0 as *const _,
                core::ptr::null(), // topk_weights
                output_dev.device_ptr(&stream).0 as *mut _,
                expert_counts_dev.device_ptr(&stream).0 as *mut _,
                expert_offsets_dev.device_ptr(&stream).0 as *mut _,
                num_experts,
                topk,
                size_m,
                size_n,
                size_k,
                dtype,
                true,                              // is_prefill
                stream.cu_stream() as *mut _ as _, // stream handle
            );
        }

        let output_host = stream.clone_dtoh(&output_dev).expect("D2H Copy failed");
        let output_f32: Vec<f32> = output_host.iter().map(|x| x.0.to_f32()).collect();

        println!("MoE BF16 hfma2 Output sample: {:?}", &output_f32[..8]);

        // Verify no NaN/Inf
        for (i, v) in output_f32.iter().enumerate() {
            assert!(!v.is_nan(), "NaN at position {} in MoE BF16 output", i);
            assert!(!v.is_infinite(), "Inf at position {} in MoE BF16 output", i);
        }

        // Basic sanity check: output should be non-zero
        assert!(
            output_f32.iter().any(|x| *x != 0.0f32),
            "MoE BF16 output is all zeros"
        );
    }
}
