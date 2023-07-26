// Copyright (c) 2023, Tri Dao.

// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

// template<>
// void run_mha_fwd_<cutlass::half_t, 32>(Flash_fwd_params &params, cudaStream_t stream) {
//     using elem_type = cutlass::half_t;
//     BOOL_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
//         run_flash_fwd<Flash_fwd_kernel_traits<32, 128, 128, 4, false, false, elem_type>, Is_dropout>(params, stream);
//         // For dropout there might be a lot of register spilling?
//         // These two are very slow due to register spilling
//         // run_flash_fwd<Flash_fwd_kernel_traits<32, 256, 128, 4, false, elem_type>>(params, stream);
//         // run_flash_fwd<Flash_fwd_kernel_traits<32, 128, 256, 4, false, elem_type>>(params, stream);
//         // This one is slightly slower
//         // run_flash_fwd<Flash_fwd_kernel_traits<32, 256, 64, 4, false, elem_type>>(params, stream);
//     });
// }
template<>
void run_mha_fwd_<cutlass::half_t, 32>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim32<cutlass::half_t>(params, stream);
}


extern "C" void run_mha(
    void *q_ptr,
    void *k_ptr,
    void *v_ptr,
    void *o_ptr,
    void *softmax_lse_ptr,

    uint32_t q_batch_stride,
    uint32_t k_batch_stride,
    uint32_t v_batch_stride,
    uint32_t o_batch_stride,

    uint32_t q_row_stride,
    uint32_t k_row_stride,
    uint32_t v_row_stride,
    uint32_t o_row_stride,

    uint32_t q_head_stride,
    uint32_t k_head_stride,
    uint32_t v_head_stride,
    uint32_t o_head_stride,

    uint32_t b,
    uint32_t h,
    uint32_t h_k,
    uint32_t d,
    uint32_t d_rounded,
    float softmax_scale,

    uint32_t seqlen_q,
    uint32_t seqlen_k,
    uint32_t seqlen_q_rounded,
    uint32_t seqlen_k_rounded,

    int is_causal
) {
    Flash_fwd_params params;
    // Reset the parameters
    memset(&params, 0, sizeof(params));

    // Set the pointers and strides.
    params.q_ptr = q_ptr;
    params.k_ptr = k_ptr;
    params.v_ptr = v_ptr;
    params.o_ptr = o_ptr;

    params.softmax_lse_ptr = softmax_lse_ptr;

    // All stride are in elements, not bytes.
    params.q_batch_stride = q_batch_stride;
    params.k_batch_stride = k_batch_stride;
    params.v_batch_stride = v_batch_stride;
    params.o_batch_stride = o_batch_stride;

    params.q_row_stride = q_row_stride;
    params.k_row_stride = k_row_stride;
    params.v_row_stride = v_row_stride;
    params.o_row_stride = o_row_stride;
    params.q_head_stride = q_head_stride;
    params.k_head_stride = k_head_stride;
    params.v_head_stride = v_head_stride;
    params.o_head_stride = o_head_stride;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;
    params.is_causal = is_causal;

    // Set the different scale values.
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;

    params.p_dropout = 1.; // probability to keep
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    params.is_bf16 = 0;
    params.cu_seqlens_q = nullptr;
    params.cu_seqlens_k = nullptr;
    params.p_ptr = nullptr;

    cudaStream_t stream = 0; // Use the default stream.
    run_mha_fwd_<cutlass::half_t, 32>(params, stream);
}
