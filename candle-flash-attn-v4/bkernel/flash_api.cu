#include "flash.h"
#include "static_switch.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <stdexcept>

// Stub implementations for FlashAttention-4 kernels.
// These will be replaced by actual CuTe-DSL generated kernels
// from https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute
//
// For now, these throw an error at runtime if called.

template<typename Element, int kHeadSize>
void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream) {
    throw std::runtime_error("FlashAttention-4 forward kernels not yet implemented. "
                             "Please build kernels from CuTe-DSL sources at "
                             "https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute");
}

template<typename Element, int kHeadSize, int kBlockH>
void run_mha_fwd_gqa_(Flash_fwd_params &params, cudaStream_t stream) {
    throw std::runtime_error("FlashAttention-4 forward GQA kernels not yet implemented. "
                             "Please build kernels from CuTe-DSL sources at "
                             "https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute");
}

template<typename Element, int kHeadSize>
void run_mha_bwd_(Flash_bwd_params &params, cudaStream_t stream) {
    throw std::runtime_error("FlashAttention-4 backward kernels not yet implemented. "
                             "Please build kernels from CuTe-DSL sources at "
                             "https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute");
}

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    int prec_type = 1;
    if (params.is_e4m3) {
        prec_type = 3;
    } else if (params.is_bf16) {
        prec_type = 2;
    }
    PREC_SWITCH(prec_type, elem_type, [&] {
        HEADDIM_SWITCH(params.d, kHeadDim, [&] {
            if (!params.use_gqa_packing) {
                run_mha_fwd_<elem_type, kHeadDim>(params, stream);
            } else {
                QUERYHEAD_SWITCH(params.h_h_k_ratio, kBlockH, [&] {
                    run_mha_fwd_gqa_<elem_type, kHeadDim, kBlockH>(params, stream);
                });
            }
        });
    });
}

void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
    int prec_type = 1;
    if (params.is_bf16) {
        prec_type = 2;
    }
    PREC_SWITCH(prec_type, elem_type, [&] {
        HEADDIM_SWITCH(params.d, kHeadDim, [&] {
            run_mha_bwd_<elem_type, kHeadDim>(params, stream);
        });
    });
}

extern "C" void run_mha_fwd(
    void *q_ptr,
    void *k_ptr,
    void *v_ptr,
    void *o_ptr,
    void *softmax_lse_ptr,
    void *alibi_slopes_ptr,

    int32_t *cu_seqlens_q_ptr,
    int32_t *cu_seqlens_k_ptr,

    uint32_t q_batch_stride,
    uint32_t k_batch_stride,
    uint32_t v_batch_stride,
    uint32_t o_batch_stride,
    uint32_t alibi_slopes_batch_stride,

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

    int is_bf16,
    int is_causal,
    int unpadded_lse,
    int use_gqa_packing,

    int window_size_left,
    int window_size_right,

    uint32_t total_q,
    uint32_t total_k,

    float rescale_threshold,
    int deterministic,
    int use_2cta_mode,
    int num_sm,

    void *stream_ptr
) {
    Flash_fwd_params params;
    memset(&params, 0, sizeof(params));

    params.q_ptr = q_ptr;
    params.k_ptr = k_ptr;
    params.v_ptr = v_ptr;
    params.o_ptr = o_ptr;

    params.softmax_lse_ptr = softmax_lse_ptr;
    params.alibi_slopes_ptr = alibi_slopes_ptr;

    params.q_batch_stride = q_batch_stride;
    params.k_batch_stride = k_batch_stride;
    params.v_batch_stride = v_batch_stride;
    params.o_batch_stride = o_batch_stride;
    params.alibi_slopes_batch_stride = alibi_slopes_batch_stride;

    params.q_row_stride = q_row_stride;
    params.k_row_stride = k_row_stride;
    params.v_row_stride = v_row_stride;
    params.o_row_stride = o_row_stride;
    params.q_head_stride = q_head_stride;
    params.k_head_stride = k_head_stride;
    params.v_head_stride = v_head_stride;
    params.o_head_stride = o_head_stride;

    params.b = b;
    params.b_k = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    __half scale_softmax_log2_half = __float2half(params.scale_softmax_log2);
    __half2 scale_softmax_log2_half2 = __half2(scale_softmax_log2_half, scale_softmax_log2_half);
    params.scale_softmax_log2_half2 = reinterpret_cast<uint32_t&>(scale_softmax_log2_half2);

    params.p_dropout = 1.f;
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;

    params.is_bf16 = is_bf16;
    params.cu_seqlens_q = cu_seqlens_q_ptr;
    params.cu_seqlens_k = cu_seqlens_k_ptr;
    params.p_ptr = nullptr;
    params.seqused_q = nullptr;
    params.seqused_k = nullptr;

    params.is_causal = is_causal;
    params.is_local = (window_size_left >= 0 || window_size_right >= 0) && !is_causal;
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    params.num_splits = 0;
    params.page_block_size = -1;

    params.total_q = total_q;
    params.total_k = total_k;

    params.unpadded_lse = unpadded_lse;
    params.use_gqa_packing = use_gqa_packing;

    params.rescale_threshold = rescale_threshold;
    params.deterministic = deterministic;
    params.use_2cta_mode = use_2cta_mode;
    params.num_sm = num_sm;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    run_mha_fwd(params, stream);
}

// C API for backward (stub for now; real kernel will be provided by CuTe-DSL)
extern "C" void run_mha_bwd(
    void *q_ptr,
    void *k_ptr,
    void *v_ptr,
    void *o_ptr,
    void *dout_ptr,
    void *dq_ptr,
    void *dk_ptr,
    void *dv_ptr,
    void *softmax_lse_ptr,
    void *dsoftmax_sum_ptr,
    void *alibi_slopes_ptr,

    int32_t *cu_seqlens_q_ptr,
    int32_t *cu_seqlens_k_ptr,

    uint32_t q_batch_stride,
    uint32_t k_batch_stride,
    uint32_t v_batch_stride,
    uint32_t o_batch_stride,
    uint32_t dout_batch_stride,
    uint32_t dq_batch_stride,
    uint32_t dk_batch_stride,
    uint32_t dv_batch_stride,
    uint32_t alibi_slopes_batch_stride,

    uint32_t q_row_stride,
    uint32_t k_row_stride,
    uint32_t v_row_stride,
    uint32_t o_row_stride,
    uint32_t dout_row_stride,
    uint32_t dq_row_stride,
    uint32_t dk_row_stride,
    uint32_t dv_row_stride,

    uint32_t q_head_stride,
    uint32_t k_head_stride,
    uint32_t v_head_stride,
    uint32_t o_head_stride,
    uint32_t dout_head_stride,
    uint32_t dq_head_stride,
    uint32_t dk_head_stride,
    uint32_t dv_head_stride,

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

    int is_bf16,
    int is_causal,
    int unpadded_lse,
    int use_gqa_packing,

    int window_size_left,
    int window_size_right,

    uint32_t total_q,
    uint32_t total_k,

    float rescale_threshold,
    int deterministic,
    int use_2cta_mode,
    int num_sm,

    void *stream_ptr
) {
    Flash_bwd_params params;
    memset(&params, 0, sizeof(params));

    params.q_ptr = q_ptr;
    params.k_ptr = k_ptr;
    params.v_ptr = v_ptr;
    params.o_ptr = o_ptr;
    params.do_ptr = dout_ptr;
    params.dq_ptr = dq_ptr;
    params.dk_ptr = dk_ptr;
    params.dv_ptr = dv_ptr;

    params.softmax_lse_ptr = softmax_lse_ptr;
    params.dsoftmax_sum = dsoftmax_sum_ptr;
    params.alibi_slopes_ptr = alibi_slopes_ptr;

    params.q_batch_stride = q_batch_stride;
    params.k_batch_stride = k_batch_stride;
    params.v_batch_stride = v_batch_stride;
    params.o_batch_stride = o_batch_stride;
    params.do_batch_stride = dout_batch_stride;
    params.dq_batch_stride = dq_batch_stride;
    params.dk_batch_stride = dk_batch_stride;
    params.dv_batch_stride = dv_batch_stride;
    params.alibi_slopes_batch_stride = alibi_slopes_batch_stride;

    params.q_row_stride = q_row_stride;
    params.k_row_stride = k_row_stride;
    params.v_row_stride = v_row_stride;
    params.o_row_stride = o_row_stride;
    params.do_row_stride = dout_row_stride;
    params.dq_row_stride = dq_row_stride;
    params.dk_row_stride = dk_row_stride;
    params.dv_row_stride = dv_row_stride;

    params.q_head_stride = q_head_stride;
    params.k_head_stride = k_head_stride;
    params.v_head_stride = v_head_stride;
    params.o_head_stride = o_head_stride;
    params.do_head_stride = dout_head_stride;
    params.dq_head_stride = dq_head_stride;
    params.dk_head_stride = dk_head_stride;
    params.dv_head_stride = dv_head_stride;

    params.b = b;
    params.b_k = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    __half scale_softmax_log2_half = __float2half(params.scale_softmax_log2);
    __half2 scale_softmax_log2_half2 = __half2(scale_softmax_log2_half, scale_softmax_log2_half);
    params.scale_softmax_log2_half2 = reinterpret_cast<uint32_t&>(scale_softmax_log2_half2);

    params.p_dropout = 1.f;
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;

    params.is_bf16 = is_bf16;
    params.cu_seqlens_q = cu_seqlens_q_ptr;
    params.cu_seqlens_k = cu_seqlens_k_ptr;
    params.p_ptr = nullptr;
    params.seqused_q = nullptr;
    params.seqused_k = nullptr;

    params.is_causal = is_causal;
    params.is_local = (window_size_left >= 0 || window_size_right >= 0) && !is_causal;
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    params.num_splits = 0;
    params.page_block_size = -1;

    params.total_q = total_q;
    params.total_k = total_k;

    params.unpadded_lse = unpadded_lse;
    params.use_gqa_packing = use_gqa_packing;

    params.rescale_threshold = rescale_threshold;
    params.deterministic = deterministic;
    params.use_2cta_mode = use_2cta_mode;
    params.num_sm = num_sm;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    run_mha_bwd(params, stream);
}
