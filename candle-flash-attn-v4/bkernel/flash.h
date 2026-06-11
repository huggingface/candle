#pragma once

#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>

#include <cmath>

struct Flash_fwd_params {
    using seqlen_t = uint32_t;

    int8_t device_id;

    seqlen_t b;
    seqlen_t b_k;
    seqlen_t h;
    seqlen_t h_k;
    seqlen_t d;
    seqlen_t d_rounded;
    seqlen_t h_h_k_ratio;

    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;
    void *__restrict__ o_ptr;

    void *__restrict__ p_ptr;
    void *__restrict__ softmax_lse_ptr;
    void *__restrict__ alibi_slopes_ptr;

    void *__restrict__ descale_q_ptr;
    void *__restrict__ descale_k_ptr;
    void *__restrict__ descale_v_ptr;

    int32_t *__restrict__ cu_seqlens_q;
    int32_t *__restrict__ cu_seqlens_k;
    int32_t *__restrict__ seqused_q;
    int32_t *__restrict__ seqused_k;

    int32_t *__restrict__ block_table;
    int32_t *__restrict__ tile_count_semaphore;

    int64_t q_batch_stride;
    int64_t k_batch_stride;
    int64_t v_batch_stride;
    int64_t o_batch_stride;
    int64_t alibi_slopes_batch_stride;

    int64_t q_row_stride;
    int64_t k_row_stride;
    int64_t v_row_stride;
    int64_t o_row_stride;

    int64_t q_head_stride;
    int64_t k_head_stride;
    int64_t v_head_stride;
    int64_t o_head_stride;

    seqlen_t seqlen_q;
    seqlen_t seqlen_k;
    seqlen_t seqlen_q_rounded;
    seqlen_t seqlen_k_rounded;

    uint32_t total_q;
    uint32_t total_k;

    int page_block_size;
    int page_num_blocks;
    int block_table_batch_stride;

    float scale_softmax;
    float scale_softmax_log2;
    uint32_t scale_softmax_log2_half2;

    float p_dropout;
    uint8_t p_dropout_in_uint8_t;
    float rp_dropout;
    float scale_softmax_rp_dropout;

    int is_bf16;
    int is_e4m3;
    int is_causal;
    int is_local;
    int is_kv_cache;
    int seqlenq_ngroups_swapped;
    int unpadded_lse;

    int window_size_left;
    int window_size_right;

    int use_gqa_packing;
    int num_splits;

    void *__restrict__ softmax_lseaccum_ptr;
    void *__restrict__ oaccum_ptr;
    int64_t oaccum_row_stride;
    int64_t oaccum_head_stride;
    int64_t oaccum_batch_stride;
    int64_t oaccum_split_stride;

    float rescale_threshold;
    int deterministic;
    int use_2cta_mode;
    int num_sm;
};

struct Flash_bwd_params : public Flash_fwd_params {
    void *__restrict__ do_ptr;
    void *__restrict__ dq_ptr;
    void *__restrict__ dk_ptr;
    void *__restrict__ dv_ptr;

    int64_t do_batch_stride;
    int64_t dq_batch_stride;
    int64_t dk_batch_stride;
    int64_t dv_batch_stride;

    int64_t do_row_stride;
    int64_t dq_row_stride;
    int64_t dk_row_stride;
    int64_t dv_row_stride;

    int64_t do_head_stride;
    int64_t dq_head_stride;
    int64_t dk_head_stride;
    int64_t dv_head_stride;

    void *__restrict__ dq_accum_ptr;
    void *__restrict__ dk_accum_ptr;
    void *__restrict__ dv_accum_ptr;

    void *__restrict__ dsoftmax_sum;
    void *__restrict__ dq_semaphore_ptr;
};
