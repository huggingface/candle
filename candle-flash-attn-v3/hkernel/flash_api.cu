/* 
 * Copyright (c) 2024 Michael Feil
 * originally published at https://github.com/Dao-AILab/flash-attention/tree/main/hopper Tri Dao, BSD-3-Clause License
 *
 * Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
 * http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
 * <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
 * option. This file may not be copied, modified, or distributed
 * except according to those terms.

 * Authors explaination: Provide a copy of the first two lines in each
 redistributed version.
 */

#include "flash_fwd_launch_template.h"
#include "flash.h"
#include "static_switch.h"


// Helper to read/print small FP16 arrays from device
void read_and_print_fp16(const void* dev_ptr, size_t num_elements, const char* name) {
    if (!dev_ptr) {
        printf("  %s is null.\n", name);
        return;
    }
    // We copy `num_elements` __half from GPU -> CPU
    std::vector<__half> host_data(num_elements);
    cudaMemcpy(host_data.data(), dev_ptr,
               sizeof(__half) * num_elements, cudaMemcpyDeviceToHost);

    printf("  %s first %zu FP16 elements:\n    ", name, num_elements);
    for (size_t i = 0; i < num_elements; i++) {
        // Convert each __half to float for printing
        float val = __half2float(host_data[i]);
        printf("%9.6f ", val);
    }
    printf("\n");
}

// Helper to read/print small int32 arrays from device
void read_and_print_int32(const int32_t* dev_ptr, size_t num_elements, const char* name) {
    if (!dev_ptr) {
        printf("  %s is null.\n", name);
        return;
    }
    std::vector<int32_t> host_data(num_elements);
    cudaMemcpy(host_data.data(), dev_ptr,
               sizeof(int32_t) * num_elements, cudaMemcpyDeviceToHost);

    printf("  %s first %zu int32 values:\n    ", name, num_elements);
    for (size_t i = 0; i < num_elements; i++) {
        printf("%d ", host_data[i]);
    }
    printf("\n");
}

// Prints all fields from Flash_fwd_params, plus optionally reads small data from pointers
void print_params(const Flash_fwd_params &p) {
    printf("\n===== Flash_fwd_params Dump =====\n");

    // Basic geometry
    printf("  b                 = %lu\n", p.b);
    printf("  b_k               = %lu\n", p.b_k);
    printf("  h                 = %lu\n", p.h);
    printf("  h_k               = %lu\n", p.h_k);
    printf("  d                 = %lu\n", p.d);
    printf("  d_rounded         = %lu\n", p.d_rounded);
    printf("  h_h_k_ratio       = %lu\n", p.h_h_k_ratio);

    // Sequence lengths
    printf("  seqlen_q          = %lu\n", p.seqlen_q);
    printf("  seqlen_k          = %lu\n", p.seqlen_k);
    printf("  seqlen_q_rounded  = %lu\n", p.seqlen_q_rounded);
    printf("  seqlen_k_rounded  = %lu\n", p.seqlen_k_rounded);
    printf("  total_q           = %u\n", p.total_q);
    printf("  total_k           = %u\n", p.total_k);

    // Strides
    printf("  q_batch_stride    = %lu\n", (unsigned long)p.q_batch_stride);
    printf("  q_row_stride      = %lu\n", (unsigned long)p.q_row_stride);
    printf("  q_head_stride     = %lu\n", (unsigned long)p.q_head_stride);
    printf("  k_batch_stride    = %lu\n", (unsigned long)p.k_batch_stride);
    printf("  k_row_stride      = %lu\n", (unsigned long)p.k_row_stride);
    printf("  k_head_stride     = %lu\n", (unsigned long)p.k_head_stride);
    printf("  v_batch_stride    = %lu\n", (unsigned long)p.v_batch_stride);
    printf("  v_row_stride      = %lu\n", (unsigned long)p.v_row_stride);
    printf("  v_head_stride     = %lu\n", (unsigned long)p.v_head_stride);
    printf("  o_batch_stride    = %lu\n", (unsigned long)p.o_batch_stride);
    printf("  o_row_stride      = %lu\n", (unsigned long)p.o_row_stride);
    printf("  o_head_stride     = %lu\n", (unsigned long)p.o_head_stride);

    // Pointer addresses
    printf("\n  Pointer addresses:\n");
    printf("    q_ptr           = %p\n", p.q_ptr);
    printf("    k_ptr           = %p\n", p.k_ptr);
    printf("    v_ptr           = %p\n", p.v_ptr);
    printf("    o_ptr           = %p\n", p.o_ptr);
    printf("    p_ptr           = %p\n", p.p_ptr);
    printf("    softmax_lse_ptr = %p\n", p.softmax_lse_ptr);
    printf("    alibi_slopes_ptr= %p\n", p.alibi_slopes_ptr);
    printf("    descale_q_ptr   = %p\n", p.descale_q_ptr);
    printf("    descale_k_ptr   = %p\n", p.descale_k_ptr);
    printf("    descale_v_ptr   = %p\n", p.descale_v_ptr);

    // (varlen / kv-cache) pointer addresses
    printf("    cu_seqlens_q    = %p\n", p.cu_seqlens_q);
    printf("    cu_seqlens_k    = %p\n", p.cu_seqlens_k);
    printf("    seqused_q       = %p\n", p.seqused_q);
    printf("    seqused_k       = %p\n", p.seqused_k);
    printf("    block_table     = %p\n", p.block_table);
    printf("    tile_count_semaphore = %p\n", p.tile_count_semaphore);

    // Additional KV cache / GQA
    printf("  page_block_size   = %d\n", p.page_block_size);
    printf("  page_num_blocks   = %d\n", p.page_num_blocks);
    printf("  use_gqa_packing   = %d\n", p.use_gqa_packing);
    printf("  num_splits        = %d\n", p.num_splits);

    // Softmax & dropout scales
    printf("\n  Softmax / dropout:\n");
    printf("    scale_softmax            = %f\n", p.scale_softmax);
    printf("    scale_softmax_log2       = %f\n", p.scale_softmax_log2);
    printf("    scale_softmax_log2_half2 = 0x%08x (raw bits)\n", p.scale_softmax_log2_half2);
    printf("    p_dropout                = %f\n", p.p_dropout);
    printf("    p_dropout_in_uint8_t     = %u\n", p.p_dropout_in_uint8_t);
    printf("    rp_dropout               = %f\n", p.rp_dropout);
    printf("    scale_softmax_rp_dropout = %f\n", p.scale_softmax_rp_dropout);

    // Booleans / flags
    printf("\n  Flags:\n");
    printf("    is_bf16      = %d\n", p.is_bf16);
    printf("    is_e4m3      = %d\n", p.is_e4m3);
    printf("    is_causal    = %d\n", p.is_causal);
    printf("    is_local     = %d\n", p.is_local);
    printf("    is_kv_cache  = %d\n", p.is_kv_cache);
    printf("    seqlenq_ngroups_swapped = %d\n", p.seqlenq_ngroups_swapped);
    printf("    unpadded_lse = %d\n", p.unpadded_lse);

    // Window / block sizes
    printf("  window_size_left  = %d\n", p.window_size_left);
    printf("  window_size_right = %d\n", p.window_size_right);

    printf("===== End of Flash_fwd_params Dump =====\n\n");

    // Optional: read small data from pointers. 
    // Adjust the "4" or "2" below for however many elements you want to debug.

    // For example, if q_ptr is not null, try reading 4 elements as FP16
    if (p.q_ptr) {
        read_and_print_fp16(p.q_ptr, 4, "q_ptr");
    }
    if (p.k_ptr) {
        read_and_print_fp16(p.k_ptr, 4, "k_ptr");
    }
    if (p.v_ptr) {
        read_and_print_fp16(p.v_ptr, 4, "v_ptr");
    }
    if (p.o_ptr) {
        read_and_print_fp16(p.o_ptr, 4, "o_ptr");
    }
    if (p.softmax_lse_ptr) {
        read_and_print_fp16(p.softmax_lse_ptr, 4, "softmax_lse_ptr");
    }

    // For cu_seqlens_q and cu_seqlens_k, read 2 int32_t elements, for example
    if (p.cu_seqlens_q) {
        read_and_print_int32(p.cu_seqlens_q, 2, "cu_seqlens_q");
    }
    if (p.cu_seqlens_k) {
        read_and_print_int32(p.cu_seqlens_k, 2, "cu_seqlens_k");
    }
}


void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    // Select a numeric code for precision:
    //   3 = cutlass::float_e4m3_t  (fp8)
    //   2 = cutlass::bfloat16_t    (bf16)
    //   1 = cutlass::half_t        (fp16)
    int prec_type = 1; // default = fp16
    if (params.is_e4m3) {
        prec_type = 3;
    } else if (params.is_bf16) {
        prec_type = 2;
    }
    // TODO: no GQA switch
    PREC_SWITCH(prec_type, elem_type, [&] {
        HEADDIM_SWITCH(params.d, kHeadDim, [&] {
            // run_mha_fwd_<elem_type, kHeadDim>(params, stream);
            if(!params.use_gqa_packing) {
                run_mha_fwd_<elem_type, kHeadDim>(params, stream);
            } else {
                QUERYHEAD_SWITCH(params.h_h_k_ratio, kBlockH, [&] {
                    run_mha_fwd_gqa_<elem_type, kHeadDim, kBlockH>(params, stream);
                });
            }
        });
        
    });
}

extern "C" void run_mha(
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
    uint32_t total_k
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
    params.alibi_slopes_ptr = alibi_slopes_ptr;

    // All stride are in elements, not bytes.
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

    // Set the dimensions.
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

    // Set the different scale values.
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    __half scale_softmax_log2_half = __float2half(params.scale_softmax_log2);
    __half2 scale_softmax_log2_half2 = __half2(scale_softmax_log2_half, scale_softmax_log2_half);
    params.scale_softmax_log2_half2 = reinterpret_cast<uint32_t&>(scale_softmax_log2_half2);

    params.p_dropout = 1.; // probability to keep
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    params.is_bf16 = is_bf16;
    params.cu_seqlens_q = cu_seqlens_q_ptr;
    params.cu_seqlens_k = cu_seqlens_k_ptr;
    params.p_ptr = nullptr; // used for `return_softmax`.
    params.seqused_q = nullptr;
    params.seqused_k = nullptr;

    params.is_causal = is_causal;
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    params.num_splits = 0;
    params.page_block_size = -1;

    params.total_q = total_q;
    params.total_k = total_k;

    params.unpadded_lse = unpadded_lse;
    params.use_gqa_packing = use_gqa_packing;

    // print_params(params);
    
    cudaStream_t stream = 0; // Use the default stream.
    run_mha_fwd(params, stream);
}