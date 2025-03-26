// Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/flash_api.cpp

#include "flash_fwd_mla_kernel.h"
#include "flash_mla.h"
#include "static_switch.h"

#include <assert.h>

#include <cuda.h>
#include <cuda_bf16.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

extern "C" void get_mla_metadata(
    int32_t* seqlens_k_ptr,
    int32_t* tile_scheduler_metadata_ptr, // [num_sm_parts, TileSchedulerMetaDataSize]
    int32_t* num_splits_ptr, // [batch_size + 1]
    const int batch_size,
    const int num_sm_parts,
    const cudaStream_t stream
) {
    // This should match the logic in the MLA kernel.
    // static constexpr int block_size_m = 64; MOVED TO lib.rs
    static constexpr int block_size_n = 64;
    static constexpr int fixed_overhead_num_blocks = 5;

    Mla_metadata_params params = {};
    params.seqlens_k_ptr = seqlens_k_ptr;
    params.tile_scheduler_metadata_ptr = tile_scheduler_metadata_ptr;
    params.num_splits_ptr = num_splits_ptr;
    params.batch_size = batch_size;
    params.block_size_n = block_size_n;
    params.fixed_overhead_num_blocks = fixed_overhead_num_blocks;
    params.num_sm_parts = num_sm_parts;
    get_mla_metadata_func(params, stream);

    return;
}

extern "C" void mha_fwd_kvcache_mla(
    Flash_fwd_mla_params params,
    const cudaStream_t stream
) {
    assert(params.d == 576);
    run_mha_fwd_splitkv_mla<cute::bfloat16_t, 576>(params, stream);
}
