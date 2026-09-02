// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

// hd512 (Gemma 4) uses no alibi/softcap/local; disabling shrinks the template cross-product so ptxas does not take hours
#define FLASHATTENTION_DISABLE_ALIBI
#define FLASHATTENTION_DISABLE_SOFTCAP
#define FLASHATTENTION_DISABLE_LOCAL

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_splitkv_paged_<cutlass::bfloat16_t, 512, false>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_splitkv_paged_hdim512<cutlass::bfloat16_t, false>(params, stream);
}
