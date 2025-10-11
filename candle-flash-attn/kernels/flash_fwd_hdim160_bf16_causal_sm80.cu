// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_<cutlass::bfloat16_t, 160, true>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim160<cutlass::bfloat16_t, true>(params, stream);
}
