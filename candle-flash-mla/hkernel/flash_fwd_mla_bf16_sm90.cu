#include "flash_fwd_mla_kernel.h"

template void run_mha_fwd_splitkv_mla<cutlass::bfloat16_t, 576>(Flash_fwd_mla_params &params, cudaStream_t stream);
