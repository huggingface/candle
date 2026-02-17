/**
 * @brief Standalone compilation unit for MOE utility kernels.
 *
 * This file exists as a compilation unit for the utility kernels defined
 * in moe_utils.cuh (count_tokens_per_expert_kernel, expert_prefix_sum_kernel).
 * Including the header here causes these extern "C" __global__ kernels to be
 * compiled into the PTX output for this file.
 */
#include "moe_utils.cuh"
