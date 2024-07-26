#ifndef _GPU_OPS_KERNELS_H_
#define _GPU_OPS_KERNELS_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include<stdlib.h>
#include<stdint.h>

namespace gpu_ops {

struct MHAParams {
  uint32_t q_batch_stride;
  uint32_t k_batch_stride;
  uint32_t v_batch_stride;
  uint32_t o_batch_stride;

  uint32_t q_row_stride;
  uint32_t k_row_stride;
  uint32_t v_row_stride;
  uint32_t o_row_stride;

  uint32_t q_head_stride;
  uint32_t k_head_stride;
  uint32_t v_head_stride;
  uint32_t o_head_stride;

  uint32_t b;
  uint32_t h;
  uint32_t h_k;
  uint32_t d;
  uint32_t d_rounded;
  float softmax_scale;
  float softcap;

  uint32_t seqlen_q;
  uint32_t seqlen_k;
  uint32_t seqlen_q_rounded;
  uint32_t seqlen_k_rounded;

  int window_size_left;
  int window_size_right;

  int is_causal;
  int is_bf16;
};

void run_mha_fwd_j(cudaStream_t stream, void **buffers,
                   const char *opaque,
                   std::size_t opaque_len);
void run_mha_bwd_j(cudaStream_t stream, void **buffers,
                   const char *opaque,
                   std::size_t opaque_len);
}

#endif
