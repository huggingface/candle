#include "cuda_utils.cuh"
#include<stdint.h>

#define BITWISE_OP(TYPENAME, FN_NAME, OP) \
extern "C" __global__ void FN_NAME(const TYPENAME *d_in1, const TYPENAME *d_in2, TYPENAME *d_out, \
                                    const uint32_t N) { \
  const int idx = blockIdx.x * blockDim.x + threadIdx.x; \
  if (idx < N) { \
    d_out[idx] = d_in1[idx] OP d_in2[idx]; \
  } \
} \

BITWISE_OP(uint32_t, bitwise_or_u32, |)
BITWISE_OP(uint32_t, bitwise_and_u32, &)
BITWISE_OP(uint32_t, bitwise_xor_u32, ^)

BITWISE_OP(uint8_t, bitwise_or_u8, |)
BITWISE_OP(uint8_t, bitwise_and_u8, &)
BITWISE_OP(uint8_t, bitwise_xor_u8, ^)

BITWISE_OP(int64_t, bitwise_or_i64, |)
BITWISE_OP(int64_t, bitwise_and_i64, &)
BITWISE_OP(int64_t, bitwise_xor_i64, ^)
