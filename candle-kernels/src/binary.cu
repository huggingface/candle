#include "binary_op_macros.cuh"
#include<stdint.h>

#if __CUDA_ARCH__ >= 800
BINARY_OP(__nv_bfloat16, badd_bf16, x + y)
BINARY_OP(__nv_bfloat16, bdiv_bf16, x / y)
BINARY_OP(__nv_bfloat16, bmul_bf16, x * y)
BINARY_OP(__nv_bfloat16, bsub_bf16, x - y)
#endif

#if __CUDA_ARCH__ >= 530
BINARY_OP(__half, badd_f16, x + y)
BINARY_OP(__half, bdiv_f16, x / y)
BINARY_OP(__half, bmul_f16, x * y)
BINARY_OP(__half, bsub_f16, x - y)
#endif

BINARY_OP(float, badd_f32, x + y)
BINARY_OP(double, badd_f64, x + y);
BINARY_OP(uint8_t, badd_u8, x + y);
BINARY_OP(uint32_t, badd_u32, x + y);
BINARY_OP(float, bdiv_f32, x / y)
BINARY_OP(double, bdiv_f64, x / y);
BINARY_OP(uint8_t, bdiv_u8, x / y);
BINARY_OP(uint32_t, bdiv_u32, x / y);
BINARY_OP(float, bmul_f32, x * y)
BINARY_OP(double, bmul_f64, x * y);
BINARY_OP(uint8_t, bmul_u8, x * y);
BINARY_OP(uint32_t, bmul_u32, x * y);
BINARY_OP(float, bsub_f32, x - y)
BINARY_OP(double, bsub_f64, x - y);
BINARY_OP(uint8_t, bsub_u8, x - y);
BINARY_OP(uint32_t, bsub_u32, x - y);
