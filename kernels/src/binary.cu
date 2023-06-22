#include "binary_op_macros.cuh"

#if __CUDA_ARCH__ >= 530
BINARY_OP(__half, badd_f16, x + y)
BINARY_OP(__half, bdiv_f16, x / y)
BINARY_OP(__half, bmul_f16, x * y)
BINARY_OP(__half, bsub_f16, x - y)
#endif

BINARY_OP(float, badd_f32, x + y)
BINARY_OP(double, badd_fwd_f64, x + y);
BINARY_OP(float, bdiv_f32, x / y)
BINARY_OP(double, bdiv_f64, x / y);
BINARY_OP(float, bmul_f32, x * y)
BINARY_OP(double, bmul_f64, x * y);
BINARY_OP(float, bsub_f32, x - y)
BINARY_OP(double, bsub_f64, x - y);
