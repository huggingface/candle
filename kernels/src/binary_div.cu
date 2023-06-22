#include "binary_op_macros.cuh"

#if __CUDA_ARCH__ >= 530
BINARY_OP(__half, bdiv_f16, x / y)
#endif

BINARY_OP(float, bdiv_f32, x / y)
BINARY_OP(double, bdiv_f64, x / y);
