#include "binary_op_macros.cuh"

#if __CUDA_ARCH__ >= 530
BINARY_OP(__half, bmul_f16, x * y)
#endif

BINARY_OP(float, bmul_f32, x * y)
BINARY_OP(double, bmul_f64, x * y);
