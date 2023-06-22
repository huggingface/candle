#include "binary_op_macros.cuh"

#if __CUDA_ARCH__ >= 530
BINARY_OP(__half, badd_f16, x + y)
#endif

BINARY_OP(float, badd_f32, x + y)
BINARY_OP(double, badd_fwd_f64, x + y);
