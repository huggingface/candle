#include "binary_op_macros.cuh"

#if __CUDA_ARCH__ >= 530
BINARY_OP(__half, bsub_f16, x - y)
#endif

BINARY_OP(float, bsub_f32, x - y)
BINARY_OP(double, bsub_f64, x - y);
