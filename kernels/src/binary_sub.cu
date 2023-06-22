#include "binary_op_macros.cuh"

struct BinarySubKernelOp {};

#if __CUDA_ARCH__ >= 530
BINARY_OP(__half, bsub_fwd_f16, bsub_bwd_lhs_f16, bsub_bwd_rhs_f16, BinarySubKernelOp,
    x - y,
    1.0,
    -1.0)
#endif

BINARY_OP(float, bsub_fwd_f32, bsub_bwd_lhs_f32, bsub_bwd_rhs_f32, BinarySubKernelOp,
    x - y,
    1.0,
    -1.0)

BINARY_OP(double, bsub_fwd_f64, bsub_bwd_lhs_f64, bsub_bwd_rhs_f64, BinarySubKernelOp,
    x - y,
    1.0,
    -1.0)
   
