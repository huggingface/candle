#include "binary_op_macros.cuh"

struct BinaryMulKernalOp {};

#if __CUDA_ARCH__ >= 530
BINARY_OP(__half, bmul_fwd_f16, bmul_bwd_lhs_f16, bmul_bwd_rhs_f16, BinaryMulKernalOp,
    x * y,
    y,
    x)
#endif

BINARY_OP(float, bmul_fwd_f32, bmul_bwd_lhs_f32, bmul_bwd_rhs_f32, BinaryMulKernalOp,
    x * y,
    y,
    x)

BINARY_OP(double, bmul_fwd_f64, bmul_bwd_lhs_f64, bmul_bwd_rhs_f64, BinaryMulKernalOp,
    x * y,
    y,
    x)
   
