#include "binary_op_macros.cuh"

struct BinaryDivOp {};

#if __CUDA_ARCH__ >= 530
BINARY_OP(__half, bdiv_fwd_f16, bdiv_bwd_lhs_f16, bdiv_bwd_rhs_f16, BinaryDivOp,
    x / y,
    recipg(y),
    -x / (y * y))
#endif

BINARY_OP(float, bdiv_fwd_f32, bdiv_bwd_lhs_f32, bdiv_bwd_rhs_f32, BinaryDivOp,
    x / y,
    recipg(y),
    -x / (y * y))

BINARY_OP(double, bdiv_fwd_f64, bdiv_bwd_lhs_f64, bdiv_bwd_rhs_f64, BinaryDivOp,
    x / y,
    recipg(y),
    -x / (y * y))
   
