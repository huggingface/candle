#include "binary_op_macros.cuh"

struct BinaryAddOp {};

#if __CUDA_ARCH__ >= 530
BINARY_OP(__half, badd_fwd_f16, badd_bwd_lhs_f16, badd_bwd_rhs_f16, BinaryAddOp,
    x + y,
    1.0,
    1.0)
#endif

BINARY_OP(float, badd_fwd_f32, badd_bwd_lhs_f32, badd_bwd_rhs_f32, BinaryAddOp,
    x + y,
    1.0,
    1.0)

BINARY_OP(double, badd_fwd_f64, badd_bwd_lhs_f64, badd_bwd_rhs_f64, BinaryAddOp,
    x + y,
    1.0,
    1.0)
