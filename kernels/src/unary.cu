#include "cuda_utils.cuh"

#define UNARY_OP(TYPENAME, FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const TYPENAME *inp, \
    TYPENAME *out \
) { \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
        TYPENAME x = inp ? inp[i] : out[i]; \
        out[i] = FUNC; \
    } \
} \

#if __CUDA_ARCH__ >= 530
UNARY_OP(__half, uneg_f16, -x)
UNARY_OP(__half, usqr_f16, x*x)
UNARY_OP(__half, usqrt_f16, sqrtg(x))
#endif

UNARY_OP(float, uneg_f32, -x)
UNARY_OP(float, uneg_f64, -x)
UNARY_OP(float, usqr_f32, x*x)
UNARY_OP(float, usqr_f64, x*x)
UNARY_OP(float, usqrt_f32, sqrtg(x))
UNARY_OP(float, usqrt_f64, sqrtg(x))
