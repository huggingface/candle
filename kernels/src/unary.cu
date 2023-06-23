#include "cuda_utils.cuh"

#define UNARY_OP(TYPENAME, FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const TYPENAME *inp, \
    TYPENAME *out \
) { \
    const size_t *dims = info; \
    const size_t *strides = info + num_dims; \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
        unsigned strided_i = get_strided_index(i, num_dims, dims, strides); \
        TYPENAME x = inp ? inp[strided_i] : out[i]; \
        out[i] = FUNC; \
    } \
} \

#if __CUDA_ARCH__ >= 530
UNARY_OP(__half, ucopy_f16, x)
UNARY_OP(__half, uneg_f16, -x)
UNARY_OP(__half, usqr_f16, x*x)
UNARY_OP(__half, usqrt_f16, sqrtg(x))
#endif

UNARY_OP(float, ucopy_f32, x)
UNARY_OP(float, ucopy_f64, x)
UNARY_OP(float, uneg_f32, -x)
UNARY_OP(float, uneg_f64, -x)
UNARY_OP(float, usqr_f32, x*x)
UNARY_OP(float, usqr_f64, x*x)
UNARY_OP(float, usqrt_f32, sqrtg(x))
UNARY_OP(float, usqrt_f64, sqrtg(x))
