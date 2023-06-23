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

template<typename T>
__device__ T gelu_fwd(T x) {
    T x_sq = x * x;
    T x_cube = x_sq * x;
    T alpha = x + static_cast<T>(0.044715) * x_cube;
    return static_cast<T>(0.5) * x * (static_cast<T>(1.0) + tanhg(static_cast<T>(M_2_SQRTPI * M_SQRT1_2) * alpha));
}


#if __CUDA_ARCH__ >= 530
UNARY_OP(__half, ucopy_f16, x)
UNARY_OP(__half, uneg_f16, -x)
UNARY_OP(__half, usqr_f16, x*x)
UNARY_OP(__half, usqrt_f16, sqrtg(x))
UNARY_OP(__half, gelu_f16, gelu_fwd(x))
#endif

UNARY_OP(float, ucopy_f32, x)
UNARY_OP(double, ucopy_f64, x)
UNARY_OP(float, uneg_f32, -x)
UNARY_OP(double, uneg_f64, -x)
UNARY_OP(float, usqr_f32, x*x)
UNARY_OP(double, usqr_f64, x*x)
UNARY_OP(float, usqrt_f32, sqrtg(x))
UNARY_OP(double, usqrt_f64, sqrtg(x))
UNARY_OP(float, gelu_f32, gelu_fwd(x))
UNARY_OP(double, gelu_f64, gelu_fwd(x))
