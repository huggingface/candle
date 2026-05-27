// Convolution kernels for ROCm

template <typename T>
__device__ void im2col1d(
    size_t threads,
    size_t l_out,
    size_t l_k,
    size_t stride,
    size_t padding,
    size_t dilation,
    const size_t *dims,
    const size_t *strides,
    const T *src,
    T *dst
) {
    size_t b = dims[0];
    size_t c_in = dims[1];
    size_t l_in = dims[2];

    size_t src_s0 = strides[0];
    size_t src_s1 = strides[1];
    size_t src_s2 = strides[2];

    size_t dst_s0 = c_in * l_k;
    size_t dst_s1 = l_k;

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= threads) return;

    size_t out_idx = tid;
    size_t out_ch = out_idx % c_in;
    out_idx /= c_in;
    size_t l_idx = out_idx % l_out;
    out_idx /= l_out;
    size_t b_idx = out_idx;

    for (size_t k_idx = 0; k_idx < l_k; k_idx++) {
        size_t l_in_idx = l_idx * stride + k_idx * dilation;
        if (l_in_idx >= padding && l_in_idx < l_in + padding) {
            size_t src_idx = b_idx * src_s0 + out_ch * src_s1 + (l_in_idx - padding) * src_s2;
            size_t dst_idx = (b_idx * l_out + l_idx) * dst_s0 + out_ch * dst_s1 + k_idx;
            dst[dst_idx] = src[src_idx];
        } else {
            size_t dst_idx = (b_idx * l_out + l_idx) * dst_s0 + out_ch * dst_s1 + k_idx;
            dst[dst_idx] = 0;
        }
    }
}

#define IM2COL1D(T, name) \
    extern "C" __global__ void im2col1d_##name( \
        size_t threads, \
        size_t l_out, \
        size_t l_k, \
        size_t stride, \
        size_t padding, \
        size_t dilation, \
        const size_t *dims, \
        const size_t *strides, \
        const T *src, \
        T *dst \
    ) { \
        im2col1d<T>(threads, l_out, l_k, stride, padding, dilation, dims, strides, src, dst); \
    }

IM2COL1D(float, f32)
IM2COL1D(double, f64)
IM2COL1D(half, f16)

template <typename T, typename A>
__device__ void conv_transpose1d(
    const size_t src_numel,
    const size_t l_out,
    const size_t stride,
    const size_t padding,
    const size_t out_padding,
    const size_t dilation,
    const size_t *info,
    const T *src,
    const T *kernel,
    T *dst
) {
    const size_t dst_i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t *src_dims = info;
    const size_t *src_s = info + 3;
    const size_t *k_dims = info + 6;
    const size_t *k_s = info + 9;
    const size_t l_k = k_dims[2];
    const size_t c_out = k_dims[1];
    const size_t c_in = src_dims[1];
    const size_t l_in = src_dims[2];
    if (dst_i >= src_dims[0] * c_out * l_out) {
        return;
    }

    const size_t b_idx = dst_i / (l_out * c_out);
    const size_t dst_c_idx = (dst_i / l_out) % c_out;
    const size_t out_x = dst_i % l_out;

    const size_t src_idx0 = b_idx * src_s[0];
    A d = 0;
    for (int k_x = 0; k_x < (int)l_k; ++k_x) {
        int inp_x_stride = (int)(out_x + padding) - k_x * dilation;
        if (inp_x_stride < 0 || inp_x_stride % (int)stride) {
            continue;
        }
        int inp_x = inp_x_stride / (int)stride;
        if (inp_x >= (int)l_in) continue;
        for (size_t src_c_idx = 0; src_c_idx < c_in; ++src_c_idx) {
            const size_t src_idx = src_idx0 + src_c_idx * src_s[1] + inp_x * src_s[2];
            const size_t k_idx = src_c_idx * k_s[0] + dst_c_idx * k_s[1] + k_x * k_s[2];
            d += static_cast<A>(src[src_idx]) * static_cast<A>(kernel[k_idx]);
        }
    }
    dst[dst_i] = static_cast<T>(d);
}

#define CONV_TRANSPOSE1D(T, A, name) \
extern "C" __global__ void conv_transpose1d_##name( \
    const size_t src_numel, \
    const size_t l_out, \
    const size_t stride, \
    const size_t padding, \
    const size_t out_padding, \
    const size_t dilation, \
    const size_t *info, \
    const T *src, \
    const T *kernel, \
    T *dst \
) { \
    conv_transpose1d<T, A>(src_numel, l_out, stride, padding, out_padding, dilation, info, src, kernel, dst); \
}

CONV_TRANSPOSE1D(float, float, f32)
CONV_TRANSPOSE1D(double, double, f64)
CONV_TRANSPOSE1D(half, float, f16)

#include <hip/hip_bfloat16.h>
__device__ __forceinline__ float bfloat162float(hip_bfloat16 v) { return (float)v; }
__device__ __forceinline__ hip_bfloat16 float2bfloat16(float v) { return hip_bfloat16(v); }

template <>
__device__ void conv_transpose1d<hip_bfloat16, float>(
    const size_t src_numel,
    const size_t l_out,
    const size_t stride,
    const size_t padding,
    const size_t out_padding,
    const size_t dilation,
    const size_t *info,
    const hip_bfloat16 *src,
    const hip_bfloat16 *kernel,
    hip_bfloat16 *dst
) {
    const size_t dst_i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t *src_dims = info;
    const size_t *src_s = info + 3;
    const size_t *k_dims = info + 6;
    const size_t *k_s = info + 9;
    const size_t l_k = k_dims[2];
    const size_t c_out = k_dims[1];
    const size_t c_in = src_dims[1];
    const size_t l_in = src_dims[2];
    if (dst_i >= src_dims[0] * c_out * l_out) {
        return;
    }

    const size_t b_idx = dst_i / (l_out * c_out);
    const size_t dst_c_idx = (dst_i / l_out) % c_out;
    const size_t out_x = dst_i % l_out;

    const size_t src_idx0 = b_idx * src_s[0];
    float d = 0;
    for (int k_x = 0; k_x < (int)l_k; ++k_x) {
        int inp_x_stride = (int)(out_x + padding) - k_x * dilation;
        if (inp_x_stride < 0 || inp_x_stride % (int)stride) {
            continue;
        }
        int inp_x = inp_x_stride / (int)stride;
        if (inp_x >= (int)l_in) continue;
        for (size_t src_c_idx = 0; src_c_idx < c_in; ++src_c_idx) {
            const size_t src_idx = src_idx0 + src_c_idx * src_s[1] + inp_x * src_s[2];
            const size_t k_idx = src_c_idx * k_s[0] + dst_c_idx * k_s[1] + k_x * k_s[2];
            d += bfloat162float(src[src_idx]) * bfloat162float(kernel[k_idx]);
        }
    }
    dst[dst_i] = float2bfloat16(d);
}

extern "C" __global__ void conv_transpose1d_bf16(
    const size_t src_numel,
    const size_t l_out,
    const size_t stride,
    const size_t padding,
    const size_t out_padding,
    const size_t dilation,
    const size_t *info,
    const hip_bfloat16 *src,
    const hip_bfloat16 *kernel,
    hip_bfloat16 *dst
) {
    conv_transpose1d<hip_bfloat16, float>(src_numel, l_out, stride, padding, out_padding, dilation, info, src, kernel, dst);
}
