#ifndef __HIPCC__
#define __device__
#define __global__
#define __forceinline__
#else
#include <hip/hip_runtime.h>
#endif

#include <stddef.h>
#include <stdint.h>
#include <math.h>

__device__ bool is_contiguous(
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides
) {
    size_t acc = 1;
    for (unsigned int d = 0; d < num_dims; d++) {
        unsigned int dim_idx = num_dims - 1 - d;
        if (dims[dim_idx] > 1 && acc != strides[dim_idx]) {
            return false;
        }
        acc *= dims[dim_idx];
    }
    return true;
}

__device__ __forceinline__ float maxg_f(float a, float b) { return fmaxf(a, b); }
__device__ __forceinline__ double maxg_d(double a, double b) { return fmax(a, b); }
__device__ __forceinline__ uint8_t maxg_u8(uint8_t a, uint8_t b) { return a > b ? a : b; }
__device__ __forceinline__ uint32_t maxg_u32(uint32_t a, uint32_t b) { return a > b ? a : b; }

template <typename T, typename A>
__device__ void conv1d(
    const size_t src_numel,
    const size_t l_out,
    const size_t stride,
    const size_t padding,
    const size_t dilation,
    const size_t *info,
    const T *src,
    const T *kernel,
    T *dst
) {
    const size_t *src_dims = info;
    const size_t *src_s = info + 3;
    const size_t *k_dims = info + 6;
    const size_t *k_s = info + 9;
    const size_t dst_i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t k_size = k_dims[2];
    const size_t c_out = k_dims[0];
    const size_t c_in = src_dims[1];
    const size_t l_in = src_dims[2];
    if (dst_i >= src_dims[0] * c_out * l_out) {
        return;
    }

    const size_t b_idx = dst_i / (l_out * c_out);
    const size_t dst_c_idx = (dst_i / l_out) % c_out;
    const size_t dst_l = dst_i % l_out;

    const size_t src_idx0 = b_idx * src_s[0];
    A d = 0;
    for (size_t offset = 0; offset < k_size; ++offset) {
        size_t src_l = (stride * dst_l + offset) * dilation;
        if (src_l < padding || src_l >= padding + l_in) {
            continue;
        }
        src_l -= padding;
        for (size_t src_c_idx = 0; src_c_idx < c_in; ++src_c_idx) {
            const size_t src_idx = src_idx0 + src_c_idx * src_s[1] + src_l * src_s[2];
            const size_t k_idx = dst_c_idx * k_s[0] + src_c_idx * k_s[1] + offset * k_s[2];
            d += static_cast<A>(src[src_idx]) * static_cast<A>(kernel[k_idx]);
        }
    }
    dst[dst_i] = static_cast<T>(d);
}

#define CONV1D_OP(TYPENAME, TYPEACC, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t src_numel, \
    const size_t l_out, \
    const size_t stride, \
    const size_t padding, \
    const size_t dilation, \
    const size_t *info, \
    const TYPENAME *src, \
    const TYPENAME *kernel, \
    TYPENAME *dst \
) { \
  conv1d<TYPENAME, TYPEACC>(src_numel, l_out, stride, padding, dilation, info, src, kernel, dst); \
}

template <typename T, typename A>
__device__ void conv2d(
    const size_t src_numel,
    const size_t w_out,
    const size_t h_out,
    const size_t stride,
    const size_t padding,
    const size_t dilation,
    const size_t *info,
    const T *src,
    const T *kernel,
    T *dst
) {
    const size_t dst_i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t *src_dims = info;
    const size_t *src_s = info + 4;
    const size_t *k_dims = info + 8;
    const size_t *k_s = info + 12;
    const size_t h_k = k_dims[2];
    const size_t w_k = k_dims[3];
    const size_t c_out = k_dims[0];
    const size_t c_in = src_dims[1];
    const size_t h_in = src_dims[2];
    const size_t w_in = src_dims[3];
    if (dst_i >= src_dims[0] * c_out * w_out * h_out) {
        return;
    }

    const size_t b_idx = dst_i / (w_out * h_out * c_out);
    const size_t dst_c_idx = (dst_i / (w_out * h_out)) % c_out;
    const size_t dst_h = (dst_i / w_out) % h_out;
    const size_t dst_w = dst_i % w_out;

    const size_t src_idx0 = b_idx * src_s[0];
    A d = 0;
    for (size_t w_offset = 0; w_offset < w_k; ++w_offset) {
        size_t src_w = stride * dst_w + w_offset * dilation;
        if (src_w < padding || src_w >= w_in + padding) {
            continue;
        }
        src_w -= padding;
        for (size_t h_offset = 0; h_offset < h_k; ++h_offset) {
            size_t src_h = stride * dst_h + h_offset * dilation;
            if (src_h < padding || src_h >= h_in + padding) {
                continue;
            }
            src_h -= padding;
            for (size_t src_c_idx = 0; src_c_idx < c_in; ++src_c_idx) {
                const size_t src_idx = src_idx0 + src_c_idx * src_s[1] + src_h * src_s[2] + src_w * src_s[3];
                const size_t k_idx = dst_c_idx * k_s[0] + src_c_idx * k_s[1] + h_offset * k_s[2] + w_offset * k_s[3];
                d += static_cast<A>(src[src_idx]) * static_cast<A>(kernel[k_idx]);
            }
        }
    }
    dst[dst_i] = static_cast<T>(d);
}

#define CONV2D_OP(TYPENAME, TYPEACC, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t src_numel, \
    const size_t w_out, \
    const size_t h_out, \
    const size_t stride, \
    const size_t padding, \
    const size_t dilation, \
    const size_t *info, \
    const TYPENAME *src, \
    const TYPENAME *kernel, \
    TYPENAME *dst \
) { \
  conv2d<TYPENAME, TYPEACC>(src_numel, w_out, h_out, stride, padding, dilation, info, src, kernel, dst); \
}

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

#define CONVT1D_OP(TYPENAME, TYPEACC, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t src_numel, \
    const size_t l_out, \
    const size_t stride, \
    const size_t padding, \
    const size_t out_padding, \
    const size_t dilation, \
    const size_t *info, \
    const TYPENAME *src, \
    const TYPENAME *kernel, \
    TYPENAME *dst \
) { \
  conv_transpose1d<TYPENAME, TYPEACC>(src_numel, l_out, stride, padding, out_padding, dilation, info, src, kernel, dst); \
}

template <typename T, typename A>
__device__ void conv_transpose2d(
    const size_t src_numel,
    const size_t w_out,
    const size_t h_out,
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
    const size_t *src_s = info + 4;
    const size_t *k_dims = info + 8;
    const size_t *k_s = info + 12;
    const size_t h_k = k_dims[2];
    const size_t w_k = k_dims[3];
    const size_t c_out = k_dims[1];
    const size_t c_in = src_dims[1];
    const size_t h_in = src_dims[2];
    const size_t w_in = src_dims[3];
    if (dst_i >= src_dims[0] * c_out * w_out * h_out) {
        return;
    }

    const size_t b_idx = dst_i / (w_out * h_out * c_out);
    const size_t dst_c_idx = (dst_i / (w_out * h_out)) % c_out;
    const size_t out_y = (dst_i / w_out) % h_out;
    const size_t out_x = dst_i % w_out;

    const size_t src_idx0 = b_idx * src_s[0];
    A d = 0;
    for (int k_x = 0; k_x < (int)w_k; ++k_x) {
        int inp_x_stride = (int)(out_x + padding) - k_x * dilation;
        if (inp_x_stride < 0 || inp_x_stride % (int)stride) {
            continue;
        }
        int inp_x = inp_x_stride / (int)stride;
        if (inp_x >= (int)w_in) continue;
        for (int k_y = 0; k_y < (int)h_k; ++k_y) {
            int inp_y_stride = (int)(out_y + padding) - k_y * dilation;
            if (inp_y_stride < 0 || inp_y_stride % (int)stride) {
                continue;
            }
            int inp_y = inp_y_stride / (int)stride;
            if (inp_y >= (int)h_in) continue;
            for (size_t src_c_idx = 0; src_c_idx < c_in; ++src_c_idx) {
                const size_t src_idx = src_idx0 + src_c_idx * src_s[1] + inp_y * src_s[2] + inp_x * src_s[3];
                const size_t k_idx = src_c_idx * k_s[0] + dst_c_idx * k_s[1] + k_y * k_s[2] + k_x * k_s[3];
                d += static_cast<A>(src[src_idx]) * static_cast<A>(kernel[k_idx]);
            }
        }
    }
    dst[dst_i] = static_cast<T>(d);
}

#define CONVT2D_OP(TYPENAME, TYPEACC, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t src_numel, \
    const size_t w_out, \
    const size_t h_out, \
    const size_t stride, \
    const size_t padding, \
    const size_t out_padding, \
    const size_t dilation, \
    const size_t *info, \
    const TYPENAME *src, \
    const TYPENAME *kernel, \
    TYPENAME *dst \
) { \
  conv_transpose2d<TYPENAME, TYPEACC>(src_numel, w_out, h_out, stride, padding, out_padding, dilation, info, src, kernel, dst); \
}

template <typename T, typename A>
__device__ void avg_pool2d(
    const size_t src_numel,
    const size_t w_k,
    const size_t h_k,
    const size_t w_stride,
    const size_t h_stride,
    const size_t *info,
    const T *src,
    T *dst
) {
    const size_t dst_i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t *src_dims = info;
    const size_t *src_s = info + 4;

    const size_t c = src_dims[1];
    const size_t w_in = src_dims[2];
    const size_t h_in = src_dims[3];

    const size_t w_out = (w_in - w_k) / w_stride + 1;
    const size_t h_out = (h_in - h_k) / h_stride + 1;
    if (dst_i >= src_dims[0] * c * w_out * h_out) {
        return;
    }

    const size_t b_idx = dst_i / (w_out * h_out * c);
    const size_t c_idx = (dst_i / (w_out * h_out)) % c;
    const size_t dst_w = (dst_i / h_out) % w_out;
    const size_t dst_h = dst_i % h_out;

    const size_t src_idx0 = b_idx * src_s[0];
    const float scale = 1.0 / (w_k * h_k);
    A d = 0;
    for (size_t w_offset = 0; w_offset < w_k; ++w_offset) {
        size_t src_w = w_stride * dst_w + w_offset;
        if (src_w >= w_in) {
            continue;
        }
        for (size_t h_offset = 0; h_offset < h_k; ++h_offset) {
            size_t src_h = h_stride * dst_h + h_offset;
            if (src_h >= h_in) {
                continue;
            }
            const size_t src_idx = src_idx0 + c_idx * src_s[1] + src_w * src_s[2] + src_h * src_s[3];
            d += static_cast<A>(src[src_idx]);
        }
    }
    dst[dst_i] = static_cast<T>(d * scale);
}

#define AVG_POOL2D_OP(TYPENAME, TYPEACC, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t src_numel, \
    const size_t w_k, \
    const size_t h_k, \
    const size_t w_stride, \
    const size_t h_stride, \
    const size_t *info, \
    const TYPENAME *src, \
    TYPENAME *dst \
) { \
  avg_pool2d<TYPENAME, TYPEACC>(src_numel, w_k, h_k, w_stride, h_stride, info, src, dst); \
}

template <typename T>
__device__ void max_pool2d(
    const size_t src_numel,
    const size_t w_k,
    const size_t h_k,
    const size_t w_stride,
    const size_t h_stride,
    const size_t *info,
    const T *src,
    T *dst
) {
    const size_t dst_i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t *src_dims = info;
    const size_t *src_s = info + 4;

    const size_t c = src_dims[1];
    const size_t w_in = src_dims[2];
    const size_t h_in = src_dims[3];

    const size_t w_out = (w_in - w_k) / w_stride + 1;
    const size_t h_out = (h_in - h_k) / h_stride + 1;
    if (dst_i >= src_dims[0] * c * w_out * h_out) {
        return;
    }

    const size_t b_idx = dst_i / (w_out * h_out * c);
    const size_t c_idx = (dst_i / (w_out * h_out)) % c;
    const size_t dst_w = (dst_i / h_out) % w_out;
    const size_t dst_h = dst_i % h_out;

    const size_t src_idx0 = b_idx * src_s[0];
    T d = 0;
    bool set = false;
    for (size_t w_offset = 0; w_offset < w_k; ++w_offset) {
        size_t src_w = w_stride * dst_w + w_offset;
        if (src_w >= w_in) {
            continue;
        }
        for (size_t h_offset = 0; h_offset < h_k; ++h_offset) {
            size_t src_h = h_stride * dst_h + h_offset;
            if (src_h >= h_in) {
                continue;
            }
            const size_t src_idx = src_idx0 + c_idx * src_s[1] + src_w * src_s[2] + src_h * src_s[3];
            if (set) {
                d = (src[src_idx] > d) ? src[src_idx] : d;
            }
            else {
                d = src[src_idx];
                set = true;
            }
        }
    }
    dst[dst_i] = d;
}

#define MAX_POOL2D_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t src_numel, \
    const size_t w_k, \
    const size_t h_k, \
    const size_t w_stride, \
    const size_t h_stride, \
    const size_t *info, \
    const TYPENAME *src, \
    TYPENAME *dst \
) { \
  max_pool2d<TYPENAME>(src_numel, w_k, h_k, w_stride, h_stride, info, src, dst); \
}

template <typename T>
__device__ void upsample_nearest2d(
    const size_t w_out,
    const size_t h_out,
    const double w_scale,
    const double h_scale,
    const size_t *info,
    const T *src,
    T *dst
) {
    const size_t dst_i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t *src_dims = info;
    const size_t *src_s = info + 4;

    const size_t c = src_dims[1];
    const size_t w_in = src_dims[2];
    const size_t h_in = src_dims[3];

    if (dst_i >= src_dims[0] * c * w_out * h_out) {
        return;
    }

    const size_t b_idx = dst_i / (w_out * h_out * c);
    const size_t c_idx = (dst_i / (w_out * h_out)) % c;
    const size_t dst_w = (dst_i / h_out) % w_out;
    const size_t dst_h = dst_i % h_out;

    size_t src_w = static_cast<size_t>(dst_w * w_scale);
    size_t src_h = static_cast<size_t>(dst_h * h_scale);
    if (src_w >= w_in) {
        src_w = w_in - 1;
    }
    if (src_h >= h_in) {
        src_h = h_in - 1;
    }

    const size_t src_i = b_idx * src_s[0] + c_idx * src_s[1] + src_w * src_s[2] + src_h * src_s[3];
    dst[dst_i] = src[src_i];
}

#define UPSAMPLE_NEAREST2D_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t w_out, \
    const size_t h_out, \
    const double w_scale, \
    const double h_scale, \
    const size_t *info, \
    const TYPENAME *src, \
    TYPENAME *dst \
) { \
  upsample_nearest2d<TYPENAME>(w_out, h_out, w_scale, h_scale, info, src, dst); \
}

template <typename scalar_t>
__device__ void upsample_bilinear2d(
    const size_t w_out,
    const size_t h_out,
    const bool align_corners,
    const bool has_scale_h,
    const double scale_h_factor,
    const bool has_scale_w,
    const double scale_w_factor,
    const size_t *info,
    const scalar_t *src,
    scalar_t *dst
) {
    const size_t dst_i = blockIdx.x * blockDim.x + threadIdx.x;

    const size_t *src_dims = info;
    const size_t *src_s = info + 4;

    const size_t c = src_dims[1];
    const size_t h_in = src_dims[2];
    const size_t w_in = src_dims[3];

    if (dst_i >= src_dims[0] * c * h_out * w_out) {
        return;
    }

    const size_t b_idx = dst_i / (h_out * w_out * c);
    const size_t c_idx = (dst_i / (h_out * w_out)) % c;
    const size_t dst_h = (dst_i / w_out) % h_out;
    const size_t dst_w = dst_i % w_out;

    double h_scale, w_scale;
    if (align_corners) {
        h_scale = (h_out > 1) ? static_cast<double>(h_in - 1) / (h_out - 1) : 0.0;
        w_scale = (w_out > 1) ? static_cast<double>(w_in - 1) / (w_out - 1) : 0.0;
    } else {
        h_scale = has_scale_h ? (1.0 / scale_h_factor) : (static_cast<double>(h_in) / h_out);
        w_scale = has_scale_w ? (1.0 / scale_w_factor) : (static_cast<double>(w_in) / w_out);
    }

    double src_h_fp, src_w_fp;
    if (align_corners) {
        src_h_fp = h_scale * dst_h;
        src_w_fp = w_scale * dst_w;
    } else {
        src_h_fp = h_scale * (dst_h + 0.5) - 0.5;
        src_w_fp = w_scale * (dst_w + 0.5) - 0.5;
    }

    src_h_fp = fmax(0.0, src_h_fp);
    src_w_fp = fmax(0.0, src_w_fp);

    size_t h0 = static_cast<size_t>(floor(src_h_fp));
    size_t w0 = static_cast<size_t>(floor(src_w_fp));
    size_t h1 = min(h0 + 1, h_in - 1);
    size_t w1 = min(w0 + 1, w_in - 1);

    double weight_h = src_h_fp - h0;
    double weight_w = src_w_fp - w0;
    weight_h = fmin(fmax(weight_h, 0.0), 1.0);
    weight_w = fmin(fmax(weight_w, 0.0), 1.0);

    const size_t base = b_idx * src_s[0] + c_idx * src_s[1];

    const scalar_t v00 = src[base + h0 * src_s[2] + w0 * src_s[3]];
    const scalar_t v10 = src[base + h0 * src_s[2] + w1 * src_s[3]];
    const scalar_t v01 = src[base + h1 * src_s[2] + w0 * src_s[3]];
    const scalar_t v11 = src[base + h1 * src_s[2] + w1 * src_s[3]];

    const double v00_d = static_cast<double>(v00);
    const double v10_d = static_cast<double>(v10);
    const double v01_d = static_cast<double>(v01);
    const double v11_d = static_cast<double>(v11);

    const double v_top = v00_d * (1.0 - weight_w) + v10_d * weight_w;
    const double v_bottom = v01_d * (1.0 - weight_w) + v11_d * weight_w;
    const double value = v_top * (1.0 - weight_h) + v_bottom * weight_h;

    dst[dst_i] = static_cast<scalar_t>(value);
}

#define UPSAMPLE_BILINEAR2D_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t w_out, \
    const size_t h_out, \
    const bool align_corners, \
    const bool has_scale_h, \
    const double scale_h_factor, \
    const bool has_scale_w, \
    const double scale_w_factor, \
    const size_t *info, \
    const TYPENAME *src, \
    TYPENAME *dst \
) { \
  upsample_bilinear2d<TYPENAME>(w_out, h_out, align_corners, has_scale_h, scale_h_factor, has_scale_w, scale_w_factor, info, src, dst); \
}

CONV1D_OP(float, float, conv1d_f32)
CONV1D_OP(double, double, conv1d_f64)
CONV1D_OP(uint8_t, uint8_t, conv1d_u8)
CONV1D_OP(uint32_t, uint32_t, conv1d_u32)

CONV2D_OP(float, float, conv2d_f32)
CONV2D_OP(double, double, conv2d_f64)
CONV2D_OP(uint8_t, uint8_t, conv2d_u8)
CONV2D_OP(uint32_t, uint32_t, conv2d_u32)

CONVT1D_OP(float, float, conv_transpose1d_f32)
CONVT1D_OP(double, double, conv_transpose1d_f64)
CONVT1D_OP(uint8_t, uint8_t, conv_transpose1d_u8)
CONVT1D_OP(uint32_t, uint32_t, conv_transpose1d_u32)

CONVT2D_OP(float, float, conv_transpose2d_f32)
CONVT2D_OP(double, double, conv_transpose2d_f64)
CONVT2D_OP(uint8_t, uint8_t, conv_transpose2d_u8)
CONVT2D_OP(uint32_t, uint32_t, conv_transpose2d_u32)

AVG_POOL2D_OP(float, float, avg_pool2d_f32)
AVG_POOL2D_OP(double, double, avg_pool2d_f64)
AVG_POOL2D_OP(uint8_t, uint8_t, avg_pool2d_u8)
AVG_POOL2D_OP(uint32_t, uint32_t, avg_pool2d_u32)

MAX_POOL2D_OP(float, max_pool2d_f32)
MAX_POOL2D_OP(double, max_pool2d_f64)
MAX_POOL2D_OP(uint8_t, max_pool2d_u8)
MAX_POOL2D_OP(uint32_t, max_pool2d_u32)

UPSAMPLE_NEAREST2D_OP(float, upsample_nearest2d_f32)
UPSAMPLE_NEAREST2D_OP(double, upsample_nearest2d_f64)
UPSAMPLE_NEAREST2D_OP(uint8_t, upsample_nearest2d_u8)
UPSAMPLE_NEAREST2D_OP(uint32_t, upsample_nearest2d_u32)

UPSAMPLE_BILINEAR2D_OP(float, upsample_bilinear2d_f32)
UPSAMPLE_BILINEAR2D_OP(double, upsample_bilinear2d_f64)
UPSAMPLE_BILINEAR2D_OP(uint8_t, upsample_bilinear2d_u8)
UPSAMPLE_BILINEAR2D_OP(uint32_t, upsample_bilinear2d_u32)
