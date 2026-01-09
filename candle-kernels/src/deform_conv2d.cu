// Deformable im2col kernel for CUDA
// Reference: refs/vision/torchvision/csrc/ops/cuda/deform_conv2d_kernel.cu

#include "cuda_utils.cuh"
#include <stdint.h>

// Bilinear interpolation for deformable convolution
// Use float for intermediate calculations to avoid type ambiguity with __half and __nv_bfloat16
template <typename scalar_t>
__device__ scalar_t bilinear_interpolate_deform(
    const scalar_t* in,
    int height,
    int width,
    scalar_t h,
    scalar_t w
) {
    // Convert to float for intermediate calculations to avoid type ambiguity
    float h_f = static_cast<float>(h);
    float w_f = static_cast<float>(w);
    
    if (h_f <= -1.0f || static_cast<float>(height) <= h_f || 
        w_f <= -1.0f || static_cast<float>(width) <= w_f) {
        return scalar_t(0.0f);
    }

    int h_low = static_cast<int>(floor(h_f));
    int w_low = static_cast<int>(floor(w_f));
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    float lh = h_f - static_cast<float>(h_low);
    float lw = w_f - static_cast<float>(w_low);
    float hh = 1.0f - lh;
    float hw = 1.0f - lw;

    float v1 = 0.0f;
    if (h_low >= 0 && w_low >= 0)
        v1 = static_cast<float>(in[h_low * width + w_low]);
    float v2 = 0.0f;
    if (h_low >= 0 && w_high <= width - 1)
        v2 = static_cast<float>(in[h_low * width + w_high]);
    float v3 = 0.0f;
    if (h_high <= height - 1 && w_low >= 0)
        v3 = static_cast<float>(in[h_high * width + w_low]);
    float v4 = 0.0f;
    if (h_high <= height - 1 && w_high <= width - 1)
        v4 = static_cast<float>(in[h_high * width + w_high]);

    float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
    return static_cast<scalar_t>(w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
}

// Deformable im2col kernel
template <typename scalar_t>
__device__ void deformable_im2col_kernel(
    size_t n,
    const scalar_t* input_ptr,
    const scalar_t* offset_ptr,
    const scalar_t* mask_ptr,
    size_t height,
    size_t width,
    size_t weight_h,
    size_t weight_w,
    size_t pad_h,
    size_t pad_w,
    size_t stride_h,
    size_t stride_w,
    size_t dilation_h,
    size_t dilation_w,
    size_t batch_sz,
    size_t n_in_channels,
    size_t n_offset_grps,
    size_t out_h,
    size_t out_w,
    bool use_mask,
    scalar_t* columns_ptr
) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;

    const size_t out_x = index % out_w;
    const size_t out_y = (index / out_w) % out_h;
    const size_t out_b = (index / (out_w * out_h)) % batch_sz;
    const size_t in_c = index / (out_w * out_h * batch_sz);
    const size_t out_c = in_c * weight_h * weight_w;

    size_t c_per_offset_grp = n_in_channels / n_offset_grps;
    const size_t grp_idx = in_c / c_per_offset_grp;

    columns_ptr += (out_c * (batch_sz * out_h * out_w) 
                  + out_b * (out_h * out_w) 
                  + out_y * out_w + out_x);

    input_ptr += (out_b * (n_in_channels * height * width) 
                + in_c * (height * width));

    offset_ptr += (out_b * n_offset_grps + grp_idx) 
                * 2 * weight_h * weight_w * out_h * out_w;

    if (use_mask) {
        mask_ptr += (out_b * n_offset_grps + grp_idx) 
                  * weight_h * weight_w * out_h * out_w;
    }

    for (size_t i = 0; i < weight_h; ++i) {
        for (size_t j = 0; j < weight_w; ++j) {
            const size_t mask_idx = i * weight_w + j;
            const size_t offset_idx = 2 * mask_idx;

            float mask_value = 1.0f;
            if (use_mask) {
                mask_value = static_cast<float>(mask_ptr[mask_idx * (out_h * out_w) 
                                    + out_y * out_w + out_x]);
            }

            const float offset_h = static_cast<float>(offset_ptr[offset_idx * (out_h * out_w) 
                                               + out_y * out_w + out_x]);
            const float offset_w = static_cast<float>(offset_ptr[(offset_idx + 1) * (out_h * out_w) 
                                               + out_y * out_w + out_x]);
            
            // Use float for intermediate calculations to avoid type ambiguity with __half and __nv_bfloat16
            const float y = static_cast<float>(out_y * stride_h) - static_cast<float>(pad_h) 
                          + static_cast<float>(i * dilation_h) + offset_h;
            const float x = static_cast<float>(out_x * stride_w) - static_cast<float>(pad_w) 
                          + static_cast<float>(j * dilation_w) + offset_w;
            
            *columns_ptr = static_cast<scalar_t>(mask_value * static_cast<float>(bilinear_interpolate_deform(
                input_ptr, static_cast<int>(height), static_cast<int>(width), 
                static_cast<scalar_t>(y), static_cast<scalar_t>(x)
            )));
            columns_ptr += batch_sz * out_h * out_w;
        }
    }
}

#define DEFORM_IM2COL_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    size_t n, \
    const TYPENAME* input, \
    const TYPENAME* offset, \
    const TYPENAME* mask, \
    size_t height, \
    size_t width, \
    size_t weight_h, \
    size_t weight_w, \
    size_t pad_h, \
    size_t pad_w, \
    size_t stride_h, \
    size_t stride_w, \
    size_t dilation_h, \
    size_t dilation_w, \
    size_t batch_sz, \
    size_t n_in_channels, \
    size_t n_offset_grps, \
    size_t out_h, \
    size_t out_w, \
    bool use_mask, \
    TYPENAME* columns \
) { \
    deformable_im2col_kernel<TYPENAME>( \
        n, input, offset, mask, \
        height, width, weight_h, weight_w, \
        pad_h, pad_w, stride_h, stride_w, \
        dilation_h, dilation_w, \
        batch_sz, n_in_channels, n_offset_grps, \
        out_h, out_w, use_mask, columns \
    ); \
}

// F32 implementation
DEFORM_IM2COL_OP(float, deformable_im2col_f32)

// F64 implementation
DEFORM_IM2COL_OP(double, deformable_im2col_f64)

// F16 implementation (requires CUDA arch >= 530)
#if __CUDA_ARCH__ >= 530
DEFORM_IM2COL_OP(__half, deformable_im2col_f16)
#endif

// BF16 implementation (requires CUDA arch >= 800)
#if __CUDA_ARCH__ >= 800
DEFORM_IM2COL_OP(__nv_bfloat16, deformable_im2col_bf16)
#endif
