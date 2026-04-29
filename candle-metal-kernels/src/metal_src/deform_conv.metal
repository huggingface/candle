// Deformable Convolution 2D for Apple Silicon (Metal)
// Ported from mps-deform-conv: https://github.com/mpsops/mps-deform-conv
//
// Operations:
// 1. deformable_im2col: Convert image to columns with offset-based sampling
// 2. bilinear interpolation at arbitrary positions
// 3. Backward pass for input, offsets, and mask gradients

#include <metal_stdlib>
using namespace metal;

// Atomic float add using compare-and-swap (works on all Metal versions)
inline void atomic_add_float(device atomic_uint* addr, float value) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    float current_val = as_type<float>(expected);
    float new_val = current_val + value;
    uint new_bits = as_type<uint>(new_val);

    while (!atomic_compare_exchange_weak_explicit(
        addr, &expected, new_bits,
        memory_order_relaxed, memory_order_relaxed)) {
        current_val = as_type<float>(expected);
        new_val = current_val + value;
        new_bits = as_type<uint>(new_val);
    }
}

// =============================================================================
// Bilinear Interpolation
// =============================================================================

template<typename T>
inline T bilinear_interpolate(
    device const T* in,
    int height,
    int width,
    T h,
    T w
) {
    if (h <= T(-1) || T(height) <= h || w <= T(-1) || T(width) <= w) {
        return T(0);
    }

    int h_low = int(floor(float(h)));
    int w_low = int(floor(float(w)));
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    T lh = h - T(h_low);
    T lw = w - T(w_low);
    T hh = T(1) - lh;
    T hw = T(1) - lw;

    T v1 = T(0);
    if (h_low >= 0 && w_low >= 0)
        v1 = in[h_low * width + w_low];

    T v2 = T(0);
    if (h_low >= 0 && w_high <= width - 1)
        v2 = in[h_low * width + w_high];

    T v3 = T(0);
    if (h_high <= height - 1 && w_low >= 0)
        v3 = in[h_high * width + w_low];

    T v4 = T(0);
    if (h_high <= height - 1 && w_high <= width - 1)
        v4 = in[h_high * width + w_high];

    return hh * hw * v1 + hh * lw * v2 + lh * hw * v3 + lh * lw * v4;
}

// =============================================================================
// Coordinate Weight for Backward Pass
// =============================================================================

template<typename T>
inline T get_coordinate_weight(
    device const T* im_data,
    int height,
    int width,
    T y,
    T x,
    bool is_y_direction
) {
    int y_l = int(floor(float(y)));
    int x_l = int(floor(float(x)));
    int y_h = y_l + 1;
    int x_h = x_l + 1;

    bool valid_y_l = 0 <= y_l && y_l < height;
    bool valid_y_h = 0 <= y_h && y_h < height;
    bool valid_x_l = 0 <= x_l && x_l < width;
    bool valid_x_h = 0 <= x_h && x_h < width;

    T zero = T(0);
    T v_yx = (valid_y_l && valid_x_l) ? im_data[y_l * width + x_l] : zero;
    T v_yX = (valid_y_l && valid_x_h) ? im_data[y_l * width + x_h] : zero;
    T v_Yx = (valid_y_h && valid_x_l) ? im_data[y_h * width + x_l] : zero;
    T v_YX = (valid_y_h && valid_x_h) ? im_data[y_h * width + x_h] : zero;

    if (is_y_direction) {
        T dx = x - T(x_l);
        return dx * (v_YX - v_yX) + (T(1) - dx) * (v_Yx - v_yx);
    } else {
        T dy = y - T(y_l);
        return dy * (v_YX - v_Yx) + (T(1) - dy) * (v_yX - v_yx);
    }
}

// =============================================================================
// Forward: Deformable im2col (float32)
// =============================================================================

kernel void deformable_im2col_f32(
    device const float* input       [[buffer(0)]],
    device const float* offset      [[buffer(1)]],
    device const float* mask        [[buffer(2)]],
    device float* columns           [[buffer(3)]],
    constant int& height            [[buffer(4)]],
    constant int& width             [[buffer(5)]],
    constant int& weight_h          [[buffer(6)]],
    constant int& weight_w          [[buffer(7)]],
    constant int& pad_h             [[buffer(8)]],
    constant int& pad_w             [[buffer(9)]],
    constant int& stride_h          [[buffer(10)]],
    constant int& stride_w          [[buffer(11)]],
    constant int& dilation_h        [[buffer(12)]],
    constant int& dilation_w        [[buffer(13)]],
    constant int& batch_sz          [[buffer(14)]],
    constant int& n_in_channels     [[buffer(15)]],
    constant int& n_offset_grps     [[buffer(16)]],
    constant int& out_h             [[buffer(17)]],
    constant int& out_w             [[buffer(18)]],
    constant int& use_mask          [[buffer(19)]],
    uint gid [[thread_position_in_grid]]
) {
    int n = n_in_channels * out_h * out_w * batch_sz;
    if (int(gid) >= n) return;

    int index = int(gid);

    int out_x = index % out_w;
    int out_y = (index / out_w) % out_h;
    int out_b = (index / (out_w * out_h)) % batch_sz;
    int in_c = index / (out_w * out_h * batch_sz);
    int out_c = in_c * weight_h * weight_w;

    int c_per_offset_grp = n_in_channels / n_offset_grps;
    int grp_idx = in_c / c_per_offset_grp;

    device float* col_ptr = columns +
        (out_c * (batch_sz * out_h * out_w) + out_b * (out_h * out_w) +
         out_y * out_w + out_x);

    device const float* in_ptr = input +
        (out_b * (n_in_channels * height * width) + in_c * (height * width));

    device const float* offset_ptr = offset +
        (out_b * n_offset_grps + grp_idx) * 2 * weight_h * weight_w * out_h * out_w;

    device const float* mask_ptr = mask +
        (out_b * n_offset_grps + grp_idx) * weight_h * weight_w * out_h * out_w;

    for (int i = 0; i < weight_h; ++i) {
        for (int j = 0; j < weight_w; ++j) {
            int mask_idx = i * weight_w + j;
            int offset_idx = 2 * mask_idx;

            float mask_value = 1.0f;
            if (use_mask) {
                mask_value = mask_ptr[mask_idx * (out_h * out_w) + out_y * out_w + out_x];
            }

            float offset_h = offset_ptr[offset_idx * (out_h * out_w) + out_y * out_w + out_x];
            float offset_w = offset_ptr[(offset_idx + 1) * (out_h * out_w) + out_y * out_w + out_x];

            float y = float(out_y * stride_h - pad_h) + float(i * dilation_h) + offset_h;
            float x = float(out_x * stride_w - pad_w) + float(j * dilation_w) + offset_w;

            *col_ptr = mask_value * bilinear_interpolate(in_ptr, height, width, y, x);
            col_ptr += batch_sz * out_h * out_w;
        }
    }
}

// =============================================================================
// Forward: Deformable im2col (float16)
// =============================================================================

kernel void deformable_im2col_f16(
    device const half* input        [[buffer(0)]],
    device const half* offset       [[buffer(1)]],
    device const half* mask         [[buffer(2)]],
    device half* columns            [[buffer(3)]],
    constant int& height            [[buffer(4)]],
    constant int& width             [[buffer(5)]],
    constant int& weight_h          [[buffer(6)]],
    constant int& weight_w          [[buffer(7)]],
    constant int& pad_h             [[buffer(8)]],
    constant int& pad_w             [[buffer(9)]],
    constant int& stride_h          [[buffer(10)]],
    constant int& stride_w          [[buffer(11)]],
    constant int& dilation_h        [[buffer(12)]],
    constant int& dilation_w        [[buffer(13)]],
    constant int& batch_sz          [[buffer(14)]],
    constant int& n_in_channels     [[buffer(15)]],
    constant int& n_offset_grps     [[buffer(16)]],
    constant int& out_h             [[buffer(17)]],
    constant int& out_w             [[buffer(18)]],
    constant int& use_mask          [[buffer(19)]],
    uint gid [[thread_position_in_grid]]
) {
    int n = n_in_channels * out_h * out_w * batch_sz;
    if (int(gid) >= n) return;

    int index = int(gid);

    int out_x = index % out_w;
    int out_y = (index / out_w) % out_h;
    int out_b = (index / (out_w * out_h)) % batch_sz;
    int in_c = index / (out_w * out_h * batch_sz);
    int out_c = in_c * weight_h * weight_w;

    int c_per_offset_grp = n_in_channels / n_offset_grps;
    int grp_idx = in_c / c_per_offset_grp;

    device half* col_ptr = columns +
        (out_c * (batch_sz * out_h * out_w) + out_b * (out_h * out_w) +
         out_y * out_w + out_x);

    device const half* in_ptr = input +
        (out_b * (n_in_channels * height * width) + in_c * (height * width));

    device const half* offset_ptr = offset +
        (out_b * n_offset_grps + grp_idx) * 2 * weight_h * weight_w * out_h * out_w;

    device const half* mask_ptr = mask +
        (out_b * n_offset_grps + grp_idx) * weight_h * weight_w * out_h * out_w;

    for (int i = 0; i < weight_h; ++i) {
        for (int j = 0; j < weight_w; ++j) {
            int mask_idx = i * weight_w + j;
            int offset_idx = 2 * mask_idx;

            half mask_value = half(1.0);
            if (use_mask) {
                mask_value = mask_ptr[mask_idx * (out_h * out_w) + out_y * out_w + out_x];
            }

            half offset_h = offset_ptr[offset_idx * (out_h * out_w) + out_y * out_w + out_x];
            half offset_w = offset_ptr[(offset_idx + 1) * (out_h * out_w) + out_y * out_w + out_x];

            half y = half(out_y * stride_h - pad_h) + half(i * dilation_h) + offset_h;
            half x = half(out_x * stride_w - pad_w) + half(j * dilation_w) + offset_w;

            *col_ptr = mask_value * bilinear_interpolate(in_ptr, height, width, y, x);
            col_ptr += batch_sz * out_h * out_w;
        }
    }
}

#if defined(__HAVE_BFLOAT__)
kernel void deformable_im2col_bf16(
    device const bfloat* input      [[buffer(0)]],
    device const bfloat* offset     [[buffer(1)]],
    device const bfloat* mask       [[buffer(2)]],
    device bfloat* columns          [[buffer(3)]],
    constant int& height            [[buffer(4)]],
    constant int& width             [[buffer(5)]],
    constant int& weight_h          [[buffer(6)]],
    constant int& weight_w          [[buffer(7)]],
    constant int& pad_h             [[buffer(8)]],
    constant int& pad_w             [[buffer(9)]],
    constant int& stride_h          [[buffer(10)]],
    constant int& stride_w          [[buffer(11)]],
    constant int& dilation_h        [[buffer(12)]],
    constant int& dilation_w        [[buffer(13)]],
    constant int& batch_sz          [[buffer(14)]],
    constant int& n_in_channels     [[buffer(15)]],
    constant int& n_offset_grps     [[buffer(16)]],
    constant int& out_h             [[buffer(17)]],
    constant int& out_w             [[buffer(18)]],
    constant int& use_mask          [[buffer(19)]],
    uint gid [[thread_position_in_grid]]
) {
    int n = n_in_channels * out_h * out_w * batch_sz;
    if (int(gid) >= n) return;

    int index = int(gid);

    int out_x = index % out_w;
    int out_y = (index / out_w) % out_h;
    int out_b = (index / (out_w * out_h)) % batch_sz;
    int in_c = index / (out_w * out_h * batch_sz);
    int out_c = in_c * weight_h * weight_w;

    int c_per_offset_grp = n_in_channels / n_offset_grps;
    int grp_idx = in_c / c_per_offset_grp;

    device bfloat* col_ptr = columns +
        (out_c * (batch_sz * out_h * out_w) + out_b * (out_h * out_w) +
         out_y * out_w + out_x);

    device const bfloat* in_ptr = input +
        (out_b * (n_in_channels * height * width) + in_c * (height * width));

    device const bfloat* offset_ptr = offset +
        (out_b * n_offset_grps + grp_idx) * 2 * weight_h * weight_w * out_h * out_w;

    device const bfloat* mask_ptr = mask +
        (out_b * n_offset_grps + grp_idx) * weight_h * weight_w * out_h * out_w;

    for (int i = 0; i < weight_h; ++i) {
        for (int j = 0; j < weight_w; ++j) {
            int mask_idx = i * weight_w + j;
            int offset_idx = 2 * mask_idx;

            bfloat mask_value = bfloat(1.0);
            if (use_mask) {
                mask_value = mask_ptr[mask_idx * (out_h * out_w) + out_y * out_w + out_x];
            }

            bfloat offset_h = offset_ptr[offset_idx * (out_h * out_w) + out_y * out_w + out_x];
            bfloat offset_w = offset_ptr[(offset_idx + 1) * (out_h * out_w) + out_y * out_w + out_x];

            bfloat y = bfloat(out_y * stride_h - pad_h) + bfloat(i * dilation_h) + offset_h;
            bfloat x = bfloat(out_x * stride_w - pad_w) + bfloat(j * dilation_w) + offset_w;

            *col_ptr = mask_value * bilinear_interpolate(in_ptr, height, width, y, x);
            col_ptr += batch_sz * out_h * out_w;
        }
    }
}
#endif

// =============================================================================
// Backward: Gradient for input (col2im)
// =============================================================================

kernel void deformable_col2im_f32(
    device const float* col         [[buffer(0)]],
    device const float* offset      [[buffer(1)]],
    device const float* mask        [[buffer(2)]],
    device float* grad_im           [[buffer(3)]],
    constant int& channels          [[buffer(4)]],
    constant int& height            [[buffer(5)]],
    constant int& width             [[buffer(6)]],
    constant int& kernel_h          [[buffer(7)]],
    constant int& kernel_w          [[buffer(8)]],
    constant int& pad_h             [[buffer(9)]],
    constant int& pad_w             [[buffer(10)]],
    constant int& stride_h          [[buffer(11)]],
    constant int& stride_w          [[buffer(12)]],
    constant int& dilation_h        [[buffer(13)]],
    constant int& dilation_w        [[buffer(14)]],
    constant int& batch_sz          [[buffer(15)]],
    constant int& n_offset_grps     [[buffer(16)]],
    constant int& out_h             [[buffer(17)]],
    constant int& out_w             [[buffer(18)]],
    constant int& use_mask          [[buffer(19)]],
    uint gid [[thread_position_in_grid]]
) {
    int n = channels * kernel_h * kernel_w * out_h * out_w * batch_sz;
    if (int(gid) >= n) return;

    int index = int(gid);

    int out_x = index % out_w;
    int out_y = (index / out_w) % out_h;
    int b = (index / (out_w * out_h)) % batch_sz;
    int j = (index / (out_w * out_h * batch_sz)) % kernel_w;
    int i = (index / (out_w * out_h * batch_sz * kernel_w)) % kernel_h;
    int c = index / (out_w * out_h * batch_sz * kernel_w * kernel_h);

    int c_per_offset_grp = channels / n_offset_grps;
    int offset_grp = c / c_per_offset_grp;

    device const float* offset_ptr = offset +
        (b * n_offset_grps + offset_grp) * 2 * kernel_h * kernel_w * out_h * out_w;

    device const float* mask_ptr = mask +
        (b * n_offset_grps + offset_grp) * kernel_h * kernel_w * out_h * out_w;

    int mask_idx = i * kernel_w + j;
    int offset_idx = 2 * mask_idx;

    float offset_h = offset_ptr[offset_idx * out_h * out_w + out_y * out_w + out_x];
    float offset_w = offset_ptr[(offset_idx + 1) * out_h * out_w + out_y * out_w + out_x];

    float mask_value = 1.0f;
    if (use_mask) {
        mask_value = mask_ptr[mask_idx * out_h * out_w + out_y * out_w + out_x];
    }

    float y = float(out_y * stride_h - pad_h) + float(i * dilation_h) + offset_h;
    float x = float(out_x * stride_w - pad_w) + float(j * dilation_w) + offset_w;

    float col_val = col[index];

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int yp = int(y) + dy;
            int xp = int(x) + dx;

            if (0 <= yp && yp < height && 0 <= xp && xp < width &&
                abs(y - float(yp)) < 1.0f && abs(x - float(xp)) < 1.0f) {

                int grad_pos = ((b * channels + c) * height + yp) * width + xp;
                float weight = (1.0f - abs(y - float(yp))) * (1.0f - abs(x - float(xp)));

                atomic_add_float((device atomic_uint*)&grad_im[grad_pos], mask_value * weight * col_val);
            }
        }
    }
}

// =============================================================================
// Backward: Gradient for offsets and mask
// =============================================================================

kernel void deformable_col2im_coord_f32(
    device const float* col         [[buffer(0)]],
    device const float* im          [[buffer(1)]],
    device const float* offset      [[buffer(2)]],
    device const float* mask        [[buffer(3)]],
    device float* grad_offset       [[buffer(4)]],
    device float* grad_mask         [[buffer(5)]],
    constant int& channels          [[buffer(6)]],
    constant int& height            [[buffer(7)]],
    constant int& width             [[buffer(8)]],
    constant int& weight_h          [[buffer(9)]],
    constant int& weight_w          [[buffer(10)]],
    constant int& pad_h             [[buffer(11)]],
    constant int& pad_w             [[buffer(12)]],
    constant int& stride_h          [[buffer(13)]],
    constant int& stride_w          [[buffer(14)]],
    constant int& dilation_h        [[buffer(15)]],
    constant int& dilation_w        [[buffer(16)]],
    constant int& batch_sz          [[buffer(17)]],
    constant int& n_offset_grps     [[buffer(18)]],
    constant int& out_h             [[buffer(19)]],
    constant int& out_w             [[buffer(20)]],
    constant int& use_mask          [[buffer(21)]],
    uint gid [[thread_position_in_grid]]
) {
    int offset_channels = 2 * weight_h * weight_w * n_offset_grps;
    int n = out_h * out_w * offset_channels * batch_sz;
    if (int(gid) >= n) return;

    int index = int(gid);

    float grad_offset_val = 0.0f;
    float grad_mask_val = 0.0f;

    int w = index % out_w;
    int h = (index / out_w) % out_h;
    int w_w = (index / (out_w * out_h * 2)) % weight_w;
    int w_h = (index / (out_w * out_h * 2 * weight_w)) % weight_h;
    int c = (index / (out_w * out_h)) % offset_channels;
    int b = index / (out_w * out_h * offset_channels);

    int offset_grp = c / (2 * weight_h * weight_w);
    int col_step = weight_h * weight_w;
    int c_per_offset_grp = channels / n_offset_grps;

    device const float* col_ptr = col +
        offset_grp * c_per_offset_grp * weight_h * weight_w * batch_sz * out_w * out_h;
    device const float* im_ptr = im +
        (b * n_offset_grps + offset_grp) * c_per_offset_grp * height * width;
    device const float* offset_ptr = offset +
        (b * n_offset_grps + offset_grp) * 2 * weight_h * weight_w * out_h * out_w;
    device const float* mask_ptr = mask +
        (b * n_offset_grps + offset_grp) * weight_h * weight_w * out_h * out_w;

    int offset_c = c - offset_grp * 2 * weight_h * weight_w;
    bool is_y_direction = (offset_c % 2) == 0;

    int c_bound = c_per_offset_grp * weight_h * weight_w;

    for (int col_c = offset_c / 2; col_c < c_bound; col_c += col_step) {
        int col_pos = (((col_c * batch_sz + b) * out_h) + h) * out_w + w;

        int out_x = col_pos % out_w;
        int out_y = (col_pos / out_w) % out_h;
        int jj = (col_pos / (out_w * out_h * batch_sz)) % weight_w;
        int ii = (col_pos / (out_w * out_h * batch_sz * weight_w)) % weight_h;

        int mask_idx = ii * weight_w + jj;

        float offset_h = offset_ptr[(2 * mask_idx) * out_h * out_w + out_y * out_w + out_x];
        float offset_w = offset_ptr[(2 * mask_idx + 1) * out_h * out_w + out_y * out_w + out_x];

        float mask_value = 1.0f;
        if (use_mask) {
            mask_value = mask_ptr[mask_idx * out_h * out_w + out_y * out_w + out_x];
        }

        float y = float(out_y * stride_h - pad_h) + float(ii * dilation_h) + offset_h;
        float x = float(out_x * stride_w - pad_w) + float(jj * dilation_w) + offset_w;

        float weight = get_coordinate_weight(im_ptr, height, width, y, x, is_y_direction);
        grad_offset_val += mask_value * weight * col_ptr[col_pos];

        if (use_mask && is_y_direction) {
            grad_mask_val += col_ptr[col_pos] * bilinear_interpolate(im_ptr, height, width, y, x);
        }

        im_ptr += height * width;
    }

    grad_offset[index] = grad_offset_val;

    if (use_mask && is_y_direction) {
        int idx = ((((b * n_offset_grps + offset_grp) * weight_h + w_h) * weight_w + w_w) * out_h + h) * out_w + w;
        grad_mask[idx] = grad_mask_val;
    }
}
