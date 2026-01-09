// Deformable im2col kernel for Metal
// Reference: refs/vision/torchvision/csrc/ops/mps/mps_kernels.h deformable_im2col_kernel

#include <metal_stdlib>
using namespace metal;

/// Bilinear interpolation (consistent with torchvision MPS implementation)
template <typename T>
inline T bilinear_interpolate_deform(
    constant T* input,
    int height,
    int width,
    T y,
    T x
) {
    if (y <= T(-1) || y >= T(height) || x <= T(-1) || x >= T(width)) {
        return T(0);
    }
    
    int y_low = static_cast<int>(floor(y));
    int x_low = static_cast<int>(floor(x));
    int y_high = y_low + 1;
    int x_high = x_low + 1;

    T ly = y - T(y_low);
    T lx = x - T(x_low);
    T hh = T(1) - ly;
    T hw = T(1) - lx;

    T v1 = T(0);
    if (y_low >= 0 && x_low >= 0)
        v1 = input[y_low * width + x_low];
    
    T v2 = T(0);
    if (y_low >= 0 && x_high <= width - 1)
        v2 = input[y_low * width + x_high];
    
    T v3 = T(0);
    if (y_high <= height - 1 && x_low >= 0)
        v3 = input[y_high * width + x_low];
    
    T v4 = T(0);
    if (y_high <= height - 1 && x_high <= width - 1)
        v4 = input[y_high * width + x_high];

    T w1 = hh * hw;
    T w2 = hh * lx;
    T w3 = ly * hw;
    T w4 = ly * lx;

    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

/// Deformable im2col kernel
/// Parameters are passed individually (following Candle's set_params! convention)
template<typename T>
kernel void deformable_im2col(
    constant T*           input_ptr     [[ buffer(0) ]],
    constant T*           offset_ptr    [[ buffer(1) ]],
    constant T*           mask_ptr      [[ buffer(2) ]],
    device T*             columns_ptr   [[ buffer(3) ]],
    constant uint&        height        [[ buffer(4) ]],
    constant uint&        width         [[ buffer(5) ]],
    constant uint&        weight_h      [[ buffer(6) ]],
    constant uint&        weight_w      [[ buffer(7) ]],
    constant uint&        pad_h         [[ buffer(8) ]],
    constant uint&        pad_w         [[ buffer(9) ]],
    constant uint&        stride_h      [[ buffer(10) ]],
    constant uint&        stride_w      [[ buffer(11) ]],
    constant uint&        dilation_h    [[ buffer(12) ]],
    constant uint&        dilation_w    [[ buffer(13) ]],
    constant uint&        batch_sz      [[ buffer(14) ]],
    constant uint&        n_in_channels [[ buffer(15) ]],
    constant uint&        n_offset_grps [[ buffer(16) ]],
    constant uint&        out_h         [[ buffer(17) ]],
    constant uint&        out_w         [[ buffer(18) ]],
    constant bool&        use_mask      [[ buffer(19) ]],
    uint                  tid           [[ thread_position_in_grid ]]
) {
    uint total = out_w * out_h * batch_sz * n_in_channels;
    if (tid >= total) {
        return;
    }

    uint out_x = tid % out_w;
    uint out_y = (tid / out_w) % out_h;
    uint out_b = (tid / (out_w * out_h)) % batch_sz;
    uint in_c  = tid / (out_w * out_h * batch_sz);
    uint out_c = in_c * weight_h * weight_w;
    
    uint c_per_offset_grp = n_in_channels / n_offset_grps;
    uint grp_idx = in_c / c_per_offset_grp;
    
    // Calculate pointer offsets
    uint col_offset = out_c * (batch_sz * out_h * out_w)
                    + out_b * (out_h * out_w)
                    + out_y * out_w + out_x;
    device T* local_columns_ptr = columns_ptr + col_offset;
    
    uint input_offset = out_b * (n_in_channels * height * width)
                      + in_c * (height * width);
    constant T* local_input_ptr = input_ptr + input_offset;
    
    uint offset_offset = (out_b * n_offset_grps + grp_idx) 
                       * 2 * weight_h * weight_w * out_h * out_w;
    constant T* local_offset_ptr = offset_ptr + offset_offset;
    
    constant T* local_mask_ptr = nullptr;
    uint mask_offset = 0;
    if (use_mask) {
        mask_offset = (out_b * n_offset_grps + grp_idx) 
                     * weight_h * weight_w * out_h * out_w;
        local_mask_ptr = mask_ptr + mask_offset;
    }
    
    for (uint i = 0; i < weight_h; ++i) {
        for (uint j = 0; j < weight_w; ++j) {
            uint mask_index = i * weight_w + j;
            uint offset_index = 2 * mask_index;
            
            T mask_value = T(1);
            if (use_mask) {
                mask_value = local_mask_ptr[mask_index * (out_h * out_w) 
                                          + out_y * out_w + out_x];
            }
            
            T offset_h_val = local_offset_ptr[offset_index * (out_h * out_w) 
                                            + out_y * out_w + out_x];
            T offset_w_val = local_offset_ptr[(offset_index + 1) * (out_h * out_w) 
                                            + out_y * out_w + out_x];
            
            T y = T(out_y * stride_h) - T(pad_h) + T(i * dilation_h) + offset_h_val;
            T x = T(out_x * stride_w) - T(pad_w) + T(j * dilation_w) + offset_w_val;
            
            T interp = bilinear_interpolate_deform(
                local_input_ptr, int(height), int(width), y, x
            );
            
            *local_columns_ptr = mask_value * interp;
            local_columns_ptr += batch_sz * out_h * out_w;
        }
    }
}

// Template instantiation for F32, F16, BF16
template [[host_name("deformable_im2col_f32")]]
kernel void deformable_im2col<float>(
    constant float*, constant float*, constant float*, device float*,
    constant uint&, constant uint&, constant uint&, constant uint&,
    constant uint&, constant uint&, constant uint&, constant uint&,
    constant uint&, constant uint&, constant uint&, constant uint&,
    constant uint&, constant uint&, constant uint&, constant bool&,
    uint);

template [[host_name("deformable_im2col_f16")]]
kernel void deformable_im2col<half>(
    constant half*, constant half*, constant half*, device half*,
    constant uint&, constant uint&, constant uint&, constant uint&,
    constant uint&, constant uint&, constant uint&, constant uint&,
    constant uint&, constant uint&, constant uint&, constant uint&,
    constant uint&, constant uint&, constant uint&, constant bool&,
    uint);

#if __METAL_VERSION__ >= 310
template [[host_name("deformable_im2col_bf16")]]
kernel void deformable_im2col<bfloat>(
    constant bfloat*, constant bfloat*, constant bfloat*, device bfloat*,
    constant uint&, constant uint&, constant uint&, constant uint&,
    constant uint&, constant uint&, constant uint&, constant uint&,
    constant uint&, constant uint&, constant uint&, constant uint&,
    constant uint&, constant uint&, constant uint&, constant bool&,
    uint);
#endif
