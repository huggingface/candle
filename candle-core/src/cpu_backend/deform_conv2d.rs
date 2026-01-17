//! Deformable Convolution 2D CPU implementation
//! Reference: refs/vision/torchvision/csrc/ops/cpu/deform_conv2d_kernel.cpp

use crate::WithDType;

/// Bilinear interpolation (consistent with torchvision CPU implementation)
/// Source: refs/vision/torchvision/csrc/ops/cpu/deform_conv2d_kernel.cpp bilinear_interpolate
fn bilinear_interpolate<T: WithDType>(
    input: &[T],
    height: usize,
    width: usize,
    h: f64,
    w: f64,
) -> T {
    // Boundary check: return 0 if coordinate is completely outside
    if h <= -1.0 || h >= height as f64 || w <= -1.0 || w >= width as f64 {
        return T::zero();
    }

    let h_low = h.floor() as i64;
    let w_low = w.floor() as i64;
    let h_high = h_low + 1;
    let w_high = w_low + 1;

    let lh = h - h_low as f64;
    let lw = w - w_low as f64;
    let hh = 1.0 - lh;
    let hw = 1.0 - lw;

    // Get values at four corners, 0 if outside boundary
    let v1 = if h_low >= 0 && w_low >= 0 {
        input[(h_low as usize) * width + (w_low as usize)].to_f64()
    } else {
        0.0
    };
    let v2 = if h_low >= 0 && w_high <= (width - 1) as i64 {
        input[(h_low as usize) * width + (w_high as usize)].to_f64()
    } else {
        0.0
    };
    let v3 = if h_high <= (height - 1) as i64 && w_low >= 0 {
        input[(h_high as usize) * width + (w_low as usize)].to_f64()
    } else {
        0.0
    };
    let v4 = if h_high <= (height - 1) as i64 && w_high <= (width - 1) as i64 {
        input[(h_high as usize) * width + (w_high as usize)].to_f64()
    } else {
        0.0
    };

    // Bilinear interpolation weights
    let w1 = hh * hw;
    let w2 = hh * lw;
    let w3 = lh * hw;
    let w4 = lh * lw;

    T::from_f64(w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4)
}

/// Deformable im2col core algorithm
/// Source: refs/vision/torchvision/csrc/ops/cpu/deform_conv2d_kernel.cpp deformable_im2col_kernel
#[allow(clippy::too_many_arguments)]
pub fn deformable_im2col_kernel<T: WithDType>(
    input: &[T],
    offset: &[T],
    mask: Option<&[T]>,
    height: usize,
    width: usize,
    weight_h: usize,
    weight_w: usize,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    batch_sz: usize,
    n_in_channels: usize,
    n_offset_grps: usize,
    out_h: usize,
    out_w: usize,
    columns: &mut [T],
) {
    let n = n_in_channels * out_h * out_w * batch_sz;
    let c_per_offset_grp = n_in_channels / n_offset_grps;

    for index in 0..n {
        let out_x = index % out_w;
        let out_y = (index / out_w) % out_h;
        let out_b = (index / (out_w * out_h)) % batch_sz;
        let in_c = index / (out_w * out_h * batch_sz);
        let out_c = in_c * weight_h * weight_w;

        let grp_idx = in_c / c_per_offset_grp;

        // Calculate pointer offsets
        let col_base =
            out_c * (batch_sz * out_h * out_w) + out_b * (out_h * out_w) + out_y * out_w + out_x;

        let input_base = out_b * (n_in_channels * height * width) + in_c * (height * width);

        let offset_base =
            (out_b * n_offset_grps + grp_idx) * 2 * weight_h * weight_w * out_h * out_w;

        let mask_base = (out_b * n_offset_grps + grp_idx) * weight_h * weight_w * out_h * out_w;

        let input_slice = &input[input_base..input_base + height * width];

        for i in 0..weight_h {
            for j in 0..weight_w {
                let mask_idx = i * weight_w + j;
                let offset_idx = 2 * mask_idx;

                // Get mask value (DCNv2)
                let mask_value = if let Some(m) = mask {
                    m[mask_base + mask_idx * (out_h * out_w) + out_y * out_w + out_x].to_f64()
                } else {
                    1.0
                };

                // Get offset values
                let offset_h = offset
                    [offset_base + offset_idx * (out_h * out_w) + out_y * out_w + out_x]
                    .to_f64();
                let offset_w = offset
                    [offset_base + (offset_idx + 1) * (out_h * out_w) + out_y * out_w + out_x]
                    .to_f64();

                // Calculate sampling coordinates
                let y =
                    (out_y * stride_h) as f64 - pad_h as f64 + (i * dilation_h) as f64 + offset_h;
                let x =
                    (out_x * stride_w) as f64 - pad_w as f64 + (j * dilation_w) as f64 + offset_w;

                // Bilinear interpolation sampling
                let sampled = bilinear_interpolate(input_slice, height, width, y, x);

                // Write to columns
                let col_idx = col_base + (i * weight_w + j) * (batch_sz * out_h * out_w);
                columns[col_idx] = T::from_f64(mask_value * sampled.to_f64());
            }
        }
    }
}
