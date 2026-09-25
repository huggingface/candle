//! Deformable Convolution v2 Layer.
use candle::{bail, DType, Result, Tensor};

// ------------------------------------------------------------------
// Config
// ------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeformConv2dConfig {
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
}

impl Default for DeformConv2dConfig {
    fn default() -> Self {
        Self {
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
        }
    }
}

// ------------------------------------------------------------------
// Validation (private)
// ------------------------------------------------------------------

/// Validated parameters extracted from input tensor shapes.
#[allow(dead_code)] // groups/offset_groups validated but not yet used (only =1 supported)
struct DeformConv2dParams {
    batch_size: usize,
    in_channels: usize,
    in_h: usize,
    in_w: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    out_h: usize,
    out_w: usize,
    groups: usize,
    offset_groups: usize,
}

#[allow(clippy::too_many_arguments)] // Mirrors deform_conv2d() signature
fn validate_and_extract(
    input: &Tensor,
    offset: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    mask: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Result<DeformConv2dParams> {
    let input_dims = input.dims();
    let weight_dims = weight.dims();
    let offset_dims = offset.dims();

    if input_dims.len() != 4 {
        bail!(
            "deform_conv2d: input must be 4D [B, C_in, H, W], got {}D",
            input_dims.len()
        );
    }
    if weight_dims.len() != 4 {
        bail!(
            "deform_conv2d: weight must be 4D [C_out, C_in/groups, kH, kW], got {}D",
            weight_dims.len()
        );
    }
    if offset_dims.len() != 4 {
        bail!(
            "deform_conv2d: offset must be 4D, got {}D",
            offset_dims.len()
        );
    }

    let batch_size = input_dims[0];
    let in_channels = input_dims[1];
    let in_h = input_dims[2];
    let in_w = input_dims[3];

    let out_channels = weight_dims[0];
    let kernel_h = weight_dims[2];
    let kernel_w = weight_dims[3];

    // Infer groups from tensor shapes (matches PyTorch convention)
    let channels_per_group = weight_dims[1];
    if in_channels == 0
        || channels_per_group == 0
        || !in_channels.is_multiple_of(channels_per_group)
    {
        bail!(
            "deform_conv2d: C_in ({in_channels}) must be divisible by weight.shape[1] ({channels_per_group})"
        );
    }
    let groups = in_channels / channels_per_group;

    if !out_channels.is_multiple_of(groups) {
        bail!("deform_conv2d: C_out ({out_channels}) must be divisible by groups ({groups})");
    }

    // Infer offset_groups from offset shape
    let offset_channels = offset_dims[1];
    let kernel_size = kernel_h * kernel_w;
    if kernel_size == 0 || !offset_channels.is_multiple_of(2 * kernel_size) {
        bail!(
            "deform_conv2d: offset.shape[1] ({offset_channels}) must be divisible by 2 * kH * kW ({})",
            2 * kernel_size
        );
    }
    let offset_groups = offset_channels / (2 * kernel_size);

    // Compute expected output spatial dimensions
    let (stride_h, stride_w) = stride;
    let (pad_h, pad_w) = padding;
    let (dil_h, dil_w) = dilation;

    let out_h = (in_h + 2 * pad_h - dil_h * (kernel_h - 1) - 1) / stride_h + 1;
    let out_w = (in_w + 2 * pad_w - dil_w * (kernel_w - 1) - 1) / stride_w + 1;

    if out_h == 0 || out_w == 0 {
        bail!("deform_conv2d: computed output size is zero (out_h={out_h}, out_w={out_w})");
    }

    // Validate offset spatial dims match expected output
    if offset_dims[0] != batch_size || offset_dims[2] != out_h || offset_dims[3] != out_w {
        bail!(
            "deform_conv2d: offset spatial dims ({}, {}) don't match expected ({out_h}, {out_w})",
            offset_dims[2],
            offset_dims[3]
        );
    }

    // Validate mask if present
    if let Some(m) = mask {
        let mask_dims = m.dims();
        if mask_dims.len() != 4 {
            bail!("deform_conv2d: mask must be 4D, got {}D", mask_dims.len());
        }
        let expected_mask_channels = offset_groups * kernel_size;
        if mask_dims[1] != expected_mask_channels {
            bail!(
                "deform_conv2d: mask.shape[1] ({}) must equal offset_groups * kH * kW ({expected_mask_channels})",
                mask_dims[1]
            );
        }
        if mask_dims[0] != batch_size || mask_dims[2] != out_h || mask_dims[3] != out_w {
            bail!(
                "deform_conv2d: mask spatial dims ({}, {}) don't match expected ({out_h}, {out_w})",
                mask_dims[2],
                mask_dims[3]
            );
        }
    }

    // Validate bias if present
    if let Some(b) = bias {
        let bias_dims = b.dims();
        if bias_dims.len() != 1 || bias_dims[0] != out_channels {
            bail!(
                "deform_conv2d: bias must be 1D with length C_out ({out_channels}), got shape {:?}",
                bias_dims
            );
        }
    }

    // First implementation constraints
    if dilation != (1, 1) {
        bail!(
            "deform_conv2d: dilation {:?} not yet supported, only (1, 1) is currently implemented",
            dilation
        );
    }
    if groups != 1 {
        bail!(
            "deform_conv2d: groups={groups} not yet supported, only groups=1 is currently implemented"
        );
    }
    if offset_groups != 1 {
        bail!(
            "deform_conv2d: offset_groups={offset_groups} not yet supported, only offset_groups=1 is currently implemented"
        );
    }

    Ok(DeformConv2dParams {
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        kernel_h,
        kernel_w,
        out_h,
        out_w,
        groups,
        offset_groups,
    })
}

// ------------------------------------------------------------------
// Core algorithm (public op function + private helpers)
// ------------------------------------------------------------------

/// Performs Deformable Convolution v2 (forward only).
///
/// When `mask` is `Some`, performs DCNv2 (modulated deformable convolution).
/// When `mask` is `None`, performs DCNv1.
///
/// `groups` and `offset_groups` are inferred from tensor shapes:
///   - groups = C_in / weight.shape\[1\]
///   - offset_groups = offset.shape\[1\] / (2 * kH * kW)
#[allow(clippy::too_many_arguments)] // Matches torchvision.ops.deform_conv2d signature
pub fn deform_conv2d(
    input: &Tensor,
    offset: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    mask: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Result<Tensor> {
    let params =
        validate_and_extract(input, offset, weight, bias, mask, stride, padding, dilation)?;
    let columns = deformable_im2col(input, offset, mask, &params, stride, padding)?;
    matmul_and_bias(weight, bias, &columns, &params)
}

/// Bilinear-interpolation-based im2col with deformable offsets.
///
/// Produces `[B, C_in * kH * kW, out_h * out_w]` columns tensor.
fn deformable_im2col(
    input: &Tensor,
    offset: &Tensor,
    mask: Option<&Tensor>,
    params: &DeformConv2dParams,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<Tensor> {
    let batch_size = params.batch_size;
    let in_channels = params.in_channels;
    let kernel_h = params.kernel_h;
    let kernel_w = params.kernel_w;
    let out_h = params.out_h;
    let out_w = params.out_w;

    let (stride_h, stride_w) = stride;
    let (pad_h, pad_w) = padding;
    let device = input.device();
    let dtype = input.dtype();
    let kernel_size = kernel_h * kernel_w;
    let n_samples = kernel_size * out_h * out_w;

    // Build base sampling grid: [kH*kW, out_h, out_w] for y and x coordinates
    let mut base_y_data = vec![0f64; kernel_size * out_h * out_w];
    let mut base_x_data = vec![0f64; kernel_size * out_h * out_w];

    for ky in 0..kernel_h {
        for kx in 0..kernel_w {
            let k_idx = ky * kernel_w + kx;
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let flat = k_idx * (out_h * out_w) + oy * out_w + ox;
                    base_y_data[flat] = (oy * stride_h + ky) as f64 - pad_h as f64;
                    base_x_data[flat] = (ox * stride_w + kx) as f64 - pad_w as f64;
                }
            }
        }
    }

    // [1, kH*kW, out_h, out_w] — broadcasts over batch
    let base_y =
        Tensor::from_vec(base_y_data, (1, kernel_size, out_h, out_w), device)?.to_dtype(dtype)?;
    let base_x =
        Tensor::from_vec(base_x_data, (1, kernel_size, out_h, out_w), device)?.to_dtype(dtype)?;

    // PyTorch interleaves offsets as [h0, w0, h1, w1, ...] in channel dim.
    // Extract even indices (y offsets) and odd indices (x offsets).
    let y_indices = Tensor::from_vec(
        (0..kernel_size).map(|i| (2 * i) as u32).collect::<Vec<_>>(),
        kernel_size,
        device,
    )?;
    let x_indices = Tensor::from_vec(
        (0..kernel_size)
            .map(|i| (2 * i + 1) as u32)
            .collect::<Vec<_>>(),
        kernel_size,
        device,
    )?;
    let offset_y = offset.index_select(&y_indices, 1)?; // [B, kH*kW, out_h, out_w]
    let offset_x = offset.index_select(&x_indices, 1)?;

    // Sampling coordinates
    let sample_y = base_y.broadcast_add(&offset_y)?;
    let sample_x = base_x.broadcast_add(&offset_x)?;

    // Bilinear interpolation
    let sampled = bilinear_sample(input, &sample_y, &sample_x, params)?;

    // Apply modulation mask if present
    let sampled = if let Some(m) = mask {
        // mask: [B, kH*kW, out_h, out_w] -> [B, 1, n_samples]
        let m = m.reshape((batch_size, 1, n_samples))?;
        sampled.broadcast_mul(&m)?
    } else {
        sampled
    };

    // Reshape to columns: [B, C_in * kH * kW, out_h * out_w]
    sampled.reshape((batch_size, in_channels * kernel_size, out_h * out_w))
}

/// Sample `input` at fractional `(sample_y, sample_x)` coordinates using bilinear interpolation.
///
/// Returns `[B, C_in, kH*kW * out_h * out_w]` — sampled values for each channel.
fn bilinear_sample(
    input: &Tensor,
    sample_y: &Tensor,
    sample_x: &Tensor,
    params: &DeformConv2dParams,
) -> Result<Tensor> {
    let DeformConv2dParams {
        batch_size,
        in_channels,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        out_h,
        out_w,
        ..
    } = *params;

    let device = input.device();
    let dtype = input.dtype();
    let kernel_size = kernel_h * kernel_w;
    let n_samples = kernel_size * out_h * out_w;

    // Floor/ceil for 4-corner sampling
    let y0 = sample_y.floor()?;
    let x0 = sample_x.floor()?;
    let y1 = (&y0 + 1.0)?;
    let x1 = (&x0 + 1.0)?;

    // Interpolation weights (fractional parts)
    let wy1 = sample_y.sub(&y0)?;
    let wx1 = sample_x.sub(&x0)?;
    let wy0 = (1.0 - &wy1)?;
    let wx0 = (1.0 - &wx1)?;

    let w_tl = wy0.mul(&wx0)?;
    let w_tr = wy0.mul(&wx1)?;
    let w_bl = wy1.mul(&wx0)?;
    let w_br = wy1.mul(&wx1)?;

    // Convert to integer indices for gather
    let y0_i64 = y0.to_dtype(DType::I64)?;
    let x0_i64 = x0.to_dtype(DType::I64)?;
    let y1_i64 = y1.to_dtype(DType::I64)?;
    let x1_i64 = x1.to_dtype(DType::I64)?;

    let h_max = (in_h as i64) - 1;
    let w_max = (in_w as i64) - 1;
    let zero = Tensor::zeros(y0_i64.shape(), DType::I64, device)?;
    let h_max_t = Tensor::new(h_max, device)?.broadcast_as(y0_i64.shape())?;
    let w_max_t = Tensor::new(w_max, device)?.broadcast_as(x0_i64.shape())?;

    // Boundary validity masks: 1.0 where in-bounds, 0.0 where out-of-bounds
    let valid_y0 = y0_i64
        .ge(&zero)?
        .to_dtype(dtype)?
        .mul(&y0_i64.le(&h_max_t)?.to_dtype(dtype)?)?;
    let valid_x0 = x0_i64
        .ge(&zero)?
        .to_dtype(dtype)?
        .mul(&x0_i64.le(&w_max_t)?.to_dtype(dtype)?)?;
    let valid_y1 = y1_i64
        .ge(&zero)?
        .to_dtype(dtype)?
        .mul(&y1_i64.le(&h_max_t)?.to_dtype(dtype)?)?;
    let valid_x1 = x1_i64
        .ge(&zero)?
        .to_dtype(dtype)?
        .mul(&x1_i64.le(&w_max_t)?.to_dtype(dtype)?)?;

    let mask_tl = valid_y0.mul(&valid_x0)?;
    let mask_tr = valid_y0.mul(&valid_x1)?;
    let mask_bl = valid_y1.mul(&valid_x0)?;
    let mask_br = valid_y1.mul(&valid_x1)?;

    // Clamp indices to [0, max] for safe indexing (out-of-bounds masked to 0 anyway)
    let y0_safe = y0_i64.clamp(0i64, h_max)?.to_dtype(DType::U32)?;
    let x0_safe = x0_i64.clamp(0i64, w_max)?.to_dtype(DType::U32)?;
    let y1_safe = y1_i64.clamp(0i64, h_max)?.to_dtype(DType::U32)?;
    let x1_safe = x1_i64.clamp(0i64, w_max)?.to_dtype(DType::U32)?;

    // Flatten spatial indices: idx = y * W + x
    let in_w_t = Tensor::new(in_w as u32, device)?.broadcast_as(y0_safe.shape())?;
    let idx_tl = y0_safe.mul(&in_w_t)?.add(&x0_safe)?;
    let idx_tr = y0_safe.mul(&in_w_t)?.add(&x1_safe)?;
    let idx_bl = y1_safe.mul(&in_w_t)?.add(&x0_safe)?;
    let idx_br = y1_safe.mul(&in_w_t)?.add(&x1_safe)?;

    // input: [B, C_in, H, W] -> [B, C_in, H*W]
    let input_flat = input.reshape((batch_size, in_channels, in_h * in_w))?;

    // idx: [B, kH*kW, out_h, out_w] -> [B, n_samples]
    let idx_tl_flat = idx_tl.reshape((batch_size, n_samples))?;
    let idx_tr_flat = idx_tr.reshape((batch_size, n_samples))?;
    let idx_bl_flat = idx_bl.reshape((batch_size, n_samples))?;
    let idx_br_flat = idx_br.reshape((batch_size, n_samples))?;

    // Expand idx to [B, C_in, n_samples] for gather along spatial dim
    let idx_tl_exp =
        idx_tl_flat
            .unsqueeze(1)?
            .broadcast_as((batch_size, in_channels, n_samples))?;
    let idx_tr_exp =
        idx_tr_flat
            .unsqueeze(1)?
            .broadcast_as((batch_size, in_channels, n_samples))?;
    let idx_bl_exp =
        idx_bl_flat
            .unsqueeze(1)?
            .broadcast_as((batch_size, in_channels, n_samples))?;
    let idx_br_exp =
        idx_br_flat
            .unsqueeze(1)?
            .broadcast_as((batch_size, in_channels, n_samples))?;

    // Gather 4 corner values: [B, C_in, n_samples]
    let val_tl = input_flat.gather(&idx_tl_exp.contiguous()?, 2)?;
    let val_tr = input_flat.gather(&idx_tr_exp.contiguous()?, 2)?;
    let val_bl = input_flat.gather(&idx_bl_exp.contiguous()?, 2)?;
    let val_br = input_flat.gather(&idx_br_exp.contiguous()?, 2)?;

    // Combine weights with boundary masks: [B, kH*kW, out_h, out_w] -> [B, 1, n_samples]
    let w_tl = w_tl.mul(&mask_tl)?.reshape((batch_size, 1, n_samples))?;
    let w_tr = w_tr.mul(&mask_tr)?.reshape((batch_size, 1, n_samples))?;
    let w_bl = w_bl.mul(&mask_bl)?.reshape((batch_size, 1, n_samples))?;
    let w_br = w_br.mul(&mask_br)?.reshape((batch_size, 1, n_samples))?;

    // Bilinear interpolation: weighted sum of 4 corners -> [B, C_in, n_samples]
    let result = val_tl
        .broadcast_mul(&w_tl)?
        .add(&val_tr.broadcast_mul(&w_tr)?)?
        .add(&val_bl.broadcast_mul(&w_bl)?)?
        .add(&val_br.broadcast_mul(&w_br)?)?;

    Ok(result)
}

/// Weight matmul + bias addition.
///
/// columns: `[B, C_in * kH * kW, out_h * out_w]`
/// Returns: `[B, C_out, out_h, out_w]`
fn matmul_and_bias(
    weight: &Tensor,
    bias: Option<&Tensor>,
    columns: &Tensor,
    params: &DeformConv2dParams,
) -> Result<Tensor> {
    let DeformConv2dParams {
        batch_size,
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        out_h,
        out_w,
        ..
    } = *params;

    // weight: [C_out, C_in, kH, kW] -> [C_out, C_in*kH*kW]
    let weight_flat = weight.reshape((out_channels, in_channels * kernel_h * kernel_w))?;

    // [C_out, C_in*kH*kW] x [B, C_in*kH*kW, out_h*out_w] -> [B, C_out, out_h*out_w]
    let output = weight_flat.broadcast_matmul(columns)?;

    // Reshape to spatial: [B, C_out, out_h, out_w]
    let output = output.reshape((batch_size, out_channels, out_h, out_w))?;

    if let Some(b) = bias {
        let b = b.reshape((1, out_channels, 1, 1))?;
        output.broadcast_add(&b)
    } else {
        Ok(output)
    }
}

// ------------------------------------------------------------------
// Module
// ------------------------------------------------------------------

/// Deformable Convolution v2 module owning convolution weights.
///
/// Does not implement the standard `Module` trait because `forward` requires
/// additional inputs (`offset`, `mask`) beyond a single tensor.
#[derive(Clone, Debug)]
pub struct DeformConv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    config: DeformConv2dConfig,
}

impl DeformConv2d {
    pub fn new(weight: Tensor, bias: Option<Tensor>, config: DeformConv2dConfig) -> Self {
        Self {
            weight,
            bias,
            config,
        }
    }

    pub fn config(&self) -> &DeformConv2dConfig {
        &self.config
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    /// `offset_groups` is inferred from `offset.shape[1] / (2 * kH * kW)`.
    pub fn forward(
        &self,
        input: &Tensor,
        offset: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        deform_conv2d(
            input,
            offset,
            &self.weight,
            self.bias.as_ref(),
            mask,
            self.config.stride,
            self.config.padding,
            self.config.dilation,
        )
    }
}

// ------------------------------------------------------------------
// Factory functions (named `_layer` to avoid collision with the op)
// ------------------------------------------------------------------

pub fn deform_conv2d_layer(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: DeformConv2dConfig,
    vb: crate::VarBuilder,
) -> Result<DeformConv2d> {
    let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints(
        (out_channels, in_channels, kernel_size, kernel_size),
        "weight",
        init_ws,
    )?;
    let bound = 1. / (in_channels as f64).sqrt();
    let init_bs = crate::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let bs = vb.get_with_hints(out_channels, "bias", init_bs)?;
    Ok(DeformConv2d::new(ws, Some(bs), cfg))
}

pub fn deform_conv2d_layer_no_bias(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: DeformConv2dConfig,
    vb: crate::VarBuilder,
) -> Result<DeformConv2d> {
    let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints(
        (out_channels, in_channels, kernel_size, kernel_size),
        "weight",
        init_ws,
    )?;
    Ok(DeformConv2d::new(ws, None, cfg))
}

// ------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    #[test]
    fn deform_conv2d_zero_offset_equals_conv2d() -> Result<()> {
        // When offset=0 and mask=1, deform_conv2d should equal regular conv2d
        let dev = &Device::Cpu;
        let input = Tensor::randn(0f32, 1.0, (1, 3, 8, 8), dev)?;
        let weight = Tensor::randn(0f32, 1.0, (8, 3, 3, 3), dev)?;
        let bias = Tensor::randn(0f32, 1.0, (8,), dev)?;
        let offset = Tensor::zeros((1, 18, 8, 8), DType::F32, dev)?;
        let mask = Tensor::ones((1, 9, 8, 8), DType::F32, dev)?;

        let dcn_out = deform_conv2d(
            &input,
            &offset,
            &weight,
            Some(&bias),
            Some(&mask),
            (1, 1),
            (1, 1),
            (1, 1),
        )?;
        let conv_out = input.conv2d(&weight, 1, 1, 1, 1)?;
        let conv_out = conv_out.broadcast_add(&bias.reshape((1, 8, 1, 1))?)?;

        let diff = dcn_out
            .sub(&conv_out)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(
            diff < 1e-5,
            "zero-offset deform_conv2d differs from conv2d: {diff}"
        );
        Ok(())
    }

    #[test]
    fn deform_conv2d_output_shape() -> Result<()> {
        let dev = &Device::Cpu;
        let input = Tensor::zeros((2, 3, 8, 8), DType::F32, dev)?;
        let weight = Tensor::zeros((8, 3, 3, 3), DType::F32, dev)?;
        let offset = Tensor::zeros((2, 18, 8, 8), DType::F32, dev)?;
        let result = deform_conv2d(
            &input, &offset, &weight, None, None, (1, 1), (1, 1), (1, 1),
        )?;
        assert_eq!(result.dims(), &[2, 8, 8, 8]);
        Ok(())
    }

    #[test]
    fn deform_conv2d_module_forward() -> Result<()> {
        let dev = &Device::Cpu;
        let varmap = crate::VarMap::new();
        let vb = crate::VarBuilder::from_varmap(&varmap, DType::F32, dev);
        let module = deform_conv2d_layer(
            3,
            8,
            3,
            DeformConv2dConfig {
                padding: (1, 1),
                ..Default::default()
            },
            vb,
        )?;
        let input = Tensor::randn(0f32, 1.0, (1, 3, 8, 8), dev)?;
        let offset = Tensor::zeros((1, 18, 8, 8), DType::F32, dev)?;
        let output = module.forward(&input, &offset, None)?;
        assert_eq!(output.dims(), &[1, 8, 8, 8]);
        Ok(())
    }

    #[test]
    fn deform_conv2d_zero_mask_is_bias_only() -> Result<()> {
        let dev = &Device::Cpu;
        let input = Tensor::randn(0f32, 1.0, (1, 3, 8, 8), dev)?;
        let weight = Tensor::randn(0f32, 1.0, (8, 3, 3, 3), dev)?;
        let bias = Tensor::randn(0f32, 1.0, (8,), dev)?;
        let offset = Tensor::randn(0f32, 1.0, (1, 18, 8, 8), dev)?;
        let mask = Tensor::zeros((1, 9, 8, 8), DType::F32, dev)?;

        let result = deform_conv2d(
            &input,
            &offset,
            &weight,
            Some(&bias),
            Some(&mask),
            (1, 1),
            (1, 1),
            (1, 1),
        )?;
        let expected = bias.reshape((1, 8, 1, 1))?.broadcast_as(result.shape())?;
        let diff = result
            .sub(&expected)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(
            diff < 1e-6,
            "zero-mask output should equal bias: {diff}"
        );
        Ok(())
    }
}
