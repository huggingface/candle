//! Swin Transformer backbone implementation for BiRefNet
//!
//! This module implements the Swin Transformer architecture used as the backbone
//! for feature extraction in BiRefNet.

use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{layer_norm, Conv2d, Conv2dConfig, LayerNorm, Linear, VarBuilder};

/// Image to Patch Embedding
///
/// Splits input image into patches and projects them to embedding dimension.
#[derive(Debug, Clone)]
pub struct PatchEmbed {
    proj: Conv2d,
    norm: Option<LayerNorm>,
    patch_size: (usize, usize),
}

impl PatchEmbed {
    pub fn new(
        in_channels: usize,
        embed_dim: usize,
        patch_size: usize,
        use_norm: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let proj = candle_nn::conv2d(
            in_channels,
            embed_dim,
            patch_size,
            Conv2dConfig {
                stride: patch_size,
                ..Default::default()
            },
            vb.pp("proj"),
        )?;

        let norm = if use_norm {
            Some(layer_norm(embed_dim, 1e-5, vb.pp("norm"))?)
        } else {
            None
        };

        Ok(Self {
            proj,
            norm,
            patch_size: (patch_size, patch_size),
        })
    }
}

impl Module for PatchEmbed {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = xs.dims4()?;

        // Pad to multiple of patch_size
        let pad_h = (self.patch_size.0 - h % self.patch_size.0) % self.patch_size.0;
        let pad_w = (self.patch_size.1 - w % self.patch_size.1) % self.patch_size.1;

        let xs = if pad_h > 0 || pad_w > 0 {
            xs.pad_with_zeros(2, 0, pad_w)?
                .pad_with_zeros(3, 0, pad_h)?
        } else {
            xs.clone()
        };

        // Conv projection: (B, C, H, W) -> (B, embed_dim, H/patch_size, W/patch_size)
        let xs = self.proj.forward(&xs)?;

        // Optional LayerNorm
        if let Some(norm) = &self.norm {
            let (b, c, wh, ww) = xs.dims4()?;
            // (B, C, H, W) -> (B, H*W, C) -> norm -> (B, H*W, C) -> (B, C, H, W)
            let xs = xs.flatten(2, 3)?.transpose(1, 2)?;
            let xs = norm.forward(&xs)?;
            xs.transpose(1, 2)?.reshape((b, c, wh, ww))
        } else {
            Ok(xs)
        }
    }
}

/// Window partition function
///
/// Partitions feature map into non-overlapping windows.
///
/// # Arguments
/// * `xs` - Input tensor (B, H, W, C)
/// * `window_size` - Size of each window
///
/// # Returns
/// (windows, (Hp, Wp)) where windows has shape (num_windows*B, window_size, window_size, C)
fn window_partition(xs: &Tensor, window_size: usize) -> Result<(Tensor, (usize, usize))> {
    let (b, h, w, c) = xs.dims4()?;
    let pad_h = (window_size - h % window_size) % window_size;
    let pad_w = (window_size - w % window_size) % window_size;

    let xs = if pad_h > 0 {
        xs.pad_with_zeros(1, 0, pad_h)?
    } else {
        xs.clone()
    };
    let xs = if pad_w > 0 {
        xs.pad_with_zeros(2, 0, pad_w)?
    } else {
        xs
    };

    let (h_p, w_p) = (h + pad_h, w + pad_w);
    let windows = xs
        .reshape((
            b,
            h_p / window_size,
            window_size,
            w_p / window_size,
            window_size,
            c,
        ))?
        .transpose(2, 3)?
        .contiguous()?
        .flatten_to(2)?;
    Ok((windows, (h_p, w_p)))
}

/// Window reverse function
///
/// Merges windows back into feature map.
fn window_reverse(
    windows: &Tensor,
    window_size: usize,
    (h_p, w_p): (usize, usize),
    (h, w): (usize, usize),
) -> Result<Tensor> {
    let b = windows.dim(0)? / (h_p * w_p / window_size / window_size);
    let c = windows.elem_count() / windows.dim(0)? / window_size / window_size;
    let xs = windows
        .reshape((
            b,
            h_p / window_size,
            w_p / window_size,
            window_size,
            window_size,
            c,
        ))?
        .transpose(2, 3)?
        .contiguous()?
        .reshape((b, h_p, w_p, c))?;
    let xs = if h_p > h { xs.narrow(1, 0, h)? } else { xs };
    let xs = if w_p > w { xs.narrow(2, 0, w)? } else { xs };
    Ok(xs)
}

/// Multi-Layer Perceptron for Swin Transformer
#[derive(Debug, Clone)]
pub struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    pub fn new(
        in_features: usize,
        hidden_features: usize,
        out_features: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let fc1 = candle_nn::linear(in_features, hidden_features, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(hidden_features, out_features, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?;
        let xs = xs.gelu_erf()?;
        self.fc2.forward(&xs)
    }
}

/// Window Attention module
#[derive(Debug, Clone)]
pub struct WindowAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    scale: f64,
    window_size: (usize, usize),
    relative_position_bias_table: Tensor,
    relative_position_index: Tensor,
}

impl WindowAttention {
    pub fn new(
        dim: usize,
        window_size: (usize, usize),
        num_heads: usize,
        qkv_bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = dim / num_heads;
        let scale = (head_dim as f64).powf(-0.5);

        let qkv = if qkv_bias {
            candle_nn::linear(dim, dim * 3, vb.pp("qkv"))?
        } else {
            candle_nn::linear_no_bias(dim, dim * 3, vb.pp("qkv"))?
        };
        let proj = candle_nn::linear(dim, dim, vb.pp("proj"))?;

        // Relative position bias table
        let num_relative_positions = (2 * window_size.0 - 1) * (2 * window_size.1 - 1);
        let relative_position_bias_table = vb.get(
            (num_relative_positions, num_heads),
            "relative_position_bias_table",
        )?;

        // Compute relative position index
        let relative_position_index =
            Self::compute_relative_position_index(window_size, vb.device())?;

        Ok(Self {
            qkv,
            proj,
            num_heads,
            scale,
            window_size,
            relative_position_bias_table,
            relative_position_index,
        })
    }

    fn compute_relative_position_index(
        window_size: (usize, usize),
        device: &Device,
    ) -> Result<Tensor> {
        let (wh, ww) = window_size;

        // Create coordinate grids
        let coords_h = Tensor::arange(0i64, wh as i64, device)?;
        let coords_w = Tensor::arange(0i64, ww as i64, device)?;

        // Stack coordinates
        let coords_h = coords_h.unsqueeze(1)?.broadcast_as((wh, ww))?;
        let coords_w = coords_w.unsqueeze(0)?.broadcast_as((wh, ww))?;

        let coords_h_flat = coords_h.flatten_all()?;
        let coords_w_flat = coords_w.flatten_all()?;
        let coords = Tensor::stack(&[coords_h_flat, coords_w_flat], 0)?;

        // Compute relative coordinates
        let coords_flatten = coords.clone();
        let relative_coords = coords_flatten
            .unsqueeze(2)?
            .broadcast_sub(&coords_flatten.unsqueeze(1)?)?;
        let relative_coords = relative_coords.permute((1, 2, 0))?;

        // Shift to start from 0
        let relative_coords_0 =
            ((relative_coords.i((.., .., 0))? + (wh as f64 - 1.0))? * (2.0 * ww as f64 - 1.0))?;
        let relative_coords_1 = (relative_coords.i((.., .., 1))? + (ww as f64 - 1.0))?;
        let relative_position_index = (relative_coords_0 + relative_coords_1)?;

        relative_position_index.to_dtype(DType::U32)
    }

    /// Forward with optional attention mask for SW-MSA
    pub fn forward_with_mask(&self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b_, n, c) = xs.dims3()?;
        let qkv = self
            .qkv
            .forward(xs)?
            .reshape((b_, n, 3, self.num_heads, c / self.num_heads))?
            .permute((2, 0, 3, 1, 4))?;

        let q = qkv.i(0)?;
        let k = qkv.i(1)?;
        let v = qkv.i(2)?;

        let q = (q * self.scale)?;
        let attn = q.matmul(&k.t()?)?;

        // Add relative position bias
        let wh_ww = self.window_size.0 * self.window_size.1;
        let relative_position_bias = self
            .relative_position_bias_table
            .index_select(&self.relative_position_index.flatten_all()?, 0)?
            .reshape((wh_ww, wh_ww, self.num_heads))?
            .permute((2, 0, 1))?;

        let attn = attn.broadcast_add(&relative_position_bias.unsqueeze(0)?)?;

        // Apply attention mask (SW-MSA)
        let attn = if let Some(mask) = mask {
            let nw = mask.dim(0)?;
            let attn = attn.reshape((b_ / nw, nw, self.num_heads, n, n))?;
            let attn = attn.broadcast_add(&mask.unsqueeze(1)?.unsqueeze(0)?)?;
            attn.reshape((b_, self.num_heads, n, n))?
        } else {
            attn
        };

        let attn = candle_nn::ops::softmax_last_dim(&attn)?;

        let x = attn.matmul(&v)?.transpose(1, 2)?.reshape((b_, n, c))?;

        self.proj.forward(&x)
    }
}

impl Module for WindowAttention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_with_mask(xs, None)
    }
}

/// Swin Transformer Block
#[derive(Debug, Clone)]
pub struct SwinTransformerBlock {
    norm1: LayerNorm,
    attn: WindowAttention,
    norm2: LayerNorm,
    mlp: Mlp,
    window_size: usize,
    shift_size: usize,
}

impl SwinTransformerBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dim: usize,
        num_heads: usize,
        window_size: usize,
        shift_size: usize,
        mlp_ratio: f64,
        qkv_bias: bool,
        layer_norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm1 = layer_norm(dim, layer_norm_eps, vb.pp("norm1"))?;
        let attn = WindowAttention::new(
            dim,
            (window_size, window_size),
            num_heads,
            qkv_bias,
            vb.pp("attn"),
        )?;
        let norm2 = layer_norm(dim, layer_norm_eps, vb.pp("norm2"))?;
        let mlp_hidden_dim = (dim as f64 * mlp_ratio) as usize;
        let mlp = Mlp::new(dim, mlp_hidden_dim, dim, vb.pp("mlp"))?;

        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
            window_size,
            shift_size,
        })
    }

    /// Forward pass with spatial dimensions and attention mask
    pub fn forward_with_size(
        &self,
        xs: &Tensor,
        h: usize,
        w: usize,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, l, c) = xs.dims3()?;
        assert_eq!(l, h * w, "input feature has wrong size");

        let shortcut = xs.clone();
        let xs = self.norm1.forward(xs)?;
        let xs = xs.reshape((b, h, w, c))?;

        // Pad to multiple of window_size
        let pad_r = (self.window_size - w % self.window_size) % self.window_size;
        let pad_b = (self.window_size - h % self.window_size) % self.window_size;

        let xs = if pad_r > 0 || pad_b > 0 {
            xs.pad_with_zeros(2, 0, pad_r)?
                .pad_with_zeros(1, 0, pad_b)?
        } else {
            xs
        };

        let (_, hp, wp, _) = xs.dims4()?;

        // Cyclic shift for SW-MSA using Candle's native roll function
        let (shifted_x, mask) = if self.shift_size > 0 {
            let shift = -(self.shift_size as i32);
            let shifted = xs.roll(shift, 1)?.roll(shift, 2)?;
            (shifted, attn_mask)
        } else {
            (xs, None)
        };

        // Window partition
        let (x_windows, _) = window_partition(&shifted_x, self.window_size)?;
        let x_windows = x_windows.reshape(((), self.window_size * self.window_size, c))?;

        // W-MSA/SW-MSA with mask
        let attn_windows = self.attn.forward_with_mask(&x_windows, mask)?;

        // Merge windows
        let attn_windows = attn_windows.reshape(((), self.window_size, self.window_size, c))?;
        let shifted_x = window_reverse(&attn_windows, self.window_size, (hp, wp), (hp, wp))?;

        // Reverse cyclic shift
        let xs = if self.shift_size > 0 {
            let shift = self.shift_size as i32;
            shifted_x.roll(shift, 1)?.roll(shift, 2)?
        } else {
            shifted_x
        };

        // Remove padding
        let xs = if pad_r > 0 || pad_b > 0 {
            xs.narrow(1, 0, h)?.narrow(2, 0, w)?
        } else {
            xs
        };

        let xs = xs.reshape((b, h * w, c))?;

        // FFN with residual connections (DropPath omitted in inference)
        let xs = (xs + shortcut)?;
        let mlp_out = self.mlp.forward(&self.norm2.forward(&xs)?)?;
        &xs + mlp_out
    }
}

/// Patch Merging layer for downsampling
#[derive(Debug, Clone)]
pub struct PatchMerging {
    reduction: Linear,
    norm: LayerNorm,
}

impl PatchMerging {
    pub fn new(dim: usize, layer_norm_eps: f64, vb: VarBuilder) -> Result<Self> {
        let reduction = candle_nn::linear_no_bias(4 * dim, 2 * dim, vb.pp("reduction"))?;
        let norm = layer_norm(4 * dim, layer_norm_eps, vb.pp("norm"))?;
        Ok(Self { reduction, norm })
    }

    pub fn forward_with_size(&self, xs: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let (b, l, c) = xs.dims3()?;
        assert_eq!(l, h * w, "input feature has wrong size");

        let xs = xs.reshape((b, h, w, c))?;

        // Pad to even dimensions
        let xs = if h % 2 == 1 || w % 2 == 1 {
            let pad_h = h % 2;
            let pad_w = w % 2;
            xs.pad_with_zeros(1, 0, pad_h)?
                .pad_with_zeros(2, 0, pad_w)?
        } else {
            xs
        };

        let (_, hp, wp, _) = xs.dims4()?;

        // Downsample: take 2x2 regions using reshape + narrow
        let xs_reshaped = xs.reshape((b, hp / 2, 2, wp / 2, 2, c))?;

        // x0 = x[:, 0::2, 0::2, :]
        let x0 = xs_reshaped
            .narrow(2, 0, 1)?
            .narrow(4, 0, 1)?
            .reshape((b, hp / 2, wp / 2, c))?;

        // x1 = x[:, 1::2, 0::2, :]
        let x1 = xs_reshaped
            .narrow(2, 1, 1)?
            .narrow(4, 0, 1)?
            .reshape((b, hp / 2, wp / 2, c))?;

        // x2 = x[:, 0::2, 1::2, :]
        let x2 = xs_reshaped
            .narrow(2, 0, 1)?
            .narrow(4, 1, 1)?
            .reshape((b, hp / 2, wp / 2, c))?;

        // x3 = x[:, 1::2, 1::2, :]
        let x3 = xs_reshaped
            .narrow(2, 1, 1)?
            .narrow(4, 1, 1)?
            .reshape((b, hp / 2, wp / 2, c))?;

        // Concatenate: B H/2 W/2 4*C
        let xs = Tensor::cat(&[x0, x1, x2, x3], D::Minus1)?;
        let xs = xs.flatten(1, 2)?; // B H/2*W/2 4*C

        let xs = self.norm.forward(&xs)?;
        self.reduction.forward(&xs)
    }
}

/// Basic Layer containing multiple Swin Transformer blocks
#[derive(Debug, Clone)]
pub struct BasicLayer {
    blocks: Vec<SwinTransformerBlock>,
    downsample: Option<PatchMerging>,
    window_size: usize,
    shift_size: usize,
}

impl BasicLayer {
    pub fn new(
        dim: usize,
        depth: usize,
        num_heads: usize,
        window_size: usize,
        downsample: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let shift_size = window_size / 2;

        let blocks: Vec<_> = (0..depth)
            .map(|i| {
                SwinTransformerBlock::new(
                    dim,
                    num_heads,
                    window_size,
                    if i % 2 == 0 { 0 } else { shift_size }, // Alternate W-MSA and SW-MSA
                    4.0,                                     // mlp_ratio
                    true,                                    // qkv_bias
                    1e-5,                                    // layer_norm_eps
                    vb.pp(format!("blocks.{}", i)),
                )
            })
            .collect::<Result<_>>()?;

        let downsample = if downsample {
            Some(PatchMerging::new(dim, 1e-5, vb.pp("downsample"))?)
        } else {
            None
        };

        Ok(Self {
            blocks,
            downsample,
            window_size,
            shift_size,
        })
    }

    /// Compute attention mask for SW-MSA
    fn compute_attention_mask(
        &self,
        hp: usize,
        wp: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        // Create img_mask: (1, Hp, Wp, 1)
        let mut img_mask_data = vec![0f32; hp * wp];

        // Define slice regions - convert negative indices to positive
        let h_slices = [
            (0, hp.saturating_sub(self.window_size)),
            (
                hp.saturating_sub(self.window_size),
                hp.saturating_sub(self.shift_size),
            ),
            (hp.saturating_sub(self.shift_size), hp),
        ];
        let w_slices = [
            (0, wp.saturating_sub(self.window_size)),
            (
                wp.saturating_sub(self.window_size),
                wp.saturating_sub(self.shift_size),
            ),
            (wp.saturating_sub(self.shift_size), wp),
        ];

        // Fill mask values - each region gets a different cnt value
        let mut cnt = 0f32;
        for (h_start, h_end) in &h_slices {
            for (w_start, w_end) in &w_slices {
                for h in *h_start..*h_end {
                    for w in *w_start..*w_end {
                        img_mask_data[h * wp + w] = cnt;
                    }
                }
                cnt += 1.0;
            }
        }

        // Create tensor: (1, Hp, Wp, 1)
        let img_mask = Tensor::from_vec(img_mask_data, (1, hp, wp, 1), device)?;

        // Window partition: (nW, window_size, window_size, 1)
        let (mask_windows, _) = window_partition(&img_mask, self.window_size)?;

        // Flatten: (nW, window_size*window_size)
        let mask_windows = mask_windows.reshape(((), self.window_size * self.window_size))?;

        // Compute attention mask
        let attn_mask = mask_windows
            .unsqueeze(1)?
            .broadcast_sub(&mask_windows.unsqueeze(2)?)?;

        // Set non-zero values to -100.0, zero values stay 0.0
        let zeros = attn_mask.zeros_like()?;
        let neg_100 = (zeros.ones_like()? * (-100.0f64))?;
        let mask_ne_zero = attn_mask.ne(&zeros)?;
        let attn_mask = mask_ne_zero.where_cond(&neg_100, &zeros)?;

        attn_mask.to_dtype(dtype)
    }

    /// Forward pass
    ///
    /// Returns (x_out, h, w, x_down, h_down, w_down)
    pub fn forward_with_size(
        &self,
        xs: &Tensor,
        h: usize,
        w: usize,
    ) -> Result<(Tensor, usize, usize, Tensor, usize, usize)> {
        // Compute padded dimensions
        let hp = h.div_ceil(self.window_size) * self.window_size;
        let wp = w.div_ceil(self.window_size) * self.window_size;

        // Compute attention mask for SW-MSA
        let attn_mask = self.compute_attention_mask(hp, wp, xs.device(), xs.dtype())?;

        let mut xs = xs.clone();
        for blk in &self.blocks {
            xs = blk.forward_with_size(&xs, h, w, Some(&attn_mask))?;
        }

        let x_out = xs.clone();

        let (x_down, h_down, w_down) = if let Some(downsample) = &self.downsample {
            let x_down = downsample.forward_with_size(&xs, h, w)?;
            (x_down, h / 2, w / 2)
        } else {
            (xs, h, w)
        };

        Ok((x_out, h, w, x_down, h_down, w_down))
    }
}

/// Swin Transformer backbone
#[derive(Debug, Clone)]
pub struct SwinTransformer {
    patch_embed: PatchEmbed,
    layers: Vec<BasicLayer>,
    norms: Vec<LayerNorm>,
    num_features: Vec<usize>,
    out_indices: Vec<usize>,
}

impl SwinTransformer {
    /// Create Swin-L configuration
    pub fn swin_v1_l(vb: VarBuilder) -> Result<Self> {
        Self::new(
            192,                 // embed_dim
            vec![2, 2, 18, 2],   // depths
            vec![6, 12, 24, 48], // num_heads
            12,                  // window_size
            vb,
        )
    }

    /// Create Swin-B configuration
    pub fn swin_v1_b(vb: VarBuilder) -> Result<Self> {
        Self::new(
            128,                // embed_dim
            vec![2, 2, 18, 2],  // depths
            vec![4, 8, 16, 32], // num_heads
            12,                 // window_size
            vb,
        )
    }

    pub fn new(
        embed_dim: usize,
        depths: Vec<usize>,
        num_heads: Vec<usize>,
        window_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let num_layers = depths.len();
        // Swin Transformer uses PatchEmbed with LayerNorm
        let patch_embed = PatchEmbed::new(3, embed_dim, 4, true, vb.pp("patch_embed"))?;

        let mut layers = Vec::with_capacity(num_layers);
        let mut norms = Vec::with_capacity(num_layers);
        let mut num_features = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let dim = embed_dim * (1 << i);
            num_features.push(dim);

            let layer = BasicLayer::new(
                dim,
                depths[i],
                num_heads[i],
                window_size,
                i < num_layers - 1, // Last layer doesn't downsample
                vb.pp(format!("layers.{}", i)),
            )?;
            layers.push(layer);

            let norm = layer_norm(dim, 1e-5, vb.pp(format!("norm{}", i)))?;
            norms.push(norm);
        }

        Ok(Self {
            patch_embed,
            layers,
            norms,
            num_features,
            out_indices: vec![0, 1, 2, 3],
        })
    }

    /// Forward pass returning multi-scale features
    pub fn forward_features(&self, xs: &Tensor) -> Result<Vec<Tensor>> {
        let xs = self.patch_embed.forward(xs)?;
        let (mut wh, mut ww) = (xs.dim(2)?, xs.dim(3)?);

        let mut outs = Vec::new();
        let mut xs = xs.flatten(2, 3)?.transpose(1, 2)?; // B, H*W, C

        for (i, layer) in self.layers.iter().enumerate() {
            let (x_out, h, w, xs_new, wh_new, ww_new) = layer.forward_with_size(&xs, wh, ww)?;

            if self.out_indices.contains(&i) {
                let x_out = self.norms[i].forward(&x_out)?;
                let out = x_out
                    .reshape(((), h, w, self.num_features[i]))?
                    .permute((0, 3, 1, 2))?
                    .contiguous()?;
                outs.push(out);
            }

            xs = xs_new;
            wh = wh_new;
            ww = ww_new;
        }

        Ok(outs)
    }
}

impl Module for SwinTransformer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let features = self.forward_features(xs)?;
        // Return the last feature map
        features
            .last()
            .cloned()
            .ok_or_else(|| candle::Error::Msg("No features produced".to_string()))
    }
}
