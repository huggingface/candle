//! Swin Transformer v1 implementation.
//!
//! Swin Transformer is a hierarchical vision transformer using shifted windows.
//! It produces hierarchical feature maps and has linear complexity to image size.
//!
//! Key characteristics:
//! - Hierarchical feature maps (like CNNs)
//! - Shifted window attention for cross-window connections
//! - Linear complexity relative to image size
//! - Relative position bias in attention
//!
//! References:
//! - [Swin Transformer Paper](https://arxiv.org/abs/2103.14030)
//! - [Microsoft Implementation](https://github.com/microsoft/Swin-Transformer)
//! - [HuggingFace Model](https://huggingface.co/microsoft/swin-tiny-patch4-window7-224)

use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{conv2d, layer_norm, linear, Conv2d, Conv2dConfig, LayerNorm, Linear, VarBuilder};
use serde::Deserialize;

/// Represents image size as either a single value (square) or [height, width].
#[derive(Debug, Clone)]
pub struct ImageSize {
    pub height: usize,
    pub width: usize,
}

impl<'de> Deserialize<'de> for ImageSize {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, SeqAccess, Visitor};

        struct ImageSizeVisitor;

        impl<'de> Visitor<'de> for ImageSizeVisitor {
            type Value = ImageSize;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a usize or an array of two usize values [height, width]")
            }

            fn visit_u64<E>(self, value: u64) -> std::result::Result<ImageSize, E>
            where
                E: de::Error,
            {
                let size = value as usize;
                Ok(ImageSize {
                    height: size,
                    width: size,
                })
            }

            fn visit_seq<A>(self, mut seq: A) -> std::result::Result<ImageSize, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let height: usize = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let width: usize = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                Ok(ImageSize { height, width })
            }
        }

        deserializer.deserialize_any(ImageSizeVisitor)
    }
}

impl Default for ImageSize {
    fn default() -> Self {
        Self {
            height: 224,
            width: 224,
        }
    }
}

fn default_image_size() -> ImageSize {
    ImageSize::default()
}

fn default_num_classes() -> usize {
    1000
}

fn default_hidden_dropout() -> f64 {
    0.0
}

fn default_attention_dropout() -> f64 {
    0.0
}

/// Swin Transformer configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default = "default_image_size")]
    pub image_size: ImageSize,
    #[serde(default)]
    pub patch_size: usize,
    pub num_channels: usize,
    pub embed_dim: usize,
    pub depths: Vec<usize>,
    pub num_heads: Vec<usize>,
    pub window_size: usize,
    pub mlp_ratio: f64,
    #[serde(default)]
    pub qkv_bias: bool,
    #[serde(default = "default_hidden_dropout", alias = "hidden_dropout_prob")]
    pub hidden_dropout: f64,
    #[serde(
        default = "default_attention_dropout",
        alias = "attention_probs_dropout_prob"
    )]
    pub attention_dropout: f64,
    pub drop_path_rate: f64,
    #[serde(default = "default_num_classes")]
    pub num_classes: usize,
    pub layer_norm_eps: f64,
}

impl Config {
    /// Swin-Tiny configuration for 224x224 images.
    pub fn tiny_224() -> Self {
        Self {
            image_size: ImageSize {
                height: 224,
                width: 224,
            },
            patch_size: 4,
            num_channels: 3,
            embed_dim: 96,
            depths: vec![2, 2, 6, 2],
            num_heads: vec![3, 6, 12, 24],
            window_size: 7,
            mlp_ratio: 4.0,
            qkv_bias: true,
            hidden_dropout: 0.0,
            attention_dropout: 0.0,
            drop_path_rate: 0.1,
            num_classes: 1000,
            layer_norm_eps: 1e-5,
        }
    }

    /// Swin-Small configuration for 224x224 images.
    pub fn small_224() -> Self {
        Self {
            depths: vec![2, 2, 18, 2],
            ..Self::tiny_224()
        }
    }

    /// Swin-Base configuration for 224x224 images.
    pub fn base_224() -> Self {
        Self {
            embed_dim: 128,
            depths: vec![2, 2, 18, 2],
            num_heads: vec![4, 8, 16, 32],
            ..Self::tiny_224()
        }
    }

    /// Swin-Base configuration for 384x384 images.
    pub fn base_384() -> Self {
        Self {
            image_size: ImageSize {
                height: 384,
                width: 384,
            },
            window_size: 12,
            ..Self::base_224()
        }
    }

    /// Swin-Large configuration for 224x224 images.
    pub fn large_224() -> Self {
        Self {
            embed_dim: 192,
            depths: vec![2, 2, 18, 2],
            num_heads: vec![6, 12, 24, 48],
            ..Self::tiny_224()
        }
    }

    /// Swin-Large configuration for 384x384 images.
    pub fn large_384() -> Self {
        Self {
            image_size: ImageSize {
                height: 384,
                width: 384,
            },
            window_size: 12,
            ..Self::large_224()
        }
    }

    /// Donut-base Swin encoder configuration.
    /// For document understanding with 1280x960 input images.
    /// Based on Swin-B with modifications for document processing.
    pub fn donut_base() -> Self {
        Self {
            image_size: ImageSize {
                height: 1280,
                width: 960,
            },
            patch_size: 4,
            num_channels: 3,
            embed_dim: 128,
            depths: vec![2, 2, 14, 2],
            num_heads: vec![4, 8, 16, 32],
            window_size: 10,
            mlp_ratio: 4.0,
            qkv_bias: true,
            hidden_dropout: 0.0,
            attention_dropout: 0.0,
            drop_path_rate: 0.1,
            num_classes: 0, // Not used for encoder-only
            layer_norm_eps: 1e-5,
        }
    }

    /// Create config for arbitrary image size (for VisionEncoderDecoder models).
    /// Height and width should be divisible by patch_size * 2^(num_stages-1).
    pub fn with_image_size(mut self, height: usize, width: usize) -> Self {
        self.image_size = ImageSize { height, width };
        self
    }

    /// Get image height.
    pub fn image_height(&self) -> usize {
        self.image_size.height
    }

    /// Get image width.
    pub fn image_width(&self) -> usize {
        self.image_size.width
    }
}

/// Partition feature map into non-overlapping windows.
/// Input: (B, H, W, C)
/// Output: (B * num_windows, window_size, window_size, C)
fn window_partition(x: &Tensor, window_size: usize) -> Result<Tensor> {
    let (b, h, w, c) = x.dims4()?;
    let num_h = h / window_size;
    let num_w = w / window_size;

    x.reshape((b, num_h, window_size, num_w, window_size, c))?
        .permute((0, 1, 3, 2, 4, 5))?
        .contiguous()?
        .reshape((b * num_h * num_w, window_size, window_size, c))
}

/// Reverse window partition.
/// Input: (B * num_windows, window_size, window_size, C)
/// Output: (B, H, W, C)
fn window_reverse(windows: &Tensor, window_size: usize, h: usize, w: usize) -> Result<Tensor> {
    let c = windows.dim(D::Minus1)?;
    let num_h = h / window_size;
    let num_w = w / window_size;
    let b = windows.dim(0)? / (num_h * num_w);

    windows
        .reshape((b, num_h, num_w, window_size, window_size, c))?
        .permute((0, 1, 3, 2, 4, 5))?
        .contiguous()?
        .reshape((b, h, w, c))
}

/// Create attention mask for shifted window attention.
fn create_attention_mask(
    h: usize,
    w: usize,
    window_size: usize,
    shift_size: usize,
    device: &Device,
) -> Result<Tensor> {
    // Create image mask with regions labeled 0-8
    let mut img_mask = vec![vec![0i64; w]; h];

    let h_slices = [
        (0, h.saturating_sub(window_size)),
        (h.saturating_sub(window_size), h.saturating_sub(shift_size)),
        (h.saturating_sub(shift_size), h),
    ];
    let w_slices = [
        (0, w.saturating_sub(window_size)),
        (w.saturating_sub(window_size), w.saturating_sub(shift_size)),
        (w.saturating_sub(shift_size), w),
    ];

    let mut cnt: i64 = 0;
    for (h_start, h_end) in &h_slices {
        for (w_start, w_end) in &w_slices {
            for row in img_mask.iter_mut().take(*h_end).skip(*h_start) {
                for cell in row.iter_mut().take(*w_end).skip(*w_start) {
                    *cell = cnt;
                }
            }
            cnt += 1;
        }
    }

    // Flatten and convert to tensor
    let mask_data: Vec<i64> = img_mask.into_iter().flatten().collect();
    let mask = Tensor::from_vec(mask_data, (1, h, w, 1), device)?;

    // Partition into windows
    let mask_windows = window_partition(&mask.to_dtype(DType::F32)?, window_size)?;
    let ws2 = window_size * window_size;
    let mask_windows = mask_windows.reshape(((), ws2))?;

    // Create attention mask: where regions differ, use -100.0
    let num_win = mask_windows.dim(0)?;
    let mask_windows_1 = mask_windows.unsqueeze(1)?;
    let mask_windows_2 = mask_windows.unsqueeze(2)?;
    // Explicitly broadcast to same shape before subtraction
    let mask_windows_1 = mask_windows_1.broadcast_as((num_win, ws2, ws2))?;
    let mask_windows_2 = mask_windows_2.broadcast_as((num_win, ws2, ws2))?;
    let attn_mask = (&mask_windows_1 - &mask_windows_2)?;

    // Convert non-zero to -100.0, zero stays 0.0
    let attn_mask = attn_mask.ne(0.0f32)?.to_dtype(DType::F32)?;
    let attn_mask = (attn_mask * (-100.0))?;

    Ok(attn_mask)
}

/// Patch embedding layer.
#[derive(Debug, Clone)]
struct PatchEmbed {
    projection: Conv2d,
}

impl PatchEmbed {
    fn new(patch_size: usize, in_chans: usize, embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let config = Conv2dConfig {
            stride: patch_size,
            ..Default::default()
        };
        let projection = conv2d(in_chans, embed_dim, patch_size, config, vb.pp("projection"))?;
        Ok(Self { projection })
    }
}

impl Module for PatchEmbed {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Input: (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        let x = self.projection.forward(x)?;
        let (b, c, h, w) = x.dims4()?;
        // (B, C, H, W) -> (B, H, W, C) -> (B, H*W, C)
        x.permute((0, 2, 3, 1))?.reshape((b, h * w, c))
    }
}

/// Window-based multi-head self attention with relative position bias.
/// Uses separate Q, K, V projections to match HuggingFace format.
#[derive(Debug, Clone)]
struct WindowAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    proj: Linear,
    relative_position_bias_table: Tensor,
    relative_position_index: Tensor,
    num_heads: usize,
    scale: f64,
    window_size: usize,
}

impl WindowAttention {
    fn new(
        dim: usize,
        window_size: usize,
        num_heads: usize,
        qkv_bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = dim / num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();

        // Separate Q, K, V projections (HuggingFace format)
        let vb_self = vb.pp("self");
        let query = if qkv_bias {
            linear(dim, dim, vb_self.pp("query"))?
        } else {
            candle_nn::linear_no_bias(dim, dim, vb_self.pp("query"))?
        };
        let key = if qkv_bias {
            linear(dim, dim, vb_self.pp("key"))?
        } else {
            candle_nn::linear_no_bias(dim, dim, vb_self.pp("key"))?
        };
        let value = if qkv_bias {
            linear(dim, dim, vb_self.pp("value"))?
        } else {
            candle_nn::linear_no_bias(dim, dim, vb_self.pp("value"))?
        };

        // Output projection
        let proj = linear(dim, dim, vb.pp("output").pp("dense"))?;

        // Relative position bias table from weights
        let coords_range = 2 * window_size - 1;
        let relative_position_bias_table = vb_self.get(
            (coords_range * coords_range, num_heads),
            "relative_position_bias_table",
        )?;

        // Relative position index from weights (stored in HuggingFace format)
        let relative_position_index = vb_self.get(
            (window_size * window_size, window_size * window_size),
            "relative_position_index",
        )?;

        Ok(Self {
            query,
            key,
            value,
            proj,
            relative_position_bias_table,
            relative_position_index,
            num_heads,
            scale,
            window_size,
        })
    }

    fn get_relative_position_bias(&self) -> Result<Tensor> {
        let ws2 = self.window_size * self.window_size;
        let index = self
            .relative_position_index
            .flatten_all()?
            .to_dtype(DType::U32)?;

        self.relative_position_bias_table
            .index_select(&index, 0)?
            .reshape((ws2, ws2, self.num_heads))?
            .permute((2, 0, 1))?
            .unsqueeze(0)
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b_nw, n, c) = x.dims3()?;
        let head_dim = c / self.num_heads;

        // Separate Q, K, V projections
        let q = self.query.forward(x)?;
        let k = self.key.forward(x)?;
        let v = self.value.forward(x)?;

        // Reshape for multi-head attention: (B*nW, N, C) -> (B*nW, num_heads, N, head_dim)
        let q = q
            .reshape((b_nw, n, self.num_heads, head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let k = k
            .reshape((b_nw, n, self.num_heads, head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let v = v
            .reshape((b_nw, n, self.num_heads, head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;

        // Scale Q and compute attention
        let q = (q * self.scale)?;
        let attn = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;

        // Add relative position bias
        let relative_bias = self.get_relative_position_bias()?;
        let attn = attn.broadcast_add(&relative_bias)?;

        // Apply mask for shifted window attention
        let attn = match mask {
            Some(m) => {
                // mask shape: (num_windows, ws*ws, ws*ws)
                // attn shape: (B*num_windows, num_heads, ws*ws, ws*ws)
                let num_windows = m.dim(0)?;
                let attn = attn.reshape((b_nw / num_windows, num_windows, self.num_heads, n, n))?;
                let m = m.unsqueeze(1)?.unsqueeze(0)?;
                let attn = attn.broadcast_add(&m)?;
                attn.reshape((b_nw, self.num_heads, n, n))?
            }
            None => attn,
        };

        // Softmax and apply to values
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let x = attn.matmul(&v)?;

        // Reshape back: (B*nW, num_heads, N, head_dim) -> (B*nW, N, C)
        let x = x.transpose(1, 2)?.contiguous()?.reshape((b_nw, n, c))?;

        self.proj.forward(&x)
    }
}

/// Swin Transformer block with W-MSA or SW-MSA.
#[derive(Debug, Clone)]
struct SwinBlock {
    layernorm_before: LayerNorm,
    attention: WindowAttention,
    layernorm_after: LayerNorm,
    intermediate: Linear,
    output: Linear,
    input_resolution: (usize, usize),
    shift_size: usize,
    window_size: usize,
    attn_mask: Option<Tensor>,
}

impl SwinBlock {
    #[allow(clippy::too_many_arguments)]
    fn new(
        dim: usize,
        input_resolution: (usize, usize),
        num_heads: usize,
        window_size: usize,
        shift_size: usize,
        mlp_ratio: f64,
        qkv_bias: bool,
        layer_norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let layernorm_before = layer_norm(dim, layer_norm_eps, vb.pp("layernorm_before"))?;
        let layernorm_after = layer_norm(dim, layer_norm_eps, vb.pp("layernorm_after"))?;

        let attention =
            WindowAttention::new(dim, window_size, num_heads, qkv_bias, vb.pp("attention"))?;

        let mlp_hidden = (dim as f64 * mlp_ratio) as usize;
        let intermediate = linear(dim, mlp_hidden, vb.pp("intermediate").pp("dense"))?;
        let output = linear(mlp_hidden, dim, vb.pp("output").pp("dense"))?;

        // Create attention mask for shifted windows
        let attn_mask = if shift_size > 0 {
            Some(create_attention_mask(
                input_resolution.0,
                input_resolution.1,
                window_size,
                shift_size,
                vb.device(),
            )?)
        } else {
            None
        };

        Ok(Self {
            layernorm_before,
            attention,
            layernorm_after,
            intermediate,
            output,
            input_resolution,
            shift_size,
            window_size,
            attn_mask,
        })
    }

    fn roll(x: &Tensor, shift_h: i32, shift_w: i32) -> Result<Tensor> {
        if shift_h == 0 && shift_w == 0 {
            return Ok(x.clone());
        }
        let (_b, h, w, _c) = x.dims4()?;

        // Roll H dimension
        // torch.roll(x, shifts=k) moves elements k positions to the right (positive k)
        // or k positions to the left (negative k)
        // For shift_h=-5: element at index 5 should move to index 0
        let x = if shift_h != 0 {
            let shift_h = ((-shift_h).rem_euclid(h as i32)) as usize;
            if shift_h > 0 && shift_h < h {
                // Split at shift_h and swap: [shift_h:h] then [0:shift_h]
                let first_part = x.narrow(1, shift_h, h - shift_h)?; // rows [shift_h:h]
                let second_part = x.narrow(1, 0, shift_h)?; // rows [0:shift_h]
                Tensor::cat(&[&first_part, &second_part], 1)?
            } else {
                x.clone()
            }
        } else {
            x.clone()
        };

        // Roll W dimension
        let x = if shift_w != 0 {
            let shift_w = ((-shift_w).rem_euclid(w as i32)) as usize;
            if shift_w > 0 && shift_w < w {
                // Split at shift_w and swap: [shift_w:w] then [0:shift_w]
                let first_part = x.narrow(2, shift_w, w - shift_w)?; // cols [shift_w:w]
                let second_part = x.narrow(2, 0, shift_w)?; // cols [0:shift_w]
                Tensor::cat(&[&first_part, &second_part], 2)?
            } else {
                x
            }
        } else {
            x
        };

        Ok(x)
    }
}

impl Module for SwinBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, _l, c) = x.dims3()?;
        let (h, w) = self.input_resolution;

        let shortcut = x.clone();
        let x = self.layernorm_before.forward(x)?;

        // Reshape to image: (B, H, W, C)
        let x = x.reshape((b, h, w, c))?;

        // Cyclic shift for SW-MSA
        let shifted_x = if self.shift_size > 0 {
            Self::roll(&x, -(self.shift_size as i32), -(self.shift_size as i32))?
        } else {
            x.clone()
        };

        // Partition windows
        let x_windows = window_partition(&shifted_x, self.window_size)?;
        let ws2 = self.window_size * self.window_size;
        let x_windows = x_windows.reshape(((), ws2, c))?;

        // Window attention
        let attn_windows = self
            .attention
            .forward(&x_windows, self.attn_mask.as_ref())?;

        // Merge windows back
        let attn_windows = attn_windows.reshape(((), self.window_size, self.window_size, c))?;
        let shifted_x = window_reverse(&attn_windows, self.window_size, h, w)?;

        // Reverse cyclic shift
        let x = if self.shift_size > 0 {
            Self::roll(&shifted_x, self.shift_size as i32, self.shift_size as i32)?
        } else {
            shifted_x
        };

        // Reshape back and residual
        let x = x.reshape((b, h * w, c))?;
        let x = (&shortcut + &x)?;

        // FFN with residual
        let x_norm = self.layernorm_after.forward(&x)?;
        let x_mlp = self.intermediate.forward(&x_norm)?;
        let x_mlp = x_mlp.gelu_erf()?;
        let x_mlp = self.output.forward(&x_mlp)?;
        &x + &x_mlp
    }
}

/// Patch merging layer for downsampling.
#[derive(Debug, Clone)]
struct PatchMerging {
    reduction: Linear,
    norm: LayerNorm,
    input_resolution: (usize, usize),
}

impl PatchMerging {
    fn new(
        input_resolution: (usize, usize),
        dim: usize,
        layer_norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm = layer_norm(dim * 4, layer_norm_eps, vb.pp("norm"))?;
        let reduction = candle_nn::linear_no_bias(dim * 4, dim * 2, vb.pp("reduction"))?;
        Ok(Self {
            reduction,
            norm,
            input_resolution,
        })
    }
}

/// Extract strided slice: x[:, start::2, :, :] for dim 1
fn stride2_dim1(x: &Tensor, start: usize, h: usize) -> Result<Tensor> {
    // Reshape (B, H, W, C) -> (B, H/2, 2, W, C) and select
    let (b, _h, w, c) = x.dims4()?;
    let x = x.reshape((b, h / 2, 2, w, c))?;
    x.i((.., .., start, .., ..))?.contiguous()
}

/// Extract strided slice: x[:, :, start::2, :] for dim 2
fn stride2_dim2(x: &Tensor, start: usize, w: usize) -> Result<Tensor> {
    // Reshape (B, H, W, C) -> (B, H, W/2, 2, C) and select
    let (b, h, _w, c) = x.dims4()?;
    let x = x.reshape((b, h, w / 2, 2, c))?;
    x.i((.., .., .., start, ..))?.contiguous()
}

impl Module for PatchMerging {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, _l, c) = x.dims3()?;
        let (h, w) = self.input_resolution;

        // Reshape to (B, H, W, C)
        let x = x.reshape((b, h, w, c))?;

        // Extract strided slices like PyTorch does:
        // x0 = x[:, 0::2, 0::2, :]  # even row, even col
        // x1 = x[:, 1::2, 0::2, :]  # odd row, even col
        // x2 = x[:, 0::2, 1::2, :]  # even row, odd col
        // x3 = x[:, 1::2, 1::2, :]  # odd row, odd col
        let x_even_rows = stride2_dim1(&x, 0, h)?; // (B, H/2, W, C)
        let x_odd_rows = stride2_dim1(&x, 1, h)?; // (B, H/2, W, C)

        let x0 = stride2_dim2(&x_even_rows, 0, w)?; // even row, even col
        let x1 = stride2_dim2(&x_odd_rows, 0, w)?; // odd row, even col
        let x2 = stride2_dim2(&x_even_rows, 1, w)?; // even row, odd col
        let x3 = stride2_dim2(&x_odd_rows, 1, w)?; // odd row, odd col

        // Concat like PyTorch: [x0, x1, x2, x3] along last dim
        let x = Tensor::cat(&[&x0, &x1, &x2, &x3], D::Minus1)?;

        let x = x.reshape((b, (h / 2) * (w / 2), 4 * c))?;
        let x = self.norm.forward(&x)?;
        self.reduction.forward(&x)
    }
}

/// Basic layer containing multiple Swin blocks and optional downsampling.
#[derive(Debug, Clone)]
struct BasicLayer {
    blocks: Vec<SwinBlock>,
    downsample: Option<PatchMerging>,
}

impl BasicLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        dim: usize,
        input_resolution: (usize, usize),
        depth: usize,
        num_heads: usize,
        window_size: usize,
        mlp_ratio: f64,
        qkv_bias: bool,
        downsample: bool,
        layer_norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut blocks = Vec::with_capacity(depth);

        for i in 0..depth {
            // Alternate between W-MSA (shift_size=0) and SW-MSA (shift_size=window_size/2)
            let shift_size = if i % 2 == 0 { 0 } else { window_size / 2 };

            blocks.push(SwinBlock::new(
                dim,
                input_resolution,
                num_heads,
                window_size,
                shift_size,
                mlp_ratio,
                qkv_bias,
                layer_norm_eps,
                vb.pp(format!("blocks.{}", i)),
            )?);
        }

        let downsample = if downsample {
            Some(PatchMerging::new(
                input_resolution,
                dim,
                layer_norm_eps,
                vb.pp("downsample"),
            )?)
        } else {
            None
        };

        Ok(Self { blocks, downsample })
    }
}

impl Module for BasicLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        if let Some(ds) = &self.downsample {
            x = ds.forward(&x)?;
        }

        Ok(x)
    }
}

/// Swin Transformer for image classification.
#[derive(Debug, Clone)]
pub struct Swin {
    patch_embed: PatchEmbed,
    embeddings_norm: LayerNorm,
    layers: Vec<BasicLayer>,
    norm: LayerNorm,
    head: Linear,
}

impl Swin {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_swin = vb.pp("swin");

        // Patch embedding
        let patch_embed = PatchEmbed::new(
            config.patch_size,
            config.num_channels,
            config.embed_dim,
            vb_swin.pp("embeddings").pp("patch_embeddings"),
        )?;

        // Norm after embeddings (separate from patch_embeddings in HF format)
        let embeddings_norm = layer_norm(
            config.embed_dim,
            config.layer_norm_eps,
            vb_swin.pp("embeddings").pp("norm"),
        )?;

        // Build stages
        let mut layers = Vec::new();
        let mut current_dim = config.embed_dim;
        let mut current_res = (
            config.image_size.height / config.patch_size,
            config.image_size.width / config.patch_size,
        );

        for (i, &depth) in config.depths.iter().enumerate() {
            layers.push(BasicLayer::new(
                current_dim,
                current_res,
                depth,
                config.num_heads[i],
                config.window_size,
                config.mlp_ratio,
                config.qkv_bias,
                i < config.depths.len() - 1, // downsample except last stage
                config.layer_norm_eps,
                vb_swin.pp(format!("encoder.layers.{}", i)),
            )?);

            if i < config.depths.len() - 1 {
                current_dim *= 2;
                current_res = (current_res.0 / 2, current_res.1 / 2);
            }
        }

        let norm = layer_norm(current_dim, config.layer_norm_eps, vb_swin.pp("layernorm"))?;
        let head = linear(current_dim, config.num_classes, vb.pp("classifier"))?;

        Ok(Self {
            patch_embed,
            embeddings_norm,
            layers,
            norm,
            head,
        })
    }
}

impl Module for Swin {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Patch embedding
        let x = self.patch_embed.forward(x)?;
        let mut x = self.embeddings_norm.forward(&x)?;

        // Through stages
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }

        // Global average pooling and classification
        let x = self.norm.forward(&x)?;
        let x = x.mean(1)?; // Average over sequence dimension
        self.head.forward(&x)
    }
}

/// Build a Swin model for image classification.
pub fn swin(config: &Config, vb: VarBuilder) -> Result<Swin> {
    Swin::new(config, vb)
}

// ============================================================================
// Swin Encoder for VisionEncoderDecoder models (Donut, etc.)
// ============================================================================

/// Patch embedding layer for encoder (supports non-square images).
#[derive(Debug, Clone)]
struct EncoderPatchEmbed {
    projection: Conv2d,
}

impl EncoderPatchEmbed {
    fn new(patch_size: usize, in_chans: usize, embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let config = Conv2dConfig {
            stride: patch_size,
            ..Default::default()
        };
        let projection = conv2d(in_chans, embed_dim, patch_size, config, vb.pp("projection"))?;
        Ok(Self { projection })
    }

    /// Forward pass returning hidden states and (H, W) grid dimensions.
    fn forward_with_size(&self, x: &Tensor) -> Result<(Tensor, usize, usize)> {
        // Input: (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        let x = self.projection.forward(x)?;
        let (b, c, h, w) = x.dims4()?;
        // (B, C, H, W) -> (B, H, W, C) -> (B, H*W, C)
        let x = x.permute((0, 2, 3, 1))?.reshape((b, h * w, c))?;
        Ok((x, h, w))
    }
}

/// Basic layer for encoder (supports non-square resolutions).
#[derive(Debug, Clone)]
struct EncoderBasicLayer {
    blocks: Vec<SwinBlock>,
    downsample: Option<PatchMerging>,
}

impl EncoderBasicLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        dim: usize,
        input_resolution: (usize, usize),
        depth: usize,
        num_heads: usize,
        window_size: usize,
        mlp_ratio: f64,
        qkv_bias: bool,
        downsample: bool,
        layer_norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut blocks = Vec::with_capacity(depth);

        for i in 0..depth {
            let shift_size = if i % 2 == 0 { 0 } else { window_size / 2 };

            blocks.push(SwinBlock::new(
                dim,
                input_resolution,
                num_heads,
                window_size,
                shift_size,
                mlp_ratio,
                qkv_bias,
                layer_norm_eps,
                vb.pp(format!("blocks.{}", i)),
            )?);
        }

        let downsample = if downsample {
            Some(PatchMerging::new(
                input_resolution,
                dim,
                layer_norm_eps,
                vb.pp("downsample"),
            )?)
        } else {
            None
        };

        Ok(Self { blocks, downsample })
    }
}

impl Module for EncoderBasicLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        if let Some(ds) = &self.downsample {
            x = ds.forward(&x)?;
        }

        Ok(x)
    }
}

/// Swin Transformer encoder for VisionEncoderDecoder models.
///
/// Unlike the classification model, this returns sequence hidden states
/// suitable for cross-attention in a decoder (e.g., BART).
///
/// Output shape: (batch_size, seq_len, hidden_dim)
/// where seq_len = (H / patch_size / 2^(num_stages-1)) * (W / patch_size / 2^(num_stages-1))
#[derive(Debug, Clone)]
pub struct SwinEncoder {
    patch_embed: EncoderPatchEmbed,
    embeddings_norm: LayerNorm,
    layers: Vec<EncoderBasicLayer>,
    norm: Option<LayerNorm>,
    config: Config,
    /// Final hidden dimension after all stages.
    pub hidden_size: usize,
}

impl SwinEncoder {
    /// Load encoder from VisionEncoderDecoder checkpoint.
    /// Expects weights at: encoder.* (Donut format).
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        Self::new_with_prefix(config, vb, "encoder")
    }

    /// Load encoder with custom weight prefix.
    pub fn new_with_prefix(config: &Config, vb: VarBuilder, prefix: &str) -> Result<Self> {
        let vb_enc = vb.pp(prefix);

        let patch_embed = EncoderPatchEmbed::new(
            config.patch_size,
            config.num_channels,
            config.embed_dim,
            vb_enc.pp("embeddings").pp("patch_embeddings"),
        )?;

        let embeddings_norm = layer_norm(
            config.embed_dim,
            config.layer_norm_eps,
            vb_enc.pp("embeddings").pp("norm"),
        )?;

        // Build stages - for encoder, we compute resolution dynamically
        let mut layers = Vec::new();
        let mut current_dim = config.embed_dim;

        // Use reference resolution for building the model
        // Actual resolution is determined at runtime
        let mut current_res = (
            config.image_size.height / config.patch_size,
            config.image_size.width / config.patch_size,
        );

        for (i, &depth) in config.depths.iter().enumerate() {
            layers.push(EncoderBasicLayer::new(
                current_dim,
                current_res,
                depth,
                config.num_heads[i],
                config.window_size,
                config.mlp_ratio,
                config.qkv_bias,
                i < config.depths.len() - 1, // downsample except last stage
                config.layer_norm_eps,
                vb_enc.pp(format!("encoder.layers.{}", i)),
            )?);

            if i < config.depths.len() - 1 {
                current_dim *= 2;
                current_res = (current_res.0 / 2, current_res.1 / 2);
            }
        }

        // Final layernorm is optional (Donut encoder doesn't have one)
        let norm = layer_norm(current_dim, config.layer_norm_eps, vb_enc.pp("layernorm")).ok();

        Ok(Self {
            patch_embed,
            embeddings_norm,
            layers,
            norm,
            config: config.clone(),
            hidden_size: current_dim,
        })
    }

    /// Get the output hidden dimension.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Compute output sequence length for given input image dimensions.
    pub fn output_seq_len(&self, height: usize, width: usize) -> usize {
        let h = height / self.config.patch_size;
        let w = width / self.config.patch_size;
        let downsample_factor = 1 << (self.config.depths.len() - 1);
        (h / downsample_factor) * (w / downsample_factor)
    }
}

impl Module for SwinEncoder {
    /// Forward pass: (B, C, H, W) -> (B, seq_len, hidden_dim)
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Patch embedding with size tracking
        let (x, h, w) = self.patch_embed.forward_with_size(x)?;
        let mut x = self.embeddings_norm.forward(&x)?;

        // Track resolution through stages
        let mut current_h = h;
        let mut current_w = w;

        // Through stages
        for (i, layer) in self.layers.iter().enumerate() {
            // Reshape for this stage's blocks (they expect specific resolution)
            let (b, _seq_len, c) = x.dims3()?;
            x = x.reshape((b, current_h, current_w, c))?;
            x = x.reshape((b, current_h * current_w, c))?;

            x = layer.forward(&x)?;

            // Update resolution after downsampling
            if i < self.layers.len() - 1 {
                current_h /= 2;
                current_w /= 2;
            }
        }

        // Final layer norm (if present)
        match &self.norm {
            Some(norm) => norm.forward(&x),
            None => Ok(x),
        }
    }
}
