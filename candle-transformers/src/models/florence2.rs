//! Florence-2 DaViT Vision Encoder
//!
//! Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks.
//! - [Paper](https://arxiv.org/abs/2311.06242)
//! - [HuggingFace](https://huggingface.co/microsoft/Florence-2-base)
//!
//! The DaViT (Dual Attention Vision Transformer) backbone uses a hierarchical
//! architecture with alternating spatial (window) attention and channel attention
//! within each stage, plus convolutional patch embeddings for downsampling.

use candle::{DType, IndexOp, Result, Tensor, D};
use candle_nn::{
    conv2d, layer_norm, linear, ops::softmax, Conv2d, Conv2dConfig, LayerNorm, Linear, Module,
    VarBuilder,
};

// ---- Configuration ----

fn default_drop_path_rate() -> f32 {
    0.1
}

fn default_window_size() -> usize {
    12
}

fn default_projection_dim() -> usize {
    768
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Florence2VisionConfig {
    pub dim_embed: Vec<usize>,
    pub num_heads: Vec<usize>,
    pub num_groups: Vec<usize>,
    pub depths: Vec<usize>,
    pub patch_size: Vec<usize>,
    pub patch_stride: Vec<usize>,
    pub patch_padding: Vec<usize>,
    pub patch_prenorm: Vec<bool>,
    #[serde(default = "default_drop_path_rate")]
    pub drop_path_rate: f32,
    #[serde(default = "default_window_size")]
    pub window_size: usize,
    #[serde(default = "default_projection_dim")]
    pub projection_dim: usize,
}

impl Florence2VisionConfig {
    /// Florence-2-base default vision config.
    pub fn florence2_base() -> Self {
        Self {
            dim_embed: vec![128, 256, 512, 1024],
            num_heads: vec![4, 8, 16, 32],
            num_groups: vec![4, 8, 16, 32],
            depths: vec![1, 1, 9, 1],
            patch_size: vec![7, 3, 3, 3],
            patch_stride: vec![4, 2, 2, 2],
            patch_padding: vec![3, 1, 1, 1],
            patch_prenorm: vec![false, true, true, true],
            drop_path_rate: 0.1,
            window_size: 12,
            projection_dim: 768,
        }
    }
}

fn default_d_model() -> usize {
    768
}

fn default_encoder_layers() -> usize {
    6
}

fn default_decoder_layers() -> usize {
    6
}

fn default_encoder_attention_heads() -> usize {
    12
}

fn default_decoder_attention_heads() -> usize {
    12
}

fn default_encoder_ffn_dim() -> usize {
    3072
}

fn default_decoder_ffn_dim() -> usize {
    3072
}

fn default_activation() -> candle_nn::Activation {
    candle_nn::Activation::Gelu
}

fn default_dropout() -> f64 {
    0.1
}

fn default_vocab_size() -> usize {
    51289
}

fn default_max_position_embeddings() -> usize {
    1024
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Florence2TextConfig {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_d_model")]
    pub d_model: usize,
    #[serde(default = "default_encoder_layers")]
    pub encoder_layers: usize,
    #[serde(default = "default_decoder_layers")]
    pub decoder_layers: usize,
    #[serde(default = "default_encoder_attention_heads")]
    pub encoder_attention_heads: usize,
    #[serde(default = "default_decoder_attention_heads")]
    pub decoder_attention_heads: usize,
    #[serde(default = "default_encoder_ffn_dim")]
    pub encoder_ffn_dim: usize,
    #[serde(default = "default_decoder_ffn_dim")]
    pub decoder_ffn_dim: usize,
    #[serde(default = "default_activation")]
    pub activation_function: candle_nn::Activation,
    #[serde(default = "default_dropout")]
    pub dropout: f64,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub scale_embedding: bool,
    #[serde(default = "default_pad_token_id")]
    pub pad_token_id: u32,
    #[serde(default)]
    pub decoder_start_token_id: u32,
}

fn default_pad_token_id() -> u32 {
    1
}

fn default_projection_dim_top() -> usize {
    768
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Florence2Config {
    pub vision_config: Florence2VisionConfig,
    pub text_config: Florence2TextConfig,
    #[serde(default = "default_projection_dim_top")]
    pub projection_dim: usize,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
}

impl Florence2Config {
    pub fn florence2_base() -> Self {
        Self {
            vision_config: Florence2VisionConfig::florence2_base(),
            text_config: Florence2TextConfig {
                vocab_size: 51289,
                d_model: 768,
                encoder_layers: 6,
                decoder_layers: 6,
                encoder_attention_heads: 12,
                decoder_attention_heads: 12,
                encoder_ffn_dim: 3072,
                decoder_ffn_dim: 3072,
                activation_function: candle_nn::Activation::Gelu,
                dropout: 0.1,
                max_position_embeddings: 1024,
                scale_embedding: false,
                pad_token_id: 1,
                decoder_start_token_id: 2,
            },
            projection_dim: 768,
            vocab_size: 51289,
        }
    }
}

// ---- DaViT Building Blocks ----

/// Convolutional patch embedding for each DaViT stage.
struct ConvEmbed {
    proj: Conv2d,
    norm: Option<LayerNorm>,
    pre_norm: bool,
}

impl ConvEmbed {
    fn new(
        in_chans: usize,
        embed_dim: usize,
        patch_size: usize,
        stride: usize,
        padding: usize,
        pre_norm: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv2dConfig {
            stride,
            padding,
            ..Default::default()
        };
        let proj = conv2d(in_chans, embed_dim, patch_size, cfg, vb.pp("proj"))?;
        let dim_norm = if pre_norm { in_chans } else { embed_dim };
        let norm = Some(layer_norm(dim_norm, 1e-5, vb.pp("norm"))?);
        Ok(Self {
            proj,
            norm,
            pre_norm,
        })
    }

    fn forward(&self, xs: &Tensor, size: (usize, usize)) -> Result<(Tensor, (usize, usize))> {
        let (h, w) = size;
        let xs = if xs.rank() == 3 {
            // (B, H*W, C) -> need pre-norm then reshape
            let xs = if self.pre_norm {
                if let Some(norm) = &self.norm {
                    norm.forward(&xs)?
                } else {
                    xs.clone()
                }
            } else {
                xs.clone()
            };
            let b = xs.dim(0)?;
            let c = xs.dim(2)?;
            xs.reshape((b, h, w, c))?.permute((0, 3, 1, 2))?
        } else {
            xs.clone()
        };

        let xs = self.proj.forward(&xs)?;
        let (_, _, new_h, new_w) = xs.dims4()?;
        // (B, C, H, W) -> (B, H*W, C)
        let xs = xs.flatten_from(2)?.transpose(1, 2)?;

        let xs = if !self.pre_norm {
            if let Some(norm) = &self.norm {
                norm.forward(&xs)?
            } else {
                xs
            }
        } else {
            xs
        };

        Ok((xs, (new_h, new_w)))
    }
}

/// Depthwise separable convolution operating on (B, N, C) sequences.
struct DepthWiseConv2d {
    dw: Conv2d,
}

impl DepthWiseConv2d {
    fn new(dim: usize, kernel_size: usize, padding: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding,
            groups: dim,
            ..Default::default()
        };
        let dw = conv2d(dim, dim, kernel_size, cfg, vb.pp("dw"))?;
        Ok(Self { dw })
    }

    fn forward(&self, xs: &Tensor, size: (usize, usize)) -> Result<(Tensor, (usize, usize))> {
        let (h, w) = size;
        let (b, _n, c) = xs.dims3()?;
        let xs = xs.transpose(1, 2)?.reshape((b, c, h, w))?;
        let xs = self.dw.forward(&xs)?;
        let new_h = xs.dim(2)?;
        let new_w = xs.dim(3)?;
        let xs = xs.flatten_from(2)?.transpose(1, 2)?;
        Ok((xs, (new_h, new_w)))
    }
}

/// MLP block: fc1 -> gelu -> fc2
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn new(in_features: usize, hidden_features: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(in_features, hidden_features, vb.pp("net.fc1"))?;
        let fc2 = linear(hidden_features, in_features, vb.pp("net.fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.fc1)?.gelu()?.apply(&self.fc2)
    }
}

/// Window attention: spatially partitions input into windows, applies multi-head attention.
struct WindowAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    scale: f64,
    window_size: usize,
}

impl WindowAttention {
    fn new(dim: usize, num_heads: usize, window_size: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;
        let scale = (head_dim as f64).powf(-0.5);
        let qkv = linear(dim, dim * 3, vb.pp("qkv"))?;
        let proj = linear(dim, dim, vb.pp("proj"))?;
        Ok(Self {
            qkv,
            proj,
            num_heads,
            scale,
            window_size,
        })
    }

    fn forward(&self, xs: &Tensor, size: (usize, usize)) -> Result<(Tensor, (usize, usize))> {
        let (h, w) = size;
        let (b, _l, c) = xs.dims3()?;
        let ws = self.window_size;

        // Pad to multiple of window_size
        let pad_b = (ws - h % ws) % ws;
        let pad_r = (ws - w % ws) % ws;

        // Reshape to (B, H, W, C)
        let xs = xs.reshape((b, h, w, c))?;

        let xs = if pad_b > 0 || pad_r > 0 {
            // Pad: (B, H, W, C) -> (B, H+pad_b, W+pad_r, C)
            xs.pad_with_zeros(1, 0, pad_b)?
                .pad_with_zeros(2, 0, pad_r)?
        } else {
            xs
        };

        let (_, hp, wp, _) = xs.dims4()?;

        // Window partition: (B, nH, ws, nW, ws, C) -> (B*nH*nW, ws*ws, C)
        let n_h = hp / ws;
        let n_w = wp / ws;
        let xs = xs
            .reshape((b, n_h, ws, n_w, ws, c))?
            .permute((0, 1, 3, 2, 4, 5))?
            .reshape((b * n_h * n_w, ws * ws, c))?
            .contiguous()?;

        // QKV
        let (b_w, n, _) = xs.dims3()?;
        let qkv = self
            .qkv
            .forward(&xs)?
            .reshape((b_w, n, 3, self.num_heads, c / self.num_heads))?
            .permute((2, 0, 3, 1, 4))?;
        let q = (qkv.i(0)? * self.scale)?;
        let k = qkv.i(1)?;
        let v = qkv.i(2)?;

        let attn = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let attn = softmax(&attn, D::Minus1)?;
        let xs = attn.matmul(&v)?;

        // (B_w, num_heads, N, head_dim) -> (B_w, N, C)
        let xs = xs.transpose(1, 2)?.reshape((b_w, n, c))?;
        let xs = self.proj.forward(&xs)?;

        // Window reverse: (B*nH*nW, ws, ws, C) -> (B, Hp, Wp, C) -> (B, H, W, C)
        let xs = xs.reshape((b_w, ws, ws, c))?;
        let xs = xs
            .reshape((b, n_h, n_w, ws, ws, c))?
            .permute((0, 1, 3, 2, 4, 5))?
            .reshape((b, hp, wp, c))?;

        // Remove padding
        let xs = if pad_b > 0 || pad_r > 0 {
            xs.i((.., ..h, ..w, ..))?.contiguous()?
        } else {
            xs
        };

        let xs = xs.reshape((b, h * w, c))?;
        Ok((xs, size))
    }
}

/// Channel attention: groups channels and applies attention across spatial tokens.
struct ChannelAttention {
    qkv: Linear,
    proj: Linear,
    groups: usize,
}

impl ChannelAttention {
    fn new(dim: usize, groups: usize, vb: VarBuilder) -> Result<Self> {
        let qkv = linear(dim, dim * 3, vb.pp("qkv"))?;
        let proj = linear(dim, dim, vb.pp("proj"))?;
        Ok(Self { qkv, proj, groups })
    }

    fn forward(&self, xs: &Tensor, _size: (usize, usize)) -> Result<(Tensor, (usize, usize))> {
        let (b, n, c) = xs.dims3()?;
        let group_dim = c / self.groups;

        let qkv = self
            .qkv
            .forward(&xs)?
            .reshape((b, n, 3, self.groups, group_dim))?
            .permute((2, 0, 3, 1, 4))?;
        let q = qkv.i(0)?;
        let k = qkv.i(1)?;
        let v = qkv.i(2)?;

        // Channel attention: (B, G, N, D) -> attend over tokens for each channel group
        let scale = (n as f64).powf(-0.5);
        let q = (q * scale)?;
        let attn = q
            .transpose(D::Minus2, D::Minus1)?
            .matmul(&k)?;
        let attn = softmax(&attn, D::Minus1)?;
        let xs = attn
            .matmul(&v.transpose(D::Minus2, D::Minus1)?)?
            .transpose(D::Minus2, D::Minus1)?;

        let xs = xs.transpose(1, 2)?.reshape((b, n, c))?;
        let xs = self.proj.forward(&xs)?;
        Ok((xs, _size))
    }
}

/// Spatial block: optional depthwise conv + window attention + optional depthwise conv + FFN.
struct SpatialBlock {
    conv1: Option<DepthWiseConv2d>,
    window_attn_norm: LayerNorm,
    window_attn: WindowAttention,
    conv2: Option<DepthWiseConv2d>,
    ffn_norm: LayerNorm,
    ffn: Mlp,
}

impl SpatialBlock {
    fn new(
        dim: usize,
        num_heads: usize,
        window_size: usize,
        mlp_ratio: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv1 = Some(DepthWiseConv2d::new(dim, 3, 1, vb.pp("conv1.fn"))?);
        let window_attn_norm = layer_norm(dim, 1e-5, vb.pp("window_attn.norm"))?;
        let window_attn =
            WindowAttention::new(dim, num_heads, window_size, vb.pp("window_attn.fn"))?;
        let conv2 = Some(DepthWiseConv2d::new(dim, 3, 1, vb.pp("conv2.fn"))?);
        let ffn_norm = layer_norm(dim, 1e-5, vb.pp("ffn.norm"))?;
        let ffn = Mlp::new(dim, (dim as f64 * mlp_ratio) as usize, vb.pp("ffn.fn"))?;
        Ok(Self {
            conv1,
            window_attn_norm,
            window_attn,
            conv2,
            ffn_norm,
            ffn,
        })
    }

    fn forward(&self, xs: &Tensor, size: (usize, usize)) -> Result<(Tensor, (usize, usize))> {
        let (mut xs, mut size) = if let Some(conv1) = &self.conv1 {
            let (out, s) = conv1.forward(xs, size)?;
            ((&out + xs)?, s)
        } else {
            (xs.clone(), size)
        };

        // Window attention with residual
        let residual = &xs;
        let (attn_out, new_size) =
            self.window_attn
                .forward(&self.window_attn_norm.forward(&xs)?, size)?;
        xs = (residual + attn_out)?;
        size = new_size;

        let (xs, size) = if let Some(conv2) = &self.conv2 {
            let residual = &xs;
            let (out, s) = conv2.forward(&xs, size)?;
            ((&out + residual)?, s)
        } else {
            (xs.clone(), size)
        };

        // FFN with residual
        let residual = &xs;
        let xs = (residual + self.ffn.forward(&self.ffn_norm.forward(&xs)?)?)?;
        Ok((xs, size))
    }
}

/// Channel block: optional depthwise conv + channel attention + optional depthwise conv + FFN.
struct ChannelBlock {
    conv1: Option<DepthWiseConv2d>,
    channel_attn_norm: LayerNorm,
    channel_attn: ChannelAttention,
    conv2: Option<DepthWiseConv2d>,
    ffn_norm: LayerNorm,
    ffn: Mlp,
}

impl ChannelBlock {
    fn new(dim: usize, num_groups: usize, mlp_ratio: f64, vb: VarBuilder) -> Result<Self> {
        let conv1 = Some(DepthWiseConv2d::new(dim, 3, 1, vb.pp("conv1.fn"))?);
        let channel_attn_norm = layer_norm(dim, 1e-5, vb.pp("channel_attn.norm"))?;
        let channel_attn = ChannelAttention::new(dim, num_groups, vb.pp("channel_attn.fn"))?;
        let conv2 = Some(DepthWiseConv2d::new(dim, 3, 1, vb.pp("conv2.fn"))?);
        let ffn_norm = layer_norm(dim, 1e-5, vb.pp("ffn.norm"))?;
        let ffn = Mlp::new(dim, (dim as f64 * mlp_ratio) as usize, vb.pp("ffn.fn"))?;
        Ok(Self {
            conv1,
            channel_attn_norm,
            channel_attn,
            conv2,
            ffn_norm,
            ffn,
        })
    }

    fn forward(&self, xs: &Tensor, size: (usize, usize)) -> Result<(Tensor, (usize, usize))> {
        let (mut xs, mut size) = if let Some(conv1) = &self.conv1 {
            let (out, s) = conv1.forward(xs, size)?;
            ((&out + xs)?, s)
        } else {
            (xs.clone(), size)
        };

        // Channel attention with residual
        let residual = &xs;
        let (attn_out, new_size) = self.channel_attn.forward(
            &self.channel_attn_norm.forward(&xs)?,
            size,
        )?;
        xs = (residual + attn_out)?;
        size = new_size;

        let (xs, size) = if let Some(conv2) = &self.conv2 {
            let residual = &xs;
            let (out, s) = conv2.forward(&xs, size)?;
            ((&out + residual)?, s)
        } else {
            (xs.clone(), size)
        };

        // FFN with residual
        let residual = &xs;
        let xs = (residual + self.ffn.forward(&self.ffn_norm.forward(&xs)?)?)?;
        Ok((xs, size))
    }
}

/// A single DaViT stage: pairs of (SpatialBlock, ChannelBlock) repeated `depth` times.
struct DaViTStage {
    blocks: Vec<(SpatialBlock, ChannelBlock)>,
}

impl DaViTStage {
    fn new(
        dim: usize,
        depth: usize,
        num_heads: usize,
        num_groups: usize,
        window_size: usize,
        mlp_ratio: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut blocks = Vec::with_capacity(depth);
        for j in 0..depth {
            let spatial =
                SpatialBlock::new(dim, num_heads, window_size, mlp_ratio, vb.pp(format!("{j}"))
                    .pp("spatial_block"))?;
            let channel =
                ChannelBlock::new(dim, num_groups, mlp_ratio, vb.pp(format!("{j}"))
                    .pp("channel_block"))?;
            blocks.push((spatial, channel));
        }
        Ok(Self { blocks })
    }

    fn forward(&self, xs: &Tensor, size: (usize, usize)) -> Result<(Tensor, (usize, usize))> {
        let mut xs = xs.clone();
        let mut size = size;
        for (spatial, channel) in &self.blocks {
            let (out, s) = spatial.forward(&xs, size)?;
            xs = out;
            size = s;
            let (out, s) = channel.forward(&xs, size)?;
            xs = out;
            size = s;
        }
        Ok((xs, size))
    }
}

/// DaViT: Dual Attention Vision Transformer backbone.
pub struct DaViT {
    convs: Vec<ConvEmbed>,
    stages: Vec<DaViTStage>,
    norms: LayerNorm,
}

impl DaViT {
    pub fn new(config: &Florence2VisionConfig, vb: VarBuilder) -> Result<Self> {
        let num_stages = config.dim_embed.len();
        let mut convs = Vec::with_capacity(num_stages);
        let mut stages = Vec::with_capacity(num_stages);

        for i in 0..num_stages {
            let in_chans = if i == 0 { 3 } else { config.dim_embed[i - 1] };
            let conv = ConvEmbed::new(
                in_chans,
                config.dim_embed[i],
                config.patch_size[i],
                config.patch_stride[i],
                config.patch_padding[i],
                config.patch_prenorm[i],
                vb.pp("convs").pp(i),
            )?;
            convs.push(conv);

            let stage = DaViTStage::new(
                config.dim_embed[i],
                config.depths[i],
                config.num_heads[i],
                config.num_groups[i],
                config.window_size,
                4.0,
                vb.pp("blocks").pp(i),
            )?;
            stages.push(stage);
        }

        let norms = layer_norm(
            *config.dim_embed.last().unwrap(),
            1e-5,
            vb.pp("norms"),
        )?;

        Ok(Self {
            convs,
            stages,
            norms,
        })
    }

    /// Forward pass returning unpooled spatial features: (B, num_tokens, dim_embed[-1]).
    pub fn forward_features_unpool(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = xs.dims4()?;
        let mut xs = xs.clone();
        let mut input_size = (h, w);

        for (conv, stage) in self.convs.iter().zip(self.stages.iter()) {
            let (out, s) = conv.forward(&xs, input_size)?;
            let (out, s) = stage.forward(&out, s)?;
            xs = out;
            input_size = s;
        }

        Ok(xs)
    }

    /// Forward with global average pooling + layer norm.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.forward_features_unpool(xs)?;
        // (B, N, C) -> avg pool -> (B, C)
        let xs = xs.mean(1)?;
        self.norms.forward(&xs)
    }
}

// ---- Positional Embeddings ----

/// Learned 2D absolute position embedding (row + column).
pub struct LearnedAbsolutePositionEmbedding2D {
    row_embeddings: candle_nn::Embedding,
    col_embeddings: candle_nn::Embedding,
}

impl LearnedAbsolutePositionEmbedding2D {
    fn new(embedding_dim: usize, num_pos: usize, vb: VarBuilder) -> Result<Self> {
        let half_dim = embedding_dim / 2;
        let other_half = embedding_dim - half_dim;
        let row_embeddings = candle_nn::embedding(num_pos, half_dim, vb.pp("row_embeddings"))?;
        let col_embeddings =
            candle_nn::embedding(num_pos, other_half, vb.pp("column_embeddings"))?;
        Ok(Self {
            row_embeddings,
            col_embeddings,
        })
    }

    /// Input: (B, H, W, C). Output: (B, H, W, embedding_dim).
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, h, w, _) = xs.dims4()?;
        let device = xs.device();
        let width_ids = Tensor::arange(0u32, w as u32, device)?;
        let height_ids = Tensor::arange(0u32, h as u32, device)?;

        let x_emb = self.col_embeddings.forward(&width_ids)?; // (W, D/2)
        let y_emb = self.row_embeddings.forward(&height_ids)?; // (H, D/2)

        // Broadcast: (H, W, D)
        let x_emb = x_emb.unsqueeze(0)?.broadcast_as((h, w, x_emb.dim(1)?))?;
        let y_emb = y_emb.unsqueeze(1)?.broadcast_as((h, w, y_emb.dim(1)?))?;
        let pos = Tensor::cat(&[x_emb, y_emb], 2)?; // (H, W, D)

        let pos = pos.unsqueeze(0)?.broadcast_as((b, h, w, pos.dim(2)?))?;
        Ok(pos)
    }
}

/// Cosine 1D positional embedding.
pub struct PositionalEmbeddingCosine1D {
    pos_embed: Tensor,
}

impl PositionalEmbeddingCosine1D {
    fn new(embed_dim: usize, max_seq_len: usize, vb: VarBuilder) -> Result<Self> {
        let device = vb.device();
        let factor = (10000f64).ln();
        let denom: Vec<f64> = (0..embed_dim)
            .step_by(2)
            .map(|i| (-factor * i as f64 / embed_dim as f64).exp())
            .collect();
        let denom = Tensor::from_vec(denom, embed_dim / 2, device)?;

        let positions = Tensor::arange(0f32, max_seq_len as f32, device)?
            .to_dtype(DType::F64)?
            .reshape((max_seq_len, 1))?;
        let freqs = positions.broadcast_mul(&denom.unsqueeze(0)?)?;

        let sin_vals = freqs.sin()?;
        let cos_vals = freqs.cos()?;

        // Interleave sin and cos
        let mut pos_data = vec![0f64; max_seq_len * embed_dim];
        let sin_flat = sin_vals.flatten_all()?.to_vec1::<f64>()?;
        let cos_flat = cos_vals.flatten_all()?.to_vec1::<f64>()?;
        for t in 0..max_seq_len {
            for i in 0..embed_dim / 2 {
                pos_data[t * embed_dim + 2 * i] = sin_flat[t * (embed_dim / 2) + i];
                pos_data[t * embed_dim + 2 * i + 1] = cos_flat[t * (embed_dim / 2) + i];
            }
        }
        let pos_embed = Tensor::from_vec(pos_data, (max_seq_len, embed_dim), device)?;

        Ok(Self { pos_embed })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let seq_len = xs.dim(xs.rank() - 2)?;
        let pos = self.pos_embed.i(..seq_len)?;
        if xs.rank() == 3 {
            Ok(pos.unsqueeze(0)?)
        } else {
            Ok(pos)
        }
    }
}

// ---- Florence-2 Vision Model with Projection ----

/// Florence-2 vision model: DaViT backbone + 2D positional embeddings + linear projection.
pub struct Florence2VisionModel {
    vision_tower: DaViT,
    image_projection: Tensor,
    image_proj_norm: LayerNorm,
    image_pos_embed: LearnedAbsolutePositionEmbedding2D,
    visual_temporal_embed: PositionalEmbeddingCosine1D,
}

impl Florence2VisionModel {
    pub fn new(config: &Florence2Config, vb: VarBuilder) -> Result<Self> {
        let vision_config = &config.vision_config;
        let image_dim_out = *vision_config.dim_embed.last().unwrap();
        let dim_projection = vision_config.projection_dim;

        let vision_tower = DaViT::new(vision_config, vb.pp("vision_tower"))?;

        let image_projection =
            vb.get((image_dim_out, dim_projection), "image_projection")?;
        let image_proj_norm = layer_norm(dim_projection, 1e-5, vb.pp("image_proj_norm"))?;
        let image_pos_embed =
            LearnedAbsolutePositionEmbedding2D::new(image_dim_out, 50, vb.pp("image_pos_embed"))?;
        let visual_temporal_embed = PositionalEmbeddingCosine1D::new(
            image_dim_out,
            100,
            vb.pp("visual_temporal_embed"),
        )?;

        Ok(Self {
            vision_tower,
            image_projection,
            image_proj_norm,
            image_pos_embed,
            visual_temporal_embed,
        })
    }

    /// Encode a batch of images to projected features for the language model.
    /// Input: (B, C, H, W) pixel values. Output: (B, num_visual_tokens, projection_dim).
    pub fn encode_image(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let (b, _c, _h, _w) = pixel_values.dims4()?;

        // DaViT forward
        let xs = self.vision_tower.forward_features_unpool(pixel_values)?;
        // xs: (B, num_tokens, dim_embed[-1])

        let num_tokens = xs.dim(1)?;
        let dim = xs.dim(2)?;
        let hw = (num_tokens as f64).sqrt() as usize;

        // Add 2D positional embeddings
        let xs = xs.reshape((b, hw, hw, dim))?;
        let pos_embed = self.image_pos_embed.forward(&xs)?;
        let xs = (xs + pos_embed)?;
        let xs = xs.reshape((b, num_tokens, dim))?;

        // Add temporal embedding (T=1 for single image)
        let temporal_token = xs.i((.., 0..1, ..))?.unsqueeze(1)?; // (B, 1, 1, D)
        let temporal_embed = self.visual_temporal_embed.forward(
            &temporal_token.squeeze(1)?,
        )?;
        let xs = (&xs + temporal_embed.i((.., ..1, ..))?.broadcast_as(xs.shape())?)?;

        // Spatial avg pool + temporal avg pool (for T=1, they're the same)
        let spatial_avg = xs.mean(1)?; // (B, D)
        let temporal_avg = xs.mean(1)?; // (B, D), same for T=1

        // Concatenate features: [spatial_avg_pool, temporal_avg_pool] -> (B, 2, D)
        let spatial_avg = spatial_avg.unsqueeze(1)?;
        let temporal_avg = temporal_avg.unsqueeze(1)?;
        let xs = Tensor::cat(&[spatial_avg, temporal_avg], 1)?;

        // Project: (B, 2, D) @ (D, proj_dim) -> (B, 2, proj_dim)
        let xs = xs.matmul(&self.image_projection)?;
        let xs = self.image_proj_norm.forward(&xs)?;

        Ok(xs)
    }
}
