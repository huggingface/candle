//! Siglip2 NaFlex model implementation.
//!
//! NaFlex variants of SigLIP2 (e.g., `siglip2-base-patch16-naflex`,
//! `siglip2-so400m-patch16-naflex`) support variable-resolution inputs that
//! preserve native aspect ratios. Unlike fixed-resolution v1/v2 variants which
//! use a `Conv2d` patch embedding, NaFlex variants use a `Linear` projection
//! over flattened patches: input pixel values arrive pre-patchified as
//! `(batch, max_num_patches, num_channels * patch_size * patch_size)`,
//! along with per-input `spatial_shapes` declaring the actual `(h_patches,
//! w_patches)` of each input and a `pixel_attention_mask` indicating padded
//! positions.
//!
//! References:
//! - 🤗 [SigLIP 2 paper](https://arxiv.org/abs/2502.14786)
//! - 🤗 [Model Card](https://huggingface.co/google/siglip2-base-patch16-naflex)
//! - HF transformers reference: `transformers/src/transformers/models/siglip2/modeling_siglip2.py`
//!
//! ## Architecture deltas vs `siglip` (v1) model
//!
//! - **VisionEmbeddings** uses `Linear` over flattened patches instead of `Conv2d`.
//!   Mathematically equivalent for non-overlapping patches; the safetensors
//!   checkpoint stores the Linear shape `[hidden_size, num_channels * patch_size^2]`
//!   directly (vs Conv2d `[hidden_size, num_channels, patch_size, patch_size]`
//!   for fixed-resolution variants).
//! - **Position embedding** is stored as a flat `[num_patches, hidden_size]` grid
//!   (e.g. `[256, 768]` for the base 16x16 grid) and is **bilinear-resized** at
//!   inference time to match each input's `(h_patches, w_patches)`.
//! - **Forward signature** takes `(pixel_values, pixel_attention_mask, spatial_shapes)`.
//! - **Attention mask** is plumbed through `Encoder`, `EncoderLayer`, `Attention`,
//!   and the `MultiheadAttentionPoolingHead`.
//!
//! ## Note on bicubic vs bilinear interpolation
//!
//! HF transformers' reference implementation uses
//! `F.interpolate(mode="bicubic", antialias=True)` for `resize_positional_embeddings`.
//! Candle's `Tensor::interpolate2d` is bilinear-only and lacks an antialias mode.
//! This implementation falls back to bilinear, which introduces small numerical
//! drift at non-base resolutions — typically `< 5e-3` max abs diff at fp32
//! versus the PyTorch reference. Adding bicubic+antialias to candle-core is
//! tracked separately and out of scope for this model file.

use candle::{DType, IndexOp, Module, Result, Tensor, D};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, VarBuilder};

// ============================================================================
// Config
// ============================================================================

fn default_text_vocab_size() -> usize {
    256000
}

fn default_text_hidden_size() -> usize {
    768
}

fn default_text_intermediate_size() -> usize {
    3072
}

fn default_text_num_hidden_layers() -> usize {
    12
}

fn default_text_num_attention_heads() -> usize {
    12
}

fn default_text_max_position_embeddings() -> usize {
    64
}

fn default_text_layer_norm_eps() -> f64 {
    1e-6
}

fn default_text_pad_token_id() -> u32 {
    1
}

fn default_text_bos_token_id() -> u32 {
    49406
}

fn default_text_eos_token_id() -> u32 {
    49407
}

fn default_text_projection_size() -> usize {
    768
}

fn default_text_hidden_act() -> candle_nn::Activation {
    candle_nn::Activation::GeluPytorchTanh
}

#[derive(serde::Deserialize, Clone, Debug)]
pub struct TextConfig {
    #[serde(default = "default_text_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_text_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_text_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_text_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_text_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_text_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_text_hidden_act")]
    pub hidden_act: candle_nn::Activation,
    #[serde(default = "default_text_layer_norm_eps")]
    pub layer_norm_eps: f64,
    #[serde(default = "default_text_pad_token_id")]
    pub pad_token_id: u32,
    #[serde(default = "default_text_bos_token_id")]
    pub bos_token_id: u32,
    #[serde(default = "default_text_eos_token_id")]
    pub eos_token_id: u32,
    #[serde(default = "default_text_projection_size")]
    pub projection_size: usize,
}

fn default_vision_hidden_size() -> usize {
    768
}

fn default_vision_intermediate_size() -> usize {
    3072
}

fn default_vision_num_hidden_layers() -> usize {
    12
}

fn default_vision_num_attention_heads() -> usize {
    12
}

fn default_vision_num_channels() -> usize {
    3
}

fn default_vision_patch_size() -> usize {
    16
}

fn default_vision_num_patches() -> usize {
    256
}

fn default_vision_layer_norm_eps() -> f64 {
    1e-6
}

fn default_vision_hidden_act() -> candle_nn::Activation {
    candle_nn::Activation::GeluPytorchTanh
}

#[derive(serde::Deserialize, Clone, Debug)]
pub struct VisionConfig {
    #[serde(default = "default_vision_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_vision_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_vision_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_vision_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_vision_num_channels")]
    pub num_channels: usize,
    #[serde(default = "default_vision_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_vision_num_patches")]
    pub num_patches: usize,
    #[serde(default = "default_vision_hidden_act")]
    pub hidden_act: candle_nn::Activation,
    #[serde(default = "default_vision_layer_norm_eps")]
    pub layer_norm_eps: f64,
}

trait TransformerConfig {
    fn hidden_size(&self) -> usize;
    fn intermediate_size(&self) -> usize;
    fn num_attention_heads(&self) -> usize;
    fn num_hidden_layers(&self) -> usize;
    fn layer_norm_eps(&self) -> f64;
    fn hidden_act(&self) -> candle_nn::Activation;
}

impl TransformerConfig for TextConfig {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }
    fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }
    fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }
    fn layer_norm_eps(&self) -> f64 {
        self.layer_norm_eps
    }
    fn hidden_act(&self) -> candle_nn::Activation {
        self.hidden_act
    }
}

impl TransformerConfig for VisionConfig {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }
    fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }
    fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }
    fn layer_norm_eps(&self) -> f64 {
        self.layer_norm_eps
    }
    fn hidden_act(&self) -> candle_nn::Activation {
        self.hidden_act
    }
}

#[derive(serde::Deserialize, Clone, Debug)]
pub struct Config {
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
}

// ============================================================================
// Shared building blocks (mirrored from siglip.rs but with attention_mask plumbing)
// ============================================================================

#[derive(Debug, Clone)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
    activation: candle_nn::Activation,
}

impl Mlp {
    fn new<C: TransformerConfig>(cfg: &C, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(cfg.hidden_size(), cfg.intermediate_size(), vb.pp("fc1"))?;
        let fc2 = linear(cfg.intermediate_size(), cfg.hidden_size(), vb.pp("fc2"))?;
        Ok(Self {
            fc1,
            fc2,
            activation: cfg.hidden_act(),
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.fc1)?.apply(&self.activation)?.apply(&self.fc2)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    fn new<C: TransformerConfig>(cfg: &C, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size();
        let num_heads = cfg.num_attention_heads();
        let head_dim = h / num_heads;
        Ok(Self {
            q_proj: linear(h, h, vb.pp("q_proj"))?,
            k_proj: linear(h, h, vb.pp("k_proj"))?,
            v_proj: linear(h, h, vb.pp("v_proj"))?,
            out_proj: linear(h, h, vb.pp("out_proj"))?,
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    fn forward(&self, xs: &Tensor, attn_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, n, _h) = xs.dims3()?;
        let q = xs
            .apply(&self.q_proj)?
            .reshape((b, n, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let k = xs
            .apply(&self.k_proj)?
            .reshape((b, n, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let v = xs
            .apply(&self.v_proj)?
            .reshape((b, n, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        // attn_weights: (b, num_heads, n, n)
        let attn_weights = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?.contiguous()?)? * self.scale)?;
        let attn_weights = match attn_mask {
            Some(m) => attn_weights.broadcast_add(m)?,
            None => attn_weights,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let out = attn_weights.matmul(&v)?;
        let out = out.permute((0, 2, 1, 3))?.contiguous()?.reshape((b, n, ()))?;
        out.apply(&self.out_proj)
    }
}

#[derive(Debug, Clone)]
struct EncoderLayer {
    layer_norm1: LayerNorm,
    self_attn: Attention,
    layer_norm2: LayerNorm,
    mlp: Mlp,
}

impl EncoderLayer {
    fn new<C: TransformerConfig>(cfg: &C, vb: VarBuilder) -> Result<Self> {
        let layer_norm1 = layer_norm(cfg.hidden_size(), cfg.layer_norm_eps(), vb.pp("layer_norm1"))?;
        let self_attn = Attention::new(cfg, vb.pp("self_attn"))?;
        let layer_norm2 = layer_norm(cfg.hidden_size(), cfg.layer_norm_eps(), vb.pp("layer_norm2"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        Ok(Self {
            layer_norm1,
            self_attn,
            layer_norm2,
            mlp,
        })
    }

    fn forward(&self, xs: &Tensor, attn_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs;
        let xs = xs.apply(&self.layer_norm1)?;
        let xs = self.self_attn.forward(&xs, attn_mask)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs2 = xs.apply(&self.layer_norm2)?;
        let xs2 = xs2.apply(&self.mlp)?;
        residual + xs2
    }
}

#[derive(Debug, Clone)]
struct Encoder {
    layers: Vec<EncoderLayer>,
}

impl Encoder {
    fn new<C: TransformerConfig>(cfg: &C, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers());
        for i in 0..cfg.num_hidden_layers() {
            layers.push(EncoderLayer::new(cfg, vb.pp(format!("layers.{i}")))?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, xs: &Tensor, attn_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in &self.layers {
            xs = layer.forward(&xs, attn_mask)?;
        }
        Ok(xs)
    }
}

// ============================================================================
// Vision tower (NaFlex-specific)
// ============================================================================

#[derive(Debug, Clone)]
struct VisionEmbeddings {
    patch_embedding: Linear,
    position_embedding: candle_nn::Embedding,
    base_grid_size: usize,
    hidden_size: usize,
}

impl VisionEmbeddings {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let in_features = cfg.num_channels * cfg.patch_size * cfg.patch_size;
        let patch_embedding = linear(in_features, cfg.hidden_size, vb.pp("patch_embedding"))?;
        let position_embedding =
            candle_nn::embedding(cfg.num_patches, cfg.hidden_size, vb.pp("position_embedding"))?;
        // base_grid_size = sqrt(num_patches); we expect num_patches to be a perfect square
        let base_grid_size = (cfg.num_patches as f64).sqrt().round() as usize;
        if base_grid_size * base_grid_size != cfg.num_patches {
            candle::bail!(
                "expected num_patches ({}) to be a perfect square for naflex base position grid",
                cfg.num_patches
            );
        }
        Ok(Self {
            patch_embedding,
            position_embedding,
            base_grid_size,
            hidden_size: cfg.hidden_size,
        })
    }

    /// Resize the base position embedding `[num_patches, hidden]` to a target
    /// `(target_h, target_w, hidden)` via bilinear interpolation, then flatten
    /// back to `[target_h * target_w, hidden]`.
    fn resize_positional_embeddings(
        &self,
        target_h: usize,
        target_w: usize,
        device: &candle::Device,
    ) -> Result<Tensor> {
        // base_pos: [num_patches, hidden] = [base*base, hidden]
        let base_pos = self.position_embedding.embeddings(); // [num_patches, hidden]
        // reshape to (1, hidden, base, base) for interpolate2d (NCHW)
        let base = self.base_grid_size;
        let base_pos = base_pos
            .reshape((base, base, self.hidden_size))?
            .permute((2, 0, 1))? // (hidden, base, base)
            .unsqueeze(0)?; // (1, hidden, base, base)
        let resized = if target_h == base && target_w == base {
            base_pos
        } else {
            base_pos.interpolate2d(target_h, target_w)?
        };
        // back to (target_h * target_w, hidden)
        let resized = resized
            .squeeze(0)? // (hidden, target_h, target_w)
            .permute((1, 2, 0))? // (target_h, target_w, hidden)
            .reshape((target_h * target_w, self.hidden_size))?;
        Ok(resized.to_device(device)?)
    }

    /// Forward for the simple "batch-uniform shape" case: every input in the
    /// batch has the same (target_h, target_w) and there are no padded
    /// positions. `pixel_values` shape is `(B, num_patches, num_channels *
    /// patch_size * patch_size)` and `num_patches == target_h * target_w`.
    fn forward_uniform(
        &self,
        pixel_values: &Tensor,
        target_h: usize,
        target_w: usize,
    ) -> Result<Tensor> {
        let (_b, n, _patch_dim) = pixel_values.dims3()?;
        if n != target_h * target_w {
            candle::bail!(
                "pixel_values num_patches ({}) does not match target_h * target_w ({})",
                n,
                target_h * target_w
            );
        }
        let patch_embeds = pixel_values.apply(&self.patch_embedding)?; // (B, N, hidden)
        let pos = self.resize_positional_embeddings(target_h, target_w, pixel_values.device())?; // (N, hidden)
        let pos = pos.unsqueeze(0)?; // (1, N, hidden) for broadcast
        patch_embeds.broadcast_add(&pos)
    }

    /// Variable-shape forward: each input in the batch may have a different
    /// `(h_i, w_i)`. `pixel_values` is padded to `max_num_patches` along the
    /// patch dimension; `pixel_attention_mask` (B, max_num_patches) marks
    /// real patches with 1 and padding with 0; `spatial_shapes` (B, 2) gives
    /// `[h_patches, w_patches]` per input. The position embedding is resized
    /// per-input and zero-padded to `max_num_patches`.
    fn forward_variable(
        &self,
        pixel_values: &Tensor,
        spatial_shapes: &Tensor,
    ) -> Result<Tensor> {
        let (b, max_n, _patch_dim) = pixel_values.dims3()?;
        let device = pixel_values.device();
        // patch_embeds: (B, max_n, hidden)
        let patch_embeds = pixel_values.apply(&self.patch_embedding)?;
        // For each input, build a (max_n, hidden) position embedding tensor
        // by resizing the base grid to (h_i, w_i) and zero-padding to max_n.
        let shapes_vec = spatial_shapes.to_vec2::<i64>()?;
        let mut per_input_pos: Vec<Tensor> = Vec::with_capacity(b);
        for shape_row in shapes_vec.iter() {
            let h_i = shape_row[0] as usize;
            let w_i = shape_row[1] as usize;
            let n_i = h_i * w_i;
            if n_i > max_n {
                candle::bail!(
                    "spatial_shapes implies {} patches for an input but max_num_patches is {}",
                    n_i,
                    max_n
                );
            }
            let pos = self.resize_positional_embeddings(h_i, w_i, device)?; // (n_i, hidden)
            let pos = if n_i < max_n {
                let pad = Tensor::zeros((max_n - n_i, self.hidden_size), pos.dtype(), device)?;
                Tensor::cat(&[&pos, &pad], 0)?
            } else {
                pos
            };
            per_input_pos.push(pos);
        }
        let pos_batch = Tensor::stack(&per_input_pos, 0)?; // (B, max_n, hidden)
        patch_embeds + pos_batch
    }
}

#[derive(Debug, Clone)]
struct MultiheadAttentionPoolingHead {
    probe: Tensor,
    in_proj_weight: Tensor, // (3*hidden, hidden) for combined Q/K/V
    in_proj_bias: Tensor,   // (3*hidden,)
    out_proj: Linear,
    layernorm: LayerNorm,
    mlp: Mlp,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl MultiheadAttentionPoolingHead {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let head_dim = h / num_heads;
        // probe: (1, 1, hidden) — single tensor at `<head>.probe`
        let probe = vb.get((1, 1, h), "probe")?;
        // The HF checkpoint stores `attention.in_proj_weight` (3H x H) and
        // `attention.in_proj_bias` (3H), corresponding to
        // `torch.nn.MultiheadAttention`'s combined Q/K/V projection.
        let in_proj_weight = vb.pp("attention").get((3 * h, h), "in_proj_weight")?;
        let in_proj_bias = vb.pp("attention").get(3 * h, "in_proj_bias")?;
        let out_proj = linear(h, h, vb.pp("attention").pp("out_proj"))?;
        let layernorm = layer_norm(h, cfg.layer_norm_eps, vb.pp("layernorm"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        Ok(Self {
            probe,
            in_proj_weight,
            in_proj_bias,
            out_proj,
            layernorm,
            mlp,
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    fn forward(&self, xs: &Tensor, attn_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, n, h) = xs.dims3()?;
        // Probe: (B, 1, H)
        let probe = self.probe.broadcast_as((b, 1, h))?;
        // Combined linear: (3H, H) split into Q, K, V projections.
        // For multihead attention on the probe (Q) attending to xs (K, V):
        //   Q = linear(probe, W_q, b_q) where W_q is in_proj_weight[0:H]
        //   K = linear(xs,    W_k, b_k) where W_k is in_proj_weight[H:2H]
        //   V = linear(xs,    W_v, b_v) where W_v is in_proj_weight[2H:3H]
        let w_q = self.in_proj_weight.narrow(0, 0, h)?;
        let w_k = self.in_proj_weight.narrow(0, h, h)?;
        let w_v = self.in_proj_weight.narrow(0, 2 * h, h)?;
        let b_q = self.in_proj_bias.narrow(0, 0, h)?;
        let b_k = self.in_proj_bias.narrow(0, h, h)?;
        let b_v = self.in_proj_bias.narrow(0, 2 * h, h)?;
        // Apply linear: y = x @ W^T + b
        let q = probe
            .broadcast_matmul(&w_q.t()?.contiguous()?)?
            .broadcast_add(&b_q)?
            .reshape((b, 1, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()?; // (B, num_heads, 1, head_dim)
        let k = xs
            .broadcast_matmul(&w_k.t()?.contiguous()?)?
            .broadcast_add(&b_k)?
            .reshape((b, n, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()?; // (B, num_heads, n, head_dim)
        let v = xs
            .broadcast_matmul(&w_v.t()?.contiguous()?)?
            .broadcast_add(&b_v)?
            .reshape((b, n, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let attn_weights = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?.contiguous()?)? * self.scale)?;
        // attn_mask shape expected: (B, 1, 1, n) or (B, num_heads, 1, n) — broadcast-compatible
        let attn_weights = match attn_mask {
            Some(m) => attn_weights.broadcast_add(m)?,
            None => attn_weights,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attended = attn_weights.matmul(&v)?; // (B, num_heads, 1, head_dim)
        let attended = attended
            .permute((0, 2, 1, 3))? // (B, 1, num_heads, head_dim)
            .contiguous()?
            .reshape((b, 1, h))?;
        let attended = attended.apply(&self.out_proj)?;
        // MLP block with residual
        let residual = &attended;
        let normed = attended.apply(&self.layernorm)?;
        let mlp_out = normed.apply(&self.mlp)?;
        let out = (residual + mlp_out)?;
        // squeeze to (B, hidden)
        out.i((.., 0, ..))
    }
}

#[derive(Debug, Clone)]
struct VisionTransformer {
    embeddings: VisionEmbeddings,
    encoder: Encoder,
    post_layernorm: LayerNorm,
    head: MultiheadAttentionPoolingHead,
}

impl VisionTransformer {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let embeddings = VisionEmbeddings::new(cfg, vb.pp("embeddings"))?;
        let encoder = Encoder::new(cfg, vb.pp("encoder"))?;
        let post_layernorm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("post_layernorm"))?;
        let head = MultiheadAttentionPoolingHead::new(cfg, vb.pp("head"))?;
        Ok(Self {
            embeddings,
            encoder,
            post_layernorm,
            head,
        })
    }

    /// `pixel_values` shape `(B, max_num_patches, num_channels*patch_size^2)`,
    /// `pixel_attention_mask` shape `(B, max_num_patches)` (1 = real, 0 = pad)
    /// or `None` for batch-uniform real patches,
    /// `spatial_shapes` shape `(B, 2)` giving `[h_patches, w_patches]` per input
    /// or `None` for the batch-uniform case where the caller provides a
    /// `(target_h, target_w)`.
    pub fn forward(
        &self,
        pixel_values: &Tensor,
        pixel_attention_mask: Option<&Tensor>,
        spatial_shapes: Option<&Tensor>,
        target_uniform: Option<(usize, usize)>,
    ) -> Result<Tensor> {
        // Embeddings
        let xs = match (spatial_shapes, target_uniform) {
            (Some(shapes), _) => self.embeddings.forward_variable(pixel_values, shapes)?,
            (None, Some((h, w))) => self.embeddings.forward_uniform(pixel_values, h, w)?,
            (None, None) => candle::bail!(
                "must provide either `spatial_shapes` (variable) or `target_uniform` (uniform)"
            ),
        };
        // Convert pixel_attention_mask to additive attention bias (B, 1, 1, N)
        let attn_bias = match pixel_attention_mask {
            Some(m) => Some(make_4d_additive_mask(m, xs.dtype())?),
            None => None,
        };
        let xs = self.encoder.forward(&xs, attn_bias.as_ref())?;
        let xs = xs.apply(&self.post_layernorm)?;
        let pooled = self.head.forward(&xs, attn_bias.as_ref())?;
        Ok(pooled)
    }
}

/// Convert a 2D pixel_attention_mask `(B, N)` of 1/0 to an additive 4D mask
/// `(B, 1, 1, N)` of 0 / -inf suitable for adding to attention logits before
/// softmax. Mirrors HF transformers' `_create_4d_causal_attention_mask`-style
/// helper for the bidirectional case.
fn make_4d_additive_mask(mask_2d: &Tensor, dtype: DType) -> Result<Tensor> {
    // Convert 1/0 mask to 0 / -1e4 additive bias. Using -1e4 keeps fp16 safe;
    // for fp32 it is well within representable range and produces effectively
    // zero contribution after softmax.
    let mask = mask_2d.to_dtype(dtype)?;
    let neg_large = (mask.zeros_like()? - 1e4f64)?;
    let one_minus_mask = (mask.ones_like()? - mask)?;
    let additive = one_minus_mask.broadcast_mul(&neg_large)?;
    additive.unsqueeze(1)?.unsqueeze(1)
}

#[derive(Debug, Clone)]
pub struct VisionModel {
    transformer: VisionTransformer,
}

impl VisionModel {
    pub fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let transformer = VisionTransformer::new(cfg, vb)?;
        Ok(Self { transformer })
    }

    pub fn forward(
        &self,
        pixel_values: &Tensor,
        pixel_attention_mask: Option<&Tensor>,
        spatial_shapes: Option<&Tensor>,
        target_uniform: Option<(usize, usize)>,
    ) -> Result<Tensor> {
        self.transformer.forward(pixel_values, pixel_attention_mask, spatial_shapes, target_uniform)
    }
}

// ============================================================================
// Text tower (essentially identical to v1)
// ============================================================================

#[derive(Debug, Clone)]
struct TextEmbeddings {
    token_embedding: candle_nn::Embedding,
    position_embedding: candle_nn::Embedding,
    position_ids: Tensor,
}

impl TextEmbeddings {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let token_embedding =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("token_embedding"))?;
        let position_embedding = candle_nn::embedding(
            cfg.max_position_embeddings,
            cfg.hidden_size,
            vb.pp("position_embedding"),
        )?;
        let position_ids =
            Tensor::arange(0u32, cfg.max_position_embeddings as u32, vb.device())?.unsqueeze(0)?;
        Ok(Self {
            token_embedding,
            position_embedding,
            position_ids,
        })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_len = input_ids.dim(D::Minus1)?;
        let pos_ids = self.position_ids.narrow(1, 0, seq_len)?;
        let token_emb = input_ids.apply(&self.token_embedding)?;
        let pos_emb = pos_ids.apply(&self.position_embedding)?;
        token_emb.broadcast_add(&pos_emb)
    }
}

#[derive(Debug, Clone)]
pub struct TextTransformer {
    embeddings: TextEmbeddings,
    encoder: Encoder,
    final_layer_norm: LayerNorm,
    head: Linear,
}

impl TextTransformer {
    pub fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let embeddings = TextEmbeddings::new(cfg, vb.pp("embeddings"))?;
        let encoder = Encoder::new(cfg, vb.pp("encoder"))?;
        let final_layer_norm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("final_layer_norm"))?;
        let head = linear(cfg.hidden_size, cfg.projection_size, vb.pp("head"))?;
        Ok(Self {
            embeddings,
            encoder,
            final_layer_norm,
            head,
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let xs = self.embeddings.forward(input_ids)?;
        // SigLIP text encoder uses no attention mask (full attention over fixed-length input)
        let xs = self.encoder.forward(&xs, None)?;
        let xs = xs.apply(&self.final_layer_norm)?;
        // Pool last token: xs[:, -1, :]
        let last = xs.i((.., xs.dim(1)? - 1, ..))?;
        last.apply(&self.head)
    }
}

#[derive(Debug, Clone)]
pub struct TextModel {
    transformer: TextTransformer,
}

impl TextModel {
    pub fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            transformer: TextTransformer::new(cfg, vb)?,
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.transformer.forward(input_ids)
    }
}

// ============================================================================
// Top-level Model
// ============================================================================

#[derive(Debug, Clone)]
pub struct Model {
    text_model: TextModel,
    vision_model: VisionModel,
    logit_scale: Tensor,
    logit_bias: Tensor,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let text_model = TextModel::new(&cfg.text_config, vb.pp("text_model"))?;
        let vision_model = VisionModel::new(&cfg.vision_config, vb.pp("vision_model"))?;
        let logit_scale = vb.get(1, "logit_scale")?;
        let logit_bias = vb.get(1, "logit_bias")?;
        Ok(Self {
            text_model,
            vision_model,
            logit_scale,
            logit_bias,
        })
    }

    /// Forward pass returning `(logits_per_text, logits_per_image)`.
    pub fn forward(
        &self,
        pixel_values: &Tensor,
        pixel_attention_mask: Option<&Tensor>,
        spatial_shapes: Option<&Tensor>,
        target_uniform: Option<(usize, usize)>,
        input_ids: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let image_embeds = self.vision_model.forward(
            pixel_values,
            pixel_attention_mask,
            spatial_shapes,
            target_uniform,
        )?;
        let text_embeds = self.text_model.forward(input_ids)?;
        // Normalize
        let image_embeds = l2_norm(&image_embeds)?;
        let text_embeds = l2_norm(&text_embeds)?;
        // logits_per_text = text_embeds @ image_embeds.T
        let logits_per_text = text_embeds.matmul(&image_embeds.t()?)?;
        // Apply scale + bias
        let logit_scale = self.logit_scale.exp()?;
        let logits_per_text = logits_per_text.broadcast_mul(&logit_scale)?;
        let logits_per_text = logits_per_text.broadcast_add(&self.logit_bias)?;
        let logits_per_image = logits_per_text.t()?;
        Ok((logits_per_text, logits_per_image))
    }
}

fn l2_norm(xs: &Tensor) -> Result<Tensor> {
    let norm = xs.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    xs.broadcast_div(&norm)
}
