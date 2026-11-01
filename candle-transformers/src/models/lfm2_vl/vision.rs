//! SigLIP2 NaFlex vision tower, as used by LFM2-VL (no pooling head).

use candle::{DType, Module, Result, Tensor};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, VarBuilder};

use super::config::Siglip2VisionConfig;

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip2/modeling_siglip2.py
#[derive(Debug, Clone)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
    activation_fn: candle_nn::Activation,
}

impl Mlp {
    fn new(cfg: &Siglip2VisionConfig, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("fc1"))?;
        let fc2 = linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("fc2"))?;
        Ok(Self {
            fc1,
            fc2,
            activation_fn: cfg.hidden_act,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.fc1)?
            .apply(&self.activation_fn)?
            .apply(&self.fc2)
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
    fn new(cfg: &Siglip2VisionConfig, vb: VarBuilder) -> Result<Self> {
        let embed_dim = cfg.hidden_size;
        let q_proj = linear(embed_dim, embed_dim, vb.pp("q_proj"))?;
        let k_proj = linear(embed_dim, embed_dim, vb.pp("k_proj"))?;
        let v_proj = linear(embed_dim, embed_dim, vb.pp("v_proj"))?;
        let out_proj = linear(embed_dim, embed_dim, vb.pp("out_proj"))?;
        let num_heads = cfg.num_attention_heads;
        let head_dim = embed_dim / num_heads;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (batch_size, q_len, _) = xs.dims3()?;
        let query_states = xs.apply(&self.q_proj)?;
        let key_states = xs.apply(&self.k_proj)?;
        let value_states = xs.apply(&self.v_proj)?;

        let shape = (batch_size, q_len, self.num_heads, self.head_dim);
        let query_states = query_states.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let key_states = key_states.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let value_states = value_states.reshape(shape)?.transpose(1, 2)?.contiguous()?;

        let attn_weights = (query_states.matmul(&key_states.t()?)? * self.scale)?;
        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };
        // The original implementation upcasts to f32 but candle_nn::ops::softmax should handle this properly.
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_outputs = attn_weights
            .matmul(&value_states)?
            .transpose(1, 2)?
            .reshape((batch_size, q_len, ()))?
            .apply(&self.out_proj)?;
        Ok(attn_outputs)
    }
}

#[derive(Debug, Clone)]
struct EncoderLayer {
    self_attn: Attention,
    layer_norm1: LayerNorm,
    mlp: Mlp,
    layer_norm2: LayerNorm,
}

impl EncoderLayer {
    fn new(cfg: &Siglip2VisionConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let layer_norm_eps = cfg.layer_norm_eps;
        let self_attn = Attention::new(cfg, vb.pp("self_attn"))?;
        let layer_norm1 = layer_norm(hidden_size, layer_norm_eps, vb.pp("layer_norm1"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let layer_norm2 = layer_norm(hidden_size, layer_norm_eps, vb.pp("layer_norm2"))?;
        Ok(Self {
            self_attn,
            layer_norm1,
            mlp,
            layer_norm2,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs;
        let xs = xs.apply(&self.layer_norm1)?;
        let xs = self.self_attn.forward(&xs, attention_mask)?;
        let xs = (residual + xs)?;
        let residual = &xs;
        let xs = xs.apply(&self.layer_norm2)?.apply(&self.mlp)?;
        let xs = (xs + residual)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct Encoder {
    layers: Vec<EncoderLayer>,
}

impl Encoder {
    fn new(cfg: &Siglip2VisionConfig, vb: VarBuilder) -> Result<Self> {
        let mut layers = vec![];
        let vb = vb.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = EncoderLayer::new(cfg, vb.pp(layer_idx))?;
            layers.push(layer)
        }
        Ok(Self { layers })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, attention_mask)?
        }
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct VisionEmbeddings {
    patch_embedding: Linear,
    /// Learned `(num_patches, hidden_size)` table, read as a square grid and resized to each
    /// image's patch grid.
    position_embedding: Tensor,
    /// Side of the square position grid (16 for 256 positions).
    position_embedding_size: usize,
}

impl VisionEmbeddings {
    fn new(cfg: &Siglip2VisionConfig, vb: VarBuilder) -> Result<Self> {
        // NaFlex patches are already flattened, so the patch embedding is a linear layer.
        let patch_embedding = linear(
            cfg.num_channels * cfg.patch_size.pow(2),
            cfg.hidden_size,
            vb.pp("patch_embedding"),
        )?;
        let position_embedding = vb
            .pp("position_embedding")
            .get((cfg.num_patches, cfg.hidden_size), "weight")?;
        let position_embedding_size = cfg.position_embedding_size();
        Ok(Self {
            patch_embedding,
            position_embedding,
            position_embedding_size,
        })
    }

    fn forward(&self, pixel_values: &Tensor, spatial_shapes: &[(usize, usize)]) -> Result<Tensor> {
        let patch_embeds = pixel_values.apply(&self.patch_embedding)?;
        // Position embeddings resized to each image's grid, padded to the sequence length.
        let max_len = pixel_values.dim(1)?;
        let positions = spatial_shapes
            .iter()
            .map(|&(h, w)| self.resized_positions(h, w, max_len))
            .collect::<Result<Vec<_>>>()?;
        let positions = Tensor::stack(&positions, 0)?;
        patch_embeds + positions
    }

    fn resized_positions(&self, h: usize, w: usize, max_len: usize) -> Result<Tensor> {
        let pos = self.position_embedding.reshape((
            self.position_embedding_size,
            self.position_embedding_size,
            (),
        ))?;
        let resized = pos
            .permute((2, 0, 1))?
            .unsqueeze(0)?
            .to_dtype(DType::F32)?
            // transformers uses `antialias=True`, which candle lacks. Matches it
            // when upsampling (h, w >= 16, e.g. 512px tiles), but diverges when a
            // side has fewer than 16 patches (max abs diff ~1.0 for 8x40).
            .upsample_bilinear2d(h, w, false)?
            .squeeze(0)?
            .reshape(((), h * w))?
            .t()?
            .to_dtype(self.position_embedding.dtype())?;

        // Padded positions reuse the first one, as in transformers.
        let n = h * w;
        if n == max_len {
            return resized.contiguous();
        }
        let pad = resized.narrow(0, 0, 1)?.repeat((max_len - n, 1))?;
        Tensor::cat(&[&resized, &pad], 0)
    }
}
/// Turns a `(N, L)` padding mask of 1/0 into an additive `(N, 1, 1, L)` mask: 0 for real
/// patches, a large negative value for padding, so padded keys get no attention weight.
fn get_extended_attention_mask(attention_mask: &Tensor, dtype: DType) -> Result<Tensor> {
    // Computed in f32: `f32::MIN` rounds to -inf in bf16/f16, and `0 * -inf` would be NaN.
    attention_mask
        .unsqueeze(1)?
        .unsqueeze(1)?
        .to_dtype(DType::F32)?
        .affine(-(f32::MIN as f64), f32::MIN as f64)?
        .to_dtype(dtype)
}

#[derive(Debug, Clone)]
pub struct Siglip2VisionModel {
    embeddings: VisionEmbeddings,
    encoder: Encoder,
    post_layernorm: LayerNorm,
}

impl Siglip2VisionModel {
    pub fn new(cfg: &Siglip2VisionConfig, vb: VarBuilder) -> Result<Self> {
        let embeddings = VisionEmbeddings::new(cfg, vb.pp("embeddings"))?;
        let encoder = Encoder::new(cfg, vb.pp("encoder"))?;
        let post_layernorm =
            layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("post_layernorm"))?;
        Ok(Self {
            embeddings,
            encoder,
            post_layernorm,
        })
    }

    /// `pixel_values`: `(N, max_num_patches, C * patch_size^2)`, `spatial_shapes`: `(h, w)` in
    /// patches per image, `pixel_attention_mask`: `(N, max_num_patches)` with 1 for real patches.
    /// Returns the last hidden state `(N, max_num_patches, hidden_size)`.
    pub fn forward(
        &self,
        pixel_values: &Tensor,
        spatial_shapes: &[(usize, usize)],
        pixel_attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let xs = self.embeddings.forward(pixel_values, spatial_shapes)?;
        let mask = get_extended_attention_mask(pixel_attention_mask, xs.dtype())?;
        self.encoder
            .forward(&xs, Some(&mask))?
            .apply(&self.post_layernorm)
    }
}
