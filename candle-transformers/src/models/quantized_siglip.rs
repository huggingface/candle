//! Quantized SigLIP vision encoder for GGUF weights.
//!
//! Loads from llama.cpp GGUF mmproj files using the standard `v.blk.*` tensor naming.
//! Reusable by any Idefics3/SigLIP-based multimodal model (Granite-Docling, SmolVLM, etc.).

use candle::{Module, Result, Tensor};
use crate::quantized_nn::{self, Linear};
use crate::quantized_var_builder::VarBuilder;

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

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
    fn new(hidden_size: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        let q_proj = quantized_nn::linear(hidden_size, hidden_size, vb.pp("attn_q"))?;
        let k_proj = quantized_nn::linear(hidden_size, hidden_size, vb.pp("attn_k"))?;
        let v_proj = quantized_nn::linear(hidden_size, hidden_size, vb.pp("attn_v"))?;
        let out_proj = quantized_nn::linear(hidden_size, hidden_size, vb.pp("attn_out"))?;
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

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, seq_len, _) = xs.dims3()?;
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let shape = (b, seq_len, self.num_heads, self.head_dim);
        let q = q.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape(shape)?.transpose(1, 2)?.contiguous()?;

        let attn = (q.matmul(&k.t()?)? * self.scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        attn.matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, seq_len, ()))?
            .apply(&self.out_proj)
    }
}

// ---------------------------------------------------------------------------
// MLP
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = quantized_nn::linear(hidden_size, intermediate_size, vb.pp("ffn_up"))?;
        let fc2 = quantized_nn::linear(intermediate_size, hidden_size, vb.pp("ffn_down"))?;
        Ok(Self { fc1, fc2 })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.fc1)?
            .apply(&candle_nn::Activation::GeluPytorchTanh)?
            .apply(&self.fc2)
    }
}

// ---------------------------------------------------------------------------
// Encoder Layer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct EncoderLayer {
    self_attn: Attention,
    layer_norm1: candle_nn::LayerNorm,
    mlp: Mlp,
    layer_norm2: candle_nn::LayerNorm,
}

impl EncoderLayer {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
        layer_norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn = Attention::new(hidden_size, num_heads, vb.clone())?;
        let layer_norm1 = quantized_nn::layer_norm(hidden_size, layer_norm_eps, vb.pp("ln1"))?;
        let mlp = Mlp::new(hidden_size, intermediate_size, vb.clone())?;
        let layer_norm2 = quantized_nn::layer_norm(hidden_size, layer_norm_eps, vb.pp("ln2"))?;
        Ok(Self {
            self_attn,
            layer_norm1,
            mlp,
            layer_norm2,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = xs.apply(&self.layer_norm1)?;
        let xs = self.self_attn.forward(&xs)?;
        let xs = (residual + xs)?;
        let residual = &xs;
        let xs = xs.apply(&self.layer_norm2)?.apply(&self.mlp)?;
        residual + xs
    }
}

// ---------------------------------------------------------------------------
// Vision Embeddings
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct VisionEmbeddings {
    patch_embedding_weight: Tensor,
    patch_embedding_bias: Tensor,
    position_embedding: Tensor,
    patch_size: usize,
    hidden_size: usize,
}

impl VisionEmbeddings {
    fn new(
        hidden_size: usize,
        image_size: usize,
        patch_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let num_patches_per_side = image_size / patch_size;
        // Patch embedding stored as (patch_size, patch_size, 3, hidden_size) in GGUF
        let patch_embedding_weight = vb
            .get_no_shape("patch_embd.weight")?
            .dequantize(vb.device())?;
        let patch_embedding_bias = vb
            .get_no_shape("patch_embd.bias")?
            .dequantize(vb.device())?;
        // Position embedding stored as (hidden_size, num_patches) in GGUF
        let position_embedding = vb
            .get_no_shape("position_embd.weight")?
            .dequantize(vb.device())?;
        // Reshape position embedding: (hidden_size, num_patches) -> (1, num_patches, hidden_size)
        let position_embedding = if position_embedding.dim(0)? == hidden_size {
            position_embedding.t()?.unsqueeze(0)?
        } else {
            position_embedding.unsqueeze(0)?
        };

        // GGUF stores patch_embd.weight with dims [16, 16, 3, 768] in GGUF order.
        // Candle loads this as shape (768, 3, 16, 16) which is already Conv2d layout
        // (out_channels, in_channels, kH, kW). No reshape needed.
        let patch_embedding_weight = if patch_embedding_weight.dims() == [hidden_size, 3, patch_size, patch_size] {
            patch_embedding_weight
        } else {
            // Fallback: reshape from GGUF (patch_size, patch_size, 3, hidden_size) -> Conv2d
            patch_embedding_weight
                .reshape((patch_size, patch_size, 3, hidden_size))?
                .permute((3, 2, 0, 1))?
                .contiguous()?
        };

        let _ = num_patches_per_side; // used implicitly via position_embedding shape
        Ok(Self {
            patch_embedding_weight,
            patch_embedding_bias,
            position_embedding,
            patch_size,
            hidden_size,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_b, _c, _h, _w) = xs.dims4()?;
        // Manual Conv2d with stride = patch_size
        let embeddings = xs.conv2d(
            &self.patch_embedding_weight,
            0,              // padding
            self.patch_size, // stride
            1,              // dilation
            1,              // groups
        )?;
        let embeddings = embeddings.broadcast_add(
            &self.patch_embedding_bias.reshape((1, self.hidden_size, 1, 1))?,
        )?;
        // (B, hidden, H/p, W/p) -> (B, num_patches, hidden)
        let embeddings = embeddings.flatten_from(2)?.transpose(1, 2)?;
        // Add position embeddings
        embeddings.broadcast_add(&self.position_embedding)
    }
}

// ---------------------------------------------------------------------------
// Vision Model (public)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct VisionModel {
    embeddings: VisionEmbeddings,
    layers: Vec<EncoderLayer>,
    post_layernorm: candle_nn::LayerNorm,
}

impl VisionModel {
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        num_hidden_layers: usize,
        num_attention_heads: usize,
        image_size: usize,
        patch_size: usize,
        layer_norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let embeddings =
            VisionEmbeddings::new(hidden_size, image_size, patch_size, vb.clone())?;

        let vb_layers = vb.pp("blk");
        let mut layers = Vec::with_capacity(num_hidden_layers);
        for i in 0..num_hidden_layers {
            layers.push(EncoderLayer::new(
                hidden_size,
                intermediate_size,
                num_attention_heads,
                layer_norm_eps,
                vb_layers.pp(i),
            )?);
        }

        let post_layernorm =
            quantized_nn::layer_norm(hidden_size, layer_norm_eps, vb.pp("post_ln"))?;

        Ok(Self {
            embeddings,
            layers,
            post_layernorm,
        })
    }
}

impl Module for VisionModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut hidden = self.embeddings.forward(xs)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }
        hidden.apply(&self.post_layernorm)
    }
}
