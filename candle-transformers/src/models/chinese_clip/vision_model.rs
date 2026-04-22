//! Chinese contrastive Language-Image Pre-Training
//!
//! Chinese contrastive Language-Image Pre-Training (CLIP) is an architecture trained on
//! pairs of images with related texts.
//!
//! - 💻 [Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)
//! - 💻 [GH](https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/chinese_clip/modeling_chinese_clip.py_

use candle::{Context, DType, IndexOp, Module, Result, Shape, Tensor, D};
use candle_nn as nn;

use super::{Activation, EncoderConfig};

#[derive(Clone, Debug)]
pub struct ChineseClipVisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub projection_dim: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub hidden_act: Activation,
    pub layer_norm_eps: f64,
    pub attention_dropout: f32,
    pub initializer_range: f32,
    pub initializer_factor: f32,
}

impl Default for ChineseClipVisionConfig {
    fn default() -> Self {
        ChineseClipVisionConfig {
            hidden_size: 768,
            intermediate_size: 3072,
            projection_dim: 512,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            num_channels: 3,
            image_size: 224,
            patch_size: 32,
            hidden_act: Activation::QuickGelu,
            layer_norm_eps: 1e-5,
            attention_dropout: 0.0,
            initializer_range: 0.02,
            initializer_factor: 1.0,
        }
    }
}

impl ChineseClipVisionConfig {
    /// [referer](https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16/blob/main/config.json)
    pub fn clip_vit_base_patch16() -> Self {
        Self {
            hidden_size: 768,
            intermediate_size: 3072,
            projection_dim: 512,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            num_channels: 3,
            image_size: 224,
            patch_size: 16,
            hidden_act: Activation::QuickGelu,
            layer_norm_eps: 1e-5,
            attention_dropout: 0.0,
            initializer_range: 0.02,
            initializer_factor: 1.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ChineseClipVisionEmbeddings {
    patch_embedding: nn::Conv2d,
    position_ids: Tensor,
    class_embedding: Tensor,
    position_embedding: nn::Embedding,
}

impl ChineseClipVisionEmbeddings {
    pub fn new(var: nn::VarBuilder, config: &ChineseClipVisionConfig) -> Result<Self> {
        let embed_dim = config.hidden_size;
        // originally nn.Parameter
        let class_embedding = if var.contains_tensor("class_embedding") {
            var.get(embed_dim, "class_embedding")?
        } else {
            Tensor::randn(0f32, 1f32, embed_dim, var.device())?
        };

        let num_patches = (config.image_size / config.patch_size).pow(2);
        let num_positions = num_patches + 1;
        let position_ids = Tensor::arange(0, num_positions as i64, var.device())?;

        let conv2dconfig = nn::Conv2dConfig {
            stride: config.patch_size,
            ..Default::default()
        };
        let position_embedding =
            nn::embedding(num_positions, embed_dim, var.pp("position_embedding"))?;
        let patch_embedding = nn::conv2d_no_bias(
            config.num_channels,
            embed_dim,
            config.patch_size,
            conv2dconfig,
            var.pp("patch_embedding"),
        )?;
        Ok(Self {
            patch_embedding,
            position_ids,
            class_embedding,
            position_embedding,
        })
    }
}

impl Module for ChineseClipVisionEmbeddings {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let batch_size = xs.shape().dims();
        let patch_embeds = self
            .patch_embedding
            .forward(xs)?
            .flatten_from(2)?
            .transpose(1, 2)?;
        let shape = Shape::from((batch_size[0], 1, self.class_embedding.dim(D::Minus1)?));
        let class_embeds = self.class_embedding.expand(shape)?;
        let embeddings = Tensor::cat(&[class_embeds, patch_embeds], 1)?;
        let position_embedding = self.position_embedding.forward(&self.position_ids)?;
        embeddings.broadcast_add(&position_embedding)
    }
}

#[derive(Clone, Debug)]
struct ChineseClipVisionAttention {
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    q_proj: nn::Linear,
    out_proj: nn::Linear,
    head_dim: usize,
    scale: f64,
    num_attention_heads: usize,
}

impl ChineseClipVisionAttention {
    fn new(var: nn::VarBuilder, config: &EncoderConfig) -> Result<Self> {
        let embed_dim = config.embed_dim();
        let num_attention_heads = config.num_attention_heads();
        let k_proj = nn::linear(embed_dim, embed_dim, var.pp("k_proj"))?;
        let v_proj = nn::linear(embed_dim, embed_dim, var.pp("v_proj"))?;
        let q_proj = nn::linear(embed_dim, embed_dim, var.pp("q_proj"))?;
        let out_proj = nn::linear(embed_dim, embed_dim, var.pp("out_proj"))?;
        let head_dim = embed_dim / num_attention_heads;
        let scale = (head_dim as f64).powf(-0.5);

        Ok(ChineseClipVisionAttention {
            k_proj,
            v_proj,
            q_proj,
            out_proj,
            head_dim,
            scale,
            num_attention_heads,
        })
    }

    fn shape(&self, xs: &Tensor, seq_len: usize, bsz: usize) -> Result<Tensor> {
        xs.reshape((bsz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn forward(&self, xs: &Tensor, causal_attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let in_dtype = xs.dtype();
        let (bsz, seq_len, embed_dim) = xs.dims3()?;

        let proj_shape = (bsz * self.num_attention_heads, seq_len, self.head_dim);
        let query_states = self
            .shape(&(self.q_proj.forward(xs)? * self.scale)?, seq_len, bsz)?
            .reshape(proj_shape)?
            .to_dtype(DType::F32)?;
        let key_states = self
            .shape(&self.k_proj.forward(xs)?, seq_len, bsz)?
            .reshape(proj_shape)?
            .to_dtype(DType::F32)?;
        let value_states = self
            .shape(&self.v_proj.forward(xs)?, seq_len, bsz)?
            .reshape(proj_shape)?
            .to_dtype(DType::F32)?;

        let attn_weights = query_states.matmul(&key_states.transpose(1, 2)?)?;

        let src_len = key_states.dim(1)?;

        let attn_weights = if let Some(causal_attention_mask) = causal_attention_mask {
            attn_weights
                .reshape((bsz, self.num_attention_heads, seq_len, src_len))?
                .broadcast_add(causal_attention_mask)?
                .reshape((bsz * self.num_attention_heads, seq_len, src_len))?
        } else {
            attn_weights
        };

        let attn_weights = nn::ops::softmax(&attn_weights, D::Minus1)?;

        let attn_output = attn_weights.matmul(&value_states)?.to_dtype(in_dtype)?;
        let attn_output = attn_output
            .reshape((bsz, self.num_attention_heads, seq_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((bsz, seq_len, embed_dim))?;
        self.out_proj.forward(&attn_output)
    }
}

#[derive(Clone, Debug)]
struct ChineseClipVisionMlp {
    fc1: nn::Linear,
    fc2: nn::Linear,
    activation: Activation,
}

impl ChineseClipVisionMlp {
    fn new(var: nn::VarBuilder, config: &EncoderConfig) -> Result<Self> {
        let fc1 = nn::linear(
            config.embed_dim(),
            config.intermediate_size(),
            var.pp("fc1"),
        )?;
        let fc2 = nn::linear(
            config.intermediate_size(),
            config.embed_dim(),
            var.pp("fc2"),
        )?;

        Ok(ChineseClipVisionMlp {
            fc1,
            fc2,
            activation: config.activation(),
        })
    }
}

impl ChineseClipVisionMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?;
        self.fc2.forward(&self.activation.forward(&xs)?)
    }
}

#[derive(Clone, Debug)]
struct ChineseClipVisionEncoderLayer {
    self_attn: ChineseClipVisionAttention,
    layer_norm1: nn::LayerNorm,
    mlp: ChineseClipVisionMlp,
    layer_norm2: nn::LayerNorm,
}

impl ChineseClipVisionEncoderLayer {
    fn new(var: nn::VarBuilder, config: &EncoderConfig) -> Result<Self> {
        let self_attn = ChineseClipVisionAttention::new(var.pp("self_attn"), config)?;
        let layer_norm1 = nn::layer_norm(
            config.embed_dim(),
            config.layer_norm_eps(),
            var.pp("layer_norm1"),
        )?;
        let mlp = ChineseClipVisionMlp::new(var.pp("mlp"), config)?;
        let layer_norm2 = nn::layer_norm(
            config.embed_dim(),
            config.layer_norm_eps(),
            var.pp("layer_norm2"),
        )?;

        Ok(ChineseClipVisionEncoderLayer {
            self_attn,
            layer_norm1,
            mlp,
            layer_norm2,
        })
    }

    fn forward(&self, xs: &Tensor, causal_attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs;
        let xs = self.layer_norm1.forward(xs)?;
        let xs = self.self_attn.forward(&xs, causal_attention_mask)?;
        let xs = (xs + residual)?;

        let residual = &xs;
        let xs = self.layer_norm2.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        xs + residual
    }
}

#[derive(Clone, Debug)]
pub struct ChineseClipVisionEncoder {
    layers: Vec<ChineseClipVisionEncoderLayer>,
}

impl ChineseClipVisionEncoder {
    pub fn new(var: nn::VarBuilder, config: &EncoderConfig) -> Result<Self> {
        let vs = var.pp("layers");
        let mut layers: Vec<ChineseClipVisionEncoderLayer> = Vec::new();
        for index in 0..config.num_hidden_layers() {
            let layer = ChineseClipVisionEncoderLayer::new(vs.pp(index.to_string()), config)?;
            layers.push(layer)
        }
        Ok(ChineseClipVisionEncoder { layers })
    }

    pub fn forward(&self, xs: &Tensor, causal_attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, causal_attention_mask)?;
        }
        Ok(xs)
    }

    // required by LLaVA
    pub fn output_hidden_states(
        &self,
        xs: &Tensor,
        causal_attention_mask: Option<&Tensor>,
    ) -> Result<Vec<Tensor>> {
        let mut xs = xs.clone();
        let mut hidden_states = Vec::new();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, causal_attention_mask)?;
            hidden_states.push(xs.clone());
        }
        Ok(hidden_states)
    }
}

#[derive(Clone, Debug)]
pub struct ChineseClipVisionTransformer {
    embeddings: ChineseClipVisionEmbeddings,
    encoder: ChineseClipVisionEncoder,
    pre_layer_norm: nn::LayerNorm,
    final_layer_norm: nn::LayerNorm,
}

impl ChineseClipVisionTransformer {
    pub fn new(var: nn::VarBuilder, config: &ChineseClipVisionConfig) -> Result<Self> {
        let embed_dim = config.hidden_size;
        let embeddings = ChineseClipVisionEmbeddings::new(var.pp("embeddings"), config)?;
        let pre_layer_norm =
            nn::layer_norm(embed_dim, config.layer_norm_eps, var.pp("pre_layrnorm"))?;
        let encoder = ChineseClipVisionEncoder::new(
            var.pp("encoder"),
            &EncoderConfig::Vision(config.clone()),
        )?;
        let final_layer_norm =
            nn::layer_norm(embed_dim, config.layer_norm_eps, var.pp("post_layernorm"))?;
        Ok(Self {
            embeddings,
            encoder,
            final_layer_norm,
            pre_layer_norm,
        })
    }
    // required by LLaVA
    pub fn output_hidden_states(&self, pixel_values: &Tensor) -> Result<Vec<Tensor>> {
        let hidden_states = pixel_values
            .apply(&self.embeddings)?
            .apply(&self.pre_layer_norm)?;

        let mut result = self.encoder.output_hidden_states(&hidden_states, None)?;
        let encoder_outputs = result.last().context("no last")?;
        let pooled_output = encoder_outputs.i((.., 0, ..))?;
        result.push(self.final_layer_norm.forward(&pooled_output)?.clone());
        Ok(result)
    }
}

impl Module for ChineseClipVisionTransformer {
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let hidden_states = pixel_values
            .apply(&self.embeddings)?
            .apply(&self.pre_layer_norm)?;

        let encoder_outputs = self.encoder.forward(&hidden_states, None)?;

        // referer: https://github.com/huggingface/transformers/blob/f6fa0f0bf0796ac66f201f23bdb8585de1609add/src/transformers/models/clip/modeling_clip.py#L787
        let pooled_output = encoder_outputs.i((.., 0, ..))?;
        self.final_layer_norm.forward(&pooled_output)
    }
}
