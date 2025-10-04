//! Based on the BLIP paper from Salesforce Research.
//!
//! The blip-image-captioning model can generate captions for an input image.
//!
//! - ⚡ [Interactive Wasm Example](https://huggingface.co/spaces/radames/Candle-BLIP-Image-Captioning)
//! - 💻 [GH Link](https://github.com/salesforce/BLIP)
//! - 🤗 [HF Link](https://huggingface.co/Salesforce/blip-image-captioning-base)
//! - 📝 [Paper](https://arxiv.org/abs/2201.12086)
//!

use super::blip_text;
use super::with_tracing::{conv2d, linear, Conv2d, Linear};
use candle::{Module, Result, Tensor, D};
use candle_nn::{layer_norm, Conv2dConfig, LayerNorm, VarBuilder};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub projection_dim: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub hidden_act: candle_nn::Activation,
    pub layer_norm_eps: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub text_config: blip_text::Config,
    pub vision_config: VisionConfig,
    pub projection_dim: usize,
    pub image_text_hidden_size: usize,
}

impl Config {
    pub fn image_captioning_large() -> Self {
        let text_config = blip_text::Config {
            vocab_size: 30524,
            hidden_size: 768,
            encoder_hidden_size: 1024,
            intermediate_size: 3072,
            projection_dim: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            max_position_embeddings: 512,
            hidden_act: candle_nn::Activation::Gelu,
            layer_norm_eps: 1e-12,
            is_decoder: true,
        };
        let vision_config = VisionConfig {
            hidden_size: 1024,
            intermediate_size: 4096,
            projection_dim: 512,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            image_size: 384,
            patch_size: 16,
            hidden_act: candle_nn::Activation::Gelu,
            layer_norm_eps: 1e-5,
        };
        Self {
            text_config,
            vision_config,
            projection_dim: 512,
            image_text_hidden_size: 256,
        }
    }
}

#[derive(Debug, Clone)]
struct VisionEmbeddings {
    class_embedding: Tensor,
    patch_embedding: Conv2d,
    position_embedding: Tensor,
}

impl VisionEmbeddings {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let class_embedding = vb.get((1, 1, cfg.hidden_size), "class_embedding")?;
        let conv_cfg = Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        let patch_embedding = conv2d(
            3,
            cfg.hidden_size,
            cfg.patch_size,
            conv_cfg,
            vb.pp("patch_embedding"),
        )?;
        let num_patches1 = cfg.image_size / cfg.patch_size;
        let num_patches = num_patches1 * num_patches1;
        let num_positions = num_patches + 1;
        let position_embedding =
            vb.get((1, num_positions, cfg.hidden_size), "position_embedding")?;
        Ok(Self {
            class_embedding,
            patch_embedding,
            position_embedding,
        })
    }
}

impl Module for VisionEmbeddings {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let target_dtype = xs.dtype();
        let b_size = xs.dim(0)?;
        let patch_embeds = xs.apply(&self.patch_embedding)?.flatten_from(2)?.t()?;
        let d = self.class_embedding.dim(D::Minus1)?;
        let class_embeds = self
            .class_embedding
            .broadcast_as((b_size, 1, d))?
            .to_dtype(target_dtype)?;
        let embeddings = Tensor::cat(&[&class_embeds, &patch_embeds], 1)?;
        let position_embedding = self.position_embedding.narrow(1, 0, embeddings.dim(1)?)?;
        embeddings.broadcast_add(&position_embedding)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    qkv: Linear,
    projection: Linear,
    scale: f64,
    num_heads: usize,
}

impl Attention {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let embed_dim = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let head_dim = embed_dim / num_heads;
        let scale = 1f64 / (head_dim as f64).sqrt();
        let qkv = linear(embed_dim, 3 * embed_dim, vb.pp("qkv"))?;
        let projection = linear(embed_dim, embed_dim, vb.pp("projection"))?;
        Ok(Self {
            qkv,
            projection,
            scale,
            num_heads,
        })
    }

    fn forward(&self, xs: &Tensor, attn_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b_sz, tgt_len, embed_dim) = xs.dims3()?;
        let mixed_qkv = xs
            .apply(&self.qkv)?
            .reshape((b_sz, tgt_len, 3, self.num_heads, embed_dim / self.num_heads))?
            .permute((2, 0, 3, 1, 4))?;
        let query = mixed_qkv.get(0)?;
        let key = mixed_qkv.get(1)?;
        let value = mixed_qkv.get(2)?;
        let attention_scores = query.matmul(&key.t()?)?;
        let attention_scores = (attention_scores * self.scale)?;
        let attention_probs = candle_nn::ops::softmax_last_dim(&attention_scores)?;
        let attention_probs = match attn_mask {
            None => attention_probs,
            Some(attn_mask) => (attention_probs * attn_mask)?,
        };
        attention_probs
            .matmul(&value)?
            .permute((0, 2, 1, 3))?
            .flatten_from(D::Minus2)?
            .apply(&self.projection)
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    activation_fn: candle_nn::Activation,
    fc1: Linear,
    fc2: Linear,
}

impl MLP {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("fc1"))?;
        let fc2 = linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("fc2"))?;
        Ok(Self {
            activation_fn: cfg.hidden_act,
            fc1,
            fc2,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.fc1)?
            .apply(&self.activation_fn)?
            .apply(&self.fc2)
    }
}

#[derive(Debug, Clone)]
struct EncoderLayer {
    self_attn: Attention,
    layer_norm1: LayerNorm,
    mlp: MLP,
    layer_norm2: LayerNorm,
}

impl EncoderLayer {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let embed_dim = cfg.hidden_size;
        let self_attn = Attention::new(cfg, vb.pp("self_attn"))?;
        let layer_norm1 = layer_norm(embed_dim, cfg.layer_norm_eps, vb.pp("layer_norm1"))?;
        let layer_norm2 = layer_norm(embed_dim, cfg.layer_norm_eps, vb.pp("layer_norm2"))?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
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
        let xs = (xs + residual)?;

        let residual = &xs;
        let xs = xs.apply(&self.layer_norm2)?.apply(&self.mlp)?;
        xs + residual
    }
}

#[derive(Debug, Clone)]
struct Encoder {
    layers: Vec<EncoderLayer>,
}

impl Encoder {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb = vb.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            let layer = EncoderLayer::new(cfg, vb.pp(i))?;
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
pub struct VisionModel {
    embeddings: VisionEmbeddings,
    encoder: Encoder,
    post_layernorm: LayerNorm,
}

impl VisionModel {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
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
}

impl Module for VisionModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.apply(&self.embeddings)?;
        let encoder_outputs = self.encoder.forward(&xs, None)?;
        // Return the last hidden state rather than pooled outputs.
        encoder_outputs.apply(&self.post_layernorm)
    }
}

#[derive(Debug, Clone)]
pub struct BlipForConditionalGeneration {
    vision_model: VisionModel,
    text_decoder: blip_text::TextLMHeadModel,
}

impl BlipForConditionalGeneration {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vision_model = VisionModel::new(&cfg.vision_config, vb.pp("vision_model"))?;
        let text_decoder =
            blip_text::TextLMHeadModel::new(&cfg.text_config, vb.pp("text_decoder"))?;
        Ok(Self {
            vision_model,
            text_decoder,
        })
    }

    pub fn vision_model(&self) -> &VisionModel {
        &self.vision_model
    }

    pub fn text_decoder(&mut self) -> &mut blip_text::TextLMHeadModel {
        &mut self.text_decoder
    }

    pub fn reset_kv_cache(&mut self) {
        self.text_decoder.reset_kv_cache();
    }
}
