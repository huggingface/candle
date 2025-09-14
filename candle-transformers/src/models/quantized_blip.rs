//! BLIP model implementation with quantization support.
//!
//! BLIP is a vision-language model for image understanding and generation tasks.
//! This implementation provides quantization for reduced memory and compute.
//!
//! Key characteristics:
//! - Vision encoder using ViT architecture
//! - Text decoder using BERT-style transformer
//! - Cross-attention between vision and text features
//! - Support for 8-bit quantization
//!
//! References:
//! - [BLIP Paper](https://arxiv.org/abs/2201.12086)
//! - [Hugging Face Implementation](https://huggingface.co/docs/transformers/model_doc/blip)
//!

use super::quantized_blip_text as blip_text;
use crate::quantized_nn::{layer_norm, linear, Linear};
pub use crate::quantized_var_builder::VarBuilder;
use candle::{quantized::QuantizedBackend, Module, Result, Tensor, D};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm};

pub type VisionConfig = super::blip::VisionConfig;
pub type Config = super::blip::Config;

#[derive(Debug, Clone)]
struct VisionEmbeddings<QB: QuantizedBackend> {
    class_embedding: Tensor<QB::Storage>,
    patch_embedding: Conv2d<QB::Storage>,
    position_embedding: Tensor<QB::Storage>,
}

impl<QB: QuantizedBackend> VisionEmbeddings<QB> {
    fn new(cfg: &VisionConfig, vb: VarBuilder<QB>) -> Result<Self> {
        let class_embedding = vb
            .get((1, 1, cfg.hidden_size), "class_embedding")?
            .dequantize(vb.device())?;
        let conv_cfg = Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        let pe_vb = vb.pp("patch_embedding");
        let pe_weight = pe_vb
            .get(
                (cfg.hidden_size, 3, cfg.patch_size, cfg.patch_size),
                "weight",
            )?
            .dequantize(vb.device())?;
        let pe_bias = pe_vb
            .get(cfg.hidden_size, "bias")?
            .dequantize(vb.device())?;

        let patch_embedding = Conv2d::new(pe_weight, Some(pe_bias), conv_cfg);
        let num_patches1 = cfg.image_size / cfg.patch_size;
        let num_patches = num_patches1 * num_patches1;
        let num_positions = num_patches + 1;
        let position_embedding = vb
            .get((1, num_positions, cfg.hidden_size), "position_embedding")?
            .dequantize(vb.device())?;
        Ok(Self {
            class_embedding,
            patch_embedding,
            position_embedding,
        })
    }
}

impl<QB: QuantizedBackend> Module<QB::Storage> for VisionEmbeddings<QB> {
    fn forward(&self, xs: &Tensor<QB::Storage>) -> Result<Tensor<QB::Storage>> {
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
struct Attention<QB: QuantizedBackend> {
    qkv: Linear<QB>,
    projection: Linear<QB>,
    scale: f64,
    num_heads: usize,
}

impl<QB: QuantizedBackend> Attention<QB> {
    fn new(cfg: &VisionConfig, vb: VarBuilder<QB>) -> Result<Self> {
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

    fn forward(
        &self,
        xs: &Tensor<QB::Storage>,
        attn_mask: Option<&Tensor<QB::Storage>>,
    ) -> Result<Tensor<QB::Storage>>
    where
        Linear<QB>: Module<QB::Storage>,
    {
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
struct MLP<QB: QuantizedBackend> {
    activation_fn: candle_nn::Activation,
    fc1: Linear<QB>,
    fc2: Linear<QB>,
}

impl<QB: QuantizedBackend> MLP<QB> {
    fn new(cfg: &VisionConfig, vb: VarBuilder<QB>) -> Result<Self> {
        let fc1 = linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("fc1"))?;
        let fc2 = linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("fc2"))?;
        Ok(Self {
            activation_fn: cfg.hidden_act,
            fc1,
            fc2,
        })
    }
}

impl<QB: QuantizedBackend> Module<QB::Storage> for MLP<QB>
where
    Linear<QB>: Module<QB::Storage>,
{
    fn forward(&self, xs: &Tensor<QB::Storage>) -> Result<Tensor<QB::Storage>> {
        xs.apply(&self.fc1)?
            .apply(&self.activation_fn)?
            .apply(&self.fc2)
    }
}

#[derive(Debug, Clone)]
struct EncoderLayer<QB: QuantizedBackend> {
    self_attn: Attention<QB>,
    layer_norm1: LayerNorm<QB::Storage>,
    mlp: MLP<QB>,
    layer_norm2: LayerNorm<QB::Storage>,
}

impl<QB: QuantizedBackend> EncoderLayer<QB> {
    fn new(cfg: &VisionConfig, vb: VarBuilder<QB>) -> Result<Self> {
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

    fn forward(
        &self,
        xs: &Tensor<QB::Storage>,
        attention_mask: Option<&Tensor<QB::Storage>>,
    ) -> Result<Tensor<QB::Storage>>
    where
        Linear<QB>: Module<QB::Storage>,
    {
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
struct Encoder<QB: QuantizedBackend> {
    layers: Vec<EncoderLayer<QB>>,
}

impl<QB: QuantizedBackend> Encoder<QB> {
    fn new(cfg: &VisionConfig, vb: VarBuilder<QB>) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb = vb.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            let layer = EncoderLayer::new(cfg, vb.pp(i))?;
            layers.push(layer)
        }
        Ok(Self { layers })
    }

    fn forward(
        &self,
        xs: &Tensor<QB::Storage>,
        attention_mask: Option<&Tensor<QB::Storage>>,
    ) -> Result<Tensor<QB::Storage>>
    where
        Linear<QB>: Module<QB::Storage>,
    {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, attention_mask)?
        }
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
pub struct VisionModel<QB: QuantizedBackend> {
    embeddings: VisionEmbeddings<QB>,
    encoder: Encoder<QB>,
    post_layernorm: LayerNorm<QB::Storage>,
}

impl<QB: QuantizedBackend> VisionModel<QB> {
    fn new(cfg: &VisionConfig, vb: VarBuilder<QB>) -> Result<Self> {
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

impl<QB: QuantizedBackend> Module<QB::Storage> for VisionModel<QB>
where
    Linear<QB>: Module<QB::Storage>,
{
    fn forward(&self, xs: &Tensor<QB::Storage>) -> Result<Tensor<QB::Storage>> {
        let xs = xs.apply(&self.embeddings)?;
        let encoder_outputs = self.encoder.forward(&xs, None)?;
        // Return the last hidden state rather than pooled outputs.
        encoder_outputs.apply(&self.post_layernorm)
    }
}

#[derive(Debug, Clone)]
pub struct BlipForConditionalGeneration<QB: QuantizedBackend> {
    vision_model: VisionModel<QB>,
    text_decoder: blip_text::TextLMHeadModel<QB>,
}

impl<QB: QuantizedBackend> BlipForConditionalGeneration<QB> {
    pub fn new(cfg: &Config, vb: VarBuilder<QB>) -> Result<Self> {
        let vision_model = VisionModel::new(&cfg.vision_config, vb.pp("vision_model"))?;
        let text_decoder =
            blip_text::TextLMHeadModel::new(&cfg.text_config, vb.pp("text_decoder"))?;
        Ok(Self {
            vision_model,
            text_decoder,
        })
    }

    pub fn vision_model(&self) -> &VisionModel<QB> {
        &self.vision_model
    }

    pub fn text_decoder(&mut self) -> &mut blip_text::TextLMHeadModel<QB> {
        &mut self.text_decoder
    }
    pub fn reset_kv_cache(&mut self) {
        self.text_decoder.reset_kv_cache();
    }
}
