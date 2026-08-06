//! BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models.
//!
//! BLIP-2 introduces Q-Former, a lightweight Querying Transformer that bridges
//! the modality gap between a frozen image encoder and a frozen LLM. The Q-Former
//! uses a set of learnable query vectors to extract visual features that are most
//! relevant to the text.
//!
//! References:
//! - [Paper](https://arxiv.org/abs/2301.12597)
//! - [HuggingFace Model Card](https://huggingface.co/Salesforce/blip2-opt-2.7b)
//!

use candle::{IndexOp, Module, Result, Tensor, D};
use candle_nn::{layer_norm, linear, Activation, LayerNorm, Linear, VarBuilder};
use serde::Deserialize;

// ----- Config -----

fn default_vision_hidden_size() -> usize {
    1408
}
fn default_vision_intermediate_size() -> usize {
    6144
}
fn default_vision_num_hidden_layers() -> usize {
    39
}
fn default_vision_num_attention_heads() -> usize {
    16
}
fn default_vision_image_size() -> usize {
    224
}
fn default_vision_patch_size() -> usize {
    14
}
fn default_vision_layer_norm_eps() -> f64 {
    1e-6
}
fn default_vision_hidden_act() -> Activation {
    Activation::Gelu
}
fn default_qformer_hidden_size() -> usize {
    768
}
fn default_qformer_num_hidden_layers() -> usize {
    12
}
fn default_qformer_num_attention_heads() -> usize {
    12
}
fn default_qformer_intermediate_size() -> usize {
    3072
}
fn default_qformer_hidden_act() -> Activation {
    Activation::Gelu
}
fn default_qformer_layer_norm_eps() -> f64 {
    1e-12
}
fn default_qformer_max_position_embeddings() -> usize {
    512
}
fn default_qformer_cross_attention_frequency() -> usize {
    2
}
fn default_qformer_encoder_hidden_size() -> usize {
    1408
}
fn default_num_query_tokens() -> usize {
    32
}

#[derive(Debug, Clone, Deserialize)]
pub struct Blip2VisionConfig {
    #[serde(default = "default_vision_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_vision_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_vision_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_vision_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_vision_image_size")]
    pub image_size: usize,
    #[serde(default = "default_vision_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_vision_hidden_act")]
    pub hidden_act: Activation,
    #[serde(default = "default_vision_layer_norm_eps")]
    pub layer_norm_eps: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Blip2QFormerConfig {
    #[serde(default = "default_qformer_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_qformer_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_qformer_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_qformer_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_qformer_hidden_act")]
    pub hidden_act: Activation,
    #[serde(default = "default_qformer_layer_norm_eps")]
    pub layer_norm_eps: f64,
    #[serde(default = "default_qformer_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_qformer_cross_attention_frequency")]
    pub cross_attention_frequency: usize,
    #[serde(default = "default_qformer_encoder_hidden_size")]
    pub encoder_hidden_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Blip2Config {
    pub vision_config: Blip2VisionConfig,
    pub qformer_config: Blip2QFormerConfig,
    #[serde(default = "default_num_query_tokens")]
    pub num_query_tokens: usize,
}

// ----- Vision Encoder -----

struct VisionEmbeddings {
    class_embedding: Tensor,
    patch_embedding: candle_nn::Conv2d,
    position_embedding: Tensor,
}

impl VisionEmbeddings {
    fn new(cfg: &Blip2VisionConfig, vb: VarBuilder) -> Result<Self> {
        let class_embedding = vb.get((1, 1, cfg.hidden_size), "class_embedding")?;
        let conv_cfg = candle_nn::Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        let patch_embedding =
            candle_nn::conv2d(3, cfg.hidden_size, cfg.patch_size, conv_cfg, vb.pp("patch_embedding"))?;
        let num_patches = (cfg.image_size / cfg.patch_size).pow(2);
        let position_embedding =
            vb.get((1, num_patches + 1, cfg.hidden_size), "position_embedding")?;
        Ok(Self {
            class_embedding,
            patch_embedding,
            position_embedding,
        })
    }
}

impl Module for VisionEmbeddings {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let b_size = xs.dim(0)?;
        let target_dtype = xs.dtype();
        let patch_embeds = xs
            .apply(&self.patch_embedding)?
            .flatten_from(2)?
            .transpose(1, 2)?;
        let cls = self
            .class_embedding
            .broadcast_as((b_size, 1, self.class_embedding.dim(D::Minus1)?))?
            .to_dtype(target_dtype)?;
        let embeddings = Tensor::cat(&[&cls, &patch_embeds], 1)?;
        let pos = self.position_embedding.narrow(1, 0, embeddings.dim(1)?)?;
        embeddings.broadcast_add(&pos)
    }
}

struct VisionAttention {
    qkv: Linear,
    projection: Linear,
    num_heads: usize,
    scale: f64,
}

impl VisionAttention {
    fn new(cfg: &Blip2VisionConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.hidden_size;
        let qkv = linear(d, d * 3, vb.pp("qkv"))?;
        let projection = linear(d, d, vb.pp("projection"))?;
        let num_heads = cfg.num_attention_heads;
        let scale = 1.0 / ((d / num_heads) as f64).sqrt();
        Ok(Self {
            qkv,
            projection,
            num_heads,
            scale,
        })
    }
}

impl Module for VisionAttention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, n, c) = xs.dims3()?;
        let head_dim = c / self.num_heads;
        let qkv = self
            .qkv
            .forward(xs)?
            .reshape((b, n, 3, self.num_heads, head_dim))?
            .permute((2, 0, 3, 1, 4))?;
        let q = (qkv.i(0)? * self.scale)?;
        let k = qkv.i(1)?;
        let v = qkv.i(2)?;
        let attn = candle_nn::ops::softmax(&q.matmul(&k.t()?)?, D::Minus1)?;
        let out = attn.matmul(&v)?.transpose(1, 2)?.reshape((b, n, c))?;
        self.projection.forward(&out)
    }
}

struct VisionMlp {
    fc1: Linear,
    fc2: Linear,
    act: Activation,
}

impl VisionMlp {
    fn new(cfg: &Blip2VisionConfig, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("fc1"))?;
        let fc2 = linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("fc2"))?;
        Ok(Self {
            fc1,
            fc2,
            act: cfg.hidden_act,
        })
    }
}

impl Module for VisionMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.fc1)?.apply(&self.act)?.apply(&self.fc2)
    }
}

struct VisionEncoderLayer {
    self_attn: VisionAttention,
    layer_norm1: LayerNorm,
    mlp: VisionMlp,
    layer_norm2: LayerNorm,
}

impl VisionEncoderLayer {
    fn new(cfg: &Blip2VisionConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.hidden_size;
        Ok(Self {
            self_attn: VisionAttention::new(cfg, vb.pp("self_attn"))?,
            layer_norm1: layer_norm(d, cfg.layer_norm_eps, vb.pp("layer_norm1"))?,
            mlp: VisionMlp::new(cfg, vb.pp("mlp"))?,
            layer_norm2: layer_norm(d, cfg.layer_norm_eps, vb.pp("layer_norm2"))?,
        })
    }
}

impl Module for VisionEncoderLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = xs.apply(&self.layer_norm1)?.apply(&self.self_attn)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.layer_norm2)?.apply(&self.mlp)?;
        xs + residual
    }
}

/// BLIP-2 Vision Encoder using a ViT backbone.
pub struct Blip2VisionModel {
    embeddings: VisionEmbeddings,
    layers: Vec<VisionEncoderLayer>,
    post_layernorm: LayerNorm,
}

impl Blip2VisionModel {
    pub fn new(cfg: &Blip2VisionConfig, vb: VarBuilder) -> Result<Self> {
        let embeddings = VisionEmbeddings::new(cfg, vb.pp("embeddings"))?;
        let vb_enc = vb.pp("encoder").pp("layers");
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| VisionEncoderLayer::new(cfg, vb_enc.pp(i)))
            .collect::<Result<Vec<_>>>()?;
        let post_layernorm =
            layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("post_layernorm"))?;
        Ok(Self {
            embeddings,
            layers,
            post_layernorm,
        })
    }
}

impl Module for Blip2VisionModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut hidden = xs.apply(&self.embeddings)?;
        for layer in &self.layers {
            hidden = hidden.apply(layer)?;
        }
        hidden.apply(&self.post_layernorm)
    }
}

// ----- Q-Former -----

struct QFormerAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    dense: Linear,
    layer_norm: LayerNorm,
    num_heads: usize,
    head_dim: usize,
}

impl QFormerAttention {
    fn new(hidden_size: usize, num_heads: usize, kv_hidden_size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        Ok(Self {
            query: linear(hidden_size, hidden_size, vb.pp("attention").pp("query"))?,
            key: linear(kv_hidden_size, hidden_size, vb.pp("attention").pp("key"))?,
            value: linear(kv_hidden_size, hidden_size, vb.pp("attention").pp("value"))?,
            dense: linear(hidden_size, hidden_size, vb.pp("output").pp("dense"))?,
            layer_norm: layer_norm(hidden_size, eps, vb.pp("output").pp("LayerNorm"))?,
            num_heads,
            head_dim,
        })
    }

    fn forward(&self, hidden_states: &Tensor, encoder_hidden_states: Option<&Tensor>) -> Result<Tensor> {
        let (b, seq_len, _) = hidden_states.dims3()?;
        let q = self.query.forward(hidden_states)?
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let kv_input = encoder_hidden_states.unwrap_or(hidden_states);
        let kv_len = kv_input.dim(1)?;
        let k = self.key.forward(kv_input)?
            .reshape((b, kv_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self.value.forward(kv_input)?
            .reshape((b, kv_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn = candle_nn::ops::softmax(&(q.matmul(&k.t()?)? * scale)?, D::Minus1)?;
        let out = attn
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, seq_len, self.num_heads * self.head_dim))?;

        let out = self.dense.forward(&out)?;
        // Residual + LayerNorm
        self.layer_norm.forward(&(out + hidden_states)?)
    }
}

struct QFormerLayer {
    self_attention: QFormerAttention,
    cross_attention: Option<QFormerAttention>,
    intermediate: Linear,
    output_dense: Linear,
    output_layernorm: LayerNorm,
    act: Activation,
}

impl QFormerLayer {
    fn new(
        cfg: &Blip2QFormerConfig,
        has_cross_attention: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let h = cfg.hidden_size;
        let self_attention =
            QFormerAttention::new(h, cfg.num_attention_heads, h, cfg.layer_norm_eps, vb.pp("attention"))?;
        let cross_attention = if has_cross_attention {
            Some(QFormerAttention::new(
                h,
                cfg.num_attention_heads,
                cfg.encoder_hidden_size,
                cfg.layer_norm_eps,
                vb.pp("crossattention"),
            )?)
        } else {
            None
        };
        let intermediate = linear(h, cfg.intermediate_size, vb.pp("intermediate").pp("dense"))?;
        let output_dense = linear(cfg.intermediate_size, h, vb.pp("output").pp("dense"))?;
        let output_layernorm =
            layer_norm(h, cfg.layer_norm_eps, vb.pp("output").pp("LayerNorm"))?;
        Ok(Self {
            self_attention,
            cross_attention,
            intermediate,
            output_dense,
            output_layernorm,
            act: cfg.hidden_act,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<Tensor> {
        let attention_output = self.self_attention.forward(hidden_states, None)?;
        let attention_output = if let Some(cross_attn) = &self.cross_attention {
            cross_attn.forward(&attention_output, encoder_hidden_states)?
        } else {
            attention_output
        };
        // FFN
        let intermediate = self
            .intermediate
            .forward(&attention_output)?
            .apply(&self.act)?;
        let output = self.output_dense.forward(&intermediate)?;
        self.output_layernorm.forward(&(output + &attention_output)?)
    }
}

/// Q-Former: Querying Transformer that bridges vision and language.
///
/// Uses learnable query vectors and cross-attention to extract visual features
/// that are most relevant for language generation.
pub struct Blip2QFormer {
    embeddings_layernorm: LayerNorm,
    embeddings_position: candle_nn::Embedding,
    layers: Vec<QFormerLayer>,
}

impl Blip2QFormer {
    pub fn new(cfg: &Blip2QFormerConfig, vb: VarBuilder) -> Result<Self> {
        let embeddings_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.layer_norm_eps,
            vb.pp("embeddings").pp("LayerNorm"),
        )?;
        let embeddings_position = candle_nn::embedding(
            cfg.max_position_embeddings,
            cfg.hidden_size,
            vb.pp("embeddings").pp("position_embeddings"),
        )?;
        let vb_layers = vb.pp("encoder").pp("layer");
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                let has_cross = i % cfg.cross_attention_frequency == 0;
                QFormerLayer::new(cfg, has_cross, vb_layers.pp(i))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embeddings_layernorm,
            embeddings_position,
            layers,
        })
    }

    /// Run the Q-Former with the given query embeddings and vision encoder output.
    ///
    /// `query_embeds`: (batch, num_queries, hidden_size)
    /// `encoder_hidden_states`: (batch, vision_seq_len, vision_hidden_size)
    pub fn forward(
        &self,
        query_embeds: &Tensor,
        encoder_hidden_states: &Tensor,
    ) -> Result<Tensor> {
        let seq_len = query_embeds.dim(1)?;
        let device = query_embeds.device();
        let position_ids = Tensor::arange(0u32, seq_len as u32, device)?;
        let position_embeddings = self.embeddings_position.forward(&position_ids)?;
        let mut hidden_states = query_embeds
            .broadcast_add(&position_embeddings)?
            .apply(&self.embeddings_layernorm)?;

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, Some(encoder_hidden_states))?;
        }
        Ok(hidden_states)
    }
}

// ----- Full BLIP-2 Model -----

/// BLIP-2 model combining a frozen vision encoder with Q-Former for
/// visual feature extraction and optional language model projection.
pub struct Blip2Model {
    vision_model: Blip2VisionModel,
    query_tokens: Tensor,
    qformer: Blip2QFormer,
    language_projection: Linear,
}

impl Blip2Model {
    pub fn new(cfg: &Blip2Config, vb: VarBuilder) -> Result<Self> {
        let vision_model = Blip2VisionModel::new(&cfg.vision_config, vb.pp("vision_model"))?;
        let query_tokens = vb.get(
            (1, cfg.num_query_tokens, cfg.qformer_config.hidden_size),
            "query_tokens",
        )?;
        let qformer = Blip2QFormer::new(&cfg.qformer_config, vb.pp("qformer"))?;
        let language_projection = linear(
            cfg.qformer_config.hidden_size,
            cfg.vision_config.hidden_size,
            vb.pp("language_projection"),
        )?;
        Ok(Self {
            vision_model,
            query_tokens,
            qformer,
            language_projection,
        })
    }

    /// Extract Q-Former visual features from pixel values.
    ///
    /// `pixel_values`: (batch, 3, image_size, image_size)
    /// Returns: (batch, num_query_tokens, hidden_size)
    pub fn get_qformer_features(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let vision_outputs = self.vision_model.forward(pixel_values)?;
        let b = pixel_values.dim(0)?;
        let query_tokens = self.query_tokens.broadcast_as((
            b,
            self.query_tokens.dim(1)?,
            self.query_tokens.dim(2)?,
        ))?;
        self.qformer.forward(&query_tokens, &vision_outputs)
    }

    /// Get language model inputs by projecting Q-Former features.
    ///
    /// `pixel_values`: (batch, 3, image_size, image_size)
    /// Returns: (batch, num_query_tokens, language_model_hidden_size)
    pub fn get_language_model_inputs(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let qformer_output = self.get_qformer_features(pixel_values)?;
        self.language_projection.forward(&qformer_output)
    }

    pub fn vision_model(&self) -> &Blip2VisionModel {
        &self.vision_model
    }
}
