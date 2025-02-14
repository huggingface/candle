//! Siglip model implementation.
//!
//! Siglip architecture combining vision and language for zero-shot tasks.
//!
//! References:
//! - 🤗 [Model Card](https://huggingface.co/google/siglip-base-patch16-224)
//!

use crate::models::clip::div_l2_norm;
use candle::{IndexOp, Module, Result, Tensor, D};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, VarBuilder};

// https://github.com/huggingface/transformers/blob/2e24ee4dfa39cc0bc264b89edbccc373c8337086/src/transformers/models/siglip/configuration_siglip.py#L27
#[derive(serde::Deserialize, Clone, Debug)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    pub hidden_act: candle_nn::Activation,
    pub layer_norm_eps: f64,
    pub pad_token_id: u32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
}

// https://github.com/huggingface/transformers/blob/2e24ee4dfa39cc0bc264b89edbccc373c8337086/src/transformers/models/siglip/configuration_siglip.py#L132
#[derive(serde::Deserialize, Clone, Debug)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub hidden_act: candle_nn::Activation,
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

impl VisionConfig {
    pub fn paligemma_3b_224() -> Self {
        Self {
            // https://huggingface.co/google/paligemma-3b-pt-224/blob/main/config.json
            patch_size: 14,
            num_attention_heads: 16,
            num_hidden_layers: 27,
            hidden_size: 1152,
            intermediate_size: 4304,
            image_size: 224, // num_image_tokens: (224 / 14)^2 = 256
            // Default values.
            num_channels: 3,
            hidden_act: candle_nn::Activation::GeluPytorchTanh,
            layer_norm_eps: 1e-6,
        }
    }

    pub fn paligemma_3b_448() -> Self {
        Self {
            // https://huggingface.co/google/paligemma-3b-pt-448/blob/main/config.json
            patch_size: 14,
            num_attention_heads: 16,
            num_hidden_layers: 27,
            hidden_size: 1152,
            intermediate_size: 4304,
            image_size: 448, // num_image_tokens: (448 / 14)^2 = 1024
            // Default values.
            num_channels: 3,
            hidden_act: candle_nn::Activation::GeluPytorchTanh,
            layer_norm_eps: 1e-6,
        }
    }

    pub fn paligemma_3b_896() -> Self {
        Self {
            // https://huggingface.co/google/paligemma-3b-pt-448/blob/main/config.json
            patch_size: 14,
            num_attention_heads: 16,
            num_hidden_layers: 27,
            hidden_size: 1152,
            intermediate_size: 4304,
            image_size: 896, // num_image_tokens: (896 / 14)^2 = 4096
            // Default values.
            num_channels: 3,
            hidden_act: candle_nn::Activation::GeluPytorchTanh,
            layer_norm_eps: 1e-6,
        }
    }

    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size).pow(2)
    }
}

// https://github.com/huggingface/transformers/blob/2e24ee4dfa39cc0bc264b89edbccc373c8337086/src/transformers/models/siglip/configuration_siglip.py#L228
#[derive(serde::Deserialize, Clone, Debug)]
pub struct Config {
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
}

impl Config {
    pub fn base_patch16_224() -> Self {
        let text_config = TextConfig {
            // https://huggingface.co/google/siglip-base-patch16-224/blob/main/config.json
            hidden_size: 768,
            intermediate_size: 3072,
            num_attention_heads: 12,
            vocab_size: 32000,
            // Default values.
            pad_token_id: 1,
            bos_token_id: 49406,
            eos_token_id: 49407,
            layer_norm_eps: 1e-6,
            hidden_act: candle_nn::Activation::GeluPytorchTanh,
            max_position_embeddings: 64,
            num_hidden_layers: 12,
        };
        let vision_config = VisionConfig {
            patch_size: 16,
            // Default values.
            hidden_size: 768,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            num_channels: 3,
            image_size: 224,
            hidden_act: candle_nn::Activation::GeluPytorchTanh,
            layer_norm_eps: 1e-6,
        };
        Self {
            text_config,
            vision_config,
        }
    }
}

#[derive(Clone, Debug)]
struct MultiheadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
}

impl MultiheadAttention {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let w_in_proj = vb.get((3 * h, h), "in_proj_weight")?.chunk(3, 0)?;
        let b_in_proj = vb.get(3 * h, "in_proj_bias")?.chunk(3, 0)?;
        let q_proj = Linear::new(w_in_proj[0].clone(), Some(b_in_proj[0].clone()));
        let k_proj = Linear::new(w_in_proj[1].clone(), Some(b_in_proj[1].clone()));
        let v_proj = Linear::new(w_in_proj[2].clone(), Some(b_in_proj[2].clone()));
        let out_proj = linear(h, h, vb.pp("out_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
        })
    }

    fn separate_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n, c) = x.dims3()?;
        x.reshape((b, n, self.num_heads, c / self.num_heads))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn recombine_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (b, n_heads, n_tokens, c_per_head) = x.dims4()?;
        x.transpose(1, 2)?
            .reshape((b, n_tokens, n_heads * c_per_head))
    }

    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let q = self.q_proj.forward(&q.contiguous()?)?;
        let k = self.k_proj.forward(&k.contiguous()?)?;
        let v = self.v_proj.forward(&v.contiguous()?)?;

        let q = self.separate_heads(&q)?;
        let k = self.separate_heads(&k)?;
        let v = self.separate_heads(&v)?;

        let (_, _, _, c_per_head) = q.dims4()?;
        let attn = (q.matmul(&k.t()?)? / (c_per_head as f64).sqrt())?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;

        let out = attn.matmul(&v)?;
        self.recombine_heads(&out)?.apply(&self.out_proj)
    }
}

#[derive(Debug, Clone)]
struct MultiheadAttentionPoolingHead {
    probe: Tensor,
    attention: MultiheadAttention,
    layernorm: LayerNorm,
    mlp: Mlp,
}

impl MultiheadAttentionPoolingHead {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let layernorm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layernorm"))?;
        let probe = vb.get((1, 1, cfg.hidden_size), "probe")?;
        let attention = MultiheadAttention::new(cfg, vb.pp("attention"))?;
        Ok(Self {
            probe,
            attention,
            layernorm,
            mlp,
        })
    }
}

impl Module for MultiheadAttentionPoolingHead {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let batch_size = xs.dim(0)?;
        let probe = self.probe.repeat((batch_size, 1, 1))?;
        let xs = self.attention.forward(&probe, xs, xs)?;
        let residual = &xs;
        let xs = xs.apply(&self.layernorm)?.apply(&self.mlp)?;
        (xs + residual)?.i((.., 0))
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
        let embed_dim = cfg.hidden_size();
        let q_proj = linear(embed_dim, embed_dim, vb.pp("q_proj"))?;
        let k_proj = linear(embed_dim, embed_dim, vb.pp("k_proj"))?;
        let v_proj = linear(embed_dim, embed_dim, vb.pp("v_proj"))?;
        let out_proj = linear(embed_dim, embed_dim, vb.pp("out_proj"))?;
        let num_heads = cfg.num_attention_heads();
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

// https://github.com/huggingface/transformers/blob/2e24ee4dfa39cc0bc264b89edbccc373c8337086/src/transformers/models/siglip/modeling_siglip.py#L599
#[derive(Debug, Clone)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
    activation_fn: candle_nn::Activation,
}

impl Mlp {
    fn new<C: TransformerConfig>(cfg: &C, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size();
        let intermediate_size = cfg.intermediate_size();
        let fc1 = candle_nn::linear(hidden_size, intermediate_size, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(intermediate_size, hidden_size, vb.pp("fc2"))?;
        Ok(Self {
            fc1,
            fc2,
            activation_fn: cfg.hidden_act(),
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &candle::Tensor) -> Result<candle::Tensor> {
        xs.apply(&self.fc1)?
            .apply(&self.activation_fn)?
            .apply(&self.fc2)
    }
}

// https://github.com/huggingface/transformers/blob/2e24ee4dfa39cc0bc264b89edbccc373c8337086/src/transformers/models/siglip/modeling_siglip.py#L614
#[derive(Debug, Clone)]
struct EncoderLayer {
    self_attn: Attention,
    layer_norm1: LayerNorm,
    mlp: Mlp,
    layer_norm2: LayerNorm,
}

impl EncoderLayer {
    fn new<C: TransformerConfig>(cfg: &C, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size();
        let layer_norm_eps = cfg.layer_norm_eps();
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
    fn new<C: TransformerConfig>(cfg: &C, vb: VarBuilder) -> Result<Self> {
        let mut layers = vec![];
        let vb = vb.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers() {
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
    patch_embedding: candle_nn::Conv2d,
    position_embedding: Tensor,
    patch_size: usize,
    base_num_patches_per_side: usize,
}

impl VisionEmbeddings {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let conv2d_cfg = candle_nn::Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        let patch_embedding = candle_nn::conv2d(
            cfg.num_channels,
            cfg.hidden_size,
            cfg.patch_size,
            conv2d_cfg,
            vb.pp("patch_embedding"),
        )?;
        let num_patches_per_side = cfg.image_size / cfg.patch_size;
        let embedder = candle_nn::embedding(
            num_patches_per_side.pow(2),
            cfg.hidden_size(),
            vb.pp("position_embedding"),
        )?;
        let position_embedding = embedder.embeddings();
        let position_embedding = position_embedding
            .reshape((
                1,
                num_patches_per_side,
                num_patches_per_side,
                cfg.hidden_size(),
            ))?
            .permute((0, 3, 1, 2))?;
        Ok(Self {
            patch_embedding,
            position_embedding,
            patch_size: cfg.patch_size,
            base_num_patches_per_side: num_patches_per_side,
        })
    }
}

impl Module for VisionEmbeddings {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        //embed tokens
        let (_batch, _channels, _height, _width) = xs.dims4()?;
        let embeddings = xs.apply(&self.patch_embedding)?;
        // interpolate position embeddings for the current image size (if needed)
        let num_patches_h = _height / self.patch_size;
        let num_patches_w = _width / self.patch_size;
        let resized_position_embedding = if num_patches_w == self.base_num_patches_per_side
            && num_patches_h == self.base_num_patches_per_side
        {
            self.position_embedding.clone()
        } else {
            self.position_embedding
                .interpolate2d(num_patches_h, num_patches_w)?
        };
        // Add position embeddings to tokens and flatten from 2D patches to 1D sequence
        let embeddings = embeddings
            .broadcast_add(&resized_position_embedding)?
            .flatten_from(2)?
            .transpose(1, 2)?;
        Ok(embeddings)
    }
}

#[derive(Debug, Clone)]
struct VisionTransformer {
    embeddings: VisionEmbeddings,
    encoder: Encoder,
    post_layernorm: LayerNorm,
    head: Option<MultiheadAttentionPoolingHead>,
}

impl VisionTransformer {
    fn new(cfg: &VisionConfig, use_head: bool, vb: VarBuilder) -> Result<Self> {
        let embeddings = VisionEmbeddings::new(cfg, vb.pp("embeddings"))?;
        let encoder = Encoder::new(cfg, vb.pp("encoder"))?;
        let post_layernorm =
            layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("post_layernorm"))?;
        let head = if use_head {
            Some(MultiheadAttentionPoolingHead::new(cfg, vb.pp("head"))?)
        } else {
            None
        };
        Ok(Self {
            embeddings,
            encoder,
            post_layernorm,
            head,
        })
    }
}

impl Module for VisionTransformer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.apply(&self.embeddings)?;
        let xs = self.encoder.forward(&xs, None)?;
        let xs = xs.apply(&self.post_layernorm)?;
        match self.head.as_ref() {
            None => Ok(xs),
            Some(h) => xs.apply(h),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VisionModel {
    vision_model: VisionTransformer,
}

impl VisionModel {
    pub fn new(cfg: &VisionConfig, use_head: bool, vb: VarBuilder) -> Result<Self> {
        let vision_model = VisionTransformer::new(cfg, use_head, vb)?;
        Ok(Self { vision_model })
    }
}

impl Module for VisionModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.vision_model)
    }
}

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
}

impl Module for TextEmbeddings {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_length = input_ids.dim(D::Minus1)?;
        let inputs_embeds = self.token_embedding.forward(input_ids)?;
        let position_ids = self.position_ids.narrow(1, 0, seq_length)?;
        let position_embedding = self.position_embedding.forward(&position_ids)?;
        inputs_embeds.broadcast_add(&position_embedding)
    }
}

#[derive(Debug, Clone)]
pub struct TextTransformer {
    embeddings: TextEmbeddings,
    encoder: Encoder,
    final_layer_norm: LayerNorm,
    pub head: Linear,
}

impl TextTransformer {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let embeddings = TextEmbeddings::new(cfg, vb.pp("embeddings"))?;
        let encoder = Encoder::new(cfg, vb.pp("encoder"))?;
        let final_layer_norm = layer_norm(
            cfg.hidden_size,
            cfg.layer_norm_eps,
            vb.pp("final_layer_norm"),
        )?;
        let head = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("head"))?;
        Ok(Self {
            embeddings,
            encoder,
            final_layer_norm,
            head,
        })
    }
}
impl Module for TextTransformer {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_bsz, seq_len) = input_ids.dims2()?;
        let input_ids = self.embeddings.forward(input_ids)?;
        let input_ids = self.encoder.forward(&input_ids, None)?;
        let last_hidden_state = self.final_layer_norm.forward(&input_ids)?;
        last_hidden_state
            .i((.., seq_len - 1, ..))?
            .contiguous()?
            .apply(&self.head)
    }
}

#[derive(Debug, Clone)]
pub struct TextModel {
    pub text_model: TextTransformer,
}

impl TextModel {
    pub fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let text_model = TextTransformer::new(cfg, vb)?;
        Ok(Self { text_model })
    }
}

impl Module for TextModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.text_model)
    }
}

#[derive(Clone, Debug)]
pub struct Model {
    text_model: TextModel,
    vision_model: VisionModel,
    logit_bias: Tensor,
    logit_scale: Tensor,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let text_model = TextModel::new(&cfg.text_config, vb.pp("text_model"))?;
        let vision_model = VisionModel::new(&cfg.vision_config, true, vb.pp("vision_model"))?;
        let logit_scale = vb.get(&[1], "logit_scale")?;
        let logit_bias = vb.get(&[1], "logit_bias")?;
        Ok(Self {
            text_model,
            vision_model,
            logit_bias,
            logit_scale,
        })
    }

    pub fn get_text_features(&self, input_ids: &Tensor) -> Result<Tensor> {
        input_ids.apply(&self.text_model)
    }

    pub fn get_image_features(&self, pixel_values: &Tensor) -> Result<Tensor> {
        pixel_values.apply(&self.vision_model)
    }

    pub fn forward(&self, pixel_values: &Tensor, input_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let image_features = self.get_image_features(pixel_values)?;
        let text_features = self.get_text_features(input_ids)?;
        let image_features_normalized = div_l2_norm(&image_features)?;
        let text_features_normalized = div_l2_norm(&text_features)?;
        let logits_per_text = text_features_normalized.matmul(&image_features_normalized.t()?)?;
        let logit_scale = self.logit_scale.exp()?;
        let logits_per_text = logits_per_text
            .broadcast_mul(&logit_scale)?
            .broadcast_add(&self.logit_bias)?;
        let logits_per_image = logits_per_text.t()?;
        Ok((logits_per_text, logits_per_image))
    }
}
