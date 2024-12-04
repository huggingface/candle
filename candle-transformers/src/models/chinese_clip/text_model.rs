//! Chinese contrastive Language-Image Pre-Training
//!
//! Chinese contrastive Language-Image Pre-Training (CLIP) is an architecture trained on
//! pairs of images with related texts.
//!
//! - 💻 [Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)
//! - 💻 [HF](https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/chinese_clip/modeling_chinese_clip.py)

use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn as nn;

use super::Activation;

/// Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
/// positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
/// [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
/// For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
/// with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
#[derive(Clone, Debug)]
pub enum PositionEmbeddingType {
    Absolute,
    RelativeKey,
    RelativeKeyQuery,
}

#[derive(Clone, Debug)]
pub struct ChineseClipTextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: Activation,
    pub hidden_dropout_prob: f32,
    pub attention_probs_dropout_prob: f64,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub initializer_range: f64,
    pub initializer_factor: f64,
    pub layer_norm_eps: f64,
    pub pad_token_id: usize,
    pub position_embedding_type: PositionEmbeddingType,
    pub use_cache: bool,
}

impl Default for ChineseClipTextConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: Activation::Gelu,
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            initializer_factor: 1.0,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: PositionEmbeddingType::Absolute,
            use_cache: true,
        }
    }
}

impl ChineseClipTextConfig {
    /// [referer](https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16/blob/main/config.json)
    pub fn clip_vit_base_patch16() -> Self {
        Self {
            vocab_size: 21128,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: Activation::Gelu,
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            initializer_factor: 1.0,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: PositionEmbeddingType::Absolute,
            use_cache: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ChineseClipTextEmbeddings {
    word_embeddings: nn::Embedding,
    position_embeddings: nn::Embedding,
    token_type_embeddings: nn::Embedding,
    layer_norm: nn::LayerNorm,
    dropout: nn::Dropout,
    position_embedding_type: PositionEmbeddingType,
    position_ids: Tensor,
    token_type_ids: Tensor,
}

impl ChineseClipTextEmbeddings {
    pub fn new(var: nn::VarBuilder, config: &ChineseClipTextConfig) -> Result<Self> {
        let word_embeddings = nn::embedding(
            config.vocab_size,
            config.hidden_size,
            var.pp("word_embeddings"),
        )?;
        let position_embeddings = nn::embedding(
            config.max_position_embeddings,
            config.hidden_size,
            var.pp("position_embeddings"),
        )?;
        let token_type_embeddings = nn::embedding(
            config.type_vocab_size,
            config.hidden_size,
            var.pp("token_type_embeddings"),
        )?;
        let layer_norm = nn::layer_norm::<f64>(
            config.hidden_size,
            config.layer_norm_eps,
            var.pp("LayerNorm"),
        )?;
        let dropout = nn::Dropout::new(config.hidden_dropout_prob);
        let position_ids =
            Tensor::arange(0u32, config.max_position_embeddings as u32, var.device())?
                .unsqueeze(0)?;
        let token_type_ids = Tensor::zeros(position_ids.shape(), DType::I64, var.device())?;

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout,
            position_embedding_type: config.position_embedding_type.clone(),
            position_ids,
            token_type_ids,
        })
    }

    fn forward(&self, xs: &Tensor, token_type_ids: Option<&Tensor>) -> Result<Tensor> {
        let (_batch_size, seq_length) = xs.dims2()?;
        let position_ids = (0..seq_length as u32).collect::<Vec<_>>();
        let position_ids = self.position_ids.index_select(
            &Tensor::new(&position_ids[..], self.position_ids.device())?,
            1,
        )?;

        let word_embeddings = self.word_embeddings.forward(xs)?;

        let token_type_ids = match token_type_ids {
            Some(token_type_ids) => token_type_ids,
            None => &self.token_type_ids.i((.., 0..seq_length))?,
        };
        let token_type_ids = token_type_ids.expand(xs.shape())?;
        let token_type_embeddings = self.token_type_embeddings.forward(&token_type_ids)?;

        let embeddings = (&word_embeddings + token_type_embeddings)?;
        let embeddings = match self.position_embedding_type {
            PositionEmbeddingType::Absolute => {
                let position_embeddings = self.position_embeddings.forward(&position_ids)?;
                let position_embeddings = position_embeddings.expand(embeddings.shape())?;
                (embeddings + position_embeddings)?
            }
            _ => embeddings,
        };
        let embeddings = self.layer_norm.forward(&embeddings)?;
        let embeddings = self.dropout.forward(&embeddings, false)?;
        Ok(embeddings)
    }
}

/// Copied from [`crate::models::bert::BertSelfOutput`] to [`ChineseClipTextSelfOutput`]
#[derive(Clone, Debug)]
struct ChineseClipTextSelfOutput {
    dense: nn::Linear,
    layer_norm: nn::LayerNorm,
    dropout: nn::Dropout,
    span: tracing::Span,
}

impl ChineseClipTextSelfOutput {
    fn new(var: nn::VarBuilder, config: &ChineseClipTextConfig) -> Result<Self> {
        let dense = nn::linear(config.hidden_size, config.hidden_size, var.pp("dense"))?;
        let layer_norm = nn::layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            var.pp("LayerNorm"),
        )?;
        let dropout = nn::Dropout::new(config.hidden_dropout_prob);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
            span: tracing::span!(tracing::Level::TRACE, "self-out"),
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states, false)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

/// Copied from [`crate::models::bert::BertSelfAttention`] to [`ChineseClipTextSelfAttention`]
#[derive(Clone, Debug)]
struct ChineseClipTextSelfAttention {
    query: nn::Linear,
    key: nn::Linear,
    value: nn::Linear,
    dropout: nn::Dropout,
    num_attention_heads: usize,
    attention_head_size: usize,
    span: tracing::Span,
    span_softmax: tracing::Span,
}

impl ChineseClipTextSelfAttention {
    fn new(var: nn::VarBuilder, config: &ChineseClipTextConfig) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        let dropout = nn::Dropout::new(config.hidden_dropout_prob);
        let hidden_size = config.hidden_size;
        let query = nn::linear(hidden_size, all_head_size, var.pp("query"))?;
        let value = nn::linear(hidden_size, all_head_size, var.pp("value"))?;
        let key = nn::linear(hidden_size, all_head_size, var.pp("key"))?;
        Ok(Self {
            query,
            key,
            value,
            dropout,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            span: tracing::span!(tracing::Level::TRACE, "self-attn"),
            span_softmax: tracing::span!(tracing::Level::TRACE, "softmax"),
        })
    }

    fn transpose_for_scores(&self, xs: &Tensor) -> Result<Tensor> {
        let mut new_x_shape = xs.dims().to_vec();
        new_x_shape.pop();
        new_x_shape.push(self.num_attention_heads);
        new_x_shape.push(self.attention_head_size);
        let xs = xs.reshape(new_x_shape.as_slice())?.transpose(1, 2)?;
        xs.contiguous()
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let query_layer = self.query.forward(hidden_states)?;
        let key_layer = self.key.forward(hidden_states)?;
        let value_layer = self.value.forward(hidden_states)?;

        let query_layer = self.transpose_for_scores(&query_layer)?;
        let key_layer = self.transpose_for_scores(&key_layer)?;
        let value_layer = self.transpose_for_scores(&value_layer)?;

        let attention_scores = query_layer.matmul(&key_layer.t()?)?;
        let attention_scores = (attention_scores / (self.attention_head_size as f64).sqrt())?;
        let attention_scores = attention_scores.broadcast_add(attention_mask)?;
        let attention_probs = {
            let _enter_sm = self.span_softmax.enter();
            nn::ops::softmax(&attention_scores, candle::D::Minus1)?
        };
        let attention_probs = self.dropout.forward(&attention_probs, false)?;

        let context_layer = attention_probs.matmul(&value_layer)?;
        let context_layer = context_layer.transpose(1, 2)?.contiguous()?;
        let context_layer = context_layer.flatten_from(candle::D::Minus2)?;
        Ok(context_layer)
    }
}

/// Copied from [`crate::models::bert::BertAttention`] to [`ChineseClipTextAttention`]
#[derive(Clone, Debug)]
struct ChineseClipTextAttention {
    self_attention: ChineseClipTextSelfAttention,
    self_output: ChineseClipTextSelfOutput,
    span: tracing::Span,
}

impl ChineseClipTextAttention {
    fn new(var: nn::VarBuilder, config: &ChineseClipTextConfig) -> Result<Self> {
        let self_attention = ChineseClipTextSelfAttention::new(var.pp("self"), config)?;
        let self_output = ChineseClipTextSelfOutput::new(var.pp("output"), config)?;
        Ok(Self {
            self_attention,
            self_output,
            span: tracing::span!(tracing::Level::TRACE, "attn"),
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let self_outputs = self.self_attention.forward(hidden_states, attention_mask)?;
        let attention_output = self.self_output.forward(&self_outputs, hidden_states)?;
        Ok(attention_output)
    }
}

type HiddenActLayer = Activation;

/// Copied from [`crate::models::bert::BertIntermediate`] to [`ChineseClipTextIntermediate`]
#[derive(Clone, Debug)]
struct ChineseClipTextIntermediate {
    dense: nn::Linear,
    intermediate_act: HiddenActLayer,
    span: tracing::Span,
}

impl ChineseClipTextIntermediate {
    fn new(var: nn::VarBuilder, config: &ChineseClipTextConfig) -> Result<Self> {
        let dense = nn::linear(
            config.hidden_size,
            config.intermediate_size,
            var.pp("dense"),
        )?;
        Ok(Self {
            dense,
            intermediate_act: config.hidden_act,
            span: tracing::span!(tracing::Level::TRACE, "inter"),
        })
    }
}

impl Module for ChineseClipTextIntermediate {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states = self.dense.forward(hidden_states)?;
        let ys = self.intermediate_act.forward(&hidden_states)?;
        Ok(ys)
    }
}

/// Copied from [`crate::models::bert::BertOutput`] to [`ChineseClipTextOutput`]
#[derive(Clone, Debug)]
struct ChineseClipTextOutput {
    dense: nn::Linear,
    layer_norm: nn::LayerNorm,
    dropout: nn::Dropout,
    span: tracing::Span,
}

impl ChineseClipTextOutput {
    fn new(var: nn::VarBuilder, config: &ChineseClipTextConfig) -> Result<Self> {
        let dense = nn::linear(
            config.intermediate_size,
            config.hidden_size,
            var.pp("dense"),
        )?;
        let layer_norm = nn::layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            var.pp("LayerNorm"),
        )?;
        let dropout = nn::Dropout::new(config.hidden_dropout_prob);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
            span: tracing::span!(tracing::Level::TRACE, "out"),
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states, false)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

/// Copied from [`crate::models::bert::BertLayer`] to [`ChineseClipTextLayer`]
#[derive(Clone, Debug)]
struct ChineseClipTextLayer {
    attention: ChineseClipTextAttention,
    intermediate: ChineseClipTextIntermediate,
    output: ChineseClipTextOutput,
    span: tracing::Span,
}

impl ChineseClipTextLayer {
    fn new(var: nn::VarBuilder, config: &ChineseClipTextConfig) -> Result<Self> {
        let attention = ChineseClipTextAttention::new(var.pp("attention"), config)?;
        let intermediate = ChineseClipTextIntermediate::new(var.pp("intermediate"), config)?;
        let output = ChineseClipTextOutput::new(var.pp("output"), config)?;
        Ok(Self {
            attention,
            intermediate,
            output,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let attention_output = self.attention.forward(hidden_states, attention_mask)?;
        // https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L523
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        let layer_output = self
            .output
            .forward(&intermediate_output, &attention_output)?;
        Ok(layer_output)
    }
}

#[derive(Clone, Debug)]
struct Tanh;

impl Tanh {
    pub fn new() -> Self {
        Self {}
    }
}
impl Module for Tanh {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.tanh()
    }
}

#[derive(Clone, Debug)]
struct ChineseClipTextPooler {
    dense: nn::Linear,
    activation: Tanh,
}

impl ChineseClipTextPooler {
    pub fn new(var: nn::VarBuilder, config: &ChineseClipTextConfig) -> Result<Self> {
        let dense = nn::linear(config.hidden_size, config.hidden_size, var.pp("dense"))?;
        let activation = Tanh::new();
        Ok(Self { dense, activation })
    }
}

impl Module for ChineseClipTextPooler {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let first_token_tensor = hidden_states.i((.., 0))?;
        let pooled_output = self.dense.forward(&first_token_tensor)?;
        let pooled_output = self.activation.forward(&pooled_output)?;
        Ok(pooled_output)
    }
}

#[derive(Clone, Debug)]
struct ChineseClipTextEncoder {
    layers: Vec<ChineseClipTextLayer>,
    span: tracing::Span,
}

impl ChineseClipTextEncoder {
    fn new(var: nn::VarBuilder, config: &ChineseClipTextConfig) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|index| ChineseClipTextLayer::new(var.pp(format!("layer.{index}")), config))
            .collect::<Result<Vec<_>>>()?;
        let span = tracing::span!(tracing::Level::TRACE, "encoder");
        Ok(ChineseClipTextEncoder { layers, span })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let mut hidden_states = hidden_states.clone();
        // Use a loop rather than a fold as it's easier to modify when adding debug/...
        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states, attention_mask)?
        }
        Ok(hidden_states)
    }
}

#[derive(Clone, Debug)]
pub struct ChineseClipTextTransformer {
    embeddings: ChineseClipTextEmbeddings,
    encoder: ChineseClipTextEncoder,
    pooler: Option<ChineseClipTextPooler>,
    pub device: Device,
    span: tracing::Span,
}

impl ChineseClipTextTransformer {
    pub fn new(var: nn::VarBuilder, config: &ChineseClipTextConfig) -> Result<Self> {
        let embeddings = ChineseClipTextEmbeddings::new(var.pp("embeddings"), config)?;
        let encoder = ChineseClipTextEncoder::new(var.pp("encoder"), config)?;
        // see: https://github.com/huggingface/transformers/blob/e40bb4845e0eefb52ec1e9cac9c2446ab36aef81/src/transformers/models/chinese_clip/modeling_chinese_clip.py#L1362
        // In the original Python version of the code, the pooler is not used, and there are no parameters for the pooler in the weight file.
        let pooler = if var.contains_tensor("pooler") {
            Some(ChineseClipTextPooler::new(var.pp("pooler"), config)?)
        } else {
            None
        };
        Ok(Self {
            embeddings,
            encoder,
            pooler,
            device: var.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let embedding_output = self.embeddings.forward(input_ids, token_type_ids)?;
        let attention_mask = match attention_mask {
            Some(attention_mask) => attention_mask.clone(),
            None => input_ids.ones_like()?,
        };
        // https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L995
        let attention_mask = get_extended_attention_mask(&attention_mask, DType::F32)?;
        let encoder_outputs = self.encoder.forward(&embedding_output, &attention_mask)?;
        let encoder_output = encoder_outputs.i((.., 0, ..))?;
        let pooled_output = match &self.pooler {
            Some(pooler) => pooler.forward(&encoder_output)?,
            None => encoder_output,
        };

        Ok(pooled_output)
    }
}

fn get_extended_attention_mask(attention_mask: &Tensor, dtype: DType) -> Result<Tensor> {
    let attention_mask = match attention_mask.rank() {
        3 => attention_mask.unsqueeze(1)?,
        2 => attention_mask.unsqueeze(1)?.unsqueeze(1)?,
        _ => candle::bail!("Wrong shape for input_ids or attention_mask"),
    };
    let attention_mask = attention_mask.to_dtype(dtype)?;
    // torch.finfo(dtype).min
    (attention_mask.ones_like()? - &attention_mask)?
        .broadcast_mul(&Tensor::try_from(f32::MIN)?.to_device(attention_mask.device())?)
}
