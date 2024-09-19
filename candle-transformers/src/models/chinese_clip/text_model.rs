use candle::{DType, IndexOp, Module, Result, Tensor};
use candle_nn::{layer_norm, linear, Dropout, LayerNorm, Linear, VarBuilder};

use super::Activation;

/// Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
/// positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
/// [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
/// For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
/// with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
#[derive(Clone, Debug)]
pub enum PositionEmbeddingType {
    // TODO: 2024/09/19 11:04:35 使用 serde 转成下划线命名
    Absolute,
    RelativeKey,
    RelativeKeyQuery,
}

#[derive(Clone, Debug)]
pub struct ChineseClipTextConfig {
    //  vocab_size=30522,
    //     hidden_size=768,
    //     num_hidden_layers=12,
    //     num_attention_heads=12,
    //     intermediate_size=3072,
    //     hidden_act="gelu",
    //     hidden_dropout_prob=0.1,
    //     attention_probs_dropout_prob=0.1,
    //     max_position_embeddings=512,
    //     type_vocab_size=2,
    //     initializer_range=0.02,
    //     initializer_factor=1.0,
    //     layer_norm_eps=1e-12,
    //     pad_token_id=0,
    //     position_embedding_type="absolute",
    //     use_cache=True,
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
    // "architectures": [
    //   "ChineseCLIPTextModel"
    // ],
    // "attention_probs_dropout_prob": 0.1,
    // "bos_token_id": 0,
    // "directionality": "bidi",
    // "eos_token_id": 2,
    // "hidden_act": "gelu",
    // "hidden_dropout_prob": 0.1,
    // "hidden_size": 768,
    // "initializer_range": 0.02,
    // "intermediate_size": 3072,
    // "layer_norm_eps": 1e-12,
    // "max_position_embeddings": 512,
    // "model_type": "chinese_clip_text_model",
    // "num_attention_heads": 12,
    // "num_hidden_layers": 12,
    // "output_past": true,
    // "pad_token_id": 0,
    // "pooler_fc_size": 768,
    // "pooler_num_attention_heads": 12,
    // "pooler_num_fc_layers": 3,
    // "pooler_size_per_head": 128,
    // "pooler_type": "first_token_transform",
    // "type_vocab_size": 2,
    // "vocab_size": 21128

    /// referer: https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16/blob/main/config.json
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

pub struct ChineseClipTextEmbeddings {
    word_embeddings: candle_nn::Embedding,
    position_embeddings: candle_nn::Embedding,
    token_type_embeddings: candle_nn::Embedding,
    layer_norm: candle_nn::LayerNorm,
    dropout: candle_nn::Dropout,
    position_embedding_type: PositionEmbeddingType,
    position_ids: candle::Tensor,
    token_type_ids: candle::Tensor,
}

impl ChineseClipTextEmbeddings {
    pub fn new(var: candle_nn::VarBuilder, config: &ChineseClipTextConfig) -> Result<Self> {
        let word_embeddings = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            var.pp("word_embeddings"),
        )?;
        let position_embeddings = candle_nn::embedding(
            config.max_position_embeddings,
            config.hidden_size,
            var.pp("position_embeddings"),
        )?;
        let token_type_embeddings = candle_nn::embedding(
            config.type_vocab_size,
            config.hidden_size,
            var.pp("token_type_embeddings"),
        )?;
        let layer_norm = candle_nn::layer_norm::<f64>(
            config.hidden_size,
            config.layer_norm_eps.into(),
            var.pp("layer_norm"),
        )?;
        let dropout = candle_nn::Dropout::new(config.hidden_dropout_prob);
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
            token_type_ids: token_type_ids,
        })
    }
}

impl Module for ChineseClipTextEmbeddings {
    fn forward(&self, xs: &candle::Tensor) -> Result<Tensor> {
        // let seq_length = input_ids.dim(D::Minus1)?;
        // let inputs_embeds = self.token_embedding.forward(input_ids)?;
        // let position_ids = self.position_ids.narrow(1, 0, seq_length)?;
        // let position_embedding = self.position_embedding.forward(&position_ids)?;
        // inputs_embeds.broadcast_add(&position_embedding)

        let input_shape = xs.shape();
        let seq_length = input_shape.dims1()?;
        let position_ids = (0..seq_length as u32).collect::<Vec<_>>();
        let position_ids = self.position_ids.index_select(
            &Tensor::new(&position_ids[..], self.position_ids.device())?,
            1,
        )?;

        let word_embeddings = self.word_embeddings.forward(&xs)?;
        let token_type_embeddings = self.token_type_embeddings.forward(&self.token_type_ids)?;
        let embeddings = (word_embeddings + token_type_embeddings)?;
        let embeddings = match self.position_embedding_type {
            PositionEmbeddingType::Absolute => {
                let position_embeddings = self.position_embeddings.forward(&position_ids)?;
                (embeddings + position_embeddings)?
            }
            _ => embeddings,
        };
        let embeddings = self.layer_norm.forward(&embeddings)?;
        let embeddings = self.dropout.forward(&embeddings, false)?;
        Ok(embeddings)
    }
}

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

struct ChineseClipTextPooler {
    dense: candle_nn::Linear,
    activation: Tanh,
}

impl ChineseClipTextPooler {
    pub fn new(var: candle_nn::VarBuilder, config: &ChineseClipTextConfig) -> Result<Self> {
        let dense = candle_nn::linear(config.hidden_size, config.hidden_size, var.pp("dense"))?;
        let activation = Tanh::new();
        Ok(Self { dense, activation })
    }
}

impl Module for ChineseClipTextPooler {
    fn forward(&self, hidden_states: &candle::Tensor) -> Result<Tensor> {
        let first_token_tensor = hidden_states.i((.., 0))?;
        let pooled_output = self.dense.forward(&first_token_tensor)?;
        let pooled_output = self.activation.forward(&pooled_output)?;
        Ok(pooled_output)
    }
}

/// Copied from [`crate::models:models::bert::BertSelfOutput`] to [`ChineseClipTextSelfOutput`]
struct ChineseClipTextSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
    span: tracing::Span,
}

impl ChineseClipTextSelfOutput {
    fn new(var: VarBuilder, config: &ChineseClipTextConfig) -> Result<Self> {
        let dense = linear(config.hidden_size, config.hidden_size, var.pp("dense"))?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            var.pp("LayerNorm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob);
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

/// Copied from [`crate::models:models::bert::BertSelfAttention`] to [`ChineseClipTextSelfAttention`]
struct ChineseClipTextSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    dropout: Dropout,
    num_attention_heads: usize,
    attention_head_size: usize,
    span: tracing::Span,
    span_softmax: tracing::Span,
}

impl ChineseClipTextSelfAttention {
    fn new(var: VarBuilder, config: &ChineseClipTextConfig) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let hidden_size = config.hidden_size;
        let query = linear(hidden_size, all_head_size, var.pp("query"))?;
        let value = linear(hidden_size, all_head_size, var.pp("value"))?;
        let key = linear(hidden_size, all_head_size, var.pp("key"))?;
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
            candle_nn::ops::softmax(&attention_scores, candle::D::Minus1)?
        };
        let attention_probs = self.dropout.forward(&attention_probs, false)?;

        let context_layer = attention_probs.matmul(&value_layer)?;
        let context_layer = context_layer.transpose(1, 2)?.contiguous()?;
        let context_layer = context_layer.flatten_from(candle::D::Minus2)?;
        Ok(context_layer)
    }
}

/// Copied from [`crate::models:models::bert::BertAttention`] to [`ChineseClipTextSelfAttention`]
struct ChineseClipTextAttention {
    self_attention: ChineseClipTextSelfAttention,
    self_output: ChineseClipTextSelfOutput,
    span: tracing::Span,
}

impl ChineseClipTextAttention {
    fn new(vb: VarBuilder, config: &ChineseClipTextConfig) -> Result<Self> {
        let self_attention = ChineseClipTextSelfAttention::new(vb.pp("self"), config)?;
        let self_output = ChineseClipTextSelfOutput::new(vb.pp("output"), config)?;
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

/// Copied from [`crate::models:models::bert::BertIntermediate`] to [`ChineseClipTextIntermediate`]
struct ChineseClipTextIntermediate {
    dense: Linear,
    intermediate_act: HiddenActLayer,
    span: tracing::Span,
}

impl ChineseClipTextIntermediate {
    fn new(var: VarBuilder, config: &ChineseClipTextConfig) -> Result<Self> {
        let dense = linear(
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

/// Copied from [`crate::models:models::bert::BertOutput`] to [`ChineseClipTextOutput`]
struct ChineseClipTextOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
    span: tracing::Span,
}

impl ChineseClipTextOutput {
    fn new(var: VarBuilder, config: &ChineseClipTextConfig) -> Result<Self> {
        let dense = linear(
            config.intermediate_size,
            config.hidden_size,
            var.pp("dense"),
        )?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            var.pp("LayerNorm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob);
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

struct BertLayer {
    attention: ChineseClipTextAttention,
    intermediate: ChineseClipTextIntermediate,
    output: ChineseClipTextOutput,
    span: tracing::Span,
}

impl BertLayer {
    fn load(var: VarBuilder, config: &ChineseClipTextConfig) -> Result<Self> {
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
        // TODO: Support cross-attention?
        // https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L523
        // TODO: Support something similar to `apply_chunking_to_forward`?
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        let layer_output = self
            .output
            .forward(&intermediate_output, &attention_output)?;
        Ok(layer_output)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use candle::{Device, IndexOp};

    #[test]
    pub fn test_tmp() {
        let data = candle::Tensor::arange(0.0, 100.0, &Device::Cpu).unwrap();
        println!("{:?}", data);
        println!("{:?}", data.shape());

        let data = data.unsqueeze(0).unwrap();
        println!("{:?}", data);
        println!("{:?}", data.shape());

        let seq_length = 10;
        let position_ids = (0..seq_length as u32).collect::<Vec<_>>();
        let position_ids = data
            .index_select(&Tensor::new(&position_ids[..], data.device()).unwrap(), 1)
            .unwrap();
        println!("{:?}", position_ids);
        println!("{:?}", position_ids.shape());

        let data = candle::Tensor::rand(1.0, 10.0, vec![2, 3], &Device::Cpu).unwrap();
        println!("---> {}", data.to_string());
        let data = data.i((.., 1..=2)).unwrap();
        print!("{}", data.to_string());
    }
}
