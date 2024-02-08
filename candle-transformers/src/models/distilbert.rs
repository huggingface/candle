use super::with_tracing::{layer_norm, linear, LayerNorm, Linear};
use candle::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};
use serde::Deserialize;

pub const DTYPE: DType = DType::F32;

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum HiddenAct {
    Gelu,
    Relu,
}

struct HiddenActLayer {
    act: HiddenAct,
    span: tracing::Span,
}

impl HiddenActLayer {
    fn new(act: HiddenAct) -> Self {
        let span = tracing::span!(tracing::Level::TRACE, "hidden-act");
        Self { act, span }
    }
}

impl Module for HiddenActLayer {
    fn forward(&self, xs: &Tensor) -> candle::Result<Tensor> {
        let _enter = self.span.enter();
        match self.act {
            // https://github.com/huggingface/transformers/blob/cd4584e3c809bb9e1392ccd3fe38b40daba5519a/src/transformers/activations.py#L213
            HiddenAct::Gelu => xs.gelu(),
            HiddenAct::Relu => xs.relu(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
enum PositionEmbeddingType {
    #[default]
    Absolute,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    vocab_size: usize,
    dim: usize,
    n_layers: usize,
    n_heads: usize,
    hidden_dim: usize,
    activation: HiddenAct,
    max_position_embeddings: usize,
    initializer_range: f64,
    pad_token_id: usize,
    #[serde(default)]
    position_embedding_type: PositionEmbeddingType,
    #[serde(default)]
    use_cache: bool,
    model_type: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 30522,
            dim: 768,
            n_layers: 12,
            n_heads: 12,
            hidden_dim: 3072,
            activation: HiddenAct::Gelu,
            max_position_embeddings: 512,
            initializer_range: 0.02,
            pad_token_id: 0,
            position_embedding_type: PositionEmbeddingType::Absolute,
            use_cache: true,
            model_type: Some("distilbert".to_string()),
        }
    }
}

struct Embeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    layer_norm: LayerNorm,
    span: tracing::Span,
}

impl Embeddings {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let word_embeddings =
            candle_nn::embedding(config.vocab_size, config.dim, vb.pp("word_embeddings"))?;
        let position_embeddings = candle_nn::embedding(
            config.max_position_embeddings,
            config.dim,
            vb.pp("position_embeddings"),
        )?;
        let layer_norm = layer_norm(config.dim, 1e-12, vb.pp("LayerNorm"))?;
        Ok(Self {
            word_embeddings,
            position_embeddings,
            layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "embeddings"),
        })
    }

    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (_bsize, seq_len) = input_ids.dims2()?;
        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let position_ids = (0..seq_len as u32).collect::<Vec<_>>();
        let position_ids = Tensor::new(&position_ids[..], input_ids.device())?;
        let embeddings =
            input_embeddings.broadcast_add(&self.position_embeddings.forward(&position_ids)?)?;

        let embeddings = self.layer_norm.forward(&embeddings)?;
        Ok(embeddings)
    }
}

struct MultiHeadSelfAttention {
    q_lin: Linear,
    k_lin: Linear,
    v_lin: Linear,
    out_lin: Linear,
    n_heads: usize,
    attention_head_size: usize,
    span: tracing::Span,
}

impl MultiHeadSelfAttention {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let attention_head_size = config.dim / config.n_heads;
        let all_head_size = config.n_heads * attention_head_size;
        let dim = config.dim;
        let q_lin = linear(dim, all_head_size, vb.pp("q_lin"))?;
        let v_lin = linear(dim, all_head_size, vb.pp("v_lin"))?;
        let k_lin = linear(dim, all_head_size, vb.pp("k_lin"))?;
        let out_lin = linear(all_head_size, dim, vb.pp("out_lin"))?;
        Ok(Self {
            q_lin,
            k_lin,
            v_lin,
            out_lin,
            n_heads: config.n_heads,
            attention_head_size,
            span: tracing::span!(tracing::Level::TRACE, "attention"),
        })
    }
}

impl MultiHeadSelfAttention {
    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (bs, q_length, _dim) = hidden_states.dims3()?;

        let dim_per_head = self.attention_head_size;
        let q = self.q_lin.forward(hidden_states)?;
        let k = self.k_lin.forward(hidden_states)?;
        let v = self.v_lin.forward(hidden_states)?;

        let q = q
            .reshape((bs, q_length, self.n_heads, dim_per_head))?
            .transpose(1, 2)?;
        let k = k
            .reshape((bs, q_length, self.n_heads, dim_per_head))?
            .transpose(1, 2)?;
        let v = v
            .reshape((bs, q_length, self.n_heads, dim_per_head))?
            .transpose(1, 2)?;

        let q: Tensor = (q / (dim_per_head as f64).sqrt())?;
        let scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let mask = attention_mask.broadcast_as(scores.shape())?;

        let scores = masked_fill(&scores.to_dtype(DType::F32)?, &mask, f32::NEG_INFINITY)?;
        let weights = candle_nn::ops::softmax(&scores, candle::D::Minus1)?;

        let context = weights.matmul(&v.contiguous()?)?;
        let context = context
            .transpose(1, 2)?
            .reshape((bs, q_length, self.n_heads * dim_per_head))?
            .contiguous()?;
        let context = self.out_lin.forward(&context)?;

        Ok(context)
    }
}

#[allow(clippy::upper_case_acronyms)]
struct FFN {
    lin1: Linear,
    lin2: Linear,
    activation: HiddenActLayer,
    span: tracing::Span,
}

impl FFN {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let lin1 = linear(config.dim, config.hidden_dim, vb.pp("lin1"))?;
        let lin2 = linear(config.hidden_dim, config.dim, vb.pp("lin2"))?;
        Ok(Self {
            lin1,
            lin2,
            activation: HiddenActLayer::new(config.activation),
            span: tracing::span!(tracing::Level::TRACE, "ffn"),
        })
    }
}

impl Module for FFN {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        hidden_states
            .apply(&self.lin1)?
            .apply(&self.activation)?
            .apply(&self.lin2)
    }
}

struct TransformerBlock {
    attention: MultiHeadSelfAttention,
    sa_layer_norm: LayerNorm,
    ffn: FFN,
    output_layer_norm: LayerNorm,
    span: tracing::Span,
}

impl TransformerBlock {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let attention = MultiHeadSelfAttention::load(vb.pp("attention"), config)?;
        let sa_layer_norm = layer_norm(config.dim, 1e-12, vb.pp("sa_layer_norm"))?;
        let ffn = FFN::load(vb.pp("ffn"), config)?;
        let output_layer_norm = layer_norm(config.dim, 1e-12, vb.pp("output_layer_norm"))?;
        Ok(Self {
            attention,
            sa_layer_norm,
            ffn,
            output_layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }
}

impl TransformerBlock {
    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let sa_output = self.attention.forward(hidden_states, attention_mask)?;
        // TODO: Support cross-attention?
        // https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L523
        // TODO: Support something similar to `apply_chunking_to_forward`?
        let sa_output = sa_output.broadcast_add(hidden_states)?;
        let sa_output = self.sa_layer_norm.forward(&sa_output)?;

        let ffn_output = self.ffn.forward(&sa_output)?;
        let ffn_output = (&ffn_output + sa_output)?;
        let output = self.output_layer_norm.forward(&ffn_output)?;
        Ok(output)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L556
struct Transformer {
    layers: Vec<TransformerBlock>,
    span: tracing::Span,
}

impl Transformer {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let layers = (0..config.n_layers)
            .map(|index| TransformerBlock::load(vb.pp(&format!("layer.{index}")), config))
            .collect::<Result<Vec<_>>>()?;
        let span = tracing::span!(tracing::Level::TRACE, "encoder");
        Ok(Transformer { layers, span })
    }
}

impl Transformer {
    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let mut hidden_states = hidden_states.clone();
        // Use a loop rather than a fold as it's easier to modify when adding debug/...
        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
        }
        Ok(hidden_states)
    }
}

pub struct DistilBertModel {
    embeddings: Embeddings,
    transformer: Transformer,
    pub device: Device,
    span: tracing::Span,
}

impl DistilBertModel {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let (embeddings, transformer) = match (
            Embeddings::load(vb.pp("embeddings"), config),
            Transformer::load(vb.pp("transformer"), config),
        ) {
            (Ok(embeddings), Ok(encoder)) => (embeddings, encoder),
            (Err(err), _) | (_, Err(err)) => {
                if let Some(model_type) = &config.model_type {
                    if let (Ok(embeddings), Ok(encoder)) = (
                        Embeddings::load(vb.pp(&format!("{model_type}.embeddings")), config),
                        Transformer::load(vb.pp(&format!("{model_type}.transformer")), config),
                    ) {
                        (embeddings, encoder)
                    } else {
                        return Err(err);
                    }
                } else {
                    return Err(err);
                }
            }
        };
        Ok(Self {
            embeddings,
            transformer,
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let embedding_output = self.embeddings.forward(input_ids)?;
        let sequence_output = self
            .transformer
            .forward(&embedding_output, attention_mask)?;
        Ok(sequence_output)
    }
}
