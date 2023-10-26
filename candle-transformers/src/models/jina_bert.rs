use super::with_tracing::{linear, linear_no_bias, Embedding, Linear};
use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{layer_norm, LayerNorm, Module, VarBuilder};
use serde::Deserialize;

pub const DTYPE: DType = DType::F32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PositionEmbeddingType {
    Absolute,
    Alibi,
}

// https://huggingface.co/jinaai/jina-bert-implementation/blob/main/configuration_bert.py
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: candle_nn::Activation,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub initializer_range: f64,
    pub layer_norm_eps: f64,
    pub pad_token_id: usize,
    pub position_embedding_type: PositionEmbeddingType,
}

impl Config {
    pub fn v2_base() -> Self {
        // https://huggingface.co/jinaai/jina-embeddings-v2-base-en/blob/main/config.json
        Self {
            vocab_size: 30528,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: candle_nn::Activation::Gelu,
            max_position_embeddings: 8192,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: PositionEmbeddingType::Alibi,
        }
    }
}

#[derive(Clone, Debug)]
struct BertEmbeddings {
    word_embeddings: Embedding,
    // no position_embeddings as we only support alibi.
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    span: tracing::Span,
}

impl BertEmbeddings {
    fn new(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let word_embeddings =
            Embedding::new(cfg.vocab_size, cfg.hidden_size, vb.pp("word_embeddings"))?;
        let token_type_embeddings = Embedding::new(
            cfg.type_vocab_size,
            cfg.hidden_size,
            vb.pp("token_type_embeddings"),
        )?;
        let layer_norm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))?;
        Ok(Self {
            word_embeddings,
            token_type_embeddings,
            layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "embeddings"),
        })
    }
}

impl Module for BertEmbeddings {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b_size, seq_len) = input_ids.dims2()?;
        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let token_type_embeddings = Tensor::zeros(seq_len, DType::U32, input_ids.device())?
            .broadcast_left(b_size)?
            .apply(&self.token_type_embeddings)?;
        let embeddings = (&input_embeddings + token_type_embeddings)?;
        let embeddings = self.layer_norm.forward(&embeddings)?;
        Ok(embeddings)
    }
}

#[derive(Clone, Debug)]
struct BertSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    num_attention_heads: usize,
    attention_head_size: usize,
    span: tracing::Span,
    span_softmax: tracing::Span,
}

impl BertSelfAttention {
    fn new(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let attention_head_size = cfg.hidden_size / cfg.num_attention_heads;
        let all_head_size = cfg.num_attention_heads * attention_head_size;
        let hidden_size = cfg.hidden_size;
        let query = linear(hidden_size, all_head_size, vb.pp("query"))?;
        let value = linear(hidden_size, all_head_size, vb.pp("value"))?;
        let key = linear(hidden_size, all_head_size, vb.pp("key"))?;
        Ok(Self {
            query,
            key,
            value,
            num_attention_heads: cfg.num_attention_heads,
            attention_head_size,
            span: tracing::span!(tracing::Level::TRACE, "self-attn"),
            span_softmax: tracing::span!(tracing::Level::TRACE, "softmax"),
        })
    }

    fn transpose_for_scores(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x_shape = xs.dims().to_vec();
        x_shape.pop();
        x_shape.push(self.num_attention_heads);
        x_shape.push(self.attention_head_size);
        xs.reshape(x_shape)?.transpose(1, 2)?.contiguous()
    }

    fn forward(&self, xs: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let query_layer = self.query.forward(xs)?;
        let key_layer = self.key.forward(xs)?;
        let value_layer = self.value.forward(xs)?;

        let query_layer = self.transpose_for_scores(&query_layer)?;
        let key_layer = self.transpose_for_scores(&key_layer)?;
        let value_layer = self.transpose_for_scores(&value_layer)?;

        let attention_scores = query_layer.matmul(&key_layer.t()?)?;
        let attention_scores = (attention_scores / (self.attention_head_size as f64).sqrt())?;
        let attention_scores = attention_scores.broadcast_add(bias)?;
        let attention_probs = {
            let _enter_sm = self.span_softmax.enter();
            candle_nn::ops::softmax_last_dim(&attention_scores)?
        };
        let context_layer = attention_probs.matmul(&value_layer)?;
        let context_layer = context_layer.transpose(1, 2)?.contiguous()?;
        let context_layer = context_layer.flatten_from(D::Minus2)?;
        Ok(context_layer)
    }
}

#[derive(Clone, Debug)]
struct BertSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    span: tracing::Span,
}

impl BertSelfOutput {
    fn new(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let dense = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))?;
        Ok(Self {
            dense,
            layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "self-out"),
        })
    }

    fn forward(&self, xs: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let xs = self.dense.forward(xs)?;
        self.layer_norm.forward(&(xs + input_tensor)?)
    }
}

#[derive(Clone, Debug)]
struct BertAttention {
    self_attention: BertSelfAttention,
    self_output: BertSelfOutput,
    span: tracing::Span,
}

impl BertAttention {
    fn new(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let self_attention = BertSelfAttention::new(vb.pp("self"), cfg)?;
        let self_output = BertSelfOutput::new(vb.pp("output"), cfg)?;
        Ok(Self {
            self_attention,
            self_output,
            span: tracing::span!(tracing::Level::TRACE, "attn"),
        })
    }

    fn forward(&self, xs: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let self_outputs = self.self_attention.forward(xs, bias)?;
        let attention_output = self.self_output.forward(&self_outputs, xs)?;
        Ok(attention_output)
    }
}

#[derive(Clone, Debug)]
struct BertGLUMLP {
    gated_layers: Linear,
    act: candle_nn::Activation,
    wo: Linear,
    layernorm: LayerNorm,
    intermediate_size: usize,
}

impl BertGLUMLP {
    fn new(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let gated_layers = linear_no_bias(
            cfg.hidden_size,
            cfg.intermediate_size * 2,
            vb.pp("gated_layers"),
        )?;
        let act = candle_nn::Activation::Gelu; // geglu
        let wo = linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("wo"))?;
        let layernorm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layernorm"))?;
        Ok(Self {
            gated_layers,
            act,
            wo,
            layernorm,
            intermediate_size: cfg.intermediate_size,
        })
    }
}

impl Module for BertGLUMLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = xs.apply(&self.gated_layers)?;
        let gated = xs.narrow(D::Minus1, 0, self.intermediate_size)?;
        let non_gated = xs.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
        let xs = (gated.apply(&self.act) * non_gated)?.apply(&self.wo);
        (xs + residual)?.apply(&self.layernorm)
    }
}

#[derive(Clone, Debug)]
struct BertLayer {
    attention: BertAttention,
    mlp: BertGLUMLP,
    span: tracing::Span,
}

impl BertLayer {
    fn new(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let attention = BertAttention::new(vb.pp("attention"), cfg)?;
        let mlp = BertGLUMLP::new(vb.pp("mlp"), cfg)?;
        Ok(Self {
            attention,
            mlp,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    fn forward(&self, xs: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.attention.forward(xs, bias)?.apply(&self.mlp)
    }
}

fn build_alibi_bias(cfg: &Config) -> Result<Tensor> {
    let n_heads = cfg.num_attention_heads;
    let seq_len = cfg.max_position_embeddings;
    let alibi_bias = Tensor::arange(0, seq_len as i64, &Device::Cpu)?.to_dtype(DType::F32)?;
    let alibi_bias = {
        let a1 = alibi_bias.reshape((1, seq_len))?;
        let a2 = alibi_bias.reshape((seq_len, 1))?;
        a1.broadcast_sub(&a2)?.abs()?.broadcast_left(n_heads)?
    };
    let mut n_heads2 = 1;
    while n_heads2 < n_heads {
        n_heads2 *= 2
    }
    let slopes = (1..=n_heads2)
        .map(|v| -1f32 / 2f32.powf((v * 8) as f32 / n_heads2 as f32))
        .collect::<Vec<_>>();
    let slopes = if n_heads2 == n_heads {
        slopes
    } else {
        slopes
            .iter()
            .skip(1)
            .step_by(2)
            .chain(slopes.iter().step_by(2))
            .take(n_heads)
            .cloned()
            .collect::<Vec<f32>>()
    };
    let slopes = Tensor::new(slopes, &Device::Cpu)?.reshape((1, (), 1, 1))?;
    alibi_bias.to_dtype(DType::F32)?.broadcast_mul(&slopes)
}

#[derive(Clone, Debug)]
struct BertEncoder {
    alibi: Tensor,
    layers: Vec<BertLayer>,
    span: tracing::Span,
}

impl BertEncoder {
    fn new(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        if cfg.position_embedding_type != PositionEmbeddingType::Alibi {
            candle::bail!("only alibi is supported as a position-embedding-type")
        }
        let layers = (0..cfg.num_hidden_layers)
            .map(|index| BertLayer::new(vb.pp(&format!("layer.{index}")), cfg))
            .collect::<Result<Vec<_>>>()?;
        let span = tracing::span!(tracing::Level::TRACE, "encoder");
        let alibi = build_alibi_bias(cfg)?.to_device(vb.device())?;
        Ok(Self {
            alibi,
            layers,
            span,
        })
    }
}

impl Module for BertEncoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let seq_len = xs.dim(1)?;
        let alibi_bias = self.alibi.i((.., .., ..seq_len, ..seq_len))?;
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, &alibi_bias)?
        }
        Ok(xs)
    }
}

#[derive(Clone, Debug)]
pub struct BertModel {
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
    pub device: Device,
    span: tracing::Span,
}

impl BertModel {
    pub fn new(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let embeddings = BertEmbeddings::new(vb.pp("embeddings"), cfg)?;
        let encoder = BertEncoder::new(vb.pp("encoder"), cfg)?;
        Ok(Self {
            embeddings,
            encoder,
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }
}

impl Module for BertModel {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let embedding_output = self.embeddings.forward(input_ids)?;
        let sequence_output = self.encoder.forward(&embedding_output)?;
        Ok(sequence_output)
    }
}
