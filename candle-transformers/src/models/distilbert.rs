use super::with_tracing::{layer_norm, linear, LayerNorm, Linear};
use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::ops::softmax;
use candle_nn::{Activation, Dropout, Embedding, Module, VarBuilder};
use serde::Deserialize;
use std::collections::HashMap;

pub const DTYPE: DType = DType::F32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
enum PositionEmbeddingType {
    #[default]
    Absolute,
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/configuration_bert.py#L1
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    vocab_size: usize,
    max_position_embeddings: usize,
    sinusoidal_pos_embds: bool,
    n_layers: usize,
    n_heads: usize,
    dim: usize,
    hidden_dim: usize,
    dropout: f32,
    attention_dropout: f32,
    activation: Activation,
    initializer_range: f32,
    qa_dropout: f32,
    #[serde(default)]
    seq_classif_dropout: f32,
    pad_token_id: usize,
    model_type: Option<String>,
    #[serde(default)]
    #[serde(flatten)]
    classifier_config: Option<ClassifierConfig>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
pub struct ClassifierConfig {
    id2label: HashMap<String, String>,
    label2id: HashMap<String, i64>,
}

impl ClassifierConfig {
    fn num_labels(&self) -> Result<usize> {
        if self.label2id.len() != self.id2label.len() {
            return Err(candle::Error::Msg(
                "incorrect label2id or id2label".to_string(),
            ));
        }

        Ok(self.label2id.len())
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 30522,
            max_position_embeddings: 512,
            sinusoidal_pos_embds: false,
            n_layers: 6,
            n_heads: 12,
            dim: 768,
            hidden_dim: 4 * 786,
            dropout: 0.1,
            attention_dropout: 0.1,
            activation: Activation::Gelu,
            initializer_range: 0.002,
            qa_dropout: 0.1,
            seq_classif_dropout: 0.2,
            pad_token_id: 0,
            model_type: Some("distilbert".to_string()),
            classifier_config: None,
        }
    }
}

struct DistilBertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
    span: tracing::Span,
}

impl DistilBertEmbeddings {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let word_embeddings = embedding(config.vocab_size, config.dim, vb.pp("word_embeddings"))?;
        let position_embeddings = embedding(
            config.max_position_embeddings,
            config.dim,
            vb.pp("position_embeddings"),
        )?;
        let layer_norm = layer_norm(config.dim, 1e-12, vb.pp("LayerNorm"))?;
        Ok(Self {
            word_embeddings,
            position_embeddings,
            layer_norm,
            dropout: Dropout::new(config.dropout),
            span: tracing::span!(tracing::Level::TRACE, "embeddings"),
        })
    }
}

impl Module for DistilBertEmbeddings {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let input_embeds = self.word_embeddings.forward(&input_ids)?;
        let seq_length = input_embeds.dim(1)?;

        let position_ids = Tensor::arange(0, seq_length as i64, input_ids.device())?;
        let position_ids = position_ids.unsqueeze(0)?.expand(input_ids.shape())?;
        let position_embeddings = position_ids.apply(&self.position_embeddings)?;

        input_embeds
            .add(&position_embeddings)?
            .apply(&self.layer_norm)?
            .apply_t(&self.dropout, false)
    }
}

struct MultiHeadSelfAttention {
    n_heads: usize,
    dim: usize,
    dropout: Dropout,
    q_lin: Linear,
    k_lin: Linear,
    v_lin: Linear,
    out_lin: Linear,
    span: tracing::Span,
}

impl MultiHeadSelfAttention {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dim = config.dim;
        let n_heads = config.n_heads;

        assert_eq!(dim % n_heads, 0, "n_heads must divide dim evenly");

        Ok(Self {
            n_heads: config.n_heads,
            dim: config.dim,
            dropout: Dropout::new(config.attention_dropout),
            q_lin: linear(dim, dim, vb.pp("q_lin"))?,
            k_lin: linear(dim, dim, vb.pp("k_lin"))?,
            v_lin: linear(dim, dim, vb.pp("v_lin"))?,
            out_lin: linear(dim, dim, vb.pp("out_lin"))?,
            span: tracing::span!(tracing::Level::TRACE, "attentions"),
        })
    }

    fn shape(&self, x: &Tensor, bs: usize, dim_per_head: usize) -> Result<Tensor> {
        x.reshape((bs, (), self.n_heads, dim_per_head))?
            .transpose(1, 2)
    }

    fn unshape(&self, x: &Tensor, bs: usize, dim_per_head: usize) -> Result<Tensor> {
        x.transpose(1, 2)?
            .contiguous()?
            .reshape((bs, (), self.n_heads * dim_per_head))
    }
}

impl Module for MultiHeadSelfAttention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let bs = xs.dim(0)?;

        let dim_per_head = self.dim / self.n_heads;

        let query_layer = xs.apply(&self.q_lin)?;
        let key_layer = xs.apply(&self.k_lin)?;
        let value_layer = xs.apply(&self.v_lin)?;

        let q = self.shape(&query_layer, bs, dim_per_head)?;
        let k = self.shape(&key_layer, bs, dim_per_head)?;
        let v = self.shape(&value_layer, bs, dim_per_head)?;

        let q = (q / (dim_per_head as f64).sqrt())?;

        let scores = q.matmul(&k.t()?)?;

        let weights = softmax(&scores, D::Minus1)?.apply_t(&self.dropout, false)?;

        let context = weights.matmul(&v)?;
        let context = self
            .unshape(&context, bs, dim_per_head)?
            .apply(&self.out_lin)?;

        Ok(context)
    }
}

struct FFN {
    dropout: Dropout,
    lin1: Linear,
    lin2: Linear,
    activation: Activation,
    span: tracing::Span,
}

impl FFN {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        Ok(Self {
            dropout: Dropout::new(config.attention_dropout),
            lin1: linear(config.dim, config.hidden_dim, vb.pp("lin1"))?,
            lin2: linear(config.hidden_dim, config.dim, vb.pp("lin2"))?,
            activation: config.activation,
            span: tracing::span!(tracing::Level::TRACE, "ffn"),
        })
    }
}

impl Module for FFN {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        xs.apply(&self.lin1)?
            .apply(&self.activation)?
            .apply(&self.lin2)?
            .apply_t(&self.dropout, false)
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
        Ok(Self {
            attention: MultiHeadSelfAttention::load(vb.pp("attention"), config)?,
            sa_layer_norm: layer_norm(config.dim, 1e-12, vb.pp("sa_layer_norm"))?,
            ffn: FFN::load(vb.pp("ffn"), config)?,
            output_layer_norm: layer_norm(config.dim, 1e-12, vb.pp("output_layer_norm"))?,
            span: tracing::span!(tracing::Level::TRACE, "TransformerBlock"),
        })
    }
}

impl Module for TransformerBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        xs.apply(&self.attention)?
            .add(xs)?
            .apply(&self.sa_layer_norm)?
            .apply(&self.ffn)?
            .apply(&self.output_layer_norm)
    }
}

struct Transformer {
    layer: Vec<TransformerBlock>,
}

impl Transformer {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        Ok(Self {
            layer: (0..config.n_layers)
                .map(|i| TransformerBlock::load(vb.push_prefix("layer").pp(i), config))
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

impl Module for Transformer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.layer
            .iter()
            .try_fold(x.copy()?, |hidden_state, layer| hidden_state.apply(layer))
    }
}

pub struct DistilBertModel {
    embeddings: DistilBertEmbeddings,
    transformer: Transformer,
    pub device: Device,
    span: tracing::Span,
}

impl DistilBertModel {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        Ok(Self {
            embeddings: DistilBertEmbeddings::load(vb.pp("embeddings"), config)?,
            transformer: Transformer::load(vb.pp("transformer"), config)?,
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }
}

impl Module for DistilBertModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        xs.apply(&self.embeddings)?.apply(&self.transformer)
    }
}

pub struct DistilBertForSequenceClassification {
    distilbert: DistilBertModel,
    pre_classifier: Linear,
    classifier: Linear,
    dropout: Dropout,
    pub device: Device,
    classifier_config: ClassifierConfig,
    span: tracing::Span,
}

impl DistilBertForSequenceClassification {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let classifier_config = config
            .classifier_config
            .clone()
            .ok_or_else(|| candle::Error::Msg("label2id or id2label not found".to_string()))?;
        Ok(Self {
            distilbert: DistilBertModel::load(vb.pp("distilbert"), config)?,
            pre_classifier: linear(config.dim, config.dim, vb.pp("pre_classifier"))?,
            classifier: linear(
                config.dim,
                classifier_config.num_labels()?,
                vb.pp("classifier"),
            )?,
            dropout: Dropout::new(config.seq_classif_dropout),
            device: vb.device().clone(),
            classifier_config,
            span: tracing::span!(tracing::Level::TRACE, "model_for_multiple_choice"),
        })
    }
}

impl Module for DistilBertForSequenceClassification {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();

        let logits = xs
            .apply(&self.distilbert)?
            .i(0)?
            .i((.., 0))?
            .apply(&self.pre_classifier)?
            .relu()?
            .apply_t(&self.dropout, false)?
            .apply(&self.classifier)?;

        Ok(logits)
    }
}

fn embedding(vocab_size: usize, hidden_size: usize, vb: VarBuilder) -> Result<Embedding> {
    let embeddings = vb.get((vocab_size, hidden_size), "weight")?;
    Ok(Embedding::new(embeddings, hidden_size))
}
