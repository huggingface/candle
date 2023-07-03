#![allow(dead_code)]

use anyhow::Result as R;
use candle::{DType, Device, Result, Tensor};

const DTYPE: DType = DType::F32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HiddenAct {
    Gelu,
    Relu,
}

impl HiddenAct {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Gelu => xs.gelu(),
            Self::Relu => xs.relu(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PositionEmbeddingType {
    Absolute,
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/configuration_bert.py#L1
#[derive(Debug, Clone, PartialEq)]
struct Config {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    hidden_act: HiddenAct,
    hidden_dropout_prob: f64,
    max_position_embeddings: usize,
    type_vocab_size: usize,
    initializer_range: f64,
    layer_norm_eps: f64,
    pad_token_id: usize,
    position_embedding_type: PositionEmbeddingType,
    use_cache: bool,
    classifier_dropout: Option<f64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: HiddenAct::Gelu,
            hidden_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: PositionEmbeddingType::Absolute,
            use_cache: true,
            classifier_dropout: None,
        }
    }
}

struct Embedding {
    embeddings: Tensor,
}

impl Embedding {
    fn new(embeddings: Tensor) -> Self {
        Self { embeddings }
    }

    fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        Tensor::embedding(indexes, &self.embeddings)
    }
}

struct Linear {
    weight: Tensor,
}

impl Linear {
    fn new(weight: Tensor) -> Self {
        Self { weight }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.matmul(&self.weight.t()?)?;
        Ok(x)
    }
}

struct Dropout {
    pr: f64,
}

impl Dropout {
    fn new(pr: f64) -> Self {
        Self { pr }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // TODO
        Ok(x.clone())
    }
}

struct LayerNorm {
    scale: Tensor,
}

impl LayerNorm {
    fn new(scale: Tensor) -> Self {
        Self { scale }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (seq_len, hidden_size) = x.shape().r2()?;
        let norm_x = ((x * x)?.sum(&[1])? / hidden_size as f64)?;
        let norm_x = norm_x.broadcast_as((seq_len, hidden_size))?;
        let x_normed = (x / (norm_x + 1e-5)?.sqrt()?)?;
        let size = self.scale.shape().r1()?;
        let scale = self.scale.broadcast_as((seq_len, size))?;
        let x = (scale * x_normed)?;
        Ok(x)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L180
struct BertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Option<Embedding>,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
    position_ids: Tensor,
    token_type_ids: Tensor,
}

impl BertEmbeddings {
    fn load(device: &Device, config: &Config) -> Result<Self> {
        let word_embeddings =
            Tensor::zeros((config.vocab_size, config.hidden_size), DTYPE, device)?;
        let position_embeddings = Tensor::zeros(
            (config.max_position_embeddings, config.hidden_size),
            DTYPE,
            device,
        )?;
        let token_type_embeddings =
            Tensor::zeros((config.type_vocab_size, config.hidden_size), DTYPE, device)?;
        let layer_norm = Tensor::zeros((), DTYPE, device)?;
        let position_ids: Vec<_> = (0..config.max_position_embeddings as u32).collect();
        let position_ids = Tensor::new(&position_ids[..], device)?.unsqueeze(0)?;
        let token_type_ids = position_ids.zeros_like()?;
        Ok(Self {
            word_embeddings: Embedding::new(word_embeddings),
            position_embeddings: Some(Embedding::new(position_embeddings)),
            token_type_embeddings: Embedding::new(token_type_embeddings),
            layer_norm: LayerNorm::new(layer_norm),
            dropout: Dropout::new(config.hidden_dropout_prob),
            position_ids,
            token_type_ids,
        })
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;
        let mut embeddings = (input_embeddings + token_type_embeddings)?;
        if let Some(position_embeddings) = &self.position_embeddings {
            embeddings = (&embeddings + position_embeddings.forward(&embeddings))?
        }
        let embeddings = self.layer_norm.forward(&embeddings)?;
        let embeddings = self.dropout.forward(&embeddings)?;
        Ok(embeddings)
    }
}

struct BertSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    dropout: Dropout,
    num_attention_heads: usize,
    attention_head_size: usize,
}

impl BertSelfAttention {
    fn load(device: &Device, config: &Config) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let query = Tensor::zeros((config.hidden_size, all_head_size), DTYPE, device)?;
        let query = Linear::new(query);
        let value = Tensor::zeros((config.hidden_size, all_head_size), DTYPE, device)?;
        let value = Linear::new(value);
        let key = Tensor::zeros((config.hidden_size, all_head_size), DTYPE, device)?;
        let key = Linear::new(key);
        Ok(Self {
            query,
            key,
            value,
            dropout,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
        })
    }

    fn transpose_for_scores(&self, xs: &Tensor) -> Result<Tensor> {
        let mut new_x_shape = xs.dims().to_vec();
        new_x_shape.pop();
        new_x_shape.push(self.num_attention_heads);
        new_x_shape.push(self.attention_head_size);
        xs.reshape(new_x_shape.as_slice())?.transpose(1, 2)
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let query_layer = self.query.forward(hidden_states)?;
        let key_layer = self.key.forward(hidden_states)?;
        let value_layer = self.value.forward(hidden_states)?;

        let query_layer = self.transpose_for_scores(&query_layer)?;
        let key_layer = self.transpose_for_scores(&key_layer)?;
        let value_layer = self.transpose_for_scores(&value_layer)?;

        let attention_scores = query_layer.matmul(&key_layer.t()?)?;
        let attention_scores = (attention_scores / (self.attention_head_size as f64).sqrt())?;
        let attention_probs = attention_scores.softmax(attention_scores.rank() - 1)?;
        let attention_probs = self.dropout.forward(&attention_probs)?;

        let context_layer = attention_probs.matmul(&value_layer)?;
        let context_layer = context_layer.transpose(1, 2)?.contiguous()?;
        let context_layer = context_layer.flatten(Some(context_layer.rank() - 2), None)?;
        Ok(context_layer)
    }
}

struct BertSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl BertSelfOutput {
    fn load(device: &Device, config: &Config) -> Result<Self> {
        let dense = Tensor::zeros((config.hidden_size, config.hidden_size), DTYPE, device)?;
        let dense = Linear::new(dense);
        let layer_norm = Tensor::zeros((), DTYPE, device)?;
        let layer_norm = LayerNorm::new(layer_norm);
        let dropout = Dropout::new(config.hidden_dropout_prob);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L392
struct BertAttention {
    self_attention: BertSelfAttention,
    self_output: BertSelfOutput,
}

impl BertAttention {
    fn load(device: &Device, config: &Config) -> Result<Self> {
        let self_attention = BertSelfAttention::load(device, config)?;
        let self_output = BertSelfOutput::load(device, config)?;
        Ok(Self {
            self_attention,
            self_output,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let self_outputs = self.self_attention.forward(hidden_states)?;
        let attention_output = self.self_output.forward(&self_outputs, hidden_states)?;
        Ok(attention_output)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L441
struct BertIntermediate {
    dense: Linear,
    intermediate_act: HiddenAct,
}

impl BertIntermediate {
    fn load(device: &Device, config: &Config) -> Result<Self> {
        let dense = Tensor::zeros(
            (config.hidden_size, config.intermediate_size),
            DTYPE,
            device,
        )?;
        let dense = Linear::new(dense);
        Ok(Self {
            dense,
            intermediate_act: config.hidden_act,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        self.intermediate_act.forward(&hidden_states)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L456
struct BertOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl BertOutput {
    fn load(device: &Device, config: &Config) -> Result<Self> {
        let dense = Tensor::zeros(
            (config.intermediate_size, config.hidden_size),
            DTYPE,
            device,
        )?;
        let dense = Linear::new(dense);
        let layer_norm = Tensor::zeros((), DTYPE, device)?;
        let layer_norm = LayerNorm::new(layer_norm);
        let dropout = Dropout::new(config.hidden_dropout_prob);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L470
struct BertLayer {
    attention: BertAttention,
    intermediate: BertIntermediate,
    output: BertOutput,
}

impl BertLayer {
    fn load(device: &Device, config: &Config) -> Result<Self> {
        let attention = BertAttention::load(device, config)?;
        let intermediate = BertIntermediate::load(device, config)?;
        let output = BertOutput::load(device, config)?;
        Ok(Self {
            attention,
            intermediate,
            output,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let attention_output = self.attention.forward(hidden_states)?;
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

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L556
struct BertEncoder {
    layers: Vec<BertLayer>,
}

impl BertEncoder {
    fn load(device: &Device, config: &Config) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|_index| BertLayer::load(device, config))
            .collect::<Result<Vec<_>>>()?;
        Ok(BertEncoder { layers })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();
        // Use a loop rather than a fold as it's easier to modify when adding debug/...
        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states)?
        }
        Ok(hidden_states)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L874
struct BertModel {
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
}

impl BertModel {
    fn load(device: &Device, config: &Config) -> Result<Self> {
        let embeddings = BertEmbeddings::load(device, config)?;
        let encoder = BertEncoder::load(device, config)?;
        Ok(Self {
            embeddings,
            encoder,
        })
    }

    fn forward(&self, input_ids: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        let embedding_output = self.embeddings.forward(input_ids, position_ids)?;
        let sequence_output = self.encoder.forward(&embedding_output)?;
        Ok(sequence_output)
    }
}

fn main() -> R<()> {
    Ok(())
}
