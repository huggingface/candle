#![allow(dead_code)]
// The tokenizer.json and weights should be retrieved from:
// https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

use anyhow::{Error as E, Result};
use candle::{safetensors::SafeTensors, DType, Device, Shape, Tensor};
use clap::Parser;
use std::collections::HashMap;

const DTYPE: DType = DType::F32;

struct VarBuilder<'a> {
    safetensors: Option<(HashMap<String, usize>, Vec<SafeTensors<'a>>)>,
    dtype: DType,
    device: Device,
}

impl<'a> VarBuilder<'a> {
    pub fn from_safetensors(
        safetensors: Vec<SafeTensors<'a>>,
        dtype: DType,
        device: Device,
    ) -> Self {
        let mut routing = HashMap::new();
        for (index, sf) in safetensors.iter().enumerate() {
            for k in sf.names() {
                routing.insert(k.to_string(), index);
            }
        }
        Self {
            safetensors: Some((routing, safetensors)),
            device,
            dtype,
        }
    }

    pub fn zeros(dtype: DType, device: Device) -> Self {
        Self {
            safetensors: None,
            device,
            dtype,
        }
    }

    pub fn get<S: Into<Shape>>(&self, s: S, tensor_name: &str) -> candle::Result<Tensor> {
        let s: Shape = s.into();
        match &self.safetensors {
            None => Tensor::zeros(s, self.dtype, &self.device),
            Some((routing, safetensors)) => {
                // Unwrap or 0  just to let the proper error flow.
                let index = routing.get(tensor_name).unwrap_or(&0);
                let tensor = safetensors[*index]
                    .tensor(tensor_name, &self.device)?
                    .to_dtype(self.dtype)?;
                if *tensor.shape() != s {
                    let msg = format!("shape mismatch for {tensor_name}");
                    Err(candle::Error::UnexpectedShape {
                        msg,
                        expected: s,
                        got: tensor.shape().clone(),
                    })?
                }
                Ok(tensor)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HiddenAct {
    Gelu,
    Relu,
}

impl HiddenAct {
    fn forward(&self, xs: &Tensor) -> candle::Result<Tensor> {
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

impl Config {
    fn all_mini_lm_l6_v2() -> Self {
        // https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/config.json
        Self {
            vocab_size: 30522,
            hidden_size: 384,
            num_hidden_layers: 6,
            num_attention_heads: 12,
            intermediate_size: 1536,
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
    hidden_size: usize,
}

impl Embedding {
    fn new(embeddings: Tensor, hidden_size: usize) -> Self {
        Self {
            embeddings,
            hidden_size,
        }
    }

    fn load(vocab_size: usize, hidden_size: usize, p: &str, vb: &VarBuilder) -> Result<Self> {
        let embeddings = vb.get((vocab_size, hidden_size), &format!("{p}.weight"))?;
        Ok(Self::new(embeddings, hidden_size))
    }

    fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        let mut final_dims = indexes.dims().to_vec();
        final_dims.push(self.hidden_size);
        let indexes = indexes.flatten_all()?;
        let values = Tensor::embedding(&indexes, &self.embeddings)?;
        let values = values.reshape(final_dims)?;
        Ok(values)
    }
}

struct Linear {
    weight: Tensor,
    bias: Tensor,
}

impl Linear {
    fn new(weight: Tensor, bias: Tensor) -> Self {
        Self { weight, bias }
    }

    fn load(size1: usize, size2: usize, p: &str, vb: &VarBuilder) -> Result<Self> {
        let weight = vb.get((size2, size1), &format!("{p}.weight"))?;
        let bias = vb.get(size2, &format!("{p}.bias"))?;
        Ok(Self::new(weight, bias))
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.matmul(&self.weight.t()?)?;
        let x = x.broadcast_add(&self.bias)?;
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

// This layer norm version handles both weight and bias so removes the mean.
struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    fn new(weight: Tensor, bias: Tensor, eps: f64) -> Self {
        Self { weight, bias, eps }
    }

    fn load(size: usize, eps: f64, p: &str, vb: &VarBuilder) -> Result<Self> {
        let weight = vb.get(size, &format!("{p}.weight"))?;
        let bias = vb.get(size, &format!("{p}.bias"))?;
        Ok(Self { weight, bias, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (seq_len, hidden_size) = x.shape().r2()?;
        let mean_x = (x.sum(&[1])? / hidden_size as f64)?;
        let x = x.broadcast_sub(&mean_x)?;
        let norm_x = ((&x * &x)?.sum(&[1])? / hidden_size as f64)?;
        let norm_x = norm_x.broadcast_as((seq_len, hidden_size))?;
        let x_normed = (x / (norm_x + self.eps)?.sqrt()?)?;
        let x = x_normed
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)?;
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
    fn load(p: &str, vb: &VarBuilder, config: &Config) -> Result<Self> {
        let word_embeddings = Embedding::load(
            config.vocab_size,
            config.hidden_size,
            &format!("{p}.word_embeddings"),
            vb,
        )?;
        let position_embeddings = Embedding::load(
            config.max_position_embeddings,
            config.hidden_size,
            &format!("{p}.position_embeddings"),
            vb,
        )?;
        let token_type_embeddings = Embedding::load(
            config.type_vocab_size,
            config.hidden_size,
            &format!("{p}.token_type_embeddings"),
            vb,
        )?;
        let layer_norm = LayerNorm::load(
            config.hidden_size,
            config.layer_norm_eps,
            &format!("{p}.LayerNorm"),
            vb,
        )?;
        let position_ids: Vec<_> = (0..config.max_position_embeddings as u32).collect();
        let position_ids = Tensor::new(&position_ids[..], &vb.device)?.unsqueeze(0)?;
        let token_type_ids = position_ids.zeros_like()?;
        Ok(Self {
            word_embeddings,
            position_embeddings: Some(position_embeddings),
            token_type_embeddings,
            layer_norm,
            dropout: Dropout::new(config.hidden_dropout_prob),
            position_ids,
            token_type_ids,
        })
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let seq_len = input_ids.shape().r1()?;
        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;
        let mut embeddings = (&input_embeddings + token_type_embeddings)?;
        if let Some(position_embeddings) = &self.position_embeddings {
            // TODO: Proper absolute positions?
            let position_ids = (0..seq_len as u32).collect::<Vec<_>>();
            let position_ids = Tensor::new(&position_ids[..], &input_ids.device())?;
            embeddings = (&embeddings + position_embeddings.forward(&position_ids)?)?
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
    fn load(p: &str, vb: &VarBuilder, config: &Config) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let hidden_size = config.hidden_size;
        let query = Linear::load(hidden_size, all_head_size, &format!("{p}.query"), vb)?;
        let value = Linear::load(hidden_size, all_head_size, &format!("{p}.value"), vb)?;
        let key = Linear::load(hidden_size, all_head_size, &format!("{p}.key"), vb)?;
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
        // Be cautious about the transposition if adding a batch dim!
        let xs = xs.reshape(new_x_shape.as_slice())?.transpose(0, 1)?;
        Ok(xs.contiguous()?)
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
        let context_layer = context_layer.transpose(0, 1)?.contiguous()?;
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
    fn load(p: &str, vb: &VarBuilder, config: &Config) -> Result<Self> {
        let dense = Linear::load(
            config.hidden_size,
            config.hidden_size,
            &format!("{p}.dense"),
            vb,
        )?;
        let layer_norm = LayerNorm::load(
            config.hidden_size,
            config.layer_norm_eps,
            &format!("{p}.LayerNorm"),
            vb,
        )?;
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
    fn load(p: &str, vb: &VarBuilder, config: &Config) -> Result<Self> {
        let self_attention = BertSelfAttention::load(&format!("{p}.self"), vb, config)?;
        let self_output = BertSelfOutput::load(&format!("{p}.output"), vb, config)?;
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
    fn load(p: &str, vb: &VarBuilder, config: &Config) -> Result<Self> {
        let dense = Linear::load(
            config.hidden_size,
            config.intermediate_size,
            &format!("{p}.dense"),
            vb,
        )?;
        Ok(Self {
            dense,
            intermediate_act: config.hidden_act,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let ys = self.intermediate_act.forward(&hidden_states)?;
        Ok(ys)
    }
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L456
struct BertOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl BertOutput {
    fn load(p: &str, vb: &VarBuilder, config: &Config) -> Result<Self> {
        let dense = Linear::load(
            config.intermediate_size,
            config.hidden_size,
            &format!("{p}.dense"),
            vb,
        )?;
        let layer_norm = LayerNorm::load(
            config.hidden_size,
            config.layer_norm_eps,
            &format!("{p}.LayerNorm"),
            vb,
        )?;
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
    fn load(p: &str, vb: &VarBuilder, config: &Config) -> Result<Self> {
        let attention = BertAttention::load(&format!("{p}.attention"), vb, config)?;
        let intermediate = BertIntermediate::load(&format!("{p}.intermediate"), vb, config)?;
        let output = BertOutput::load(&format!("{p}.output"), vb, config)?;
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
    fn load(p: &str, vb: &VarBuilder, config: &Config) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|index| {
                let p = format!("{p}.layer.{index}");
                BertLayer::load(&p, vb, config)
            })
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
    fn load(vb: &VarBuilder, config: &Config) -> Result<Self> {
        let embeddings = BertEmbeddings::load("embeddings", vb, config)?;
        let encoder = BertEncoder::load("encoder", vb, config)?;
        Ok(Self {
            embeddings,
            encoder,
        })
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let embedding_output = self.embeddings.forward(input_ids, token_type_ids)?;
        let sequence_output = self.encoder.forward(&embedding_output)?;
        Ok(sequence_output)
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    tokenizer_config: String,

    #[arg(long)]
    weights: String,
}

fn main() -> Result<()> {
    use tokenizers::Tokenizer;

    let args = Args::parse();
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_cuda(0)?
    };

    let mut tokenizer = Tokenizer::from_file(args.tokenizer_config).map_err(E::msg)?;
    let tokenizer = tokenizer.with_padding(None).with_truncation(None);

    let weights = unsafe { candle::safetensors::MmapedFile::new(args.weights)? };
    let weights = weights.deserialize()?;
    let vb = VarBuilder::from_safetensors(vec![weights], DTYPE, device.clone());
    let config = Config::all_mini_lm_l6_v2();
    let model = BertModel::load(&vb, &config)?;

    let tokens = tokenizer
        .encode("This is an example sentence", true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let token_ids = Tensor::new(&tokens[..], &device)?;
    println!("{token_ids}");
    let token_type_ids = token_ids.zeros_like()?;
    let ys = model.forward(&token_ids, &token_type_ids)?;
    println!("{ys}");
    Ok(())
}
