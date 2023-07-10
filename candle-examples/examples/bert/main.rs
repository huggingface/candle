#![allow(dead_code)]

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{anyhow, Error as E, Result};
use candle::{DType, Device, Tensor};
use candle_hub::{api::sync::Api, Cache, Repo, RepoType};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder};
use clap::Parser;
use serde::Deserialize;
use tokenizers::{PaddingParams, Tokenizer};

const DTYPE: DType = DType::F32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum HiddenAct {
    Gelu,
    Relu,
}

impl HiddenAct {
    fn forward(&self, xs: &Tensor) -> candle::Result<Tensor> {
        match self {
            // TODO: The all-MiniLM-L6-v2 model uses "gelu" whereas this is "gelu_new", this explains some
            // small numerical difference.
            // https://github.com/huggingface/transformers/blob/cd4584e3c809bb9e1392ccd3fe38b40daba5519a/src/transformers/activations.py#L213
            Self::Gelu => xs.gelu(),
            Self::Relu => xs.relu(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
enum PositionEmbeddingType {
    #[default]
    Absolute,
}

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/configuration_bert.py#L1
#[derive(Debug, Clone, PartialEq, Deserialize)]
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
    #[serde(default)]
    position_embedding_type: PositionEmbeddingType,
    #[serde(default)]
    use_cache: bool,
    classifier_dropout: Option<f64>,
    model_type: Option<String>,
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
            model_type: Some("bert".to_string()),
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
            model_type: Some("bert".to_string()),
        }
    }
}

fn embedding(vocab_size: usize, hidden_size: usize, p: &str, vb: &VarBuilder) -> Result<Embedding> {
    let embeddings = vb.get((vocab_size, hidden_size), &format!("{p}.weight"))?;
    Ok(Embedding::new(embeddings, hidden_size))
}

fn linear(size1: usize, size2: usize, p: &str, vb: &VarBuilder) -> Result<Linear> {
    let weight = vb.get((size2, size1), &format!("{p}.weight"))?;
    let bias = vb.get(size2, &format!("{p}.bias"))?;
    Ok(Linear::new(weight, Some(bias)))
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

fn layer_norm(size: usize, eps: f64, p: &str, vb: &VarBuilder) -> Result<LayerNorm> {
    let (weight, bias) = match (
        vb.get(size, &format!("{p}.weight")),
        vb.get(size, &format!("{p}.bias")),
    ) {
        (Ok(weight), Ok(bias)) => (weight, bias),
        (Err(err), _) | (_, Err(err)) => {
            if let (Ok(weight), Ok(bias)) = (
                vb.get(size, &format!("{p}.gamma")),
                vb.get(size, &format!("{p}.beta")),
            ) {
                (weight, bias)
            } else {
                return Err(err.into());
            }
        }
    };
    Ok(LayerNorm::new(weight, bias, eps))
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
        let word_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            &format!("{p}.word_embeddings"),
            vb,
        )?;
        let position_embeddings = embedding(
            config.max_position_embeddings,
            config.hidden_size,
            &format!("{p}.position_embeddings"),
            vb,
        )?;
        let token_type_embeddings = embedding(
            config.type_vocab_size,
            config.hidden_size,
            &format!("{p}.token_type_embeddings"),
            vb,
        )?;
        let layer_norm = layer_norm(
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
        let (_bsize, seq_len) = input_ids.shape().r2()?;
        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;
        let mut embeddings = (&input_embeddings + token_type_embeddings)?;
        if let Some(position_embeddings) = &self.position_embeddings {
            // TODO: Proper absolute positions?
            let position_ids = (0..seq_len as u32).collect::<Vec<_>>();
            let position_ids = Tensor::new(&position_ids[..], &input_ids.device())?;
            embeddings = embeddings.broadcast_add(&position_embeddings.forward(&position_ids)?)?
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
        let query = linear(hidden_size, all_head_size, &format!("{p}.query"), vb)?;
        let value = linear(hidden_size, all_head_size, &format!("{p}.value"), vb)?;
        let key = linear(hidden_size, all_head_size, &format!("{p}.key"), vb)?;
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
        let xs = xs.reshape(new_x_shape.as_slice())?.transpose(1, 2)?;
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
        let attention_probs = attention_scores.softmax(candle::D::Minus1)?;
        let attention_probs = self.dropout.forward(&attention_probs)?;

        let context_layer = attention_probs.matmul(&value_layer)?;
        let context_layer = context_layer.transpose(1, 2)?.contiguous()?;
        let context_layer = context_layer.flatten_from(candle::D::Minus2)?;
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
        let dense = linear(
            config.hidden_size,
            config.hidden_size,
            &format!("{p}.dense"),
            vb,
        )?;
        let layer_norm = layer_norm(
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
        Ok(self.layer_norm.forward(&(hidden_states + input_tensor)?)?)
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
        let dense = linear(
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
        let dense = linear(
            config.intermediate_size,
            config.hidden_size,
            &format!("{p}.dense"),
            vb,
        )?;
        let layer_norm = layer_norm(
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
        Ok(self.layer_norm.forward(&(hidden_states + input_tensor)?)?)
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
    device: Device,
}

impl BertModel {
    fn load(vb: &VarBuilder, config: &Config) -> Result<Self> {
        let (embeddings, encoder) = match (
            BertEmbeddings::load("embeddings", vb, config),
            BertEncoder::load("encoder", vb, config),
        ) {
            (Ok(embeddings), Ok(encoder)) => (embeddings, encoder),
            (Err(err), _) | (_, Err(err)) => {
                if let Some(model_type) = &config.model_type {
                    if let (Ok(embeddings), Ok(encoder)) = (
                        BertEmbeddings::load(&format!("{model_type}.embeddings"), vb, config),
                        BertEncoder::load(&format!("{model_type}.encoder"), vb, config),
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
            encoder,
            device: vb.device.clone(),
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

    /// Run offline (you must have the files already cached)
    #[arg(long)]
    offline: bool,

    /// The model to use, check out available models: https://huggingface.co/models?library=sentence-transformers&sort=trending
    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    /// When set, compute embeddings for this prompt.
    #[arg(long)]
    prompt: Option<String>,

    /// The number of times to run the prompt.
    #[arg(long, default_value = "1")]
    n: usize,
}

impl Args {
    fn build_model_and_tokenizer(&self) -> Result<(BertModel, Tokenizer)> {
        let device = if self.cpu {
            Device::Cpu
        } else {
            Device::new_cuda(0)?
        };
        let default_model = "sentence-transformers/all-MiniLM-L6-v2".to_string();
        let default_revision = "refs/pr/21".to_string();
        let (model_id, revision) = match (self.model_id.to_owned(), self.revision.to_owned()) {
            (Some(model_id), Some(revision)) => (model_id, revision),
            (Some(model_id), None) => (model_id, "main".to_string()),
            (None, Some(revision)) => (default_model, revision),
            (None, None) => (default_model, default_revision),
        };

        let repo = Repo::with_revision(model_id, RepoType::Model, revision);
        let (config_filename, tokenizer_filename, weights_filename) = if self.offline {
            let cache = Cache::default();
            (
                cache
                    .get(&repo, "config.json")
                    .ok_or(anyhow!("Missing config file in cache"))?,
                cache
                    .get(&repo, "tokenizer.json")
                    .ok_or(anyhow!("Missing tokenizer file in cache"))?,
                cache
                    .get(&repo, "model.safetensors")
                    .ok_or(anyhow!("Missing weights file in cache"))?,
            )
        } else {
            let api = Api::new()?;
            (
                api.get(&repo, "config.json")?,
                api.get(&repo, "tokenizer.json")?,
                api.get(&repo, "model.safetensors")?,
            )
        };
        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let weights = unsafe { candle::safetensors::MmapedFile::new(weights_filename)? };
        let weights = weights.deserialize()?;
        let vb = VarBuilder::from_safetensors(vec![weights], DTYPE, &device);
        let model = BertModel::load(&vb, &config)?;
        Ok((model, tokenizer))
    }
}

fn main() -> Result<()> {
    let start = std::time::Instant::now();

    let args = Args::parse();
    let (model, mut tokenizer) = args.build_model_and_tokenizer()?;
    let device = &model.device;

    if let Some(prompt) = args.prompt {
        let tokenizer = tokenizer.with_padding(None).with_truncation(None);
        let tokens = tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;
        println!("Loaded and encoded {:?}", start.elapsed());
        for _ in 0..args.n {
            let start = std::time::Instant::now();
            let _ys = model.forward(&token_ids, &token_type_ids)?;
            println!("Took {:?}", start.elapsed());
        }
    } else {
        let sentences = [
            "The cat sits outside",
            "A man is playing guitar",
            "I love pasta",
            "The new movie is awesome",
            "The cat plays in the garden",
            "A woman watches TV",
            "The new movie is so great",
            "Do you like pizza?",
        ];
        let n_sentences = sentences.len();
        if let Some(pp) = tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest
        } else {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
        }
        let tokens = tokenizer
            .encode_batch(sentences.to_vec(), true)
            .map_err(E::msg)?;
        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Ok(Tensor::new(tokens.as_slice(), device)?)
            })
            .collect::<Result<Vec<_>>>()?;

        let token_ids = Tensor::stack(&token_ids, 0)?;
        let token_type_ids = token_ids.zeros_like()?;
        println!("running inference on batch {:?}", token_ids.shape());
        let embeddings = model.forward(&token_ids, &token_type_ids)?;
        println!("generated embeddings {:?}", embeddings.shape());
        // Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
        let (_n_sentence, n_tokens, _hidden_size) = embeddings.shape().r3()?;
        let embeddings = (embeddings.sum(&[1])? / (n_tokens as f64))?.squeeze(1)?;
        println!("pooled embeddings {:?}", embeddings.shape());
        let mut similarities = vec![];
        for i in 0..n_sentences {
            let e_i = embeddings.get(i)?;
            for j in (i + 1)..n_sentences {
                let e_j = embeddings.get(j)?;
                let sum_ij = (&e_i * &e_j)?.sum_all()?.reshape(())?.to_scalar::<f32>()?;
                let sum_i2 = (&e_i * &e_i)?.sum_all()?.reshape(())?.to_scalar::<f32>()?;
                let sum_j2 = (&e_j * &e_j)?.sum_all()?.reshape(())?.to_scalar::<f32>()?;
                let cosine_similarity = sum_ij / (sum_i2 * sum_j2).sqrt();
                similarities.push((cosine_similarity, i, j))
            }
        }
        similarities.sort_by(|u, v| v.0.total_cmp(&u.0));
        for &(score, i, j) in similarities[..5].iter() {
            println!("score: {score:.2} '{}' '{}'", sentences[i], sentences[j])
        }
    }
    Ok(())
}
