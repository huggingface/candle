use crate::bart_decoder::masked_fill;
use crate::layer_state::LayerState;
use candle::Error;
use candle::Tensor;
use candle::Var;
use candle::{DType, Result};
use candle_nn as nn;
use candle_nn::activation::Activation;
use candle_nn::embedding;
use candle_nn::{linear, Dropout, Linear};
use candle_nn::{Embedding, Module, VarBuilder};
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::collections::HashMap;
use std::ops::Div;
use std::sync::Arc;
use std::sync::Mutex;

use crate::bart_attention::BartAttention;
// use crate::bart_encoder::nn::LayerNorm;
use crate::{Config, DEVICE};
use candle_nn::LayerNorm;

pub struct EncoderLayer {
    self_attention: BartAttention,
    self_attention_layer_norm: nn::LayerNorm,
    dropout: Dropout,
    activation_dropout: Dropout,
    activation: Activation,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: nn::LayerNorm,
}

fn layer_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<LayerNorm> {
    let (weight, bias) = match (vb.get(size, "weight"), vb.get(size, "bias")) {
        (Ok(weight), Ok(bias)) => (weight, bias),
        (Err(err), _) | (_, Err(err)) => {
            if let (Ok(weight), Ok(bias)) = (vb.get(size, "gamma"), vb.get(size, "beta")) {
                (weight, bias)
            } else {
                return Err(err);
            }
        }
    };
    Ok(LayerNorm::new(weight, bias, eps))
}

impl EncoderLayer {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let layer_norm_config = nn::LayerNormConfig {
            eps: 1e-5,
            ..Default::default()
        };
        let output_attention = config.output_attentions.unwrap_or(false);

        let self_attention = BartAttention::load(vb.pp("self-attn"), config)?;

        let self_attention_layer_norm = nn::layer_norm(
            config.d_model,
            layer_norm_config,
            vb.pp("self_attn_layer_norm"),
        )?;
        let dropout = Dropout::new(config.dropout);
        let activation_dropout = Dropout::new(config.activation_dropout);
        let activation = match &config.activation_function {
            Some(act_function) => *act_function,
            None => Activation::Gelu,
        };

        let fc1 = nn::linear(config.d_model, config.encoder_ffn_dim, vb.pp("fc1"))?;
        let fc2 = nn::linear(config.encoder_ffn_dim, config.d_model, vb.pp("fc2"))?;

        let final_layer_norm =
            nn::layer_norm(config.d_model, layer_norm_config, vb.pp("final_layer_norm"))?;

        Ok(EncoderLayer {
            self_attention,
            self_attention_layer_norm,
            dropout,
            activation_dropout,
            activation,
            fc1,
            fc2,
            final_layer_norm,
        })
    }

    pub fn forward_t(
        &self,
        x: &Tensor,
        encoder_attention_mask: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (output, attention_weights, _) =
            self.self_attention
                .forward_t(x, None, encoder_attention_mask, None, train)?;

        let output = (self.dropout.forward(&output, train)? + x)?;
        let output = output.apply(&self.self_attention_layer_norm)?;

        let residual = output.copy()?;
        let output = self.activation.forward(&output.apply(&self.fc1)?)?;

        let output = self.dropout.forward(
            &self
                .activation_dropout
                .forward(&output, train)?
                .apply(&self.fc2)?,
            train,
        )?;

        let output = (output + residual)?;
        Ok((output.apply(&self.final_layer_norm)?, attention_weights))
    }
}

pub struct BartEncoder {
    dropout: Dropout,
    layer_norm_embedding: Option<nn::LayerNorm>,
    layers: Vec<EncoderLayer>,
    embed_positions: EmbeddingOption,
    output_attentions: bool,
    output_hidden_states: bool,
    scale_embedding: f64,
}

impl BartEncoder {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let output_attentions = config.output_attentions.unwrap_or(false);
        let output_hidden_states = config.output_hidden_states.unwrap_or(false);
        let normalize_embedding = config.normalize_embedding.unwrap_or(true);
        let static_position_embeddings = config.static_position_embeddings.unwrap_or(false);
        let scale_embedding = match config.scale_embedding {
            Some(value) => {
                if value {
                    (config.d_model as f64).sqrt()
                } else {
                    1.0
                }
            }
            None => 1.0,
        };

        let dropout = Dropout::new(config.dropout);

        let layer_norm_embedding = if normalize_embedding {
            let layer_norm_config = nn::LayerNormConfig {
                eps: 1e-5,
                ..Default::default()
            };

            Some(nn::layer_norm(
                config.d_model,
                layer_norm_config,
                vb.pp("layernorm_embedding"),
            )?)
        } else {
            None
        };

        let embed_positions = if static_position_embeddings {
            EmbeddingOption::SinusoidalPositionalEmbedding(SinusoidalPositionalEmbedding::load(
                vb.pp("embed_positions"),
                config,
            )?)
        } else {
            EmbeddingOption::LearnedPositionalEmbedding(LearnedPositionalEmbedding::load(
                vb.pp("embed_positions"),
                config,
            )?)
        };

        let mut layers: Vec<EncoderLayer> = vec![];
        let p_layers = vb.pp("layers");
        for layer_index in 0..config.encoder_layers {
            layers.push(EncoderLayer::load(vb.pp(layer_index), config)?);
        }

        Ok(BartEncoder {
            dropout,
            layer_norm_embedding,
            layers,
            embed_positions,
            output_attentions,
            output_hidden_states,
            scale_embedding,
        })
    }

    pub fn forward_t(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        embeddings: &nn::Embedding,
        train: bool,
    ) -> Result<BartEncoderOutput> {
        let x = (input_ids.apply(embeddings)? * self.scale_embedding)?;

        let x = (x + &self.embed_positions.forward(input_ids, 0)?)?;
        let x = if let Some(layer_norm_embedding) = &self.layer_norm_embedding {
            x.apply(layer_norm_embedding)?
        } else {
            x
        };
        let attention_mask = attention_mask
            .map(|mask| _expand_mask(mask, None, mask.dtype()))
            .unwrap()?;
        // let attention_mask = attention_mask.map(|mask| _expand_mask(mask, None, x.kind()));
        let mut hidden_state = self.dropout.forward(&x, train)?;

        let mut all_hidden_states: Option<Vec<Tensor>> = if self.output_hidden_states {
            Some(vec![])
        } else {
            None
        };
        let mut all_attentions: Option<Vec<Tensor>> = if self.output_attentions {
            Some(vec![])
        } else {
            None
        };

        let mut attention_weights: Option<Tensor>;

        for layer in &self.layers {
            let (hidden_state, attention_weights) =
                layer.forward_t(&hidden_state, Some(&attention_mask), train)?;

            if let Some(ref mut attentions) = all_attentions {
                if let Some(weights) = attention_weights {
                    attentions.push(weights);
                }
            };
            if let Some(ref mut hidden_states) = all_hidden_states {
                hidden_states.push(hidden_state.copy()?);
            };
        }

        Ok(BartEncoderOutput {
            hidden_state,
            all_hidden_states,
            all_attentions,
        })
    }
}

pub struct BartEncoderOutput {
    /// Last encoder layer hidden state
    pub hidden_state: Tensor,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}

pub enum EmbeddingOption {
    /// PositionalEmbedding
    LearnedPositionalEmbedding(LearnedPositionalEmbedding),
    SinusoidalPositionalEmbedding(SinusoidalPositionalEmbedding),
}

impl EmbeddingOption {
    pub fn forward(&self, input: &Tensor, past_key_values_length: usize) -> Result<Tensor> {
        match self {
            EmbeddingOption::LearnedPositionalEmbedding(embedding) => {
                embedding.forward(input, past_key_values_length)
            }
            EmbeddingOption::SinusoidalPositionalEmbedding(embedding) => {
                embedding.forward(input, past_key_values_length)
            }
        }
    }
}

#[derive(Debug)]
pub struct LearnedPositionalEmbedding {
    embedding: nn::Embedding,
    offset: usize,
}

impl LearnedPositionalEmbedding {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let offset = 2;

        let num_embeddings = config.max_position_embeddings + offset;

        let embedding = embedding(
            config.max_position_embeddings,
            config.max_position_embeddings,
            vb,
        )?;

        Ok(LearnedPositionalEmbedding { embedding, offset })
    }

    pub fn forward(&self, input: &Tensor, past_key_values_length: usize) -> Result<Tensor> {
        let (_, sequence_length) = input.dims2()?;
        let end = past_key_values_length + sequence_length;
        let positions = Tensor::arange(sequence_length as i64, end as i64, &DEVICE)?;
        positions.apply(&self.embedding)
    }
}

#[derive(Debug)]
pub struct SinusoidalPositionalEmbedding {
    embedding: nn::Embedding,
}

impl SinusoidalPositionalEmbedding {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let embedding = embedding(
            config.max_position_embeddings,
            config.max_position_embeddings,
            vb,
        )?;

        Ok(SinusoidalPositionalEmbedding { embedding })
    }

    pub fn forward(&self, input: &Tensor, past_key_values_length: usize) -> Result<Tensor> {
        let (_, sequence_length) = input.dims2()?;
        let (_, check, _) = input.dims3()?;
        assert_eq!(sequence_length, check);

        let end = past_key_values_length + sequence_length;
        let positions = Tensor::arange(sequence_length as i64, end as i64, DEVICE)?;
        positions.apply(&self.embedding)
    }
}

pub(crate) fn _expand_mask(
    mask: &Tensor,
    target_length: Option<usize>,
    dtype: DType,
) -> Result<Tensor> {
    let (batch_size, source_length) = mask.dims2()?;
    let target_length = target_length.unwrap_or(source_length);
    let expanded_mask = mask
        .unsqueeze(1)?
        .unsqueeze(1)?
        .expand((batch_size, 1, target_length, source_length))?
        .to_dtype(dtype)?;

    let ones = Tensor::ones((batch_size, 1, target_length, source_length), dtype, DEVICE)?;
    let zeros = Tensor::zeros((batch_size, 1, target_length, source_length), dtype, DEVICE)?;

    let inverted_mask = (ones - expanded_mask)?;
    let bool_mask = inverted_mask.ge(0i64)?;

    masked_fill(&inverted_mask, &bool_mask, &zeros)
    // inverted_mask.masked_fill(&inverted_mask.to_kind(Kind::Bool), get_min(dtype).unwrap())
}
