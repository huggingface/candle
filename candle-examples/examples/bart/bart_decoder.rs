use candle::Var;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use candle::Tensor;
use candle_nn as nn;
use candle_nn::{linear, Dropout, Linear};
use std::borrow::Borrow;
use std::ops::Div;

use candle::{DType, Result};
use candle_nn::activation::Activation;
use candle_nn::embedding;
use candle_nn::{Embedding, Module, VarBuilder};
use serde::{Deserialize, Serialize};

use crate::bart_attention::BartAttention;
use crate::bart_encoder::EmbeddingOption;
use crate::bart_encoder::_expand_mask;
use crate::bart_encoder::{LearnedPositionalEmbedding, SinusoidalPositionalEmbedding};
use crate::layer_state::LayerState;
use crate::{Config, DEVICE};
use candle::Device;
use candle_nn::LayerNorm;

pub struct DecoderLayer {
    self_attention: BartAttention,
    encoder_attention: BartAttention,
    self_attention_layer_norm: LayerNorm,
    encoder_attention_layer_norm: LayerNorm,
    dropout: Dropout,
    activation_dropout: Dropout,
    activation: Activation,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
}

impl DecoderLayer {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        // pub fn new<'p, P>(p: P, config: &BartConfig) -> DecoderLayer

        let layer_norm_config = nn::LayerNormConfig {
            eps: 1e-5,
            ..Default::default()
        };
        let output_attention = config.output_attentions.unwrap_or(false);
        let self_attention = BartAttention::load(vb.pp("self_attn"), config)?;
        let encoder_attention = BartAttention::load(vb.pp("encoder_attn"), config)?;

        let self_attention_layer_norm = nn::layer_norm(
            config.d_model,
            layer_norm_config,
            vb.pp("self_attn_layer_norm"),
        )?;

        let encoder_attention_layer_norm = nn::layer_norm(
            config.d_model,
            layer_norm_config,
            vb.pp("encoder_attn_layer_norm"),
        )?;

        let dropout = Dropout::new(config.dropout);
        let activation_dropout = Dropout::new(config.activation_dropout);
        let activation = match &config.activation_function {
            Some(act_function) => *act_function,
            None => Activation::Gelu,
        };

        let fc1 = nn::linear(config.d_model, config.decoder_ffn_dim, vb.pp("fc1"))?;
        let fc2 = nn::linear(config.decoder_ffn_dim, config.d_model, vb.pp("fc2"))?;
        let final_layer_norm =
            nn::layer_norm(config.d_model, layer_norm_config, vb.pp("final_layer_norm"))?;

        Ok(DecoderLayer {
            self_attention,
            encoder_attention,
            self_attention_layer_norm,
            encoder_attention_layer_norm,
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
        encoder_hidden_states: &Tensor,
        encoder_attention_mask: Option<&Tensor>,
        decoder_attention_mask: Option<&Tensor>,
        layer_states: (Option<LayerState>, Option<LayerState>),
        train: bool,
    ) -> Result<(
        Tensor,
        Option<Tensor>,
        (Option<LayerState>, Option<LayerState>),
    )> {
        let (output, attention_weights, new_self_layer_states) = self.self_attention.forward_t(
            x,
            None,
            decoder_attention_mask,
            layer_states.0,
            train,
        )?;

        let output = (self.dropout.forward(&output, train)? + x)?;
        let output = output.apply(&self.self_attention_layer_norm)?;

        let (output1, _, new_encoder_layer_states) = self.encoder_attention.forward_t(
            &output,
            Some(encoder_hidden_states),
            encoder_attention_mask,
            layer_states.1,
            train,
        )?;
        let output1 = (self.dropout.forward(&output1, train)? + output)?;

        let output1 = output1.apply(&self.encoder_attention_layer_norm)?;
        let output2 = self.activation.forward(&output1.apply(&self.fc1)?)?;

        let output2 = self.dropout.forward(
            &self
                .activation_dropout
                .forward(&output2, train)?
                .apply(&self.fc2)?,
            train,
        )?;

        let output2: Tensor = (output2 + output1)?;
        Ok((
            output2.apply(&self.final_layer_norm)?,
            attention_weights,
            (new_self_layer_states, new_encoder_layer_states),
        ))
    }
}

pub struct BartDecoder {
    dropout: Dropout,
    layer_norm_embedding: Option<LayerNorm>,
    layers: Vec<DecoderLayer>,
    embed_positions: EmbeddingOption,
    output_attentions: bool,
    output_hidden_states: bool,
    output_past: bool,
    scale_embedding: f64,
}

impl BartDecoder {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let output_past = config.output_past.unwrap_or(true);
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

        let mut layers: Vec<DecoderLayer> = vec![];
        let p_layers = vb.pp("layers");
        for layer_index in 0..config.encoder_layers {
            layers.push(DecoderLayer::load(
                vb.pp(format!("layers.{layer_index}")),
                config,
            )?);
        }

        Ok(BartDecoder {
            dropout,
            layer_norm_embedding,
            layers,
            embed_positions,
            output_attentions,
            output_hidden_states,
            output_past,
            scale_embedding,
        })
    }

    pub fn forward_t(
        &self,
        input_ids: &Tensor,
        encoder_hidden_states: &Tensor,
        encoder_attention_mask: Option<&Tensor>,
        decoder_attention_mask: Option<&Tensor>,
        embeddings: &nn::Embedding,
        old_layer_states: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
        train: bool,
    ) -> Result<BartDecoderOutput> {
        let past_key_values_length = if let Some(old_layer_states_values) = &old_layer_states {
            if let Some(old_value_state) = &old_layer_states_values[0].0 {
                old_value_state.prev_key.dim(2)?
            } else {
                0
            }
        } else {
            0
        };

        let positions = self
            .embed_positions
            .forward(input_ids, past_key_values_length)?;

        let x: Tensor = ((input_ids.apply(embeddings)? * self.scale_embedding)? + positions)?;

        let decoder_attention_mask = _prepare_decoder_attention_mask(
            decoder_attention_mask,
            input_ids.dims(),
            &x,
            past_key_values_length,
        );

        let encoder_attention_mask = encoder_attention_mask
            .map(|mask| _expand_mask(mask, Some(*input_ids.dims().last().unwrap()), x.dtype()))
            .transpose()?;

        let x = if let Some(layer_norm_embedding) = &self.layer_norm_embedding {
            x.apply(layer_norm_embedding)?
        } else {
            x
        };

        let mut hidden_state = self.dropout.forward(&x, train)?;
        // let mut hidden_state = x.apply_t(&self.dropout, train);
        let mut all_hidden_states: Option<Vec<Tensor>> = if self.output_hidden_states {
            Some(Vec::with_capacity(self.layers.len()))
        } else {
            None
        };
        let mut all_attentions: Option<Vec<Tensor>> = if self.output_attentions {
            Some(Vec::with_capacity(self.layers.len()))
        } else {
            None
        };
        let mut next_decoder_cache: Option<Vec<(Option<LayerState>, Option<LayerState>)>> =
            if self.output_past {
                if old_layer_states.is_some() {
                    old_layer_states
                } else {
                    Some(vec![(None, None); self.layers.len()])
                }
            } else {
                None
            };

        let mut attention_weights: Option<Tensor>;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let layer_state = match &next_decoder_cache {
                Some(values) => values[layer_idx].to_owned(),
                None => (None, None),
            };
            let (hidden_state, attention_weights, last) = layer.forward_t(
                &hidden_state,
                encoder_hidden_states,
                encoder_attention_mask.as_ref(),
                decoder_attention_mask.as_ref(),
                layer_state,
                train,
            )?;

            if let Some(ref mut hidden_states) = all_hidden_states {
                hidden_states.push(hidden_state.as_ref().copy()?);
            };
            if let Some(ref mut attentions) = all_attentions {
                if let Some(weight) = attention_weights {
                    attentions.push(weight);
                }
            };
            if let Some(value) = &mut next_decoder_cache {
                value[layer_idx] = last
            };
        }

        Ok(BartDecoderOutput {
            hidden_state,
            encoder_attention_mask,
            next_decoder_cache,
            all_hidden_states,
            all_attentions,
        })
    }
}

pub struct BartDecoderOutput {
    /// last decoder layer hidden state
    pub hidden_state: Tensor,
    /// Padding mask for the encoder positions to attend to
    pub encoder_attention_mask: Option<Tensor>,
    /// Cached outputs of the model (attention layers keys and values) if the model is used for generation
    pub next_decoder_cache: Option<Vec<(Option<LayerState>, Option<LayerState>)>>,
    /// Hidden states for all intermediate layers
    pub all_hidden_states: Option<Vec<Tensor>>,
    /// Attention weights for all intermediate layers
    pub all_attentions: Option<Vec<Tensor>>,
}

pub(crate) fn _prepare_decoder_attention_mask(
    attention_mask: Option<&Tensor>,
    input_shape: &[usize],
    input_embeds: &Tensor,
    past_key_values_length: usize,
) -> Option<Tensor> {
    let last_input_shape_dim = *input_shape.last().unwrap();
    let mut combined_attention_mask = if last_input_shape_dim > 1 {
        Some(
            _make_causal_mask(
                input_shape,
                input_embeds.dtype(),
                DEVICE,
                past_key_values_length,
            )
            .unwrap(),
        )
    } else {
        None
    };

    if let Some(attention_mask) = attention_mask {
        let expanded_attention_mask = _expand_mask(
            attention_mask,
            Some(last_input_shape_dim),
            input_embeds.dtype(),
        )
        .unwrap();
        combined_attention_mask = match combined_attention_mask {
            Some(value) => Some((value + expanded_attention_mask).unwrap()),
            None => Some(expanded_attention_mask),
        };
    }

    combined_attention_mask
}

pub(crate) fn _make_causal_mask(
    input_ids_shape: &[usize],
    dtype: DType,
    device: &Device,
    past_key_values_length: usize,
) -> Result<Tensor> {
    let batch_size = input_ids_shape[0];
    let target_length = input_ids_shape[1];

    let mut mask = Tensor::zeros((target_length, target_length), dtype, DEVICE)?;
    // let mut mask = Tensor::full(
    //     [target_length, target_length],
    //     get_min(dtype).unwrap(),
    //     (dtype, device),
    // );

    let mask_cond = Tensor::arange(0i64, target_length as i64, DEVICE)?;
    // let mask_cond = Tensor::arange(target_length, (dtype, device));

    mask = masked_fill(
        &mask_cond.lt(&(&mask_cond + 1f64)?.reshape((target_length, 1))?)?,
        &mask,
        &mask,
    )?;

    // let _ = mask.masked_fill_(
    //     &mask_cond.lt_tensor(&(&mask_cond + 1).view([target_length, 1])),
    //     0,
    // );

    if past_key_values_length > 0 {
        mask = Tensor::cat(
            &[
                Tensor::zeros((target_length, past_key_values_length), dtype, DEVICE)?,
                mask,
            ],
            3,
        )?;
    }
    // if past_key_values_length > 0 {
    //     mask = Tensor::cat(
    //         &[
    //             Tensor::zeros([target_length, past_key_values_length], (dtype, device)),
    //             mask,
    //         ],
    //         -1,
    //     );
    // }

    mask.unsqueeze(0)?.unsqueeze(0)?.expand((
        batch_size,
        1,
        target_length,
        target_length + past_key_values_length,
    ))

    // mask.unsqueeze(0).unsqueeze(0).expand(
    //     [
    //         batch_size,
    //         1,
    //         target_length,
    //         target_length + past_key_values_length,
    //     ],
    //     true,
    // )
}

pub fn masked_fill(in_tensor: &Tensor, mask: &Tensor, value: &Tensor) -> Result<Tensor> {
    mask.where_cond(value, in_tensor)
}
