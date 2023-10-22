use candle::Var;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use candle::Tensor;
use candle_nn as nn;
use candle_nn::{linear, Dropout, Linear};
use std::borrow::Borrow;
use std::ops::Div;

use crate::layer_state::LayerState;
use crate::Config;
use candle::{DType, Device, Result};
use candle_nn::activation::Activation;
use candle_nn::{Embedding, Module, VarBuilder};
use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub struct BartAttention {
    num_heads: usize,
    head_dim: usize,
    dropout: Dropout,
    scaling: f64,
    encoder_decoder_attention: bool,
    output_attentions: bool,
    key: Linear,
    value: Linear,
    query: Linear,
    out_proj: Linear,
    store_cache: bool,
}

impl BartAttention {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let attention_head_size = config.num_hidden_layers / config.encoder_attention_heads;
        let all_head_size = config.encoder_attention_heads * config.encoder_ffn_dim;
        let dropout = Dropout::new(config.dropout);
        let hidden_size = config.num_hidden_layers;
        let embed_dim = config.d_model;

        let query = linear(embed_dim, embed_dim, vb.pp("query"))?;
        let value = linear(embed_dim, embed_dim, vb.pp("value"))?;
        let key = linear(embed_dim, embed_dim, vb.pp("key"))?;
        let out_proj = linear(embed_dim, embed_dim, vb.pp("out_proj"))?;

        let head_dim = embed_dim / config.encoder_attention_heads;
        let scaling = (head_dim as f64).powf(-0.5);

        Ok(Self {
            num_heads: config.decoder_attention_heads,
            head_dim: config.encoder_ffn_dim,
            dropout,
            scaling,
            encoder_decoder_attention: true,
            output_attentions: true,
            key,
            value,
            query,
            out_proj,
            store_cache: false,
        })
    }

    pub fn forward_t(
        &self,
        hidden_states: &Tensor,
        key_value_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        layer_state: Option<LayerState>,
        train: bool,
    ) -> Result<(Tensor, Option<Tensor>, Option<LayerState>)> {
        let (bs, target_length, embed_dim) = hidden_states.shape().dims3().unwrap();

        let query_states = (hidden_states.apply(&self.query).unwrap() * self.scaling).unwrap();

        let (key_states, value_states) = if self.encoder_decoder_attention {
            if let Some(layer_state_value) = layer_state {
                (layer_state_value.prev_key, layer_state_value.prev_value)
            } else {
                (
                    key_value_states
                        .unwrap()
                        .apply(&self.key)?
                        .reshape((bs, (), self.num_heads, self.head_dim))?
                        .transpose(1, 2)?
                        .contiguous()?,
                    key_value_states
                        .unwrap()
                        .apply(&self.value)?
                        .reshape((bs, (), self.num_heads, self.head_dim))?
                        .transpose(1, 2)?
                        .contiguous()?,
                )
            }
        } else if let Some(layer_state_value) = layer_state {
            let key_states = hidden_states
                .apply(&self.key)
                .unwrap()
                .reshape((bs, (), self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;

            let value_states = hidden_states
                .apply(&self.value)
                .unwrap()
                .reshape((bs, (), self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;

            (
                Tensor::cat(&[layer_state_value.prev_key, key_states], 2).unwrap(),
                Tensor::cat(&[layer_state_value.prev_value, value_states], 2).unwrap(),
            )
        } else {
            (
                hidden_states
                    .apply(&self.key)?
                    .reshape((bs, (), self.num_heads, self.head_dim))?
                    .transpose(1, 2)?
                    .contiguous()?,
                hidden_states
                    .apply(&self.value)?
                    .reshape((bs, (), self.num_heads, self.head_dim))?
                    .transpose(1, 2)?
                    .contiguous()?,
            )
        };

        let new_layer_state = if self.store_cache {
            Some(LayerState {
                prev_key: key_states.copy().unwrap(),
                prev_value: value_states.copy().unwrap(),
            })
        } else {
            None
        };

        let query_states = query_states
            .reshape((bs, target_length, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((bs * self.num_heads, (), self.head_dim))?;

        let key_states = key_states.reshape((bs * self.num_heads, (), self.head_dim))?;
        let value_states = value_states.reshape((bs * self.num_heads, (), self.head_dim))?;

        let source_length = key_states.shape().dims1()?;

        let mut attention_weights = query_states.broadcast_matmul(&key_states.transpose(1, 2)?)?;

        if let Some(attention_mask_value) = attention_mask {
            attention_weights = attention_weights
                .reshape((bs, self.num_heads, target_length, source_length))?
                .add(attention_mask_value)?;
            attention_weights =
                attention_weights.reshape((bs * self.num_heads, target_length, source_length))?;
        };

        let attention_weights = candle_nn::ops::softmax_last_dim(&attention_weights)?;

        let saved_attention_weights = if self.output_attentions {
            Some(attention_weights.reshape((bs, self.num_heads, target_length, source_length))?)
        } else {
            None
        };

        let attention_probas = self.dropout.forward(&attention_weights, train)?;

        let attention_output = attention_probas
            .broadcast_matmul(&value_states)?
            .reshape((bs, self.num_heads, target_length, self.head_dim))?
            .transpose(1, 2)?
            .reshape((bs, target_length, embed_dim))?
            .apply(&self.out_proj)?;

        Ok((attention_output, saved_attention_weights, new_layer_state))
    }
}

// #[derive(Debug)]
// /// # Cache for BART attention layers
// /// Stores the cached value of key, value and key padding mask to avoid recalculation (e.g. at each generation step)
// pub struct LayerState {
//     /// Cached keys
//     pub prev_key: Tensor,
//     /// Cached values
//     pub prev_value: Tensor,
// }

// impl Clone for LayerState {
//     fn clone(&self) -> Self {
//         LayerState {
//             prev_key: self.prev_key.copy().unwrap(),
//             prev_value: self.prev_value.copy().unwrap(),
//         }
//     }
// }

// impl LayerState {
//     pub(crate) fn reorder_cache(&mut self, new_indices: &Tensor) {
//         self.prev_key = self.prev_key.index_select(new_indices, 0).unwrap();
//         self.prev_value = self.prev_value.index_select(new_indices, 0).unwrap();
//     }
// }
