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
use candle_nn::{Module, VarBuilder};
use serde::{Deserialize, Serialize};

use crate::bart_attention::BartAttention;
use crate::bart_encoder::EmbeddingOption;
use crate::bart_encoder::_expand_mask;
use crate::bart_encoder::{LearnedPositionalEmbedding, SinusoidalPositionalEmbedding};
use crate::layer_state::LayerState;
use crate::{Config, DEVICE};
use candle::Device;
use candle_nn::LayerNorm;

#[derive(Debug, Clone, Copy)]
pub struct EmbeddingConfig {
    pub sparse: bool,
    pub scale_grad_by_freq: bool,
    // pub ws_init: super::Init,
    pub padding_idx: i64,
}

// #[derive(Debug)]
// pub struct Embedding {
//     pub ws: Tensor,
//     config: EmbeddingConfig,
// }

// pub fn embedding<'a, T: Borrow<super::Path<'a>>>(
//     vs: T,
//     num_embeddings: i64,
//     embedding_dim: i64,
//     config: EmbeddingConfig,
// ) -> Embedding {
//     let vs = vs.borrow();
//     Embedding {
//         ws: vs.var("weight", &[num_embeddings, embedding_dim], config.ws_init),
//         config,
//     }
// }

// impl super::module::Module for Embedding {
//     fn forward(&self, xs: &Tensor) -> Tensor {
//         Tensor::embedding(
//             &self.ws,
//             xs,
//             self.config.padding_idx,
//             self.config.scale_grad_by_freq,
//             self.config.sparse,
//         )
//     }
// }
