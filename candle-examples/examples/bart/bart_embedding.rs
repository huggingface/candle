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
