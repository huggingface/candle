use candle::Var;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use candle::Tensor;
use candle_nn as nn;
use candle_nn::{linear, Dropout, Linear};
use std::borrow::Borrow;
use std::ops::Div;

use candle::{DType, Device, Result};
use candle_nn::activation::Activation;
use candle_nn::{Embedding, Module, VarBuilder};
use serde::{Deserialize, Serialize};

use crate::Config;

#[derive(Debug)]
/// # Cache for BART attention layers
/// Stores the cached value of key, value and key padding mask to avoid recalculation (e.g. at each generation step)
pub struct LayerState {
    /// Cached keys
    pub prev_key: Tensor,
    /// Cached values
    pub prev_value: Tensor,
}

impl Clone for LayerState {
    fn clone(&self) -> Self {
        LayerState {
            prev_key: self.prev_key.copy().unwrap(),
            prev_value: self.prev_value.copy().unwrap(),
        }
    }
}

impl LayerState {
    pub(crate) fn reorder_cache(&mut self, new_indices: &Tensor) {
        self.prev_key = self.prev_key.index_select(new_indices, 0).unwrap();
        self.prev_value = self.prev_value.index_select(new_indices, 0).unwrap();
    }
}
