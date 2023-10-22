// Copyright 2020 The Facebook AI Research Team Authors
// Copyright 2020-present, the HuggingFace Inc. team.
// Copyright 2020 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]
use anyhow::{Error, Result};
use candle::{Device, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use crate::bart::BartModel;
use crate::bart_config::Config;

pub const DTYPE: DType = DType::F32;

mod bart;
mod bart_attention;
mod bart_config;
mod bart_decoder;
mod bart_embedding;
mod bart_encoder;
mod layer_state;

const DEVICE: &Device = &Device::Cpu;

fn main() -> Result<(), Error> {
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&["models/bart/model.safetensors"], DTYPE, DEVICE)
            .unwrap()
    };

    let config: String = std::fs::read_to_string("models/bart/config.json")?;
    let config: Config = serde_json::from_str(&config)?;

    let model = BartModel::load(vb, &config)?;

    Ok(())
}
