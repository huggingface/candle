use candle::{DType, Device, Result, Tensor};
use candle_nn as nn;
use candle_nn::{Embedding, Module, VarBuilder};
use serde::Deserialize;

use crate::bart_config::Config;
use crate::bart_decoder::BartDecoder;
use crate::bart_embedding::EmbeddingConfig;
use crate::bart_encoder::BartEncoder;
use candle_nn::embedding;
pub const DTYPE: DType = DType::F32;

pub struct BartModel {
    pub(crate) encoder: BartEncoder,
    decoder: BartDecoder,
    pub(crate) embeddings: nn::Embedding,
    pad_token_id: usize,
}
impl BartModel {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let pad_token_id = config.pad_token_id.unwrap_or(1);
        // let embedding_config = EmbeddingConfig {
        //     padding_idx: pad_token_id,
        //     ..Default::default()
        // };
        // let embeddings = embedding(config.vocab_size, config.d_model, vb.pp("embeddings"))?;
        let embeddings = embedding(config.vocab_size, config.d_model, vb.pp("model.shared"))?;

        let encoder = BartEncoder::load(vb.pp("model.encoder"), config)?;
        let decoder = BartDecoder::load(vb.pp("model.decoder"), config)?;

        Ok(BartModel {
            encoder,
            decoder,
            embeddings,
            pad_token_id,
        })
    }
}
