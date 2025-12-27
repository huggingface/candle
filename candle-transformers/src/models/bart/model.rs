use candle::Result;
use candle_nn::{embedding, Embedding, VarBuilder};

use crate::models::bart::{config::BartConfig, decode::BartDecoder, encode::BartEncoder};

/// Full BART encoder-decoder model with shared embeddings.
#[derive(Debug, Clone)]
pub struct BartModel {
    pub shared: Embedding,
    encoder: BartEncoder,
    decoder: BartDecoder,
}

impl BartModel {
    /// Load full BART encoder-decoder model.
    /// Handles two weight formats:
    /// - model.shared (older format)
    /// - model.decoder.embed_tokens (weight-tied format, e.g., bart-large-cnn)
    pub fn new(cfg: &BartConfig, vb: VarBuilder) -> Result<Self> {
        // Try to load shared embedding from model.shared first
        // If not found, fall back to model.decoder.embed_tokens (weight-tied format)
        let shared = if vb.pp("model").pp("shared").contains_tensor("weight") {
            embedding(cfg.vocab_size, cfg.d_model, vb.pp("model").pp("shared"))?
        } else {
            // Weight-tied format: shared embedding is stored in decoder.embed_tokens
            embedding(
                cfg.vocab_size,
                cfg.d_model,
                vb.pp("model").pp("decoder").pp("embed_tokens"),
            )?
        };

        // Pass cloned embedding to encoder and decoder
        // Tensor clone is cheap (Arc underneath) - ensures true weight sharing
        let encoder = BartEncoder::new_with_shared_embeddings(
            cfg,
            shared.clone(),
            vb.pp("model").pp("encoder"),
        )?;

        let decoder = BartDecoder::new_with_shared_embeddings(
            cfg,
            shared.clone(),
            vb.pp("model").pp("decoder"),
        )?;

        Ok(Self {
            shared,
            encoder,
            decoder,
        })
    }

    pub fn shared(&self) -> &Embedding {
        &self.shared
    }

    pub fn encoder(&self) -> &BartEncoder {
        &self.encoder
    }

    pub fn encoder_mut(&mut self) -> &mut BartEncoder {
        &mut self.encoder
    }

    pub fn decoder(&self) -> &BartDecoder {
        &self.decoder
    }

    pub fn decoder_mut(&mut self) -> &mut BartDecoder {
        &mut self.decoder
    }
}
