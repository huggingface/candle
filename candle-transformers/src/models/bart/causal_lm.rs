use candle::{Device, Result, Tensor};
use candle_nn::{embedding, linear, Linear, VarBuilder};

use crate::models::bart::{
    config::{BartConfig, BartWeightPrefix},
    decode::BartDecoder,
};

/// BART decoder for causal language modeling (with LM head).
/// Use for VisionEncoderDecoder models (Donut, TrOCR) or decoder-only inference.
#[derive(Debug, Clone)]
pub struct BartForCausalLM {
    decoder: BartDecoder,
    lm_head: Linear,
}

impl BartForCausalLM {
    /// Load decoder for VisionEncoderDecoder models (Donut, TrOCR).
    /// Expects weights at: decoder.model.decoder.*, decoder.lm_head
    ///
    /// Handles weight tying where embed_tokens shares weights with lm_head.
    /// Some models store only lm_head.weight (not embed_tokens.weight).
    pub fn new(cfg: &BartConfig, vb: VarBuilder) -> Result<Self> {
        let decoder_vb = vb.pp("decoder.model.decoder");

        if cfg.tie_word_embeddings {
            // For tied embeddings, some models only store lm_head.weight.
            // Try embed_tokens first, fall back to lm_head.
            let embed_tokens = if decoder_vb.contains_tensor("embed_tokens.weight") {
                embedding(cfg.vocab_size, cfg.d_model, decoder_vb.pp("embed_tokens"))?
            } else {
                // Load from lm_head and transpose for embedding use
                embedding(cfg.vocab_size, cfg.d_model, vb.pp("decoder.lm_head"))?
            };
            let lm_head = candle_nn::Linear::new(embed_tokens.embeddings().clone(), None);
            let decoder = BartDecoder::new_with_shared_embeddings(cfg, embed_tokens, decoder_vb)?;
            Ok(Self { decoder, lm_head })
        } else {
            let decoder = BartDecoder::new_internal(cfg, decoder_vb, None)?;
            let lm_head = linear(cfg.d_model, cfg.vocab_size, vb.pp("decoder.lm_head"))?;
            Ok(Self { decoder, lm_head })
        }
    }

    /// Load decoder from full BART/mBART checkpoint.
    /// Expects weights at: model.decoder.*, model.shared (for weight tying)
    pub fn new_from_full_model(cfg: &BartConfig, vb: VarBuilder) -> Result<Self> {
        let decoder = BartDecoder::new_internal(cfg, vb.pp("model").pp("decoder"), None)?;

        let lm_head = if cfg.tie_word_embeddings {
            // Load shared embedding from model.shared for weight tying
            let shared = embedding(cfg.vocab_size, cfg.d_model, vb.pp("model").pp("shared"))?;
            candle_nn::Linear::new(shared.embeddings().clone(), None)
        } else {
            linear(cfg.d_model, cfg.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self { decoder, lm_head })
    }

    /// Load decoder with automatic prefix detection.
    pub fn new_with_prefix(
        cfg: &BartConfig,
        vb: VarBuilder,
        prefix: BartWeightPrefix,
    ) -> Result<Self> {
        let decoder_vb = vb.pp(prefix.decoder_prefix());
        let decoder = BartDecoder::new_internal(cfg, decoder_vb, None)?;

        let lm_head = if cfg.tie_word_embeddings {
            candle_nn::Linear::new(decoder.embed_tokens().embeddings().clone(), None)
        } else {
            linear(cfg.d_model, cfg.vocab_size, vb.pp(prefix.lm_head_prefix()))?
        };

        Ok(Self { decoder, lm_head })
    }

    pub fn decoder(&self) -> &BartDecoder {
        &self.decoder
    }

    pub fn decoder_mut(&mut self) -> &mut BartDecoder {
        &mut self.decoder
    }

    pub fn reset_kv_cache(&mut self) {
        self.decoder.reset_kv_cache();
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        encoder_xs: Option<&Tensor>,
        past_kv_len: usize,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let xs = self
            .decoder
            .forward(xs, encoder_xs, past_kv_len, attn_mask)?;
        xs.apply(&self.lm_head)
    }

    /// Generate a causal attention mask.
    pub fn create_causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 }))
            .collect();
        Tensor::from_vec(mask, (seq_len, seq_len), device)
    }

    /// Convenience method for decoding with encoder output.
    pub fn decode(
        &mut self,
        decoder_input_ids: &Tensor,
        encoder_hidden_states: &Tensor,
        past_kv_len: usize,
    ) -> Result<Tensor> {
        let seq_len = decoder_input_ids.dim(1)?;
        let device = decoder_input_ids.device();

        // For incremental decoding with cache, we only need mask for current position
        let attn_mask = if past_kv_len > 0 && seq_len == 1 {
            // Single token with cache: no mask needed (all previous tokens are valid)
            None
        } else {
            // Full sequence: use causal mask
            Some(Self::create_causal_mask(seq_len, device)?)
        };

        self.forward(
            decoder_input_ids,
            Some(encoder_hidden_states),
            past_kv_len,
            attn_mask.as_ref(),
        )
    }
}
