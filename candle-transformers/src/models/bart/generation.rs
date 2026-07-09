use candle::{Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

use crate::models::bart::{
    beam_search::BatchedKVCache, causal_lm::BartForCausalLM, config::BartConfig, model::BartModel,
};

/// BART model for sequence-to-sequence generation (full encoder-decoder).
#[derive(Debug, Clone)]
pub struct BartForConditionalGeneration {
    model: BartModel,
    lm_head: Linear,
    final_logits_bias: Option<Tensor>,
    /// Track encoder output identity for cross-attention cache invalidation.
    encoder_output_id: Option<candle::TensorId>,
}

impl BartForConditionalGeneration {
    /// Load BartForConditionalGeneration from weights.
    /// Expects weights at: model.*, lm_head (or tied to model.shared)
    pub fn new(cfg: &BartConfig, vb: VarBuilder) -> Result<Self> {
        let model = BartModel::new(cfg, vb.clone())?;

        // Weight tying: when tie_word_embeddings=true (default), reuse shared embedding
        // IMPORTANT: Embedding is already (vocab_size, d_model) - NO transpose needed!
        let lm_head = if cfg.tie_word_embeddings {
            candle_nn::Linear::new(model.shared.embeddings().clone(), None)
        } else {
            linear(cfg.d_model, cfg.vocab_size, vb.pp("lm_head"))?
        };

        // Optional bias added to logits (most BART checkpoints have this)
        let final_logits_bias = vb.get((1, cfg.vocab_size), "final_logits_bias").ok();

        Ok(Self {
            model,
            lm_head,
            final_logits_bias,
            encoder_output_id: None,
        })
    }

    pub fn model(&self) -> &BartModel {
        &self.model
    }

    pub fn model_mut(&mut self) -> &mut BartModel {
        &mut self.model
    }

    /// Reset all KV caches (both self-attention and cross-attention).
    pub fn reset_kv_cache(&mut self) {
        self.model.decoder_mut().reset_kv_cache();
        self.encoder_output_id = None;
    }

    /// Encode input tokens to hidden states.
    pub fn encode(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        self.model.encoder_mut().forward(input_ids)
    }

    /// Decode with encoder output, applying cross-attention cache invalidation.
    pub fn decode(
        &mut self,
        decoder_input_ids: &Tensor,
        encoder_output: &Tensor,
        past_kv_len: usize,
    ) -> Result<Tensor> {
        // Invalidate cross-attention cache if encoder output changed
        let current_id = encoder_output.id();
        if self.encoder_output_id != Some(current_id) {
            self.model.decoder_mut().reset_cross_attn_cache();
            self.encoder_output_id = Some(current_id);
        }

        let seq_len = decoder_input_ids.dim(1)?;
        let device = decoder_input_ids.device();

        // For incremental decoding with cache, we only need mask for current position
        let attn_mask = if past_kv_len > 0 && seq_len == 1 {
            None
        } else {
            Some(BartForCausalLM::create_causal_mask(seq_len, device)?)
        };

        let hidden = self.model.decoder_mut().forward(
            decoder_input_ids,
            Some(encoder_output),
            past_kv_len,
            attn_mask.as_ref(),
        )?;

        let logits = hidden.apply(&self.lm_head)?;

        // Apply final_logits_bias if present
        if let Some(bias) = &self.final_logits_bias {
            logits.broadcast_add(bias)
        } else {
            Ok(logits)
        }
    }

    /// Full forward pass: encode input and decode with decoder input.
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        decoder_input_ids: &Tensor,
        past_kv_len: usize,
    ) -> Result<Tensor> {
        let encoder_output = self.encode(input_ids)?;
        self.decode(decoder_input_ids, &encoder_output, past_kv_len)
    }

    /// Get the LM head for direct access (used in batched beam search).
    pub fn lm_head(&self) -> &Linear {
        &self.lm_head
    }

    /// Get the final logits bias for direct access (used in batched beam search).
    pub fn final_logits_bias(&self) -> Option<&Tensor> {
        self.final_logits_bias.as_ref()
    }

    /// Decode with external cache for batched beam search.
    /// Model remains immutable (&self).
    pub fn decode_with_cache(
        &self,
        decoder_input_ids: &Tensor,
        encoder_output: &Tensor,
        cache: &mut BatchedKVCache,
    ) -> Result<Tensor> {
        let seq_len = decoder_input_ids.dim(1)?;
        let device = decoder_input_ids.device();
        let past_kv_len = cache.get_past_kv_len();

        // For incremental decoding with cache, we only need mask for current position
        let attn_mask = if past_kv_len > 0 && seq_len == 1 {
            None
        } else {
            Some(BartForCausalLM::create_causal_mask(seq_len, device)?)
        };

        let hidden = self.model.decoder().forward_with_cache(
            decoder_input_ids,
            Some(encoder_output),
            cache,
            attn_mask.as_ref(),
        )?;

        let logits = hidden.apply(&self.lm_head)?;

        // Apply final_logits_bias if present
        if let Some(bias) = &self.final_logits_bias {
            logits.broadcast_add(bias)
        } else {
            Ok(logits)
        }
    }
}
