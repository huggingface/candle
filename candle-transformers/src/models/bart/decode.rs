use candle::{Result, Tensor};
use candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, Module, VarBuilder};

use crate::models::bart::{
    attention::BartAttention,
    beam_search::BatchedKVCache,
    config::{BartConfig, BartLearnedPositionalEmbedding, BartWeightPrefix, LayerNormOrder},
};

/// BART decoder layer with self-attention, cross-attention, and FFN.
#[derive(Debug, Clone)]
pub struct BartDecoderLayer {
    self_attn: BartAttention,
    self_attn_layer_norm: LayerNorm,
    encoder_attn: BartAttention,
    encoder_attn_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
    activation_fn: candle_nn::Activation,
    layer_norm_order: LayerNormOrder,
}

impl BartDecoderLayer {
    fn load(vb: VarBuilder, cfg: &BartConfig) -> Result<Self> {
        let embed_dim = cfg.d_model;
        // Decoder self-attention uses cache for incremental decoding
        let self_attn = BartAttention::load(vb.pp("self_attn"), cfg, false, true)?;
        let self_attn_layer_norm = layer_norm(embed_dim, 1e-5, vb.pp("self_attn_layer_norm"))?;
        // Decoder cross-attention uses cache for encoder K/V
        let encoder_attn = BartAttention::load(vb.pp("encoder_attn"), cfg, true, true)?;
        let encoder_attn_layer_norm =
            layer_norm(embed_dim, 1e-5, vb.pp("encoder_attn_layer_norm"))?;
        let fc1 = linear(embed_dim, cfg.decoder_ffn_dim, vb.pp("fc1"))?;
        let fc2 = linear(cfg.decoder_ffn_dim, embed_dim, vb.pp("fc2"))?;
        let final_layer_norm = layer_norm(embed_dim, 1e-5, vb.pp("final_layer_norm"))?;

        Ok(Self {
            self_attn,
            self_attn_layer_norm,
            encoder_attn,
            encoder_attn_layer_norm,
            fc1,
            fc2,
            final_layer_norm,
            activation_fn: cfg.activation_function,
            layer_norm_order: cfg.layer_norm_order(),
        })
    }

    fn reset_kv_cache(&mut self) {
        self.self_attn.reset_kv_cache();
        self.encoder_attn.reset_kv_cache();
    }

    /// Reset only the cross-attention cache (when encoder output changes).
    fn reset_cross_attn_cache(&mut self) {
        self.encoder_attn.reset_kv_cache();
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<Tensor> {
        match self.layer_norm_order {
            LayerNormOrder::Pre => self.forward_pre_norm(xs, attention_mask, encoder_hidden_states),
            LayerNormOrder::Post => {
                self.forward_post_norm(xs, attention_mask, encoder_hidden_states)
            }
        }
    }

    fn forward_pre_norm(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<Tensor> {
        // PRE-LAYERNORM: norm before attention (MBart, Donut)

        // Self-attention with pre-norm
        let residual = xs.clone();
        let xs = self.self_attn_layer_norm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, None, attention_mask)?;
        let mut xs = (xs + residual)?;

        // Cross-attention to encoder with pre-norm
        if let Some(encoder_hidden_states) = encoder_hidden_states {
            let residual = xs.clone();
            let xs_normed = self.encoder_attn_layer_norm.forward(&xs)?;
            xs = self
                .encoder_attn
                .forward(&xs_normed, Some(encoder_hidden_states), None)?;
            xs = (xs + residual)?;
        }

        // FFN with pre-norm
        let residual = xs.clone();
        let xs = self.final_layer_norm.forward(&xs)?;
        let xs = self.fc1.forward(&xs)?;
        let xs = self.activation_fn.forward(&xs)?;
        let xs = self.fc2.forward(&xs)?;
        xs + residual
    }

    fn forward_post_norm(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<Tensor> {
        // POST-LAYERNORM: norm after residual add (BART)

        // Self-attention with post-norm
        let residual = xs.clone();
        let xs = self.self_attn.forward(xs, None, attention_mask)?;
        let xs = (xs + residual)?;
        let mut xs = self.self_attn_layer_norm.forward(&xs)?;

        // Cross-attention to encoder with post-norm
        if let Some(encoder_hidden_states) = encoder_hidden_states {
            let residual = xs.clone();
            xs = self
                .encoder_attn
                .forward(&xs, Some(encoder_hidden_states), None)?;
            xs = (xs + residual)?;
            xs = self.encoder_attn_layer_norm.forward(&xs)?;
        }

        // FFN with post-norm
        let residual = xs.clone();
        let xs = self.fc1.forward(&xs)?;
        let xs = self.activation_fn.forward(&xs)?;
        let xs = self.fc2.forward(&xs)?;
        let xs = (xs + residual)?;
        self.final_layer_norm.forward(&xs)
    }

    /// Forward pass with external cache.
    ///
    /// # Arguments
    /// * `layer_idx` - Index of this layer (for cache lookup)
    /// * `cache` - External batched cache
    pub fn forward_with_cache(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        layer_idx: usize,
        cache: &mut BatchedKVCache,
    ) -> Result<Tensor> {
        // Take cache entries for this layer
        let mut self_attn_cache = cache.take_self_attn(layer_idx);
        let mut cross_attn_cache = cache.take_cross_attn(layer_idx);

        let result = match self.layer_norm_order {
            LayerNormOrder::Pre => self.forward_pre_norm_with_cache(
                xs,
                attention_mask,
                encoder_hidden_states,
                &mut self_attn_cache,
                &mut cross_attn_cache,
            ),
            LayerNormOrder::Post => self.forward_post_norm_with_cache(
                xs,
                attention_mask,
                encoder_hidden_states,
                &mut self_attn_cache,
                &mut cross_attn_cache,
            ),
        };

        // Store updated cache
        if let Some(kv) = self_attn_cache {
            cache.set_self_attn(layer_idx, kv);
        }
        if let Some(kv) = cross_attn_cache {
            cache.set_cross_attn(layer_idx, kv);
        }

        result
    }

    fn forward_pre_norm_with_cache(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        self_attn_cache: &mut Option<(Tensor, Tensor)>,
        cross_attn_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        // PRE-LAYERNORM: norm before attention (MBart, Donut)

        // Self-attention with pre-norm
        let residual = xs.clone();
        let xs = self.self_attn_layer_norm.forward(xs)?;
        let xs = self
            .self_attn
            .forward_with_cache(&xs, None, attention_mask, self_attn_cache)?;
        let mut xs = (xs + residual)?;

        // Cross-attention to encoder with pre-norm
        if let Some(encoder_hidden_states) = encoder_hidden_states {
            let residual = xs.clone();
            let xs_normed = self.encoder_attn_layer_norm.forward(&xs)?;
            xs = self.encoder_attn.forward_with_cache(
                &xs_normed,
                Some(encoder_hidden_states),
                None,
                cross_attn_cache,
            )?;
            xs = (xs + residual)?;
        }

        // FFN with pre-norm
        let residual = xs.clone();
        let xs = self.final_layer_norm.forward(&xs)?;
        let xs = self.fc1.forward(&xs)?;
        let xs = self.activation_fn.forward(&xs)?;
        let xs = self.fc2.forward(&xs)?;
        xs + residual
    }

    fn forward_post_norm_with_cache(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        self_attn_cache: &mut Option<(Tensor, Tensor)>,
        cross_attn_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        // POST-LAYERNORM: norm after residual add (BART)

        // Self-attention with post-norm
        let residual = xs.clone();
        let xs = self
            .self_attn
            .forward_with_cache(xs, None, attention_mask, self_attn_cache)?;
        let xs = (xs + residual)?;
        let mut xs = self.self_attn_layer_norm.forward(&xs)?;

        // Cross-attention to encoder with post-norm
        if let Some(encoder_hidden_states) = encoder_hidden_states {
            let residual = xs.clone();
            xs = self.encoder_attn.forward_with_cache(
                &xs,
                Some(encoder_hidden_states),
                None,
                cross_attn_cache,
            )?;
            xs = (xs + residual)?;
            xs = self.encoder_attn_layer_norm.forward(&xs)?;
        }

        // FFN with post-norm
        let residual = xs.clone();
        let xs = self.fc1.forward(&xs)?;
        let xs = self.activation_fn.forward(&xs)?;
        let xs = self.fc2.forward(&xs)?;
        let xs = (xs + residual)?;
        self.final_layer_norm.forward(&xs)
    }
}

/// BART decoder with embeddings and transformer layers.
#[derive(Debug, Clone)]
pub struct BartDecoder {
    embed_tokens: Embedding,
    embed_positions: BartLearnedPositionalEmbedding,
    layernorm_embedding: LayerNorm,
    layers: Vec<BartDecoderLayer>,
    layer_norm: Option<LayerNorm>,
    embed_scale: Option<f64>,
}

impl BartDecoder {
    /// Internal constructor: loads decoder at the given VarBuilder path.
    /// Does NOT apply any path prefix.
    pub fn new_internal(
        cfg: &BartConfig,
        vb: VarBuilder,
        embed_tokens: Option<Embedding>,
    ) -> Result<Self> {
        let embed_tokens = match embed_tokens {
            Some(e) => e,
            None => embedding(cfg.vocab_size, cfg.d_model, vb.pp("embed_tokens"))?,
        };
        let embed_positions = BartLearnedPositionalEmbedding::load(vb.pp("embed_positions"), cfg)?;
        let layernorm_embedding = layer_norm(cfg.d_model, 1e-5, vb.pp("layernorm_embedding"))?;

        let mut layers = Vec::with_capacity(cfg.decoder_layers);
        let vb_l = vb.pp("layers");
        for idx in 0..cfg.decoder_layers {
            let layer = BartDecoderLayer::load(vb_l.pp(idx), cfg)?;
            layers.push(layer);
        }

        let layer_norm = if cfg.add_final_layer_norm {
            Some(layer_norm(cfg.d_model, 1e-5, vb.pp("layer_norm"))?)
        } else {
            None
        };

        let embed_scale = if cfg.scale_embedding {
            Some((cfg.d_model as f64).sqrt())
        } else {
            None
        };

        Ok(Self {
            embed_tokens,
            embed_positions,
            layernorm_embedding,
            layers,
            layer_norm,
            embed_scale,
        })
    }

    /// Legacy constructor for VisionEncoderDecoder models (Donut, TrOCR, etc.).
    /// Applies "decoder.model.decoder" path prefix for backward compatibility.
    pub fn new(cfg: &BartConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_internal(cfg, vb.pp("decoder.model.decoder"), None)
    }

    /// Load decoder using shared embeddings (for full BART models with weight tying).
    /// The VarBuilder should already point to the decoder path (e.g., "model.decoder").
    pub fn new_with_shared_embeddings(
        cfg: &BartConfig,
        embed_tokens: Embedding,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::new_internal(cfg, vb, Some(embed_tokens))
    }

    /// Load decoder with explicit weight prefix.
    pub fn new_with_prefix(
        cfg: &BartConfig,
        vb: VarBuilder,
        prefix: BartWeightPrefix,
    ) -> Result<Self> {
        Self::new_internal(cfg, vb.pp(prefix.decoder_prefix()), None)
    }

    pub fn embed_tokens(&self) -> &Embedding {
        &self.embed_tokens
    }

    pub fn reset_kv_cache(&mut self) {
        self.layers.iter_mut().for_each(|l| l.reset_kv_cache());
    }

    /// Reset only the cross-attention cache (when encoder output changes).
    pub fn reset_cross_attn_cache(&mut self) {
        self.layers
            .iter_mut()
            .for_each(|l| l.reset_cross_attn_cache());
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        encoder_xs: Option<&Tensor>,
        past_kv_len: usize,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let embed_pos = self.embed_positions.forward(xs, past_kv_len)?;
        let xs = xs.apply(&self.embed_tokens)?;

        let xs = match self.embed_scale {
            None => xs,
            Some(scale) => (xs * scale)?,
        };

        let mut xs = xs.broadcast_add(&embed_pos)?;
        xs = self.layernorm_embedding.forward(&xs)?;

        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attn_mask, encoder_xs)?;
        }

        if let Some(layer_norm) = &self.layer_norm {
            xs = layer_norm.forward(&xs)?;
        }

        Ok(xs)
    }

    /// Forward pass with external cache for batched decoding.
    pub fn forward_with_cache(
        &self,
        xs: &Tensor,
        encoder_xs: Option<&Tensor>,
        cache: &mut BatchedKVCache,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let past_kv_len = cache.get_past_kv_len();

        let embed_pos = self.embed_positions.forward(xs, past_kv_len)?;
        let xs = xs.apply(&self.embed_tokens)?;

        let xs = match self.embed_scale {
            None => xs,
            Some(scale) => (xs * scale)?,
        };

        let mut xs = xs.broadcast_add(&embed_pos)?;
        xs = self.layernorm_embedding.forward(&xs)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_with_cache(&xs, attn_mask, encoder_xs, layer_idx, cache)?;
        }

        if let Some(layer_norm) = &self.layer_norm {
            xs = layer_norm.forward(&xs)?;
        }

        Ok(xs)
    }
}
