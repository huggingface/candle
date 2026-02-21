use candle::{Result, Tensor};
use candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, Module, VarBuilder};

use crate::models::bart::{
    attention::BartAttention,
    config::{BartConfig, BartLearnedPositionalEmbedding, LayerNormOrder},
};

/// BART encoder layer with self-attention and FFN (no cross-attention).
#[derive(Debug, Clone)]
pub struct BartEncoderLayer {
    self_attn: BartAttention,
    self_attn_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
    activation_fn: candle_nn::Activation,
    layer_norm_order: LayerNormOrder,
}

impl BartEncoderLayer {
    fn load(vb: VarBuilder, cfg: &BartConfig) -> Result<Self> {
        let embed_dim = cfg.d_model;
        // Encoder self-attention does NOT use cache (processes full sequence once)
        let self_attn = BartAttention::load_for_encoder(vb.pp("self_attn"), cfg)?;
        let self_attn_layer_norm = layer_norm(embed_dim, 1e-5, vb.pp("self_attn_layer_norm"))?;
        let fc1 = linear(embed_dim, cfg.encoder_ffn_dim(), vb.pp("fc1"))?;
        let fc2 = linear(cfg.encoder_ffn_dim(), embed_dim, vb.pp("fc2"))?;
        let final_layer_norm = layer_norm(embed_dim, 1e-5, vb.pp("final_layer_norm"))?;

        Ok(Self {
            self_attn,
            self_attn_layer_norm,
            fc1,
            fc2,
            final_layer_norm,
            activation_fn: cfg.activation_function,
            layer_norm_order: cfg.layer_norm_order(),
        })
    }

    fn forward(&mut self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        match self.layer_norm_order {
            LayerNormOrder::Pre => self.forward_pre_norm(xs, attention_mask),
            LayerNormOrder::Post => self.forward_post_norm(xs, attention_mask),
        }
    }

    fn forward_pre_norm(&mut self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // PRE-LAYERNORM: norm before attention (MBart, Donut)
        let residual = xs.clone();
        let xs = self.self_attn_layer_norm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, None, attention_mask)?;
        let xs = (xs + residual)?;

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
    ) -> Result<Tensor> {
        // POST-LAYERNORM: norm after residual add (BART)
        let residual = xs.clone();
        let xs = self.self_attn.forward(xs, None, attention_mask)?;
        let xs = (xs + residual)?;
        let xs = self.self_attn_layer_norm.forward(&xs)?;

        let residual = xs.clone();
        let xs = self.fc1.forward(&xs)?;
        let xs = self.activation_fn.forward(&xs)?;
        let xs = self.fc2.forward(&xs)?;
        let xs = (xs + residual)?;
        self.final_layer_norm.forward(&xs)
    }
}

/// BART encoder with embeddings and transformer layers.
#[derive(Debug, Clone)]
pub struct BartEncoder {
    embed_tokens: Embedding,
    embed_positions: BartLearnedPositionalEmbedding,
    /// Optional layernorm_embedding, controlled by normalize_embedding config.
    layernorm_embedding: Option<LayerNorm>,
    layers: Vec<BartEncoderLayer>,
    /// Final layer norm for PRE-LAYERNORM models (mBART).
    /// POST-LAYERNORM models (BART) don't need this because each layer already
    /// normalizes at the end.
    layer_norm: Option<LayerNorm>,
    embed_scale: Option<f64>,
}

impl BartEncoder {
    /// Load encoder from weights at the given VarBuilder path.
    pub fn new(cfg: &BartConfig, vb: VarBuilder) -> Result<Self> {
        let embed_tokens = embedding(cfg.vocab_size, cfg.d_model, vb.pp("embed_tokens"))?;
        Self::new_with_shared_embeddings(cfg, embed_tokens, vb)
    }

    /// Load encoder using shared embeddings (for weight tying with decoder).
    pub fn new_with_shared_embeddings(
        cfg: &BartConfig,
        embed_tokens: Embedding,
        vb: VarBuilder,
    ) -> Result<Self> {
        let embed_positions = BartLearnedPositionalEmbedding::load(vb.pp("embed_positions"), cfg)?;

        // layernorm_embedding is optional, controlled by normalize_embedding config
        let layernorm_embedding = if cfg.normalize_embedding.unwrap_or(true) {
            Some(layer_norm(cfg.d_model, 1e-5, vb.pp("layernorm_embedding"))?)
        } else {
            None
        };

        let num_layers = cfg.encoder_layers();
        let mut layers = Vec::with_capacity(num_layers);
        let vb_l = vb.pp("layers");
        for idx in 0..num_layers {
            let layer = BartEncoderLayer::load(vb_l.pp(idx), cfg)?;
            layers.push(layer);
        }

        // PRE-LAYERNORM models (mBART) need a final layer_norm on the encoder output.
        // POST-LAYERNORM models (BART) don't need this because each layer already
        // applies norm after the residual connection.
        let layer_norm = match cfg.layer_norm_order() {
            LayerNormOrder::Pre => Some(layer_norm(cfg.d_model, 1e-5, vb.pp("layer_norm"))?),
            LayerNormOrder::Post => None,
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

    pub fn embed_tokens(&self) -> &Embedding {
        &self.embed_tokens
    }

    pub fn forward(&mut self, xs: &Tensor) -> Result<Tensor> {
        let embed_pos = self.embed_positions.forward(xs, 0)?;
        let xs = xs.apply(&self.embed_tokens)?;

        let xs = match self.embed_scale {
            None => xs,
            Some(scale) => (xs * scale)?,
        };

        let mut xs = xs.broadcast_add(&embed_pos)?;

        if let Some(layernorm_embedding) = &self.layernorm_embedding {
            xs = layernorm_embedding.forward(&xs)?;
        }

        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, None)?;
        }

        // Final layer norm for PRE-LAYERNORM models (mBART)
        if let Some(layer_norm) = &self.layer_norm {
            xs = layer_norm.forward(&xs)?;
        }

        Ok(xs)
    }

    /// Forward pass with attention mask.
    pub fn forward_with_mask(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let embed_pos = self.embed_positions.forward(xs, 0)?;
        let xs = xs.apply(&self.embed_tokens)?;

        let xs = match self.embed_scale {
            None => xs,
            Some(scale) => (xs * scale)?,
        };

        let mut xs = xs.broadcast_add(&embed_pos)?;

        if let Some(layernorm_embedding) = &self.layernorm_embedding {
            xs = layernorm_embedding.forward(&xs)?;
        }

        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask)?;
        }

        // Final layer norm for PRE-LAYERNORM models (mBART)
        if let Some(layer_norm) = &self.layer_norm {
            xs = layer_norm.forward(&xs)?;
        }

        Ok(xs)
    }
}
