use candle::{Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

use crate::models::bart::config::BartConfig;

/// Multi-head attention with optional KV-cache.
#[derive(Debug, Clone)]
pub struct BartAttention {
    head_dim: usize,
    num_heads: usize,
    scaling: f64,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    kv_cache: Option<(Tensor, Tensor)>,
    /// Controls whether KV caching is enabled.
    /// - Encoder self-attention: false (processes full sequence once)
    /// - Decoder self-attention: true (incremental decoding)
    /// - Decoder cross-attention: true (cache encoder K/V)
    enable_kv_cache: bool,
}

impl BartAttention {
    pub fn load(
        vb: VarBuilder,
        cfg: &BartConfig,
        is_cross_attention: bool,
        enable_kv_cache: bool,
    ) -> Result<Self> {
        let embed_dim = cfg.d_model;
        let num_heads = cfg.decoder_attention_heads;
        let head_dim = embed_dim / num_heads;

        // For cross-attention, k and v projections take encoder_hidden_size as input
        // If cross_attention_hidden_size is None, use d_model (encoder output matches decoder)
        let kv_input_dim = if is_cross_attention {
            cfg.cross_attention_hidden_size.unwrap_or(embed_dim)
        } else {
            embed_dim
        };

        let q_proj = linear(embed_dim, embed_dim, vb.pp("q_proj"))?;
        let k_proj = linear(kv_input_dim, embed_dim, vb.pp("k_proj"))?;
        let v_proj = linear(kv_input_dim, embed_dim, vb.pp("v_proj"))?;
        let out_proj = linear(embed_dim, embed_dim, vb.pp("out_proj"))?;

        Ok(Self {
            head_dim,
            num_heads,
            scaling: 1.0 / (head_dim as f64).sqrt(),
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            kv_cache: None,
            enable_kv_cache,
        })
    }

    /// Load attention for encoder (no KV cache).
    pub fn load_for_encoder(vb: VarBuilder, cfg: &BartConfig) -> Result<Self> {
        Self::load(vb, cfg, false, false)
    }

    pub fn reset_kv_cache(&mut self) {
        self.kv_cache = None;
    }

    fn _shape(&self, tensor: &Tensor, bsz: usize) -> Result<Tensor> {
        tensor
            .reshape((bsz, (), self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        kv_states: Option<&Tensor>,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, tgt_len, _) = xs.dims3()?;
        let query_states = (xs.apply(&self.q_proj)? * self.scaling)?;

        let (key_states, value_states) = match kv_states {
            None => {
                // Self-attention: compute K/V from input
                let key_states = self._shape(&xs.apply(&self.k_proj)?, b_sz)?;
                let value_states = self._shape(&xs.apply(&self.v_proj)?, b_sz)?;

                // Use KV-cache if enabled (decoder self-attention)
                if self.enable_kv_cache {
                    let kv_states = match &self.kv_cache {
                        None => (key_states, value_states),
                        Some((p_key_states, p_value_states)) => {
                            let key_states = Tensor::cat(&[p_key_states, &key_states], 2)?;
                            let value_states = Tensor::cat(&[p_value_states, &value_states], 2)?;
                            (key_states, value_states)
                        }
                    };
                    self.kv_cache = Some(kv_states.clone());
                    kv_states
                } else {
                    (key_states, value_states)
                }
            }
            Some(kv_states) => {
                // Cross-attention: use encoder hidden states for K/V
                // If caching enabled, reuse cached K/V if available
                if self.enable_kv_cache {
                    if let Some((cached_k, cached_v)) = &self.kv_cache {
                        (cached_k.clone(), cached_v.clone())
                    } else {
                        let key_states = self._shape(&kv_states.apply(&self.k_proj)?, b_sz)?;
                        let value_states = self._shape(&kv_states.apply(&self.v_proj)?, b_sz)?;
                        self.kv_cache = Some((key_states.clone(), value_states.clone()));
                        (key_states, value_states)
                    }
                } else {
                    let key_states = self._shape(&kv_states.apply(&self.k_proj)?, b_sz)?;
                    let value_states = self._shape(&kv_states.apply(&self.v_proj)?, b_sz)?;
                    (key_states, value_states)
                }
            }
        };

        let proj_shape = (b_sz * self.num_heads, (), self.head_dim);
        let query_states = self._shape(&query_states, b_sz)?.reshape(proj_shape)?;
        let key_states = key_states.reshape(proj_shape)?;
        let value_states = value_states.reshape(proj_shape)?;

        let attn_weights = query_states.matmul(&key_states.transpose(1, 2)?)?;
        let attn_weights = match attn_mask {
            None => attn_weights,
            Some(attn_mask) => attn_weights.broadcast_add(attn_mask)?,
        };
        let attn_probs = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_probs.matmul(&value_states)?;

        attn_output
            .reshape((b_sz, self.num_heads, tgt_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((b_sz, tgt_len, self.head_dim * self.num_heads))?
            .apply(&self.out_proj)
    }

    /// Forward pass with external cache management.
    /// Model remains immutable (&self), cache is passed explicitly.
    ///
    /// # Arguments
    /// * `xs` - Query input, shape (batch_beams, seq_len, d_model)
    /// * `kv_states` - For cross-attention: encoder hidden states
    /// * `attn_mask` - Optional causal mask
    /// * `cache` - Mutable reference to external (K, V) cache for this layer
    ///
    /// # Returns
    /// Attention output, shape (batch_beams, seq_len, d_model)
    pub fn forward_with_cache(
        &self,
        xs: &Tensor,
        kv_states: Option<&Tensor>,
        attn_mask: Option<&Tensor>,
        cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b_sz, tgt_len, _) = xs.dims3()?;
        let query_states = (xs.apply(&self.q_proj)? * self.scaling)?;

        let (key_states, value_states) = match kv_states {
            None => {
                // Self-attention: compute K/V from input
                let key_states = self._shape(&xs.apply(&self.k_proj)?, b_sz)?;
                let value_states = self._shape(&xs.apply(&self.v_proj)?, b_sz)?;

                // Concatenate with cached K/V if present
                let (key_states, value_states) = match cache.as_ref() {
                    None => (key_states, value_states),
                    Some((cached_k, cached_v)) => {
                        let key_states = Tensor::cat(&[cached_k, &key_states], 2)?;
                        let value_states = Tensor::cat(&[cached_v, &value_states], 2)?;
                        (key_states, value_states)
                    }
                };

                // Update cache with new K/V
                *cache = Some((key_states.clone(), value_states.clone()));
                (key_states, value_states)
            }
            Some(kv_states) => {
                // Cross-attention: use encoder hidden states for K/V
                // Cache is populated on first call, then reused
                match cache.as_ref() {
                    Some((cached_k, cached_v)) => {
                        // Reuse cached encoder K/V
                        (cached_k.clone(), cached_v.clone())
                    }
                    None => {
                        // First call: compute and cache encoder K/V
                        let key_states = self._shape(&kv_states.apply(&self.k_proj)?, b_sz)?;
                        let value_states = self._shape(&kv_states.apply(&self.v_proj)?, b_sz)?;
                        *cache = Some((key_states.clone(), value_states.clone()));
                        (key_states, value_states)
                    }
                }
            }
        };

        // Rest of attention computation
        let proj_shape = (b_sz * self.num_heads, (), self.head_dim);
        let query_states = self._shape(&query_states, b_sz)?.reshape(proj_shape)?;
        let key_states = key_states.reshape(proj_shape)?;
        let value_states = value_states.reshape(proj_shape)?;

        let attn_weights = query_states.matmul(&key_states.transpose(1, 2)?)?;
        let attn_weights = match attn_mask {
            None => attn_weights,
            Some(attn_mask) => attn_weights.broadcast_add(attn_mask)?,
        };
        let attn_probs = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_probs.matmul(&value_states)?;

        attn_output
            .reshape((b_sz, self.num_heads, tgt_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((b_sz, tgt_len, self.head_dim * self.num_heads))?
            .apply(&self.out_proj)
    }
}
