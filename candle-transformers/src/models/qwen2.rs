//! Qwen2 model implementation with quantization support.
//!
//! Qwen2 is a large language model from Alibaba optimized for efficiency.
//! This implementation provides quantization for reduced memory and compute.
//!
//! Key characteristics:
//! - Streaming decode support
//! - Grouped query attention (GQA)
//! - RMSNorm for layer normalization
//! - Rotary positional embeddings (RoPE)
//! - Support for 8-bit quantization
//!
//! References:
//! - 🤗 [Qwen2 Model](https://huggingface.co/Qwen/Qwen2-7B)
//!

use crate::models::with_tracing::{linear, linear_no_bias, Linear, RmsNorm};
use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Activation, VarBuilder};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    pub max_window_layers: usize,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub use_sliding_window: bool,
    pub hidden_act: Activation,
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / (cfg.rope_theta as f32).powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();

        // Keep in f32 for precision
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?; // f32
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)? // f32, not model dtype
            .reshape((max_seq_len, 1))?;

        // Compute freqs, sin, cos all in f32 for precision
        let freqs = t.matmul(&inv_freq)?; // f32 matmul

        // Only cast to model dtype at the very end
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_sz, hidden_sz, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = hidden_sz / num_heads;
        let q_proj = linear(hidden_sz, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_sz, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_sz, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_sz, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size: hidden_sz,
            rotary_emb,
            kv_cache: None,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)?;

        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        let key_states = crate::utils::repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states =
            crate::utils::repeat_kv(value_states, self.num_kv_groups)?.contiguous()?;

        let attn_output = {
            let scale = 1f64 / f64::sqrt(self.head_dim as f64);
            let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;

            let attn_weights = match attention_mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(mask)?,
            };
            // Metal uses online softmax which handles numerical stability internally.
            // CUDA accumulates in float for f16/bf16 inputs.
            // TODO: Implement online softmax for CUDA to match Metal's single-pass approach.
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&value_states)?
        };
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.hidden_size))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }

    /// Extract the current KV cache (returns owned copy)
    pub fn extract_kv_cache(&self) -> Option<(Tensor, Tensor)> {
        self.kv_cache.clone()
    }

    /// Restore a previously extracted KV cache
    pub fn restore_kv_cache(&mut self, cache: Option<(Tensor, Tensor)>) {
        self.kv_cache = cache;
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask, seqlen_offset)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }

    pub fn extract_kv_cache(&self) -> Option<(Tensor, Tensor)> {
        self.self_attn.extract_kv_cache()
    }

    pub fn restore_kv_cache(&mut self, cache: Option<(Tensor, Tensor)>) {
        self.self_attn.restore_kv_cache(cache);
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    sliding_window: usize,
    device: Device,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            sliding_window: cfg.sliding_window,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn prepare_causal_attention_mask(
        &self,
        _b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        // Total key/value length (cached + current input)
        let total_len = seqlen_offset + tgt_len;

        // Create mask of shape [tgt_len, total_len] where:
        // - Query positions are 0..tgt_len (relative), absolute = seqlen_offset + i
        // - Key positions are 0..total_len (absolute)
        //
        // For each query at absolute position (seqlen_offset + i):
        // - Causal: can attend to key positions j where j <= seqlen_offset + i
        // - Sliding window: can attend to key positions j where j >= (seqlen_offset + i) - sliding_window
        // Use f32::MIN (finite large negative) like PyTorch's torch.finfo(dtype).min
        // This is consistent with prepare_4d_causal_attention_mask_with_cache_position
        let min_dtype = f32::MIN;

        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                let abs_query_pos = seqlen_offset + i;
                (0..total_len).map(move |j| {
                    // Causal check: can't attend to future positions
                    let is_future = j > abs_query_pos;
                    // Sliding window check: can't attend to positions too far in the past
                    let is_too_old = j + self.sliding_window < abs_query_pos;

                    if is_future || is_too_old {
                        min_dtype
                    } else {
                        0.
                    }
                })
            })
            .collect();

        let mask = Tensor::from_slice(&mask, (tgt_len, total_len), &self.device)?;
        // Expand to [1, 1, tgt_len, total_len] - will broadcast to batch size during attention
        let final_mask = mask
            .expand((1, 1, tgt_len, total_len))?
            .to_dtype(self.dtype)?;
        Ok(final_mask)
    }

    fn prepare_attention_mask(&self, attn_mask: &Tensor) -> Result<Tensor> {
        let (b_sz, sql_len) = attn_mask.dims2()?;
        let mut mask: Vec<Tensor> = vec![];
        for b in 0..b_sz {
            mask.push(attn_mask.i((b, ..))?.expand((1, 1, sql_len, sql_len))?);
        }
        let mask = Tensor::cat(&mask, 0)?;
        let on_true = mask.zeros_like()?.to_dtype(self.dtype)?;
        // Use f32::MIN like PyTorch's torch.finfo(dtype).min for consistency
        let on_false = Tensor::new(f32::MIN, &self.device)?
            .broadcast_as(mask.shape())?
            .to_dtype(self.dtype)?;
        mask.where_cond(&on_true, &on_false)
    }

    /// Creates a 4D causal attention mask that properly handles cached decoding with
    /// selective attention (e.g., only attending to specific positions).
    ///
    /// This matches PyTorch's `_prepare_4d_causal_attention_mask_with_cache_position`.
    ///
    /// # Arguments
    /// * `attention_mask` - Optional 2D mask [batch, total_seq_len] where 1=attend, 0=mask
    /// * `query_length` - Number of query tokens (usually 1 during decode)
    /// * `key_length` - Total K/V length (cache + current)
    /// * `cache_position` - Absolute positions of query tokens [query_length]
    /// * `batch_size` - Batch size for expansion
    ///
    /// # Returns
    /// 4D attention mask [batch, 1, query_length, key_length]
    pub fn prepare_4d_causal_attention_mask_with_cache_position(
        &self,
        attention_mask: Option<&Tensor>,
        query_length: usize,
        key_length: usize,
        cache_position: &Tensor,
        batch_size: usize,
    ) -> Result<Tensor> {
        // Get cache_position as a vector for iteration
        let cache_pos_vec: Vec<i64> = cache_position.to_vec1()?;

        // Create base causal mask [query_length, key_length]
        // For each query at cache_position[i], it can attend to key positions j
        // where j <= cache_position[i] (causal) and j >= cache_position[i] - sliding_window
        let mut mask_data: Vec<f32> = Vec::with_capacity(query_length * key_length);

        // Use f32::MIN (finite large negative) like PyTorch's torch.finfo(dtype).min
        // This avoids 0 * -inf = NaN issues when combining masks
        let min_dtype = f32::MIN;

        for &abs_query_pos in cache_pos_vec.iter().take(query_length) {
            let abs_query_pos = abs_query_pos as usize;
            for j in 0..key_length {
                // Causal: can't attend to future positions
                let is_future = j > abs_query_pos;
                // Sliding window: can't attend to positions too far in the past
                let is_too_old = j + self.sliding_window < abs_query_pos;

                if is_future || is_too_old {
                    mask_data.push(min_dtype);
                } else {
                    mask_data.push(0.0);
                }
            }
        }

        let causal_mask = Tensor::from_slice(&mask_data, (query_length, key_length), &self.device)?;

        // If attention_mask is provided (2D), combine with causal mask
        // Matching PyTorch lines 722-728:
        //   padding_mask = causal_mask + attention_mask[:, None, None, :]
        //   padding_mask = padding_mask == 0
        //   causal_mask.masked_fill(padding_mask, min_dtype)
        if let Some(attn_mask) = attention_mask {
            // attn_mask is [batch, total_seq_len] where total_seq_len >= key_length
            // We need to slice it to [batch, key_length] if necessary
            let mask_len = attn_mask.dim(1)?;
            let attn_mask = if mask_len > key_length {
                attn_mask.narrow(1, 0, key_length)?
            } else {
                attn_mask.clone()
            };

            // Expand causal_mask to [1, 1, query_length, key_length]
            let causal_4d = causal_mask.unsqueeze(0)?.unsqueeze(0)?;

            // Expand attention_mask [batch, key_length] to [batch, 1, 1, key_length]
            // attention_mask: 1=attend, 0=mask (as f32)
            let attn_4d = attn_mask.to_dtype(DType::F32)?.unsqueeze(1)?.unsqueeze(2)?;

            // Add: causal (0=attend, MIN=mask) + attn (1=attend, 0=mask)
            // Result: 1=attend (causal ok, attn ok), 0=deny (causal ok, attn deny), MIN=mask (causal deny)
            let sum = causal_4d.broadcast_add(&attn_4d)?;

            // padding_mask = True where sum == 0 (causal allows but attention denies)
            let padding_mask = sum.eq(0.0)?;

            // Set those positions to MIN using where_cond (like PyTorch's masked_fill)
            // CRITICAL: Use causal_4d as fallback, NOT sum! PyTorch's masked_fill keeps original
            // causal values where padding_mask is False, not the sum of causal + attn.
            let min_val = Tensor::new(min_dtype, &self.device)?.broadcast_as(causal_4d.shape())?;
            let combined = padding_mask.where_cond(&min_val, &causal_4d)?;

            // Result is [batch, 1, query_length, key_length]
            combined.to_dtype(self.dtype)
        } else {
            // No attention mask, just expand causal mask
            causal_mask
                .unsqueeze(0)?
                .unsqueeze(0)?
                .expand((batch_size, 1, query_length, key_length))?
                .to_dtype(self.dtype)
        }
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.forward_with_cache_position(input_ids, seqlen_offset, attn_mask, None)
    }

    /// Forward pass with explicit cache_position tracking for selective attention.
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs [batch, seq_len]
    /// * `seqlen_offset` - Offset for RoPE and cache (number of cached tokens)
    /// * `attn_mask` - Optional 2D attention mask [batch, total_seq_len] for selective attention
    /// * `cache_position` - Optional absolute positions of input tokens for mask creation
    ///
    /// When `cache_position` is provided, uses `prepare_4d_causal_attention_mask_with_cache_position`
    /// which properly handles the 2D attention mask for cached decoding.
    pub fn forward_with_cache_position(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        attn_mask: Option<&Tensor>,
        cache_position: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let key_length = seqlen_offset + seq_len;

        // Create attention mask based on whether cache_position is provided
        let attention_mask: Option<Tensor> = match cache_position {
            Some(cache_pos) => {
                // Use the new function that properly handles 2D attention masks
                Some(self.prepare_4d_causal_attention_mask_with_cache_position(
                    attn_mask, seq_len, key_length, cache_pos, b_size,
                )?)
            }
            None => {
                // Legacy path: no cache_position, use old mask preparation
                match attn_mask {
                    Some(mask) => Some(self.prepare_attention_mask(mask)?),
                    None => {
                        Some(self.prepare_causal_attention_mask(b_size, seq_len, seqlen_offset)?)
                    }
                }
            }
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;

        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?;
        }
        xs.apply(&self.norm)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }

    /// Extract all KV caches from all layers
    pub fn extract_kv_cache(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.layers
            .iter()
            .map(|layer| layer.extract_kv_cache())
            .collect()
    }

    /// Restore KV caches to all layers
    pub fn restore_kv_cache(&mut self, caches: Vec<Option<(Tensor, Tensor)>>) {
        for (layer, cache) in self.layers.iter_mut().zip(caches.into_iter()) {
            layer.restore_kv_cache(cache);
        }
    }

    /// Shift KV cache: copy position 0 to last position for all layers
    /// This is used for negative prompt refresh in VibeVoice when SPEECH_START is generated
    pub fn shift_kv_cache_first_to_last(&mut self) -> Result<()> {
        let mut new_caches = Vec::new();

        for layer in &self.layers {
            if let Some((k_cache, v_cache)) = layer.extract_kv_cache() {
                // k_cache and v_cache shape: [batch, num_heads, seq_len, head_dim]
                let seq_len = k_cache.dim(2)?;
                if seq_len > 1 {
                    // Extract position 0: shape [batch, num_heads, head_dim]
                    let k_pos_0 = k_cache.i((.., .., 0, ..))?;
                    let v_pos_0 = v_cache.i((.., .., 0, ..))?;

                    // Add dimension: [batch, num_heads, 1, head_dim]
                    let k_pos_0 = k_pos_0.unsqueeze(2)?;
                    let v_pos_0 = v_pos_0.unsqueeze(2)?;

                    // Replace last position with pos 0
                    // Take positions [0, seq_len-1), then concat pos_0
                    let k_prefix = k_cache.narrow(2, 0, seq_len - 1)?;
                    let v_prefix = v_cache.narrow(2, 0, seq_len - 1)?;

                    let k_new = Tensor::cat(&[&k_prefix, &k_pos_0], 2)?;
                    let v_new = Tensor::cat(&[&v_prefix, &v_pos_0], 2)?;

                    new_caches.push(Some((k_new, v_new)));
                } else {
                    new_caches.push(Some((k_cache, v_cache)));
                }
            } else {
                new_caches.push(None);
            }
        }

        self.restore_kv_cache(new_caches);
        Ok(())
    }

    pub fn forward_from_embeds(
        &mut self,
        embeds: &Tensor,
        seqlen_offset: usize,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.forward_from_embeds_with_cache_position(embeds, seqlen_offset, attn_mask, None)
    }

    /// Forward pass from embeddings with explicit cache_position tracking.
    /// See `forward_with_cache_position` for details on the cache_position parameter.
    pub fn forward_from_embeds_with_cache_position(
        &mut self,
        embeds: &Tensor,
        seqlen_offset: usize,
        attn_mask: Option<&Tensor>,
        cache_position: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_size, seq_len, _hidden_size) = embeds.dims3()?;
        let key_length = seqlen_offset + seq_len;

        // Create attention mask based on whether cache_position is provided
        let attention_mask: Option<Tensor> = match cache_position {
            Some(cache_pos) => Some(self.prepare_4d_causal_attention_mask_with_cache_position(
                attn_mask, seq_len, key_length, cache_pos, b_size,
            )?),
            None => match attn_mask {
                Some(mask) => Some(self.prepare_attention_mask(mask)?),
                None => Some(self.prepare_causal_attention_mask(b_size, seq_len, seqlen_offset)?),
            },
        };

        // Use the provided embeddings directly instead of looking up tokens
        let mut xs = embeds.clone();

        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?;
        }
        xs.apply(&self.norm)
    }
}

#[derive(Debug, Clone)]
pub struct ModelForCausalLM {
    base_model: Model,
    lm_head: Linear,
}

impl ModelForCausalLM {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let base_model = Model::new(cfg, vb.clone())?;
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        } else {
            Linear::from_weights(base_model.embed_tokens.embeddings().clone(), None)
        };
        Ok(Self {
            base_model,
            lm_head,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;
        self.base_model
            .forward(input_ids, seqlen_offset, None)?
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.lm_head)
    }

    pub fn clear_kv_cache(&mut self) {
        self.base_model.clear_kv_cache()
    }
}
