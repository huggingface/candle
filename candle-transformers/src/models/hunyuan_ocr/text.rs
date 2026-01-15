//! Text decoder for HunyuanOCR.
//!
//! Contains:
//! - RotaryEmbedding: xDRoPE (Extended Dynamic Rotary Position Embedding)
//! - Attention: Multi-head attention with Q/K normalization
//! - Mlp: SwiGLU MLP
//! - DecoderLayer: Transformer decoder layer
//! - TextModel: Complete text decoder

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{embedding, linear_b, rms_norm, Embedding, Linear, Module, RmsNorm, VarBuilder};

use super::config::TextConfig;

// ============================================================================
// Cache
// ============================================================================

/// Cache for autoregressive generation.
///
/// Each layer stores its own (key, value) pair. The cache is passed mutably
/// to each forward call, following the candle-transformers convention.
pub struct Cache {
    /// KV cache for each layer: Vec<Option<(key, value)>>
    pub kvs: Vec<Option<(Tensor, Tensor)>>,
}

impl Cache {
    /// Create a new cache for the specified number of layers.
    pub fn new(num_layers: usize) -> Self {
        Self {
            kvs: vec![None; num_layers],
        }
    }

    /// Clear all cached key-value pairs.
    pub fn clear(&mut self) {
        for kv in self.kvs.iter_mut() {
            *kv = None;
        }
    }
}

// ============================================================================
// Rotary Embedding (xDRoPE)
// ============================================================================

/// HunyuanVL Rotary Embedding with xDRoPE support.
pub struct RotaryEmbedding {
    cos_cached: Tensor,
    sin_cached: Tensor,
    xdrope_section: Option<Vec<usize>>,
}

impl RotaryEmbedding {
    pub fn new(cfg: &TextConfig, device: &Device) -> Result<Self> {
        let dim = cfg.head_dim();
        let max_position_embeddings = cfg.max_position_embeddings;
        let base = cfg.rope_theta;

        // Calculate inv_freq with alpha scaling
        let adjusted_base = if let Some(ref rope_scaling) = cfg.rope_scaling {
            if let Some(alpha) = rope_scaling.alpha {
                base * alpha.powf(dim as f64 / (dim as f64 - 2.0))
            } else {
                base
            }
        } else {
            base
        };

        let inv_freq = Self::compute_inv_freq(dim, adjusted_base, device)?;
        let (cos_cached, sin_cached) =
            Self::compute_cos_sin_cache(max_position_embeddings, &inv_freq, device)?;

        let xdrope_section = cfg
            .rope_scaling
            .as_ref()
            .and_then(|s| s.xdrope_section.clone());

        Ok(Self {
            cos_cached,
            sin_cached,
            xdrope_section,
        })
    }

    fn compute_inv_freq(dim: usize, base: f64, device: &Device) -> Result<Tensor> {
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / base.powf(i as f64 / dim as f64) as f32)
            .collect();

        Tensor::from_vec(inv_freq, (dim / 2,), device)
    }

    fn compute_cos_sin_cache(
        max_position_embeddings: usize,
        inv_freq: &Tensor,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let t: Vec<f32> = (0..max_position_embeddings).map(|i| i as f32).collect();
        let t = Tensor::from_vec(t, (max_position_embeddings, 1), device)?;

        let inv_freq_expanded = inv_freq.unsqueeze(0)?;
        let freqs = t.broadcast_mul(&inv_freq_expanded)?;

        // emb = cat([freqs, freqs], dim=-1)
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;

        let cos_cached = emb.cos()?;
        let sin_cached = emb.sin()?;

        Ok((cos_cached, sin_cached))
    }

    /// Apply xDRoPE to query and key tensors (Prefill stage).
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch, num_heads, seq_len, head_dim]
    /// * `k` - Key tensor [batch, num_heads, seq_len, head_dim]
    /// * `position_ids` - Position IDs [batch, x_dim, seq_len]
    pub fn apply_xdrope(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_ids: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (batch, _num_heads, _seq_len, _head_dim) = q.dims4()?;

        if let Some(ref xdrope_section) = self.xdrope_section {
            let x_dim = xdrope_section.len();

            // Validate position_ids shape
            let pos_dims = position_ids.dims();
            if pos_dims.len() != 3 {
                candle::bail!(
                    "position_ids must be 3D [batch, x_dim, seq_len], got shape {:?}",
                    pos_dims
                );
            }
            if pos_dims[1] != x_dim {
                candle::bail!(
                    "position_ids dim 1 must equal x_dim ({}), got {}",
                    x_dim,
                    pos_dims[1]
                );
            }

            // Step 1: Index cos/sin based on multi-dim position_ids
            let mut cos_parts = Vec::new();
            let mut sin_parts = Vec::new();

            for b in 0..batch {
                let mut cos_dims = Vec::new();
                let mut sin_dims = Vec::new();

                for d in 0..x_dim {
                    let pos_ids_bd = position_ids.i((b, d))?;
                    let pos_ids_bd = if pos_ids_bd.dtype() != DType::I64 {
                        pos_ids_bd.to_dtype(DType::I64)?
                    } else {
                        pos_ids_bd
                    };

                    let cos_bd = self.cos_cached.index_select(&pos_ids_bd, 0)?;
                    let sin_bd = self.sin_cached.index_select(&pos_ids_bd, 0)?;

                    cos_dims.push(cos_bd);
                    sin_dims.push(sin_bd);
                }

                let cos_b = Tensor::stack(&cos_dims, 0)?;
                let sin_b = Tensor::stack(&sin_dims, 0)?;

                cos_parts.push(cos_b);
                sin_parts.push(sin_b);
            }

            let cos = Tensor::stack(&cos_parts, 0)?;
            let sin = Tensor::stack(&sin_parts, 0)?;

            // Step 2: permute(0, 2, 1, 3) -> [batch, seq_len, x_dim, head_dim]
            let cos = cos.permute((0, 2, 1, 3))?;
            let sin = sin.permute((0, 2, 1, 3))?;

            let cos = cos.contiguous()?;
            let sin = sin.contiguous()?;

            // Step 3: Apply xdrope_section transform
            let (cos, sin) = self.apply_xdrope_section_transform(cos, sin, xdrope_section)?;

            // Step 4: Expand to num_heads dimension
            let cos = cos.unsqueeze(1)?;
            let sin = sin.unsqueeze(1)?;

            // Step 5: Apply rotation
            let q_out = apply_rotary_emb(q, &cos, &sin)?;
            let k_out = apply_rotary_emb(k, &cos, &sin)?;

            Ok((q_out, k_out))
        } else {
            // Standard RoPE (no xdrope_section)
            let pos_dims = position_ids.dims();
            let position_ids_2d = if pos_dims.len() == 3 {
                position_ids.i((.., 0))?
            } else if pos_dims.len() == 2 {
                position_ids.clone()
            } else {
                candle::bail!("position_ids must be 2D or 3D, got shape {:?}", pos_dims);
            };

            let mut cos_parts = Vec::new();
            let mut sin_parts = Vec::new();

            for b in 0..batch {
                let pos_ids_b = position_ids_2d.i(b)?;
                let pos_ids_b = if pos_ids_b.dtype() != DType::I64 {
                    pos_ids_b.to_dtype(DType::I64)?
                } else {
                    pos_ids_b
                };

                let cos_b = self.cos_cached.index_select(&pos_ids_b, 0)?;
                let sin_b = self.sin_cached.index_select(&pos_ids_b, 0)?;
                cos_parts.push(cos_b);
                sin_parts.push(sin_b);
            }

            let cos = Tensor::stack(&cos_parts, 0)?;
            let sin = Tensor::stack(&sin_parts, 0)?;

            let cos = cos.unsqueeze(1)?;
            let sin = sin.unsqueeze(1)?;

            let q_out = apply_rotary_emb(q, &cos, &sin)?;
            let k_out = apply_rotary_emb(k, &cos, &sin)?;

            Ok((q_out, k_out))
        }
    }

    /// Apply xdrope_section dimension reorganization.
    ///
    /// Input: cos/sin shape [batch, seq_len, x_dim, head_dim]
    /// Output: cos/sin shape [batch, seq_len, head_dim]
    fn apply_xdrope_section_transform(
        &self,
        cos: Tensor,
        sin: Tensor,
        xdrope_section: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let (_batch, _seq_len, x_dim, _last_dim) = cos.dims4()?;

        // Step 1: Calculate xdrope_section_doubled
        let xdrope_section_doubled: Vec<usize> = xdrope_section.iter().map(|&x| x * 2).collect();

        // Step 2: Split by xdrope_section_doubled and reorganize
        let mut cos_parts = Vec::new();
        let mut sin_parts = Vec::new();
        let mut start_idx = 0;

        for (i, &section_size) in xdrope_section_doubled.iter().enumerate() {
            let dim_idx = i % x_dim;

            // Extract section from cos/sin
            let cos_section = cos.narrow(3, start_idx, section_size)?;
            let sin_section = sin.narrow(3, start_idx, section_size)?;

            // Select the dim_idx dimension
            let cos_part = cos_section.i((.., .., dim_idx))?;
            let sin_part = sin_section.i((.., .., dim_idx))?;

            cos_parts.push(cos_part);
            sin_parts.push(sin_part);
            start_idx += section_size;
        }

        // Step 3: Concatenate all parts
        let cos_out = Tensor::cat(&cos_parts.iter().collect::<Vec<_>>(), D::Minus1)?;
        let sin_out = Tensor::cat(&sin_parts.iter().collect::<Vec<_>>(), D::Minus1)?;

        Ok((cos_out, sin_out))
    }

    /// Apply standard RoPE (Decode stage).
    pub fn apply_standard_rope(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_id: i64,
    ) -> Result<(Tensor, Tensor)> {
        let cos = self.cos_cached.get(position_id as usize)?;
        let sin = self.sin_cached.get(position_id as usize)?;

        let cos = cos.unsqueeze(0)?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?.unsqueeze(0)?;

        let q_out = apply_rotary_emb(q, &cos, &sin)?;
        let k_out = apply_rotary_emb(k, &cos, &sin)?;

        Ok((q_out, k_out))
    }
}

/// Apply rotary embedding: x * cos + rotate_half(x) * sin
fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let original_dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let cos_f32 = cos.to_dtype(DType::F32)?;
    let sin_f32 = sin.to_dtype(DType::F32)?;

    let x_rotated = rotate_half(&x_f32)?;
    let result = (x_f32.broadcast_mul(&cos_f32)? + x_rotated.broadcast_mul(&sin_f32)?)?;

    result.to_dtype(original_dtype)
}

/// Rotate half of the tensor: cat([-x[..., half:], x[..., :half]], dim=-1)
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let head_dim = x.dim(D::Minus1)?;
    let half_dim = head_dim / 2;

    let x1 = x.narrow(D::Minus1, 0, half_dim)?;
    let x2 = x.narrow(D::Minus1, half_dim, half_dim)?;

    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

// ============================================================================
// Attention
// ============================================================================

/// Attention layer with Q/K normalization and Flash Attention support.
struct Attention {
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    n_kv_groups: usize,
    softmax_scale: f64,
    use_flash_attn: bool,

    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,

    query_layernorm: RmsNorm,
    key_layernorm: RmsNorm,

    rotary_emb: Arc<RotaryEmbedding>,
}

impl Attention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &TextConfig,
        use_flash_attn: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();

        // HunyuanOCR uses no bias (attention_bias=false)
        let q_proj = linear_b(hidden_size, num_heads * head_dim, false, vb.pp("q_proj"))?;
        let k_proj = linear_b(hidden_size, num_kv_heads * head_dim, false, vb.pp("k_proj"))?;
        let v_proj = linear_b(hidden_size, num_kv_heads * head_dim, false, vb.pp("v_proj"))?;
        let o_proj = linear_b(num_heads * head_dim, hidden_size, false, vb.pp("o_proj"))?;

        // Q/K LayerNorms (unique to HunyuanOCR)
        let query_layernorm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("query_layernorm"))?;
        let key_layernorm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("key_layernorm"))?;

        Ok(Self {
            num_heads,
            num_kv_heads,
            head_dim,
            n_kv_groups: num_heads / num_kv_heads,
            softmax_scale: 1.0 / (head_dim as f64).sqrt(),
            use_flash_attn,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            query_layernorm,
            key_layernorm,
            rotary_emb,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = xs.dims3()?;

        // Q/K/V projections
        let mut q = self.q_proj.forward(xs)?;
        let mut k = self.k_proj.forward(xs)?;
        let mut v = self.v_proj.forward(xs)?;

        // Reshape to multi-head format
        q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        k = k
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        v = v
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply RoPE (xDRoPE for prefill, standard for decode)
        if seqlen_offset == 0 {
            (q, k) = self.rotary_emb.apply_xdrope(&q, &k, position_ids)?;
        } else {
            let position_id = seqlen_offset as i64;
            (q, k) = self.rotary_emb.apply_standard_rope(&q, &k, position_id)?;
        }

        // Q/K Normalization (HunyuanOCR specific)
        q = q.apply(&self.query_layernorm)?;
        k = k.apply(&self.key_layernorm)?;

        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        // Update KV cache (following candle-transformers convention)
        let (k, v) = if let Some((prev_k, prev_v)) = kv_cache.as_ref() {
            let k = Tensor::cat(&[prev_k, &k], 2)?.contiguous()?;
            let v = Tensor::cat(&[prev_v, &v], 2)?.contiguous()?;
            (k, v)
        } else {
            (k, v)
        };
        *kv_cache = Some((k.clone(), v.clone()));

        // Compute attention
        let attn_output =
            self.compute_attention(&q, &k, &v, attention_mask, self.use_flash_attn)?;

        // Reshape and output projection
        let attn_output = attn_output.transpose(1, 2)?.reshape((
            b_sz,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        self.o_proj.forward(&attn_output)
    }

    fn compute_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
        #[allow(unused_variables)] use_flash_attn: bool,
    ) -> Result<Tensor> {
        let seq_len = q.dim(2)?;

        // Try Flash Attention on CUDA if enabled
        #[cfg(feature = "flash-attn")]
        if use_flash_attn {
            return self.flash_attn_cuda(q, k, v, seq_len > 1);
        }

        // Try SDPA on Metal for decode (seq_len == 1)
        if q.device().is_metal() && seq_len == 1 {
            return self.sdpa_metal(q, k, v);
        }

        // Standard attention fallback
        self.compute_standard_attention(q, k, v, attention_mask)
    }

    #[cfg(feature = "flash-attn")]
    fn flash_attn_cuda(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        causal: bool,
    ) -> Result<Tensor> {
        // Repeat KV for GQA before flash attention
        let k = crate::utils::repeat_kv(k.clone(), self.n_kv_groups)?.contiguous()?;
        let v = crate::utils::repeat_kv(v.clone(), self.n_kv_groups)?.contiguous()?;

        // candle-flash-attn expects (batch, seq_len, num_heads, head_dim)
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let softmax_scale = self.softmax_scale as f32;
        let output = candle_flash_attn::flash_attn(&q, &k, &v, softmax_scale, causal)?;

        // Transpose back to (batch, num_heads, seq_len, head_dim)
        output.transpose(1, 2)
    }

    /// SDPA on Metal - efficient for decode phase (seq_len == 1)
    fn sdpa_metal(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        // SDPA handles MQA/GQA internally on Metal
        candle_nn::ops::sdpa(
            q,
            k,
            v,
            None,
            true, // causal
            self.softmax_scale as f32,
            1.0,
        )
    }

    fn compute_standard_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Repeat KV for GQA
        let k = crate::utils::repeat_kv(k.clone(), self.n_kv_groups)?.contiguous()?;
        let v = crate::utils::repeat_kv(v.clone(), self.n_kv_groups)?.contiguous()?;

        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * self.softmax_scale)?;

        let attn_weights = match attention_mask {
            Some(mask) => attn_weights.broadcast_add(mask)?,
            None => attn_weights,
        };

        // Softmax in F32 for numerical stability
        let original_dtype = attn_weights.dtype();
        let attn_weights = if original_dtype != DType::F32 {
            let attn_f32 = attn_weights.to_dtype(DType::F32)?;
            candle_nn::ops::softmax_last_dim(&attn_f32)?.to_dtype(original_dtype)?
        } else {
            candle_nn::ops::softmax_last_dim(&attn_weights)?
        };

        attn_weights.matmul(&v)
    }
}

// ============================================================================
// MLP
// ============================================================================

/// SwiGLU MLP layer.
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn new(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let intermediate_size = cfg.intermediate_size;

        // HunyuanOCR MLP uses no bias
        let gate_proj = linear_b(hidden_size, intermediate_size, false, vb.pp("gate_proj"))?;
        let up_proj = linear_b(hidden_size, intermediate_size, false, vb.pp("up_proj"))?;
        let down_proj = linear_b(intermediate_size, hidden_size, false, vb.pp("down_proj"))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ============================================================================
// Decoder Layer
// ============================================================================

/// Transformer decoder layer.
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &TextConfig,
        use_flash_attn: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;

        let self_attn = Attention::new(rotary_emb, cfg, use_flash_attn, vb.pp("self_attn"))?;

        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        // Pre-norm with residual
        let residual = xs.clone();
        let xs = self.input_layernorm.forward(xs)?;
        let xs =
            self.self_attn
                .forward(&xs, attention_mask, position_ids, seqlen_offset, kv_cache)?;
        let xs = (xs + residual)?;

        let residual = xs.clone();
        let xs = self
            .mlp
            .forward(&xs.apply(&self.post_attention_layernorm)?)?;
        xs + residual
    }
}

// ============================================================================
// Text Model
// ============================================================================

/// Complete text decoder model.
pub struct TextModel {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    /// Hidden size of the model.
    #[allow(dead_code)]
    pub hidden_size: usize,
    /// Data type of the model.
    #[allow(dead_code)]
    pub dtype: DType,
    num_layers: usize,
}

impl TextModel {
    pub fn new(cfg: &TextConfig, use_flash_attn: bool, vb: VarBuilder) -> Result<Self> {
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;

        let rotary_emb = Arc::new(RotaryEmbedding::new(cfg, vb.device())?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                use_flash_attn,
                vb.pp(format!("layers.{}", i)),
            )?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;

        // LM head may be tied to embedding weights
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            linear_b(cfg.hidden_size, cfg.vocab_size, false, vb.pp("lm_head"))?
        } else {
            // Tied weights
            Linear::new(embed_tokens.embeddings().clone(), None)
        };

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            hidden_size: cfg.hidden_size,
            dtype: vb.dtype(),
            num_layers: cfg.num_hidden_layers,
        })
    }

    /// Get the number of layers (for cache initialization).
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn forward(
        &self,
        input_embeds: Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
        seqlen_offset: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let mut xs = input_embeds;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(
                &xs,
                attention_mask,
                position_ids,
                seqlen_offset,
                &mut cache.kvs[layer_idx],
            )?;
        }

        xs = xs.apply(&self.norm)?;
        self.lm_head.forward(&xs)
    }
}
