//! Qwen3-ASR model: audio encoder + text decoder + multimodal RoPE.
//!
//! Based on alan890104/qwen3-asr-rs (MIT).

use super::{
    audio_encoder::AudioEncoder,
    kv_cache::KVCache,
    rope::{apply_multimodal_rotary_pos_emb, MultimodalRotaryEmbedding},
    Config, TextConfig,
};
use crate::models::with_tracing::{linear_b, linear_no_bias, Linear};
use candle::{DType, Device, Result, Tensor};
use candle_nn::{embedding, rms_norm, Embedding, Module, RmsNorm, VarBuilder};

#[cfg(feature = "flash-attn")]
use candle_flash_attn::{flash_attn, flash_attn_varlen};

// ── Attention helpers ────────────────────────────────────────────────

pub fn repeat_kv(hidden_states: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(hidden_states.clone());
    }
    let (batch, kv_heads, seq_len, head_dim) = hidden_states.dims4()?;
    let expanded = hidden_states.unsqueeze(2)?;
    let expanded = expanded.broadcast_as((batch, kv_heads, n_rep, seq_len, head_dim))?;
    expanded.reshape((batch, kv_heads * n_rep, seq_len, head_dim))
}

pub fn make_causal_mask_cached(
    attention_mask: Option<&Tensor>,
    batch_size: usize,
    cache_len: usize,
    new_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let total_len = cache_len.saturating_add(new_len);
    if new_len == 0 {
        return Tensor::zeros((batch_size, 1usize, 0usize, total_len), dtype, device);
    }
    let total_len_u32 = u32::try_from(total_len)
        .map_err(|_| candle::Error::Msg("total_len overflows u32".to_string()))?;
    let cache_len_u32 = u32::try_from(cache_len)
        .map_err(|_| candle::Error::Msg("cache_len overflows u32".to_string()))?;
    let q_end = cache_len_u32
        .checked_add(
            u32::try_from(new_len)
                .map_err(|_| candle::Error::Msg("new_len overflows u32".to_string()))?,
        )
        .ok_or_else(|| candle::Error::Msg("cache_len + new_len overflows u32".to_string()))?;
    let kv = Tensor::arange(0u32, total_len_u32, device)?;
    let q = Tensor::arange(cache_len_u32, q_end, device)?;
    let kv = kv.unsqueeze(0)?.broadcast_as((new_len, total_len))?;
    let q = q.unsqueeze(1)?.broadcast_as((new_len, total_len))?;
    let allowed = kv.le(&q)?;
    let allowed = allowed
        .unsqueeze(0)?
        .broadcast_as((batch_size, new_len, total_len))?;
    let allowed = match attention_mask {
        None => allowed,
        Some(m) => {
            let (b, s) = m.dims2()?;
            if b != batch_size || s != total_len {
                candle::bail!(
                    "attention_mask dims mismatch: expected=({batch_size},{total_len}), got=({b},{s})"
                );
            }
            let key_padding = m.ne(0u32)?;
            let key_padding = key_padding
                .unsqueeze(1)?
                .broadcast_as((batch_size, new_len, total_len))?;
            (&allowed * &key_padding)?
        }
    };
    let allowed = allowed.unsqueeze(1)?;
    let shape = (batch_size, 1usize, new_len, total_len);
    let zeros = Tensor::zeros(shape, DType::F32, device)?;
    let neg = Tensor::full(f32::MIN, shape, device)?;
    let mask = allowed.where_cond(&zeros, &neg)?;
    if dtype == DType::F32 {
        Ok(mask)
    } else {
        mask.to_dtype(dtype)
    }
}

// ── Flash-attn helpers ───────────────────────────────────────────────

#[cfg(feature = "flash-attn")]
fn seqlens_from_left_padded_attention_mask(mask: &Tensor, seq_len: usize) -> Result<Vec<usize>> {
    let (batch, t2) = mask.dims2()?;
    if t2 != seq_len {
        candle::bail!("attention_mask seq_len mismatch: expected={seq_len}, got={t2}");
    }
    let mask_u8 = mask.ne(0u32)?;
    let lens_f32 = mask_u8.to_dtype(DType::F32)?.sum(1)?;
    let lens = lens_f32.to_vec1::<f32>()?;
    if lens.len() != batch {
        candle::bail!("attention_mask lens count mismatch");
    }
    let mut out: Vec<usize> = Vec::with_capacity(batch);
    for (i, v) in lens.into_iter().enumerate() {
        if !v.is_finite() || v < 0.0 {
            candle::bail!("invalid attention_mask length at {i}: {v}");
        }
        let len = v.round() as usize;
        out.push(len);
    }
    validate_left_padded(mask, seq_len, &out).map_err(|msg| candle::Error::Msg(msg.to_string()))?;
    Ok(out)
}

#[cfg(feature = "flash-attn")]
fn validate_left_padded(
    mask: &Tensor,
    seq_len: usize,
    lengths: &[usize],
) -> std::result::Result<(), String> {
    let mask_data = mask.to_vec2::<u32>().map_err(|e| format!("{e:?}"))?;
    for (b, (&len, row)) in lengths.iter().zip(mask_data.iter()).enumerate() {
        let pad = seq_len - len;
        for j in 0..pad {
            if row[j] != 0 {
                return Err(format!(
                    "attention_mask at batch {b} position {j}: expected 0 for left-padding, got {}; \
                     this helper requires a left-padded mask (valid tokens at the right)",
                    row[j]
                ));
            }
        }
        if len > 0 && row[pad] == 0 {
            return Err(format!(
                "attention_mask at batch {b} position {pad}: expected non-zero first valid token \
                 for left-padding; this helper requires a left-padded mask"
            ));
        }
    }
    Ok(())
}

#[cfg(feature = "flash-attn")]
fn cu_seqlens_u32(lengths: &[usize], device: &Device) -> Result<(Tensor, usize, u32)> {
    let mut cu: Vec<u32> = Vec::with_capacity(lengths.len() + 1);
    cu.push(0);
    let mut total: u32 = 0;
    let mut max_len: usize = 0;
    for (i, &len) in lengths.iter().enumerate() {
        let len_u32 = u32::try_from(len)
            .map_err(|_| candle::Error::Msg(format!("seq len overflows u32 at {i}")))?;
        total = total
            .checked_add(len_u32)
            .ok_or_else(|| candle::Error::Msg(format!("cu overflow at {i}")))?;
        cu.push(total);
        max_len = max_len.max(len);
    }
    let cu_t = Tensor::from_vec(cu, (lengths.len() + 1,), device)?;
    Ok((cu_t, max_len, total))
}

#[cfg(feature = "flash-attn")]
fn left_pad_indices_u32(seq_len: usize, lengths: &[usize]) -> Result<Vec<u32>> {
    let batch = lengths.len();
    let total: usize = lengths.iter().sum();
    let mut idxs: Vec<u32> = Vec::with_capacity(total);
    for (b, &len) in lengths.iter().enumerate() {
        if len > seq_len {
            candle::bail!("sequence length exceeds seq_len for batch {b}");
        }
        let pad = seq_len - len;
        let base = b * seq_len;
        for j in 0..len {
            idxs.push(
                u32::try_from(base + pad + j)
                    .map_err(|_| candle::Error::Msg("index overflows u32".to_string()))?,
            );
        }
    }
    Ok(idxs)
}

// ── Rotary Embedding ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct TextRotaryEmbedding {
    rope: MultimodalRotaryEmbedding,
    mrope_section: Vec<usize>,
    interleaved: bool,
    interleave_masks: Option<[Tensor; 3]>,
}

impl TextRotaryEmbedding {
    fn load(cfg: &TextConfig, device: &Device) -> Result<Self> {
        let rope = MultimodalRotaryEmbedding::new(cfg.head_dim, cfg.rope_theta, device)?;
        let (mrope_section, interleaved) = match cfg.rope_scaling.as_ref() {
            None => (vec![24usize, 20, 20], true),
            Some(s) => (
                if s.mrope_section.is_empty() {
                    vec![24usize, 20, 20]
                } else {
                    s.mrope_section.clone()
                },
                s.mrope_interleaved || s.interleaved,
            ),
        };
        let interleave_masks = if interleaved && mrope_section.len() == 3 {
            let half_dim = cfg.head_dim / 2;
            let modality_num = mrope_section.len();
            let m1_end = (mrope_section[1] * modality_num).min(half_dim);
            let m2_end = (mrope_section[2] * modality_num).min(half_dim);
            let mut mask_m0 = vec![1.0f32; half_dim];
            let mut mask_m1 = vec![0.0f32; half_dim];
            let mut mask_m2 = vec![0.0f32; half_dim];
            for pos in 0..half_dim {
                if pos >= 1 && pos < m1_end && (pos - 1) % modality_num == 0 {
                    mask_m0[pos] = 0.0;
                    mask_m1[pos] = 1.0;
                } else if pos >= 2 && pos < m2_end && (pos - 2) % modality_num == 0 {
                    mask_m0[pos] = 0.0;
                    mask_m2[pos] = 1.0;
                }
            }
            let mask_m0 = Tensor::from_vec(mask_m0, (1, 1, half_dim), device)?;
            let mask_m1 = Tensor::from_vec(mask_m1, (1, 1, half_dim), device)?;
            let mask_m2 = Tensor::from_vec(mask_m2, (1, 1, half_dim), device)?;
            Some([mask_m0, mask_m1, mask_m2])
        } else {
            None
        };
        Ok(Self {
            rope,
            mrope_section,
            interleaved,
            interleave_masks,
        })
    }

    fn forward(&self, x: &Tensor, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        self.rope.forward(x, position_ids)
    }
}

// ── MLP ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct TextMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl TextMLP {
    fn load(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
        })
    }
}

impl Module for TextMLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(xs)?)?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&gate.broadcast_mul(&up)?)
    }
}

// ── Attention ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct TextAttention {
    use_flash_attn: bool,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_key_value_groups: usize,
    head_dim: usize,
    scaling: f64,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
}

impl TextAttention {
    fn load(cfg: &TextConfig, vb: VarBuilder, use_flash_attn: bool) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let num_attention_heads = cfg.num_attention_heads;
        let num_key_value_heads = cfg.num_key_value_heads;
        if num_key_value_heads == 0 {
            candle::bail!("num_key_value_heads must be > 0");
        }
        if !num_attention_heads.is_multiple_of(num_key_value_heads) {
            candle::bail!(
                "num_attention_heads ({num_attention_heads}) must be divisible by num_key_value_heads ({num_key_value_heads})"
            );
        }
        let num_key_value_groups = num_attention_heads / num_key_value_heads;
        let q_out = num_attention_heads * head_dim;
        let kv_out = num_key_value_heads * head_dim;
        Ok(Self {
            use_flash_attn,
            num_attention_heads,
            num_key_value_heads,
            num_key_value_groups,
            head_dim,
            scaling: (head_dim as f64).powf(-0.5),
            q_proj: linear_b(cfg.hidden_size, q_out, cfg.attention_bias, vb.pp("q_proj"))?,
            k_proj: linear_b(cfg.hidden_size, kv_out, cfg.attention_bias, vb.pp("k_proj"))?,
            v_proj: linear_b(cfg.hidden_size, kv_out, cfg.attention_bias, vb.pp("v_proj"))?,
            o_proj: linear_b(q_out, cfg.hidden_size, cfg.attention_bias, vb.pp("o_proj"))?,
            q_norm: rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_with_kv_cache(
        &self,
        hidden_states: &Tensor,
        position_embeddings: (&Tensor, &Tensor),
        attention_mask: Option<&Tensor>,
        token_attention_mask: &Tensor,
        rope: &TextRotaryEmbedding,
        kv_cache: &mut KVCache,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let (batch, seq_len, _hidden) = hidden_states.dims3()?;
        let q = self.q_proj.forward(hidden_states)?;
        let q = q.reshape((batch, seq_len, self.num_attention_heads, self.head_dim))?;
        let q = self.q_norm.forward(&q)?.transpose(1, 2)?;
        let k = self.k_proj.forward(hidden_states)?;
        let k = k.reshape((batch, seq_len, self.num_key_value_heads, self.head_dim))?;
        let k = self.k_norm.forward(&k)?.transpose(1, 2)?;
        let v = self.v_proj.forward(hidden_states)?;
        let v = v.reshape((batch, seq_len, self.num_key_value_heads, self.head_dim))?;
        let v = v.transpose(1, 2)?;

        let (cos, sin) = position_embeddings;
        let (q, k) = apply_multimodal_rotary_pos_emb(
            &q,
            &k,
            cos,
            sin,
            rope.mrope_section.as_slice(),
            rope.interleaved,
            rope.interleave_masks.as_ref(),
        )?;

        let (k, v) = kv_cache.update(layer_idx, &k, &v)?;

        // Flash-attn cached path
        if self.use_flash_attn && attention_mask.is_none() {
            #[cfg(not(feature = "flash-attn"))]
            {
                let _ = (token_attention_mask, kv_cache, layer_idx);
                candle::bail!("flash-attn not enabled in this build");
            }
            #[cfg(feature = "flash-attn")]
            {
                let softmax_scale = self.scaling as f32;
                let (_b2, total_len) = token_attention_mask.dims2()?;
                let cache_len = total_len - seq_len;
                let q4 = q.transpose(1, 2)?.contiguous()?;
                let k4 = k.transpose(1, 2)?.contiguous()?;
                let v4 = v.transpose(1, 2)?.contiguous()?;
                let out = if cache_len == 0 {
                    let seqlens =
                        seqlens_from_left_padded_attention_mask(token_attention_mask, seq_len)?;
                    let (cu, max_len, total) = cu_seqlens_u32(&seqlens, hidden_states.device())?;
                    let flat_total = batch * seq_len;
                    let total_usize = usize::try_from(total)
                        .map_err(|_| candle::Error::Msg("total overflows usize".to_string()))?;
                    if total_usize == flat_total {
                        flash_attn(&q4, &k4, &v4, softmax_scale, true)?
                    } else {
                        let idxs = left_pad_indices_u32(seq_len, &seqlens)?;
                        let idx = Tensor::from_vec(idxs, (total_usize,), hidden_states.device())?;
                        let q_flat =
                            q4.reshape((flat_total, self.num_attention_heads, self.head_dim))?;
                        let k_flat =
                            k4.reshape((flat_total, self.num_key_value_heads, self.head_dim))?;
                        let v_flat =
                            v4.reshape((flat_total, self.num_key_value_heads, self.head_dim))?;
                        let q_unpad = q_flat.index_select(&idx, 0)?.contiguous()?;
                        let k_unpad = k_flat.index_select(&idx, 0)?.contiguous()?;
                        let v_unpad = v_flat.index_select(&idx, 0)?.contiguous()?;
                        let out_unpad = flash_attn_varlen(
                            &q_unpad,
                            &k_unpad,
                            &v_unpad,
                            &cu,
                            &cu,
                            max_len,
                            max_len,
                            softmax_scale,
                            true,
                        )?;
                        let zeros = Tensor::zeros(
                            (flat_total, self.num_attention_heads, self.head_dim),
                            out_unpad.dtype(),
                            out_unpad.device(),
                        )?;
                        let flat = zeros.index_add(&idx, &out_unpad, 0)?;
                        flat.reshape((batch, seq_len, self.num_attention_heads, self.head_dim))?
                    }
                } else {
                    let k_total = batch * total_len;
                    let q_total = batch * seq_len;
                    let q3 = q4.reshape((q_total, self.num_attention_heads, self.head_dim))?;
                    let seqlens_k =
                        seqlens_from_left_padded_attention_mask(token_attention_mask, total_len)?;
                    let (cu_k, max_k, total_k) =
                        cu_seqlens_u32(&seqlens_k, hidden_states.device())?;
                    let total_k = usize::try_from(total_k)
                        .map_err(|_| candle::Error::Msg("total_k overflows usize".to_string()))?;
                    let seqlens_q = vec![seq_len; batch];
                    let (cu_q, max_q, _) = cu_seqlens_u32(&seqlens_q, hidden_states.device())?;
                    let idxs_k = left_pad_indices_u32(total_len, &seqlens_k)?;
                    let idx_k = Tensor::from_vec(idxs_k, (total_k,), hidden_states.device())?;
                    let k3 = k4
                        .reshape((k_total, self.num_key_value_heads, self.head_dim))?
                        .index_select(&idx_k, 0)?
                        .contiguous()?;
                    let v3 = v4
                        .reshape((k_total, self.num_key_value_heads, self.head_dim))?
                        .index_select(&idx_k, 0)?
                        .contiguous()?;
                    flash_attn_varlen(
                        &q3,
                        &k3,
                        &v3,
                        &cu_q,
                        &cu_k,
                        max_q,
                        max_k,
                        softmax_scale,
                        true,
                    )?
                };
                let out =
                    out.reshape((batch, seq_len, self.num_attention_heads * self.head_dim))?;
                return self.o_proj.forward(&out);
            }
        }

        // Eager cached path
        let k = repeat_kv(&k, self.num_key_value_groups)?;
        let v = repeat_kv(&v, self.num_key_value_groups)?;
        let k_t = k.transpose(2, 3)?;
        let mut attn_weights = q.matmul(&k_t)?.affine(self.scaling, 0.0)?;
        if let Some(mask) = attention_mask {
            attn_weights = attn_weights.broadcast_add(mask)?;
        }
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;
        let attn_output = attn_output.transpose(1, 2)?;
        let attn_output =
            attn_output.reshape((batch, seq_len, self.num_attention_heads * self.head_dim))?;
        self.o_proj.forward(&attn_output)
    }
}

// ── Decoder Layer ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct TextDecoderLayer {
    self_attn: TextAttention,
    mlp: TextMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl TextDecoderLayer {
    fn load(cfg: &TextConfig, vb: VarBuilder, use_flash_attn: bool) -> Result<Self> {
        Ok(Self {
            self_attn: TextAttention::load(cfg, vb.pp("self_attn"), use_flash_attn)?,
            mlp: TextMLP::load(cfg, vb.pp("mlp"))?,
            input_layernorm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attention_layernorm: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_with_kv_cache(
        &self,
        hidden_states: &Tensor,
        position_embeddings: (&Tensor, &Tensor),
        attention_mask: Option<&Tensor>,
        token_attention_mask: &Tensor,
        rope: &TextRotaryEmbedding,
        kv_cache: &mut KVCache,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let x = self.input_layernorm.forward(hidden_states)?;
        let x = self.self_attn.forward_with_kv_cache(
            &x,
            position_embeddings,
            attention_mask,
            token_attention_mask,
            rope,
            kv_cache,
            layer_idx,
        )?;
        let x = (&residual + &x)?;
        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        &residual + &x
    }
}

// ── Text Model ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TextModel {
    embed_tokens: Embedding,
    layers: Vec<TextDecoderLayer>,
    norm: RmsNorm,
    rotary_emb: TextRotaryEmbedding,
    pub hidden_size: usize,
    use_flash_attn: bool,
}

impl TextModel {
    pub fn load(
        cfg: &TextConfig,
        vb: VarBuilder,
        device: &Device,
        use_flash_attn: bool,
    ) -> Result<Self> {
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(TextDecoderLayer::load(
                cfg,
                vb.pp("layers").pp(i.to_string()),
                use_flash_attn,
            )?);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let rotary_emb = TextRotaryEmbedding::load(cfg, device)?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            hidden_size: cfg.hidden_size,
            use_flash_attn,
        })
    }

    pub fn embed_tokens_weight(&self) -> &Tensor {
        self.embed_tokens.embeddings()
    }

    pub fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn forward_with_kv_cache(
        &self,
        attention_mask: &Tensor,
        position_ids: &Tensor,
        inputs_embeds: &Tensor,
        kv_cache: &mut KVCache,
    ) -> Result<Tensor> {
        let (batch, seq_len, hidden) = inputs_embeds.dims3()?;
        if hidden != self.hidden_size {
            candle::bail!(
                "inputs_embeds hidden mismatch: expected={}, got={hidden}",
                self.hidden_size
            );
        }
        let (b2, total_len) = attention_mask.dims2()?;
        if b2 != batch {
            candle::bail!("attention_mask batch mismatch");
        }
        let cache_len = kv_cache.seq_len();
        if total_len != cache_len.saturating_add(seq_len) {
            candle::bail!(
                "attention_mask total_len mismatch: total_len={total_len} cache_len={cache_len} new_len={seq_len}"
            );
        }
        let device = inputs_embeds.device();
        let dtype = inputs_embeds.dtype();
        let causal_mask = if self.use_flash_attn {
            None
        } else {
            Some(make_causal_mask_cached(
                Some(attention_mask),
                batch,
                cache_len,
                seq_len,
                dtype,
                device,
            )?)
        };
        let (cos, sin) = self.rotary_emb.forward(inputs_embeds, position_ids)?;
        let pos_emb = (&cos, &sin);
        let mut hidden_states = inputs_embeds.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward_with_kv_cache(
                &hidden_states,
                pos_emb,
                causal_mask.as_ref(),
                attention_mask,
                &self.rotary_emb,
                kv_cache,
                i,
            )?;
        }
        self.norm.forward(&hidden_states)
    }
}

// ── Position and audio feature helpers ───────────────────────────────

pub fn get_rope_index(attention_mask: &Tensor) -> Result<Tensor> {
    let (batch, seq_len) = attention_mask.dims2()?;
    if seq_len == 0 {
        candle::bail!("attention_mask seq_len must be > 0");
    }
    let device = attention_mask.device();
    let seq_len_u32 = u32::try_from(seq_len)
        .map_err(|_| candle::Error::Msg("seq_len overflows u32".to_string()))?;
    let mask_u8 = attention_mask.ne(0u32)?;
    let mask_f32 = mask_u8.to_dtype(DType::F32)?;
    let sum_mask = mask_f32.sum(1)?;
    let pad = ((seq_len as f64) - &sum_mask)?;
    let pos = Tensor::arange(0u32, seq_len_u32, device)?.to_dtype(DType::F32)?;
    let pos = pos.unsqueeze(0)?.broadcast_as((batch, seq_len))?;
    let pad = pad.unsqueeze(1)?.broadcast_as((batch, seq_len))?;
    let shifted = (&pos - &pad)?;
    let ones = Tensor::ones((batch, seq_len), DType::F32, device)?;
    let base_pos = mask_u8.where_cond(&shifted, &ones)?;
    base_pos
        .round()?
        .to_dtype(DType::I64)?
        .unsqueeze(0)?
        .broadcast_as((3usize, batch, seq_len))
}

pub fn merge_audio_features(
    input_ids: &Tensor,
    inputs_embeds: &Tensor,
    audio_token_id: u32,
    audio_features: &Tensor,
    audio_placeholder_count: usize,
) -> Result<Tensor> {
    let (batch, seq_len) = input_ids.dims2()?;
    let (b2, s2, hidden) = inputs_embeds.dims3()?;
    if (b2, s2) != (batch, seq_len) {
        candle::bail!("inputs_embeds dims mismatch");
    }
    let (num_audio, audio_hidden) = audio_features.dims2()?;
    if audio_hidden != hidden {
        candle::bail!("audio_features hidden mismatch");
    }
    if audio_placeholder_count == 0 || num_audio == 0 {
        return Ok(inputs_embeds.clone());
    }
    if audio_placeholder_count != num_audio {
        candle::bail!("audio placeholder count mismatch: {audio_placeholder_count} vs {num_audio}");
    }

    let mask_u8 = input_ids.eq(audio_token_id)?;
    let mask_f32 = mask_u8.to_dtype(DType::F32)?;
    let in_row = ((mask_f32.cumsum(1)? - 1f64)? * &mask_f32)?;
    let row_counts = mask_f32.sum(1)?;
    let offsets = (row_counts.cumsum(0)? - &row_counts)?;
    let offsets = offsets.unsqueeze(1)?.broadcast_as((batch, seq_len))?;
    let total = batch * seq_len;
    let idx = ((&offsets + &in_row)? * &mask_f32)?.reshape((total,))?;
    let idx = idx.to_dtype(DType::U32)?;
    let audio_features = audio_features.to_dtype(inputs_embeds.dtype())?;
    let audio_at_pos = audio_features
        .embedding(&idx)?
        .reshape((batch, seq_len, hidden))?;
    let cond = mask_u8
        .unsqueeze(2)?
        .broadcast_as((batch, seq_len, hidden))?;
    cond.where_cond(&audio_at_pos, inputs_embeds)
}

// ── Full Model ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Model {
    pub audio_encoder: AudioEncoder,
    pub text_model: TextModel,
    pub lm_head: Linear,
    pub audio_token_id: u32,
}

impl Model {
    pub fn load(
        cfg: &Config,
        vb: VarBuilder,
        device: &Device,
        use_flash_attn: bool,
    ) -> Result<Self> {
        let audio_token_id = cfg.thinker_config.audio_token_id.ok_or_else(|| {
            candle::Error::Msg("thinker_config.audio_token_id is required".to_string())
        })?;
        let vb_thinker = vb.pp("thinker");
        let audio_encoder = AudioEncoder::load(
            &cfg.thinker_config.audio_config,
            vb_thinker.pp("audio_tower"),
            use_flash_attn,
        )?;
        let text_model = TextModel::load(
            &cfg.thinker_config.text_config,
            vb_thinker.pp("model"),
            device,
            use_flash_attn,
        )?;
        let lm_head = if cfg.thinker_config.text_config.tie_word_embeddings {
            Linear::from_weights(text_model.embed_tokens_weight().clone(), None)
        } else {
            linear_no_bias(
                cfg.thinker_config.text_config.hidden_size,
                cfg.thinker_config.text_config.vocab_size,
                vb_thinker.pp("lm_head"),
            )?
        };
        Ok(Self {
            audio_encoder,
            text_model,
            lm_head,
            audio_token_id,
        })
    }

    pub fn get_audio_features(&self, input_features: &Tensor) -> Result<Tensor> {
        self.audio_encoder.forward(input_features)
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.text_model.embed(input_ids)
    }

    pub fn inputs_embeds_with_audio_features(
        &self,
        input_ids: &Tensor,
        audio_features: Option<&Tensor>,
        audio_placeholder_count: usize,
    ) -> Result<Tensor> {
        let inputs_embeds = self.embed_tokens(input_ids)?;
        match audio_features {
            None => Ok(inputs_embeds),
            Some(af) => merge_audio_features(
                input_ids,
                &inputs_embeds,
                self.audio_token_id,
                af,
                audio_placeholder_count,
            ),
        }
    }

    pub fn forward_with_kv_cache(
        &self,
        attention_mask: &Tensor,
        position_ids: &Tensor,
        inputs_embeds: &Tensor,
        kv_cache: &mut KVCache,
    ) -> Result<Tensor> {
        let hidden_states = self.text_model.forward_with_kv_cache(
            attention_mask,
            position_ids,
            inputs_embeds,
            kv_cache,
        )?;
        self.lm_head.forward(&hidden_states)
    }
}

#[cfg(test)]
mod tests {
    use super::{make_causal_mask_cached, repeat_kv};

    #[test]
    fn test_repeat_kv() -> anyhow::Result<()> {
        let device = candle::Device::Cpu;
        let x = candle::Tensor::arange(0f32, 2.0 * 3.0 * 4.0, &device)?.reshape((2, 3, 2, 2))?;
        let y = repeat_kv(&x, 2)?;
        assert_eq!(y.dims(), &[2, 6, 2, 2]);
        Ok(())
    }

    #[test]
    fn test_make_causal_mask_cached() -> anyhow::Result<()> {
        let device = candle::Device::Cpu;
        let mask = make_causal_mask_cached(None, 1, 3, 2, candle::DType::F32, &device)?;
        assert_eq!(mask.dims(), &[1, 1, 2, 5]);
        Ok(())
    }

    #[test]
    fn test_config_deserialize_real_format() -> anyhow::Result<()> {
        use super::super::Config as AsrConfig;
        let json = r#"{
            "thinker_config": {
                "model_type": "qwen3_asr",
                "audio_token_id": 151668,
                "audio_config": {
                    "d_model": 896,
                    "encoder_attention_heads": 14,
                    "encoder_ffn_dim": 3584,
                    "encoder_layers": 18,
                    "output_dim": 2048,
                    "num_mel_bins": 128
                },
                "text_config": {
                    "vocab_size": 151936,
                    "hidden_size": 2048,
                    "intermediate_size": 11008,
                    "num_hidden_layers": 28,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 4,
                    "head_dim": 128,
                    "rope_theta": 5000000.0,
                    "rms_norm_eps": 1e-6,
                    "rope_scaling": {
                        "mrope_section": [24, 20, 20],
                        "interleaved": true
                    }
                }
            }
        }"#;
        let cfg: AsrConfig = serde_json::from_str(json)?;
        assert_eq!(cfg.thinker_config.audio_token_id, Some(151668));
        assert_eq!(cfg.thinker_config.audio_config.d_model, 896);
        assert_eq!(cfg.thinker_config.audio_config.encoder_layers, 18);
        assert_eq!(cfg.thinker_config.audio_config.output_dim, 2048);
        assert_eq!(cfg.thinker_config.text_config.hidden_size, 2048);
        assert_eq!(cfg.thinker_config.text_config.num_hidden_layers, 28);
        assert!(cfg.thinker_config.text_config.rope_scaling.is_some());
        Ok(())
    }

    #[test]
    fn test_audio_encoder_forward_shape() -> anyhow::Result<()> {
        use super::super::audio_encoder::AudioEncoder;
        use super::super::AudioEncoderConfig;

        let device = candle::Device::Cpu;
        let cfg = AudioEncoderConfig {
            num_mel_bins: 4,
            d_model: 4,
            encoder_attention_heads: 1,
            encoder_ffn_dim: 8,
            encoder_layers: 1,
            output_dim: 6,
            downsample_hidden_size: 2,
            max_source_positions: 64,
            n_window: 5,
            n_window_infer: 20,
            conv_chunksize: 8,
            ..Default::default()
        };

        let vb = candle_nn::VarBuilder::zeros(candle::DType::F32, &device);
        let encoder = AudioEncoder::load(&cfg, vb, false)?;

        let batch = 1usize;
        let frames = 10usize;
        let input_features = candle::Tensor::ones(
            (batch, cfg.num_mel_bins, frames),
            candle::DType::F32,
            &device,
        )?;

        let output = encoder.forward(&input_features)?;
        let (out_frames, out_dim) = output.dims2()?;
        assert_eq!(out_dim, cfg.output_dim);
        assert!(out_frames > 0, "encoder output must have at least 1 frame");

        let output2 = encoder.forward_with_lens(&input_features, &[frames])?;
        assert_eq!(output.dims(), output2.dims());
        Ok(())
    }

    #[test]
    fn test_model_creation_with_zeros() -> anyhow::Result<()> {
        use super::super::Config as AsrConfig;
        let device = candle::Device::Cpu;
        let json = r#"{
            "thinker_config": {
                "audio_token_id": 151668,
                "audio_config": {
                    "num_mel_bins": 4,
                    "d_model": 4,
                    "encoder_attention_heads": 1,
                    "encoder_ffn_dim": 8,
                    "encoder_layers": 1,
                    "output_dim": 6,
                    "downsample_hidden_size": 2,
                    "max_source_positions": 64,
                    "n_window": 5,
                    "n_window_infer": 20,
                    "conv_chunksize": 8
                },
                "text_config": {
                    "vocab_size": 1000,
                    "hidden_size": 64,
                    "intermediate_size": 256,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 16,
                    "rope_theta": 10000.0,
                    "rms_norm_eps": 1e-6,
                    "rope_scaling": {
                        "mrope_section": [4, 2, 2],
                        "interleaved": true
                    }
                }
            }
        }"#;
        let cfg: AsrConfig = serde_json::from_str(json)?;
        let vb = candle_nn::VarBuilder::zeros(candle::DType::F32, &device);
        let model = super::Model::load(&cfg, vb, &device, false)?;
        assert_eq!(model.audio_token_id, 151668);
        Ok(())
    }
}
