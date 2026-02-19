//! Thinker text model (decoder-only transformer with mRoPE).
//!
//! This is a faithful port of the "thinker.model" module in the official
//! Qwen3-ASR implementation:
//! `Qwen3-ASR/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py`.

use candle::{Result, Tensor};
use candle_nn::{
    embedding, linear_b, linear_no_bias, rms_norm, Embedding, Linear, Module, RmsNorm, VarBuilder,
};

use crate::config::{RopeScaling, TextConfig};
use crate::model::kv_cache::KVCache;
use crate::model::{attention, rope};

#[cfg(feature = "flash-attn")]
use candle::{DType, Device};

#[cfg(feature = "flash-attn")]
use candle_flash_attn::{flash_attn, flash_attn_varlen};

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
        candle::bail!(
            "internal error: attention_mask lens mismatch: expected={batch}, got={}",
            lens.len()
        );
    }

    let mut out: Vec<usize> = Vec::with_capacity(batch);
    for (i, v) in lens.into_iter().enumerate() {
        if !v.is_finite() || v < 0.0 {
            candle::bail!("invalid attention_mask length at {i}: {v}");
        }
        if v > seq_len as f32 + 0.01 {
            candle::bail!(
                "attention_mask length at {i} exceeds seq_len: len={v} seq_len={seq_len}"
            );
        }
        let len = v.round() as usize;
        if (len as f32 - v).abs() > 0.01 {
            candle::bail!("attention_mask length at {i} is not integral: {v}");
        }
        out.push(len);
    }
    Ok(out)
}

#[cfg(feature = "flash-attn")]
fn cu_seqlens_u32(lengths: &[usize], device: &Device) -> Result<(Tensor, usize, u32)> {
    let mut cu: Vec<u32> = Vec::with_capacity(lengths.len().saturating_add(1));
    cu.push(0);

    let mut total: u32 = 0;
    let mut max_len: usize = 0;
    for (i, &len) in lengths.iter().enumerate() {
        let len_u32 = u32::try_from(len).map_err(|_| {
            candle::Error::Msg(format!(
                "sequence length overflows u32 at index {i}: len={len}"
            ))
        })?;
        total = total.checked_add(len_u32).ok_or_else(|| {
            candle::Error::Msg(format!(
                "cumulative sequence length overflows u32 at index {i}: total={total} len={len}"
            ))
        })?;
        cu.push(total);
        max_len = max_len.max(len);
    }

    let cu_t = Tensor::from_vec(cu, (lengths.len().saturating_add(1),), device)?;
    Ok((cu_t, max_len, total))
}

#[cfg(feature = "flash-attn")]
fn left_pad_indices_u32(seq_len: usize, lengths: &[usize]) -> Result<Vec<u32>> {
    let batch = lengths.len();

    let mut total: usize = 0;
    for &len in lengths {
        total = total
            .checked_add(len)
            .ok_or_else(|| candle::Error::Msg("index capacity overflow".to_string()))?;
    }

    let mut idxs: Vec<u32> = Vec::with_capacity(total);
    for (b, &len) in lengths.iter().enumerate() {
        if len > seq_len {
            candle::bail!(
                "sequence length exceeds seq_len for batch index {b}: len={len} seq_len={seq_len}"
            );
        }
        let pad = seq_len.saturating_sub(len);
        let base = b
            .checked_mul(seq_len)
            .ok_or_else(|| candle::Error::Msg("index overflow".to_string()))?;
        for j in 0..len {
            let pos = base
                .checked_add(pad.saturating_add(j))
                .ok_or_else(|| candle::Error::Msg("index overflow".to_string()))?;
            idxs.push(
                u32::try_from(pos)
                    .map_err(|_| candle::Error::Msg(format!("index overflows u32: pos={pos}")))?,
            );
        }
    }

    if idxs.is_empty() && batch > 0 {
        candle::bail!("attention_mask contains no valid tokens");
    }

    Ok(idxs)
}

#[derive(Debug, Clone)]
struct ThinkerTextRotaryEmbedding {
    rope: rope::mrope::MultimodalRotaryEmbedding,
    mrope_section: Vec<usize>,
    interleaved: bool,
}

impl ThinkerTextRotaryEmbedding {
    fn load(cfg: &TextConfig, device: &candle::Device) -> Result<Self> {
        let scaling = cfg.rope_scaling.as_ref();
        let rope = match scaling {
            None => rope::mrope::MultimodalRotaryEmbedding::new(
                cfg.head_dim,
                cfg.max_position_embeddings,
                cfg.rope_theta,
                device,
            )?,
            Some(s) => rope::mrope::MultimodalRotaryEmbedding::with_scaling(
                cfg.head_dim,
                cfg.max_position_embeddings,
                cfg.rope_theta,
                s,
                device,
            )?,
        };

        let (mrope_section, interleaved) = match scaling {
            None => (vec![24usize, 20, 20], false),
            Some(s) => (
                if s.mrope_section.is_empty() {
                    vec![24usize, 20, 20]
                } else {
                    s.mrope_section.clone()
                },
                s.mrope_interleaved || s.interleaved,
            ),
        };

        Ok(Self {
            rope,
            mrope_section,
            interleaved,
        })
    }

    fn forward(&self, x: &Tensor, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        self.rope.forward(x, position_ids)
    }
}

#[derive(Debug, Clone)]
struct ThinkerTextMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    hidden_act: String,
}

impl ThinkerTextMlp {
    fn load(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            hidden_act: cfg.hidden_act.clone(),
        })
    }
}

impl Module for ThinkerTextMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?;
        let up = self.up_proj.forward(xs)?;
        let gate = match self.hidden_act.as_str() {
            "silu" | "swish" => candle_nn::ops::silu(&gate)?,
            other => candle::bail!("unsupported hidden_act={other:?}"),
        };
        let hidden = gate.broadcast_mul(&up)?;
        self.down_proj.forward(&hidden)
    }
}

#[derive(Debug, Clone)]
struct ThinkerTextAttention {
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

impl ThinkerTextAttention {
    fn load(cfg: &TextConfig, vb: VarBuilder, use_flash_attn: bool) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let num_attention_heads = cfg.num_attention_heads;
        let num_key_value_heads = cfg.num_key_value_heads;
        let num_key_value_groups = num_attention_heads / num_key_value_heads;

        let q_out = num_attention_heads * head_dim;
        let kv_out = num_key_value_heads * head_dim;

        let q_proj = linear_b(cfg.hidden_size, q_out, cfg.attention_bias, vb.pp("q_proj"))?;
        let k_proj = linear_b(cfg.hidden_size, kv_out, cfg.attention_bias, vb.pp("k_proj"))?;
        let v_proj = linear_b(cfg.hidden_size, kv_out, cfg.attention_bias, vb.pp("v_proj"))?;
        let o_proj = linear_b(q_out, cfg.hidden_size, cfg.attention_bias, vb.pp("o_proj"))?;

        let q_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            use_flash_attn,
            num_attention_heads,
            num_key_value_heads,
            num_key_value_groups,
            head_dim,
            scaling: (head_dim as f64).powf(-0.5),
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        position_embeddings: (&Tensor, &Tensor),
        attention_mask: Option<&Tensor>,
        token_attention_mask: Option<&Tensor>,
        rope_scaling: &ThinkerTextRotaryEmbedding,
    ) -> Result<Tensor> {
        let (batch, seq_len, _hidden) = hidden_states.dims3()?;

        let q = self.q_proj.forward(hidden_states)?;
        let q = q.reshape((batch, seq_len, self.num_attention_heads, self.head_dim))?;
        let q = self.q_norm.forward(&q)?;
        let q = q.transpose(1, 2)?; // (b, h, s, d)

        let k = self.k_proj.forward(hidden_states)?;
        let k = k.reshape((batch, seq_len, self.num_key_value_heads, self.head_dim))?;
        let k = self.k_norm.forward(&k)?;
        let k = k.transpose(1, 2)?; // (b, kv, s, d)

        let v = self.v_proj.forward(hidden_states)?;
        let v = v.reshape((batch, seq_len, self.num_key_value_heads, self.head_dim))?;
        let v = v.transpose(1, 2)?; // (b, kv, s, d)

        let (cos, sin) = position_embeddings;
        let (q, k) = rope::mrope::apply_multimodal_rotary_pos_emb(
            &q,
            &k,
            cos,
            sin,
            rope_scaling.mrope_section.as_slice(),
            rope_scaling.interleaved,
        )?;

        if self.use_flash_attn && attention_mask.is_none() {
            #[cfg(not(feature = "flash-attn"))]
            {
                let _ = token_attention_mask;
                candle::bail!("flash-attn support is not enabled in this build");
            }
            #[cfg(feature = "flash-attn")]
            {
                let softmax_scale = self.scaling as f32;

                let q4 = q.transpose(1, 2)?.contiguous()?; // (b, s, h, d)
                let k4 = k.transpose(1, 2)?.contiguous()?; // (b, s, kv, d)
                let v4 = v.transpose(1, 2)?.contiguous()?; // (b, s, kv, d)

                let Some(tok_mask) = token_attention_mask else {
                    let attn = flash_attn(&q4, &k4, &v4, softmax_scale, true)?;
                    let attn =
                        attn.reshape((batch, seq_len, self.num_attention_heads * self.head_dim))?;
                    return self.o_proj.forward(&attn);
                };

                let seqlens = seqlens_from_left_padded_attention_mask(tok_mask, seq_len)?;
                let (cu, max_len, total_u32) =
                    cu_seqlens_u32(seqlens.as_slice(), hidden_states.device())?;
                let total = usize::try_from(total_u32).map_err(|_| {
                    candle::Error::Msg(format!(
                        "total sequence length overflows usize: total={total_u32}"
                    ))
                })?;

                let flat_total = batch.checked_mul(seq_len).ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "batch*seq_len overflow: batch={batch} seq_len={seq_len}"
                    ))
                })?;

                if total == flat_total {
                    let attn = flash_attn(&q4, &k4, &v4, softmax_scale, true)?;
                    let attn =
                        attn.reshape((batch, seq_len, self.num_attention_heads * self.head_dim))?;
                    return self.o_proj.forward(&attn);
                }

                let idxs = left_pad_indices_u32(seq_len, seqlens.as_slice())?;
                if idxs.len() != total {
                    candle::bail!(
                        "internal error: index len mismatch: idxs={} total={total}",
                        idxs.len()
                    );
                }

                let idx = Tensor::from_vec(idxs, (total,), hidden_states.device())?;

                let q_flat = q4.reshape((flat_total, self.num_attention_heads, self.head_dim))?;
                let k_flat = k4.reshape((flat_total, self.num_key_value_heads, self.head_dim))?;
                let v_flat = v4.reshape((flat_total, self.num_key_value_heads, self.head_dim))?;

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

                let out = {
                    let zeros = Tensor::zeros(
                        (flat_total, self.num_attention_heads, self.head_dim),
                        out_unpad.dtype(),
                        out_unpad.device(),
                    )?;
                    let flat = zeros.index_add(&idx, &out_unpad, 0)?;
                    flat.reshape((batch, seq_len, self.num_attention_heads, self.head_dim))?
                };

                let out =
                    out.reshape((batch, seq_len, self.num_attention_heads * self.head_dim))?;
                return self.o_proj.forward(&out);
            }
        }

        let k = attention::repeat_kv(&k, self.num_key_value_groups)?;
        let v = attention::repeat_kv(&v, self.num_key_value_groups)?;

        let k_t = k.transpose(2, 3)?; // (b, h, d, s)
        let mut attn_weights = q.matmul(&k_t)?.affine(self.scaling, 0.0)?;

        if let Some(mask) = attention_mask {
            attn_weights = attn_weights.broadcast_add(mask)?;
        }

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?; // (b, h, s, d)
        let attn_output = attn_output.transpose(1, 2)?; // (b, s, h, d)
        let attn_output =
            attn_output.reshape((batch, seq_len, self.num_attention_heads * self.head_dim))?;
        self.o_proj.forward(&attn_output)
    }

    fn forward_with_kv_cache(
        &self,
        hidden_states: &Tensor,
        position_embeddings: (&Tensor, &Tensor),
        attention_mask: Option<&Tensor>,
        token_attention_mask: &Tensor,
        rope_scaling: &ThinkerTextRotaryEmbedding,
        layer_cache: (&mut KVCache, usize),
    ) -> Result<Tensor> {
        let (batch, seq_len, _hidden) = hidden_states.dims3()?;
        let (kv_cache, layer_idx) = layer_cache;

        let q = self.q_proj.forward(hidden_states)?;
        let q = q.reshape((batch, seq_len, self.num_attention_heads, self.head_dim))?;
        let q = self.q_norm.forward(&q)?;
        let q = q.transpose(1, 2)?; // (b, h, s, d)

        let k = self.k_proj.forward(hidden_states)?;
        let k = k.reshape((batch, seq_len, self.num_key_value_heads, self.head_dim))?;
        let k = self.k_norm.forward(&k)?;
        let k = k.transpose(1, 2)?; // (b, kv, s, d)

        let v = self.v_proj.forward(hidden_states)?;
        let v = v.reshape((batch, seq_len, self.num_key_value_heads, self.head_dim))?;
        let v = v.transpose(1, 2)?; // (b, kv, s, d)

        let (cos, sin) = position_embeddings;
        let (q, k) = rope::mrope::apply_multimodal_rotary_pos_emb(
            &q,
            &k,
            cos,
            sin,
            rope_scaling.mrope_section.as_slice(),
            rope_scaling.interleaved,
        )?;

        // Cache rotated keys to match the attention math.
        let (k, v) = kv_cache.update(layer_idx, &k, &v)?;

        if self.use_flash_attn && attention_mask.is_none() {
            #[cfg(not(feature = "flash-attn"))]
            {
                let _ = token_attention_mask;
                candle::bail!("flash-attn support is not enabled in this build");
            }
            #[cfg(feature = "flash-attn")]
            {
                let softmax_scale = self.scaling as f32;

                let (_b2, total_len) = token_attention_mask.dims2()?;
                if _b2 != batch {
                    candle::bail!(
                        "token_attention_mask batch mismatch: expected={batch}, got={_b2}"
                    );
                }
                let cache_len = total_len.checked_sub(seq_len).ok_or_else(|| {
                    candle::Error::Msg(format!(
                        "attention mask shorter than new seq_len: total_len={total_len} new_len={seq_len}"
                    ))
                })?;

                let q4 = q.transpose(1, 2)?.contiguous()?; // (b, s_q, h, d)
                let k4 = k.transpose(1, 2)?.contiguous()?; // (b, s_k, kv, d)
                let v4 = v.transpose(1, 2)?.contiguous()?; // (b, s_k, kv, d)

                if cache_len == 0 {
                    let seqlens =
                        seqlens_from_left_padded_attention_mask(token_attention_mask, seq_len)?;
                    let (cu, max_len, total_u32) =
                        cu_seqlens_u32(seqlens.as_slice(), hidden_states.device())?;
                    let total = usize::try_from(total_u32).map_err(|_| {
                        candle::Error::Msg(format!(
                            "total sequence length overflows usize: total={total_u32}"
                        ))
                    })?;

                    let flat_total = batch.checked_mul(seq_len).ok_or_else(|| {
                        candle::Error::Msg(format!(
                            "batch*seq_len overflow: batch={batch} seq_len={seq_len}"
                        ))
                    })?;

                    if total == flat_total {
                        let attn = flash_attn(&q4, &k4, &v4, softmax_scale, true)?;
                        let attn = attn.reshape((
                            batch,
                            seq_len,
                            self.num_attention_heads * self.head_dim,
                        ))?;
                        return self.o_proj.forward(&attn);
                    }

                    let idxs = left_pad_indices_u32(seq_len, seqlens.as_slice())?;
                    if idxs.len() != total {
                        candle::bail!(
                            "internal error: index len mismatch: idxs={} total={total}",
                            idxs.len()
                        );
                    }
                    let idx = Tensor::from_vec(idxs, (total,), hidden_states.device())?;

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

                    let out = {
                        let zeros = Tensor::zeros(
                            (flat_total, self.num_attention_heads, self.head_dim),
                            out_unpad.dtype(),
                            out_unpad.device(),
                        )?;
                        let flat = zeros.index_add(&idx, &out_unpad, 0)?;
                        flat.reshape((batch, seq_len, self.num_attention_heads, self.head_dim))?
                    };

                    let out =
                        out.reshape((batch, seq_len, self.num_attention_heads * self.head_dim))?;
                    return self.o_proj.forward(&out);
                }

                // Cached decode step: q has no padding, but k/v include left padding.
                let k_total = batch
                    .checked_mul(total_len)
                    .ok_or_else(|| candle::Error::Msg("k/v size overflow".to_string()))?;
                let q_total = batch
                    .checked_mul(seq_len)
                    .ok_or_else(|| candle::Error::Msg("q size overflow".to_string()))?;

                let q3 = q4.reshape((q_total, self.num_attention_heads, self.head_dim))?;

                let seqlens_k =
                    seqlens_from_left_padded_attention_mask(token_attention_mask, total_len)?;
                let (cu_k, max_k, total_k_u32) =
                    cu_seqlens_u32(seqlens_k.as_slice(), hidden_states.device())?;
                let total_k = usize::try_from(total_k_u32).map_err(|_| {
                    candle::Error::Msg(format!("total_k overflows usize: total_k={total_k_u32}"))
                })?;

                let seqlens_q = vec![seq_len; batch];
                let (cu_q, max_q, total_q_u32) =
                    cu_seqlens_u32(seqlens_q.as_slice(), hidden_states.device())?;
                let total_q = usize::try_from(total_q_u32).map_err(|_| {
                    candle::Error::Msg(format!("total_q overflows usize: total_q={total_q_u32}"))
                })?;

                if total_q != q_total {
                    candle::bail!(
                        "internal error: q total mismatch: expected={q_total}, got={total_q}"
                    );
                }

                let idxs_k = left_pad_indices_u32(total_len, seqlens_k.as_slice())?;
                if idxs_k.len() != total_k {
                    candle::bail!(
                        "internal error: k index len mismatch: idxs={} total_k={total_k}",
                        idxs_k.len()
                    );
                }
                let idx_k = Tensor::from_vec(idxs_k, (total_k,), hidden_states.device())?;

                let k3 = k4
                    .reshape((k_total, self.num_key_value_heads, self.head_dim))?
                    .index_select(&idx_k, 0)?
                    .contiguous()?;
                let v3 = v4
                    .reshape((k_total, self.num_key_value_heads, self.head_dim))?
                    .index_select(&idx_k, 0)?
                    .contiguous()?;

                let out = flash_attn_varlen(
                    &q3,
                    &k3,
                    &v3,
                    &cu_q,
                    &cu_k,
                    max_q,
                    max_k,
                    softmax_scale,
                    true,
                )?;
                let out =
                    out.reshape((batch, seq_len, self.num_attention_heads * self.head_dim))?;
                return self.o_proj.forward(&out);
            }
        }

        let k = attention::repeat_kv(&k, self.num_key_value_groups)?;
        let v = attention::repeat_kv(&v, self.num_key_value_groups)?;

        let k_t = k.transpose(2, 3)?; // (b, h, d, s)
        let mut attn_weights = q.matmul(&k_t)?.affine(self.scaling, 0.0)?;

        if let Some(mask) = attention_mask {
            attn_weights = attn_weights.broadcast_add(mask)?;
        }

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?; // (b, h, s, d)
        let attn_output = attn_output.transpose(1, 2)?; // (b, s, h, d)
        let attn_output =
            attn_output.reshape((batch, seq_len, self.num_attention_heads * self.head_dim))?;
        self.o_proj.forward(&attn_output)
    }
}

#[derive(Debug, Clone)]
struct ThinkerTextDecoderLayer {
    self_attn: ThinkerTextAttention,
    mlp: ThinkerTextMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl ThinkerTextDecoderLayer {
    fn load(cfg: &TextConfig, vb: VarBuilder, use_flash_attn: bool) -> Result<Self> {
        let self_attn = ThinkerTextAttention::load(cfg, vb.pp("self_attn"), use_flash_attn)?;
        let mlp = ThinkerTextMlp::load(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
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
        &self,
        hidden_states: &Tensor,
        position_embeddings: (&Tensor, &Tensor),
        attention_mask: Option<&Tensor>,
        token_attention_mask: Option<&Tensor>,
        rope_scaling: &ThinkerTextRotaryEmbedding,
    ) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let x = self.input_layernorm.forward(hidden_states)?;
        let x = self.self_attn.forward(
            &x,
            position_embeddings,
            attention_mask,
            token_attention_mask,
            rope_scaling,
        )?;
        let x = (&residual + &x)?;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        &residual + &x
    }

    fn forward_with_kv_cache(
        &self,
        hidden_states: &Tensor,
        position_embeddings: (&Tensor, &Tensor),
        attention_mask: Option<&Tensor>,
        token_attention_mask: &Tensor,
        rope_scaling: &ThinkerTextRotaryEmbedding,
        layer_cache: (&mut KVCache, usize),
    ) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let x = self.input_layernorm.forward(hidden_states)?;
        let x = self.self_attn.forward_with_kv_cache(
            &x,
            position_embeddings,
            attention_mask,
            token_attention_mask,
            rope_scaling,
            layer_cache,
        )?;
        let x = (&residual + &x)?;

        let residual = x.clone();
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        &residual + &x
    }
}

/// Text part of Qwen3-ASR thinker (decoder-only transformer).
#[derive(Debug, Clone)]
pub struct ThinkerTextModel {
    embed_tokens: Embedding,
    layers: Vec<ThinkerTextDecoderLayer>,
    norm: RmsNorm,
    rotary_emb: ThinkerTextRotaryEmbedding,
    hidden_size: usize,
    use_flash_attn: bool,
}

impl ThinkerTextModel {
    pub fn load(
        cfg: &TextConfig,
        vb: VarBuilder,
        device: &candle::Device,
        use_flash_attn: bool,
    ) -> Result<Self> {
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            layers.push(ThinkerTextDecoderLayer::load(
                cfg,
                vb.pp("layers").pp(idx.to_string()),
                use_flash_attn,
            )?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let rotary_emb = ThinkerTextRotaryEmbedding::load(cfg, device)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            hidden_size: cfg.hidden_size,
            use_flash_attn,
        })
    }

    pub fn embed_tokens(&self) -> &Embedding {
        &self.embed_tokens
    }

    pub fn embed_tokens_weight(&self) -> &Tensor {
        self.embed_tokens.embeddings()
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn forward(
        &self,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
        inputs_embeds: &Tensor,
    ) -> Result<Tensor> {
        let (batch, seq_len, hidden) = inputs_embeds.dims3()?;
        if hidden != self.hidden_size {
            candle::bail!(
                "inputs_embeds hidden mismatch: expected={}, got={hidden}",
                self.hidden_size
            );
        }

        let device = inputs_embeds.device();
        let dtype = inputs_embeds.dtype();
        let causal_mask = if self.use_flash_attn {
            None
        } else {
            Some(attention::make_causal_mask(
                attention_mask,
                batch,
                seq_len,
                dtype,
                device,
            )?)
        };

        // Shape is used only for dtype/device in the rope kernel, matching the official behavior.
        let (cos, sin) = self.rotary_emb.forward(inputs_embeds, position_ids)?;
        let position_embeddings = (&cos, &sin);

        let mut hidden_states = inputs_embeds.clone();
        for layer in &self.layers {
            hidden_states = layer.forward(
                &hidden_states,
                position_embeddings,
                causal_mask.as_ref(),
                attention_mask,
                &self.rotary_emb,
            )?;
        }

        self.norm.forward(&hidden_states)
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
            candle::bail!("attention_mask batch mismatch: expected={batch}, got={b2}");
        }

        let cache_len = kv_cache.seq_len();
        if total_len != cache_len.saturating_add(seq_len) {
            candle::bail!(
                "attention_mask total_len mismatch vs cache: total_len={total_len} cache_len={cache_len} new_len={seq_len}"
            );
        }

        let device = inputs_embeds.device();
        let dtype = inputs_embeds.dtype();
        let causal_mask = if self.use_flash_attn {
            None
        } else {
            Some(attention::make_causal_mask_cached(
                Some(attention_mask),
                batch,
                cache_len,
                seq_len,
                dtype,
                device,
            )?)
        };

        let (cos, sin) = self.rotary_emb.forward(inputs_embeds, position_ids)?;
        let position_embeddings = (&cos, &sin);

        let mut hidden_states = inputs_embeds.clone();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward_with_kv_cache(
                &hidden_states,
                position_embeddings,
                causal_mask.as_ref(),
                attention_mask,
                &self.rotary_emb,
                (&mut *kv_cache, layer_idx),
            )?;
        }

        self.norm.forward(&hidden_states)
    }
}

fn _require_mrope_enabled(cfg: &TextConfig) -> Result<&RopeScaling> {
    cfg.rope_scaling.as_ref().ok_or_else(|| {
        candle::Error::Msg(
            "text_config.rope_scaling is required (Qwen3-ASR uses mRoPE)".to_string(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::ThinkerTextRotaryEmbedding;

    #[test]
    fn test_rotary_embedding_loads_without_rope_scaling() -> anyhow::Result<()> {
        let device = candle::Device::Cpu;
        let cfg = crate::config::TextConfig::default();
        let _ = ThinkerTextRotaryEmbedding::load(&cfg, &device)?;
        Ok(())
    }
}
