//! Qwen3-TTS core model (talker + code predictor) for inference-only generation.

use std::sync::Arc;

use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{kv_cache::ConcatKvCache, Activation, Embedding, VarBuilder};

use crate::{
    generation::LogitsProcessor,
    models::with_tracing::{linear_b, linear_no_bias, Linear, RmsNorm},
    utils::repeat_kv,
};

use super::config::{LayerType, Qwen3TtsCodePredictorConfig, Qwen3TtsConfig, Qwen3TtsTalkerConfig};
use super::speaker::{mel_spectrogram, Qwen3TtsSpeakerEncoder};

fn build_causal_mask(
    bsz: usize,
    seq_len: usize,
    seqlen_offset: usize,
    sliding_window: Option<usize>,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let mut mask = vec![0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let disallow = if i < j {
                true
            } else if let Some(window) = sliding_window {
                j + window < i
            } else {
                false
            };
            if disallow {
                mask[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask = Tensor::from_vec(mask, (seq_len, seq_len), device)?;
    let mask = if seqlen_offset > 0 {
        let mask0 = Tensor::zeros((seq_len, seqlen_offset), dtype, device)?;
        Tensor::cat(&[&mask0, &mask], D::Minus1)?
    } else {
        mask
    };
    mask.expand((bsz, 1, seq_len, seq_len + seqlen_offset))?
        .to_dtype(dtype)
}

fn apply_repetition_penalty(logits: &Tensor, generated: &[i64], penalty: f32) -> Result<Tensor> {
    if generated.is_empty() || (penalty - 1.0).abs() < f32::EPSILON {
        return Ok(logits.clone());
    }
    let device = logits.device();
    let shape = logits.dims().to_vec();
    let flat = logits.flatten_all()?;
    let mut vals = flat.to_vec1::<f32>()?;
    for &token in generated {
        let idx = token as usize;
        if idx < vals.len() {
            if vals[idx] > 0.0 {
                vals[idx] /= penalty;
            } else {
                vals[idx] *= penalty;
            }
        }
    }
    Tensor::from_vec(vals, shape, device)
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Qwen3TtsCodePredictorConfig, device: &Device) -> Result<Self> {
        let dim = cfg.head_dim();
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_len), device)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct MropeEmbedding {
    cos: Tensor,
    sin: Tensor,
    mrope_section: Vec<usize>,
    head_dim: usize,
    interleaved: bool,
}

impl MropeEmbedding {
    fn new(dtype: DType, cfg: &Qwen3TtsTalkerConfig, device: &Device) -> Result<Self> {
        let dim = cfg.head_dim();
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_len), device)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
            mrope_section: cfg.mrope_section(),
            head_dim: dim,
            interleaved: cfg.mrope_interleaved(),
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let (three, _batch, _seq_len) = position_ids.dims3()?;
        if three != 3 {
            candle::bail!("position_ids must have shape [3, batch, seq_len]");
        }
        let (cos_3d, sin_3d) = self.compute_3d_embeddings(position_ids)?;
        let (cos, sin) = if self.interleaved {
            let (cos, sin) = self.apply_interleaved_sections(&cos_3d, &sin_3d)?;
            (cos, sin)
        } else {
            self.apply_mrope_sections(&cos_3d, &sin_3d)?
        };
        let cos = cos.unsqueeze(1)?;
        let sin = sin.unsqueeze(1)?;
        let q_embed = apply_rope_to_tensor(q, &cos, &sin)?;
        let k_embed = apply_rope_to_tensor(k, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }

    fn compute_3d_embeddings(&self, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let (three, batch, seq_len) = position_ids.dims3()?;
        let half_dim = self.head_dim / 2;
        let mut cos_parts = Vec::with_capacity(three);
        let mut sin_parts = Vec::with_capacity(three);
        for dim_idx in 0..three {
            let pos = position_ids.i(dim_idx)?; // [batch, seq_len]
            let pos_flat = pos.flatten_all()?; // [batch*seq_len]
            let cos_g = self.cos.index_select(&pos_flat, 0)?;
            let sin_g = self.sin.index_select(&pos_flat, 0)?;
            let cos_dim = cos_g.reshape((batch, seq_len, half_dim))?;
            let sin_dim = sin_g.reshape((batch, seq_len, half_dim))?;
            let cos_full = Tensor::cat(&[&cos_dim, &cos_dim], D::Minus1)?;
            let sin_full = Tensor::cat(&[&sin_dim, &sin_dim], D::Minus1)?;
            cos_parts.push(cos_full);
            sin_parts.push(sin_full);
        }
        Ok((Tensor::stack(&cos_parts, 0)?, Tensor::stack(&sin_parts, 0)?))
    }

    fn apply_mrope_sections(&self, cos_3d: &Tensor, sin_3d: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut sections = Vec::new();
        sections.extend_from_slice(&self.mrope_section);
        sections.extend_from_slice(&self.mrope_section);
        let mut cos_parts = Vec::new();
        let mut sin_parts = Vec::new();
        let mut offset = 0usize;
        for (i, &sec) in sections.iter().enumerate() {
            let dim_idx = i % 3;
            let cos_slice = cos_3d.i(dim_idx)?.narrow(D::Minus1, offset, sec)?;
            let sin_slice = sin_3d.i(dim_idx)?.narrow(D::Minus1, offset, sec)?;
            cos_parts.push(cos_slice);
            sin_parts.push(sin_slice);
            offset += sec;
        }
        Ok((
            Tensor::cat(&cos_parts, D::Minus1)?,
            Tensor::cat(&sin_parts, D::Minus1)?,
        ))
    }

    fn apply_interleaved_sections(
        &self,
        cos_3d: &Tensor,
        sin_3d: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let half_dim = self.head_dim / 2;
        let cos_half = cos_3d.narrow(D::Minus1, 0, half_dim)?;
        let sin_half = sin_3d.narrow(D::Minus1, 0, half_dim)?;
        let cos_inter = interleave_modalities(&cos_half, &self.mrope_section)?;
        let sin_inter = interleave_modalities(&sin_half, &self.mrope_section)?;
        let cos = Tensor::cat(&[&cos_inter, &cos_inter], D::Minus1)?;
        let sin = Tensor::cat(&[&sin_inter, &sin_inter], D::Minus1)?;
        Ok((cos, sin))
    }
}

fn interleave_modalities(x: &Tensor, mrope_section: &[usize]) -> Result<Tensor> {
    let (modalities, batch, seq_len, dim) = x.dims4()?;
    if modalities == 0 {
        candle::bail!("mrope interleaved expects at least one modality");
    }
    let x_flat = x.to_vec1::<f32>()?;
    let modality_num = mrope_section.len();
    let mut out = vec![0f32; batch * seq_len * dim];
    for b in 0..batch {
        for s in 0..seq_len {
            let mut row = vec![0f32; dim];
            let base0 = (0 * batch * seq_len + b * seq_len + s) * dim;
            row.copy_from_slice(&x_flat[base0..base0 + dim]);
            for (i, &n) in mrope_section.iter().enumerate().skip(1) {
                let beg = i;
                let end = n.saturating_mul(modality_num);
                let mut idx = beg;
                while idx < end && idx < dim {
                    let base = (i * batch * seq_len + b * seq_len + s) * dim;
                    row[idx] = x_flat[base + idx];
                    idx += modality_num;
                }
            }
            let base = (b * seq_len + s) * dim;
            out[base..base + dim].copy_from_slice(&row[..]);
        }
    }
    let out = Tensor::from_vec(out, (batch, seq_len, dim), x.device())?;
    out.to_dtype(x.dtype())
}

fn apply_rope_to_tensor(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let x = x.contiguous()?;
    let head_dim = x.dim(D::Minus1)?;
    let half_dim = head_dim / 2;
    let x1 = x.narrow(D::Minus1, 0, half_dim)?;
    let x2 = x.narrow(D::Minus1, half_dim, half_dim)?;
    let x_rot = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
    x.broadcast_mul(cos)? + x_rot.broadcast_mul(sin)?
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act: Activation,
}

impl Mlp {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        act: Activation,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?,
            act,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct ResizeMlp {
    linear1: Linear,
    linear2: Linear,
    act: Activation,
}

impl ResizeMlp {
    fn new(
        input: usize,
        intermediate: usize,
        output: usize,
        act: Activation,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            linear1: linear_b(input, intermediate, true, vb.pp("linear_fc1"))?,
            linear2: linear_b(intermediate, output, true, vb.pp("linear_fc2"))?,
            act,
        })
    }
}

impl Module for ResizeMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.linear1)?
            .apply(&self.act)?
            .apply(&self.linear2)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary: Arc<RotaryEmbedding>,
    kv_cache: ConcatKvCache,
    sliding_window: Option<usize>,
}

impl Attention {
    fn new(
        cfg: &Qwen3TtsCodePredictorConfig,
        rotary: Arc<RotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_kv_heads();
        let num_kv_groups = num_heads / num_kv_heads;
        let q_proj = linear_b(
            cfg.hidden_size,
            num_heads * head_dim,
            cfg.attention_bias,
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_b(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            cfg.attention_bias,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_b(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            cfg.attention_bias,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_b(
            num_heads * head_dim,
            cfg.hidden_size,
            cfg.attention_bias,
            vb.pp("o_proj"),
        )?;
        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size: cfg.hidden_size,
            rotary,
            kv_cache: ConcatKvCache::new(2),
            sliding_window: if cfg.use_sliding_window {
                cfg.sliding_window
            } else {
                None
            },
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attn_mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let (b, l, _) = xs.dims3()?;
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;
        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let q = self
            .q_norm
            .forward(&q_flat)?
            .reshape((b, self.num_heads, l, self.head_dim))?;
        let k = self
            .k_norm
            .forward(&k_flat)?
            .reshape((b, self.num_kv_heads, l, self.head_dim))?;
        let (q, k) = self.rotary.apply(&q, &k, offset)?;
        let (k, v) = self.kv_cache.append(&k, &v)?;
        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(mask) = attn_mask {
            scores = scores.broadcast_add(mask)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;
        ctx.transpose(1, 2)?
            .reshape((b, l, self.hidden_size))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

#[derive(Debug, Clone)]
struct TalkerAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary: Arc<MropeEmbedding>,
    kv_cache: ConcatKvCache,
}

impl TalkerAttention {
    fn new(
        cfg: &Qwen3TtsTalkerConfig,
        rotary: Arc<MropeEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_kv_heads();
        let num_kv_groups = num_heads / num_kv_heads;
        let q_proj = linear_b(
            cfg.hidden_size,
            num_heads * head_dim,
            cfg.attention_bias,
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_b(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            cfg.attention_bias,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_b(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            cfg.attention_bias,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_b(
            num_heads * head_dim,
            cfg.hidden_size,
            cfg.attention_bias,
            vb.pp("o_proj"),
        )?;
        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size: cfg.hidden_size,
            rotary,
            kv_cache: ConcatKvCache::new(2),
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        position_ids: &Tensor,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, l, _) = xs.dims3()?;
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;
        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let q = self
            .q_norm
            .forward(&q_flat)?
            .reshape((b, self.num_heads, l, self.head_dim))?;
        let k = self
            .k_norm
            .forward(&k_flat)?
            .reshape((b, self.num_kv_heads, l, self.head_dim))?;
        let (q, k) = self.rotary.apply(&q, &k, position_ids)?;
        let (k, v) = self.kv_cache.append(&k, &v)?;
        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(mask) = attn_mask {
            scores = scores.broadcast_add(mask)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;
        ctx.transpose(1, 2)?
            .reshape((b, l, self.hidden_size))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl DecoderLayer {
    fn new(
        cfg: &Qwen3TtsCodePredictorConfig,
        rotary: Arc<RotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::new(cfg, rotary, vb.pp("self_attn"))?,
            mlp: Mlp::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                cfg.hidden_act,
                vb.pp("mlp"),
            )?,
            ln1: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            ln2: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attn_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.ln1.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attn_mask, seqlen_offset)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.ln2)?.apply(&self.mlp)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
struct TalkerLayer {
    self_attn: TalkerAttention,
    mlp: Mlp,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl TalkerLayer {
    fn new(
        cfg: &Qwen3TtsTalkerConfig,
        rotary: Arc<MropeEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            self_attn: TalkerAttention::new(cfg, rotary, vb.pp("self_attn"))?,
            mlp: Mlp::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                cfg.hidden_act,
                vb.pp("mlp"),
            )?,
            ln1: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            ln2: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        position_ids: &Tensor,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.ln1.forward(xs)?;
        let xs = self.self_attn.forward(&xs, position_ids, attn_mask)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.ln2)?.apply(&self.mlp)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
struct CodePredictorModel {
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    rotary: Arc<RotaryEmbedding>,
    codec_embedding: Vec<Embedding>,
    device: Device,
    dtype: DType,
    layer_types: Vec<LayerType>,
    sliding_window: Option<usize>,
}

impl CodePredictorModel {
    fn new(
        cfg: &Qwen3TtsCodePredictorConfig,
        embedding_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb_m = vb.clone();
        let rotary = Arc::new(RotaryEmbedding::new(vb_m.dtype(), cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, rotary.clone(), vb_l.pp(layer_idx))?);
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let mut codec_embedding = Vec::with_capacity(cfg.num_code_groups.saturating_sub(1));
        let vb_e = vb_m.pp("codec_embedding");
        for idx in 0..cfg.num_code_groups.saturating_sub(1) {
            codec_embedding.push(candle_nn::embedding(
                cfg.vocab_size,
                embedding_dim,
                vb_e.pp(idx),
            )?);
        }
        Ok(Self {
            layers,
            norm,
            rotary,
            codec_embedding,
            device: vb.device().clone(),
            dtype: vb.dtype(),
            layer_types: cfg.layer_types(),
            sliding_window: if cfg.use_sliding_window {
                cfg.sliding_window
            } else {
                None
            },
        })
    }

    fn forward(&mut self, inputs_embeds: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b, seq_len, _) = inputs_embeds.dims3()?;
        let full_mask = if seq_len > 1 {
            Some(build_causal_mask(
                b,
                seq_len,
                seqlen_offset,
                None,
                &self.device,
                self.dtype,
            )?)
        } else {
            None
        };
        let sliding_mask = if seq_len > 1 && self.sliding_window.is_some() {
            Some(build_causal_mask(
                b,
                seq_len,
                seqlen_offset,
                self.sliding_window,
                &self.device,
                self.dtype,
            )?)
        } else {
            None
        };
        let mut xs = inputs_embeds.clone();
        for (layer, layer_type) in self.layers.iter_mut().zip(self.layer_types.iter()) {
            let mask = match layer_type {
                LayerType::FullAttention => full_mask.as_ref(),
                LayerType::SlidingAttention => sliding_mask.as_ref().or(full_mask.as_ref()),
            };
            xs = layer.forward(&xs, mask, seqlen_offset)?;
        }
        xs.apply(&self.norm)
    }

    fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }
}

#[derive(Debug, Clone)]
struct CodePredictor {
    model: CodePredictorModel,
    lm_head: Vec<Linear>,
    small_to_mtp: Option<Linear>,
    num_code_groups: usize,
}

impl CodePredictor {
    fn new(
        cfg: &Qwen3TtsCodePredictorConfig,
        talker_hidden_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let model = CodePredictorModel::new(cfg, talker_hidden_size, vb.pp("model"))?;
        let mut lm_head = Vec::with_capacity(cfg.num_code_groups.saturating_sub(1));
        let vb_lm = vb.pp("lm_head");
        for idx in 0..cfg.num_code_groups.saturating_sub(1) {
            lm_head.push(linear_b(
                cfg.hidden_size,
                cfg.vocab_size,
                false,
                vb_lm.pp(idx),
            )?);
        }
        let small_to_mtp = if cfg.hidden_size != talker_hidden_size {
            Some(linear_b(
                talker_hidden_size,
                cfg.hidden_size,
                true,
                vb.pp("small_to_mtp_projection"),
            )?)
        } else {
            None
        };
        Ok(Self {
            model,
            lm_head,
            small_to_mtp,
            num_code_groups: cfg.num_code_groups,
        })
    }

    fn get_embedding(&self, idx: usize) -> Result<&Embedding> {
        self.model
            .codec_embedding
            .get(idx)
            .ok_or_else(|| candle::Error::Msg("codec embedding index out of range".into()))
    }

    fn generate(
        &mut self,
        inputs_embeds: &Tensor,
        do_sample: bool,
        top_k: Option<usize>,
        top_p: Option<f64>,
        temperature: f64,
        seed: u64,
    ) -> Result<Vec<i64>> {
        let max_new_tokens = self.num_code_groups.saturating_sub(1);
        let mut out: Vec<i64> = Vec::with_capacity(max_new_tokens);
        self.model.clear_kv_cache();

        let mut lp = if do_sample {
            LogitsProcessor::from_sampling(
                seed,
                match (top_k, top_p) {
                    (Some(k), Some(p)) => {
                        crate::generation::Sampling::TopKThenTopP { k, p, temperature }
                    }
                    (Some(k), None) => crate::generation::Sampling::TopK { k, temperature },
                    (None, Some(p)) => crate::generation::Sampling::TopP { p, temperature },
                    (None, None) => crate::generation::Sampling::All { temperature },
                },
            )
        } else {
            LogitsProcessor::from_sampling(seed, crate::generation::Sampling::ArgMax)
        };

        let mut seqlen_offset = 0usize;
        for step in 0..max_new_tokens {
            let step_embeds = if step == 0 {
                if let Some(proj) = &self.small_to_mtp {
                    inputs_embeds.apply(proj)?
                } else {
                    inputs_embeds.clone()
                }
            } else {
                let token = out[step - 1];
                let token_t = Tensor::from_vec(vec![token], (1, 1), inputs_embeds.device())?;
                let emb = self.get_embedding(step - 1)?.forward(&token_t)?;
                if let Some(proj) = &self.small_to_mtp {
                    emb.apply(proj)?
                } else {
                    emb
                }
            };
            let hs = self.model.forward(&step_embeds, seqlen_offset)?;
            let last = hs.i((.., hs.dim(1)? - 1, ..))?.unsqueeze(1)?;
            let logits = last.apply(&self.lm_head[step])?;
            let logits = logits.squeeze(0)?.squeeze(0)?;
            let token = lp.sample(&logits)? as i64;
            out.push(token);
            seqlen_offset += step_embeds.dim(1)?;
        }
        Ok(out)
    }
}

#[derive(Debug, Clone)]
struct TalkerModel {
    codec_embedding: Embedding,
    text_embedding: Embedding,
    layers: Vec<TalkerLayer>,
    norm: RmsNorm,
    rotary: Arc<MropeEmbedding>,
    device: Device,
    dtype: DType,
}

impl TalkerModel {
    fn new(cfg: &Qwen3TtsTalkerConfig, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.clone();
        let codec_embedding =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("codec_embedding"))?;
        let text_embedding = candle_nn::embedding(
            cfg.text_vocab_size,
            cfg.text_hidden_size,
            vb_m.pp("text_embedding"),
        )?;
        let rotary = Arc::new(MropeEmbedding::new(vb_m.dtype(), cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            layers.push(TalkerLayer::new(cfg, rotary.clone(), vb_l.pp(layer_idx))?);
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        Ok(Self {
            codec_embedding,
            text_embedding,
            layers,
            norm,
            rotary,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn embed_codec(&self, ids: &Tensor) -> Result<Tensor> {
        self.codec_embedding.forward(ids)
    }

    fn embed_text(&self, ids: &Tensor) -> Result<Tensor> {
        self.text_embedding.forward(ids)
    }

    fn forward(
        &mut self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = inputs_embeds.dims3()?;
        let attn_mask = if seq_len > 1 {
            Some(build_causal_mask(
                b,
                seq_len,
                seqlen_offset,
                None,
                &self.device,
                self.dtype,
            )?)
        } else {
            None
        };
        let mut xs = inputs_embeds.clone();
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, position_ids, attn_mask.as_ref())?;
        }
        xs.apply(&self.norm)
    }

    fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }
}

#[derive(Debug, Clone)]
struct Talker {
    model: TalkerModel,
    text_projection: ResizeMlp,
    codec_head: Linear,
    code_predictor: CodePredictor,
    cfg: Qwen3TtsTalkerConfig,
    device: Device,
}

impl Talker {
    fn new(cfg: &Qwen3TtsTalkerConfig, vb: VarBuilder) -> Result<Self> {
        let model = TalkerModel::new(cfg, vb.pp("model"))?;
        let text_projection = ResizeMlp::new(
            cfg.text_hidden_size,
            cfg.text_hidden_size,
            cfg.hidden_size,
            cfg.hidden_act,
            vb.pp("text_projection"),
        )?;
        let codec_head = linear_b(cfg.hidden_size, cfg.vocab_size, false, vb.pp("codec_head"))?;
        let code_predictor = CodePredictor::new(
            &cfg.code_predictor_config,
            cfg.hidden_size,
            vb.pp("code_predictor"),
        )?;
        Ok(Self {
            model,
            text_projection,
            codec_head,
            code_predictor,
            cfg: cfg.clone(),
            device: vb.device().clone(),
        })
    }
}

/// Top-level Qwen3-TTS wrapper.
pub struct Qwen3Tts {
    pub config: Qwen3TtsConfig,
    talker: Talker,
    speaker_encoder: Option<Qwen3TtsSpeakerEncoder>,
    device: Device,
    dtype: DType,
}

pub struct GenerationParams {
    pub do_sample: bool,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub temperature: f64,
    pub repetition_penalty: f32,
    pub subtalker_do_sample: bool,
    pub subtalker_top_k: Option<usize>,
    pub subtalker_top_p: Option<f64>,
    pub subtalker_temperature: f64,
    pub max_new_tokens: usize,
    pub seed: u64,
}

#[derive(Debug, Clone)]
pub struct VoiceClonePromptItem {
    pub ref_code: Option<Tensor>,
    pub ref_spk_embedding: Tensor,
    pub x_vector_only_mode: bool,
    pub icl_mode: bool,
}

impl Qwen3Tts {
    pub fn new(cfg: Qwen3TtsConfig, vb: VarBuilder) -> Result<Self> {
        let talker = Talker::new(&cfg.talker_config, vb.pp("talker"))?;
        let speaker_encoder = match cfg.speaker_encoder_config.as_ref() {
            Some(enc_cfg) => Some(Qwen3TtsSpeakerEncoder::new(
                enc_cfg,
                vb.pp("speaker_encoder"),
            )?),
            None => None,
        };
        Ok(Self {
            config: cfg,
            talker,
            speaker_encoder,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn clear_kv_cache(&mut self) {
        self.talker.model.clear_kv_cache();
        self.talker.code_predictor.model.clear_kv_cache();
    }

    pub fn extract_speaker_embedding(&self, audio: &[f32], sample_rate: usize) -> Result<Tensor> {
        let enc_cfg = self
            .config
            .speaker_encoder_config
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("speaker encoder not configured".into()))?;
        if sample_rate != enc_cfg.sample_rate {
            candle::bail!(
                "speaker encoder expects {} Hz audio, got {} Hz",
                enc_cfg.sample_rate,
                sample_rate
            );
        }
        let encoder = self
            .speaker_encoder
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("speaker encoder not initialized".into()))?;
        let (mel, frames) = mel_spectrogram(audio, enc_cfg)?;
        let mel = Tensor::from_vec(mel, (1, frames, enc_cfg.mel_dim), &self.device)?;
        let emb = encoder.forward(&mel)?;
        emb.i(0)
    }

    fn make_position_ids(&self, seq_len: usize, offset: usize) -> Result<Tensor> {
        let ids = (offset..offset + seq_len)
            .map(|v| v as i64)
            .collect::<Vec<_>>();
        let ids = Tensor::from_vec(ids, (1, seq_len), &self.device)?;
        let ids = ids.unsqueeze(0)?; // [1,1,seq]
        let ids = ids.broadcast_as((3, 1, seq_len))?;
        Ok(ids)
    }

    fn build_talker_input(
        &self,
        input_ids: &Tensor,
        language: &str,
        speaker: Option<&str>,
        speaker_embed: Option<&Tensor>,
        icl: Option<(&Tensor, &Tensor)>,
        non_streaming: bool,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let seq_len = input_ids.dim(1)?;
        if seq_len < 8 {
            candle::bail!("input_ids too short for Qwen3-TTS prompt format");
        }

        let speaker_name = speaker.unwrap_or("");
        let spk_key = speaker_name.to_lowercase();
        let speaker_embed = if let Some(embed) = speaker_embed {
            let embed = if embed.dims().len() == 1 {
                embed.unsqueeze(0)?.unsqueeze(0)?
            } else if embed.dims().len() == 2 {
                embed.unsqueeze(0)?
            } else {
                embed.clone()
            };
            Some(embed.to_device(&self.device)?.to_dtype(self.dtype)?)
        } else if speaker_name.is_empty() {
            None
        } else {
            let spk_ids =
                self.talker.cfg.spk_id.get(&spk_key).ok_or_else(|| {
                    candle::Error::Msg(format!("unknown speaker: {speaker_name}"))
                })?;
            let spk_id = *spk_ids.first().ok_or_else(|| {
                candle::Error::Msg(format!("speaker id list empty for {speaker_name}"))
            })?;
            let spk_t = Tensor::from_vec(vec![spk_id], (1, 1), &self.device)?;
            Some(self.talker.model.embed_codec(&spk_t)?)
        };

        let language_id = if language.eq_ignore_ascii_case("auto") {
            None
        } else {
            let key = language.to_lowercase();
            self.talker.cfg.codec_language_id.get(&key).copied()
        };
        let language_id = if (language.eq_ignore_ascii_case("chinese")
            || language.eq_ignore_ascii_case("auto"))
            && !speaker_name.is_empty()
        {
            if let Some(val) = self.talker.cfg.spk_is_dialect.get(&spk_key) {
                if let Some(dialect) = val.as_str() {
                    self.talker
                        .cfg
                        .codec_language_id
                        .get(dialect)
                        .copied()
                        .or(language_id)
                } else {
                    language_id
                }
            } else {
                language_id
            }
        } else {
            language_id
        };

        let tts_ids = Tensor::from_vec(
            vec![
                self.config.tts_bos_token_id,
                self.config.tts_eos_token_id,
                self.config.tts_pad_token_id,
            ],
            (1, 3),
            &self.device,
        )?;
        let tts_embeds = self
            .talker
            .text_projection
            .forward(&self.talker.model.embed_text(&tts_ids)?)?;
        let tts_bos = tts_embeds.narrow(1, 0, 1)?;
        let tts_eos = tts_embeds.narrow(1, 1, 1)?;
        let tts_pad = tts_embeds.narrow(1, 2, 1)?;

        let mut codec_prefill: Vec<i64> = Vec::new();
        if let Some(lang_id) = language_id {
            codec_prefill.push(self.talker.cfg.codec_think_id);
            codec_prefill.push(self.talker.cfg.codec_think_bos_id);
            codec_prefill.push(lang_id);
            codec_prefill.push(self.talker.cfg.codec_think_eos_id);
        } else {
            codec_prefill.push(self.talker.cfg.codec_nothink_id);
            codec_prefill.push(self.talker.cfg.codec_think_bos_id);
            codec_prefill.push(self.talker.cfg.codec_think_eos_id);
        }
        let codec_prefill_len = codec_prefill.len();
        let codec_prefill = Tensor::from_vec(codec_prefill, (1, codec_prefill_len), &self.device)?;
        let codec_prefill = self.talker.model.embed_codec(&codec_prefill)?;
        let codec_tail = Tensor::from_vec(
            vec![self.talker.cfg.codec_pad_id, self.talker.cfg.codec_bos_id],
            (1, 2),
            &self.device,
        )?;
        let codec_tail = self.talker.model.embed_codec(&codec_tail)?;
        let codec_embeds = if let Some(spk) = speaker_embed {
            Tensor::cat(&[&codec_prefill, &spk, &codec_tail], 1)?
        } else {
            Tensor::cat(&[&codec_prefill, &codec_tail], 1)?
        };

        let role_ids = input_ids.narrow(1, 0, 3)?;
        let role_embed = self
            .talker
            .text_projection
            .forward(&self.talker.model.embed_text(&role_ids)?)?;

        let pad_len = codec_embeds.dim(1)? - 2;
        let pad_block = tts_pad.broadcast_as((1, pad_len, tts_pad.dim(2)?))?;
        let prefill = Tensor::cat(&[&pad_block, &tts_bos], 1)?;
        let prefill = (prefill + codec_embeds.narrow(1, 0, codec_embeds.dim(1)? - 1)?)?;
        let mut talker_input = Tensor::cat(&[&role_embed, &prefill], 1)?;

        let first_text = input_ids.narrow(1, 3, 1)?;
        let first_text = self
            .talker
            .text_projection
            .forward(&self.talker.model.embed_text(&first_text)?)?;
        let last_codec = codec_embeds.narrow(1, codec_embeds.dim(1)? - 1, 1)?;
        if icl.is_none() {
            talker_input = Tensor::cat(&[&talker_input, &(first_text + last_codec)?], 1)?;
        }

        let text_body_len = seq_len.saturating_sub(8);
        let text_body = if text_body_len > 0 {
            input_ids.narrow(1, 3, text_body_len)?
        } else {
            Tensor::zeros((1, 0), DType::I64, &self.device)?
        };
        if let Some((ref_ids, ref_codes)) = icl {
            let (icl_input, trailing) = self.build_icl_prompt(
                &text_body,
                ref_ids,
                ref_codes,
                &tts_pad,
                &tts_eos,
                non_streaming,
            )?;
            talker_input = Tensor::cat(&[&talker_input, &icl_input], 1)?;
            Ok((talker_input, trailing, tts_pad))
        } else if non_streaming {
            let text_embed = self
                .talker
                .text_projection
                .forward(&self.talker.model.embed_text(&text_body)?)?;
            let text_embed = Tensor::cat(&[&text_embed, &tts_eos], 1)?;
            let pad_ids = vec![self.talker.cfg.codec_pad_id; text_body_len + 1];
            let pad_len = pad_ids.len();
            let pad_ids = Tensor::from_vec(pad_ids, (1, pad_len), &self.device)?;
            let pad_embed = self.talker.model.embed_codec(&pad_ids)?;
            let text_embed = (text_embed + pad_embed)?;
            let bos_id =
                Tensor::from_vec(vec![self.talker.cfg.codec_bos_id], (1, 1), &self.device)?;
            let bos_embed = self.talker.model.embed_codec(&bos_id)?;
            let tail = (tts_pad.broadcast_as((1, 1, tts_pad.dim(2)?))? + bos_embed)?;
            talker_input = talker_input.narrow(1, 0, talker_input.dim(1)? - 1)?;
            talker_input = Tensor::cat(&[&talker_input, &text_embed, &tail], 1)?;
            Ok((talker_input, tts_pad.clone(), tts_pad))
        } else {
            let trailing = if text_body_len > 1 {
                let rest = input_ids.narrow(1, 4, text_body_len - 1)?;
                let rest_embed = self
                    .talker
                    .text_projection
                    .forward(&self.talker.model.embed_text(&rest)?)?;
                Tensor::cat(&[&rest_embed, &tts_eos], 1)?
            } else {
                tts_eos.clone()
            };
            Ok((talker_input, trailing, tts_pad))
        }
    }

    fn build_icl_prompt(
        &self,
        text_id: &Tensor,
        ref_id: &Tensor,
        ref_code: &Tensor,
        tts_pad: &Tensor,
        tts_eos: &Tensor,
        non_streaming: bool,
    ) -> Result<(Tensor, Tensor)> {
        let text_cat = Tensor::cat(&[ref_id, text_id], 1)?;
        let text_embed = self
            .talker
            .text_projection
            .forward(&self.talker.model.embed_text(&text_cat)?)?;
        let text_embed = Tensor::cat(&[&text_embed, tts_eos], 1)?;

        let ref_code = match ref_code.dims().len() {
            2 => ref_code.clone(),
            3 => ref_code.squeeze(0)?,
            _ => candle::bail!("ref_code must be 2D or 3D"),
        };
        let mut codec_hiddens: Vec<Tensor> =
            Vec::with_capacity(self.config.talker_config.num_code_groups);
        for i in 0..self.config.talker_config.num_code_groups {
            let ids = ref_code.narrow(1, i, 1)?;
            let emb = if i == 0 {
                self.talker.model.embed_codec(&ids)?
            } else {
                self.talker
                    .code_predictor
                    .get_embedding(i - 1)?
                    .forward(&ids)?
            };
            codec_hiddens.push(emb);
        }
        let mut sum = codec_hiddens[0].clone();
        for h in codec_hiddens.iter().skip(1) {
            sum = (sum + h)?;
        }
        let codec_embed = sum.squeeze(1)?.unsqueeze(0)?;
        let bos_id = Tensor::from_vec(vec![self.talker.cfg.codec_bos_id], (1, 1), &self.device)?;
        let bos_embed = self.talker.model.embed_codec(&bos_id)?;
        let codec_embed = Tensor::cat(&[&bos_embed, &codec_embed], 1)?;

        let text_lens = text_embed.dim(1)?;
        let codec_lens = codec_embed.dim(1)?;
        if non_streaming {
            let pad_ids = vec![self.talker.cfg.codec_pad_id; text_lens];
            let pad_len = pad_ids.len();
            let pad_ids = Tensor::from_vec(pad_ids, (1, pad_len), &self.device)?;
            let pad_embed = self.talker.model.embed_codec(&pad_ids)?;
            let icl_input = (text_embed + pad_embed)?;
            let icl_input = Tensor::cat(&[&icl_input, &(codec_embed + tts_pad)?], 1)?;
            Ok((icl_input, tts_pad.clone()))
        } else if text_lens > codec_lens {
            let icl_input = (text_embed.narrow(1, 0, codec_lens)? + codec_embed)?;
            let trailing = text_embed.narrow(1, codec_lens, text_lens - codec_lens)?;
            Ok((icl_input, trailing))
        } else {
            let pad_len = codec_lens.saturating_sub(text_lens);
            let pad_block = tts_pad.broadcast_as((1, pad_len, tts_pad.dim(2)?))?;
            let text_embed = Tensor::cat(&[&text_embed, &pad_block], 1)?;
            let icl_input = (text_embed + codec_embed)?;
            Ok((icl_input, tts_pad.clone()))
        }
    }

    pub fn generate_custom_voice_codes(
        &mut self,
        input_ids: &Tensor,
        language: &str,
        speaker: &str,
        instruct_ids: Option<&Tensor>,
        non_streaming: bool,
        params: &GenerationParams,
    ) -> Result<Vec<Vec<i64>>> {
        let speaker = if speaker.is_empty() {
            None
        } else {
            Some(speaker)
        };
        self.generate_one(
            input_ids,
            language,
            speaker,
            None,
            instruct_ids,
            None,
            non_streaming,
            params,
        )
    }

    pub fn generate_custom_voice_codes_batch(
        &mut self,
        input_ids: &[Tensor],
        languages: &[String],
        speakers: &[String],
        instruct_ids: Option<&[Option<Tensor>]>,
        non_streaming: bool,
        params: &GenerationParams,
    ) -> Result<Vec<Vec<Vec<i64>>>> {
        let batch = input_ids.len();
        if languages.len() != batch {
            candle::bail!(
                "languages length {} does not match batch {}",
                languages.len(),
                batch
            );
        }
        if speakers.len() != batch {
            candle::bail!(
                "speakers length {} does not match batch {}",
                speakers.len(),
                batch
            );
        }
        if let Some(ins) = instruct_ids {
            if ins.len() != batch {
                candle::bail!(
                    "instruct_ids length {} does not match batch {}",
                    ins.len(),
                    batch
                );
            }
        }
        let mut out = Vec::with_capacity(batch);
        for idx in 0..batch {
            let speaker = if speakers[idx].is_empty() {
                None
            } else {
                Some(speakers[idx].as_str())
            };
            let ins = instruct_ids.and_then(|v| v[idx].as_ref());
            let codes = self.generate_one(
                &input_ids[idx],
                &languages[idx],
                speaker,
                None,
                ins,
                None,
                non_streaming,
                params,
            )?;
            out.push(codes);
        }
        Ok(out)
    }

    pub fn generate_voice_design_codes_batch(
        &mut self,
        input_ids: &[Tensor],
        languages: &[String],
        instruct_ids: &[Option<Tensor>],
        non_streaming: bool,
        params: &GenerationParams,
    ) -> Result<Vec<Vec<Vec<i64>>>> {
        let batch = input_ids.len();
        if languages.len() != batch {
            candle::bail!(
                "languages length {} does not match batch {}",
                languages.len(),
                batch
            );
        }
        if instruct_ids.len() != batch {
            candle::bail!(
                "instruct_ids length {} does not match batch {}",
                instruct_ids.len(),
                batch
            );
        }
        let mut out = Vec::with_capacity(batch);
        for idx in 0..batch {
            let ins = instruct_ids[idx].as_ref();
            let codes = self.generate_one(
                &input_ids[idx],
                &languages[idx],
                None,
                None,
                ins,
                None,
                non_streaming,
                params,
            )?;
            out.push(codes);
        }
        Ok(out)
    }

    pub fn generate_voice_clone_codes_batch(
        &mut self,
        input_ids: &[Tensor],
        languages: &[String],
        ref_ids: Option<&[Option<Tensor>]>,
        voice_clone_prompt: &[VoiceClonePromptItem],
        non_streaming: bool,
        params: &GenerationParams,
    ) -> Result<Vec<Vec<Vec<i64>>>> {
        let batch = input_ids.len();
        if languages.len() != batch {
            candle::bail!(
                "languages length {} does not match batch {}",
                languages.len(),
                batch
            );
        }
        if voice_clone_prompt.len() != batch {
            candle::bail!(
                "voice_clone_prompt length {} does not match batch {}",
                voice_clone_prompt.len(),
                batch
            );
        }
        if let Some(ids) = ref_ids {
            if ids.len() != batch {
                candle::bail!(
                    "ref_ids length {} does not match batch {}",
                    ids.len(),
                    batch
                );
            }
        }
        let mut out = Vec::with_capacity(batch);
        for idx in 0..batch {
            let prompt = &voice_clone_prompt[idx];
            let speaker_embed = if prompt.x_vector_only_mode || prompt.icl_mode {
                Some(&prompt.ref_spk_embedding)
            } else {
                None
            };
            let mut icl_store: Option<(Tensor, Tensor)> = None;
            let icl = if prompt.icl_mode {
                let ref_ids = ref_ids
                    .and_then(|v| v[idx].as_ref())
                    .ok_or_else(|| candle::Error::Msg("ref_ids required for icl_mode".into()))?;
                let ref_len = ref_ids.dim(1)?;
                let body_len = ref_len.saturating_sub(5);
                let body = if body_len > 0 {
                    ref_ids.narrow(1, 3, body_len)?
                } else {
                    Tensor::zeros((1, 0), DType::I64, &self.device)?
                };
                let code = prompt
                    .ref_code
                    .as_ref()
                    .ok_or_else(|| candle::Error::Msg("ref_code required for icl_mode".into()))?;
                icl_store = Some((body, code.clone()));
                let (body, code) = icl_store.as_ref().unwrap();
                Some((body, code))
            } else {
                None
            };
            let codes = self.generate_one(
                &input_ids[idx],
                &languages[idx],
                None,
                speaker_embed,
                None,
                icl,
                non_streaming,
                params,
            )?;
            out.push(codes);
        }
        Ok(out)
    }

    fn generate_one(
        &mut self,
        input_ids: &Tensor,
        language: &str,
        speaker: Option<&str>,
        speaker_embed: Option<&Tensor>,
        instruct_ids: Option<&Tensor>,
        icl: Option<(&Tensor, &Tensor)>,
        non_streaming: bool,
        params: &GenerationParams,
    ) -> Result<Vec<Vec<i64>>> {
        if input_ids.dim(0)? != 1 {
            candle::bail!("input_ids must be shape [1, seq_len] for single-sample generation");
        }

        self.clear_kv_cache();

        let mut icl_store: Option<(Tensor, Tensor)> = None;
        let icl_ref = if let Some((ref_ids, ref_code)) = icl {
            icl_store = Some((
                ref_ids.to_device(&self.device)?,
                ref_code.to_device(&self.device)?,
            ));
            let (ids, codes) = icl_store.as_ref().unwrap();
            Some((ids, codes))
        } else {
            None
        };

        let (talker_input, trailing_text, tts_pad) = self.build_talker_input(
            input_ids,
            language,
            speaker,
            speaker_embed,
            icl_ref,
            non_streaming,
        )?;
        let mut inputs_embeds = talker_input.clone();
        if let Some(ins) = instruct_ids {
            let instruct_embed = self
                .talker
                .text_projection
                .forward(&self.talker.model.embed_text(ins)?)?;
            inputs_embeds = Tensor::cat(&[&instruct_embed, &inputs_embeds], 1)?;
        }

        let mut seqlen_offset = 0usize;
        let pos = self.make_position_ids(inputs_embeds.dim(1)?, seqlen_offset)?;
        let hs = self
            .talker
            .model
            .forward(&inputs_embeds, &pos, seqlen_offset)?;
        seqlen_offset += inputs_embeds.dim(1)?;
        let last = hs.i((.., hs.dim(1)? - 1, ..))?.unsqueeze(1)?;
        let mut logits = last.apply(&self.talker.codec_head)?;
        let mut past_hidden = last.clone();
        let suppress_from = self.talker.cfg.vocab_size.saturating_sub(1024);
        let suppress_eos = self.talker.cfg.codec_eos_token_id as usize;
        let mut logits_vec = logits.squeeze(0)?.squeeze(0)?.to_vec1::<f32>()?;
        if suppress_from < self.talker.cfg.vocab_size {
            for idx in suppress_from..self.talker.cfg.vocab_size {
                if idx != suppress_eos {
                    logits_vec[idx] = f32::NEG_INFINITY;
                }
            }
        }
        logits = Tensor::from_vec(logits_vec, (self.talker.cfg.vocab_size,), &self.device)?
            .unsqueeze(0)?
            .unsqueeze(0)?;

        let mut generated_tokens: Vec<i64> = Vec::new();
        let mut lp = if params.do_sample {
            LogitsProcessor::from_sampling(
                params.seed,
                match (params.top_k, params.top_p) {
                    (Some(k), Some(p)) => crate::generation::Sampling::TopKThenTopP {
                        k,
                        p,
                        temperature: params.temperature,
                    },
                    (Some(k), None) => crate::generation::Sampling::TopK {
                        k,
                        temperature: params.temperature,
                    },
                    (None, Some(p)) => crate::generation::Sampling::TopP {
                        p,
                        temperature: params.temperature,
                    },
                    (None, None) => crate::generation::Sampling::All {
                        temperature: params.temperature,
                    },
                },
            )
        } else {
            LogitsProcessor::from_sampling(params.seed, crate::generation::Sampling::ArgMax)
        };

        let mut token0 = lp.sample(&logits.squeeze(0)?.squeeze(0)?)? as i64;
        generated_tokens.push(token0);
        let mut output_codes: Vec<Vec<i64>> = Vec::new();

        for step in 0..params.max_new_tokens {
            let last_id = Tensor::from_vec(vec![token0], (1, 1), &self.device)?;
            let last_hidden = self.talker.model.embed_codec(&last_id)?;
            let predictor_in = Tensor::cat(&[&past_hidden, &last_hidden], 1)?;
            let sub_tokens = self.talker.code_predictor.generate(
                &predictor_in,
                params.subtalker_do_sample,
                params.subtalker_top_k,
                params.subtalker_top_p,
                params.subtalker_temperature,
                params.seed + step as u64 + 1,
            )?;
            let mut step_codes = Vec::with_capacity(1 + sub_tokens.len());
            step_codes.push(token0);
            step_codes.extend_from_slice(&sub_tokens);
            output_codes.push(step_codes);

            if token0 == self.talker.cfg.codec_eos_token_id {
                break;
            }

            let mut codec_hiddens: Vec<Tensor> = Vec::with_capacity(1 + sub_tokens.len());
            codec_hiddens.push(last_hidden);
            for (i, tok) in sub_tokens.iter().enumerate() {
                let tok_t = Tensor::from_vec(vec![*tok], (1, 1), &self.device)?;
                codec_hiddens.push(
                    self.talker
                        .code_predictor
                        .get_embedding(i)?
                        .forward(&tok_t)?,
                );
            }
            let mut sum = codec_hiddens[0].clone();
            for h in codec_hiddens.iter().skip(1) {
                sum = (sum + h)?;
            }
            let mut step_embed = sum;
            if step < trailing_text.dim(1)? {
                let t = trailing_text.narrow(1, step, 1)?;
                step_embed = (step_embed + t)?;
            } else {
                step_embed = (step_embed + tts_pad.clone())?;
            }

            let pos = self.make_position_ids(1, seqlen_offset)?;
            let hs_step = self
                .talker
                .model
                .forward(&step_embed, &pos, seqlen_offset)?;
            seqlen_offset += 1;
            let last = hs_step.i((.., hs_step.dim(1)? - 1, ..))?.unsqueeze(1)?;
            past_hidden = last.clone();
            let logits = last.apply(&self.talker.codec_head)?;
            let logits = logits.squeeze(0)?.squeeze(0)?;
            let mut logits =
                apply_repetition_penalty(&logits, &generated_tokens, params.repetition_penalty)?;
            if suppress_from < self.talker.cfg.vocab_size {
                let mut logits_vec = logits.to_vec1::<f32>()?;
                for idx in suppress_from..self.talker.cfg.vocab_size {
                    if idx != suppress_eos {
                        logits_vec[idx] = f32::NEG_INFINITY;
                    }
                }
                logits = Tensor::from_vec(logits_vec, (self.talker.cfg.vocab_size,), &self.device)?;
            }
            let token = lp.sample(&logits)? as i64;
            generated_tokens.push(token);
            token0 = token;
        }

        Ok(output_codes)
    }
}
