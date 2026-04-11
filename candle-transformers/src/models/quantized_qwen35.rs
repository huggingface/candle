//! Qwen3.5 implementation with quantization support.
//!
//! Qwen3.5 is a hybrid SSM-Transformer architecture with combined QKV projections
//! and State Space Model (SSM) layers.
//!
//! References:
//! - [Qwen3 Models](https://huggingface.co/Qwen/Qwen3-0.6B)
//!
use super::with_tracing::QMatMul;
use crate::{quantized_nn::RmsNorm, utils::repeat_kv};
use candle::quantized::{gguf_file, QTensor};
use candle::{DType, Device, Result, Tensor, D};
use candle_nn::{kv_cache::ConcatKvCache, Activation, Embedding, Module};
use std::io::{Read, Seek};
use std::sync::Arc;

/// Reader wrapper for Qwen3.5 GGUF tensors and metadata.
struct Gguf<R: Read + Seek> {
    ct: gguf_file::Content,
    reader: R,
    device: Device,
}

impl<R: Read + Seek> Gguf<R> {
    /// Creates a GGUF reader on `device`.
    fn new(ct: gguf_file::Content, reader: R, device: Device) -> Self {
        Self { ct, reader, device }
    }

    /// Returns the current target device.
    fn device(&self) -> &Device {
        &self.device
    }

    /// Loads `name` as a quantized matrix multiplication weight.
    fn qmatmul(&mut self, name: &str) -> Result<QMatMul> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        QMatMul::from_weights(ws.into())
    }

    /// Loads `name` as an RMSNorm weight with the supplied epsilon.
    fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        RmsNorm::from_qtensor(ws, eps)
    }

    /// Returns the raw GGUF metadata map.
    fn metadata(&self) -> &std::collections::HashMap<String, gguf_file::Value> {
        &self.ct.metadata
    }

    /// Loads a raw GGUF tensor by name.
    fn tensor(&mut self, name: &str) -> Result<QTensor> {
        self.ct.tensor(&mut self.reader, name, &self.device)
    }

    /// Loads a raw GGUF tensor if it is present.
    fn tensor_or_none(&mut self, name: &str) -> Option<Result<QTensor>> {
        if !self.ct.tensor_infos.contains_key(name) {
            return None;
        }
        Some(self.tensor(name))
    }
}

#[derive(Debug, Clone)]
struct SSMWeights {
    attn_qkv: QMatMul,
    attn_gate: QMatMul,
    ssm_a: Tensor,
    ssm_conv1d: Tensor,
    ssm_dt_bias: Tensor,
    ssm_alpha: QMatMul,
    ssm_beta: QMatMul,
    ssm_norm: RmsNorm,
    ssm_out: QMatMul,
    group_count: usize,
    state_size: usize,
    inner_size: usize,
    value_head_count: usize,
    value_head_dim: usize,
    conv_kernel: usize,
    conv_state: Option<Tensor>,
    recurrent_state: Option<Tensor>,
    span: tracing::Span,
}

impl SSMWeights {
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        prefix: &str,
        rms_norm_eps: f64,
        group_count: usize,
        state_size: usize,
        inner_size: usize,
        conv_kernel: usize,
    ) -> Result<Self> {
        let attn_qkv = gg.qmatmul(&format!("{prefix}.attn_qkv.weight"))?;
        let attn_gate = gg.qmatmul(&format!("{prefix}.attn_gate.weight"))?;
        let ssm_a = gg.tensor(&format!("{prefix}.ssm_a"))?;
        let ssm_conv1d = gg.tensor(&format!("{prefix}.ssm_conv1d.weight"))?;
        let ssm_dt_bias = gg.tensor(&format!("{prefix}.ssm_dt.bias"))?;
        let ssm_alpha = gg.qmatmul(&format!("{prefix}.ssm_alpha.weight"))?;
        let ssm_beta = gg.qmatmul(&format!("{prefix}.ssm_beta.weight"))?;
        // Use the model epsilon from GGUF metadata here. Qwen3.5 has many
        // Gated DeltaNet layers, so even small normalization drift compounds
        // across the hybrid stack.
        let ssm_norm = gg.rms_norm(&format!("{prefix}.ssm_norm.weight"), rms_norm_eps)?;
        let ssm_out = gg.qmatmul(&format!("{prefix}.ssm_out.weight"))?;
        let ssm_a = ssm_a.dequantize(&gg.device)?;
        let value_head_count = ssm_a.elem_count();
        if value_head_count == 0 || !inner_size.is_multiple_of(value_head_count) {
            candle::bail!(
                "Qwen3.5 SSM value heads do not divide inner size: inner_size={}, value_heads={}",
                inner_size,
                value_head_count
            );
        }
        let value_head_dim = inner_size / value_head_count;

        let span = tracing::span!(tracing::Level::TRACE, "ssm");

        Ok(Self {
            attn_qkv,
            attn_gate,
            ssm_a,
            ssm_conv1d: ssm_conv1d.dequantize(&gg.device)?,
            ssm_dt_bias: ssm_dt_bias.dequantize(&gg.device)?,
            ssm_alpha,
            ssm_beta,
            ssm_norm,
            ssm_out,
            group_count,
            state_size,
            inner_size,
            value_head_count,
            value_head_dim,
            conv_kernel,
            conv_state: None,
            recurrent_state: None,
            span,
        })
    }

    fn apply_conv1d(&mut self, x: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, channels) = x.dims3()?;
        let conv_weight = self.ssm_conv1d.to_device(x.device())?.to_dtype(x.dtype())?;
        let (d0, d1) = conv_weight.dims2()?;
        let (weight, kernel) = if d0 == channels {
            (conv_weight, d1)
        } else if d1 == channels {
            (conv_weight.t()?, d0)
        } else {
            candle::bail!(
                "Qwen3.5 linear-attention conv layout mismatch: channels={}, raw_weight_dims=({}, {})",
                channels,
                d0,
                d1
            )
        };
        debug_assert_eq!(kernel, self.conv_kernel);

        let hist_len = kernel.saturating_sub(1);
        let history = match &self.conv_state {
            Some(state) if state.dims() == [b_sz, hist_len, channels] => state.clone(),
            _ => Tensor::zeros((b_sz, hist_len, channels), x.dtype(), x.device())?,
        };
        let padded = if hist_len == 0 {
            x.clone()
        } else {
            Tensor::cat(&[&history, x], 1)?
        };

        let mut outputs = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let mut acc = Tensor::zeros((b_sz, channels), x.dtype(), x.device())?;
            for k in 0..kernel {
                let src = padded.narrow(1, t + k, 1)?.squeeze(1)?;
                let weight = weight
                    .narrow(1, k, 1)?
                    .squeeze(1)?
                    .broadcast_as((b_sz, channels))?;
                acc = (&acc + src.broadcast_mul(&weight)?)?;
            }
            outputs.push(acc);
        }

        if hist_len > 0 {
            let total_len = hist_len + seq_len;
            self.conv_state = Some(padded.narrow(1, total_len - hist_len, hist_len)?);
        }
        Tensor::stack(&outputs, 1)
    }

    fn l2_normalize(xs: &Tensor) -> Result<Tensor> {
        let norm = (xs.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
        xs.broadcast_div(&norm)
    }

    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let span = self.span.clone();
        let _enter = span.enter();
        let input_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let (b_sz, seq_len, _hidden_size) = x.dims3()?;
        let key_dim = self.group_count * self.state_size;

        let mixed_qkv = self.attn_qkv.forward(&x)?;
        let conv = candle_nn::ops::silu(&self.apply_conv1d(&mixed_qkv)?)?;
        let q = conv.narrow(D::Minus1, 0, key_dim)?;
        let k = conv.narrow(D::Minus1, key_dim, key_dim)?;
        let v = conv.narrow(D::Minus1, key_dim * 2, self.inner_size)?;
        let z = self.attn_gate.forward(&x)?;

        let q = q.reshape((b_sz, seq_len, self.group_count, self.state_size))?;
        let k = k.reshape((b_sz, seq_len, self.group_count, self.state_size))?;
        let v = v.reshape((b_sz, seq_len, self.value_head_count, self.value_head_dim))?;
        let z = z.reshape((b_sz, seq_len, self.value_head_count, self.value_head_dim))?;

        let q = if self.value_head_count > self.group_count {
            let repeat_factor = self.value_head_count / self.group_count;
            q.unsqueeze(3)?
                .broadcast_as((
                    b_sz,
                    seq_len,
                    self.group_count,
                    repeat_factor,
                    self.state_size,
                ))?
                .transpose(2, 3)?
                .reshape((b_sz, seq_len, self.value_head_count, self.state_size))?
        } else {
            q
        };
        let k = if self.value_head_count > self.group_count {
            let repeat_factor = self.value_head_count / self.group_count;
            k.unsqueeze(3)?
                .broadcast_as((
                    b_sz,
                    seq_len,
                    self.group_count,
                    repeat_factor,
                    self.state_size,
                ))?
                .transpose(2, 3)?
                .reshape((b_sz, seq_len, self.value_head_count, self.state_size))?
        } else {
            k
        };

        let q = Self::l2_normalize(&q)?;
        let q = (&q * (1f64 / (self.state_size as f64).sqrt()))?;
        let k = Self::l2_normalize(&k)?;

        let dt_bias = self
            .ssm_dt_bias
            .to_device(x.device())?
            .to_dtype(x.dtype())?
            .reshape((1, 1, self.value_head_count))?
            .broadcast_as((b_sz, seq_len, self.value_head_count))?;
        let g = self.ssm_alpha.forward(&x)?;
        let g = ((&g + &dt_bias)?.exp()? + 1.)?.log()?;
        let a = self
            .ssm_a
            .to_device(x.device())?
            .to_dtype(x.dtype())?
            .reshape((1, 1, self.value_head_count))?
            .broadcast_as((b_sz, seq_len, self.value_head_count))?;
        let g = g.broadcast_mul(&a)?;
        let beta = candle_nn::ops::sigmoid(&self.ssm_beta.forward(&x)?)?;
        let state_shape = (
            b_sz,
            self.value_head_count,
            self.state_size,
            self.value_head_dim,
        );
        let mut state = match &self.recurrent_state {
            Some(state)
                if state.dims()
                    == [
                        b_sz,
                        self.value_head_count,
                        self.state_size,
                        self.value_head_dim,
                    ] =>
            {
                state.clone()
            }
            _ => Tensor::zeros(state_shape, DType::F32, x.device())?,
        };

        let mut outputs = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let q_t = q.narrow(1, t, 1)?.squeeze(1)?;
            let k_t = k.narrow(1, t, 1)?.squeeze(1)?;
            let v_t = v.narrow(1, t, 1)?.squeeze(1)?;
            let g_t = g.narrow(1, t, 1)?.squeeze(1)?.exp()?;
            let beta_t = beta.narrow(1, t, 1)?.squeeze(1)?;
            let g = g_t.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?;
            let beta = beta_t.unsqueeze(D::Minus1)?;
            state = state.broadcast_mul(&g)?;
            let retrieved = k_t
                .unsqueeze(D::Minus2)?
                .matmul(&state)?
                .squeeze(D::Minus2)?;
            let delta = (&v_t - &retrieved)?.broadcast_mul(&beta)?;
            let update = k_t
                .unsqueeze(D::Minus1)?
                .broadcast_mul(&delta.unsqueeze(D::Minus2)?)?;
            state = (&state + &update)?;
            let y_t = q_t
                .unsqueeze(D::Minus2)?
                .matmul(&state)?
                .squeeze(D::Minus2)?;
            outputs.push(y_t);
        }
        self.recurrent_state = Some(state);

        let y = Tensor::stack(&outputs, 1)?;
        let y = y.reshape((b_sz * seq_len * self.value_head_count, self.value_head_dim))?;
        let y = self.ssm_norm.forward(&y)?;
        let y = y.reshape((b_sz, seq_len, self.value_head_count, self.value_head_dim))?;
        let z = candle_nn::ops::silu(&z)?;
        let y = y.broadcast_mul(&z)?;
        let y = y.reshape((b_sz, seq_len, self.inner_size))?;
        let y = self.ssm_out.forward(&y)?;
        y.to_dtype(input_dtype)
    }

    fn clear_state(&mut self) {
        self.conv_state = None;
        self.recurrent_state = None;
    }
}

#[derive(Debug, Clone)]
struct MlpWeights {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
    act_fn: Activation,
    span: tracing::Span,
}

impl MlpWeights {
    fn new<R: Read + Seek>(gg: &mut Gguf<R>, prefix: &str) -> Result<Self> {
        let gate_proj = gg.qmatmul(&format!("{prefix}.ffn_gate.weight"))?;
        let up_proj = gg.qmatmul(&format!("{prefix}.ffn_up.weight"))?;
        let down_proj = gg.qmatmul(&format!("{prefix}.ffn_down.weight"))?;
        let act_fn = Activation::Silu;
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn,
            span,
        })
    }
}

impl Module for MlpWeights {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let gate = self.gate_proj.forward(x)?.apply(&self.act_fn)?;
        let up = self.up_proj.forward(x)?;
        let gated = (gate * up)?;
        self.down_proj.forward(&gated)
    }
}

/// Rotary position embedding table for Qwen3.5 text attention.
#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    rotary_dim: usize,
}

impl RotaryEmbedding {
    /// Builds the rotary sine/cosine cache from GGUF metadata.
    fn new(
        dtype: DType,
        rotary_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        dev: &Device,
    ) -> Result<Self> {
        let dim = rotary_dim;
        let max_seq_len = max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f64 / rope_theta.powf(i as f64 / dim as f64) as f64)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
            rotary_dim,
        })
    }

    /// Applies partial rotary embeddings to query and key tensors at `offset`.
    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self
            .cos
            .narrow(0, offset, seq_len)?
            .to_device(q.device())?
            .to_dtype(q.dtype())?;
        let sin = self
            .sin
            .narrow(0, offset, seq_len)?
            .to_device(q.device())?
            .to_dtype(q.dtype())?;
        let q_rot = q.narrow(D::Minus1, 0, self.rotary_dim)?;
        let k_rot = k.narrow(D::Minus1, 0, self.rotary_dim)?;
        let q_pass = q.narrow(D::Minus1, self.rotary_dim, q.dims()[3] - self.rotary_dim)?;
        let k_pass = k.narrow(D::Minus1, self.rotary_dim, k.dims()[3] - self.rotary_dim)?;
        let q_rot = candle_nn::rotary_emb::rope(&q_rot.contiguous()?, &cos, &sin)?;
        let k_rot = candle_nn::rotary_emb::rope(&k_rot.contiguous()?, &cos, &sin)?;
        let q_embed = Tensor::cat(&[&q_rot, &q_pass], D::Minus1)?.contiguous()?;
        let k_embed = Tensor::cat(&[&k_rot, &k_pass], D::Minus1)?.contiguous()?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
enum QKVStyle {
    Combined(QMatMul),
    Separate { q: QMatMul, k: QMatMul, v: QMatMul },
}

#[derive(Debug, Clone)]
struct AttentionWeights {
    qkv_style: QKVStyle,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    o_proj: QMatMul,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: ConcatKvCache,
    span_attn: tracing::Span,
}

impl AttentionWeights {
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary_emb: Arc<RotaryEmbedding>,
        prefix: &str,
    ) -> Result<Self> {
        let num_kv_groups = num_heads / num_kv_heads;

        // Qwen3.5 uses either combined QKV or separate Q/K/V projections
        // Check which style this block uses
        let qkv_style = match gg.tensor_or_none(&format!("{prefix}.attn_qkv.weight")) {
            Some(Ok(t)) => QKVStyle::Combined(QMatMul::from_weights(t.into())?),
            _ => {
                // Use separate Q, K, V projections
                let q = gg.qmatmul(&format!("{prefix}.attn_q.weight"))?;
                let k = gg.qmatmul(&format!("{prefix}.attn_k.weight"))?;
                let v = gg.qmatmul(&format!("{prefix}.attn_v.weight"))?;
                QKVStyle::Separate { q, k, v }
            }
        };

        let o_proj = gg.qmatmul(&format!("{prefix}.attn_output.weight"))?;

        // Q/K normalization (if separate projections)
        let q_norm = match gg.tensor_or_none(&format!("{prefix}.attn_q_norm.weight")) {
            Some(Ok(t)) => Some(RmsNorm::from_qtensor(t, rms_norm_eps)?),
            _ => None,
        };
        let k_norm = match gg.tensor_or_none(&format!("{prefix}.attn_k_norm.weight")) {
            Some(Ok(t)) => Some(RmsNorm::from_qtensor(t, rms_norm_eps)?),
            _ => None,
        };

        let kv_cache = ConcatKvCache::new(2);

        let span_attn = tracing::span!(tracing::Level::TRACE, "attn");

        Ok(Self {
            qkv_style,
            q_norm,
            k_norm,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            rotary_emb,
            kv_cache,
            span_attn,
        })
    }

    fn forward(&mut self, x: &Tensor, attn_mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b, l, _hidden_dim) = x.dims3()?;
        // Handle both QKV styles
        let (q, k, v, gate) = match &self.qkv_style {
            QKVStyle::Combined(qkv_proj) => {
                let qkv = qkv_proj.forward(x)?;
                let qkv =
                    qkv.reshape((b, l, self.num_heads + 2 * self.num_kv_heads, self.head_dim))?;
                let qkv = qkv.transpose(1, 2)?;
                let q = qkv.narrow(1, 0, self.num_heads)?;
                let k = qkv.narrow(1, self.num_heads, self.num_kv_heads)?;
                let v = qkv.narrow(1, self.num_heads + self.num_kv_heads, self.num_kv_heads)?;
                (q, k, v, None)
            }
            QKVStyle::Separate { q, k, v } => {
                // Q packs both query and output gate as [num_heads, head_dim * 2].
                let q_out = q.forward(x)?;
                let k_out = k.forward(x)?;
                let v_out = v.forward(x)?;
                let q_out_dim = q_out.dims()[2];
                let k_out_dim = k_out.dims()[2];
                let q_heads_actual = q_out_dim / (self.head_dim * 2);
                let kv_heads_actual = k_out_dim / self.head_dim;
                let q_and_gate = q_out.reshape((b, l, q_heads_actual, self.head_dim * 2))?;
                let q_full = q_and_gate
                    .narrow(D::Minus1, 0, self.head_dim)?
                    .transpose(1, 2)?;
                let q = q_full.narrow(1, 0, self.num_heads)?;
                let gate = q_and_gate
                    .narrow(2, 0, self.num_heads)?
                    .narrow(D::Minus1, self.head_dim, self.head_dim)?
                    .reshape((b, l, self.num_heads * self.head_dim))?;

                let k = k_out
                    .reshape((b, l, kv_heads_actual, self.head_dim))?
                    .transpose(1, 2)?;
                let v = v_out
                    .reshape((b, l, kv_heads_actual, self.head_dim))?
                    .transpose(1, 2)?;
                (q, k, v, Some(gate))
            }
        };

        // Apply Q/K norms if present (only for Separate mode where they exist)
        // For Separate mode, Q may have more heads than self.num_heads
        let actual_q_heads = q.dims()[1];
        let actual_kv_heads = k.dims()[1];
        let q = if let Some(ref q_norm) = self.q_norm {
            let q_flat = q.flatten(0, 2)?;
            let q_flat = q_norm.forward(&q_flat)?;
            q_flat.reshape((b, actual_q_heads, l, self.head_dim))?
        } else {
            q
        };
        let k = if let Some(ref k_norm) = self.k_norm {
            let k_flat = k.flatten(0, 2)?;
            let k_flat = k_norm.forward(&k_flat)?;
            k_flat.reshape((b, actual_kv_heads, l, self.head_dim))?
        } else {
            k
        };

        // For Separate mode with different Q and KV heads, compute correct grouping
        let num_kv_groups = if actual_q_heads != actual_kv_heads {
            actual_q_heads / actual_kv_heads
        } else {
            self.num_kv_groups
        };

        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;

        let (k, v) = self.kv_cache.append(&k, &v)?;

        let k = repeat_kv(k, num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, num_kv_groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = attn_mask {
            let m_dtype = m.dtype();
            let scores_dtype = scores.dtype();
            let mask = if m_dtype != scores_dtype {
                m.to_dtype(scores_dtype)?
            } else {
                m.clone()
            };
            scores = scores.broadcast_add(&mask)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;
        let ctx = ctx.transpose(1, 2)?;
        let (b_ctx, l_ctx, actual_heads, head_dim) = ctx.dims4()?;
        let reshaped_ctx = ctx.reshape((b_ctx, l_ctx, actual_heads * head_dim))?;
        let gated_ctx = if let Some(gate) = gate {
            reshaped_ctx.broadcast_mul(&candle_nn::ops::sigmoid(&gate)?)?
        } else {
            reshaped_ctx
        };

        self.o_proj.forward(&gated_ctx)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    self_attn: Option<AttentionWeights>,
    linear_attn: Option<SSMWeights>,
    mlp: MlpWeights,
    ln1: RmsNorm,
    ln2: RmsNorm,
    device: Device,
}

impl LayerWeights {
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary: Arc<RotaryEmbedding>,
        layer_idx: usize,
        ssm_group_count: usize,
        ssm_state_size: usize,
        ssm_inner_size: usize,
        ssm_conv_kernel: usize,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");

        // Qwen3.5 uses attn_norm instead of attention_norm
        let ln1 = gg.rms_norm(&format!("{prefix}.attn_norm.weight"), rms_norm_eps)?;
        // Qwen3.5 uses post_attention_norm instead of ffn_norm
        let ln2 = gg.rms_norm(
            &format!("{prefix}.post_attention_norm.weight"),
            rms_norm_eps,
        )?;

        let mlp = MlpWeights::new(gg, &prefix)?;
        let is_linear_attention =
            matches!(gg.tensor_or_none(&format!("{prefix}.ssm_a")), Some(Ok(_)));
        let (self_attn, linear_attn) = if is_linear_attention {
            (
                None,
                Some(SSMWeights::new(
                    gg,
                    &prefix,
                    rms_norm_eps,
                    ssm_group_count,
                    ssm_state_size,
                    ssm_inner_size,
                    ssm_conv_kernel,
                )?),
            )
        } else {
            (
                Some(AttentionWeights::new(
                    gg,
                    num_attention_heads,
                    num_key_value_heads,
                    head_dim,
                    rms_norm_eps,
                    rotary,
                    &prefix,
                )?),
                None,
            )
        };

        Ok(Self {
            self_attn,
            linear_attn,
            mlp,
            ln1,
            ln2,
            device: gg.device().clone(),
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let x = if x.device().same_device(&self.device) {
            x.clone()
        } else {
            x.to_device(&self.device)?
        };
        let x = if self.device.is_cpu() && x.dtype() != DType::F32 {
            x.to_dtype(DType::F32)?
        } else {
            x
        };
        let mask = match mask {
            Some(mask) if mask.device().same_device(&self.device) => Some(mask.clone()),
            Some(mask) => Some(mask.to_device(&self.device)?),
            None => None,
        };
        let h = self
            .ln1
            .forward(&x)
            .map_err(|e| candle::Error::msg(format!("ln1 failed: {e}")))?;
        let h = if h.dtype() == DType::F32 {
            h
        } else {
            h.to_dtype(DType::F32)?
        };
        let h = match (&mut self.self_attn, &mut self.linear_attn) {
            (Some(attn), None) => attn
                .forward(&h, mask.as_ref(), offset)
                .map_err(|e| candle::Error::msg(format!("attention mixer failed: {e}")))?,
            (None, Some(linear_attn)) => linear_attn
                .forward(&h)
                .map_err(|e| candle::Error::msg(format!("ssm mixer failed: {e}")))?,
            _ => candle::bail!("invalid Qwen3.5 layer mixer configuration"),
        };
        let h = if h.dtype() == x.dtype() {
            h
        } else {
            h.to_dtype(x.dtype())?
        };
        let x = (&x + h)?;

        let h2 = self
            .ln2
            .forward(&x)
            .map_err(|e| candle::Error::msg(format!("ln2 failed: {e}")))?;
        let h2 = if h2.dtype() == DType::F32 {
            h2
        } else {
            h2.to_dtype(DType::F32)?
        };
        let h2 = h2
            .apply(&self.mlp)
            .map_err(|e| candle::Error::msg(format!("mlp failed: {e}")))?;
        let h2 = if h2.dtype() == x.dtype() {
            h2
        } else {
            h2.to_dtype(x.dtype())?
        };
        x + h2
    }

    fn clear_kv_cache(&mut self) {
        if let Some(self_attn) = &mut self.self_attn {
            self_attn.clear_kv_cache();
        }
        if let Some(linear_attn) = &mut self.linear_attn {
            linear_attn.clear_state();
        }
    }
}

/// Quantized Qwen3.5 model weights loaded from GGUF.
#[derive(Debug, Clone)]
pub struct ModelWeights {
    embed_tokens: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    lm_head: QMatMul,
    device: Device,
    dtype: DType,
    span: tracing::Span,
    span_output: tracing::Span,
}

impl ModelWeights {
    /// Loads Qwen3.5 weights and architecture metadata from a GGUF file.
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let mut gg = Gguf::new(ct, reader, device.clone());

        let md_get = |s: &str| match gg.metadata().get(s) {
            None => candle::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        let num_attention_heads = md_get("qwen35.attention.head_count")?.to_u32()? as usize;
        let head_dim = md_get("qwen35.attention.key_length")?.to_u32()? as usize;
        let num_layers = md_get("qwen35.block_count")?.to_u32()? as usize;
        let hidden_size = md_get("qwen35.embedding_length")?.to_u32()? as usize;

        let num_kv_heads = md_get("qwen35.attention.head_count_kv")?.to_u32()? as usize;
        let max_position_embeddings = md_get("qwen35.context_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("qwen35.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("qwen35.rope.freq_base")?.to_f32()? as f64;
        let rope_dim = md_get("qwen35.rope.dimension_count")?.to_u32()? as usize;
        let ssm_group_count = md_get("qwen35.ssm.group_count")?.to_u32()? as usize;
        let ssm_state_size = md_get("qwen35.ssm.state_size")?.to_u32()? as usize;
        let ssm_inner_size = md_get("qwen35.ssm.inner_size")?.to_u32()? as usize;
        let ssm_conv_kernel = md_get("qwen35.ssm.conv_kernel")?.to_u32()? as usize;

        let dtype = match gg.metadata().get("general.dtype") {
            Some(v) => match v.to_u32() {
                Ok(0) => DType::F32,
                Ok(1) => DType::F16,
                _ => DType::F16,
            },
            None => DType::F16,
        };

        let embed_tensor = gg.tensor("token_embd.weight")?;
        let embed_weight = if device.is_cuda() || device.is_metal() {
            match embed_tensor.dequantize_f16(device) {
                Ok(weight) => weight,
                Err(_) => embed_tensor
                    .dequantize(device)?
                    .to_dtype(DType::F16)?,
            }
        } else {
            embed_tensor.dequantize(device)?
        };
        let embed_tokens = Embedding::new(embed_weight, hidden_size);
        let rotary = Arc::new(RotaryEmbedding::new(
            dtype,
            rope_dim,
            max_position_embeddings,
            rope_freq_base,
            device,
        )?);

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(LayerWeights::new(
                &mut gg,
                num_attention_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
                rotary.clone(),
                i,
                ssm_group_count,
                ssm_state_size,
                ssm_inner_size,
                ssm_conv_kernel,
            )?);
        }

        let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;
        let lm_head_tensor = match gg.tensor("output.weight") {
            Ok(tensor) => tensor,
            Err(_) => gg.tensor("token_embd.weight")?,
        };
        let lm_head = QMatMul::from_weights(lm_head_tensor.into())?;
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype,
            span,
            span_output,
        })
    }

    fn causal_mask(
        &self,
        b: usize,
        tgt: usize,
        offset: usize,
        sw: Option<usize>,
    ) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| {
                (0..(tgt + offset)).map(move |j| {
                    let past_ok = j <= i + offset;
                    let sw_ok = match sw {
                        Some(w) => (i + offset) as i64 - j as i64 <= w as i64,
                        None => true,
                    };
                    if past_ok && sw_ok {
                        0.
                    } else {
                        minf
                    }
                })
            })
            .collect();
        Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?.to_dtype(self.dtype)
    }

    /// Runs a forward pass and returns logits for the final input token.
    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b, l) = input.dims2()?;
        let embed_device = self.embed_tokens.embeddings().device().clone();
        let input = if input.device().same_device(&embed_device) {
            input.clone()
        } else {
            input.to_device(&embed_device)?
        };
        let mut h = self.embed_tokens.forward(&input)?;
        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset, None)?)
        };
        for (idx, layer) in self.layers.iter_mut().enumerate() {
            h = layer
                .forward(&h, causal_mask.as_ref(), offset)
                .map_err(|e| candle::Error::msg(format!("layer {idx} failed: {e}")))?;
        }
        if !h.device().same_device(&self.device) {
            h = h.to_device(&self.device)?;
        }
        let h = self.norm.forward(&h)?;
        let _enter = self.span_output.enter();
        let last_hidden = h.narrow(1, l - 1, 1)?;
        self.lm_head.forward(&last_hidden)?.squeeze(1)
    }

    /// Clears transformer KV-cache and SSM recurrent state for all layers.
    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}
