//! Qwen3 implementation with quantization support.
//!
//! Based on the Qwen3 architecture and implemented with quantized weights
//! for reduced memory usage and faster inference on compatible hardware.
//!
//! References:
//! - [Qwen3 Models](https://huggingface.co/Qwen/Qwen3-0.6B) (architecture based on official implementations)
//!
use super::with_tracing::QMatMul;
use crate::{quantized_nn::RmsNorm, utils::repeat_kv};
use candle::quantized::{gguf_file, QTensor};
use candle::{DType, Device, Result, Storage, Tensor};
use candle_nn::attention::cpu_flash::causal::causal_decode_f32_interleaved;
use candle_nn::attention::{flash_attn, AttnMask};
use candle_nn::kv_cache::{ConcatKvCache, InterleavedKvCache, RawInterleavedKvCache};
use candle_nn::{Activation, Embedding, Module};
use std::io::{Read, Seek};
use std::sync::Arc;

pub struct Gguf<R: Read + Seek> {
    ct: gguf_file::Content,
    reader: R,
    device: Device,
}

impl<R: Read + Seek> Gguf<R> {
    pub fn new(ct: gguf_file::Content, reader: R, device: Device) -> Self {
        Self { ct, reader, device }
    }

    pub fn qmatmul(&mut self, name: &str) -> Result<QMatMul> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        QMatMul::from_weights(ws.into())
    }

    pub fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        RmsNorm::from_qtensor(ws, eps)
    }

    pub fn metadata(&self) -> &std::collections::HashMap<String, gguf_file::Value> {
        &self.ct.metadata
    }

    pub fn tensor(&mut self, name: &str) -> Result<QTensor> {
        self.ct.tensor(&mut self.reader, name, &self.device)
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

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    /// Pre-extracted flat f32 cos/sin for fused decode (zero allocation)
    cos_f32: Vec<f32>,
    sin_f32: Vec<f32>,
    half_d: usize,
}

impl RotaryEmbedding {
    pub fn new(
        dtype: DType,
        head_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        dev: &Device,
    ) -> Result<Self> {
        let dim = head_dim;
        let max_seq_len = max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let sin_t = freqs.sin()?;
        let cos_t = freqs.cos()?;
        let cos_f32 = cos_t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let sin_f32 = sin_t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        Ok(Self {
            sin: sin_t,
            cos: cos_t,
            cos_f32,
            sin_f32,
            half_d: dim / 2,
        })
    }

    /// Apply RoPE (q, k shape: B x H x L x D)
    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let sin = self.sin.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }

    /// Zero-allocation cos/sin slices for a single position.
    #[inline]
    pub fn cos_sin_at(&self, pos: usize) -> (&[f32], &[f32]) {
        let start = pos * self.half_d;
        let end = start + self.half_d;
        (&self.cos_f32[start..end], &self.sin_f32[start..end])
    }
}

#[derive(Debug, Clone)]
struct AttentionWeights {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: ConcatKvCache,
    interleaved_cache: InterleavedKvCache,
    raw_cache: RawInterleavedKvCache,
    use_flash_attn: bool,
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
        use_flash_attn: bool,
        prefix: &str,
    ) -> Result<Self> {
        let num_kv_groups = num_heads / num_kv_heads;
        let hidden_size = num_heads * head_dim;

        let q_proj = gg.qmatmul(&format!("{prefix}.attn_q.weight"))?;
        let k_proj = gg.qmatmul(&format!("{prefix}.attn_k.weight"))?;
        let v_proj = gg.qmatmul(&format!("{prefix}.attn_v.weight"))?;
        let o_proj = gg.qmatmul(&format!("{prefix}.attn_output.weight"))?;

        let q_norm = gg.rms_norm(&format!("{prefix}.attn_q_norm.weight"), rms_norm_eps)?;
        let k_norm = gg.rms_norm(&format!("{prefix}.attn_k_norm.weight"), rms_norm_eps)?;

        let kv_cache = ConcatKvCache::new(2);
        let interleaved_cache = InterleavedKvCache::new(head_dim);
        let raw_cache = RawInterleavedKvCache::new(num_kv_heads, head_dim, 4096);

        let span_attn = tracing::span!(tracing::Level::TRACE, "attn");

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
            hidden_size,
            rotary_emb,
            kv_cache,
            interleaved_cache,
            raw_cache,
            use_flash_attn,
            span_attn,
        })
    }

    fn forward(&mut self, x: &Tensor, attn_mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b, l, _) = x.dims3()?;

        // QKV projections
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Per-head Q/K norms (must stay as tensor ops)
        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.num_kv_heads, l, self.head_dim))?;

        // RoPE
        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;

        if self.use_flash_attn && x.device().is_cpu() {
            let scale = 1.0 / (self.head_dim as f32).sqrt();

            if l == 1 && b == 1 && q.dtype() == DType::F32 {
                // ── Fused decode: extract raw slices → raw cache → kernel ──
                // Q: (1, H_q, 1, D) → flat (H_q * D)
                let q_cont = q.squeeze(0)?.squeeze(1)?.contiguous()?;
                let (q_g, q_l) = q_cont.storage_and_layout();
                let q_data: &[f32] = match &*q_g {
                    Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[q_l.start_offset()..],
                    _ => candle::bail!("Expected CPU storage"),
                };

                // K: (1, H_kv, 1, D) → flat (H_kv * D)
                let k_cont = k.squeeze(0)?.squeeze(1)?.contiguous()?;
                let (k_g, k_l) = k_cont.storage_and_layout();
                let k_data: &[f32] = match &*k_g {
                    Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[k_l.start_offset()..],
                    _ => candle::bail!("Expected CPU storage"),
                };

                // V: (1, H_kv, 1, D) → flat (H_kv * D)
                let v_cont = v.squeeze(0)?.squeeze(1)?.contiguous()?;
                let (v_g, v_l) = v_cont.storage_and_layout();
                let v_data: &[f32] = match &*v_g {
                    Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[v_l.start_offset()..],
                    _ => candle::bail!("Expected CPU storage"),
                };

                // Write K, V into raw cache (no tensor allocation)
                let k_len = self.num_kv_heads * self.head_dim;
                self.raw_cache.write_kv(&k_data[..k_len], &v_data[..k_len]);

                // Run interleaved decode kernel
                let kv_len = self.raw_cache.len();
                let q_len = self.num_heads * self.head_dim;
                let ctx = causal_decode_f32_interleaved(
                    &q_data[..q_len],
                    self.raw_cache.data(),
                    self.num_heads, self.num_kv_heads, self.head_dim, kv_len,
                    scale,
                )?;

                let ctx = ctx.unsqueeze(0)?.transpose(1, 2)?;
                ctx.reshape((b, l, self.hidden_size))?.apply(&self.o_proj)
            } else {
                // ── Prefill: use InterleavedKvCache + flash_attn, populate raw cache ──
                let kv = self.interleaved_cache.append(&k, &v)?;

                // Populate raw cache for subsequent decode steps
                {
                    let k_cont = k.squeeze(0)?.transpose(0, 1)?.contiguous()?;
                    let v_cont = v.squeeze(0)?.transpose(0, 1)?.contiguous()?;
                    let (kg, kl) = k_cont.storage_and_layout();
                    let k_d: &[f32] = match &*kg {
                        Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[kl.start_offset()..],
                        _ => candle::bail!("Expected CPU"),
                    };
                    let (vg, vl) = v_cont.storage_and_layout();
                    let v_d: &[f32] = match &*vg {
                        Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[vl.start_offset()..],
                        _ => candle::bail!("Expected CPU"),
                    };
                    self.raw_cache.write_kv_batch(k_d, v_d, l);
                }

                let kv_k = kv.narrow(2, 0, self.head_dim)?.unsqueeze(0)?;
                let kv_v = kv.narrow(2, self.head_dim, self.head_dim)?.unsqueeze(0)?;

                let q = q.transpose(1, 2)?.contiguous()?;
                let k = kv_k.contiguous()?;
                let v = kv_v.contiguous()?;

                let ctx = flash_attn::<f32>(
                    &q, &k, &v, scale,
                    AttnMask::causal_with_offset(offset),
                    None, None,
                )?;
                let ctx = ctx.transpose(1, 2)?;
                ctx.reshape((b, l, self.hidden_size))?.apply(&self.o_proj)
            }
        } else {
            // Standard matmul attention (no flash)
            let (k, v) = self.kv_cache.append(&k, &v)?;

            let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
            let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

            let scale = 1.0 / (self.head_dim as f64).sqrt();
            let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
            if let Some(m) = attn_mask {
                let scores_dtype = scores.dtype();
                let mask = if m.dtype() != scores_dtype {
                    m.to_dtype(scores_dtype)?
                } else {
                    m.clone()
                };
                scores = scores.broadcast_add(&mask)?;
            }
            let probs = candle_nn::ops::softmax_last_dim(&scores)?;
            let ctx = probs.matmul(&v)?;
            let reshaped_ctx = ctx
                .transpose(1, 2)?
                .reshape((b, l, self.hidden_size))?;
            self.o_proj.forward(&reshaped_ctx)
        }
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
        self.interleaved_cache.reset();
        self.raw_cache.reset();
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    self_attn: AttentionWeights,
    mlp: MlpWeights,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl LayerWeights {
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary: Arc<RotaryEmbedding>,
        use_flash_attn: bool,
        layer_idx: usize,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");

        let ln1 = gg.rms_norm(&format!("{prefix}.attn_norm.weight"), rms_norm_eps)?;
        let ln2 = gg.rms_norm(&format!("{prefix}.ffn_norm.weight"), rms_norm_eps)?;
        let self_attn = AttentionWeights::new(
            gg,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            rotary,
            use_flash_attn,
            &prefix,
        )?;
        let mlp = MlpWeights::new(gg, &prefix)?;
        Ok(Self {
            self_attn,
            mlp,
            ln1,
            ln2,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.self_attn.forward(&h, mask, offset)?;
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = h2.apply(&self.mlp)?;
        x + h2
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    embed_tokens: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    lm_head: QMatMul,
    device: Device,
    dtype: DType,
    use_flash_attn: bool,
    span: tracing::Span,
    span_output: tracing::Span,
}

impl ModelWeights {
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        use_flash_attn: bool,
    ) -> Result<Self> {
        let mut gg = Gguf::new(ct, reader, device.clone());
        let md_get = |s: &str| match gg.metadata().get(s) {
            None => candle::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        let num_attention_heads = md_get("qwen3.attention.head_count")?.to_u32()? as usize;
        let num_kv_heads = md_get("qwen3.attention.head_count_kv")?.to_u32()? as usize;
        let head_dim = md_get("qwen3.attention.key_length")?.to_u32()? as usize;
        let num_layers = md_get("qwen3.block_count")?.to_u32()? as usize;
        let hidden_size = md_get("qwen3.embedding_length")?.to_u32()? as usize;
        let max_position_embeddings = md_get("qwen3.context_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("qwen3.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("qwen3.rope.freq_base")?.to_f32()? as f64;

        let dtype = match gg.metadata().get("general.dtype") {
            Some(v) => match v.to_u32() {
                Ok(0) => DType::F32,
                Ok(1) => DType::F16,
                _ => DType::F16,
            },
            None => DType::F16,
        };

        let embed_tensor = gg.tensor("token_embd.weight")?;
        let embed_tokens = Embedding::new(embed_tensor.dequantize(device)?, hidden_size);

        let rotary = Arc::new(RotaryEmbedding::new(
            dtype,
            head_dim,
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
                use_flash_attn,
                i,
            )?);
        }

        let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;
        // Load output projection tensor, falling back to tied embeddings like gemma3
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
            use_flash_attn,
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

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;
        // Skip mask materialization when using CPU flash attention
        let causal_mask = if l == 1 || (self.use_flash_attn && self.device.is_cpu()) {
            None
        } else {
            Some(self.causal_mask(b, l, offset, None)?)
        };
        for layer in &mut self.layers {
            h = layer.forward(&h, causal_mask.as_ref(), offset)?;
        }
        let h = self.norm.forward(&h)?;
        let _enter = self.span_output.enter();
        let last_hidden = h.narrow(1, l - 1, 1)?;
        self.lm_head.forward(&last_hidden)?.squeeze(1)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}
