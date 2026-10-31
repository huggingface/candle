//! Muse-Glimmer implementation with quantization support.
//!
//! Muse-Glimmer is a 30B-parameter model from Meta with a hybrid
//! sliding-window / full-attention transformer architecture.
//!
//! Key characteristics:
//! - 52 layers, hidden 6656, FFN 19968, vocabulary 202048
//! - Group-Query Attention: 32 query heads, 2 KV heads, head_dim 128
//! - Sliding-window attention (window 2048, RoPE base 5e5) on most layers;
//!   every 4th layer is full attention without RoPE (the per-layer pattern
//!   comes from the GGUF metadata `muse-glimmer.attention.sliding_window_pattern`)
//! - Per-head QK-norm (RMS over head_dim; the q-norm weight carries the
//!   qk_scale_factor baked in at conversion time) and a per-layer
//!   attention output gate: `sigmoid(attn_gate @ pre_norm) * attn_out`
//!   applied before the output projection
//! - The embeddings are RMS-normalized (weightless) before the layers
//! - Post-attention / post-FFN norms: weightless RMS norm (eps 1e-8,
//!   different from the 1e-5 layer-norm eps) followed by multiplication
//!   by the stored weights
//! - Logits scaled by `muse-glimmer.logit_scale` and tanh-soft-capped at
//!   `muse-glimmer.final_logit_softcapping` (20)
//!
//! References:
//! - [Muse-Glimmer Models](https://huggingface.co/meta-models/Muse-Glimmer-30B)
//! - [GGUF quantizations](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF)

use super::with_tracing::QMatMul;
use crate::{
    quantized_nn::{cat_quantized_weights, RmsNorm},
    utils::{build_additive_causal_mask, repeat_kv},
};
use candle::quantized::gguf_file;
use candle::quantized::gguf_file::Value;
use candle::quantized::QStorage;
use candle::quantized::QTensor;
use candle::{DType, Device, IndexOp, Result, Storage, Tensor};
use candle_nn::attention::cpu_flash::causal::causal_decode_f32_interleaved;
use candle_nn::kv_cache::{PreallocKvCache, RawInterleavedKvCache};
use candle_nn::{Activation, Embedding, Module, RmsNorm as CnnRmsNorm};
use std::io::{Read, Seek};
use std::sync::Arc;

/// Maximum number of cached positions in the (pre-allocated) KV cache;
/// generation fails with a clear error past this limit.
pub const MAX_KV_CACHE_LEN: usize = 8192;

/// Epsilon of the post-attention / post-FFN (weightless) norms, as in the
/// reference implementation (distinct from the 1e-5 layer-norm eps).
const POST_NORM_EPS: f64 = 1e-8;

struct Gguf<R: Read + Seek> {
    ct: gguf_file::Content,
    reader: R,
    device: Device,
}

impl<R: Read + Seek> Gguf<R> {
    fn new(ct: gguf_file::Content, reader: R, device: Device) -> Self {
        Self { ct, reader, device }
    }

    fn qmatmul(&mut self, name: &str) -> Result<QMatMul> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        QMatMul::from_weights(ws.into())
    }

    fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        RmsNorm::from_qtensor(ws, eps)
    }

    fn metadata(&self) -> &std::collections::HashMap<String, Value> {
        &self.ct.metadata
    }

    fn tensor(&mut self, name: &str) -> Result<QTensor> {
        self.ct.tensor(&mut self.reader, name, &self.device)
    }
}

/// The Q/K/V projections: either one fused matmul over the concatenated
/// weight rows (one dispatch per token instead of three) or three
/// separate matmuls when the weights cannot be fused at load time.
#[derive(Debug, Clone)]
enum QkvProj {
    Fused(QMatMul),
    Separate(QMatMul, QMatMul, QMatMul),
}

impl QkvProj {
    fn forward(&self, x: &Tensor, q_dim: usize, kv_dim: usize) -> Result<(Tensor, Tensor, Tensor)> {
        match self {
            Self::Fused(w) => {
                let out = w.forward(x)?;
                // The fused output's rows are strided (stride q+2kv); the
                // Vulkan binary/unary op kernels only index dense or
                // broadcast layouts, so make each projection's view
                // contiguous (no-op at decode, one small copy at prefill).
                Ok((
                    out.narrow(2, 0, q_dim)?.contiguous()?,
                    out.narrow(2, q_dim, kv_dim)?.contiguous()?,
                    out.narrow(2, q_dim + kv_dim, kv_dim)?.contiguous()?,
                ))
            }
            Self::Separate(q, k, v) => Ok((q.forward(x)?, k.forward(x)?, v.forward(x)?)),
        }
    }
}

/// The MLP gate/up projections: either one fused matmul over the
/// concatenated weight rows (one dispatch per token instead of two) or
/// two separate matmuls.
#[derive(Debug, Clone)]
enum GateUpProj {
    Fused(QMatMul, usize),
    Separate(QMatMul, QMatMul),
}

impl GateUpProj {
    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Fused(w, inter) => {
                let out = w.forward(x)?;
                // Make the strided narrow views contiguous: the Vulkan
                // binary-op kernel (silu, mul, ...) only indexes dense or
                // broadcast layouts, so gappy views must be materialized.
                Ok((
                    out.narrow(2, 0, *inter)?.contiguous()?,
                    out.narrow(2, *inter, *inter)?.contiguous()?,
                ))
            }
            Self::Separate(gate, up) => Ok((gate.forward(x)?, up.forward(x)?)),
        }
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_up: GateUpProj,
    down: QMatMul,
    span: tracing::Span,
}

impl Mlp {
    fn new<R: Read + Seek>(gg: &mut Gguf<R>, prefix: &str) -> Result<Self> {
        let gate_up = match cat_quantized_weights(
            &gg.ct,
            &mut gg.reader,
            &[
                format!("{prefix}.ffn_gate.weight"),
                format!("{prefix}.ffn_up.weight"),
            ],
            &gg.device,
        )? {
            Some((w, ns)) => {
                if ns.len() != 2 || ns[0] != ns[1] {
                    candle::bail!("ffn gate/up weights {:?} have mismatched dims", ns);
                }
                GateUpProj::Fused(QMatMul::from_weights(w.into())?, ns[0])
            }
            None => GateUpProj::Separate(
                gg.qmatmul(&format!("{prefix}.ffn_gate.weight"))?,
                gg.qmatmul(&format!("{prefix}.ffn_up.weight"))?,
            ),
        };
        let down = gg.qmatmul(&format!("{prefix}.ffn_down.weight"))?;
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        Ok(Self {
            gate_up,
            down,
            span,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (gate, up) = self.gate_up.forward(x)?;
        let gate = gate.silu()?;
        let gated = (gate * up)?;
        self.down.forward(&gated)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(
        dtype: DType,
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        dev: &Device,
    ) -> Result<Self> {
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / head_dim as f64) as f32)
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
        })
    }

    /// Apply RoPE (q, k shape: B x H x L x D). Muse-Glimmer uses the
    /// interleaved (rotate-pairs) convention: llama.cpp runs it with
    /// `rope type 0` (GGML_ROPE_TYPE_NORMAL), which rotates adjacent
    /// value pairs (2i, 2i+1) rather than the NeoX half-split.
    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let sin = self.sin.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let q_embed = candle_nn::rotary_emb::rope_i(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope_i(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    attention_qkv: QkvProj,
    attention_gate: QMatMul,
    attention_q_norm: RmsNorm,
    attention_k_norm: RmsNorm,
    attention_wo: QMatMul,
    attention_norm: RmsNorm,
    post_attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
    post_ffn_norm: RmsNorm,
    mlp: Mlp,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    q_dim: usize,
    /// Sliding-window size for SWA layers; `None` for full-attention
    /// layers (which also have no RoPE).
    sliding_window_size: Option<usize>,
    use_rope: bool,
    rotary_embedding: Arc<RotaryEmbedding>,
    // Pre-allocated KV cache (both backends): f16 on Vulkan, f32 on CPU.
    kv_cache: PreallocKvCache,
    // CPU: raw interleaved cache backing the fused decode kernel.
    raw_cache: Option<RawInterleavedKvCache>,
    span_attn: tracing::Span,
    span_mlp: tracing::Span,
}

impl LayerWeights {
    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b_sz, seq_len, _) = x.dims3()?;

        let (q, k, v) =
            self.attention_qkv
                .forward(x, self.q_dim, self.n_kv_head * self.head_dim)?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;

        // Per-head QK-norm (RMS over head_dim); the q-norm weight carries
        // the qk_scale_factor baked in at conversion time.
        let q = self.attention_q_norm.forward(&q.contiguous()?)?;
        let k = self.attention_k_norm.forward(&k.contiguous()?)?;

        // RoPE on the sliding-window layers only; full layers are NoPE.
        let (q, k) = if self.use_rope {
            self.rotary_embedding.apply(&q, &k, index_pos)?
        } else {
            (q, k)
        };

        // Attention output gate: sigmoid(attn_gate @ pre_norm).
        let gate = self
            .attention_gate
            .forward(x)?
            .apply(&Activation::Sigmoid)?;

        // CPU single-query decode: raw interleaved cache + fused
        // multi-threaded, prefetched, GQA-aware decode kernel (no tensor
        // allocations). Causal-only, so windowed layers past their window
        // (mask = Some) fall through to the matmul chain.
        if b_sz == 1 && seq_len == 1 && q.dtype() == DType::F32 && mask.is_none() {
            if let Some(rc) = self.raw_cache.as_mut() {
                // Keep the pre-allocated tensor cache in sync: once the
                // context grows past the sliding window, the matmul chain
                // below needs every cached row.
                self.kv_cache.append(&k, &v)?;
                let q_cont = q.squeeze(0)?.squeeze(1)?.contiguous()?;
                let (q_g, q_l) = q_cont.storage_and_layout();
                let q_data: &[f32] = match &*q_g {
                    Storage::Cpu(c) => &c.as_slice::<f32>()?[q_l.start_offset()..],
                    _ => candle::bail!("expected CPU storage"),
                };
                let k_cont = k.squeeze(0)?.squeeze(1)?.contiguous()?;
                let (k_g, k_l) = k_cont.storage_and_layout();
                let k_data: &[f32] = match &*k_g {
                    Storage::Cpu(c) => &c.as_slice::<f32>()?[k_l.start_offset()..],
                    _ => candle::bail!("expected CPU storage"),
                };
                let v_cont = v.squeeze(0)?.squeeze(1)?.contiguous()?;
                let (v_g, v_l) = v_cont.storage_and_layout();
                let v_data: &[f32] = match &*v_g {
                    Storage::Cpu(c) => &c.as_slice::<f32>()?[v_l.start_offset()..],
                    _ => candle::bail!("expected CPU storage"),
                };
                let k_len = self.n_kv_head * self.head_dim;
                rc.write_kv(&k_data[..k_len], &v_data[..k_len]);
                let ctx = causal_decode_f32_interleaved(
                    &q_data[..self.n_head * self.head_dim],
                    rc.data(),
                    self.n_head,
                    self.n_kv_head,
                    self.head_dim,
                    rc.len(),
                    1.0 / (self.head_dim as f32).sqrt(),
                )?;
                let ctx = ctx.reshape((b_sz, seq_len, self.q_dim))?;
                let ctx = (ctx * gate)?;
                return self.attention_wo.forward(&ctx);
            }
        }

        // CPU: populate the raw cache with the new rows so later decode
        // steps can use the fused raw kernel (the tensor cache below
        // serves the matmul prefill).
        if b_sz == 1 && q.dtype() == DType::F32 {
            if let Some(rc) = self.raw_cache.as_mut() {
                let k_cont = k.squeeze(0)?.transpose(0, 1)?.contiguous()?;
                let (k_g, k_l) = k_cont.storage_and_layout();
                let k_data: &[f32] = match &*k_g {
                    Storage::Cpu(c) => &c.as_slice::<f32>()?[k_l.start_offset()..],
                    _ => candle::bail!("expected CPU storage"),
                };
                let v_cont = v.squeeze(0)?.transpose(0, 1)?.contiguous()?;
                let (v_g, v_l) = v_cont.storage_and_layout();
                let v_data: &[f32] = match &*v_g {
                    Storage::Cpu(c) => &c.as_slice::<f32>()?[v_l.start_offset()..],
                    _ => candle::bail!("expected CPU storage"),
                };
                rc.write_kv_batch(k_data, v_data, seq_len);
            }
        }

        // The Vulkan KV cache is f16. For the prefill (seq_len > 1) the
        // matmul chain must run on the exact f32 K/V, not the f16 cache
        // round-trip — the f16 rounding of K/V diverges enough over the
        // deep stack to turn generation into a repetition loop. The decode
        // uses the f16 cache views (required by the fused decode kernel),
        // so the f32 copy is only needed for the prefill.
        let (k_prefill, v_prefill) = if seq_len > 1 {
            (Some(k.clone()), Some(v.clone()))
        } else {
            (None, None)
        };

        // Pre-allocated KV cache: in-place row append (a one-row copy
        // during decode) and strided views of the accumulated k/v; no
        // per-token cat or reallocation.
        let (k, v) = self.kv_cache.append(&k, &v)?;

        // Single-query decode on Vulkan: fused GQA-native attention (one
        // kernel pair; no repeat_kv, no matmul chain). Causal-only:
        // windowed layers past their window use the matmul chain.
        if b_sz == 1
            && seq_len == 1
            && mask.is_none()
            && q.dtype() == DType::F32
            && matches!(q.device(), Device::Vulkan(_))
        {
            let q2 = q.contiguous()?.reshape((self.n_head, self.head_dim))?;
            // squeeze(0) keeps the pre-allocated cache's row stride
            // (MAX_KV_CACHE_LEN * head_dim) as a pure view; the kernel
            // consumes it via its kv_stride parameter. A reshape would
            // materialize the whole (kv_len, n_kv_head, head_dim) matrix
            // every token (O(n^2) copy traffic).
            let k2 = k.squeeze(0)?;
            let v2 = v.squeeze(0)?;
            let scale = (1.0 / (self.head_dim as f64).sqrt()) as f32;
            let ctx = q2.fused_decode_attention(&k2, &v2, scale, self.n_kv_head)?;
            let ctx = ctx.reshape((b_sz, seq_len, self.q_dim))?;
            let ctx = (ctx * gate)?;
            return self.attention_wo.forward(&ctx);
        }

        // Prefill path (and windowed decodes past the sliding window):
        // repeat KV for GQA, then the matmul chain. Prefill runs the
        // matmul chain in the activation dtype; the cache is f16 on
        // Vulkan, so convert the views back if needed. For the prefill
        // use the exact f32 pre-cache K/V (see above); windowed decodes
        // (seq_len == 1) still read the f16 cache views.
        let (k, v) = if seq_len > 1 {
            (k_prefill.unwrap(), v_prefill.unwrap())
        } else {
            (
                if k.dtype() != q.dtype() {
                    k.to_dtype(q.dtype())?
                } else {
                    k
                },
                if v.dtype() != q.dtype() {
                    v.to_dtype(q.dtype())?
                } else {
                    v
                },
            )
        };
        let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        if let Some(mask) = mask {
            let mask_dtype = attn_weights.dtype();
            let mask = if mask.dtype() != mask_dtype {
                mask.to_dtype(mask_dtype)?
            } else {
                mask.clone()
            };
            attn_weights = attn_weights.broadcast_add(&mask)?;
        }

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.q_dim))?;
        let attn_output = (attn_output * gate)?;
        self.attention_wo.forward(&attn_output)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
        if let Some(rc) = &mut self.raw_cache {
            rc.reset();
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    tok_embeddings: Embedding,
    embd_norm: CnnRmsNorm,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    output: QMatMul,
    logit_scale: f32,
    logit_softcap: f32,
    span: tracing::Span,
    span_output: tracing::Span,
}

impl ModelWeights {
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

        let block_count = md_get("muse-glimmer.block_count")?.to_u32()? as usize;
        let embedding_length = md_get("muse-glimmer.embedding_length")?.to_u32()? as usize;
        let n_head = md_get("muse-glimmer.attention.head_count")?.to_u32()? as usize;
        let n_kv_head = md_get("muse-glimmer.attention.head_count_kv")?.to_u32()? as usize;
        let head_dim = md_get("muse-glimmer.attention.key_length")?.to_u32()? as usize;
        let rms_eps = md_get("muse-glimmer.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_base = md_get("muse-glimmer.rope.freq_base")?.to_f32()? as f64;
        let sliding_window = md_get("muse-glimmer.attention.sliding_window")?.to_u32()? as usize;
        // Per-layer SWA pattern, mirroring llama.cpp's
        // `load_swa_pattern(ml, 4)`: a per-layer array is used as-is; a
        // scalar key is honored only when it is a uint32 `n` (a layer is
        // SWA when `n == 0 || il % n < n - 1`); anything else (e.g. the
        // bool flag in the Muse GGUFs, which the llama.cpp loader ignores)
        // falls back to the default `n = 4` -> every 4th layer is full
        // attention without RoPE.
        let sliding_window_pattern: Vec<bool> =
            match md_get("muse-glimmer.attention.sliding_window_pattern")? {
                Value::Array(vals) => {
                    let v: Vec<bool> = vals
                        .iter()
                        .map(|v| matches!(v, Value::Bool(true)))
                        .collect();
                    (0..block_count)
                        .map(|il| v.get(il).copied().unwrap_or(false))
                        .collect()
                }
                Value::U32(n) => {
                    let n = *n as usize;
                    (0..block_count)
                        .map(|il| n == 0 || il % n < n - 1)
                        .collect()
                }
                _ => (0..block_count).map(|il| il % 4 < 3).collect(),
            };
        let logit_scale = md_get("muse-glimmer.logit_scale")?.to_f32()?;
        let logit_softcap = md_get("muse-glimmer.final_logit_softcapping")?.to_f32()?;

        let q_dim = n_head * head_dim;

        // RoPE applies only to the sliding-window layers; the tables are
        // bounded by the pre-allocated KV cache length.
        let rotary = Arc::new(RotaryEmbedding::new(
            DType::F32,
            head_dim,
            MAX_KV_CACHE_LEN,
            rope_base,
            device,
        )?);

        // The KV cache is f16 on Vulkan (halves VRAM and the decode
        // traffic); f32 on CPU (the raw decode kernel consumes f32 rows).
        let kv_dtype = if device.is_vulkan() {
            DType::F16
        } else {
            DType::F32
        };

        let mut layers = Vec::with_capacity(block_count);
        for (i, &is_swa) in sliding_window_pattern.iter().enumerate() {
            let prefix = format!("blk.{i}");

            let attention_norm = gg.rms_norm(&format!("{prefix}.attn_norm.weight"), rms_eps)?;
            let post_attention_norm = gg.rms_norm(
                &format!("{prefix}.post_attention_norm.weight"),
                POST_NORM_EPS,
            )?;
            let ffn_norm = gg.rms_norm(&format!("{prefix}.ffn_norm.weight"), rms_eps)?;
            let post_ffn_norm =
                gg.rms_norm(&format!("{prefix}.post_ffw_norm.weight"), POST_NORM_EPS)?;

            let attention_qkv = match cat_quantized_weights(
                &gg.ct,
                &mut gg.reader,
                &[
                    format!("{prefix}.attn_q.weight"),
                    format!("{prefix}.attn_k.weight"),
                    format!("{prefix}.attn_v.weight"),
                ],
                &gg.device,
            )? {
                Some((w, ns)) => {
                    if ns.len() != 3 || ns[0] != ns[1] || ns[1] != ns[2] {
                        candle::bail!("attn q/k/v weights {:?} have mismatched dims", ns);
                    }
                    QkvProj::Fused(QMatMul::from_weights(w.into())?)
                }
                None => QkvProj::Separate(
                    gg.qmatmul(&format!("{prefix}.attn_q.weight"))?,
                    gg.qmatmul(&format!("{prefix}.attn_k.weight"))?,
                    gg.qmatmul(&format!("{prefix}.attn_v.weight"))?,
                ),
            };

            let mlp = Mlp::new(&mut gg, &prefix)?;

            let span_attn = tracing::span!(tracing::Level::TRACE, "attention");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "mlp");
            layers.push(LayerWeights {
                attention_qkv,
                attention_gate: gg.qmatmul(&format!("{prefix}.attn_gate.weight"))?,
                attention_q_norm: gg.rms_norm(&format!("{prefix}.attn_q_norm.weight"), rms_eps)?,
                attention_k_norm: gg.rms_norm(&format!("{prefix}.attn_k_norm.weight"), rms_eps)?,
                attention_wo: gg.qmatmul(&format!("{prefix}.attn_output.weight"))?,
                attention_norm,
                post_attention_norm,
                ffn_norm,
                post_ffn_norm,
                mlp,
                n_head,
                n_kv_head,
                head_dim,
                q_dim,
                sliding_window_size: is_swa.then_some(sliding_window),
                use_rope: is_swa,
                rotary_embedding: rotary.clone(),
                kv_cache: PreallocKvCache::new(
                    1,
                    n_kv_head,
                    MAX_KV_CACHE_LEN,
                    head_dim,
                    kv_dtype,
                    device,
                )?,
                raw_cache: (device.is_cpu())
                    .then(|| RawInterleavedKvCache::new(n_kv_head, head_dim, MAX_KV_CACHE_LEN)),
                span_attn,
                span_mlp,
            });
        }

        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");

        Ok(Self {
            // Token embeddings: dequantizing to f32 on the device would
            // allocate one buffer past the 4 GiB `max_buffer_size` limit
            // (radv) for this model (202048 x 6656 x 4B = 5.4 GB). On
            // Vulkan, dequantize on the CPU and upload as f16 (2.7 GB).
            tok_embeddings: if device.is_vulkan() {
                let (bytes, dtype, shape) =
                    gg.ct.tensor_raw(&mut gg.reader, "token_embd.weight")?;
                let cpu_dev = Device::Cpu;
                let qt = QTensor::new(
                    QStorage::from_data(std::borrow::Cow::Owned(bytes), &cpu_dev, dtype)?,
                    shape,
                )?;
                let embd = qt
                    .dequantize(&cpu_dev)?
                    .to_dtype(DType::F16)?
                    .to_device(device)?;
                Embedding::new(embd, embedding_length)
            } else {
                Embedding::new(
                    gg.tensor("token_embd.weight")?.dequantize(device)?,
                    embedding_length,
                )
            },
            embd_norm: CnnRmsNorm::new(
                Tensor::ones(embedding_length, DType::F32, device)?,
                rms_eps,
            ),
            layers,
            norm: gg.rms_norm("output_norm.weight", rms_eps)?,
            output: QMatMul::from_weights(gg.tensor("output.weight")?.into())?,
            logit_scale,
            logit_softcap,
            span,
            span_output,
        })
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (_b, seq_len) = input.dims2()?;

        let layer_in = self.tok_embeddings.forward(input)?;
        // The embeddings are f16 on Vulkan; the rest of the network is f32.
        let mut layer_in = layer_in.to_dtype(DType::F32)?;
        // The embeddings are RMS-normalized (weightless) before the layers.
        layer_in = self.embd_norm.forward(&layer_in)?;

        for (li, layer) in self.layers.iter_mut().enumerate() {
            // Causal mask for prefill; sliding-window layers also need a
            // mask once the context grows past their window (the fast
            // decode kernels are causal-only).
            let attention_mask = if seq_len == 1
                && !layer
                    .sliding_window_size
                    .is_some_and(|w| offset + seq_len > w)
            {
                None
            } else {
                Some(build_additive_causal_mask(
                    seq_len,
                    offset,
                    layer.sliding_window_size,
                    input.device(),
                    input.dtype(),
                )?)
            };

            // Attention block
            let residual = &layer_in;
            let x = layer.attention_norm.forward(&layer_in)?;
            let x = layer.forward_attn(&x, attention_mask.as_ref(), offset)?;
            // Fused post-norm + residual add: one kernel instead of
            // RMSNorm + add (Vulkan f32); falls back to the decomposed
            // path on other backends.
            let x = layer.post_attention_norm.post_add_forward(residual, &x)?;

            // Feed-forward block
            let _enter = layer.span_mlp.enter();
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp.forward(&x)?;
            let x = layer.post_ffn_norm.post_add_forward(residual, &x)?;
            drop(_enter);

            layer_in = x;
            let _ = li;
        }

        let _enter = self.span_output.enter();
        let x = layer_in.i((.., seq_len - 1, ..))?;
        let x = self.norm.forward(&x)?;
        let output = self.output.forward(&x)?;
        // Logit scale + tanh softcap (from the reference implementation).
        let output = (output * (self.logit_scale as f64))?;
        let output = (output / (self.logit_softcap as f64))?;
        let output = (output.tanh()? * (self.logit_softcap as f64))?;

        Ok(output)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}
