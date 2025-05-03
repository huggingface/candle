//! Qwen3 implementation with quantization support.
//!
//! Based on the Qwen3 architecture and implemented with quantized weights
//! for reduced memory usage and faster inference on compatible hardware.
//!
//! References:
//! - [Qwen3 Models](https://huggingface.co/Qwen/Qwen3-0.6B) (architecture based on official implementations)
//!
use crate::{quantized_nn::RmsNorm, utils::repeat_kv};
use candle::quantized::gguf_file;
use candle::quantized::QTensor;
use candle::{DType, Device, Result, Tensor};
use candle_nn::{kv_cache::KvCache, Activation, Embedding, Module};
use std::io::{Read, Seek};
use std::sync::Arc;

#[derive(Debug, Clone)]
struct QMatMul {
    inner: candle::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        let inner = candle::quantized::QMatMul::from_qtensor(qtensor)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MlpWeights {
    gate_proj: QMatMulWrapper,
    up_proj: QMatMulWrapper,
    down_proj: QMatMulWrapper,
    act_fn: Activation,
    span: tracing::Span,
}

impl MlpWeights {
    pub(crate) fn new<R: Read + Seek>(
        ct: &gguf_file::Content,
        reader: &mut R,
        prefix: &str,
        device: &Device,
    ) -> Result<Self> {
        let gate_proj = QMatMulWrapper::from_qtensor(ct.tensor(
            reader,
            &format!("{prefix}.ffn_gate.weight"),
            device,
        )?)?;
        let up_proj = QMatMulWrapper::from_qtensor(ct.tensor(
            reader,
            &format!("{prefix}.ffn_up.weight"),
            device,
        )?)?;
        let down_proj = QMatMulWrapper::from_qtensor(ct.tensor(
            reader,
            &format!("{prefix}.ffn_down.weight"),
            device,
        )?)?;

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

// Re-use the Config struct from the unquantized version
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub attention_bias: bool,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub hidden_act: Activation,
}

// Wrapper for QMatMul to include tracing span
#[derive(Debug, Clone)]
struct QMatMulWrapper {
    inner: QMatMul,
    span: tracing::Span,
}

impl QMatMulWrapper {
    fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        let inner = QMatMul::from_qtensor(qtensor)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub(crate) fn new(
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
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    /// Apply RoPE (q, k shape: B x H x L x D)
    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let sin = self.sin.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct AttentionWeights {
    q_proj: QMatMulWrapper,
    k_proj: QMatMulWrapper,
    v_proj: QMatMulWrapper,
    o_proj: QMatMulWrapper,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: KvCache,
    span_attn: tracing::Span,
}

impl AttentionWeights {
    pub(crate) fn new<R: Read + Seek>(
        ct: &gguf_file::Content,
        reader: &mut R,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary_emb: Arc<RotaryEmbedding>,
        prefix: &str,
        device: &Device,
    ) -> Result<Self> {
        let num_kv_groups = num_heads / num_kv_heads;

        let q_proj = QMatMulWrapper::from_qtensor(ct.tensor(
            reader,
            &format!("{prefix}.attn_q.weight"),
            device,
        )?)?;
        let k_proj = QMatMulWrapper::from_qtensor(ct.tensor(
            reader,
            &format!("{prefix}.attn_k.weight"),
            device,
        )?)?;
        let v_proj = QMatMulWrapper::from_qtensor(ct.tensor(
            reader,
            &format!("{prefix}.attn_v.weight"),
            device,
        )?)?;
        let o_proj = QMatMulWrapper::from_qtensor(ct.tensor(
            reader,
            &format!("{prefix}.attn_output.weight"),
            device,
        )?)?;

        let q_norm = RmsNorm::from_qtensor(
            ct.tensor(reader, &format!("{prefix}.attn_q_norm.weight"), device)?,
            rms_norm_eps,
        )?;
        let k_norm = RmsNorm::from_qtensor(
            ct.tensor(reader, &format!("{prefix}.attn_k_norm.weight"), device)?,
            rms_norm_eps,
        )?;

        let max_position_embeddings = ct
            .metadata
            .get("qwen3.context_length")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(4096) as usize;
        let kv_cache = KvCache::new(2, max_position_embeddings);

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
            rotary_emb,
            kv_cache,
            span_attn,
        })
    }

    pub(crate) fn forward(
        &mut self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b, l, _) = x.dims3()?;

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

        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;

        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.num_kv_heads, l, self.head_dim))?;

        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;

        // Reset KV cache if we're at the first position
        if offset == 0 {
            self.kv_cache.reset();
        }
        let (k, v) = self.kv_cache.append(&k.contiguous()?, &v.contiguous()?)?;

        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

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
        let ctx = probs.matmul(&v)?; // (B, H, L, D)
        let reshaped_ctx = ctx
            .transpose(1, 2)?
            .reshape((b, l, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&reshaped_ctx)
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
        ct: &gguf_file::Content,
        reader: &mut R,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary: Arc<RotaryEmbedding>,
        layer_idx: usize,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");

        let ln1 = RmsNorm::from_qtensor(
            ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?,
            rms_norm_eps,
        )?;
        let ln2 = RmsNorm::from_qtensor(
            ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?,
            rms_norm_eps,
        )?;

        let self_attn = AttentionWeights::new(
            ct,
            reader,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            rotary,
            &prefix,
            device,
        )?;
        let mlp = MlpWeights::new(ct, reader, &prefix, device)?;

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
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    embed_tokens: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    lm_head: QMatMulWrapper,
    device: Device,
    dtype: DType,
    span: tracing::Span,
    span_output: tracing::Span,
}

impl ModelWeights {
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let md_get = |s: &str| match ct.metadata.get(s) {
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

        let dtype = match ct.metadata.get("general.dtype") {
            Some(v) => match v.to_u32() {
                Ok(0) => DType::F32,
                Ok(1) => DType::F16,
                _ => DType::F16,
            },
            None => DType::F16,
        };

        let embed_tensor = ct.tensor(reader, "token_embd.weight", device)?;
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
                &ct,
                reader,
                num_attention_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
                rotary.clone(),
                i,
                device,
            )?);
        }

        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;

        // Load output projection tensor, falling back to tied embeddings like gemma3
        let lm_head_tensor = match ct.tensor(reader, "output.weight", device) {
            Ok(tensor) => tensor,
            Err(_) => ct.tensor(reader, "token_embd.weight", device)?,
        };
        let lm_head = QMatMulWrapper::from_qtensor(lm_head_tensor)?;

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

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;

        let causal = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset, None)?)
        };

        for layer in &mut self.layers {
            h = layer.forward(&h, causal.as_ref(), offset)?;
        }

        let h = self.norm.forward(&h)?;

        let _enter = self.span_output.enter();
        let last_hidden = h.narrow(1, l - 1, 1)?;

        self.lm_head.forward(&last_hidden)?.squeeze(1)
    }
}
