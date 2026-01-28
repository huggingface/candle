//! GLM4 implementation with quantization support.
//!
//! Based on the GLM4 architecture and implemented with quantized weights
//! for reduced memory usage and faster inference on compatible hardware.
//!
//! References:
//! - [GLM4-0414 Models](THUDM/GLM-4-9B-0414) (architecture based on official implementations)
//!
use super::with_tracing::QMatMul;
use crate::{quantized_nn::RmsNorm, utils::repeat_kv};
use candle::quantized::{gguf_file, QTensor};
use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{kv_cache::KvCache, Activation, Embedding, Module};
use std::io::{Read, Seek};
use std::sync::Arc;

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

    fn metadata(&self) -> &std::collections::HashMap<String, gguf_file::Value> {
        &self.ct.metadata
    }

    fn tensor(&mut self, name: &str) -> Result<QTensor> {
        self.ct.tensor(&mut self.reader, name, &self.device)
    }

    fn unquantized_tensor(&mut self, name: &str, dtype: DType) -> Option<Tensor> {
        let t = self.ct.tensor(&mut self.reader, name, &self.device);
        if let Ok(t) = &t {
            t.dequantize(&self.device).unwrap().to_dtype(dtype).ok()
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_up_proj: QMatMul,
    down_proj: QMatMul,
    act_fn: Activation,
}

impl Mlp {
    fn new<R: Read + Seek>(gg: &mut Gguf<R>, prefix: &str) -> Result<Self> {
        //ffn_gate and ffn_up combined into ffn_up
        let gate_up_proj = gg.qmatmul(&format!("{prefix}.ffn_up.weight"))?;
        let down_proj = gg.qmatmul(&format!("{prefix}.ffn_down.weight"))?;
        let act_fn = Activation::Silu;
        Ok(Self {
            gate_up_proj,
            down_proj,
            act_fn,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w = self.gate_up_proj.forward(xs)?;
        let dim = w.dims().len() - 1;
        let gate = w
            .narrow(dim, 0, w.dim(dim)? / 2)?
            .contiguous()?
            .apply(&self.act_fn)?;
        let up_states = w
            .narrow(dim, w.dim(dim)? / 2, w.dim(dim)? / 2)?
            .contiguous()?;
        self.down_proj.forward(&(gate * up_states)?)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    rotary_dim: usize,
}

impl RotaryEmbedding {
    pub(crate) fn new(
        dtype: DType,
        head_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        partial_rotary_factor: Option<f32>,
        dev: &Device,
    ) -> Result<Self> {
        let rotary_dim = if let Some(factor) = partial_rotary_factor {
            (factor * head_dim as f32) as usize
        } else {
            head_dim
        };
        let max_seq_len = max_position_embeddings;
        let inv_freq: Vec<_> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / rotary_dim as f64) as f32)
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

    pub(crate) fn apply(&self, xs: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, _, seq_len, _) = xs.dims4()?;
        let (s, e) = (offset, offset + seq_len);
        let cos = self.cos.i((s..e, ..))?.contiguous()?;
        let sin = self.sin.i((s..e, ..))?.contiguous()?;
        let xs_rot = xs
            .i((0, .., .., ..self.rotary_dim))?
            .unsqueeze(0)?
            .contiguous()?;
        let xs_pass = xs.i((0, .., .., self.rotary_dim..))?.unsqueeze(0)?;
        let xs_rot = candle_nn::rotary_emb::rope_i(&xs_rot, &cos, &sin).unwrap();
        Tensor::cat(&[&xs_rot, &xs_pass], D::Minus1)?.contiguous()
    }
}

#[derive(Debug, Clone)]
struct AttentionWeights {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    attention_bq: Option<Tensor>,
    attention_bk: Option<Tensor>,
    attention_bv: Option<Tensor>,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: KvCache,
    dtype: DType,
    span_attn: tracing::Span,
}

impl AttentionWeights {
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rotary_emb: Arc<RotaryEmbedding>,
        prefix: &str,
        dtype: DType,
    ) -> Result<Self> {
        let num_kv_groups = num_heads / num_kv_heads;

        let q_proj = gg.qmatmul(&format!("{prefix}.attn_q.weight"))?;
        let k_proj = gg.qmatmul(&format!("{prefix}.attn_k.weight"))?;
        let v_proj = gg.qmatmul(&format!("{prefix}.attn_v.weight"))?;
        let o_proj = gg.qmatmul(&format!("{prefix}.attn_output.weight"))?;

        let attention_bq = gg.unquantized_tensor(&format!("{prefix}.attn_q.bias"), DType::F32);
        let attention_bk = gg.unquantized_tensor(&format!("{prefix}.attn_k.bias"), DType::F32);
        let attention_bv = gg.unquantized_tensor(&format!("{prefix}.attn_v.bias"), DType::F32);

        // Initialize KV cache with 512 tokens capacity to reduce initial memory allocation.
        // The cache will grow in chunks of 512 tokens when needed.
        let kv_cache = KvCache::new(2, 512);

        let span_attn = tracing::span!(tracing::Level::TRACE, "attn");

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            attention_bq,
            attention_bk,
            attention_bv,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            rotary_emb,
            kv_cache,
            dtype,
            span_attn,
        })
    }

    fn forward(&mut self, x: &Tensor, attn_mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b, l, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        let q = if let Some(bq) = &self.attention_bq {
            q.broadcast_add(bq)?
        } else {
            q
        };

        let k = if let Some(bk) = &self.attention_bk {
            k.broadcast_add(bk)?
        } else {
            k
        };

        let v = if let Some(bv) = &self.attention_bv {
            v.broadcast_add(bv)?
        } else {
            v
        };

        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.rotary_emb.apply(&q, offset)?;
        let k = self.rotary_emb.apply(&k, offset)?;

        let (q, k, v) = (
            q.to_dtype(self.dtype)?,
            k.to_dtype(self.dtype)?,
            v.to_dtype(self.dtype)?,
        );
        // Reset KV cache if we're at the first position
        if offset == 0 {
            self.kv_cache.reset();
        }

        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let (k, v) = self.kv_cache.append(&k.contiguous()?, &v.contiguous()?)?;

        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(mask) = attn_mask {
            scores = scores.broadcast_add(mask)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?; // (B, H, L, D)
        let reshaped_ctx = ctx
            .transpose(1, 2)?
            .reshape((b, l, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&reshaped_ctx.to_dtype(x.dtype())?)
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    self_attn: AttentionWeights,
    mlp: Mlp,
    ffn_norm: RmsNorm,
    attn_norm: RmsNorm,
    post_ffw_norm: RmsNorm,
    post_attention_norm: RmsNorm,
}

impl LayerWeights {
    #[allow(clippy::too_many_arguments)]
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary: Arc<RotaryEmbedding>,
        layer_idx: usize,
        dtype: DType,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");

        let attn_norm = gg.rms_norm(&format!("{prefix}.attn_norm.weight"), rms_norm_eps)?;
        let ffn_norm = gg.rms_norm(&format!("{prefix}.ffn_norm.weight"), rms_norm_eps)?;

        let post_ffw_norm = gg.rms_norm(&format!("{prefix}.post_ffw_norm.weight"), rms_norm_eps)?;
        let post_attention_norm = gg.rms_norm(
            &format!("{prefix}.post_attention_norm.weight"),
            rms_norm_eps,
        )?;

        let self_attn = AttentionWeights::new(
            gg,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rotary,
            &prefix,
            dtype,
        )?;
        let mlp = Mlp::new(gg, &prefix)?;
        Ok(Self {
            self_attn,
            mlp,
            attn_norm,
            ffn_norm,
            post_ffw_norm,
            post_attention_norm,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let residual = x;
        let x = self.attn_norm.forward(x)?;
        let attn = self.self_attn.forward(&x, mask, offset)?;
        let attn = self.post_attention_norm.forward(&attn)?;
        let x = (attn + residual)?;

        // MLP
        let residual = &x;
        let x = self.ffn_norm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        let x = self.post_ffw_norm.forward(&x)?;
        x + residual
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
    span: tracing::Span,
    span_output: tracing::Span,
}

impl ModelWeights {
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let mut gg = Gguf::new(ct, reader, device.clone());
        let md_get = |s: &str| match gg.metadata().get(s) {
            None => candle::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        let num_attention_heads = md_get("glm4.attention.head_count")?.to_u32()? as usize;
        let num_kv_heads = md_get("glm4.attention.head_count_kv")?.to_u32()? as usize;
        let head_dim = md_get("glm4.attention.key_length")?.to_u32()? as usize;
        let num_layers = md_get("glm4.block_count")?.to_u32()? as usize;
        let hidden_size = md_get("glm4.embedding_length")?.to_u32()? as usize;
        let max_position_embeddings = md_get("glm4.context_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("glm4.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("glm4.rope.freq_base")?.to_f32()? as f64;

        let embed_tensor = gg.tensor("token_embd.weight")?;
        let embed_tokens = Embedding::new(embed_tensor.dequantize(device)?, hidden_size);

        let rotary = Arc::new(RotaryEmbedding::new(
            DType::F32,
            head_dim,
            max_position_embeddings,
            rope_freq_base,
            Some(0.5), //partial rotary factor not embedded in gguf
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
                dtype,
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

        let causal_mask = if l == 1 {
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
}
