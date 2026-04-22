use crate::models::with_tracing::QMatMul;
use crate::quantized_var_builder::VarBuilder;
use candle::quantized::gguf_file;
use candle::{DType, Device, Module, Result, Storage, Tensor};
use candle_nn::attention::cpu_flash::causal::causal_decode_f32_interleaved;
use candle_nn::attention::{flash_attn, AttnMask};
use candle_nn::kv_cache::{rope_i_inplace, InterleavedKvCache, KvCache, RawInterleavedKvCache};
use candle_nn::Activation;
use std::io::Write;
use std::sync::Arc;

const MAX_SEQ_LEN: usize = 4096;
use candle::IndexOp;

// ===== RECONSTRUCTION FUNCTION =====
fn reconstruct_qk_weights(gguf_weight: &Tensor, _num_heads: usize) -> Result<Tensor> {
    let total_rows = gguf_weight.dim(0)?;
    let half_rows = total_rows / 2;
    let chunk_size = 128;
    let chunks_per_half = half_rows / chunk_size;

    let mut heads = Vec::new();

    // First half
    for chunk_idx in 0..chunks_per_half {
        let chunk_start = chunk_idx * chunk_size;

        // Even rows
        let mut head_even = Vec::new();
        for i in (chunk_start..chunk_start + chunk_size).step_by(2) {
            head_even.push(gguf_weight.i(i)?);
        }
        heads.push(Tensor::stack(&head_even, 0)?);

        // Odd rows
        let mut head_odd = Vec::new();
        for i in (chunk_start + 1..chunk_start + chunk_size).step_by(2) {
            head_odd.push(gguf_weight.i(i)?);
        }
        heads.push(Tensor::stack(&head_odd, 0)?);
    }

    // Second half
    for chunk_idx in 0..chunks_per_half {
        let chunk_start = half_rows + chunk_idx * chunk_size;

        // Even rows
        let mut head_even = Vec::new();
        for i in (chunk_start..chunk_start + chunk_size).step_by(2) {
            head_even.push(gguf_weight.i(i)?);
        }
        heads.push(Tensor::stack(&head_even, 0)?);

        // Odd rows
        let mut head_odd = Vec::new();
        for i in (chunk_start + 1..chunk_start + chunk_size).step_by(2) {
            head_odd.push(gguf_weight.i(i)?);
        }
        heads.push(Tensor::stack(&head_odd, 0)?);
    }

    Tensor::cat(&heads, 0)
}

#[derive(Debug, Clone)]
pub struct QuantizedConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub rope_dimension_count: usize,
    pub no_rope_layer_interval: Option<usize>,
}

impl QuantizedConfig {
    /// Load config from GGUF metadata
    pub fn from_gguf(ct: &gguf_file::Content) -> Result<Self> {
        let metadata = &ct.metadata;

        // Helper to get required metadata
        let get_u32 = |key: &str| -> Result<usize> {
            metadata
                .get(key)
                .and_then(|v| v.to_u32().ok())
                .map(|v| v as usize)
                .ok_or_else(|| {
                    candle::Error::Msg(format!("Missing or invalid metadata key: {}", key))
                })
        };

        let get_f32 = |key: &str| -> Result<f64> {
            metadata
                .get(key)
                .and_then(|v| v.to_f32().ok())
                .map(|v| v as f64)
                .ok_or_else(|| {
                    candle::Error::Msg(format!("Missing or invalid metadata key: {}", key))
                })
        };

        Ok(Self {
            vocab_size: get_u32("smollm3.vocab_size")?,
            hidden_size: get_u32("smollm3.embedding_length")?,
            intermediate_size: get_u32("smollm3.feed_forward_length")?,
            num_hidden_layers: get_u32("smollm3.block_count")?,
            num_attention_heads: get_u32("smollm3.attention.head_count")?,
            num_key_value_heads: get_u32("smollm3.attention.head_count_kv")?,
            max_position_embeddings: get_u32("smollm3.context_length").unwrap_or(MAX_SEQ_LEN),
            rope_theta: get_f32("smollm3.rope.freq_base")?,
            rms_norm_eps: get_f32("smollm3.attention.layer_norm_rms_epsilon")?,
            rope_dimension_count: get_u32("smollm3.rope.dimension_count")?,
            no_rope_layer_interval: Some(4),
        })
    }

    pub fn should_skip_rope(&self, layer_idx: usize) -> bool {
        if let Some(interval) = self.no_rope_layer_interval {
            return (layer_idx + 1).is_multiple_of(interval);
        }
        false
    }

    pub fn head_dim(&self) -> usize {
        self.rope_dimension_count
    }
}

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(candle::D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let norm_x = (x.sqr()?.sum_keepdim(candle::D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        let result = x_normed.broadcast_mul(&self.weight)?;
        result.to_dtype(x_dtype)
    }
}

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    /// Pre-extracted flat f32 arrays for fused in-place RoPE (no tensor ops at decode time)
    cos_f32: Vec<f32>,
    sin_f32: Vec<f32>,
    half_d: usize,
    /// When true, use interleaved RoPE (pairs adjacent elements 2i,2i+1)
    use_interleaved: bool,
}

impl RotaryEmbedding {
    pub fn new(
        dtype: DType,
        cfg: &QuantizedConfig,
        dev: &Device,
        use_interleaved: bool,
    ) -> Result<Self> {
        let dim = cfg.head_dim();
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let sin_t = freqs.sin()?;
        let cos_t = freqs.cos()?;
        // Flat f32 RoPE tables used by the fused decode path.
        let cos_f32 = cos_t
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let sin_f32 = sin_t
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let half_d = dim / 2;
        Ok(Self {
            sin: sin_t.to_dtype(dtype)?,
            cos: cos_t.to_dtype(dtype)?,
            cos_f32,
            sin_f32,
            half_d,
            use_interleaved,
        })
    }

    pub fn apply_rotary_emb(
        &self,
        q: &Tensor,
        k: &Tensor,
        offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let (q_embed, k_embed) = if self.use_interleaved {
            // Interleaved RoPE: pairs adjacent elements (2i, 2i+1)
            // Correct for interlaced GGUF weight layout
            let q_embed = candle_nn::rotary_emb::rope_i(&q.contiguous()?, &cos, &sin)?;
            let k_embed = candle_nn::rotary_emb::rope_i(&k.contiguous()?, &cos, &sin)?;
            (q_embed, k_embed)
        } else {
            let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
            let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
            (q_embed, k_embed)
        };
        Ok((q_embed, k_embed))
    }

    /// Get raw f32 cos/sin slices for a single position. Zero allocation.
    #[inline]
    pub fn cos_sin_at(&self, pos: usize) -> (&[f32], &[f32]) {
        let start = pos * self.half_d;
        let end = start + self.half_d;
        (&self.cos_f32[start..end], &self.sin_f32[start..end])
    }
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(x)
    } else {
        let (b, n_kv_heads, seq_len, head_dim) = x.dims4()?;
        x.unsqueeze(2)?
            .expand(&[b, n_kv_heads, n_rep, seq_len, head_dim])?
            .reshape(&[b, n_kv_heads * n_rep, seq_len, head_dim])
    }
}

#[derive(Debug, Clone)]
struct QuantizedMLP {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
}

impl QuantizedMLP {
    fn new(vb: VarBuilder, _layer_idx: usize) -> Result<Self> {
        // VarBuilder.get_no_shape() returns Arc<QTensor> which QMatMul::from_weights expects
        let gate_proj = QMatMul::from_weights(vb.get_no_shape("ffn_gate.weight")?)?;
        let up_proj = QMatMul::from_weights(vb.get_no_shape("ffn_up.weight")?)?;
        let down_proj = QMatMul::from_weights(vb.get_no_shape("ffn_down.weight")?)?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.apply(&Activation::Silu)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

#[derive(Debug, Clone)]
struct QuantizedAttention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Option<Arc<RotaryEmbedding>>,
    skip_rope: bool,
    use_flash_attn: bool,
    kv_cache: Option<KvCache>,
    interleaved_cache: Option<InterleavedKvCache>,
    raw_cache: Option<RawInterleavedKvCache>,
    /// Pre-allocated buffers for in-place RoPE during fused decode
    q_rope_buf: Vec<f32>,
    k_rope_buf: Vec<f32>,
}

impl QuantizedAttention {
    fn new(
        vb: VarBuilder,
        cfg: &QuantizedConfig,
        layer_idx: usize,
        rotary_emb: Option<Arc<RotaryEmbedding>>,
        use_flash_attn: bool,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;

        let v_proj = QMatMul::from_weights(vb.get_no_shape("attn_v.weight")?)?;

        let device = vb.device();
        let cpu = Device::Cpu;
        let o_proj = QMatMul::from_weights(vb.get_no_shape("attn_output.weight")?)?;

        let (q_proj, k_proj) = if use_flash_attn {
            // Interlaced path: keep Q/K in native GGUF order. The interleaved RoPE
            // handles pairing; dot products are order-independent so QK scores match.
            // V is not interlaced, so attention output stays in standard order.
            let q_proj = QMatMul::from_weights(vb.get_no_shape("attn_q.weight")?)?;
            let k_proj = QMatMul::from_weights(vb.get_no_shape("attn_k.weight")?)?;
            (q_proj, k_proj)
        } else {
            // Standard path: dequantize, reconstruct, requantize.
            use candle::quantized::{GgmlDType, QTensor};
            let q_weight_qtensor = vb.get_no_shape("attn_q.weight")?;
            let q_weight_raw = q_weight_qtensor.dequantize(&cpu)?;
            let q_weight = reconstruct_qk_weights(&q_weight_raw, num_heads)?;
            let q_weight = q_weight.to_device(device)?;
            let q_weight_qtensor = QTensor::quantize(&q_weight, GgmlDType::Q8_0)?;
            drop(q_weight_raw);
            drop(q_weight);

            let k_weight_qtensor = vb.get_no_shape("attn_k.weight")?;
            let k_weight_raw = k_weight_qtensor.dequantize(&cpu)?;
            let k_weight = reconstruct_qk_weights(&k_weight_raw, num_kv_heads)?;
            let k_weight = k_weight.to_device(device)?;
            let k_weight_qtensor = QTensor::quantize(&k_weight, GgmlDType::Q8_0)?;
            drop(k_weight_raw);
            drop(k_weight);

            let q_proj = QMatMul::from_weights(Arc::new(q_weight_qtensor))?;
            let k_proj = QMatMul::from_weights(Arc::new(k_weight_qtensor))?;
            (q_proj, k_proj)
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            hidden_size: num_heads * head_dim,
            rotary_emb,
            skip_rope: cfg.should_skip_rope(layer_idx),
            use_flash_attn,
            kv_cache: if use_flash_attn {
                None
            } else {
                Some(KvCache::new(2, 512))
            },
            interleaved_cache: if use_flash_attn {
                Some(InterleavedKvCache::new(head_dim))
            } else {
                None
            },
            raw_cache: if use_flash_attn {
                Some(RawInterleavedKvCache::new(
                    num_kv_heads,
                    head_dim,
                    MAX_SEQ_LEN,
                ))
            } else {
                None
            },
            q_rope_buf: vec![0f32; num_heads * head_dim],
            k_rope_buf: vec![0f32; num_kv_heads * head_dim],
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;

        // Fused decode: raw f32, no tensor ops in hot path.
        if self.use_flash_attn
            && x.device().is_cpu()
            && seq_len == 1
            && b == 1
            && x.dtype() == DType::F32
        {
            // 1. QKV projections (raw f32 output slices)
            let q_proj_out = self.q_proj.forward(x)?; // (1, 1, H_q * D)
            let k_proj_out = self.k_proj.forward(x)?; // (1, 1, H_kv * D)
            let v_proj_out = self.v_proj.forward(x)?; // (1, 1, H_kv * D)

            // Extract flat f32 slices
            let (q_g, q_l) = q_proj_out.storage_and_layout();
            let q_flat: &[f32] = match &*q_g {
                Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[q_l.start_offset()..],
                _ => candle::bail!("Expected CPU storage"),
            };
            let (k_g, k_l) = k_proj_out.storage_and_layout();
            let k_flat: &[f32] = match &*k_g {
                Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[k_l.start_offset()..],
                _ => candle::bail!("Expected CPU storage"),
            };
            let (v_g, v_l) = v_proj_out.storage_and_layout();
            let v_flat: &[f32] = match &*v_g {
                Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[v_l.start_offset()..],
                _ => candle::bail!("Expected CPU storage"),
            };

            // 2. Copy Q and K into pre-allocated buffers for in-place RoPE (no allocation)
            let q_len = self.num_heads * self.head_dim;
            let k_len = self.num_kv_heads * self.head_dim;
            self.q_rope_buf[..q_len].copy_from_slice(&q_flat[..q_len]);
            self.k_rope_buf[..k_len].copy_from_slice(&k_flat[..k_len]);

            // 3. Apply RoPE in-place (no tensor ops, no allocation)
            if !self.skip_rope {
                if let Some(rope) = &self.rotary_emb {
                    let (cos, sin) = rope.cos_sin_at(offset);
                    rope_i_inplace(
                        &mut self.q_rope_buf[..q_len],
                        cos,
                        sin,
                        self.num_heads,
                        self.head_dim,
                    );
                    rope_i_inplace(
                        &mut self.k_rope_buf[..k_len],
                        cos,
                        sin,
                        self.num_kv_heads,
                        self.head_dim,
                    );
                }
            }

            // 4. Write K, V directly into raw cache (no tensor allocation)
            let rc = self.raw_cache.as_mut().unwrap();
            rc.write_kv(&self.k_rope_buf[..k_len], &v_flat[..k_len]);

            // 5. Run interleaved decode kernel
            let scale = 1.0 / (self.head_dim as f32).sqrt();
            let kv_len = rc.len();
            let ctx = causal_decode_f32_interleaved(
                &self.q_rope_buf[..q_len],
                rc.data(),
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                kv_len,
                scale,
            )?;

            // 6. Output projection
            let ctx = ctx.unsqueeze(0)?.transpose(1, 2)?;
            return ctx
                .reshape((b, seq_len, self.hidden_size))?
                .apply(&self.o_proj);
        }

        // Standard path: prefill, non-f32, or non-flash.
        let q = self
            .q_proj
            .forward(x)?
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = if self.skip_rope {
            (q, k)
        } else if let Some(rope) = &self.rotary_emb {
            rope.apply_rotary_emb(&q, &k, offset)?
        } else {
            (q, k)
        };

        if self.use_flash_attn && x.device().is_cpu() && b == 1 {
            // Prefill (B=1 only): use InterleavedKvCache + flash_attn
            let kv = self.interleaved_cache.as_mut().unwrap().append(&k, &v)?;
            // Also populate raw cache for subsequent decode steps
            {
                let k_cont = k.squeeze(0)?.transpose(0, 1)?.contiguous()?;
                let v_cont = v.squeeze(0)?.transpose(0, 1)?.contiguous()?;
                let (kg, kl) = k_cont.storage_and_layout();
                let k_data: &[f32] = match &*kg {
                    Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[kl.start_offset()..],
                    _ => candle::bail!("Expected CPU"),
                };
                let (vg, vl) = v_cont.storage_and_layout();
                let v_data: &[f32] = match &*vg {
                    Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[vl.start_offset()..],
                    _ => candle::bail!("Expected CPU"),
                };
                self.raw_cache
                    .as_mut()
                    .unwrap()
                    .write_kv_batch(k_data, v_data, seq_len);
            }

            let scale = 1.0 / (self.head_dim as f32).sqrt();
            let kv_k = kv.narrow(2, 0, self.head_dim)?.unsqueeze(0)?;
            let kv_v = kv.narrow(2, self.head_dim, self.head_dim)?.unsqueeze(0)?;

            let q = q.transpose(1, 2)?.contiguous()?;
            let k = kv_k.contiguous()?;
            let v = kv_v.contiguous()?;

            let ctx = flash_attn::<f32>(
                &q,
                &k,
                &v,
                scale,
                AttnMask::causal_with_offset(offset),
                None,
                None,
            )?;
            let ctx = ctx.transpose(1, 2)?;
            ctx.reshape((b, seq_len, self.hidden_size))?
                .apply(&self.o_proj)
        } else {
            // Standard matmul attention (no flash)
            let (k, v) = self
                .kv_cache
                .as_mut()
                .unwrap()
                .append(&k.contiguous()?, &v.contiguous()?)?;

            let k = repeat_kv(k, self.num_kv_groups)?;
            let v = repeat_kv(v, self.num_kv_groups)?;

            let scale = 1.0 / (self.head_dim as f64).sqrt();
            let q = q.contiguous()?;
            let attn_weights = (q.matmul(&k.t()?)? * scale)?;

            let attn_weights = match mask {
                Some(mask) => attn_weights.broadcast_add(mask)?,
                None => attn_weights,
            };

            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            let attn_output = attn_weights.matmul(&v)?;

            attn_output
                .transpose(1, 2)?
                .reshape((b, seq_len, self.hidden_size))?
                .apply(&self.o_proj)
        }
    }

    fn clear_kv_cache(&mut self) {
        if let Some(c) = &mut self.kv_cache {
            c.reset();
        }
        if let Some(c) = &mut self.interleaved_cache {
            c.reset();
        }
        if let Some(c) = &mut self.raw_cache {
            c.reset();
        }
    }
}

#[derive(Debug, Clone)]
struct QuantizedDecoderLayer {
    self_attn: QuantizedAttention,
    mlp: QuantizedMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl QuantizedDecoderLayer {
    fn new(
        vb: VarBuilder,
        cfg: &QuantizedConfig,
        layer_idx: usize,
        rotary_emb: Option<Arc<RotaryEmbedding>>,
        use_flash_attn: bool,
    ) -> Result<Self> {
        let attn_vb = vb.pp(format!("blk.{layer_idx}"));

        Ok(Self {
            self_attn: QuantizedAttention::new(
                attn_vb.clone(),
                cfg,
                layer_idx,
                rotary_emb,
                use_flash_attn,
            )?,
            mlp: QuantizedMLP::new(attn_vb.clone(), layer_idx)?,
            input_layernorm: RmsNorm::new(
                attn_vb
                    .get_no_shape("attn_norm.weight")?
                    .dequantize(vb.device())?,
                cfg.rms_norm_eps,
            ),
            post_attention_layernorm: RmsNorm::new(
                attn_vb
                    .get_no_shape("ffn_norm.weight")?
                    .dequantize(vb.device())?,
                cfg.rms_norm_eps,
            ),
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(&x, mask, offset)?;
        let x = (residual + x)?;

        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        residual + x
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
pub struct QuantizedModelForCausalLM {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<QuantizedDecoderLayer>,
    norm: RmsNorm,
    lm_head: QMatMul,
    device: Device,
    use_flash_attn: bool,
    config: QuantizedConfig,
}

impl QuantizedModelForCausalLM {
    pub fn from_gguf<P: AsRef<std::path::Path>>(
        path: P,
        device: &Device,
        use_flash_attn: bool,
    ) -> Result<Self> {
        use candle::quantized::{GgmlDType, QTensor};

        // Open file once to read metadata
        let mut file = std::fs::File::open(path.as_ref())?;
        let content = gguf_file::Content::read(&mut file)?;
        let config = QuantizedConfig::from_gguf(&content)?;

        // Create VarBuilder for tensor loading
        let vb = VarBuilder::from_gguf(path, device)?;

        // Load embedding tensor - dequantize on CPU first to save VRAM
        // (will be used for both embed_tokens and lm_head - tied embeddings)
        let cpu = Device::Cpu;
        let embed_tensor = vb.get_no_shape("token_embd.weight")?.dequantize(&cpu)?;
        let embed_tensor_gpu = embed_tensor.to_device(device)?; // Move to GPU for embedding layer
        let embed_tokens = candle_nn::Embedding::new(embed_tensor_gpu, config.hidden_size);

        // Create rotary embedding if needed
        let needs_rope = (0..config.num_hidden_layers).any(|i| !config.should_skip_rope(i));
        let rotary_emb = if needs_rope {
            Some(Arc::new(RotaryEmbedding::new(
                DType::F32,
                &config,
                device,
                use_flash_attn,
            )?))
        } else {
            None
        };

        // Load decoder layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        println!("Loading {} decoder layers...", config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            if layer_idx % 4 == 0 || layer_idx == config.num_hidden_layers - 1 {
                print!(
                    "  Layer {}/{}...\r",
                    layer_idx + 1,
                    config.num_hidden_layers
                );
                std::io::stdout().flush().ok();
            }
            layers.push(QuantizedDecoderLayer::new(
                vb.clone(),
                &config,
                layer_idx,
                rotary_emb.clone(),
                use_flash_attn,
            )?);
        }
        println!(
            "  Layer {}/{} - Done!    ",
            config.num_hidden_layers, config.num_hidden_layers
        );

        // Load output norm
        let norm = RmsNorm::new(
            vb.get_no_shape("output_norm.weight")?.dequantize(device)?,
            config.rms_norm_eps,
        );

        // Load LM head - move CPU embedding tensor to GPU, then quantize
        let embed_tensor_for_lm = embed_tensor.to_device(device)?;
        let embed_qtensor = QTensor::quantize(&embed_tensor_for_lm, GgmlDType::Q8_0)?;
        let lm_head = QMatMul::from_weights(Arc::new(embed_qtensor))?;
        drop(embed_tensor); // Free CPU memory
        drop(embed_tensor_for_lm);

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            use_flash_attn,
            config,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, offset: usize) -> Result<Tensor> {
        let (batch_size, seq_len) = input_ids.dims2()?;

        // Embed tokens
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        // Skip mask materialization when using CPU flash attention
        let mask = if seq_len > 1 && !(self.use_flash_attn && self.device.is_cpu()) {
            Some(self.create_causal_mask(batch_size, seq_len, offset)?)
        } else {
            None
        };

        // Forward through decoder layers
        for layer in &mut self.layers {
            hidden_states = layer.forward(&hidden_states, mask.as_ref(), offset)?;
        }

        // Final norm
        hidden_states = self.norm.forward(&hidden_states)?;

        // LM head (only last token for generation)
        let last_hidden = hidden_states.narrow(1, seq_len - 1, 1)?;
        let logits = last_hidden.apply(&self.lm_head)?;

        Ok(logits)
    }

    fn create_causal_mask(
        &self,
        batch_size: usize,
        tgt_len: usize,
        offset: usize,
    ) -> Result<Tensor> {
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len + offset).map(move |j| {
                    if j <= i + offset {
                        0f32
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();

        Tensor::from_slice(
            &mask,
            (batch_size, 1, tgt_len, tgt_len + offset),
            &self.device,
        )
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }

    pub fn config(&self) -> &QuantizedConfig {
        &self.config
    }
}
