//! Qwen3.5 implementation with quantization support.
//!
//! Qwen3.5 is a hybrid architecture combining Gated DeltaNet (Linear Attention)
//! and Gated Attention (Full Softmax Attention).
//!
//! Based on the Qwen 3.5 architecture and implemented with quantized weights
//! for reduced memory usage and faster inference.
//!
use super::with_tracing::QMatMul;
use crate::{quantized_nn::RmsNorm, utils::repeat_kv};
use candle::quantized::{gguf_file, QTensor};
use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{kv_cache::ConcatKvCache, Activation, Embedding};
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

    pub fn qmatmul(&mut self, name: &str) -> Result<(QMatMul, usize)> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        let out_dim = ws.shape().dims()[0];
        Ok((QMatMul::from_weights(ws.into())?, out_dim))
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
}

impl MlpWeights {
    fn new<R: Read + Seek>(gg: &mut Gguf<R>, prefix: &str) -> Result<Self> {
        let (gate_proj, _) = gg.qmatmul(&format!("{prefix}.ffn_gate.weight"))?;
        let (up_proj, _) = gg.qmatmul(&format!("{prefix}.ffn_up.weight"))?;
        let (down_proj, _) = gg.qmatmul(&format!("{prefix}.ffn_down.weight"))?;
        let act_fn = Activation::Silu;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn,
        })
    }
}

impl Module for MlpWeights {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
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
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let sin = self.sin.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
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
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: ConcatKvCache,
}

impl AttentionWeights {
    #[allow(clippy::too_many_arguments)]
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        _num_heads: usize,
        _num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary_emb: Arc<RotaryEmbedding>,
        prefix: &str,
    ) -> Result<Self> {
        let (q_proj, q_out) = gg.qmatmul(&format!("{prefix}.attn_q.weight"))?;
        let (k_proj, k_out) = gg.qmatmul(&format!("{prefix}.attn_k.weight"))?;
        let (v_proj, _) = gg.qmatmul(&format!("{prefix}.attn_v.weight"))?;
        let (o_proj, _) = gg.qmatmul(&format!("{prefix}.attn_output.weight"))?;

        let num_heads = q_out / (head_dim * 2); // Q + Gate
        let num_kv_heads = k_out / head_dim;
        let num_kv_groups = num_heads / num_kv_heads;

        let q_norm = gg.rms_norm(&format!("{prefix}.attn_q_norm.weight"), rms_norm_eps)?;
        let k_norm = gg.rms_norm(&format!("{prefix}.attn_k_norm.weight"), rms_norm_eps)?;

        let kv_cache = ConcatKvCache::new(2);

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
        })
    }

    fn forward(&mut self, x: &Tensor, attn_mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        let q_out = self.q_proj.forward(x)?;
        let q_out = q_out.reshape((b, l, self.num_heads, self.head_dim * 2))?;
        
        let q = q_out.narrow(D::Minus1, 0, self.head_dim)?.transpose(1, 2)?;
        let gate = q_out.narrow(D::Minus1, self.head_dim, self.head_dim)?;
        
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

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

        let (k, v) = self.kv_cache.append(&k, &v)?;

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
        let ctx = probs.matmul(&v)?;
        
        let ctx = ctx.transpose(1, 2)?;
        // Apply sigmoid gate
        let gate_sigmoid = candle_nn::ops::sigmoid(&gate.to_dtype(DType::F32)?)?.to_dtype(ctx.dtype())?;
        let ctx = ctx.broadcast_mul(&gate_sigmoid)?;
        
        let reshaped_ctx = ctx.reshape((b, l, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&reshaped_ctx)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

#[derive(Debug, Clone)]
struct GatedDeltaNetWeights {
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_dim: usize,
    conv_kernel_size: usize,
    
    in_proj_qkv: QMatMul,
    in_proj_z: QMatMul,
    in_proj_b: QMatMul,
    in_proj_a: QMatMul,
    out_proj: QMatMul,
    
    conv1d_weight: Tensor,
    dt_bias_f32: Tensor,
    neg_a_f32: Tensor,
    norm_weight: Tensor,
    norm_eps: f64,
    
    conv_state: Option<Tensor>,
    recurrent_state: Option<Tensor>,
}

impl GatedDeltaNetWeights {
    #[allow(clippy::too_many_arguments)]
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        _hidden_size: usize,
        num_v_heads: usize,
        num_k_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        conv_kernel_size: usize,
        rms_norm_eps: f64,
        prefix: &str,
    ) -> Result<Self> {
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_dim = key_dim * 2 + value_dim;

        let (in_proj_qkv, _) = gg.qmatmul(&format!("{prefix}.attn_qkv.weight"))?;
        let (in_proj_z, _) = gg.qmatmul(&format!("{prefix}.attn_gate.weight"))?;
        let (in_proj_b, _) = gg.qmatmul(&format!("{prefix}.ssm_beta.weight"))?;
        let (in_proj_a, _) = gg.qmatmul(&format!("{prefix}.ssm_alpha.weight"))?;
        let (out_proj, _) = gg.qmatmul(&format!("{prefix}.ssm_out.weight"))?;

        let conv1d_weight = gg.tensor(&format!("{prefix}.ssm_conv1d.weight"))?
            .dequantize(&gg.device)?
            .unsqueeze(1)?;
        
        let a_log = gg.tensor(&format!("{prefix}.ssm_a"))?.dequantize(&gg.device)?;
        let dt_bias = gg.tensor(&format!("{prefix}.ssm_dt.bias"))?.dequantize(&gg.device)?;
        
        let dt_bias_f32 = dt_bias.to_dtype(DType::F32)?;
        let neg_a_f32 = (&a_log.to_dtype(DType::F32)?.exp()? * -1.0)?;
        
        let norm_weight = gg.tensor(&format!("{prefix}.ssm_norm.weight"))?.dequantize(&gg.device)?;

        Ok(Self {
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_dim,
            conv_kernel_size,
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            out_proj,
            conv1d_weight,
            dt_bias_f32,
            neg_a_f32,
            norm_weight,
            norm_eps: rms_norm_eps,
            conv_state: None,
            recurrent_state: None,
        })
    }

    fn l2_norm(xs: &Tensor) -> Result<Tensor> {
        let eps = 1e-6;
        let norm = (xs.sqr()?.sum_keepdim(D::Minus1)? + eps)?.sqrt()?;
        xs.broadcast_div(&norm)
    }

    fn rms_norm_gated(&self, x: &Tensor, g: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let g_f32 = g.to_dtype(DType::F32)?;
        
        let gate = g_f32.silu()?;
        let norm_x = (x_f32.sqr()?.mean_keepdim(D::Minus1)? + self.norm_eps)?.sqrt()?;
        let x_normed = x_f32.broadcast_div(&norm_x)?;
        (x_normed * gate)?.broadcast_mul(&self.norm_weight.to_dtype(DType::F32)?)?.to_dtype(x_dtype)
    }

    fn torch_recurrent_gated_delta_rule(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        initial_state: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let query = Self::l2_norm(query)?;
        let key = Self::l2_norm(key)?;
        
        let query = query.transpose(1, 2)?;
        let key = key.transpose(1, 2)?;
        let value = value.transpose(1, 2)?;
        let beta = beta.transpose(1, 2)?;
        let g = g.transpose(1, 2)?;
        
        let (batch_size, num_heads, seq_len, k_head_dim) = key.dims4()?;
        let v_head_dim = value.dim(3)?;
        
        let scale = 1.0 / (query.dim(3)? as f64).sqrt();
        let query = (query * scale)?;
        
        let mut core_attn_out_vec = Vec::with_capacity(seq_len);
        
        let mut last_recurrent_state = match initial_state {
            Some(state) => state.to_dtype(DType::F32)?,
            None => Tensor::zeros((batch_size, num_heads, k_head_dim, v_head_dim), DType::F32, query.device())?,
        };
        
        let query_f32 = query.to_dtype(DType::F32)?;
        let key_f32 = key.to_dtype(DType::F32)?;
        let value_f32 = value.to_dtype(DType::F32)?;
        let g_exp = g.to_dtype(DType::F32)?.exp()?.unsqueeze(3)?;
        let beta_f32 = beta.to_dtype(DType::F32)?.unsqueeze(3)?;

        for i in 0..seq_len {
            let q_t = query_f32.narrow(2, i, 1)?;
            let k_t = key_f32.narrow(2, i, 1)?;
            let v_t = value_f32.narrow(2, i, 1)?;
            let g_t = g_exp.narrow(2, i, 1)?;
            let beta_t = beta_f32.narrow(2, i, 1)?;

            last_recurrent_state = last_recurrent_state.broadcast_mul(&g_t)?;
            
            let kv_mem = k_t.matmul(&last_recurrent_state)?;
            let delta = (&v_t - &kv_mem)?.broadcast_mul(&beta_t)?;
            
            let delta_k = k_t.transpose(2, 3)?.matmul(&delta)?;
            last_recurrent_state = (&last_recurrent_state + &delta_k)?;
            
            let out_t = q_t.matmul(&last_recurrent_state)?;
            core_attn_out_vec.push(out_t);
        }
        
        let core_attn_out = Tensor::cat(&core_attn_out_vec, 2)?;
        let core_attn_out = core_attn_out.transpose(1, 2)?;
        Ok((core_attn_out, last_recurrent_state))
    }

    fn forward(&mut self, hidden_states: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        let initial_dtype = hidden_states.dtype();
        
        let mixed_qkv = self.in_proj_qkv.forward(hidden_states)?.transpose(1, 2)?; 
        
        let z = self.in_proj_z.forward(hidden_states)?;
        let z = z.reshape((batch_size, seq_len, (), self.head_v_dim))?;
        
        let b = self.in_proj_b.forward(hidden_states)?;
        let a = self.in_proj_a.forward(hidden_states)?;
        
        let use_precomputed_states = self.conv_state.is_some() && seq_len == 1;
        
        let mixed_qkv = if use_precomputed_states {
            let conv_state = self.conv_state.as_mut().unwrap();
            let conv_state_data = Tensor::cat(&[conv_state.as_ref(), &mixed_qkv], 2)?;
            *conv_state = conv_state_data.narrow(2, 1, self.conv_kernel_size - 1)?;
            let out = conv_state_data.conv1d(&self.conv1d_weight, 0, 1, 1, self.conv_dim)?;
            candle_nn::ops::silu(&out)?
        } else {
            let pad = self.conv_kernel_size - 1;
            let padding = Tensor::zeros((batch_size, self.conv_dim, pad), mixed_qkv.dtype(), mixed_qkv.device())?;
            let padded_qkv = Tensor::cat(&[&padding, &mixed_qkv], 2)?;
            self.conv_state = Some(padded_qkv.narrow(2, seq_len, pad)?);
            let out = padded_qkv.conv1d(&self.conv1d_weight, 0, 1, 1, self.conv_dim)?;
            candle_nn::ops::silu(&out)?
        };
        
        let mixed_qkv = mixed_qkv.transpose(1, 2)?;
        
        let q = mixed_qkv.narrow(D::Minus1, 0, self.key_dim)?;
        let k = mixed_qkv.narrow(D::Minus1, self.key_dim, self.key_dim)?;
        let v = mixed_qkv.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?;
        
        let q = q.reshape((batch_size, seq_len, (), self.head_k_dim))?;
        let k = k.reshape((batch_size, seq_len, (), self.head_k_dim))?;
        let v = v.reshape((batch_size, seq_len, (), self.head_v_dim))?;
        
        let beta = candle_nn::ops::sigmoid(&b)?;
        let g = {
            let a_f32 = a.to_dtype(DType::F32)?;
            let a_plus_dt = a_f32.broadcast_add(&self.dt_bias_f32)?;
            let softplus = (a_plus_dt.exp()? + 1.0)?.log()?;
            self.neg_a_f32.broadcast_mul(&softplus)?
        };
        
        let repeat_n = self.num_v_heads / self.num_k_heads;
        let q = repeat_interleave(&q, repeat_n, 2)?;
        let k = repeat_interleave(&k, repeat_n, 2)?;
        
        let initial_state = if use_precomputed_states {
            self.recurrent_state.as_ref()
        } else {
            None
        };
        
        let (core_attn_out, new_state) = self.torch_recurrent_gated_delta_rule(&q, &k, &v, &g, &beta, initial_state)?;
        self.recurrent_state = Some(new_state);
        
        let core_attn_out = core_attn_out.to_dtype(initial_dtype)?;
        let core_attn_out = core_attn_out.reshape((batch_size, seq_len, self.num_v_heads, self.head_v_dim))?;
        let core_attn_out = core_attn_out.reshape(((), self.head_v_dim))?;
        let z_flat = z.reshape(((), self.head_v_dim))?;
        let core_attn_out = self.rms_norm_gated(&core_attn_out, &z_flat)?;
        let core_attn_out = core_attn_out.reshape((batch_size, seq_len, self.num_v_heads * self.head_v_dim))?;
        
        self.out_proj.forward(&core_attn_out)
    }

    fn clear_kv_cache(&mut self) {
        self.conv_state = None;
        self.recurrent_state = None;
    }
}

fn repeat_interleave(img: &Tensor, repeats: usize, dim: usize) -> Result<Tensor> {
    if repeats == 1 {
        return Ok(img.clone());
    }
    let mut dims = img.dims().to_vec();
    dims[dim] *= repeats;
    let final_dims = dims.clone();
    let img = img.unsqueeze(dim + 1)?;
    let mut expand_dims = img.dims().to_vec();
    expand_dims[dim + 1] = repeats;
    let expanded = img.expand(expand_dims.as_slice())?;
    expanded.reshape(final_dims.as_slice())
}

#[derive(Debug, Clone)]
enum TokenMixer {
    FullAttention(AttentionWeights),
    LinearAttention(GatedDeltaNetWeights),
}

#[derive(Debug, Clone)]
struct LayerWeights {
    token_mixer: TokenMixer,
    mlp: MlpWeights,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl LayerWeights {
    #[allow(clippy::too_many_arguments)]
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        layer_idx: usize,
        num_attention_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary: Arc<RotaryEmbedding>,
        full_attention_interval: usize,
        // Linear attention specifics
        linear_num_key_heads: usize,
        linear_num_value_heads: usize,
        linear_key_head_dim: usize,
        linear_value_head_dim: usize,
        linear_conv_kernel_dim: usize,
        hidden_size: usize,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");
        let is_full_attention = (layer_idx + 1) % full_attention_interval == 0;

        let token_mixer = if is_full_attention {
            TokenMixer::FullAttention(AttentionWeights::new(
                gg,
                num_attention_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
                rotary,
                &prefix,
            )?)
        } else {
            TokenMixer::LinearAttention(GatedDeltaNetWeights::new(
                gg,
                hidden_size,
                linear_num_value_heads,
                linear_num_key_heads,
                linear_key_head_dim,
                linear_value_head_dim,
                linear_conv_kernel_dim,
                rms_norm_eps,
                &prefix,
            )?)
        };

        let mlp = MlpWeights::new(gg, &prefix)?;
        let input_layernorm = gg.rms_norm(&format!("{prefix}.attn_norm.weight"), rms_norm_eps)?;
        let post_attention_layernorm = gg.rms_norm(&format!("{prefix}.post_attention_norm.weight"), rms_norm_eps)?;

        Ok(Self {
            token_mixer,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = match &mut self.token_mixer {
            TokenMixer::FullAttention(attn) => attn.forward(&x, mask, offset)?,
            TokenMixer::LinearAttention(attn) => attn.forward(&x)?,
        };
        let x = (residual + x)?;
        
        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = x.apply(&self.mlp)?;
        x + residual
    }

    fn clear_kv_cache(&mut self) {
        match &mut self.token_mixer {
            TokenMixer::FullAttention(attn) => attn.clear_kv_cache(),
            TokenMixer::LinearAttention(attn) => attn.clear_kv_cache(),
        }
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
}

impl ModelWeights {
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let mut gg = Gguf::new(ct, reader, device.clone());
        let md_get = |s: &str| {
            match gg.metadata().get(s) {
                Some(v) => Ok(v),
                None => {
                    let s35 = s.replace("qwen3.", "qwen35.");
                    match gg.metadata().get(&s35) {
                        Some(v) => Ok(v),
                        None => candle::bail!("cannot find {s} or {s35} in metadata"),
                    }
                }
            }
        };

        let num_attention_heads = md_get("qwen3.attention.head_count")?.to_u32()? as usize;
        let num_kv_heads = md_get("qwen3.attention.head_count_kv")?.to_u32()? as usize;
        let key_length = md_get("qwen3.attention.key_length")?.to_u32()? as usize;
        
        let head_dim = key_length; 
        let num_layers = md_get("qwen3.block_count")?.to_u32()? as usize;
        let hidden_size = md_get("qwen3.embedding_length")?.to_u32()? as usize;
        let max_position_embeddings = md_get("qwen3.context_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("qwen3.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("qwen3.rope.freq_base")?.to_f32()? as f64;
        
        let full_attention_interval = md_get("qwen3.full_attention_interval")?.to_u32()? as usize;
        
        // Linear attention specifics
        let linear_num_key_heads = md_get("qwen3.ssm.group_count")?.to_u32()? as usize;
        let ssm_inner_size = md_get("qwen3.ssm.inner_size")?.to_u32()? as usize;
        let linear_key_head_dim = md_get("qwen3.ssm.state_size")?.to_u32()? as usize;
        let linear_value_head_dim = linear_key_head_dim;
        let linear_num_value_heads = ssm_inner_size / linear_value_head_dim;
        let linear_conv_kernel_dim = md_get("qwen3.ssm.conv_kernel")?.to_u32()? as usize;

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
                i,
                num_attention_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
                rotary.clone(),
                full_attention_interval,
                linear_num_key_heads,
                linear_num_value_heads,
                linear_key_head_dim,
                linear_value_head_dim,
                linear_conv_kernel_dim,
                hidden_size,
            )?);
        }

        let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;
        let lm_head_tensor = match gg.tensor("output.weight") {
            Ok(tensor) => tensor,
            Err(_) => gg.tensor("token_embd.weight")?,
        };
        let lm_head = QMatMul::from_weights(lm_head_tensor.into())?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype,
        })
    }

    fn causal_mask(
        &self,
        b: usize,
        tgt: usize,
        offset: usize,
    ) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| {
                (0..(tgt + offset)).map(move |j| {
                    if j <= i + offset {
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
        let (b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;
        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset)?)
        };
        for layer in &mut self.layers {
            h = layer.forward(&h, causal_mask.as_ref(), offset)?;
        }
        let h = self.norm.forward(&h)?;
        let last_hidden = h.narrow(1, l - 1, 1)?;
        self.lm_head.forward(&last_hidden)?.squeeze(1)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}
