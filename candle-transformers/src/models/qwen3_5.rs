use crate::models::with_tracing::{linear_b, linear_no_bias, Linear};
use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Activation, Conv1d, Conv1dConfig, VarBuilder, conv1d_no_bias};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    pub text_config: TextConfig,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: Option<usize>,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_parameters: RopeParameters,
    pub attention_bias: bool,
    pub hidden_act: Activation,
    
    // Qwen 3.5 specific
    pub layer_types: Vec<String>,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f64,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.text_config.head_dim.unwrap_or(self.text_config.hidden_size / self.text_config.num_attention_heads)
    }

    pub fn rope_theta(&self) -> f64 {
        self.text_config.rope_parameters.rope_theta
    }
}



#[derive(Debug, Clone)]
pub struct Qwen3_5RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl Qwen3_5RmsNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_states = x.to_dtype(internal_dtype)?;
        let variance = hidden_states.sqr()?.mean_keepdim(D::Minus1)?;
        let hidden_states = hidden_states.broadcast_mul(&(&variance + self.eps)?.sqrt()?.recip()?)?;
        let weight = (&self.weight.to_dtype(internal_dtype)? + 1.0)?;
        hidden_states.broadcast_mul(&weight)?.to_dtype(x_dtype)
    }
}

fn l2_norm(x: &Tensor) -> Result<Tensor> {
    let inv_norm = (x.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?.recip()?;
    x.broadcast_mul(&inv_norm)
}

fn repeat_interleave(x: &Tensor, n: usize, dim: usize) -> Result<Tensor> {
    if n == 1 {
        return Ok(x.clone());
    }
    let mut dims = x.dims().to_vec();
    dims.insert(dim + 1, n);
    let expanded = x.unsqueeze(dim + 1)?.broadcast_as(dims.as_slice())?;
    let mut final_dims = x.dims().to_vec();
    final_dims[dim] *= n;
    expanded.reshape(final_dims.as_slice())
}

#[derive(Debug, Clone)]
pub struct Qwen3_5GatedDeltaNet {
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel_size: usize,
    conv_dim: usize,
    
    conv1d: Conv1d,
    dt_bias_f32: Tensor,
    neg_a_f32: Tensor,
    
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_b: Linear,
    in_proj_a: Linear,
    out_proj: Linear,
    
    norm_weight: Tensor,
    norm_eps: f64,

    conv_state: Option<Tensor>,
    recurrent_state: Option<Tensor>,
}

impl Qwen3_5GatedDeltaNet {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.text_config.hidden_size;
        let num_v_heads = cfg.text_config.linear_num_value_heads;
        let num_k_heads = cfg.text_config.linear_num_key_heads;
        let head_k_dim = cfg.text_config.linear_key_head_dim;
        let head_v_dim = cfg.text_config.linear_value_head_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        
        let conv_dim = key_dim * 2 + value_dim;
        let conv_kernel_size = cfg.text_config.linear_conv_kernel_dim;
        
        let conv1d_cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: conv_dim,
            ..Default::default()
        };
        // wait, we manually pad, so padding is 0 for the conv1d module itself
        let conv1d = conv1d_no_bias(conv_dim, conv_dim, conv_kernel_size, conv1d_cfg, vb.pp("conv1d"))?;
        
        let dt_bias = vb.get(num_v_heads, "dt_bias")?;
        let a_log = vb.get(num_v_heads, "A_log")?;
        
        let dt_bias_f32 = dt_bias.to_dtype(DType::F32)?;
        let neg_a_f32 = (&a_log.to_dtype(DType::F32)?.exp()? * -1.0)?;
        
        let in_proj_qkv = linear_no_bias(hidden_size, conv_dim, vb.pp("in_proj_qkv"))?;
        let in_proj_z = linear_no_bias(hidden_size, value_dim, vb.pp("in_proj_z"))?;
        let in_proj_b = linear_no_bias(hidden_size, num_v_heads, vb.pp("in_proj_b"))?;
        let in_proj_a = linear_no_bias(hidden_size, num_v_heads, vb.pp("in_proj_a"))?;
        let out_proj = linear_no_bias(value_dim, hidden_size, vb.pp("out_proj"))?;
        
        let norm_weight = vb.get(head_v_dim, "norm.weight")?;
        
        Ok(Self {
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_kernel_size,
            conv_dim,
            conv1d,
            dt_bias_f32,
            neg_a_f32,
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            out_proj,
            norm_weight,
            norm_eps: cfg.text_config.rms_norm_eps,
            conv_state: None,
            recurrent_state: None,
        })
    }

    fn rms_norm_gated(&self, hidden_states: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let input_dtype = hidden_states.dtype();
        let hidden_states = hidden_states.to_dtype(DType::F32)?;
        let variance = hidden_states.sqr()?.mean_keepdim(D::Minus1)?;
        let hidden_states = hidden_states.broadcast_mul(&(&variance + self.norm_eps)?.sqrt()?.recip()?)?;
        let hidden_states = hidden_states.broadcast_mul(&self.norm_weight.to_dtype(DType::F32)?)?;
        let hidden_states = hidden_states.to_dtype(input_dtype)?;
        let gate_silu = candle_nn::ops::silu(&gate.to_dtype(DType::F32)?)?.to_dtype(input_dtype)?;
        hidden_states.broadcast_mul(&gate_silu)
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
        let query = l2_norm(query)?;
        let key = l2_norm(key)?;
        
        // Keep in initial_dtype (usually BF16) if possible, but core update might need F32 for stability
        let query = query.transpose(1, 2)?; // (batch, heads, seq, dim)
        let key = key.transpose(1, 2)?;
        let value = value.transpose(1, 2)?;
        let beta = beta.transpose(1, 2)?;
        let g = g.transpose(1, 2)?;
        
        let (batch_size, num_heads, seq_len, k_head_dim) = key.dims4()?;
        let v_head_dim = value.dim(3)?;
        
        let scale = 1.0 / (query.dim(3)? as f64).sqrt();
        let query = (query * scale)?;
        
        let mut core_attn_out_vec = Vec::with_capacity(seq_len);
        
        // We MUST use F32 for the recurrent state to avoid numerical drift
        let mut last_recurrent_state = match initial_state {
            Some(state) => state.to_dtype(DType::F32)?,
            None => Tensor::zeros((batch_size, num_heads, k_head_dim, v_head_dim), DType::F32, query.device())?,
        };
        
        // Move loop-invariant conversions outside and prepare dimensions
        let query_f32 = query.to_dtype(DType::F32)?;
        let key_f32 = key.to_dtype(DType::F32)?;
        let value_f32 = value.to_dtype(DType::F32)?;
        let g_exp = g.to_dtype(DType::F32)?.exp()?.unsqueeze(3)?; // (B, H, S, 1)
        let beta_f32 = beta.to_dtype(DType::F32)?.unsqueeze(3)?; // (B, H, S, 1)

        for i in 0..seq_len {
            let q_t = query_f32.narrow(2, i, 1)?; // (B, H, 1, K)
            let k_t = key_f32.narrow(2, i, 1)?;   // (B, H, 1, K)
            let v_t = value_f32.narrow(2, i, 1)?; // (B, H, 1, V)
            let g_t = g_exp.narrow(2, i, 1)?;     // (B, H, 1, 1)
            let beta_t = beta_f32.narrow(2, i, 1)?; // (B, H, 1, 1)

            last_recurrent_state = last_recurrent_state.broadcast_mul(&g_t)?;
            
            // k_t.matmul(&state) is equivalent to broadcast_mul + sum(2)
            let kv_mem = k_t.matmul(&last_recurrent_state)?; // (B, H, 1, V)
            let delta = (&v_t - &kv_mem)?.broadcast_mul(&beta_t)?; // (B, H, 1, V)
            
            // k_t.T.matmul(&delta) is equivalent to outer product
            let delta_k = k_t.transpose(2, 3)?.matmul(&delta)?; // (B, H, K, 1) x (B, H, 1, V) = (B, H, K, V)
            last_recurrent_state = (&last_recurrent_state + &delta_k)?;
            
            let out_t = q_t.matmul(&last_recurrent_state)?; // (B, H, 1, V)
            core_attn_out_vec.push(out_t);
        }
        
        let core_attn_out = Tensor::cat(&core_attn_out_vec, 2)?; // (B, H, S, V)
        let core_attn_out = core_attn_out.transpose(1, 2)?; // (B, S, H, V)
        Ok((core_attn_out, last_recurrent_state))
    }

    pub fn forward(&mut self, hidden_states: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        let hidden_states = hidden_states.contiguous()?;
        let initial_dtype = hidden_states.dtype();
        
        let mixed_qkv = self.in_proj_qkv.forward(&hidden_states)?.transpose(1, 2)?; 
        
        let z = self.in_proj_z.forward(&hidden_states)?;
        let z = z.reshape((batch_size, seq_len, (), self.head_v_dim))?;
        
        let b = self.in_proj_b.forward(&hidden_states)?;
        let a = self.in_proj_a.forward(&hidden_states)?;
        
        let use_precomputed_states = self.conv_state.is_some() && seq_len == 1;
        
        let mixed_qkv = if use_precomputed_states {
            let conv_state = self.conv_state.as_mut().unwrap();
            let conv_state_data = Tensor::cat(&[conv_state.as_ref(), &mixed_qkv], 2)?;
            *conv_state = conv_state_data.narrow(2, 1, self.conv_kernel_size - 1)?;
            let out = conv_state_data.conv1d(&self.conv1d.weight(), 0, 1, 1, self.conv_dim)?;
            candle_nn::ops::silu(&out)?
        } else {
            let pad = self.conv_kernel_size - 1;
            let padding = Tensor::zeros((batch_size, self.conv_dim, pad), mixed_qkv.dtype(), mixed_qkv.device())?;
            let padded_qkv = Tensor::cat(&[&padding, &mixed_qkv], 2)?;
            self.conv_state = Some(padded_qkv.narrow(2, seq_len, pad)?);
            let out = padded_qkv.conv1d(&self.conv1d.weight(), 0, 1, 1, self.conv_dim)?;
            candle_nn::ops::silu(&out)?
        };
        
        let mixed_qkv = mixed_qkv.transpose(1, 2)?; // (batch, seq, conv_dim)
        
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
        let core_attn_out = core_attn_out.reshape(((), self.head_v_dim))?;
        let z_flat = z.reshape(((), self.head_v_dim))?;
        let core_attn_out = self.rms_norm_gated(&core_attn_out, &z_flat)?;
        let core_attn_out = core_attn_out.reshape((batch_size, seq_len, ()))?;
        
        self.out_proj.forward(&core_attn_out)
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3_5TextRotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl Qwen3_5TextRotaryEmbedding {
    pub fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim();
        let max_seq_len = cfg.text_config.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta().powf(i as f64 / dim as f64) as f32)
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

    pub fn apply(&self, q: &Tensor, k: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3_5Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Qwen3_5RmsNorm,
    k_norm: Qwen3_5RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<Qwen3_5TextRotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Qwen3_5Attention {
    pub fn new(
        cfg: &Config,
        rotary_emb: Arc<Qwen3_5TextRotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let num_heads = cfg.text_config.num_attention_heads;
        let num_kv_heads = cfg.text_config.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;

        let q_proj = linear_b(cfg.text_config.hidden_size, num_heads * head_dim * 2, cfg.text_config.attention_bias, vb.pp("q_proj"))?;
        let k_proj = linear_b(cfg.text_config.hidden_size, num_kv_heads * head_dim, cfg.text_config.attention_bias, vb.pp("k_proj"))?;
        let v_proj = linear_b(cfg.text_config.hidden_size, num_kv_heads * head_dim, cfg.text_config.attention_bias, vb.pp("v_proj"))?;
        let o_proj = linear_b(num_heads * head_dim, cfg.text_config.hidden_size, cfg.text_config.attention_bias, vb.pp("o_proj"))?;

        let q_norm = Qwen3_5RmsNorm::new(head_dim, cfg.text_config.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = Qwen3_5RmsNorm::new(head_dim, cfg.text_config.rms_norm_eps, vb.pp("k_norm"))?;

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
            kv_cache: None,
        })
    }

    pub fn forward(
        &mut self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = hidden_states.dims3()?;
        
        let q_proj_out = self.q_proj.forward(hidden_states)?;
        let q_proj_out = q_proj_out.reshape((b_sz, q_len, self.num_heads, self.head_dim * 2))?;
        
        let query_states = q_proj_out.narrow(D::Minus1, 0, self.head_dim)?;
        let gate = q_proj_out.narrow(D::Minus1, self.head_dim, self.head_dim)?;
        
        let query_states = self.q_norm.forward(&query_states)?.transpose(1, 2)?;
        
        let key_states = self.k_proj.forward(hidden_states)?;
        let key_states = key_states.reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?;
        let key_states = self.k_norm.forward(&key_states)?.transpose(1, 2)?;
        
        let value_states = self.v_proj.forward(hidden_states)?;
        let value_states = value_states.reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        
        let (query_states, key_states) = self.rotary_emb.apply(&query_states, &key_states, seqlen_offset)?;
        
        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &key_states], 2)?;
                let v = Tensor::cat(&[prev_v, &value_states], 2)?;
                (k, v)
            }
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));
        
        let key_states = crate::utils::repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states = crate::utils::repeat_kv(value_states, self.num_kv_groups)?.contiguous()?;
        
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;
        
        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&value_states)?;
        
        let attn_output = attn_output.transpose(1, 2)?.reshape((b_sz, q_len, self.num_heads, self.head_dim))?;
        
        let attn_output = attn_output.broadcast_mul(&candle_nn::ops::sigmoid(&gate.to_dtype(DType::F32)?)?.to_dtype(attn_output.dtype())?)?;
        
        let attn_output = attn_output.reshape((b_sz, q_len, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&attn_output)
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3_5MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Qwen3_5MLP {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.text_config.hidden_size;
        let intermediate_sz = cfg.text_config.intermediate_size;
        let gate_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_sz, hidden_sz, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.text_config.hidden_act,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
enum TokenMixer {
    FullAttention(Qwen3_5Attention),
    LinearAttention(Qwen3_5GatedDeltaNet),
}

#[derive(Debug, Clone)]
struct Qwen3_5DecoderLayer {
    token_mixer: TokenMixer,
    mlp: Qwen3_5MLP,
    input_layernorm: Qwen3_5RmsNorm,
    post_attention_layernorm: Qwen3_5RmsNorm,
}

impl Qwen3_5DecoderLayer {
    fn new(cfg: &Config, layer_idx: usize, rotary_emb: Arc<Qwen3_5TextRotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        let layer_type = cfg.text_config.layer_types.get(layer_idx).cloned().unwrap_or_else(|| "full_attention".to_string());
        
        let token_mixer = if layer_type == "linear_attention" {
            TokenMixer::LinearAttention(Qwen3_5GatedDeltaNet::new(cfg, vb.pp("linear_attn"))?)
        } else {
            TokenMixer::FullAttention(Qwen3_5Attention::new(cfg, rotary_emb, vb.pp("self_attn"))?)
        };
        
        let mlp = Qwen3_5MLP::new(cfg, vb.pp("mlp"))?;
        let input_layernorm = Qwen3_5RmsNorm::new(cfg.text_config.hidden_size, cfg.text_config.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = Qwen3_5RmsNorm::new(cfg.text_config.hidden_size, cfg.text_config.rms_norm_eps, vb.pp("post_attention_layernorm"))?;
        
        Ok(Self {
            token_mixer,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
    
    fn forward(&mut self, hidden_states: &Tensor, attention_mask: Option<&Tensor>, seqlen_offset: usize) -> Result<Tensor> {
        let residual = hidden_states;
        let norm_hidden_states = self.input_layernorm.forward(hidden_states)?;
        
        let mixed_states = match &mut self.token_mixer {
            TokenMixer::FullAttention(attn) => attn.forward(&norm_hidden_states, attention_mask, seqlen_offset)?,
            TokenMixer::LinearAttention(attn) => attn.forward(&norm_hidden_states)?, 
        };
        
        let hidden_states = (residual + mixed_states)?;
        let residual = &hidden_states;
        let norm_hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let mlp_states = self.mlp.forward(&norm_hidden_states)?;
        residual + mlp_states
    }
    
    fn clear_kv_cache(&mut self) {
        match &mut self.token_mixer {
            TokenMixer::FullAttention(attn) => attn.kv_cache = None,
            TokenMixer::LinearAttention(attn) => {
                attn.conv_state = None;
                attn.recurrent_state = None;
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Qwen3_5DecoderLayer>,
    norm: Qwen3_5RmsNorm,
    device: Device,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model").pp("language_model");
        let embed_tokens = candle_nn::embedding(cfg.text_config.vocab_size, cfg.text_config.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(Qwen3_5TextRotaryEmbedding::new(vb.dtype(), cfg, vb_m.device())?);
        
        let mut layers = Vec::with_capacity(cfg.text_config.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.text_config.num_hidden_layers {
            let layer = Qwen3_5DecoderLayer::new(cfg, layer_idx, rotary_emb.clone(), vb_l.pp(layer_idx))?;
            layers.push(layer);
        }
        
        let norm = Qwen3_5RmsNorm::new(cfg.text_config.hidden_size, cfg.text_config.rms_norm_eps, vb_m.pp("norm"))?;
        
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn prepare_causal_attention_mask(&self, b_size: usize, tgt_len: usize, seqlen_offset: usize) -> Result<Tensor> {
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), self.dtype, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?.to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(self.prepare_causal_attention_mask(b_size, seq_len, seqlen_offset)?)
        };

        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?;
        }
        self.norm.forward(&xs)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelForCausalLM {
    pub base_model: Model,
    pub lm_head: Linear,
}

impl ModelForCausalLM {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let base_model = Model::new(cfg, vb.clone())?;
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            linear_no_bias(cfg.text_config.hidden_size, cfg.text_config.vocab_size, vb.pp("lm_head"))?
        } else {
            Linear::from_weights(base_model.embed_tokens.embeddings().clone(), None)
        };
        Ok(Self {
            base_model,
            lm_head,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;
        self.base_model
            .forward(input_ids, seqlen_offset)?
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.lm_head)
    }

    pub fn clear_kv_cache(&mut self) {
        self.base_model.clear_kv_cache();
    }
}
