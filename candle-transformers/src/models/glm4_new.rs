use crate::models::glm4::EosTokenId;
use crate::{
    models::with_tracing::{linear_b, linear_no_bias, Linear, RmsNorm},
    utils::repeat_kv,
};
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{kv_cache::IncrementalKvCache, Activation, VarBuilder};
use std::sync::Arc;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: Option<usize>,
    pub partial_rotary_factor: Option<f32>,
    pub attention_bias: Option<bool>,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub hidden_act: Activation,
    pub eos_token_id: Option<EosTokenId>,
}

#[derive(Debug, Clone)]
pub(crate) struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    rotary_dim: usize,
}

impl RotaryEmbedding {
    pub(crate) fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg
            .head_dim
            .unwrap_or(cfg.hidden_size / cfg.num_attention_heads);
        let rotary_dim = if cfg.partial_rotary_factor.is_some() {
            (cfg.partial_rotary_factor.unwrap() * dim as f32) as usize
        } else {
            dim
        };
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..rotary_dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / rotary_dim as f64) as f32)
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
pub(crate) struct Mlp {
    gate_up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Mlp {
    pub(crate) fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_up_proj: linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size * 2,
                vb.pp("gate_up_proj"),
            )?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = self.gate_up_proj.forward(x)?;
        let dim = w.dims().len() - 1;
        let gate = w.narrow(dim, 0, w.dim(dim)? / 2)?.contiguous()?;
        let gate = gate.apply(&self.act_fn)?;
        let up_states = w
            .narrow(dim, w.dim(dim)? / 2, w.dim(dim)? / 2)?
            .contiguous()?;
        self.down_proj.forward(&(gate * up_states)?)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: IncrementalKvCache,
}

impl Attention {
    pub(crate) fn new(
        cfg: &Config,
        rotary_emb: Arc<RotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = cfg
            .head_dim
            .unwrap_or(cfg.hidden_size / cfg.num_attention_heads);
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;

        let q_proj = linear_b(
            cfg.hidden_size,
            num_heads * head_dim,
            cfg.attention_bias.unwrap_or(false),
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_b(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            cfg.attention_bias.unwrap_or(false),
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_b(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            cfg.attention_bias.unwrap_or(false),
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_b(
            num_heads * head_dim,
            cfg.hidden_size,
            false,
            vb.pp("o_proj"),
        )?;

        // Necessary because the hidden_size in the config isn't always accurate
        let hidden_size = head_dim * cfg.num_attention_heads;

        // Initialize KV cache with 512 tokens capacity to reduce initial memory allocation.
        // The cache will grow in chunks of 512 tokens when needed.
        let kv_cache = IncrementalKvCache::new(2, 512);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size,
            rotary_emb,
            kv_cache,
        })
    }

    pub(crate) fn forward(
        &mut self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
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

        let q = self.rotary_emb.apply(&q, offset)?;
        let k = self.rotary_emb.apply(&k, offset)?;

        let (k, v) = self.kv_cache.append(&k.contiguous()?, &v.contiguous()?)?;

        let k = repeat_kv(k, self.num_kv_groups)?;
        let v = repeat_kv(v, self.num_kv_groups)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = attn_mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;

        ctx.transpose(1, 2)?
            .reshape((b, l, self.hidden_size))?
            .apply(&self.o_proj)
    }

    pub(crate) fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    post_mlp_layernorm: RmsNorm,
    post_self_attn_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &Config, rotary: Arc<RotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(cfg, rotary, vb.pp("self_attn"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;

        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let post_self_attn_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_self_attn_layernorm"),
        )?;
        let post_mlp_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_mlp_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            post_self_attn_layernorm,
            post_mlp_layernorm,
        })
    }

    fn forward(&mut self, xs: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let residual = xs;
        let hidden_states = self.input_layernorm.forward(xs)?;
        let hidden_states = self.self_attn.forward(&hidden_states, mask, offset)?;
        let hidden_states = self.post_self_attn_layernorm.forward(&hidden_states)?;
        let hidden_states = (residual + hidden_states)?;
        let residual = &hidden_states;
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        let hidden_states = self.post_mlp_layernorm.forward(&hidden_states)?;
        residual + hidden_states
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    device: Device,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let rotary = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("model.layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, rotary.clone(), vb_l.pp(i))?);
        }
        Ok(Self {
            embed_tokens,
            layers,
            norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn clear_kv_cache(&mut self) {
        for l in &mut self.layers {
            l.clear_kv_cache();
        }
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
        self.norm.forward(&h)
    }
}

#[derive(Debug, Clone)]
pub struct ModelForCausalLM {
    base: Model,
    lm_head: Linear,
}

impl ModelForCausalLM {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let base = Model::new(cfg, vb.clone())?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::from_weights(base.embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        Ok(Self { base, lm_head })
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, l) = input.dims2()?;
        self.base
            .forward(input, offset)?
            .narrow(1, l - 1, 1)?
            .apply(&self.lm_head)
    }

    pub fn clear_kv_cache(&mut self) {
        self.base.clear_kv_cache();
    }
}
