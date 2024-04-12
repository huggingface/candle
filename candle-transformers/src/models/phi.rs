use crate::models::with_tracing::{layer_norm, linear, Embedding, LayerNorm, Linear};
/// Phi model.
/// https://huggingface.co/microsoft/phi-2
/// There is an alternative implementation of the phi model in mixformers.rs.
/// This corresponds to the model update made with the following commit:
/// https://huggingface.co/microsoft/phi-2/commit/cb2f4533604d8b67de604e7df03bfe6f3ca22869
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Activation, VarBuilder};
use serde::Deserialize;

// https://huggingface.co/microsoft/phi-2/blob/main/configuration_phi.py
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: Option<usize>,
    pub(crate) hidden_act: Activation,
    pub(crate) max_position_embeddings: usize,
    pub(crate) layer_norm_eps: f64,
    pub(crate) tie_word_embeddings: bool,
    pub(crate) rope_theta: f32,
    pub(crate) partial_rotary_factor: f64,
    pub(crate) qk_layernorm: bool,
}

impl Config {
    fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    dim: usize,
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = (cfg.partial_rotary_factor * cfg.head_dim() as f64) as usize;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        Ok(Self {
            dim,
            sin: emb.sin()?,
            cos: emb.cos()?,
        })
    }

    fn apply_rotary_emb(&self, xs: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (_b_size, _num_heads, seq_len, _headdim) = xs.dims4()?;
        let xs_rot = xs.i((.., .., .., ..self.dim))?;
        let xs_pass = xs.i((.., .., .., self.dim..))?;
        let xs12 = xs_rot.chunk(2, D::Minus1)?;
        let (xs1, xs2) = (&xs12[0], &xs12[1]);
        let c = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let s = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let rotate_half = Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)?;
        let xs_rot = (xs_rot.broadcast_mul(&c)? + rotate_half.broadcast_mul(&s)?)?;
        Tensor::cat(&[&xs_rot, &xs_pass], D::Minus1)
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    fc1: Linear,
    fc2: Linear,
    act: Activation,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("fc1"))?;
        let fc2 = linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("fc2"))?;
        Ok(Self {
            fc1,
            fc2,
            // This does not match the mixformers implementation where Gelu is used rather than
            // GeluNew.
            act: cfg.hidden_act,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.fc1)?.apply(&self.act)?.apply(&self.fc2)
    }
}

#[derive(Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    dense: Linear,
    kv_cache: Option<(Tensor, Tensor)>,
    q_layernorm: Option<LayerNorm>,
    k_layernorm: Option<LayerNorm>,
    rotary_emb: RotaryEmbedding,
    softmax_scale: f64,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    span: tracing::Span,
}

fn get_mask(size: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<_> = (0..size)
        .flat_map(|i| (0..size).map(move |j| u8::from(j > i)))
        .collect();
    Tensor::from_slice(&mask, (size, size), device)
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

impl Attention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads();
        let head_dim = cfg.head_dim();
        let q_proj = linear(cfg.hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let dense = linear(num_heads * head_dim, cfg.hidden_size, vb.pp("dense"))?;
        // Alternative rope scalings are not supported.
        let rotary_emb = RotaryEmbedding::new(cfg, vb.device())?;
        let (q_layernorm, k_layernorm) = if cfg.qk_layernorm {
            let q_layernorm = layer_norm(head_dim, cfg.layer_norm_eps, vb.pp("q_layernorm"))?;
            let k_layernorm = layer_norm(head_dim, cfg.layer_norm_eps, vb.pp("k_layernorm"))?;
            (Some(q_layernorm), Some(k_layernorm))
        } else {
            (None, None)
        };
        let softmax_scale = 1f64 / (head_dim as f64).sqrt();
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            dense,
            kv_cache: None,
            q_layernorm,
            k_layernorm,
            rotary_emb,
            softmax_scale,
            num_heads,
            num_kv_heads,
            head_dim,
            span: tracing::span!(tracing::Level::TRACE, "attention"),
        })
    }

    fn repeat_kv(&self, xs: Tensor) -> Result<Tensor> {
        crate::utils::repeat_kv(xs, self.num_heads / self.num_kv_heads)
    }

    fn forward(&mut self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b_size, seq_len, _n_embd) = xs.dims3()?;
        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = match &self.q_layernorm {
            None => query_states,
            Some(ln) => query_states.apply(ln)?,
        };
        let key_states = match &self.k_layernorm {
            None => key_states,
            Some(ln) => key_states.apply(ln)?,
        };

        let query_states = query_states
            .reshape((b_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Rotary embeddings.
        let seqlen_offset = match &self.kv_cache {
            None => 0,
            Some((prev_k, _)) => prev_k.dim(2)?,
        };
        let query_states = self
            .rotary_emb
            .apply_rotary_emb(&query_states, seqlen_offset)?;
        let key_states = self
            .rotary_emb
            .apply_rotary_emb(&key_states, seqlen_offset)?;

        // KV cache.
        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &key_states], 2)?;
                let v = Tensor::cat(&[prev_v, &value_states], 2)?;
                (k, v)
            }
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        // Repeat kv.
        let key_states = self.repeat_kv(key_states)?.contiguous()?;
        let value_states = self.repeat_kv(value_states)?.contiguous()?;

        let attn_weights = (query_states
            .to_dtype(DType::F32)?
            .contiguous()?
            .matmul(&key_states.to_dtype(DType::F32)?.t()?)?
            * self.softmax_scale)?;
        let attn_weights = match mask {
            None => attn_weights,
            Some(mask) => masked_fill(
                &attn_weights,
                &mask.broadcast_left((b_size, self.num_heads))?,
                f32::NEG_INFINITY,
            )?,
        };
        let attn_weights =
            candle_nn::ops::softmax_last_dim(&attn_weights)?.to_dtype(value_states.dtype())?;
        let attn_output = attn_weights.matmul(&value_states)?;
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_size, seq_len, ()))?;
        attn_output.apply(&self.dense)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

#[derive(Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: LayerNorm,
    span: tracing::Span,
}

impl DecoderLayer {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(cfg, vb.pp("self_attn"))?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        let input_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.layer_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            span: tracing::span!(tracing::Level::TRACE, "block"),
        })
    }

    fn forward(&mut self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();
        let residual = xs;
        let xs = xs.apply(&self.input_layernorm)?;
        let attn_outputs = self.self_attn.forward(&xs, mask)?;
        let feed_forward_hidden_states = self.mlp.forward(&xs)?;
        attn_outputs + feed_forward_hidden_states + residual
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
}

#[derive(Clone)]
pub struct Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    final_layernorm: LayerNorm,
    lm_head: Linear,
    span: tracing::Span,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            Embedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let final_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.layer_norm_eps,
            vb_m.pp("final_layernorm"),
        )?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_m = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(cfg, vb_m.pp(layer_idx))?;
            layers.push(layer)
        }
        let lm_head = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        Ok(Self {
            embed_tokens,
            layers,
            final_layernorm,
            lm_head,
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    pub fn forward(&mut self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (_b_size, seq_len) = xs.dims2()?;
        let mut xs = xs.apply(&self.embed_tokens)?;
        let mask = if seq_len <= 1 {
            None
        } else {
            Some(get_mask(seq_len, xs.device())?)
        };
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, mask.as_ref())?;
        }
        xs.apply(&self.final_layernorm)?
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.lm_head)?
            .squeeze(1)
    }

    pub fn clear_kv_cache(&mut self) {
        self.layers.iter_mut().for_each(|b| b.clear_kv_cache())
    }
}
