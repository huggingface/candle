//! Native LFM3 hybrid transformer/convolution backbone.
//!
//! Tensor operations remain on the selected Candle device. Quantized GGUF linear
//! weights stay quantized; convolution/norm weights are materialized as F32.
use crate::quantized_var_builder;
use crate::utils::repeat_kv;
use candle::{quantized::QMatMul, DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{Embedding, RmsNorm, VarBuilder};
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    Conv,
    FullAttention,
    SlidingAttention,
}
#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f64,
    #[serde(default = "one")]
    pub partial_rotary_factor: f64,
    #[serde(default)]
    pub rope_type: String,
}
fn one() -> f64 {
    1.
}
fn epsilon() -> f64 {
    1e-5
}
fn width() -> usize {
    7
}
fn window() -> usize {
    256
}
fn context() -> usize {
    128000
}
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: Option<usize>,
    #[serde(default = "context")]
    pub max_position_embeddings: usize,
    #[serde(default = "epsilon")]
    pub norm_eps: f64,
    #[serde(default = "width", rename = "conv_L_cache", alias = "conv_l_cache")]
    pub conv_l_cache: usize,
    #[serde(default)]
    pub conv_bias: bool,
    #[serde(default = "window")]
    pub sliding_window: usize,
    #[serde(default)]
    pub use_sandwich_norm: bool,
    pub layer_types: Vec<LayerType>,
    pub rope_parameters: serde_json::Value,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}
impl Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
            || self.num_hidden_layers != self.layer_types.len()
            || self.conv_l_cache == 0
            || self.intermediate_size == 0
            || self.max_position_embeddings == 0
        {
            candle::bail!("invalid LFM3 dimensions/layer configuration")
        }
        if !self.head_dim().is_multiple_of(2) {
            candle::bail!("LFM3 attention head dimension must be even")
        }
        Ok(())
    }
    fn rope(&self, kind: LayerType) -> Result<RopeParameters> {
        let name = if kind == LayerType::SlidingAttention {
            "sliding_attention"
        } else {
            "full_attention"
        };
        let value = self
            .rope_parameters
            .get(name)
            .unwrap_or(&self.rope_parameters)
            .clone();
        serde_json::from_value(value).map_err(candle::Error::wrap)
    }
}

/// Weight source supporting native safetensors and quantization-preserving GGUF.
#[derive(Clone)]
pub enum Weights<'a> {
    Dense(VarBuilder<'a>),
    Gguf(quantized_var_builder::VarBuilder),
}
impl<'a> Weights<'a> {
    pub fn device(&self) -> &Device {
        match self {
            Self::Dense(v) => v.device(),
            Self::Gguf(v) => v.device(),
        }
    }
    pub fn dense(
        &self,
        shape: impl Into<candle::Shape>,
        dense: &str,
        gguf: &str,
    ) -> Result<Tensor> {
        match self {
            Self::Dense(v) => v.get(shape, dense),
            Self::Gguf(v) => {
                let shape = shape.into();
                let t = v.get_no_shape(gguf)?.dequantize(v.device())?;
                if t.elem_count() != shape.elem_count() {
                    candle::bail!(
                        "LFM3 tensor {gguf} shape mismatch {:?} vs {shape:?}",
                        t.shape()
                    )
                }
                t.reshape(shape)
            }
        }
    }
    pub fn linear(
        &self,
        input: usize,
        output: usize,
        dense: &str,
        gguf: &str,
        bias: bool,
    ) -> Result<Linear> {
        let weight = match self {
            Self::Dense(v) => QMatMul::Tensor(v.get((output, input), &format!("{dense}.weight"))?),
            Self::Gguf(v) => QMatMul::from_arc(v.get((output, input), &format!("{gguf}.weight"))?)?,
        };
        let bias = if bias {
            Some(self.dense(output, &format!("{dense}.bias"), &format!("{gguf}.bias"))?)
        } else {
            None
        };
        Ok(Linear { weight, bias })
    }
    pub fn norm(&self, size: usize, eps: f64, dense: &str, gguf: &str) -> Result<RmsNorm> {
        Ok(RmsNorm::new(
            self.dense(size, &format!("{dense}.weight"), &format!("{gguf}.weight"))?,
            eps,
        ))
    }
}
#[derive(Debug, Clone)]
pub struct Linear {
    weight: QMatMul,
    bias: Option<Tensor>,
}
impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.weight.forward(x)?;
        match &self.bias {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }
}

/// Causal gated depthwise convolution with a bounded width-1 input history.
/// Shared by the language model, audio encoder and LiquidCodec decoder.
pub fn causal_conv(
    projected: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    cache: &mut Option<Tensor>,
) -> Result<Tensor> {
    let (batch, time, three) = projected.dims3()?;
    let (hidden, groups, width) = weight.dims3()?;
    if three != 3 * hidden || groups != 1 || width == 0 || time == 0 {
        candle::bail!("invalid LFM3 gated convolution shapes")
    }
    let x = projected
        .narrow(2, 0, hidden)?
        .broadcast_mul(&projected.narrow(2, 2 * hidden, hidden)?)?
        .transpose(1, 2)?;
    let history = match cache.as_ref() {
        Some(h) => h.clone(),
        None => Tensor::zeros((batch, hidden, width - 1), x.dtype(), x.device())?,
    };
    let all = Tensor::cat(&[&history, &x], 2)?.contiguous()?;
    let mut y = all.conv1d(weight, 0, 1, 1, hidden)?;
    if let Some(b) = bias {
        y = y.broadcast_add(&b.reshape((1, hidden, 1))?)?;
    }
    *cache = Some(all.narrow(2, time, width - 1)?.contiguous()?);
    y.transpose(1, 2)?
        .broadcast_mul(&projected.narrow(2, hidden, hidden)?)
}
#[derive(Clone)]
struct Conv {
    input: Linear,
    output: Linear,
    weight: Tensor,
    bias: Option<Tensor>,
    cache: Option<Tensor>,
}
impl Conv {
    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        self.output.forward(&causal_conv(
            &self.input.forward(x)?,
            &self.weight,
            self.bias.as_ref(),
            &mut self.cache,
        )?)
    }
}

#[derive(Clone)]
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    gate: Option<Linear>,
    qn: RmsNorm,
    kn: RmsNorm,
    heads: usize,
    kv_heads: usize,
    dim: usize,
    rope: RopeParameters,
    window: Option<usize>,
    cache: Option<(Tensor, Tensor)>,
}
impl Attention {
    fn rotary(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        if !self.rope.rope_type.is_empty() && self.rope.rope_type != "default" {
            candle::bail!("unsupported LFM3 RoPE type {}", self.rope.rope_type)
        }
        let length = x.dim(2)?;
        let pairs = (self.dim as f64 * self.rope.partial_rotary_factor) as usize / 2;
        let inv = (0..self.dim / 2)
            .map(|i| {
                if i < pairs {
                    1f32 / (self.rope.rope_theta as f32).powf((2 * i) as f32 / self.dim as f32)
                } else {
                    0.
                }
            })
            .collect::<Vec<_>>();
        let freq = Tensor::arange(offset as u32, (offset + length) as u32, x.device())?
            .to_dtype(DType::F32)?
            .reshape((length, 1))?
            .matmul(&Tensor::from_vec(inv, (1, self.dim / 2), x.device())?)?;
        candle_nn::rotary_emb::rope(
            &x.contiguous()?,
            &freq.cos()?.to_dtype(x.dtype())?,
            &freq.sin()?.to_dtype(x.dtype())?,
        )
    }
    fn forward(
        &mut self,
        x: &Tensor,
        offset: usize,
        causal: bool,
        cache_enabled: bool,
    ) -> Result<Tensor> {
        let (batch, time, _) = x.dims3()?;
        let q = self
            .qn
            .forward(
                &self
                    .q
                    .forward(x)?
                    .reshape((batch, time, self.heads, self.dim))?,
            )?
            .transpose(1, 2)?;
        let k = self
            .kn
            .forward(
                &self
                    .k
                    .forward(x)?
                    .reshape((batch, time, self.kv_heads, self.dim))?,
            )?
            .transpose(1, 2)?;
        let q = self.rotary(&q, offset)?;
        let k = self.rotary(&k, offset)?;
        let v = self
            .v
            .forward(x)?
            .reshape((batch, time, self.kv_heads, self.dim))?
            .transpose(1, 2)?;
        let (k, v) = match (self.cache.as_ref(), cache_enabled) {
            (Some((pk, pv)), true) => (Tensor::cat(&[pk, &k], 2)?, Tensor::cat(&[pv, &v], 2)?),
            _ => (k, v),
        };
        let count = k.dim(2)?;
        let past = count - time;
        let window = self.window;
        let mask = (0..time)
            .flat_map(|i| {
                (0..count).map(move |j| {
                    if (causal && j > past + i) || window.is_some_and(|w| j + w <= past + i) {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect::<Vec<_>>();
        let k_full = repeat_kv(k.clone(), self.heads / self.kv_heads)?.contiguous()?;
        let v_full = repeat_kv(v.clone(), self.heads / self.kv_heads)?.contiguous()?;
        let scores = (q.matmul(&k_full.transpose(2, 3)?)? / (self.dim as f64).sqrt())?
            .broadcast_add(&Tensor::from_vec(mask, (1, 1, time, count), x.device())?)?;
        let y = candle_nn::ops::softmax_last_dim(&scores)?
            .matmul(&v_full)?
            .transpose(1, 2)?
            .contiguous()?;
        let y = if let Some(gate) = &self.gate {
            y.broadcast_mul(&candle_nn::ops::sigmoid(&gate.forward(x)?)?.unsqueeze(D::Minus1)?)?
        } else {
            y
        };
        if cache_enabled {
            let retain = self.window.unwrap_or(count).min(count);
            self.cache = Some((
                k.narrow(2, count - retain, retain)?.contiguous()?,
                v.narrow(2, count - retain, retain)?.contiguous()?,
            ));
        }
        self.out
            .forward(&y.reshape((batch, time, self.heads * self.dim))?)
    }
}
#[derive(Clone)]
enum Operator {
    Conv(Conv),
    Attention(Box<Attention>),
}
#[derive(Clone)]
struct Layer {
    operator: Operator,
    norm: RmsNorm,
    ffn_norm: RmsNorm,
    post_operator: Option<RmsNorm>,
    post_ffn: Option<RmsNorm>,
    w1: Linear,
    w2: Linear,
    w3: Linear,
}
impl Layer {
    fn forward(
        &mut self,
        x: &Tensor,
        offset: usize,
        causal: bool,
        cache_enabled: bool,
    ) -> Result<Tensor> {
        let normalized = self.norm.forward(x)?;
        let y = match &mut self.operator {
            Operator::Conv(c) => c.forward(&normalized)?,
            Operator::Attention(a) => a.forward(&normalized, offset, causal, cache_enabled)?,
        };
        let y = match &self.post_operator {
            Some(n) => n.forward(&y)?,
            None => y,
        };
        let x = (x + y)?;
        let normalized = self.ffn_norm.forward(&x)?;
        let y = self.w2.forward(
            &self
                .w1
                .forward(&normalized)?
                .silu()?
                .broadcast_mul(&self.w3.forward(&normalized)?)?,
        )?;
        let y = match &self.post_ffn {
            Some(n) => n.forward(&y)?,
            None => y,
        };
        x + y
    }
}

/// A lightweight checkpoint of conversation tensors, sharing their immutable
/// storage and excluding all model weights. Restore after cancelled/failed turns.
#[derive(Clone, Debug)]
pub struct CacheState {
    position: usize,
    layers: Vec<LayerCache>,
}
#[derive(Clone, Debug)]
enum LayerCache {
    Conv(Option<Tensor>),
    Attention(Option<(Tensor, Tensor)>),
}
impl CacheState {
    pub fn position(&self) -> usize {
        self.position
    }
}

/// Hybrid stack reusable for LFM3 text, depth decoder and codec decoder.
#[derive(Clone)]
pub struct Stack {
    layers: Vec<Layer>,
    norm: RmsNorm,
    position: usize,
    max_positions: usize,
    causal: bool,
    cache_enabled: bool,
}
impl Stack {
    pub fn new(
        cfg: &Config,
        weights: &Weights<'_>,
        dense_root: &str,
        gguf_root: &str,
        causal: bool,
        cache_enabled: bool,
        gated_attention: bool,
    ) -> Result<Self> {
        cfg.validate()?;
        let h = cfg.hidden_size;
        let dim = cfg.head_dim();
        let mut layers = Vec::new();
        for (index, &kind) in cfg.layer_types.iter().enumerate() {
            let d = format!("{dense_root}.layers.{index}");
            let g = format!("{gguf_root}blk.{index}");
            let text_stack = gguf_root.is_empty();
            let conv_in = if text_stack {
                "shortconv.in_proj"
            } else {
                "conv_pw1"
            };
            let conv_out = if text_stack {
                "shortconv.out_proj"
            } else {
                "conv_pw2"
            };
            let conv_dw = if text_stack {
                "shortconv.conv"
            } else {
                "conv_dw"
            };
            let attn_out = if text_stack {
                "attn_output"
            } else {
                "attn_out"
            };
            let op_norm = if text_stack { "attn_norm" } else { "ln1" };
            let ff_norm = if text_stack { "ffn_norm" } else { "ln2" };
            let post_op = if text_stack {
                "post_attention_norm"
            } else {
                "attn_post_norm"
            };
            let post_ff = if text_stack {
                "post_ffw_norm"
            } else {
                "ffn_post_norm"
            };
            let operator = if kind == LayerType::Conv {
                let input = weights.linear(
                    h,
                    3 * h,
                    &format!("{d}.conv.in_proj"),
                    &format!("{g}.{conv_in}"),
                    cfg.conv_bias,
                )?;
                let output = weights.linear(
                    h,
                    h,
                    &format!("{d}.conv.out_proj"),
                    &format!("{g}.{conv_out}"),
                    cfg.conv_bias,
                )?;
                let weight = weights.dense(
                    (h, 1, cfg.conv_l_cache),
                    &format!("{d}.conv.conv.weight"),
                    &format!("{g}.{conv_dw}.weight"),
                )?;
                let bias = if cfg.conv_bias {
                    Some(weights.dense(
                        h,
                        &format!("{d}.conv.conv.bias"),
                        &format!("{g}.{conv_dw}.bias"),
                    )?)
                } else {
                    None
                };
                Operator::Conv(Conv {
                    input,
                    output,
                    weight,
                    bias,
                    cache: None,
                })
            } else {
                let q = weights.linear(
                    h,
                    cfg.num_attention_heads * dim,
                    &format!("{d}.self_attn.q_proj"),
                    &format!("{g}.attn_q"),
                    false,
                )?;
                let k = weights.linear(
                    h,
                    cfg.num_key_value_heads * dim,
                    &format!("{d}.self_attn.k_proj"),
                    &format!("{g}.attn_k"),
                    false,
                )?;
                let v = weights.linear(
                    h,
                    cfg.num_key_value_heads * dim,
                    &format!("{d}.self_attn.v_proj"),
                    &format!("{g}.attn_v"),
                    false,
                )?;
                let out_name = if gated_attention {
                    "o_proj"
                } else {
                    "out_proj"
                };
                let out = weights.linear(
                    cfg.num_attention_heads * dim,
                    h,
                    &format!("{d}.self_attn.{out_name}"),
                    &format!("{g}.{attn_out}"),
                    false,
                )?;
                let gate = if gated_attention {
                    Some(weights.linear(
                        h,
                        cfg.num_attention_heads,
                        &format!("{d}.self_attn.gate_proj"),
                        &format!("{g}.attn_gate"),
                        false,
                    )?)
                } else {
                    None
                };
                let qn_name = if gated_attention {
                    "q_norm"
                } else {
                    "q_layernorm"
                };
                let kn_name = if gated_attention {
                    "k_norm"
                } else {
                    "k_layernorm"
                };
                let qn = weights.norm(
                    dim,
                    cfg.norm_eps,
                    &format!("{d}.self_attn.{qn_name}"),
                    &format!("{g}.attn_q_norm"),
                )?;
                let kn = weights.norm(
                    dim,
                    cfg.norm_eps,
                    &format!("{d}.self_attn.{kn_name}"),
                    &format!("{g}.attn_k_norm"),
                )?;
                Operator::Attention(Box::new(Attention {
                    q,
                    k,
                    v,
                    out,
                    gate,
                    qn,
                    kn,
                    heads: cfg.num_attention_heads,
                    kv_heads: cfg.num_key_value_heads,
                    dim,
                    rope: cfg.rope(kind)?,
                    window: (kind == LayerType::SlidingAttention).then_some(cfg.sliding_window),
                    cache: None,
                }))
            };
            let norm = weights.norm(
                h,
                cfg.norm_eps,
                &format!("{d}.operator_norm"),
                &format!("{g}.{op_norm}"),
            )?;
            let ffn_norm = weights.norm(
                h,
                cfg.norm_eps,
                &format!("{d}.ffn_norm"),
                &format!("{g}.{ff_norm}"),
            )?;
            let post_operator = cfg
                .use_sandwich_norm
                .then(|| {
                    weights.norm(
                        h,
                        cfg.norm_eps,
                        &format!("{d}.post_operator_norm"),
                        &format!("{g}.{post_op}"),
                    )
                })
                .transpose()?;
            let post_ffn = cfg
                .use_sandwich_norm
                .then(|| {
                    weights.norm(
                        h,
                        cfg.norm_eps,
                        &format!("{d}.post_ffn_norm"),
                        &format!("{g}.{post_ff}"),
                    )
                })
                .transpose()?;
            let w1 = weights.linear(
                h,
                cfg.intermediate_size,
                &format!("{d}.feed_forward.w1"),
                &format!("{g}.ffn_gate"),
                false,
            )?;
            let w2 = weights.linear(
                cfg.intermediate_size,
                h,
                &format!("{d}.feed_forward.w2"),
                &format!("{g}.ffn_down"),
                false,
            )?;
            let w3 = weights.linear(
                h,
                cfg.intermediate_size,
                &format!("{d}.feed_forward.w3"),
                &format!("{g}.ffn_up"),
                false,
            )?;
            layers.push(Layer {
                operator,
                norm,
                ffn_norm,
                post_operator,
                post_ffn,
                w1,
                w2,
                w3,
            });
        }
        let norm_name = if gguf_root == "a." {
            "a.post_ln".to_owned()
        } else {
            format!("{gguf_root}output_norm")
        };
        let norm = weights.norm(
            h,
            cfg.norm_eps,
            &format!("{dense_root}.embedding_norm"),
            &norm_name,
        )?;
        Ok(Self {
            layers,
            norm,
            position: 0,
            max_positions: cfg.max_position_embeddings,
            causal,
            cache_enabled,
        })
    }
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn snapshot_cache(&self) -> CacheState {
        CacheState {
            position: self.position,
            layers: self
                .layers
                .iter()
                .map(|l| match &l.operator {
                    Operator::Conv(c) => LayerCache::Conv(c.cache.clone()),
                    Operator::Attention(a) => LayerCache::Attention(a.cache.clone()),
                })
                .collect(),
        }
    }
    pub fn restore_cache(&mut self, state: &CacheState) -> Result<()> {
        if state.position > self.max_positions || state.layers.len() != self.layers.len() {
            candle::bail!("incompatible LFM3 cache checkpoint")
        }
        for (layer, cache) in self.layers.iter().zip(&state.layers) {
            match (&layer.operator, cache) {
                (Operator::Conv(c), LayerCache::Conv(Some(t))) => {
                    let (_, h, w) = t.dims3()?;
                    if h != c.weight.dim(0)? || w + 1 != c.weight.dim(2)? {
                        candle::bail!("incompatible LFM3 convolution cache shape")
                    }
                }
                (Operator::Attention(a), LayerCache::Attention(Some((k, v)))) => {
                    let (_, h, n, d) = k.dims4()?;
                    if k.shape() != v.shape() || h != a.kv_heads || d != a.dim || n > state.position
                    {
                        candle::bail!("incompatible LFM3 attention cache shape")
                    }
                }
                (Operator::Conv(_), LayerCache::Conv(None))
                | (Operator::Attention(_), LayerCache::Attention(None)) => {}
                _ => candle::bail!("incompatible LFM3 cache layer type"),
            }
        }
        self.position = state.position;
        for (layer, cache) in self.layers.iter_mut().zip(&state.layers) {
            match (&mut layer.operator, cache) {
                (Operator::Conv(c), LayerCache::Conv(t)) => c.cache = t.clone(),
                (Operator::Attention(a), LayerCache::Attention(t)) => a.cache = t.clone(),
                _ => unreachable!(),
            }
        }
        Ok(())
    }

    pub fn reset(&mut self) {
        self.position = 0;
        for layer in &mut self.layers {
            match &mut layer.operator {
                Operator::Conv(c) => c.cache = None,
                Operator::Attention(a) => a.cache = None,
            }
        }
    }
    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        if !self.cache_enabled {
            self.reset();
        }
        let time = x.dim(1)?;
        if time == 0
            || self
                .position
                .checked_add(time)
                .is_none_or(|n| n > self.max_positions)
        {
            candle::bail!("LFM3 context limit exceeded or empty sequence")
        }
        let mut y = x.clone();
        for layer in &mut self.layers {
            y = layer.forward(&y, self.position, self.causal, self.cache_enabled)?;
        }
        self.position += time;
        self.norm.forward(&y)
    }
}

#[derive(Clone)]
pub struct Model {
    embedding: Embedding,
    stack: Stack,
    head: Linear,
}
impl Model {
    pub fn from_gguf(
        path: impl AsRef<std::path::Path>,
        cfg: &Config,
        device: &Device,
    ) -> Result<Self> {
        let w = Weights::Gguf(quantized_var_builder::VarBuilder::from_gguf(path, device)?);
        Self::new(cfg, &w, "lfm.model", "")
    }
    pub fn new(cfg: &Config, w: &Weights<'_>, dense_root: &str, gguf_root: &str) -> Result<Self> {
        let embedding = Embedding::new(
            w.dense(
                (cfg.vocab_size, cfg.hidden_size),
                &format!("{dense_root}.embed_tokens.weight"),
                &format!("{gguf_root}token_embd.weight"),
            )?,
            cfg.hidden_size,
        );
        let head = if cfg.tie_word_embeddings {
            match w {
                Weights::Gguf(v) => Linear {
                    weight: QMatMul::from_arc(v.get(
                        (cfg.vocab_size, cfg.hidden_size),
                        &format!("{gguf_root}token_embd.weight"),
                    )?)?,
                    bias: None,
                },
                _ => Linear {
                    weight: QMatMul::Tensor(embedding.embeddings().clone()),
                    bias: None,
                },
            }
        } else {
            w.linear(
                cfg.hidden_size,
                cfg.vocab_size,
                "lfm.lm_head",
                &format!("{gguf_root}output"),
                false,
            )?
        };
        let stack = Stack::new(cfg, w, dense_root, gguf_root, true, true, true)?;
        Ok(Self {
            embedding,
            stack,
            head,
        })
    }
    pub fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        self.embedding.forward(tokens)
    }
    pub fn forward_hidden(&mut self, embeddings: &Tensor) -> Result<Tensor> {
        self.stack.forward(embeddings)
    }
    pub fn project(&self, hidden: &Tensor) -> Result<Tensor> {
        self.head.forward(hidden)
    }
    pub fn forward(&mut self, tokens: &Tensor) -> Result<Tensor> {
        let hidden = self.forward_hidden(&self.embed(tokens)?)?;
        self.project(&hidden.i((.., hidden.dim(1)? - 1.., ..))?)
    }
    pub fn reset(&mut self) {
        self.stack.reset()
    }
    pub fn position(&self) -> usize {
        self.stack.position()
    }
    pub fn snapshot_cache(&self) -> CacheState {
        self.stack.snapshot_cache()
    }
    pub fn restore_cache(&mut self, state: &CacheState) -> Result<()> {
        self.stack.restore_cache(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    fn stack() -> Result<Stack> {
        let cfg:Config=serde_json::from_value(serde_json::json!({"vocab_size":16,"hidden_size":4,"intermediate_size":8,"num_hidden_layers":3,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":2,"conv_L_cache":3,"sliding_window":2,"max_position_embeddings":32,"use_sandwich_norm":true,"layer_types":["conv","full_attention","sliding_attention"],"rope_parameters":{"full_attention":{"rope_theta":1000000.,"partial_rotary_factor":0.5},"sliding_attention":{"rope_theta":10000.}}})).map_err(candle::Error::wrap)?;
        let mut map = HashMap::new();
        let mut add = |name: String, shape: &[usize], norm: bool| -> Result<()> {
            let count = shape.iter().product::<usize>();
            let data = (0..count)
                .map(|i| {
                    if norm {
                        1.
                    } else {
                        ((i as f32 + 1.) * 0.41).sin() * 0.15
                    }
                })
                .collect::<Vec<_>>();
            map.insert(name, Tensor::from_vec(data, shape, &Device::Cpu)?);
            Ok(())
        };
        add("model.embedding_norm.weight".into(), &[4], true)?;
        for i in 0..3 {
            let root = format!("model.layers.{i}");
            for n in [
                "operator_norm",
                "ffn_norm",
                "post_operator_norm",
                "post_ffn_norm",
            ] {
                add(format!("{root}.{n}.weight"), &[4], true)?;
            }
            for (n, shape) in [
                ("feed_forward.w1", vec![8, 4]),
                ("feed_forward.w2", vec![4, 8]),
                ("feed_forward.w3", vec![8, 4]),
            ] {
                add(format!("{root}.{n}.weight"), &shape, false)?;
            }
            if i == 0 {
                add(format!("{root}.conv.in_proj.weight"), &[12, 4], false)?;
                add(format!("{root}.conv.out_proj.weight"), &[4, 4], false)?;
                add(format!("{root}.conv.conv.weight"), &[4, 1, 3], false)?;
            } else {
                for (n, shape) in [
                    ("q_proj", vec![4, 4]),
                    ("k_proj", vec![2, 4]),
                    ("v_proj", vec![2, 4]),
                    ("o_proj", vec![4, 4]),
                    ("gate_proj", vec![2, 4]),
                ] {
                    add(format!("{root}.self_attn.{n}.weight"), &shape, false)?;
                }
                for n in ["q_norm", "k_norm"] {
                    add(format!("{root}.self_attn.{n}.weight"), &[2], true)?;
                }
            }
        }
        Stack::new(
            &cfg,
            &Weights::Dense(VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)),
            "model",
            "",
            true,
            true,
            true,
        )
    }
    #[test]
    fn causal_hybrid_chunking_and_reset_match() -> Result<()> {
        let input = Tensor::from_vec(
            (0..28)
                .map(|i| ((i as f32) * 0.29).cos())
                .collect::<Vec<_>>(),
            (1, 7, 4),
            &Device::Cpu,
        )?;
        let mut whole = stack()?;
        let expected = whole.forward(&input)?;
        let mut streamed = stack()?;
        let mut out = Vec::new();
        for (start, len) in [(0, 2), (2, 1), (3, 4)] {
            out.push(streamed.forward(&input.narrow(1, start, len)?)?);
        }
        let actual = Tensor::cat(&out, 1)?;
        let error = (&expected - actual)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(error < 2e-5, "chunked cache mismatch {error}");
        assert_eq!(streamed.position(), 7);
        streamed.reset();
        let _ = streamed.forward(&input.narrow(1, 0, 2)?)?;
        let checkpoint = streamed.snapshot_cache();
        let continuation = streamed.forward(&input.narrow(1, 2, 5)?)?;
        streamed.restore_cache(&checkpoint)?;
        assert_eq!(streamed.position(), 2);
        let replay = streamed.forward(&input.narrow(1, 2, 5)?)?;
        let checkpoint_error = (&continuation - replay)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(
            checkpoint_error < 2e-5,
            "restored conversation differs {checkpoint_error}"
        );
        streamed.reset();
        let error = (&expected - streamed.forward(&input)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(error < 2e-5, "reset mismatch {error}");
        Ok(())
    }
}
