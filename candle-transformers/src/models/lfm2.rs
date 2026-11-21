//! LFM2 (Liquid Foundation Model 2) implementation.
//!
//! LFM2 is a hybrid architecture that combines attention and short convolution layers.
//! See [LiquidAI](https://www.liquid.ai/) for more information.
//!
//! This implementation supports the `Lfm2ForCausalLM` and `Lfm2MoeForCausalLM`
//! architectures from HuggingFace transformers, which cover the LFM2 and LFM2.5
//! text models.

use crate::models::with_tracing::{linear_no_bias as linear, Embedding, Linear, RmsNorm};
use crate::utils::repeat_kv;
use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::VarBuilder;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    FullAttention,
    Conv,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f32,
}

/// Raw `config.json` for both `lfm2` and `lfm2_moe` checkpoints. Missing fields
/// use the same defaults as `Lfm2Config` / `Lfm2MoeConfig` in transformers.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Lfm2Config {
    pub model_type: Option<String>,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: Option<usize>,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// Newer configs store the rope base here instead of `rope_theta`.
    pub rope_parameters: Option<RopeParameters>,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_conv_l_cache", alias = "conv_L_cache")]
    pub conv_l_cache: usize,
    #[serde(default)]
    pub conv_bias: bool,
    pub layer_types: Option<Vec<LayerType>>,
    /// Older configs list the attention layers instead of `layer_types`.
    pub full_attn_idxs: Option<Vec<usize>>,
    pub tie_embedding: Option<bool>,
    pub tie_word_embeddings: Option<bool>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    // FFN dimension configuration (`lfm2` only)
    pub block_ff_dim: Option<usize>,
    #[serde(default = "default_true")]
    pub block_auto_adjust_ff_dim: bool,
    #[serde(default = "default_ffn_dim_multiplier")]
    pub block_ffn_dim_multiplier: Option<f32>,
    #[serde(default = "default_block_multiple_of")]
    pub block_multiple_of: usize,
    // MoE configuration (`lfm2_moe` only)
    #[serde(default = "default_num_experts")]
    pub num_experts: usize,
    #[serde(default = "default_num_experts_per_tok")]
    pub num_experts_per_tok: usize,
    #[serde(default = "default_moe_intermediate_size")]
    pub moe_intermediate_size: usize,
    #[serde(default = "default_num_dense_layers")]
    pub num_dense_layers: usize,
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
    #[serde(default = "default_true")]
    pub use_expert_bias: bool,
    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f64,
}

fn default_num_key_value_heads() -> usize {
    8
}

fn default_norm_eps() -> f64 {
    1e-5
}

fn default_rope_theta() -> f32 {
    1_000_000.0
}

fn default_max_position_embeddings() -> usize {
    128000
}

fn default_conv_l_cache() -> usize {
    3
}

fn default_ffn_dim_multiplier() -> Option<f32> {
    Some(1.0)
}

fn default_block_multiple_of() -> usize {
    256
}

fn default_true() -> bool {
    true
}

fn default_num_experts() -> usize {
    32
}

fn default_num_experts_per_tok() -> usize {
    4
}

fn default_moe_intermediate_size() -> usize {
    1792
}

fn default_num_dense_layers() -> usize {
    2
}

fn default_routed_scaling_factor() -> f64 {
    1.0
}

impl Lfm2Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn is_moe(&self) -> bool {
        self.model_type.as_deref() == Some("lfm2_moe")
    }

    /// FFN size of the dense layers, following `Lfm2MLP` in transformers.
    fn compute_intermediate_size(&self) -> usize {
        if self.is_moe() {
            return self.intermediate_size.unwrap_or(7168);
        }
        let mut size = self
            .block_ff_dim
            .or(self.intermediate_size)
            .unwrap_or(12288);
        if self.block_auto_adjust_ff_dim {
            size = 2 * size / 3;
            if let Some(multiplier) = self.block_ffn_dim_multiplier {
                size = (multiplier * size as f32) as usize;
                size = size.div_ceil(self.block_multiple_of) * self.block_multiple_of;
            }
        }
        size
    }

    fn compute_layer_types(&self) -> Vec<LayerType> {
        if let Some(layer_types) = &self.layer_types {
            return layer_types.clone();
        }
        (0..self.num_hidden_layers)
            .map(|i| match &self.full_attn_idxs {
                Some(idxs) if !idxs.contains(&i) => LayerType::Conv,
                _ => LayerType::FullAttention,
            })
            .collect()
    }

    pub fn into_config(self, use_flash_attn: bool) -> Config {
        let intermediate_size = self.compute_intermediate_size();
        let layer_types = self.compute_layer_types();
        let rope_theta = self
            .rope_parameters
            .as_ref()
            .map_or(self.rope_theta, |r| r.rope_theta);
        let tie_embedding = self
            .tie_embedding
            .or(self.tie_word_embeddings)
            .unwrap_or(true);
        let moe = self.is_moe().then_some(MoeConfig {
            num_experts: self.num_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            moe_intermediate_size: self.moe_intermediate_size,
            num_dense_layers: self.num_dense_layers,
            norm_topk_prob: self.norm_topk_prob,
            use_expert_bias: self.use_expert_bias,
            routed_scaling_factor: self.routed_scaling_factor,
        });
        Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            norm_eps: self.norm_eps,
            rope_theta,
            max_position_embeddings: self.max_position_embeddings,
            conv_l_cache: self.conv_l_cache,
            conv_bias: self.conv_bias,
            layer_types,
            tie_embedding,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            moe,
            use_flash_attn,
        }
    }
}

/// Sparse MoE settings for `lfm2_moe` models (e.g. LFM2.5-8B-A1B).
#[derive(Debug, Clone)]
pub struct MoeConfig {
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    /// The first `num_dense_layers` layers use a dense MLP.
    pub num_dense_layers: usize,
    pub norm_topk_prob: bool,
    pub use_expert_bias: bool,
    pub routed_scaling_factor: f64,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub conv_l_cache: usize,
    pub conv_bias: bool,
    pub layer_types: Vec<LayerType>,
    pub tie_embedding: bool,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub moe: Option<MoeConfig>,
    pub use_flash_attn: bool,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// Cache for LFM2 model supporting both attention KV cache and convolution state cache.
#[derive(Debug, Clone)]
pub struct Cache {
    masks: HashMap<(usize, usize), Tensor>,
    pub use_kv_cache: bool,
    // KV cache for attention layers: (key, value) per layer
    kvs: Vec<Option<(Tensor, Tensor)>>,
    // Conv state cache for convolution layers
    conv_states: Vec<Option<Tensor>>,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

fn calculate_default_inv_freq(cfg: &Config) -> Vec<f32> {
    let head_dim = cfg.head_dim();
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}

impl Cache {
    pub fn new(use_kv_cache: bool, dtype: DType, config: &Config, device: &Device) -> Result<Self> {
        let theta = calculate_default_inv_freq(config);
        let theta = Tensor::new(theta, device)?;

        let idx_theta = Tensor::arange(0, config.max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((config.max_position_embeddings, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;

        let num_layers = config.num_hidden_layers;
        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kvs: vec![None; num_layers],
            conv_states: vec![None; num_layers],
            device: device.clone(),
            cos,
            sin,
        })
    }

    fn mask(&mut self, seq_len: usize, index_pos: usize) -> Result<Tensor> {
        let kv_len = index_pos + seq_len;
        if let Some(mask) = self.masks.get(&(seq_len, kv_len)) {
            Ok(mask.clone())
        } else {
            let mask = crate::utils::build_causal_mask(seq_len, index_pos, &self.device)?;
            self.masks.insert((seq_len, kv_len), mask.clone());
            Ok(mask)
        }
    }

    pub fn clear(&mut self) {
        self.kvs.iter_mut().for_each(|v| *v = None);
        self.conv_states.iter_mut().for_each(|v| *v = None);
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

/// MLP layer with SwiGLU activation.
#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    span: tracing::Span,
}

impl Mlp {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        // LFM2 uses w1 (gate), w3 (up), w2 (down) naming convention
        let gate_proj = linear(hidden_size, intermediate_size, vb.pp("w1"))?;
        let up_proj = linear(hidden_size, intermediate_size, vb.pp("w3"))?;
        let down_proj = linear(intermediate_size, hidden_size, vb.pp("w2"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

/// Sigmoid-routed top-k MoE, shared with `quantized_lfm2`. `expert_bias` only
/// changes which experts are picked, not their weights.
pub(crate) fn sparse_moe_forward(
    xs: &Tensor,
    gate: &impl Module,
    experts: &[impl Module],
    expert_bias: Option<&[f32]>,
    top_k: usize,
    norm_topk_prob: bool,
    routed_scaling_factor: f64,
) -> Result<Tensor> {
    let (b_sz, seq_len, hidden_dim) = xs.dims3()?;
    let xs = xs.reshape(((), hidden_dim))?;
    let scores = candle_nn::ops::sigmoid(&gate.forward(&xs)?.to_dtype(DType::F32)?)?;

    // Pick the experts on the host and group the tokens by expert.
    let mut rows = vec![vec![]; experts.len()];
    let mut weights = vec![vec![]; experts.len()];
    for (row, s) in scores.to_vec2::<f32>()?.iter().enumerate() {
        let biased = |i: usize| s[i] + expert_bias.map_or(0.0, |b| b[i]);
        let mut ids: Vec<usize> = (0..s.len()).collect();
        ids.sort_by(|&i, &j| biased(j).total_cmp(&biased(i)));
        let ids = &ids[..top_k];
        let mut scale = routed_scaling_factor as f32;
        if norm_topk_prob {
            scale /= ids.iter().map(|&i| s[i]).sum::<f32>() + 1e-6;
        }
        for &i in ids {
            rows[i].push(row as u32);
            weights[i].push(s[i] * scale);
        }
    }

    let mut ys = xs.zeros_like()?;
    for (expert, (rows, weights)) in experts.iter().zip(rows.iter().zip(weights.iter())) {
        if rows.is_empty() {
            continue;
        }
        let rows = Tensor::new(rows.as_slice(), xs.device())?;
        let weights = Tensor::new(weights.as_slice(), xs.device())?
            .reshape(((), 1))?
            .to_dtype(xs.dtype())?;
        let out = expert.forward(&xs.index_select(&rows, 0)?)?;
        ys = ys.index_add(&rows, &out.broadcast_mul(&weights)?, 0)?;
    }
    ys.reshape((b_sz, seq_len, hidden_dim))
}

#[derive(Debug, Clone)]
struct SparseMoe {
    gate: Linear,
    experts: Vec<Mlp>,
    expert_bias: Option<Vec<f32>>,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
    routed_scaling_factor: f64,
    span: tracing::Span,
}

impl SparseMoe {
    fn new(cfg: &Config, moe: &MoeConfig, vb: VarBuilder) -> Result<Self> {
        if moe.num_experts_per_tok == 0 || moe.num_experts_per_tok > moe.num_experts {
            candle::bail!(
                "num_experts_per_tok must be in 1..={}, got {}",
                moe.num_experts,
                moe.num_experts_per_tok
            )
        }
        let gate = linear(cfg.hidden_size, moe.num_experts, vb.pp("gate"))?;
        let vb_e = vb.pp("experts");
        let experts = (0..moe.num_experts)
            .map(|i| Mlp::new(cfg.hidden_size, moe.moe_intermediate_size, vb_e.pp(i)))
            .collect::<Result<Vec<_>>>()?;
        let expert_bias = if moe.use_expert_bias {
            let bias = vb.get_with_hints_dtype(
                moe.num_experts,
                "expert_bias",
                Default::default(),
                DType::F32,
            )?;
            Some(bias.to_vec1::<f32>()?)
        } else {
            None
        };
        Ok(Self {
            gate,
            experts,
            expert_bias,
            num_experts_per_tok: moe.num_experts_per_tok,
            norm_topk_prob: moe.norm_topk_prob,
            routed_scaling_factor: moe.routed_scaling_factor,
            span: tracing::span!(tracing::Level::TRACE, "moe"),
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        sparse_moe_forward(
            xs,
            &self.gate,
            &self.experts,
            self.expert_bias.as_deref(),
            self.num_experts_per_tok,
            self.norm_topk_prob,
            self.routed_scaling_factor,
        )
    }
}

#[derive(Debug, Clone)]
enum FeedForward {
    Dense(Mlp),
    Moe(SparseMoe),
}

impl FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(mlp) => mlp.forward(xs),
            Self::Moe(moe) => moe.forward(xs),
        }
    }
}

/// Attention layer with per-head QK normalization and RoPE.
#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    use_flash_attn: bool,
    span: tracing::Span,
    span_rot: tracing::Span,
}

impl Attention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_attention_heads = cfg.num_attention_heads;
        let num_key_value_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();

        let q_proj = linear(hidden_size, num_attention_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_size, num_key_value_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_size, num_key_value_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear(
            num_attention_heads * head_dim,
            hidden_size,
            vb.pp("out_proj"),
        )?;

        let q_norm = RmsNorm::new(head_dim, cfg.norm_eps, vb.pp("q_layernorm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.norm_eps, vb.pp("k_layernorm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            use_flash_attn: cfg.use_flash_attn,
            span: tracing::span!(tracing::Level::TRACE, "attn"),
            span_rot: tracing::span!(tracing::Level::TRACE, "attn-rot"),
        })
    }

    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize, cache: &Cache) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_, _, seq_len, _) = x.dims4()?;
        let cos = cache.cos.narrow(0, index_pos, seq_len)?;
        let sin = cache.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to (batch, seq, num_heads, head_dim) then transpose to (batch, num_heads, seq, head_dim)
        let q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Apply per-head QK normalization
        let q = self.q_norm.forward(&q.contiguous()?)?;
        let k = self.k_norm.forward(&k.contiguous()?)?;

        // Apply rotary embeddings
        let q = self.apply_rotary_emb(&q, index_pos, cache)?;
        let k = self.apply_rotary_emb(&k, index_pos, cache)?;

        // Handle KV cache
        let (k, v) = if cache.use_kv_cache {
            match &cache.kvs[block_idx] {
                Some((k_cache, v_cache)) if index_pos > 0 => {
                    let k = Tensor::cat(&[k_cache, &k], 2)?.contiguous()?;
                    let v = Tensor::cat(&[v_cache, &v], 2)?.contiguous()?;
                    (k, v)
                }
                _ => (k, v),
            }
        } else {
            (k, v)
        };

        if cache.use_kv_cache {
            cache.kvs[block_idx] = Some((k.clone(), v.clone()));
        }

        // Expand KV heads to match query heads
        let k = repeat_kv(k, self.num_attention_heads / self.num_key_value_heads)?;
        let v = repeat_kv(v, self.num_attention_heads / self.num_key_value_heads)?;

        let y = if self.use_flash_attn {
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
            flash_attn(&q, &k, &v, softmax_scale, seq_len > 1)?.transpose(1, 2)?
        } else {
            let in_dtype = q.dtype();
            let q = q.to_dtype(DType::F32)?;
            let k = k.to_dtype(DType::F32)?;
            let v = v.to_dtype(DType::F32)?;
            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = if seq_len == 1 {
                att
            } else {
                let mask = cache.mask(seq_len, index_pos)?.broadcast_as(att.shape())?;
                masked_fill(&att, &mask, f32::NEG_INFINITY)?
            };
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?
        };

        let y = y.transpose(1, 2)?.reshape((
            b_sz,
            seq_len,
            self.num_attention_heads * self.head_dim,
        ))?;
        self.o_proj.forward(&y)
    }
}

/// Causal depthwise conv, `xs` is (batch, channels, seq_len) and `weight` is
/// (channels, kernel_size). Much faster than a grouped `Conv1d`, which candle
/// runs one channel at a time.
pub(crate) fn causal_conv1d(xs: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let (_, _, seq_len) = xs.dims3()?;
    let kernel_size = weight.dim(1)?;
    if kernel_size == 0 {
        candle::bail!("conv kernel size must be at least 1")
    }
    // Accumulate in f32 like the conv kernels do.
    let dtype = xs.dtype();
    let weight = weight.to_dtype(DType::F32)?;
    let xs = xs
        .to_dtype(DType::F32)?
        .pad_with_zeros(2, kernel_size - 1, 0)?;
    let tap = |k: usize| {
        xs.narrow(2, k, seq_len)?
            .broadcast_mul(&weight.narrow(1, k, 1)?.unsqueeze(0)?)
    };
    (1..kernel_size)
        .try_fold(tap(0)?, |ys, k| ys + tap(k)?)?
        .to_dtype(dtype)
}

/// Short convolution layer for efficient sequence processing.
#[derive(Debug, Clone)]
struct ShortConv {
    in_proj: Linear,
    out_proj: Linear,
    conv_weight: Tensor,
    l_cache: usize,
    hidden_size: usize,
    span: tracing::Span,
}

impl ShortConv {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let l_cache = cfg.conv_l_cache;

        // in_proj projects to 3 * hidden_size for B, C, X components
        let in_proj = linear(hidden_size, 3 * hidden_size, vb.pp("in_proj"))?;
        let out_proj = linear(hidden_size, hidden_size, vb.pp("out_proj"))?;

        // Conv weight shape: (hidden_size, 1, l_cache) or (hidden_size, l_cache)
        let conv_weight = vb.get((hidden_size, 1, l_cache), "conv.weight")?;

        Ok(Self {
            in_proj,
            out_proj,
            conv_weight,
            l_cache,
            hidden_size,
            span: tracing::span!(tracing::Level::TRACE, "shortconv"),
        })
    }

    fn forward(&self, x: &Tensor, block_idx: usize, cache: &mut Cache) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b_sz, seq_len, _) = x.dims3()?;

        // Project input to B, C, X components
        let bcx = self.in_proj.forward(x)?.transpose(1, 2)?;
        let b = bcx.narrow(1, 0, self.hidden_size)?;
        let c = bcx.narrow(1, self.hidden_size, self.hidden_size)?;
        let x_proj = bcx.narrow(1, 2 * self.hidden_size, self.hidden_size)?;

        // Element-wise multiply B and X
        let bx = (b * &x_proj)?.contiguous()?;

        let conv_weight = self.conv_weight.squeeze(1)?;

        let conv_out = if seq_len == 1 {
            // Token-by-token generation: use cached state
            let mut state = match &cache.conv_states[block_idx] {
                Some(s) => s.clone(),
                None => Tensor::zeros(
                    (b_sz, self.hidden_size, self.l_cache),
                    bx.dtype(),
                    bx.device(),
                )?,
            };

            // Shift cache and add new token
            if self.l_cache > 1 {
                let tail = state.narrow(2, 1, self.l_cache - 1)?;
                state = Tensor::cat(&[tail, bx.clone()], 2)?;
            } else {
                state = bx.clone();
            }

            if cache.use_kv_cache {
                cache.conv_states[block_idx] = Some(state.clone());
            }

            // Apply convolution as element-wise multiply and sum
            (state * conv_weight.unsqueeze(0)?)?
                .sum_keepdim(2)?
                .contiguous()?
        } else {
            let out = causal_conv1d(&bx, &conv_weight)?;

            // Update cache with last l_cache tokens
            if cache.use_kv_cache && self.l_cache > 0 {
                let start = seq_len.saturating_sub(self.l_cache);
                let cache_len = seq_len - start;
                let mut cache_src = bx.narrow(2, start, cache_len)?;
                if cache_len < self.l_cache {
                    let pad = self.l_cache - cache_len;
                    let zeros = Tensor::zeros(
                        (b_sz, self.hidden_size, pad),
                        cache_src.dtype(),
                        cache_src.device(),
                    )?;
                    cache_src = Tensor::cat(&[zeros, cache_src], 2)?;
                }
                cache.conv_states[block_idx] = Some(cache_src);
            }

            out
        };

        // Multiply by C and project output
        let conv_out = (c * &conv_out)?;
        let conv_out = conv_out.transpose(1, 2)?.contiguous()?;
        self.out_proj.forward(&conv_out)
    }
}

/// Unified decoder layer supporting both attention and convolution.
#[derive(Debug, Clone)]
enum LayerKind {
    Attention(Box<Attention>),
    ShortConv(ShortConv),
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    feed_forward: FeedForward,
    kind: LayerKind,
    span: tracing::Span,
}

impl DecoderLayer {
    fn new(cfg: &Config, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        // LFM2 uses operator_norm and ffn_norm naming
        let input_layernorm = RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb.pp("operator_norm"))?;
        let post_attention_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb.pp("ffn_norm"))?;
        let vb_ff = vb.pp("feed_forward");
        let feed_forward = match &cfg.moe {
            Some(moe) if layer_idx >= moe.num_dense_layers => {
                FeedForward::Moe(SparseMoe::new(cfg, moe, vb_ff)?)
            }
            _ => FeedForward::Dense(Mlp::new(cfg.hidden_size, cfg.intermediate_size, vb_ff)?),
        };

        let layer_type = cfg
            .layer_types
            .get(layer_idx)
            .copied()
            .unwrap_or(LayerType::FullAttention);
        let kind = match layer_type {
            LayerType::FullAttention => {
                LayerKind::Attention(Box::new(Attention::new(cfg, vb.pp("self_attn"))?))
            }
            LayerType::Conv => LayerKind::ShortConv(ShortConv::new(cfg, vb.pp("conv"))?),
        };

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            feed_forward,
            kind,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let residual = x;
        let x = self.input_layernorm.forward(x)?;

        let x = match &self.kind {
            LayerKind::Attention(attn) => attn.forward(&x, index_pos, block_idx, cache)?,
            LayerKind::ShortConv(conv) => conv.forward(&x, block_idx, cache)?,
        };

        let x = (x + residual)?;
        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.feed_forward.forward(&x)?;
        x + residual
    }
}

/// LFM2 model for causal language modeling.
#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    embedding_norm: RmsNorm,
    lm_head: Linear,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");

        let embed_tokens =
            Embedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(cfg, layer_idx, vb_l.pp(layer_idx))?;
            layers.push(layer);
        }

        let embedding_norm =
            RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb_m.pp("embedding_norm"))?;

        let lm_head = if cfg.tie_embedding {
            Linear::from_weights(embed_tokens.embeddings().clone(), None)
        } else {
            linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            embed_tokens,
            layers,
            embedding_norm,
            lm_head,
            dtype: vb.dtype(),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        for (block_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(&hidden_states, index_pos, block_idx, cache)?;
        }

        let hidden_states = self.embedding_norm.forward(&hidden_states)?;
        let hidden_states = hidden_states.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&hidden_states)?;
        logits.to_dtype(DType::F32)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::{Conv1d, Conv1dConfig};

    #[test]
    fn derived_config() -> Result<()> {
        use LayerType::{Conv, FullAttention as Attn};
        let base = r#""vocab_size": 64, "hidden_size": 8, "num_hidden_layers": 2, "num_attention_heads": 2"#;
        // (config keys, intermediate size, rope theta, layer types, is moe)
        let cases = [
            // LFM2.5-230M: FFN size used as is.
            (
                r#""model_type": "lfm2", "intermediate_size": 2560, "block_ff_dim": 2560,
                "block_auto_adjust_ff_dim": false, "layer_types": ["conv", "full_attention"],
                "rope_parameters": {"rope_theta": 1000000.0}, "tie_embedding": true"#,
                2560,
                1e6,
                [Conv, Attn],
                false,
            ),
            // LFM2.5-350M: 2/3 of `block_ff_dim`, rounded up to 256.
            (
                r#""model_type": "lfm2", "intermediate_size": 6656, "block_ff_dim": 6656,
                "layer_types": ["full_attention", "conv"], "rope_theta": 1000000.0"#,
                4608,
                1e6,
                [Attn, Conv],
                false,
            ),
            // LFM2.5-2.6B: rope base only in `rope_parameters`.
            (
                r#""model_type": "lfm2", "intermediate_size": 10752,
                "block_auto_adjust_ff_dim": false, "layer_types": ["conv", "conv"],
                "rope_parameters": {"rope_theta": 10000000.0}, "tie_word_embeddings": true"#,
                10752,
                1e7,
                [Conv, Conv],
                false,
            ),
            // LFM2-1.2B (v1): `block_ff_dim` and `full_attn_idxs` only.
            (
                r#""model_type": "lfm2", "block_ff_dim": 12288, "full_attn_idxs": [1]"#,
                8192,
                1e6,
                [Conv, Attn],
                false,
            ),
            // LFM2.5-8B-A1B: MoE configs use `intermediate_size` as is.
            (
                r#""model_type": "lfm2_moe", "intermediate_size": 7168,
                "layer_types": ["conv", "full_attention"],
                "rope_parameters": {"rope_theta": 5000000}, "num_experts": 32,
                "num_experts_per_tok": 4, "moe_intermediate_size": 1792, "num_dense_layers": 2"#,
                7168,
                5e6,
                [Conv, Attn],
                true,
            ),
        ];
        for (keys, intermediate_size, rope_theta, layer_types, is_moe) in cases {
            let cfg: Lfm2Config = serde_json::from_str(&format!("{{{base}, {keys}}}"))
                .map_err(candle::Error::wrap)?;
            let cfg = cfg.into_config(false);
            assert_eq!(cfg.intermediate_size, intermediate_size, "{keys}");
            assert_eq!(cfg.rope_theta, rope_theta, "{keys}");
            assert_eq!(cfg.layer_types, layer_types, "{keys}");
            assert_eq!(cfg.moe.is_some(), is_moe, "{keys}");
            assert!(cfg.tie_embedding, "{keys}");
        }
        Ok(())
    }

    #[test]
    fn causal_conv1d_matches_grouped_conv() -> Result<()> {
        let (channels, kernel_size) = (4, 3);
        let xs = Tensor::randn(0f32, 1., (2, channels, 5), &Device::Cpu)?;
        let weight = Tensor::randn(0f32, 1., (channels, kernel_size), &Device::Cpu)?;
        let conv = Conv1d::new(
            weight.reshape((channels, 1, kernel_size))?,
            None,
            Conv1dConfig {
                padding: kernel_size - 1,
                groups: channels,
                ..Default::default()
            },
        );
        let expected = conv.forward(&xs)?.narrow(2, 0, 5)?;
        let diff = (causal_conv1d(&xs, &weight)? - expected)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-5, "max diff {diff}");
        Ok(())
    }

    #[test]
    fn expert_bias_only_changes_the_selection() -> Result<()> {
        // One token, three experts: expert `e` scales its input by `e + 1` and
        // the router scores are sigmoid(2), sigmoid(1), sigmoid(0).
        let device = Device::Cpu;
        let xs = Tensor::new(&[[[1f32]]], &device)?;
        let gate = candle_nn::Linear::new(Tensor::new(&[[2f32], [1.], [0.]], &device)?, None);
        let experts: Vec<_> = (0..3)
            .map(|e| candle_nn::func(move |xs: &Tensor| xs.affine((e + 1) as f64, 0.)))
            .collect();
        let sigmoid = |x: f32| 1. / (1. + (-x).exp());
        let (s0, s1, s2) = (sigmoid(2.), sigmoid(1.), sigmoid(0.));
        // (expert bias, top k, normalize, expected output)
        let cases = [
            (None, 1, false, s0),
            // The bias picks expert 2, its weight is still the unbiased score.
            (Some([0., 0., 1.]), 1, false, 3. * s2),
            (None, 2, true, (s0 + 2. * s1) / (s0 + s1 + 1e-6)),
        ];
        for (bias, top_k, norm, expected) in cases {
            let ys = sparse_moe_forward(
                &xs,
                &gate,
                &experts,
                bias.as_ref().map(|b| &b[..]),
                top_k,
                norm,
                1.0,
            )?;
            let ys = ys.flatten_all()?.to_vec1::<f32>()?[0];
            assert!(
                (ys - expected).abs() < 1e-5,
                "{bias:?} {top_k} {norm}: {ys} vs {expected}"
            );
        }
        Ok(())
    }
}
