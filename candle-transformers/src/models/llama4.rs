//! Llama 4 (text-only decoder) inference implementation.
//!
//! Implements the Scout/Maverick MoE architecture: alternating NoPE/RoPE
//! attention layers (iRoPE), parameterless QK-norm, attention temperature
//! tuning on NoPE layers, and a top-k sigmoid-gated MoE with an always-on
//! shared expert.
//!
//! Reference: <https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/modeling_llama4.py>

use super::with_tracing::{linear_b, linear_no_bias, Linear, RmsNorm};
use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{kv_cache::ConcatKvCache, Activation, VarBuilder};

fn default_rope_theta() -> f64 {
    500_000.0
}

fn default_rms_norm_eps() -> f64 {
    1e-5
}

fn default_true() -> bool {
    true
}

fn default_attn_scale() -> f64 {
    0.1
}

fn default_floor_scale() -> f64 {
    8192.0
}

fn default_interleave_moe_layer_step() -> usize {
    1
}

fn default_no_rope_layer_interval() -> usize {
    4
}

fn default_hidden_act() -> Activation {
    Activation::Silu
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(untagged)]
pub enum EosToks {
    Single(u32),
    Multiple(Vec<u32>),
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    #[serde(default)]
    pub intermediate_size_mlp: Option<usize>,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub max_position_embeddings: usize,
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    #[serde(default = "default_interleave_moe_layer_step")]
    pub interleave_moe_layer_step: usize,
    #[serde(default = "default_true")]
    pub use_qk_norm: bool,
    #[serde(default = "default_no_rope_layer_interval")]
    pub no_rope_layer_interval: usize,
    #[serde(default)]
    pub no_rope_layers: Option<Vec<usize>>,
    #[serde(default)]
    pub attn_temperature_tuning: bool,
    #[serde(default = "default_attn_scale")]
    pub attn_scale: f64,
    #[serde(default = "default_floor_scale")]
    pub floor_scale: f64,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: Activation,
    #[serde(default)]
    pub eos_token_id: Option<EosToks>,
}

impl Config {
    fn intermediate_size_mlp(&self) -> usize {
        self.intermediate_size_mlp.unwrap_or(self.intermediate_size)
    }

    /// Whether layer `layer_idx` (0-indexed) gets RoPE applied. Layers where this is
    /// false use no positional embedding at all (NoPE), per the iRoPE design.
    fn use_rope(&self, layer_idx: usize) -> bool {
        match &self.no_rope_layers {
            Some(v) => v.get(layer_idx).copied().unwrap_or(1) != 0,
            None => !(layer_idx + 1).is_multiple_of(self.no_rope_layer_interval),
        }
    }

    /// Whether layer `layer_idx` (0-indexed) is a sparse MoE layer, as opposed to a
    /// dense MLP layer.
    fn is_moe_layer(&self, layer_idx: usize) -> bool {
        self.interleave_moe_layer_step != 0
            && (layer_idx + 1).is_multiple_of(self.interleave_moe_layer_step)
    }
}

/// Parameterless L2 normalization, used for QK-norm.
#[derive(Debug, Clone)]
struct L2Norm {
    eps: f64,
}

impl L2Norm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x32 = x.to_dtype(DType::F32)?;
        let norm = (x32.sqr()?.mean_keepdim(D::Minus1)? + self.eps)?.sqrt()?;
        x32.broadcast_div(&norm)?.to_dtype(dtype)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(cfg: &Config, dtype: DType, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / (cfg.rope_theta as f32).powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
        })
    }

    /// Apply the interleaved-pair (complex/polar style) RoPE used by Llama 4.
    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope_i(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope_i(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Mlp {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        act: Activation,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?,
            act_fn: act,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let lhs = x.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = x.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct MoeBlock {
    router: Linear,
    experts: Vec<Mlp>,
    shared_expert: Mlp,
    num_experts_per_tok: usize,
}

impl MoeBlock {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let router = linear_no_bias(cfg.hidden_size, cfg.num_local_experts, vb.pp("router"))?;
        let mut experts = Vec::with_capacity(cfg.num_local_experts);
        let vb_e = vb.pp("experts");
        for idx in 0..cfg.num_local_experts {
            experts.push(Mlp::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                cfg.hidden_act,
                vb_e.pp(idx),
            )?);
        }
        let shared_expert = Mlp::new(
            cfg.hidden_size,
            cfg.intermediate_size_mlp(),
            cfg.hidden_act,
            vb.pp("shared_expert"),
        )?;
        Ok(Self {
            router,
            experts,
            shared_expert,
            num_experts_per_tok: cfg.num_experts_per_tok,
        })
    }
}

impl Module for MoeBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs = xs.reshape(((), hidden_dim))?;

        let shared = self.shared_expert.forward(&xs)?;

        let router_logits = xs.apply(&self.router)?;
        let top_idxs = router_logits
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.num_experts_per_tok)?
            .contiguous()?;
        let top_logits = router_logits.gather(&top_idxs, D::Minus1)?;
        // Llama 4 gates each selected expert independently with a sigmoid of its own
        // logit (not a softmax over the top-k group), so per-entry weights don't need
        // to be renormalized against each other.
        let routing_weights = candle_nn::ops::sigmoid(&top_logits.to_dtype(DType::F32)?)?;
        let routing_weights = routing_weights.to_vec2::<f32>()?;
        let top_idxs = top_idxs.to_vec2::<u32>()?;

        let mut top_x = vec![vec![]; self.experts.len()];
        let mut selected_weights = vec![vec![]; self.experts.len()];
        for (row_idx, (rw, expert_idxs)) in routing_weights.iter().zip(top_idxs.iter()).enumerate()
        {
            for (&rw, &expert_idx) in rw.iter().zip(expert_idxs.iter()) {
                top_x[expert_idx as usize].push(row_idx as u32);
                selected_weights[expert_idx as usize].push(rw);
            }
        }

        let mut ys = xs.zeros_like()?;
        for (expert_idx, expert_layer) in self.experts.iter().enumerate() {
            let rows = &top_x[expert_idx];
            if rows.is_empty() {
                continue;
            }
            let rows = Tensor::new(rows.as_slice(), xs.device())?;
            let weights = Tensor::new(selected_weights[expert_idx].as_slice(), xs.device())?
                .reshape(((), 1))?
                .to_dtype(xs.dtype())?;
            let current_state = xs.index_select(&rows, 0)?.reshape(((), hidden_dim))?;
            let current_hidden_states = expert_layer.forward(&current_state)?;
            let current_hidden_states = current_hidden_states.broadcast_mul(&weights)?;
            ys = ys.index_add(&rows, &current_hidden_states, 0)?;
        }

        (ys + shared)?.reshape((b_size, seq_len, hidden_dim))
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Option<L2Norm>,
    k_norm: Option<L2Norm>,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    use_rope: bool,
    attn_temperature_tuning: bool,
    attn_scale: f64,
    floor_scale: f64,
    rotary_emb: std::sync::Arc<RotaryEmbedding>,
    kv_cache: ConcatKvCache,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        cfg: &Config,
        layer_idx: usize,
        rotary_emb: std::sync::Arc<RotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let hidden_size = head_dim * num_heads;

        let q_proj = linear_b(
            cfg.hidden_size,
            num_heads * head_dim,
            cfg.attention_bias,
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_b(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            cfg.attention_bias,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_b(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            cfg.attention_bias,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_b(
            num_heads * head_dim,
            cfg.hidden_size,
            cfg.attention_bias,
            vb.pp("o_proj"),
        )?;

        let (q_norm, k_norm) = if cfg.use_qk_norm && cfg.use_rope(layer_idx) {
            (Some(L2Norm { eps: 1e-6 }), Some(L2Norm { eps: 1e-6 }))
        } else {
            (None, None)
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            hidden_size,
            use_rope: cfg.use_rope(layer_idx),
            attn_temperature_tuning: cfg.attn_temperature_tuning,
            attn_scale: cfg.attn_scale,
            floor_scale: cfg.floor_scale,
            rotary_emb,
            kv_cache: ConcatKvCache::new(2),
        })
    }

    fn temperature_tune(&self, q: &Tensor, offset: usize, seq_len: usize) -> Result<Tensor> {
        let scales: Vec<f32> = (0..seq_len)
            .map(|i| {
                let pos = (offset + i) as f64;
                let scaled =
                    (((pos + 1.0) / self.floor_scale).floor() + 1.0).ln() * self.attn_scale + 1.0;
                scaled as f32
            })
            .collect();
        let scales =
            Tensor::from_vec(scales, (1, 1, seq_len, 1), q.device())?.to_dtype(q.dtype())?;
        q.broadcast_mul(&scales)
    }

    fn forward(&mut self, x: &Tensor, attn_mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let mut q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        if let (Some(q_norm), Some(k_norm)) = (&self.q_norm, &self.k_norm) {
            q = q_norm.forward(&q)?;
            k = k_norm.forward(&k)?;
        }

        if self.use_rope {
            let (q_r, k_r) = self.rotary_emb.apply(&q, &k, offset)?;
            q = q_r;
            k = k_r;
        } else if self.attn_temperature_tuning {
            q = self.temperature_tune(&q, offset, l)?;
        }

        let (k, v) = self.kv_cache.append(&k, &v)?;

        let k = crate::utils::repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = crate::utils::repeat_kv(v, self.num_kv_groups)?.contiguous()?;

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

    fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

#[derive(Debug, Clone)]
enum FeedForward {
    Dense(Mlp),
    Moe(MoeBlock),
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(m) => m.forward(xs),
            Self::Moe(m) => m.forward(xs),
        }
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    feed_forward: FeedForward,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(
        cfg: &Config,
        layer_idx: usize,
        rotary_emb: std::sync::Arc<RotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn = Attention::new(cfg, layer_idx, rotary_emb, vb.pp("self_attn"))?;
        let feed_forward = if cfg.is_moe_layer(layer_idx) {
            FeedForward::Moe(MoeBlock::new(cfg, vb.pp("feed_forward"))?)
        } else {
            FeedForward::Dense(Mlp::new(
                cfg.hidden_size,
                cfg.intermediate_size_mlp(),
                cfg.hidden_act,
                vb.pp("feed_forward"),
            )?)
        };
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            feed_forward,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let h = self.input_layernorm.forward(x)?;
        let h = self.self_attn.forward(&h, mask, offset)?;
        let x = (x + h)?;
        let h2 = self.post_attention_layernorm.forward(&x)?;
        let h2 = self.feed_forward.forward(&h2)?;
        x + h2
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
        let rotary_emb = std::sync::Arc::new(RotaryEmbedding::new(cfg, vb.dtype(), vb.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("model.layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, i, rotary_emb.clone(), vb_l.pp(i))?);
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

    fn causal_mask(&self, b: usize, tgt: usize, offset: usize) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| (0..(tgt + offset)).map(move |j| if j <= i + offset { 0. } else { minf }))
            .collect();
        Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?.to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;

        let causal = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset)?)
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle::DType;
    use std::collections::HashMap;

    fn test_config() -> Config {
        Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 12,
            intermediate_size_mlp: Some(20),
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            rope_theta: 500_000.0,
            rms_norm_eps: 1e-5,
            attention_bias: false,
            tie_word_embeddings: false,
            max_position_embeddings: 64,
            num_local_experts: 4,
            num_experts_per_tok: 2,
            interleave_moe_layer_step: 1,
            use_qk_norm: true,
            no_rope_layer_interval: 4,
            no_rope_layers: None,
            attn_temperature_tuning: true,
            attn_scale: 0.1,
            floor_scale: 8192.0,
            hidden_act: Activation::Silu,
            eos_token_id: None,
        }
    }

    fn var(shape: &[usize], dev: &Device) -> Tensor {
        let elems: usize = shape.iter().product();
        let data: Vec<f32> = (0..elems).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
        Tensor::from_vec(data, shape, dev).unwrap()
    }

    fn build_weights(cfg: &Config, dev: &Device) -> HashMap<String, Tensor> {
        let mut w = HashMap::new();
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = cfg.head_dim;
        w.insert(
            "model.embed_tokens.weight".to_string(),
            var(&[cfg.vocab_size, h], dev),
        );
        w.insert("model.norm.weight".to_string(), var(&[h], dev));
        w.insert("lm_head.weight".to_string(), var(&[cfg.vocab_size, h], dev));
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            w.insert(format!("{p}.input_layernorm.weight"), var(&[h], dev));
            w.insert(
                format!("{p}.post_attention_layernorm.weight"),
                var(&[h], dev),
            );
            w.insert(
                format!("{p}.self_attn.q_proj.weight"),
                var(&[nh * hd, h], dev),
            );
            w.insert(
                format!("{p}.self_attn.k_proj.weight"),
                var(&[nkv * hd, h], dev),
            );
            w.insert(
                format!("{p}.self_attn.v_proj.weight"),
                var(&[nkv * hd, h], dev),
            );
            w.insert(
                format!("{p}.self_attn.o_proj.weight"),
                var(&[h, nh * hd], dev),
            );
            if cfg.is_moe_layer(i) {
                w.insert(
                    format!("{p}.feed_forward.router.weight"),
                    var(&[cfg.num_local_experts, h], dev),
                );
                for e in 0..cfg.num_local_experts {
                    let ep = format!("{p}.feed_forward.experts.{e}");
                    w.insert(
                        format!("{ep}.gate_proj.weight"),
                        var(&[cfg.intermediate_size, h], dev),
                    );
                    w.insert(
                        format!("{ep}.up_proj.weight"),
                        var(&[cfg.intermediate_size, h], dev),
                    );
                    w.insert(
                        format!("{ep}.down_proj.weight"),
                        var(&[h, cfg.intermediate_size], dev),
                    );
                }
                let sp = format!("{p}.feed_forward.shared_expert");
                let im = cfg.intermediate_size_mlp();
                w.insert(format!("{sp}.gate_proj.weight"), var(&[im, h], dev));
                w.insert(format!("{sp}.up_proj.weight"), var(&[im, h], dev));
                w.insert(format!("{sp}.down_proj.weight"), var(&[h, im], dev));
            } else {
                let mp = format!("{p}.feed_forward");
                let im = cfg.intermediate_size_mlp();
                w.insert(format!("{mp}.gate_proj.weight"), var(&[im, h], dev));
                w.insert(format!("{mp}.up_proj.weight"), var(&[im, h], dev));
                w.insert(format!("{mp}.down_proj.weight"), var(&[h, im], dev));
            }
        }
        w
    }

    #[test]
    fn forward_smoke_test() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = test_config();
        let weights = build_weights(&cfg, &dev);
        let vb = VarBuilder::from_tensors(weights, DType::F32, &dev);
        let mut model = ModelForCausalLM::new(&cfg, vb)?;

        let input = Tensor::new(&[1u32, 2, 3, 4, 5], &dev)?.unsqueeze(0)?;
        let logits = model.forward(&input, 0)?;
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
        let logits = logits.flatten_all()?.to_vec1::<f32>()?;
        assert!(logits.iter().all(|v| v.is_finite()));

        let next = Tensor::new(&[6u32], &dev)?.unsqueeze(0)?;
        let logits = model.forward(&next, 5)?;
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
        let logits = logits.flatten_all()?.to_vec1::<f32>()?;
        assert!(logits.iter().all(|v| v.is_finite()));

        model.clear_kv_cache();
        Ok(())
    }
}
