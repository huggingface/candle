#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! DeepSeek-V3.
//!
//! DeepSeek-V3 reuses the Multi-head Latent Attention (MLA) mechanism and the
//! YaRN-scaled rotary embeddings introduced by DeepSeek-V2 (see [`crate::models::deepseek2`]),
//! so this module pulls those pieces in directly rather than duplicating them. The
//! parts that actually differ are the MoE gate:
//!   - the gating score is a sigmoid (not a softmax) of the router logits,
//!   - expert/group selection is "auxiliary-loss-free": a learned per-expert bias
//!     (`e_score_correction_bias`) is added to the scores before the top-k/group
//!     selection is performed, but the *weights* used to combine expert outputs
//!     come from the un-biased sigmoid scores,
//!   - groups are scored using the sum of their two highest-scoring experts rather
//!     than the single highest one,
//!   - the `routed_scaling_factor` is always applied to the selected weights,
//!     whether or not they get re-normalized.
//!
//! Multi-token prediction (MTP) and FP8 quantization, which DeepSeek-V3 also
//! introduces, are not implemented here as they are training-time / storage-format
//! concerns that do not change the (dequantized) forward pass used for inference.

use std::sync::Arc;

use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{embedding, rms_norm, Activation, Embedding, Linear, Module, RmsNorm, VarBuilder};
use serde::Deserialize;

use crate::models::deepseek2::{
    masked_fill, Attention, BincountOp, DeepSeekV2Config, DeepSeekV2RopeConfig,
    DeepSeekV2RopeScaling, DeepSeekV2RotaryEmbedding, Mlp, NonZeroOp, ScoringFunc, TopKLastDimOp,
    TopKOutput, TopkMethod,
};
use crate::serde_default_fn;

serde_default_fn!(f64, routed_scaling_factor, 1.0);
serde_default_fn!(usize, moe_layer_freq, 1);
serde_default_fn!(usize, first_k_dense_replace, 0);
serde_default_fn!(bool, norm_topk_prob, false);
serde_default_fn!(Activation, hidden_act, Activation::Silu);
serde_default_fn!(bool, tie_word_embeddings, false);
serde_default_fn!(TopkMethod3, topk_method, TopkMethod3::NoAuxTc);
serde_default_fn!(ScoringFunc3, scoring_func, ScoringFunc3::Sigmoid);

#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TopkMethod3 {
    #[serde(rename = "noaux_tc")]
    NoAuxTc,
    #[serde(rename = "greedy")]
    Greedy,
    #[serde(rename = "group_limited_greedy")]
    GroupLimitedGreedy,
}

#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScoringFunc3 {
    #[serde(rename = "sigmoid")]
    Sigmoid,
    #[serde(rename = "softmax")]
    Softmax,
}

#[derive(Deserialize, Clone, Debug)]
pub struct DeepSeekV3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub n_shared_experts: Option<usize>,
    pub n_routed_experts: Option<usize>,
    #[serde(default = "routed_scaling_factor")]
    pub routed_scaling_factor: f64,
    #[serde(default = "topk_method")]
    pub(crate) topk_method: TopkMethod3,
    pub num_experts_per_tok: Option<usize>,
    #[serde(default = "moe_layer_freq")]
    pub moe_layer_freq: usize,
    #[serde(default = "first_k_dense_replace")]
    pub first_k_dense_replace: usize,
    #[serde(default = "norm_topk_prob")]
    pub norm_topk_prob: bool,
    #[serde(default = "scoring_func")]
    pub(crate) scoring_func: ScoringFunc3,
    #[serde(default = "hidden_act")]
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    #[serde(default = "tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    pub rope_theta: f32,
    pub rope_scaling: Option<DeepSeekV2RopeScaling>,
    pub attention_bias: bool,
    pub q_lora_rank: Option<usize>,
    pub qk_rope_head_dim: usize,
    pub kv_lora_rank: usize,
    pub v_head_dim: usize,
    pub qk_nope_head_dim: usize,
    pub n_group: usize,
    pub topk_group: usize,
}

/// The attention block and the dense MLP are unchanged from DeepSeek-V2, so we
/// build the (private) `DeepSeekV2Config` they expect out of the fields they
/// actually use. The MoE-gate-only fields are filled in with inert placeholders
/// since `Attention`/`Mlp` never read them.
impl From<&DeepSeekV3Config> for DeepSeekV2Config {
    fn from(cfg: &DeepSeekV3Config) -> Self {
        DeepSeekV2Config {
            vocab_size: cfg.vocab_size,
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            moe_intermediate_size: cfg.moe_intermediate_size,
            num_hidden_layers: cfg.num_hidden_layers,
            num_attention_heads: cfg.num_attention_heads,
            n_shared_experts: None,
            n_routed_experts: None,
            routed_scaling_factor: cfg.routed_scaling_factor,
            topk_method: TopkMethod::Greedy,
            num_experts_per_tok: cfg.num_experts_per_tok,
            moe_layer_freq: cfg.moe_layer_freq,
            first_k_dense_replace: cfg.first_k_dense_replace,
            norm_topk_prob: cfg.norm_topk_prob,
            scoring_func: ScoringFunc::Softmax,
            hidden_act: cfg.hidden_act,
            max_position_embeddings: cfg.max_position_embeddings,
            rms_norm_eps: cfg.rms_norm_eps,
            tie_word_embeddings: cfg.tie_word_embeddings,
            rope_theta: cfg.rope_theta,
            rope_scaling: cfg.rope_scaling.clone(),
            attention_bias: cfg.attention_bias,
            q_lora_rank: cfg.q_lora_rank,
            qk_rope_head_dim: cfg.qk_rope_head_dim,
            kv_lora_rank: cfg.kv_lora_rank,
            v_head_dim: cfg.v_head_dim,
            qk_nope_head_dim: cfg.qk_nope_head_dim,
            n_group: cfg.n_group,
            topk_group: cfg.topk_group,
        }
    }
}

/// Auxiliary-loss-free MoE gate ("noaux_tc"): expert/group selection is biased by
/// a learned correction term, but the combination weights use the raw scores.
struct MoeGate {
    weight: Tensor,
    e_score_correction_bias: Tensor,
    top_k: usize,
    n_routed_experts: usize,
    n_group: usize,
    topk_group: usize,
    norm_topk_prob: bool,
    routed_scaling_factor: f64,
}

impl MoeGate {
    fn new(cfg: &DeepSeekV3Config, vb: VarBuilder, n_routed_experts: usize) -> Result<Self> {
        if cfg.topk_method != TopkMethod3::NoAuxTc {
            candle::bail!(
                "DeepSeek-V3 only supports the \"noaux_tc\" topk_method, got {:?}",
                cfg.topk_method
            );
        }
        if cfg.scoring_func != ScoringFunc3::Sigmoid {
            candle::bail!(
                "DeepSeek-V3 only supports the \"sigmoid\" scoring_func, got {:?}",
                cfg.scoring_func
            );
        }
        let weight = vb.get((n_routed_experts, cfg.hidden_size), "weight")?;
        let e_score_correction_bias = vb.get(n_routed_experts, "e_score_correction_bias")?;
        Ok(Self {
            weight,
            e_score_correction_bias,
            top_k: cfg.num_experts_per_tok.unwrap(),
            n_routed_experts,
            n_group: cfg.n_group,
            topk_group: cfg.topk_group,
            norm_topk_prob: cfg.norm_topk_prob,
            routed_scaling_factor: cfg.routed_scaling_factor,
        })
    }

    /// (topk_idx, topk_weight)
    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let (bs, seq_len, h) = xs.dims3()?;
        let n = bs * seq_len;
        let xs = xs.reshape((n, h))?;
        let logits = xs
            .to_dtype(DType::F32)?
            .matmul(&self.weight.t()?.to_dtype(DType::F32)?)?;
        let scores = candle_nn::ops::sigmoid(&logits)?;

        let scores_for_choice =
            scores.broadcast_add(&self.e_score_correction_bias.to_dtype(DType::F32)?)?;

        // Score each group by the sum of its two highest-scoring experts.
        let group_scores = scores_for_choice
            .reshape((n, self.n_group, ()))?
            .topk_unsorted(2)?
            .values
            .sum(D::Minus1)?;
        // Pick the top groups.
        let group_idx = group_scores.topk_unsorted(self.topk_group)?.indices;
        let group_mask = group_scores.zeros_like()?.scatter_add(
            &group_idx,
            &group_idx.ones_like()?.to_dtype(group_scores.dtype())?,
            1,
        )?;
        let score_mask = group_mask
            .unsqueeze(D::Minus1)?
            .expand((n, self.n_group, self.n_routed_experts / self.n_group))?
            .reshape((n, self.n_routed_experts))?;
        // Zero out experts that belong to a non-selected group.
        let tmp_scores = masked_fill(&scores_for_choice, &(1. - &score_mask.ne(0.)?)?, 0.)?;
        let TopKOutput {
            indices: topk_idx, ..
        } = tmp_scores.topk_unsorted(self.top_k)?;
        // The combination weights come from the *uncorrected* sigmoid scores.
        let mut topk_weight = scores.gather(&topk_idx, 1)?;

        if self.top_k > 1 && self.norm_topk_prob {
            let denominator = (topk_weight.sum_keepdim(D::Minus1)? + 1e-20)?;
            topk_weight = topk_weight.broadcast_div(&denominator)?;
        }
        topk_weight = (topk_weight * self.routed_scaling_factor)?;

        Ok((topk_idx, topk_weight))
    }
}

struct Moe {
    experts: Vec<Mlp>,
    shared_experts: Option<Mlp>,
    gate: MoeGate,
}

impl Moe {
    fn new(
        cfg: &DeepSeekV3Config,
        vb: VarBuilder,
        n_shared_experts: Option<usize>,
        n_routed_experts: usize,
    ) -> Result<Self> {
        let v2_cfg: DeepSeekV2Config = cfg.into();
        let mut experts = Vec::with_capacity(n_routed_experts);
        for i in 0..n_routed_experts {
            let vb_e = vb.pp("experts").pp(i);
            experts.push(Mlp::new(
                &v2_cfg,
                vb_e,
                None,
                Some(cfg.moe_intermediate_size),
            )?);
        }
        let shared_experts = if let Some(n_shared_experts) = n_shared_experts {
            let intermediate_size = cfg.moe_intermediate_size * n_shared_experts;
            Some(Mlp::new(
                &v2_cfg,
                vb.pp("shared_experts"),
                None,
                Some(intermediate_size),
            )?)
        } else {
            None
        };
        let gate = MoeGate::new(cfg, vb.pp("gate"), n_routed_experts)?;
        Ok(Self {
            experts,
            shared_experts,
            gate,
        })
    }

    fn moe_infer(&self, xs: &Tensor, topk_ids: &Tensor, topk_weight: &Tensor) -> Result<Tensor> {
        let mut y = xs.zeros_like()?;
        let counts = topk_ids
            .flatten_all()?
            .bincount(self.experts.len() as u32)?;
        for (i, expert) in self.experts.iter().enumerate() {
            if counts[i] == 0 {
                continue;
            }
            let idx_top = topk_ids.eq(i as f64)?.nonzero()?.t()?;
            let idx = &idx_top.i(0)?.contiguous()?;
            let top = &idx_top.i(1)?.contiguous()?;

            y = y.index_add(
                idx,
                &expert.forward(&xs.index_select(idx, 0)?)?.broadcast_mul(
                    &topk_weight
                        .index_select(idx, 0)?
                        .gather(&top.unsqueeze(1)?, 1)?
                        .squeeze(1)?
                        .unsqueeze(D::Minus1)?
                        .to_dtype(xs.dtype())?,
                )?,
                0,
            )?;
        }

        Ok(y)
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let identity = xs.clone();
        let orig_shape = xs.shape();
        let (topk_idx, topk_weight) = self.gate.forward(xs)?;
        let xs = xs.reshape(((), xs.dim(D::Minus1)?))?;

        let mut y = self
            .moe_infer(&xs, &topk_idx, &topk_weight)?
            .reshape(orig_shape)?;
        if let Some(ref shared_experts) = self.shared_experts {
            y = (y + shared_experts.forward(&identity)?)?;
        }
        Ok(y)
    }
}

enum MoeOrMlp {
    Moe(Box<Moe>),
    Mlp(Box<Mlp>),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(mlp) => mlp.forward(xs),
            Self::Moe(moe) => moe.forward(xs),
        }
    }
}

struct DecoderLayer {
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    attn: Attention,
    moe_or_mlp: MoeOrMlp,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
        cfg: &DeepSeekV3Config,
        vb: VarBuilder,
        layer_idx: usize,
    ) -> Result<Self> {
        let v2_cfg: DeepSeekV2Config = cfg.into();
        let attn = Attention::new(rotary_emb, &v2_cfg, vb.pp("self_attn"))?;
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let moe_or_mlp = if let Some(n_routed_experts) = cfg.n_routed_experts {
            if layer_idx >= cfg.first_k_dense_replace
                && layer_idx.is_multiple_of(cfg.moe_layer_freq)
            {
                MoeOrMlp::Moe(
                    Moe::new(cfg, vb.pp("mlp"), cfg.n_shared_experts, n_routed_experts)?.into(),
                )
            } else {
                MoeOrMlp::Mlp(Mlp::new(&v2_cfg, vb.pp("mlp"), None, None)?.into())
            }
        } else {
            MoeOrMlp::Mlp(Mlp::new(&v2_cfg, vb.pp("mlp"), None, None)?.into())
        };

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            attn,
            moe_or_mlp,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.attn.forward(&xs, attention_mask, seqlen_offset)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .moe_or_mlp
            .forward(&xs.apply(&self.post_attention_layernorm)?)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.attn.clear_kv_cache();
    }
}

pub struct DeepSeekV3 {
    lm_head: Linear,
    embed_tokens: Embedding,
    norm: RmsNorm,
    layers: Vec<DecoderLayer>,
    dtype: DType,
    device: Device,
}

impl DeepSeekV3 {
    pub fn new(cfg: &DeepSeekV3Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");

        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let lm_head = if !cfg.tie_word_embeddings {
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        } else {
            candle_nn::Linear::new(embed_tokens.embeddings().clone(), None)
        };
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let rope_cfg = DeepSeekV2RopeConfig {
            rope_scaling: cfg.rope_scaling.clone(),
            max_position_embeddings: cfg.max_position_embeddings,
            rope_theta: cfg.rope_theta,
            qk_rope_head_dim: cfg.qk_rope_head_dim,
        };
        let rotary_emb = Arc::new(DeepSeekV2RotaryEmbedding::new(
            &rope_cfg,
            vb.dtype(),
            vb.device(),
        )?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx), layer_idx)?;
            layers.push(layer)
        }

        Ok(Self {
            lm_head,
            embed_tokens,
            norm,
            layers,
            dtype: vb.dtype(),
            device: vb.device().clone(),
        })
    }

    fn prepare_decoder_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0. }))
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (bs, seq_len) = input_ids.dims2()?;
        let mut xs = self.embed_tokens.forward(input_ids)?;
        let attention_mask = if seq_len == 1 {
            None
        } else {
            let mask = self.prepare_decoder_attention_mask(bs, seq_len, seqlen_offset)?;
            Some(mask)
        };
        for layer in &mut self.layers {
            xs = layer.forward(
                &xs,
                attention_mask
                    .as_ref()
                    .map(|m| m.to_device(xs.device()).unwrap())
                    .as_ref(),
                seqlen_offset,
            )?;
        }
        let xs = xs.apply(&self.norm)?;
        let xs = xs.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&xs)?;
        logits.to_dtype(DType::F32)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Tiny config (one dense layer followed by one MoE layer) just large
    /// enough to exercise every weight tensor / shape transformation in the
    /// forward pass, including the noaux_tc group-limited MoE gate.
    fn test_config() -> DeepSeekV3Config {
        DeepSeekV3Config {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 16,
            moe_intermediate_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            n_shared_experts: Some(1),
            n_routed_experts: Some(4),
            routed_scaling_factor: 1.0,
            topk_method: TopkMethod3::NoAuxTc,
            num_experts_per_tok: Some(2),
            moe_layer_freq: 1,
            first_k_dense_replace: 1,
            norm_topk_prob: true,
            scoring_func: ScoringFunc3::Sigmoid,
            hidden_act: Activation::Silu,
            max_position_embeddings: 16,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: false,
            rope_theta: 10000.0,
            rope_scaling: None,
            attention_bias: false,
            q_lora_rank: None,
            qk_rope_head_dim: 4,
            kv_lora_rank: 4,
            v_head_dim: 4,
            qk_nope_head_dim: 4,
            n_group: 2,
            topk_group: 1,
        }
    }

    fn rand_tensor(shape: &[usize], dev: &Device) -> Result<Tensor> {
        Tensor::randn(0f32, 1f32, shape, dev)
    }

    fn build_var_map(cfg: &DeepSeekV3Config, dev: &Device) -> Result<HashMap<String, Tensor>> {
        let mut m = HashMap::new();
        let h = cfg.hidden_size;
        let q_head_dim = cfg.qk_rope_head_dim + cfg.qk_nope_head_dim;

        m.insert(
            "model.embed_tokens.weight".to_string(),
            rand_tensor(&[cfg.vocab_size, h], dev)?,
        );
        m.insert(
            "lm_head.weight".to_string(),
            rand_tensor(&[cfg.vocab_size, h], dev)?,
        );
        m.insert("model.norm.weight".to_string(), rand_tensor(&[h], dev)?);

        for layer_idx in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{layer_idx}");
            m.insert(
                format!("{p}.input_layernorm.weight"),
                rand_tensor(&[h], dev)?,
            );
            m.insert(
                format!("{p}.post_attention_layernorm.weight"),
                rand_tensor(&[h], dev)?,
            );

            // Attention (MLA) weights, shared shape regardless of layer kind.
            m.insert(
                format!("{p}.self_attn.q_proj.weight"),
                rand_tensor(&[cfg.num_attention_heads * q_head_dim, h], dev)?,
            );
            m.insert(
                format!("{p}.self_attn.kv_a_proj_with_mqa.weight"),
                rand_tensor(&[cfg.kv_lora_rank + cfg.qk_rope_head_dim, h], dev)?,
            );
            m.insert(
                format!("{p}.self_attn.kv_a_layernorm.weight"),
                rand_tensor(&[cfg.kv_lora_rank], dev)?,
            );
            m.insert(
                format!("{p}.self_attn.kv_b_proj.weight"),
                rand_tensor(
                    &[
                        cfg.num_attention_heads
                            * (q_head_dim - cfg.qk_rope_head_dim + cfg.v_head_dim),
                        cfg.kv_lora_rank,
                    ],
                    dev,
                )?,
            );
            m.insert(
                format!("{p}.self_attn.o_proj.weight"),
                rand_tensor(&[h, cfg.num_attention_heads * cfg.v_head_dim], dev)?,
            );

            let is_moe = layer_idx >= cfg.first_k_dense_replace
                && layer_idx.is_multiple_of(cfg.moe_layer_freq);
            if is_moe {
                let n_routed_experts = cfg.n_routed_experts.unwrap();
                m.insert(
                    format!("{p}.mlp.gate.weight"),
                    rand_tensor(&[n_routed_experts, h], dev)?,
                );
                m.insert(
                    format!("{p}.mlp.gate.e_score_correction_bias"),
                    rand_tensor(&[n_routed_experts], dev)?,
                );
                for e in 0..n_routed_experts {
                    let ep = format!("{p}.mlp.experts.{e}");
                    m.insert(
                        format!("{ep}.gate_proj.weight"),
                        rand_tensor(&[cfg.moe_intermediate_size, h], dev)?,
                    );
                    m.insert(
                        format!("{ep}.up_proj.weight"),
                        rand_tensor(&[cfg.moe_intermediate_size, h], dev)?,
                    );
                    m.insert(
                        format!("{ep}.down_proj.weight"),
                        rand_tensor(&[h, cfg.moe_intermediate_size], dev)?,
                    );
                }
                let shared_intermediate = cfg.moe_intermediate_size * cfg.n_shared_experts.unwrap();
                m.insert(
                    format!("{p}.mlp.shared_experts.gate_proj.weight"),
                    rand_tensor(&[shared_intermediate, h], dev)?,
                );
                m.insert(
                    format!("{p}.mlp.shared_experts.up_proj.weight"),
                    rand_tensor(&[shared_intermediate, h], dev)?,
                );
                m.insert(
                    format!("{p}.mlp.shared_experts.down_proj.weight"),
                    rand_tensor(&[h, shared_intermediate], dev)?,
                );
            } else {
                m.insert(
                    format!("{p}.mlp.gate_proj.weight"),
                    rand_tensor(&[cfg.intermediate_size, h], dev)?,
                );
                m.insert(
                    format!("{p}.mlp.up_proj.weight"),
                    rand_tensor(&[cfg.intermediate_size, h], dev)?,
                );
                m.insert(
                    format!("{p}.mlp.down_proj.weight"),
                    rand_tensor(&[h, cfg.intermediate_size], dev)?,
                );
            }
        }

        Ok(m)
    }

    #[test]
    fn forward_pass_shapes_and_kv_cache() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = test_config();
        let map = build_var_map(&cfg, &dev)?;
        let vb = VarBuilder::from_tensors(map, DType::F32, &dev);
        let mut model = DeepSeekV3::new(&cfg, vb)?;

        let prompt = Tensor::new(&[1u32, 2, 3, 4], &dev)?.unsqueeze(0)?;
        let logits = model.forward(&prompt, 0)?;
        assert_eq!(logits.dims(), &[1, cfg.vocab_size]);

        // A second, single-token forward pass exercises the kv-cache path.
        let next = Tensor::new(&[5u32], &dev)?.unsqueeze(0)?;
        let logits = model.forward(&next, 4)?;
        assert_eq!(logits.dims(), &[1, cfg.vocab_size]);

        model.clear_kv_cache();
        Ok(())
    }

    #[test]
    fn rejects_non_noaux_tc_topk_method() -> Result<()> {
        let dev = Device::Cpu;
        let mut cfg = test_config();
        cfg.topk_method = TopkMethod3::Greedy;
        let map = build_var_map(&cfg, &dev)?;
        let vb = VarBuilder::from_tensors(map, DType::F32, &dev);
        assert!(DeepSeekV3::new(&cfg, vb).is_err());
        Ok(())
    }
}
