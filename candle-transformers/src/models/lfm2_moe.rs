//! LFM2-MoE (Liquid Foundation Model 2, Mixture-of-Experts) implementation.
//!
//! LFM2-MoE keeps the hybrid attention + short-convolution backbone of LFM2 and
//! replaces the dense feed-forward of the later layers with a sparse
//! Mixture-of-Experts block (DeepSeek-V3 style sigmoid routing with an expert bias).
//!
//! The first `num_dense_layers` layers keep a regular dense SwiGLU MLP; every
//! subsequent layer routes each token to `num_experts_per_tok` experts out of
//! `num_experts`.
//!
//! The token-mixing operators (attention, short convolution), the RoPE cache and
//! the dense MLP are reused as-is from the [`crate::models::lfm2`] module, in the
//! same spirit as `qwen3_moe` reusing `qwen3`.

use crate::models::lfm2::{Attention, Mlp, ShortConv};
use crate::models::with_tracing::{linear_no_bias as linear, Embedding, Linear, RmsNorm};
use candle::{DType, IndexOp, Module, Result, Tensor, D};
use candle_nn::VarBuilder;

// Reused verbatim from the dense LFM2 model.
pub use crate::models::lfm2::{Cache, Config as Lfm2Config, LayerType};

/// Nested RoPE parameters introduced by transformers v5 configs (e.g. LFM2.5),
/// which moved `rope_theta` from the top level into a `rope_parameters` object.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f32,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Lfm2MoeConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    // When present (transformers v5 configs such as LFM2.5), the nested value
    // takes precedence over the flat `rope_theta` field.
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_conv_l_cache", alias = "conv_L_cache")]
    pub conv_l_cache: usize,
    #[serde(default)]
    pub conv_bias: bool,
    pub layer_types: Vec<LayerType>,
    // HF uses `tie_word_embeddings` (defaults to true for LFM2-MoE); the dense
    // model used `tie_embedding`. Accept either key and default to tied.
    #[serde(default = "default_tie_embedding", alias = "tie_word_embeddings")]
    pub tie_embedding: bool,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    // Dense feed-forward dimension (used by the first `num_dense_layers` layers).
    pub intermediate_size: usize,
    // --- MoE-specific fields ---
    // Intermediate size of each expert (smaller than the dense `intermediate_size`).
    pub moe_intermediate_size: usize,
    // Total number of experts in each MoE layer.
    pub num_experts: usize,
    // Number of experts each token is routed to (top-k).
    pub num_experts_per_tok: usize,
    // Number of leading dense layers before MoE layers start.
    pub num_dense_layers: usize,
    // Whether the router adds a per-expert bias when selecting the top-k.
    #[serde(default = "default_use_expert_bias")]
    pub use_expert_bias: bool,
    // Whether to renormalize the top-k routing weights so they sum to 1.
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
    // Final scaling factor applied to the routing weights.
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

fn default_tie_embedding() -> bool {
    true
}

fn default_use_expert_bias() -> bool {
    true
}

fn default_norm_topk_prob() -> bool {
    true
}

fn default_routed_scaling_factor() -> f64 {
    1.0
}

impl Lfm2MoeConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn into_config(self, use_flash_attn: bool) -> Config {
        let rope_theta = self
            .rope_parameters
            .as_ref()
            .map_or(self.rope_theta, |p| p.rope_theta);
        Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            moe_intermediate_size: self.moe_intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            norm_eps: self.norm_eps,
            rope_theta,
            max_position_embeddings: self.max_position_embeddings,
            conv_l_cache: self.conv_l_cache,
            conv_bias: self.conv_bias,
            layer_types: self.layer_types,
            tie_embedding: self.tie_embedding,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            num_experts: self.num_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            num_dense_layers: self.num_dense_layers,
            use_expert_bias: self.use_expert_bias,
            norm_topk_prob: self.norm_topk_prob,
            routed_scaling_factor: self.routed_scaling_factor,
            use_flash_attn,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
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
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub num_dense_layers: usize,
    pub use_expert_bias: bool,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f64,
    pub use_flash_attn: bool,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// Bridge our MoE config to the dense LFM2 config so we can reuse the dense
/// token-mixing operators (`Attention`, `ShortConv`) and the RoPE `Cache`.
impl From<&Config> for Lfm2Config {
    fn from(cfg: &Config) -> Self {
        Lfm2Config {
            vocab_size: cfg.vocab_size,
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            num_hidden_layers: cfg.num_hidden_layers,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            norm_eps: cfg.norm_eps,
            rope_theta: cfg.rope_theta,
            max_position_embeddings: cfg.max_position_embeddings,
            conv_l_cache: cfg.conv_l_cache,
            conv_bias: cfg.conv_bias,
            layer_types: cfg.layer_types.clone(),
            tie_embedding: cfg.tie_embedding,
            bos_token_id: cfg.bos_token_id,
            eos_token_id: cfg.eos_token_id,
            use_flash_attn: cfg.use_flash_attn,
        }
    }
}

/// Router (gating network) for a sparse MoE layer.
///
/// LFM2-MoE uses DeepSeek-V3 style routing: the router logits go through a
/// `sigmoid` (not a softmax), and an optional per-expert bias is added *only* to
/// pick which experts win the top-k. The weights that actually scale each
/// expert's output are read back from the unbiased sigmoid scores.
#[derive(Debug, Clone)]
struct MoeGate {
    gate: Linear,
    // Per-expert bias of shape (num_experts,). Only used to select the top-k.
    expert_bias: Option<Tensor>,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
    routed_scaling_factor: f64,
    span: tracing::Span,
}

impl MoeGate {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let gate = linear(cfg.hidden_size, cfg.num_experts, vb.pp("gate"))?;
        let expert_bias = if cfg.use_expert_bias {
            Some(vb.get(cfg.num_experts, "expert_bias")?)
        } else {
            None
        };
        Ok(Self {
            gate,
            expert_bias,
            num_experts_per_tok: cfg.num_experts_per_tok,
            norm_topk_prob: cfg.norm_topk_prob,
            routed_scaling_factor: cfg.routed_scaling_factor,
            span: tracing::span!(tracing::Level::TRACE, "moe-gate"),
        })
    }

    /// `xs`: (num_tokens, hidden_size).
    /// Returns `(selected_experts, routing_weights)`, both of shape (num_tokens, top_k);
    /// `selected_experts` holds u32 expert indices, `routing_weights` the scaling factors.
    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let _enter = self.span.enter();
        // (num_tokens, num_experts)
        let router_logits = xs.apply(&self.gate)?;
        let routing_weights = candle_nn::ops::sigmoid(&router_logits)?;

        // Pick the top-k experts. The bias (if any) influences *selection* only.
        let selected_experts = match &self.expert_bias {
            Some(bias) => routing_weights.broadcast_add(bias)?,
            None => routing_weights.clone(),
        }
        .arg_sort_last_dim(false)? // descending: largest scores first
        .narrow(D::Minus1, 0, self.num_experts_per_tok)?
        .contiguous()?;

        // Gather the *unbiased* sigmoid weights for the selected experts.
        let mut routing_weights = routing_weights.gather(&selected_experts, D::Minus1)?;

        if self.norm_topk_prob {
            let denom = (routing_weights.sum_keepdim(D::Minus1)? + 1e-6)?;
            routing_weights = routing_weights.broadcast_div(&denom)?;
        }
        let routing_weights = (routing_weights * self.routed_scaling_factor)?;
        Ok((selected_experts, routing_weights))
    }
}

/// Sparse Mixture-of-Experts feed-forward block.
///
/// Each token is routed to `num_experts_per_tok` experts; every expert is a plain
/// LFM2 SwiGLU MLP (reused from the dense model) sized to `moe_intermediate_size`.
#[derive(Debug, Clone)]
struct SparseMoeBlock {
    gate: MoeGate,
    experts: Vec<Mlp>,
    span: tracing::Span,
}

impl SparseMoeBlock {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let gate = MoeGate::new(cfg, vb.clone())?;

        // Each expert is structurally identical to the dense MLP (w1/w3/w2),
        // just narrower. Build a dense config whose intermediate size is the
        // expert size so we can reuse `lfm2::Mlp` verbatim.
        let mut expert_cfg: Lfm2Config = cfg.into();
        expert_cfg.intermediate_size = cfg.moe_intermediate_size;

        let vb_e = vb.pp("experts");
        let mut experts = Vec::with_capacity(cfg.num_experts);
        for idx in 0..cfg.num_experts {
            experts.push(Mlp::new(&expert_cfg, vb_e.pp(idx))?);
        }
        Ok(Self {
            gate,
            experts,
            span: tracing::span!(tracing::Level::TRACE, "moe-block"),
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        // Flatten the batch/sequence dims: route each token independently.
        let xs = xs.reshape(((), hidden_dim))?;

        let (selected_experts, routing_weights) = self.gate.forward(&xs)?;

        // Move routing decisions to the CPU so we can build per-expert token lists.
        let routing_weights = routing_weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let selected_experts = selected_experts.to_vec2::<u32>()?;

        // For each expert, collect the row indices routed to it and their weights.
        let mut token_rows = vec![vec![]; self.experts.len()];
        let mut token_weights = vec![vec![]; self.experts.len()];
        for (row_idx, (weights, experts)) in routing_weights
            .iter()
            .zip(selected_experts.iter())
            .enumerate()
        {
            for (&weight, &expert_idx) in weights.iter().zip(experts.iter()) {
                token_rows[expert_idx as usize].push(row_idx as u32);
                token_weights[expert_idx as usize].push(weight);
            }
        }

        // Run each expert once on its batch of tokens, then scatter-add back.
        let mut ys = xs.zeros_like()?;
        for (expert_idx, expert) in self.experts.iter().enumerate() {
            let rows = &token_rows[expert_idx];
            if rows.is_empty() {
                continue;
            }
            let rows = Tensor::new(rows.as_slice(), xs.device())?;
            let weights = Tensor::new(token_weights[expert_idx].as_slice(), xs.device())?
                .reshape(((), 1))?
                .to_dtype(xs.dtype())?;

            let tokens = xs.index_select(&rows, 0)?;
            let out = expert.forward(&tokens)?;
            let out = out.broadcast_mul(&weights)?;
            ys = ys.index_add(&rows, &out, 0)?;
        }

        ys.reshape((b_size, seq_len, hidden_dim))
    }
}

/// Feed-forward of a decoder layer: either a dense MLP or a sparse MoE block.
#[derive(Debug, Clone)]
enum FeedForward {
    Dense(Mlp),
    Sparse(SparseMoeBlock),
}

impl FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            FeedForward::Dense(mlp) => mlp.forward(x),
            FeedForward::Sparse(moe) => moe.forward(x),
        }
    }
}

/// Token-mixing operator of a decoder layer (reused from the dense model).
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
        // LFM2 uses operator_norm and ffn_norm naming.
        let input_layernorm = RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb.pp("operator_norm"))?;
        let post_attention_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.norm_eps, vb.pp("ffn_norm"))?;

        // The first `num_dense_layers` layers keep a dense SwiGLU MLP; the rest
        // route tokens through a sparse MoE block. This mirrors the Python:
        //   feed_forward = Lfm2MoeMLP(...) if layer_idx < num_dense_layers
        //                  else Lfm2MoeSparseMoeBlock(...)
        let lfm2_cfg: Lfm2Config = cfg.into();
        let feed_forward = if layer_idx < cfg.num_dense_layers {
            FeedForward::Dense(Mlp::new(&lfm2_cfg, vb.pp("feed_forward"))?)
        } else {
            FeedForward::Sparse(SparseMoeBlock::new(cfg, vb.pp("feed_forward"))?)
        };

        let layer_type = cfg
            .layer_types
            .get(layer_idx)
            .copied()
            .unwrap_or(LayerType::FullAttention);
        let kind = match layer_type {
            LayerType::FullAttention => {
                LayerKind::Attention(Box::new(Attention::new(&lfm2_cfg, vb.pp("self_attn"))?))
            }
            LayerType::Conv => LayerKind::ShortConv(ShortConv::new(&lfm2_cfg, vb.pp("conv"))?),
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

/// LFM2-MoE model for causal language modeling.
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
    use candle::Device;
    use candle_nn::{VarBuilder, VarMap};

    /// Build a tiny LFM2-MoE config that exercises every code path:
    /// dense layers, MoE layers, attention layers and conv layers.
    fn tiny_config() -> Config {
        Config {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 16,
            moe_intermediate_size: 8,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: 64,
            conv_l_cache: 3,
            conv_bias: false,
            // layers 0,1 = dense (and conv); layer 2 = MoE + attention; layer 3 = MoE + conv.
            layer_types: vec![
                LayerType::Conv,
                LayerType::Conv,
                LayerType::FullAttention,
                LayerType::Conv,
            ],
            tie_embedding: true,
            bos_token_id: None,
            eos_token_id: None,
            num_experts: 4,
            num_experts_per_tok: 2,
            num_dense_layers: 2,
            use_expert_bias: true,
            norm_topk_prob: true,
            routed_scaling_factor: 1.0,
            use_flash_attn: false,
        }
    }

    /// LFM2 (transformers v4) exposes `rope_theta` at the top level while LFM2.5
    /// (transformers v5) nests it inside `rope_parameters`; both must parse to the
    /// right value.
    #[test]
    fn config_rope_theta_formats() {
        let base = r#"
            "vocab_size": 128000, "hidden_size": 2048, "num_hidden_layers": 24,
            "num_attention_heads": 32, "layer_types": ["conv", "full_attention"],
            "intermediate_size": 7168, "moe_intermediate_size": 1792,
            "num_experts": 32, "num_experts_per_tok": 4, "num_dense_layers": 2,
            "bos_token_id": 1, "eos_token_id": 7
        "#;
        let flat: Lfm2MoeConfig =
            serde_json::from_str(&format!("{{ \"rope_theta\": 1000000.0, {base} }}")).unwrap();
        assert_eq!(flat.into_config(false).rope_theta, 1_000_000.0);
        let nested: Lfm2MoeConfig = serde_json::from_str(&format!(
            "{{ \"rope_parameters\": {{ \"rope_theta\": 5000000, \"rope_type\": \"default\" }}, {base} }}"
        ))
        .unwrap();
        assert_eq!(nested.into_config(false).rope_theta, 5_000_000.0);
    }

    /// Smoke test: build the model with random weights and run a prefill followed
    /// by a single-token decode step. This checks the full plumbing (routing,
    /// expert dispatch, dense/MoE wiring, attention + conv caches) without needing
    /// the real multi-gigabyte checkpoint.
    #[test]
    fn forward_shapes() -> Result<()> {
        let device = Device::Cpu;
        let cfg = tiny_config();

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = Model::new(&cfg, vb)?;

        let mut cache = Cache::new(true, DType::F32, &Lfm2Config::from(&cfg), &device)?;

        // Prefill three tokens.
        let input = Tensor::new(&[[1u32, 2u32, 3u32]], &device)?;
        let logits = model.forward(&input, 0, &mut cache)?;
        assert_eq!(logits.dims(), &[1, cfg.vocab_size]);

        // Decode one more token, reusing the caches (index_pos = 3).
        let next = Tensor::new(&[[4u32]], &device)?;
        let logits = model.forward(&next, 3, &mut cache)?;
        assert_eq!(logits.dims(), &[1, cfg.vocab_size]);

        Ok(())
    }

    /// The router must select exactly `num_experts_per_tok` experts per token and
    /// produce matching, finite routing weights.
    #[test]
    fn router_topk_shapes() -> Result<()> {
        let device = Device::Cpu;
        let cfg = tiny_config();

        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let gate = MoeGate::new(&cfg, vb)?;

        let n_tokens = 5;
        let xs = Tensor::randn(0f32, 1f32, (n_tokens, cfg.hidden_size), &device)?;
        let (selected, weights) = gate.forward(&xs)?;

        assert_eq!(selected.dims(), &[n_tokens, cfg.num_experts_per_tok]);
        assert_eq!(weights.dims(), &[n_tokens, cfg.num_experts_per_tok]);

        // Every selected index must be a valid expert id.
        for row in selected.to_vec2::<u32>()? {
            for idx in row {
                assert!((idx as usize) < cfg.num_experts);
            }
        }
        // With norm_topk_prob the per-token weights sum to ~1.
        for row in weights.to_vec2::<f32>()? {
            let sum: f32 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "weights should sum to 1, got {sum}"
            );
        }
        Ok(())
    }
}
