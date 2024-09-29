#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle::{
    CpuStorage, CustomOp1, DType, Device, Error, IndexOp, Layout, Module, Result, Shape, Tensor,
    WithDType, D,
};
use candle_nn::{layer_norm, LayerNorm, Linear, VarBuilder};
use std::sync::Arc;

use crate::layers::{PhiRopeConfig, PhiRopeScalingConfig, PhiRotaryEmbedding};

fn default_use_flash_attn() -> bool {
    false
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_act: candle_nn::Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_scaling: Option<PhiRopeScalingConfig>,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>,
    pub original_max_position_embeddings: usize,
    pub lm_head_bias: bool,
    pub attention_bias: bool,
    pub num_local_experts: usize,
    pub router_jitter_noise: f64,
    #[serde(default = "default_use_flash_attn")]
    pub use_flash_attn: bool,
}

impl From<Config> for PhiRopeConfig {
    fn from(val: Config) -> Self {
        PhiRopeConfig {
            rope_scaling: val.rope_scaling,
            max_position_embeddings: val.max_position_embeddings,
            original_max_position_embeddings: val.original_max_position_embeddings,
            rope_theta: val.rope_theta,
            head_dim: val.hidden_size / val.num_attention_heads,
        }
    }
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn with_flash_attn(mut self, use_flash_attn: bool) -> Self {
        self.use_flash_attn = use_flash_attn;
        self
    }
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

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<PhiRotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
    use_flash_attn: bool,
}

impl Attention {
    fn new(rotary_emb: Arc<PhiRotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();

        let q_proj = candle_nn::linear_b(
            cfg.hidden_size,
            num_heads * head_dim,
            cfg.attention_bias,
            vb.pp("q_proj"),
        )?;
        let k_proj = candle_nn::linear_b(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            cfg.attention_bias,
            vb.pp("k_proj"),
        )?;
        let v_proj = candle_nn::linear_b(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            cfg.attention_bias,
            vb.pp("v_proj"),
        )?;
        let o_proj = candle_nn::linear_b(
            num_heads * head_dim,
            cfg.hidden_size,
            cfg.attention_bias,
            vb.pp("o_proj"),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary_emb,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_cache: None,
            use_flash_attn: cfg.use_flash_attn,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        position_id: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let (q, k, v) = if q_len != 1 {
            let q = q
                .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.num_heads, q_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?;
            (q, k, v)
        };

        let (q, k) = self
            .rotary_emb
            .forward(&q, &k, &[seqlen_offset], &[position_id])?;

        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &k], 2)?;
                let value_states = Tensor::cat(&[prev_v, &v], 2)?;
                (key_states, value_states)
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let attn_output = if self.use_flash_attn {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
            flash_attn(&q, &k, &v, softmax_scale, q_len > 1)?.transpose(1, 2)?
        } else {
            let scale = 1f64 / f64::sqrt(self.head_dim as f64);
            let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

            let attn_weights = match attention_mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(mask)?,
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&v)?
        };

        let attn_output = if attention_mask.is_some() {
            attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?
        } else {
            attn_output.reshape((b_sz, q_len, ()))?
        };
        self.o_proj.forward(&attn_output)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

#[derive(Clone)]
struct Mlp {
    w1: Linear,
    w2: Linear,
    w3: Linear,
    act_fn: candle_nn::Activation,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;

        let w1 = candle_nn::linear_no_bias(hidden_size, i_size, vb.pp("w1"))?;
        let w2 = candle_nn::linear_no_bias(i_size, hidden_size, vb.pp("w2"))?;
        let w3 = candle_nn::linear_no_bias(hidden_size, i_size, vb.pp("w3"))?;

        Ok(Self {
            w1,
            w2,
            w3,
            act_fn: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut current_hidden_states = self.w1.forward(xs)?.apply(&self.act_fn)?;
        let rhs = self.w3.forward(xs)?;
        current_hidden_states = current_hidden_states.broadcast_mul(&rhs)?;
        self.w2.forward(&current_hidden_states)
    }
}

struct NonZero {}

impl NonZero {
    // Sequential CPU version
    fn nonzero<T: WithDType>(&self, vs: &[T], layout: &Layout) -> Vec<u32> {
        let n = layout.dims().len();
        let mut result = Vec::new();
        let mut indices = vec![0u32; n];
        for (i, v) in vs.iter().enumerate() {
            if !v.is_zero() {
                let mut idx = i;
                for (dim_index, dim) in layout.dims().iter().enumerate().rev() {
                    let d = idx % dim;
                    indices[dim_index] = u32::try_from(d).unwrap();
                    idx /= dim;
                }
                result.extend_from_slice(&indices);
            }
        }
        result
    }
}

impl CustomOp1 for NonZero {
    fn name(&self) -> &'static str {
        "nonzero"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        if !layout.is_contiguous() {
            return Err(Error::RequiresContiguous { op: "nonzero" });
        }
        let result = match storage {
            candle::CpuStorage::U8(vs) => self.nonzero(vs, layout),
            candle::CpuStorage::U32(vs) => self.nonzero(vs, layout),
            candle::CpuStorage::I64(vs) => self.nonzero(vs, layout),
            candle::CpuStorage::BF16(vs) => self.nonzero(vs, layout),
            candle::CpuStorage::F16(vs) => self.nonzero(vs, layout),
            candle::CpuStorage::F32(vs) => self.nonzero(vs, layout),
            candle::CpuStorage::F64(vs) => self.nonzero(vs, layout),
        };
        let index_len = layout.dims().len();
        let result_len = result.len() / index_len;
        let result = CpuStorage::U32(result);
        let shape = Shape::from_dims(&[result_len, index_len]);
        Ok((result, shape))
    }
}

pub trait NonZeroOp {
    fn nonzero(&self) -> Result<Tensor>;
}

impl NonZeroOp for Tensor {
    fn nonzero(&self) -> Result<Tensor> {
        if !self.is_contiguous() {
            return Err(candle::Error::RequiresContiguous { op: "nonzero" });
        }
        let original_device = self.device();
        self.to_device(&candle::Device::Cpu)?
            .apply_op1_no_bwd(&NonZero {})?
            .to_device(original_device)
    }
}

struct MoeMlp {
    gate: candle_nn::Linear,
    experts: Vec<Mlp>,
    router_jitter_noise: f64,
    num_experts: usize,
}

// https://github.com/mokeyish/candle-ext/blob/main/src/masked_fill.rs
/// xs are on false (0), value is on true (1)
pub fn masked_fill<D: WithDType>(xs: &Tensor, mask: &Tensor, value: D) -> Result<Tensor> {
    let on_true = Tensor::full(value, xs.shape(), xs.device())?.to_dtype(xs.dtype())?;
    let on_false = xs;
    let res = mask
        .broadcast_as(xs.shape())?
        .where_cond(&on_true, on_false)?;
    Ok(res)
}

impl MoeMlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let num_experts = cfg.num_local_experts;
        let gate = candle_nn::linear_no_bias(cfg.hidden_size, num_experts, vb.pp("gate"))?;

        let experts_vb = vb.pp("experts");
        let mut experts = Vec::with_capacity(num_experts);
        for i in 0..num_experts {
            experts.push(Mlp::new(cfg, experts_vb.pp(i))?);
        }

        Ok(Self {
            gate,
            experts,
            router_jitter_noise: cfg.router_jitter_noise,
            num_experts,
        })
    }

    fn sparsemixer(&self, scores: &Tensor, jitter_eps: f64) -> Result<(Tensor, Tensor)> {
        // Compute mask for sparsity
        let selected_experts = scores.argmax_keepdim(D::Minus1)?;
        let mask_logits_threshold = scores.gather(&selected_experts, D::Minus1)?;
        let factor = scores.abs()?.broadcast_minimum(&mask_logits_threshold)?;
        let mask_logits_threshold = mask_logits_threshold
            .broadcast_sub(scores)?
            .broadcast_div(&factor)?
            .gt(2. * jitter_eps)?;

        // Apply mask
        let masked_gates = masked_fill(scores, &mask_logits_threshold, f64::NEG_INFINITY)?;

        // Compute scores
        let masked_gates = candle_nn::ops::softmax_last_dim(&masked_gates)?;
        let multiplier = masked_gates.gather(&selected_experts, D::Minus1)?;

        // Mask out first expert
        let masked_scores = scores.scatter_add(
            &selected_experts
                .broadcast_as(scores.shape())?
                .contiguous()?,
            &(scores.ones_like()? * f64::NEG_INFINITY)?,
            D::Minus1,
        )?;

        // Compute mask for sparsity
        let selected_experts_top2 = masked_scores.argmax_keepdim(D::Minus1)?;
        let mask_logits_threshold = masked_scores.gather(&selected_experts_top2, D::Minus1)?;
        let factor = scores.abs()?.broadcast_minimum(&mask_logits_threshold)?;
        let mask_logits_threshold = mask_logits_threshold
            .broadcast_sub(scores)?
            .broadcast_div(&factor)?
            .gt(2. * jitter_eps)?;

        // Apply mask
        let masked_gates_top2 =
            masked_fill(&masked_scores, &mask_logits_threshold, f64::NEG_INFINITY)?;
        let masked_gates_top2 = candle_nn::ops::softmax_last_dim(&masked_gates_top2)?;
        let multiplier_top2 = masked_gates_top2.gather(&selected_experts_top2, D::Minus1)?;

        let multiplier = Tensor::cat(&[multiplier, multiplier_top2], D::Minus1)?;
        let selected_experts = Tensor::cat(&[selected_experts, selected_experts_top2], D::Minus1)?;

        Ok((multiplier, selected_experts))
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (bs, seq, hidden) = xs.dims3()?;
        let xs = xs.reshape(((), hidden))?;
        let xs_dev = xs.device();
        let xs = xs.to_device(&Device::Cpu)?;

        // Sparse MoE block accumulates hidden states on CPU, but MLP and gate weights are untouched (maybe on GPU)

        let router_logits = self
            .gate
            .forward(&xs.to_device(xs_dev)?)?
            .to_device(&Device::Cpu)?;
        let (routing_weights, selected_experts) = self.sparsemixer(
            &router_logits.to_device(&Device::Cpu)?,
            self.router_jitter_noise,
        )?;

        let mut final_hidden_states = Tensor::zeros((bs * seq, hidden), xs.dtype(), xs.device())?;

        // One hot encode the selected experts to create an expert mask
        // this will be used to easily index which expert to activate
        let experts_mask =
            candle_nn::encoding::one_hot(selected_experts, self.num_experts, 1u8, 0u8)?
                .permute((2, 1, 0))?;

        // Loop over all avail experts in the model and perform the computation on each expert
        for expert_idx in 0..self.num_experts {
            let expert = &self.experts[expert_idx];
            let expert_mask = experts_mask.i(expert_idx)?;
            assert_eq!(expert_mask.rank(), 2);
            let nonzero_mask = expert_mask.contiguous()?.nonzero()?;
            let idx = nonzero_mask.i((.., 0))?;
            let top_x = nonzero_mask.i((.., 1))?;

            if top_x.dim(0)? == 0 {
                continue;
            }

            // Index the correct hidden staters and compute the expert hidden state
            // for the current expert, we need to make sure to multiply the output hidden
            // states by `routing_weights` on the corresponding tokens (top-1, top-2)
            let current_state = xs.index_select(&top_x, 0)?.reshape(((), hidden))?;
            let current_routing_weights = routing_weights
                .index_select(&top_x, 0)?
                .gather(&idx.unsqueeze(1)?.contiguous()?, 1)?;
            let exp_out = expert
                .forward(&current_state.to_device(xs_dev)?)?
                .to_device(&Device::Cpu)?;

            let current_hidden_states = exp_out.broadcast_mul(&current_routing_weights)?;

            final_hidden_states = final_hidden_states.index_add(
                &top_x.contiguous()?,
                &current_hidden_states.to_dtype(xs.dtype())?,
                0,
            )?;
        }

        final_hidden_states
            .reshape((bs, seq, hidden))?
            .to_device(xs_dev)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: MoeMlp,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(rotary_emb: Arc<PhiRotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?;
        let mlp = MoeMlp::new(cfg, vb.pp("block_sparse_moe"))?;
        let input_layernorm =
            layer_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
        position_id: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward(&xs, attention_mask, seqlen_offset, position_id)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&xs.apply(&self.post_attention_layernorm)?)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
}

pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: LayerNorm,
    lm_head: Linear,
    device: Device,
    sliding_window: Option<usize>,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");

        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let rotary_emb = Arc::new(PhiRotaryEmbedding::new(
                vb.dtype(),
                cfg.clone(),
                vb.device(),
            )?);
            let layer = DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        let norm = layer_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = candle_nn::linear_b(
            cfg.hidden_size,
            cfg.vocab_size,
            cfg.lm_head_bias,
            vb.pp("lm_head"),
        )?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
            sliding_window: cfg.sliding_window,
        })
    }

    fn prepare_decoder_attention_mask(
        &self,
        tgt_len: usize,
        seqlen_offset: usize,
        dtype: DType,
    ) -> Result<Tensor> {
        let sliding_window = self.sliding_window.unwrap_or(tgt_len + 1);
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((1, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(dtype)
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        position_id: usize,
    ) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids)?;

        let seq_len = xs.dim(1)?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            let mask = self.prepare_decoder_attention_mask(seq_len, seqlen_offset, xs.dtype())?;
            Some(mask)
        };

        for layer in self.layers.iter_mut() {
            xs = layer.forward(
                &xs,
                attention_mask
                    .as_ref()
                    .map(|m| m.to_device(xs.device()).unwrap())
                    .as_ref(),
                seqlen_offset,
                position_id,
            )?
        }

        self.lm_head.forward(&xs.apply(&self.norm)?)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }
}
