use core::f32;

use crate::config::MllamaTextConfig;
use candle::{shape::Dim, DType, IndexOp, Module, Result, Tensor, D};
use candle_nn::{linear, Linear, VarBuilder};
use candle_transformers::utils::repeat_kv;

pub trait AttentionDecoderLayer {
    fn forward(
        hidden_states: &Tensor,
        cross_attention_states: Option<Tensor>,
        cross_attention_mask: Option<Tensor>,
        attention_mask: Option<Tensor>,
        full_text_row_masked_out_mask: Option<(Tensor, Tensor)>,
        position_ids: Option<Tensor>,
        output_attentions: Option<bool>,
        cache_position: Option<Tensor>,
        position_embeddings: Option<(Tensor, Tensor)>,
    ) -> Result<Vec<Tensor>>;
}
pub struct MllamaTextRMSNorm {
    weight: Tensor,
    variance_epsilon: f32,
}
impl MllamaTextRMSNorm {
    pub fn new(vb: VarBuilder, hidden_size: usize, eps: Option<f32>) -> Result<Self> {
        let weight = vb.get(hidden_size, "weight")?;
        let variance_epsilon = eps.unwrap_or(1e-6);
        Ok(Self {
            weight,
            variance_epsilon,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let input_dtype = hidden_states.dtype();
        let hidden_states = hidden_states.to_dtype(DType::F32)?;
        let variance = hidden_states
            .pow(&Tensor::from_vec(vec![2.0], 1, hidden_states.device())?)? // is there better way to build tensor from scalar value
            // or even easier way to apply broadcast operation using scalares
            .mean_keepdim(hidden_states.dims()[hidden_states.dims().len() - 1])?; // there should be better way to get last dim, is there?

        // torch.rsqrt, but what if we use simple candle_nn::LayerNorm??!
        let eps_variance = variance.broadcast_add(&Tensor::from_vec(
            vec![self.variance_epsilon],
            1,
            variance.device(),
        )?)?;
        let hidden_states = Tensor::ones(1, eps_variance.dtype(), eps_variance.device())?
            .broadcast_div(&eps_variance)?
            .sqrt()?;

        let hidden_states = (&self.weight * hidden_states)?.to_dtype(input_dtype)?;
        Ok(hidden_states)
    }
}

pub struct MllamaTextCrossSdpaAttention {
    num_heads: usize,
    num_key_value_heads: usize,
    dropout: f32,
    hidden_size: usize,
    head_dim: usize,
    num_key_value_groups: usize,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: MllamaTextRMSNorm,
    k_norm: MllamaTextRMSNorm,
}
impl MllamaTextCrossSdpaAttention {
    pub fn new(vb: VarBuilder, cfg: &MllamaTextConfig) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_key_value_heads = cfg.num_key_value_heads;
        let dropout = cfg.dropout;
        let hidden_size = cfg.hidden_size;
        let head_dim = cfg.hidden_size / num_heads;
        let num_key_value_groups = num_heads / num_key_value_heads;

        let q_proj = linear(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_size, num_key_value_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_size, num_key_value_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        let q_norm = MllamaTextRMSNorm::new(vb.pp("q_norm"), head_dim, Some(cfg.rms_norm_eps))?;
        let k_norm = MllamaTextRMSNorm::new(vb.pp("k_norm"), head_dim, Some(cfg.rms_norm_eps))?;
        Ok(Self {
            num_heads,
            num_key_value_heads,
            dropout,
            hidden_size,
            head_dim,
            num_key_value_groups,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        cross_attention_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let [bsz, q_len, _] = *hidden_states.dims() else {
            panic!("wrong shape")
        };
        let query_states = self.q_proj.forward(&hidden_states)?;
        let query_states = query_states
            .reshape((bsz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let query_states = self.q_norm.forward(&query_states)?;

        let (key_states, value_states) = match cross_attention_states {
            Some(cross_attention_states) => {
                let key_states = self.k_proj.forward(cross_attention_states)?;
                let value_states = self.v_proj.forward(cross_attention_states)?;

                let key_states = key_states
                    .reshape((
                        bsz,
                        key_states.elem_count() / (bsz * self.num_key_value_heads * self.head_dim), // again! the should a better way
                        self.num_key_value_heads,
                        self.head_dim,
                    ))?
                    .transpose(1, 2)?;

                let value_states = value_states
                    .reshape((
                        bsz,
                        value_states.elem_count()
                            / (bsz * self.num_key_value_heads * self.head_dim),
                        self.num_key_value_heads,
                        self.head_dim,
                    ))?
                    .transpose(1, 2)?;

                (key_states, value_states)
            }
            _ => panic!("in case of mllama we should have the cross attention states"),
        };

        let key_states = repeat_kv(key_states, self.num_key_value_groups)?;
        let value_states = repeat_kv(value_states, self.num_key_value_groups)?;

        let key_states = self.k_norm.forward(&key_states)?;

        let att = scaled_dot_product_attention(
            &query_states,
            &key_states,
            &value_states,
            attention_mask,
        )?;

        let hidden_states = candle_nn::ops::softmax_last_dim(&att)?.matmul(&value_states)?;
        Ok(hidden_states)
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let dim = q.dim(D::Minus1)?;
    let scale_factor = 1.0 / (dim as f64).sqrt();
    let attn_weights = (q.matmul(&k.t()?)? * scale_factor)?;

    let attn_weights = match mask {
        None => attn_weights,
        Some(mask) => masked_fill(&attn_weights, mask, f32::NEG_INFINITY)?,
    };
    let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

    Ok(attn_weights.matmul(&v)?)
}

pub struct MllamaTextMLP {
    hidden_size: usize,
    intermediate_size: usize,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}
impl MllamaTextMLP {
    pub fn new(vb: VarBuilder, cfg: &MllamaTextConfig) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let intermediate_size = cfg.intermediate_size;
        let gate_proj = linear(hidden_size, intermediate_size, vb.pp("mlp.gate_proj"))?;
        let up_proj = linear(hidden_size, intermediate_size, vb.pp("mlp.up_proj"))?;
        let down_proj = linear(intermediate_size, hidden_size, vb.pp("mlp.down_proj"))?;

        Ok(Self {
            hidden_size,
            intermediate_size,
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let down_proj = self
            .down_proj
            .forward(&self.gate_proj.forward(x)?.silu()?)?;
        Ok(down_proj)
    }
}

pub struct MllamaCrossAttentionDecoderLayer {
    cross_attn: MllamaTextCrossSdpaAttention,
    input_layernorm: MllamaTextRMSNorm,
    cross_attn_attn_gate: Tensor,
    mlp: MllamaTextMLP,
    post_attention_layernorm: MllamaTextRMSNorm,
    cross_attn_mlp_gate: Tensor,
}
impl MllamaCrossAttentionDecoderLayer {
    pub fn new(vb: VarBuilder, cfg: &MllamaTextConfig) -> Result<Self> {
        let cross_attn = MllamaTextCrossSdpaAttention::new(vb.pp("cross_attn"), cfg)?;

        let input_layernorm = MllamaTextRMSNorm::new(
            vb.pp("input_layernorm"),
            cfg.hidden_size,
            Some(cfg.rms_norm_eps),
        )?;
        let cross_attn_attn_gate = vb.get(1, "cross_attn_attn_gate")?;

        let mlp = MllamaTextMLP::new(vb.pp("mlp"), cfg)?;
        let post_attention_layernorm = MllamaTextRMSNorm::new(
            vb.pp("post_attention_layernorm"),
            cfg.hidden_size,
            Some(cfg.rms_norm_eps),
        )?;
        let cross_attn_mlp_gate = vb.get(1, "cross_attn_mlp_gate")?;

        Ok(Self {
            cross_attn,
            input_layernorm,
            cross_attn_attn_gate,
            mlp,
            post_attention_layernorm,
            cross_attn_mlp_gate,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        cross_attention_states: &Tensor,
        cross_attention_mask: &Tensor,
        full_text_row_masked_out_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = hidden_states.clone();

        let hidden_states = self.input_layernorm.forward(&hidden_states)?;

        let hidden_states = self.cross_attn.forward(
            &hidden_states,
            Some(&cross_attention_states),
            Some(&cross_attention_mask),
        )?;

        let hidden_states = (residual + self.cross_attn_attn_gate.tanh()? * hidden_states)?;

        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;

        let hidden_states = match full_text_row_masked_out_mask {
            Some(full_text_row_masked_out_mask) => {
                let mask = full_text_row_masked_out_mask
                    .i((.., 0))?
                    .broadcast_as(hidden_states.shape())?;
                (mask * hidden_states)?
            }
            None => hidden_states,
        };

        let hidden_states = (residual + self.cross_attn_mlp_gate.tanh()? * hidden_states)?;
        Ok(hidden_states)
    }
}
