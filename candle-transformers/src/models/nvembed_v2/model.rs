// Copyright (c) NVIDIA CORPORATION, all rights reserved.
// This source code is licensed under the CC-BY-NC-4.0 license.
// See https://spdx.org/licenses/CC-BY-NC-4.0 for details.

use super::decoder::Model as MistralModel;
use crate::models::{
    mistral::Config,
    with_tracing::{layer_norm, linear, linear_no_bias, LayerNorm, Linear},
};
use candle::{DType, Device, Result, Tensor, D};
use candle_nn::{ops::softmax_last_dim, Module, VarBuilder};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
struct LatentAttentionConfig {
    num_latents_value: usize,
    num_cross_heads: usize,
    output_normalize: bool,
    hidden_dim: usize,
    latent_dim: usize,
    cross_dim_head: usize,
    hidden_size: usize,
}

impl LatentAttentionConfig {
    fn new(hidden_size: usize, output_normalize: bool) -> Self {
        Self {
            num_latents_value: 512,
            num_cross_heads: 8,
            output_normalize,
            hidden_dim: 4096,
            latent_dim: 4096,
            cross_dim_head: 4096,
            hidden_size,
        }
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct GEGLU {}

impl GEGLU {
    fn new() -> Self {
        Self {}
    }
}
impl Module for GEGLU {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let last_dim = x.dims().len() - 1;
        let chunks = x.chunk(2, last_dim)?;
        let (x, gates) = (chunks[0].clone(), chunks[1].clone());

        let gates = gates.gelu()?;

        x * gates
    }
}

#[derive(Debug, Clone)]
struct FeedForward {
    linear1: Linear,
    gelu: GEGLU,
    linear2: Linear,
}

impl FeedForward {
    fn new(dim: usize, vb1: VarBuilder, vb2: VarBuilder) -> Result<Self> {
        let linear1 = linear(dim, dim * 4 * 2, vb1)?;
        let gelu = GEGLU::new();
        let linear2 = linear(dim * 4, dim, vb2)?;

        Ok(Self {
            linear1,
            gelu,
            linear2,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.linear1.forward(xs)?;
        let xs = self.gelu.forward(&xs)?;
        let xs = self.linear2.forward(&xs)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    heads: usize,
    to_q: Linear,
    to_kv: Linear,
    to_out: Linear,
    dim_head: usize,
}

#[allow(clippy::too_many_arguments)]
impl Attention {
    fn new(
        query_dim: usize,
        context_dim: Option<usize>,
        heads: Option<usize>,
        dim_head: Option<usize>,
        vb_to_q: VarBuilder,
        vb_to_kv: VarBuilder,
        vb_to_out: VarBuilder,
    ) -> Result<Self> {
        let heads = heads.unwrap_or(8);
        let dim_head = dim_head.unwrap_or(64);
        let inner_dim = dim_head * heads;
        let context_dim = context_dim.unwrap_or(query_dim);

        let to_q = linear_no_bias(query_dim, inner_dim, vb_to_q)?;
        let to_kv = linear_no_bias(context_dim, inner_dim * 2, vb_to_kv)?;
        let to_out = linear_no_bias(inner_dim, query_dim, vb_to_out)?;
        Ok(Self {
            heads,
            to_q,
            to_kv,
            to_out,
            dim_head,
        })
    }

    // Cross attn takes queries from the mistral decoder and kv from latent attention model
    fn forward(&self, x: &Tensor, context: &Tensor) -> Result<Tensor> {
        let h = self.heads;
        let q = self.to_q.forward(x)?;
        let kv_chunks = self
            .to_kv
            .forward(context)?
            .chunk(2, context.shape().dims().len() - 1)?;
        let (k, v) = (kv_chunks[0].clone(), kv_chunks[1].clone());

        let (b_sz, q_len, _) = q.dims3()?;
        let q = q
            .reshape((b_sz, q_len, h, self.dim_head))?
            .transpose(1, 2)?
            .contiguous()?;

        let (_, q_len, _) = k.dims3()?;
        let k = k
            .reshape((b_sz, q_len, h, self.dim_head))?
            .transpose(1, 2)?
            .contiguous()?;

        let (_, q_len, _) = v.dims3()?;
        let v = v
            .reshape((b_sz, q_len, h, self.dim_head))?
            .transpose(1, 2)?
            .contiguous()?;

        let scale = 1f64 / f64::sqrt(self.dim_head as f64);

        let attn_weight = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let attn_weight = softmax_last_dim(&attn_weight)?;

        let out = attn_weight.matmul(&v)?;

        let (_, _, q_len, _) = out.dims4()?;
        let out = out
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.dim_head * h))?;

        self.to_out.forward(&out)
    }
}

#[derive(Debug, Clone)]
enum PreNormInnerLayer {
    Attention(Attention),
    FeedForward(FeedForward),
}

#[derive(Debug, Clone)]
struct PreNorm {
    norm: LayerNorm,
    norm_context: Option<LayerNorm>,
    inner_layer: PreNormInnerLayer,
}

impl PreNorm {
    fn new(
        dim: usize,
        context_dim: Option<usize>,
        inner_layer: PreNormInnerLayer,
        norm_vb: VarBuilder,
        norm_context_vb: Option<VarBuilder>,
    ) -> Result<Self> {
        let norm = layer_norm(dim, candle_nn::LayerNormConfig::default(), norm_vb)?;

        let norm_context = match context_dim {
            Some(context_dim) => {
                let norm_context_vb = norm_context_vb
                    .expect("norm_context_vb must be passed if context_dim is passed");
                match layer_norm(
                    context_dim,
                    candle_nn::LayerNormConfig::default(),
                    norm_context_vb,
                ) {
                    Ok(norm_context) => Some(norm_context),
                    Err(e) => return Err(e),
                }
            }
            None => None,
        };
        Ok(Self {
            norm,
            norm_context,
            inner_layer,
        })
    }

    // Applies a layernorm to the input before passing to cross attn or feed forward
    fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let xs = self.norm.forward(xs)?;

        let mut normed_context = None;
        if let Some(norm_context) = &self.norm_context {
            if let Some(context) = context {
                normed_context = Some(norm_context.forward(context)?);
            }
        }

        match &self.inner_layer {
            PreNormInnerLayer::Attention(attn) => attn.forward(&xs, &normed_context.unwrap()),
            PreNormInnerLayer::FeedForward(ff) => ff.forward(&xs),
        }
    }
}

#[derive(Debug, Clone)]
struct LatentAttentionModel {
    cross_attn: PreNorm,
    ff: PreNorm,
    output_normalize: bool,
    latents: Tensor,
}

impl LatentAttentionModel {
    fn new(vb: VarBuilder, config: LatentAttentionConfig) -> Result<Self> {
        let vb_cross = vb.pp("cross_attend_blocks");

        let num_latents = config.num_latents_value;
        let latent_dim = config.latent_dim;
        let cross_heads = config.num_cross_heads;
        let cross_dim_head = config.cross_dim_head;
        let dim = config.hidden_dim;
        let hidden_size = config.hidden_size;

        let cross_attn = PreNorm::new(
            latent_dim,
            Some(hidden_size),
            PreNormInnerLayer::Attention(Attention::new(
                latent_dim,
                Some(dim),
                Some(cross_heads),
                Some(cross_dim_head),
                vb_cross.pp("0.fn.to_q"),
                vb_cross.pp("0.fn.to_kv"),
                vb_cross.pp("0.fn.to_out"),
            )?),
            vb_cross.pp("0.norm"),
            Some(vb_cross.pp("0.norm_context")),
        )?;

        let ff = PreNorm::new(
            latent_dim,
            None,
            PreNormInnerLayer::FeedForward(FeedForward::new(
                latent_dim,
                vb_cross.pp("1.fn.net.0"),
                vb_cross.pp("1.fn.net.2"),
            )?),
            vb_cross.pp("1.norm"),
            None,
        )?;

        let output_normalize = config.output_normalize;
        let latents = vb.get((num_latents, latent_dim), "latents")?;

        Ok(Self {
            cross_attn,
            ff,
            output_normalize,
            latents,
        })
    }

    fn forward(&self, hiddens: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let b = hiddens.dims()[0];
        let x = self.latents.unsqueeze(0)?.repeat((b, 1, 1))?;

        let hiddens = (self.cross_attn.forward(hiddens, Some(&x))? + hiddens)?;
        let hiddens = (self.ff.forward(&hiddens, None)? + hiddens)?;

        // Mean pooling
        let hiddens_masked = hiddens.broadcast_mul(&attention_mask.unsqueeze(D::Minus1)?)?;
        let s = hiddens_masked.sum(1)?;
        let d = attention_mask.sum_keepdim(1)?;
        let hiddens = s.broadcast_div(&d)?;

        if self.output_normalize {
            let hiddens = div_l2_norm(&hiddens)?;

            Ok(hiddens)
        } else {
            Ok(hiddens)
        }
    }
}

#[derive(Debug, Clone)]
pub struct NVEmbedModel {
    latent_attention_model: LatentAttentionModel,
    embedding_model: MistralModel,
    pub device: Device,
    pub dtype: DType,
}

impl NVEmbedModel {
    pub fn new(vb: VarBuilder, output_normalize: bool) -> Result<Self> {
        let cfg = Config::config_7b_v0_1(false);
        let embedding_model = MistralModel::new(&cfg, vb.pp("embedding_model"))?;
        let hidden_size = embedding_model.cfg.hidden_size;
        let latent_attention_model = LatentAttentionModel::new(
            vb.pp("latent_attention_model"),
            LatentAttentionConfig::new(hidden_size, output_normalize),
        )?;

        Ok(Self {
            latent_attention_model,
            embedding_model,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        attn_mask: &Tensor,
        pool_mask: &Tensor,
    ) -> Result<Tensor> {
        let outputs = self
            .embedding_model
            .forward(attn_mask, input_ids, self.dtype)?;

        self.latent_attention_model.forward(&outputs, pool_mask)
    }
}

fn div_l2_norm(v: &Tensor) -> Result<Tensor> {
    let l2_norm = v.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    v.broadcast_div(&l2_norm)
}
