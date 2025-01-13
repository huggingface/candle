use super::embedding::Model as EmbeddingModel;
use crate::models::{
    mistral::Config,
    with_tracing::{layer_norm, linear, linear_no_bias, LayerNorm, Linear},
};
use candle::{DType, Device, Result, Tensor, D};
use candle_nn::{ops::softmax_last_dim, LayerNormConfig, Module, VarBuilder};

// Geglu and feedforward from candle-transformers/src/models/stable_diffusion/attention.rs
#[derive(Debug)]
struct GeGlu {
    proj: Linear,
    span: tracing::Span,
}

impl GeGlu {
    fn new(vs: VarBuilder, dim_in: usize, dim_out: usize) -> Result<Self> {
        let proj = linear(dim_in, dim_out * 2, vs)?;
        let span = tracing::span!(tracing::Level::TRACE, "geglu");
        Ok(Self { proj, span })
    }
}

impl Module for GeGlu {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states_and_gate = self.proj.forward(xs)?.chunk(2, D::Minus1)?;
        &hidden_states_and_gate[0] * hidden_states_and_gate[1].gelu()?
    }
}

#[derive(Debug)]
struct FeedForward {
    project_in: GeGlu,
    linear: Linear,
    span: tracing::Span,
}

impl FeedForward {
    fn new(vs: VarBuilder, dim: usize, dim_out: Option<usize>, mult: usize) -> Result<Self> {
        let inner_dim = dim * mult;
        let dim_out = dim_out.unwrap_or(dim);
        let vs = vs.pp("net");
        let project_in = GeGlu::new(vs.pp("0"), dim, inner_dim)?;
        let linear = linear(inner_dim, dim_out, vs.pp("2"))?;
        let span = tracing::span!(tracing::Level::TRACE, "ff");
        Ok(Self {
            project_in,
            linear,
            span,
        })
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let xs = self.project_in.forward(xs)?;
        self.linear.forward(&xs)
    }
}

// CrossAttention from candle-transformers/src/models/stable_diffusion/attention.rs
#[derive(Debug)]
struct CrossAttention {
    to_q: Linear,
    to_kv: Linear,
    to_out: Linear,
    heads: usize,
    scale: f64,
    span: tracing::Span,
    span_attn: tracing::Span,
    span_softmax: tracing::Span,
}

impl CrossAttention {
    fn new(
        vs: VarBuilder,
        query_dim: usize,
        context_dim: Option<usize>,
        heads: usize,
        dim_head: usize,
    ) -> Result<Self> {
        let inner_dim = dim_head * heads;
        let context_dim = context_dim.unwrap_or(query_dim);
        let scale = 1.0 / f64::sqrt(dim_head as f64);
        let to_q = linear_no_bias(query_dim, inner_dim, vs.pp("to_q"))?;
        let to_kv = linear_no_bias(context_dim, inner_dim * 2, vs.pp("to_kv"))?;
        let to_out = linear_no_bias(inner_dim, query_dim, vs.pp("to_out"))?;
        let span = tracing::span!(tracing::Level::TRACE, "xa");
        let span_attn = tracing::span!(tracing::Level::TRACE, "xa-attn");
        let span_softmax = tracing::span!(tracing::Level::TRACE, "xa-softmax");
        Ok(Self {
            to_q,
            to_kv,
            to_out,
            heads,
            scale,
            span,
            span_attn,
            span_softmax,
        })
    }

    fn reshape_heads_to_batch_dim(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, dim) = xs.dims3()?;
        xs.reshape((batch_size, seq_len, self.heads, dim / self.heads))?
            .transpose(1, 2)?
            .reshape((batch_size * self.heads, seq_len, dim / self.heads))
    }

    fn reshape_batch_dim_to_heads(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, dim) = xs.dims3()?;
        xs.reshape((batch_size / self.heads, self.heads, seq_len, dim))?
            .transpose(1, 2)?
            .reshape((batch_size / self.heads, seq_len, dim * self.heads))
    }

    fn attention(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> Result<Tensor> {
        let _enter = self.span_attn.enter();

        let in_dtype = query.dtype();
        let query = query.to_dtype(DType::F32)?;
        let key = key.to_dtype(DType::F32)?;
        let value = value.to_dtype(DType::F32)?;
        let xs = query.matmul(&(key.t()? * self.scale)?)?;
        let xs = {
            let _enter = self.span_softmax.enter();
            softmax_last_dim(&xs)?
        };
        let xs = xs.matmul(&value)?.to_dtype(in_dtype)?;

        self.reshape_batch_dim_to_heads(&xs)
    }

    fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();
        let query = self.to_q.forward(xs)?;
        let context = context.unwrap_or(xs).contiguous()?;
        let kv_chunks = self
            .to_kv
            .forward(&context)?
            .chunk(2, context.shape().dims().len() - 1)?;
        let (key, value) = (kv_chunks[0].clone(), kv_chunks[1].clone());
        let query = self.reshape_heads_to_batch_dim(&query)?;
        let key = self.reshape_heads_to_batch_dim(&key)?;
        let value = self.reshape_heads_to_batch_dim(&value)?;

        let xs = self.attention(&query, &key, &value)?;
        self.to_out.forward(&xs)
    }
}

#[derive(Debug)]
pub struct Model {
    embedding_model: EmbeddingModel,
    cross_attn: CrossAttention,
    cross_attn_norm: LayerNorm,
    cross_attn_context_norm: LayerNorm,
    ff: FeedForward,
    ff_norm: LayerNorm,
    latents: Tensor,
    pub device: Device,
    pub dtype: DType,
}

impl Model {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        // Embedding model
        let cfg = Config::config_7b_v0_1(false);
        let embedding_model = EmbeddingModel::new(&cfg, vb.pp("embedding_model"))?;

        // Latent attention
        let dim = 4096;
        let vb = vb.pp("latent_attention_model");
        let latents = vb.get((512, dim), "latents")?;

        // Cross attend blocks
        let vb = vb.pp("cross_attend_blocks");
        let cross_attn_norm = layer_norm(dim, LayerNormConfig::default(), vb.pp("0.norm"))?;
        let cross_attn_context_norm = layer_norm(
            dim,
            candle_nn::LayerNormConfig::default(),
            vb.pp("0.norm_context"),
        )?;
        let cross_attn = CrossAttention::new(vb.pp("0.fn"), dim, None, 8, 4096)?;

        let ff_norm = layer_norm(dim, LayerNormConfig::default(), vb.pp("1.norm"))?;
        let ff = FeedForward::new(vb.pp("1.fn"), dim, None, 4)?;

        Ok(Self {
            embedding_model,
            cross_attn,
            cross_attn_norm,
            cross_attn_context_norm,
            ff,
            ff_norm,
            latents,
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
        // Embedding model
        let hiddens = self
            .embedding_model
            .forward(attn_mask, input_ids, self.dtype)?;

        // Latent attention
        let b = hiddens.dims()[0];
        let x = self.latents.unsqueeze(0)?.repeat((b, 1, 1))?;
        let original_hiddens = &hiddens;

        let hiddens = self.cross_attn_norm.forward(original_hiddens)?;
        let x = self.cross_attn_context_norm.forward(&x)?;
        let cross_hiddens = (self.cross_attn.forward(&hiddens, Some(&x))? + original_hiddens)?;

        let hiddens = self.ff_norm.forward(&cross_hiddens)?;
        let hiddens = (self.ff.forward(&hiddens)? + cross_hiddens)?;

        // Mean pooling
        let hiddens_masked = hiddens.broadcast_mul(&pool_mask.unsqueeze(D::Minus1)?)?;
        let s = hiddens_masked.sum(1)?;
        let d = pool_mask.sum_keepdim(1)?;
        s.broadcast_div(&d)
    }
}
