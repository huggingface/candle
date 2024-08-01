//! Attention Based Building Blocks
use candle::{DType, IndexOp, Result, Tensor, D};
use candle_nn as nn;
use candle_nn::Module;

#[derive(Debug)]
struct GeGlu {
    proj: nn::Linear,
    span: tracing::Span,
}

impl GeGlu {
    fn new(vs: nn::VarBuilder, dim_in: usize, dim_out: usize) -> Result<Self> {
        let proj = nn::linear(dim_in, dim_out * 2, vs.pp("proj"))?;
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

/// A feed-forward layer.
#[derive(Debug)]
struct FeedForward {
    project_in: GeGlu,
    linear: nn::Linear,
    span: tracing::Span,
}

impl FeedForward {
    // The glu parameter in the python code is unused?
    // https://github.com/huggingface/diffusers/blob/d3d22ce5a894becb951eec03e663951b28d45135/src/diffusers/models/attention.py#L347
    /// Creates a new feed-forward layer based on some given input dimension, some
    /// output dimension, and a multiplier to be used for the intermediary layer.
    fn new(vs: nn::VarBuilder, dim: usize, dim_out: Option<usize>, mult: usize) -> Result<Self> {
        let inner_dim = dim * mult;
        let dim_out = dim_out.unwrap_or(dim);
        let vs = vs.pp("net");
        let project_in = GeGlu::new(vs.pp("0"), dim, inner_dim)?;
        let linear = nn::linear(inner_dim, dim_out, vs.pp("2"))?;
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

#[derive(Debug)]
pub struct CrossAttention {
    to_q: nn::Linear,
    to_k: nn::Linear,
    to_v: nn::Linear,
    to_out: nn::Linear,
    heads: usize,
    scale: f64,
    slice_size: Option<usize>,
    span: tracing::Span,
    span_attn: tracing::Span,
    span_softmax: tracing::Span,
    use_flash_attn: bool,
}

impl CrossAttention {
    // Defaults should be heads = 8, dim_head = 64, context_dim = None
    pub fn new(
        vs: nn::VarBuilder,
        query_dim: usize,
        context_dim: Option<usize>,
        heads: usize,
        dim_head: usize,
        slice_size: Option<usize>,
        use_flash_attn: bool,
    ) -> Result<Self> {
        let inner_dim = dim_head * heads;
        let context_dim = context_dim.unwrap_or(query_dim);
        let scale = 1.0 / f64::sqrt(dim_head as f64);
        let to_q = nn::linear_no_bias(query_dim, inner_dim, vs.pp("to_q"))?;
        let to_k = nn::linear_no_bias(context_dim, inner_dim, vs.pp("to_k"))?;
        let to_v = nn::linear_no_bias(context_dim, inner_dim, vs.pp("to_v"))?;
        let to_out = nn::linear(inner_dim, query_dim, vs.pp("to_out.0"))?;
        let span = tracing::span!(tracing::Level::TRACE, "xa");
        let span_attn = tracing::span!(tracing::Level::TRACE, "xa-attn");
        let span_softmax = tracing::span!(tracing::Level::TRACE, "xa-softmax");
        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            heads,
            scale,
            slice_size,
            span,
            span_attn,
            span_softmax,
            use_flash_attn,
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

    fn sliced_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        slice_size: usize,
    ) -> Result<Tensor> {
        let batch_size_attention = query.dim(0)?;
        let mut hidden_states = Vec::with_capacity(batch_size_attention / slice_size);
        let in_dtype = query.dtype();
        let query = query.to_dtype(DType::F32)?;
        let key = key.to_dtype(DType::F32)?;
        let value = value.to_dtype(DType::F32)?;

        for i in 0..batch_size_attention / slice_size {
            let start_idx = i * slice_size;
            let end_idx = (i + 1) * slice_size;

            let xs = query
                .i(start_idx..end_idx)?
                .matmul(&(key.i(start_idx..end_idx)?.t()? * self.scale)?)?;
            let xs = nn::ops::softmax(&xs, D::Minus1)?.matmul(&value.i(start_idx..end_idx)?)?;
            hidden_states.push(xs)
        }
        let hidden_states = Tensor::stack(&hidden_states, 0)?.to_dtype(in_dtype)?;
        self.reshape_batch_dim_to_heads(&hidden_states)
    }

    fn attention(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let xs = if self.use_flash_attn {
            let init_dtype = query.dtype();
            let q = query
                .to_dtype(candle::DType::F16)?
                .unsqueeze(0)?
                .transpose(1, 2)?;
            let k = key
                .to_dtype(candle::DType::F16)?
                .unsqueeze(0)?
                .transpose(1, 2)?;
            let v = value
                .to_dtype(candle::DType::F16)?
                .unsqueeze(0)?
                .transpose(1, 2)?;
            flash_attn(&q, &k, &v, self.scale as f32, false)?
                .transpose(1, 2)?
                .squeeze(0)?
                .to_dtype(init_dtype)?
        } else {
            let in_dtype = query.dtype();
            let query = query.to_dtype(DType::F32)?;
            let key = key.to_dtype(DType::F32)?;
            let value = value.to_dtype(DType::F32)?;
            let xs = query.matmul(&(key.t()? * self.scale)?)?;
            let xs = {
                let _enter = self.span_softmax.enter();
                nn::ops::softmax_last_dim(&xs)?
            };
            xs.matmul(&value)?.to_dtype(in_dtype)?
        };
        self.reshape_batch_dim_to_heads(&xs)
    }

    pub fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();
        let query = self.to_q.forward(xs)?;
        let context = context.unwrap_or(xs).contiguous()?;
        let key = self.to_k.forward(&context)?;
        let value = self.to_v.forward(&context)?;
        let query = self.reshape_heads_to_batch_dim(&query)?;
        let key = self.reshape_heads_to_batch_dim(&key)?;
        let value = self.reshape_heads_to_batch_dim(&value)?;
        let dim0 = query.dim(0)?;
        let slice_size = self.slice_size.and_then(|slice_size| {
            if dim0 < slice_size {
                None
            } else {
                Some(slice_size)
            }
        });
        let xs = match slice_size {
            None => self.attention(&query, &key, &value)?,
            Some(slice_size) => self.sliced_attention(&query, &key, &value, slice_size)?,
        };
        self.to_out.forward(&xs)
    }
}

/// A basic Transformer block.
#[derive(Debug)]
struct BasicTransformerBlock {
    attn1: CrossAttention,
    ff: FeedForward,
    attn2: CrossAttention,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
    norm3: nn::LayerNorm,
    span: tracing::Span,
}

impl BasicTransformerBlock {
    fn new(
        vs: nn::VarBuilder,
        dim: usize,
        n_heads: usize,
        d_head: usize,
        context_dim: Option<usize>,
        sliced_attention_size: Option<usize>,
        use_flash_attn: bool,
    ) -> Result<Self> {
        let attn1 = CrossAttention::new(
            vs.pp("attn1"),
            dim,
            None,
            n_heads,
            d_head,
            sliced_attention_size,
            use_flash_attn,
        )?;
        let ff = FeedForward::new(vs.pp("ff"), dim, None, 4)?;
        let attn2 = CrossAttention::new(
            vs.pp("attn2"),
            dim,
            context_dim,
            n_heads,
            d_head,
            sliced_attention_size,
            use_flash_attn,
        )?;
        let norm1 = nn::layer_norm(dim, 1e-5, vs.pp("norm1"))?;
        let norm2 = nn::layer_norm(dim, 1e-5, vs.pp("norm2"))?;
        let norm3 = nn::layer_norm(dim, 1e-5, vs.pp("norm3"))?;
        let span = tracing::span!(tracing::Level::TRACE, "basic-transformer");
        Ok(Self {
            attn1,
            ff,
            attn2,
            norm1,
            norm2,
            norm3,
            span,
        })
    }

    fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();
        let xs = (self.attn1.forward(&self.norm1.forward(xs)?, None)? + xs)?;
        let xs = (self.attn2.forward(&self.norm2.forward(&xs)?, context)? + xs)?;
        self.ff.forward(&self.norm3.forward(&xs)?)? + xs
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SpatialTransformerConfig {
    pub depth: usize,
    pub num_groups: usize,
    pub context_dim: Option<usize>,
    pub sliced_attention_size: Option<usize>,
    pub use_linear_projection: bool,
}

impl Default for SpatialTransformerConfig {
    fn default() -> Self {
        Self {
            depth: 1,
            num_groups: 32,
            context_dim: None,
            sliced_attention_size: None,
            use_linear_projection: false,
        }
    }
}

#[derive(Debug)]
enum Proj {
    Conv2d(nn::Conv2d),
    Linear(nn::Linear),
}

// Aka Transformer2DModel
#[derive(Debug)]
pub struct SpatialTransformer {
    norm: nn::GroupNorm,
    proj_in: Proj,
    transformer_blocks: Vec<BasicTransformerBlock>,
    proj_out: Proj,
    span: tracing::Span,
    pub config: SpatialTransformerConfig,
}

impl SpatialTransformer {
    pub fn new(
        vs: nn::VarBuilder,
        in_channels: usize,
        n_heads: usize,
        d_head: usize,
        use_flash_attn: bool,
        config: SpatialTransformerConfig,
    ) -> Result<Self> {
        let inner_dim = n_heads * d_head;
        let norm = nn::group_norm(config.num_groups, in_channels, 1e-6, vs.pp("norm"))?;
        let proj_in = if config.use_linear_projection {
            Proj::Linear(nn::linear(in_channels, inner_dim, vs.pp("proj_in"))?)
        } else {
            Proj::Conv2d(nn::conv2d(
                in_channels,
                inner_dim,
                1,
                Default::default(),
                vs.pp("proj_in"),
            )?)
        };
        let mut transformer_blocks = vec![];
        let vs_tb = vs.pp("transformer_blocks");
        for index in 0..config.depth {
            let tb = BasicTransformerBlock::new(
                vs_tb.pp(&index.to_string()),
                inner_dim,
                n_heads,
                d_head,
                config.context_dim,
                config.sliced_attention_size,
                use_flash_attn,
            )?;
            transformer_blocks.push(tb)
        }
        let proj_out = if config.use_linear_projection {
            Proj::Linear(nn::linear(in_channels, inner_dim, vs.pp("proj_out"))?)
        } else {
            Proj::Conv2d(nn::conv2d(
                inner_dim,
                in_channels,
                1,
                Default::default(),
                vs.pp("proj_out"),
            )?)
        };
        let span = tracing::span!(tracing::Level::TRACE, "spatial-transformer");
        Ok(Self {
            norm,
            proj_in,
            transformer_blocks,
            proj_out,
            span,
            config,
        })
    }

    pub fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (batch, _channel, height, weight) = xs.dims4()?;
        let residual = xs;
        let xs = self.norm.forward(xs)?;
        let (inner_dim, xs) = match &self.proj_in {
            Proj::Conv2d(p) => {
                let xs = p.forward(&xs)?;
                let inner_dim = xs.dim(1)?;
                let xs = xs
                    .transpose(1, 2)?
                    .t()?
                    .reshape((batch, height * weight, inner_dim))?;
                (inner_dim, xs)
            }
            Proj::Linear(p) => {
                let inner_dim = xs.dim(1)?;
                let xs = xs
                    .transpose(1, 2)?
                    .t()?
                    .reshape((batch, height * weight, inner_dim))?;
                (inner_dim, p.forward(&xs)?)
            }
        };
        let mut xs = xs;
        for block in self.transformer_blocks.iter() {
            xs = block.forward(&xs, context)?
        }
        let xs = match &self.proj_out {
            Proj::Conv2d(p) => p.forward(
                &xs.reshape((batch, height, weight, inner_dim))?
                    .t()?
                    .transpose(1, 2)?,
            )?,
            Proj::Linear(p) => p
                .forward(&xs)?
                .reshape((batch, height, weight, inner_dim))?
                .t()?
                .transpose(1, 2)?,
        };
        xs + residual
    }
}

/// Configuration for an attention block.
#[derive(Debug, Clone, Copy)]
pub struct AttentionBlockConfig {
    pub num_head_channels: Option<usize>,
    pub num_groups: usize,
    pub rescale_output_factor: f64,
    pub eps: f64,
}

impl Default for AttentionBlockConfig {
    fn default() -> Self {
        Self {
            num_head_channels: None,
            num_groups: 32,
            rescale_output_factor: 1.,
            eps: 1e-5,
        }
    }
}

#[derive(Debug)]
pub struct AttentionBlock {
    group_norm: nn::GroupNorm,
    query: nn::Linear,
    key: nn::Linear,
    value: nn::Linear,
    proj_attn: nn::Linear,
    channels: usize,
    num_heads: usize,
    span: tracing::Span,
    config: AttentionBlockConfig,
}

impl AttentionBlock {
    pub fn new(vs: nn::VarBuilder, channels: usize, config: AttentionBlockConfig) -> Result<Self> {
        let num_head_channels = config.num_head_channels.unwrap_or(channels);
        let num_heads = channels / num_head_channels;
        let group_norm =
            nn::group_norm(config.num_groups, channels, config.eps, vs.pp("group_norm"))?;
        let (q_path, k_path, v_path, out_path) = if vs.contains_tensor("to_q.weight") {
            ("to_q", "to_k", "to_v", "to_out.0")
        } else {
            ("query", "key", "value", "proj_attn")
        };
        let query = nn::linear(channels, channels, vs.pp(q_path))?;
        let key = nn::linear(channels, channels, vs.pp(k_path))?;
        let value = nn::linear(channels, channels, vs.pp(v_path))?;
        let proj_attn = nn::linear(channels, channels, vs.pp(out_path))?;
        let span = tracing::span!(tracing::Level::TRACE, "attn-block");
        Ok(Self {
            group_norm,
            query,
            key,
            value,
            proj_attn,
            channels,
            num_heads,
            span,
            config,
        })
    }

    fn transpose_for_scores(&self, xs: Tensor) -> Result<Tensor> {
        let (batch, t, h_times_d) = xs.dims3()?;
        xs.reshape((batch, t, self.num_heads, h_times_d / self.num_heads))?
            .transpose(1, 2)
    }
}

impl Module for AttentionBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let in_dtype = xs.dtype();
        let residual = xs;
        let (batch, channel, height, width) = xs.dims4()?;
        let xs = self
            .group_norm
            .forward(xs)?
            .reshape((batch, channel, height * width))?
            .transpose(1, 2)?;

        let query_proj = self.query.forward(&xs)?;
        let key_proj = self.key.forward(&xs)?;
        let value_proj = self.value.forward(&xs)?;

        let query_states = self
            .transpose_for_scores(query_proj)?
            .to_dtype(DType::F32)?;
        let key_states = self.transpose_for_scores(key_proj)?.to_dtype(DType::F32)?;
        let value_states = self
            .transpose_for_scores(value_proj)?
            .to_dtype(DType::F32)?;

        // scale is applied twice, hence the -0.25 here rather than -0.5.
        // https://github.com/huggingface/diffusers/blob/d3d22ce5a894becb951eec03e663951b28d45135/src/diffusers/models/attention.py#L87
        let scale = f64::powf(self.channels as f64 / self.num_heads as f64, -0.25);
        let attention_scores = (query_states * scale)?.matmul(&(key_states.t()? * scale)?)?;
        let attention_probs = nn::ops::softmax(&attention_scores, D::Minus1)?;

        // TODO: revert the call to force_contiguous once the three matmul kernels have been
        // adapted to handle layout with some dims set to 1.
        let xs = attention_probs.matmul(&value_states)?;
        let xs = xs.to_dtype(in_dtype)?;
        let xs = xs.transpose(1, 2)?.contiguous()?;
        let xs = xs.flatten_from(D::Minus2)?;
        let xs = self
            .proj_attn
            .forward(&xs)?
            .t()?
            .reshape((batch, channel, height, width))?;
        (xs + residual)? / self.config.rescale_output_factor
    }
}
