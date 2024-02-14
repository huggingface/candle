#![allow(unused)]
use candle::{DType, Module, Result, Tensor, D};
use candle_nn::{conv1d, embedding, linear, Conv1d, Conv1dConfig, Embedding, Linear, VarBuilder};

pub struct AdaLayerNorm {
    eps: f64,
    dim: usize,
    scale: Embedding,
    shift: Embedding,
}

fn layer_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let x_dtype = x.dtype();
    let internal_dtype = match x_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        d => d,
    };
    let hidden_size = x.dim(D::Minus1)?;
    let x = x.to_dtype(internal_dtype)?;
    let x = {
        let mean_x = (x.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        x.broadcast_sub(&mean_x)?
    };
    let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
    let x_normed = x.broadcast_div(&(norm_x + eps)?.sqrt()?)?;
    x_normed.to_dtype(x_dtype)
}

impl AdaLayerNorm {
    pub fn new(
        num_embeddings: usize,
        embedding_dim: usize,
        eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let scale = embedding(num_embeddings, embedding_dim, vb.pp("scale"))?;
        let shift = embedding(num_embeddings, embedding_dim, vb.pp("shift"))?;
        Ok(Self {
            eps,
            dim: embedding_dim,
            scale,
            shift,
        })
    }

    pub fn forward(&self, xs: &Tensor, cond_embedding_id: &Tensor) -> Result<Tensor> {
        let scale = self.scale.forward(cond_embedding_id)?;
        let shift = self.shift.forward(cond_embedding_id)?;
        let xs = layer_norm(xs, self.eps)?;
        xs * scale + shift
    }
}

pub struct ConvNeXtBlock {
    dwconv: Conv1d,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Option<Tensor>,
}

impl ConvNeXtBlock {
    pub fn new(
        dim: usize,
        intermediate_dim: usize,
        layer_scale_init_value: f64,
        adanorm_num_embeddings: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let dwconv = {
            let cfg = Conv1dConfig {
                padding: 3,
                groups: dim,
                ..Default::default()
            };
            conv1d(dim, dim, 7, cfg, vb.pp("dwconv"))?
        };
        let pwconv1 = linear(dim, intermediate_dim, vb.pp("pwconv1"))?;
        let pwconv2 = linear(intermediate_dim, dim, vb.pp("pwconv2"))?;
        let gamma = if layer_scale_init_value > 0. {
            Some(vb.get(dim, "gamma")?)
        } else {
            None
        };
        Ok(Self {
            dwconv,
            pwconv1,
            pwconv2,
            gamma,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = xs.apply(&self.dwconv)?.transpose(1, 2)?;
        // TODO: norm
        let xs = xs.apply(&self.pwconv1)?.gelu()?.apply(&self.pwconv2)?;
        let xs = match self.gamma.as_ref() {
            Some(gamma) => (gamma * xs)?,
            None => xs,
        };
        xs.transpose(1, 2)? + residual
    }
}

struct VocosBackbone {
    embed: Conv1d,
    convnext: Vec<ConvNeXtBlock>,
    final_layer_norm: candle_nn::LayerNorm,
}

impl VocosBackbone {
    pub fn new(
        input_channels: usize,
        dim: usize,
        intermediate_dim: usize,
        num_layers: dim,
        layer_scale_init_value: f64,
        adanorm_num_embeddings: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let embed = {
            let cfg = Conv1dConfig {
                padding: 3,
                ..Default::default()
            };
            conv1d(input_channels, dim, 7, cfg, vb.pp("embed"))?
        };
        let mut convnext = Vec::with_capacity(num_layers);
        let vb_c = vb.pp("convnext");
        for i in 0..num_layers {
            let block = ConvNeXtBlock::new(
                dim,
                intermediate_dim,
                layer_scale_init_value,
                adanorm_num_embeddings,
                vb_c.pp(i),
            )?;
        }
        let final_layer_norm = candle_nn::layer_norm(dim, 1e-6, vb.pp("final_layer_norm"))?;
        Ok(Self {
            embed,
            convnext,
            final_layer_norm,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.apply(&self.embed)?;
        // TODO: norm
        let mut xs = xs.transpose(1, 2)?;
        for conv_block in self.convnext.iter() {
            xs = conv_block.forward(&xs)?
        }
        xs.apply(&self.final_layer_norm)
    }
}
