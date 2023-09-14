use candle::{Module, Result, Tensor, D};
use candle_nn::VarBuilder;

// https://github.com/huggingface/diffusers/blob/19edca82f1ff194c07317369a92b470dbae97f34/src/diffusers/pipelines/wuerstchen/modeling_wuerstchen_common.py#L22
#[derive(Debug)]
pub struct WLayerNorm {
    inner: candle_nn::LayerNorm,
}

impl WLayerNorm {
    pub fn new(size: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = candle_nn::layer_norm::LayerNormConfig {
            eps: 1e-6,
            remove_mean: true,
            affine: false,
        };
        let inner = candle_nn::layer_norm(size, cfg, vb)?;
        Ok(Self { inner })
    }
}

impl Module for WLayerNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.permute((0, 2, 3, 1))?
            .apply(&self.inner)?
            .permute((0, 3, 1, 2))
    }
}

#[derive(Debug)]
pub struct TimestepBlock {
    mapper: candle_nn::Linear,
}

impl TimestepBlock {
    pub fn new(c: usize, c_timestep: usize, vb: VarBuilder) -> Result<Self> {
        let mapper = candle_nn::linear(c_timestep, c * 2, vb.pp("mapper"))?;
        Ok(Self { mapper })
    }

    pub fn forward(&self, xs: &Tensor, t: &Tensor) -> Result<Tensor> {
        let ab = self
            .mapper
            .forward(t)?
            .unsqueeze(2)?
            .unsqueeze(3)?
            .chunk(2, 1)?;
        xs.broadcast_mul(&(&ab[0] + 1.)?)?.broadcast_add(&ab[1])
    }
}

#[derive(Debug)]
pub struct GlobalResponseNorm {
    gamma: Tensor,
    beta: Tensor,
}

impl GlobalResponseNorm {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let gamma = vb.get((1, 1, 1, 1, dim), "gamma")?;
        let beta = vb.get((1, 1, 1, 1, dim), "beta")?;
        Ok(Self { gamma, beta })
    }
}

impl Module for GlobalResponseNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let agg_norm = xs.sqr()?.sum_keepdim((1, 2))?;
        let stand_div_norm =
            agg_norm.broadcast_div(&(agg_norm.mean_keepdim(D::Minus1)? + 1e-6)?)?;
        (xs.broadcast_mul(&stand_div_norm)?
            .broadcast_mul(&self.gamma)
            + &self.beta)?
            + xs
    }
}

#[derive(Debug)]
pub struct ResBlock {
    depthwise: candle_nn::Conv2d,
    norm: WLayerNorm,
    channelwise_lin1: candle_nn::Linear,
    channelwise_grn: GlobalResponseNorm,
    channelwise_lin2: candle_nn::Linear,
}

impl ResBlock {
    pub fn new(c: usize, c_skip: usize, ksize: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = candle_nn::Conv2dConfig {
            padding: ksize / 2,
            groups: c,
            ..Default::default()
        };
        let depthwise = candle_nn::conv2d(c + c_skip, c, ksize, cfg, vb.pp("depthwise"))?;
        let norm = WLayerNorm::new(c, vb.pp("norm"))?;
        let channelwise_lin1 = candle_nn::linear(c, c * 4, vb.pp("channelwise.0"))?;
        let channelwise_grn = GlobalResponseNorm::new(c * 4, vb.pp("channelwise.2"))?;
        let channelwise_lin2 = candle_nn::linear(c * 4, c, vb.pp("channelwise.4"))?;
        Ok(Self {
            depthwise,
            norm,
            channelwise_lin1,
            channelwise_grn,
            channelwise_lin2,
        })
    }

    pub fn forward(&self, xs: &Tensor, x_skip: Option<&Tensor>) -> Result<Tensor> {
        let x_res = xs;
        let xs = match x_skip {
            None => xs.clone(),
            Some(x_skip) => Tensor::cat(&[xs, x_skip], 1)?,
        };
        let xs = xs
            .apply(&self.depthwise)?
            .apply(&self.norm)?
            .permute((0, 2, 3, 1))?;
        let xs = xs
            .apply(&self.channelwise_lin1)?
            .gelu()?
            .apply(&self.channelwise_grn)?
            .apply(&self.channelwise_lin2)?
            .permute((0, 3, 1, 2))?;
        xs + x_res
    }
}
