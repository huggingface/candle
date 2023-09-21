use candle::{DType, Module, Result, Tensor, D};
use candle_nn::VarBuilder;

// https://github.com/huggingface/diffusers/blob/19edca82f1ff194c07317369a92b470dbae97f34/src/diffusers/pipelines/wuerstchen/modeling_wuerstchen_common.py#L22
#[derive(Debug)]
pub struct WLayerNorm {
    eps: f64,
}

impl WLayerNorm {
    pub fn new(_size: usize) -> Result<Self> {
        Ok(Self { eps: 1e-6 })
    }
}

impl Module for WLayerNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.permute((0, 2, 3, 1))?;

        let x_dtype = xs.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };

        let hidden_size = xs.dim(D::Minus1)?;
        let xs = xs.to_dtype(internal_dtype)?;
        let mean_x = (xs.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let xs = xs.broadcast_sub(&mean_x)?;
        let norm_x = (xs.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        xs.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?
            .to_dtype(x_dtype)?
            .permute((0, 3, 1, 2))
    }
}

#[derive(Debug)]
pub struct LayerNormNoWeights {
    eps: f64,
}

impl LayerNormNoWeights {
    pub fn new(_size: usize) -> Result<Self> {
        Ok(Self { eps: 1e-6 })
    }
}

impl Module for LayerNormNoWeights {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x_dtype = xs.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = xs.dim(D::Minus1)?;
        let xs = xs.to_dtype(internal_dtype)?;
        let mean_x = (xs.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let xs = xs.broadcast_sub(&mean_x)?;
        let norm_x = (xs.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        xs.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?
            .to_dtype(x_dtype)
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
        let gamma = vb.get((1, 1, 1, dim), "gamma")?;
        let beta = vb.get((1, 1, 1, dim), "beta")?;
        Ok(Self { gamma, beta })
    }
}

impl Module for GlobalResponseNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let agg_norm = xs.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
        let stand_div_norm =
            agg_norm.broadcast_div(&(agg_norm.mean_keepdim(D::Minus1)? + 1e-6)?)?;
        xs.broadcast_mul(&stand_div_norm)?
            .broadcast_mul(&self.gamma)?
            .broadcast_add(&self.beta)?
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
        let norm = WLayerNorm::new(c)?;
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
            .gelu_erf()?
            .apply(&self.channelwise_grn)?
            .apply(&self.channelwise_lin2)?
            .permute((0, 3, 1, 2))?;
        xs + x_res
    }
}
use super::attention_processor::Attention;
#[derive(Debug)]
pub struct AttnBlock {
    self_attn: bool,
    norm: WLayerNorm,
    attention: Attention,
    kv_mapper_lin: candle_nn::Linear,
}

impl AttnBlock {
    pub fn new(
        c: usize,
        c_cond: usize,
        nhead: usize,
        self_attn: bool,
        use_flash_attn: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm = WLayerNorm::new(c)?;
        let attention = Attention::new(c, nhead, c / nhead, use_flash_attn, vb.pp("attention"))?;
        let kv_mapper_lin = candle_nn::linear(c_cond, c, vb.pp("kv_mapper.1"))?;
        Ok(Self {
            self_attn,
            norm,
            attention,
            kv_mapper_lin,
        })
    }

    pub fn forward(&self, xs: &Tensor, kv: &Tensor) -> Result<Tensor> {
        let kv = candle_nn::ops::silu(kv)?.apply(&self.kv_mapper_lin)?;
        let norm_xs = self.norm.forward(xs)?;
        let kv = if self.self_attn {
            let (b_size, channel, _, _) = xs.dims4()?;
            let norm_xs = norm_xs.reshape((b_size, channel, ()))?.transpose(1, 2)?;
            Tensor::cat(&[&norm_xs, &kv], 1)?.contiguous()?
        } else {
            kv
        };
        xs + self.attention.forward(&norm_xs, &kv)
    }
}
