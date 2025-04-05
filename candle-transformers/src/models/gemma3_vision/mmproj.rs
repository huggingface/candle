use candle::{DType, Result, Tensor, D};
use candle_nn::{Module, VarBuilder};

use super::config::Gemma3Config;

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        x_normed
            .to_dtype(x_dtype)?
            .broadcast_mul(&(&self.weight + 1.0)?)
    }
}

pub struct AvgPool2d {
    kernel_size: usize,
    stride: usize,
}

impl AvgPool2d {
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        Self {
            kernel_size,
            stride,
        }
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.avg_pool2d_with_stride(self.kernel_size, self.stride)
    }
}

pub struct Gemma3MultiModalProjector {
    mm_input_projection_weight: Tensor,
    mm_soft_emb_norm: RmsNorm,
    patches_per_image: usize,
    avg_pool: AvgPool2d,
}

impl Gemma3MultiModalProjector {
    pub fn new(cfg: &Gemma3Config, vb: VarBuilder) -> Result<Self> {
        let Gemma3Config::WithVision {
            text_config,
            vision_config,
            image_token_index: _,
            mm_tokens_per_image,
        } = cfg
        else {
            unreachable!()
        };

        let mm_input_projection_weight = vb.get(
            (vision_config.hidden_size, text_config.hidden_size),
            "mm_input_projection_weight",
        )?;
        let mm_soft_emb_norm = RmsNorm::new(
            vision_config.hidden_size,
            vision_config.layer_norm_eps,
            vb.pp("mm_soft_emb_norm"),
        )?;

        let patches_per_image = vision_config.image_size / vision_config.patch_size;
        let tokens_per_side = mm_tokens_per_image.isqrt();
        let kernel_size = patches_per_image / tokens_per_side;
        let avg_pool = AvgPool2d::new(kernel_size, kernel_size);

        Ok(Self {
            mm_input_projection_weight,
            mm_soft_emb_norm,
            patches_per_image,
            avg_pool,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (bs, _, seqlen) = xs.dims3()?;

        let mut reshaped_vision_outputs = xs.transpose(1, 2)?;
        reshaped_vision_outputs = reshaped_vision_outputs.reshape((
            bs,
            seqlen,
            self.patches_per_image,
            self.patches_per_image,
        ))?;
        reshaped_vision_outputs = reshaped_vision_outputs.contiguous()?;

        let mut pooled_vision_outputs = self.avg_pool.forward(&reshaped_vision_outputs)?;
        pooled_vision_outputs = pooled_vision_outputs.flatten_from(2)?;
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)?;

        let normed_vision_outputs = self.mm_soft_emb_norm.forward(&pooled_vision_outputs)?;

        normed_vision_outputs.broadcast_matmul(&self.mm_input_projection_weight)
    }
}
