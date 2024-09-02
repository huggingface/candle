use crate::models::moondream::{Config, VisionConfig};
use crate::models::quantized_mixformer::MixFormerSequentialForCausalLM as PhiModel;
use crate::quantized_nn::{layer_norm, linear_b, Linear};
use crate::quantized_var_builder::VarBuilder;
use candle::{IndexOp, Module, Result, Tensor, D};

fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let dim = q.dim(D::Minus1)?;
    let scale_factor = 1.0 / (dim as f64).sqrt();
    let attn_weights = (q.matmul(&k.t()?)? * scale_factor)?;
    candle_nn::ops::softmax_last_dim(&attn_weights)?.matmul(v)
}

#[derive(Debug, Clone)]
struct LinearPatchEmbedding {
    linear: Linear,
}

impl LinearPatchEmbedding {
    fn new(vb: VarBuilder) -> Result<Self> {
        let linear = linear_b(588, 1152, true, vb.pp("linear"))?;
        Ok(Self { linear })
    }
}

impl Module for LinearPatchEmbedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.linear)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    num_heads: usize,
    head_dim: usize,
    qkv: Linear,
    proj: Linear,
}

impl Attention {
    pub fn new(vb: VarBuilder, dim: usize, num_heads: usize) -> Result<Self> {
        let qkv = linear_b(dim, dim * 3, true, vb.pp("qkv"))?;
        let proj = linear_b(dim, dim, true, vb.pp("proj"))?;
        Ok(Self {
            num_heads,
            head_dim: dim / num_heads,
            qkv,
            proj,
        })
    }
}

impl Module for Attention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, n, c) = xs.dims3()?;
        let qkv = xs
            .apply(&self.qkv)?
            .reshape((b, n, 3, self.num_heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?;
        let (q, k, v) = (
            qkv.i(0)?.contiguous()?,
            qkv.i(1)?.contiguous()?,
            qkv.i(2)?.contiguous()?,
        );
        scaled_dot_product_attention(&q, &k, &v)?
            .transpose(1, 2)?
            .reshape((b, n, c))?
            .apply(&self.proj)
    }
}

#[derive(Debug, Clone)]
struct VitBlock {
    attn: Attention,
    mlp: Mlp,
    norm1: candle_nn::LayerNorm,
    norm2: candle_nn::LayerNorm,
}

impl VitBlock {
    fn new(vb: VarBuilder, dim: usize, num_heads: usize, cfg: &VisionConfig) -> Result<Self> {
        let attn = Attention::new(vb.pp("attn"), dim, num_heads)?;
        let mlp = Mlp::new(vb.pp("mlp"), dim, cfg.hidden_features, dim, cfg.act)?;
        let norm1 = layer_norm(dim, 1e-5, vb.pp("norm1"))?;
        let norm2 = layer_norm(dim, 1e-5, vb.pp("norm2"))?;
        Ok(Self {
            attn,
            mlp,
            norm1,
            norm2,
        })
    }
}

impl Module for VitBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = xs.apply(&self.norm1)?.apply(&self.attn)?;
        let xs = (xs + &ys)?;
        let ys = xs.apply(&self.norm2)?.apply(&self.mlp)?;
        let xs = (&xs + &ys)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct VisionTransformer {
    patch_embed: LinearPatchEmbedding,
    pos_embed: Tensor,
    blocks: Vec<VitBlock>,
    norm: candle_nn::LayerNorm,
}

impl VisionTransformer {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_embed = LinearPatchEmbedding::new(vb.pp("patch_embed"))?;
        let pos_embed = vb
            .get((1, cfg.embed_len, cfg.embed_dim), "pos_embed")?
            .dequantize(vb.device())?;
        let blocks = (0..cfg.num_blocks)
            .map(|i| {
                VitBlock::new(
                    vb.pp(format!("blocks.{}", i)),
                    cfg.embed_dim,
                    cfg.num_heads,
                    cfg,
                )
            })
            .collect::<Result<_>>()?;
        let norm = layer_norm(cfg.embed_dim, 1e-5, vb.pp("norm"))?;
        Ok(Self {
            patch_embed,
            pos_embed,
            blocks,
            norm,
        })
    }
}

impl Module for VisionTransformer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = (&xs.apply(&self.patch_embed)? + &self.pos_embed)?;
        for block in self.blocks.iter() {
            xs = xs.apply(block)?;
        }
        xs.apply(&self.norm)
    }
}

#[derive(Debug, Clone)]
pub struct Encoder {
    model: VisionTransformer,
}

impl Encoder {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let model = VisionTransformer::new(cfg, vb.pp("model.visual"))?;
        Ok(Self { model })
    }
}

impl Module for Encoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.model)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    fc1: Linear,
    act: candle_nn::Activation,
    fc2: Linear,
}

impl Mlp {
    fn new(
        vb: VarBuilder,
        in_features: usize,
        hidden_features: usize,
        out_features: usize,
        act: candle_nn::Activation,
    ) -> Result<Self> {
        let fc1 = linear_b(in_features, hidden_features, true, vb.pp("fc1"))?;
        let fc2 = linear_b(hidden_features, out_features, true, vb.pp("fc2"))?;
        Ok(Self { fc1, act, fc2 })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.fc1)?.apply(&self.act)?.apply(&self.fc2)
    }
}

#[derive(Debug, Clone)]
struct VisionProjection {
    mlp: Mlp,
}

impl VisionProjection {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let mlp = Mlp::new(
            vb.pp("mlp"),
            cfg.image_embedding_dim,
            cfg.hidden_dim,
            cfg.model_dim,
            cfg.act,
        )?;
        Ok(Self { mlp })
    }
}

impl Module for VisionProjection {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.mlp)
    }
}

#[derive(Debug, Clone)]
pub struct VisionEncoder {
    encoder: Encoder,
    projection: VisionProjection,
}

impl VisionEncoder {
    pub fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = Encoder::new(cfg, vb.pp("encoder"))?;
        let projection = VisionProjection::new(cfg, vb.pp("projection"))?;
        Ok(Self {
            encoder,
            projection,
        })
    }
}

impl Module for VisionEncoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, c, hp1, wp2) = xs.dims4()?;
        let (p1, p2) = (14, 14);
        let h = hp1 / p1;
        let w = wp2 / p2;
        xs.reshape((b, c, h, p1, h, p2))?
            .permute((0, 2, 4, 1, 3, 5))?
            .reshape((b, h * w, c * p1 * p2))?
            .apply(&self.encoder)?
            .apply(&self.projection)
    }
}

pub struct Model {
    pub text_model: PhiModel,
    pub vision_encoder: VisionEncoder,
}

impl Model {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let text_model = PhiModel::new_v2(&config.phi_config, vb.pp("text_model"))?;
        let vision_encoder = VisionEncoder::new(&config.vision_config, vb.pp("vision_encoder"))?;
        Ok(Self {
            text_model,
            vision_encoder,
        })
    }

    pub fn vision_encoder(&self) -> &VisionEncoder {
        &self.vision_encoder
    }

    pub fn text_model(&mut self) -> &mut PhiModel {
        &mut self.text_model
    }
}
