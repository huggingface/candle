use crate::models::mixformer::{Config as PhiConfig, MixFormerSequentialForCausalLM as PhiModel};
use crate::models::with_tracing::{layer_norm, linear_b, LayerNorm, Linear};
use candle::{IndexOp, Module, Result, Tensor, D};
use candle_nn::VarBuilder;

pub struct Config {
    pub phi_config: PhiConfig,
    pub vision_config: VisionConfig,
}

impl Config {
    pub fn v2() -> Self {
        Self {
            phi_config: PhiConfig::v1_5(),
            vision_config: VisionConfig::v2(),
        }
    }
}

fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let dim = q.dim(D::Minus1)?;
    let scale_factor = 1.0 / (dim as f64).sqrt();
    let attn_weights = (q.matmul(&k.t()?)? * scale_factor)?;
    candle_nn::ops::softmax_last_dim(&attn_weights)?.matmul(v)
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct VisionConfig {
    pub(crate) image_embedding_dim: usize,
    pub(crate) model_dim: usize,
    pub(crate) hidden_dim: usize,
    pub(crate) hidden_features: usize,
    pub(crate) embed_len: usize,
    pub(crate) embed_dim: usize,
    pub(crate) num_blocks: usize,
    pub(crate) num_heads: usize,
    pub(crate) act: candle_nn::Activation,
}

impl VisionConfig {
    pub fn v2() -> Self {
        Self {
            image_embedding_dim: 1152,
            model_dim: 2048,
            hidden_dim: 2048 * 4,
            hidden_features: 4304,
            embed_len: 729,
            embed_dim: 1152,
            num_blocks: 27,
            num_heads: 16,
            act: candle_nn::Activation::GeluPytorchTanh,
        }
    }
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
    span: tracing::Span,
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
            span: tracing::span!(tracing::Level::TRACE, "vit-attn"),
        })
    }
}

impl Module for Attention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
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
    norm1: LayerNorm,
    norm2: LayerNorm,
    span: tracing::Span,
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
            span: tracing::span!(tracing::Level::TRACE, "vit-block"),
        })
    }
}

impl Module for VitBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
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
    norm: LayerNorm,
    span: tracing::Span,
}

impl VisionTransformer {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_embed = LinearPatchEmbedding::new(vb.pp("patch_embed"))?;
        let pos_embed = vb.get((1, cfg.embed_len, cfg.embed_dim), "pos_embed")?;
        let blocks = (0..cfg.num_blocks)
            .map(|i| {
                VitBlock::new(
                    vb.pp(&format!("blocks.{}", i)),
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
            span: tracing::span!(tracing::Level::TRACE, "vit"),
        })
    }
}

impl Module for VisionTransformer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
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
    span: tracing::Span,
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
        Ok(Self {
            fc1,
            act,
            fc2,
            span: tracing::span!(tracing::Level::TRACE, "mlp"),
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
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

#[derive(Debug, Clone)]
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
