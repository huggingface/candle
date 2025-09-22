//! MoonDream Model vision-to-text
//!
//!
//! Moondream is a computer-vision model that can answer real-world questions about images.
//! It's lightweight with only 1.6B parameters, enabling it to run on mobile phones and edge devices.
//! [MoonDream Original Implementation](https://github.com/vikhyat/moondream)
//!
//! The model consists of:
//! - Vision encoder using a ViT-style architecture
//! - Text decoder based on Microsoft's Phi model
//! - Vision projection module to align vision and text embeddings
//!
//! # Examples
//!
//! <img src="https://raw.githubusercontent.com/vikhyat/moondream/main/assets/demo-1.jpg" width="200">
//!
//! ```bash
//! # download an example image
//! wget https://raw.githubusercontent.com/vikhyat/moondream/main/assets/demo-1.jpg
//!
//! # Now you can run Moondream from the `candle-examples` crate:
//! cargo run --example moondream \
//!   --release -- \
//!   --prompt "What is the girl eating?"
//!   --image "./demo-1.jpg"
//!
//! > avavx: false, neon: true, simd128: false, f16c: false
//! > temp: 0.00 repeat-penalty: 1.00 repeat-last-n: 64
//! > retrieved the files in 3.395583ms
//! > Running on CPU, to run on GPU(metal), build this example with `--features metal`
//! > loaded the model in 5.485493792s
//! > loaded and encoded the image Tensor[dims 3, 378, 378; f32] in 4.801396417s
//! > starting the inference loop
//! > The girl is eating a hamburger.<
//! > 9 tokens generated (0.68 token/s)
//! ```

use crate::models::mixformer::{Config as PhiConfig, MixFormerSequentialForCausalLM as PhiModel};
use crate::models::with_tracing::{layer_norm, linear_b, LayerNorm, Linear};
use candle::{BackendStorage, IndexOp, Module, Result, Tensor, D};
use candle_nn::VarBuilder;

#[derive(Debug, Clone, serde::Deserialize)]
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

fn scaled_dot_product_attention<B: BackendStorage>(
    q: &Tensor<B>,
    k: &Tensor<B>,
    v: &Tensor<B>,
) -> Result<Tensor<B>> {
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
struct LinearPatchEmbedding<B: BackendStorage> {
    linear: Linear<B>,
}

impl<B: BackendStorage> LinearPatchEmbedding<B> {
    fn new(vb: VarBuilder<B>) -> Result<Self> {
        let linear = linear_b(588, 1152, true, vb.pp("linear"))?;
        Ok(Self { linear })
    }
}

impl<B: BackendStorage> Module<B> for LinearPatchEmbedding<B> {
    fn forward(&self, xs: &Tensor<B>) -> Result<Tensor<B>> {
        xs.apply(&self.linear)
    }
}

#[derive(Debug, Clone)]
struct Attention<B: BackendStorage> {
    num_heads: usize,
    head_dim: usize,
    qkv: Linear<B>,
    proj: Linear<B>,
    span: tracing::Span,
}

impl<B: BackendStorage> Attention<B> {
    pub fn new(vb: VarBuilder<B>, dim: usize, num_heads: usize) -> Result<Self> {
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

impl<B: BackendStorage> Module<B> for Attention<B> {
    fn forward(&self, xs: &Tensor<B>) -> Result<Tensor<B>> {
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
struct VitBlock<B: BackendStorage> {
    attn: Attention<B>,
    mlp: Mlp<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    span: tracing::Span,
}

impl<B: BackendStorage> VitBlock<B> {
    fn new(vb: VarBuilder<B>, dim: usize, num_heads: usize, cfg: &VisionConfig) -> Result<Self> {
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

impl<B: BackendStorage> Module<B> for VitBlock<B> {
    fn forward(&self, xs: &Tensor<B>) -> Result<Tensor<B>> {
        let _enter = self.span.enter();
        let ys = xs.apply(&self.norm1)?.apply(&self.attn)?;
        let xs = (xs + &ys)?;
        let ys = xs.apply(&self.norm2)?.apply(&self.mlp)?;
        let xs = (&xs + &ys)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct VisionTransformer<B: BackendStorage> {
    patch_embed: LinearPatchEmbedding<B>,
    pos_embed: Tensor<B>,
    blocks: Vec<VitBlock<B>>,
    norm: LayerNorm<B>,
    span: tracing::Span,
}

impl<B: BackendStorage> VisionTransformer<B> {
    fn new(cfg: &VisionConfig, vb: VarBuilder<B>) -> Result<Self> {
        let patch_embed = LinearPatchEmbedding::new(vb.pp("patch_embed"))?;
        let pos_embed = vb.get((1, cfg.embed_len, cfg.embed_dim), "pos_embed")?;
        let blocks = (0..cfg.num_blocks)
            .map(|i| {
                VitBlock::new(
                    vb.pp(format!("blocks.{i}")),
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

impl<B: BackendStorage> Module<B> for VisionTransformer<B> {
    fn forward(&self, xs: &Tensor<B>) -> Result<Tensor<B>> {
        let _enter = self.span.enter();
        let mut xs = (&xs.apply(&self.patch_embed)? + &self.pos_embed)?;
        for block in self.blocks.iter() {
            xs = xs.apply(block)?;
        }
        xs.apply(&self.norm)
    }
}

#[derive(Debug, Clone)]
pub struct Encoder<B: BackendStorage> {
    model: VisionTransformer<B>,
}

impl<B: BackendStorage> Encoder<B> {
    fn new(cfg: &VisionConfig, vb: VarBuilder<B>) -> Result<Self> {
        let model = VisionTransformer::new(cfg, vb.pp("model.visual"))?;
        Ok(Self { model })
    }
}

impl<B: BackendStorage> Module<B> for Encoder<B> {
    fn forward(&self, xs: &Tensor<B>) -> Result<Tensor<B>> {
        xs.apply(&self.model)
    }
}

#[derive(Debug, Clone)]
struct Mlp<B: BackendStorage> {
    fc1: Linear<B>,
    act: candle_nn::Activation,
    fc2: Linear<B>,
    span: tracing::Span,
}

impl<B: BackendStorage> Mlp<B> {
    fn new(
        vb: VarBuilder<B>,
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

impl<B: BackendStorage> Module<B> for Mlp<B> {
    fn forward(&self, xs: &Tensor<B>) -> Result<Tensor<B>> {
        let _enter = self.span.enter();
        xs.apply(&self.fc1)?.apply(&self.act)?.apply(&self.fc2)
    }
}

#[derive(Debug, Clone)]
struct VisionProjection<B: BackendStorage> {
    mlp: Mlp<B>,
}

impl<B: BackendStorage> VisionProjection<B> {
    fn new(cfg: &VisionConfig, vb: VarBuilder<B>) -> Result<Self> {
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

impl<B: BackendStorage> Module<B> for VisionProjection<B> {
    fn forward(&self, xs: &Tensor<B>) -> Result<Tensor<B>> {
        xs.apply(&self.mlp)
    }
}

#[derive(Debug, Clone)]
pub struct VisionEncoder<B: BackendStorage> {
    encoder: Encoder<B>,
    projection: VisionProjection<B>,
}

impl<B: BackendStorage> VisionEncoder<B> {
    pub fn new(cfg: &VisionConfig, vb: VarBuilder<B>) -> Result<Self> {
        let encoder = Encoder::new(cfg, vb.pp("encoder"))?;
        let projection = VisionProjection::new(cfg, vb.pp("projection"))?;
        Ok(Self {
            encoder,
            projection,
        })
    }
}

impl<B: BackendStorage> Module<B> for VisionEncoder<B> {
    fn forward(&self, xs: &Tensor<B>) -> Result<Tensor<B>> {
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
pub struct Model<B: BackendStorage> {
    pub text_model: PhiModel<B>,
    pub vision_encoder: VisionEncoder<B>,
}

impl<B: BackendStorage> Model<B> {
    pub fn new(config: &Config, vb: VarBuilder<B>) -> Result<Self> {
        let text_model = PhiModel::new_v2(&config.phi_config, vb.pp("text_model"))?;
        let vision_encoder = VisionEncoder::new(&config.vision_config, vb.pp("vision_encoder"))?;
        Ok(Self {
            text_model,
            vision_encoder,
        })
    }

    pub fn vision_encoder(&self) -> &VisionEncoder<B> {
        &self.vision_encoder
    }

    pub fn text_model(&mut self) -> &mut PhiModel<B> {
        &mut self.text_model
    }
}
