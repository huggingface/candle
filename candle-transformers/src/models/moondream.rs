#![allow(unused)]
use crate::models::phi;
use candle::{Module, Result, Tensor};
use candle_nn::{linear_b, Linear, VarBuilder};

// https://github.com/vikhyat/moondream/blob/main/moondream/configuration_moondream.py
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    phi_config: phi::Config,
    vision_config: VisionConfig,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct VisionConfig {
    image_embedding_dim: usize,
    model_dim: usize,
    hidden_dim: usize,
    act: candle_nn::Activation,
}

impl VisionConfig {
    pub fn v2() -> Self {
        Self {
            image_embedding_dim: 1152,
            model_dim: 2048,
            hidden_dim: 2048 * 4,
            act: candle_nn::Activation::Silu,
        }
    }
}

impl Config {
    pub fn v2() -> Self {
        let phi_config = phi::Config {
            vocab_size: 51200,
            hidden_size: 2048,
            intermediate_size: 8192,
            num_hidden_layers: 24,
            num_attention_heads: 32,
            num_key_value_heads: None,
            hidden_act: candle_nn::Activation::NewGelu,
            max_position_embeddings: 2048,
            tie_word_embeddings: false,
            layer_norm_eps: 1e-5,
            rope_theta: 10_000.,
            partial_rotary_factor: 0.5,
            qk_layernorm: false,
        };
        let vision_config = VisionConfig::v2();
        Self {
            phi_config,
            vision_config,
        }
    }
}

#[derive(Debug, Clone)]
struct LinearPatchEmbedding {
    linear: Linear,
}

#[derive(Debug, Clone)]
struct Encoder {}

impl Encoder {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        todo!()
    }
}

impl Module for Encoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        todo!()
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
        in_f: usize,
        hidden_f: usize,
        out_f: usize,
        act: candle_nn::Activation,
        vb: VarBuilder,
    ) -> Result<Self> {
        let fc1 = linear_b(in_f, hidden_f, true, vb.pp("fc1"))?;
        let fc2 = linear_b(hidden_f, out_f, true, vb.pp("fc2"))?;
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
            cfg.image_embedding_dim,
            cfg.hidden_dim,
            cfg.model_dim,
            cfg.act,
            vb.pp("mlp"),
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
struct VisionEncoder {
    encoder: Encoder,
    projection: VisionProjection,
}

impl VisionEncoder {
    pub fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = Encoder::new(cfg, vb.pp("vision.trunk"))?;
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
        let xs = xs
            .reshape((b, c, h, p1, h, p2))?
            .permute((0, 2, 4, 1, 3, 5))?
            .reshape((b, h * w, c * p1 * p2))?;
        xs.apply(&self.encoder)?.apply(&self.projection)
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    text_model: phi::Model,
    vision_encoder: VisionEncoder,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let text_model = phi::Model::new(&cfg.phi_config, vb.pp("text_model"))?;
        let vision_encoder = VisionEncoder::new(&cfg.vision_config, vb.pp("vision_encoder"))?;
        Ok(Self {
            text_model,
            vision_encoder,
        })
    }
}
