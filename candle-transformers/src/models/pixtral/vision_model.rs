#![allow(unused)]
use candle::{Module, Result, Tensor};
use candle_nn::{linear_b, rms_norm, Linear, RmsNorm, VarBuilder};

fn default_act() -> candle_nn::Activation {
    candle_nn::Activation::Gelu
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct Config {
    hidden_size: usize,
    num_channels: usize,
    image_size: usize,
    patch_size: usize,
    rope_theta: f64,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default = "default_act")]
    hidden_act: candle_nn::Activation,
}

impl Config {
    fn pixtral_12b_2409() -> Self {
        Self {
            hidden_size: 1024,
            num_channels: 3,
            image_size: 1024,
            patch_size: 16,
            rope_theta: 10000.0,
            intermediate_size: 4096,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            // Default
            hidden_act: candle_nn::Activation::Gelu,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    scale: f64,
    num_heads: usize,
    head_dim: usize,
}

impl Attention {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let head_dim = h / num_heads;
        let q_proj = linear_b(h, h, false, vb.pp("q_proj"))?;
        let k_proj = linear_b(h, h, false, vb.pp("k_proj"))?;
        let v_proj = linear_b(h, h, false, vb.pp("v_proj"))?;
        let o_proj = linear_b(h, h, false, vb.pp("o_proj"))?;
        let scale = (head_dim as f64).powf(-0.5);
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            scale,
            num_heads,
            head_dim,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: candle_nn::Activation,
}

impl Mlp {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        let gate_proj = linear_b(h, i, false, vb.pp("gate_proj"))?;
        let up_proj = linear_b(h, i, false, vb.pp("up_proj"))?;
        let down_proj = linear_b(i, h, false, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        (xs.apply(&self.gate_proj)?.apply(&self.act_fn)? * xs.apply(&self.up_proj))?
            .apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
pub struct AttentionLayer {
    attention_norm: RmsNorm,
    feed_forward: Mlp,
    attention: Attention,
    ffn_norm: RmsNorm,
}

impl AttentionLayer {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let attention_norm = rms_norm(cfg.hidden_size, 1e-5, vb.pp("attention_norm"))?;
        let feed_forward = Mlp::new(cfg, vb.pp("feed_forward"))?;
        let attention = Attention::new(cfg, vb.pp("attention"))?;
        let ffn_norm = rms_norm(cfg.hidden_size, 1e-5, vb.pp("ffn_norm"))?;
        Ok(Self {
            attention_norm,
            feed_forward,
            attention,
            ffn_norm,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Transformer {
    layers: Vec<AttentionLayer>,
}

impl Transformer {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb = vb.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = AttentionLayer::new(cfg, vb.pp(layer_idx))?;
        }
        Ok(Self { layers })
    }
}

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {}

impl RotaryEmbedding {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        todo!()
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    patch_conv: candle_nn::Conv2d,
    ln_pre: RmsNorm,
    transformer: Transformer,
    patch_positional_embedding: RotaryEmbedding,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let conv2d_cfg = candle_nn::Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        let patch_conv = candle_nn::conv2d_no_bias(
            cfg.num_channels,
            cfg.hidden_size,
            cfg.patch_size,
            conv2d_cfg,
            vb.pp("patch_conv"),
        )?;
        let ln_pre = candle_nn::rms_norm(cfg.hidden_size, 1e-5, vb.pp("ln_pre"))?;
        let transformer = Transformer::new(cfg, vb.pp("transformer"))?;
        let patch_positional_embedding =
            RotaryEmbedding::new(cfg, vb.pp("patch_positional_embedding"))?;
        Ok(Self {
            patch_conv,
            ln_pre,
            transformer,
            patch_positional_embedding,
        })
    }
}

impl Module for Model {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        todo!()
    }
}
