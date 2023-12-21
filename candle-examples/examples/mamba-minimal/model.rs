#![allow(unused)]
/// This follows the lines of:
/// https://github.com/johnma2006/mamba-minimal/blob/master/model.py
/// Simple, minimal implementation of Mamba in one file of PyTorch.
use candle::{Result, Tensor};
use candle_nn::{RmsNorm, VarBuilder};

use candle_transformers::models::with_tracing::{linear, linear_no_bias, Linear};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    d_model: usize,
    n_layer: usize,
    vocab_size: usize,
    pad_vocab_size_multiple: usize,
}

impl Config {
    fn vocab_size(&self) -> usize {
        let pad = self.pad_vocab_size_multiple;
        (self.vocab_size + pad - 1) / pad * pad
    }

    fn dt_rank(&self) -> usize {
        (self.d_model + 15) / 16
    }

    fn d_conv(&self) -> usize {
        4
    }

    fn d_state(&self) -> usize {
        16
    }

    fn d_inner(&self) -> usize {
        self.d_model * 2
    }
}

// https://github.com/johnma2006/mamba-minimal/blob/61f01953ca153f8c4a850d7111beecbf4be9cee1/model.py#L177
#[derive(Clone, Debug)]
pub struct MambaBlock {
    in_proj: Linear,
    conv1d: candle_nn::Conv1d,
    x_proj: Linear,
    dt_proj: Linear,
    a_log: Tensor,
    d: Tensor,
    out_proj: Linear,
}

impl MambaBlock {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let d_inner = cfg.d_inner();
        let d_conv = cfg.d_conv();
        let d_state = cfg.d_state();
        let dt_rank = cfg.dt_rank();
        let in_proj = linear_no_bias(cfg.d_model, d_inner * 2, vb.pp("in_proj"))?;
        let conv_cfg = candle_nn::Conv1dConfig {
            groups: d_inner,
            padding: d_conv - 1,
            ..Default::default()
        };
        let conv1d = candle_nn::conv1d(d_inner, d_inner, d_conv, conv_cfg, vb.pp("conv1d"))?;
        let x_proj = linear_no_bias(d_inner, dt_rank + d_state * 2, vb.pp("x_proj"))?;
        let dt_proj = linear(dt_rank, d_inner, vb.pp("dt_proj"))?;
        let a_log = vb.get((d_inner, d_state), "A_log")?;
        let d = vb.get(d_inner, "D")?;
        let out_proj = linear_no_bias(d_inner, cfg.d_model, vb.pp("out_proj"))?;
        Ok(Self {
            in_proj,
            conv1d,
            x_proj,
            dt_proj,
            a_log,
            d,
            out_proj,
        })
    }
}

// https://github.com/johnma2006/mamba-minimal/blob/61f01953ca153f8c4a850d7111beecbf4be9cee1/model.py#L143
#[derive(Clone, Debug)]
pub struct ResidualBlock {
    mixer: MambaBlock,
    norm: RmsNorm,
}

impl ResidualBlock {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let norm = candle_nn::rms_norm(cfg.d_model, 1e-5, vb.pp("norm"))?;
        let mixer = MambaBlock::new(cfg, vb.pp("mixer"))?;
        Ok(Self { mixer, norm })
    }
}

// https://github.com/johnma2006/mamba-minimal/blob/61f01953ca153f8c4a850d7111beecbf4be9cee1/model.py#L56
#[derive(Clone, Debug)]
pub struct Model {
    embedding: candle_nn::Embedding,
    layers: Vec<ResidualBlock>,
    norm_f: RmsNorm,
    lm_head: Linear,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let embedding = candle_nn::embedding(cfg.vocab_size(), cfg.d_model, vb.pp("embedding"))?;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        let vb_l = vb.pp("layers");
        for layer_idx in 0..cfg.n_layer {
            let layer = ResidualBlock::new(cfg, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        let norm_f = candle_nn::rms_norm(cfg.d_model, 1e-5, vb.pp("norm_f"))?;
        let lm_head = Linear::from_weights(embedding.embeddings().clone(), None);
        Ok(Self {
            embedding,
            layers,
            norm_f,
            lm_head,
        })
    }

    pub fn forward(&mut self, _input_ids: &Tensor, _seqlen_offset: usize) -> Result<Tensor> {
        todo!()
    }
}
