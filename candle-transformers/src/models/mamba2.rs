//! Mamba2 inference implementation.
//!
//! See ["Transformers are SSMs: Generalized Models and Efficient Algorithms
//! Through Structured State Space Duality"](https://arxiv.org/abs/2405.21060)

use crate::models::with_tracing::{linear_no_bias, Linear};
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{RmsNorm, VarBuilder};

const D_CONV: usize = 4;

fn default_d_state() -> usize {
    64
}
fn default_expand() -> usize {
    2
}
fn default_headdim() -> usize {
    64
}
fn default_ngroups() -> usize {
    1
}
fn default_pad_vocab_size_multiple() -> usize {
    16
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub d_model: usize,
    pub n_layer: usize,
    pub vocab_size: usize,
    #[serde(default = "default_d_state")]
    pub d_state: usize,
    #[serde(default = "default_expand")]
    pub expand: usize,
    #[serde(default = "default_headdim")]
    pub headdim: usize,
    #[serde(default = "default_ngroups")]
    pub ngroups: usize,
    #[serde(default = "default_pad_vocab_size_multiple")]
    pub pad_vocab_size_multiple: usize,
}

impl Config {
    fn vocab_size(&self) -> usize {
        let pad = self.pad_vocab_size_multiple;
        self.vocab_size.div_ceil(pad) * pad
    }

    fn d_inner(&self) -> usize {
        self.d_model * self.expand
    }

    fn nheads(&self) -> usize {
        self.d_inner() / self.headdim
    }
}

pub struct State {
    pub hs: Vec<Tensor>,
    pub conv_states: Vec<Tensor>,
    pub pos: usize,
}

impl State {
    pub fn new(batch_size: usize, cfg: &Config, dtype: DType, device: &Device) -> Result<Self> {
        let d_inner = cfg.d_inner();
        let nheads = cfg.nheads();
        let mut hs = Vec::with_capacity(cfg.n_layer);
        let mut conv_states = Vec::with_capacity(cfg.n_layer);
        for _ in 0..cfg.n_layer {
            let h = Tensor::zeros(
                (batch_size, nheads, cfg.headdim, cfg.d_state),
                dtype,
                device,
            )?;
            let conv = Tensor::zeros((batch_size, d_inner, D_CONV), dtype, device)?;
            hs.push(h);
            conv_states.push(conv);
        }
        Ok(Self {
            hs,
            conv_states,
            pos: 0,
        })
    }
}

#[derive(Clone, Debug)]
pub struct Mamba2Block {
    in_proj: Linear,
    conv1d_weight: Tensor,
    conv1d_bias: Tensor,
    a_log: Tensor,
    d: Tensor,
    dt_bias: Tensor,
    out_proj: Linear,
    norm: RmsNorm,
    d_inner: usize,
    d_state: usize,
    headdim: usize,
    nheads: usize,
    ngroups: usize,
    layer_idx: usize,
}

impl Mamba2Block {
    pub fn new(layer_idx: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let d_inner = cfg.d_inner();
        let nheads = cfg.nheads();
        let ngroups = cfg.ngroups;
        let d_state = cfg.d_state;

        // Mamba2 projects: z, x, B, C, dt in parallel
        let proj_size = 2 * d_inner + 2 * ngroups * d_state + nheads;
        let in_proj = linear_no_bias(cfg.d_model, proj_size, vb.pp("in_proj"))?;

        let conv1d_weight = vb.get((d_inner, 1, D_CONV), "conv1d.weight")?;
        let conv1d_bias = vb.get(d_inner, "conv1d.bias")?;

        let a_log = vb.get(nheads, "A_log")?;
        let d = vb.get(nheads, "D")?;
        let dt_bias = vb.get(nheads, "dt_bias")?;

        let out_proj = linear_no_bias(d_inner, cfg.d_model, vb.pp("out_proj"))?;
        let norm = candle_nn::rms_norm(d_inner, 1e-5, vb.pp("norm"))?;

        Ok(Self {
            in_proj,
            conv1d_weight,
            conv1d_bias,
            a_log,
            d,
            dt_bias,
            out_proj,
            norm,
            d_inner,
            d_state,
            headdim: cfg.headdim,
            nheads,
            ngroups,
            layer_idx,
        })
    }

    pub fn forward(&self, xs: &Tensor, state: &mut State) -> Result<Tensor> {
        let (b_sz, _dim) = xs.dims2()?;

        let proj = self.in_proj.forward(xs)?;

        // Split: z, x, B, C, dt
        let z = proj.narrow(D::Minus1, 0, self.d_inner)?;
        let x = proj.narrow(D::Minus1, self.d_inner, self.d_inner)?;
        let bc_start = 2 * self.d_inner;
        let b = proj.narrow(D::Minus1, bc_start, self.ngroups * self.d_state)?;
        let c = proj.narrow(
            D::Minus1,
            bc_start + self.ngroups * self.d_state,
            self.ngroups * self.d_state,
        )?;
        let dt = proj.narrow(
            D::Minus1,
            bc_start + 2 * self.ngroups * self.d_state,
            self.nheads,
        )?;

        // Causal conv
        let x_conv = self.apply_conv1d(&x, &mut state.conv_states[self.layer_idx])?;
        let x_conv = candle_nn::ops::silu(&x_conv)?;

        // Softplus on dt
        let dt = ((&dt + &self.dt_bias)?.exp()? + 1.)?.log()?;

        let a = self.a_log.exp()?.neg()?;

        let y = self.ssm_step(&x_conv, &a, &b, &c, &dt, state)?;

        // Skip connection
        let d = self.d.broadcast_as((b_sz, self.nheads))?;
        let x_skip = x_conv.reshape((b_sz, self.nheads, self.headdim))?;
        let y = (&y + x_skip.broadcast_mul(&d.unsqueeze(D::Minus1)?)?)?;
        let y = y.reshape((b_sz, self.d_inner))?;

        // Norm and gate
        let y = self.norm.forward(&y)?;
        let y = (y * candle_nn::ops::silu(&z)?)?;

        self.out_proj.forward(&y)
    }

    fn apply_conv1d(&self, x: &Tensor, conv_state: &mut Tensor) -> Result<Tensor> {
        let (b_sz, d_inner) = x.dims2()?;

        // Shift state left, add new token
        let shifted = conv_state.narrow(D::Minus1, 1, D_CONV - 1)?;
        let x_expanded = x.unsqueeze(D::Minus1)?;
        *conv_state = Tensor::cat(&[shifted, x_expanded], D::Minus1)?;

        // Depthwise conv
        let mut result = self.conv1d_bias.broadcast_as((b_sz, d_inner))?;
        for i in 0..D_CONV {
            let w = self.conv1d_weight.i((.., 0, i))?;
            let x_i = conv_state.i((.., .., i))?;
            result = (result + w.broadcast_mul(&x_i)?)?;
        }
        Ok(result)
    }

    fn ssm_step(
        &self,
        x: &Tensor,
        a: &Tensor,
        b: &Tensor,
        c: &Tensor,
        dt: &Tensor,
        state: &mut State,
    ) -> Result<Tensor> {
        let (b_sz, _) = x.dims2()?;
        let h = &mut state.hs[self.layer_idx];

        let x = x.reshape((b_sz, self.nheads, self.headdim))?;

        // Expand B, C from groups to heads
        let b = b.reshape((b_sz, self.ngroups, self.d_state))?;
        let c = c.reshape((b_sz, self.ngroups, self.d_state))?;
        let heads_per_group = self.nheads / self.ngroups;
        let b =
            b.unsqueeze(2)?
                .broadcast_as((b_sz, self.ngroups, heads_per_group, self.d_state))?;
        let b = b.reshape((b_sz, self.nheads, self.d_state))?;
        let c =
            c.unsqueeze(2)?
                .broadcast_as((b_sz, self.ngroups, heads_per_group, self.d_state))?;
        let c = c.reshape((b_sz, self.nheads, self.d_state))?;

        // decay = exp(A * dt)
        let dt_a = dt.broadcast_mul(a)?;
        let decay = dt_a.exp()?;
        let decay = decay.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?;
        let decay = decay.broadcast_as((b_sz, self.nheads, self.headdim, self.d_state))?;

        // h = decay * h + dt * outer(x, B)
        let x_unsq = x.unsqueeze(D::Minus1)?;
        let b_unsq = b.unsqueeze(2)?;
        let x_b = x_unsq.broadcast_mul(&b_unsq)?;

        let dt_expanded = dt.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?;
        let dt_expanded =
            dt_expanded.broadcast_as((b_sz, self.nheads, self.headdim, self.d_state))?;

        *h = ((&*h * &decay)? + (&dt_expanded * &x_b)?)?;

        // y = C^T @ h
        let c_unsq = c.unsqueeze(2)?;
        let y = (h * &c_unsq)?.sum(D::Minus1)?;

        Ok(y)
    }
}

#[derive(Clone, Debug)]
pub struct ResidualBlock {
    mixer: Mamba2Block,
    norm: RmsNorm,
}

impl ResidualBlock {
    pub fn new(layer_idx: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let norm = candle_nn::rms_norm(cfg.d_model, 1e-5, vb.pp("norm"))?;
        let mixer = Mamba2Block::new(layer_idx, cfg, vb.pp("mixer"))?;
        Ok(Self { mixer, norm })
    }

    fn forward(&self, xs: &Tensor, state: &mut State) -> Result<Tensor> {
        self.mixer.forward(&xs.apply(&self.norm)?, state)? + xs
    }
}

#[derive(Clone, Debug)]
pub struct Model {
    embedding: candle_nn::Embedding,
    layers: Vec<ResidualBlock>,
    norm_f: RmsNorm,
    lm_head: Linear,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let embedding = candle_nn::embedding(cfg.vocab_size(), cfg.d_model, vb.pp("embedding"))?;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        let vb_l = vb.pp("layers");
        for layer_idx in 0..cfg.n_layer {
            layers.push(ResidualBlock::new(layer_idx, cfg, vb_l.pp(layer_idx))?);
        }
        let norm_f = candle_nn::rms_norm(cfg.d_model, 1e-5, vb.pp("norm_f"))?;
        let lm_head = Linear::from_weights(embedding.embeddings().clone(), None);
        Ok(Self {
            embedding,
            layers,
            norm_f,
            lm_head,
            dtype: vb.dtype(),
        })
    }

    pub fn forward(&self, input_ids: &Tensor, state: &mut State) -> Result<Tensor> {
        let _b_size = input_ids.dims1()?;
        let mut xs = self.embedding.forward(input_ids)?;
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, state)?;
        }
        state.pos += 1;
        xs.apply(&self.norm_f)?.apply(&self.lm_head)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}
