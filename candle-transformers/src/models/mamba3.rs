//! Mamba-3 inference implementation.
//!
//! See [Mamba-3: Improved Sequence Modeling using State Space Principles](https://arxiv.org/abs/2602.18424)

use crate::models::with_tracing::{linear_no_bias, Linear};
use crate::ops::mamba3::{mamba3_mimo_fwd, mamba3_mimo_step, mamba3_siso_fwd, mamba3_siso_step};
use crate::ops::mamba3::{sigmoid, softplus};
use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{RmsNorm, VarBuilder};

fn default_d_state() -> usize {
    128
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
fn default_rope_fraction() -> f32 {
    0.5
}
fn default_a_floor() -> f32 {
    1e-4
}
fn default_mimo_rank() -> usize {
    4
}
fn default_chunk_size() -> usize {
    64
}
fn default_pad_vocab_size_multiple() -> usize {
    16
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    #[serde(alias = "hidden_size")]
    pub d_model: usize,
    #[serde(alias = "num_hidden_layers")]
    pub n_layer: usize,
    pub vocab_size: usize,
    #[serde(alias = "state_size", default = "default_d_state")]
    pub d_state: usize,
    #[serde(default = "default_expand")]
    pub expand: usize,
    #[serde(alias = "head_dim", default = "default_headdim")]
    pub headdim: usize,
    #[serde(alias = "n_groups", default = "default_ngroups")]
    pub ngroups: usize,
    #[serde(default = "default_rope_fraction")]
    pub rope_fraction: f32,
    #[serde(default = "default_a_floor")]
    pub a_floor: f32,
    #[serde(default)]
    pub is_mimo: bool,
    #[serde(default = "default_mimo_rank")]
    pub mimo_rank: usize,
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
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

    fn mimo_rank(&self) -> usize {
        if self.is_mimo {
            self.mimo_rank
        } else {
            1
        }
    }

    fn num_rope_angles(&self) -> usize {
        let mut split = (self.d_state as f32 * self.rope_fraction) as usize;
        if split % 2 != 0 {
            split -= 1;
        }
        split / 2
    }

    fn in_proj_size(&self) -> usize {
        2 * self.d_inner()
            + 2 * self.d_state * self.ngroups * self.mimo_rank()
            + 3 * self.nheads()
            + self.num_rope_angles()
    }
}

pub struct State {
    pub angle_states: Vec<Tensor>,
    pub ssm_states: Vec<Tensor>,
    pub k_states: Vec<Tensor>,
    pub v_states: Vec<Tensor>,
    pub pos: usize,
}

impl State {
    pub fn new(batch_size: usize, cfg: &Config, dtype: DType, device: &Device) -> Result<Self> {
        let nheads = cfg.nheads();
        let rank = cfg.mimo_rank();
        let mut angle_states = Vec::with_capacity(cfg.n_layer);
        let mut ssm_states = Vec::with_capacity(cfg.n_layer);
        let mut k_states = Vec::with_capacity(cfg.n_layer);
        let mut v_states = Vec::with_capacity(cfg.n_layer);
        for _ in 0..cfg.n_layer {
            angle_states.push(Tensor::zeros(
                (batch_size, nheads, cfg.num_rope_angles()),
                DType::F32,
                device,
            )?);
            ssm_states.push(Tensor::zeros(
                (batch_size, nheads, cfg.headdim, cfg.d_state),
                DType::F32,
                device,
            )?);
            k_states.push(Tensor::zeros(
                (batch_size, rank, nheads, cfg.d_state),
                dtype,
                device,
            )?);
            v_states.push(Tensor::zeros(
                (batch_size, nheads, cfg.headdim),
                dtype,
                device,
            )?);
        }
        Ok(Self {
            angle_states,
            ssm_states,
            k_states,
            v_states,
            pos: 0,
        })
    }
}

#[derive(Clone, Debug)]
pub struct Mamba3Block {
    in_proj: Linear,
    dt_bias: Tensor,
    b_bias: Tensor,
    c_bias: Tensor,
    b_norm: RmsNorm,
    c_norm: RmsNorm,
    mimo_x: Option<Tensor>,
    mimo_z: Option<Tensor>,
    mimo_o: Option<Tensor>,
    d: Tensor,
    out_proj: Linear,
    d_inner: usize,
    d_state: usize,
    nheads: usize,
    ngroups: usize,
    headdim: usize,
    mimo_rank: usize,
    num_rope_angles: usize,
    is_mimo: bool,
    a_floor: f32,
    layer_idx: usize,
}

impl Mamba3Block {
    pub fn new(layer_idx: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let d_inner = cfg.d_inner();
        let nheads = cfg.nheads();
        let rank = cfg.mimo_rank();
        let in_proj = linear_no_bias(cfg.d_model, cfg.in_proj_size(), vb.pp("in_proj"))?;
        let dt_bias = vb.get(nheads, "dt_bias")?;
        let b_bias = vb.get((nheads, rank, cfg.d_state), "B_bias")?;
        let c_bias = vb.get((nheads, rank, cfg.d_state), "C_bias")?;
        let b_norm = candle_nn::rms_norm(cfg.d_state, 1e-5, vb.pp("B_norm"))?;
        let c_norm = candle_nn::rms_norm(cfg.d_state, 1e-5, vb.pp("C_norm"))?;
        let d = vb.get(nheads, "D")?;
        let out_proj = linear_no_bias(d_inner, cfg.d_model, vb.pp("out_proj"))?;

        let (mimo_x, mimo_z, mimo_o) = if cfg.is_mimo {
            (
                Some(vb.get((nheads, rank, cfg.headdim), "mimo_x")?),
                Some(vb.get((nheads, rank, cfg.headdim), "mimo_z")?),
                Some(vb.get((nheads, rank, cfg.headdim), "mimo_o")?),
            )
        } else {
            (None, None, None)
        };

        Ok(Self {
            in_proj,
            dt_bias,
            b_bias,
            c_bias,
            b_norm,
            c_norm,
            mimo_x,
            mimo_z,
            mimo_o,
            d,
            out_proj,
            d_inner,
            d_state: cfg.d_state,
            nheads,
            ngroups: cfg.ngroups,
            headdim: cfg.headdim,
            mimo_rank: rank,
            num_rope_angles: cfg.num_rope_angles(),
            is_mimo: cfg.is_mimo,
            a_floor: cfg.a_floor,
            layer_idx,
        })
    }

    fn split_proj<'a>(&self, proj: &'a Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let mut off = 0;
        let mut take = |p: &Tensor, n: usize| -> Result<Tensor> {
            let t = p.narrow(D::Minus1, off, n)?;
            off += n;
            Ok(t)
        };
        let z = take(proj, self.d_inner)?;
        let x = take(proj, self.d_inner)?;
        let b = take(proj, self.d_state * self.ngroups * self.mimo_rank)?;
        let c = take(proj, self.d_state * self.ngroups * self.mimo_rank)?;
        let dd_dt = take(proj, self.nheads)?;
        let dd_a = take(proj, self.nheads)?;
        let trap = take(proj, self.nheads)?;
        let angles = take(proj, self.num_rope_angles)?;
        Ok((z, x, b, c, dd_dt, dd_a, trap, angles))
    }

    fn preprocess(
        &self,
        dd_a: &Tensor,
        dd_dt: &Tensor,
        b: &Tensor,
        c: &Tensor,
        x: &Tensor,
        z: &Tensor,
        trap: &Tensor,
        angles: &Tensor,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let b_sz = x.dim(0)?;
        let a = {
            let sp = softplus(&dd_a.to_dtype(DType::F32)?)?;
            let neg = sp.neg()?;
            neg.clamp(f32::NEG_INFINITY, -self.a_floor)?
        };
        let dt = softplus(&dd_dt.broadcast_add(&self.dt_bias.unsqueeze(0)?)?)?;
        let trap = sigmoid(trap)?;

        let b = b.reshape((b_sz, seq_len, self.mimo_rank, self.ngroups, self.d_state))?;
        let c = c.reshape((b_sz, seq_len, self.mimo_rank, self.ngroups, self.d_state))?;
        let b = self.b_norm.forward(&b.reshape((b_sz * seq_len * self.mimo_rank * self.ngroups, self.d_state))?)?
            .reshape((b_sz, seq_len, self.mimo_rank, self.ngroups, self.d_state))?;
        let c = self.c_norm.forward(&c.reshape((b_sz * seq_len * self.mimo_rank * self.ngroups, self.d_state))?)?
            .reshape((b_sz, seq_len, self.mimo_rank, self.ngroups, self.d_state))?;

        let heads_per_group = self.nheads / self.ngroups;
        let b = b.unsqueeze(3)?.broadcast_as((b_sz, seq_len, self.mimo_rank, self.ngroups, heads_per_group, self.d_state))?
            .reshape((b_sz, seq_len, self.mimo_rank, self.nheads, self.d_state))?;
        let c = c.unsqueeze(3)?.broadcast_as((b_sz, seq_len, self.mimo_rank, self.ngroups, heads_per_group, self.d_state))?
            .reshape((b_sz, seq_len, self.mimo_rank, self.nheads, self.d_state))?;

        let x = x.reshape((b_sz, seq_len, self.nheads, self.headdim))?;
        let z = z.reshape((b_sz, seq_len, self.nheads, self.headdim))?;
        let angles = angles.unsqueeze(2)?.broadcast_as((b_sz, seq_len, self.nheads, self.num_rope_angles))?;
        let adt = a.broadcast_mul(&dt)?;
        Ok((adt, dt, trap, b, c, x, z, angles))
    }

    pub fn forward(&self, xs: &Tensor, state: &mut State) -> Result<Tensor> {
        let (b_sz, _) = xs.dims2()?;
        let proj = self.in_proj.forward(xs)?;
        let (z, x, b, c, dd_dt, dd_a, trap, angles) = self.split_proj(&proj)?;
        let (adt, dt, trap, b, c, x, z, angles) =
            self.preprocess(&dd_a, &dd_dt, &b, &c, &x, &z, &trap, &angles, 1)?;

        let li = self.layer_idx;
        let y = if self.is_mimo {
            let q = c.squeeze(1)?;
            let k = b.squeeze(1)?;
            let out = mamba3_mimo_step(
                &q,
                &k,
                &x.squeeze(1)?,
                &adt.squeeze(1)?,
                &dt.squeeze(1)?,
                &trap.squeeze(1)?,
                &self.c_bias,
                &self.b_bias,
                &angles.squeeze(1)?,
                self.mimo_x.as_ref().unwrap(),
                self.mimo_z.as_ref(),
                self.mimo_o.as_ref(),
                Some(&self.d),
                Some(&z.squeeze(1)?),
                &state.angle_states[li],
                &state.ssm_states[li],
                &state.k_states[li],
                &state.v_states[li],
            )?;
            state.angle_states[li] = out.angle_state;
            state.ssm_states[li] = out.ssm_state;
            state.k_states[li] = out.k_state;
            state.v_states[li] = x.squeeze(1)?;
            out.out
        } else {
            let out = mamba3_siso_step(
                &c.squeeze(1)?.squeeze(1)?,
                &b.squeeze(1)?.squeeze(1)?,
                &x.squeeze(1)?,
                &adt.squeeze(1)?,
                &dt.squeeze(1)?,
                &trap.squeeze(1)?,
                &self.c_bias.squeeze(1)?,
                &self.b_bias.squeeze(1)?,
                &angles.squeeze(1)?,
                Some(&self.d),
                Some(&z.squeeze(1)?),
                &state.angle_states[li],
                &state.ssm_states[li],
                &state.k_states[li].squeeze(1)?,
                &state.v_states[li],
            )?;
            state.angle_states[li] = out.angle_state;
            state.ssm_states[li] = out.ssm_state;
            state.k_states[li] = out.k_state.unsqueeze(1)?;
            state.v_states[li] = x.squeeze(1)?;
            out.out
        };

        let y = y.reshape((b_sz, self.d_inner))?;
        self.out_proj.forward(&y)
    }

    pub fn forward_prefill(&self, xs: &Tensor, state: &mut State, chunk_size: usize) -> Result<Tensor> {
        let (b_sz, seq_len, _) = xs.dims3()?;
        let _ = chunk_size;
        let proj = xs.apply(&self.in_proj)?;
        let (z, x, b, c, dd_dt, dd_a, trap, angles) = self.split_proj(&proj)?;
        let (adt, dt, trap, b, c, x, z, angles) =
            self.preprocess(&dd_a, &dd_dt, &b, &c, &x, &z, &trap, &angles, seq_len)?;

        let li = self.layer_idx;
        let y = if self.is_mimo {
            let fwd = mamba3_mimo_fwd(
                &c,
                &b,
                &x,
                &adt,
                &dt,
                &trap,
                &self.c_bias,
                &self.b_bias,
                &angles,
                self.mimo_x.as_ref().unwrap(),
                self.mimo_z.as_ref(),
                self.mimo_o.as_ref(),
                Some(&self.d),
                Some(&z),
                Some(&state.angle_states[li]),
                Some(&state.ssm_states[li]),
                Some(&state.k_states[li]),
                Some(&state.v_states[li]),
            )?;
            state.angle_states[li] = fwd.angle_state;
            state.ssm_states[li] = fwd.ssm_state;
            state.k_states[li] = fwd.k_state;
            state.v_states[li] = fwd.v_state;
            fwd.out
        } else {
            let fwd = mamba3_siso_fwd(
                &c.squeeze(2)?,
                &b.squeeze(2)?,
                &x,
                &adt,
                &dt,
                &trap,
                &self.c_bias.squeeze(1)?,
                &self.b_bias.squeeze(1)?,
                &angles,
                Some(&self.d),
                Some(&z),
                Some(&state.angle_states[li]),
                Some(&state.ssm_states[li]),
                Some(&state.k_states[li].squeeze(1)?),
                Some(&state.v_states[li]),
            )?;
            state.angle_states[li] = fwd.angle_state;
            state.ssm_states[li] = fwd.ssm_state;
            state.k_states[li] = fwd.k_state.unsqueeze(1)?;
            state.v_states[li] = fwd.v_state;
            fwd.out
        };

        let y = y.reshape((b_sz, seq_len, self.d_inner))?;
        y.apply(&self.out_proj)
    }
}

#[derive(Clone, Debug)]
pub struct ResidualBlock {
    mixer: Mamba3Block,
    norm: RmsNorm,
}

impl ResidualBlock {
    pub fn new(layer_idx: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let norm = candle_nn::rms_norm(cfg.d_model, 1e-5, vb.pp("norm"))?;
        let mixer = Mamba3Block::new(layer_idx, cfg, vb.pp("mixer"))?;
        Ok(Self { mixer, norm })
    }

    fn forward(&self, xs: &Tensor, state: &mut State) -> Result<Tensor> {
        self.mixer.forward(&xs.apply(&self.norm)?, state)? + xs
    }

    fn forward_prefill(&self, xs: &Tensor, state: &mut State, chunk_size: usize) -> Result<Tensor> {
        let normed = xs.apply(&self.norm)?;
        self.mixer.forward_prefill(&normed, state, chunk_size)? + xs
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
        let embedding = candle_nn::embedding(cfg.vocab_size(), cfg.d_model, vb.pp("embeddings"))?;
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
        let mut xs = self.embedding.forward(input_ids)?;
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, state)?;
        }
        state.pos += 1;
        xs.apply(&self.norm_f)?.apply(&self.lm_head)
    }

    pub fn forward_prefill(
        &self,
        input_ids: &Tensor,
        state: &mut State,
        chunk_size: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len) = input_ids.dims2()?;
        let mut xs = self.embedding.forward(input_ids)?;
        for layer in self.layers.iter() {
            xs = layer.forward_prefill(&xs, state, chunk_size)?;
        }
        state.pos += seq_len;
        let xs = xs.reshape((b_sz * seq_len, xs.dim(D::Minus1)?))?;
        let logits = xs.apply(&self.norm_f)?.apply(&self.lm_head)?;
        logits.reshape((b_sz, seq_len, logits.dim(D::Minus1)?))
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}
