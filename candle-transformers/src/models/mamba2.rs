//! Mamba2 inference implementation.
//!
//! See ["Transformers are SSMs: Generalized Models and Efficient Algorithms
//! Through Structured State Space Duality"](https://arxiv.org/abs/2405.21060)

use crate::models::with_tracing::{linear_no_bias, Linear};
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{RmsNorm, VarBuilder};

const D_CONV: usize = 4;

/// Segment sum for SSD: computes cumsum[i] - cumsum[j] with lower triangular mask.
/// See Algorithm 1 in the Mamba2 paper.
fn segsum(x: &Tensor) -> Result<Tensor> {
    let device = x.device();
    let dtype = x.dtype();
    let t = x.dim(D::Minus1)?;

    let x_cumsum = x.cumsum(D::Minus1)?;

    let target_shape: Vec<usize> = {
        let mut shape = x.dims().to_vec();
        shape.push(t);
        shape
    };

    let x_cumsum_row = x_cumsum.unsqueeze(D::Minus1)?.broadcast_as(target_shape.as_slice())?;
    let x_cumsum_col = x_cumsum.unsqueeze(x.rank() - 1)?.broadcast_as(target_shape.as_slice())?;
    let x_segsum = (&x_cumsum_row - &x_cumsum_col)?;

    let mask_lower = Tensor::tril2(t, DType::U8, device)?;
    let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?
        .to_dtype(dtype)?
        .broadcast_as(x_segsum.shape())?;

    mask_lower
        .broadcast_as(x_segsum.shape())?
        .where_cond(&x_segsum, &neg_inf)
}

fn pad_to_chunk_size(x: &Tensor, chunk_size: usize) -> Result<(Tensor, usize)> {
    let seq_len = x.dim(1)?;
    let pad_len = (chunk_size - (seq_len % chunk_size)) % chunk_size;
    if pad_len == 0 {
        return Ok((x.clone(), 0));
    }

    let mut pad_shape = x.dims().to_vec();
    pad_shape[1] = pad_len;
    let padding = Tensor::zeros(pad_shape, x.dtype(), x.device())?;
    Ok((Tensor::cat(&[x, &padding], 1)?, pad_len))
}

fn reshape_into_chunks(x: &Tensor, chunk_size: usize) -> Result<Tensor> {
    let dims = x.dims();
    let b = dims[0];
    let l = dims[1];
    let n_chunks = l / chunk_size;

    let mut new_shape = vec![b, n_chunks, chunk_size];
    new_shape.extend_from_slice(&dims[2..]);
    x.reshape(new_shape)
}

fn reshape_from_chunks(x: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let b = dims[0];
    let n_chunks = dims[1];
    let chunk_size = dims[2];

    let mut new_shape = vec![b, n_chunks * chunk_size];
    new_shape.extend_from_slice(&dims[3..]);
    x.reshape(new_shape)
}

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

    fn d_xbc(&self) -> usize {
        self.d_inner() + 2 * self.ngroups * self.d_state
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
        let d_xbc = cfg.d_xbc();
        let nheads = cfg.nheads();
        let mut hs = Vec::with_capacity(cfg.n_layer);
        let mut conv_states = Vec::with_capacity(cfg.n_layer);
        for _ in 0..cfg.n_layer {
            let h = Tensor::zeros(
                (batch_size, nheads, cfg.headdim, cfg.d_state),
                dtype,
                device,
            )?;
            let conv = Tensor::zeros((batch_size, d_xbc, D_CONV), dtype, device)?;
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
    d_xbc: usize,
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
        let d_xbc = d_inner + 2 * ngroups * d_state;

        let proj_size = d_inner + d_xbc + nheads;
        let in_proj = linear_no_bias(cfg.d_model, proj_size, vb.pp("in_proj"))?;

        let conv1d_weight = vb.get((d_xbc, 1, D_CONV), "conv1d.weight")?;
        let conv1d_bias = vb.get(d_xbc, "conv1d.bias")?;

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
            d_xbc,
            headdim: cfg.headdim,
            nheads,
            ngroups,
            layer_idx,
        })
    }

    pub fn forward(&self, xs: &Tensor, state: &mut State) -> Result<Tensor> {
        let (b_sz, _dim) = xs.dims2()?;

        let proj = self.in_proj.forward(xs)?;

        let z = proj.narrow(D::Minus1, 0, self.d_inner)?;
        let xbc = proj.narrow(D::Minus1, self.d_inner, self.d_xbc)?;
        let dt = proj.narrow(D::Minus1, self.d_inner + self.d_xbc, self.nheads)?;

        let xbc_conv = self.apply_conv1d(&xbc, &mut state.conv_states[self.layer_idx])?;
        let xbc_conv = candle_nn::ops::silu(&xbc_conv)?;

        let x_conv = xbc_conv.narrow(D::Minus1, 0, self.d_inner)?;
        let b = xbc_conv.narrow(D::Minus1, self.d_inner, self.ngroups * self.d_state)?;
        let c = xbc_conv.narrow(
            D::Minus1,
            self.d_inner + self.ngroups * self.d_state,
            self.ngroups * self.d_state,
        )?;

        let dt_bias = self.dt_bias.broadcast_as(dt.shape())?;
        let dt = ((&dt + &dt_bias)?.exp()? + 1.)?.log()?; // softplus

        let a = self.a_log.exp()?.neg()?;

        let y = self.ssm_step(&x_conv, &a, &b, &c, &dt, state)?;

        let d = self.d.broadcast_as((b_sz, self.nheads))?;
        let x_skip = x_conv.reshape((b_sz, self.nheads, self.headdim))?;
        let y = (&y + x_skip.broadcast_mul(&d.unsqueeze(D::Minus1)?)?)?;
        let y = y.reshape((b_sz, self.d_inner))?;

        // Mamba2 applies gate before norm (MambaRMSNormGated)
        let y = (y * candle_nn::ops::silu(&z)?)?;
        let y = self.norm.forward(&y)?;

        self.out_proj.forward(&y)
    }

    fn apply_conv1d(&self, xbc: &Tensor, conv_state: &mut Tensor) -> Result<Tensor> {
        let (b_sz, d_xbc) = xbc.dims2()?;

        let shifted = conv_state.narrow(D::Minus1, 1, D_CONV - 1)?;
        let xbc_expanded = xbc.unsqueeze(D::Minus1)?;
        *conv_state = Tensor::cat(&[shifted, xbc_expanded], D::Minus1)?;

        let mut result = self.conv1d_bias.broadcast_as((b_sz, d_xbc))?;
        for i in 0..D_CONV {
            let w = self.conv1d_weight.i((.., 0, i))?;
            let xbc_i = conv_state.i((.., .., i))?;
            result = (result + w.broadcast_mul(&xbc_i)?)?;
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

        let dt_a = dt.broadcast_mul(a)?;
        let decay = dt_a.exp()?;
        let decay = decay.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?;
        let decay = decay.broadcast_as((b_sz, self.nheads, self.headdim, self.d_state))?;

        let x_unsq = x.unsqueeze(D::Minus1)?;
        let b_unsq = b.unsqueeze(2)?;
        let x_b = x_unsq.broadcast_mul(&b_unsq)?;

        let dt_expanded = dt.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?;
        let dt_expanded =
            dt_expanded.broadcast_as((b_sz, self.nheads, self.headdim, self.d_state))?;

        // SSM recurrence: h = exp(A*dt) * h + dt * (x ⊗ B)
        *h = ((&*h * &decay)? + (&dt_expanded * &x_b)?)?;

        let c_unsq = c.unsqueeze(2)?;
        let c_broadcast = c_unsq.broadcast_as(h.shape())?;
        let y = (&*h * &c_broadcast)?.sum(D::Minus1)?;

        Ok(y)
    }

    /// Chunked SSD algorithm for parallel prefill (Algorithm 1 in Mamba2 paper).
    fn ssd_chunked(
        &self,
        x: &Tensor,
        a: &Tensor,
        b: &Tensor,
        c: &Tensor,
        chunk_size: usize,
        initial_state: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let device = x.device();
        let dtype = x.dtype();
        let (batch, seq_len, nheads, headdim) = x.dims4()?;
        let (_, _, _, d_state) = b.dims4()?;
        let n_chunks = seq_len / chunk_size;

        let x = reshape_into_chunks(x, chunk_size)?;
        let a = reshape_into_chunks(a, chunk_size)?;
        let b = reshape_into_chunks(b, chunk_size)?;
        let c = reshape_into_chunks(c, chunk_size)?;

        let a = a.permute((0, 3, 1, 2))?;
        let a_cumsum = a.cumsum(D::Minus1)?;

        // Intra-chunk (diagonal blocks)
        let l = segsum(&a)?.exp()?;

        let c_expanded = c.unsqueeze(3)?;
        let b_expanded = b.unsqueeze(2)?;
        let cb_shape = (batch, n_chunks, chunk_size, chunk_size, nheads, d_state);
        let cb = (c_expanded.broadcast_as(cb_shape)? * b_expanded.broadcast_as(cb_shape)?)?.sum(D::Minus1)?;
        let cb = cb.permute((0, 1, 4, 2, 3))?;

        let l_t = l.permute((0, 2, 1, 3, 4))?;
        let cb_l = (&cb * &l_t)?;

        let x_t = x.permute((0, 1, 3, 2, 4))?;
        let y_diag_shape = (batch, n_chunks, nheads, chunk_size, chunk_size, headdim);
        let y_diag = (cb_l.unsqueeze(D::Minus1)?.broadcast_as(y_diag_shape)?
            * x_t.unsqueeze(3)?.broadcast_as(y_diag_shape)?)?
            .sum(4)?
            .permute((0, 1, 3, 2, 4))?;

        // Intra-chunk states
        let a_last = a_cumsum.narrow(D::Minus1, chunk_size - 1, 1)?;
        let decay_states = (a_last.broadcast_as(a_cumsum.shape())? - &a_cumsum)?.exp()?;

        let decay_s = decay_states.permute((0, 2, 1, 3))?.unsqueeze(D::Minus1)?;
        let b_t = b.permute((0, 1, 3, 2, 4))?;
        let b_weighted = b_t.broadcast_mul(&decay_s)?;

        let x_t2 = x.permute((0, 1, 3, 2, 4))?;
        let states_shape = (batch, n_chunks, nheads, chunk_size, headdim, d_state);
        let states = (x_t2.unsqueeze(D::Minus1)?.broadcast_as(states_shape)?
            * b_weighted.unsqueeze(4)?.broadcast_as(states_shape)?)?
            .sum(3)?;

        // Inter-chunk recurrence
        let init_state = match initial_state {
            Some(s) => s.unsqueeze(1)?,
            None => Tensor::zeros((batch, 1, nheads, headdim, d_state), dtype, device)?,
        };
        let states_with_init = Tensor::cat(&[&init_state, &states], 1)?;

        let a_chunk = a_cumsum.narrow(D::Minus1, chunk_size - 1, 1)?.squeeze(D::Minus1)?;
        let zeros = Tensor::zeros((batch, nheads, 1), dtype, device)?;
        let a_chunk_padded = Tensor::cat(&[&zeros, &a_chunk], D::Minus1)?;
        let decay_chunk = segsum(&a_chunk_padded)?.exp()?;

        let states_p = states_with_init.permute((0, 2, 1, 3, 4))?;
        let inter_shape = (batch, nheads, n_chunks + 1, n_chunks + 1, headdim, d_state);
        let new_states = (decay_chunk.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?.broadcast_as(inter_shape)?
            * states_p.unsqueeze(2)?.broadcast_as(inter_shape)?)?
            .sum(3)?
            .permute((0, 2, 1, 3, 4))?;

        let states_out = new_states.narrow(1, 0, n_chunks)?;
        let final_state = new_states.narrow(1, n_chunks, 1)?.squeeze(1)?;

        // State-to-output (off-diagonal blocks)
        let state_decay_out = a_cumsum.exp()?;

        let c_t2 = c.permute((0, 1, 3, 2, 4))?;
        let off_shape = (batch, n_chunks, nheads, chunk_size, headdim, d_state);
        let c_states = (c_t2.unsqueeze(4)?.broadcast_as(off_shape)?
            * states_out.unsqueeze(3)?.broadcast_as(off_shape)?)?
            .sum(D::Minus1)?;

        let decay_out = state_decay_out.permute((0, 2, 1, 3))?.unsqueeze(D::Minus1)?;
        let y_off = c_states.broadcast_mul(&decay_out)?.permute((0, 1, 3, 2, 4))?;

        let y = (&y_diag + &y_off)?;
        let y = reshape_from_chunks(&y)?;

        Ok((y, final_state))
    }

    pub fn forward_prefill(
        &self,
        xs: &Tensor,
        state: &mut State,
        chunk_size: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = xs.dims3()?;

        let (xs, pad_len) = pad_to_chunk_size(xs, chunk_size)?;
        let padded_len = xs.dim(1)?;

        let proj = xs.apply(&self.in_proj)?;

        let z = proj.narrow(D::Minus1, 0, self.d_inner)?;
        let xbc = proj.narrow(D::Minus1, self.d_inner, self.d_xbc)?;
        let dt = proj.narrow(D::Minus1, self.d_inner + self.d_xbc, self.nheads)?;

        let xbc_t = xbc.transpose(1, 2)?;
        let pad = Tensor::zeros((b_sz, self.d_xbc, D_CONV - 1), xbc.dtype(), xbc.device())?;
        let xbc_padded = Tensor::cat(&[&pad, &xbc_t], D::Minus1)?;
        let xbc_conv = xbc_padded.conv1d(
            &self.conv1d_weight,
            0,
            1,
            1,
            self.d_xbc,
        )?;
        let xbc_conv = xbc_conv
            .broadcast_add(&self.conv1d_bias.reshape((1, self.d_xbc, 1))?)?
            .transpose(1, 2)?;
        let xbc_conv = candle_nn::ops::silu(&xbc_conv)?;

        // Update conv_state from real sequence tokens (not padding) for correct autoregressive behavior
        let start = seq_len.saturating_sub(D_CONV);
        let count = D_CONV.min(seq_len);
        let last_tokens = xbc.narrow(1, start, count)?;
        let last_tokens = last_tokens.transpose(1, 2)?;
        if count >= D_CONV {
            state.conv_states[self.layer_idx] = last_tokens.contiguous()?;
        } else {
            let existing = state.conv_states[self.layer_idx].narrow(D::Minus1, count, D_CONV - count)?;
            state.conv_states[self.layer_idx] = Tensor::cat(&[&existing, &last_tokens], D::Minus1)?;
        }

        let x_conv = xbc_conv.narrow(D::Minus1, 0, self.d_inner)?;
        let bc = xbc_conv.narrow(D::Minus1, self.d_inner, 2 * self.ngroups * self.d_state)?;
        let b = bc.narrow(D::Minus1, 0, self.ngroups * self.d_state)?;
        let c = bc.narrow(D::Minus1, self.ngroups * self.d_state, self.ngroups * self.d_state)?;

        let dt_bias = self.dt_bias.broadcast_as(dt.shape())?;
        let dt = ((&dt + &dt_bias)?.exp()? + 1.)?.log()?;

        let a = self.a_log.exp()?.neg()?;
        let mut a_dt = dt.broadcast_mul(&a)?;

        let mut x_ssd = x_conv.reshape((b_sz, padded_len, self.nheads, self.headdim))?;

        // Zero out padding to prevent it from affecting chunk state computation
        if pad_len > 0 {
            let mask_ones = Tensor::ones((b_sz, seq_len, self.nheads, self.headdim), x_ssd.dtype(), x_ssd.device())?;
            let mask_zeros = Tensor::zeros((b_sz, pad_len, self.nheads, self.headdim), x_ssd.dtype(), x_ssd.device())?;
            let mask = Tensor::cat(&[&mask_ones, &mask_zeros], 1)?;
            x_ssd = x_ssd.broadcast_mul(&mask)?;

            let mask_ones_a = Tensor::ones((b_sz, seq_len, self.nheads), a_dt.dtype(), a_dt.device())?;
            let mask_zeros_a = Tensor::zeros((b_sz, pad_len, self.nheads), a_dt.dtype(), a_dt.device())?;
            let mask_a = Tensor::cat(&[&mask_ones_a, &mask_zeros_a], 1)?;
            a_dt = a_dt.broadcast_mul(&mask_a)?;
        }

        let heads_per_group = self.nheads / self.ngroups;
        let b = b.reshape((b_sz, padded_len, self.ngroups, self.d_state))?;
        let b = b
            .unsqueeze(3)?
            .broadcast_as((b_sz, padded_len, self.ngroups, heads_per_group, self.d_state))?
            .reshape((b_sz, padded_len, self.nheads, self.d_state))?;
        // Discretize B: B_bar = dt * B (ZOH discretization absorbed into ssd_chunked)
        let b = b.broadcast_mul(&dt.unsqueeze(D::Minus1)?)?;
        let c = c.reshape((b_sz, padded_len, self.ngroups, self.d_state))?;
        let c = c
            .unsqueeze(3)?
            .broadcast_as((b_sz, padded_len, self.ngroups, heads_per_group, self.d_state))?
            .reshape((b_sz, padded_len, self.nheads, self.d_state))?;

        let initial_state = Some(&state.hs[self.layer_idx]);
        let (y, final_state) = self.ssd_chunked(&x_ssd, &a_dt, &b, &c, chunk_size, initial_state)?;
        state.hs[self.layer_idx] = final_state;

        let y = y.reshape((b_sz, padded_len, self.d_inner))?;

        let d = self.d.unsqueeze(0)?.unsqueeze(0)?;
        let x_skip = x_conv.reshape((b_sz, padded_len, self.nheads, self.headdim))?;
        let y = (&y.reshape((b_sz, padded_len, self.nheads, self.headdim))?
            + x_skip.broadcast_mul(&d.unsqueeze(D::Minus1)?)?)?;
        let y = y.reshape((b_sz, padded_len, self.d_inner))?;

        let y = (y * candle_nn::ops::silu(&z)?)?;
        let y = y.reshape((b_sz * padded_len, self.d_inner))?;
        let y = self.norm.forward(&y)?;
        let y = y.reshape((b_sz, padded_len, self.d_inner))?;

        let y = y.apply(&self.out_proj)?;

        if pad_len > 0 {
            y.narrow(1, 0, seq_len)
        } else {
            Ok(y)
        }
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

    fn forward_prefill(
        &self,
        xs: &Tensor,
        state: &mut State,
        chunk_size: usize,
    ) -> Result<Tensor> {
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
        let _b_size = input_ids.dims1()?;
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
