//! Recurrent Neural Networks
use candle::{DType, Device, IndexOp, Result, Tensor};

/// Trait for Recurrent Neural Networks.
#[allow(clippy::upper_case_acronyms)]
pub trait RNN {
    type State: Clone;

    /// A zero state from which the recurrent network is usually initialized.
    fn zero_state(&self, batch_dim: usize) -> Result<Self::State>;

    /// Applies a single step of the recurrent network.
    ///
    /// The input should have dimensions [batch_size, features].
    fn step(&self, input: &Tensor, state: &Self::State) -> Result<Self::State>;

    /// Applies multiple steps of the recurrent network.
    ///
    /// The input should have dimensions [batch_size, seq_len, features].
    /// The initial state is the result of applying zero_state.
    fn seq(&self, input: &Tensor) -> Result<Vec<Self::State>> {
        let batch_dim = input.dim(0)?;
        let state = self.zero_state(batch_dim)?;
        self.seq_init(input, &state)
    }

    /// Applies multiple steps of the recurrent network.
    ///
    /// The input should have dimensions [batch_size, seq_len, features].
    fn seq_init(&self, input: &Tensor, init_state: &Self::State) -> Result<Vec<Self::State>> {
        let (_b_size, seq_len, _features) = input.dims3()?;
        let mut output = Vec::with_capacity(seq_len);
        for seq_index in 0..seq_len {
            let input = input.i((.., seq_index, ..))?.contiguous()?;
            let state = if seq_index == 0 {
                self.step(&input, init_state)?
            } else {
                self.step(&input, &output[seq_index - 1])?
            };
            output.push(state);
        }
        Ok(output)
    }

    /// Converts a sequence of state to a tensor.
    fn states_to_tensor(&self, states: &[Self::State]) -> Result<Tensor>;
}

/// The state for a LSTM network, this contains two tensors.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone)]
pub struct LSTMState {
    h: Tensor,
    c: Tensor,
}

impl LSTMState {
    /// The hidden state vector, which is also the output of the LSTM.
    pub fn h(&self) -> &Tensor {
        &self.h
    }

    /// The cell state vector.
    pub fn c(&self) -> &Tensor {
        &self.c
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy)]
pub struct LSTMConfig {
    pub w_ih_init: super::Init,
    pub w_hh_init: super::Init,
    pub b_ih_init: Option<super::Init>,
    pub b_hh_init: Option<super::Init>,
    pub layer_idx: usize,
}

impl Default for LSTMConfig {
    fn default() -> Self {
        Self {
            w_ih_init: super::init::DEFAULT_KAIMING_UNIFORM,
            w_hh_init: super::init::DEFAULT_KAIMING_UNIFORM,
            b_ih_init: Some(super::Init::Const(0.)),
            b_hh_init: Some(super::Init::Const(0.)),
            layer_idx: 0,
        }
    }
}

impl LSTMConfig {
    pub fn default_no_bias() -> Self {
        Self {
            w_ih_init: super::init::DEFAULT_KAIMING_UNIFORM,
            w_hh_init: super::init::DEFAULT_KAIMING_UNIFORM,
            b_ih_init: None,
            b_hh_init: None,
            layer_idx: 0,
        }
    }
}

/// A Long Short-Term Memory (LSTM) layer.
///
/// <https://en.wikipedia.org/wiki/Long_short-term_memory>
#[allow(clippy::upper_case_acronyms, unused)]
#[derive(Clone, Debug)]
pub struct LSTM {
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Option<Tensor>,
    b_hh: Option<Tensor>,
    hidden_dim: usize,
    config: LSTMConfig,
    device: Device,
    dtype: DType,
}

/// Creates a LSTM layer.
pub fn lstm(
    in_dim: usize,
    hidden_dim: usize,
    config: LSTMConfig,
    vb: crate::VarBuilder,
) -> Result<LSTM> {
    let layer_idx = config.layer_idx;
    let w_ih = vb.get_with_hints(
        (4 * hidden_dim, in_dim),
        &format!("weight_ih_l{layer_idx}"), // Only a single layer is supported.
        config.w_ih_init,
    )?;
    let w_hh = vb.get_with_hints(
        (4 * hidden_dim, hidden_dim),
        &format!("weight_hh_l{layer_idx}"), // Only a single layer is supported.
        config.w_hh_init,
    )?;
    let b_ih = match config.b_ih_init {
        Some(init) => {
            Some(vb.get_with_hints(4 * hidden_dim, &format!("bias_ih_l{layer_idx}"), init)?)
        }
        None => None,
    };
    let b_hh = match config.b_hh_init {
        Some(init) => {
            Some(vb.get_with_hints(4 * hidden_dim, &format!("bias_hh_l{layer_idx}"), init)?)
        }
        None => None,
    };
    Ok(LSTM {
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        hidden_dim,
        config,
        device: vb.device().clone(),
        dtype: vb.dtype(),
    })
}

impl RNN for LSTM {
    type State = LSTMState;

    fn zero_state(&self, batch_dim: usize) -> Result<Self::State> {
        let zeros =
            Tensor::zeros((batch_dim, self.hidden_dim), self.dtype, &self.device)?.contiguous()?;
        Ok(Self::State {
            h: zeros.clone(),
            c: zeros.clone(),
        })
    }

    fn step(&self, input: &Tensor, in_state: &Self::State) -> Result<Self::State> {
        let w_ih = input.matmul(&self.w_ih.t()?)?;
        let w_hh = in_state.h.matmul(&self.w_hh.t()?)?;
        let w_ih = match &self.b_ih {
            None => w_ih,
            Some(b_ih) => w_ih.broadcast_add(b_ih)?,
        };
        let w_hh = match &self.b_hh {
            None => w_hh,
            Some(b_hh) => w_hh.broadcast_add(b_hh)?,
        };
        let chunks = (&w_ih + &w_hh)?.chunk(4, 1)?;
        let in_gate = crate::ops::sigmoid(&chunks[0])?;
        let forget_gate = crate::ops::sigmoid(&chunks[1])?;
        let cell_gate = chunks[2].tanh()?;
        let out_gate = crate::ops::sigmoid(&chunks[3])?;

        let next_c = ((forget_gate * &in_state.c)? + (in_gate * cell_gate)?)?;
        let next_h = (out_gate * next_c.tanh()?)?;
        Ok(LSTMState {
            c: next_c,
            h: next_h,
        })
    }

    fn states_to_tensor(&self, states: &[Self::State]) -> Result<Tensor> {
        let states = states.iter().map(|s| s.h.clone()).collect::<Vec<_>>();
        Tensor::stack(&states, 1)
    }
}

/// The state for a GRU network, this contains a single tensor.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone)]
pub struct GRUState {
    h: Tensor,
}

impl GRUState {
    /// The hidden state vector, which is also the output of the LSTM.
    pub fn h(&self) -> &Tensor {
        &self.h
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy)]
pub struct GRUConfig {
    pub w_ih_init: super::Init,
    pub w_hh_init: super::Init,
    pub b_ih_init: Option<super::Init>,
    pub b_hh_init: Option<super::Init>,
}

impl Default for GRUConfig {
    fn default() -> Self {
        Self {
            w_ih_init: super::init::DEFAULT_KAIMING_UNIFORM,
            w_hh_init: super::init::DEFAULT_KAIMING_UNIFORM,
            b_ih_init: Some(super::Init::Const(0.)),
            b_hh_init: Some(super::Init::Const(0.)),
        }
    }
}

impl GRUConfig {
    pub fn default_no_bias() -> Self {
        Self {
            w_ih_init: super::init::DEFAULT_KAIMING_UNIFORM,
            w_hh_init: super::init::DEFAULT_KAIMING_UNIFORM,
            b_ih_init: None,
            b_hh_init: None,
        }
    }
}

/// A Gated Recurrent Unit (GRU) layer.
///
/// <https://en.wikipedia.org/wiki/Gated_recurrent_unit>
#[allow(clippy::upper_case_acronyms, unused)]
#[derive(Clone, Debug)]
pub struct GRU {
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Option<Tensor>,
    b_hh: Option<Tensor>,
    hidden_dim: usize,
    config: GRUConfig,
    device: Device,
    dtype: DType,
}

/// Creates a GRU layer.
pub fn gru(
    in_dim: usize,
    hidden_dim: usize,
    config: GRUConfig,
    vb: crate::VarBuilder,
) -> Result<GRU> {
    let w_ih = vb.get_with_hints(
        (3 * hidden_dim, in_dim),
        "weight_ih_l0", // Only a single layer is supported.
        config.w_ih_init,
    )?;
    let w_hh = vb.get_with_hints(
        (3 * hidden_dim, hidden_dim),
        "weight_hh_l0", // Only a single layer is supported.
        config.w_hh_init,
    )?;
    let b_ih = match config.b_ih_init {
        Some(init) => Some(vb.get_with_hints(3 * hidden_dim, "bias_ih_l0", init)?),
        None => None,
    };
    let b_hh = match config.b_hh_init {
        Some(init) => Some(vb.get_with_hints(3 * hidden_dim, "bias_hh_l0", init)?),
        None => None,
    };
    Ok(GRU {
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        hidden_dim,
        config,
        device: vb.device().clone(),
        dtype: vb.dtype(),
    })
}

impl RNN for GRU {
    type State = GRUState;

    fn zero_state(&self, batch_dim: usize) -> Result<Self::State> {
        let h =
            Tensor::zeros((batch_dim, self.hidden_dim), self.dtype, &self.device)?.contiguous()?;
        Ok(Self::State { h })
    }

    fn step(&self, input: &Tensor, in_state: &Self::State) -> Result<Self::State> {
        let w_ih = input.matmul(&self.w_ih.t()?)?;
        let w_hh = in_state.h.matmul(&self.w_hh.t()?)?;
        let w_ih = match &self.b_ih {
            None => w_ih,
            Some(b_ih) => w_ih.broadcast_add(b_ih)?,
        };
        let w_hh = match &self.b_hh {
            None => w_hh,
            Some(b_hh) => w_hh.broadcast_add(b_hh)?,
        };
        let chunks_ih = w_ih.chunk(3, 1)?;
        let chunks_hh = w_hh.chunk(3, 1)?;
        let r_gate = crate::ops::sigmoid(&(&chunks_ih[0] + &chunks_hh[0])?)?;
        let z_gate = crate::ops::sigmoid(&(&chunks_ih[1] + &chunks_hh[1])?)?;
        let n_gate = (&chunks_ih[2] + (r_gate * &chunks_hh[2])?)?.tanh();

        let next_h = ((&z_gate * &in_state.h)? - ((&z_gate - 1.)? * n_gate)?)?;
        Ok(GRUState { h: next_h })
    }

    fn states_to_tensor(&self, states: &[Self::State]) -> Result<Tensor> {
        let states = states.iter().map(|s| s.h.clone()).collect::<Vec<_>>();
        Tensor::cat(&states, 1)
    }
}
