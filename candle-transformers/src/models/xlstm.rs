//! xLSTM (Extended Long Short-Term Memory) implementation.
//!
//! Based on Beck et al., "xLSTM: Extended Long Short-Term Memory" (NeurIPS 2024)
//! <https://arxiv.org/abs/2405.04517>
//!
//! This implementation supports the mLSTM (matrix memory LSTM) variant with:
//! - Matrix memory using covariance update rule
//! - Exponential gating with stabilization
//! - HuggingFace weight compatibility (NX-AI/xLSTM-7b)

use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Linear, RmsNorm, VarBuilder};

// ============================================================================
// GroupNormNoBias
// ============================================================================

/// GroupNorm without bias - xLSTM's multihead_norm only has weight, no bias
#[derive(Clone, Debug)]
pub struct GroupNormNoBias {
    weight: Tensor,
    eps: f64,
    num_groups: usize,
}

impl GroupNormNoBias {
    pub fn new(num_groups: usize, num_channels: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        if !num_channels.is_multiple_of(num_groups) {
            candle::bail!(
                "GroupNormNoBias: num_groups ({num_groups}) must divide num_channels ({num_channels})"
            )
        }
        let weight = vb.get(num_channels, "weight")?;
        Ok(Self {
            weight,
            eps,
            num_groups,
        })
    }
}

impl Module for GroupNormNoBias {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_shape = x.dims();
        if x_shape.len() < 3 {
            candle::bail!("input rank for GroupNormNoBias should be at least 3");
        }
        let (b_sz, n_channels) = (x_shape[0], x_shape[1]);
        let hidden_size = x_shape[2..].iter().product::<usize>() * n_channels / self.num_groups;

        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let x = x.reshape((b_sz, self.num_groups, hidden_size))?;
        let x = x.to_dtype(internal_dtype)?;
        let mean_x = (x.sum_keepdim(2)? / hidden_size as f64)?;
        let x = x.broadcast_sub(&mean_x)?;
        let norm_x = (x.sqr()?.sum_keepdim(2)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;

        let mut w_dims = vec![1; x_shape.len()];
        w_dims[1] = n_channels;
        let weight = self.weight.reshape(w_dims)?;

        x_normed
            .to_dtype(x_dtype)?
            .reshape(x_shape)?
            .broadcast_mul(&weight)
    }
}

// ============================================================================
// Config
// ============================================================================

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Embedding dimension (hidden size)
    pub embedding_dim: usize,
    /// Number of residual blocks (layers)
    pub num_blocks: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension (not directly used, derived from embedding_dim / num_heads)
    #[serde(default)]
    pub head_dim: usize,

    /// Query/Key dimension factor relative to embedding_dim
    #[serde(default = "default_qk_dim_factor")]
    pub qk_dim_factor: f64,
    /// Value dimension factor relative to embedding_dim
    #[serde(default = "default_v_dim_factor")]
    pub v_dim_factor: f64,

    /// FFN projection factor
    #[serde(default = "default_ffn_proj_factor")]
    pub ffn_proj_factor: f64,
    /// Round FFN intermediate size up to multiple of this
    #[serde(default = "default_round_up")]
    pub ffn_round_up_to_multiple_of: usize,
    /// Round mLSTM dimensions up to multiple of this
    #[serde(default = "default_round_up")]
    pub mlstm_round_up_to_multiple_of: usize,

    /// Epsilon for layer normalization
    #[serde(default = "default_eps")]
    pub norm_eps: f64,
    /// Epsilon for cell normalization (GroupNorm)
    #[serde(default = "default_eps")]
    pub cell_norm_eps: f64,

    /// Soft cap for gate pre-activations (bounds to ±cap)
    #[serde(default = "default_gate_soft_cap")]
    pub gate_soft_cap: f64,
    /// Soft cap for output logits
    #[serde(default = "default_output_soft_cap")]
    pub output_logit_soft_cap: f64,

    /// Chunk size for chunked processing (not used in step-by-step inference)
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,

    /// Whether to apply final normalization after all blocks
    #[serde(default = "default_true")]
    pub add_post_blocks_norm: bool,
    /// Whether to tie input embeddings with output LM head
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Whether to use bias in linear layers (not used, always false)
    #[serde(default)]
    pub use_bias: bool,

    /// Beginning of sequence token ID
    #[serde(default)]
    pub bos_token_id: usize,
    /// End of sequence token ID
    #[serde(default = "default_eos")]
    pub eos_token_id: usize,
    /// Padding token ID
    #[serde(default = "default_pad")]
    pub pad_token_id: usize,
}

fn default_qk_dim_factor() -> f64 {
    0.5
}
fn default_v_dim_factor() -> f64 {
    1.0
}
fn default_ffn_proj_factor() -> f64 {
    2.667
}
fn default_round_up() -> usize {
    64
}
fn default_eps() -> f64 {
    1e-6
}
fn default_gate_soft_cap() -> f64 {
    15.0
}
fn default_output_soft_cap() -> f64 {
    30.0
}
fn default_chunk_size() -> usize {
    64
}
fn default_true() -> bool {
    true
}
fn default_eos() -> usize {
    2
}
fn default_pad() -> usize {
    1
}

impl Config {
    /// Compute the query/key dimension (rounded up)
    pub fn qk_dim(&self) -> usize {
        let raw = (self.embedding_dim as f64 * self.qk_dim_factor) as usize;
        raw.div_ceil(self.mlstm_round_up_to_multiple_of) * self.mlstm_round_up_to_multiple_of
    }

    /// Compute the value dimension (rounded up)
    pub fn v_dim(&self) -> usize {
        let raw = (self.embedding_dim as f64 * self.v_dim_factor) as usize;
        raw.div_ceil(self.mlstm_round_up_to_multiple_of) * self.mlstm_round_up_to_multiple_of
    }

    /// Compute the FFN intermediate size (rounded up)
    pub fn ffn_intermediate_size(&self) -> usize {
        let raw = (self.embedding_dim as f64 * self.ffn_proj_factor) as usize;
        raw.div_ceil(self.ffn_round_up_to_multiple_of) * self.ffn_round_up_to_multiple_of
    }
}

// ============================================================================
// State Management
// ============================================================================

/// Per-layer mLSTM state for recurrent inference
#[derive(Debug, Clone)]
pub struct MLstmState {
    /// Matrix memory: (batch, num_heads, qk_head_dim, v_head_dim)
    pub c: Tensor,
    /// Normalizer vector: (batch, num_heads, qk_head_dim)
    pub n: Tensor,
    /// Max tracker for log-space stability: (batch, num_heads, 1)
    pub m: Tensor,
}

impl MLstmState {
    pub fn new(
        batch_size: usize,
        num_heads: usize,
        qk_head_dim: usize,
        v_head_dim: usize,
        _dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        // State always uses f32 for numerical stability (per HuggingFace inference_state_dtype)
        let state_dtype = DType::F32;
        Ok(Self {
            // C shape is (qk_d, v_d) to match HuggingFace: stores k ⊗ v^T
            c: Tensor::zeros(
                (batch_size, num_heads, qk_head_dim, v_head_dim),
                state_dtype,
                device,
            )?,
            n: Tensor::zeros((batch_size, num_heads, qk_head_dim), state_dtype, device)?,
            // Initialize to -inf for first max() to work correctly
            m: Tensor::full(f32::NEG_INFINITY, (batch_size, num_heads, 1), device)?,
        })
    }
}

/// Full model state containing per-layer mLSTM states
pub struct State {
    pub mlstm_states: Vec<MLstmState>,
    pub pos: usize,
}

impl State {
    pub fn new(batch_size: usize, cfg: &Config, dtype: DType, device: &Device) -> Result<Self> {
        let qk_head_dim = cfg.qk_dim() / cfg.num_heads;
        let v_head_dim = cfg.v_dim() / cfg.num_heads;
        let mlstm_states = (0..cfg.num_blocks)
            .map(|_| {
                MLstmState::new(
                    batch_size,
                    cfg.num_heads,
                    qk_head_dim,
                    v_head_dim,
                    dtype,
                    device,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            mlstm_states,
            pos: 0,
        })
    }
}

// ============================================================================
// MLstmBlock
// ============================================================================

/// mLSTM block with matrix memory and exponential gating
#[derive(Clone, Debug)]
pub struct MLstmBlock {
    // Projections (no bias)
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,

    // Gates
    igate_preact: Linear, // with bias
    fgate_preact: Linear, // with bias
    ogate_preact: Linear, // no bias

    // Normalization
    multihead_norm: GroupNormNoBias,

    // Config
    num_heads: usize,
    qk_head_dim: usize,
    v_head_dim: usize,
    gate_soft_cap: f64,
    eps: f64,
}

impl MLstmBlock {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let d = cfg.embedding_dim;
        let qk_dim = cfg.qk_dim();
        let v_dim = cfg.v_dim();
        let num_heads = cfg.num_heads;

        Ok(Self {
            q_proj: candle_nn::linear_no_bias(d, qk_dim, vb.pp("q"))?,
            k_proj: candle_nn::linear_no_bias(d, qk_dim, vb.pp("k"))?,
            v_proj: candle_nn::linear_no_bias(d, v_dim, vb.pp("v"))?,
            out_proj: candle_nn::linear_no_bias(v_dim, d, vb.pp("out_proj"))?,
            igate_preact: candle_nn::linear(d, num_heads, vb.pp("igate_preact"))?,
            fgate_preact: candle_nn::linear(d, num_heads, vb.pp("fgate_preact"))?,
            ogate_preact: candle_nn::linear_no_bias(d, v_dim, vb.pp("ogate_preact"))?,
            multihead_norm: GroupNormNoBias::new(
                num_heads,
                v_dim,
                cfg.cell_norm_eps,
                vb.pp("multihead_norm"),
            )?,
            num_heads,
            qk_head_dim: qk_dim / num_heads,
            v_head_dim: v_dim / num_heads,
            gate_soft_cap: cfg.gate_soft_cap,
            eps: cfg.cell_norm_eps,
        })
    }

    /// Single token forward with state update
    pub fn forward(&self, xs: &Tensor, state: &mut MLstmState) -> Result<Tensor> {
        let (batch, _) = xs.dims2()?;
        let h = self.num_heads;
        let qk_d = self.qk_head_dim;
        let v_d = self.v_head_dim;
        let input_dtype = xs.dtype();

        // 1. Project Q, K, V and reshape to heads
        // Convert to f32 for state operations (numerical stability)
        let q = self
            .q_proj
            .forward(xs)?
            .reshape((batch, h, qk_d))?
            .to_dtype(DType::F32)?; // (B, H, qk_d)
        let k = self
            .k_proj
            .forward(xs)?
            .reshape((batch, h, qk_d))?
            .to_dtype(DType::F32)?; // (B, H, qk_d)
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((batch, h, v_d))?
            .to_dtype(DType::F32)?; // (B, H, v_d)

        // 2. Normalize k by sqrt(d) for stable dot products
        let k = (&k / (qk_d as f64).sqrt())?;

        // 3. Compute gates in log-space (all in f32 for numerical stability)
        // Input gate: soft_cap(i_preact) used directly as log-space value
        // Forget gate: log_sigmoid(soft_cap(f_preact))
        // NOTE: Convert to f32 BEFORE soft_cap to avoid f16 precision issues
        let i_preact_f32 = self.igate_preact.forward(xs)?.to_dtype(DType::F32)?;
        let log_i = self.soft_cap(&i_preact_f32)?.unsqueeze(D::Minus1)?; // (B, H, 1)

        let f_preact_f32 = self.fgate_preact.forward(xs)?.to_dtype(DType::F32)?;
        let f_capped = self.soft_cap(&f_preact_f32)?;
        let log_f = self.log_sigmoid(&f_capped)?.unsqueeze(D::Minus1)?; // (B, H, 1)

        // 4. Stabilization: m_new = max(log_f + m_prev, log_i)
        let log_f_plus_m = (&log_f + &state.m)?;
        let m_new = log_f_plus_m.maximum(&log_i)?; // (B, H, 1)

        // 5. Compute stabilized gates
        let i_prime = (&log_i - &m_new)?.exp()?; // (B, H, 1)
        let f_prime = (&log_f_plus_m - &m_new)?.exp()?; // (B, H, 1)

        // 6. Memory update: C = f' * C + i' * outer(k, v)
        // Note: HuggingFace uses C shape (qk_d, v_d) with outer product k ⊗ v^T
        let k_col = k.unsqueeze(D::Minus1)?; // (B, H, qk_d, 1)
        let v_row = v.unsqueeze(2)?; // (B, H, 1, v_d)
        let outer_kv = k_col.matmul(&v_row)?; // (B, H, qk_d, v_d)

        let f_for_c = f_prime.unsqueeze(D::Minus1)?; // (B, H, 1, 1)
        let i_for_c = i_prime.unsqueeze(D::Minus1)?; // (B, H, 1, 1)
        state.c = (f_for_c.broadcast_mul(&state.c)? + i_for_c.broadcast_mul(&outer_kv)?)?;

        // 7. Normalizer update: n = f' * n + i' * k
        state.n = (f_prime.broadcast_mul(&state.n)? + i_prime.broadcast_mul(&k)?)?;
        state.m = m_new;

        // 8. Output: h = q^T @ C / max(|q^T @ n|, exp(-m)) + eps
        // With C shape (qk_d, v_d), we compute q^T @ C to get (v_d,)
        let q_row = q.unsqueeze(2)?; // (B, H, 1, qk_d)
        let qc = q_row.matmul(&state.c)?.squeeze(2)?; // (B, H, v_d)

        // Denominator: max(|q^T @ n|, exp(-m)) + eps per official mlstm_kernels
        // NOTE: state.m was just updated to m_new above
        let qn = (&q * &state.n)?.sum_keepdim(D::Minus1)?; // (B, H, 1)
        let qn_abs = qn.abs()?;
        let max_val = state.m.neg()?.exp()?; // exp(-m_new)
        let denom = (qn_abs.maximum(&max_val)? + self.eps)?;
        let h_raw = qc.broadcast_div(&denom)?;

        // 9. GroupNorm (requires 3D input) - convert back to input dtype
        let h_flat = h_raw
            .reshape((batch, self.num_heads * self.v_head_dim))?
            .to_dtype(input_dtype)?; // (B, v_dim)
        let h_3d = h_flat.unsqueeze(D::Minus1)?; // (B, v_dim, 1)
        let h_normed = self.multihead_norm.forward(&h_3d)?;
        let h_normed = h_normed.squeeze(D::Minus1)?; // (B, v_dim)

        // 10. Apply output gate (sigmoid gating)
        let ogate = candle_nn::ops::sigmoid(&self.ogate_preact.forward(xs)?)?; // (B, v_dim)
        let h_gated = (&h_normed * &ogate)?;

        // 11. Output projection
        self.out_proj.forward(&h_gated)
    }

    fn log_sigmoid(&self, x: &Tensor) -> Result<Tensor> {
        // log_sigmoid(x) = -softplus(-x) = log(sigmoid(x))
        // = -log(1 + exp(-x))
        let neg_x = x.neg()?;
        (neg_x.exp()? + 1.0)?.log()?.neg()
    }

    fn soft_cap(&self, x: &Tensor) -> Result<Tensor> {
        // soft_cap(x) = cap * tanh(x / cap)
        // Bounds output to ±cap, used for gate stability
        let cap = self.gate_soft_cap;
        (x / cap)?.tanh()? * cap
    }
}

// ============================================================================
// FfnBlock (SwiGLU)
// ============================================================================

/// Feed-forward network with SwiGLU activation
#[derive(Clone, Debug)]
pub struct FfnBlock {
    proj_up: Linear,
    proj_up_gate: Linear,
    proj_down: Linear,
}

impl FfnBlock {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let d = cfg.embedding_dim;
        let intermediate = cfg.ffn_intermediate_size();
        Ok(Self {
            proj_up: candle_nn::linear_no_bias(d, intermediate, vb.pp("proj_up"))?,
            proj_up_gate: candle_nn::linear_no_bias(d, intermediate, vb.pp("proj_up_gate"))?,
            proj_down: candle_nn::linear_no_bias(intermediate, d, vb.pp("proj_down"))?,
        })
    }
}

impl Module for FfnBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let up = self.proj_up.forward(xs)?;
        let gate = candle_nn::ops::silu(&self.proj_up_gate.forward(xs)?)?;
        (up * gate)?.apply(&self.proj_down)
    }
}

// ============================================================================
// ResidualBlock
// ============================================================================

/// Residual block with pre-norm mLSTM and FFN
#[derive(Clone, Debug)]
pub struct ResidualBlock {
    norm_mlstm: RmsNorm,
    mlstm: MLstmBlock,
    norm_ffn: RmsNorm,
    ffn: FfnBlock,
    layer_index: usize,
}

impl ResidualBlock {
    pub fn new(layer_index: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm_mlstm: candle_nn::rms_norm(cfg.embedding_dim, cfg.norm_eps, vb.pp("norm_mlstm"))?,
            mlstm: MLstmBlock::new(cfg, vb.pp("mlstm_layer"))?,
            norm_ffn: candle_nn::rms_norm(cfg.embedding_dim, cfg.norm_eps, vb.pp("norm_ffn"))?,
            ffn: FfnBlock::new(cfg, vb.pp("ffn"))?,
            layer_index,
        })
    }

    pub fn forward(&self, xs: &Tensor, state: &mut State) -> Result<Tensor> {
        // Pre-norm -> mLSTM -> Residual
        let h = self.mlstm.forward(
            &xs.apply(&self.norm_mlstm)?,
            &mut state.mlstm_states[self.layer_index],
        )?;
        let xs = (xs + h)?;

        // Pre-norm -> FFN -> Residual
        let h = xs.apply(&self.norm_ffn)?.apply(&self.ffn)?;
        xs + h
    }
}

// ============================================================================
// Model
// ============================================================================

/// xLSTM language model
#[derive(Clone, Debug)]
pub struct Model {
    embeddings: candle_nn::Embedding,
    blocks: Vec<ResidualBlock>,
    out_norm: Option<RmsNorm>,
    lm_head: Linear,
    output_logit_soft_cap: f64,
    dtype: DType,
    cfg: Config,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_backbone = vb.pp("backbone");

        // Token embeddings
        let embeddings = candle_nn::embedding(
            cfg.vocab_size,
            cfg.embedding_dim,
            vb_backbone.pp("embeddings"),
        )?;

        // Residual blocks
        let vb_blocks = vb_backbone.pp("blocks");
        let blocks = (0..cfg.num_blocks)
            .map(|i| ResidualBlock::new(i, cfg, vb_blocks.pp(i)))
            .collect::<Result<Vec<_>>>()?;

        // Optional output normalization
        let out_norm = if cfg.add_post_blocks_norm {
            Some(candle_nn::rms_norm(
                cfg.embedding_dim,
                cfg.norm_eps,
                vb_backbone.pp("out_norm"),
            )?)
        } else {
            None
        };

        // LM head (optionally tied to embeddings)
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embeddings.embeddings().clone(), None)
        } else {
            candle_nn::linear_no_bias(cfg.embedding_dim, cfg.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            embeddings,
            blocks,
            out_norm,
            lm_head,
            output_logit_soft_cap: cfg.output_logit_soft_cap,
            dtype: vb.dtype(),
            cfg: cfg.clone(),
        })
    }

    pub fn forward(&self, input_ids: &Tensor, state: &mut State) -> Result<Tensor> {
        let mut xs = self.embeddings.forward(input_ids)?;
        for block in &self.blocks {
            xs = block.forward(&xs, state)?;
        }
        if let Some(ref norm) = self.out_norm {
            xs = xs.apply(norm)?;
        }
        state.pos += 1;

        let logits = xs.apply(&self.lm_head)?;
        self.apply_output_soft_cap(&logits)
    }

    fn apply_output_soft_cap(&self, logits: &Tensor) -> Result<Tensor> {
        let cap = self.output_logit_soft_cap;
        (logits / cap)?.tanh()? * cap
    }

    pub fn new_state(&self, batch_size: usize, device: &Device) -> Result<State> {
        State::new(batch_size, &self.cfg, self.dtype, device)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }
}
