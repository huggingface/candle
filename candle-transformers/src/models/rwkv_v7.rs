//! RWKV v7 "Goose" (x070) model implementation.
//!
//! The [RWKV model](https://wiki.rwkv.com/) is a recurrent neural network model
//! with performance on par with transformer architectures. This implements the v7
//! architecture (codenamed "Goose"), which introduces:
//!
//! - Delta-rule state update with in-context learning
//! - Value residual stream across layers
//! - LoRA-style projections for decay, gate, and ICL parameters
//!
//! Three variants are supported:
//! - **v7**: Base architecture with linear attention + squared ReLU FFN
//! - **v7a**: Adds DeepEmbed token-dependent gating to the FFN
//! - **v7b**: Adds Deep Embedding Attention (DEA) — a full quadratic attention alongside RWKV
//!
//! # References
//!
//! - [RWKV-7 reference code](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7)

use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{embedding, Embedding, VarBuilder};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Which RWKV v7 variant to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
pub enum ModelVersion {
    V7,
    V7a,
    V7b,
}

/// Configuration for RWKV v7 models.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub version: ModelVersion,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    #[serde(default = "default_head_size")]
    pub head_size: usize,
    pub intermediate_size: Option<usize>,
    #[serde(default = "default_rescale_every")]
    pub rescale_every: usize,
}

fn default_head_size() -> usize {
    64
}

fn default_rescale_every() -> usize {
    0
}

impl Config {
    fn n_heads(&self) -> usize {
        self.hidden_size / self.head_size
    }

    fn dim_ffn(&self) -> usize {
        self.intermediate_size.unwrap_or(self.hidden_size * 4)
    }
}

/// Infer LoRA dimensions from actual weight shapes in the first block.
/// This is more robust than computing from a formula, as different
/// model sizes may use different LoRA dimensions.
fn infer_lora_dims(vb: &VarBuilder) -> Result<(usize, usize, usize, usize)> {
    let att = vb.pp("blocks").pp(0).pp("att");
    let d_decay = att.get_unchecked("w1")?.dim(1)?;
    let d_aaa = att.get_unchecked("a1")?.dim(1)?;
    let d_mv = att.get_unchecked("v1")?.dim(1)?;
    let d_gate = att.get_unchecked("g1")?.dim(1)?;
    Ok((d_decay, d_aaa, d_mv, d_gate))
}

// ─── State ───────────────────────────────────────────────────────────────────

/// Per-layer persistent state for RWKV v7 inference.
pub struct StatePerLayer {
    /// Previous token embedding for time-mix shifting. Shape: `(hidden_size,)`.
    pub att_x_prev: Tensor,
    /// WKV state matrix. Shape: `(n_heads, head_size, head_size)` in f32.
    pub att_kv: Tensor,
    /// Previous token embedding for channel-mix shifting. Shape: `(hidden_size,)`.
    pub ffn_x_prev: Tensor,
}

/// KV cache state for DEA (v7b only).
pub struct DeaState {
    /// Token IDs seen so far (growing).
    pub token_ids: Vec<u32>,
    /// Per-layer K projections cache. Each entry: `(seq_len, 32)`.
    pub k_cache: Vec<Tensor>,
    /// Per-layer V projections cache. Each entry: `(seq_len, 32)`.
    pub v_cache: Vec<Tensor>,
    /// Per-layer previous Q for token-shifting. Each entry: `(256,)`.
    pub q_prev: Vec<Tensor>,
}

/// Full inference state for RWKV v7.
pub struct State {
    pub per_layer: Vec<StatePerLayer>,
    pub dea: Option<DeaState>,
    pub pos: usize,
}

impl State {
    /// Create state with F32 precision (default, most compatible).
    pub fn new(cfg: &Config, dev: &Device) -> Result<Self> {
        Self::new_with_dtype(cfg, dev, DType::F32)
    }

    /// Create state with specified dtype (F16/BF16 for faster inference).
    ///
    /// Note: The KV state (`att_kv`) always uses F32 for numerical stability
    /// in the delta-rule accumulation. Other state tensors use the specified dtype.
    pub fn new_with_dtype(cfg: &Config, dev: &Device, dtype: DType) -> Result<Self> {
        let n_heads = cfg.n_heads();
        let mut per_layer = Vec::with_capacity(cfg.num_hidden_layers);
        for _layer_idx in 0..cfg.num_hidden_layers {
            per_layer.push(StatePerLayer {
                att_x_prev: Tensor::zeros(cfg.hidden_size, dtype, dev)?,
                // KV state stays F32 for numerical stability in accumulation
                att_kv: Tensor::zeros((n_heads, cfg.head_size, cfg.head_size), DType::F32, dev)?,
                ffn_x_prev: Tensor::zeros(cfg.hidden_size, dtype, dev)?,
            });
        }
        let dea = if cfg.version == ModelVersion::V7b {
            let mut k_cache = Vec::with_capacity(cfg.num_hidden_layers);
            let mut v_cache = Vec::with_capacity(cfg.num_hidden_layers);
            let mut q_prev = Vec::with_capacity(cfg.num_hidden_layers);
            for _ in 0..cfg.num_hidden_layers {
                k_cache.push(Tensor::zeros((0, 32), dtype, dev)?);
                v_cache.push(Tensor::zeros((0, 32), dtype, dev)?);
                q_prev.push(Tensor::zeros(256, dtype, dev)?);
            }
            Some(DeaState {
                token_ids: Vec::new(),
                k_cache,
                v_cache,
                q_prev,
            })
        } else {
            None
        };
        Ok(Self {
            per_layer,
            dea,
            pos: 0,
        })
    }
}

// ─── Tokenizer ───────────────────────────────────────────────────────────────

pub use crate::models::rwkv_v5::Tokenizer;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Layer normalization that preserves input dtype when possible.
/// All internal computation happens in F32 for numerical stability,
/// then converts back to the original dtype.
fn layer_norm(xs: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Result<Tensor> {
    let xs_dtype = xs.dtype();
    let needs_conversion = xs_dtype != DType::F32;

    // Convert to F32 for all internal computation (numerical stability)
    let xs_f32 = if needs_conversion {
        xs.to_dtype(DType::F32)?
    } else {
        xs.clone()
    };

    let dim = xs_f32.dim(candle::D::Minus1)?;
    let mean = (xs_f32.sum_keepdim(candle::D::Minus1)? / dim as f64)?;
    let centered = xs_f32.broadcast_sub(&mean)?;
    let var = (centered.sqr()?.sum_keepdim(candle::D::Minus1)? / dim as f64)?;
    let xs = centered.broadcast_div(&(var + eps)?.sqrt()?)?;

    // Convert back to original dtype if needed
    let xs = if needs_conversion {
        xs.to_dtype(xs_dtype)?
    } else {
        xs
    };
    let xs = xs.broadcast_mul(weight)?.broadcast_add(bias)?;
    Ok(xs)
}

// ─── TimeMix (Attention) ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct TimeMix {
    // Token-shift lerp mixes (pre-squeezed to 1D for efficiency)
    x_r: Tensor,
    x_w: Tensor,
    x_k: Tensor,
    x_v: Tensor,
    x_a: Tensor,
    x_g: Tensor,
    // Decay LoRA (w0 pre-squeezed)
    w0: Tensor,
    w1: Tensor,
    w2: Tensor,
    // ICL rate LoRA (a0 pre-squeezed)
    a0: Tensor,
    a1: Tensor,
    a2: Tensor,
    // Value residual LoRA (None for layer 0, v0 pre-squeezed)
    v0: Option<Tensor>,
    v1: Option<Tensor>,
    v2: Option<Tensor>,
    // Gate LoRA
    g1: Tensor,
    g2: Tensor,
    // Key processing (pre-squeezed)
    k_k: Tensor,
    k_a: Tensor,
    // Bonus term (pre-flattened to 1D)
    r_k: Tensor,
    // Linear projections (pre-transposed for efficiency)
    receptance_t: Tensor,
    key_t: Tensor,
    value_t: Tensor,
    output_t: Tensor,
    // GroupNorm weights
    ln_x_weight: Tensor,
    ln_x_bias: Tensor,
    // Metadata
    layer_id: usize,
    n_heads: usize,
    head_size: usize,
}

impl TimeMix {
    fn new(
        layer_id: usize,
        cfg: &Config,
        lora: (usize, usize, usize, usize),
        vb: VarBuilder,
    ) -> Result<Self> {
        let c = cfg.hidden_size;
        let (d_decay, d_aaa, d_mv, d_gate) = lora;
        let n_heads = cfg.n_heads();
        let head_size = cfg.head_size;

        // Pre-squeeze (1,1,C) -> (C,) at load time to avoid per-token squeeze calls
        let x_r = vb.get((1, 1, c), "x_r")?.squeeze(0)?.squeeze(0)?;
        let x_w = vb.get((1, 1, c), "x_w")?.squeeze(0)?.squeeze(0)?;
        let x_k = vb.get((1, 1, c), "x_k")?.squeeze(0)?.squeeze(0)?;
        let x_v = vb.get((1, 1, c), "x_v")?.squeeze(0)?.squeeze(0)?;
        let x_a = vb.get((1, 1, c), "x_a")?.squeeze(0)?.squeeze(0)?;
        let x_g = vb.get((1, 1, c), "x_g")?.squeeze(0)?.squeeze(0)?;

        let w0 = vb.get((1, 1, c), "w0")?.squeeze(0)?.squeeze(0)?;
        let w1 = vb.get((c, d_decay), "w1")?;
        let w2 = vb.get((d_decay, c), "w2")?;

        let a0 = vb.get((1, 1, c), "a0")?.squeeze(0)?.squeeze(0)?;
        let a1 = vb.get((c, d_aaa), "a1")?;
        let a2 = vb.get((d_aaa, c), "a2")?;

        // v0/v1/v2 exist for all layers in the weights file, but are only used for layers > 0
        // (layer 0 stores v_first instead of blending toward it).
        let (v0, v1, v2) = if layer_id > 0 {
            (
                Some(vb.get((1, 1, c), "v0")?.squeeze(0)?.squeeze(0)?),
                Some(vb.get((c, d_mv), "v1")?),
                Some(vb.get((d_mv, c), "v2")?),
            )
        } else {
            // Load and discard — these tensors exist in the file but are ignored at layer 0
            let _ = vb.get((1, 1, c), "v0");
            let _ = vb.get((c, d_mv), "v1");
            let _ = vb.get((d_mv, c), "v2");
            (None, None, None)
        };

        let g1 = vb.get((c, d_gate), "g1")?;
        let g2 = vb.get((d_gate, c), "g2")?;

        let k_k = vb.get((1, 1, c), "k_k")?.squeeze(0)?.squeeze(0)?;
        let k_a = vb.get((1, 1, c), "k_a")?.squeeze(0)?.squeeze(0)?;
        // Pre-flatten r_k to (H*N,) to avoid reshape in forward
        let r_k = vb
            .get((n_heads, head_size), "r_k")?
            .reshape(n_heads * head_size)?;

        // Linear projections — pre-transpose and make contiguous for optimal memory access
        let receptance_t = vb.get((c, c), "receptance.weight")?.t()?.contiguous()?;
        let key_t = vb.get((c, c), "key.weight")?.t()?.contiguous()?;
        let value_t = vb.get((c, c), "value.weight")?.t()?.contiguous()?;
        let output_t = vb.get((c, c), "output.weight")?.t()?.contiguous()?;

        let ln_x_weight = vb.get(c, "ln_x.weight")?;
        let ln_x_bias = vb.get(c, "ln_x.bias")?;

        Ok(Self {
            x_r,
            x_w,
            x_k,
            x_v,
            x_a,
            x_g,
            w0,
            w1,
            w2,
            a0,
            a1,
            a2,
            v0,
            v1,
            v2,
            g1,
            g2,
            k_k,
            k_a,
            r_k,
            receptance_t,
            key_t,
            value_t,
            output_t,
            ln_x_weight,
            ln_x_bias,
            layer_id,
            n_heads,
            head_size,
        })
    }

    /// Forward pass for a single token (RNN mode).
    /// Input `x` shape: `[C]` (1D). Returns `(output [C], v_first [C])`.
    fn forward(
        &self,
        x: &Tensor,
        state: &mut StatePerLayer,
        v_first: Option<Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let h = self.n_heads;
        let n = self.head_size;

        // Helper: matrix multiply for 1D vec @ 2D weight: unsqueeze, matmul, squeeze
        macro_rules! mm {
            ($x:expr, $w:expr) => {
                $x.unsqueeze(0)?.matmul($w)?.squeeze(0)?
            };
        }

        // 1. Token shift: lerp between current and previous token
        // (x_r, x_w, etc. are pre-squeezed at load time)
        let xx = (&state.att_x_prev - x)?;
        let xr = (x + xx.broadcast_mul(&self.x_r)?)?;
        let xw = (x + xx.broadcast_mul(&self.x_w)?)?;
        let xk = (x + xx.broadcast_mul(&self.x_k)?)?;
        let xv = (x + xx.broadcast_mul(&self.x_v)?)?;
        let xa = (x + xx.broadcast_mul(&self.x_a)?)?;
        let xg = (x + xx.broadcast_mul(&self.x_g)?)?;
        state.att_x_prev = x.clone();

        // 2. Linear projections (weights pre-transposed at load time)
        let r = mm!(xr, &self.receptance_t);
        let k = mm!(xk, &self.key_t);
        let v = mm!(xv, &self.value_t);

        // 3. Decay: w = exp(-0.606531 * sigmoid(w0 + tanh(xw @ w1) @ w2))
        let w = mm!(mm!(xw, &self.w1).tanh()?, &self.w2);
        let w = (&self.w0 + &w)?.to_dtype(DType::F32)?;
        let w = (w.neg()?.exp()? + 1.0)?.recip()?; // sigmoid
        let w = (w * (-0.606531))?.exp()?;

        // 4. Value residual
        let (v, v_first) = if self.layer_id == 0 {
            // Layer 0: v_first = v (only one clone needed, v is moved)
            let v_first = v.clone();
            (v, v_first)
        } else {
            let v_first = v_first.unwrap();
            if let (Some(v0), Some(v1), Some(v2)) = (&self.v0, &self.v1, &self.v2) {
                let gate = candle_nn::ops::sigmoid(&(v0 + mm!(mm!(xv, v1), v2))?)?;
                let v = (&v + (&v_first - &v)?.broadcast_mul(&gate)?)?;
                (v, v_first)
            } else {
                (v, v_first)
            }
        };

        // 5. ICL rate: a = sigmoid(a0 + (xa @ a1) @ a2)
        let a = candle_nn::ops::sigmoid(&(&self.a0 + mm!(mm!(xa, &self.a1), &self.a2))?)?;

        // 6. Gate: g = sigmoid(xg @ g1) @ g2
        let g = mm!(candle_nn::ops::sigmoid(&mm!(xg, &self.g1))?, &self.g2);

        // 7. Key processing (k_k, k_a pre-squeezed)
        // kk = L2_normalize(k * k_k, per_head)
        let kk = (&k * &self.k_k)?;
        let kk = kk.reshape((h, n))?;
        let kk_norm = (kk.sqr()?.sum_keepdim(1)?.sqrt()? + 1e-12)?;
        let kk = kk.broadcast_div(&kk_norm)?;
        let kk = kk.reshape(h * n)?;

        // k = k * (1 + (a - 1) * k_a)
        let k = (&k * (1.0 + (&a - 1.0)?.broadcast_mul(&self.k_a)?)?)?;

        // 8. State update (delta-rule core)
        // vk = v.view(H,N,1) @ k.view(H,1,N)  — outer product
        let v_hn = v.reshape((h, n, 1))?;
        let k_hn = k.reshape((h, 1, n))?;
        let vk = v_hn.matmul(&k_hn)?;

        // ab = (-kk).view(H,N,1) @ (kk*a).view(H,1,N)  — ICL correction
        let kk_h = kk.reshape((h, n))?;
        let a_h = a.reshape((h, n))?;
        let neg_kk = kk_h.neg()?.reshape((h, n, 1))?;
        let kk_a = (&kk_h * &a_h)?.reshape((h, 1, n))?;
        let ab = neg_kk.matmul(&kk_a)?;

        // state = state * w.view(H,1,N) + state @ ab + vk
        let w_h = w.reshape((h, 1, n))?;
        let att_kv = &state.att_kv;
        let new_state = (att_kv.broadcast_mul(&w_h)?
            + att_kv
                .to_dtype(DType::F32)?
                .matmul(&ab.to_dtype(DType::F32)?)?
            + vk.to_dtype(DType::F32)?)?;
        state.att_kv = new_state;

        // out = state @ r.view(H,N,1)
        let r_hn = r.reshape((h, n, 1))?;
        let out = state.att_kv.to_dtype(r.dtype())?.matmul(&r_hn)?;

        // 9. GroupNorm (H groups, eps=64e-5)
        let out = {
            let reshaped = out.reshape((h, n))?;
            let mean = reshaped.mean_keepdim(1)?;
            let centered = reshaped.broadcast_sub(&mean)?;
            let var = centered.sqr()?.mean_keepdim(1)?;
            let normed = centered.broadcast_div(&(var + 64e-5)?.sqrt()?)?;
            normed.reshape(h * n)?
        };
        let out = (out.broadcast_mul(&self.ln_x_weight)? + &self.ln_x_bias)?;

        // 10. Bonus term: (r * k * r_k).sum_per_head * v (r_k pre-flattened)
        let bonus = (&r * &k * &self.r_k)?
            .reshape((h, n))?
            .sum_keepdim(1)?
            .broadcast_mul(&v.reshape((h, n))?)?
            .reshape(h * n)?;
        let out = (out + bonus)?;

        // 11. Output (weight pre-transposed)
        let out = mm!((out * g)?, &self.output_t);

        Ok((out, v_first))
    }
}

// ─── ChannelMix (FFN) ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct ChannelMix {
    x_k: Tensor,     // Pre-squeezed to 1D
    key_t: Tensor,   // Pre-transposed
    value_t: Tensor, // Pre-transposed
    // DeepEmbed (v7a, v7b only)
    deep_embed: Option<DeepEmbed>,
}

#[derive(Debug, Clone)]
struct DeepEmbed {
    s_emb: Tensor, // (vocab_size, 1024) — pre-merged with emb @ s_emb_x^T
    s0: Tensor,    // (dim_ffn,)
    s1: Tensor,    // (hidden_size, 32)
    s2: Tensor,    // (32, dim_ffn)
}

impl ChannelMix {
    fn new(_layer_id: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let c = cfg.hidden_size;
        let dim_ffn = cfg.dim_ffn();

        // Pre-squeeze and pre-transpose at load time
        let x_k = vb.get((1, 1, c), "x_k")?.squeeze(0)?.squeeze(0)?;
        let key_t = vb.get((dim_ffn, c), "key.weight")?.t()?.contiguous()?;
        let value_t = vb.get((c, dim_ffn), "value.weight")?.t()?.contiguous()?;

        let deep_embed = if cfg.version == ModelVersion::V7a || cfg.version == ModelVersion::V7b {
            // Load s_emb — the pre-merged embedding is computed in Model::new()
            let s_emb = vb.get((cfg.vocab_size, 1024), "s_emb.weight")?;
            // s0 stored as (1, 1, dim_ffn) in weights, squeeze to 1D for efficiency
            let s0 = vb.get((1, 1, dim_ffn), "s0")?.squeeze(0)?.squeeze(0)?;
            let s1 = vb.get((c, 32), "s1")?;
            let s2 = vb.get((32, dim_ffn), "s2")?;
            Some(DeepEmbed { s_emb, s0, s1, s2 })
        } else {
            None
        };

        Ok(Self {
            x_k,
            key_t,
            value_t,
            deep_embed,
        })
    }

    /// Forward pass for a single token. Input `x` shape: `[C]`.
    /// `token_ids` is needed for DeepEmbed (v7a/v7b).
    fn forward(
        &self,
        x: &Tensor,
        state: &mut StatePerLayer,
        token_ids: Option<&[u32]>,
    ) -> Result<Tensor> {
        macro_rules! mm {
            ($x:expr, $w:expr) => {
                $x.unsqueeze(0)?.matmul($w)?.squeeze(0)?
            };
        }

        // Token shift (x_k pre-squeezed)
        let xx = (&state.ffn_x_prev - x)?;
        let k = (x + xx.broadcast_mul(&self.x_k)?)?;
        state.ffn_x_prev = x.clone();

        // Squared ReLU: relu(key(k))^2 (key pre-transposed)
        let mut k = mm!(k, &self.key_t).relu()?.sqr()?;

        // DeepEmbed gating (v7a/v7b)
        if let Some(de) = &self.deep_embed {
            let token_ids = token_ids.expect("v7a/v7b requires token_ids in forward");
            let token_id = token_ids[0] as usize;
            // ss = (x @ s1) @ s_emb[token_id].view(32, 32)
            let semb = de.s_emb.i(token_id)?;
            let ss = mm!(x, &de.s1)
                .unsqueeze(0)?
                .matmul(&semb.reshape((32, 32))?)?
                .squeeze(0)?;
            // k = k * ((ss @ s2) + s0)
            let gate = (mm!(ss, &de.s2) + &de.s0)?;
            k = (k * gate)?;
        }

        // Down-projection (value pre-transposed)
        Ok(mm!(k, &self.value_t))
    }
}

// ─── DeaAttention (v7b only) ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct DeaAttention {
    qq_weight: Tensor,  // (hidden_size, 256)
    k1: Tensor,         // (hidden_size, 32)
    k2: Tensor,         // (32, 256)
    k_emb: Tensor,      // (vocab_size, 256) — pre-merged
    v1: Tensor,         // (hidden_size, 32)
    v2: Tensor,         // (32, hidden_size)
    v_emb: Tensor,      // (vocab_size, hidden_size) — pre-merged
    x_q: Tensor,        // (256,)
    x_k: Tensor,        // (256,)
    x_v: Tensor,        // (hidden_size,)
    lnq_weight: Tensor, // (256,)
    lnq_bias: Tensor,   // (256,)
    lnk_weight: Tensor, // (256,)
    lnk_bias: Tensor,   // (256,)
    lnv_weight: Tensor, // (hidden_size,)
    lnv_bias: Tensor,   // (hidden_size,)
    layer_id: usize,
    hidden_size: usize,
}

impl DeaAttention {
    fn new(layer_id: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let c = cfg.hidden_size;
        // qq.weight stored as (256, hidden_size) in PyTorch format, transpose for matmul
        let qq_weight = vb.get((256, c), "qq.weight")?.t()?.contiguous()?;
        let k1 = vb.get((c, 32), "k1")?;
        let k2 = vb.get((32, 256), "k2")?;
        let k_emb = vb.get((cfg.vocab_size, 256), "k_emb.weight")?;
        let v1 = vb.get((c, 32), "v1")?;
        let v2 = vb.get((32, c), "v2")?;
        let v_emb = vb.get((cfg.vocab_size, c), "v_emb.weight")?;
        // Token-shift params stored as (1, 1, dim), squeeze to 1D
        let x_q = vb.get((1, 1, 256), "x_q")?.squeeze(0)?.squeeze(0)?;
        let x_k = vb.get((1, 1, 256), "x_k")?.squeeze(0)?.squeeze(0)?;
        let x_v = vb.get((1, 1, c), "x_v")?.squeeze(0)?.squeeze(0)?;

        let lnq_weight = vb.get(256, "lnq.weight")?;
        let lnq_bias = vb.get(256, "lnq.bias")?;
        let lnk_weight = vb.get(256, "lnk.weight")?;
        let lnk_bias = vb.get(256, "lnk.bias")?;
        let lnv_weight = vb.get(c, "lnv.weight")?;
        let lnv_bias = vb.get(c, "lnv.bias")?;
        Ok(Self {
            qq_weight,
            k1,
            k2,
            k_emb,
            v1,
            v2,
            v_emb,
            x_q,
            x_k,
            x_v,
            lnq_weight,
            lnq_bias,
            lnk_weight,
            lnk_bias,
            lnv_weight,
            lnv_bias,
            layer_id,
            hidden_size: c,
        })
    }

    /// Forward pass for DEA attention. Updates the KV cache in `dea_state`.
    fn forward(&self, x: &Tensor, dea_state: &mut DeaState, token_ids: &[u32]) -> Result<Tensor> {
        let dev = x.device();

        // Helper for 1D vector @ 2D matrix multiplication
        macro_rules! mm {
            ($x:expr, $w:expr) => {
                $x.unsqueeze(0)?.matmul($w)?.squeeze(0)?
            };
        }

        // Q projection
        let q = mm!(x, &self.qq_weight);

        // K: project down, cache, project up, multiply by token embedding
        let k_proj = mm!(x, &self.k1); // (32,)
        let k_proj_2d = k_proj.reshape((1, 32))?;
        let old_k = &dea_state.k_cache[self.layer_id];
        dea_state.k_cache[self.layer_id] = if old_k.dim(0)? == 0 {
            k_proj_2d.clone()
        } else {
            Tensor::cat(&[old_k, &k_proj_2d], 0)?
        };
        let all_token_ids: Vec<u32> = dea_state
            .token_ids
            .iter()
            .copied()
            .chain(token_ids.iter().copied())
            .collect();
        let ctx_tensor = Tensor::new(&all_token_ids[..], dev)?;
        let k_full = dea_state.k_cache[self.layer_id].matmul(&self.k2)?;
        let k_emb_sel = self.k_emb.index_select(&ctx_tensor, 0)?;
        let k_full = (k_full * k_emb_sel)?;

        // V: project down, cache, project up (with tanh), multiply by token embedding
        let v_proj = mm!(x, &self.v1); // (32,)
        let v_proj_2d = v_proj.reshape((1, 32))?;
        let old_v = &dea_state.v_cache[self.layer_id];
        dea_state.v_cache[self.layer_id] = if old_v.dim(0)? == 0 {
            v_proj_2d.clone()
        } else {
            Tensor::cat(&[old_v, &v_proj_2d], 0)?
        };
        let v_full = dea_state.v_cache[self.layer_id].matmul(&self.v2)?.tanh()?;
        let v_emb_sel = self.v_emb.index_select(&ctx_tensor, 0)?;
        let v_full = (v_full * v_emb_sel)?;

        // Token shifting on Q (using previous Q state)
        // Important: save ORIGINAL q before shifting (reference line 160)
        let q_prev = &dea_state.q_prev[self.layer_id];
        let q_shifted = (&q + (q_prev - &q)?.broadcast_mul(&self.x_q)?)?;
        dea_state.q_prev[self.layer_id] = q.clone(); // Save original, not shifted!
        let q = q_shifted;

        // Token shifting on K and V (pad left by 1)
        // For seq_len=1: F.pad(k, (0,0,1,-1)) produces zeros, so k = k * (1 - x_k)
        // For seq_len>1: shifted = [zeros, k[:-1]], so k = k + (shifted - k) * x_k
        let seq_len = k_full.dim(0)?;

        let k_full = if seq_len > 1 {
            let k_shifted = Tensor::cat(
                &[
                    &Tensor::zeros((1, 256), k_full.dtype(), dev)?,
                    &k_full.i(..seq_len - 1)?,
                ],
                0,
            )?;
            (&k_full + (&k_shifted - &k_full)?.broadcast_mul(&self.x_k)?)?
        } else {
            // Single token: shifted is zeros, so k = k + (0 - k) * x_k = k * (1 - x_k)
            // Note: Candle doesn't support scalar - tensor directly, use neg + scalar
            let scale = (self.x_k.neg()? + 1.0)?;

            k_full.broadcast_mul(&scale)?
        };
        let v_full = if seq_len > 1 {
            let v_shifted = Tensor::cat(
                &[
                    &Tensor::zeros((1, self.hidden_size), v_full.dtype(), dev)?,
                    &v_full.i(..seq_len - 1)?,
                ],
                0,
            )?;
            (&v_full + (&v_shifted - &v_full)?.broadcast_mul(&self.x_v)?)?
        } else {
            // Single token: v = v * (1 - x_v)
            let scale = (1.0 - &self.x_v)?;
            v_full.broadcast_mul(&scale)?
        };

        // LayerNorm on Q, K, V
        let q = layer_norm(&q.unsqueeze(0)?, &self.lnq_weight, &self.lnq_bias, 1e-5)?.squeeze(0)?;
        let k_full = layer_norm(&k_full, &self.lnk_weight, &self.lnk_bias, 1e-5)?;
        let v_full = layer_norm(&v_full, &self.lnv_weight, &self.lnv_bias, 1e-5)?;

        // Soft-capped causal attention: 64 * tanh(q @ k^T / 1024)
        let scores = q.unsqueeze(0)?.matmul(&k_full.t()?)?;
        let scores = ((scores * (1.0 / 1024.0))?.tanh()? * 64.0)?;

        // Attention output
        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = attn_weights.matmul(&v_full)?.squeeze(0)?;

        Ok(out)
    }
}

// ─── Block ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Block {
    ln0_weight: Option<Tensor>,
    ln0_bias: Option<Tensor>,
    ln1_weight: Tensor,
    ln1_bias: Tensor,
    ln2_weight: Tensor,
    ln2_bias: Tensor,
    att: TimeMix,
    ffn: ChannelMix,
    dea: Option<DeaAttention>,
    layer_id: usize,
}

impl Block {
    fn new(
        layer_id: usize,
        cfg: &Config,
        lora: (usize, usize, usize, usize),
        vb: VarBuilder,
    ) -> Result<Self> {
        let c = cfg.hidden_size;

        let (ln0_weight, ln0_bias) = if layer_id == 0 {
            (Some(vb.get(c, "ln0.weight")?), Some(vb.get(c, "ln0.bias")?))
        } else {
            (None, None)
        };

        let ln1_weight = vb.get(c, "ln1.weight")?;
        let ln1_bias = vb.get(c, "ln1.bias")?;
        let ln2_weight = vb.get(c, "ln2.weight")?;
        let ln2_bias = vb.get(c, "ln2.bias")?;

        let att = TimeMix::new(layer_id, cfg, lora, vb.pp("att"))?;
        let ffn = ChannelMix::new(layer_id, cfg, vb.pp("ffn"))?;

        let dea = if cfg.version == ModelVersion::V7b {
            Some(DeaAttention::new(layer_id, cfg, vb.pp("qkv"))?)
        } else {
            None
        };

        Ok(Self {
            ln0_weight,
            ln0_bias,
            ln1_weight,
            ln1_bias,
            ln2_weight,
            ln2_bias,
            att,
            ffn,
            dea,
            layer_id,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        state: &mut State,
        v_first: Option<Tensor>,
        token_ids: Option<&[u32]>,
    ) -> Result<(Tensor, Tensor)> {
        // Pre-norm (block 0 only) - store owned tensor if ln0 applied
        let x_owned: Option<Tensor> = if let (Some(w), Some(b)) = (&self.ln0_weight, &self.ln0_bias)
        {
            Some(layer_norm(x, w, b, 1e-5)?)
        } else {
            None
        };
        let x_ref: &Tensor = x_owned.as_ref().unwrap_or(x);

        // DEA attention (v7b only) — computed on x BEFORE ln1
        let dea_out = if let Some(dea) = &self.dea {
            let dea_state = state.dea.as_mut().expect("v7b requires DeaState");
            Some(dea.forward(x_ref, dea_state, token_ids.unwrap())?)
        } else {
            None
        };

        // Time mixing (RWKV linear attention)
        let x_ln1 = layer_norm(x_ref, &self.ln1_weight, &self.ln1_bias, 1e-5)?;
        let (att_out, v_first) =
            self.att
                .forward(&x_ln1, &mut state.per_layer[self.layer_id], v_first)?;

        // Residual: x + att_out + dea_out (clone only when needed for addition)
        let x = if let Some(dea_out) = dea_out {
            (x_ref + &att_out + dea_out)?
        } else {
            (x_ref + att_out)?
        };

        // Channel mixing (FFN)
        let x_ln2 = layer_norm(&x, &self.ln2_weight, &self.ln2_bias, 1e-5)?;
        let ffn_out = self
            .ffn
            .forward(&x_ln2, &mut state.per_layer[self.layer_id], token_ids)?;
        let x = (x + ffn_out)?;

        Ok((x, v_first))
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Model {
    embeddings: Embedding,
    blocks: Vec<Block>,
    ln_out_weight: Tensor,
    ln_out_bias: Tensor,
    head_t: Tensor, // Pre-transposed for efficiency
    pub version: ModelVersion,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let c = cfg.hidden_size;
        let lora = infer_lora_dims(&vb)?;

        let embeddings = embedding(cfg.vocab_size, c, vb.pp("emb"))?;

        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_b = vb.pp("blocks");
        for layer_id in 0..cfg.num_hidden_layers {
            blocks.push(Block::new(layer_id, cfg, lora, vb_b.pp(layer_id))?);
        }

        let ln_out_weight = vb.get(c, "ln_out.weight")?;
        let ln_out_bias = vb.get(c, "ln_out.bias")?;
        // Pre-transpose head weight at load time
        let head_t = vb
            .get((cfg.vocab_size, c), "head.weight")?
            .t()?
            .contiguous()?;

        let mut model = Self {
            embeddings,
            blocks,
            ln_out_weight,
            ln_out_bias,
            head_t,
            version: cfg.version,
        };

        // Load-time merges for DeepEmbed (v7a/v7b) and DEA (v7b)
        // IMPORTANT: Reference pre-normalizes emb.weight with ln0 BEFORE merging!
        // See rwkv_v7b_demo.py line 103:
        //   z['emb.weight'] = F.layer_norm(z['emb.weight'], ..., weight=z['blocks.0.ln0.weight'], ...)
        if cfg.version == ModelVersion::V7a || cfg.version == ModelVersion::V7b {
            // Get ln0 weights from block 0 to normalize embeddings
            let ln0_weight = &model.blocks[0]
                .ln0_weight
                .as_ref()
                .expect("v7a/v7b requires ln0");
            let ln0_bias = &model.blocks[0]
                .ln0_bias
                .as_ref()
                .expect("v7a/v7b requires ln0");

            // Normalize embeddings with ln0 (applied to each row independently)
            let emb_raw = model.embeddings.embeddings();
            let emb_normalized = layer_norm(emb_raw, ln0_weight, ln0_bias, 1e-5)?;

            // DeepEmbed merges (FFN s_emb)
            for i in 0..cfg.num_hidden_layers {
                if let Some(de) = &mut model.blocks[i].ffn.deep_embed {
                    // s_emb += normalized_emb @ s_emb_x^T
                    let s_emb_x = vb_b.pp(i).pp("ffn").get((1024, c), "s_emb_x.weight")?;
                    de.s_emb = (&de.s_emb + emb_normalized.matmul(&s_emb_x.t()?)?)?;
                }
            }

            // DEA merges (v7b only)
            if cfg.version == ModelVersion::V7b {
                for i in 0..cfg.num_hidden_layers {
                    if let Some(dea) = &mut model.blocks[i].dea {
                        let k_emb_x = vb_b.pp(i).pp("qkv").get((256, c), "k_emb_x.weight")?;
                        dea.k_emb = (&dea.k_emb + emb_normalized.matmul(&k_emb_x.t()?)?)?;

                        let v_emb_x = vb_b.pp(i).pp("qkv").get((c, c), "v_emb_x.weight")?;
                        dea.v_emb = (&dea.v_emb + emb_normalized.matmul(&v_emb_x.t()?)?)?;
                    }
                }
            }
        }

        Ok(model)
    }

    /// Run a forward pass for a single token (RNN-style inference).
    ///
    /// `token_ids` should contain the token ID(s) being processed.
    /// For v7a/v7b, these are used for DeepEmbed and DEA token-embedding lookups.
    pub fn forward(&self, xs: &Tensor, state: &mut State, token_ids: &[u32]) -> Result<Tensor> {
        let mut xs = xs.apply(&self.embeddings)?;
        // xs shape: (1, 1, hidden_size) for single token; squeeze to (hidden_size,)
        xs = xs.squeeze(0)?.squeeze(0)?;

        let token_ids_opt = if self.version == ModelVersion::V7 {
            None
        } else {
            Some(token_ids)
        };

        let mut v_first: Option<Tensor> = None;
        for block in &self.blocks {
            let (new_xs, new_v_first) = block.forward(&xs, state, v_first, token_ids_opt)?;
            xs = new_xs;
            v_first = Some(new_v_first);
        }

        // Update DEA token ID cache after all blocks processed
        if let Some(dea_state) = &mut state.dea {
            dea_state.token_ids.extend_from_slice(token_ids);
        }

        let xs = layer_norm(&xs, &self.ln_out_weight, &self.ln_out_bias, 1e-5)?;
        // head_t is pre-transposed, no .t() needed
        let xs = xs.unsqueeze(0)?.matmul(&self.head_t)?.squeeze(0)?;
        state.pos += 1;
        Ok(xs)
    }

    /// Process a sequence of tokens efficiently (batch prompt processing).
    ///
    /// This is significantly faster than calling `forward` token-by-token because:
    /// - Embeddings are computed in one batch
    /// - Linear projections are batched where possible
    ///
    /// Returns the logits for the last token only (for next-token prediction).
    pub fn forward_seq(&self, token_ids: &[u32], state: &mut State) -> Result<Tensor> {
        if token_ids.is_empty() {
            candle::bail!("token_ids cannot be empty");
        }

        // For short sequences, fall back to single-token processing
        if token_ids.len() == 1 {
            let dev = state.per_layer[0].att_x_prev.device();
            let input = Tensor::new(&[token_ids[0]], dev)?.unsqueeze(0)?;
            return self.forward(&input, state, token_ids);
        }

        let dev = state.per_layer[0].att_x_prev.device();

        // Batch embed all tokens at once: (seq_len,) -> (seq_len, hidden_size)
        let input_ids = Tensor::new(token_ids, dev)?;
        let xs = input_ids.apply(&self.embeddings)?;

        // Process each token through all layers
        // Note: RWKV state updates are sequential, but we batch the embedding lookup
        let seq_len = token_ids.len();
        let mut last_logits = None;

        for t in 0..seq_len {
            // Extract single token embedding: (hidden_size,)
            let x = xs.i(t)?;

            let token_ids_opt = if self.version == ModelVersion::V7 {
                None
            } else {
                Some(&token_ids[t..t + 1])
            };

            let mut x_out = x;
            let mut v_first: Option<Tensor> = None;

            for block in &self.blocks {
                let (new_x, new_v_first) = block.forward(&x_out, state, v_first, token_ids_opt)?;
                x_out = new_x;
                v_first = Some(new_v_first);
            }

            // Update DEA token ID cache
            if let Some(dea_state) = &mut state.dea {
                dea_state.token_ids.push(token_ids[t]);
            }

            state.pos += 1;

            // Only compute logits for the last token
            if t == seq_len - 1 {
                let x_norm = layer_norm(&x_out, &self.ln_out_weight, &self.ln_out_bias, 1e-5)?;
                last_logits = Some(x_norm.unsqueeze(0)?.matmul(&self.head_t)?.squeeze(0)?);
            }
        }

        last_logits.ok_or_else(|| candle::Error::Msg("No tokens processed".to_string()))
    }
}
