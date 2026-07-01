pub mod cpu_flash;
pub mod varlen;

use candle::{DType, Result, Tensor};
use std::sync::LazyLock;

pub use cpu_flash::flash_attn;
pub use cpu_flash::varlen::flash_attn_varlen_cpu;
pub use varlen::flash_attn_varlen_unfused;

static FLASH_ATTN_DISABLED: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("CANDLE_DISABLE_FLASH_ATTN")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
});

/// Returns true if flash attention is disabled via the `CANDLE_DISABLE_FLASH_ATTN` env var.
///
/// Set `CANDLE_DISABLE_FLASH_ATTN=1` to force fallback to standard matmul attention
/// for debugging purposes.
pub fn is_flash_attn_disabled() -> bool {
    *FLASH_ATTN_DISABLED
}

#[derive(Debug, Clone, Default)]
pub enum AttnMask {
    #[default]
    None,
    Causal {
        kv_offset: usize,
    },
    Mask(Tensor),
}

impl AttnMask {
    #[inline]
    pub fn causal() -> Self {
        AttnMask::Causal { kv_offset: 0 }
    }

    #[inline]
    pub fn causal_with_offset(kv_offset: usize) -> Self {
        AttnMask::Causal { kv_offset }
    }

    #[inline]
    pub fn is_causal(&self) -> bool {
        matches!(self, AttnMask::Causal { .. })
    }

    #[inline]
    pub fn kv_offset(&self) -> usize {
        match self {
            AttnMask::Causal { kv_offset } => *kv_offset,
            _ => 0,
        }
    }
}

/// Unified attention dispatch that automatically selects the best available path.
///
/// Selection order:
/// 1. If `CANDLE_DISABLE_FLASH_ATTN=1` → standard matmul attention
/// 2. If `flash-attn` feature enabled and tensor is on GPU → CUDA flash attention
/// 3. On CPU with supported dtype → CPU flash attention kernel
/// 4. Otherwise → standard matmul attention
///
/// # Arguments
/// * `q` - Query tensor, shape `(B, H, S, D)` (heads-first, GQA already expanded)
/// * `k` - Key tensor, shape `(B, H, KV_S, D)` (heads already repeated for GQA)
/// * `v` - Value tensor, shape `(B, H, KV_S, D)`
/// * `softmax_scale` - Scale factor (typically `1/sqrt(head_dim)`)
/// * `attn_mask` - Optional mask tensor, shape broadcastable to `(B, H, S, KV_S)`
///
/// # Returns
/// Output tensor with shape `(B, H, S, D)`
pub fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    attn_mask: Option<&Tensor>,
) -> Result<Tensor> {
    if !is_flash_attn_disabled() {
        #[cfg(feature = "flash-attn")]
        if !q.device().is_cpu() {
            let q_t = q.transpose(1, 2)?.contiguous()?;
            let k_t = k.transpose(1, 2)?.contiguous()?;
            let v_t = v.transpose(1, 2)?.contiguous()?;
            let (_, _, seq_len, _) = q.dims4()?;
            let causal = seq_len > 1 || attn_mask.is_some();
            let out = candle_flash_attn::flash_attn(&q_t, &k_t, &v_t, softmax_scale, causal)?;
            return out.transpose(1, 2);
        }

        if q.device().is_cpu() {
            if let Ok(out) = cpu_flash_dispatch(q, k, v, softmax_scale, attn_mask) {
                return Ok(out);
            }
        }
    }

    standard_attn(q, k, v, softmax_scale, attn_mask)
}

/// Attempts CPU flash attention dispatch.
///
/// Transposes from heads-first `(B, H, S, D)` to `(B, S, H, D)` for the flash kernel,
/// then transposes the output back.
fn cpu_flash_dispatch(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    attn_mask: Option<&Tensor>,
) -> Result<Tensor> {
    let q_t = q.transpose(1, 2)?.contiguous()?;
    let k_t = k.transpose(1, 2)?.contiguous()?;
    let v_t = v.transpose(1, 2)?.contiguous()?;

    let mask = if attn_mask.is_some() {
        AttnMask::causal()
    } else {
        AttnMask::None
    };

    let out = match q.dtype() {
        DType::F32 => flash_attn::<f32>(&q_t, &k_t, &v_t, softmax_scale, mask, None, None)?,
        DType::F64 => flash_attn::<f64>(&q_t, &k_t, &v_t, softmax_scale, mask, None, None)?,
        DType::BF16 => {
            let q_f32 = q_t.to_dtype(DType::F32)?;
            let k_f32 = k_t.to_dtype(DType::F32)?;
            let v_f32 = v_t.to_dtype(DType::F32)?;
            let out = flash_attn::<f32>(&q_f32, &k_f32, &v_f32, softmax_scale, mask, None, None)?;
            out.to_dtype(DType::BF16)?
        }
        DType::F16 => {
            let q_f32 = q_t.to_dtype(DType::F32)?;
            let k_f32 = k_t.to_dtype(DType::F32)?;
            let v_f32 = v_t.to_dtype(DType::F32)?;
            let out = flash_attn::<f32>(&q_f32, &k_f32, &v_f32, softmax_scale, mask, None, None)?;
            out.to_dtype(DType::F16)?
        }
        _ => candle::bail!("unsupported dtype for CPU flash attention: {:?}", q.dtype()),
    };

    // Output from CPU flash is (B, H, S, D) already based on the kernel contract
    Ok(out)
}

/// Standard matmul-based attention: Q*K^T * scale [+ mask] → softmax → *V
///
/// Input/output shape: `(B, H, S, D)` (heads-first layout).
pub fn standard_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    attn_mask: Option<&Tensor>,
) -> Result<Tensor> {
    let mut scores = (q.matmul(&k.transpose(2, 3)?)? * (softmax_scale as f64))?;
    if let Some(m) = attn_mask {
        let m_dtype = m.dtype();
        let s_dtype = scores.dtype();
        if m_dtype != s_dtype {
            scores = scores.broadcast_add(&m.to_dtype(s_dtype)?)?;
        } else {
            scores = scores.broadcast_add(m)?;
        }
    }
    let probs = crate::ops::softmax_last_dim(&scores)?;
    probs.matmul(v)
}
