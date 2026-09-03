//! CPU flash attention implementations.
//!
//! - `standard`: General-purpose with explicit mask tensor, B=1 only
//! - `causal`: Loop-bound causal masking, B=1 only
//! - `varlen`: Packed variable-length sequences (total_q, H, D), any batch size
//!
//! The top-level [`flash_attn`] function automatically dispatches:
//! - **B=1**: single-batch kernels in `standard`/`causal` (direct slice access, zero batch overhead)
//! - **B>1** (f32/f16, no softcap): packs into varlen format
//! - **B>1 unsupported config**: hard error (explicit mask + B>1, softcap + B>1, etc.)

pub mod causal;
pub(crate) mod online_softmax;
pub mod standard;
pub mod varlen;

use candle::{DType, Result, Tensor, WithDType};

use super::AttnMask;

/// Dot product of two equal-length `T` rows, returned as f32.
///
/// Q and K stream in their native dtype (no per-row dequantization).
///
/// For f32 this is thin glue over candle's architecture-tuned `VecOps::vec_dot`
/// intrinsic (NEON / AVX2 / SIMD128). For f16/bf16 we accumulate in f32 by hand:
/// `VecOps::vec_dot` narrows its internal f32 accumulator back to the half type
/// before returning, which would round the attention score (and can overflow it
/// to `inf`) *before* softmax. Keeping the score in f32 preserves its full range
/// and precision, matching the pre-standardization half kernels.
#[inline]
pub(crate) fn dot_f32<T: WithDType>(a: &[T], b: &[T]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    if matches!(T::DTYPE, DType::F16) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { dot_f16_avx2_fma(a.as_ptr() as *const u16, b.as_ptr() as *const u16, a.len()) };
            }
        }
    }
    if matches!(T::DTYPE, DType::BF16) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { dot_bf16_avx2_fma(a.as_ptr() as *const u16, b.as_ptr() as *const u16, a.len()) };
            }
        }
    }
    if matches!(T::DTYPE, DType::F16 | DType::BF16) {
        let mut acc = 0f32;
        for (&x, &y) in a.iter().zip(b.iter()) {
            acc += (x.to_f64() as f32) * (y.to_f64() as f32);
        }
        return acc;
    }
    let mut res = T::zero();
    // SAFETY: `a` and `b` are both at least `a.len()` long and `res` is a valid
    // out pointer, pre-zeroed for the scalar fallback that accumulates into it.
    unsafe { T::vec_dot(a.as_ptr(), b.as_ptr(), &mut res, a.len()) };
    res.to_f64() as f32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_f16_avx2_fma(a: *const u16, b: *const u16, len: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut i = 0;

    // 16-way unrolled AVX2 + F16C hardware decompression
    while i + 16 <= len {
        let va0 = _mm_loadu_si128(a.add(i) as *const __m128i);
        let vb0 = _mm_loadu_si128(b.add(i) as *const __m128i);
        let fa0 = _mm256_cvtph_ps(va0);
        let fb0 = _mm256_cvtph_ps(vb0);
        sum0 = _mm256_fmadd_ps(fa0, fb0, sum0);

        let va1 = _mm_loadu_si128(a.add(i + 8) as *const __m128i);
        let vb1 = _mm_loadu_si128(b.add(i + 8) as *const __m128i);
        let fa1 = _mm256_cvtph_ps(va1);
        let fb1 = _mm256_cvtph_ps(vb1);
        sum1 = _mm256_fmadd_ps(fa1, fb1, sum1);

        i += 16;
    }

    if i + 8 <= len {
        let va = _mm_loadu_si128(a.add(i) as *const __m128i);
        let vb = _mm_loadu_si128(b.add(i) as *const __m128i);
        let fa = _mm256_cvtph_ps(va);
        let fb = _mm256_cvtph_ps(vb);
        sum0 = _mm256_fmadd_ps(fa, fb, sum0);
        i += 8;
    }

    sum0 = _mm256_add_ps(sum0, sum1);

    // Horizontal reduction
    let hi128 = _mm256_extractf128_ps(sum0, 1);
    let lo128 = _mm256_castps256_ps128(sum0);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehl_ps(sum128, sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehdup_ps(sums);
    let sums2 = _mm_add_ss(sums, shuf2);
    let mut acc = _mm_cvtss_f32(sums2);

    // Residual elements
    while i < len {
        let ha = half::f16::from_bits(*a.add(i));
        let hb = half::f16::from_bits(*b.add(i));
        acc += (ha.to_f64() as f32) * (hb.to_f64() as f32);
        i += 1;
    }

    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_bf16_avx2_fma(a: *const u16, b: *const u16, len: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut i = 0;

    // 16-way unrolled AVX2 FMA for BF16 via exponent shift
    while i + 16 <= len {
        let va0_raw = _mm_loadu_si128(a.add(i) as *const __m128i);
        let vb0_raw = _mm_loadu_si128(b.add(i) as *const __m128i);
        let va0_32 = _mm256_cvtepu16_epi32(va0_raw);
        let vb0_32 = _mm256_cvtepu16_epi32(vb0_raw);
        let fa0 = _mm256_castsi256_ps(_mm256_slli_epi32(va0_32, 16));
        let fb0 = _mm256_castsi256_ps(_mm256_slli_epi32(vb0_32, 16));
        sum0 = _mm256_fmadd_ps(fa0, fb0, sum0);

        let va1_raw = _mm_loadu_si128(a.add(i + 8) as *const __m128i);
        let vb1_raw = _mm_loadu_si128(b.add(i + 8) as *const __m128i);
        let va1_32 = _mm256_cvtepu16_epi32(va1_raw);
        let vb1_32 = _mm256_cvtepu16_epi32(vb1_raw);
        let fa1 = _mm256_castsi256_ps(_mm256_slli_epi32(va1_32, 16));
        let fb1 = _mm256_castsi256_ps(_mm256_slli_epi32(vb1_32, 16));
        sum1 = _mm256_fmadd_ps(fa1, fb1, sum1);

        i += 16;
    }

    if i + 8 <= len {
        let va_raw = _mm_loadu_si128(a.add(i) as *const __m128i);
        let vb_raw = _mm_loadu_si128(b.add(i) as *const __m128i);
        let va_32 = _mm256_cvtepu16_epi32(va_raw);
        let vb_32 = _mm256_cvtepu16_epi32(vb_raw);
        let fa = _mm256_castsi256_ps(_mm256_slli_epi32(va_32, 16));
        let fb = _mm256_castsi256_ps(_mm256_slli_epi32(vb_32, 16));
        sum0 = _mm256_fmadd_ps(fa, fb, sum0);
        i += 8;
    }

    sum0 = _mm256_add_ps(sum0, sum1);

    // Horizontal reduction
    let hi128 = _mm256_extractf128_ps(sum0, 1);
    let lo128 = _mm256_castps256_ps128(sum0);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehl_ps(sum128, sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehdup_ps(sums);
    let sums2 = _mm_add_ss(sums, shuf2);
    let mut acc = _mm_cvtss_f32(sums2);

    // Residual elements
    while i < len {
        let ha = half::bf16::from_bits(*a.add(i));
        let hb = half::bf16::from_bits(*b.add(i));
        acc += (ha.to_f64() as f32) * (hb.to_f64() as f32);
        i += 1;
    }

    acc
}

/// Flash attention with automatic dispatch.
///
/// Selects optimal implementation based on batch size, mask type, and dtype:
/// - **B=1**: uses single-batch optimized kernels (direct slice access, no batch overhead)
/// - **B>1 + Causal/None + f32/f16**: packs to varlen format (avoids batch-dim stride overhead)
/// - **Explicit mask or unsupported dtype**: falls back to general-purpose batched kernel
///
/// # Arguments
/// * `q` - Query tensor, shape `(B, S, H, D)`
/// * `k` - Key tensor, shape `(B, KV_S, KV_H, D)`
/// * `v` - Value tensor, shape `(B, KV_S, KV_H, D)`
/// * `softmax_scale` - Scale factor (typically `1/sqrt(head_dim)`)
/// * `attn_mask` - Masking strategy
/// * `max_bias` - ALiBi max bias (`None` to disable)
/// * `softcap` - Logit soft-capping (`None` to disable)
///
/// # Returns
/// Output tensor with shape `(B, H, S, D)`
pub fn flash_attn<T>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    attn_mask: AttnMask,
    max_bias: Option<f32>,
    softcap: Option<f32>,
) -> Result<Tensor>
where
    T: WithDType + num_traits::Float,
{
    let b = q.dims()[0];

    if b > 1 {
        let dt = q.dtype();
        let varlen_ok = (dt == DType::F32 || dt == DType::F16) && softcap.is_none();
        let mask_ok = matches!(&attn_mask, AttnMask::Causal { .. } | AttnMask::None);

        if !varlen_ok || !mask_ok {
            candle::bail!(
                "CPU flash attention with B>1 requires: f32/f16 dtype, no softcap, \
                 and Causal or None mask. Got B={b}, dtype={dt:?}, softcap={softcap:?}, \
                 mask={}",
                match &attn_mask {
                    AttnMask::Causal { .. } => "Causal",
                    AttnMask::None => "None",
                    AttnMask::Mask(_) => "Mask(tensor)",
                }
            );
        }

        return flash_attn_via_varlen(q, k, v, softmax_scale, &attn_mask, max_bias);
    }

    // B=1: dedicated single-batch kernels (no batch indexing, direct slices)
    match attn_mask {
        AttnMask::Causal { kv_offset } => {
            causal::run_causal_attn_cpu::<T>(q, k, v, softmax_scale, kv_offset, max_bias, softcap)
        }
        AttnMask::None => {
            standard::run_flash_attn_cpu::<T>(q, k, v, None, softmax_scale, max_bias, softcap)
        }
        AttnMask::Mask(mask) => standard::run_flash_attn_cpu::<T>(
            q,
            k,
            v,
            Some(&mask),
            softmax_scale,
            max_bias,
            softcap,
        ),
    }
}

/// Reshape batched (B,S,H,D) tensors into packed varlen format and dispatch.
///
/// Returns output in (B, H, S, D) to match the standard `flash_attn` contract.
fn flash_attn_via_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    attn_mask: &AttnMask,
    max_bias: Option<f32>,
) -> Result<Tensor> {
    let q_dims = q.dims();
    let k_dims = k.dims();
    let (b, s_q, h_q, d) = (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
    let (s_kv, h_kv) = (k_dims[1], k_dims[2]);

    let causal = attn_mask.is_causal();

    let q_packed = q.contiguous()?.reshape((b * s_q, h_q, d))?;
    let k_packed = k.contiguous()?.reshape((b * s_kv, h_kv, d))?;
    let v_packed = v.contiguous()?.reshape((b * s_kv, h_kv, d))?;

    // Build uniform seqlens
    let device = q.device();
    let seqlens_q = Tensor::from_vec(vec![s_q as u32; b], b, device)?;
    let seqlens_k = Tensor::from_vec(vec![s_kv as u32; b], b, device)?;

    // ALiBi: convert max_bias to per-head slopes tensor
    let alibi_slopes = if let Some(mb) = max_bias {
        if mb > 0.0 {
            let n2 = 2_usize.pow((h_q as f32).log2().ceil() as u32);
            let slopes: Vec<f32> = (0..h_q)
                .map(|h| 2.0f32.powf(-mb * ((h + 1) as f32) / n2 as f32))
                .collect();
            Some(Tensor::from_vec(slopes, h_q, device)?)
        } else {
            None
        }
    } else {
        None
    };

    let ctx = varlen::flash_attn_varlen_cpu(
        &q_packed,
        &k_packed,
        &v_packed,
        alibi_slopes.as_ref(),
        &seqlens_q,
        &seqlens_k,
        s_q,
        s_kv,
        softmax_scale,
        causal,
        None,
        None,
    )?;

    ctx.reshape((b, s_q, h_q, d))?.transpose(1, 2)?.contiguous()
}
