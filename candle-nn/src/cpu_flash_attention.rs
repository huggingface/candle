#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle::{Device, Result, Storage, Tensor, WithDType};
use std::sync::LazyLock;
use std::{f32, iter::Sum};

use rayon::prelude::*;
use rayon::ThreadPool;

#[cfg(target_os = "macos")]
/// Elevate the thread QoS so macOS prefers running it on Performance (P) cores.
unsafe fn set_thread_affinity() {
    use libc::{pthread_set_qos_class_self_np, qos_class_t::QOS_CLASS_USER_INTERACTIVE};
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
}

#[cfg(not(target_os = "macos"))]
#[inline(always)]
unsafe fn set_thread_affinity() {
    // On non‑macOS platforms we currently leave affinity untouched.
}

/// Rayon pool used by the flash‑attention CPU kernels, with a per‑thread
/// start handler that applies our affinity hint exactly once.
static FLASH_ATTN_POOL: LazyLock<ThreadPool> = LazyLock::new(|| {
    rayon::ThreadPoolBuilder::new()
        .start_handler(|_| unsafe {
            set_thread_affinity();
        })
        .build()
        .expect("Failed to build custom Rayon thread‑pool for flash‑attention")
});

const DOT_CHUNK: usize = 4;

/// Size (in KV positions) processed by each inner‑tile job.
const TILE_KV: usize = 16;

/// Attention mask specification for CPU flash attention.
///
/// Determines how masking is applied during attention computation.
#[derive(Debug, Clone, Copy)]
pub enum AttnMask<'a> {
    /// No masking - full bidirectional attention.
    /// All query positions can attend to all key/value positions.
    None,

    /// Causal masking via efficient loop bounds.
    /// Each query position can only attend to positions at or before it.
    /// This is more efficient than an explicit mask tensor as it avoids
    /// iterating over masked positions entirely.
    ///
    /// `kv_offset`: Number of prior KV positions when using KV cache (for decode).
    Causal { kv_offset: usize },

    /// Custom mask tensor for arbitrary attention patterns.
    /// Supports sliding window, block-sparse, or any custom masking.
    ///
    /// Expected shape: `(B, Q_LEN, KV_LEN)` or broadcastable.
    /// Values should be 0.0 for positions to attend, `NEG_INFINITY` for masked.
    Mask(&'a Tensor),
}

impl<'a> AttnMask<'a> {
    /// Create a causal mask with no KV offset (for prefill).
    pub fn causal() -> Self {
        AttnMask::Causal { kv_offset: 0 }
    }

    /// Create a causal mask with the specified KV offset (for decode with KV cache).
    pub fn causal_with_offset(kv_offset: usize) -> Self {
        AttnMask::Causal { kv_offset }
    }
}

#[inline]
fn vec_dot<T: WithDType + Sum + Copy + std::ops::Mul<Output = T>>(a: &[T], b: &[T]) -> T {
    let mut sum = T::zero();
    let chunks = a.len() / DOT_CHUNK;

    for i in 0..chunks {
        let i_chunk = i * DOT_CHUNK;
        sum = sum
            + a[i_chunk] * b[i_chunk]
            + a[i_chunk + 1] * b[i_chunk + 1]
            + a[i_chunk + 2] * b[i_chunk + 2]
            + a[i_chunk + 3] * b[i_chunk + 3];
    }

    for i in (chunks * DOT_CHUNK)..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Fused attention optimized for CPU.
///
/// Computes `softmax(Q @ K^T * scale) @ V`
///
/// **Input shapes:**
/// - `q`: (B, S, H, D)
/// - `k`: (B, KV_S, KV_H, D)
/// - `v`: (B, KV_S, KV_H, D)
///
/// **Output shape:** (B, H, S, D)
///
/// **Parameters:**
/// - `softmax_scale`: Scale factor applied before softmax (typically `1/sqrt(head_dim)`)
/// - `attn_mask`: Masking strategy - `None`, `Causal`, or custom `Mask` tensor
/// - `max_bias`: ALiBi max bias (None or Some(0.0) to disable)
/// - `softcap`: Logit soft-capping value (None or Some(0.0) to disable)
///
/// # Examples
///
/// ```ignore
/// // Causal attention (optimized loop bounds)
/// run_flash_attn_cpu::<f32>(&q, &k, &v, scale, AttnMask::Causal { kv_offset: 0 }, None, None)?;
///
/// // With KV cache offset for decode
/// run_flash_attn_cpu::<f32>(&q, &k, &v, scale, AttnMask::causal_with_offset(512), None, None)?;
///
/// // Custom mask tensor
/// run_flash_attn_cpu::<f32>(&q, &k, &v, scale, AttnMask::Mask(&mask), None, None)?;
///
/// // Full attention (no masking)
/// run_flash_attn_cpu::<f32>(&q, &k, &v, scale, AttnMask::None, None, None)?;
/// ```
pub fn run_flash_attn_cpu<T>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    attn_mask: AttnMask<'_>,
    max_bias: Option<f32>,
    softcap: Option<f32>,
) -> Result<Tensor>
where
    T: WithDType + Sum + num_traits::real::Real,
{
    // Extract CPU slices for q, k, v
    let (q_guard, q_layout) = q.storage_and_layout();
    let q_data: &[T] = if let Storage::Cpu(cpu) = &*q_guard {
        let data = cpu.as_slice::<T>()?;
        &data[q_layout.start_offset()..]
    } else {
        return Err(candle::Error::Msg("Expected CPU storage for q".into()));
    };

    let (k_guard, k_layout) = k.storage_and_layout();
    let k_data: &[T] = if let Storage::Cpu(cpu) = &*k_guard {
        let data = cpu.as_slice::<T>()?;
        &data[k_layout.start_offset()..]
    } else {
        return Err(candle::Error::Msg("Expected CPU storage for k".into()));
    };

    let (v_guard, v_layout) = v.storage_and_layout();
    let v_data: &[T] = if let Storage::Cpu(cpu) = &*v_guard {
        let data = cpu.as_slice::<T>()?;
        &data[v_layout.start_offset()..]
    } else {
        return Err(candle::Error::Msg("Expected CPU storage for v".into()));
    };

    // Extract mask data if provided
    let _mask_guard;
    let mask_data: Option<&[T]> = match &attn_mask {
        AttnMask::Mask(mask) => {
            _mask_guard = Some(mask.storage_and_layout());
            if let Some((ref guard, ref layout)) = _mask_guard {
                if let Storage::Cpu(cpu) = &**guard {
                    let data = cpu.as_slice::<T>()?;
                    Some(&data[layout.start_offset()..])
                } else {
                    return Err(candle::Error::Msg("Expected CPU storage for mask".into()));
                }
            } else {
                None
            }
        }
        _ => {
            _mask_guard = None;
            None
        }
    };

    let q_stride = q.stride();
    let k_stride = k.stride();
    let v_stride = v.stride();

    let q_len = q.shape().dims()[1];

    // Fast path for decode: q_len == 1
    if q_len == 1 {
        return flash_attn_cpu_single_q(
            q_data,
            k_data,
            v_data,
            mask_data,
            q.shape().dims(),
            k.shape().dims(),
            v.shape().dims(),
            q_stride,
            k_stride,
            v_stride,
            softmax_scale,
            &attn_mask,
            max_bias.unwrap_or(0.0),
            softcap.unwrap_or(0.0),
        );
    }

    flash_attn_cpu(
        q_data,
        k_data,
        v_data,
        mask_data,
        q.shape().dims(),
        k.shape().dims(),
        v.shape().dims(),
        q_stride,
        k_stride,
        v_stride,
        softmax_scale,
        &attn_mask,
        max_bias.unwrap_or(0.0),
        softcap.unwrap_or(0.0),
    )
}

/// Optimised path for the common decode case: q_len == 1 but kv_len >> 1.
/// We drop the inner q‑position loop and parallelise over `(batch, head)`.
#[allow(clippy::too_many_arguments)]
fn flash_attn_cpu_single_q<T: WithDType + Sum + num_traits::real::Real>(
    q_data: &[T],
    k_data: &[T],
    v_data: &[T],
    mask_data: Option<&[T]>,
    qshape: &[usize],
    kshape: &[usize],
    vshape: &[usize],
    qstride: &[usize],
    kstride: &[usize],
    vstride: &[usize],
    scale: f32,
    attn_mask: &AttnMask<'_>,
    max_bias: f32,
    logit_softcap: f32,
) -> Result<Tensor> {
    // Shapes: (B, 1, H, D)
    let (b, _q_len, h, d) = (qshape[0], qshape[1], qshape[2], qshape[3]);
    let kv_len = kshape[1];
    let k_h = kshape[2];
    let v_h = vshape[2];
    let rk2 = h / k_h;
    let rv2 = h / v_h;
    let dv = d;

    let n2 = 2_usize.pow((h as f32).log2().ceil() as u32);

    // Output buffer: (B, H, 1, D)
    let mut out = vec![0f32; b * h * dv];

    // For decode with q_len == 1, the single query is at the end of the sequence.
    // It can attend to ALL prior KV positions regardless of causal setting.
    // So causal masking has no effect here - we iterate all kv_len positions.
    let kv_tiles = kv_len.div_ceil(TILE_KV);

    FLASH_ATTN_POOL.install(|| {
        out.par_chunks_mut(dv)
            .with_min_len(64)
            .enumerate()
            .for_each(|(row_idx, out_chunk)| {
                let b_i = row_idx / h;
                let h_i = row_idx % h;

                // ALiBi positional bias (standard formula)
                let slope = if max_bias > 0.0 {
                    2.0f32.powf(-max_bias * ((h_i + 1) as f32) / n2 as f32)
                } else {
                    0.0
                };

                // For grouped‑KV we collapse multiple query heads into the same K/V head.
                let k_head = h_i / rk2;
                let v_head = h_i / rv2;

                // Gather Q row (strided)
                let q_base = b_i * qstride[0] + 0 * qstride[1] + h_i * qstride[2];
                let mut q_row: Vec<T> = Vec::with_capacity(d);
                for di in 0..d {
                    q_row.push(q_data[q_base + di * qstride[3]]);
                }

                // Parallel reduce over KV tiles
                let (vkq, s_tot, _m_final) = (0..kv_tiles)
                    .into_par_iter()
                    .fold(
                        || (vec![0f32; dv], 0.0f32, f32::NEG_INFINITY),
                        |(mut vkq, mut s, mut m), tile_idx| {
                            let kv_start = tile_idx * TILE_KV;
                            let kv_end = (kv_start + TILE_KV).min(kv_len);

                            let mut k_row: Vec<T> = Vec::with_capacity(d);

                            for kv_pos in kv_start..kv_end {
                                // Get mask value based on mask type
                                let mask_val = match (attn_mask, mask_data) {
                                    (AttnMask::Mask(_), Some(mv)) => {
                                        let mval = mv[(b_i * kv_len) + kv_pos];
                                        mval.to_f64() as f32
                                    }
                                    _ => 0.0, // No masking for None or Causal (decode sees all)
                                };

                                // Skip fully masked positions
                                if mask_val == f32::NEG_INFINITY {
                                    continue;
                                }

                                // ALiBi bias based on position
                                let alibi_bias = if max_bias > 0.0 {
                                    slope * (kv_pos as f32 - (kv_len - 1) as f32)
                                } else {
                                    0.0
                                };

                                // K row (strided)
                                let k_base =
                                    b_i * kstride[0] + kv_pos * kstride[1] + k_head * kstride[2];
                                k_row.clear();
                                for di in 0..d {
                                    k_row.push(k_data[k_base + di * kstride[3]]);
                                }

                                // dot(Q, K)
                                let mut s_val = vec_dot::<T>(&q_row, &k_row).to_f64() as f32;
                                let mut scale_applied = scale;
                                if logit_softcap != 0.0 {
                                    scale_applied /= logit_softcap;
                                }
                                s_val *= scale_applied;
                                if logit_softcap != 0.0 {
                                    s_val = logit_softcap * s_val.tanh();
                                }
                                s_val += alibi_bias + mask_val;

                                // Tile‑local online softmax
                                let m_old = m;
                                let mut ms = 1.0f32;
                                let mut vs = 1.0f32;
                                if s_val > m {
                                    m = s_val;
                                    ms = (m_old - m).exp();
                                    for v in vkq.iter_mut() {
                                        *v *= ms;
                                    }
                                } else {
                                    vs = (s_val - m).exp();
                                }

                                // V row
                                let v_base =
                                    b_i * vstride[0] + kv_pos * vstride[1] + v_head * vstride[2];
                                for d_i in 0..dv {
                                    vkq[d_i] +=
                                        v_data[v_base + d_i * vstride[3]].to_f64() as f32 * vs;
                                }

                                s = s * ms + vs;
                            }

                            (vkq, s, m)
                        },
                    )
                    // Reduce two tiles
                    .reduce(
                        || (vec![0f32; dv], 0.0f32, f32::NEG_INFINITY),
                        |mut a, b| {
                            let (ref mut vkq_a, mut s_a, m_a) = a;
                            let (vkq_b, s_b, m_b) = b;
                            if m_a >= m_b {
                                let factor = (m_b - m_a).exp();
                                for (va, vb) in vkq_a.iter_mut().zip(vkq_b) {
                                    *va += vb * factor;
                                }
                                s_a += s_b * factor;
                                (vkq_a.clone(), s_a, m_a)
                            } else {
                                let factor = (m_a - m_b).exp();
                                let mut vkq_new = vkq_b;
                                for (vb, va) in vkq_new.iter_mut().zip(vkq_a) {
                                    *vb += *va * factor;
                                }
                                (vkq_new, s_b + s_a * factor, m_b)
                            }
                        },
                    );

                // Final normalisation
                let inv_s = 1.0 / s_tot;
                for v in out_chunk.iter_mut().zip(vkq.iter()) {
                    *v.0 = *v.1 * inv_s;
                }
            });
    });

    let out_shape = (b, h, 1usize, dv);
    Tensor::from_vec(out, out_shape, &Device::Cpu)
}

/// Main forward flash-attention CPU routine (prefill path).
/// Shapes follow Candle convention: (B, S, H, D)
#[allow(clippy::too_many_arguments)]
fn flash_attn_cpu<T: WithDType + Sum + num_traits::real::Real>(
    q_data: &[T],
    k_data: &[T],
    v_data: &[T],
    mask_data: Option<&[T]>,
    qshape: &[usize],
    kshape: &[usize],
    vshape: &[usize],
    qstride: &[usize],
    kstride: &[usize],
    vstride: &[usize],
    scale: f32,
    attn_mask: &AttnMask<'_>,
    max_bias: f32,
    logit_softcap: f32,
) -> Result<Tensor> {
    let (b, q_len, h, d) = (qshape[0], qshape[1], qshape[2], qshape[3]);
    let kv_len = kshape[1];

    // Head broadcasting factors for grouped-query attention
    let k_h = kshape[2];
    let v_h = vshape[2];
    let rk2 = h / k_h;
    let rv2 = h / v_h;
    let dv = d;

    // Precompute value for ALiBi slope calculation
    let n2 = 2_usize.pow((h as f32).log2().ceil() as u32);

    // Extract kv_offset for causal masking
    let kv_offset = match attn_mask {
        AttnMask::Causal { kv_offset } => *kv_offset,
        _ => 0,
    };

    let mut out = vec![0f32; b * q_len * h * dv];

    // Rayon‑parallel: each (b_i, h_i, q_pos) row is independent.
    FLASH_ATTN_POOL.install(|| {
        out.par_chunks_mut(dv)
            .with_min_len(64)
            .enumerate()
            .for_each(|(row_idx, out_chunk)| {
                // Decode flat index back to (batch, head, q_pos)
                let rows_per_batch = h * q_len;
                let b_i = row_idx / rows_per_batch;
                let rem = row_idx % rows_per_batch;
                let h_i = rem / q_len;
                let q_pos = rem % q_len;

                let slope = if max_bias > 0.0 {
                    2.0f32.powf(-max_bias * ((h_i + 1) as f32) / n2 as f32)
                } else {
                    0.0
                };

                // For grouped‑KV we collapse multiple query heads into the same K/V head.
                let k_head = h_i / rk2;
                let v_head = h_i / rv2;

                // Buffers local to this row
                let mut vkq = vec![0f32; dv];
                let mut s = 0.0f32;
                let mut m = f32::NEG_INFINITY;

                let mut q_row: Vec<T> = Vec::with_capacity(d);
                let mut k_row: Vec<T> = Vec::with_capacity(d);

                // Gather Q (strided)
                let q_base = b_i * qstride[0] + q_pos * qstride[1] + h_i * qstride[2];
                q_row.clear();
                for di in 0..d {
                    q_row.push(q_data[q_base + di * qstride[3]]);
                }

                // Determine KV iteration bounds based on mask type
                let kv_end = match attn_mask {
                    AttnMask::Causal { kv_offset } => {
                        // Causal: only attend to positions <= current position
                        (q_pos + kv_offset + 1).min(kv_len)
                    }
                    _ => kv_len, // None or Mask: iterate all (Mask filters via values)
                };

                // Iterate over keys/values
                for kv_pos in 0..kv_end {
                    // Get mask value based on mask type
                    let mask_val = match (attn_mask, mask_data) {
                        (AttnMask::Mask(_), Some(mv)) => {
                            let mval = mv[((b_i * q_len + q_pos) * kv_len) + kv_pos];
                            slope * mval.to_f64() as f32
                        }
                        _ => 0.0, // No additional mask value for None or Causal
                    };

                    // Skip fully masked positions (for explicit masks)
                    if mask_val == f32::NEG_INFINITY {
                        continue;
                    }

                    // ALiBi bias based on relative position
                    let alibi_bias = if max_bias > 0.0 {
                        slope * (kv_pos as i64 - (q_pos + kv_offset) as i64) as f32
                    } else {
                        0.0
                    };

                    // K row (strided)
                    let k_base = b_i * kstride[0] + kv_pos * kstride[1] + k_head * kstride[2];
                    k_row.clear();
                    for di in 0..d {
                        k_row.push(k_data[k_base + di * kstride[3]]);
                    }

                    // dot(Q, K)
                    let mut s_val = vec_dot::<T>(&q_row, &k_row);
                    let mut scale_applied = scale;
                    if logit_softcap != 0.0 {
                        scale_applied /= logit_softcap;
                    }
                    s_val *= T::from_f64(scale_applied as f64);
                    if logit_softcap != 0.0 {
                        s_val = T::from_f64(logit_softcap as f64 * s_val.to_f64().tanh());
                    }
                    s_val += T::from_f64((alibi_bias + mask_val) as f64);

                    // Online softmax
                    let m_old = m;
                    let mut ms = 1.0f32;
                    let mut vs = 1.0f32;
                    if s_val.to_f64() as f32 > m {
                        m = s_val.to_f64() as f32;
                        ms = (m_old - m).exp();
                        for v in vkq.iter_mut() {
                            *v *= ms;
                        }
                    } else {
                        vs = (s_val.to_f64() as f32 - m).exp();
                    }

                    // V row (strided)
                    let v_base = b_i * vstride[0] + kv_pos * vstride[1] + v_head * vstride[2];
                    for d_i in 0..dv {
                        vkq[d_i] += v_data[v_base + d_i * vstride[3]].to_f64() as f32 * vs;
                    }

                    s = s * ms + vs;
                }

                // Normalise & write out
                let inv_s = 1.0 / s;
                for v in vkq.iter_mut() {
                    *v *= inv_s;
                }
                out_chunk.copy_from_slice(&vkq);
            });
    });

    // Output shape: (B, H, S, D)
    let out_shape = (b, h, q_len, dv);
    Tensor::from_vec(out, out_shape, &Device::Cpu)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attn_mask_variants() {
        // Test that enum variants can be created
        let _none = AttnMask::None;
        let _causal = AttnMask::causal();
        let _causal_offset = AttnMask::causal_with_offset(512);
    }
}
