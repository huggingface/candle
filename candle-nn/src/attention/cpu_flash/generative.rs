//! Optimized flash attention for generative inference (decode).
//!
//! Tuned for the autoregressive generation case: single query token (q_len=1)
//! attending to a long KV sequence. Key optimizations:
//!
//! - **Tiled KV processing**: Cache-friendly tiles with parallel reduction
//! - **Nested parallelism**: Outer over (batch, head), inner over KV tiles
//! - **Loop-bound causal masking**: Skips masked positions entirely
//! - **Online softmax**: Numerically stable incremental computation

use candle::{Device, Result, Storage, Tensor, WithDType};
use rayon::prelude::*;
use std::iter::Sum;

use super::standard::{vec_dot, FLASH_ATTN_POOL};
use crate::attention::AttnMask;

/// KV tile size for cache-friendly processing.
const TILE_KV: usize = 16;

/// Flash attention optimized for generative decode (q_len=1).
pub fn flash_attn_generative<T>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    attn_mask: &AttnMask<'_>,
    max_bias: f32,
    logit_softcap: f32,
) -> Result<Tensor>
where
    T: WithDType + Sum + num_traits::real::Real,
{
    eprintln!(">>> GENERATIVE PATH: q_len={}", q.dims()[1]);
    // Extract CPU slices
    let (q_guard, q_layout) = q.storage_and_layout();
    let q_data: &[T] = match &*q_guard {
        Storage::Cpu(cpu) => &cpu.as_slice::<T>()?[q_layout.start_offset()..],
        _ => return Err(candle::Error::Msg("Expected CPU storage for q".into())),
    };

    let (k_guard, k_layout) = k.storage_and_layout();
    let k_data: &[T] = match &*k_guard {
        Storage::Cpu(cpu) => &cpu.as_slice::<T>()?[k_layout.start_offset()..],
        _ => return Err(candle::Error::Msg("Expected CPU storage for k".into())),
    };

    let (v_guard, v_layout) = v.storage_and_layout();
    let v_data: &[T] = match &*v_guard {
        Storage::Cpu(cpu) => &cpu.as_slice::<T>()?[v_layout.start_offset()..],
        _ => return Err(candle::Error::Msg("Expected CPU storage for v".into())),
    };

    flash_attn_decode_inner(
        q_data,
        k_data,
        v_data,
        q.shape().dims(),
        k.shape().dims(),
        v.shape().dims(),
        q.stride(),
        k.stride(),
        v.stride(),
        softmax_scale,
        attn_mask,
        max_bias,
        logit_softcap,
    )
}

#[allow(clippy::too_many_arguments)]
fn flash_attn_decode_inner<T>(
    q_data: &[T],
    k_data: &[T],
    v_data: &[T],
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
) -> Result<Tensor>
where
    T: WithDType + Sum + num_traits::real::Real,
{
    let (b, _q_len, h, d) = (qshape[0], qshape[1], qshape[2], qshape[3]);
    let kv_len = kshape[1];
    let k_h = kshape[2];
    let v_h = vshape[2];

    // Grouped-query attention ratios
    let rk = h / k_h;
    let rv = h / v_h;
    let dv = d;

    // ALiBi slope calculation
    let n2 = 2_usize.pow((h as f32).log2().ceil() as u32);

    // Causal parameters
    let is_causal = attn_mask.is_causal();
    let kv_offset = attn_mask.kv_offset();

    // Output buffer: (B, H, 1, D)
    let mut out = vec![0f32; b * h * dv];
    let kv_tiles = kv_len.div_ceil(TILE_KV);

    FLASH_ATTN_POOL.install(|| {
        out.par_chunks_mut(dv)
            .with_min_len(64)
            .enumerate()
            .for_each(|(row_idx, out_chunk)| {
                let b_i = row_idx / h;
                let h_i = row_idx % h;

                // ALiBi slope
                let _slope = if max_bias > 0.0 {
                    2.0f32.powf(-max_bias * ((h_i + 1) as f32) / n2 as f32)
                } else {
                    1.0
                };

                // GQA head mapping
                let k_head = h_i / rk;
                let v_head = h_i / rv;

                // Q row (contiguous for single-q)
                let q_base = b_i * qstride[0] + h_i * qstride[2];
                let q_row = &q_data[q_base..q_base + d];

                // Parallel reduce over KV tiles
                let (vkq, s_tot, _m_tot) = (0..kv_tiles)
                    .into_par_iter()
                    .map(|tile_idx| {
                        let start = tile_idx * TILE_KV;
                        let end = (start + TILE_KV).min(kv_len);

                        // Skip tiles past causal boundary
                        if is_causal && start > kv_offset {
                            return (vec![0f32; dv], 0.0f32, f32::NEG_INFINITY);
                        }

                        let mut vkq = vec![0f32; dv];
                        let mut s = 0.0f32;
                        let mut m = f32::NEG_INFINITY;

                        for kv_pos in start..end {
                            // Causal check
                            if is_causal && kv_pos > kv_offset {
                                continue;
                            }

                            // K row
                            let k_base =
                                b_i * kstride[0] + kv_pos * kstride[1] + k_head * kstride[2];
                            let k_row = &k_data[k_base..k_base + d];

                            // QK dot product
                            let mut s_val = vec_dot::<T>(q_row, k_row).to_f64() as f32;

                            // Scale + optional softcap
                            let mut scale_applied = scale;
                            if logit_softcap != 0.0 {
                                scale_applied /= logit_softcap;
                            }
                            s_val *= scale_applied;
                            if logit_softcap != 0.0 {
                                s_val = logit_softcap * s_val.tanh();
                            }

                            // Online softmax
                            let m_old = m;
                            let (ms, vs) = if s_val > m {
                                m = s_val;
                                let ms = (m_old - m).exp();
                                for v in vkq.iter_mut() {
                                    *v *= ms;
                                }
                                (ms, 1.0f32)
                            } else {
                                (1.0f32, (s_val - m).exp())
                            };

                            // Accumulate V
                            let v_base =
                                b_i * vstride[0] + kv_pos * vstride[1] + v_head * vstride[2];
                            for d_i in 0..dv {
                                vkq[d_i] += v_data[v_base + d_i * vstride[3]].to_f64() as f32 * vs;
                            }

                            s = s * ms + vs;
                        }

                        (vkq, s, m)
                    })
                    .reduce(
                        || (vec![0f32; dv], 0.0f32, f32::NEG_INFINITY),
                        |a, b| merge_softmax_accumulators(a, b),
                    );

                // Final normalization
                let inv_s = if s_tot > 0.0 { 1.0 / s_tot } else { 0.0 };
                for (out_v, acc_v) in out_chunk.iter_mut().zip(vkq.iter()) {
                    *out_v = *acc_v * inv_s;
                }
            });
    });

    Tensor::from_vec(out, (b, h, 1usize, dv), &Device::Cpu)
}

/// Merge two online softmax accumulators.
#[inline]
fn merge_softmax_accumulators(
    a: (Vec<f32>, f32, f32),
    b: (Vec<f32>, f32, f32),
) -> (Vec<f32>, f32, f32) {
    let (vkq_a, s_a, m_a) = a;
    let (vkq_b, s_b, m_b) = b;

    if m_a >= m_b {
        let factor = (m_b - m_a).exp();
        let mut vkq = vkq_a;
        for (va, vb) in vkq.iter_mut().zip(vkq_b.iter()) {
            *va += *vb * factor;
        }
        (vkq, s_a + s_b * factor, m_a)
    } else {
        let factor = (m_a - m_b).exp();
        let mut vkq = vkq_b;
        for (vb, va) in vkq.iter_mut().zip(vkq_a.iter()) {
            *vb += va * factor;
        }
        (vkq, s_b + s_a * factor, m_b)
    }
}
