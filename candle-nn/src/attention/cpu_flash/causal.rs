//! Optimized causal attention using loop-bound masking.
//!
//! Instead of materializing a mask tensor and checking each position,
//! this implementation computes the causal boundary and only iterates
//! over valid positions. This skips ~50% of work for causal attention.

use candle::{Device, Result, Storage, Tensor, WithDType};
use rayon::prelude::*;
use std::iter::Sum;

use super::standard::{vec_dot, FLASH_ATTN_POOL};

/// Size (in KV positions) processed by each inner-tile job.
const TILE_KV: usize = 16;

/// Causal attention optimized with loop-bound masking.
///
/// Dispatches to decode (q_len=1) or prefill (q_len>1) paths.
#[allow(clippy::too_many_arguments)]
pub fn run_causal_attn_cpu<T>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    kv_offset: usize,
    max_bias: Option<f32>,
    softcap: Option<f32>,
) -> Result<Tensor>
where
    T: WithDType + Sum + num_traits::real::Real,
{
    // Extract CPU slices for q, k, v
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

    let q_stride = q.stride();
    let k_stride = k.stride();
    let v_stride = v.stride();

    let q_len = q.shape().dims()[1];

    if q_len == 1 {
        causal_attn_decode(
            q_data,
            k_data,
            v_data,
            q.shape().dims(),
            k.shape().dims(),
            v.shape().dims(),
            q_stride,
            k_stride,
            v_stride,
            softmax_scale,
            kv_offset,
            max_bias.unwrap_or(0.0),
            softcap.unwrap_or(0.0),
        )
    } else {
        causal_attn_prefill(
            q_data,
            k_data,
            v_data,
            q.shape().dims(),
            k.shape().dims(),
            v.shape().dims(),
            q_stride,
            k_stride,
            v_stride,
            softmax_scale,
            kv_offset,
            max_bias.unwrap_or(0.0),
            softcap.unwrap_or(0.0),
        )
    }
}

/// Decode path: q_len == 1, attends to all kv_len positions.
///
/// For decode, the single query token is conceptually at position `kv_offset`,
/// so it can attend to all KV positions [0, kv_len).
#[allow(clippy::too_many_arguments)]
fn causal_attn_decode<T: WithDType + Sum + num_traits::real::Real>(
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
    _kv_offset: usize, // Not used for decode - query sees all KV
    max_bias: f32,
    logit_softcap: f32,
) -> Result<Tensor> {
    let (b, _q_len, h, d) = (qshape[0], qshape[1], qshape[2], qshape[3]);
    let kv_len = kshape[1];
    let k_h = kshape[2];
    let v_h = vshape[2];
    let rk = h / k_h;
    let rv = h / v_h;
    let dv = d;

    // ALiBi slope calculation
    let n2 = 2_usize.pow((h as f32).log2().ceil() as u32);

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
                let slope = if max_bias > 0.0 {
                    2.0f32.powf(-max_bias * ((h_i + 1) as f32) / n2 as f32)
                } else {
                    0.0
                };

                // GQA head mapping
                let k_head = h_i / rk;
                let v_head = h_i / rv;

                // Q row - for decode q_len=1, q_pos=0
                let q_base = b_i * qstride[0] + h_i * qstride[2];
                let q_row = &q_data[q_base..q_base + d];

                // Parallel reduce over KV tiles
                let (vkq, s_tot, _m_tot) = (0..kv_tiles)
                    .into_par_iter()
                    .map(|tile_idx| {
                        let start = tile_idx * TILE_KV;
                        let end = (start + TILE_KV).min(kv_len);

                        let mut vkq = vec![0f32; dv];
                        let mut s = 0.0f32;
                        let mut m = f32::NEG_INFINITY;

                        for kv_pos in start..end {
                            // ALiBi bias
                            let alibi_bias = if max_bias > 0.0 {
                                slope * (kv_pos as f32 - (kv_len - 1) as f32)
                            } else {
                                0.0
                            };

                            // K row
                            let k_base =
                                b_i * kstride[0] + kv_pos * kstride[1] + k_head * kstride[2];
                            let k_row = &k_data[k_base..k_base + d];

                            // QK dot product
                            let mut s_val = vec_dot::<T>(q_row, k_row).to_f64() as f32;

                            // Scale + softcap
                            let mut scale_applied = scale;
                            if logit_softcap != 0.0 {
                                scale_applied /= logit_softcap;
                            }
                            s_val *= scale_applied;
                            if logit_softcap != 0.0 {
                                s_val = logit_softcap * s_val.tanh();
                            }
                            s_val += alibi_bias;

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
                        merge_softmax_accumulators,
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

/// Prefill path: q_len > 1, uses loop bounds to skip masked positions.
///
/// Each query at position q_pos can attend to KV positions [0, q_pos + kv_offset].
#[allow(clippy::too_many_arguments)]
fn causal_attn_prefill<T: WithDType + Sum + num_traits::real::Real>(
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
    kv_offset: usize,
    max_bias: f32,
    logit_softcap: f32,
) -> Result<Tensor> {
    let (b, q_len, h, d) = (qshape[0], qshape[1], qshape[2], qshape[3]);
    let kv_len = kshape[1];
    let k_h = kshape[2];
    let v_h = vshape[2];
    let rk = h / k_h;
    let rv = h / v_h;
    let dv = d;

    // ALiBi slope calculation
    let n2 = 2_usize.pow((h as f32).log2().ceil() as u32);

    let mut out = vec![0f32; b * q_len * h * dv];

    FLASH_ATTN_POOL.install(|| {
        out.par_chunks_mut(dv)
            .with_min_len(64)
            .enumerate()
            .for_each(|(row_idx, out_chunk)| {
                // Decode flat index to (batch, head, q_pos)
                let rows_per_batch = h * q_len;
                let b_i = row_idx / rows_per_batch;
                let rem = row_idx % rows_per_batch;
                let h_i = rem / q_len;
                let q_pos = rem % q_len;

                // ALiBi slope
                let slope = if max_bias > 0.0 {
                    2.0f32.powf(-max_bias * ((h_i + 1) as f32) / n2 as f32)
                } else {
                    0.0
                };

                // GQA head mapping
                let k_head = h_i / rk;
                let v_head = h_i / rv;

                // Buffers
                let mut vkq = vec![0f32; dv];
                let mut s = 0.0f32;
                let mut m = f32::NEG_INFINITY;

                let mut q_row: Vec<T> = Vec::with_capacity(d);
                let mut k_row: Vec<T> = Vec::with_capacity(d);

                // Gather Q (strided)
                let q_base = b_i * qstride[0] + q_pos * qstride[1] + h_i * qstride[2];
                for di in 0..d {
                    q_row.push(q_data[q_base + di * qstride[3]]);
                }

                // LOOP-BOUND CAUSAL: only iterate up to causal boundary
                let kv_end = (q_pos + kv_offset + 1).min(kv_len);

                for kv_pos in 0..kv_end {
                    // ALiBi bias
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

                    // QK dot product
                    let mut s_val = vec_dot::<T>(&q_row, &k_row);

                    // Scale + softcap
                    let mut scale_applied = scale;
                    if logit_softcap != 0.0 {
                        scale_applied /= logit_softcap;
                    }
                    s_val *= T::from_f64(scale_applied as f64);
                    if logit_softcap != 0.0 {
                        s_val = T::from_f64(logit_softcap as f64 * s_val.to_f64().tanh());
                    }
                    s_val += T::from_f64(alibi_bias as f64);

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

                // Normalize & write
                let inv_s = if s > 0.0 { 1.0 / s } else { 0.0 };
                for v in vkq.iter_mut() {
                    *v *= inv_s;
                }
                out_chunk.copy_from_slice(&vkq);
            });
    });

    Tensor::from_vec(out, (b, h, q_len, dv), &Device::Cpu)
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
