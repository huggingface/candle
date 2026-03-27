//! Single-batch (B=1) causal attention using loop-bound masking.
//!
//! Instead of materializing a mask tensor and checking each position,
//! this implementation computes the causal boundary and only iterates
//! over valid positions. This skips ~50% of work for causal attention.
//!
//! All functions assume B=1 and will error on multi-batch input.
//! For B>1, the dispatcher in `mod.rs` routes to the packed varlen path.

use candle::{Device, Result, Storage, Tensor, WithDType};
use rayon::prelude::*;
use std::iter::Sum;

use super::standard::{vec_dot, FLASH_ATTN_POOL};

/// Causal attention optimized with loop-bound masking, **B=1 only**.
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
    let b = q.shape().dims()[0];
    if b != 1 {
        candle::bail!(
            "causal::run_causal_attn_cpu is B=1 only (got B={b}). \
             Multi-batch should be routed through the varlen path."
        );
    }

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
        causal_decode(
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
            max_bias.unwrap_or(0.0),
            softcap.unwrap_or(0.0),
        )
    } else {
        causal_prefill(
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

/// Decode path (B=1, q_len=1): attends to all kv_len positions.
///
/// Parallel over heads only (no nested tile parallelism). Each thread
/// reuses a single accumulator buffer via `for_each_init`, eliminating
/// per-call Vec allocations. Sequential KV loop per head keeps the
/// online softmax simple and avoids tile-reduce overhead.
#[allow(clippy::too_many_arguments)]
fn causal_decode<T: WithDType + Sum + num_traits::real::Real>(
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
    max_bias: f32,
    logit_softcap: f32,
) -> Result<Tensor> {
    let (h, d) = (qshape[2], qshape[3]);
    let kv_len = kshape[1];
    let k_h = kshape[2];
    let v_h = vshape[2];
    let rk = h / k_h;
    let rv = h / v_h;
    let dv = d;

    let n2 = 2_usize.pow((h as f32).log2().ceil() as u32);

    // Precompute scale (fold softcap into scale if active)
    let (scale_pre, do_softcap) = if logit_softcap != 0.0 {
        (scale / logit_softcap, true)
    } else {
        (scale, false)
    };

    // V contiguous last dim? Use direct slices.
    let v_contiguous = vstride[3] == 1;

    let mut out = vec![0f32; h * dv];

    FLASH_ATTN_POOL.install(|| {
        out.par_chunks_mut(dv)
            .enumerate()
            .for_each_init(
                || vec![0f32; dv], // thread-local accumulator, reused across heads
                |vkq, (h_i, out_chunk)| {
                    let slope = if max_bias > 0.0 {
                        2.0f32.powf(-max_bias * ((h_i + 1) as f32) / n2 as f32)
                    } else {
                        0.0
                    };

                    let k_head = h_i / rk;
                    let v_head = h_i / rv;

                    // B=1: no batch offset
                    let q_base = h_i * qstride[2];
                    let q_row = &q_data[q_base..q_base + d];

                    // Reset accumulator
                    vkq.fill(0.0);
                    let mut s = 0.0f32;
                    let mut m = f32::NEG_INFINITY;

                    // Sequential KV loop — no tile parallelism overhead
                    for kv_pos in 0..kv_len {
                        let alibi_bias = if max_bias > 0.0 {
                            slope * (kv_pos as f32 - (kv_len - 1) as f32)
                        } else {
                            0.0
                        };

                        let k_base = kv_pos * kstride[1] + k_head * kstride[2];
                        let k_row = &k_data[k_base..k_base + d];

                        // QK dot product — stays in f32 via vec_dot
                        let mut s_val = vec_dot::<T>(q_row, k_row).to_f32().unwrap_or(0.0);

                        s_val *= scale_pre;
                        if do_softcap {
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

                        // Accumulate V — direct slice when contiguous
                        let v_base = kv_pos * vstride[1] + v_head * vstride[2];
                        if v_contiguous {
                            let v_row = &v_data[v_base..v_base + dv];
                            for d_i in 0..dv {
                                vkq[d_i] += v_row[d_i].to_f32().unwrap_or(0.0) * vs;
                            }
                        } else {
                            for d_i in 0..dv {
                                vkq[d_i] +=
                                    v_data[v_base + d_i * vstride[3]].to_f32().unwrap_or(0.0) * vs;
                            }
                        }

                        s = s * ms + vs;
                    }

                    // Normalize & write
                    let inv_s = if s > 0.0 { 1.0 / s } else { 0.0 };
                    for (out_v, acc_v) in out_chunk.iter_mut().zip(vkq.iter()) {
                        *out_v = *acc_v * inv_s;
                    }
                },
            );
    });

    Tensor::from_vec(out, (1usize, h, 1usize, dv), &Device::Cpu)
}

/// Prefill path (B=1, q_len > 1): loop-bound causal masking, direct slices.
///
/// Each query at position q_pos can attend to KV positions [0, q_pos + kv_offset].
/// Uses thread-local buffers to avoid per-row Vec allocations.
#[allow(clippy::too_many_arguments)]
fn causal_prefill<T: WithDType + Sum + num_traits::real::Real>(
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
    let (q_len, h, d) = (qshape[1], qshape[2], qshape[3]);
    let kv_len = kshape[1];
    let k_h = kshape[2];
    let v_h = vshape[2];
    let rk = h / k_h;
    let rv = h / v_h;
    let dv = d;

    let n2 = 2_usize.pow((h as f32).log2().ceil() as u32);

    let (scale_pre, do_softcap) = if logit_softcap != 0.0 {
        (scale / logit_softcap, true)
    } else {
        (scale, false)
    };

    let v_contiguous = vstride[3] == 1;

    let mut out = vec![0f32; q_len * h * dv];

    FLASH_ATTN_POOL.install(|| {
        out.par_chunks_mut(dv)
            .with_min_len(64)
            .enumerate()
            .for_each_init(
                || vec![0f32; dv], // thread-local accumulator
                |vkq, (row_idx, out_chunk)| {
                    // Flat (h, q_pos) layout — no batch dimension
                    let h_i = row_idx / q_len;
                    let q_pos = row_idx % q_len;

                    let slope = if max_bias > 0.0 {
                        2.0f32.powf(-max_bias * ((h_i + 1) as f32) / n2 as f32)
                    } else {
                        0.0
                    };

                    let k_head = h_i / rk;
                    let v_head = h_i / rv;

                    // Reset accumulator
                    vkq.fill(0.0);
                    let mut s = 0.0f32;
                    let mut m = f32::NEG_INFINITY;

                    // Direct slice — no batch offset, contiguous last dim
                    let q_base = q_pos * qstride[1] + h_i * qstride[2];
                    let q_row = &q_data[q_base..q_base + d];

                    // LOOP-BOUND CAUSAL: only iterate up to causal boundary
                    let kv_end = (q_pos + kv_offset + 1).min(kv_len);

                    for kv_pos in 0..kv_end {
                        let alibi_bias = if max_bias > 0.0 {
                            slope * (kv_pos as i64 - (q_pos + kv_offset) as i64) as f32
                        } else {
                            0.0
                        };

                        // K row — direct slice
                        let k_base = kv_pos * kstride[1] + k_head * kstride[2];
                        let k_row = &k_data[k_base..k_base + d];

                        let mut s_val = vec_dot::<T>(q_row, k_row).to_f32().unwrap_or(0.0);

                        s_val *= scale_pre;
                        if do_softcap {
                            s_val = logit_softcap * s_val.tanh();
                        }
                        s_val += alibi_bias;

                        // Online softmax
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

                        // V row — direct slice when contiguous
                        let v_base = kv_pos * vstride[1] + v_head * vstride[2];
                        if v_contiguous {
                            let v_row = &v_data[v_base..v_base + dv];
                            for d_i in 0..dv {
                                vkq[d_i] += v_row[d_i].to_f32().unwrap_or(0.0) * vs;
                            }
                        } else {
                            for d_i in 0..dv {
                                vkq[d_i] +=
                                    v_data[v_base + d_i * vstride[3]].to_f32().unwrap_or(0.0) * vs;
                            }
                        }

                        s = s * ms + vs;
                    }

                    // Normalize & write
                    let inv_s = if s > 0.0 { 1.0 / s } else { 0.0 };
                    for v in vkq.iter_mut() {
                        *v *= inv_s;
                    }
                    out_chunk.copy_from_slice(vkq);
                },
            );
    });

    Tensor::from_vec(out, (1usize, h, q_len, dv), &Device::Cpu)
}
