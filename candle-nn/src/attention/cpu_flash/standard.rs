#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Single-batch (B=1) CPU flash attention kernels.
//!
//! These kernels assume B=1 and will error on multi-batch input.
//! For B>1, the dispatcher in `mod.rs` routes to the packed varlen path.
//! This invariant lets every inner loop use direct slice references
//! (no strided gather, no batch-dim indexing), which is the foundation
//! for future SIMD / cache-tiling optimizations.

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
pub(crate) static FLASH_ATTN_POOL: LazyLock<ThreadPool> = LazyLock::new(|| {
    rayon::ThreadPoolBuilder::new()
        .start_handler(|_| unsafe {
            set_thread_affinity();
        })
        .build()
        .expect("Failed to build custom Rayon thread‑pool for flash‑attention")
});

const DOT_CHUNK: usize = 4;

#[inline]
pub(crate) fn vec_dot<T: WithDType + Sum + Copy + std::ops::Mul<Output = T>>(
    a: &[T],
    b: &[T],
) -> T {
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

/// Fused attention optimized for CPU, **B=1 only**.
///
/// Computes softmax(qk^T*scale)v.
///
/// **Input shapes:** `(1, seq, qhead, hidden)`, `(1, kv_seq, kv_head, hidden)`, ...
///
/// **Output shape:** `(1, qhead, seq, v_hidden)`
///
/// Panics / errors if `B != 1`.
pub fn run_flash_attn_cpu<T>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    softmax_scale: f32,
    max_bias: Option<f32>,
    softcap: Option<f32>,
) -> Result<Tensor>
where
    T: WithDType + Sum + num_traits::real::Real,
{
    let b = q.shape().dims()[0];
    if b != 1 {
        candle::bail!(
            "standard::run_flash_attn_cpu is B=1 only (got B={b}). \
             Multi-batch should be routed through the varlen path."
        );
    }

    // Inline CPU slice extraction for q, k, v, and optional mask
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
    let mask_guard = mask.map(|mask| mask.storage_and_layout().0);
    let mask_data: Option<&[T]> = if let Some(mask_guard) = &mask_guard {
        let mask = mask.as_ref().unwrap();

        if let Storage::Cpu(cpu) = &**mask_guard {
            let data = cpu.as_slice::<T>()?;
            Some(&data[mask.layout().start_offset()..])
        } else {
            return Err(candle::Error::Msg("Expected CPU storage for mask".into()));
        }
    } else {
        None
    };

    let q_stride = q.stride();
    let k_stride = k.stride();
    let v_stride = v.stride();

    // Decode fast path: q_len == 1
    if q.shape().dims()[1] == 1 {
        return flash_attn_decode(
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
            max_bias.unwrap_or(0.0),
            softcap.unwrap_or(0.0),
        );
    }

    // Prefill path (q_len > 1)
    flash_attn_prefill(
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
        max_bias.unwrap_or(0.0),
        softcap.unwrap_or(0.0),
    )
}

/// Decode path (B=1, q_len=1): sequential KV loop per head, parallel over heads.
///
/// Uses thread-local accumulator via `for_each_init` to avoid per-call allocations.
#[allow(clippy::too_many_arguments)]
fn flash_attn_decode<T: WithDType + Sum + num_traits::real::Real>(
    q_data: &[T],
    k_data: &[T],
    v_data: &[T],
    mask_vec: Option<&[T]>,
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
    let rk2 = h / k_h;
    let rv2 = h / v_h;
    let dv = d;

    let n2 = 2_usize.pow((h as f32).log2().ceil() as u32);

    let (scale_pre, do_softcap) = if logit_softcap != 0.0 {
        (scale / logit_softcap, true)
    } else {
        (scale, false)
    };

    let v_contiguous = vstride[3] == 1;

    let mut out = vec![0f32; h * dv];

    FLASH_ATTN_POOL.install(|| {
        out.par_chunks_mut(dv).enumerate().for_each_init(
            || vec![0f32; dv],
            |vkq, (h_i, out_chunk)| {
                let slope = if max_bias > 0.0 {
                    2.0f32.powf(-max_bias * ((h_i + 1) as f32) / n2 as f32)
                } else {
                    1.0
                };

                let k_head = h_i / rk2;
                let v_head = h_i / rv2;

                let q_base = h_i * qstride[2];
                let q_row = &q_data[q_base..q_base + d];

                // Reset accumulator
                vkq.fill(0.0);
                let mut s = 0.0f32;
                let mut m = f32::NEG_INFINITY;

                // Sequential KV loop — no tile parallelism overhead
                for kv_pos in 0..kv_len {
                    // Mask
                    let mv = if let Some(mv_vec) = mask_vec {
                        let mval = mv_vec[kv_pos];
                        slope * mval.to_f32().unwrap_or(0.0)
                    } else {
                        0.0
                    };
                    if mv == f32::NEG_INFINITY {
                        continue;
                    }

                    let k_base = kv_pos * kstride[1] + k_head * kstride[2];
                    let k_row = &k_data[k_base..k_base + d];

                    let mut s_val = vec_dot::<T>(q_row, k_row).to_f32().unwrap_or(0.0);

                    s_val *= scale_pre;
                    if do_softcap {
                        s_val = logit_softcap * s_val.tanh();
                    }
                    s_val += mv;

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

                let inv_s = if s > 0.0 { 1.0 / s } else { 0.0 };
                for (out_v, acc_v) in out_chunk.iter_mut().zip(vkq.iter()) {
                    *out_v = *acc_v * inv_s;
                }
            },
        );
    });

    Tensor::from_vec(out, (1usize, h, 1usize, dv), &Device::Cpu)
}

/// Prefill path (B=1, q_len > 1): direct slice access, no batch indexing.
/// Uses thread-local buffers to avoid per-row Vec allocations.
#[allow(clippy::too_many_arguments)]
fn flash_attn_prefill<T: WithDType + Sum + num_traits::real::Real>(
    q_data: &[T],
    k_data: &[T],
    v_data: &[T],
    mask_vec: Option<&[T]>,
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
    let (q_len, h, d) = (qshape[1], qshape[2], qshape[3]);
    let kv_len = kshape[1];
    let k_h = kshape[2];
    let v_h = vshape[2];
    let rk2 = h / k_h;
    let rv2 = h / v_h;
    let dv = d;

    let n2 = 2_usize.pow((h as f32).log2().ceil() as u32);

    let (scale_pre, do_softcap) = if logit_softcap != 0.0 {
        (scale / logit_softcap, true)
    } else {
        (scale, false)
    };

    let v_contiguous = vstride[3] == 1;

    let mut out = vec![0f32; h * q_len * dv];

    FLASH_ATTN_POOL.install(|| {
        out.par_chunks_mut(dv)
            .with_min_len(64)
            .enumerate()
            .for_each_init(
                || vec![0f32; dv],
                |vkq, (row_idx, out_chunk)| {
                    let h_i = row_idx / q_len;
                    let q_pos = row_idx % q_len;

                    let slope = if max_bias > 0.0 {
                        2.0f32.powf(-max_bias * ((h_i + 1) as f32) / n2 as f32)
                    } else {
                        1.0
                    };

                    let k_head = h_i / rk2;
                    let v_head = h_i / rv2;

                    // Reset accumulator
                    vkq.fill(0.0);
                    let mut s = 0.0f32;
                    let mut m = f32::NEG_INFINITY;

                    let q_base = q_pos * qstride[1] + h_i * qstride[2];
                    let q_row = &q_data[q_base..q_base + d];

                    for kv_pos in 0..kv_len {
                        let mv = if let Some(mv_vec) = mask_vec {
                            let mval = mv_vec[q_pos * kv_len + kv_pos];
                            slope * mval.to_f32().unwrap_or(0.0)
                        } else {
                            0.0
                        };
                        if mv == f32::NEG_INFINITY {
                            continue;
                        }

                        let k_base = kv_pos * kstride[1] + k_head * kstride[2];
                        let k_row = &k_data[k_base..k_base + d];

                        let mut s_val = vec_dot::<T>(q_row, k_row).to_f32().unwrap_or(0.0);
                        s_val *= scale_pre;
                        if do_softcap {
                            s_val = logit_softcap * s_val.tanh();
                        }
                        s_val += mv;

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
