#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle::{CpuDevice, CpuStorage, Result, Tensor, WithDType};
use std::sync::LazyLock;
use std::{f32, iter::Sum};

use rayon::prelude::*;
use rayon::ThreadPool;

type CpuTensor = Tensor<CpuStorage>;

#[cfg(target_os = "macos")]
/// Elevate the thread QoS so macOS prefers running it on Performance (P) cores.
unsafe fn set_thread_affinity() {
    // USER_INTERACTIVE has the highest scheduling priority that user code
    // can request and is most likely to be scheduled on P‑cores.
    use libc::{pthread_set_qos_class_self_np, qos_class_t::QOS_CLASS_USER_INTERACTIVE};
    // The second argument is a relative priority within the QoS class (0 = default).
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
/// Computes softmax(qk^T*scale)v.
///
/// **Inputs shapes:**
/// - `q`: (bs, seq, qhead, hidden)
/// - `k`: (bs, kv_seq, v_head, hidden)
/// - `k`: (bs, kv_seq, kv_head_seq, v_hidden)
/// - `scale` is applied before softmax.
///
/// - This supports ALiBi with `max_bias` as well as softcapping with `softcap`.
///
/// **Output shape:** (bs, qhead, seq, v_hidden)
pub fn run_flash_attn_cpu<T>(
    q: &CpuTensor,
    k: &CpuTensor,
    v: &CpuTensor,
    mask: Option<&CpuTensor>,
    softmax_scale: f32,
    max_bias: Option<f32>,
    softcap: Option<f32>,
) -> Result<CpuTensor>
where
    T: WithDType + Sum + num_traits::real::Real,
{
    // Inline CPU slice extraction for q, k, v, and optional mask
    let (q_guard, q_layout) = q.storage_and_layout();
    let q_data: &[T] = {
        let data = q_guard.as_slice::<T>()?;
        &data[q_layout.start_offset()..]
    };
    let (k_guard, k_layout) = k.storage_and_layout();
    let k_data: &[T] = {
        let data = k_guard.as_slice::<T>()?;
        &data[k_layout.start_offset()..]
    };
    let (v_guard, v_layout) = v.storage_and_layout();
    let v_data: &[T] = {
        let data = v_guard.as_slice::<T>()?;
        &data[v_layout.start_offset()..]
    };
    let mask_guard = mask.map(|mask| mask.storage_and_layout().0);
    let mask_data: Option<&[T]> = if let Some(mask_guard) = &mask_guard {
        let mask = mask.as_ref().unwrap();
        let data = mask_guard.as_slice::<T>()?;
        Some(&data[mask.layout().start_offset()..])
    } else {
        None
    };
    // q_guard, k_guard, v_guard, and m_guard (if any) are kept in scope to hold storage alive

    let q_stride = q.stride();
    let k_stride = k.stride();
    let v_stride = v.stride();

    // Fast path for decode: q_len == 1
    if q.shape().dims()[1] == 1 {
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
        max_bias.unwrap_or(0.0),
        softcap.unwrap_or(0.0),
    )
}

/// Optimised path for the common decode case: q_len == 1 but kv_len ≫ 1.
/// We drop the inner q‑position loop and parallelise over `(batch, head)`.
#[allow(clippy::too_many_arguments)]
fn flash_attn_cpu_single_q<T: WithDType + Sum + num_traits::real::Real>(
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
) -> Result<CpuTensor> {
    // Shapes: (B, 1, H, D)
    let (b, _q_len, h, d) = (
        qshape[0], qshape[1], // == 1
        qshape[2], qshape[3],
    );
    let kv_len = kshape[1];
    let k_h = kshape[2];
    let v_h = vshape[2];
    let rk2 = h / k_h;
    let rv2 = h / v_h;
    let dv = d;

    let n2 = 2_usize.pow((h as f32).log2().ceil() as u32);

    // Output buffer: (B, H, 1, D)
    let mut out = vec![0f32; b * h * dv];

    // Expose a second dimension of work: split the KV axis into tiles that
    // fit in the last‑level cache and let Rayon schedule them.
    let kv_tiles = kv_len.div_ceil(TILE_KV);

    // SAFETY: `par_chunks_mut` hands out non‑overlapping &mut slices, so no two
    // threads write the same output area.
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
                    1.0
                };

                // For grouped‑KV we collapse multiple query heads into the same K/V head.
                let k_head = h_i / rk2;
                let v_head = h_i / rv2;

                // ------------------------------------------------------------------
                // Nested parallelism: each KV tile is mapped independently, then we
                // reduce the partial results with the correct soft‑max algebra.
                // ------------------------------------------------------------------
                let (vkq, s_tot, _m_tot) = (0..kv_tiles)
                    .into_par_iter()
                    .map(|tile_idx| {
                        // ---- per‑tile scratch -------------------------------------------------
                        let start = tile_idx * TILE_KV;
                        let end = (start + TILE_KV).min(kv_len);

                        let mut vkq = vec![0f32; dv];
                        let mut s = 0.0f32;
                        let mut m = f32::NEG_INFINITY;

                        // ---------------- single‑Q row (already contiguous) -------------------
                        let q_base =
                            b_i * qstride[0] /*batch*/ + h_i * qstride[2] /*head*/;
                        let q_row = &q_data[q_base..q_base + d];

                        // ---------------- iterate over this KV slice --------------------------
                        for kv_pos in start..end {
                            // Mask
                            let mv = if let Some(mv_vec) = mask_vec {
                                let mval = mv_vec[(b_i * kv_len) + kv_pos];
                                slope * mval.to_f64() as f32
                            } else {
                                0.0
                            };
                            if mv == f32::NEG_INFINITY {
                                continue;
                            }

                            // K row
                            let k_base =
                                b_i * kstride[0] + kv_pos * kstride[1] + k_head * kstride[2];
                            let k_row = &k_data[k_base..k_base + d];

                            // dot(Q, K)
                            let mut s_val = vec_dot::<T>(q_row, k_row).to_f64() as f32;

                            let mut scale_applied = scale;
                            if logit_softcap != 0.0 {
                                scale_applied /= logit_softcap;
                            }
                            s_val *= scale_applied;
                            if logit_softcap != 0.0 {
                                s_val = logit_softcap * s_val.tanh();
                            }
                            s_val += mv;

                            // Tile‑local online softmax ------------------------------------------
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
                                vkq[d_i] += v_data[v_base + d_i * vstride[3]].to_f64() as f32 * vs;
                            }

                            s = s * ms + vs;
                        }

                        // Return per‑tile accumulator + softmax stats
                        (vkq, s, m)
                    })
                    // -------- reduce two tiles -----------------------------------------------
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

                // ---------------- final normalisation ---------------------------------------
                let inv_s = 1.0 / s_tot;
                for v in out_chunk.iter_mut().zip(vkq.iter()) {
                    *v.0 = *v.1 * inv_s;
                }
            });
    });

    let out_shape = (b, h, 1usize, dv);
    Tensor::from_vec(out, out_shape, &CpuDevice)
}

/// Main forward flash-attention CPU routine.
/// Shapes follow Candle convention: (B, S, H, D)
#[allow(clippy::too_many_arguments)]
fn flash_attn_cpu<T: WithDType + Sum + num_traits::real::Real>(
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
) -> Result<CpuTensor> {
    let (b, q_len, h, d) = (qshape[0], qshape[1], qshape[2], qshape[3]);
    let kv_len = kshape[1];
    // --- Head broadcasting factors ----------------------------------------------------
    // Allows K and V to have fewer heads than Q (grouped‑KV); the ratio is an
    // integer factor.  rk2 = #Q‑heads / #K‑heads,  rv2 = #Q‑heads / #V‑heads.
    let k_h = kshape[2];
    let v_h = vshape[2];
    let rk2 = h / k_h; // must divide exactly; panic otherwise
    let rv2 = h / v_h;
    let dv = d; // value dim = key dim in this kernel

    // Precompute value for ALiBi slope calculation
    let n2 = 2_usize.pow((h as f32).log2().ceil() as u32);

    let mut out = vec![0f32; b * q_len * h * dv];

    // ------------------------------------------------------------------
    // Rayon‑parallel version: each (b_i, h_i, q_pos) row is independent.
    // ------------------------------------------------------------------

    let _rows = b * h * q_len; // total independent work items

    // SAFETY: `par_chunks_mut` hands out non‑overlapping &mut [f32] slices,
    // so no two threads can write the same output area.
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
                    1.0
                };

                // For grouped‑KV we collapse multiple query heads into the same K/V head.
                let k_head = h_i / rk2;
                let v_head = h_i / rv2;

                // Buffers local to this row
                let mut vkq = vec![0f32; dv];
                let mut s = 0.0f32;
                let mut m = f32::NEG_INFINITY;

                // Allocate q_row and k_row once per row
                let mut q_row: Vec<T> = Vec::with_capacity(d);
                let mut k_row: Vec<T> = Vec::with_capacity(d);

                // ------------------- gather Q (strided) --------------------
                let q_base = b_i * qstride[0] + q_pos * qstride[1] + h_i * qstride[2];
                q_row.clear();
                for di in 0..d {
                    q_row.push(q_data[q_base + di * qstride[3]]);
                }

                // ---------------- iterate over keys/values -----------------
                for kv_pos in 0..kv_len {
                    // Mask (optional)
                    let mv = if let Some(mv_vec) = mask_vec {
                        let mval = mv_vec[((b_i * q_len + q_pos) * kv_len) + kv_pos];
                        slope * mval.to_f64() as f32
                    } else {
                        0.0
                    };
                    if mv == f32::NEG_INFINITY {
                        continue;
                    }

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
                    s_val += T::from_f64(mv as f64);

                    // online softmax
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

                // ------------------- normalise & write out ------------------
                let inv_s = 1.0 / s;
                for v in vkq.iter_mut() {
                    *v *= inv_s;
                }
                out_chunk.copy_from_slice(&vkq);
            });
    });

    // Build output tensor with shape (B, H, S, D) to match standard (permute 0,2,1,3)
    let out_shape = (b, h, q_len, dv);
    Tensor::from_vec(out, out_shape, &CpuDevice)
}
