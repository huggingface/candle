//! Fused variable-length CPU flash attention.
//!
//! Operates on packed (padding-free) sequences with per-sequence lengths,
//! enabling efficient batch processing without wasting compute on padding tokens.
//!
//! Ported from <https://github.com/huggingface/candle/pull/3250>.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle::{DType, Device, Result, Storage, Tensor};
use half::f16;
use rayon::prelude::*;

use super::standard::FLASH_ATTN_POOL;

/// Fused variable-length flash attention on CPU.
///
/// **Input shapes** (packed, no batch dim):
/// - `q`: `[total_q, Hq, D]`
/// - `k`: `[total_k, Hk, D]`
/// - `v`: `[total_k, Hv, D]`
/// - `seqlens_q`, `seqlens_k`: `[B]` (u32 per-sequence lengths)
///
/// **Output shape:** `[total_q, Hq, D]`
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_varlen_cpu(
    q: &Tensor,                    // [total_q, Hq, D]
    k: &Tensor,                    // [total_k, Hk, D]
    v: &Tensor,                    // [total_k, Hv, D]
    alibi_slopes: Option<&Tensor>, // [Hq]
    seqlens_q: &Tensor,            // [B]
    seqlens_k: &Tensor,            // [B]
    _max_seqlen_q: usize,
    _max_seqlen_k: usize,
    softmax_scale: f32,
    causal: bool,
    window_left: Option<usize>,
    window_right: Option<usize>,
) -> Result<Tensor> {
    if !q.device().is_cpu() {
        candle::bail!("flash_attn_varlen_cpu is CPU only");
    }

    let dt = q.dtype();
    if k.dtype() != dt || v.dtype() != dt {
        candle::bail!(
            "q/k/v must have the same dtype (got q={:?}, k={:?}, v={:?})",
            dt,
            k.dtype(),
            v.dtype()
        );
    }
    if dt != DType::F32 && dt != DType::F16 {
        candle::bail!("flash_attn_varlen_cpu supports f32 or f16 only");
    }

    let (total_q, hq, d) = q.dims3()?;
    let (total_k, hk, dk) = k.dims3()?;
    let (_total_v, hv, dv) = v.dims3()?;
    if dk != d || dv != d {
        candle::bail!("head dim mismatch");
    }

    if hq % hk != 0 {
        candle::bail!("Invalid GQA: Hq={} not divisible by Hk={}", hq, hk);
    }
    if hq % hv != 0 {
        candle::bail!("Invalid GQA: Hq={} not divisible by Hv={}", hq, hv);
    }
    let rk = hq / hk;
    let rv = hq / hv;

    let seqlens_q_vec = seqlens_q.to_vec1::<u32>()?;
    let seqlens_k_vec = seqlens_k.to_vec1::<u32>()?;
    let bsz = seqlens_q_vec.len();
    if seqlens_k_vec.len() != bsz {
        candle::bail!("seqlens_q and seqlens_k batch mismatch");
    }

    if causal {
        for i in 0..bsz {
            let lq = seqlens_q_vec[i] as usize;
            let lk = seqlens_k_vec[i] as usize;
            if lk < lq {
                candle::bail!("causal requires lk>=lq, got lk={lk} lq={lq} at batch {i}");
            }
        }
    }

    let mut cumsum_q = vec![0usize; bsz + 1];
    let mut cumsum_k = vec![0usize; bsz + 1];
    for i in 0..bsz {
        cumsum_q[i + 1] = cumsum_q[i] + seqlens_q_vec[i] as usize;
        cumsum_k[i + 1] = cumsum_k[i] + seqlens_k_vec[i] as usize;
    }
    if cumsum_q[bsz] != total_q || cumsum_k[bsz] != total_k {
        candle::bail!("total_q/total_k mismatch with seqlens");
    }

    let slopes: Option<Vec<f32>> = if let Some(s) = alibi_slopes {
        let v = s.to_vec1::<f32>()?;
        if v.len() != hq {
            candle::bail!("alibi_slopes len {} != Hq {}", v.len(), hq);
        }
        Some(v)
    } else {
        None
    };

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;

    // Precompute q_idx -> (batch, q_pos)
    let mut batch_of_q = vec![0usize; total_q];
    let mut pos_in_b = vec![0usize; total_q];
    for b in 0..bsz {
        let start = cumsum_q[b];
        let lq = seqlens_q_vec[b] as usize;
        for t in 0..lq {
            batch_of_q[start + t] = b;
            pos_in_b[start + t] = t;
        }
    }

    #[inline(always)]
    fn alibi_bias(slope: f32, i_k: isize, j: isize, causal: bool) -> f32 {
        let dist = if causal {
            (i_k - j).max(0) as f32
        } else {
            (j - i_k).abs() as f32
        };
        -slope * dist
    }

    #[inline(always)]
    fn key_range(
        q_pos: usize,
        seq_len_k: usize,
        offset: isize,
        causal: bool,
        wl: Option<usize>,
        wr: Option<usize>,
    ) -> Option<(usize, usize)> {
        if seq_len_k == 0 {
            return Some((1, 0));
        }

        let lk = seq_len_k as isize;
        let i_k = q_pos as isize + offset;

        let mut lo: isize = 0;
        let mut hi: isize = lk - 1;

        if causal {
            hi = hi.min(i_k);
        }

        match (wl, wr) {
            (Some(left), Some(right)) => {
                lo = lo.max(i_k - left as isize);
                hi = hi.min(i_k + right as isize);
            }
            (Some(left), None) => {
                hi = hi.min(i_k);
                lo = lo.max(i_k - left as isize);
            }
            (None, None) => {}
            (None, Some(_)) => return None, // invalid config
        }

        lo = lo.max(0);
        hi = hi.min(lk - 1);

        if lo > hi {
            Some((1, 0))
        } else {
            Some((lo as usize, hi as usize))
        }
    }

    match dt {
        DType::F32 => {
            let (q_g, q_l) = q.storage_and_layout();
            let q_data: &[f32] = match &*q_g {
                Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[q_l.start_offset()..],
                _ => candle::bail!("q not cpu"),
            };

            let (k_g, k_l) = k.storage_and_layout();
            let k_data: &[f32] = match &*k_g {
                Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[k_l.start_offset()..],
                _ => candle::bail!("k not cpu"),
            };

            let (v_g, v_l) = v.storage_and_layout();
            let v_data: &[f32] = match &*v_g {
                Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[v_l.start_offset()..],
                _ => candle::bail!("v not cpu"),
            };

            let mut out = vec![0f32; total_q * hq * d];

            #[inline(always)]
            fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
                let mut s = 0.0f32;
                let mut i = 0;
                while i + 4 <= a.len() {
                    s += a[i] * b[i]
                        + a[i + 1] * b[i + 1]
                        + a[i + 2] * b[i + 2]
                        + a[i + 3] * b[i + 3];
                    i += 4;
                }
                while i < a.len() {
                    s += a[i] * b[i];
                    i += 1;
                }
                s
            }

            FLASH_ATTN_POOL.install(|| {
                out.par_chunks_mut(d).enumerate().for_each_init(
                    || vec![0f32; d],
                    |acc, (row, out_row)| {
                        let q_idx = row / hq;
                        let h = row % hq;

                        let b = batch_of_q[q_idx];
                        let q_pos = pos_in_b[q_idx];

                        let lq = seqlens_q_vec[b] as usize;
                        let lk = seqlens_k_vec[b] as usize;
                        if lq == 0 || lk == 0 {
                            out_row.fill(0.0);
                            return;
                        }

                        let start_k = cumsum_k[b];
                        let offset = lk as isize - lq as isize;

                        let Some((j0, j1)) =
                            key_range(q_pos, lk, offset, causal, window_left, window_right)
                        else {
                            out_row.fill(0.0);
                            return;
                        };
                        if j0 > j1 {
                            out_row.fill(0.0);
                            return;
                        }

                        let k_head = h / rk;
                        let v_head = h / rv;

                        let slope = slopes.as_ref().map(|s| s[h]).unwrap_or(0.0);
                        let i_k = q_pos as isize + offset;

                        let q_base = (q_idx * hq + h) * d;
                        let q_row = &q_data[q_base..q_base + d];

                        acc.fill(0.0);
                        let mut m = f32::NEG_INFINITY;
                        let mut ssum = 0.0f32;

                        for j in j0..=j1 {
                            let k_base = ((start_k + j) * hk + k_head) * d;
                            let k_row = &k_data[k_base..k_base + d];

                            let mut score = dot_f32(q_row, k_row) * softmax_scale;
                            if slopes.is_some() {
                                score += alibi_bias(slope, i_k, j as isize, causal);
                            }

                            if score > m {
                                let scale_old = (m - score).exp();
                                #[allow(clippy::needless_range_loop)]
                                for t in 0..d {
                                    acc[t] *= scale_old;
                                }
                                ssum *= scale_old;
                                m = score;

                                let v_base = ((start_k + j) * hv + v_head) * d;
                                let v_row = &v_data[v_base..v_base + d];
                                for t in 0..d {
                                    acc[t] += v_row[t];
                                }
                                ssum += 1.0;
                            } else {
                                let w = (score - m).exp();
                                let v_base = ((start_k + j) * hv + v_head) * d;
                                let v_row = &v_data[v_base..v_base + d];
                                for t in 0..d {
                                    acc[t] += v_row[t] * w;
                                }
                                ssum += w;
                            }
                        }

                        let inv = if ssum > 0.0 { 1.0 / ssum } else { 0.0 };
                        for t in 0..d {
                            out_row[t] = acc[t] * inv;
                        }
                    },
                );
            });

            Tensor::from_vec(out, (total_q, hq, d), &Device::Cpu)
        }

        DType::F16 => {
            let (q_g, q_l) = q.storage_and_layout();
            let q_data: &[f16] = match &*q_g {
                Storage::Cpu(cpu) => &cpu.as_slice::<f16>()?[q_l.start_offset()..],
                _ => candle::bail!("q not cpu"),
            };

            let (k_g, k_l) = k.storage_and_layout();
            let k_data: &[f16] = match &*k_g {
                Storage::Cpu(cpu) => &cpu.as_slice::<f16>()?[k_l.start_offset()..],
                _ => candle::bail!("k not cpu"),
            };

            let (v_g, v_l) = v.storage_and_layout();
            let v_data: &[f16] = match &*v_g {
                Storage::Cpu(cpu) => &cpu.as_slice::<f16>()?[v_l.start_offset()..],
                _ => candle::bail!("v not cpu"),
            };

            #[inline(always)]
            fn dot_qf32_kf16(q: &[f32], k: &[f16]) -> f32 {
                let mut s = 0.0f32;
                let mut i = 0usize;
                while i + 8 <= q.len() {
                    s += q[i] * k[i].to_f32()
                        + q[i + 1] * k[i + 1].to_f32()
                        + q[i + 2] * k[i + 2].to_f32()
                        + q[i + 3] * k[i + 3].to_f32()
                        + q[i + 4] * k[i + 4].to_f32()
                        + q[i + 5] * k[i + 5].to_f32()
                        + q[i + 6] * k[i + 6].to_f32()
                        + q[i + 7] * k[i + 7].to_f32();
                    i += 8;
                }
                while i < q.len() {
                    s += q[i] * k[i].to_f32();
                    i += 1;
                }
                s
            }

            let mut out = vec![f16::from_f32(0.0); total_q * hq * d];

            FLASH_ATTN_POOL.install(|| {
                out.par_chunks_mut(d).enumerate().for_each_init(
                    || (vec![0f32; d], vec![0f32; d]),
                    |(q_row_f32, acc), (row, out_row)| {
                        let q_idx = row / hq;
                        let h = row % hq;

                        let b = batch_of_q[q_idx];
                        let q_pos = pos_in_b[q_idx];

                        let lq = seqlens_q_vec[b] as usize;
                        let lk = seqlens_k_vec[b] as usize;
                        if lq == 0 || lk == 0 {
                            out_row.fill(f16::from_f32(0.0));
                            return;
                        }

                        let start_k = cumsum_k[b];
                        let offset = lk as isize - lq as isize;

                        let Some((j0, j1)) =
                            key_range(q_pos, lk, offset, causal, window_left, window_right)
                        else {
                            out_row.fill(f16::from_f32(0.0));
                            return;
                        };
                        if j0 > j1 {
                            out_row.fill(f16::from_f32(0.0));
                            return;
                        }

                        let k_head = h / rk;
                        let v_head = h / rv;

                        let slope = slopes.as_ref().map(|s| s[h]).unwrap_or(0.0);
                        let i_k = q_pos as isize + offset;

                        let q_base = (q_idx * hq + h) * d;
                        for t in 0..d {
                            q_row_f32[t] = q_data[q_base + t].to_f32();
                        }

                        acc.fill(0.0);
                        let mut m = f32::NEG_INFINITY;
                        let mut ssum = 0.0f32;

                        for j in j0..=j1 {
                            let k_base = ((start_k + j) * hk + k_head) * d;
                            let k_row = &k_data[k_base..k_base + d];

                            let mut score = dot_qf32_kf16(q_row_f32, k_row) * softmax_scale;
                            if slopes.is_some() {
                                score += alibi_bias(slope, i_k, j as isize, causal);
                            }

                            if score > m {
                                let scale_old = (m - score).exp();
                                #[allow(clippy::needless_range_loop)]
                                for t in 0..d {
                                    acc[t] *= scale_old;
                                }
                                ssum *= scale_old;
                                m = score;

                                let v_base = ((start_k + j) * hv + v_head) * d;
                                for t in 0..d {
                                    acc[t] += v_data[v_base + t].to_f32();
                                }
                                ssum += 1.0;
                            } else {
                                let w = (score - m).exp();
                                let v_base = ((start_k + j) * hv + v_head) * d;
                                for t in 0..d {
                                    acc[t] += v_data[v_base + t].to_f32() * w;
                                }
                                ssum += w;
                            }
                        }

                        let inv = if ssum > 0.0 { 1.0 / ssum } else { 0.0 };
                        for t in 0..d {
                            out_row[t] = f16::from_f32(acc[t] * inv);
                        }
                    },
                );
            });

            Tensor::from_vec(out, (total_q, hq, d), &Device::Cpu)
        }

        _ => unreachable!("dtype checked above"),
    }
}
