#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Single-batch (B=1) CPU flash attention kernels. B>1 routes to varlen via mod.rs.

use candle::{Device, Result, Storage, Tensor, WithDType};
use rayon::prelude::*;
use std::f32;

use super::dot_f32;
use super::online_softmax::online_softmax_step;

/// Fused softmax(qk^T*scale)v on CPU, B=1 only. Output shape (1, H, S, Dv).
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
    T: WithDType,
{
    let b = q.shape().dims()[0];
    if b != 1 {
        candle::bail!(
            "standard::run_flash_attn_cpu is B=1 only (got B={b}). \
             Multi-batch should be routed through the varlen path."
        );
    }

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
    // Kernel indexes mask as flat 2D (q_pos * kv_len + kv_pos), so force contiguous.
    let mask_cont = mask.map(|m| m.contiguous()).transpose()?;
    let mask_guard = mask_cont.as_ref().map(|m| m.storage_and_layout());
    let mask_data: Option<&[T]> = if let Some((guard, layout)) = &mask_guard {
        if let Storage::Cpu(cpu) = &**guard {
            let data = cpu.as_slice::<T>()?;
            Some(&data[layout.start_offset()..])
        } else {
            return Err(candle::Error::Msg("Expected CPU storage for mask".into()));
        }
    } else {
        None
    };

    let q_stride = q.stride();
    let k_stride = k.stride();
    let v_stride = v.stride();

    let max_bias = max_bias.unwrap_or(0.0);
    let logit_softcap = softcap.unwrap_or(0.0);
    let lean = mask_data.is_none() && max_bias == 0.0 && logit_softcap == 0.0;

    if q.shape().dims()[1] == 1 {
        if lean {
            return flash_attn_decode_lean::<T>(
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
            );
        }
        return flash_attn_decode::<T>(
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
            max_bias,
            logit_softcap,
        );
    }

    if lean {
        return flash_attn_prefill_lean::<T>(
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
        );
    }
    flash_attn_prefill::<T>(
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
        max_bias,
        logit_softcap,
    )
}

#[allow(clippy::too_many_arguments)]
fn flash_attn_decode_lean<T: WithDType>(
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
) -> Result<Tensor> {
    let (h, d) = (qshape[2], qshape[3]);
    let kv_len = kshape[1];
    let k_h = kshape[2];
    let v_h = vshape[2];
    let rk2 = h / k_h;
    let rv2 = h / v_h;
    let dv = d;
    let v_contiguous = vstride[3] == 1;

    let mut out = vec![0f32; h * dv];

    out.par_chunks_mut(dv).enumerate().for_each_init(
        || vec![0f32; dv],
        |acc, (h_i, out_chunk)| {
            let k_head = h_i / rk2;
            let v_head = h_i / rv2;

            let q_base = h_i * qstride[2];
            let q_row = &q_data[q_base..q_base + d];

            acc.fill(0.0);
            let mut m = f32::NEG_INFINITY;
            let mut ssum = 0.0f32;

            for kv_pos in 0..kv_len {
                let k_base = kv_pos * kstride[1] + k_head * kstride[2];
                let k_row = &k_data[k_base..k_base + d];

                let score = dot_f32(q_row, k_row) * scale;

                let v_base = kv_pos * vstride[1] + v_head * vstride[2];

                online_softmax_step(score, &mut m, &mut ssum, acc, |acc, w| {
                    if v_contiguous {
                        let v_row = &v_data[v_base..v_base + dv];
                        for d_i in 0..dv {
                            acc[d_i] += v_row[d_i].to_f64() as f32 * w;
                        }
                    } else {
                        for d_i in 0..dv {
                            acc[d_i] += v_data[v_base + d_i * vstride[3]].to_f64() as f32 * w;
                        }
                    }
                });
            }

            let inv_s = if ssum > 0.0 { 1.0 / ssum } else { 0.0 };
            for (out_v, acc_v) in out_chunk.iter_mut().zip(acc.iter()) {
                *out_v = *acc_v * inv_s;
            }
        },
    );

    Tensor::from_vec(out, (1usize, h, 1usize, dv), &Device::Cpu)
}

#[allow(clippy::too_many_arguments)]
fn flash_attn_decode<T: WithDType>(
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

    out.par_chunks_mut(dv).enumerate().for_each_init(
        || vec![0f32; dv],
        |acc, (h_i, out_chunk)| {
            let slope = if max_bias > 0.0 {
                2.0f32.powf(-max_bias * ((h_i + 1) as f32) / n2 as f32)
            } else {
                1.0
            };

            let k_head = h_i / rk2;
            let v_head = h_i / rv2;

            let q_base = h_i * qstride[2];
            let q_row = &q_data[q_base..q_base + d];

            acc.fill(0.0);
            let mut m = f32::NEG_INFINITY;
            let mut ssum = 0.0f32;

            for kv_pos in 0..kv_len {
                let mv = if let Some(mv_vec) = mask_vec {
                    let mval = mv_vec[kv_pos];
                    slope * mval.to_f64() as f32
                } else {
                    0.0
                };
                if mv == f32::NEG_INFINITY {
                    continue;
                }

                let k_base = kv_pos * kstride[1] + k_head * kstride[2];
                let k_row = &k_data[k_base..k_base + d];

                let mut score = dot_f32(q_row, k_row) * scale_pre;
                if do_softcap {
                    score = logit_softcap * score.tanh();
                }
                score += mv;

                let v_base = kv_pos * vstride[1] + v_head * vstride[2];

                online_softmax_step(score, &mut m, &mut ssum, acc, |acc, w| {
                    if v_contiguous {
                        let v_row = &v_data[v_base..v_base + dv];
                        for d_i in 0..dv {
                            acc[d_i] += v_row[d_i].to_f64() as f32 * w;
                        }
                    } else {
                        for d_i in 0..dv {
                            acc[d_i] += v_data[v_base + d_i * vstride[3]].to_f64() as f32 * w;
                        }
                    }
                });
            }

            let inv_s = if ssum > 0.0 { 1.0 / ssum } else { 0.0 };
            for (out_v, acc_v) in out_chunk.iter_mut().zip(acc.iter()) {
                *out_v = *acc_v * inv_s;
            }
        },
    );

    Tensor::from_vec(out, (1usize, h, 1usize, dv), &Device::Cpu)
}

#[allow(clippy::too_many_arguments)]
fn flash_attn_prefill_lean<T: WithDType>(
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
) -> Result<Tensor> {
    let (q_len, h, d) = (qshape[1], qshape[2], qshape[3]);
    let kv_len = kshape[1];
    let k_h = kshape[2];
    let v_h = vshape[2];
    let rk2 = h / k_h;
    let rv2 = h / v_h;
    let dv = d;
    let v_contiguous = vstride[3] == 1;

    let mut out = vec![0f32; h * q_len * dv];

    out.par_chunks_mut(dv)
        .with_min_len(64)
        .enumerate()
        .for_each_init(
            || vec![0f32; dv],
            |acc, (row_idx, out_chunk)| {
                let h_i = row_idx / q_len;
                let q_pos = row_idx % q_len;

                let k_head = h_i / rk2;
                let v_head = h_i / rv2;

                let q_base = q_pos * qstride[1] + h_i * qstride[2];
                let q_row = &q_data[q_base..q_base + d];

                acc.fill(0.0);
                let mut m = f32::NEG_INFINITY;
                let mut ssum = 0.0f32;

                for kv_pos in 0..kv_len {
                    let k_base = kv_pos * kstride[1] + k_head * kstride[2];
                    let k_row = &k_data[k_base..k_base + d];

                    let score = dot_f32(q_row, k_row) * scale;

                    let v_base = kv_pos * vstride[1] + v_head * vstride[2];

                    online_softmax_step(score, &mut m, &mut ssum, acc, |acc, w| {
                        if v_contiguous {
                            let v_row = &v_data[v_base..v_base + dv];
                            for d_i in 0..dv {
                                acc[d_i] += v_row[d_i].to_f64() as f32 * w;
                            }
                        } else {
                            for d_i in 0..dv {
                                acc[d_i] += v_data[v_base + d_i * vstride[3]].to_f64() as f32 * w;
                            }
                        }
                    });
                }

                let inv_s = if ssum > 0.0 { 1.0 / ssum } else { 0.0 };
                for (out_v, acc_v) in out_chunk.iter_mut().zip(acc.iter()) {
                    *out_v = *acc_v * inv_s;
                }
            },
        );

    Tensor::from_vec(out, (1usize, h, q_len, dv), &Device::Cpu)
}

#[allow(clippy::too_many_arguments)]
fn flash_attn_prefill<T: WithDType>(
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

    out.par_chunks_mut(dv)
        .with_min_len(64)
        .enumerate()
        .for_each_init(
            || vec![0f32; dv],
            |acc, (row_idx, out_chunk)| {
                let h_i = row_idx / q_len;
                let q_pos = row_idx % q_len;

                let slope = if max_bias > 0.0 {
                    2.0f32.powf(-max_bias * ((h_i + 1) as f32) / n2 as f32)
                } else {
                    1.0
                };

                let k_head = h_i / rk2;
                let v_head = h_i / rv2;

                let q_base = q_pos * qstride[1] + h_i * qstride[2];
                let q_row = &q_data[q_base..q_base + d];

                acc.fill(0.0);
                let mut m = f32::NEG_INFINITY;
                let mut ssum = 0.0f32;

                for kv_pos in 0..kv_len {
                    let mv = if let Some(mv_vec) = mask_vec {
                        let mval = mv_vec[q_pos * kv_len + kv_pos];
                        slope * mval.to_f64() as f32
                    } else {
                        0.0
                    };
                    if mv == f32::NEG_INFINITY {
                        continue;
                    }

                    let k_base = kv_pos * kstride[1] + k_head * kstride[2];
                    let k_row = &k_data[k_base..k_base + d];

                    let mut score = dot_f32(q_row, k_row) * scale_pre;
                    if do_softcap {
                        score = logit_softcap * score.tanh();
                    }
                    score += mv;

                    let v_base = kv_pos * vstride[1] + v_head * vstride[2];

                    online_softmax_step(score, &mut m, &mut ssum, acc, |acc, w| {
                        if v_contiguous {
                            let v_row = &v_data[v_base..v_base + dv];
                            for d_i in 0..dv {
                                acc[d_i] += v_row[d_i].to_f64() as f32 * w;
                            }
                        } else {
                            for d_i in 0..dv {
                                acc[d_i] += v_data[v_base + d_i * vstride[3]].to_f64() as f32 * w;
                            }
                        }
                    });
                }

                let inv_s = if ssum > 0.0 { 1.0 / ssum } else { 0.0 };
                for (out_v, acc_v) in out_chunk.iter_mut().zip(acc.iter()) {
                    *out_v = *acc_v * inv_s;
                }
            },
        );

    Tensor::from_vec(out, (1usize, h, q_len, dv), &Device::Cpu)
}
