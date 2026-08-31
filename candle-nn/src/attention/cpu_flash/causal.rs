// Index loops (for t in 0..d) are intentional for SIMD auto-vectorization.
#![allow(clippy::needless_range_loop)]

// Single-batch (B=1) causal attention using loop-bound masking.

use candle::{cpu::kernels::VecOps, DType, Device, Result, Storage, Tensor, WithDType};
use rayon::prelude::*;

use super::dot_f32;
use super::online_softmax::online_softmax_step;

// These utils are mostly for readability.

// Multiply-add a vector into an accumulator.
#[inline(always)]
fn vec_fmadd_scalar(acc: &mut [f32], v: &[f32], w: f32) {
    acc.iter_mut().zip(v).for_each(|(a, e)| *a += e * w);
}

// Scale an accumulator in place
#[inline(always)]
fn vec_scale(acc: &mut [f32], s: f32) {
    acc.iter_mut().for_each(|a| *a *= s);
}

// Add v into acc
#[inline(always)]
fn vec_add(acc: &mut [f32], v: &[f32]) {
    acc.iter_mut().zip(v).for_each(|(a, e)| *a += e);
}

/// Prefetch a cache line for read.
#[inline(always)]
fn prefetch_read(ptr: *const f32) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::asm!("prfm pldl1keep, [{ptr}]", ptr = in(reg) ptr, options(nostack, preserves_flags));
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        let _ = ptr;
    }
}

/// Causal attention with loop-bound masking, **B=1 only**.
///
/// Squeezes batch dim, extracts contiguous slices, dispatches to
/// f32 or generic kernel. The inner kernels operate on raw slices only.
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
    T: WithDType + num_traits::Float,
{
    let b = q.dims()[0];
    if b != 1 {
        candle::bail!(
            "causal::run_causal_attn_cpu is B=1 only (got B={b}). \
             Multi-batch should be routed through the varlen path."
        );
    }

    let q = q.squeeze(0)?.contiguous()?;
    let k = k.squeeze(0)?.contiguous()?;
    let v = v.squeeze(0)?.contiguous()?;

    let (s_q, h_q, d) = q.dims3()?;
    let (s_kv, h_kv, _) = k.dims3()?;
    let (_, h_v, _) = v.dims3()?;

    let max_bias = max_bias.unwrap_or(0.0);
    let softcap = softcap.unwrap_or(0.0);

    if q.dtype() == DType::F32 {
        let (q_g, q_l) = q.storage_and_layout();
        let q_data: &[f32] = match &*q_g {
            Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[q_l.start_offset()..],
            _ => candle::bail!("Expected CPU storage"),
        };
        let (k_g, k_l) = k.storage_and_layout();
        let k_data: &[f32] = match &*k_g {
            Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[k_l.start_offset()..],
            _ => candle::bail!("Expected CPU storage"),
        };
        let (v_g, v_l) = v.storage_and_layout();
        let v_data: &[f32] = match &*v_g {
            Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[v_l.start_offset()..],
            _ => candle::bail!("Expected CPU storage"),
        };

        let result = if s_q == 1 {
            causal_decode_f32(
                q_data,
                k_data,
                v_data,
                h_q,
                h_kv,
                h_v,
                d,
                s_kv,
                softmax_scale,
                max_bias,
                softcap,
            )
        } else {
            causal_prefill_f32(
                q_data,
                k_data,
                v_data,
                s_q,
                h_q,
                h_kv,
                h_v,
                d,
                s_kv,
                softmax_scale,
                kv_offset,
                max_bias,
                softcap,
            )
        };

        result.and_then(|t| t.unsqueeze(0))
    } else {
        let (q_g, q_l) = q.storage_and_layout();
        let q_data: &[T] = match &*q_g {
            Storage::Cpu(cpu) => &cpu.as_slice::<T>()?[q_l.start_offset()..],
            _ => candle::bail!("Expected CPU storage"),
        };
        let (k_g, k_l) = k.storage_and_layout();
        let k_data: &[T] = match &*k_g {
            Storage::Cpu(cpu) => &cpu.as_slice::<T>()?[k_l.start_offset()..],
            _ => candle::bail!("Expected CPU storage"),
        };
        let (v_g, v_l) = v.storage_and_layout();
        let v_data: &[T] = match &*v_g {
            Storage::Cpu(cpu) => &cpu.as_slice::<T>()?[v_l.start_offset()..],
            _ => candle::bail!("Expected CPU storage"),
        };

        let result = if s_q == 1 {
            causal_decode_generic(
                q_data,
                k_data,
                v_data,
                h_q,
                h_kv,
                h_v,
                d,
                s_kv,
                softmax_scale,
                max_bias,
                softcap,
            )
        } else {
            causal_prefill_generic(
                q_data,
                k_data,
                v_data,
                s_q,
                h_q,
                h_kv,
                h_v,
                d,
                s_kv,
                softmax_scale,
                kv_offset,
                max_bias,
                softcap,
            )
        };

        result.and_then(|t| t.unsqueeze(0))
    }
}

// f32 decode (q_len=1).
// Input layout is contiguous (1, S, H, D); we index past the batch dim.
// q[h] starts at h*D; k/v[pos, h] starts at pos*H_kv*D + h*D.
#[allow(clippy::too_many_arguments)]
fn causal_decode_f32(
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    h_q: usize,
    h_kv: usize,
    h_v: usize,
    d: usize,
    kv_len: usize,
    scale: f32,
    max_bias: f32,
    logit_softcap: f32,
) -> Result<Tensor> {
    // Dispatch to branchless fast path when no ALiBi / softcap
    if max_bias == 0.0 && logit_softcap == 0.0 {
        return causal_decode_f32_lean(q_data, k_data, v_data, h_q, h_kv, h_v, d, kv_len, scale);
    }

    let rk = h_q / h_kv;
    let rv = h_q / h_v;
    let n2 = 2_usize.pow((h_q as f32).log2().ceil() as u32);

    let (scale_pre, do_softcap) = if logit_softcap != 0.0 {
        (scale / logit_softcap, true)
    } else {
        (scale, false)
    };

    let k_seq_stride = h_kv * d;
    let v_seq_stride = h_v * d;

    let mut out = vec![0f32; h_q * d];

    let pool = candle::utils::barrier_pool();
    let n_total = pool.n_workers() + 1;
    let mut scratch = vec![0f32; n_total * d];
    let scratch_ptr = scratch.as_mut_ptr() as usize;
    let out_ptr = out.as_mut_ptr() as usize;
    let q_ptr = q_data.as_ptr() as usize;
    let k_ptr = k_data.as_ptr() as usize;
    let v_ptr = v_data.as_ptr() as usize;

    pool.execute(|tid| unsafe {
        let start_h = tid * h_q / n_total;
        let end_h = (tid + 1) * h_q / n_total;
        if start_h >= end_h {
            return;
        }
        let acc = std::slice::from_raw_parts_mut((scratch_ptr as *mut f32).add(tid * d), d);
        let q_data = std::slice::from_raw_parts(q_ptr as *const f32, h_q * d);
        let k_data = std::slice::from_raw_parts(k_ptr as *const f32, kv_len * k_seq_stride);
        let v_data = std::slice::from_raw_parts(v_ptr as *const f32, kv_len * v_seq_stride);

        for h_i in start_h..end_h {
            let out_chunk = std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(h_i * d), d);
            let slope = 2.0f32.powf(-max_bias * ((h_i + 1) as f32) / n2 as f32);
            let k_head_off = (h_i / rk) * d;
            let v_head_off = (h_i / rv) * d;
            let q_row = &q_data[h_i * d..(h_i + 1) * d];

            acc.fill(0.0);
            let mut m = f32::NEG_INFINITY;
            let mut ssum = 0.0f32;

            for kv_pos in 0..kv_len {
                let alibi_bias = slope * (kv_pos as f32 - (kv_len - 1) as f32);
                let k_base = kv_pos * k_seq_stride + k_head_off;
                let k_row = &k_data[k_base..k_base + d];

                if kv_pos + 1 < kv_len {
                    prefetch_read(k_data[k_base + k_seq_stride..].as_ptr());
                }

                let mut score = 0.0f32;
                f32::vec_dot(q_row.as_ptr(), k_row.as_ptr(), &mut score, q_row.len());
                score *= scale_pre;

                if do_softcap {
                    score = logit_softcap * score.tanh();
                }
                score += alibi_bias;

                let v_base = kv_pos * v_seq_stride + v_head_off;
                let v_row = &v_data[v_base..v_base + d];

                if kv_pos + 1 < kv_len {
                    prefetch_read(v_data[v_base + v_seq_stride..].as_ptr());
                }

                online_softmax_step(score, &mut m, &mut ssum, acc, |acc, w| {
                    for t in 0..d {
                        acc[t] += v_row[t] * w;
                    }
                });
            }

            let inv = if ssum > 0.0 { 1.0 / ssum } else { 0.0 };
            out_chunk.fill(0.0);
            vec_fmadd_scalar(out_chunk, acc, inv);
        }
    });

    Tensor::from_vec(out, (h_q, 1usize, d), &Device::Cpu)
}

/// Lean decode: no ALiBi, no softcap. Zero branches in the inner KV loop.
/// This is the hot path for Qwen3, SmolLM3, and most standard LLMs.
#[allow(clippy::too_many_arguments)]
fn causal_decode_f32_lean(
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    h_q: usize,
    h_kv: usize,
    h_v: usize,
    d: usize,
    kv_len: usize,
    scale: f32,
) -> Result<Tensor> {
    let rk = h_q / h_kv;
    let rv = h_q / h_v;
    let k_seq_stride = h_kv * d;
    let v_seq_stride = h_v * d;

    let mut out = vec![0f32; h_q * d];

    let pool = candle::utils::barrier_pool();
    let n_total = pool.n_workers() + 1;
    let mut scratch = vec![0f32; n_total * d];
    let scratch_ptr = scratch.as_mut_ptr() as usize;
    let out_ptr = out.as_mut_ptr() as usize;
    let q_ptr = q_data.as_ptr() as usize;
    let k_ptr = k_data.as_ptr() as usize;
    let v_ptr = v_data.as_ptr() as usize;

    pool.execute(|tid| unsafe {
        let start_h = tid * h_q / n_total;
        let end_h = (tid + 1) * h_q / n_total;
        if start_h >= end_h {
            return;
        }
        let acc = std::slice::from_raw_parts_mut((scratch_ptr as *mut f32).add(tid * d), d);
        let q_data = std::slice::from_raw_parts(q_ptr as *const f32, h_q * d);
        let k_data = std::slice::from_raw_parts(k_ptr as *const f32, kv_len * k_seq_stride);
        let v_data = std::slice::from_raw_parts(v_ptr as *const f32, kv_len * v_seq_stride);

        for h_i in start_h..end_h {
            let out_chunk = std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(h_i * d), d);
            let k_head_off = (h_i / rk) * d;
            let v_head_off = (h_i / rv) * d;
            let q_row = &q_data[h_i * d..(h_i + 1) * d];

            acc.fill(0.0);
            let mut m = f32::NEG_INFINITY;
            let mut ssum = 0.0f32;

            for kv_pos in 0..kv_len {
                let k_base = kv_pos * k_seq_stride + k_head_off;
                let k_row = &k_data[k_base..k_base + d];

                if kv_pos + 1 < kv_len {
                    prefetch_read(k_data[k_base + k_seq_stride..].as_ptr());
                }

                let mut score = 0.0f32;
                f32::vec_dot(q_row.as_ptr(), k_row.as_ptr(), &mut score, q_row.len());
                score *= scale;

                let v_base = kv_pos * v_seq_stride + v_head_off;
                let v_row = &v_data[v_base..v_base + d];

                if kv_pos + 1 < kv_len {
                    prefetch_read(v_data[v_base + v_seq_stride..].as_ptr());
                }

                online_softmax_step(score, &mut m, &mut ssum, acc, |acc, w| {
                    for t in 0..d {
                        acc[t] += v_row[t] * w;
                    }
                });
            }

            let inv = if ssum > 0.0 { 1.0 / ssum } else { 0.0 };
            out_chunk.fill(0.0);
            vec_fmadd_scalar(out_chunk, acc, inv);
        }
    });

    Tensor::from_vec(out, (h_q, 1usize, d), &Device::Cpu)
}

// Interleaved KV decode.
// KV layout is contiguous (S, H_kv, 2*D) with K=[..,:D] and V=[..,D:2D],
// so one base pointer per position and one prefetch covers both.

/// Decode with interleaved KV cache. No ALiBi, no softcap.
#[allow(clippy::too_many_arguments)]
pub fn causal_decode_f32_interleaved(
    q_data: &[f32],
    kv_data: &[f32],
    h_q: usize,
    h_kv: usize,
    d: usize,
    kv_len: usize,
    scale: f32,
) -> Result<Tensor> {
    let rk = h_q / h_kv;
    if rk == 2 {
        return causal_decode_f32_interleaved_gqa2(q_data, kv_data, h_kv, d, kv_len, scale);
    }
    let kv_head_stride = 2 * d;
    let kv_seq_stride = h_kv * kv_head_stride;

    let mut out = vec![0f32; h_q * d];

    let pool = candle::utils::barrier_pool();
    let n_total = pool.n_workers() + 1;
    let mut scratch = vec![0f32; n_total * d];
    let scratch_ptr = scratch.as_mut_ptr() as usize;
    let out_ptr = out.as_mut_ptr() as usize;
    let q_ptr = q_data.as_ptr() as usize;
    let kv_ptr = kv_data.as_ptr() as usize;

    pool.execute(|tid| unsafe {
        let start_h = tid * h_q / n_total;
        let end_h = (tid + 1) * h_q / n_total;
        if start_h >= end_h {
            return;
        }
        let acc = std::slice::from_raw_parts_mut((scratch_ptr as *mut f32).add(tid * d), d);
        let q_data = std::slice::from_raw_parts(q_ptr as *const f32, h_q * d);
        let kv_data = std::slice::from_raw_parts(kv_ptr as *const f32, kv_len * kv_seq_stride);

        for h_i in start_h..end_h {
            let out_chunk = std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(h_i * d), d);
            let kv_head_off = (h_i / rk) * kv_head_stride;
            let q_row = &q_data[h_i * d..(h_i + 1) * d];

            acc.fill(0.0);
            let mut m = f32::NEG_INFINITY;
            let mut ssum = 0.0f32;

            for kv_pos in 0..kv_len {
                let kv_base = kv_pos * kv_seq_stride + kv_head_off;
                let k_row = &kv_data[kv_base..kv_base + d];
                let v_row = &kv_data[kv_base + d..kv_base + 2 * d];

                if kv_pos + 1 < kv_len {
                    prefetch_read(kv_data[kv_base + kv_seq_stride..].as_ptr());
                }

                let mut score = 0.0f32;
                f32::vec_dot(q_row.as_ptr(), k_row.as_ptr(), &mut score, q_row.len());
                score *= scale;

                online_softmax_step(score, &mut m, &mut ssum, acc, |acc, w| {
                    for t in 0..d {
                        acc[t] += v_row[t] * w;
                    }
                });
            }

            let inv = if ssum > 0.0 { 1.0 / ssum } else { 0.0 };
            out_chunk.fill(0.0);
            vec_fmadd_scalar(out_chunk, acc, inv);
        }
    });

    Tensor::from_vec(out, (h_q, 1usize, d), &Device::Cpu)
}

/// GQA-2 specialization: each pair of query heads (h*2, h*2+1) shares a KV head.
/// Loads K and V once per position, accumulates into both heads simultaneously.
fn causal_decode_f32_interleaved_gqa2(
    q_data: &[f32],
    kv_data: &[f32],
    h_kv: usize,
    d: usize,
    kv_len: usize,
    scale: f32,
) -> Result<Tensor> {
    let h_q = h_kv * 2;
    let kv_head_stride = 2 * d;
    let kv_seq_stride = h_kv * kv_head_stride;

    let mut out = vec![0f32; h_q * d];

    let pool = candle::utils::barrier_pool();
    let n_total = pool.n_workers() + 1;
    let mut scratch = vec![0f32; n_total * 2 * d];
    let scratch_ptr = scratch.as_mut_ptr() as usize;
    let out_ptr = out.as_mut_ptr() as usize;
    let q_ptr = q_data.as_ptr() as usize;
    let kv_ptr = kv_data.as_ptr() as usize;

    pool.execute(|tid| unsafe {
        let start_h = tid * h_kv / n_total;
        let end_h = (tid + 1) * h_kv / n_total;
        if start_h >= end_h {
            return;
        }

        let acc0 = std::slice::from_raw_parts_mut((scratch_ptr as *mut f32).add(tid * 2 * d), d);
        let acc1 =
            std::slice::from_raw_parts_mut((scratch_ptr as *mut f32).add(tid * 2 * d + d), d);
        let q_data = std::slice::from_raw_parts(q_ptr as *const f32, h_q * d);
        let kv_data = std::slice::from_raw_parts(kv_ptr as *const f32, kv_len * kv_seq_stride);

        for kv_h in start_h..end_h {
            let out0 = std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(kv_h * 2 * d), d);
            let out1 =
                std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(kv_h * 2 * d + d), d);
            let kv_head_off = kv_h * kv_head_stride;
            let q0 = &q_data[kv_h * 2 * d..(kv_h * 2 + 1) * d];
            let q1 = &q_data[(kv_h * 2 + 1) * d..(kv_h * 2 + 2) * d];

            acc0.fill(0.0);
            acc1.fill(0.0);
            let mut m0 = f32::NEG_INFINITY;
            let mut m1 = f32::NEG_INFINITY;
            let mut s0 = 0.0f32;
            let mut s1 = 0.0f32;

            for kv_pos in 0..kv_len {
                let kv_base = kv_pos * kv_seq_stride + kv_head_off;
                let k_row = &kv_data[kv_base..kv_base + d];
                let v_row = &kv_data[kv_base + d..kv_base + 2 * d];

                if kv_pos + 1 < kv_len {
                    prefetch_read(kv_data[kv_base + kv_seq_stride..].as_ptr());
                }

                let mut sc0 = 0.0f32;
                let mut sc1 = 0.0f32;
                f32::vec_dot(q0.as_ptr(), k_row.as_ptr(), &mut sc0, q0.len());
                f32::vec_dot(q1.as_ptr(), k_row.as_ptr(), &mut sc1, q1.len());
                sc0 *= scale;
                sc1 *= scale;

                if sc0 > m0 {
                    let so = f32::exp(m0 - sc0);
                    vec_scale(acc0, so);
                    s0 *= so;
                    m0 = sc0;
                    vec_add(acc0, v_row);
                    s0 += 1.0;
                } else {
                    let w = f32::exp(sc0 - m0);
                    vec_fmadd_scalar(acc0, v_row, w);
                    s0 += w;
                }

                if sc1 > m1 {
                    let so = f32::exp(m1 - sc1);
                    vec_scale(acc1, so);
                    s1 *= so;
                    m1 = sc1;
                    vec_add(acc1, v_row);
                    s1 += 1.0;
                } else {
                    let w = f32::exp(sc1 - m1);
                    vec_fmadd_scalar(acc1, v_row, w);
                    s1 += w;
                }
            }

            let inv0 = if s0 > 0.0 { 1.0 / s0 } else { 0.0 };
            let inv1 = if s1 > 0.0 { 1.0 / s1 } else { 0.0 };
            out0.fill(0.0);
            out1.fill(0.0);
            vec_fmadd_scalar(out0, acc0, inv0);
            vec_fmadd_scalar(out1, acc1, inv1);
        }
    });

    Tensor::from_vec(out, (h_q, 1usize, d), &Device::Cpu)
}

// f32 prefill (q_len > 1)

#[allow(clippy::too_many_arguments)]
fn causal_prefill_f32(
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    s_q: usize,
    h_q: usize,
    h_kv: usize,
    h_v: usize,
    d: usize,
    kv_len: usize,
    scale: f32,
    kv_offset: usize,
    max_bias: f32,
    logit_softcap: f32,
) -> Result<Tensor> {
    // Dispatch to lean path for common case
    if max_bias == 0.0 && logit_softcap == 0.0 {
        return causal_prefill_f32_lean(
            q_data, k_data, v_data, s_q, h_q, h_kv, h_v, d, kv_len, scale, kv_offset,
        );
    }

    let rk = h_q / h_kv;
    let rv = h_q / h_v;
    let n2 = 2_usize.pow((h_q as f32).log2().ceil() as u32);

    let (scale_pre, do_softcap) = if logit_softcap != 0.0 {
        (scale / logit_softcap, true)
    } else {
        (scale, false)
    };

    let q_seq_stride = h_q * d;
    let k_seq_stride = h_kv * d;
    let v_seq_stride = h_v * d;

    let mut out = vec![0f32; h_q * s_q * d];

    let pool = candle::utils::barrier_pool();
    let n_total = pool.n_workers() + 1;
    let n_rows = h_q * s_q;
    let mut scratch = vec![0f32; n_total * d];
    let scratch_ptr = scratch.as_mut_ptr() as usize;
    let out_ptr = out.as_mut_ptr() as usize;
    let q_ptr = q_data.as_ptr() as usize;
    let k_ptr = k_data.as_ptr() as usize;
    let v_ptr = v_data.as_ptr() as usize;

    pool.execute(|tid| unsafe {
        let start_row = tid * n_rows / n_total;
        let end_row = (tid + 1) * n_rows / n_total;
        if start_row >= end_row {
            return;
        }
        let acc = std::slice::from_raw_parts_mut((scratch_ptr as *mut f32).add(tid * d), d);
        let q_data = std::slice::from_raw_parts(q_ptr as *const f32, s_q * h_q * d);
        let k_data = std::slice::from_raw_parts(k_ptr as *const f32, kv_len * k_seq_stride);
        let v_data = std::slice::from_raw_parts(v_ptr as *const f32, kv_len * v_seq_stride);

        for row_idx in start_row..end_row {
            let out_chunk =
                std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(row_idx * d), d);
            let h_i = row_idx / s_q;
            let q_pos = row_idx % s_q;

            let slope = if max_bias > 0.0 {
                2.0f32.powf(-max_bias * ((h_i + 1) as f32) / n2 as f32)
            } else {
                0.0
            };

            let k_head_off = (h_i / rk) * d;
            let v_head_off = (h_i / rv) * d;

            let q_base = q_pos * q_seq_stride + h_i * d;
            let q_row = &q_data[q_base..q_base + d];

            acc.fill(0.0);
            let mut m = f32::NEG_INFINITY;
            let mut ssum = 0.0f32;

            let kv_end = (q_pos + kv_offset + 1).min(kv_len);

            for kv_pos in 0..kv_end {
                let alibi_bias = if max_bias > 0.0 {
                    slope * (kv_pos as i64 - (q_pos + kv_offset) as i64) as f32
                } else {
                    0.0
                };

                let k_base = kv_pos * k_seq_stride + k_head_off;
                let k_row = &k_data[k_base..k_base + d];

                if kv_pos + 1 < kv_end {
                    prefetch_read(k_data[k_base + k_seq_stride..].as_ptr());
                }

                let mut score = 0.0f32;
                f32::vec_dot(q_row.as_ptr(), k_row.as_ptr(), &mut score, q_row.len());
                score *= scale_pre;

                if do_softcap {
                    score = logit_softcap * score.tanh();
                }
                score += alibi_bias;

                let v_base = kv_pos * v_seq_stride + v_head_off;
                let v_row = &v_data[v_base..v_base + d];

                if kv_pos + 1 < kv_end {
                    prefetch_read(v_data[v_base + v_seq_stride..].as_ptr());
                }

                online_softmax_step(score, &mut m, &mut ssum, acc, |acc, w| {
                    for t in 0..d {
                        acc[t] += v_row[t] * w;
                    }
                });
            }

            let inv = if ssum > 0.0 { 1.0 / ssum } else { 0.0 };
            out_chunk.fill(0.0);
            vec_fmadd_scalar(out_chunk, acc, inv);
        }
    });

    Tensor::from_vec(out, (h_q, s_q, d), &Device::Cpu)
}

/// Lean prefill: no ALiBi, no softcap.
#[allow(clippy::too_many_arguments)]
fn causal_prefill_f32_lean(
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
    s_q: usize,
    h_q: usize,
    h_kv: usize,
    h_v: usize,
    d: usize,
    kv_len: usize,
    scale: f32,
    kv_offset: usize,
) -> Result<Tensor> {
    let rk = h_q / h_kv;
    let rv = h_q / h_v;
    let q_seq_stride = h_q * d;
    let k_seq_stride = h_kv * d;
    let v_seq_stride = h_v * d;

    let mut out = vec![0f32; h_q * s_q * d];

    let pool = candle::utils::barrier_pool();
    let n_total = pool.n_workers() + 1;
    let n_rows = h_q * s_q;
    let mut scratch = vec![0f32; n_total * d];
    let scratch_ptr = scratch.as_mut_ptr() as usize;
    let out_ptr = out.as_mut_ptr() as usize;
    let q_ptr = q_data.as_ptr() as usize;
    let k_ptr = k_data.as_ptr() as usize;
    let v_ptr = v_data.as_ptr() as usize;

    pool.execute(|tid| unsafe {
        let start_row = tid * n_rows / n_total;
        let end_row = (tid + 1) * n_rows / n_total;
        if start_row >= end_row {
            return;
        }
        let acc = std::slice::from_raw_parts_mut((scratch_ptr as *mut f32).add(tid * d), d);
        let q_data = std::slice::from_raw_parts(q_ptr as *const f32, s_q * h_q * d);
        let k_data = std::slice::from_raw_parts(k_ptr as *const f32, kv_len * k_seq_stride);
        let v_data = std::slice::from_raw_parts(v_ptr as *const f32, kv_len * v_seq_stride);

        for row_idx in start_row..end_row {
            let out_chunk =
                std::slice::from_raw_parts_mut((out_ptr as *mut f32).add(row_idx * d), d);
            let h_i = row_idx / s_q;
            let q_pos = row_idx % s_q;

            let k_head_off = (h_i / rk) * d;
            let v_head_off = (h_i / rv) * d;

            let q_base = q_pos * q_seq_stride + h_i * d;
            let q_row = &q_data[q_base..q_base + d];

            acc.fill(0.0);
            let mut m = f32::NEG_INFINITY;
            let mut ssum = 0.0f32;

            let kv_end = (q_pos + kv_offset + 1).min(kv_len);

            for kv_pos in 0..kv_end {
                let k_base = kv_pos * k_seq_stride + k_head_off;
                let k_row = &k_data[k_base..k_base + d];

                if kv_pos + 1 < kv_end {
                    prefetch_read(k_data[k_base + k_seq_stride..].as_ptr());
                }

                let mut score = 0.0f32;
                f32::vec_dot(q_row.as_ptr(), k_row.as_ptr(), &mut score, q_row.len());
                score *= scale;

                let v_base = kv_pos * v_seq_stride + v_head_off;
                let v_row = &v_data[v_base..v_base + d];

                if kv_pos + 1 < kv_end {
                    prefetch_read(v_data[v_base + v_seq_stride..].as_ptr());
                }

                online_softmax_step(score, &mut m, &mut ssum, acc, |acc, w| {
                    for t in 0..d {
                        acc[t] += v_row[t] * w;
                    }
                });
            }

            let inv = if ssum > 0.0 { 1.0 / ssum } else { 0.0 };
            out_chunk.fill(0.0);
            vec_fmadd_scalar(out_chunk, acc, inv);
        }
    });

    Tensor::from_vec(out, (h_q, s_q, d), &Device::Cpu)
}

// Generic fallback (non-f32)

#[allow(clippy::too_many_arguments)]
fn causal_decode_generic<T: WithDType + num_traits::Float>(
    q_data: &[T],
    k_data: &[T],
    v_data: &[T],
    h_q: usize,
    h_kv: usize,
    h_v: usize,
    d: usize,
    kv_len: usize,
    scale: f32,
    max_bias: f32,
    logit_softcap: f32,
) -> Result<Tensor> {
    let rk = h_q / h_kv;
    let rv = h_q / h_v;
    let n2 = 2_usize.pow((h_q as f32).log2().ceil() as u32);
    let (scale_pre, do_softcap) = if logit_softcap != 0.0 {
        (scale / logit_softcap, true)
    } else {
        (scale, false)
    };

    let k_seq_stride = h_kv * d;
    let v_seq_stride = h_v * d;

    let mut out = vec![0f32; h_q * d];

    out.par_chunks_mut(d).enumerate().for_each_init(
        || vec![0f32; d],
        |acc, (h_i, out_chunk)| {
            let slope = if max_bias > 0.0 {
                2.0f32.powf(-max_bias * ((h_i + 1) as f32) / n2 as f32)
            } else {
                0.0
            };
            let k_head_off = (h_i / rk) * d;
            let v_head_off = (h_i / rv) * d;
            let q_row = &q_data[h_i * d..(h_i + 1) * d];

            acc.fill(0.0);
            let mut m = f32::NEG_INFINITY;
            let mut ssum = 0.0f32;

            for kv_pos in 0..kv_len {
                let alibi_bias = if max_bias > 0.0 {
                    slope * (kv_pos as f32 - (kv_len - 1) as f32)
                } else {
                    0.0
                };
                let k_base = kv_pos * k_seq_stride + k_head_off;
                let k_row = &k_data[k_base..k_base + d];
                let mut s_val_t = T::zero();
                unsafe { T::vec_dot(q_row.as_ptr(), k_row.as_ptr(), &mut s_val_t, q_row.len()) }
                let mut s_val = s_val_t.to_f32().unwrap_or(0.0);
                s_val *= scale_pre;
                if do_softcap {
                    s_val = logit_softcap * s_val.tanh();
                }
                s_val += alibi_bias;

                let v_base = kv_pos * v_seq_stride + v_head_off;
                let v_row = &v_data[v_base..v_base + d];
                online_softmax_step(s_val, &mut m, &mut ssum, acc, |acc, w| {
                    for t in 0..d {
                        acc[t] += v_row[t].to_f32().unwrap_or(0.0) * w;
                    }
                });
            }
            let inv = if ssum > 0.0 { 1.0 / ssum } else { 0.0 };
            for t in 0..d {
                out_chunk[t] = acc[t] * inv;
            }
        },
    );

    Tensor::from_vec(out, (h_q, 1usize, d), &Device::Cpu)
}

#[allow(clippy::too_many_arguments)]
fn causal_prefill_generic<T: WithDType>(
    q_data: &[T],
    k_data: &[T],
    v_data: &[T],
    s_q: usize,
    h_q: usize,
    h_kv: usize,
    h_v: usize,
    d: usize,
    kv_len: usize,
    scale: f32,
    kv_offset: usize,
    max_bias: f32,
    logit_softcap: f32,
) -> Result<Tensor> {
    let rk = h_q / h_kv;
    let rv = h_q / h_v;
    let n2 = 2_usize.pow((h_q as f32).log2().ceil() as u32);
    let (scale_pre, do_softcap) = if logit_softcap != 0.0 {
        (scale / logit_softcap, true)
    } else {
        (scale, false)
    };

    let q_seq_stride = h_q * d;
    let k_seq_stride = h_kv * d;
    let v_seq_stride = h_v * d;

    let mut out = vec![0f32; h_q * s_q * d];

    out.par_chunks_mut(d)
        .with_min_len(64)
        .enumerate()
        .for_each_init(
            || vec![0f32; d],
            |acc, (row_idx, out_chunk)| {
                let h_i = row_idx / s_q;
                let q_pos = row_idx % s_q;
                let slope = if max_bias > 0.0 {
                    2.0f32.powf(-max_bias * ((h_i + 1) as f32) / n2 as f32)
                } else {
                    0.0
                };
                let k_head_off = (h_i / rk) * d;
                let v_head_off = (h_i / rv) * d;
                let q_base = q_pos * q_seq_stride + h_i * d;
                let q_row = &q_data[q_base..q_base + d];

                acc.fill(0.0);
                let mut m = f32::NEG_INFINITY;
                let mut ssum = 0.0f32;
                let kv_end = (q_pos + kv_offset + 1).min(kv_len);

                for kv_pos in 0..kv_end {
                    let alibi_bias = if max_bias > 0.0 {
                        slope * (kv_pos as i64 - (q_pos + kv_offset) as i64) as f32
                    } else {
                        0.0
                    };
                    let k_base = kv_pos * k_seq_stride + k_head_off;
                    let k_row = &k_data[k_base..k_base + d];
                    let mut s_val = dot_f32(q_row, k_row);
                    s_val *= scale_pre;
                    if do_softcap {
                        s_val = logit_softcap * s_val.tanh();
                    }
                    s_val += alibi_bias;

                    let v_base = kv_pos * v_seq_stride + v_head_off;
                    let v_row = &v_data[v_base..v_base + d];
                    online_softmax_step(s_val, &mut m, &mut ssum, acc, |acc, w| {
                        for t in 0..d {
                            acc[t] += v_row[t].to_f64() as f32 * w;
                        }
                    });
                }
                let inv = if ssum > 0.0 { 1.0 / ssum } else { 0.0 };
                for t in 0..d {
                    out_chunk[t] = acc[t] * inv;
                }
            },
        );

    Tensor::from_vec(out, (h_q, s_q, d), &Device::Cpu)
}
