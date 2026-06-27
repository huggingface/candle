// Index loops (for t in 0..d) are intentional for SIMD auto-vectorization.
#![allow(clippy::needless_range_loop)]

// Single-batch (B=1) causal attention using loop-bound masking.

use candle::{cpu::kernels::VecOps, DType, Device, Result, Storage, Tensor, WithDType};
use rayon::prelude::*;

use super::dot_f32;
use super::online_softmax::online_softmax_step;
// f16-KV CPU flash decode helpers. The dot fn is feature-selected: f16-attn-dot
// uses a native f16.f16 dot, otherwise K is widened in-register (dot_f32_f16).
use super::axpy_f16;
#[cfg(feature = "f16-attn-dot")]
use super::dot_f16_f16;
#[cfg(not(feature = "f16-attn-dot"))]
use super::dot_f32_f16;

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

// Prefetch a cache line for read. Generic so f32 and f16-KV callers share it.
#[inline(always)]
fn prefetch_read<T>(ptr: *const T) {
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

// Opt-in (CANDLE_VEC_SOFTMAX_EXP=1) NEON polynomial exp for softmax, replacing scalar
// libm expf (~1e-6, normalized out). Default OFF to stay bit-reproducible.
#[cfg(target_arch = "aarch64")]
static VEC_SOFTMAX_EXP: std::sync::LazyLock<bool> =
    std::sync::LazyLock::new(|| std::env::var("CANDLE_VEC_SOFTMAX_EXP").is_ok());

// exp(x) via range reduction x = n*ln2 + r, exp(x) = 2^n * poly(r). Scalar form so the
// vector body and the <4 tail share one approximation.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn poly_exp_scalar(x: f32) -> f32 {
    let x = x.clamp(-87.0, 88.0);
    let n = (x * std::f32::consts::LOG2_E).round();
    let r = x - n * std::f32::consts::LN_2;
    let mut p = 1.0 / 720.0;
    p = p * r + 1.0 / 120.0;
    p = p * r + 1.0 / 24.0;
    p = p * r + 1.0 / 6.0;
    p = p * r + 0.5;
    p = p * r + 1.0;
    p = p * r + 1.0;
    p * f32::from_bits((((n as i32) + 127) << 23) as u32)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn vec_exp_sub_sum_neon(s: &mut [f32], mr: f32) -> f32 {
    use std::arch::aarch64::*;
    #[inline(always)]
    unsafe fn vexpq(x: float32x4_t) -> float32x4_t {
        let x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.0)), vdupq_n_f32(88.0));
        let n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(std::f32::consts::LOG2_E)));
        let r = vfmsq_f32(x, n, vdupq_n_f32(std::f32::consts::LN_2));
        let mut p = vdupq_n_f32(1.0 / 720.0);
        p = vfmaq_f32(vdupq_n_f32(1.0 / 120.0), p, r);
        p = vfmaq_f32(vdupq_n_f32(1.0 / 24.0), p, r);
        p = vfmaq_f32(vdupq_n_f32(1.0 / 6.0), p, r);
        p = vfmaq_f32(vdupq_n_f32(0.5), p, r);
        p = vfmaq_f32(vdupq_n_f32(1.0), p, r);
        p = vfmaq_f32(vdupq_n_f32(1.0), p, r);
        let bits = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
        vmulq_f32(p, vreinterpretq_f32_s32(bits))
    }
    let vmr = vdupq_n_f32(mr);
    let mut vsum = vdupq_n_f32(0.0);
    let n = s.len();
    let p = s.as_mut_ptr();
    let mut i = 0;
    while i + 4 <= n {
        let e = vexpq(vsubq_f32(vld1q_f32(p.add(i)), vmr));
        vst1q_f32(p.add(i), e);
        vsum = vaddq_f32(vsum, e);
        i += 4;
    }
    let mut sum = vaddvq_f32(vsum);
    while i < n {
        let e = poly_exp_scalar(*p.add(i) - mr);
        *p.add(i) = e;
        sum += e;
        i += 1;
    }
    sum
}

// For each x in s, set x = exp(x - mr) and return the sum (softmax inner loop).
#[inline(always)]
fn exp_sub_sum(s: &mut [f32], mr: f32) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if *VEC_SOFTMAX_EXP {
        return unsafe { vec_exp_sub_sum_neon(s, mr) };
    }
    let mut sum = 0.0f32;
    for x in s.iter_mut() {
        let e = (*x - mr).exp();
        *x = e;
        sum += e;
    }
    sum
}

// f16-KV interleaved causal decode (q_len=1). Reads the head-major f16 KV cache one
// head's stream at a time - half the bytes of an f32 cache.
#[allow(clippy::too_many_arguments)]
pub fn causal_decode_f16kv_interleaved(
    q_data: &[f32],
    kv_data: &[half::f16],
    head_stride: usize,
    h_q: usize,
    h_kv: usize,
    d: usize,
    kv_len: usize,
    scale: f32,
) -> Result<Tensor> {
    let rk = h_q / h_kv;

    let mut out = vec![0f32; h_q * d];

    // One task per kv head, all rk GQA query heads in one pass so each contiguous
    // K/V row is read from memory once, not once per query head.
    let process =
        |kv_h: usize, out_chunk: &mut [f32], acc: &mut [f32], m: &mut [f32], ssum: &mut [f32]| {
            let head_base = kv_h * head_stride;

            acc.fill(0.0);
            m.fill(f32::NEG_INFINITY);
            ssum.fill(0.0);

            // f16-attn-dot: narrow this kv-head's rk query rows to f16 once (amortized
            // over kv_len), so the inner score is a pure f16.f16 dot. Off by default.
            #[cfg(feature = "f16-attn-dot")]
            let q_f16: Vec<half::f16> = {
                let base = kv_h * rk * d;
                q_data[base..base + rk * d]
                    .iter()
                    .map(|&x| half::f16::from_f32(x))
                    .collect()
            };

            for kv_pos in 0..kv_len {
                // K and V share a base pointer (adjacent in memory)
                let kv_base = head_base + kv_pos * 2 * d;
                let k_row = &kv_data[kv_base..kv_base + d];
                let v_row = &kv_data[kv_base + d..kv_base + 2 * d];

                // One prefetch loads both next K and V
                if kv_pos + 1 < kv_len {
                    prefetch_read(kv_data[kv_base + 2 * d..].as_ptr());
                }

                for r in 0..rk {
                    let h_i = kv_h * rk + r;
                    #[cfg(not(feature = "f16-attn-dot"))]
                    let score = {
                        let q_row = &q_data[h_i * d..(h_i + 1) * d];
                        dot_f32_f16(q_row, k_row) * scale
                    };
                    #[cfg(feature = "f16-attn-dot")]
                    let score = {
                        let _ = h_i;
                        dot_f16_f16(&q_f16[r * d..(r + 1) * d], k_row) * scale
                    };
                    let acc_r = &mut acc[r * d..(r + 1) * d];
                    online_softmax_step(score, &mut m[r], &mut ssum[r], acc_r, |acc, w| {
                        axpy_f16(acc, v_row, w);
                    });
                }
            }

            for r in 0..rk {
                let inv = if ssum[r] > 0.0 { 1.0 / ssum[r] } else { 0.0 };
                let acc_r = &acc[r * d..(r + 1) * d];
                let out_r = &mut out_chunk[r * d..(r + 1) * d];
                for t in 0..d {
                    out_r[t] = acc_r[t] * inv;
                }
            }
        };

    // Contiguous per-thread partition of the h_kv kv-heads on the barrier pool (vs rayon's
    // par_chunks, which thrashed shared cache on N1). 0-worker pool runs f(0) inline.
    struct OutPtr(*mut f32);
    unsafe impl Sync for OutPtr {}
    let optr = OutPtr(out.as_mut_ptr());
    let pool = candle::utils::barrier_pool();
    let n_total = pool.n_workers() + 1;
    let hpt = h_kv.div_ceil(n_total);
    pool.execute(|tid| {
        let p = &optr; // capture the whole OutPtr (Sync), not the raw `.0` field
        let start = tid * hpt;
        if start < h_kv {
            let end = h_kv.min((tid + 1) * hpt);
            let mut acc = vec![0f32; rk * d];
            let mut m = vec![0f32; rk];
            let mut ssum = vec![0f32; rk];
            for kv_h in start..end {
                let out_chunk =
                    unsafe { std::slice::from_raw_parts_mut(p.0.add(kv_h * rk * d), rk * d) };
                process(kv_h, out_chunk, &mut acc, &mut m, &mut ssum);
            }
        }
    });

    Tensor::from_vec(out, (h_q, 1usize, d), &Device::Cpu)
}

// Causal prefill over the f16 head-major interleaved KV cache (same layout decode reads).
// One task per (kv head, query block), rk GQA heads sharing every K/V row, KV-blocked
// softmax; QK widens f16 K in-register, PV accumulator stays f32.
#[allow(clippy::too_many_arguments)]
pub fn causal_prefill_f16kv_headmajor(
    q_data: &[f32],
    kv_data: &[half::f16],
    head_stride: usize,
    s_q: usize,
    h_q: usize,
    h_kv: usize,
    d: usize,
    kv_len: usize,
    scale: f32,
    kv_offset: usize,
) -> Result<Tensor> {
    let rk = h_q / h_kv;
    let q_seq_stride = h_q * d;

    const KB: usize = 128;
    const QB: usize = 32;
    let n_qblocks = s_q.div_ceil(QB);

    let mut out = vec![0f32; h_q * s_q * d];
    struct OutPtr(*mut f32);
    unsafe impl Sync for OutPtr {}
    let out_ptr = OutPtr(out.as_mut_ptr());

    // Contiguous per-thread partition of (kv-head, q-block) tasks on the barrier pool.
    // 0-worker pool runs f(0) inline.
    let n_tasks = h_kv * n_qblocks;
    let pool = candle::utils::barrier_pool();
    let n_total = pool.n_workers() + 1;
    let tpt = n_tasks.div_ceil(n_total);
    pool.execute(|tid| {
        let t_start = tid * tpt;
        if t_start >= n_tasks {
            return;
        }
        let t_end = n_tasks.min((tid + 1) * tpt);
        let mut acc = vec![0f32; rk * d];
        let mut w = vec![0f32; rk * KB];
        let mut m = vec![f32::NEG_INFINITY; rk];
        let mut ssum = vec![0f32; rk];
        for task in t_start..t_end {
            let kv_h = task / n_qblocks;
            let q0 = (task % n_qblocks) * QB;
            let q1 = (q0 + QB).min(s_q);
            let head_base = kv_h * head_stride;
            let p = &out_ptr;

            // f16-attn-dot: narrow this q block's rows to f16 so the QK dot is a pure
            // f16.f16 FMLA. Default keeps Q in f32 and widens K in-register.
            #[cfg(feature = "f16-attn-dot")]
            let mut qf16 = vec![half::f16::ZERO; rk * d];

            for q_pos in q0..q1 {
                let q_pos_base = q_pos * q_seq_stride + kv_h * rk * d;
                #[cfg(feature = "f16-attn-dot")]
                for (o, &x) in qf16
                    .iter_mut()
                    .zip(&q_data[q_pos_base..q_pos_base + rk * d])
                {
                    *o = half::f16::from_f32(x);
                }

                acc.fill(0.0);
                m.fill(f32::NEG_INFINITY);
                ssum.fill(0.0);
                let kv_end = (q_pos + kv_offset + 1).min(kv_len);

                for kv0 in (0..kv_end).step_by(KB) {
                    let kb = KB.min(kv_end - kv0);

                    // Pass 1: scores for all rk heads, each K row loaded once.
                    for j in 0..kb {
                        let kv_base = head_base + (kv0 + j) * 2 * d;
                        let k_row = &kv_data[kv_base..kv_base + d];
                        if j + 1 < kb {
                            prefetch_read(kv_data[kv_base + 2 * d..].as_ptr());
                        }
                        for r in 0..rk {
                            #[cfg(not(feature = "f16-attn-dot"))]
                            let dot = {
                                let q_row = &q_data[q_pos_base + r * d..q_pos_base + (r + 1) * d];
                                dot_f32_f16(q_row, k_row)
                            };
                            #[cfg(feature = "f16-attn-dot")]
                            let dot = dot_f16_f16(&qf16[r * d..(r + 1) * d], k_row);
                            w[r * KB + j] = dot * scale;
                        }
                    }

                    // Per head: one max/rescale per block, then batched exp.
                    for r in 0..rk {
                        let s = &mut w[r * KB..r * KB + kb];
                        let bmax = s.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                        if bmax > m[r] {
                            let f = (m[r] - bmax).exp();
                            if f > 0.0 {
                                for a in &mut acc[r * d..(r + 1) * d] {
                                    *a *= f;
                                }
                                ssum[r] *= f;
                            } else {
                                acc[r * d..(r + 1) * d].fill(0.0);
                                ssum[r] = 0.0;
                            }
                            m[r] = bmax;
                        }
                        let mr = m[r];
                        ssum[r] += exp_sub_sum(s, mr);
                    }

                    // Pass 2: PV, each V row loaded once for all rk heads.
                    for j in 0..kb {
                        let kv_base = head_base + (kv0 + j) * 2 * d;
                        let v_row = &kv_data[kv_base + d..kv_base + 2 * d];
                        if j + 1 < kb {
                            prefetch_read(kv_data[kv_base + 2 * d + d..].as_ptr());
                        }
                        for r in 0..rk {
                            let wj = w[r * KB + j];
                            axpy_f16(&mut acc[r * d..(r + 1) * d], v_row, wj);
                        }
                    }
                }

                for r in 0..rk {
                    let h_i = kv_h * rk + r;
                    let inv = if ssum[r] > 0.0 { 1.0 / ssum[r] } else { 0.0 };
                    let acc_r = &acc[r * d..(r + 1) * d];
                    // SAFETY: each (h_i, q_pos) output row is written by exactly
                    // one task (kv heads and q blocks partition the rows).
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut(p.0.add(h_i * s_q * d + q_pos * d), d)
                    };
                    for t in 0..d {
                        dst[t] = acc_r[t] * inv;
                    }
                }
            }
        }
    });

    Tensor::from_vec(out, (h_q, s_q, d), &Device::Cpu)
}
