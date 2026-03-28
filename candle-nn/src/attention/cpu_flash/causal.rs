//! Single-batch (B=1) causal attention using loop-bound masking.
//!
//! Operates directly on raw f32 slices with precomputed contiguous strides.
//! No tensor operations in the hot path — all reshaping is done via pointer
//! arithmetic on the input slices.
//!
//! For B>1, the dispatcher in `mod.rs` routes to the packed varlen path.

use candle::{DType, Device, Result, Storage, Tensor, WithDType};
use rayon::prelude::*;
use std::iter::Sum;

use super::standard::{vec_dot, FLASH_ATTN_POOL};

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
    T: WithDType + Sum + num_traits::real::Real,
{
    let b = q.dims()[0];
    if b != 1 {
        candle::bail!(
            "causal::run_causal_attn_cpu is B=1 only (got B={b}). \
             Multi-batch should be routed through the varlen path."
        );
    }

    // Squeeze batch dim: (1, S, H, D) → contiguous (S, H, D)
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
                q_data, k_data, v_data,
                h_q, h_kv, h_v, d, s_kv,
                softmax_scale, max_bias, softcap,
            )
        } else {
            causal_prefill_f32(
                q_data, k_data, v_data,
                s_q, h_q, h_kv, h_v, d, s_kv,
                softmax_scale, kv_offset, max_bias, softcap,
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
                q_data, k_data, v_data,
                h_q, h_kv, h_v, d, s_kv,
                softmax_scale, max_bias, softcap,
            )
        } else {
            causal_prefill_generic(
                q_data, k_data, v_data,
                s_q, h_q, h_kv, h_v, d, s_kv,
                softmax_scale, kv_offset, max_bias, softcap,
            )
        };

        result.and_then(|t| t.unsqueeze(0))
    }
}

// ── f32 dot product ─────────────────────────────────────────────────────

#[inline(always)]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut s = 0.0f32;
    let mut i = 0;
    while i + 4 <= n {
        s += a[i] * b[i]
            + a[i + 1] * b[i + 1]
            + a[i + 2] * b[i + 2]
            + a[i + 3] * b[i + 3];
        i += 4;
    }
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

// ── f32 decode (q_len=1) ────────────────────────────────────────────────
//
// Input data layout: contiguous (1, S, H, D), we index past the batch dim.
// For q with S=1: q[h] starts at h*D.
// For k/v with S=kv_len: k[pos, h] starts at pos*H_kv*D + h*D.

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

    FLASH_ATTN_POOL.install(|| {
        out.par_chunks_mut(d)
            .enumerate()
            .for_each_init(
                || vec![0f32; d],
                |acc, (h_i, out_chunk)| {
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

                        let mut score = dot_f32(q_row, k_row) * scale_pre;
                        if do_softcap {
                            score = logit_softcap * score.tanh();
                        }
                        score += alibi_bias;

                        let v_base = kv_pos * v_seq_stride + v_head_off;
                        let v_row = &v_data[v_base..v_base + d];

                        if kv_pos + 1 < kv_len {
                            prefetch_read(v_data[v_base + v_seq_stride..].as_ptr());
                        }

                        if score > m {
                            let scale_old = f32::exp(m - score);
                            for t in 0..d { acc[t] *= scale_old; }
                            ssum *= scale_old;
                            m = score;
                            for t in 0..d { acc[t] += v_row[t]; }
                            ssum += 1.0;
                        } else {
                            let w = f32::exp(score - m);
                            for t in 0..d { acc[t] += v_row[t] * w; }
                            ssum += w;
                        }
                    }

                    let inv = if ssum > 0.0 { 1.0 / ssum } else { 0.0 };
                    for t in 0..d { out_chunk[t] = acc[t] * inv; }
                },
            );
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

    FLASH_ATTN_POOL.install(|| {
        out.par_chunks_mut(d)
            .enumerate()
            .for_each_init(
                || vec![0f32; d],
                |acc, (h_i, out_chunk)| {
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

                        let score = dot_f32(q_row, k_row) * scale;

                        let v_base = kv_pos * v_seq_stride + v_head_off;
                        let v_row = &v_data[v_base..v_base + d];

                        if kv_pos + 1 < kv_len {
                            prefetch_read(v_data[v_base + v_seq_stride..].as_ptr());
                        }

                        if score > m {
                            let scale_old = f32::exp(m - score);
                            for t in 0..d { acc[t] *= scale_old; }
                            ssum *= scale_old;
                            m = score;
                            for t in 0..d { acc[t] += v_row[t]; }
                            ssum += 1.0;
                        } else {
                            let w = f32::exp(score - m);
                            for t in 0..d { acc[t] += v_row[t] * w; }
                            ssum += w;
                        }
                    }

                    let inv = if ssum > 0.0 { 1.0 / ssum } else { 0.0 };
                    for t in 0..d { out_chunk[t] = acc[t] * inv; }
                },
            );
    });

    Tensor::from_vec(out, (h_q, 1usize, d), &Device::Cpu)
}

// ── Interleaved KV decode ────────────────────────────────────────────────
//
// KV data layout: contiguous (S, H_kv, 2*D) where K=[..,:D], V=[..,D:2D].
// One base address per KV position, one prefetch loads both K and V.

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
    let kv_head_stride = 2 * d; // each head has K (d) + V (d)
    let kv_seq_stride = h_kv * kv_head_stride; // stride between positions

    let mut out = vec![0f32; h_q * d];

    FLASH_ATTN_POOL.install(|| {
        out.par_chunks_mut(d)
            .enumerate()
            .for_each_init(
                || vec![0f32; d],
                |acc, (h_i, out_chunk)| {
                    let kv_head = h_i / rk;
                    let kv_head_off = kv_head * kv_head_stride;
                    let q_row = &q_data[h_i * d..(h_i + 1) * d];

                    acc.fill(0.0);
                    let mut m = f32::NEG_INFINITY;
                    let mut ssum = 0.0f32;

                    for kv_pos in 0..kv_len {
                        // Single base for both K and V — adjacent in memory
                        let kv_base = kv_pos * kv_seq_stride + kv_head_off;
                        let k_row = &kv_data[kv_base..kv_base + d];
                        let v_row = &kv_data[kv_base + d..kv_base + 2 * d];

                        // One prefetch loads both next K and V
                        if kv_pos + 1 < kv_len {
                            prefetch_read(kv_data[kv_base + kv_seq_stride..].as_ptr());
                        }

                        let score = dot_f32(q_row, k_row) * scale;

                        if score > m {
                            let scale_old = f32::exp(m - score);
                            for t in 0..d { acc[t] *= scale_old; }
                            ssum *= scale_old;
                            m = score;
                            for t in 0..d { acc[t] += v_row[t]; }
                            ssum += 1.0;
                        } else {
                            let w = f32::exp(score - m);
                            for t in 0..d { acc[t] += v_row[t] * w; }
                            ssum += w;
                        }
                    }

                    let inv = if ssum > 0.0 { 1.0 / ssum } else { 0.0 };
                    for t in 0..d { out_chunk[t] = acc[t] * inv; }
                },
            );
    });

    Tensor::from_vec(out, (h_q, 1usize, d), &Device::Cpu)
}

// ── f32 prefill (q_len > 1) ────────────────────────────────────────────

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

    FLASH_ATTN_POOL.install(|| {
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

                        if kv_pos + 1 < kv_end {
                            prefetch_read(k_data[k_base + k_seq_stride..].as_ptr());
                        }

                        let mut score = dot_f32(q_row, k_row) * scale_pre;
                        if do_softcap {
                            score = logit_softcap * score.tanh();
                        }
                        score += alibi_bias;

                        let v_base = kv_pos * v_seq_stride + v_head_off;
                        let v_row = &v_data[v_base..v_base + d];

                        if kv_pos + 1 < kv_end {
                            prefetch_read(v_data[v_base + v_seq_stride..].as_ptr());
                        }

                        if score > m {
                            let scale_old = f32::exp(m - score);
                            for t in 0..d {
                                acc[t] *= scale_old;
                            }
                            ssum *= scale_old;
                            m = score;

                            for t in 0..d {
                                acc[t] += v_row[t];
                            }
                            ssum += 1.0;
                        } else {
                            let w = f32::exp(score - m);
                            for t in 0..d {
                                acc[t] += v_row[t] * w;
                            }
                            ssum += w;
                        }
                    }

                    let inv = if ssum > 0.0 { 1.0 / ssum } else { 0.0 };
                    for t in 0..d {
                        out_chunk[t] = acc[t] * inv;
                    }
                },
            );
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

    FLASH_ATTN_POOL.install(|| {
        out.par_chunks_mut(d)
            .with_min_len(64)
            .enumerate()
            .for_each_init(
                || vec![0f32; d],
                |acc, (row_idx, out_chunk)| {
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

                        let score = dot_f32(q_row, k_row) * scale;

                        let v_base = kv_pos * v_seq_stride + v_head_off;
                        let v_row = &v_data[v_base..v_base + d];

                        if kv_pos + 1 < kv_end {
                            prefetch_read(v_data[v_base + v_seq_stride..].as_ptr());
                        }

                        if score > m {
                            let scale_old = f32::exp(m - score);
                            for t in 0..d { acc[t] *= scale_old; }
                            ssum *= scale_old;
                            m = score;
                            for t in 0..d { acc[t] += v_row[t]; }
                            ssum += 1.0;
                        } else {
                            let w = f32::exp(score - m);
                            for t in 0..d { acc[t] += v_row[t] * w; }
                            ssum += w;
                        }
                    }

                    let inv = if ssum > 0.0 { 1.0 / ssum } else { 0.0 };
                    for t in 0..d { out_chunk[t] = acc[t] * inv; }
                },
            );
    });

    Tensor::from_vec(out, (h_q, s_q, d), &Device::Cpu)
}

// ── Generic fallback (non-f32) ──────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn causal_decode_generic<T: WithDType + Sum + num_traits::real::Real>(
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

    FLASH_ATTN_POOL.install(|| {
        out.par_chunks_mut(d)
            .enumerate()
            .for_each_init(
                || vec![0f32; d],
                |acc, (h_i, out_chunk)| {
                    let slope = if max_bias > 0.0 {
                        2.0f32.powf(-max_bias * ((h_i + 1) as f32) / n2 as f32)
                    } else { 0.0 };
                    let k_head_off = (h_i / rk) * d;
                    let v_head_off = (h_i / rv) * d;
                    let q_row = &q_data[h_i * d..(h_i + 1) * d];

                    acc.fill(0.0);
                    let mut m = f32::NEG_INFINITY;
                    let mut ssum = 0.0f32;

                    for kv_pos in 0..kv_len {
                        let alibi_bias = if max_bias > 0.0 {
                            slope * (kv_pos as f32 - (kv_len - 1) as f32)
                        } else { 0.0 };
                        let k_base = kv_pos * k_seq_stride + k_head_off;
                        let k_row = &k_data[k_base..k_base + d];
                        let mut s_val = vec_dot::<T>(q_row, k_row).to_f32().unwrap_or(0.0);
                        s_val *= scale_pre;
                        if do_softcap { s_val = logit_softcap * s_val.tanh(); }
                        s_val += alibi_bias;

                        if s_val > m {
                            let ms = (m - s_val).exp();
                            for t in 0..d { acc[t] *= ms; }
                            ssum *= ms;
                            m = s_val;
                            let v_base = kv_pos * v_seq_stride + v_head_off;
                            let v_row = &v_data[v_base..v_base + d];
                            for t in 0..d { acc[t] += v_row[t].to_f32().unwrap_or(0.0); }
                            ssum += 1.0;
                        } else {
                            let w = (s_val - m).exp();
                            let v_base = kv_pos * v_seq_stride + v_head_off;
                            let v_row = &v_data[v_base..v_base + d];
                            for t in 0..d { acc[t] += v_row[t].to_f32().unwrap_or(0.0) * w; }
                            ssum += w;
                        }
                    }
                    let inv = if ssum > 0.0 { 1.0 / ssum } else { 0.0 };
                    for t in 0..d { out_chunk[t] = acc[t] * inv; }
                },
            );
    });

    Tensor::from_vec(out, (h_q, 1usize, d), &Device::Cpu)
}

#[allow(clippy::too_many_arguments)]
fn causal_prefill_generic<T: WithDType + Sum + num_traits::real::Real>(
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

    FLASH_ATTN_POOL.install(|| {
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
                    } else { 0.0 };
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
                        } else { 0.0 };
                        let k_base = kv_pos * k_seq_stride + k_head_off;
                        let k_row = &k_data[k_base..k_base + d];
                        let mut s_val = vec_dot::<T>(q_row, k_row).to_f32().unwrap_or(0.0);
                        s_val *= scale_pre;
                        if do_softcap { s_val = logit_softcap * s_val.tanh(); }
                        s_val += alibi_bias;

                        if s_val > m {
                            let ms = (m - s_val).exp();
                            for t in 0..d { acc[t] *= ms; }
                            ssum *= ms;
                            m = s_val;
                            let v_base = kv_pos * v_seq_stride + v_head_off;
                            let v_row = &v_data[v_base..v_base + d];
                            for t in 0..d { acc[t] += v_row[t].to_f32().unwrap_or(0.0); }
                            ssum += 1.0;
                        } else {
                            let w = (s_val - m).exp();
                            let v_base = kv_pos * v_seq_stride + v_head_off;
                            let v_row = &v_data[v_base..v_base + d];
                            for t in 0..d { acc[t] += v_row[t].to_f32().unwrap_or(0.0) * w; }
                            ssum += w;
                        }
                    }
                    let inv = if ssum > 0.0 { 1.0 / ssum } else { 0.0 };
                    for t in 0..d { out_chunk[t] = acc[t] * inv; }
                },
            );
    });

    Tensor::from_vec(out, (h_q, s_q, d), &Device::Cpu)
}
