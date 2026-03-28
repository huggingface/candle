//! Unfused variable-length attention fallback.
//!
//! Uses standard tensor operations (matmul + softmax) instead of fused kernels.
//! Supports variable-length sequences without padding, causal masking,
//! sliding window, ALiBi, and GQA.
//!
//! Prefer `cpu_flash::varlen::flash_attn_varlen_cpu` for better performance;
//! this exists as a reference / fallback for correctness testing.
//!
//! Ported from <https://github.com/huggingface/candle/pull/3250>.

use crate::ops;
use candle::Tensor;

/// Unfused variable-length flash attention (no fused kernels).
///
/// **Input shapes** (packed, no batch dim):
/// - `q`: `[total_q, Hq, D]`
/// - `k`: `[total_k, Hk, D]`
/// - `v`: `[total_k, Hv, D]`
/// - `seqlens_q`, `seqlens_k`: `[B]` (u32 per-sequence lengths)
///
/// **Output shape:** `[total_q, Hq, D]`
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_varlen_unfused(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: Option<&Tensor>,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    _max_seqlen_q: usize,
    _max_seqlen_k: usize,
    softmax_scale: f32,
    causal: bool,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor, candle::Error> {
    let device = q.device();
    let (_total_q, num_heads, head_dim) = q.dims3()?;
    let (total_k, num_kv_heads, _) = k.dims3()?;

    // Handle GQA by repeating k/v heads if needed
    let (k, v) = if num_heads != num_kv_heads {
        if num_heads % num_kv_heads != 0 {
            candle::bail!(
                "Invalid GQA config: num_heads={} not divisible by num_kv_heads={}",
                num_heads,
                num_kv_heads
            );
        }
        let repeat_factor = num_heads / num_kv_heads;

        let k = k
            .reshape((total_k, num_kv_heads, 1, head_dim))?
            .broadcast_as((total_k, num_kv_heads, repeat_factor, head_dim))?
            .reshape((total_k, num_heads, head_dim))?;
        let v = v
            .reshape((total_k, num_kv_heads, 1, head_dim))?
            .broadcast_as((total_k, num_kv_heads, repeat_factor, head_dim))?
            .reshape((total_k, num_heads, head_dim))?;
        (k, v)
    } else {
        (k.clone(), v.clone())
    };

    let batch_size = seqlens_q.dims()[0];
    let mut outputs = Vec::new();

    let seqlens_q_vec = seqlens_q.to_vec1::<u32>()?;
    let seqlens_k_vec = seqlens_k.to_vec1::<u32>()?;
    let mut cumsum_q = vec![0usize; batch_size + 1];
    let mut cumsum_k = vec![0usize; batch_size + 1];
    for i in 0..batch_size {
        cumsum_q[i + 1] = cumsum_q[i] + seqlens_q_vec[i] as usize;
        cumsum_k[i + 1] = cumsum_k[i] + seqlens_k_vec[i] as usize;
    }

    for i in 0..batch_size {
        let seq_len_q = seqlens_q_vec[i] as usize;
        let seq_len_k = seqlens_k_vec[i] as usize;

        if seq_len_q == 0 || seq_len_k == 0 {
            continue;
        }

        if causal && seq_len_k < seq_len_q {
            candle::bail!(
                "causal attention requires seq_len_k >= seq_len_q (got k={}, q={}) for batch index {}",
                seq_len_k,
                seq_len_q,
                i
            );
        }

        let start_q = cumsum_q[i];
        let start_k = cumsum_k[i];

        let q_seq = q.narrow(0, start_q, seq_len_q)?;
        let k_seq = k.narrow(0, start_k, seq_len_k)?;
        let v_seq = v.narrow(0, start_k, seq_len_k)?;

        let q_seq = q_seq.transpose(0, 1)?.contiguous()?; // [H, Sq, D]
        let k_seq = k_seq.transpose(0, 1)?.contiguous()?; // [H, Sk, D]
        let v_seq = v_seq.transpose(0, 1)?.contiguous()?; // [H, Sk, D]

        let k_seq_t = k_seq.transpose(1, 2)?.contiguous()?;
        let attention_scores = q_seq.matmul(&k_seq_t)?; // [H, Sq, Sk]

        let scale_tensor =
            Tensor::new(softmax_scale, device)?.to_dtype(attention_scores.dtype())?;
        let mut attention_scores =
            attention_scores.mul(&scale_tensor.broadcast_as(attention_scores.shape())?)?;

        if causal {
            let causal_mask = create_causal_mask_batch(seq_len_q, seq_len_k, num_heads, device)?
                .to_dtype(attention_scores.dtype())?;
            attention_scores = attention_scores.add(&causal_mask)?;
        }

        if window_size_left.is_some() || window_size_right.is_some() {
            let window_mask = create_window_mask_batch(
                seq_len_q,
                seq_len_k,
                num_heads,
                window_size_left,
                window_size_right,
                device,
            )?
            .to_dtype(attention_scores.dtype())?;
            attention_scores = attention_scores.add(&window_mask)?;
        }

        if let Some(alibi_slopes) = alibi_slopes {
            let alibi_bias = create_alibi_bias_batch(
                seq_len_q,
                seq_len_k,
                num_heads,
                alibi_slopes,
                causal,
                device,
            )?
            .to_dtype(attention_scores.dtype())?;
            attention_scores = attention_scores.add(&alibi_bias)?;
        }

        let attention_probs = ops::softmax_last_dim(&attention_scores)?;
        let context_layer = attention_probs.matmul(&v_seq)?; // [H, Sq, D]

        let seq_output = context_layer.transpose(0, 1)?; // [Sq, H, D]
        outputs.push(seq_output);
    }

    if outputs.is_empty() {
        return Tensor::zeros((0, num_heads, head_dim), q.dtype(), device);
    }

    Tensor::cat(&outputs, 0)
}

fn create_causal_mask_batch(
    seq_len_q: usize,
    seq_len_k: usize,
    num_heads: usize,
    device: &candle::Device,
) -> Result<Tensor, candle::Error> {
    let offset = seq_len_k as isize - seq_len_q as isize;

    let mask: Vec<f32> = (0..seq_len_q)
        .flat_map(|i| {
            let i = i as isize;
            (0..seq_len_k).map(move |j| {
                let j = j as isize;
                if j > i + offset {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
        })
        .collect();

    let mask = Tensor::from_vec(mask, (seq_len_q, seq_len_k), device)?;
    mask.expand((num_heads, seq_len_q, seq_len_k))
}

fn create_window_mask_batch(
    seq_len_q: usize,
    seq_len_k: usize,
    num_heads: usize,
    window_left: Option<usize>,
    window_right: Option<usize>,
    device: &candle::Device,
) -> Result<Tensor, candle::Error> {
    let offset = seq_len_k as isize - seq_len_q as isize;

    let mask: Vec<f32> = match (window_left, window_right) {
        (Some(left), Some(right)) => (0..seq_len_q)
            .flat_map(|i| {
                let i_k = i as isize + offset;
                (0..seq_len_k).map(move |j| {
                    let j = j as isize;
                    let left_dist = (i_k - j).max(0) as usize;
                    let right_dist = (j - i_k).max(0) as usize;
                    if left_dist > left || right_dist > right {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect(),

        (Some(left), None) => (0..seq_len_q)
            .flat_map(|i| {
                let i_k = i as isize + offset;
                (0..seq_len_k).map(move |j| {
                    let j = j as isize;
                    if j > i_k {
                        return f32::NEG_INFINITY;
                    }
                    let dist = (i_k - j) as usize;
                    if dist > left {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect(),

        (None, None) => vec![0.0; seq_len_q * seq_len_k],
        (None, Some(_)) => candle::bail!("window_right specified without window_left"),
    };

    let mask = Tensor::from_vec(mask, (seq_len_q, seq_len_k), device)?;
    mask.expand((num_heads, seq_len_q, seq_len_k))
}

fn create_alibi_bias_batch(
    seq_len_q: usize,
    seq_len_k: usize,
    num_heads: usize,
    alibi_slopes: &Tensor,
    causal: bool,
    device: &candle::Device,
) -> Result<Tensor, candle::Error> {
    let slopes = alibi_slopes.to_vec1::<f32>()?;
    if slopes.len() != num_heads {
        candle::bail!(
            "alibi_slopes has len {}, expected num_heads={}",
            slopes.len(),
            num_heads
        );
    }

    let offset = seq_len_k as isize - seq_len_q as isize;

    let mut head_biases = Vec::with_capacity(num_heads);
    for &slope in slopes.iter() {
        let bias: Vec<f32> = (0..seq_len_q)
            .flat_map(|i| {
                let i_k = i as isize + offset;
                (0..seq_len_k).map(move |j| {
                    let j = j as isize;
                    let dist = if causal {
                        (i_k - j).max(0) as f32
                    } else {
                        (j - i_k).abs() as f32
                    };
                    -slope * dist
                })
            })
            .collect();

        head_biases.push(Tensor::from_vec(bias, (seq_len_q, seq_len_k), device)?);
    }

    Tensor::stack(&head_biases, 0) // [H, Q, K]
}
