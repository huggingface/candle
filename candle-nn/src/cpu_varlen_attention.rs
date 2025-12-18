use crate::ops;
use candle::Tensor;

/// Cuda-free fallback implementation for variable length flash attention
//  https://github.com/Dao-AILab/flash-attention/blob/ac9b5f107f2f19cd0ca6e01548d20d072a46335c/csrc/flash_attn/flash_api.cpp#L515
/// No fused attention is used, "flash" fused, but allows for padding-free variable length attention on CPU.

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
#[allow(clippy::needless_range_loop)]
pub fn flash_attn_varlen_cpu(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: Option<&Tensor>,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    _max_seqlen_q: usize,
    _max_seqlen_k: usize,
    softmax_scale: f64,
    causal: bool,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor, candle::Error> {
    // Get device and shape information
    let device = q.device();
    let (_total_q, num_heads, head_dim) = q.dims3()?;
    let (total_k, num_kv_heads, _) = k.dims3()?;

    // Handle GQA (Grouped Query Attention) by repeating k/v heads if needed
    let (k, v) = if num_heads != num_kv_heads {
        if num_heads % num_kv_heads != 0 {
            candle::bail!(
                "Invalid GQA config: num_heads={} not divisible by num_kv_heads={}",
                num_heads,
                num_kv_heads
            );
        }
        let repeat_factor = num_heads / num_kv_heads;

        // Reshape to [total_k, num_kv_heads, 1, head_dim] for proper broadcasting
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

    // Process each sequence in the batch
    let batch_size = seqlens_q.dims()[0];
    let mut outputs = Vec::new();

    // Pre-compute cumulative sequence lengths to avoid O(n²) nested loops
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

        // Use pre-computed cumulative sequence lengths (O(1) lookup)
        let start_q = cumsum_q[i];
        let start_k = cumsum_k[i];

        // Extract Q, K, V for this sequence
        let q_seq = q.narrow(0, start_q, seq_len_q)?; // [seq_len_q, num_heads, head_dim]
        let k_seq = k.narrow(0, start_k, seq_len_k)?; // [seq_len_k, num_heads, head_dim]
        let v_seq = v.narrow(0, start_k, seq_len_k)?; // [seq_len_k, num_heads, head_dim]

        // Transpose to [seq_len_q, num_heads, head_dim] and [seq_len_k, num_heads, head_dim]
        // This matches the format used in bert.rs and qwen3.rs
        // Ensure contiguous after transpose to avoid matmul issues
        let q_seq = q_seq.transpose(0, 1)?.contiguous()?; // [num_heads, seq_len_q, head_dim]
        let k_seq = k_seq.transpose(0, 1)?.contiguous()?; // [num_heads, seq_len_k, head_dim]
        let v_seq = v_seq.transpose(0, 1)?.contiguous()?; // [num_heads, seq_len_k, head_dim]

        // Compute attention scores for all heads at once (following bert.rs/qwen3.rs pattern)
        // Reshape to [num_heads, seq_len_q, seq_len_k] for batch matrix multiplication
        // Ensure k_seq transpose is contiguous for matmul
        let k_seq_t = k_seq.transpose(1, 2)?.contiguous()?;
        let attention_scores = q_seq.matmul(&k_seq_t)?; // [num_heads, seq_len_q, seq_len_k]

        // Apply softmax scale
        let scale_tensor = Tensor::new(softmax_scale as f32, device)?;
        let mut attention_scores =
            attention_scores.mul(&scale_tensor.broadcast_as(attention_scores.shape())?)?; // [num_heads, seq_len_q, seq_len_k]

        // Apply causal mask if requested
        if causal {
            let causal_mask = create_causal_mask_batch(seq_len_q, seq_len_k, num_heads, device)?;
            attention_scores = attention_scores.add(&causal_mask)?;
        }

        // Apply windowing if specified
        if window_size_left.is_some() || window_size_right.is_some() {
            let window_mask = create_window_mask_batch(
                seq_len_q,
                seq_len_k,
                num_heads,
                window_size_left,
                window_size_right,
                device,
            )?;
            attention_scores = attention_scores.add(&window_mask)?;
        }

        // Apply ALiBi slopes if provided
        if let Some(alibi_slopes) = alibi_slopes {
            let alibi_bias = create_alibi_bias_batch(
                seq_len_q,
                seq_len_k,
                num_heads,
                alibi_slopes,
                causal,
                device,
            )?;
            attention_scores = attention_scores.add(&alibi_bias)?;
        }

        // Apply softmax along the last dimension (seq_len_k)
        let attention_probs = ops::softmax_last_dim(&attention_scores)?; // [num_heads, seq_len_q, seq_len_k]

        // Compute attention output for all heads at once
        let context_layer = attention_probs.matmul(&v_seq)?; // [num_heads, seq_len_q, head_dim]

        // Transpose back to [seq_len_q, num_heads, head_dim] to match expected output format
        let seq_output = context_layer.transpose(0, 1)?; // [seq_len_q, num_heads, head_dim]
        outputs.push(seq_output);
    }

    // Concatenate all sequence outputs
    if outputs.is_empty() {
        return Tensor::zeros((0, num_heads, head_dim), q.dtype(), device);
    }

    Tensor::cat(&outputs, 0)
}

/// Create causal attention mask for all heads at once
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
                // FlashAttn-style: allow j <= i + offset
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

/// Create window attention mask for all heads at once
/// Supports different windowing patterns:
/// - Standard: both window_left and window_right (bidirectional window)
/// - Mistral-style: only window_left (causal sliding window)
/// - Gemma3-style: bidirectional distance-based windowing
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
                let i_k = i as isize + offset; // query position in key-index space
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

        // Mistral-style causal sliding window: allow j <= i_k AND (i_k - j) <= left
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

/// Create ALiBi (Attention with Linear Biases) bias for all heads at once
/// Optimized to avoid per-head to_scalar() calls
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
                        // (i_k - j) is >= 0 for allowed positions
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

    Tensor::stack(&head_biases, 0) // [H,Q,K]
}
