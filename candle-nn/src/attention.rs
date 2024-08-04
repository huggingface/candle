use candle::{Device, Result, Tensor};

use crate::cublaslt::{setup_cublas_lt_wrapper, CUBLASLT_HANDLE};

#[cfg(feature = "flash-attn")]
pub fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
pub fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("Compile with '--features flash-attn'")
}

/// Computes softmax(QK^T*sqrt(d_k))V
fn naive_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    head_dim: usize,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let att = (&q.contiguous()?.matmul(&k.t()?.contiguous()?)? / (head_dim as f64).sqrt())?;

    let att = match mask {
        Some(m) => att.broadcast_add(m)?,
        None => att,
    };
    let att = crate::ops::softmax_last_dim(&att)?;
    // Convert to contiguous as matmul doesn't support strided vs for now.
    att.matmul(&v.contiguous()?)
}

/// Computes softmax(QK^T*sqrt(d_k))V
///
/// The attention implementation is automatically accelerated and dispatched as follows:
/// 1) If `use_flash_attn == true`, use a Flash Attention V2 kernel
/// 2) If using CUDA, it will attempt to use cuBLASlt an optimized version
/// 3) Otherwise, use the "naive" SDPA implementation - just matmuls and elementwise operations.
/// 
/// Note that there may be minute differences in output because floating point operations are not associative.
#[allow(unused_variables, clippy::too_many_arguments)]
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_attn_heads: usize,
    head_dim: usize,
    mask: Option<&Tensor>,
    use_flash_attn: bool,
    b_sz: usize,
    seq_len: usize,
) -> Result<Tensor> {
    if use_flash_attn {
        // flash-attn expects (b_sz, seq_len, nheads, head_dim)
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let softmax_scale = 1f32 / (head_dim as f32).sqrt();
        return flash_attn(&q, &k, &v, softmax_scale, seq_len > 1)?.transpose(1, 2);
    }

    // Initializiation is behind a LazyLock. So, the first call will be slightly slower.
    // No cost to the other calls.
    setup_cublas_lt_wrapper();

    if let (Device::Cuda(_), Some(cublaslt)) = (q.device(), *CUBLASLT_HANDLE.lock().unwrap()) {
        #[cfg(feature = "cuda")]
        {
            // cuBLASLt batch matmul implementation requires inputs to be dims3
            let k = k.flatten(0, 1)?;
            let q = q.flatten(0, 1)?;
            let v = v.flatten(0, 1)?;
            let attention_bias = mask.map(|mask| mask.flatten(0, 1)).transpose()?;

            // If attention_bias is set, we fuse the add by giving it as the output matrix
            // and setting beta to 1.0
            let beta = match attention_bias.is_some() {
                true => Some(1.0),
                false => None,
            };

            // Batch matrix multiplication
            // Fuse softmax scale and attention_bias add
            let attention_scores = cublaslt.batch_matmul(
                &k,
                &q,
                attention_bias.as_ref(),
                Some((1.0 / (head_dim as f64).sqrt()) as f32),
                beta,
                None,
                None,
            )?;
            let attention_probs = crate::ops::softmax_last_dim(&attention_scores)?;

            let context_layer = cublaslt.batch_matmul(
                &v.t()?.contiguous()?,
                &attention_probs,
                // We save one allocation
                Some(&q),
                None,
                None,
                None,
                None,
            )?;

            // Reshape to dims4
            context_layer.reshape((b_sz, n_attn_heads, seq_len, head_dim))
        }
        #[cfg(not(feature = "cuda"))]
        {
            candle::bail!("`cuda` feature is not enabled")
        }
    } else {
        naive_sdpa(q, k, v, head_dim, mask)
    }
}
