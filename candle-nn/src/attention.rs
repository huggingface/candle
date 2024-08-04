use candle::{Result, Tensor};

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

/// Computes (softmax(QK^T*sqrt(d_k)) + M)V. `M` is the attention mask, and is a bias (0 for unmasked, -inf for masked).
///
/// The attention implementation is automatically accelerated and dispatched as follows:
/// 1) If `use_flash_attn == true`, use a Flash Attention V2 kernel
/// 2) Otherwise, use SDPA with fusion of softmax scale and attention bias application
///
/// Note that there may be minute differences in output because floating point operations are not associative.
#[allow(unused_variables, clippy::too_many_arguments)]
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_attn_heads: usize,
    head_dim: usize,
    mask: Option<Tensor>,
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

    let att = match mask {
        Some(mut m) => {
            q.contiguous()?.matmul_with_alpha_beta(
                &k.t()?.contiguous()?,
                &mut m,
                Some(1. / (head_dim as f64).sqrt()),
            )?;
            m
        }
        None => (&q.contiguous()?.matmul(&k.t()?.contiguous()?)? / (head_dim as f64).sqrt())?,
    };

    let att = crate::ops::softmax_last_dim(&att)?;
    // Convert to contiguous as matmul doesn't support strided vs for now.
    att.matmul(&v.contiguous()?)
}
