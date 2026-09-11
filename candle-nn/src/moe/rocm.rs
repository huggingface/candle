//! `moe_gemm_gguf` on ROCm, built out of [`QTensor::indexed_moe_forward`].
//!
//! The CUDA path has two dedicated kernels in `candle-kernels/src/moe/`, both
//! compiled into a host-side static library (`libmoe.a`) that the ROCm backend
//! has no build pipeline for, and the prefill one is `nvcuda::wmma` besides. So
//! this goes the other way: `indexed_moe_forward_*_q8_1` in `quantized.cu` — a
//! module hipcc already compiles — computes the same routed mat-vecs, and the
//! rest is index bookkeeping done in tensor ops.
//!
//! # The two argument conventions
//!
//! `moe_gemm_gguf` takes the routing already sorted by expert, which is what
//! lets its kernel keep one expert's weights resident across a block:
//!
//! ```text
//! for m in 0..size_m:
//!     t = sorted_token_ids[m]            // the flat (token, slot) pair
//!     out[t] = W[experts_ids[m]] @ in[input_index(t)] * scale(t)
//! ```
//!
//! `indexed_moe_forward` takes it *unsorted*, indexed by the pair itself:
//! `out[b][j] = W[ids[b][j]] @ x[b][…]`. Since `sorted_token_ids` is a
//! permutation of `0..size_m`, the two carry identical information and
//! [`unsort`] converts between them.

use candle::quantized::QTensor;
use candle::{DType, Result, Tensor};

/// Undo the expert-major sort: return `routed[t] = experts_ids[m]` for the `m`
/// with `sorted_token_ids[m] == t`.
///
/// `sorted_token_ids` is a permutation of `0..size_m`, so arg-sorting it
/// ascending yields exactly that inverse mapping.
fn unsort(sorted_token_ids: &Tensor, experts_ids: &Tensor) -> Result<Tensor> {
    let inverse = sorted_token_ids.arg_sort_last_dim(true)?;
    experts_ids.index_select(&inverse, 0)
}

/// See [`super::moe_gemm_gguf`]. `is_prefill` has no meaning here: the CUDA
/// backend uses it to pick between a WMMA tile kernel and a mat-vec kernel, and
/// ROCm only has the latter.
pub(super) fn moe_gemm_gguf(
    input: &Tensor,
    weights: &QTensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
) -> Result<Tensor> {
    if topk == 0 {
        candle::bail!("moe_gemm_gguf: topk must be non-zero")
    }
    let (rows, k) = input.dims2()?;
    let (_, _, weight_k) = weights.shape().dims3()?;
    if weight_k != k {
        candle::bail!(
            "moe_gemm_gguf: input has k={k} but the experts are {:?}",
            weights.shape()
        )
    }

    // Without `topk_weights` each input row is shared by all `topk` of its
    // experts, so there are `topk` times as many routed pairs as rows; with it,
    // the caller has already expanded them one per pair.
    let (size_m, input_dim1) = match topk_weights {
        None => (rows * topk, 1),
        Some(_) => (rows, topk),
    };
    let batch = size_m / topk;
    if batch * topk != size_m {
        candle::bail!("moe_gemm_gguf: {size_m} routed pairs is not a multiple of topk {topk}")
    }
    for (name, t) in [
        ("sorted_token_ids", sorted_token_ids),
        ("experts_ids", experts_ids),
    ] {
        if t.dims1()? != size_m {
            candle::bail!(
                "moe_gemm_gguf: {name} is {:?}, expected [{size_m}]",
                t.shape()
            )
        }
    }

    let ids = unsort(sorted_token_ids, experts_ids)?.reshape((batch, topk))?;
    let x = input.reshape((batch, input_dim1, k))?.contiguous()?;
    let out = weights.indexed_moe_forward(&x, &ids)?;

    let out = match topk_weights {
        None => out,
        // The kernel scales by the routing weight of the pair it just computed,
        // which is the same `(batch, topk)` grid the output carries.
        Some(w) => out.broadcast_mul(&w.to_dtype(DType::F32)?.reshape((batch, topk, 1))?)?,
    };
    out.reshape((size_m, ()))
}

#[cfg(test)]
mod tests;
