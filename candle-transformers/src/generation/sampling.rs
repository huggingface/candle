use candle::{DType, Result, Tensor, D};

/// Apply repetition penalty to logits.
///
/// `context` must be a 1D U32 tensor of deduped previously seen token ids.
pub fn apply_repeat_penalty(logits: &Tensor, context: &Tensor, penalty: f32) -> Result<Tensor> {
    candle::sampling::apply_repeat_penalty(logits, context, penalty)
}

/// Zero out logits for all but the top `k` tokens (set to -inf).
/// Returns the tensor unchanged when `k >= vocab_size`.
pub fn apply_topk_mask(logits: &Tensor, k: usize) -> Result<Tensor> {
    let vocab_size = logits.elem_count();
    if k >= vocab_size {
        return Ok(logits.clone());
    }
    let (sorted, _) = logits.sort_last_dim(false)?;
    let threshold = sorted.narrow(0, k - 1, 1)?.broadcast_as(logits.shape())?;
    let neg_inf = Tensor::full(f32::NEG_INFINITY, logits.shape(), logits.device())?
        .to_dtype(logits.dtype())?;
    logits.ge(&threshold)?.where_cond(logits, &neg_inf)
}

/// Zero out logits for tokens outside the top-p nucleus (set to -inf).
///
/// Sorts by probability descending, masks all tokens whose exclusive
/// prefix-sum of probabilities is >= `p`. Ensures at least one token is
/// always kept.
pub fn apply_topp_mask(logits: &Tensor, p: f64) -> Result<Tensor> {
    let vocab_size = logits.elem_count();
    let probs = candle_nn::ops::softmax_last_dim(logits)?;
    let (sorted_probs, sorted_idx) = probs.sort_last_dim(false)?;

    let cumsum = sorted_probs.cumsum(D::Minus1)?;
    let zero = Tensor::zeros(&[1], probs.dtype(), logits.device())?;
    let shifted = Tensor::cat(&[&zero, &cumsum.narrow(0, 0, vocab_size - 1)?], 0)?;

    let keep_sorted = shifted.lt(p as f32)?;

    let keep_orig = Tensor::zeros(&[vocab_size], DType::U8, logits.device())?.scatter(
        &sorted_idx,
        &keep_sorted,
        D::Minus1,
    )?;

    let neg_inf = Tensor::full(f32::NEG_INFINITY, logits.shape(), logits.device())?
        .to_dtype(logits.dtype())?;
    keep_orig.where_cond(logits, &neg_inf)
}
