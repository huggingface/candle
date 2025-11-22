//! Loss Calculations
//!
use candle::{Result, Tensor};

/// The negative log likelihood loss.
///
/// Arguments
///
/// * [inp]: The input tensor of dimensions `N, C` where `N` is the batch size and `C` the number
///   of categories. This is expected to contain log probabilities.
/// * [target]: The ground truth labels as a tensor of u32 of dimension `N`.
///
/// The resulting tensor is a scalar containing the average value over the batch.
pub fn nll(inp: &Tensor, target: &Tensor) -> Result<Tensor> {
    let b_sz = match target.dims() {
        &[b_sz] => b_sz,
        dims => candle::bail!("the target tensor should have a single dimension ({dims:?})"),
    };
    match inp.dims() {
        &[inp_b_sz, _] => {
            if inp_b_sz != b_sz {
                candle::bail!("batch size mismatch between inp ({inp_b_sz}) and target ({b_sz})")
            }
        }
        dims => candle::bail!("the target tensor should have two dimensions ({dims:?})"),
    }
    inp.gather(&target.unsqueeze(1)?, 1)?
        .sum_all()?
        .affine(-1f64 / b_sz as f64, 0.)
}

/// Negative log-likelihood loss with support for ignoring a target index.
///
/// Arguments
///
/// * `inp`: The input tensor of dimensions `N, C` where `N` is the batch size and `C` the number
///   of categories. This is expected to contain log probabilities.
/// * `target`: The ground truth labels as a tensor of u32 of dimension `N` with values in `[0, C)`
///   or equal to `ignore_index`.
/// * `ignore_index`: Target value to ignore in loss computation (can be out of range).
///
/// Returns a scalar tensor containing the average value over the non-ignored elements. If all
/// targets are ignored, returns `0.0` (unlike PyTorch which returns `NaN`).
pub fn nll_with_ignore(inp: &Tensor, target: &Tensor, ignore_index: u32) -> Result<Tensor> {
    let b_sz = match target.dims() {
        &[b_sz] => b_sz,
        dims => candle::bail!("the target tensor should have a single dimension ({dims:?})"),
    };
    let (inp_b_sz, _c) = match inp.dims() {
        &[inp_b_sz, c] => (inp_b_sz, c),
        dims => candle::bail!("the inp tensor should have two dimensions ({dims:?})"),
    };
    if inp_b_sz != b_sz {
        candle::bail!("batch size mismatch between inp ({inp_b_sz}) and target ({b_sz})");
    }

    // Create mask BEFORE any gather to handle any out-of-range ignore_index safely.
    let keep_mask = target.ne(ignore_index)?;

    // Sanitize targets: replace ignored indices with 0 (safe index for gather).
    let zeros = Tensor::zeros_like(target)?;
    let target_safe = keep_mask.where_cond(target, &zeros)?;

    // Gather selected log-probabilities.
    let gathered = inp.gather(&target_safe.unsqueeze(1)?, 1)?.squeeze(1)?;

    // Convert mask to same dtype as `inp` for arithmetic masking.
    let mask_float = keep_mask.to_dtype(inp.dtype())?;

    // Zero-out contributions from ignored positions.
    let masked = gathered.mul(&mask_float)?;

    // Count non-ignored elements.
    let count = mask_float
        .sum_all()?
        .to_dtype(candle::DType::F64)?
        .to_scalar::<f64>()?;
    if count == 0.0 {
        return Tensor::new(0.0f64, inp.device())?.to_dtype(inp.dtype());
    }

    masked.sum_all()?.affine(-1.0 / count, 0.)
}

/// The cross-entropy loss.
///
/// Arguments
///
/// * [inp]: The input tensor of dimensions `N, C` where `N` is the batch size and `C` the number
///   of categories. This is expected to raw logits.
/// * [target]: The ground truth labels as a tensor of u32 of dimension `N`.
///
/// The resulting tensor is a scalar containing the average value over the batch.
pub fn cross_entropy(inp: &Tensor, target: &Tensor) -> Result<Tensor> {
    if inp.rank() != 2 {
        candle::bail!("cross_entropy expects an input tensor of rank 2")
    }
    let inp = crate::ops::log_softmax(inp, 1)?;
    nll(&inp, target)
}

/// Cross-entropy loss with support for ignoring a target index.
///
/// Applies `log_softmax` to input logits and then computes NLL loss with ignore_index semantics.
///
/// Arguments
///
/// * `inp`: The input tensor of dimensions `N, C` where `N` is the batch size and `C` the number
///   of categories. This is expected to contain raw logits.
/// * `target`: The ground truth labels as a tensor of u32 of dimension `N` with values in `[0, C)`
///   or equal to `ignore_index`.
/// * `ignore_index`: Target value to ignore in loss computation (can be out of range).
///
/// Returns a scalar tensor containing the average value over the non-ignored elements. If all
/// targets are ignored, returns `0.0` (unlike PyTorch which returns `NaN`).
pub fn cross_entropy_with_ignore(inp: &Tensor, target: &Tensor, ignore_index: u32) -> Result<Tensor> {
    if inp.rank() != 2 {
        candle::bail!("cross_entropy expects an input tensor of rank 2")
    }
    let inp = crate::ops::log_softmax(inp, 1)?;
    nll_with_ignore(&inp, target, ignore_index)
}

/// The mean squared error loss.
pub fn mse(inp: &Tensor, target: &Tensor) -> Result<Tensor> {
    (inp - target)?.sqr()?.mean_all()
}

/// The binary cross-entropy with logit loss.
///
/// Arguments
///
/// * [inp]: The input tensor of dimensions `N, C` where `N` is the batch size and `C` the number
///   of categories. This is expected to raw logits.
/// * [target]: The ground truth labels as a tensor of u32 of dimension `N, C` where `N` is the batch size and `C` the number
///   of categories.
///
/// The resulting tensor is a scalar containing the average value over the batch.
pub fn binary_cross_entropy_with_logit(inp: &Tensor, target: &Tensor) -> Result<Tensor> {
    let inp = crate::ops::sigmoid(inp)?;

    let left_side = target * inp.log()?;
    let right_side = (target.affine(-1., 1.))? * inp.affine(-1., 1.)?.log()?;

    let loss = left_side? + right_side?;
    let loss = loss?.neg()?.mean_all()?;

    Ok(loss)
}
