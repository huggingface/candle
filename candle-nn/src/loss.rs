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

/// HuberLoss
///
/// A robust loss function that combines `MAE` and `MSE` losses:
///
/// - When the absolute element-wise error is less than `delta`, it uses a squared term (MSE loss).
/// - When the absolute element-wise error is greater than or equal to `delta`, it uses a linear term (MAE loss scaled by `delta`).
/// # Formula
///
/// HuberLoss =
/// ```tex
/// 0.5(x_n - y_n)^2, & |x_n - y_n| < delta
/// delta(|x_n - y_n| - 0.5delta), & |x_n - y_n| >= delta
/// ```
pub fn huber(inp: &Tensor, target: &Tensor, delta: f64) -> Result<Tensor> {
    if inp.dims() != target.dims() {
        candle::bail!(
            "input and target must have the same shape, got inp: {:?}, target: {:?}",
            inp.dims(),
            target.dims()
        );
    }
    let diff = (inp - target)?;
    let abs_diff = diff.abs()?;
    let mask = abs_diff.le(delta)?;
    let squared_loss = ((&diff * &diff)? * 0.5)?;
    let linear_loss = ((abs_diff * delta)? - 0.5 * delta.powi(2))?;
    let loss = mask.where_cond(&squared_loss, &linear_loss)?;
    loss.mean_all()
}
