use candle::Result;
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder};

// Applies weight norm for inference by recomputing the weight tensor. This
// does not apply to training.
// https://pytorch.org/docs/stable/generated/torch.nn.utils.weight_norm.html
pub fn conv1d_weight_norm(
    in_c: usize,
    out_c: usize,
    kernel_size: usize,
    config: Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let weight_g = vb.get((out_c, 1, 1), "weight_g")?;
    let weight_v = vb.get((out_c, in_c, kernel_size), "weight_v")?;
    let norm_v = weight_v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    let weight = weight_v.broadcast_mul(&weight_g)?.broadcast_div(&norm_v)?;
    let bias = vb.get(out_c, "bias")?;
    Ok(Conv1d::new(weight, Some(bias), config))
}
