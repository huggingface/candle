//! Tensor operation utilities.

use candle::{Result, Tensor};

/// Perform L2 normalization along the last dimension.
///
/// This function normalizes each row vector to have unit L2 norm, which is commonly
/// used in embedding and similarity computations.
///
/// # Example  
/// ```
/// use candle_core::Tensor;
/// use candle_utils::tensor::normalize_l2;
///
/// let device = candle_utils::get_device(true, false).unwrap();
/// let x = Tensor::new(&[[0f32, 1.], [2., 3.]], &device).unwrap();
/// let normalized_x = normalize_l2(&x).unwrap();
/// ```
pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    let squared = &v.sqr()?;
    let summed = &squared.sum_keepdim(1)?;
    let norms = &summed.sqrt()?;
    v.broadcast_div(norms)
}
