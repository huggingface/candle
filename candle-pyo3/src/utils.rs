use ::candle::{Error as CandleError, Result as CandleResult};
use candle::Shape;

/// Tries to broadcast the `rhs` shape into the `lhs` shape.
pub fn broadcast_shapes(lhs: &::candle::Tensor, rhs: &::candle::Tensor) -> CandleResult<Shape> {
    // see `Shape.broadcast_shape_binary_op`
    let lhs_dims = lhs.dims();
    let rhs_dims = rhs.dims();
    let lhs_ndims = lhs_dims.len();
    let rhs_ndims = rhs_dims.len();
    let bcast_ndims = usize::max(lhs_ndims, rhs_ndims);
    let mut bcast_dims = vec![0; bcast_ndims];
    for (idx, bcast_value) in bcast_dims.iter_mut().enumerate() {
        let rev_idx = bcast_ndims - idx;
        let l_value = if lhs_ndims < rev_idx {
            1
        } else {
            lhs_dims[lhs_ndims - rev_idx]
        };
        let r_value = if rhs_ndims < rev_idx {
            1
        } else {
            rhs_dims[rhs_ndims - rev_idx]
        };
        *bcast_value = if l_value == r_value {
            l_value
        } else if l_value == 1 {
            r_value
        } else if r_value == 1 {
            l_value
        } else {
            return Err(CandleError::BroadcastIncompatibleShapes {
                src_shape: lhs.shape().clone(),
                dst_shape: rhs.shape().clone(),
            }
            .bt());
        }
    }
    Ok(Shape::from(bcast_dims))
}
