use ::candle::{Error, Result};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub fn wrap_err(err: ::candle::Error) -> PyErr {
    PyErr::new::<PyValueError, _>(format!("{err:?}"))
}

/// Checks if the given shape is compatible with the given layout and returns an error if not.
pub fn can_broadcast(lhs: &::candle::Tensor, rhs: &::candle::Tensor) -> Result<()> {
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
            return Err(Error::BroadcastIncompatibleShapes {
                src_shape: lhs.shape().clone(),
                dst_shape: rhs.shape().clone(),
            }
            .bt());
        }
    }
    Ok(())
}

/// Check if we need to broadcast the lhs tensor into the rhs tensor
pub fn must_broadcast(lhs: &::candle::Tensor, rhs: &::candle::Tensor) -> Result<bool> {
    let rhs_shape = rhs.shape();
    let lhs_shape = lhs.shape();

    // first check if the shapes are equal
    if lhs_shape == rhs_shape {
        return Ok(false);
    }

    can_broadcast(lhs, rhs)?;

    Ok(true)
}
