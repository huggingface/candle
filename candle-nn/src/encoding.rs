//! Encoding Utilities. (e.g., one-hot/cold encoding)
//!
//! # Examples
//!
//! ## One-hot encoding
//!
//! ```rust
//! use candle::{Shape, Tensor, Device};
//! use candle_nn::encoding::one_hot;
//!
//! let device = candle::Device::Cpu;
//!
//! let indices = Tensor::new(vec![vec![0f32, 2.], vec![1., -1.]], &device)?;
//! let depth = 4;
//!
//! let allow_f64 = true;
//! let on_value = None; // defaults to 1.0
//! let off_value = None; // defaults to 0.0
//!
//! let one_hot = one_hot::<f32, f32>(indices, depth, allow_f64, on_value, off_value)?;
//!
//! let expected_matrix = [
//!     [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
//!     [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
//! ];
//!
//! assert_eq!(one_hot.shape(), &Shape::from((2, 2, depth)));
//!
//! let matrix = one_hot.to_vec3::<f32>()?;
//!
//! assert_eq!(matrix, expected_matrix);
//!```
//! ## One-cold Encoding
//!
//! ```rust
//! use candle::{Shape, Tensor, Device};
//! use candle_nn::encoding::one_hot;
//!
//!
//! let device = candle::Device::Cpu;
//! let depth = 4;
//! let indices = Tensor::new(vec![vec![0i64, 2], vec![1, 3]], &device)?;
//!
//! let allow_f64 = false;
//! let on_value = Some(0);
//! let off_value = Some(1);
//!
//! let one_cold = one_hot::<i64, u8>(indices, depth, allow_f64, on_value, off_value)?;
//!
//! let expected_matrix = [[[0, 1, 1, 1], [1, 1, 0, 1]], [[1, 0, 1, 1], [1, 1, 1, 0]]];
//!
//! assert_eq!(one_cold.shape(), &Shape::from((2, 2, depth)));
//!
//! let matrix = one_cold.to_vec3::<u8>()?;
//!
//! assert_eq!(matrix, expected_matrix);
//! ```
//!

use candle::{bail, Result, Tensor, WithDType};

const INVALID_ONE_HOT_INDEX_MSG: &str =
    "one_hot: Invalid negative index value. Expected a positive index value or `-1` value to ignore.";

const INDEX_OUT_OF_BOUNDS_MSG: &str = "one_hot: Index out of bounds.";

const INDEX_EXCEEDS_DEPTH_MSG: &str = "one_hot: Index value exceeds the depth value.";

/// One-hot/cold encoding.
///
/// Given an input tensor of indices, this function returns a tensor of the same shape as the input
/// tensor with an additional dimension of the given depth size. The values in the returned tensor are
/// all set to the `off_value` except for the positions represented by the indices, which are set to the `on_value`.
///
/// This method returns a tensor with a rank that is one rank larger than the input tensor.
///
/// As an example, the following tensor will be encoded to a one-hot matrix:
///
/// ```
/// [[0., 2.], [1., -1.]]
/// ```
///
/// with a depth of 4 will be encoded to:
///
/// ```
/// [[[1., 0., 0., 0.], [0., 0., 1., 0.]], [[0., 1., 0., 0.], [0., 0., 0., 0.]]]
/// ```
///
/// When the input tensor index has a value of -1, the corresponding one-hot vector will be ignored,
/// resulting in a vector of values set to the `off_value`.
///
///
/// This method supports one-cold encoding by setting `on_value` to `0.0` and `off_value` to `1.0`.
/// By default `on_value` is `1.0` and `off_value` is `0.0`.
///
/// Other encoding values can be used by setting `on_value` and `off_value` to the desired values.
///
/// # Examples
///
/// ## One-hot encoding
///
/// ```rust
/// use candle::{Shape, Tensor, Device};
/// use candle_nn::encoding::one_hot;
///
/// let device = candle::Device::Cpu;
///
/// let indices = Tensor::new(vec![vec![0f32, 2.], vec![1., -1.]], &device)?;
/// let depth = 4;
///
/// let allow_f64 = true;
/// let on_value = None; // defaults to 1.0
/// let off_value = None; // defaults to 0.0
///
/// let one_hot = one_hot::<f32, f32>(indices, depth, allow_f64, on_value, off_value)?;
///
/// let expected_matrix = [
///     [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
///     [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
/// ];
///
/// assert_eq!(one_hot.shape(), &Shape::from((2, 2, depth)));
///
/// let matrix = one_hot.to_vec3::<f32>()?;
///
/// assert_eq!(matrix, expected_matrix);
///```
/// ## One-cold Encoding
///
/// ```rust
/// use candle::{Shape, Tensor, Device};
/// use candle_nn::encoding::one_hot;
///
///
/// let device = candle::Device::Cpu;
/// let depth = 4;
/// let indices = Tensor::new(vec![vec![0i64, 2], vec![1, 3]], &device)?;
///
/// let allow_f64 = false;
/// let on_value = Some(0);
/// let off_value = Some(1);
///
/// let one_cold = one_hot::<i64, u8>(indices, depth, allow_f64, on_value, off_value)?;
///
/// let expected_matrix = [[[0, 1, 1, 1], [1, 1, 0, 1]], [[1, 0, 1, 1], [1, 1, 1, 0]]];
///
/// assert_eq!(one_cold.shape(), &Shape::from((2, 2, depth)));
///
/// let matrix = one_cold.to_vec3::<u8>()?;
///
/// assert_eq!(matrix, expected_matrix);
/// ```
///
///
/// # Bails
///
/// This method will bail on tensors with a rank greater than 3.
///
/// This method will bail if the index value is less than -1.
///
/// This method will bail if the index value is greater than or equal to the depth value.
///
/// This method will bail if the input data type does not equal the generic data type provided.
///
/// This method will bail if the input data type is `f64` and `allow_f64` is false.
///
/// # API Design
///
/// The api design for this method is loosely based on the [TensorFlow One-Hot](https://www.tensorflow.org/api_docs/python/tf/one_hot) method.
pub fn one_hot<I: WithDType, O: WithDType>(
    indices: Tensor,
    depth: usize,
    allow_f64: bool,
    on_value: Option<O>,
    off_value: Option<O>,
) -> Result<Tensor> {
    let on_value = on_value.unwrap_or(O::from_f64(1.));
    let off_value = off_value.unwrap_or(O::from_f64(0.));

    if I::DTYPE != indices.dtype() {
        bail!(
            "one_hot: indices dtype {} does not match expected dtype {}",
            indices.dtype().as_str(),
            I::DTYPE.as_str()
        )
    }

    if !allow_f64 && I::DTYPE.as_str() == "f64" {
        bail!("one_hot: f64 is not supported")
    }

    let rank = indices.rank();
    match rank {
        0 => {
            let mut v = vec![off_value; depth];
            let index = indices.to_vec0::<I>()?;
            let vi = index.to_f64();

            check_value(vi, depth)?;

            if vi >= 0. {
                v[vi as usize] = on_value;
            }

            Tensor::new(v, indices.device())
        }
        1 => {
            let dim1 = indices.dims1()?;
            let mut v = vec![off_value; depth * dim1];
            let index = indices.to_vec1::<I>()?;
            for i in 0..dim1 {
                let vi = index[i].to_f64();

                if vi == -1. {
                    continue;
                }

                check_value(vi, depth)?;

                let idx = i * depth + vi as usize;

                if idx >= v.len() {
                    bail!(
                        "{} Expected index value, {}, to be less than {}.",
                        INDEX_OUT_OF_BOUNDS_MSG,
                        idx,
                        v.len()
                    );
                }

                v[idx] = on_value;
            }

            Tensor::new(v, indices.device())?.reshape(&[dim1, depth])
        }
        2 => {
            let (dim1, dim2) = indices.dims2()?;
            let mut v = vec![off_value; depth * dim1 * dim2];
            let index = indices.to_vec2::<I>()?;
            for i in 0..dim1 {
                for j in 0..dim2 {
                    let vij = index[i][j].to_f64();

                    if vij == -1. {
                        continue;
                    }

                    check_value(vij, depth)?;

                    let idx = i * depth * dim2 + j * depth + vij as usize;

                    if idx >= v.len() {
                        bail!(
                            "{} Expected index value, {}, to be less than {}.",
                            INDEX_OUT_OF_BOUNDS_MSG,
                            idx,
                            v.len()
                        );
                    }

                    v[idx] = on_value;
                }
            }

            Tensor::new(v, indices.device())?.reshape(&[dim1, dim2, depth])
        }
        3 => {
            let (dim1, dim2, dim3) = indices.dims3()?;
            let mut v = vec![off_value; depth * dim1 * dim2 * dim3];
            let index = indices.to_vec3::<I>()?;
            for i in 0..dim1 {
                for j in 0..dim2 {
                    for k in 0..dim3 {
                        let vijk = index[i][j][k].to_f64();

                        if vijk == -1. {
                            continue;
                        }

                        check_value(vijk, depth)?; // bail on invalid index value

                        let idx =
                            i * depth * dim2 * dim3 + j * depth * dim3 + k * depth + vijk as usize;

                        if idx >= v.len() {
                            bail!(
                                "{} Expected index value, {}, to be less than {}.",
                                INDEX_OUT_OF_BOUNDS_MSG,
                                idx,
                                v.len()
                            );
                        }

                        v[idx] = on_value;
                    }
                }
            }

            Tensor::new(v, indices.device())?.reshape(&[dim1, dim2, dim3, depth])
        }
        _ => {
            bail!("one_hot: rank {} is not supported. ", rank)
        }
    }
}

fn check_value(value: f64, depth: usize) -> Result<()> {
    if value < -1. {
        bail!("{}. Received {}", INVALID_ONE_HOT_INDEX_MSG, value);
    }

    if value as usize >= depth {
        bail!(
            "{} Index value, {}, exceeds the depth value, {}.",
            INDEX_EXCEEDS_DEPTH_MSG,
            value,
            depth
        )
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use candle::Shape;

    use super::*;

    #[test]
    pub fn test_f64_one_hot() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let device = candle::Device::Cpu;

        let indices = Tensor::new(vec![vec![0f32, 2.], vec![1., -1.]], &device)?;
        let depth = 4;

        let allow_f64 = true;
        let on_value = None; // defaults to 1.0
        let off_value = None; // defaults to 0.0

        let one_hot = one_hot::<f32, f32>(indices, depth, allow_f64, on_value, off_value)?;

        let expected_matrix = [
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        ];

        assert_eq!(one_hot.shape(), &Shape::from((2, 2, depth)));

        let matrix = one_hot.to_vec3::<f32>()?;

        assert_eq!(matrix, expected_matrix);

        Ok(())
    }

    #[test]
    pub fn test_u8_one_cold() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let device = candle::Device::Cpu;
        let depth = 4;
        let indices = Tensor::new(vec![vec![0i64, 2], vec![1, -1]], &device)?;

        let allow_f64 = false;
        let on_value = Some(0);
        let off_value = Some(1);

        let one_cold = one_hot::<i64, u8>(indices, depth, allow_f64, on_value, off_value)?;

        let expected_matrix = [[[0, 1, 1, 1], [1, 1, 0, 1]], [[1, 0, 1, 1], [1, 1, 1, 1]]];

        assert_eq!(one_cold.shape(), &Shape::from((2, 2, depth)));

        let matrix = one_cold.to_vec3::<u8>()?;

        assert_eq!(matrix, expected_matrix);

        Ok(())
    }
}
