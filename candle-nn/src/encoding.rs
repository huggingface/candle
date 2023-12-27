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
//! let device = Device::Cpu;
//! let depth = 4;
//! let indices = Tensor::new(vec![vec![0f32, 2.], vec![1., -1.]], &device).unwrap();
//!
//! let on_value = Some(1.0); // default
//! let off_value = Some(0.0); // default
//!
//! let one_hot = one_hot::<f32>(indices, depth, on_value, off_value).unwrap();
//! let expected_matrix = [
//!     [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
//!     [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
//! ];
//! assert_eq!(one_hot.shape(), &Shape::from((2, 2, 4)));
//!
//! let matrix = one_hot.to_vec3::<f32>().unwrap();
//!
//! assert_eq!(matrix, expected_matrix);
//! ```
//!

use candle::{bail, Result, Tensor, WithDType};

/// One-hot/cold encoding.
///
/// Given an input tensor of indices, this function returns a tensor of the same shape as the input
/// tensor with an additional dimension of the given depth size. The values in the returned tensor are
/// all zeros except for the positions represented by the indices.
///
/// This method returns a tensor with a rank that is one larger than the input tensor.
///
/// As an example, the following tensor will be converted to a one-hot matrix:
///
/// ```
/// [[0., 2.], [1., -1.]]
/// ```
///
/// with a depth of 4 will be converted to:
///
/// ```
/// [[[1., 0., 0., 0.], [0., 0., 1., 0.]], [[0., 1., 0., 0.], [0., 0., 0., 0.]]]
/// ```
///
/// When the input tensor index has a value of -1, the corresponding one-hot vector will be all zeros.
///
///
/// This method supports one-cold encoding by setting `on_value` to `0.0` and `off_value` to `1.0`.
/// By default `on_value` is `1.0` and `off_value` is `0.0`.
///
/// Other encoding values can be used by setting `on_value` and `off_value` to the desired values.
///
/// # Examples
///
///```rust
/// use candle::{Shape, Tensor, Device};
/// use candle_nn::encoding::one_hot;
///
/// let device = Device::Cpu;
/// let depth = 4;
/// let indices = Tensor::new(vec![vec![0f32, 2.], vec![1., -1.]], &device).unwrap();
///
/// let on_value = Some(1.0); // default
/// let off_value = Some(0.0); // default
///
/// let one_hot = one_hot::<f32>(indices, depth, on_value, off_value).unwrap();
/// let expected_matrix = [
///     [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
///     [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
/// ];
/// assert_eq!(one_hot.shape(), &Shape::from((2, 2, 4)));
///
/// let matrix = one_hot.to_vec3::<f32>().unwrap();
///
/// assert_eq!(matrix, expected_matrix);
/// ```
///
/// # Bails
///
/// This method will bail on tensors with a rank greater than 3.
///
/// # API Design
///
/// The api design for this method is loosely based on the [TensorFlow One-Hot](https://www.tensorflow.org/api_docs/python/tf/one_hot) method.
pub fn one_hot<D: WithDType>(
    indices: Tensor,
    depth: usize,
    on_value: Option<D>,
    off_value: Option<D>,
) -> Result<Tensor> {
    let on_value = on_value.unwrap_or(D::from_f64(1.0));
    let off_value = off_value.unwrap_or(D::from_f64(0.0));

    let rank = indices.rank();
    let dims = indices.dims();
    match rank {
        0 => {
            let mut v = vec![off_value; depth];
            let index = indices.to_vec0::<f32>()?;
            v[index as usize] = on_value;

            Tensor::new(v, indices.device())
        }
        1 => {
            let mut v = vec![off_value; depth * dims[0]];
            let index = indices.to_vec1::<f32>()?;
            for i in 0..dims[0] {
                if index[i] < 0.0 {
                    continue;
                }

                v[(i * depth + index[i] as usize) as usize] = on_value;
            }

            Tensor::new(v, indices.device())?.reshape(&[dims[0], depth])
        }
        2 => {
            let mut v = vec![off_value; depth * dims[0] * dims[1]];
            let index = indices.to_vec2::<f32>()?;
            for i in 0..dims[0] {
                for j in 0..dims[1] {
                    if index[i][j] < 0.0 {
                        continue;
                    }

                    v[(i * depth * dims[1] + j * depth + index[i][j] as usize) as usize] = on_value;
                }
            }

            Tensor::new(v, indices.device())?.reshape(&[dims[0], dims[1], depth])
        }
        3 => {
            let mut v = vec![off_value; depth * dims[0] * dims[1] * dims[2]];
            let index = indices.to_vec3::<f32>()?;
            for i in 0..dims[0] {
                for j in 0..dims[1] {
                    for k in 0..dims[2] {
                        if index[i][j][k] < 0.0 {
                            continue;
                        }

                        v[(i * depth * dims[1] * dims[2]
                            + j * depth * dims[2]
                            + k * depth
                            + index[i][j][k] as usize) as usize] = on_value;
                    }
                }
            }

            Tensor::new(v, indices.device())?.reshape(&[dims[0], dims[1], dims[2], depth])
        }
        _ => {
            bail!("one_hot: rank {} is not supported", rank)
        }
    }
}

#[cfg(test)]
mod tests {
    use candle::Shape;

    use super::*;

    #[test]
    pub fn test_one_hot() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let device = candle::Device::Cpu;
        let depth = 4;
        let indices = Tensor::new(vec![vec![0f32, 2.], vec![1., -1.]], &device)?;

        let one_hot = one_hot::<f32>(indices, depth, None, None)?;

        let expected_matrix = [
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        ];

        assert_eq!(one_hot.shape(), &Shape::from((2, 2, 4)));

        let matrix = one_hot.to_vec3::<f32>()?;

        assert_eq!(matrix, expected_matrix);

        Ok(())
    }
}
