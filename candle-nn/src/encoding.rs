//! Encoding Utilities. (e.g., one-hot/cold encoding)

use candle::{bail, DType, Result, Tensor, WithDType};

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
/// `[[0i64, 2], [1, -1]]`
///
/// with a depth of 4 will be encoded to:
///
/// `[[[1, 0, 0, 0], [0, 0, 1, 0]], [[0, 1, 0, 0], [0, 0, 0, 0]]]`
///
/// When the input tensor index has a value of -1, the corresponding one-hot vector will be ignored,
/// resulting in a vector of values set to the `off_value`.
///
///
/// This method supports one-cold encoding by setting `on_value` to `0` and `off_value` to `1`.
/// By default `on_value` is `1` and `off_value` is `0`.
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
/// let indices = Tensor::new(vec![vec![0i64, 2], vec![1, -1]], &device).unwrap();
/// let depth = 4;
/// let one_hot = one_hot(indices, depth, 1f32, 0f32).unwrap();
///
/// let expected_matrix = [
///     [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
///     [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
/// ];
///
/// assert_eq!(one_hot.shape(), &Shape::from((2, 2, depth)));
///
/// let matrix = one_hot.to_vec3::<f32>().unwrap();
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
/// let indices = Tensor::new(vec![vec![0u8, 2], vec![1, 3]], &device).unwrap();
/// let one_cold = one_hot(indices, depth, 0u8, 1u8).unwrap();
///
/// let expected_matrix = [[[0, 1, 1, 1], [1, 1, 0, 1]], [[1, 0, 1, 1], [1, 1, 1, 0]]];
///
/// assert_eq!(one_cold.shape(), &Shape::from((2, 2, depth)));
///
/// let matrix = one_cold.to_vec3::<u8>().unwrap();
///
/// assert_eq!(matrix, expected_matrix);
/// ```
///
///
/// # Bails
///
/// This method bails if:
/// - One of the index value is less than -1.
/// - One of the index value is greater than or equal to the depth value.
/// - The input data type is not `U8`, `U32`, or `I64`.
///
/// # API Design
///
/// The api design for this method is loosely based on the [TensorFlow One-Hot](https://www.tensorflow.org/api_docs/python/tf/one_hot) method.
pub fn one_hot<D: WithDType>(
    indices: Tensor,
    depth: usize,
    on_value: D,
    off_value: D,
) -> Result<Tensor> {
    let mut target_shape = indices.dims().to_vec();
    target_shape.push(depth);
    let indices = indices.flatten_all()?;
    let mut out = vec![off_value; depth * indices.elem_count()];
    match indices.dtype() {
        DType::U8 => {
            let indices = indices.to_vec1::<u8>()?;
            for (i, &index) in indices.iter().enumerate() {
                set_at_index(index, i * depth, depth, &mut out, on_value)?;
            }
        }
        DType::U32 => {
            let indices = indices.to_vec1::<u32>()?;
            for (i, &index) in indices.iter().enumerate() {
                set_at_index(index, i * depth, depth, &mut out, on_value)?;
            }
        }
        DType::I64 => {
            let indices = indices.to_vec1::<i64>()?;
            for (i, &index) in indices.iter().enumerate() {
                set_at_index(index, i * depth, depth, &mut out, on_value)?;
            }
        }
        dtype => {
            bail!("one_hot: unsupported data type {dtype:?}, expected U8, U32, or I64")
        }
    };
    Tensor::from_vec(out, target_shape, indices.device())
}

fn set_at_index<D: WithDType, I: Into<i64>>(
    value: I,
    offset: usize,
    depth: usize,
    v: &mut [D],
    on_value: D,
) -> Result<()> {
    let value = value.into();
    // Skip for an entire row of off_values
    if value == -1 {
        return Ok(());
    }
    if value < -1 {
        bail!(
            "one_hot: invalid negative index value {value}, expected a positive index value or -1"
        );
    }
    let value = value as usize;
    if value >= depth {
        bail!("one_hot: index value {value} exceeds depth {depth}")
    }
    let idx = offset + value;
    if idx >= v.len() {
        bail!("one_hot: index out of bounds {idx}, len {}", v.len());
    }
    v[idx] = on_value;
    Ok(())
}
