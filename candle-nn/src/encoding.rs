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
/// - The input tensor has a rank greater than 3.
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
    let dtype = indices.dtype();
    let rank = indices.rank();

    match rank {
        0 => {
            let mut v = vec![off_value; depth];
            match dtype {
                DType::U8 => {
                    let vi = indices.to_vec0::<u8>()?;
                    set_usize_value(vi as usize, 0, depth, &mut v, on_value)?;
                }
                DType::U32 => {
                    let vi = indices.to_vec0::<u32>()?;
                    set_usize_value(vi as usize, 0, depth, &mut v, on_value)?;
                }
                DType::I64 => {
                    let vi = indices.to_vec0::<i64>()?;
                    set_int64_value(vi, 0, depth, &mut v, on_value)?;
                }
                d => unsupported_dtype(d)?,
            };
            Tensor::from_vec(v, (depth,), indices.device())
        }
        1 => {
            let dim1 = indices.dims1()?;
            let mut v = vec![off_value; depth * dim1];

            match dtype {
                DType::U8 => {
                    let indices = indices.to_vec1::<i64>()?;
                    for (i, &index) in indices.iter().enumerate() {
                        set_usize_value(index as usize, i * depth, depth, &mut v, on_value)?;
                    }
                }
                DType::U32 => {
                    let indices = indices.to_vec1::<i64>()?;
                    for (i, &index) in indices.iter().enumerate() {
                        set_usize_value(index as usize, i * depth, depth, &mut v, on_value)?;
                    }
                }
                DType::I64 => {
                    let indices = indices.to_vec1::<i64>()?;
                    for (i, &index) in indices.iter().enumerate() {
                        set_int64_value(index, i * depth, depth, &mut v, on_value)?;
                    }
                }
                d => unsupported_dtype(d)?,
            };
            Tensor::from_vec(v, (dim1, depth), indices.device())
        }
        2 => {
            let (dim1, dim2) = indices.dims2()?;
            let mut v = vec![off_value; depth * dim1 * dim2];
            let idx = |i: usize, j: usize, depth: usize, dim2: usize| -> usize {
                i * depth * dim2 + j * depth
            };
            let iter = (0..dim1).flat_map(|i| (0..dim2).map(move |j| (i, j)));
            match dtype {
                DType::U8 => {
                    let index = indices.to_vec2::<u8>()?;
                    for (i, j) in iter {
                        set_usize_value(
                            index[i][j] as usize,
                            idx(i, j, depth, dim2),
                            depth,
                            &mut v,
                            on_value,
                        )?;
                    }
                }
                DType::U32 => {
                    let index = indices.to_vec2::<u32>()?;
                    for (i, j) in iter {
                        set_usize_value(
                            index[i][j] as usize,
                            idx(i, j, depth, dim2),
                            depth,
                            &mut v,
                            on_value,
                        )?;
                    }
                }
                DType::I64 => {
                    let index = indices.to_vec2::<i64>()?;
                    for (i, j) in iter {
                        set_int64_value(
                            index[i][j],
                            idx(i, j, depth, dim2),
                            depth,
                            &mut v,
                            on_value,
                        )?;
                    }
                }
                d => unsupported_dtype(d)?,
            };
            Tensor::from_vec(v, (dim1, dim2, depth), indices.device())
        }
        3 => {
            let (dim1, dim2, dim3) = indices.dims3()?;
            let mut v = vec![off_value; depth * dim1 * dim2 * dim3];
            let idx =
                |i: usize, j: usize, k: usize, depth: usize, dim2: usize, dim3: usize| -> usize {
                    i * depth * dim2 * dim3 + j * depth * dim3 + k * depth
                };
            let iter = (0..dim1)
                .flat_map(|i| (0..dim2).flat_map(move |j| (0..dim3).map(move |k| (i, j, k))));
            match dtype {
                DType::U8 => {
                    let index = indices.to_vec3::<u8>()?;
                    for (i, j, k) in iter {
                        set_usize_value(
                            index[i][j][k] as usize,
                            idx(i, j, k, depth, dim2, dim3),
                            depth,
                            &mut v,
                            on_value,
                        )?;
                    }
                }
                DType::U32 => {
                    let index = indices.to_vec3::<u32>()?;
                    for (i, j, k) in iter {
                        set_usize_value(
                            index[i][j][k] as usize,
                            idx(i, j, k, depth, dim2, dim3),
                            depth,
                            &mut v,
                            on_value,
                        )?;
                    }
                }
                DType::I64 => {
                    let index = indices.to_vec3::<i64>()?;
                    for (i, j, k) in iter {
                        set_int64_value(
                            index[i][j][k],
                            idx(i, j, k, depth, dim2, dim3),
                            depth,
                            &mut v,
                            on_value,
                        )?;
                    }
                }
                d => unsupported_dtype(d)?,
            };
            Tensor::from_vec(v, (dim1, dim2, dim3, depth), indices.device())
        }
        _ => {
            bail!("one_hot: rank {} is not supported.", rank)
        }
    }
}

fn unsupported_dtype(dtype: DType) -> Result<()> {
    bail!("one_hot: unsupported data type {dtype:?}, expected U8, U32, or I64")
}

// Set unsigned usize index values to the given value.
fn set_usize_value<D: WithDType>(
    value: usize,
    idx: usize,
    depth: usize,
    v: &mut Vec<D>,
    on_value: D,
) -> Result<()> {
    if value >= depth {
        bail!("one_hot: index value {value} exceeds depth {depth}")
    }
    let idx = idx + value;
    if idx >= v.len() {
        bail!("one_hot: index out of bounds {idx}, len {}", v.len());
    }
    v[idx] = on_value;
    Ok(())
}

// Set signed integer index values to the given value.
// Signed integer values are only permitted for `-1` values.
// Otherwise, the value must be positive for unsigned usize values.
// This method will only case i64 values to usize values if the value is positive,
// otherwise the method will bail.
fn set_int64_value<D: WithDType>(
    value: i64,
    idx: usize,
    depth: usize,
    v: &mut Vec<D>,
    on_value: D,
) -> Result<()> {
    // Skip for an entire row of off_values
    if value == -1 {
        return Ok(());
    }
    if value < -1 {
        bail!(
            "one_hot: invalid negative index value {value}, expected a positive index value or -1"
        );
    }
    set_usize_value(value as usize, idx, depth, v, on_value)
}
