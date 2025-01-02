#![allow(unused_imports, unexpected_cfgs)]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;
#[cfg(not(target_arch = "wasm32"))]
use tokio::test as test;
use candle_wasm_tests::{
    to_vec0_round_async, to_vec1_round_async, to_vec2_round_async, to_vec3_round_async,
};
use candle::{Result, Shape, Tensor};
use candle_nn::encoding::one_hot_async;
#[test]
async fn test_i64_one_hot() -> Result<()> {
    let device = candle::Device::Cpu;
    let indices = Tensor::new(vec![vec![0i64, 2], vec![1, - 1]], &device)?;
    let depth = 4;
    let on_value = 1.0;
    let off_value = 0.0;
    let one_hot = one_hot_async::<f32>(indices, depth, on_value, off_value).await?;
    let expected_matrix = [
        [[1., 0., 0., 0.], [0., 0., 1., 0.]],
        [[0., 1., 0., 0.], [0., 0., 0., 0.]],
    ];
    assert_eq!(one_hot.shape(), & Shape::from((2, 2, depth)));
    let matrix = one_hot.to_vec3_async::<f32>().await?;
    assert_eq!(matrix, expected_matrix);
    Ok(())
}
#[test]
async fn test_rank_3_one_hot() -> Result<()> {
    let device = candle::Device::Cpu;
    let indices = Tensor::new(
        vec![vec![vec![0i64, 1], vec![2, 3]], vec![vec![3, 1], vec![1, - 1]],],
        &device,
    )?;
    let depth = 4;
    let on_value = 1.0;
    let off_value = 0.0;
    let one_hot = one_hot_async::<f32>(indices, depth, on_value, off_value).await?;
    let expected_matrix = Tensor::new(
        vec![
            vec![vec![vec![1f32, 0., 0., 0.], vec![0., 1., 0., 0.]], vec![vec![0., 0.,
            1., 0.], vec![0., 0., 0., 1.]],], vec![vec![vec![0., 0., 0., 1.], vec![0.,
            1., 0., 0.]], vec![vec![0., 1., 0., 0.], vec![0., 0., 0., 0.]],],
        ],
        &device,
    )?;
    assert_eq!(one_hot.shape(), expected_matrix.shape());
    assert_eq!(one_hot.dims(), expected_matrix.dims());
    let matrix = one_hot.get(1)?.to_vec3_async::<f32>().await?;
    let expected_matrix = expected_matrix.get(1)?.to_vec3_async::<f32>().await?;
    assert_eq!(matrix, expected_matrix);
    Ok(())
}
#[test]
async fn test_u8_one_cold() -> Result<()> {
    let device = candle::Device::Cpu;
    let depth = 4;
    let indices = Tensor::new(vec![vec![0i64, 2], vec![1, - 1]], &device)?;
    let on_value = 0u8;
    let off_value = 1;
    let one_cold = one_hot_async(indices, depth, on_value, off_value).await?;
    let expected_matrix = [[[0, 1, 1, 1], [1, 1, 0, 1]], [[1, 0, 1, 1], [1, 1, 1, 1]]];
    assert_eq!(one_cold.shape(), & Shape::from((2, 2, depth)));
    let matrix = one_cold.to_vec3_async::<u8>().await?;
    assert_eq!(matrix, expected_matrix);
    Ok(())
}
#[test]
async fn test_iter() -> Result<()> {
    let device = candle::Device::Cpu;
    let depth = 4;
    let indices = Tensor::new(vec![vec![0i64, 2], vec![1, - 1]], &device)?;
    let matrix = indices.to_vec2_async::<i64>().await?;
    let (dim1, dim2) = indices.dims2()?;
    let iter = (0..dim1).flat_map(|i| (0..dim2).map(move |j| (i, j)));
    let mut v = vec![0; depth * dim1 * dim2];
    for (i, j) in iter {
        let idx = i * depth * dim2 + j * depth;
        v[idx] = matrix[i][j];
    }
    for (i, row) in matrix.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            let idx = i * depth * dim2 + j * depth;
            assert_eq!(v[idx], value);
        }
    }
    Ok(())
}
