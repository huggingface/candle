#![allow(unused)]
use candle::{DType, Device, Result, Tensor};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

const LEN: usize = 20;

fn repeat(values: [f32; 4]) -> Vec<f32> {
    values.into_iter().cycle().take(LEN).collect()
}

fn as_f32(tensor: &Tensor) -> Result<Vec<f32>> {
    tensor.to_dtype(DType::F32)?.to_vec1::<f32>()
}

fn add_vectors_with_simd_block_and_tail(dtype: DType) -> Result<()> {
    let lhs = repeat([0.1, 0.3, 0.7, -0.2]);
    let rhs = repeat([0.2, -0.1, 0.05, 1.3]);

    let lhs = Tensor::from_slice(&lhs, LEN, &Device::Cpu)?.to_dtype(dtype)?;
    let rhs = Tensor::from_slice(&rhs, LEN, &Device::Cpu)?.to_dtype(dtype)?;
    let expected = lhs
        .to_dtype(DType::F32)?
        .add(&rhs.to_dtype(DType::F32)?)?
        .to_dtype(dtype)?;
    let sum = lhs.add(&rhs)?;
    assert_eq!(as_f32(&sum)?, as_f32(&expected)?);

    Ok(())
}

fn add_scalar_with_simd_block_and_tail(dtype: DType) -> Result<()> {
    let lhs = repeat([0.1, 0.3, 0.7, -0.2]);

    let lhs = Tensor::from_slice(&lhs, LEN, &Device::Cpu)?.to_dtype(dtype)?;
    let scalar = Tensor::new(0.2f32, &Device::Cpu)?.to_dtype(dtype)?;
    let expected = lhs
        .to_dtype(DType::F32)?
        .broadcast_add(&scalar.to_dtype(DType::F32)?)?
        .to_dtype(dtype)?;
    let sum = lhs.broadcast_add(&scalar)?;
    assert_eq!(as_f32(&sum)?, as_f32(&expected)?);

    Ok(())
}

#[wasm_bindgen_test]
fn add_f16_vectors_with_simd128() -> Result<()> {
    add_vectors_with_simd_block_and_tail(DType::F16)
}

#[wasm_bindgen_test]
fn add_f16_scalar_with_simd128() -> Result<()> {
    add_scalar_with_simd_block_and_tail(DType::F16)
}

#[wasm_bindgen_test]
fn add_bf16_vectors_with_simd128() -> Result<()> {
    add_vectors_with_simd_block_and_tail(DType::BF16)
}

#[wasm_bindgen_test]
fn add_bf16_scalar_with_simd128() -> Result<()> {
    add_scalar_with_simd_block_and_tail(DType::BF16)
}
