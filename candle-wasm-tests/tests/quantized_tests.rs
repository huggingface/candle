#![allow(unused)]
use candle::{
    quantized::{self, k_quants, GgmlDType, GgmlType},
    test_utils::to_vec2_round,
    Device, Module, Result, Tensor,
};

use wasm_bindgen_test::*;
wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn quantized_matmul_neg() -> Result<()> {
    let cpu = &Device::Cpu;
    let (m, k, n) = (3, 64, 4);
    let lhs = (0..(m * k))
        .map(|v| v as f32 - (m * k) as f32 / 2.0)
        .collect::<Vec<_>>();
    let tensor_lhs = Tensor::from_slice(&lhs, (m, k), cpu)?;
    let mut dst = vec![42.; 3 * 4];
    let mut rhs_t = vec![k_quants::BlockQ4_0::zeros(); 8];
    let rhs = (0..k * n)
        .map(|v| v as f32 - (k * n) as f32 / 3.0)
        .collect::<Vec<_>>();
    let tensor_rhs = Tensor::from_slice(&rhs, (n, k), cpu)?.t()?;
    k_quants::BlockQ4_0::from_float(&rhs, &mut rhs_t)?;
    k_quants::matmul((m, k, n), &lhs, &rhs_t, &mut dst)?;
    assert_eq!(
        dst.iter().map(|x| x.round()).collect::<Vec<_>>(),
        &[
            243524.0, -19596.0, -285051.0, -549815.0, 23777.0, 21651.0, 19398.0, 18367.0,
            -196472.0, 63012.0, 324585.0, 587902.0
        ]
    );
    let mm = tensor_lhs.matmul(&tensor_rhs)?;
    assert_eq!(
        to_vec2_round(&mm, 0)?,
        &[
            [244064.0, -20128.0, -284320.0, -548512.0],
            [23563.0, 21515.0, 19467.0, 17419.0],
            [-196939.0, 63157.0, 323253.0, 583349.0]
        ]
    );

    let qtensor = quantized::QTensor::new(quantized::QStorage::Cpu(Box::new(rhs_t)), (4, 64))?;
    let matmul = quantized::QMatMul::from_qtensor(qtensor)?;
    let res = matmul.forward(&tensor_lhs)?;
    assert_eq!(
        to_vec2_round(&res, 0)?,
        &[
            [243524.0, -19596.0, -285051.0, -549815.0],
            [23777.0, 21651.0, 19398.0, 18367.0],
            [-196472.0, 63012.0, 324585.0, 587902.0]
        ]
    );

    Ok(())
}

/// Creates a vector similarly to the one used in GGML unit tests: https://github.com/ggerganov/llama.cpp/blob/master/tests/test-quantize-fns.cpp#L26-L30
fn create_ggml_like_vector(offset: f32) -> Vec<f32> {
    const GGML_TEST_SIZE: usize = 32 * 128;
    (0..GGML_TEST_SIZE)
        .map(|i| 0.1 + 2.0 * (i as f32 + offset).cos())
        .collect()
}

/// Very simple dot product implementation
fn vec_dot_reference(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// Returns the error achieved by the GGML matmul unit test.
fn ggml_reference_matmul_error(dtype: GgmlDType) -> Result<f32> {
    let err = match dtype {
        GgmlDType::F16 => 0.000010,
        GgmlDType::Q2K => 0.004086,
        GgmlDType::Q3K => 0.016148,
        GgmlDType::Q4K => 0.002425,
        GgmlDType::Q5K => 0.000740,
        GgmlDType::Q6K => 0.000952,
        GgmlDType::Q4_0 => 0.001143,
        GgmlDType::Q4_1 => 0.007784,
        GgmlDType::Q5_0 => 0.001353,
        GgmlDType::Q5_1 => 0.001363,
        GgmlDType::Q8_0 => 0.000092,

        // Not from the ggml repo.
        GgmlDType::Q8K => 0.00065,
        _ => candle::bail!("No GGML results for quantization type {dtype:?}",),
    };
    Ok(err)
}

/// Mirrores the GGML matmul unit test: https://github.com/ggerganov/llama.cpp/blob/master/tests/test-quantize-fns.cpp#L76-L91
fn ggml_matmul_error_test<T: GgmlType>() -> Result<()> {
    const GGML_MAX_DOT_PRODUCT_ERROR: f32 = 0.02;
    let a = create_ggml_like_vector(0.0);
    let b = create_ggml_like_vector(1.0);
    let length = a.len();

    let mut a_quant = vec![T::zeros(); length / T::BLCK_SIZE];
    let mut b_quant = vec![T::VecDotType::zeros(); length / T::VecDotType::BLCK_SIZE];
    T::from_float(&a, &mut a_quant)?;
    T::VecDotType::from_float(&b, &mut b_quant)?;

    let result = T::vec_dot(length, &a_quant, &b_quant)?;
    let result_unopt = T::vec_dot_unopt(length, &a_quant, &b_quant)?;
    let reference_result = vec_dot_reference(&a, &b);

    if (result - result_unopt).abs() / length as f32 > 1e-6 {
        candle::bail!(
            "the opt and unopt vec-dot returned different values, opt {result}, unopt {result_unopt}"
        )
    }

    let error = (result - reference_result).abs() / length as f32;

    let ggml_error = ggml_reference_matmul_error(T::DTYPE)?;

    if !error.is_finite() || error > GGML_MAX_DOT_PRODUCT_ERROR {
        candle::bail!(
            "Dot product error {} exceeds max error {}",
            error,
            GGML_MAX_DOT_PRODUCT_ERROR
        );
    }

    // We diverge slightly due to different rounding behavior / f16 to f32 conversions in GGML
    // => we use a slightly higher error threshold
    const ERROR_LENIENCY: f32 = 0.00001;
    if error - ERROR_LENIENCY > ggml_error {
        candle::bail!(
            "Dot product error {} exceeds ggml reference error {}",
            error,
            ggml_error
        );
    }
    Ok(())
}

#[wasm_bindgen_test]
fn quantized_matmul_q40() -> Result<()> {
    ggml_matmul_error_test::<candle::quantized::k_quants::BlockQ4_0>()?;
    Ok(())
}

#[wasm_bindgen_test]
fn quantized_matmul_q50() -> Result<()> {
    ggml_matmul_error_test::<candle::quantized::k_quants::BlockQ5_0>()?;
    Ok(())
}

#[wasm_bindgen_test]
fn quantized_matmul_q80() -> Result<()> {
    ggml_matmul_error_test::<candle::quantized::k_quants::BlockQ8_0>()?;
    Ok(())
}

#[wasm_bindgen_test]
fn quantized_matmul_q2k() -> Result<()> {
    ggml_matmul_error_test::<candle::quantized::k_quants::BlockQ2K>()?;
    Ok(())
}

#[wasm_bindgen_test]
fn quantized_matmul_q3k() -> Result<()> {
    ggml_matmul_error_test::<candle::quantized::k_quants::BlockQ3K>()?;
    Ok(())
}

#[wasm_bindgen_test]
fn quantized_matmul_q4k() -> Result<()> {
    ggml_matmul_error_test::<candle::quantized::k_quants::BlockQ4K>()?;
    Ok(())
}

#[wasm_bindgen_test]
fn quantized_matmul_q5k() -> Result<()> {
    ggml_matmul_error_test::<candle::quantized::k_quants::BlockQ5K>()?;
    Ok(())
}

#[wasm_bindgen_test]
fn quantized_matmul_q6k() -> Result<()> {
    ggml_matmul_error_test::<candle::quantized::k_quants::BlockQ6K>()?;
    Ok(())
}

#[wasm_bindgen_test]
fn quantized_matmul_q8k() -> Result<()> {
    ggml_matmul_error_test::<candle::quantized::k_quants::BlockQ8K>()?;
    Ok(())
}
