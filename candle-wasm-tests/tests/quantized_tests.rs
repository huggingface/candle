use candle::{
    quantized::{self, k_quants, GgmlType},
    test_utils::to_vec2_round,
    Device, Result, Tensor,
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

    let qtensor = quantized::QTensor::new(rhs_t, (4, 64))?;
    let matmul = quantized::QMatMul::from_qtensor(qtensor);
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
