use candle_core::{
    bail,
    quantized::{self, GgmlDType},
    test_device,
    test_utils::to_vec2_round,
    DType, Device, IndexOp, Module, Result, Tensor, Var,
};
use quantized::{k_quants, GgmlType};
use rand::prelude::*;
use std::borrow::Cow;

const GGML_TEST_SIZE: usize = 32 * 128;

const GGML_MAX_QUANTIZATION_TOTAL_ERROR: f32 = 0.002;
const GGML_MAX_QUANTIZATION_TOTAL_ERROR_2BITS: f32 = 0.0075;
const GGML_MAX_QUANTIZATION_TOTAL_ERROR_3BITS: f32 = 0.0040;
const GGML_MAX_DOT_PRODUCT_ERROR: f32 = 0.02;

fn test_matmul(
    device: &Device,
    (b, m, n, k): (usize, usize, usize, usize),
    dtype: GgmlDType,
) -> Result<()> {
    if (device.is_cuda() || device.is_metal())
        && (dtype == GgmlDType::Q8_1 || dtype == GgmlDType::Q8K)
    {
        return Ok(());
    }

    let lhs = (0..(m * k))
        .map(|v| v as f32 / (m * k) as f32)
        .collect::<Vec<_>>();
    let rhs = (0..(k * n))
        .map(|v| v as f32 / (n * k) as f32)
        .collect::<Vec<_>>();

    let lhs = Tensor::from_slice(&lhs, (m, k), device)?;
    let rhs = Tensor::from_slice(&rhs, (k, n), device)?;
    let mm = lhs.matmul(&rhs)?;
    let qtensor = quantized::QTensor::quantize(&rhs.t()?, dtype)?;
    let matmul = quantized::QMatMul::from_qtensor(qtensor)?;
    let res = matmul.forward(&lhs)?;

    let error: f32 = ((&mm - &res)?.abs()? / &mm.abs()?)?
        .sum_all()?
        .to_scalar()?;
    let error = error / (b * m * n) as f32;
    assert!(
        error <= 0.02,
        "Error {error} is too big. \nExpected:\n {mm} \nFound:\n {res}\n for {dtype:?}"
    );

    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn test_matmul_mm() -> Result<()> {
    let dtype = GgmlDType::Q8_0;
    let device = Device::new_metal(0)?;

    let m = 32;
    let n = 32;
    let k = 32;
    let lhs = (0..(m * k))
        .map(|v| v as f32 / (m * k) as f32)
        .collect::<Vec<_>>();
    let rhs = (0..(k * n))
        .map(|v| v as f32 / (n * k) as f32)
        .collect::<Vec<_>>();

    let lhs = Tensor::from_slice(&lhs, (m, k), &device)?;
    let rhs = Tensor::from_slice(&rhs, (1, 1, k, n), &device)?.repeat((5, 20, 1, 1))?;
    let mm = lhs.broadcast_matmul(&rhs)?;
    let qtensor = quantized::QTensor::quantize(&lhs.t()?, dtype)?;
    let matmul = quantized::QMatMul::from_qtensor(qtensor)?;
    let res = matmul.forward(&rhs)?;

    let error: f32 = ((&mm - &res)?.abs()? / &mm.abs()?)?
        .sum_all()?
        .to_scalar()?;

    let error = error / res.elem_count() as f32;
    assert!(
        error <= 0.001,
        "Error {error} is too big. \nExpected:\n {mm} \nFound:\n {res}\n for {dtype:?}"
    );

    Ok(())
}

fn quantized_matmul(device: &Device) -> Result<()> {
    let (m, k, n) = (3, 64, 4);
    let lhs_s = (0..(m * k)).map(|v| v as f32).collect::<Vec<_>>();
    let lhs = Tensor::from_slice(&lhs_s, (m, k), device)?;
    let mut dst = vec![42.; 3 * 4];
    let mut rhs_t = vec![k_quants::BlockQ4_0::zeros(); 8];
    let rhs = (0..(k * n)).map(|v| v as f32).collect::<Vec<_>>();
    k_quants::BlockQ4_0::from_float(&rhs, &mut rhs_t);
    k_quants::matmul((m, k, n), &lhs_s, &rhs_t, &mut dst)?;
    assert_eq!(
        dst.iter().map(|x| x.round()).collect::<Vec<_>>(),
        &[
            85120.0, 214562.0, 345455.0, 474748.0, 213475.0, 604465.0, 1000686.0, 1388317.0,
            341876.0, 994283.0, 1655709.0, 2301518.0
        ]
    );
    let tensor_rhs = Tensor::from_slice(&rhs, (n, k), device)?.t()?;
    let mm = lhs.matmul(&tensor_rhs)?;
    assert_eq!(
        mm.to_vec2::<f32>()?,
        &[
            [85344.0, 214368.0, 343392.0, 472416.0],
            [214368.0, 605536.0, 996704.0, 1387872.0],
            [343392.0, 996704.0, 1650016.0, 2303328.0]
        ]
    );

    let qtensor = quantized::QTensor::quantize(&tensor_rhs.t()?, GgmlDType::Q4_0)?;
    let matmul = quantized::QMatMul::from_qtensor(qtensor)?;
    let res = matmul.forward(&lhs)?;
    match device {
        Device::Metal(_) => assert_eq!(
            to_vec2_round(&res, 0)?,
            &[
                [84946.0, 214126.0, 344757.0, 473798.0],
                [213458.0, 604350.0, 1000469.0, 1387990.0],
                [341970.0, 994574.0, 1656181.0, 2302182.0]
            ]
        ),
        Device::Cuda(_) => assert_eq!(
            to_vec2_round(&res, 0)?,
            &[
                [84866.0, 214045.0, 344676.0, 473707.0],
                [213425.0, 604313.0, 1000431.0, 1387960.0],
                [342030.0, 994630.0, 1656248.0, 2302250.0]
            ]
        ),
        Device::Cpu => assert_eq!(
            to_vec2_round(&res, 0)?,
            &[
                [85120.0, 214562.0, 345455.0, 474748.0],
                [213475.0, 604465.0, 1000686.0, 1388317.0],
                [341876.0, 994283.0, 1655709.0, 2301518.0]
            ]
        ),
    }
    test_matmul(device, (1, 3, 4, 256), GgmlDType::Q4_0)?;
    Ok(())
}

fn quantized_matmul_neg(device: &Device) -> Result<()> {
    let (m, k, n) = (3, 64, 4);
    let lhs_s = (0..(m * k))
        .map(|v| v as f32 - (m * k) as f32 / 2.0)
        .collect::<Vec<_>>();
    let lhs = Tensor::from_slice(&lhs_s, (m, k), device)?;
    let mut dst = vec![42.; 3 * 4];
    let mut rhs_t = vec![k_quants::BlockQ4_0::zeros(); 8];
    let rhs = (0..k * n)
        .map(|v| v as f32 - (k * n) as f32 / 3.0)
        .collect::<Vec<_>>();
    let tensor_rhs = Tensor::from_slice(&rhs, (n, k), device)?.t()?;
    k_quants::BlockQ4_0::from_float(&rhs, &mut rhs_t);
    k_quants::matmul((m, k, n), &lhs_s, &rhs_t, &mut dst)?;
    assert_eq!(
        dst.iter().map(|x| x.round()).collect::<Vec<_>>(),
        &[
            243524.0, -19596.0, -285051.0, -549815.0, 23777.0, 21651.0, 19398.0, 18367.0,
            -196472.0, 63012.0, 324585.0, 587902.0
        ]
    );
    let mm = lhs.matmul(&tensor_rhs)?;
    assert_eq!(
        to_vec2_round(&mm, 0)?,
        &[
            [244064.0, -20128.0, -284320.0, -548512.0],
            [23563.0, 21515.0, 19467.0, 17419.0],
            [-196939.0, 63157.0, 323253.0, 583349.0]
        ]
    );

    let qtensor = quantized::QTensor::quantize(&tensor_rhs.t()?, GgmlDType::Q4_0)?;
    let matmul = quantized::QMatMul::from_qtensor(qtensor)?;
    let res = matmul.forward(&lhs)?;
    match device {
        Device::Metal(_) => assert_eq!(
            to_vec2_round(&res, 0)?,
            &[
                [243659.0, -19716.0, -285444.0, -550439.0],
                [23779.0, 21653.0, 19404.0, 18349.0],
                [-196101.0, 63021.0, 324252.0, 587137.0]
            ]
        ),
        Device::Cuda(_) => assert_eq!(
            to_vec2_round(&res, 0)?,
            &[
                [243740.0, -19762.0, -285476.0, -550498.0],
                [23774.0, 21645.0, 19395.0, 18364.0],
                [-196045.0, 63030.0, 324120.0, 587079.0]
            ]
        ),
        Device::Cpu => assert_eq!(
            to_vec2_round(&res, 0)?,
            &[
                [243524.0, -19596.0, -285051.0, -549815.0],
                [23777.0, 21651.0, 19398.0, 18367.0],
                [-196472.0, 63012.0, 324585.0, 587902.0]
            ]
        ),
    }
    let lhs2 = Tensor::stack(&[&lhs, &lhs], 0)?;
    let res2 = matmul.forward(&lhs2)?;
    let res2 = res2.i(1)?;
    let diff = (&res - res2)?.abs()?.mean_all()?.to_vec0::<f32>()? / res.elem_count() as f32;
    if device.is_cuda() {
        assert!(diff < 0.1);
    } else {
        assert!(diff < 0.96);
    }
    Ok(())
}

fn qmm_batch(dev: &Device) -> Result<()> {
    let (lhs, rhs, _mm) = get_random_tensors(2, 256, 6, dev)?;
    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q2K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;
    assert_eq!(mm.shape().dims(), [2, 6]);
    let lhs2 = Tensor::cat(&[&lhs, &lhs], 0)?;
    let mm2 = rhs.forward(&lhs2)?;
    assert_eq!(mm2.shape().dims(), [4, 6]);
    let diff2 = (mm2.i(2..)? - &mm)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert_eq!(diff2, 0.0);
    let lhs3 = Tensor::cat(&[&lhs2, &lhs], 0)?;
    let mm3 = rhs.forward(&lhs3)?;
    assert_eq!(mm3.shape().dims(), [6, 6]);
    let diff3 = (mm3.i(2..4)? - &mm)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert_eq!(diff3, 0.0);
    let diff3 = (mm3.i(4..)? - &mm)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    assert_eq!(diff3, 0.0);
    let lhs4 = Tensor::cat(&[&lhs3, &lhs3], 0)?;
    let mm4 = rhs.forward(&lhs4)?;
    assert_eq!(mm4.shape().dims(), [12, 6]);
    let diff4 = (mm4.i(..6)? - &mm3)?.abs()?.sum_all()?.to_vec0::<f32>()?;
    if dev.is_cuda() {
        // We use a different kernel for sizes from 1 to 8 on cuda which explains
        // the difference here.
        assert!(0. < diff4 && diff4 < 1e-4)
    } else {
        assert_eq!(diff4, 0.0)
    };
    let diff4 = (mm4.i(6..)? - &mm4.i(..6)?)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff4, 0.0);
    Ok(())
}

test_device!(quantized_matmul, qmm_cpu, qmm_cuda, qmm_metal);
test_device!(quantized_matmul_neg, qmm_n_cpu, qmm_n_cuda, qmm_n_metal);
test_device!(qmm_batch, qmm_b_cpu, qmm_b_cuda, qmm_b_metal);

fn quantize_q4_0(device: &Device) -> Result<()> {
    let src = (0..32 * 4).map(|v| v as f32).collect::<Vec<_>>();

    let src = Tensor::from_slice(&src, (32 * 4,), device)?;
    let quant = quantized::QTensor::quantize(&src, GgmlDType::Q4_0)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);
    assert_eq!(
        dst.to_vec1::<f32>()?,
        &[
            -0.0, -0.0, 3.875, 3.875, 3.875, 3.875, 7.75, 7.75, 7.75, 7.75, 11.625, 11.625, 11.625,
            11.625, 15.5, 15.5, 15.5, 15.5, 19.375, 19.375, 19.375, 19.375, 23.25, 23.25, 23.25,
            23.25, 27.125, 27.125, 27.125, 27.125, 31.0, 31.0, 31.5, 31.5, 31.5, 31.5, 39.375,
            39.375, 39.375, 39.375, 39.375, 39.375, 39.375, 39.375, 47.25, 47.25, 47.25, 47.25,
            47.25, 47.25, 47.25, 47.25, 55.125, 55.125, 55.125, 55.125, 55.125, 55.125, 55.125,
            55.125, 63.0, 63.0, 63.0, 63.0, 59.375, 59.375, 71.25, 71.25, 71.25, 71.25, 71.25,
            71.25, 71.25, 71.25, 71.25, 71.25, 71.25, 71.25, 83.125, 83.125, 83.125, 83.125,
            83.125, 83.125, 83.125, 83.125, 83.125, 83.125, 83.125, 83.125, 95.0, 95.0, 95.0, 95.0,
            95.0, 95.0, 95.25, 95.25, 95.25, 95.25, 95.25, 95.25, 95.25, 95.25, 111.125, 111.125,
            111.125, 111.125, 111.125, 111.125, 111.125, 111.125, 111.125, 111.125, 111.125,
            111.125, 111.125, 111.125, 111.125, 111.125, 127.0, 127.0, 127.0, 127.0, 127.0, 127.0,
            127.0, 127.0
        ]
    );
    ggml_quantization_error_test(GgmlDType::Q4_0, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

fn quantize_q4_1(device: &Device) -> Result<()> {
    let src = (0..32 * 4).map(|v| v as f32).collect::<Vec<_>>();
    let src = Tensor::from_slice(&src, (32 * 4,), device)?;
    let quant = quantized::QTensor::quantize(&src, GgmlDType::Q4_1)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);
    assert_eq!(
        round_vector(&dst.to_vec1::<f32>()?),
        &[
            0.0, 0.0, 2.066, 2.066, 4.133, 4.133, 6.199, 6.199, 8.266, 8.266, 10.332, 10.332,
            12.398, 12.398, 14.465, 14.465, 16.531, 16.531, 18.598, 18.598, 20.664, 20.664, 22.73,
            22.73, 24.797, 24.797, 26.863, 26.863, 28.93, 28.93, 30.996, 30.996, 32.0, 32.0,
            34.066, 34.066, 36.133, 36.133, 38.199, 38.199, 40.266, 40.266, 42.332, 42.332, 44.398,
            44.398, 46.465, 46.465, 48.531, 48.531, 50.598, 50.598, 52.664, 52.664, 54.73, 54.73,
            56.797, 56.797, 58.863, 58.863, 60.93, 60.93, 62.996, 62.996, 64.0, 64.0, 66.066,
            66.066, 68.133, 68.133, 70.199, 70.199, 72.266, 72.266, 74.332, 74.332, 76.398, 76.398,
            78.465, 78.465, 80.531, 80.531, 82.598, 82.598, 84.664, 84.664, 86.73, 86.73, 88.797,
            88.797, 90.863, 90.863, 92.93, 92.93, 94.996, 94.996, 96.0, 96.0, 98.066, 98.066,
            100.133, 100.133, 102.199, 102.199, 104.266, 104.266, 106.332, 106.332, 108.398,
            108.398, 110.465, 110.465, 112.531, 112.531, 114.598, 114.598, 116.664, 116.664,
            118.73, 118.73, 120.797, 120.797, 122.863, 122.863, 124.93, 124.93, 126.996, 126.996
        ]
    );
    ggml_quantization_error_test(GgmlDType::Q4_1, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

fn quantize_q5_0(device: &Device) -> Result<()> {
    let src = (0..32 * 4).map(|v| v as f32).collect::<Vec<_>>();
    let src = Tensor::from_slice(&src, (32 * 4,), device)?;
    let quant = quantized::QTensor::quantize(&src, GgmlDType::Q5_0)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);
    assert_eq!(
        round_vector(&dst.to_vec1::<f32>()?),
        &[
            -0.0, 1.938, 1.938, 3.875, 3.875, 5.813, 5.813, 7.75, 7.75, 9.688, 9.688, 11.625,
            11.625, 13.563, 13.563, 15.5, 15.5, 17.438, 17.438, 19.375, 19.375, 21.313, 21.313,
            23.25, 23.25, 25.188, 25.188, 27.125, 27.125, 29.063, 29.063, 31.0, 31.5, 31.5, 35.438,
            35.438, 35.438, 35.438, 39.375, 39.375, 39.375, 39.375, 43.313, 43.313, 43.313, 43.313,
            47.25, 47.25, 47.25, 47.25, 51.188, 51.188, 51.188, 51.188, 55.125, 55.125, 55.125,
            55.125, 59.063, 59.063, 59.063, 59.063, 63.0, 63.0, 65.313, 65.313, 65.313, 65.313,
            65.313, 71.25, 71.25, 71.25, 71.25, 71.25, 71.25, 77.188, 77.188, 77.188, 77.188,
            77.188, 77.188, 83.125, 83.125, 83.125, 83.125, 83.125, 83.125, 89.063, 89.063, 89.063,
            89.063, 89.063, 89.063, 95.0, 95.0, 95.0, 95.25, 95.25, 95.25, 95.25, 103.188, 103.188,
            103.188, 103.188, 103.188, 103.188, 103.188, 103.188, 111.125, 111.125, 111.125,
            111.125, 111.125, 111.125, 111.125, 111.125, 119.063, 119.063, 119.063, 119.063,
            119.063, 119.063, 119.063, 119.063, 127.0, 127.0, 127.0, 127.0
        ]
    );
    ggml_quantization_error_test(GgmlDType::Q5_0, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

fn quantize_q5_1(device: &Device) -> Result<()> {
    let src = (0..32 * 4).map(|v| v as f32).collect::<Vec<_>>();
    let src = Tensor::from_slice(&src, (32 * 4,), device)?;
    let quant = quantized::QTensor::quantize(&src, GgmlDType::Q5_1)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);
    assert_eq!(
        round_vector(&dst.to_vec1::<f32>()?),
        &[
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
            30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0,
            44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0,
            58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0,
            72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0,
            86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0,
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
            112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0,
            124.0, 125.0, 126.0, 127.0
        ]
    );
    ggml_quantization_error_test(GgmlDType::Q5_1, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

fn quantize_mxfp4(device: &Device) -> Result<()> {
    let dtype = GgmlDType::MXFP4;
    let src = get_test_vector2(8.0, 1024, device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    ggml_quantization_error_test(dtype, device, 0.03)?;
    Ok(())
}

#[test]
fn indexed_moe_mxfp4_cpu_reference() -> Result<()> {
    let device = &Device::Cpu;
    let (num_experts, n, k) = (3usize, 8usize, 32usize);
    let (batch, topk) = (2usize, 2usize);

    let w = (0..num_experts * n * k)
        .map(|i| ((i as f32) * 0.011).sin() * 2.0)
        .collect::<Vec<_>>();
    let w = Tensor::from_vec(w, (num_experts, n, k), device)?;
    let q = quantized::QTensor::quantize(&w, GgmlDType::MXFP4)?;

    let x = (0..batch * k)
        .map(|i| ((i as f32) * 0.017).cos())
        .collect::<Vec<_>>();
    let x = Tensor::from_vec(x, (batch, k), device)?.to_dtype(DType::F32)?;
    let ids = Tensor::from_vec(vec![0u32, 2, 1, 0], (batch, topk), device)?;

    let y = q
        .indexed_moe_forward(&x, &ids)?
        .to_dtype(DType::F32)?
        .to_vec3::<f32>()?;

    let w_dq = q.dequantize(device)?.to_vec3::<f32>()?;
    let x = x.to_vec2::<f32>()?;
    let ids = ids.to_vec2::<u32>()?;
    let mut y_ref = vec![0f32; batch * topk * n];
    for b in 0..batch {
        for t in 0..topk {
            let e = ids[b][t] as usize;
            for r in 0..n {
                let mut acc = 0f32;
                for c in 0..k {
                    acc += x[b][c] * w_dq[e][r][c];
                }
                y_ref[(b * topk + t) * n + r] = acc;
            }
        }
    }

    let y_ref = Tensor::from_vec(y_ref, (batch, topk, n), device)?.to_vec3::<f32>()?;
    let mut max_abs = 0f32;
    for b in 0..batch {
        for t in 0..topk {
            for r in 0..n {
                let d = (y[b][t][r] - y_ref[b][t][r]).abs();
                if d > max_abs {
                    max_abs = d;
                }
            }
        }
    }

    // The CPU indexed_moe path quantizes activations for vec-dot, so it should
    // stay close (but not bit-exact) to f32 x dequantized-weight reference.
    assert!(max_abs < 0.35, "max_abs={max_abs}");
    Ok(())
}

fn indexed_moe_mxfp4_parity_with_device(device: &Device) -> Result<()> {
    let cpu = &Device::Cpu;
    let (num_experts, n, k) = (4usize, 256usize, 256usize);
    let (batch, topk) = (4usize, 2usize);

    let w = (0..num_experts * n * k)
        .map(|i| ((i as f32) * 0.0031).sin() * 1.5)
        .collect::<Vec<_>>();
    let w = Tensor::from_vec(w, (num_experts, n, k), cpu)?;
    let q_cpu = quantized::QTensor::quantize(&w, GgmlDType::MXFP4)?;
    let q_data = q_cpu.data()?.into_owned();
    let mut max_e = 0u8;
    for (i, block) in q_data.chunks_exact(17).enumerate() {
        let e = block[0];
        assert!(e != 255, "invalid MXFP4 exponent 255 in block {i}");
        if e > max_e {
            max_e = e;
        }
    }
    assert!(max_e < 200, "unexpectedly large MXFP4 exponent {max_e}");
    let q_dev_storage =
        quantized::QStorage::from_data(Cow::Owned(q_data.clone()), device, GgmlDType::MXFP4)?;
    let q_dev = quantized::QTensor::new(q_dev_storage, (num_experts, n, k))?;
    let q_dev_data = q_dev.data()?.into_owned();
    if q_data != q_dev_data {
        let first_diff = q_data
            .iter()
            .zip(q_dev_data.iter())
            .position(|(a, b)| a != b)
            .unwrap_or_else(|| q_data.len().min(q_dev_data.len()));
        let q_head = &q_data[..usize::min(32, q_data.len())];
        let d_head = &q_dev_data[..usize::min(32, q_dev_data.len())];
        panic!(
            "device MXFP4 bytes differ from CPU source: cpu_len={} dev_len={} first_diff={} cpu_head={:?} dev_head={:?}",
            q_data.len(),
            q_dev_data.len(),
            first_diff,
            q_head,
            d_head
        );
    }
    let q_dev_dq = q_dev
        .dequantize(device)?
        .to_device(cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if let Some((idx, v)) = q_dev_dq
        .iter()
        .copied()
        .enumerate()
        .find(|(_, v)| !v.is_finite())
    {
        panic!("device dequantized weight non-finite at {idx}: {v}");
    }

    let x = (0..batch * k)
        .map(|i| ((i as f32) * 0.013).cos())
        .collect::<Vec<_>>();
    let x_cpu = Tensor::from_vec(x, (batch, k), cpu)?.to_dtype(DType::F32)?;
    let x_dev = x_cpu.to_device(device)?;

    let ids_cpu = Tensor::from_vec(vec![0u32, 1, 2, 3, 3, 2, 1, 0], (batch, topk), cpu)?;
    let ids_dev = ids_cpu.to_device(device)?;

    let y_cpu = q_cpu
        .indexed_moe_forward(&x_cpu, &ids_cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let y_dev = q_dev
        .indexed_moe_forward(&x_dev, &ids_dev)?
        .to_device(cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    if let Some((idx, v)) = y_cpu
        .iter()
        .copied()
        .enumerate()
        .find(|(_, v)| !v.is_finite())
    {
        panic!("cpu output non-finite at {idx}: {v}");
    }
    if let Some((idx, v)) = y_dev
        .iter()
        .copied()
        .enumerate()
        .find(|(_, v)| !v.is_finite())
    {
        panic!("device output non-finite at {idx}: {v}");
    }

    let mut max_abs = 0f32;
    let mut mean_abs = 0f32;
    let mut mean_abs_ref = 0f32;
    for (&a, &b) in y_cpu.iter().zip(y_dev.iter()) {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
        mean_abs += d;
        mean_abs_ref += a.abs();
    }
    mean_abs /= y_cpu.len() as f32;
    mean_abs_ref /= y_cpu.len() as f32;

    // Quantized backend kernels are not bit-identical, but should remain close
    // on average for deterministic inputs and identical packed weights.
    let mean_tol = 0.03 * mean_abs_ref + 1e-4;
    let max_tol = 0.20 * mean_abs_ref + 1e-3;
    assert!(
        mean_abs <= mean_tol && max_abs <= max_tol,
        "indexed_moe parity failed: max_abs={max_abs} mean_abs={mean_abs} max_tol={max_tol} mean_tol={mean_tol} cpu_head={:?} dev_head={:?}",
        &y_cpu[..usize::min(8, y_cpu.len())],
        &y_dev[..usize::min(8, y_dev.len())]
    );
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn indexed_moe_mxfp4_cpu_cuda_parity() -> Result<()> {
    let dev = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    indexed_moe_mxfp4_parity_with_device(&dev)
}

#[cfg(feature = "metal")]
#[test]
fn indexed_moe_mxfp4_cpu_metal_parity() -> Result<()> {
    let dev = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    indexed_moe_mxfp4_parity_with_device(&dev)
}

fn get_test_vector2(bound: f32, size: usize, device: &Device) -> Result<Tensor> {
    assert!(
        size.is_multiple_of(crate::quantized::k_quants::QK_K),
        "size must be a multiple of {}",
        crate::quantized::k_quants::QK_K
    );

    let src = (0..size)
        .map(|v| (v as f32 - size as f32 / 2.) * bound / (size as f32 / 2.))
        .collect::<Vec<_>>();
    assert_eq!([src[0], src[size / 2]], [-bound, 0.0]);
    Tensor::from_vec(src, (size,), device)
}

/// Round a vector
fn round_vector(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|x| (1000. * x).round() / 1000.)
        .collect::<Vec<_>>()
}

fn compare_with_error(values: &[f32], expected: &[f32], tolerance: f32) {
    for (i, (value, expected_value)) in values.iter().zip(expected.iter()).enumerate() {
        let difference = (value - expected_value).abs();

        assert!(
            difference < tolerance,
            "Error at index {i}: value = {value}, expected = {expected_value}. Difference = {difference} exceeds tolerance = {tolerance}."
        );
    }
}

/// Creates a vector similar to the ones used in GGML unit tests:
/// https://github.com/ggerganov/llama.cpp/blob/master/tests/test-quantize-fns.cpp#L26-L30
fn create_ggml_like_vector(offset: f32) -> Vec<f32> {
    (0..GGML_TEST_SIZE)
        .map(|i| 0.1 + 2.0 * (i as f32 + offset).cos())
        .collect()
}

/// Calculates the root mean square error between two vectors
fn calculate_rmse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let sum = a
        .iter()
        .zip(b)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    sum / a.len() as f32
}

/// Similar to the GGML quantization unit test:
/// https://github.com/ggerganov/llama.cpp/blob/master/tests/test-quantize-fns.cpp#L43-L50
fn ggml_quantization_error_test(dtype: GgmlDType, device: &Device, max_error: f32) -> Result<()> {
    let src = create_ggml_like_vector(0.0);
    let src = Tensor::from_slice(&src, (GGML_TEST_SIZE,), device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);
    let error = calculate_rmse(&src.to_vec1::<f32>()?, &dst.to_vec1::<f32>()?);
    if error > max_error {
        bail!(
            "Quantization error {} exceeds max error {}",
            error,
            max_error
        );
    }
    Ok(())
}

#[test]
fn imatrix_quantize_q6k() -> Result<()> {
    let cpu = &Device::Cpu;

    let mut row_counts = 0f64;
    let mut ncall = 0f64;
    let mut values = Tensor::zeros((768,), DType::F32, cpu)?;

    for _ in 0..10 {
        let lhs = Var::from_tensor(&Tensor::randn(0f32, 1f32, (1024, 512), cpu)?)?;
        let rhs = Var::from_tensor(&Tensor::randn(0f32, 1f32, (512, 768), cpu)?)?;
        let res = lhs.matmul(&rhs)?;

        // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L180-L186
        values = (values + res.sqr()?.sum(0)?)?;
        row_counts += res.dim(0)? as f64;
        ncall += 1.;
    }

    // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L275
    let out = ((values / row_counts)? * ncall)?;
    let imatrix = out.to_vec1::<f32>()?;

    let xs = Tensor::randn(0f32, 1f32, (1024, 768), cpu)?;

    let quant1 = quantized::QTensor::quantize(&xs, GgmlDType::Q6K)?;
    let quant2 = quantized::QTensor::quantize_imatrix(&xs, &imatrix, GgmlDType::Q6K)?;

    let dequant1 = quant1.dequantize(cpu)?;
    let dequant2 = quant2.dequantize(cpu)?;

    let err1 = (dequant1 - &xs)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    let err2 = (dequant2 - &xs)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    assert!(err2 < err1, "err2 {err2} > err1 {err1}");

    Ok(())
}

#[test]
fn imatrix_quantize_q5k() -> Result<()> {
    let cpu = &Device::Cpu;

    let mut row_counts = 0f64;
    let mut ncall = 0f64;
    let mut values = Tensor::zeros((768,), DType::F32, cpu)?;

    for _ in 0..10 {
        let lhs = Var::from_tensor(&Tensor::randn(0f32, 1f32, (1024, 512), cpu)?)?;
        let rhs = Var::from_tensor(&Tensor::randn(0f32, 1f32, (512, 768), cpu)?)?;
        let res = lhs.matmul(&rhs)?;

        // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L180-L186
        values = (values + res.sqr()?.sum(0)?)?;
        row_counts += res.dim(0)? as f64;
        ncall += 1.;
    }

    // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L275
    let out = ((values / row_counts)? * ncall)?;
    let imatrix = out.to_vec1::<f32>()?;

    let xs = Tensor::randn(0f32, 1f32, (1024, 768), cpu)?;

    let quant1 = quantized::QTensor::quantize(&xs, GgmlDType::Q5K)?;
    let quant2 = quantized::QTensor::quantize_imatrix(&xs, &imatrix, GgmlDType::Q5K)?;

    let dequant1 = quant1.dequantize(cpu)?;
    let dequant2 = quant2.dequantize(cpu)?;

    let err1 = (dequant1 - &xs)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    let err2 = (dequant2 - &xs)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    assert!(err2 < err1, "err2 {err2} > err1 {err1}");

    Ok(())
}

#[test]
fn imatrix_quantize_q4k() -> Result<()> {
    // let data =
    //     quantized::imatrix_file::load_imatrix("../Llama-3.2-3B-Instruct.imatrix").unwrap();
    // for (name, weights) in &data {
    //     println!("{name}, {} elems", weights.len());
    // }
    // dbg!(&data["blk.0.attn_q.weight"].len());

    let cpu = &Device::Cpu;

    let mut row_counts = 0f64;
    let mut ncall = 0f64;
    let mut values = Tensor::zeros((768,), DType::F32, cpu)?;

    for _ in 0..10 {
        let lhs = Var::from_tensor(&Tensor::randn(0f32, 1f32, (1024, 512), cpu)?)?;
        let rhs = Var::from_tensor(&Tensor::randn(0f32, 1f32, (512, 768), cpu)?)?;
        let res = lhs.matmul(&rhs)?;

        // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L180-L186
        values = (values + res.sqr()?.sum(0)?)?;
        row_counts += res.dim(0)? as f64;
        ncall += 1.;
    }

    // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L275
    let out = ((values / row_counts)? * ncall)?;
    let imatrix = out.to_vec1::<f32>()?;

    let xs = Tensor::randn(0f32, 1f32, (1024, 768), cpu)?;

    let quant1 = quantized::QTensor::quantize(&xs, GgmlDType::Q4K)?;
    let quant2 = quantized::QTensor::quantize_imatrix(&xs, &imatrix, GgmlDType::Q4K)?;

    let dequant1 = quant1.dequantize(cpu)?;
    let dequant2 = quant2.dequantize(cpu)?;

    let err1 = (dequant1 - &xs)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    let err2 = (dequant2 - &xs)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    assert!(err2 < err1, "err2 {err2} > err1 {err1}");

    Ok(())
}

#[test]
fn imatrix_quantize_q3k() -> Result<()> {
    let cpu = &Device::Cpu;

    let mut row_counts = 0f64;
    let mut ncall = 0f64;
    let mut values = Tensor::zeros((768,), DType::F32, cpu)?;

    for _ in 0..10 {
        let lhs = Var::from_tensor(&Tensor::randn(0f32, 1f32, (1024, 512), cpu)?)?;
        let rhs = Var::from_tensor(&Tensor::randn(0f32, 1f32, (512, 768), cpu)?)?;
        let res = lhs.matmul(&rhs)?;

        // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L180-L186
        values = (values + res.sqr()?.sum(0)?)?;
        row_counts += res.dim(0)? as f64;
        ncall += 1.;
    }

    // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L275
    let out = ((values / row_counts)? * ncall)?;
    let imatrix = out.to_vec1::<f32>()?;

    let xs = Tensor::randn(0f32, 1f32, (1024, 768), cpu)?;

    let quant1 = quantized::QTensor::quantize(&xs, GgmlDType::Q3K)?;
    let quant2 = quantized::QTensor::quantize_imatrix(&xs, &imatrix, GgmlDType::Q3K)?;

    let dequant1 = quant1.dequantize(cpu)?;
    let dequant2 = quant2.dequantize(cpu)?;

    let err1 = (dequant1 - &xs)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    let err2 = (dequant2 - &xs)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    assert!(err2 < err1, "err2 {err2} > err1 {err1}");

    Ok(())
}

#[test]
fn imatrix_quantize_q2k() -> Result<()> {
    let cpu = &Device::Cpu;

    let mut row_counts = 0f64;
    let mut ncall = 0f64;
    let mut values = Tensor::zeros((768,), DType::F32, cpu)?;

    for _ in 0..10 {
        let lhs = Var::from_tensor(&Tensor::randn(0f32, 1f32, (1024, 512), cpu)?)?;
        let rhs = Var::from_tensor(&Tensor::randn(0f32, 1f32, (512, 768), cpu)?)?;
        let res = lhs.matmul(&rhs)?;

        // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L180-L186
        values = (values + res.sqr()?.sum(0)?)?;
        row_counts += res.dim(0)? as f64;
        ncall += 1.;
    }

    // https://github.com/ggerganov/llama.cpp/blob/678d7994f4da0af3d29046be99950ac999ee9762/examples/imatrix/imatrix.cpp#L275
    let out = ((values / row_counts)? * ncall)?;
    let imatrix = out.to_vec1::<f32>()?;

    let xs = Tensor::randn(0f32, 1f32, (1024, 768), cpu)?;

    let quant1 = quantized::QTensor::quantize(&xs, GgmlDType::Q2K)?;
    let quant2 = quantized::QTensor::quantize_imatrix(&xs, &imatrix, GgmlDType::Q2K)?;

    let dequant1 = quant1.dequantize(cpu)?;
    let dequant2 = quant2.dequantize(cpu)?;

    let err1 = (dequant1 - &xs)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    let err2 = (dequant2 - &xs)?.abs()?.mean_all()?.to_scalar::<f32>()?;
    assert!(err2 < err1, "err2 {err2} > err1 {err1}");

    Ok(())
}

fn quantize_q2k(device: &Device) -> Result<()> {
    let dtype = GgmlDType::Q2K;

    let src = get_test_vector2(0.5, 1024, device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src = src.to_vec1::<f32>()?;
    let dst = dst.to_vec1::<f32>()?;
    compare_with_error(dst.as_slice(), src.as_slice(), 0.1);

    // Test some specific values
    assert_eq!(
        [src[0], src[128], src[256], src[512], src[800], src[1023]],
        [-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344]
    );
    let dst = round_vector(&dst);
    assert_eq!(
        [dst[0], dst[128], dst[256], dst[512], dst[800], dst[1023]],
        [-0.499, -0.366, -0.249, 0.0, 0.295, 0.492]
    );

    let src_big = get_test_vector2(128.0, 1024, device)?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let dst_big_f16 = quant_big.dequantize_f16(device)?;
    let diff = (dst_big.to_dtype(DType::F16)? - dst_big_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src_big = src_big.to_vec1::<f32>()?;
    let dst_big = dst_big.to_vec1::<f32>()?;
    compare_with_error(dst_big.as_slice(), src_big.as_slice(), 6.0);

    ggml_quantization_error_test(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR_2BITS)?;
    Ok(())
}

fn quantize_q3k(device: &Device) -> Result<()> {
    let dtype = GgmlDType::Q3K;
    let src = get_test_vector2(0.5, 1024, device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src = src.to_vec1::<f32>()?;
    let dst = dst.to_vec1::<f32>()?;
    compare_with_error(dst.as_slice(), src.as_slice(), 0.03);

    // Test some specific values
    assert_eq!(
        [src[0], src[128], src[256], src[512], src[800], src[1023]],
        [-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344]
    );
    let dst = round_vector(&dst);
    assert_eq!(
        [dst[0], dst[128], dst[256], dst[512], dst[800], dst[1023]],
        [-0.493, -0.37, -0.243, -0.0, 0.292, 0.492]
    );

    let src_big = get_test_vector2(128.0, 1024, device)?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let dst_big_f16 = quant_big.dequantize_f16(device)?;
    let diff = (dst_big.to_dtype(DType::F16)? - dst_big_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src_big = src_big.to_vec1::<f32>()?;
    let dst_big = dst_big.to_vec1::<f32>()?;
    compare_with_error(dst_big.as_slice(), src_big.as_slice(), 3.5);

    ggml_quantization_error_test(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR_3BITS)?;
    Ok(())
}

fn quantize_q4k(device: &Device) -> Result<()> {
    let dtype = GgmlDType::Q4K;
    let src = get_test_vector2(0.5, 1024, device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src = src.to_vec1::<f32>()?;
    let dst = dst.to_vec1::<f32>()?;
    compare_with_error(dst.as_slice(), src.as_slice(), 0.017);

    // Test some specific values
    assert_eq!(
        [src[0], src[128], src[256], src[512], src[800], src[1023]],
        [-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344]
    );
    let dst = round_vector(&dst);
    assert_eq!(
        [dst[0], dst[128], dst[256], dst[512], dst[800], dst[1023]],
        [-0.5, -0.373, -0.25, 0.0, 0.288, 0.498]
    );

    let src_big = get_test_vector2(128.0, 1024, device)?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let dst_big_f16 = quant_big.dequantize_f16(device)?;
    let diff = (dst_big.to_dtype(DType::F16)? - dst_big_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src_big = src_big.to_vec1::<f32>()?;
    let dst_big = dst_big.to_vec1::<f32>()?;
    compare_with_error(dst_big.as_slice(), src_big.as_slice(), 4.5);

    ggml_quantization_error_test(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

fn quantize_q5k(device: &Device) -> Result<()> {
    let dtype = GgmlDType::Q5K;
    let src = get_test_vector2(0.5, 1024, device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src = src.to_vec1::<f32>()?;
    let dst = dst.to_vec1::<f32>()?;
    compare_with_error(dst.as_slice(), src.as_slice(), 0.009);

    // Test some specific values
    assert_eq!(
        [src[0], src[128], src[256], src[512], src[800], src[1023]],
        [-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344]
    );
    let dst = round_vector(&dst);
    assert_eq!(
        [dst[0], dst[128], dst[256], dst[512], dst[800], dst[1023]],
        [-0.5, -0.373, -0.25, 0.0, 0.279, 0.499]
    );

    let src_big = get_test_vector2(128.0, 1024, device)?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let dst_big_f16 = quant_big.dequantize_f16(device)?;
    let diff = (dst_big.to_dtype(DType::F16)? - dst_big_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src_big = src_big.to_vec1::<f32>()?;
    let dst_big = dst_big.to_vec1::<f32>()?;
    compare_with_error(dst_big.as_slice(), src_big.as_slice(), 2.5);

    ggml_quantization_error_test(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

fn quantize_q6k(device: &Device) -> Result<()> {
    let dtype = GgmlDType::Q6K;
    let src = get_test_vector2(0.5, 1024, device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src = src.to_vec1::<f32>()?;
    let dst = dst.to_vec1::<f32>()?;
    compare_with_error(dst.as_slice(), src.as_slice(), 0.008);

    // Test some specific values
    assert_eq!(
        [src[0], src[128], src[256], src[512], src[800], src[1023]],
        [-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344]
    );
    let dst = round_vector(&dst);
    assert_eq!(
        [dst[0], dst[128], dst[256], dst[512], dst[800], dst[1023]],
        [-0.497, -0.372, -0.25, -0.0, 0.284, 0.5]
    );

    let src_big = get_test_vector2(128.0, 1024, device)?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let dst_big_f16 = quant_big.dequantize_f16(device)?;
    let diff = (dst_big.to_dtype(DType::F16)? - dst_big_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src_big = src_big.to_vec1::<f32>()?;
    let dst_big = dst_big.to_vec1::<f32>()?;
    compare_with_error(dst_big.as_slice(), src_big.as_slice(), 2.0);

    ggml_quantization_error_test(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

fn quantize_q8k(device: &Device) -> Result<()> {
    let dtype = GgmlDType::Q8K;
    let src = get_test_vector2(0.5, 1024, device)?;
    let quant = quantized::QTensor::quantize(&src, dtype)?;
    let dst = quant.dequantize(device)?;
    let dst_f16 = quant.dequantize_f16(device)?;
    let diff = (dst.to_dtype(DType::F16)? - dst_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src = src.to_vec1::<f32>()?;
    let dst = dst.to_vec1::<f32>()?;
    compare_with_error(dst.as_slice(), src.as_slice(), 0.008);

    // Test some specific values
    assert_eq!(
        [src[0], src[128], src[256], src[512], src[800], src[1023]],
        [-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344]
    );
    let dst = round_vector(&dst);
    assert_eq!(
        [dst[0], dst[128], dst[256], dst[512], dst[800], dst[1023]],
        [-0.5, -0.375, -0.25, -0.0, 0.281, 0.499]
    );

    let src_big = get_test_vector2(128.0, 1024, device)?;
    let quant_big = quantized::QTensor::quantize(&src_big, dtype)?;
    let dst_big = quant_big.dequantize(device)?;
    let dst_big_f16 = quant_big.dequantize_f16(device)?;
    let diff = (dst_big.to_dtype(DType::F16)? - dst_big_f16)?
        .to_dtype(DType::F32)?
        .abs()?
        .sum_all()?
        .to_vec0::<f32>()?;
    assert_eq!(diff, 0.);

    let src_big = src_big.to_vec1::<f32>()?;
    let dst_big = dst_big.to_vec1::<f32>()?;
    compare_with_error(dst_big.as_slice(), src_big.as_slice(), 0.6);

    ggml_quantization_error_test(dtype, device, GGML_MAX_QUANTIZATION_TOTAL_ERROR)?;
    Ok(())
}

test_device!(
    quantize_q4_0,
    quantize_q4_0_cpu,
    quantize_q4_0_cuda,
    quantize_q4_0_metal
);
test_device!(
    quantize_q4_1,
    quantize_q4_1_cpu,
    quantize_q4_1_cuda,
    quantize_q4_1_metal
);
test_device!(
    quantize_q5_0,
    quantize_q5_0_cpu,
    quantize_q5_0_cuda,
    quantize_q5_0_metal
);
test_device!(
    quantize_q5_1,
    quantize_q5_1_cpu,
    quantize_q5_1_cuda,
    quantize_q5_1_metal
);
test_device!(
    quantize_mxfp4,
    quantize_mxfp4_cpu,
    quantize_mxfp4_cuda,
    quantize_mxfp4_metal
);
test_device!(
    quantize_q2k,
    quantize_q2k_cpu,
    quantize_q2k_cuda,
    quantize_q2k_metal
);
test_device!(
    quantize_q3k,
    quantize_q3k_cpu,
    quantize_q3k_cuda,
    quantize_q3k_metal
);
test_device!(
    quantize_q4k,
    quantize_q4k_cpu,
    quantize_q4k_cuda,
    quantize_q4k_metal
);
test_device!(
    quantize_q5k,
    quantize_q5k_cpu,
    quantize_q5k_cuda,
    quantize_q5k_metal
);
test_device!(
    quantize_q6k,
    quantize_q6k_cpu,
    quantize_q6k_cuda,
    quantize_q6k_metal
);
test_device!(
    quantize_q8k,
    quantize_q8k_cpu,
    quantize_q8k_cuda,
    quantize_q8k_metal
);

/// Very simple dot product implementation
fn vec_dot_reference(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// Returns the error achieved by the GGML matmul unit test.
fn ggml_reference_matmul_error(dtype: GgmlDType) -> Result<f32> {
    let err = match dtype {
        GgmlDType::F32 => 0.000000,
        GgmlDType::F16 => 0.000010,
        GgmlDType::BF16 => 0.000200,
        GgmlDType::Q2K => 0.004086,
        GgmlDType::Q3K => 0.016148,
        GgmlDType::Q4K => 0.002425,
        GgmlDType::Q5K => 0.000740,
        GgmlDType::Q6K => 0.000952,
        GgmlDType::Q4_0 => 0.001143,
        GgmlDType::Q4_1 => 0.008,
        GgmlDType::MXFP4 => 0.012,
        GgmlDType::Q5_0 => 0.001353,
        GgmlDType::Q5_1 => 0.00149,
        GgmlDType::Q8_0 => 0.000092,
        GgmlDType::Q8_1 => 0.000092,

        // Not from the ggml repo.
        GgmlDType::Q8K => 0.00065,
    };
    Ok(err)
}

/// Similar to the GGML matmul unit test:
/// https://github.com/ggerganov/llama.cpp/blob/master/tests/test-quantize-fns.cpp#L76-L91
fn ggml_matmul_error_test<T: GgmlType>() -> Result<()> {
    let a = create_ggml_like_vector(0.0);
    let b = create_ggml_like_vector(1.0);
    ggml_matmul_error_test_::<T>(a.as_slice(), b.as_slice(), 1.0)?;
    // Another example that is more likely to trigger the overflow reported in #1526
    let a = (0..GGML_TEST_SIZE)
        .map(|i| i as f32 / GGML_TEST_SIZE as f32)
        .collect::<Vec<_>>();
    let b = (0..GGML_TEST_SIZE)
        .map(|i| i as f32 / GGML_TEST_SIZE as f32)
        .collect::<Vec<_>>();
    ggml_matmul_error_test_::<T>(a.as_slice(), b.as_slice(), 2.0)?;
    Ok(())
}

fn ggml_matmul_error_test_<T: GgmlType>(a: &[f32], b: &[f32], err_m: f32) -> Result<()> {
    let length = a.len();

    let mut a_quant = vec![T::zeros(); length / T::BLCK_SIZE];
    let mut b_quant = vec![T::VecDotType::zeros(); length / T::VecDotType::BLCK_SIZE];
    T::from_float(a, &mut a_quant);
    T::VecDotType::from_float(b, &mut b_quant);

    let result = T::vec_dot(length, &a_quant, &b_quant);
    let result_unopt = T::vec_dot_unopt(length, &a_quant, &b_quant);

    if (result - result_unopt).abs() / length as f32 > 1e-6 {
        bail!(
            "the opt and unopt vec-dot returned different values, opt: {result} vs unopt: {result_unopt}"
        )
    }

    let mut dst = vec![0.0f32; 1];
    crate::k_quants::matmul((1, length, 1), b, &a_quant, &mut dst)?;
    let result_matmul = dst[0];

    if (result_matmul - result).abs() / length as f32 > 1e-6 {
        bail!(
            "calling matmul vs calling vec-dot directly returned different values, matmul: {result_matmul} vs vec-dot: {result}"
        )
    }

    let reference_result = vec_dot_reference(a, b);

    let verify_result = |result: f32, source: &str| {
        let error = (result - reference_result).abs() / length as f32;
        let ggml_error = ggml_reference_matmul_error(T::DTYPE)? * err_m;
        if !error.is_finite() || error > GGML_MAX_DOT_PRODUCT_ERROR {
            bail!("Dot product with dtype {:?} error {error} exceeds max error {GGML_MAX_DOT_PRODUCT_ERROR}. Source: {source}", T::DTYPE);
        }
        // We diverge slightly due to different rounding behavior / f16 to f32 conversions in GGML
        // => we use a slightly higher error threshold
        const ERROR_LENIENCY: f32 = 0.00001;
        if error - ERROR_LENIENCY > ggml_error {
            bail!(
                "Dot product with dtype {:?} error {error} exceeds ggml reference error {ggml_error}. Source: {source}",
                T::DTYPE,
            );
        }
        Ok(())
    };

    verify_result(result, "vec-dot")?;
    verify_result(result_matmul, "matmul")?;
    Ok(())
}

#[test]
fn quantized_mm() -> Result<()> {
    ggml_matmul_error_test::<f32>()?;
    ggml_matmul_error_test::<half::f16>()?;
    //ggml_matmul_error_test::<half::bf16>()?; TODO: Fails on ubuntu and windows. Check CpuBF16 impl
    ggml_matmul_error_test::<k_quants::BlockQ4_0>()?;
    ggml_matmul_error_test::<k_quants::BlockQ4_1>()?;
    ggml_matmul_error_test::<k_quants::BlockQ5_0>()?;
    ggml_matmul_error_test::<k_quants::BlockQ5_1>()?;
    ggml_matmul_error_test::<k_quants::BlockQ8_0>()?;
    ggml_matmul_error_test::<k_quants::BlockQ8_1>()?;
    Ok(())
}

/// generates random tensors of size `m x k` and `n x k` and calculates their expected matrix multiplication result.
fn get_random_tensors(
    m: usize,
    k: usize,
    n: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let mut rng = StdRng::seed_from_u64(314159265358979);

    let lhs = (0..m * k)
        .map(|_| rng.random::<f32>() - 0.5)
        .collect::<Vec<_>>();
    let rhs = (0..n * k)
        .map(|_| rng.random::<f32>() - 0.5)
        .collect::<Vec<_>>();

    let lhs = Tensor::from_vec(lhs, (m, k), device)?;
    let rhs = Tensor::from_vec(rhs, (n, k), device)?;

    let mm = lhs.matmul(&rhs.t()?)?;
    Ok((lhs, rhs, mm))
}

#[macro_export]
macro_rules! quantized_matmul {
    // TODO: Switch to generating the two last arguments automatically once concat_idents is
    // stable. https://github.com/rust-lang/rust/issues/29599
    ($fn_name: ident, $fn_name_cpu: ident, $fn_name_cuda: ident, $fn_name_metal: ident, $dtype: expr) => {
        fn $fn_name(device: &Device) -> Result<()> {
            test_matmul(device, (1, 3, 4, 256), $dtype)?;
            Ok(())
        }

        test_device!($fn_name, $fn_name_cpu, $fn_name_cuda, $fn_name_metal);
    };
}

quantized_matmul!(
    quantized_matmul_q4_0_bis,
    quantized_matmul_q4_0_cpu,
    quantized_matmul_q4_0_cuda,
    quantized_matmul_q4_0_metal,
    GgmlDType::Q4_0
);
quantized_matmul!(
    quantized_matmul_q4_1_bis,
    quantized_matmul_q4_1_cpu,
    quantized_matmul_q4_1_cuda,
    quantized_matmul_q4_1_metal,
    GgmlDType::Q4_1
);
quantized_matmul!(
    quantized_matmul_q5_0_bis,
    quantized_matmul_q5_0_cpu,
    quantized_matmul_q5_0_cuda,
    quantized_matmul_q5_0_metal,
    GgmlDType::Q5_0
);
quantized_matmul!(
    quantized_matmul_q5_1_bis,
    quantized_matmul_q5_1_cpu,
    quantized_matmul_q5_1_cuda,
    quantized_matmul_q5_1_metal,
    GgmlDType::Q5_1
);
quantized_matmul!(
    quantized_matmul_q8_0_bis,
    quantized_matmul_q8_0_cpu,
    quantized_matmul_q8_0_cuda,
    quantized_matmul_q8_0_metal,
    GgmlDType::Q8_0
);
quantized_matmul!(
    quantized_matmul_q8_1_bis,
    quantized_matmul_q8_1_cpu,
    quantized_matmul_q8_1_cuda,
    quantized_matmul_q8_1_metal,
    GgmlDType::Q8_1
);
quantized_matmul!(
    quantized_matmul_q2k_bis,
    quantized_matmul_q2k_cpu,
    quantized_matmul_q2k_cuda,
    quantized_matmul_q2k_metal,
    GgmlDType::Q2K
);
quantized_matmul!(
    quantized_matmul_q3k_bis,
    quantized_matmul_q3k_cpu,
    quantized_matmul_q3k_cuda,
    quantized_matmul_q3k_metal,
    GgmlDType::Q3K
);
quantized_matmul!(
    quantized_matmul_q4k_bis,
    quantized_matmul_q4k_cpu,
    quantized_matmul_q4k_cuda,
    quantized_matmul_q4k_metal,
    GgmlDType::Q4K
);
quantized_matmul!(
    quantized_matmul_q5k_bis,
    quantized_matmul_q5k_cpu,
    quantized_matmul_q5k_cuda,
    quantized_matmul_q5k_metal,
    GgmlDType::Q5K
);
quantized_matmul!(
    quantized_matmul_q6k_bis,
    quantized_matmul_q6k_cpu,
    quantized_matmul_q6k_cuda,
    quantized_matmul_q6k_metal,
    GgmlDType::Q6K
);
// Not implemented on metal
quantized_matmul!(
    quantized_matmul_q8k_bis,
    quantized_matmul_q8k_cpu,
    quantized_matmul_q8k_cuda,
    quantized_matmul_q8k_metal,
    GgmlDType::Q8K
);

#[test]
fn quantized_matmul_q2k() -> Result<()> {
    use k_quants::BlockQ2K;

    let cpu = &Device::Cpu;
    let (m, k, n) = (11, 512, 21);
    let (lhs, rhs, mm) = get_random_tensors(m, k, n, cpu)?;
    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.262, 1.513, -0.208, 1.702]);

    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q2K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;

    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [0.916, 0.422, 0.215, 1.668]);

    ggml_matmul_error_test::<BlockQ2K>()?;

    Ok(())
}

#[test]
fn quantized_matmul_q3k() -> Result<()> {
    use k_quants::BlockQ3K;

    let cpu = &Device::Cpu;
    let (m, k, n) = (11, 512, 21);
    let (lhs, rhs, mm) = get_random_tensors(m, k, n, cpu)?;
    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.262, 1.513, -0.208, 1.702]);

    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q3K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;

    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.029, 1.418, -0.314, 1.495]);

    ggml_matmul_error_test::<BlockQ3K>()?;

    Ok(())
}

#[test]
fn quantized_matmul_q4k() -> Result<()> {
    use k_quants::BlockQ4K;

    let cpu = &Device::Cpu;
    let (m, k, n) = (11, 512, 21);
    let (lhs, rhs, mm) = get_random_tensors(m, k, n, cpu)?;
    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.262, 1.513, -0.208, 1.702]);

    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q4K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;

    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.125, 1.435, -0.201, 1.589]);

    ggml_matmul_error_test::<BlockQ4K>()?;

    Ok(())
}

#[test]
fn quantized_matmul_q5k() -> Result<()> {
    use k_quants::BlockQ5K;

    let cpu = &Device::Cpu;
    let (m, k, n) = (11, 512, 21);
    let (lhs, rhs, mm) = get_random_tensors(m, k, n, cpu)?;
    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.262, 1.513, -0.208, 1.702]);

    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q5K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;

    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.192, 1.491, -0.18, 1.743]);

    //Expected: 0.000740408897
    ggml_matmul_error_test::<BlockQ5K>()?;

    Ok(())
}

#[test]
fn quantized_matmul_q6k() -> Result<()> {
    use k_quants::BlockQ6K;

    let cpu = &Device::Cpu;
    let (m, k, n) = (11, 512, 21);
    let (lhs, rhs, mm) = get_random_tensors(m, k, n, cpu)?;
    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.262, 1.513, -0.208, 1.702]);

    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q6K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;

    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.324, 1.49, -0.164, 1.741]);

    ggml_matmul_error_test::<BlockQ6K>()?;
    Ok(())
}

#[test]
fn quantized_matmul_q8k() -> Result<()> {
    use k_quants::BlockQ8K;

    let cpu = &Device::Cpu;
    let (m, k, n) = (11, 512, 21);
    let (lhs, rhs, mm) = get_random_tensors(m, k, n, cpu)?;
    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.262, 1.513, -0.208, 1.702]);

    let rhs = quantized::QTensor::quantize(&rhs, GgmlDType::Q8K)?;
    let rhs = quantized::QMatMul::from_qtensor(rhs)?;
    let mm = rhs.forward(&lhs)?;

    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = round_vector(&[dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]);
    assert_eq!(dst, [1.266, 1.504, -0.204, 1.7]);

    ggml_matmul_error_test::<BlockQ8K>()?;
    Ok(())
}
