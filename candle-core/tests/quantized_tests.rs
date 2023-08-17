use candle_core::{quantized, Device, Result, Tensor};
use quantized::{k_quants, GgmlType};
mod test_utils;
use test_utils::to_vec2_round;

#[test]
fn quantized_matmul() -> Result<()> {
    let cpu = &Device::Cpu;
    let (m, k, n) = (3, 64, 4);
    let lhs = (0..(m * k)).map(|v| v as f32).collect::<Vec<_>>();
    let tensor_lhs = Tensor::from_slice(&lhs, (m, k), cpu)?;
    let mut dst = vec![42.; 3 * 4];
    let mut rhs_t = vec![k_quants::BlockQ4_0::zeros(); 8];
    let rhs = (0..(k * n)).map(|v| v as f32).collect::<Vec<_>>();
    let tensor_rhs = Tensor::from_slice(&rhs, (n, k), cpu)?.t()?;
    k_quants::BlockQ4_0::from_float(&rhs, &mut rhs_t)?;
    k_quants::matmul((m, k, n), &lhs, &rhs_t, &mut dst)?;
    assert_eq!(
        dst.iter().map(|x| x.round()).collect::<Vec<_>>(),
        &[
            85120.0, 214562.0, 345455.0, 474748.0, 213475.0, 604465.0, 1000686.0, 1388317.0,
            341876.0, 994283.0, 1655709.0, 2301518.0
        ]
    );
    let mm = tensor_lhs.matmul(&tensor_rhs)?;
    assert_eq!(
        mm.to_vec2::<f32>()?,
        &[
            [85344.0, 214368.0, 343392.0, 472416.0],
            [214368.0, 605536.0, 996704.0, 1387872.0],
            [343392.0, 996704.0, 1650016.0, 2303328.0]
        ]
    );

    let qtensor = quantized::QTensor::new(rhs_t, (4, 64));
    let matmul = quantized::QMatMul::from_qtensor(qtensor);
    let res = matmul.forward(&tensor_lhs)?;
    assert_eq!(
        to_vec2_round(&res, 0)?,
        &[
            [85120.0, 214562.0, 345455.0, 474748.0],
            [213475.0, 604465.0, 1000686.0, 1388317.0],
            [341876.0, 994283.0, 1655709.0, 2301518.0]
        ]
    );

    Ok(())
}

#[test]
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

    let qtensor = quantized::QTensor::new(rhs_t, (4, 64));
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

#[test]
fn quantize_q4_0() -> Result<()> {
    use k_quants::BlockQ4_0;

    let src = (0..32 * 4).map(|v| v as f32).collect::<Vec<_>>();
    let mut dst = vec![0f32; 32 * 4];
    let mut quant = vec![BlockQ4_0::zeros(); 4];
    BlockQ4_0::from_float(&src, &mut quant)?;
    BlockQ4_0::to_float(&quant, dst.as_mut_slice())?;
    assert_eq!(
        dst,
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
    Ok(())
}

#[test]
fn quantize_q8k() -> Result<()> {
    use k_quants::BlockQ8K;

    let src = (0..256 * 4)
        .map(|v| (v as f32 - 512.) / 1024.)
        .collect::<Vec<_>>();
    let mut dst = vec![0f32; 256 * 4];
    let mut quant = vec![BlockQ8K::zeros(); 4];
    BlockQ8K::from_float(&src, &mut quant)?;
    BlockQ8K::to_float(&quant, dst.as_mut_slice())?;
    assert_eq!(
        [src[0], src[128], src[256], src[512], src[800], src[1023]],
        [-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344]
    );
    assert_eq!(
        [dst[0], dst[128], dst[256], dst[512], dst[800], dst[1023]],
        [-0.5, -0.375, -0.25, -0.0, 0.28070068, 0.49902344]
    );
    Ok(())
}

#[test]
fn quantize_q6k() -> Result<()> {
    use k_quants::BlockQ6K;

    let src = (0..256 * 4)
        .map(|v| (v as f32 - 512.) / 1024.)
        .collect::<Vec<_>>();
    let mut dst = vec![0f32; 256 * 4];
    let mut quant = vec![BlockQ6K::zeros(); 4];
    BlockQ6K::from_float(&src, &mut quant)?;
    BlockQ6K::to_float(&quant, dst.as_mut_slice())?;
    assert_eq!(
        [src[0], src[128], src[256], src[512], src[800], src[1023]],
        [-0.5, -0.375, -0.25, 0.0, 0.28125, 0.49902344]
    );
    let dst = dst
        .iter()
        .map(|x| (1000. * x).round() / 1000.)
        .collect::<Vec<_>>();
    assert_eq!(
        [dst[0], dst[128], dst[256], dst[512], dst[800], dst[1023]],
        [-0.497, -0.372, -0.25, -0.0, 0.284, 0.5]
    );
    Ok(())
}

#[test]
fn quantized_matmul_q6k() -> Result<()> {
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(314159265358979);

    let cpu = &Device::Cpu;
    let (m, k, n) = (256, 512, 256);
    let lhs = (0..m * k).map(|_| rng.gen::<f32>()).collect::<Vec<_>>();
    let rhs = (0..n * k).map(|_| rng.gen::<f32>()).collect::<Vec<_>>();

    let lhs = Tensor::from_vec(lhs, (m, k), cpu)?;
    let rhs = Tensor::from_vec(rhs, (n, k), cpu)?;

    let mm = lhs.matmul(&rhs.t()?)?;
    assert_eq!(mm.dims(), [m, n]);
    let dst = mm.flatten_all()?.to_vec1::<f32>()?;
    let dst = [dst[0], dst[m * n / 3], dst[m * n * 2 / 3], dst[m * n - 1]]
        .iter()
        .map(|x| (1000. * x).round() / 1000.)
        .collect::<Vec<_>>();
    assert_eq!(dst, [130.499, 131.039, 129.785, 135.439]);

    Ok(())
}
