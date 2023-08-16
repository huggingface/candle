use candle_core::{quantized, Device, Result, Tensor};
use quantized::{k_quants, GgmlType};

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
        dst,
        &[
            85120.43, 214561.61, 345454.9, 474748.1, 213474.94, 604465.25, 1000686.4, 1388317.3,
            341875.88, 994283.0, 1655708.8, 2301518.3
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
        res.to_vec2::<f32>()?,
        &[
            [85120.43, 214561.61, 345454.9, 474748.1],
            [213474.94, 604465.25, 1000686.4, 1388317.3],
            [341875.88, 994283.0, 1655708.8, 2301518.3]
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
