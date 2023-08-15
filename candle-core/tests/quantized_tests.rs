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
    Ok(())
}
