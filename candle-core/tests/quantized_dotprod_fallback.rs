use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Module, Result, Tensor};

#[test]
fn quantized_matmul_without_dotprod() -> Result<()> {
    let device = Device::Cpu;
    let (out, k) = (64usize, 256usize);
    let weights = (0..out * k)
        .map(|i| (((i * 7919) % 1000) as f32 / 1000.0 - 0.5) * 0.5)
        .collect::<Vec<_>>();
    let input = (0..k)
        .map(|i| ((i * 104729) % 997) as f32 / 997.0 - 0.5)
        .collect::<Vec<_>>();
    let weights = Tensor::from_vec(weights, (out, k), &device)?;
    let input = Tensor::from_vec(input, (1, k), &device)?;

    for dtype in [
        GgmlDType::Q4_0,
        GgmlDType::Q8_0,
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
        GgmlDType::Q8K,
    ] {
        let quantized = QTensor::quantize(&weights, dtype)?;
        let expected = input.matmul(&quantized.dequantize(&device)?.t()?)?;
        let actual = QMatMul::from_qtensor(quantized)?.forward(&input)?;
        let difference = (&actual - &expected)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(
            difference < 0.01,
            "{dtype:?} max absolute error: {difference}"
        );
    }
    Ok(())
}
