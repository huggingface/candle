#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use candle::{test_utils, Device, Tensor};
use candle_nn::{LayerNorm, Module};

fn run_layer_norm_test(device: &Device) -> Result<()> {
    let w = Tensor::new(&[3f32], device)?;
    let b = Tensor::new(&[0.5f32], device)?;
    let ln2 = LayerNorm::new(Tensor::cat(&[&w, &w], 0)?, Tensor::cat(&[&b, &b], 0)?, 1e-8);
    let ln3 = LayerNorm::new(
        Tensor::cat(&[&w, &w, &w], 0)?,
        Tensor::cat(&[&b, &b, &b], 0)?,
        1e-8,
    );
    let ln = LayerNorm::new(w, b, 1e-8);

    let two = Tensor::new(&[[[2f32]]], device)?;
    let res = ln.forward(&two)?.flatten_all()?;
    assert_eq!(res.to_vec1::<f32>()?, [0.5f32]);

    let inp = Tensor::new(&[[[4f32, 0f32]]], device)?;
    let res = ln2.forward(&inp)?;
    assert_eq!(res.to_vec3::<f32>()?, [[[3.5f32, -2.5]]]);

    let inp = Tensor::new(&[[[1f32, 2., 3.], [4., 5., 6.], [9., 8., 7.]]], device)?;
    let res = ln3.forward(&inp)?;
    assert_eq!(
        test_utils::to_vec3_round(&res, 4)?,
        [[
            [-3.1742, 0.5, 4.1742],
            [-3.1742, 0.5, 4.1742],
            [4.1742, 0.5, -3.1742]
        ]]
    );
    let mean = (res.sum_keepdim(2)? / 3.0)?;
    // The average value should be `b`.
    assert_eq!(
        test_utils::to_vec3_round(&mean, 4)?,
        [[[0.5], [0.5], [0.5]]]
    );
    let std = (res.broadcast_sub(&mean)?.sqr()?.sum_keepdim(2)?.sqrt()? / 3.0)?;
    // The standard deviation should be sqrt(`w`).
    assert_eq!(
        test_utils::to_vec3_round(&std, 4)?,
        [[[1.7321], [1.7321], [1.7321]]]
    );
    Ok(())
}

#[test]
fn layer_norm() -> Result<()> {
    let device = &Device::Cpu;
    run_layer_norm_test(device)
}

#[cfg(feature = "metal")]
#[test]
fn layer_norm_metal() -> Result<()> {
    let device = &Device::new_metal(0)?;
    run_layer_norm_test(device)
}

#[cfg(feature = "metal")]
#[test]
fn layer_norm_metal_f16() -> Result<()> {
    use candle::DType;

    let device = &Device::new_metal(0)?;
    let w = Tensor::new(&[3f32], device)?.to_dtype(DType::F16)?;
    let b = Tensor::new(&[0.5f32], device)?.to_dtype(DType::F16)?;
    let ln3 = LayerNorm::new(
        Tensor::cat(&[&w, &w, &w], 0)?,
        Tensor::cat(&[&b, &b, &b], 0)?,
        1e-5,
    );

    let inp = Tensor::new(&[[[1f32, 2., 3.], [4., 5., 6.], [9., 8., 7.]]], device)?
        .to_dtype(DType::F16)?;
    let res = ln3.forward(&inp)?.to_dtype(DType::F32)?;

    // F16 precision is lower, so use relaxed comparison (round to 1 decimal place)
    assert_eq!(
        test_utils::to_vec3_round(&res, 1)?,
        [[[-3.2, 0.5, 4.2], [-3.2, 0.5, 4.2], [4.2, 0.5, -3.2]]]
    );
    Ok(())
}

#[cfg(feature = "metal")]
#[test]
fn layer_norm_metal_large() -> Result<()> {
    use candle::DType;

    let device = &Device::new_metal(0)?;
    let batch_size = 8;
    let seq_len = 128;
    let hidden_size = 768; // BERT-base hidden size

    // Create layer norm with random weights and bias
    let w = Tensor::ones(hidden_size, DType::F32, device)?;
    let b = Tensor::zeros(hidden_size, DType::F32, device)?;
    let ln = LayerNorm::new(w, b, 1e-5);

    // Create random input
    let inp = Tensor::randn(0f32, 1f32, (batch_size, seq_len, hidden_size), device)?;

    // Run layer norm on Metal
    let metal_result = ln.forward(&inp)?;

    // Run same computation on CPU for comparison
    let inp_cpu = inp.to_device(&Device::Cpu)?;
    let w_cpu = Tensor::ones(hidden_size, DType::F32, &Device::Cpu)?;
    let b_cpu = Tensor::zeros(hidden_size, DType::F32, &Device::Cpu)?;
    let ln_cpu = LayerNorm::new(w_cpu, b_cpu, 1e-5);
    let cpu_result = ln_cpu.forward(&inp_cpu)?;

    // Compare results - should be very close
    let metal_values = metal_result
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let cpu_values = cpu_result.flatten_all()?.to_vec1::<f32>()?;

    // Check that results match within tolerance
    let max_diff = metal_values
        .iter()
        .zip(cpu_values.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < 1e-4,
        "Max difference {} exceeds tolerance",
        max_diff
    );
    Ok(())
}
