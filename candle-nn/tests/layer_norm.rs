#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use candle::{test_utils, DType, Device, Tensor};
use candle_nn::{LayerNorm, LayerNormConfig, Module, VarBuilder};
use std::collections::HashMap;

#[test]
fn layer_norm() -> Result<()> {
    let device = &Device::Cpu;
    let w = Tensor::new(&[3f32], device)?;
    let b = Tensor::new(&[0.5f32], device)?;
    let ln2 = LayerNorm::new(Tensor::cat(&[&w, &w], 0)?, Tensor::cat(&[&b, &b], 0)?, 1e-8);
    let ln3 = LayerNorm::new(
        Tensor::cat(&[&w, &w, &w], 0)?,
        Tensor::cat(&[&b, &b, &b], 0)?,
        1e-8,
    );
    let ln = LayerNorm::new(w, b, 1e-8);
    assert_eq!(ln.eps(), 1e-8);
    assert!(ln.remove_mean());

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

    // Verify that rms_norm sets remove_mean to false.
    let rms = LayerNorm::rms_norm(Tensor::new(&[1f32], device)?, 1e-5);
    assert_eq!(rms.eps(), 1e-5);
    assert!(!rms.remove_mean());

    Ok(())
}

#[test]
fn layer_norm_from_var_builder() -> Result<()> {
    let device = &Device::Cpu;
    let weight = Tensor::new(&[3f32, 3f32], device)?;
    let bias = Tensor::new(&[0.5f32, 0.5f32], device)?;

    // A checkpoint that ships no bias still loads when the norm does not use one.
    let tensors = HashMap::from([("weight".to_string(), weight.clone())]);
    let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
    let ln = candle_nn::layer_norm_no_bias(2, 1e-8, vb)?;
    assert!(ln.bias().is_none());

    // The gamma and beta spelling is still picked up for an affine norm.
    let tensors = HashMap::from([
        ("gamma".to_string(), weight.clone()),
        ("beta".to_string(), bias.clone()),
    ]);
    let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
    let ln = candle_nn::layer_norm(2, LayerNormConfig::default(), vb)?;
    assert_eq!(ln.weight().to_vec1::<f32>()?, [3f32, 3f32]);
    assert_eq!(ln.bias().unwrap().to_vec1::<f32>()?, [0.5f32, 0.5f32]);

    // An affine norm still reports a missing bias rather than silently dropping it.
    let tensors = HashMap::from([("weight".to_string(), weight)]);
    let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
    assert!(candle_nn::layer_norm(2, LayerNormConfig::default(), vb).is_err());

    Ok(())
}
