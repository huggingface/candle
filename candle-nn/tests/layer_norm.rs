#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use candle::{test_utils, DType, Device, Tensor};
use candle_nn::{LayerNorm, Module, VarBuilder, VarMap};
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
fn rms_norm_from_varmap() -> Result<()> {
    // Regression for #3972: fresh VarMap must allow lazy Init, and affine=false
    // must not probe bias/beta names.
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let rms = candle_nn::rms_norm(8, 1e-6, vb.pp("norm"))?;
    assert_eq!(rms.weight().dims(), &[8]);
    assert!(varmap.data().lock().unwrap().contains_key("norm.weight"));
    assert!(!varmap.data().lock().unwrap().contains_key("norm.bias"));
    Ok(())
}

#[test]
fn layer_norm_from_varmap() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let ln = candle_nn::layer_norm(4, 1e-5, vb.pp("ln"))?;
    assert_eq!(ln.weight().dims(), &[4]);
    assert!(ln.bias().is_some());
    assert_eq!(ln.bias().unwrap().dims(), &[4]);
    let data = varmap.data().lock().unwrap();
    assert!(data.contains_key("ln.weight"));
    assert!(data.contains_key("ln.bias"));
    Ok(())
}

#[test]
fn layer_norm_gamma_beta_aliases() -> Result<()> {
    let device = &Device::Cpu;
    let gamma = Tensor::new(&[1f32, 1., 1., 1.], device)?;
    let beta = Tensor::new(&[0f32, 0., 0., 0.], device)?;
    let tensors: HashMap<String, Tensor> = [
        ("ln.gamma".to_string(), gamma),
        ("ln.beta".to_string(), beta),
    ]
    .into_iter()
    .collect();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
    let ln = candle_nn::layer_norm(4, 1e-5, vb.pp("ln"))?;
    assert_eq!(ln.weight().dims(), &[4]);
    assert!(ln.bias().is_some());
    assert_eq!(ln.bias().unwrap().dims(), &[4]);
    Ok(())
}
