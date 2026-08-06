#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use candle::{test_utils, Device, Tensor};
use candle_nn::{LayerNorm, Module};

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

/// Gradients must flow through the fused layer_norm and match the gradients of
/// the pure-tensor layer_norm_slow implementation, for x, alpha, and beta.
#[test]
fn layer_norm_grad() -> Result<()> {
    use candle::{Device, Var};

    let device = &Device::Cpu;
    let (rows, n) = (4, 6);
    let xs: Vec<f32> = (0..rows * n)
        .map(|i| (i as f32 * 0.37).sin() * 3.0)
        .collect();
    let alpha: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32 * 0.11).cos()).collect();
    let beta: Vec<f32> = (0..n).map(|i| (i as f32 * 0.23).sin()).collect();
    let weight: Vec<f32> = (0..rows * n).map(|i| (i as f32 * 0.53).cos()).collect();
    let xs = Tensor::from_vec(xs, (rows, n), device)?;
    let alpha = Tensor::from_vec(alpha, n, device)?;
    let beta = Tensor::from_vec(beta, n, device)?;
    let weight = Tensor::from_vec(weight, (rows, n), device)?;

    let grads_of = |fused: bool| -> Result<(Tensor, Tensor, Tensor)> {
        let x_var = Var::from_tensor(&xs)?;
        let a_var = Var::from_tensor(&alpha)?;
        let b_var = Var::from_tensor(&beta)?;
        let y = if fused {
            candle_nn::ops::layer_norm(
                x_var.as_tensor(),
                a_var.as_tensor(),
                b_var.as_tensor(),
                1e-5,
            )?
        } else {
            candle_nn::ops::layer_norm_slow(
                x_var.as_tensor(),
                a_var.as_tensor(),
                b_var.as_tensor(),
                1e-5,
            )?
        };
        let loss = (y * &weight)?.sum_all()?;
        let grads = loss.backward()?;
        let err = || candle::Error::Msg("missing gradient".to_string());
        Ok((
            grads.get(&x_var).cloned().ok_or_else(err)?,
            grads.get(&a_var).cloned().ok_or_else(err)?,
            grads.get(&b_var).cloned().ok_or_else(err)?,
        ))
    };

    let (gx_f, ga_f, gb_f) = grads_of(true)?;
    let (gx_s, ga_s, gb_s) = grads_of(false)?;
    for (name, f, s) in [
        ("x", gx_f, gx_s),
        ("alpha", ga_f, ga_s),
        ("beta", gb_f, gb_s),
    ] {
        let diff = (&f - &s)?.abs()?.max_all()?.to_vec0::<f32>()?;
        assert!(diff < 1e-4, "grad mismatch for {name}: {diff}");
    }
    Ok(())
}
