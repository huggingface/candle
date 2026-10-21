// Gradient parity for the fused ops that gained a backward pass: softmax over
// the last dim, rms_norm and layer_norm. Each is checked against the composed
// (autograd-by-construction) formulation on the same input.
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::{test_device, DType, Device, Result, Tensor, Var, D};

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()
}

fn check(name: &str, a: &Tensor, b: &Tensor, tol: f32) -> Result<()> {
    let d = max_abs_diff(a, b)?;
    assert!(d <= tol, "{name}: max abs diff {d} > {tol}");
    Ok(())
}

fn softmax_bwd(device: &Device) -> Result<()> {
    let x = Var::randn(0f32, 2.0, (2, 3, 17), device)?;
    let w = Tensor::randn(0f32, 1.0, (2, 3, 17), device)?;

    let y = candle_nn::ops::softmax_last_dim(x.as_tensor())?;
    let g = (&y * &w)?.sum_all()?.backward()?;
    let gx = g.get(&x).expect("no grad for x through softmax_last_dim");

    let y_ref = candle_nn::ops::softmax(x.as_tensor(), D::Minus1)?;
    let g_ref = (&y_ref * &w)?.sum_all()?.backward()?;
    let gx_ref = g_ref.get(&x).unwrap();

    check("softmax fwd", &y, &y_ref, 1e-6)?;
    check("softmax dx", gx, gx_ref, 1e-5)?;
    Ok(())
}

fn softmax_bwd_f64(device: &Device) -> Result<()> {
    // The f64 path accumulates in f64: tighter tolerance than the f32 test.
    let x = Var::randn(0f64, 2.0, (2, 3, 17), device)?;
    let w = Tensor::randn(0f64, 1.0, (2, 3, 17), device)?;
    let y = candle_nn::ops::softmax_last_dim(x.as_tensor())?;
    let g = (&y * &w)?.sum_all()?.backward()?;
    let y_ref = candle_nn::ops::softmax(x.as_tensor(), D::Minus1)?;
    let g_ref = (&y_ref * &w)?.sum_all()?.backward()?;
    let d = (g.get(&x).unwrap() - g_ref.get(&x).unwrap())?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f64>()?;
    assert!(d <= 1e-12, "softmax dx f64: max abs diff {d}");
    Ok(())
}

fn softmax_bwd_noncontig_grad(device: &Device) -> Result<()> {
    // The incoming gradient reaches the bwd through a transpose, so the op
    // has to make it contiguous itself.
    let x = Var::randn(0f32, 2.0, (4, 6), device)?;
    let w = Tensor::randn(0f32, 1.0, (6, 4), device)?;
    let y = candle_nn::ops::softmax_last_dim(x.as_tensor())?;
    let g = (y.t()? * &w)?.sum_all()?.backward()?;
    let gx = g.get(&x).unwrap();
    let y_ref = candle_nn::ops::softmax(x.as_tensor(), D::Minus1)?;
    let g_ref = (y_ref.t()? * &w)?.sum_all()?.backward()?;
    check(
        "softmax dx (transposed grad)",
        gx,
        g_ref.get(&x).unwrap(),
        1e-5,
    )
}

fn softmax_bf16_fwd(device: &Device) -> Result<()> {
    // The fused kernel rounds once at the store: it must match an f32 softmax
    // rounded to bf16 to within one bf16 ulp of a value in [0, 1].
    let x = Tensor::randn(0f32, 3.0, (8, 64), device)?;
    let y = candle_nn::ops::softmax_last_dim(&x.to_dtype(DType::BF16)?)?;
    let y_ref = candle_nn::ops::softmax_last_dim(&x.to_dtype(DType::BF16)?.to_dtype(DType::F32)?)?;
    check("bf16 softmax vs f32", &y, &y_ref, 4e-3)
}

fn rms_norm_bwd(device: &Device) -> Result<()> {
    let x = Var::randn(0f32, 1.5, (2, 5, 32), device)?;
    let alpha = Var::randn(1f32, 0.3, 32, device)?;
    let w = Tensor::randn(0f32, 1.0, (2, 5, 32), device)?;
    let eps = 1e-5;

    let y = candle_nn::ops::rms_norm(x.as_tensor(), alpha.as_tensor(), eps)?;
    let g = (&y * &w)?.sum_all()?.backward()?;
    let y_ref = candle_nn::ops::rms_norm_slow(x.as_tensor(), alpha.as_tensor(), eps)?;
    let g_ref = (&y_ref * &w)?.sum_all()?.backward()?;

    check("rms_norm fwd", &y, &y_ref, 1e-5)?;
    check(
        "rms_norm dx",
        g.get(&x).unwrap(),
        g_ref.get(&x).unwrap(),
        1e-4,
    )?;
    check(
        "rms_norm dalpha",
        g.get(&alpha).unwrap(),
        g_ref.get(&alpha).unwrap(),
        1e-4,
    )?;
    Ok(())
}

fn layer_norm_bwd(device: &Device) -> Result<()> {
    let x = Var::randn(0f32, 1.5, (2, 5, 32), device)?;
    let alpha = Var::randn(1f32, 0.3, 32, device)?;
    let beta = Var::randn(0f32, 0.3, 32, device)?;
    let w = Tensor::randn(0f32, 1.0, (2, 5, 32), device)?;
    let eps = 1e-5;

    let y = candle_nn::ops::layer_norm(x.as_tensor(), alpha.as_tensor(), beta.as_tensor(), eps)?;
    let g = (&y * &w)?.sum_all()?.backward()?;
    let y_ref =
        candle_nn::ops::layer_norm_slow(x.as_tensor(), alpha.as_tensor(), beta.as_tensor(), eps)?;
    let g_ref = (&y_ref * &w)?.sum_all()?.backward()?;

    check("layer_norm fwd", &y, &y_ref, 1e-5)?;
    check(
        "layer_norm dx",
        g.get(&x).unwrap(),
        g_ref.get(&x).unwrap(),
        1e-4,
    )?;
    check(
        "layer_norm dalpha",
        g.get(&alpha).unwrap(),
        g_ref.get(&alpha).unwrap(),
        1e-4,
    )?;
    check(
        "layer_norm dbeta",
        g.get(&beta).unwrap(),
        g_ref.get(&beta).unwrap(),
        1e-4,
    )?;
    Ok(())
}

test_device!(
    softmax_bwd,
    softmax_bwd_cpu,
    softmax_bwd_gpu,
    softmax_bwd_metal
);
test_device!(
    softmax_bwd_noncontig_grad,
    softmax_bwd_noncontig_grad_cpu,
    softmax_bwd_noncontig_grad_gpu,
    softmax_bwd_noncontig_grad_metal
);
test_device!(
    softmax_bf16_fwd,
    softmax_bf16_fwd_cpu,
    softmax_bf16_fwd_gpu,
    softmax_bf16_fwd_metal
);
test_device!(
    rms_norm_bwd,
    rms_norm_bwd_cpu,
    rms_norm_bwd_gpu,
    rms_norm_bwd_metal
);
test_device!(
    layer_norm_bwd,
    layer_norm_bwd_cpu,
    layer_norm_bwd_gpu,
    layer_norm_bwd_metal
);
test_device!(
    softmax_bwd_f64,
    softmax_bwd_f64_cpu,
    softmax_bwd_f64_gpu,
    softmax_bwd_f64_metal
);
