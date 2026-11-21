// Forward and gradient parity for `gelu_tanh` against `Tensor::gelu` and its
// composed backward, plus the single-rounding property for bf16.
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::{test_device, DType, Device, Result, Tensor, Var};

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

fn gelu_tanh_bwd(device: &Device) -> Result<()> {
    let x = Var::randn(0f32, 3.0, (3, 7, 16), device)?;
    let w = Tensor::randn(0f32, 1.0, (3, 7, 16), device)?;

    let y = candle_nn::ops::gelu_tanh(x.as_tensor())?;
    let g = (&y * &w)?.sum_all()?.backward()?;
    let y_ref = x.as_tensor().gelu()?;
    let g_ref = (&y_ref * &w)?.sum_all()?.backward()?;

    check("gelu_tanh fwd", &y, &y_ref, 1e-5)?;
    check(
        "gelu_tanh dx",
        g.get(&x).unwrap(),
        g_ref.get(&x).unwrap(),
        1e-4,
    )?;

    // Half precision: one rounding at the store, so within one bf16 ulp of
    // the f32 evaluation of the same bf16 inputs.
    let xb = x.as_tensor().to_dtype(DType::BF16)?;
    let yb = candle_nn::ops::gelu_tanh(&xb)?;
    let yb_ref = candle_nn::ops::gelu_tanh(&xb.to_dtype(DType::F32)?)?;
    let scale = yb_ref.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    check("gelu_tanh bf16 fwd", &yb, &yb_ref, 8e-3 * scale.max(1.0))?;
    Ok(())
}

test_device!(
    gelu_tanh_bwd,
    gelu_tanh_bwd_cpu,
    gelu_tanh_bwd_gpu,
    gelu_tanh_bwd_metal
);
