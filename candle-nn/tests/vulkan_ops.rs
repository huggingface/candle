//! Parity of the candle-nn custom ops on Vulkan against the CPU reference.
//! Skips (passes) when no hardware Vulkan device is available.
#![cfg(feature = "vulkan")]

use candle::{DType, Device, Result, Tensor, D};

fn vulkan() -> Option<Device> {
    match Device::new_vulkan(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("skipping vulkan test: {e}");
            None
        }
    }
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.to_device(&Device::Cpu)?.flatten_all()?;
    let b = b.to_device(&Device::Cpu)?.flatten_all()?;
    (a - b)?.abs()?.max(0)?.to_scalar::<f32>()
}

#[test]
fn vulkan_nn_ops_match_cpu() -> Result<()> {
    let Some(vk) = vulkan() else { return Ok(()) };
    let cpu = Device::Cpu;
    candle::vulkan_backend::reset_stats();

    let x = Tensor::randn(0f32, 1.0, (2, 4, 7, 64), &cpu)?;
    let xv = x.to_device(&vk)?;

    // softmax over the last dim
    let d = max_abs_diff(
        &candle_nn::ops::softmax_last_dim(&xv)?,
        &candle_nn::ops::softmax_last_dim(&x)?,
    )?;
    assert!(d < 1e-6, "softmax_last_dim diff {d}");

    // rms norm
    let alpha = Tensor::randn(0f32, 1.0, 64, &cpu)?;
    let d = max_abs_diff(
        &candle_nn::ops::rms_norm(&xv, &alpha.to_device(&vk)?, 1e-6)?,
        &candle_nn::ops::rms_norm(&x, &alpha, 1e-6)?,
    )?;
    assert!(d < 1e-4, "rms_norm diff {d}");

    // sigmoid
    let d = max_abs_diff(
        &candle_nn::ops::sigmoid(&xv)?,
        &candle_nn::ops::sigmoid(&x)?,
    )?;
    assert!(d < 1e-6, "sigmoid diff {d}");

    // rope (both pairings)
    let cos = Tensor::randn(0f32, 1.0, (7, 32), &cpu)?;
    let sin = Tensor::randn(0f32, 1.0, (7, 32), &cpu)?;
    let (cv, sv) = (cos.to_device(&vk)?, sin.to_device(&vk)?);
    let d = max_abs_diff(
        &candle_nn::rotary_emb::rope(&xv, &cv, &sv)?,
        &candle_nn::rotary_emb::rope(&x, &cos, &sin)?,
    )?;
    assert!(d < 1e-5, "rope diff {d}");
    let d = max_abs_diff(
        &candle_nn::rotary_emb::rope_i(&xv, &cv, &sv)?,
        &candle_nn::rotary_emb::rope_i(&x, &cos, &sin)?,
    )?;
    assert!(d < 1e-5, "rope_i diff {d}");

    // matmul + silu + reductions, as in a transformer MLP
    let w = Tensor::randn(0f32, 0.1, (64, 96), &cpu)?;
    let y = x.reshape((56, 64))?.matmul(&w)?;
    let yv = xv.reshape((56, 64))?.matmul(&w.to_device(&vk)?)?;
    assert!(max_abs_diff(&yv, &y)? < 1e-4, "matmul");
    let d = max_abs_diff(&candle_nn::ops::silu(&yv)?, &candle_nn::ops::silu(&y)?)?;
    assert!(d < 1e-4, "silu diff {d}");
    let d = max_abs_diff(&yv.sum_keepdim(D::Minus1)?, &y.sum_keepdim(D::Minus1)?)?;
    assert!(d < 1e-3, "sum diff {d}");
    let _ = yv.to_dtype(DType::F32)?;

    let stats = candle::vulkan_backend::stats();
    eprintln!("{stats}");
    assert_eq!(stats.cpu_fallbacks, 0, "unexpected CPU fallback: {stats}");
    assert!(stats.native_dispatches > 0);
    Ok(())
}
