use candle::{Device, Result, Tensor};
use candle_nn::{LinearTernary, Module};

fn cpu() -> &'static Device {
    &Device::Cpu
}

#[test]
fn weights_are_ternary_after_quantization() -> Result<()> {
    // 2-out, 3-in weight matrix; quantize with explicit threshold
    let w = Tensor::new(&[[2.0f32, -3.0, 0.1], [0.0, 1.5, -0.5]], cpu())?;
    let layer = LinearTernary::from_tensor(w, None, Some(1.0))?;
    let vals = layer.weight().to_vec2::<f32>()?;
    for row in &vals {
        for &v in row {
            assert!(v == -1.0 || v == 0.0 || v == 1.0, "non-ternary weight: {v}");
        }
    }
    Ok(())
}

#[test]
fn threshold_zero_yields_no_zeros() -> Result<()> {
    let w = Tensor::new(&[[2.0f32, -3.0], [1.5, -0.5]], cpu())?;
    let layer = LinearTernary::from_tensor(w, None, Some(0.0))?;
    let vals = layer.weight().flatten_all()?.to_vec1::<f32>()?;
    assert!(!vals.contains(&0.0), "threshold=0 should produce no zero weights");
    for v in vals {
        assert!(v == -1.0 || v == 1.0);
    }
    Ok(())
}

#[test]
fn threshold_large_yields_all_zeros() -> Result<()> {
    let w = Tensor::new(&[[2.0f32, -3.0], [1.5, -0.5]], cpu())?;
    let layer = LinearTernary::from_tensor(w, None, Some(1e9))?;
    let vals = layer.weight().flatten_all()?.to_vec1::<f32>()?;
    assert!(vals.iter().all(|&v| v == 0.0), "threshold=1e9 should zero all weights");
    Ok(())
}

#[test]
fn forward_matches_manual() -> Result<()> {
    // Inject known ternary weights directly
    let w = Tensor::new(&[[1.0f32, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 1.0]], cpu())?;
    let layer = LinearTernary::from_tensor(w, None, Some(1e9))?; // threshold kills all; then override
    // Override weight with exact ternary values (threshold=1e9 → zeros, so inject manually)
    let w_known = Tensor::new(&[[1.0f32, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 1.0]], cpu())?;
    let layer = LinearTernary::from_tensor(w_known, None, Some(0.5))?;
    let xs = Tensor::new(&[[2.0f32, 3.0, 4.0, 5.0]], cpu())?;
    let ys = layer.forward(&xs)?;
    let result = ys.to_vec2::<f32>()?;
    // row 0: 1*2 + 0*3 + (-1)*4 + 0*5 = -2
    // row 1: 0*2 + 1*3 + 0*4 + 1*5  = 8
    assert!((result[0][0] - (-2.0)).abs() < 1e-5, "row 0 mismatch: {}", result[0][0]);
    assert!((result[0][1] - 8.0).abs() < 1e-5, "row 1 mismatch: {}", result[0][1]);
    Ok(())
}

#[test]
fn sparsity_is_correct() -> Result<()> {
    // weights = [[1, -1], [0, 0]] → 50% zeros
    let w = Tensor::new(&[[2.0f32, -2.0], [0.1, 0.1]], cpu())?;
    // threshold = 1.0 → |2| > 1 → ±1, |0.1| < 1 → 0 → 2 zeros out of 4
    let layer = LinearTernary::from_tensor(w, None, Some(1.0))?;
    let s = layer.sparsity()?;
    assert!((s - 0.5).abs() < 1e-5, "expected sparsity 0.5, got {s}");
    Ok(())
}

#[test]
fn default_threshold_uses_mean_abs() -> Result<()> {
    // mean(|w|) = mean(1, 0.1) = 0.55
    // |1.0| > 0.55 → +1; |0.1| < 0.55 → 0
    let w = Tensor::new(&[[1.0f32, 0.1]], cpu())?;
    let layer = LinearTernary::from_tensor(w, None, None)?;
    let vals = layer.weight().to_vec2::<f32>()?;
    assert!((vals[0][0] - 1.0).abs() < 1e-5);
    assert!((vals[0][1] - 0.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn output_shape_with_batch() -> Result<()> {
    let w = Tensor::zeros((4, 8), candle::DType::F32, cpu())?;
    let layer = LinearTernary::from_tensor(w, None, Some(0.0))?;
    let xs = Tensor::zeros((2, 8), candle::DType::F32, cpu())?;
    let ys = layer.forward(&xs)?;
    assert_eq!(ys.dims(), &[2, 4]);
    Ok(())
}
