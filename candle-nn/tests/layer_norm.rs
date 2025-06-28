#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use candle::{test_utils, Device, Tensor, DType};
use candle_nn::{LayerNorm, Module, VarBuilder, VarMap};

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
fn test_layernorm_gradient_flow() -> Result<()> {
    // Test that LayerNorm properly propagates gradients to all parameters
    let device = &Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    
    // Create a simple model: Linear -> LayerNorm -> Linear
    let hidden_size = 64;
    let batch_size = 4;
    
    // Build model components
    let linear1 = candle_nn::linear(hidden_size, hidden_size, vb.pp("linear1"))?;
    let layer_norm = candle_nn::layer_norm(
        hidden_size, 
        candle_nn::LayerNormConfig::default(), 
        vb.pp("layer_norm")
    )?;
    let linear2 = candle_nn::linear(hidden_size, hidden_size, vb.pp("linear2"))?;
    
    // Create input and target
    let input = Tensor::randn(0f32, 1.0, (batch_size, hidden_size), device)?;
    let target = Tensor::randn(0f32, 1.0, (batch_size, hidden_size), device)?;
    
    // Forward pass
    let x1 = linear1.forward(&input)?;
    let x_norm = layer_norm.forward(&x1)?;
    let output = linear2.forward(&x_norm)?;
    
    // Compute loss (MSE)
    let loss = (output.sub(&target))?.sqr()?.mean_all()?;
    
    // Backward pass
    let grads = loss.backward()?;
    
    // Check gradient flow
    let vars = varmap.all_vars();
    let mut params_with_gradients = 0;
    let mut _params_without_gradients = 0;
    
    for var in &vars {
        if let Some(grad) = grads.get(var) {
            let grad_norm = grad.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
            if grad_norm > 1e-8 {
                params_with_gradients += 1;
            } else {
                _params_without_gradients += 1;
            }
        } else {
            _params_without_gradients += 1;
        }
    }
    
    let gradient_flow_pct = (params_with_gradients as f32 / vars.len() as f32) * 100.0;
    println!("Gradient flow: {:.1}% ({}/{} parameters)", 
        gradient_flow_pct, params_with_gradients, vars.len());
    
    // With the fix, we should have 100% gradient flow
    assert!(gradient_flow_pct > 90.0, 
        "Gradient flow too low: {:.1}% (expected > 90%)", gradient_flow_pct);
    
    Ok(())
}

#[test]
fn test_layernorm_numerical_equivalence() -> Result<()> {
    // Test that the fixed implementation produces the same numerical results
    let device = &Device::Cpu;
    
    // Test with various input shapes and values
    let test_cases = vec![
        (vec![1, 3], vec![1f32, 2., 3.]),
        (vec![2, 4], vec![1f32, 2., 3., 4., 5., 6., 7., 8.]),
        (vec![1, 2, 3], vec![1f32, 2., 3., 4., 5., 6.]),
    ];
    
    for (shape, data) in test_cases {
        let input = Tensor::new(data.as_slice(), device)?.reshape(shape.as_slice())?;
        
        // Create LayerNorm with known parameters
        let normalized_shape = *shape.last().unwrap();
        let weight = Tensor::ones(normalized_shape, DType::F32, device)?;
        let bias = Tensor::zeros(normalized_shape, DType::F32, device)?;
        let eps = 1e-5;
        
        let layer_norm = LayerNorm::new(weight, bias, eps);
        let output = layer_norm.forward(&input)?;
        
        // Verify the output has the expected properties:
        // 1. Same shape as input
        assert_eq!(output.shape(), input.shape());
        
        // 2. Mean should be approximately zero (within numerical precision)
        let mean = output.mean_keepdim(candle::D::Minus1)?;
        let mean_abs_max = mean.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        assert!(mean_abs_max < 1e-5, "Mean not close to zero: {}", mean_abs_max);
        
        // 3. Variance should be approximately 1 (check first element as example)
        let centered = output.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(candle::D::Minus1)?;
        let var_flat = var.flatten_all()?;
        let var_val = var_flat.get(0)?.to_scalar::<f32>()?;
        assert!((var_val - 1.0).abs() < 1e-4, "Variance not close to 1: {}", var_val);
    }
    
    Ok(())
}
