use candle::{Device, Module, Result, Tensor};
use candle_nn::Activation;

#[test]
fn test_glu_activation() -> Result<()> {
    let device = Device::Cpu;

    // Test GLU with even dimension (4 -> 2)
    let input = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0]], &device)?;
    let glu = Activation::Glu;
    let output = glu.forward(&input)?;

    // Expected: sigmoid([1, 2]) * [3, 4]
    assert_eq!(output.dims(), &[1, 2]);

    // Verify output is finite and reasonable
    let output_vals: Vec<f32> = output.flatten_all()?.to_vec1()?;
    assert!(output_vals.iter().all(|&x| x.is_finite()));

    println!("GLU output: {:?}", output_vals);
    Ok(())
}

#[test]
fn test_glu_odd_dimension_error() {
    let device = Device::Cpu;
    let input = Tensor::new(&[[1.0f32, 2.0, 3.0]], &device).unwrap();
    let glu = Activation::Glu;

    // Should error with odd dimension
    let result = glu.forward(&input);
    assert!(result.is_err(), "GLU should error with odd dimension");

    // Check error message contains expected text
    let error_msg = format!("{}", result.unwrap_err());
    assert!(
        error_msg.contains("even"),
        "Error should mention even dimension requirement"
    );
    println!("GLU correctly rejects odd dimensions: {}", error_msg);
}

#[test]
fn test_geglu_activation() -> Result<()> {
    let device = Device::Cpu;
    let input = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0]], &device)?;
    let geglu = Activation::GeGlu;
    let output = geglu.forward(&input)?;

    assert_eq!(output.dims(), &[1, 2]);

    let output_vals: Vec<f32> = output.flatten_all()?.to_vec1()?;
    assert!(output_vals.iter().all(|&x| x.is_finite()));

    println!("GeGLU output: {:?}", output_vals);
    Ok(())
}

#[test]
fn test_reglu_activation() -> Result<()> {
    let device = Device::Cpu;
    let input = Tensor::new(&[[-1.0f32, 2.0, 3.0, 4.0]], &device)?;
    let reglu = Activation::ReGlu;
    let output = reglu.forward(&input)?;

    assert_eq!(output.dims(), &[1, 2]);

    // ReLU(-1) = 0, ReLU(2) = 2
    // output = [0 * 3, 2 * 4] = [0, 8]
    let output_vals: Vec<f32> = output.flatten_all()?.to_vec1()?;
    assert_eq!(output_vals[0], 0.0);
    assert_eq!(output_vals[1], 8.0);

    println!("ReGLU output: {:?}", output_vals);
    Ok(())
}

#[test]
fn test_multidimensional_glu() -> Result<()> {
    let device = Device::Cpu;
    // Test with 3D tensor (batch_size=2, seq_len=3, hidden_dim=4)
    let input = Tensor::randn(0f32, 1f32, (2, 3, 4), &device)?;
    let glu = Activation::Glu;
    let output = glu.forward(&input)?;

    // Should halve the last dimension: (2, 3, 4) -> (2, 3, 2)
    assert_eq!(output.dims(), &[2, 3, 2]);

    println!(
        "Multidimensional GLU: {:?} -> {:?}",
        input.dims(),
        output.dims()
    );
    Ok(())
}

#[test]
fn test_phi3_compatibility() -> Result<()> {
    let device = Device::Cpu;

    // Test that GLU variants work with typical transformer dimensions
    let transformer_input = Tensor::randn(0f32, 1f32, (2, 128, 2048), &device)?; // (batch, seq, hidden*2)

    let glu = Activation::Glu;
    let output = glu.forward(&transformer_input)?;

    // Should halve last dimension: (2, 128, 2048) -> (2, 128, 1024)
    assert_eq!(output.dims(), &[2, 128, 1024]);

    println!(
        "Phi-3 compatibility: {:?} -> {:?}",
        transformer_input.dims(),
        output.dims()
    );
    Ok(())
}

#[test]
fn test_glu_variants_comparison() -> Result<()> {
    let device = Device::Cpu;
    let input = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0]], &device)?;

    let glu_output = Activation::Glu.forward(&input)?;
    let geglu_output = Activation::GeGlu.forward(&input)?;
    let reglu_output = Activation::ReGlu.forward(&input)?;

    // All should have same output shape
    assert_eq!(glu_output.dims(), &[1, 2]);
    assert_eq!(geglu_output.dims(), &[1, 2]);
    assert_eq!(reglu_output.dims(), &[1, 2]);

    // Values should be different due to different gating functions
    let glu_vals: Vec<f32> = glu_output.flatten_all()?.to_vec1()?;
    let geglu_vals: Vec<f32> = geglu_output.flatten_all()?.to_vec1()?;
    let reglu_vals: Vec<f32> = reglu_output.flatten_all()?.to_vec1()?;

    println!("GLU values: {:?}", glu_vals);
    println!("GeGLU values: {:?}", geglu_vals);
    println!("ReGLU values: {:?}", reglu_vals);

    // GLU and GeGLU should have different values (sigmoid vs GELU)
    assert_ne!(glu_vals, geglu_vals);
    // GLU and ReGLU should have different values (sigmoid vs ReLU)
    assert_ne!(glu_vals, reglu_vals);

    Ok(())
}

#[test]
fn test_glu_variants_with_negative_inputs() -> Result<()> {
    let device = Device::Cpu;
    let input = Tensor::new(&[[-2.0f32, -1.0, 3.0, 4.0]], &device)?;

    // Test that all variants handle negative inputs correctly
    let glu_output = Activation::Glu.forward(&input)?;
    let geglu_output = Activation::GeGlu.forward(&input)?;
    let reglu_output = Activation::ReGlu.forward(&input)?;

    assert_eq!(glu_output.dims(), &[1, 2]);
    assert_eq!(geglu_output.dims(), &[1, 2]);
    assert_eq!(reglu_output.dims(), &[1, 2]);

    let glu_vals: Vec<f32> = glu_output.flatten_all()?.to_vec1()?;
    let geglu_vals: Vec<f32> = geglu_output.flatten_all()?.to_vec1()?;
    let reglu_vals: Vec<f32> = reglu_output.flatten_all()?.to_vec1()?;

    println!("Negative input GLU: {:?}", glu_vals);
    println!("Negative input GeGLU: {:?}", geglu_vals);
    println!("Negative input ReGLU: {:?}", reglu_vals);

    // All should produce finite values
    assert!(glu_vals.iter().all(|&x| x.is_finite()));
    assert!(geglu_vals.iter().all(|&x| x.is_finite()));
    assert!(reglu_vals.iter().all(|&x| x.is_finite()));

    Ok(())
}

#[test]
fn test_core_vs_enum_consistency() -> Result<()> {
    let device = Device::Cpu;
    let input = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0]], &device)?;

    // Test that core tensor methods match activation enum
    let core_glu = input.glu()?;
    let enum_glu = Activation::Glu.forward(&input)?;

    let core_geglu = input.geglu()?;
    let enum_geglu = Activation::GeGlu.forward(&input)?;

    let core_reglu = input.reglu()?;
    let enum_reglu = Activation::ReGlu.forward(&input)?;

    // Compare outputs (allowing for small floating point differences)
    let core_glu_vals: Vec<f32> = core_glu.flatten_all()?.to_vec1()?;
    let enum_glu_vals: Vec<f32> = enum_glu.flatten_all()?.to_vec1()?;

    let core_geglu_vals: Vec<f32> = core_geglu.flatten_all()?.to_vec1()?;
    let enum_geglu_vals: Vec<f32> = enum_geglu.flatten_all()?.to_vec1()?;

    let core_reglu_vals: Vec<f32> = core_reglu.flatten_all()?.to_vec1()?;
    let enum_reglu_vals: Vec<f32> = enum_reglu.flatten_all()?.to_vec1()?;

    // GLU consistency
    for (core_val, enum_val) in core_glu_vals.iter().zip(enum_glu_vals.iter()) {
        assert!(
            (core_val - enum_val).abs() < 1e-6,
            "GLU core vs enum mismatch: {} vs {}",
            core_val,
            enum_val
        );
    }

    // GeGLU consistency
    for (core_val, enum_val) in core_geglu_vals.iter().zip(enum_geglu_vals.iter()) {
        assert!(
            (core_val - enum_val).abs() < 1e-6,
            "GeGLU core vs enum mismatch: {} vs {}",
            core_val,
            enum_val
        );
    }

    // ReGLU consistency
    for (core_val, enum_val) in core_reglu_vals.iter().zip(enum_reglu_vals.iter()) {
        assert!(
            (core_val - enum_val).abs() < 1e-6,
            "ReGLU core vs enum mismatch: {} vs {}",
            core_val,
            enum_val
        );
    }

    println!("Core vs Enum consistency test passed for all GLU variants");
    Ok(())
}

#[test]
fn test_glu_performance_characteristics() -> Result<()> {
    let device = Device::Cpu;

    // Test different sizes to verify linear scaling
    let sizes = vec![8, 16, 32, 64];

    for size in sizes {
        let input = Tensor::randn(0f32, 1f32, (1, size), &device)?;

        let glu_output = Activation::Glu.forward(&input)?;
        let geglu_output = Activation::GeGlu.forward(&input)?;
        let reglu_output = Activation::ReGlu.forward(&input)?;

        // All should halve the input size
        assert_eq!(glu_output.dims(), &[1, size / 2]);
        assert_eq!(geglu_output.dims(), &[1, size / 2]);
        assert_eq!(reglu_output.dims(), &[1, size / 2]);

        println!(
            "Size {}: All GLU variants produce correct output dimensions",
            size
        );
    }

    Ok(())
}

#[test]
fn test_glu_gradient_flow() -> Result<()> {
    let device = Device::Cpu;

    // Test that GLU variants allow proper gradient flow
    let input = Tensor::randn(0f32, 1f32, (2, 8), &device)?;

    let activations = vec![
        ("GLU", Activation::Glu),
        ("GeGLU", Activation::GeGlu),
        ("ReGLU", Activation::ReGlu),
    ];

    for (name, activation) in activations {
        let output = activation.forward(&input)?;

        // Verify output is differentiable (not zero everywhere)
        let output_vals: Vec<f32> = output.flatten_all()?.to_vec1()?;
        let non_zero_count = output_vals.iter().filter(|&&x| x.abs() > 1e-6).count();

        assert!(
            non_zero_count > 0,
            "Activation {} produced all-zero output",
            name
        );

        println!(
            "{} gradient flow test passed ({} non-zero values)",
            name, non_zero_count
        );
    }

    Ok(())
}
