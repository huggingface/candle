use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::var_map::ConcurrentVarMap;
use candle_nn::{VarBuilder, VarMap};

#[test]
fn test_with_neural_network_layers() -> Result<()> {
    let device = Device::Cpu;

    // Test with original VarMap
    let varmap1 = VarMap::new();
    let vb1 = VarBuilder::from_varmap(&varmap1, DType::F32, &device);
    let layer1 = candle_nn::linear(768, 512, vb1.pp("layer1"))?;

    // Test with updated VarMap
    let varmap2 = VarMap::new();
    let vb2 = VarBuilder::from_varmap(&varmap2, DType::F32, &device);
    let layer2 = candle_nn::linear(768, 512, vb2.pp("layer1"))?;

    // Test with ConcurrentVarMap - now we need to handle it differently
    // since from_varmap expects VarMap specifically
    let varmap3 = ConcurrentVarMap::new();
    let vb3 = VarBuilder::from_backend(Box::new(varmap3.clone()), DType::F32, device.clone());
    let layer3 = candle_nn::linear(768, 512, vb3.pp("layer1"))?;

    // All should work identically
    let input = Tensor::randn(0f32, 1f32, (32, 768), &device)?;

    let out1 = layer1.forward(&input)?;
    let out2 = layer2.forward(&input)?;
    let out3 = layer3.forward(&input)?;

    assert_eq!(out1.shape(), out2.shape());
    assert_eq!(out2.shape(), out3.shape());

    Ok(())
}
