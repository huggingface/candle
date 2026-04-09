use candle::{DType, Result, Shape};
use candle_nn::{VarBuilder, VarMap};
use candle_nn::embedding;

#[test]
fn test_embedding() -> Result<()> {
    let device = candle::Device::Cpu;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let embed = embedding(10, 20, vb)?;

    assert_eq!(embed.embeddings().shape(), &Shape::from((10, 20)));

    Ok(())
}