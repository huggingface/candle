use candle_core::{Device, Tensor};
use candle_core::Result;

#[test]
fn test_reduction() -> Result<()> {
    let dev = &pollster::block_on(Device::new_webgpu(0))?; 

    let inputa = Tensor::arange(0.0f32, 24.0, dev)?.reshape((2,3,4))?;
    let inputb = Tensor::arange(0.0f32, 24.0, &Device::Cpu)?.reshape((2,3,4))?;

    let r1a = inputa.sum(0)?; println!("r1a {r1a}");
    let r1b = inputb.sum(0)?; println!("r1b {r1b}");
    let r2a = inputa.sum(1)?; println!("r2a {r2a}");
    let r2b = inputb.sum(1)?; println!("r2b {r2b}");
    let r3a = inputa.sum(2)?; println!("r3a {r3a}");
    let r3b = inputb.sum(2)?; println!("r3b {r3b}");
    let r4a = inputa.sum((1,2))?;  println!("r4a {r4a}");
    let r4b = inputb.sum((1,2))?; println!("r4b {r4b}");
    let r5a = inputa.sum((0,2))?; println!("r5a {r5a}");
    let r5b = inputb.sum((0,2))?; println!("r5b {r5b}");
    
    assert_eq!(r1a.to_vec2::<f32>()?, r1b.to_vec2::<f32>()?);
    assert_eq!(r2a.to_vec2::<f32>()?, r2b.to_vec2::<f32>()?);
    assert_eq!(r3a.to_vec2::<f32>()?, r3b.to_vec2::<f32>()?);
    assert_eq!(r4a.to_vec1::<f32>()?, r4b.to_vec1::<f32>()?);
    assert_eq!(r5a.to_vec1::<f32>()?, r5b.to_vec1::<f32>()?);

    //test transpose:
    let inputa = inputa.transpose(0, 1)?;
    let inputb = inputb.transpose(0, 1)?;

    let r1a = inputa.sum(0)?; println!("r1a {r1a}");
    let r1b = inputb.sum(0)?; println!("r1b {r1b}");
    let r2a = inputa.sum(1)?; println!("r2a {r2a}");
    let r2b = inputb.sum(1)?; println!("r2b {r2b}");
    let r3a = inputa.sum(2)?; println!("r3a {r3a}");
    let r3b = inputb.sum(2)?; println!("r3b {r3b}");
    let r4a = inputa.sum((1,2))?;  println!("r4a {r4a}");
    let r4b = inputb.sum((1,2))?; println!("r4b {r4b}");
    let r5a = inputa.sum((0,2))?; println!("r5a {r5a}");
    let r5b = inputb.sum((0,2))?; println!("r5b {r5b}");
    
    assert_eq!(r1a.to_vec2::<f32>()?, r1b.to_vec2::<f32>()?);
    assert_eq!(r2a.to_vec2::<f32>()?, r2b.to_vec2::<f32>()?);
    assert_eq!(r3a.to_vec2::<f32>()?, r3b.to_vec2::<f32>()?);
    assert_eq!(r4a.to_vec1::<f32>()?, r4b.to_vec1::<f32>()?);
    assert_eq!(r5a.to_vec1::<f32>()?, r5b.to_vec1::<f32>()?);
    Ok(())
}