use anyhow::Result;
use candle::{Device, Tensor};

fn main() -> Result<()> {
    let x = Tensor::var(&[3f32, 1., 4.], Device::Cpu)?;
    let y = (((&x * &x)? + &x * 5f64)? + 4f64)?;
    println!("{:?}", y.to_vec1::<f32>()?);
    Ok(())
}
