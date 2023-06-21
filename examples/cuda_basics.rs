use anyhow::Result;
use candle::{DType, Device, Tensor};

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let x = Tensor::zeros(4, DType::F32, device)?;
    println!("{:?}", x.to_vec1::<f32>()?);
    Ok(())
}
