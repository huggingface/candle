use anyhow::Result;
use candle::{Device, Tensor};

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let ids = Tensor::new(&[0u32, 2u32, 1u32, 0u32, 1u32], &device)?;
    let t = Tensor::new(&[[0f32, 1f32], [2f32, 3f32], [4f32, 5f32]], &device)?;
    let hs = Tensor::embedding(&ids, &t)?;
    println!("> {:?}", hs.to_vec2::<f32>());
    let hs = hs.sum(&[0, 1])?.reshape(&[])?;
    println!("> {:?}", hs.to_scalar::<f32>());

    let ids = Tensor::new(&[0u32, 2u32, 1u32, 0u32, 1u32], &Device::Cpu)?;
    let t = Tensor::new(&[[0f32, 1f32], [2f32, 3f32], [4f32, 5f32]], &Device::Cpu)?;
    let hs = Tensor::embedding(&ids, &t)?;
    println!("> {:?}", hs.to_vec2::<f32>());
    let hs = hs.sum(&[0, 1])?.reshape(&[])?;
    println!("> {:?}", hs.to_scalar::<f32>());

    let x = Tensor::new(&[3f32, 1., 4., 1., 5.], &device)?;
    println!("{:?}", x.to_vec1::<f32>()?);
    let y = Tensor::new(&[2f32, 7., 1., 8., 2.], &device)?;
    let z = (y + x * 3.)?;
    println!("{:?}", z.to_vec1::<f32>()?);
    println!("{:?}", z.sqrt()?.to_vec1::<f32>()?);
    let x = Tensor::new(&[[11f32, 22.], [33., 44.], [55., 66.], [77., 78.]], &device)?;
    let y = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &device)?;
    println!("{:?}", y.to_vec2::<f32>()?);
    let z = x.matmul(&y)?;
    println!("{:?}", z.to_vec2::<f32>()?);
    let x = Tensor::new(
        &[[11f32, 22.], [33., 44.], [55., 66.], [77., 78.]],
        &Device::Cpu,
    )?;
    let y = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
    println!("{:?}", y.to_vec2::<f32>()?);
    let z = x.matmul(&y)?;
    println!("{:?}", z.to_vec2::<f32>()?);
    Ok(())
}
