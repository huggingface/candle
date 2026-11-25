use candle::{DType, Device, Result, Tensor};
use std::time::Instant;

#[test]
fn bf16_lfm_op_timings() -> Result<()> {
    let dev = Device::Cpu;
    // LFM ShortConv shape: depthwise conv over [1, 2048, 8192], k=3
    let x = Tensor::rand(-1f32, 1f32, (1, 2048, 8192), &dev)?;
    let w = Tensor::rand(-1f32, 1f32, (2048, 1, 3), &dev)?;
    for dt in [DType::F32, DType::BF16] {
        let xd = x.to_dtype(dt)?;
        let wd = w.to_dtype(dt)?;
        let t = Instant::now();
        for _ in 0..3 {
            let _ = xd.conv1d(&wd, 2, 1, 1, 2048)?;
        }
        println!("conv1d {dt:?}: {:?}", t.elapsed() / 3);
    }
    // rms_norm at [8192, 2048]
    let x = Tensor::rand(-1f32, 1f32, (1, 8192, 2048), &dev)?;
    let w = Tensor::rand(0.5f32, 1.5f32, (2048,), &dev)?;
    for dt in [DType::F32, DType::BF16] {
        let xd = x.to_dtype(dt)?;
        let wd = w.to_dtype(dt)?;
        let t = Instant::now();
        for _ in 0..5 {
            let _ = candle_nn::ops::rms_norm(&xd, &wd, 1e-6)?;
        }
        println!("rms_norm {dt:?}: {:?}", t.elapsed() / 5);
    }
    // silu + mul at [8192, 6144]
    let a = Tensor::rand(-1f32, 1f32, (8192, 6144), &dev)?;
    for dt in [DType::F32, DType::BF16] {
        let ad = a.to_dtype(dt)?;
        let t = Instant::now();
        for _ in 0..5 {
            let _ = (candle_nn::ops::silu(&ad)? * &ad)?;
        }
        println!("silu*mul {dt:?}: {:?}", t.elapsed() / 5);
    }
    Ok(())
}
