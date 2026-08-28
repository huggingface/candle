// Per-op CPU (enqueue) overhead probe for the SYCL backend.
use candle_core::{Device, Tensor};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dev = Device::new_sycl(0)?;
    let x = Tensor::randn(0f32, 1.0, (1, 1024), &dev)?;
    let w = Tensor::randn(0f32, 0.1, (1024, 1024), &dev)?;
    for _ in 0..50 {
        let _ = x.affine(2.0, 1.0)?;
    }
    dev.synchronize()?;

    let t = Instant::now();
    for _ in 0..1000 {
        let _y = x.affine(2.0, 1.0)?;
    }
    let enq = t.elapsed().as_secs_f64() * 1e3;
    dev.synchronize()?;
    let tot = t.elapsed().as_secs_f64() * 1e3;
    println!(
        "affine x1000: {enq:.2} ms enqueue ({:.1} us/op)  {tot:.2} ms total",
        enq * 1000.0 / 1000.0
    );

    for _ in 0..20 {
        let _ = x.matmul(&w)?;
    }
    dev.synchronize()?;
    let t = Instant::now();
    for _ in 0..500 {
        let _y = x.matmul(&w)?;
    }
    let enq = t.elapsed().as_secs_f64() * 1e3;
    dev.synchronize()?;
    let tot = t.elapsed().as_secs_f64() * 1e3;
    println!(
        "matmul(1x1024 @ 1024x1024) x500: {enq:.2} ms enqueue ({:.1} us/op)  {tot:.2} ms total",
        enq * 1000.0 / 500.0
    );
    Ok(())
}
