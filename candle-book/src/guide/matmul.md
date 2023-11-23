# Simple Matrix Multiplication

Write the following to your `myapp/src/main.rs` file:
```rust
use candle_core::{Device, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    let a = Tensor::randn(0f32, 1., (2, 3), &device)?;
    let b = Tensor::randn(0f32, 1., (3, 4), &device)?;

    let c = a.matmul(&b)?;
    println!("{c}");
    Ok(())
}
```

`cargo run` should display a tensor of shape `Tensor[[2, 4], f32]`.

Having installed `candle` with Cuda support, simply define the `device` to be on GPU:

```diff
- let device = Device::Cpu;
+ let device = Device::new_cuda(0)?;
```