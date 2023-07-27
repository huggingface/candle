# Hello world !

We will now create the hello world of the ML world, building a model capable of solving MNIST dataset.

Open `src/main.rs` and fill in with these contents:

```rust
# extern crate candle;
use candle::{DType, Device, Result, Tensor, xx};

struct Model {
    first: Tensor,
    second: Tensor,
}

impl Model {
    fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let x = image.matmul(&self.first)?;
        let x = x.relu()?;
        x.matmul(&self.second)
    }
}

fn main() -> Result<()> {
    // Use Device::new_cuda(0)?; to use the GPU.
    let device = Device::Cpu;

    let first = Tensor::zeros((784, 100), DType::F32, &device)?;
    let second = Tensor::zeros((100, 10), DType::F32, &device)?;
    let model = Model { first, second };

    let dummy_image = Tensor::zeros((1, 784), DType::F32, &device)?;

    let digit = model.forward(&dummy_image)?;
    println!("Digit {digit:?} digit");
    Ok(())
}
```

Everything should now run with:

```bash
cargo run --release
```

Now that we have the running dummy code we can get to more advanced topics:


- [For PyTorch users](./guide/cheatsheet.md)
- [Running existing models](./inference/README.md)
- [Training models](./training/README.md)


