# Candle Mnist Tutorial

## Modeling

Open `src/main.rs` in your project folder and paste the following

```rust
use candle_core::{Device, Result, Tensor};

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

    let first = Tensor::randn(0f32, 1.0, (784, 100), &device)?;
    let second = Tensor::randn(0f32, 1.0, (100, 10), &device)?;
    let model = Model { first, second };

    let dummy_image = Tensor::randn(0f32, 1.0, (1, 784), &device)?;

    let digit = model.forward(&dummy_image)?;
    println!("Digit {digit:?} digit");
    Ok(())
}
```

Everything should now run with:

```bash
$ cargo run --release

> Digit Tensor[dims 1, 10; f32] digit
```

Since we're giving random inputs we should expect an incoherent output.

## Making a `Linear` layer.

Now that we have a basic output, lets create a more complex layer type. If we add a `bias` to the weight we
can build the classic `Linear` layer.

You can replace all the code in `src/main.rs` with the following:

```rust
use candle_core::{Device, Result, Tensor};

struct Linear{
    weight: Tensor,
    bias: Tensor,
}
impl Linear{
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.matmul(&self.weight)?;
        x.broadcast_add(&self.bias)
    }
}

struct Model {
    first: Linear,
    second: Linear,
}

impl Model {
    fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let x = self.first.forward(image)?;
        let x = x.relu()?;
        self.second.forward(&x)
    }
}

fn main() -> Result<()> {
    // Use Device::new_cuda(0)?; to use the GPU.
    // Use Device::Cpu; to use the CPU.
    let device = Device::cuda_if_available(0)?;

    // Creating a dummy model
    let weight = Tensor::randn(0f32, 1.0, (784, 100), &device)?;
    let bias = Tensor::randn(0f32, 1.0, (100, ), &device)?;
    let first = Linear{weight, bias};
    let weight = Tensor::randn(0f32, 1.0, (100, 10), &device)?;
    let bias = Tensor::randn(0f32, 1.0, (10, ), &device)?;
    let second = Linear{weight, bias};
    let model = Model { first, second };

    let dummy_image = Tensor::randn(0f32, 1.0, (1, 784), &device)?;

    // Inference on the model
    let digit = model.forward(&dummy_image)?;
    println!("Digit {digit:?} digit");
    Ok(())
}
```

Give it another run with:

```bash
$ cargo run --release

> Digit Tensor[dims 1, 10; f32] digit
```

## Using `candle_nn`.

Most of the classical layers (like [Linear](https://github.com/huggingface/candle/blob/main/candle-nn/src/linear.rs)) are already implemented in [candle-nn](https://github.com/huggingface/candle/tree/main/candle-nn).

This Linear is coded with PyTorch layout in mind, to reuse better existing models out there, so it uses the transpose of the weights and not the weights directly.

Lets simplify our example!

First add `candle-nn` as a dependency:

```bash
cargo add --git https://github.com/huggingface/candle.git candle-nn
```

Now, lets replace all the code in `src/main.rs` with:

```rust
use candle_core::{Device, Result, Tensor};
use candle_nn::{Linear, Module};

struct Model {
    first: Linear,
    second: Linear,
}

impl Model {
    fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let x = self.first.forward(image)?;
        let x = x.relu()?;
        self.second.forward(&x)
    }
}

fn main() -> Result<()> {
    // Use Device::new_cuda(0)?; to use the GPU.
    let device = Device::Cpu;

    // This has changed (784, 100) -> (100, 784) !
    let weight = Tensor::randn(0f32, 1.0, (100, 784), &device)?;
    let bias = Tensor::randn(0f32, 1.0, (100, ), &device)?;
    let first = Linear::new(weight, Some(bias));
    let weight = Tensor::randn(0f32, 1.0, (10, 100), &device)?;
    let bias = Tensor::randn(0f32, 1.0, (10, ), &device)?;
    let second = Linear::new(weight, Some(bias));
    let model = Model { first, second };

    let dummy_image = Tensor::randn(0f32, 1.0, (1, 784), &device)?;

    let digit = model.forward(&dummy_image)?;
    println!("Digit {digit:?} digit");
    Ok(())
}
```

```bash
$ cargo run --release

> Digit Tensor[dims 1, 10; f32] digit
```