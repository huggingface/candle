# Candle MNIST Tutorial

## Modeling

Open `src/main.rs` in your project folder and insert the following code:

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
    // Use Device::new_cuda(0)?; to utilize GPU acceleration.
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

Execute the program with:

```bash
$ cargo run --release

> Digit Tensor[dims 1, 10; f32] digit
```

Since random inputs are provided, expect an incoherent output.

## Implementing a `Linear` Layer

To create a more sophisticated layer type, add a `bias` to the weight to construct the standard `Linear` layer.

Replace the entire content of `src/main.rs` with:

```rust
use candle_core::{Device, Result, Tensor};

struct Linear {
    weight: Tensor,
    bias: Tensor,
}

impl Linear {
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
    // Use Device::new_cuda(0)?; for GPU acceleration.
    // Use Device::Cpu; for CPU computation.
    let device = Device::cuda_if_available(0)?;

    // Initialize model parameters
    let weight = Tensor::randn(0f32, 1.0, (784, 100), &device)?;
    let bias = Tensor::randn(0f32, 1.0, (100, ), &device)?;
    let first = Linear { weight, bias };
    let weight = Tensor::randn(0f32, 1.0, (100, 10), &device)?;
    let bias = Tensor::randn(0f32, 1.0, (10, ), &device)?;
    let second = Linear { weight, bias };
    let model = Model { first, second };

    let dummy_image = Tensor::randn(0f32, 1.0, (1, 784), &device)?;

    // Perform inference
    let digit = model.forward(&dummy_image)?;
    println!("Digit {digit:?} digit");
    Ok(())
}
```

Execute again with:

```bash
$ cargo run --release

> Digit Tensor[dims 1, 10; f32] digit
```

## Utilizing `candle_nn`

Many classical layers (such as [Linear](https://github.com/huggingface/candle/blob/main/candle-nn/src/linear.rs)) are already implemented in [candle-nn](https://github.com/huggingface/candle/tree/main/candle-nn).

This `Linear` implementation follows PyTorch conventions for improved compatibility with existing models, utilizing the transpose of weights rather than direct weights.

Let's simplify our implementation. First, add `candle-nn` as a dependency:

```bash
$ cargo add --git https://github.com/huggingface/candle.git candle-nn
```

Now, replace the entire content of `src/main.rs` with:

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
    // Use Device::new_cuda(0)?; for GPU acceleration.
    let device = Device::Cpu;

    // Note the dimension change: (784, 100) -> (100, 784)
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

Execute the final version:

```bash
$ cargo run --release

> Digit Tensor[dims 1, 10; f32] digit
```