# Hello world!

We will now create the hello world of the ML world, building a model capable of solving MNIST dataset.

Open `src/main.rs` and fill in this content:

```rust
# extern crate candle_core;
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
cargo run --release
```

## Using a `Linear` layer.

Now that we have this, we might want to complexify things a bit, for instance by adding `bias` and creating
the classical `Linear` layer. We can do as such

```rust
# extern crate candle_core;
# use candle_core::{Device, Result, Tensor};
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
```

This will change the model running code into a new function

```rust
# extern crate candle_core;
# use candle_core::{Device, Result, Tensor};
# struct Linear{
#     weight: Tensor,
#     bias: Tensor,
# }
# impl Linear{
#     fn forward(&self, x: &Tensor) -> Result<Tensor> {
#         let x = x.matmul(&self.weight)?;
#         x.broadcast_add(&self.bias)
#     }
# }
# 
# struct Model {
#     first: Linear,
#     second: Linear,
# }
# 
# impl Model {
#     fn forward(&self, image: &Tensor) -> Result<Tensor> {
#         let x = self.first.forward(image)?;
#         let x = x.relu()?;
#         self.second.forward(&x)
#     }
# }
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

Now it works, it is a great way to create your own layers.
But most of the classical layers are already implemented in [candle-nn](https://github.com/huggingface/candle/tree/main/candle-nn).

## Using `candle_nn`.

For instance [Linear](https://github.com/huggingface/candle/blob/main/candle-nn/src/linear.rs) is already there.
This Linear is coded with PyTorch layout in mind, to reuse better existing models out there, so it uses the transpose of the weights and not the weights directly.

So instead we can simplify our example:

```bash
cargo add --git https://github.com/huggingface/candle.git candle-nn
```

And rewrite our examples using it

```rust
# extern crate candle_core;
# extern crate candle_nn;
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

Feel free to modify this example to use `Conv2d` to create a classical convnet instead.


Now that we have the running dummy code we can get to more advanced topics:

- [For PyTorch users](../guide/cheatsheet.md)
- [Running existing models](../inference/inference.md)
- [Training models](../training/training.md)


