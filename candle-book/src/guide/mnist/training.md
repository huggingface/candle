# Candle Mnist Tutorial

## Training

First lets create a new function `make_linear` that takes in a `VarBuilder` and outputs a initalized linear layer. A `VarBuilder` builds a `VarMap` which is the datastructure that holds our trainable parameters.

```rust
use candle_core::{Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, VarMap};

fn make_linear(vs: VarBuilder, in_dim: usize, out_dim: usize) -> Result<Linear>{
    let ws = vs.get_with_hints(
        (out_dim, in_dim),
        "weight",
        candle_nn::init::DEFAULT_KAIMING_NORMAL,
    )?;
    let bound = 1. / (in_dim as f64).sqrt();
    let bs = vs.get_with_hints(
        out_dim,
        "bias",
        candle_nn::Init::Uniform {
            lo: -bound,
            up: bound,
        },
    )?;
    Ok(Linear::new(ws, Some(bs)))
}
```

First lets create a function on our model class called `new` to accept a `VarBuilder` and initalize the model. We use `VarBuilder::pp` to "push prefix" so that the first layer weights are `first.weight` and`first.bias`, and the second layer weights are `second.weight` and `second.bias`.

```rust
impl Model {
    fn new(vs: VarBuilder) -> Result<Self> {
        const IMAGE_DIM: usize = 784;
        const HIDDEN_DIM: usize = 100;
        const LABELS: usize = 10;

        let first = make_linear(vs.pp("first"), IMAGE_DIM, HIDDEN_DIM)?;
        let second = make_linear(vs.pp("second"), HIDDEN_DIM, LABELS)?;

        Ok(Self { first, second })
    }

    fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let x = self.first.forward(image)?;
        let x = x.relu()?;
        self.second.forward(&x)
    }
}
```

Now lets add `candle-datasets` to our project to use mnist.

```bash
cargo add --git https://github.com/huggingface/candle.git candle-datasets
```

Now we can write our training loop.

```rust
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{loss, ops, Linear, Module, Optimizer, VarBuilder, VarMap};

fn training_loop(
    m: candle_datasets::vision::Dataset,
) -> anyhow::Result<()> {
    let dev = Device::cuda_if_available(0)?;

    let train_labels = m.train_labels;
    let train_images = m.train_images.to_device(&dev)?;
    let train_labels = train_labels.to_dtype(DType::U32)?.to_device(&dev)?;

    // Create a new varmap for a trainable parameters
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = Model::new(vs.clone())?;

    let learning_rate = 0.05;
    let epochs = 10;

    // Crease a stochastic gradient descent object, this is what will update our parameters.
    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), learning_rate)?;
    let test_images = m.test_images.to_device(&dev)?;
    let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    for epoch in 1..epochs {
        // Standard mnsit forward pass
        let logits = model.forward(&train_images)?;
        let log_sm = ops::log_softmax(&logits, D::Minus1)?;
        
        // compute Negitive Log Likelyhood loss
        let loss = loss::nll(&log_sm, &train_labels)?;

        // Do backward pass and update weights
        sgd.backward_step(&loss)?;

        // evaluate model on test set
        let test_logits = model.forward(&test_images)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let test_accuracy = sum_ok / test_labels.dims1()? as f32;
        println!(
            "{epoch:4} train loss: {:8.5} test acc: {:5.2}%",
            loss.to_scalar::<f32>()?,
            1.   * test_accuracy
        );
    }
    Ok(())
}
```

Now for our main function:

``` rust
pub fn main() -> anyhow::Result<()> {
    let m = candle_datasets::vision::mnist::load()?;
    return training_loop(m);
}
```

Now lets run it:

```bash
$ cargo run --release

> 1 train loss:  2.35449 test acc:  0.12%
> 2 train loss:  2.30760 test acc:  0.15%
> ...
```