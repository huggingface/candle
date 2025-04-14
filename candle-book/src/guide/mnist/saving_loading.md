# Candle Mnist Tutorial


## Saving an Loading

Now that we have a trained model, we likely want to save and load.
Saving and loading happens inside the `VarMap` and is stored on disk in the
[safetensors](https://huggingface.co/docs/safetensors/index) format.

### Saving

Lets modify our `training_loop` to save the weights:

```rust
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
    
    // Save the weights to this file
    varmap.save("model_weights.safetensors")?;
    Ok(())
}
```

```bash
$ cargo run --release

> 1 train loss:  2.40485 test acc:  0.11%
> 2 train loss:  2.34161 test acc:  0.14%
> 3 train loss:  2.28841 test acc:  0.17%
> 4 train loss:  2.24158 test acc:  0.19%
> 5 train loss:  2.19898 test acc:  0.23%
> 6 train loss:  2.15927 test acc:  0.26%
> 7 train loss:  2.12161 test acc:  0.29%
> 8 train loss:  2.08549 test acc:  0.32%
> 9 train loss:  2.05053 test acc:  0.35%
```

### Loading

Now that we have a saved model, lets load it. The main change we must make is to make `varmap` mutable.

```rust
fn training_loop(
    m: candle_datasets::vision::Dataset,
) -> anyhow::Result<()> {
    let dev = Device::cuda_if_available(0)?;

    let train_labels = m.train_labels;
    let train_images = m.train_images.to_device(&dev)?;
    let train_labels = train_labels.to_dtype(DType::U32)?.to_device(&dev)?;

    // Create a new varmap for a trainable parameters
    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = Model::new(vs.clone())?;

    // Load weights from this file and overwrite weights in our varmap.
    varmap.load("model_weights.safetensors")?;

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
    
    // Save the weights to this file
    varmap.save("model_weights.safetensors")?;
    Ok(())
}
```

```bash
$ cargo run --release

> 1 train loss:  2.01645 test acc:  0.38%
> 2 train loss:  1.98300 test acc:  0.41%
> 3 train loss:  1.95008 test acc:  0.44%
> 4 train loss:  1.91754 test acc:  0.47%
> 5 train loss:  1.88534 test acc:  0.50%
> 6 train loss:  1.85349 test acc:  0.53%
> 7 train loss:  1.82198 test acc:  0.56%
> 8 train loss:  1.79077 test acc:  0.59%
> 9 train loss:  1.75989 test acc:  0.61%
```

Of course this function will fail if the file doesn't exist or 
is malformed for the current model, 
so adding a check to see if the file exists could be beneficial.

