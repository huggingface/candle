# Training


Training starts with data. We're going to use the huggingface hub and 
start with the Hello world dataset of machine learning, MNIST.

Let's start with downloading `MNIST` from [huggingface](https://huggingface.co/datasets/mnist).


```rust
use candle_datasets::from_hub;


let dataset = from_hub("mnist")?;
```

This uses the standardized `parquet` files from the `refs/convert/parquet` branch on every dataset.
