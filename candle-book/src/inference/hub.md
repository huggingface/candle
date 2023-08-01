# Using the hub

Install the [`hf-hub`](https://github.com/huggingface/hf-hub) crate:

```bash
cargo add hf-hub
```

Then let's start by downloading the [model file](https://huggingface.co/bert-base-uncased/tree/main).


```rust
# extern crate candle;
# extern crate hf_hub;
use hf_hub::api::sync::Api;
use candle::Device;

let api = Api::new().unwrap();
let repo = api.model("bert-base-uncased".to_string());

let weights = repo.get("model.safetensors").unwrap();

let weights = candle::safetensors::load(weights, &Device::Cpu);
```

We now have access to all the [tensors](https://huggingface.co/bert-base-uncased?show_tensors=true) within the file.


## Using async 

`hf-hub` comes with an async API.

```bash
cargo add hf-hub --features tokio
```

```rust,ignore
# extern crate candle;
# extern crate hf_hub;
use hf_hub::api::tokio::Api;
use candle::Device;

let api = Api::new().unwrap();
let repo = api.model("bert-base-uncased".to_string());

let weights = repo.get("model.safetensors").await.unwrap();

let weights = candle::safetensors::load(weights, &Device::Cpu);
```


## Using in a real model.

Now that we have our weights, we can use them in our bert architecture:

```rust
# extern crate candle;
# extern crate candle_nn;
# extern crate hf_hub;
# use hf_hub::api::sync::Api;
# 
# let api = Api::new().unwrap();
# let repo = api.model("bert-base-uncased".to_string());
# 
# let weights = repo.get("model.safetensors").unwrap();
use candle::{Device, Tensor, DType};
use candle_nn::Linear;

let weights = candle::safetensors::load(weights, &Device::Cpu).unwrap();

let weight = weights.get("bert.encoder.layer.0.attention.self.query.weight").unwrap();
let bias = weights.get("bert.encoder.layer.0.attention.self.query.bias").unwrap();

let linear = Linear::new(weight.clone(), Some(bias.clone()));

let input_ids = Tensor::zeros((3, 7680), DType::F32, &Device::Cpu).unwrap();
let output = linear.forward(&input_ids);
```

For a full reference, you can check out the full [bert](https://github.com/LaurentMazare/candle/tree/main/candle-examples/examples/bert) example.
