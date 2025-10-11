# Using the hub

Install the [`hf-hub`](https://github.com/huggingface/hf-hub) crate:

```bash
cargo add hf-hub
```

Then let's start by downloading the [model file](https://huggingface.co/bert-base-uncased/tree/main).


```rust
# extern crate candle_core;
# extern crate hf_hub;
use hf_hub::api::sync::Api;
use candle_core::Device;

let api = Api::new().unwrap();
let repo = api.model("bert-base-uncased".to_string());

let weights = repo.get("model.safetensors").unwrap();

let weights = candle_core::safetensors::load(weights, &Device::Cpu);
```

We now have access to all the [tensors](https://huggingface.co/bert-base-uncased?show_tensors=true) within the file.

You can check all the names of the tensors [here](https://huggingface.co/bert-base-uncased?show_tensors=true)


## Using async 

`hf-hub` comes with an async API.

```bash
cargo add hf-hub --features tokio
```

```rust,ignore
# This is tested directly in examples crate because it needs external dependencies unfortunately:
# See [this](https://github.com/rust-lang/mdBook/issues/706)
{{#include ../lib.rs:book_hub_1}}
```


## Using in a real model.

Now that we have our weights, we can use them in our bert architecture:

```rust
# extern crate candle_core;
# extern crate candle_nn;
# extern crate hf_hub;
# use hf_hub::api::sync::Api;
# 
# let api = Api::new().unwrap();
# let repo = api.model("bert-base-uncased".to_string());
# 
# let weights = repo.get("model.safetensors").unwrap();
use candle_core::{Device, Tensor, DType};
use candle_nn::{Linear, Module};

let weights = candle_core::safetensors::load(weights, &Device::Cpu).unwrap();

let weight = weights.get("bert.encoder.layer.0.attention.self.query.weight").unwrap();
let bias = weights.get("bert.encoder.layer.0.attention.self.query.bias").unwrap();

let linear = Linear::new(weight.clone(), Some(bias.clone()));

let input_ids = Tensor::zeros((3, 768), DType::F32, &Device::Cpu).unwrap();
let output = linear.forward(&input_ids).unwrap();
```

For a full reference, you can check out the full [bert](https://github.com/LaurentMazare/candle/tree/main/candle-examples/examples/bert) example.

## Memory mapping

For more efficient loading, instead of reading the file, you could use [`memmap2`](https://docs.rs/memmap2/latest/memmap2/)

**Note**: Be careful about memory mapping it seems to cause issues on [Windows, WSL](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/5893)
and will definitely be slower on network mounted disk, because it will issue more read calls.

```rust,ignore
{{#include ../lib.rs:book_hub_2}}
```

**Note**: This operation is **unsafe**. [See the safety notice](https://docs.rs/memmap2/latest/memmap2/struct.Mmap.html#safety).
In practice model files should never be modified, and the mmaps should be mostly READONLY anyway, so the caveat most likely does not apply, but always keep it in mind.


## Tensor Parallel Sharding

When using multiple GPUs to use in Tensor Parallel in order to get good latency, you can load only the part of the Tensor you need.

For that you need to use [`safetensors`](https://crates.io/crates/safetensors) directly.

```bash
cargo add safetensors
```


```rust,ignore
{{#include ../lib.rs:book_hub_3}}
```
