# Candle Flash Attention v3 Layer

Flash Attention v3 Layer for Hopper (compatible nvidia `sm90a` arch) and the candle framework. 

Work supported by Baseten (https://github.com/basetenlabs)
If you are working on the intersection of CUDA / LLMs and Inference already, feel free to reach out, [we are hiring.](https://www.baseten.co/careers/)

### Usage

```rust
use baseten_candle_flash_attn_v3;
use anyhow::Result;
use candle::{DType, Device, IndexOp, Tensor, D};

fn flash_attn_acausal() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let q = Tensor::arange(0u32, 3 * 2 * 64, &device)?
        .to_dtype(DType::F16)?
        .reshape((1, 3, 2, 64))?; // batch, head, seqlen, hidden_dim
    let k = (&q / 400.)?;
    let v = (&q / 500.)?;
    let q = (&q / 300.)?;

    let att = {
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        baseten_candle_flash_attn_v3::flash_attn(&q, &k, &v, 0.5, false, false)?.transpose(1, 2)?
    };
```

### Install instructions

```
[dependencies]
candle = { version = "*", package = "candle-core", default-features = false }
candle-nn = { version = "*" }
candle-transformers = { version = "*" }
baseten-candle-flash-attn-v3 = { git = "https://github.com/michaelfeil/candle-flash-attn-v3", rev = "main", optional = true }
````