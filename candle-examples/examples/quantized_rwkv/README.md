## candle-rwkv

use [tensor-tools](https://github.com/huggingface/candle/blob/main/candle-core/examples/tensor-tools.rs) to quantize safetensors

```bash
$ cargo run --example quantized_rwkv --release -- --weight-files quantized.gguf --prompt "The smallest prime is "
avx: true, neon: false, simd128: false, f16c: true
temp: 0.00 repeat-penalty: 1.10 repeat-last-n: 64
The smallest prime is ϕ(2) = 2.
The smallest composite is ϕ(3) = 3.
The smallest perfect number is ϕ(5) = 5.
The smallest perfect square is ϕ(4) = 4.
The smallest perfect cube is ϕ(6) = 6.
```
