## candle-rwkv

The [RWKV model](https://wiki.rwkv.com/) is a recurrent neural network model
with performance on par with transformer architectures. Several variants are
available, candle implements the v5 and v6 versions and can be used with
Eagle 7B([blog post](https://blog.rwkv.com/p/eagle-7b-soaring-past-transformers)).

```bash
$ cargo run --example rwkv --release -- --prompt "The smallest prime is "
avx: true, neon: false, simd128: false, f16c: true
temp: 0.00 repeat-penalty: 1.10 repeat-last-n: 64
The smallest prime is ϕ(2) = 2.
The smallest composite is ϕ(3) = 3.
The smallest perfect number is ϕ(5) = 5.
The smallest perfect square is ϕ(4) = 4.
The smallest perfect cube is ϕ(6) = 6.
```
