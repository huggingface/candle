# bge

BAAI general encoders

## Sentence embeddings

Bert is used to compute the sentence embeddings for a prompt. The model weights
are downloaded from the hub on the first run.

```bash
cargo run --example bge --release -- --prompt "The cat sits outside"

> Loaded and encoded 244.278562ms
> [[-3.6233e-2, -1.0165e-1,  2.7755e-2,  8.0149e-3,  3.2675e-2,  1.3506e-2,
> ...
>    6.9497e-3, -2.0638e-2,  3.9833e-3,  1.0049e-1,  2.4590e-2,  3.6645e-2]]
> Tensor[[1, 768], f32]
> Took 58.801007ms
```
