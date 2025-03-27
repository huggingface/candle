# candle-musicgen

Candle implementation of musicgen from [Simple and Controllable Music Generation](https://arxiv.org/pdf/2306.05284).

## Running an example

```bash
$ cargo run --example musicgen -- --prompt "90s rock song with loud guitars and heavy drums"

> tokens: [2777, 7, 2480, 2324, 28, 8002, 5507, 7, 11, 2437, 5253, 7, 1]
> Tensor[dims 1, 13; u32]
> [[[ 0.0902,  0.1256, -0.0585, ...,  0.1057, -0.5141, -0.4675],
>   [ 0.1972, -0.0268, -0.3368, ..., -0.0495, -0.3597, -0.3940],
>   [-0.0855, -0.0007,  0.2225, ..., -0.2804, -0.5360, -0.2436],
>   ...
>   [ 0.0515,  0.0235, -0.3855, ..., -0.4728, -0.6858, -0.2923],
>   [-0.3728, -0.1442, -0.1179, ..., -0.4388, -0.0287, -0.3242],
>   [ 0.0163,  0.0012, -0.0020, ...,  0.0142,  0.0173, -0.0103]]]
> Tensor[[1, 13, 768], f32]
```