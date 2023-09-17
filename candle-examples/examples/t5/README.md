# candle-t5

## Encoder-decoder example:

```bash
$ cargo run --example t5 --release -- --model-id "t5-small" --prompt "translate to German: A beautiful candle." --decode
...
Running on CPU, to run on GPU, build this example with `--features cuda`
 Eine schöne Kerze.
9 tokens generated (2.42 token/s)
```

## Sentence embedding example:

```bash
$ cargo run --example t5 --release -- --model-id "t5-small" --prompt "A beautiful candle."
...
[[[ 0.0515, -0.0541, -0.0761, ..., -0.0392,  0.1511, -0.0265],
  [-0.0974,  0.0998, -0.1659, ..., -0.2450,  0.1738, -0.0164],
  [ 0.0624, -0.1024,  0.0430, ..., -0.1388,  0.0564, -0.2962],
  [-0.0389, -0.1173,  0.0026, ...,  0.1064, -0.1065,  0.0990],
  [ 0.1300,  0.0027, -0.0326, ...,  0.0026, -0.0317,  0.0851]]]
Tensor[[1, 5, 512], f32]
Took 303.766583ms
```
