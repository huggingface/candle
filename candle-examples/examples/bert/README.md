# candle-bert

Bert is a general large language model. In this example it can be used for two
different tasks:

- Compute sentence embeddings for a prompt.
- Compute similarities between a set of sentences.

## Sentence embeddings

Bert is used to compute the sentence embeddings for a prompt. The model weights
are downloaded from the hub on the first run.

```bash
cargo run --example bert --release -- --prompt "Here is a test sentence"

> [[[ 0.0798, -0.0665, -0.0247, ..., -0.1082, -0.1000, -0.2751],
>   [ 0.4218,  0.2690,  0.2740, ...,  0.3889,  1.3503,  0.9908],
>   [ 0.0466,  0.3041, -0.1143, ...,  0.4427,  0.6926, -0.1515],
>   ...
>   [ 0.3396,  0.4320, -0.4408, ...,  0.9212,  0.2331, -0.6777],
>   [ 0.2789,  0.7539,  0.4306, ..., -0.0095,  0.3375, -1.7529],
>   [ 0.6737,  0.7882,  0.0548, ...,  0.1836,  0.7299, -0.6617]]]
> Tensor[[1, 7, 384], f32]
```

### Custom models

You can specify different models, such as BGE, with the `--model-id` flag:

```bash
cargo run  --example bert --release -- \
--model-id BAAI/bge-large-zh-v1.5 \
--prompt "Here is a test sentence"
Loaded and encoded 435.70775ms
[[[ 3.0944e-1, -7.8455e-5,  -1.2768e0, ...,  1.3755e-2, -3.2371e-1,  2.3819e-1],
  [-2.8506e-1,  1.9953e-1,  -1.3076e0, ...,  6.9819e-2,  1.0833e-2,  -1.1512e0],
  [ 3.9892e-1,  2.0000e-1, -9.3178e-1, ..., -4.1393e-1, -4.9644e-2, -3.3786e-1],
  ...
  [ 6.0345e-1,  3.5744e-1,  -1.2672e0, ..., -6.9165e-1, -3.4973e-3, -8.4214e-1],
  [ 3.9218e-1, -3.2735e-1,  -1.3123e0, ..., -4.9318e-1, -5.1334e-1, -3.6391e-1],
  [ 3.0978e-1,  2.5662e-4,  -1.2773e0, ...,  1.3357e-2, -3.2390e-1,  2.3858e-1]]]
Tensor[[1, 9, 1024], f32]
Took 176.744667ms
```

### Gelu approximation

You can get a speedup by using an approximation of the gelu activation, with a
small loss of precision, by passing the `--approximate-gelu` flag:

```bash
$ cargo run  --example bert --release -- \
--model-id BAAI/bge-large-zh-v1.5 \
--prompt "Here is a test sentence" \
--approximate-gelu
Loaded and encoded 244.388042ms
[[[ 3.1048e-1, -6.0339e-4,  -1.2758e0, ...,  1.3718e-2, -3.2362e-1,  2.3775e-1],
  [-2.8354e-1,  1.9984e-1,  -1.3077e0, ...,  6.9390e-2,  9.9681e-3,  -1.1531e0],
  [ 3.9947e-1,  1.9917e-1, -9.3178e-1, ..., -4.1301e-1, -5.0719e-2, -3.3955e-1],
  ...
  [ 6.0499e-1,  3.5664e-1,  -1.2642e0, ..., -6.9134e-1, -3.4581e-3, -8.4471e-1],
  [ 3.9311e-1, -3.2812e-1,  -1.3105e0, ..., -4.9291e-1, -5.1270e-1, -3.6543e-1],
  [ 3.1082e-1, -2.6737e-4,  -1.2762e0, ...,  1.3319e-2, -3.2381e-1,  2.3815e-1]]]
Tensor[[1, 9, 1024], f32]
Took 116.840791ms
```

## Similarities

In this example, Bert is used to compute the sentence embeddings for a set of
sentences (hardcoded in the examples). Then cosine similarities are computed for
each sentence pair and they are reported by decreasing values, hence the first
reported pair contains the two sentences that have the highest similarity score.
The sentence embeddings are computed using average pooling through all the
sentence tokens, including some potential padding.

```bash
cargo run --example bert --release

> score: 0.85 'The new movie is awesome' 'The new movie is so great'
> score: 0.61 'The cat sits outside' 'The cat plays in the garden'
> score: 0.52 'I love pasta' 'Do you like pizza?'
> score: 0.23 'The new movie is awesome' 'Do you like pizza?'
> score: 0.22 'I love pasta' 'The new movie is awesome'
```
