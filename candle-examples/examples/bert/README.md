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
