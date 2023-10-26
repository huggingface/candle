# candle-jina-bert

Jina-Bert is a general large language model with a context size of 8192, [model
card](https://huggingface.co/jinaai/jina-embeddings-v2-base-en). In this example
it can be used for two different tasks:
- Compute sentence embeddings for a prompt.
- Compute similarities between a set of sentences.


## Sentence embeddings

Jina-Bert is used to compute the sentence embeddings for a prompt. The model weights
are downloaded from the hub on the first run.

```bash
cargo run --example jina-bert --release -- --prompt "Here is a test sentence"

> [[[ 0.1595, -0.9885,  0.6494, ...,  0.3003, -0.6901, -1.2355],
>   [ 0.0374, -0.1798,  1.3359, ...,  0.6731,  0.2133, -1.6807],
>   [ 0.1700, -0.8534,  0.8924, ..., -0.1785, -0.0727, -1.5087],
>   ...
>   [-0.3113, -1.3665,  0.2027, ..., -0.2519,  0.1711, -1.5811],
>   [ 0.0907, -1.0492,  0.5382, ...,  0.0242, -0.7077, -1.0830],
>   [ 0.0369, -0.6343,  0.6105, ...,  0.0671,  0.3778, -1.1505]]]
> Tensor[[1, 7, 768], f32]
```

## Similarities

In this example, Jina-Bert is used to compute the sentence embeddings for a set of
sentences (hardcoded in the examples). Then cosine similarities are computed for
each sentence pair and they are reported by decreasing values, hence the first
reported pair contains the two sentences that have the highest similarity score.
The sentence embeddings are computed using average pooling through all the
sentence tokens, including some potential padding.

```bash
cargo run --example jina-bert --release

> score: 0.94 'The new movie is awesome' 'The new movie is so great'
> score: 0.81 'The cat sits outside' 'The cat plays in the garden'
> score: 0.78 'I love pasta' 'Do you like pizza?'
> score: 0.68 'I love pasta' 'The new movie is awesome'
> score: 0.67 'A man is playing guitar' 'A woman watches TV'
```
