# candle-distilbert

DistilBert is a distiled version of the Bert model.

## Sentence embeddings

DistilBert is used to compute the sentence embeddings for a prompt. The model weights
are downloaded from the hub on the first run.

```bash
cargo run --example distilbert --release -- --prompt "Here is a test sentence"

> [[[ 0.5109,  0.1280, -0.2635, ...,  0.3462, -1.0434,  0.1441],
>   [ 0.1735,  0.0818, -0.5549, ...,  0.3472, -0.8264, -0.0244],
>   [ 0.0702, -0.1311, -0.4914, ...,  0.3483, -0.6194,  0.1829],
>   ...
>   [ 0.2993, -0.0106, -0.4640, ...,  0.2844, -0.6732,  0.0042],
>   [ 0.1066, -0.0081, -0.4299, ...,  0.3435, -0.7729,  0.0190],
>   [ 0.8903,  0.2055, -0.2541, ...,  0.3208, -0.6585,  0.0586]]]
> Tensor[[1, 7, 768], f32]

```
