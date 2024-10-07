# candle-stella-en-v5: Implementation of [stella_en_1.5B_v5](https://huggingface.co/dunzhang/stella_en_1.5B_v5) embedding model

As of 7th Oct 2024, *Stella_en_1.5B_v5* is one of the top ranking model on `retrieval` and `reranking` tasks in [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard.

[Model card](https://huggingface.co/dunzhang/stella_en_1.5B_v5) on the HuggingFace Hub.

## Running the example

Stella_en_1.5B_v5 is used to generate text embeddings embeddings for a prompt. The model weights
are downloaded from the hub on the first run.

```bash
$ cargo run --example stella-en-v5 --release  -- --query "What are safetensors?"

> [[ 0.3905, -0.0130,  0.2072, ..., -0.1100, -0.0086,  0.6002]]
>  Tensor[[1, 1024], f32]
```

Stella_en_1.5B_v5 is trained by [MRL](https://arxiv.org/abs/2205.13147) enabling multiple embedding dimensions.

The following reproduces the example in the [model card](https://huggingface.co/dunzhang/stella_en_1.5B_v5) for a retrieval task (s2p).

```bash
$ cargo run --example stella-en-v5 --release --features <metal | cuda>
```