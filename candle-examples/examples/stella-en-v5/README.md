# candle-stella-en-v5: Implementation of [stella_en_1.5B_v5](https://huggingface.co/dunzhang/stella_en_1.5B_v5) embedding model

As of 7th Oct 2024, *Stella_en_1.5B_v5* is one of the top ranking model on `retrieval` and `reranking` tasks in [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard.

[Model card](https://huggingface.co/dunzhang/stella_en_1.5B_v5) on the HuggingFace Hub.

## Running the example

Stella_en_1.5B_v5 is used to generate text embeddings embeddings for a prompt. The model weights
are downloaded from the hub on the first run.

```bash
$ cargo run --example stella-en-v5 --release  -- --query "What are safetensors?" --which 1.5b

> [[ 0.3905, -0.0130,  0.2072, ..., -0.1100, -0.0086,  0.6002]]
>  Tensor[[1, 1024], f32]
```

Stella_en_1.5B_v5 is trained by [MRL](https://arxiv.org/abs/2205.13147) enabling multiple embedding dimensions.

The following reproduces the example in the [model card](https://huggingface.co/dunzhang/stella_en_1.5B_v5) for a retrieval task (s2p). The sample queries and docs are hardcoded in the example.

```bash
$ cargo run --example stella-en-v5 --release --features <metal | cuda> -- --which 1.5b

>
> Score: 0.8178786
> Query: What are some ways to reduce stress?
> Answer: There are many effective ways to reduce stress. Some common techniques include deep breathing, meditation, and physical activity. Engaging in hobbies, spending
> time in nature, and connecting with loved ones can also help alleviate stress. Additionally, setting boundaries, practicing self-care, and learning to say no can prevent
> stress from building up.
>
>
> Score: 0.7853528
> Query: What are the benefits of drinking green tea?
> Answer: Green tea has been consumed for centuries and is known for its potential health benefits. It contains antioxidants that may help protect the body against damage 
> caused by free radicals. Regular consumption of green tea has been associated with improved heart health, enhanced cognitive function, and a reduced risk of certain types >
> of cancer. The polyphenols in green tea may also have anti-inflammatory and weight loss properties.
>

$ cargo run --example stella-en-v5 --release --features <metal | cuda> -- --which 400m

>
> Score: 0.8397539
> Query: What are some ways to reduce stress?
> Answer: There are many effective ways to reduce stress. Some common techniques include deep breathing, meditation, and physical activity. Engaging in hobbies, spending
> time in nature, and connecting with loved ones can also help alleviate stress. Additionally, setting boundaries, practicing self-care, and learning to say no can prevent
> stress from building up.
>
>
>
> Score: 0.809545
> Query: What are the benefits of drinking green tea?
> Answer: Green tea has been consumed for centuries and is known for its potential health benefits. It contains antioxidants that may help protect the body against damage
> caused by free radicals. Regular consumption of green tea has been associated with improved heart health, enhanced cognitive function, and a reduced risk of certain types
> of cancer. The polyphenols in green tea may also have anti-inflammatory and weight loss properties.
>
```

## Supported options:
- `Stella_en_v5` has 2 model variants published - a 1.5B variant and 400M variant. This is enabled through the flag `--which`. E.g. `--which 400m` or `--which 1.5b`.

- `Stella_en_v5` supports 256, 768, 1024, 2048, 4096, 6144 and 8192 embedding dimensions (though the model card mentions 512, I couldn't find weights for the same). In the example run this is supported with `--embed-dim` option. E.g. `... --embed-dim 4096`. Defaults to `1024`.

- As per the [model card](https://huggingface.co/dunzhang/stella_en_1.5B_v5), the model has been primarily trained on `s2s` (similarity) and `s2p` (retrieval) tasks. These require a slightly different `query` preprocessing (a different prompt template for each). In this example this is enabled though `--task` option.