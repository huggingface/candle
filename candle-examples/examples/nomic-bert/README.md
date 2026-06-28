# candle-nomic-bert

[nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) is
a text embedding model based on the NomicBert architecture. It produces 768-dimensional
embeddings suitable for semantic search, retrieval, and clustering.

Key architectural differences from standard BERT:
- Rotary position embeddings (RoPE) supporting up to 8192 tokens
- SwiGLU feed-forward layers
- Fused QKV attention projections with no biases

## Sentence embeddings

Compute the embedding for a single prompt. Model weights are downloaded from the
hub on the first run.

```bash
cargo run --example nomic-bert --release -- --prompt "Here is a test sentence"

> Embedding (first 10 dims):
>   [0] -0.030893
>   [1] 0.038772
>   [2] -0.171375
>   ...
```

## Similarities

When run without `--prompt`, the example computes cosine similarities between a
set of hardcoded sentences and reports the most similar pairs.

```bash
cargo run --example nomic-bert --release

> Top cosine similarities:
>   0.9664  'The new movie is awesome' <-> 'The new movie is so great'
>   0.7377  'The cat sits outside' <-> 'The cat plays in the garden'
>   0.5764  'I love pasta' <-> 'Do you like pizza?'
>   0.5031  'A man is playing guitar' <-> 'A woman watches TV'
>   0.4781  'A man is playing guitar' <-> 'The cat plays in the garden'
```

## Task prefixes

nomic-embed-text-v1.5 was trained with task prefixes. Adding them is optional but
improves retrieval quality. Use `--prefix` to prepend a prefix to every input:

```bash
# For documents/passages to be searched over
cargo run --example nomic-bert --release -- \
  --prefix "search_document: " \
  --prompt "Dragonwell is a classic Chinese green tea."

# For search queries
cargo run --example nomic-bert --release -- \
  --prefix "search_query: " \
  --prompt "sweet floral white tea"
```

Available prefixes: `search_document: `, `search_query: `, `clustering: `,
`classification: `.