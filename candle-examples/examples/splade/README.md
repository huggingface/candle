# candle-splade

 SPLADE is a neural retrieval model which learns query/document sparse expansion via the BERT MLM head and sparse regularization. Sparse representations benefit from several advantages compared to dense approaches: efficient use of inverted index, explicit lexical match, interpretability... They also seem to be better at generalizing on out-of-domain data. In this example we can do the following two tasks:

- Compute sparse embedding for a given query.
- Compute similarities between a set of sentences using sparse embeddings.

## Sparse Sentence embeddings

SPLADE is used to compute the sparse embedding for a given query. The model weights
are downloaded from the hub on the first run. This makes use of the BertForMaskedLM model. 

```bash
cargo run --example splade --release -- --prompt "Here is a test sentence"

> "the out there still house inside position outside stay standing hotel sitting dog animal sit bird cat statue cats"
> [0.10270107, 0.269471, 0.047469813, 0.0016636598, 0.05394874, 0.23105666, 0.037475716, 0.45949644, 0.009062732, 0.06790692, 0.0327835, 0.33122346, 0.16863061, 0.12688516, 0.340983, 0.044972017, 0.47724655, 0.01765311, 0.37331146]
```

```bash
cargo run --example splade --release --features

> score: 0.47 'The new movie is awesome' 'The new movie is so great'
> score: 0.43 'The cat sits outside' 'The cat plays in the garden'
> score: 0.14 'I love pasta' 'Do you like pizza?'
> score: 0.11 'A man is playing guitar' 'The cat plays in the garden'
> score: 0.05 'A man is playing guitar' 'A woman watches TV'
```
