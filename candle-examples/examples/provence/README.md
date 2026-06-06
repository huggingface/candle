## provence

This is a port of the [Provence](https://huggingface.co/naver/provence-reranker-debertav3-v1) model. Provence is based on DebertaV3.

> Provence is a lightweight context pruning model for retrieval-augmented generation, particularly optimized for question answering. Given a user question and a retrieved passage, Provence removes sentences from the passage that are not relevant to the user question. This speeds up generation and reduces context noise, in a plug-and-play manner for any LLM.

## Examples

Note that all examples here use the `metal` feature flag provided by the `candle-examples` crate. You may need to adjust this to match your environment.

Also, the `candle-transformers-provence-process` feature flag is required to enable the model's `process` helper function. This will enable additional dependencies, like `tokenizer`. If you only need to run `forward` and not `process`, then no additional dependencies are needed and the `candle-transformers-provence-process` feature flag is not needed.

### Single Questing and Context

```bash
cargo run  --example provence --release --features=metal,candle-transformers-provence-process -- -q "What is used to thicken a classic béchamel sauce?" -c "Béchamel sauce. Basics. Béchamel is one of the five mother sauces of French cuisine. It is a simple white sauce made from a roux of butter and flour, to which milk is gradually added while whisking to avoid lumps. The roux acts as the thickening agent, giving the sauce a smooth, creamy texture. Variations. Some chefs add a pinch of nutmeg for flavor. In Italian cuisine, a similar sauce called besciamella is often used in lasagna. Modern adaptations may substitute butter with olive oil or milk with plant-based alternatives, but the thickening principle with flour remains the same." -t="0.35"
```

### Running on CPU

To run the example on CPU, supply the `--cpu` flag.
