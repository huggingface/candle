# candle-helium: 2b LLM with CC-BY licensed weights

Helium-1 is a lightweight model with around 2B parameters, the preview version
currently supports 6 languages, showing strong capabilities in those languages
compared to existing open weights models.

- [Blog Post](https://kyutai.org/2025/01/13/helium.html) announcing the model
  release.
- [Model card](https://huggingface.co/kyutai/helium-1-preview-2b) on the HuggingFace Hub.

## Running the example

```bash
$ cargo run --example helium --release --features cuda -- --prompt 'Write helloworld code in Rust' --sample-len 150
```


