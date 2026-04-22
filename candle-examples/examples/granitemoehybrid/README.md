# candle-granite 4.0 Micro (GraniteMoeHybrid)

This example runs IBM's [Granite 4.0 Micro](https://huggingface.co/ibm-granite/granite-4.0-micro) hybrid Mixture-of-Experts model with Candle's `GraniteMoeHybrid` implementation. It mirrors the Granite example workflow while showcasing the embedding/logit scaling and hybrid attention stack specific to the 4.0 release.

## Running the example

```bash
cargo run --example granitemoehybrid --features metal -r -- \
  --prompt "Summarize the architectural differences between Granite 3.x and Granite 4.0 Micro."
```

Key flags:
- `--model-id` selects a Hugging Face repo or a local directory containing `config.json`, `tokenizer.json`, and the `model.safetensors` shards (defaults to `ibm-granite/granite-4.0-micro`).
- `--cpu` forces CPU execution; omit to use CUDA/Metal when available. Combine with `--dtype bf16|f16|f32` to override the default precision.
- `--no_kv_cache` disables reuse of attention key/value tensors. Leave it off for faster decoding.
- `--use_flash_attn` turns on Flash Attention kernels when Candle is built with the feature.
- Sampling controls such as `--temperature`, `--top-p`, `--top-k`, `--repeat-penalty`, and `--repeat-last-n` match the Granite example.

The inline prompt builder wraps your text in the chat template expected by Granite 4.0 Micro (`<|start_of_role|>user ...`). Generation stops when the EOS token (`100257`) is produced or after `sample_len` tokens.

## Tips

- Download the model locally with `huggingface-cli download ibm-granite/granite-4.0-micro` and pass the directory via `--model-id ./granite-4.0-micro` to avoid repeated hub calls.
- Enable `--tracing` to emit a Chrome trace (`trace-timestamp.json`) when profiling hybrid block performance.
- If you experiment with longer outputs, raise `--sample_len` and consider `--repeat-penalty` tuning to reduce repetition.
