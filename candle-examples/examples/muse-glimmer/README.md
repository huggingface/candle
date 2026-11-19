# candle-muse-glimmer

[Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) is a 30B-parameter model from Meta with a hybrid sliding-window / full-attention transformer architecture.

This example loads the [GGUF quantizations](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF) and generates text with them.

## Running the example

```bash
cargo run --example muse-glimmer --release -- --prompt "Write a short story about a lighthouse keeper."
```

The `q4_k_m` quantization is used by default; `q4_k_xl` is available via the `--which` argument.

```bash
cargo run --example muse-glimmer --release -- --which q4_k_xl --prompt "Explain in great detail the compilation process of a Rust program down to an ELF binary."
```
