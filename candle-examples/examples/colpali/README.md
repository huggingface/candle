# Colpali

[HuggingFace Model Card](https://huggingface.co/akshayballal/colpali-merged)

```bash
cargo run --features cuda,pdf2image --release --example colpali -- --prompt "What is Positional Encoding" --pdf "candle-examples/examples/colpali/assets/attention.pdf"
```

```
Prompt: what is position encoding?
top 3 page numbers that contain similarity to the prompt
-----------------------------------
Page: 6
Page: 11
Page: 15
-----------------------------------
```