# Colpali

[HuggingFace Model Card](https://huggingface.co/vidore/colpali-v1.2-merged)

```
wget https://huggingface.co/papers/1706.03762
cargo run --features cuda,pdf2image --release --example colpali -- --prompt "What is Positional Encoding" --pdf "1706.03762.pdf"
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