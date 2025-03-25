# candle-chatglm

Uses `THUDM/chatglm3-6b` to generate chinese text. Will not generate text for english (usually).
 
## Text Generation

```bash
cargo run --example chatglm --release  -- --prompt "部署门槛较低等众多优秀特 "

> 部署门槛较低等众多优秀特 点，使得其成为了一款备受欢迎的AI助手。
> 
> 作为一款人工智能助手，ChatGLM3-6B
```