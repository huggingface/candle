## GLM4
GLM-4-9B-0414 is a new architecture in the GLM-4 series developed by Zhipu AI. This model is not compatible with previous versions of GLM-4, such as THUDM/glm-4-9b, due to differences in model architecture and internal implementation. Users must explicitly specify the correct model type when loading it, as using the wrong configuration may lead to initialization errors or runtime failures.

### GLM4-0414 Arch:

- [GLM4-0414 Collection](https://huggingface.co/collections/THUDM/glm-4-0414-67f3cbcb34dd9d252707cb2e)
- [GLM-4-9B-0414 Weight](https://huggingface.co/THUDM/GLM-4-9B-0414)

### Old GLM4 Arch:

- [GitHub](https://github.com/THUDM/GLM4)
- [GLM-4-9B Weight](https://huggingface.co/THUDM/glm-4-9b)

### Running with CUDA 
Use `--which` to distinguish two archs

```bash
cargo run --example glm4 --release --features cuda -- --which "glm4-new" --model-id THUDM/GLM-4-9B-0414 --prompt "How are you today?"
cargo run --example glm4 --release --features cuda -- --which "glm4-old" --model-id THUDM/glm-4-9b --prompt "How are you today?"
```

### Running with local file (CUDA)

```bash
cargo run --example glm4 --release --features cuda -- --which "glm4-new" --weight-path /path/GLM-4-9B-0414 --prompt "How are you today?"
cargo run --example glm4 --release --features cuda -- --which "glm4-old" --weight-path /path/glm-4-9b --prompt "How are you today?"
```

### Running with local file (Metal)

```bash
cargo run --example glm4 --release --features metal -- --which "glm4-new" --weight-path /path/GLM-4-9B-0414 --prompt "How are you today?"
cargo run --example glm4 --release --features metal -- --which "glm4-old" --weight-path /path/glm-4-9b --prompt "How are you today?"
```

### Running with CPU
```bash
cargo run --example glm4 --release -- --cpu --which "glm4-new" --model-id THUDM/GLM-4-9B-0414 --prompt "How are you today?"
```

### Output Example (GLM-4-9B-0414)
```
avx: true, neon: false, simd128: false, f16c: true
temp: 0.80 repeat-penalty: 1.20 repeat-last-n: 64
retrieved the files in 158.728989ms
loaded the model in 3.714556129s
starting the inference loop
How are you today?
I'm just a computer program, so I don't have feelings or emotions. But thank you for asking! How can I assist you today?

31 tokens generated (28.77 token/s)
```