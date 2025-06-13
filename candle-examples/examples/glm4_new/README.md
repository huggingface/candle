## GLM4
GLM-4-9B-0414 is the open-source variant of the latest generation of pre-trained models in the GLM-4 series, developed by Zhipu AI.

- [GLM4-0414 Collection](https://huggingface.co/collections/THUDM/glm-4-0414-67f3cbcb34dd9d252707cb2e)
- [GLM-4-9B-0414 Weight](https://huggingface.co/THUDM/GLM-4-9B-0414)

### Running with CUDA

``` shell
  cargo run --example glm4_new --release --features cuda -- --model-id THUDM/GLM-4-9B-0414 --prompt "How are you today?"
```

### Running with local file (CUDA)

``` shell
  cargo run --example glm4_new --release --features cuda -- --weight-path /home/data/GLM-4-9B-0414 --prompt "How are you today?"
```

### Running with local file (Metal)

``` shell
  cargo run --example glm4_new --release --features metal -- --weight-path /home/data/GLM-4-9B-0414 --prompt "How are you today?"
```

### Running with CPU
``` shell
  cargo run --example glm4_new --release -- --cpu --model-id THUDM/GLM-4-9B-0414 --prompt "How are you today?"
```

### Output Example
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