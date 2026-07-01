# candle-quantized-gptq-qwen2: GPTQ-quantized Qwen2 from the Hugging Face Hub

End-to-end example that downloads a GPTQ-quantized Qwen2 checkpoint (in the AutoGPTQ /
GPTQModel safetensors layout: packed `qweight`/`qzeros`/`scales`/`g_idx` tensors per linear
layer) from the Hugging Face Hub and runs text generation through it, using
[`candle_transformers::models::gptq_qwen2`] and the GPTQ kernels in `candle-gptq-kernels` /
[`candle_transformers::quantized_gptq`].

The model architecture mirrors the dense [`qwen2`](../qwen2) model, but every attention and MLP
projection (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) is
dequantized from the checkpoint's packed GPTQ tensors via `gptq_linear` at load time, instead of
being read as plain dense weights.

## Running the example

```bash
$ cargo run --example quantized-gptq-qwen2 --release -- \
    --prompt "Give me three tips for staying focused while studying."
```

By default this downloads
[`Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4`](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4).
A different GPTQ-quantized Qwen2 checkpoint can be selected with `--model-id`:

```bash
$ cargo run --example quantized-gptq-qwen2 --release -- \
    --model-id "<org>/<some-other-qwen2-gptq-checkpoint>" \
    --prompt "Hello there "
```

This example always uses the portable CPU dequantize-at-load path (`gptq_linear`), which runs on
CPU, CUDA, and Metal without requiring the `gptq-cuda`/`gptq-metal` feature flags. For the fused
dequantize+GEMM kernels (which keep the checkpoint packed and dequantize on every forward pass
instead of once at load time), see `candle_transformers::quantized_gptq::cuda::GptqLinearCuda` /
`metal::GptqLinearMetal`, gated behind the `gptq-cuda`/`gptq-metal` features on
`candle-transformers`.
