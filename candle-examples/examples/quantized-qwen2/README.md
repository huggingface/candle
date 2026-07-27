# candle-quantized-qwen2: GPTQ/AWQ-quantized Qwen2 from the Hugging Face Hub

End-to-end example that downloads a GPTQ- or AWQ-quantized Qwen2 checkpoint from the Hugging Face
Hub and runs text generation through it, using
[`candle_transformers::models::quant_linear_qwen2`] and the fused/CPU kernels in
`candle-gptq-kernels` / `candle-awq-kernels` via
[`candle_transformers::quantized_linear::QuantizedLinear`].

The checkpoint's `config.json` carries a `quantization_config` block whose `quant_method` field
(`"gptq"` or `"awq"`) this example reads to pick the right format automatically — no separate
example or flag is needed per format.

The model architecture mirrors the dense [`qwen2`](../qwen2) model, but every attention and MLP
projection (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) is
dequantized from the checkpoint's packed GPTQ/AWQ tensors at load time, instead of being read as
plain dense weights. Whether a given projection carries a `bias` tensor is detected per-checkpoint
(GPTQ exports tend to add `bias` to every quantized projection; AWQ exports tend to follow the
dense model's own bias layout, i.e. only on `q`/`k`/`v_proj`), rather than assumed.

## Running the example

```bash
$ cargo run --example quantized-qwen2 --release -- \
    --prompt "Give me three tips for staying focused while studying."
```

By default this downloads
[`Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4`](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4)
(GPTQ). To run against an AWQ checkpoint instead, pass `--model-id`:

```bash
$ cargo run --example quantized-qwen2 --release -- \
    --model-id "Qwen/Qwen2-0.5B-Instruct-AWQ" \
    --prompt "Give me three tips for staying focused while studying."
```

Any other GPTQ- or AWQ-quantized Qwen2 checkpoint on the Hub works the same way, as long as its
`config.json` has a `quantization_config.quant_method` of `"gptq"` or `"awq"`:

```bash
$ cargo run --example quantized-qwen2 --release -- \
    --model-id "<org>/<some-other-qwen2-gptq-or-awq-checkpoint>" \
    --prompt "Hello there "
```

### FP8 is not demonstrated here

`QuantizedLinear` also supports a third format, block-wise FP8 (DeepSeek-V3-style, with a
per-128x128-block `weight_scale_inv` tensor). The small public Qwen2 FP8 checkpoints on the Hub
(e.g. `RedHatAI/Qwen2-0.5B-Instruct-FP8`) don't use that layout — they use per-tensor static scalar
scales (the vLLM / compressed-tensors layout), which `candle_transformers::quantized_fp8` does not
implement. There is currently no small public checkpoint to demonstrate the block-wise FP8 path
end-to-end against; see `candle_transformers::quantized_fp8`'s own unit tests for coverage instead.

## Implementation notes

This example always uses the portable CPU dequantize-at-load path (`gptq_linear`/`awq_linear`),
which runs on CPU, CUDA, and Metal without requiring any of the `{gptq,awq}-{cuda,metal}` feature
flags. For the fused dequantize+GEMM kernels (which keep the checkpoint packed and dequantize on
every forward pass instead of once at load time), see
`candle_transformers::quantized_gptq::cuda::GptqLinearCuda` /
`candle_transformers::quantized_awq::cuda::AwqLinearCuda` and their `metal` counterparts, gated
behind the `gptq-cuda`/`gptq-metal`/`awq-cuda`/`awq-metal` features on `candle-transformers`.
