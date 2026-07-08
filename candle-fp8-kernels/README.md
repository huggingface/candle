# candle-fp8-kernels

Fused dequantize+GEMM CUDA/Metal kernel for block-wise FP8 (E4M3, DeepSeek-V3-style) quantized
linear layers, used from
[`candle_transformers::quantized_fp8`](../candle-transformers/src/quantized_fp8.rs) when the
`fp8-cuda`/`fp8-metal` feature is enabled on `candle-transformers`.

See [`candle-gptq-kernels`'s README](../candle-gptq-kernels/README.md) for why this lives in its
own crate rather than inside [`candle-kernels`](../candle-kernels).
