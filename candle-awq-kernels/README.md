# candle-awq-kernels

Fused dequantize+GEMM CUDA/Metal kernel for AWQ-quantized linear layers, used from
[`candle_transformers::quantized_awq`](../candle-transformers/src/quantized_awq.rs) when the
`awq-cuda`/`awq-metal` feature is enabled on `candle-transformers`.

See [`candle-gptq-kernels`'s README](../candle-gptq-kernels/README.md) for why this lives in its
own crate rather than inside [`candle-kernels`](../candle-kernels).
