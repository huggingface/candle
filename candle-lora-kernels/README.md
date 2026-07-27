# candle-lora-kernels

Heterogeneous multi-LoRA batching CUDA kernels (S-LoRA / Punica style BGMV) for
the candle ML framework.

This crate is the fused-kernel fast path for
`candle_nn::lora::LoraLinear::forward_with_adapters`. A decode-step batch in
which every row selects its own LoRA adapter is applied as two gather-GEMV
passes — "shrink" (`in -> r`) then "expand" (`r -> out`) — that read the stacked
adapter matrices in place through a per-row slot index, instead of materializing
the per-row gather that the pure-tensor reference path builds before its batched
matmuls.

It is CUDA-only (requires `nvcc`) and is enabled through candle-nn's `lora-cuda`
feature; without that feature candle-nn uses the equivalent pure-tensor path on
every backend.
