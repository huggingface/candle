# Candle Flash Attention v4 Layer

Flash Attention v4 Layer for Blackwell (compatible nvidia `sm100a` arch) and the candle framework.

This implements the FlashAttention-4 algorithm from the paper
"FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling"
by Zadouri et al.

Key features over FlashAttention-3:
- **Conditional softmax rescaling**: Skips unnecessary rescaling operations when
  the running max change is below a threshold τ, reducing non-matmul operations.
- **Deterministic backward pass**: Optional deterministic execution mode for
  reproducible training (important for RL applications), using semaphore-locked
  global reductions.
- **2-CTA MMA mode**: Reduces shared memory traffic and halves the number of
  global atomic reductions in the backward pass on Blackwell GPUs.
- **Software-emulated exponential**: Uses polynomial approximation on FMA units
  to increase exponential throughput, alleviating a key bottleneck.
- **LPT scheduling**: Longest-processing-time-first scheduling for improved load
  balancing with causal masking and variable sequence lengths.

## Build Requirements

- NVIDIA Blackwell GPU (B200, GB200, etc.) with compute capability >= 100
- CUDA Toolkit with Blackwell support
- CUTLASS (automatically fetched by the build system)

## Usage

```rust
use candle_flash_attn_v4::flash_attn;

let output = flash_attn(&q, &k, &v, softmax_scale, causal, 8.0)?;
```

For conditional rescaling with custom threshold:
```rust
use candle_flash_attn_v4::flash_attn_with_rescale_threshold;

let output = flash_attn_with_rescale_threshold(
    &q, &k, &v, softmax_scale, causal, 8.0
)?;
```

For deterministic backward pass (training with RL):
```rust
use candle_flash_attn_v4::flash_attn_deterministic;

let output = flash_attn_deterministic(
    &q, &k, &v, softmax_scale, causal, 8.0
)?;
```
