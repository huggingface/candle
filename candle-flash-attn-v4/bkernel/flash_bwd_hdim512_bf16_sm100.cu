// FlashAttention-4 backward kernel instantiation for Blackwell (sm_100a)
// Kernel implementations using CuTe-DSL compiled kernels with:
// - 2-CTA MMA mode for reduced shared memory traffic
// - Tensor memory for storing more intermediate results
// - DSMEM exchange for dS between CTA pairs
// - Deterministic execution mode with semaphore locks
// - LPT (longest-processing-time-first) scheduling
//
// This file is compiled as part of the candle-flash-attn-v4 build.
// The actual kernel implementations are provided by the FlashAttention-4
// CuTe-DSL framework from https://github.com/Dao-AILab/flash-attention
