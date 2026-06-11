// FlashAttention-4 forward kernel instantiation for Blackwell (sm_100a)
// Kernel implementations using CuTe-DSL compiled kernels with:
// - Redesigned pipeline for maximum overlap of MMA and softmax
// - Software-emulated exponential via polynomial approximation
// - Conditional softmax rescaling (skip when m_j - m_{j-1} <= tau)
// - Tensor memory (TMEM) for intermediate results
// - 2-CTA MMA mode for reduced shared memory traffic
//
// This file is compiled as part of the candle-flash-attn-v4 build.
// The actual kernel implementations are provided by the FlashAttention-4
// CuTe-DSL framework from https://github.com/Dao-AILab/flash-attention
