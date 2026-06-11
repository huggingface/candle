/*
 * Stub replacements for moe_wmma.cu and moe_wmma_gguf.cu on Pascal (sm_60)
 * and other pre-Volta GPUs without tensor cores.
 *
 * Why this file exists
 * --------------------
 *
 * `moe_wmma.cu` and `moe_wmma_gguf.cu` use `nvcuda::wmma` tensor-core
 * fragments (Volta+, sm_70+). Compiling them for sm_60 fails with
 * "unsupported function" / undeclared identifier errors because the
 * `wmma` namespace and matrix fragment intrinsics simply don't exist
 * for that target.
 *
 * Dense models (Llama, Qwen, Gemma, Phi, etc.) do not exercise the MoE
 * GEMM path at all, so we provide stubs that abort with a clear error
 * message in the unlikely event a Pascal user actually invokes a MoE
 * model. This keeps the static link satisfied without forcing every
 * Pascal user to manually patch their build.
 *
 * Mixtral and other MoE models cannot run on Pascal regardless of this
 * stub — the abort just makes the failure mode obvious instead of a
 * cryptic linker error at build time.
 *
 * The build system (`candle-kernels/build.rs`) substitutes this file
 * for `moe_wmma.cu` + `moe_wmma_gguf.cu` when `CUDA_COMPUTE_CAP < 70`.
 */

#include <cstdio>
#include <cstdlib>

extern "C" void moe_gemm_wmma(
    const void * /*input*/,
    const void * /*weights*/,
    const int *  /*sorted_token_ids*/,
    const int *  /*expert_ids*/,
    const float *  /*topk_weights*/,
    void * /*output*/,
    int *  /*expert_counts*/,
    int *  /*expert_offsets*/,
    int /*num_experts*/,
    int /*topk*/,
    int /*size_m*/,
    int /*size_n*/,
    int /*size_k*/,
    int /*dtype*/,
    bool /*is_prefill*/,
    long long /*stream*/) {
    fprintf(stderr,
        "candle-kernels: moe_gemm_wmma called on a GPU without tensor cores "
        "(sm < 70). MoE models like Mixtral are not supported on Pascal. "
        "Aborting.\n");
    std::abort();
}

extern "C" void moe_gemm_gguf_prefill(
    const void * /*input*/,
    const unsigned char * /*weights*/,
    const int *  /*sorted_token_ids*/,
    const int *  /*expert_ids*/,
    const float *  /*topk_weights*/,
    void * /*output*/,
    int /*num_experts*/,
    int /*topk*/,
    int /*size_m*/,
    int /*size_n*/,
    int /*size_k*/,
    int /*input_dtype*/,
    int /*gguf_dtype*/,
    long long /*stream*/) {
    fprintf(stderr,
        "candle-kernels: moe_gemm_gguf_prefill called on a GPU without "
        "tensor cores (sm < 70). MoE prefill is not supported on Pascal. "
        "Aborting.\n");
    std::abort();
}
