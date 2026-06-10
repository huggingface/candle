// candle-flash-attn host dispatch: thin extern "C" wrapper around
// Tri Dao's flash_fwd kernel templates.
//
// PR-FA-2 update: extend the dispatcher to mirror v2.8.3's
// `run_mha_fwd(params, stream, force_split_kernel)` shape — branch on
// `num_splits <= 1 && !force_split_kernel` to choose between the dense
// `run_mha_fwd_<>` and `run_mha_fwd_splitkv_dispatch<>` kernel
// templates. The FFI `extern "C" run_mha` exposes the new params
// (`num_splits`, `softmax_lseaccum_ptr`, `oaccum_ptr`,
// `force_split_kernel`) so Rust callers can drive splitkv. PR-FA-2
// keeps Rust-side defaults at `num_splits=1` and null accumulator
// pointers, so the dense path is taken and existing behavior is
// unchanged. PR-FA-3 wires the Rust-side `set_params_splitkv`
// equivalent (heuristic + accumulator buffer allocation).
//
// PR-FA-1 (already merged): vendored kernels were bumped from the
// post-Dec-2024 state to upstream v2.8.3 (commit 060c918, 2025-08-14).
// v2.8.3 wraps the kernel templates in `namespace flash`, so
// `run_mha_fwd_<>` and `Flash_fwd_params` live under `FLASH_NAMESPACE`
// (= `flash`).

#include <cstdio>
#include <cstdlib>

#include "kernels.h"
#include "kernel_helpers.h"
#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

// Suppress implicit instantiation of `run_mha_fwd_splitkv_dispatch<>` in this
// TU. Without these declarations cicc would expand all 24 tuples
// (2 dtypes × 6 hdims × 2 causal) of the splitkv dispatcher inline here, and
// each tuple instantiates ~142 kernel specialisations through the seven nested
// SWITCH macros in `run_flash_splitkv_fwd<>` — a single-TU compile that ran
// >30 minutes at ~18 GB RSS before being killed during PR-FA-2 development.
//
// The corresponding explicit instantiation definitions live in the per-hdim
// `flash_fwd_split_hdim*_*_sm80.cu` files (e.g.
// `template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 64, false>(...)`),
// which compile in parallel as 24 independent TUs (~30s each on this machine).
// The linker resolves the calls in this TU to those out-of-line definitions.
//
// Note: `run_mha_fwd_<>` (the dense-path counterpart) is forward-declared in
// `flash.h` without a primary template definition; cicc therefore never tries
// to implicitly instantiate it here, and no extern declarations are required.
extern template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 32, false>(Flash_fwd_params &params, cudaStream_t stream);
extern template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 32, true>(Flash_fwd_params &params, cudaStream_t stream);
extern template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 64, false>(Flash_fwd_params &params, cudaStream_t stream);
extern template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 64, true>(Flash_fwd_params &params, cudaStream_t stream);
extern template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 96, false>(Flash_fwd_params &params, cudaStream_t stream);
extern template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 96, true>(Flash_fwd_params &params, cudaStream_t stream);
extern template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 128, false>(Flash_fwd_params &params, cudaStream_t stream);
extern template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 128, true>(Flash_fwd_params &params, cudaStream_t stream);
extern template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 192, false>(Flash_fwd_params &params, cudaStream_t stream);
extern template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 192, true>(Flash_fwd_params &params, cudaStream_t stream);
extern template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 256, false>(Flash_fwd_params &params, cudaStream_t stream);
extern template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 256, true>(Flash_fwd_params &params, cudaStream_t stream);
extern template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 32, false>(Flash_fwd_params &params, cudaStream_t stream);
extern template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 32, true>(Flash_fwd_params &params, cudaStream_t stream);
extern template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 64, false>(Flash_fwd_params &params, cudaStream_t stream);
extern template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 64, true>(Flash_fwd_params &params, cudaStream_t stream);
extern template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 96, false>(Flash_fwd_params &params, cudaStream_t stream);
extern template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 96, true>(Flash_fwd_params &params, cudaStream_t stream);
extern template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 128, false>(Flash_fwd_params &params, cudaStream_t stream);
extern template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 128, true>(Flash_fwd_params &params, cudaStream_t stream);
extern template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 192, false>(Flash_fwd_params &params, cudaStream_t stream);
extern template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 192, true>(Flash_fwd_params &params, cudaStream_t stream);
extern template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 256, false>(Flash_fwd_params &params, cudaStream_t stream);
extern template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 256, true>(Flash_fwd_params &params, cudaStream_t stream);

// Templated dispatch wrapper. Mirrors v2.8.3's
// `flash_api.cpp::run_mha_fwd` — chooses between the dense
// `run_mha_fwd_<elem_type, kHeadDim, Is_causal>` specialisation and
// the `run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>`
// specialisation based on `params.num_splits` and the explicit
// `force_split_kernel` override (used upstream for paged-KV / cached-K
// paths; not yet plumbed in candle but exposed for FFI symmetry).
void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream,
                 bool force_split_kernel = false) {
  FP16_SWITCH(!params.is_bf16, [&] {
      HEADDIM_SWITCH(params.d, [&] {
          BOOL_SWITCH(params.is_causal, Is_causal, [&] {
              if (params.num_splits <= 1 && !force_split_kernel) {
                  run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
              } else {
                  run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
              }
          });
      });
  });
}

} // namespace FLASH_NAMESPACE

extern "C" void run_mha(
    void *q_ptr,
    void *k_ptr,
    void *v_ptr,
    void *o_ptr,
    void *softmax_lse_ptr,
    void *alibi_slopes_ptr,

    int32_t *cu_seqlens_q_ptr,
    int32_t *cu_seqlens_k_ptr,

    uint32_t q_batch_stride,
    uint32_t k_batch_stride,
    uint32_t v_batch_stride,
    uint32_t o_batch_stride,
    uint32_t alibi_slopes_batch_stride,

    uint32_t q_row_stride,
    uint32_t k_row_stride,
    uint32_t v_row_stride,
    uint32_t o_row_stride,

    uint32_t q_head_stride,
    uint32_t k_head_stride,
    uint32_t v_head_stride,
    uint32_t o_head_stride,

    uint32_t b,
    uint32_t h,
    uint32_t h_k,
    uint32_t d,
    uint32_t d_rounded,
    float softmax_scale,

    uint32_t seqlen_q,
    uint32_t seqlen_k,
    uint32_t seqlen_q_rounded,
    uint32_t seqlen_k_rounded,

    int is_bf16,
    int is_causal,
    int unpadded_lse,

    int window_size_left,
    int window_size_right,

    float softcap,

    // PR-FA-2: split-KV dispatch surface. `num_splits<=1 && !force_split_kernel`
    // takes the dense path (existing behavior); `num_splits>1` or
    // `force_split_kernel != 0` enters `run_mha_fwd_splitkv_dispatch<>`. The
    // `_accum_ptr` buffers must be fp32 of shape
    // `(num_splits, b, h, seqlen_q[, d_rounded])`; the caller (Rust side) is
    // responsible for allocation. Defaults of `num_splits=1`,
    // `softmax_lseaccum_ptr=nullptr`, `oaccum_ptr=nullptr`, `force_split_kernel=0`
    // reproduce PR-FA-1 behavior exactly.
    int num_splits,
    void *softmax_lseaccum_ptr,
    void *oaccum_ptr,
    int force_split_kernel
) {
    FLASH_NAMESPACE::Flash_fwd_params params;
    // Reset the parameters
    memset(&params, 0, sizeof(params));

    // Set the pointers and strides.
    params.q_ptr = q_ptr;
    params.k_ptr = k_ptr;
    params.v_ptr = v_ptr;
    params.o_ptr = o_ptr;

    params.softmax_lse_ptr = softmax_lse_ptr;
    params.alibi_slopes_ptr = alibi_slopes_ptr;

    // All stride are in elements, not bytes.
    params.q_batch_stride = q_batch_stride;
    params.k_batch_stride = k_batch_stride;
    params.v_batch_stride = v_batch_stride;
    params.o_batch_stride = o_batch_stride;
    params.alibi_slopes_batch_stride = alibi_slopes_batch_stride;

    params.q_row_stride = q_row_stride;
    params.k_row_stride = k_row_stride;
    params.v_row_stride = v_row_stride;
    params.o_row_stride = o_row_stride;
    params.q_head_stride = q_head_stride;
    params.k_head_stride = k_head_stride;
    params.v_head_stride = v_head_stride;
    params.o_head_stride = o_head_stride;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.
    if (softcap > 0.0) {
        params.softcap = softmax_scale / softcap;
        params.scale_softmax = softcap;
        params.scale_softmax_log2 = softcap * M_LOG2E;
    } else{
        // Remove potential NaN
        params.softcap = 0.0;
        params.scale_softmax = softmax_scale;
        params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    }

    params.p_dropout = 1.; // probability to keep
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    params.is_bf16 = is_bf16;
    params.cu_seqlens_q = cu_seqlens_q_ptr;
    params.cu_seqlens_k = cu_seqlens_k_ptr;
    params.p_ptr = nullptr; // used for `return_softmax`.
    params.seqused_k = nullptr;

    params.is_causal = is_causal;
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    params.is_seqlens_k_cumulative = true;
    params.num_splits = num_splits;
    params.softmax_lseaccum_ptr = softmax_lseaccum_ptr;
    params.oaccum_ptr = oaccum_ptr;
    params.unpadded_lse = unpadded_lse;

    // Tripwire: candle-flash-attn does not support dropout. `philox_unpack.cuh`
    // is a stubbed replacement (returns a fake seed/offset pair) so the dropout
    // codepath inside `flash_fwd_kernel.h` compiles, but executing it would
    // silently produce garbage. Dropout is currently impossible to reach because
    // `params.p_dropout` is hard-set to 1.0 above and is not an FFI input —
    // this check catches anyone re-introducing a dropout path without also
    // wiring a real philox state. Unconditional (not `assert`) so the guard
    // remains active in release builds.
    if (params.p_dropout != 1.f) {
        std::fprintf(stderr,
                     "candle-flash-attn: dropout is not supported "
                     "(philox_unpack.cuh is stubbed); got p_dropout=%f\n",
                     params.p_dropout);
        std::abort();
    }

    cudaStream_t stream = 0; // Use the default stream.
    FLASH_NAMESPACE::run_mha_fwd(params, stream, force_split_kernel != 0);
}
