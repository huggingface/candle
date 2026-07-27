/*
 * extern "C" shim around the vendored Marlin FP16xINT4 kernel so it can be driven over FFI
 * from Rust (see `src/marlin.rs`). The kernel itself lives in `marlin_cuda_kernel.cu` and is
 * vendored verbatim from https://github.com/IST-DASLab/marlin (Apache-2.0); the upstream entry
 * point `marlin_cuda(...)` has C++ linkage and takes raw `void*`/`cudaStream_t`, so we just
 * forward-declare it and wrap it in an `extern "C"` function with a stable name.
 *
 * We deliberately do NOT vendor upstream's `marlin_cuda.cpp` PyBind/torch wrapper; this shim
 * replaces it.
 */

#include <cuda_runtime.h>

// Defined in marlin_cuda_kernel.cu (C++ linkage; signature must match exactly so the mangled
// names line up at link time).
int marlin_cuda(
  const void* A,
  const void* B,
        void* C,
        void* s,
  int prob_m,
  int prob_n,
  int prob_k,
  void* workspace,
  int groupsize,
  int dev,
  cudaStream_t stream,
  int thread_k,
  int thread_n,
  int sms,
  int max_par
);

// Returns the upstream status code: 0 = ok, 1 = ERR_PROB_SHAPE, 2 = ERR_KERN_SHAPE.
//   A:         fp16 activations, row-major [prob_m, prob_k]
//   B:         repacked int4 weights in Marlin layout, int32 [prob_k/16, prob_n*2]
//   C:         fp16 output, row-major [prob_m, prob_n]
//   s:         fp16 scales in Marlin layout [prob_k/groupsize, prob_n]
//   workspace: int32 scratch, at least prob_n/128*max_par entries, zero-initialized
//   groupsize: -1 (per-output-channel) or 128
extern "C" int run_marlin_gemm(
  const void* A,
  const void* B,
        void* C,
  const void* s,
  int prob_m,
  int prob_n,
  int prob_k,
  void* workspace,
  int groupsize,
  int dev,
  int max_par
) {
  // thread_k/thread_n/sms = -1 lets Marlin pick its auto-tuned tiling; default stream (0) keeps
  // the same convention as the other kernels in this crate.
  return marlin_cuda(
    A, B, C, const_cast<void*>(s),
    prob_m, prob_n, prob_k,
    workspace,
    groupsize,
    dev,
    /*stream=*/0,
    /*thread_k=*/-1,
    /*thread_n=*/-1,
    /*sms=*/-1,
    max_par
  );
}
