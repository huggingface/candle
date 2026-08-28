// Batched GEMM via oneMKL (the path the feasibility report measured at ~3.15
// TFLOP/s f32 / ~6.25 TFLOP/s f16 on Meteor Lake Arc).
#include "common.hpp"
#include <oneapi/mkl.hpp>

using oneapi::mkl::transpose;

namespace {

template <typename T>
int gemm_impl(CandleSyclQueue *q, int transa, int transb, int64_t m, int64_t n,
              int64_t k, double alpha, double beta, const void *a, const void *b,
              void *c, int64_t batch, int64_t stride_a, int64_t stride_b,
              int64_t stride_c, int64_t off_a, int64_t off_b) {
  // Row-major C[m,n] = op(A) @ op(B). oneMKL column-major trick: compute
  // C^T[n,m] = op(B)^T @ op(A)^T by swapping the operands and dims.
  auto opa = transa ? transpose::trans : transpose::nontrans;
  auto opb = transb ? transpose::trans : transpose::nontrans;
  int64_t lda = transa ? m : k;
  int64_t ldb = transb ? k : n;
  int64_t ldc = n;
  T al = static_cast<T>(alpha);
  T be = static_cast<T>(beta);
  const T *A = static_cast<const T *>(a) + off_a;
  const T *B = static_cast<const T *>(b) + off_b;
  T *C = static_cast<T *>(c);
  try {
    if (batch <= 1) {
      oneapi::mkl::blas::row_major::gemm(q->q, opa, opb, m, n, k, al, A, lda, B,
                                         ldb, be, C, ldc);
    } else {
      oneapi::mkl::blas::row_major::gemm_batch(q->q, opa, opb, m, n, k, al, A,
                                               lda, stride_a, B, ldb, stride_b,
                                               be, C, ldc, stride_c, batch);
    }
    /* enqueue-only; sync at to_cpu / synchronize */
    return CANDLE_SYCL_OK;
  } catch (const std::exception &) {
    return CANDLE_SYCL_ERR_EXCEPTION;
  }
}

} // namespace

extern "C" int candle_sycl_gemm(CandleSyclQueue *q, CandleSyclDType dt, int transa,
                                int transb, int64_t m, int64_t n, int64_t k,
                                double alpha, double beta, const void *a,
                                const void *b, void *c, int64_t batch,
                                int64_t stride_a, int64_t stride_b, int64_t stride_c,
                                int64_t off_a, int64_t off_b) {
  switch (dt) {
  case CANDLE_SYCL_F32:
    return gemm_impl<float>(q, transa, transb, m, n, k, alpha, beta, a, b, c,
                            batch, stride_a, stride_b, stride_c, off_a, off_b);
  case CANDLE_SYCL_F64:
    return gemm_impl<double>(q, transa, transb, m, n, k, alpha, beta, a, b, c,
                             batch, stride_a, stride_b, stride_c, off_a, off_b);
  case CANDLE_SYCL_F16:
    return gemm_impl<f16>(q, transa, transb, m, n, k, alpha, beta, a, b, c, batch,
                          stride_a, stride_b, stride_c, off_a, off_b);
  case CANDLE_SYCL_BF16:
    return gemm_impl<bf16>(q, transa, transb, m, n, k, alpha, beta, a, b, c,
                           batch, stride_a, stride_b, stride_c, off_a, off_b);
  default:
    return CANDLE_SYCL_ERR_UNSUPPORTED_DTYPE;
  }
}
