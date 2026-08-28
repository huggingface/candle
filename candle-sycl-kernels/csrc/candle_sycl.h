// C ABI for the candle SYCL backend. Deliberately flat, opaque-handle, and
// error-code based (§6e of the feasibility report: a small auditable FFI layer
// in place of a dependency on an unvetted SYCL binding crate). Every function
// that can fail returns 0 on success and a non-zero CandleSyclStatus otherwise.
#ifndef CANDLE_SYCL_H
#define CANDLE_SYCL_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  CANDLE_SYCL_OK = 0,
  CANDLE_SYCL_ERR_NO_DEVICE = 1,
  CANDLE_SYCL_ERR_ALLOC = 2,
  CANDLE_SYCL_ERR_LAUNCH = 3,
  CANDLE_SYCL_ERR_UNSUPPORTED_DTYPE = 4,
  CANDLE_SYCL_ERR_UNSUPPORTED_OP = 5,
  CANDLE_SYCL_ERR_INVALID = 6,
  CANDLE_SYCL_ERR_EXCEPTION = 7,
} CandleSyclStatus;

// Keep in sync with `DType` mapping in src/lib.rs.
typedef enum {
  CANDLE_SYCL_U8 = 0,
  CANDLE_SYCL_U32 = 1,
  CANDLE_SYCL_I64 = 2,
  CANDLE_SYCL_F16 = 3,
  CANDLE_SYCL_BF16 = 4,
  CANDLE_SYCL_F32 = 5,
  CANDLE_SYCL_F64 = 6,
} CandleSyclDType;

typedef struct CandleSyclQueue CandleSyclQueue;

#define CANDLE_SYCL_MAX_DIMS 8
// num_dims == 0 is the sentinel for "dense, contiguous, offset 0".
typedef struct {
  uint32_t num_dims;
  int64_t offset; // in elements
  int64_t dims[CANDLE_SYCL_MAX_DIMS];
  int64_t strides[CANDLE_SYCL_MAX_DIMS];
} CandleSyclLayout;

// ---- runtime ----------------------------------------------------------------
int candle_sycl_device_count(void);
// Returns NULL on failure.
CandleSyclQueue *candle_sycl_queue_new(int ordinal);
void candle_sycl_queue_free(CandleSyclQueue *q);
int candle_sycl_synchronize(CandleSyclQueue *q);
// The underlying `sycl::queue *` as an opaque pointer, for out-of-tree kernels
// (e.g. crane's fused GDN launcher) that link their own `.so` and need to
// submit onto candle's in-order queue. Cast back to `sycl::queue *`.
void *candle_sycl_queue_native(CandleSyclQueue *q);

typedef struct {
  char name[256];
  uint64_t global_mem_bytes;
  uint32_t max_compute_units;
  uint32_t max_clock_khz;
  int32_t is_integrated;
  int32_t supports_fp16;
  int32_t supports_fp64;
} CandleSyclDeviceInfo;
int candle_sycl_device_info(CandleSyclQueue *q, CandleSyclDeviceInfo *out);

// ---- memory ----------------------------------------------------------------
// USM device allocation. Returns NULL on failure.
void *candle_sycl_malloc(CandleSyclQueue *q, size_t bytes);
void candle_sycl_free(CandleSyclQueue *q, void *ptr);
int candle_sycl_memcpy_htod(CandleSyclQueue *q, void *dst, const void *src, size_t bytes);
int candle_sycl_memcpy_dtoh(CandleSyclQueue *q, void *dst, const void *src, size_t bytes);
int candle_sycl_memcpy_dtod(CandleSyclQueue *q, void *dst, const void *src, size_t bytes);
// Byte-wise memset (used for zeroing).
int candle_sycl_memset(CandleSyclQueue *q, void *dst, int value, size_t bytes);

// ---- elementwise ----------------------------------------------------------
// op codes: keep in sync with src/lib.rs UnaryOp / BinaryOp.
int candle_sycl_fill(CandleSyclQueue *q, CandleSyclDType dt, void *dst, size_t numel, double value);
int candle_sycl_fill_strided(CandleSyclQueue *q, CandleSyclDType dt, const CandleSyclLayout *lin,
                             void *dst, size_t numel, double value);

int candle_sycl_affine(CandleSyclQueue *q, CandleSyclDType dt, const CandleSyclLayout *lin,
                       const void *inp, void *out, size_t numel, double mul, double add);
int candle_sycl_elu(CandleSyclQueue *q, CandleSyclDType dt, const CandleSyclLayout *lin,
                    const void *inp, void *out, size_t numel, double alpha);
int candle_sycl_powf(CandleSyclQueue *q, CandleSyclDType dt, const CandleSyclLayout *lin,
                     const void *inp, void *out, size_t numel, double exponent);

int candle_sycl_unary(CandleSyclQueue *q, uint32_t op, CandleSyclDType dt,
                      const CandleSyclLayout *lin, const void *inp, void *out, size_t numel);

int candle_sycl_binary(CandleSyclQueue *q, uint32_t op, CandleSyclDType dt,
                       const CandleSyclLayout *lhs_l, const void *lhs,
                       const CandleSyclLayout *rhs_l, const void *rhs, void *out, size_t numel);

int candle_sycl_cast(CandleSyclQueue *q, CandleSyclDType src_dt, CandleSyclDType dst_dt,
                     const CandleSyclLayout *lin, const void *inp, void *out, size_t numel);

// contiguous copy of a strided source into a dense destination at dst_offset.
int candle_sycl_copy_strided(CandleSyclQueue *q, CandleSyclDType dt, const CandleSyclLayout *lin,
                             const void *inp, void *out, size_t dst_offset, size_t numel);

// cudaMemcpy2D-style: a d1 x d2 block, inner stride 1, outer strides given.
int candle_sycl_copy2d(CandleSyclQueue *q, CandleSyclDType dt, const void *inp, void *out,
                       size_t d1, size_t d2, size_t src_stride1, size_t dst_stride1,
                       size_t src_offset, size_t dst_offset);

// ---- reduce / compare / select ----------------------------------------
// op: 0 sum, 1 min, 2 max, 3 argmin, 4 argmax. `lin` has the reduced dims moved
// last; output is `out_el` elements (T for sum/min/max, u32 for arg*).
int candle_sycl_reduce(CandleSyclQueue *q, uint32_t op, CandleSyclDType dt,
                       const CandleSyclLayout *lin, const void *inp, void *out,
                       size_t out_el, size_t reduce_el);

// op: 0 eq,1 ne,2 lt,3 le,4 gt,5 ge. Output is u8.
int candle_sycl_cmp(CandleSyclQueue *q, uint32_t op, CandleSyclDType dt,
                    const CandleSyclLayout *lhs_l, const void *lhs,
                    const CandleSyclLayout *rhs_l, const void *rhs, void *out, size_t numel);

int candle_sycl_where(CandleSyclQueue *q, CandleSyclDType cond_dt, CandleSyclDType val_dt,
                      const CandleSyclLayout *cond_l, const void *cond,
                      const CandleSyclLayout *t_l, const void *t_vals,
                      const CandleSyclLayout *f_l, const void *f_vals, void *out, size_t numel);

// ---- indexing --------------------------------------------------------
int candle_sycl_index_select(CandleSyclQueue *q, CandleSyclDType dt, CandleSyclDType idt,
                             const CandleSyclLayout *lin, const void *inp, const void *ids,
                             void *out, size_t left_size, size_t src_dim_size,
                             size_t ids_dim_size, size_t right_size);
int candle_sycl_gather(CandleSyclQueue *q, CandleSyclDType dt, CandleSyclDType idt,
                       const CandleSyclLayout *lin, const void *inp, const void *ids,
                       void *out, size_t left_size, size_t src_dim_size, size_t ids_dim_size,
                       size_t right_size);
int candle_sycl_scatter(CandleSyclQueue *q, int add, CandleSyclDType dt, CandleSyclDType idt,
                        void *out, const void *ids, const void *src, size_t left_size,
                        size_t src_dim_size, size_t dst_dim_size, size_t right_size);
int candle_sycl_index_add(CandleSyclQueue *q, CandleSyclDType dt, CandleSyclDType idt, void *out,
                          const void *ids, const void *src, size_t left_size,
                          size_t ids_dim_size, size_t dst_dim_size, size_t right_size);
int candle_sycl_argsort(CandleSyclQueue *q, CandleSyclDType dt, int ascending, const void *inp,
                        uint32_t *out, size_t nrows, size_t ncols);

// ---- pooling / upsampling (NCHW) -------------------------------------
// `src` = [offset, stride_b, stride_c, stride_h, stride_w, b, c, h, w].
int candle_sycl_avg_pool2d(CandleSyclQueue *q, CandleSyclDType dt, const void *inp, void *out,
                           const int64_t *src, size_t k_h, size_t k_w, size_t s_h, size_t s_w,
                           size_t h_out, size_t w_out);
int candle_sycl_max_pool2d(CandleSyclQueue *q, CandleSyclDType dt, const void *inp, void *out,
                           const int64_t *src, size_t k_h, size_t k_w, size_t s_h, size_t s_w,
                           size_t h_out, size_t w_out);
int candle_sycl_upsample_nearest2d(CandleSyclQueue *q, CandleSyclDType dt, const void *inp,
                                   void *out, const int64_t *src, size_t dst_h, size_t dst_w);
// `src` = [offset, stride_b, stride_c, stride_w, b, c, w_in].
int candle_sycl_upsample_nearest1d(CandleSyclQueue *q, CandleSyclDType dt, const void *inp,
                                   void *out, const int64_t *src, size_t dst_w);
int candle_sycl_upsample_bilinear2d(CandleSyclQueue *q, CandleSyclDType dt, const void *inp,
                                    void *out, const int64_t *src, size_t dst_h, size_t dst_w,
                                    int align_corners, double scale_h, double scale_w);

// ---- im2col (conv path) --------------------------------------------
int candle_sycl_im2col2d(CandleSyclQueue *q, CandleSyclDType dt, const void *inp, void *col,
                         const int64_t *meta, size_t k_h, size_t k_w, size_t stride,
                         size_t padding, size_t dilation, size_t out_h, size_t out_w);
int candle_sycl_im2col1d(CandleSyclQueue *q, CandleSyclDType dt, const void *inp, void *col,
                         const int64_t *meta, size_t k, size_t stride, size_t padding,
                         size_t dilation, size_t out_l);

// ---- transposed convolution --------------------------------------
int candle_sycl_conv_transpose2d(CandleSyclQueue *q, CandleSyclDType dt, const void *inp,
                                 const int64_t *im, const void *ker, void *out, size_t b,
                                 size_t c_in, size_t c_out, size_t ih, size_t iw, size_t kh,
                                 size_t kw, size_t out_h, size_t out_w, size_t stride,
                                 size_t padding, size_t dilation);
int candle_sycl_conv_transpose1d(CandleSyclQueue *q, CandleSyclDType dt, const void *inp,
                                 const int64_t *im, const void *ker, void *out, size_t b,
                                 size_t c_in, size_t c_out, size_t il, size_t kl, size_t out_l,
                                 size_t stride, size_t padding, size_t dilation);

// ---- GGUF dequantization ----------------------------------------
// ggml_dtype: candle-source-order id (F32=0,F16=1,BF16=2,Q4_0=3,...,Q8K=14).
int candle_sycl_dequantize(CandleSyclQueue *q, uint32_t ggml_dtype, const void *src,
                           void *dst_f32, size_t n_blocks);
// Fused quantized mat-vec: weight (n rows x k/blk blocks) @ act^T (f32, m x k),
// m <= 8. out is f32 (m x n).
int candle_sycl_mmvq(CandleSyclQueue *q, uint32_t dt, const void *w, const float *act,
                     float *out, size_t n, size_t k, size_t m);

// ---- candle-nn fused ops -----------------------------------------
int candle_sycl_softmax_lastdim(CandleSyclQueue *q, CandleSyclDType dt, const void *inp,
                                void *out, size_t rows, size_t d);
int candle_sycl_rms_norm(CandleSyclQueue *q, CandleSyclDType dt, const void *inp,
                         const void *alpha, void *out, size_t rows, size_t d, float eps);
// mode: 0 interleaved (bhtd), 1 half-split (bhtd), 2 half-split (bthd).
int candle_sycl_rope(CandleSyclQueue *q, uint32_t mode, CandleSyclDType dt, const void *inp,
                     const void *cosb, const void *sinb, void *out, size_t b, size_t h,
                     size_t t, size_t d, int cos_batched);

// ---- gemm (oneMKL) -------------------------------------------------------
// Row-major batched C = alpha * op(A) @ op(B) + beta * C.
int candle_sycl_gemm(CandleSyclQueue *q, CandleSyclDType dt, int transa, int transb,
                     int64_t m, int64_t n, int64_t k, double alpha, double beta,
                     const void *a, const void *b, void *c, int64_t batch,
                     int64_t stride_a, int64_t stride_b, int64_t stride_c,
                     int64_t off_a, int64_t off_b);

#ifdef __cplusplus
}
#endif
#endif
