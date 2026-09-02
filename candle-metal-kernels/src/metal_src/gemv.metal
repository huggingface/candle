// GEMV kernel adapted from MLX:
// https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/gemv.metal
// Copyright © 2023-2024 Apple Inc.

#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

#define MLX_MTL_CONST static constant constexpr const
#define MLX_MTL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

// elem_to_loc for nc=1 batch handling
template <typename stride_t>
METAL_FUNC stride_t gemv_elem_to_loc(
    uint elem,
    constant const int* shape,
    constant const stride_t* strides,
    int ndim) {
  stride_t loc = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    loc += (elem % shape[i]) * strides[i];
    elem /= shape[i];
  }
  return loc;
}

template <typename U>
struct GemvDefaultAccT {
  using type = float;
};

///////////////////////////////////////////////////////////////////////////////
/// Matrix-vector: mat [M_out, K] × vec [K] → out [M_out]
/// (mat rows = output dimension, mat cols = K)
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    const int BM,       // Threadgroup rows (in simdgroups)
    const int BN,       // Threadgroup cols (in simdgroups)
    const int SM,       // Simdgroup rows (in threads)
    const int SN,       // Simdgroup cols (in threads)
    const int TM,       // Thread rows (in elements)
    const int TN,       // Thread cols (in elements)
    const bool kDoAxpby,
    typename AccT = typename GemvDefaultAccT<T>::type>
struct GEMVKernel {
  using acc_type = AccT;

  MLX_MTL_CONST int threadsM = BM * SM;
  MLX_MTL_CONST int threadsN = BN * SN;

  MLX_MTL_CONST int blockM = threadsM * TM;
  MLX_MTL_CONST int blockN = threadsN * TN;

  static_assert(SM * SN == 32, "simdgroup must have 32 threads");
  static_assert(SN == 4 || SN == 8 || SN == 16 || SN == 32,
                "gemv block must have width 4, 8, 16, or 32");

  MLX_MTL_CONST short tgp_mem_size = BN > 1 ? BN * (blockM + TM) : 0;
  MLX_MTL_CONST bool needs_tgp_reduction = BN > 1;

  template <typename U = T>
  static METAL_FUNC void load_unsafe(
      const device T* src, thread U dst[TN], const int src_offset = 0) {
    MLX_MTL_PRAGMA_UNROLL
    for (int tn = 0; tn < TN; tn++) {
      dst[tn] = static_cast<U>(src[src_offset + tn]);
    }
  }

  template <typename U = T>
  static METAL_FUNC void load_safe(
      const device T* src,
      thread U dst[TN],
      const int src_offset = 0,
      const int src_size = TN) {
    if (src_offset + TN <= src_size) {
      MLX_MTL_PRAGMA_UNROLL
      for (int tn = 0; tn < TN; tn++) {
        dst[tn] = static_cast<U>(src[src_offset + tn]);
      }
    } else {
      MLX_MTL_PRAGMA_UNROLL
      for (int tn = 0; tn < TN; tn++) {
        dst[tn] = src_offset + tn < src_size
            ? static_cast<U>(src[src_offset + tn])
            : U(0);
      }
    }
  }

  static METAL_FUNC void run(
      const device T* mat,
      const device T* in_vec,
      const device T* bias,
      device T* out_vec,
      const int in_vec_size,
      const int out_vec_size,
      const int matrix_ld,
      const float alpha,
      const float beta,
      const int bias_stride,
      threadgroup AccT* tgp_memory,
      uint3 tid,
      uint3 lid,
      uint simd_gid,
      uint simd_lid) {
    (void)lid;

    thread AccT result[TM] = {0};
    thread T inter[TN];
    thread AccT v_coeff[TN];

    const int thrM = SN != 32 ? (int)(simd_lid / SN) : 0;
    const int thrN = SN != 32 ? (int)(simd_lid % SN) : (int)simd_lid;

    const int sgN = BN != 1 ? (int)(simd_gid % BN) : 0;
    const int simdM = BN != 1 ? SM * (int)(simd_gid / BN) : (int)(SM * simd_gid);
    const int simdN = BN != 1 ? SN * (int)(simd_gid % BN) : 0;

    int bm = (simdM + thrM) * TM;
    int bn = (simdN + thrN) * TN;

    // Block position (output row)
    int out_row = tid.x * blockM + bm;

    if (out_row >= out_vec_size) return;

    out_row = out_row + TM <= out_vec_size ? out_row : out_vec_size - TM;

    mat += out_row * matrix_ld;

    const int loop_stride = blockN;
    const int in_size = in_vec_size;
    const int n_iter = in_size / loop_stride;
    const int last_iter = loop_stride * n_iter;
    const int leftover = in_size - last_iter;

    for (int i = 0; i < n_iter; ++i) {
      load_unsafe<AccT>(in_vec, v_coeff, bn);

      int mat_offset = 0;
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        load_unsafe(mat, inter, mat_offset + bn);
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          result[tm] += inter[tn] * v_coeff[tn];
        }
        mat_offset += matrix_ld;
      }
      bn += blockN;
    }

    if (leftover > 0) {
      load_safe<AccT>(in_vec, v_coeff, bn, in_size);
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        load_safe(&mat[tm * matrix_ld], inter, bn, in_size);
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          result[tm] += inter[tn] * v_coeff[tn];
        }
      }
    }

    MLX_MTL_PRAGMA_UNROLL
    for (int tm = 0; tm < TM; tm++) {
      MLX_MTL_PRAGMA_UNROLL
      for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
        result[tm] += simd_shuffle_down(result[tm], sn);
      }
    }

    if (needs_tgp_reduction) {
      threadgroup AccT* tgp_results = tgp_memory + sgN * (blockM + TM) + bm;
      if (thrN == 0) {
        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
          tgp_results[tm] = result[tm];
        }
        threadgroup_barrier(mem_flags::mem_none);
        if (sgN == 0) {
          MLX_MTL_PRAGMA_UNROLL
          for (int sgn = 1; sgn < BN; sgn++) {
            MLX_MTL_PRAGMA_UNROLL
            for (int tm = 0; tm < TM; tm++) {
              result[tm] += tgp_results[sgn * (blockM + TM) + tm];
            }
          }
        }
      }
    }

    if (simdN == 0 && thrN == 0) {
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        if (kDoAxpby) {
          out_vec[out_row + tm] =
              static_cast<T>(alpha) * static_cast<T>(result[tm]) +
              static_cast<T>(beta) * bias[(out_row + tm) * bias_stride];
        } else {
          out_vec[out_row + tm] = static_cast<T>(result[tm]);
        }
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
/// Vector-matrix: vec [K] × mat [K, N_out] → out [N_out]
/// (mat rows = K, mat cols = output dimension)
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    const int BM,
    const int BN,
    const int SM,
    const int SN,
    const int TM,
    const int TN,
    const bool kDoAxpby,
    typename AccT = typename GemvDefaultAccT<T>::type>
struct GEMVTKernel {
  using acc_type = AccT;

  MLX_MTL_CONST int threadsM = BM * SM;
  MLX_MTL_CONST int threadsN = BN * SN;

  MLX_MTL_CONST int blockM = threadsM * TM;
  MLX_MTL_CONST int blockN = threadsN * TN;

  static_assert(SM * SN == 32, "simdgroup must have 32 threads");

  MLX_MTL_CONST short tgp_mem_size = BM > 1 ? BM * (blockN + TN) : 0;
  MLX_MTL_CONST bool needs_tgp_reduction = BM > 1;

  static METAL_FUNC void run(
      const device T* mat,
      const device T* in_vec,
      const device T* bias,
      device T* out_vec,
      const int in_vec_size,
      const int out_vec_size,
      const int marix_ld,
      const float alpha,
      const float beta,
      const int bias_stride,
      threadgroup AccT* tgp_memory,
      uint3 tid,
      uint3 lid,
      uint simd_gid,
      uint simd_lid) {
    (void)lid;

    AccT result[TN] = {0};
    T inter[TN];
    AccT v_coeff[TM];

    const int thrM = SN != 32 ? (int)(simd_lid / SN) : 0;
    const int thrN = SN != 32 ? (int)(simd_lid % SN) : (int)simd_lid;

    const int sgM = BN != 1 ? (int)(simd_gid / BN) : (int)simd_gid;
    const int sgN = BN != 1 ? (int)(simd_gid % BN) : 0;

    const int simdM = SM * sgM;
    const int simdN = SN * sgN;

    int cm = (simdM + thrM);
    int cn = (simdN + thrN);

    int bm = cm * TM;
    int bn = cn * TN;

    int out_col = tid.x * blockN + bn;

    const int loop_stride = blockM;
    const int in_size = in_vec_size;
    const int n_iter = in_size / loop_stride;
    const int last_iter = loop_stride * n_iter;
    const int leftover = in_size - last_iter;

    if (out_col < out_vec_size) {
      out_col = out_col + TN <= out_vec_size ? out_col : out_vec_size - TN;

      for (int i = 0; i < n_iter; ++i) {
        threadgroup_barrier(mem_flags::mem_none);

        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
          v_coeff[tm] = static_cast<AccT>(in_vec[bm + tm]);
        }

        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
          auto vc = static_cast<AccT>(v_coeff[tm]);
          MLX_MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < TN; tn++) {
            inter[tn] = mat[(bm + tm) * marix_ld + out_col + tn];
          }
          MLX_MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < TN; tn++) {
            result[tn] += vc * inter[tn];
          }
        }

        bm += blockM;
      }

      if (leftover > 0) {
        for (int tm = 0; tm < TM && bm + tm < in_vec_size; tm++) {
          v_coeff[tm] = static_cast<AccT>(in_vec[bm + tm]);
          MLX_MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < TN; tn++) {
            inter[tn] = mat[(bm + tm) * marix_ld + out_col + tn];
          }
          MLX_MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < TN; tn++) {
            result[tn] += v_coeff[tm] * inter[tn];
          }
        }
      }
    }

    MLX_MTL_PRAGMA_UNROLL
    for (int tn = 0; tn < TN; tn++) {
      MLX_MTL_PRAGMA_UNROLL
      for (ushort sm = (SM / 2); sm >= 1; sm >>= 1) {
        result[tn] += simd_shuffle_down(result[tn], SN * sm);
      }
    }

    if (needs_tgp_reduction) {
      threadgroup AccT* tgp_results = tgp_memory + sgM * (blockN + TN) + bn;
      if (thrM == 0) {
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          tgp_results[tn] = result[tn];
        }
        threadgroup_barrier(mem_flags::mem_none);
        if (sgM == 0) {
          MLX_MTL_PRAGMA_UNROLL
          for (int sgm = 1; sgm < BM; sgm++) {
            MLX_MTL_PRAGMA_UNROLL
            for (int tn = 0; tn < TN; tn++) {
              result[tn] += tgp_results[sgm * (blockN + TN) + tn];
            }
          }
        }
      }
    }

    if (cm == 0 && out_col < out_vec_size) {
      MLX_MTL_PRAGMA_UNROLL
      for (int j = 0; j < TN; j++) {
        if (kDoAxpby) {
          out_vec[out_col + j] =
              static_cast<T>(alpha) * static_cast<T>(result[j]) +
              static_cast<T>(beta) * bias[(out_col + j) * bias_stride];
        } else {
          out_vec[out_col + j] = static_cast<T>(result[j]);
        }
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
/// gemv kernel: mat [M, K] × vec [K] → out [M]
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    const int BM, const int BN,
    const int SM, const int SN,
    const int TM, const int TN,
    const bool kDoNCBatch,
    const bool kDoAxpby>
[[kernel, max_total_threads_per_threadgroup(BM * BN * 32)]]
void gemv(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    const device T* bias [[buffer(2)]],
    device T* out_vec [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    const constant int& matrix_ld [[buffer(6)]],
    const constant float& alpha [[buffer(7)]],
    const constant float& beta [[buffer(8)]],
    const constant int& batch_ndim [[buffer(9)]],
    const constant int* batch_shape [[buffer(10)]],
    const constant int64_t* vector_batch_stride [[buffer(11)]],
    const constant int64_t* matrix_batch_stride [[buffer(12)]],
    const constant int64_t* bias_batch_stride [[buffer(13)]],
    const constant int& bias_stride [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using kernel_t = GEMVKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>;
  threadgroup typename kernel_t::acc_type tgp_memory[kernel_t::tgp_mem_size == 0 ? 1 : kernel_t::tgp_mem_size];

  if (kDoNCBatch) {
    in_vec += gemv_elem_to_loc(tid.z, batch_shape, vector_batch_stride, batch_ndim);
    mat    += gemv_elem_to_loc(tid.z, batch_shape, matrix_batch_stride, batch_ndim);
    if (kDoAxpby)
      bias += gemv_elem_to_loc(tid.z, batch_shape, bias_batch_stride, batch_ndim);
  } else {
    in_vec += tid.z * vector_batch_stride[0];
    mat    += tid.z * matrix_batch_stride[0];
    if (kDoAxpby)
      bias += tid.z * bias_batch_stride[0];
  }
  out_vec += tid.z * out_vec_size;

  kernel_t::run(mat, in_vec, bias, out_vec,
                in_vec_size, out_vec_size, matrix_ld,
                alpha, beta, bias_stride,
                kernel_t::tgp_mem_size == 0 ? nullptr : tgp_memory,
                tid, lid, simd_gid, simd_lid);
}

///////////////////////////////////////////////////////////////////////////////
/// gemv_t kernel: vec [K] × mat [K, N] → out [N]
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    const int BM, const int BN,
    const int SM, const int SN,
    const int TM, const int TN,
    const bool kDoNCBatch,
    const bool kDoAxpby>
[[kernel, max_total_threads_per_threadgroup(BM * BN * 32)]]
void gemv_t(
    const device T* mat [[buffer(0)]],
    const device T* in_vec [[buffer(1)]],
    const device T* bias [[buffer(2)]],
    device T* out_vec [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    const constant int& matrix_ld [[buffer(6)]],
    const constant float& alpha [[buffer(7)]],
    const constant float& beta [[buffer(8)]],
    const constant int& batch_ndim [[buffer(9)]],
    const constant int* batch_shape [[buffer(10)]],
    const constant int64_t* vector_batch_stride [[buffer(11)]],
    const constant int64_t* matrix_batch_stride [[buffer(12)]],
    const constant int64_t* bias_batch_stride [[buffer(13)]],
    const constant int& bias_stride [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using kernel_t = GEMVTKernel<T, BM, BN, SM, SN, TM, TN, kDoAxpby>;
  threadgroup typename kernel_t::acc_type tgp_memory[kernel_t::tgp_mem_size == 0 ? 1 : kernel_t::tgp_mem_size];

  if (kDoNCBatch) {
    in_vec += gemv_elem_to_loc(tid.z, batch_shape, vector_batch_stride, batch_ndim);
    mat    += gemv_elem_to_loc(tid.z, batch_shape, matrix_batch_stride, batch_ndim);
    if (kDoAxpby)
      bias += gemv_elem_to_loc(tid.z, batch_shape, bias_batch_stride, batch_ndim);
  } else {
    in_vec += tid.z * vector_batch_stride[0];
    mat    += tid.z * matrix_batch_stride[0];
    if (kDoAxpby)
      bias += tid.z * bias_batch_stride[0];
  }
  out_vec += tid.z * out_vec_size;

  kernel_t::run(mat, in_vec, bias, out_vec,
                in_vec_size, out_vec_size, matrix_ld,
                alpha, beta, bias_stride,
                kernel_t::tgp_mem_size == 0 ? nullptr : tgp_memory,
                tid, lid, simd_gid, simd_lid);
}

///////////////////////////////////////////////////////////////////////////////
/// Instantiations
///////////////////////////////////////////////////////////////////////////////

// Use decltype-based instantiation (MLX defines.h pattern):
//   template [[host_name(...)]] [[kernel]] decltype(func<...>) func<...>;
// This avoids redeclaring parameter attributes, which Metal rejects.
#define instantiate_gemv_helper(func, nm, itype, bm, bn, sm, sn, tm, tn, nc, axpby) \
  template [[host_name(                                                               \
      #func "_" #nm "_bm" #bm "_bn" #bn "_sm" #sm "_sn" #sn                         \
            "_tm" #tm "_tn" #tn "_nc" #nc "_axpby" #axpby)]]                         \
  [[kernel]] decltype(func<itype, bm, bn, sm, sn, tm, tn, (bool)nc, (bool)axpby>)   \
             func<itype, bm, bn, sm, sn, tm, tn, (bool)nc, (bool)axpby>;

#define instantiate_gemv_nc_axpby(func, nm, itype, bm, bn, sm, sn, tm, tn) \
  instantiate_gemv_helper(func, nm, itype, bm, bn, sm, sn, tm, tn, 0, 0)   \
  instantiate_gemv_helper(func, nm, itype, bm, bn, sm, sn, tm, tn, 0, 1)   \
  instantiate_gemv_helper(func, nm, itype, bm, bn, sm, sn, tm, tn, 1, 0)   \
  instantiate_gemv_helper(func, nm, itype, bm, bn, sm, sn, tm, tn, 1, 1)

// gemv blocks: mat×vec (output size = M)
// bm=4/8 for large output; bm=1,bn=8 for K-heavy; bm=1,bn=1,sm=8,sn=4 for small K
#define instantiate_gemv_blocks(nm, itype)                              \
  instantiate_gemv_nc_axpby(gemv, nm, itype, 1,  8, 1, 32, 4, 4)      \
  instantiate_gemv_nc_axpby(gemv, nm, itype, 1,  8, 1, 32, 1, 4)      \
  instantiate_gemv_nc_axpby(gemv, nm, itype, 1,  1, 8,  4, 4, 4)      \
  instantiate_gemv_nc_axpby(gemv, nm, itype, 1,  1, 8,  4, 1, 4)      \
  instantiate_gemv_nc_axpby(gemv, nm, itype, 4,  1, 1, 32, 1, 4)      \
  instantiate_gemv_nc_axpby(gemv, nm, itype, 4,  1, 1, 32, 4, 4)      \
  instantiate_gemv_nc_axpby(gemv, nm, itype, 8,  1, 1, 32, 4, 4)

// gemv_t blocks: vec×mat (output size = N)
// bn=2/4/16 for various output sizes; sm/sn tuned for K size
#define instantiate_gemv_t_blocks(nm, itype)                              \
  instantiate_gemv_nc_axpby(gemv_t, nm, itype, 1,  2,  8, 4, 4, 1)      \
  instantiate_gemv_nc_axpby(gemv_t, nm, itype, 1,  2,  8, 4, 4, 4)      \
  instantiate_gemv_nc_axpby(gemv_t, nm, itype, 1,  4,  8, 4, 4, 4)      \
  instantiate_gemv_nc_axpby(gemv_t, nm, itype, 1, 16,  8, 4, 4, 4)      \
  instantiate_gemv_nc_axpby(gemv_t, nm, itype, 1, 16,  4, 8, 4, 4)

instantiate_gemv_blocks(float32, float)
instantiate_gemv_blocks(float16, half)
instantiate_gemv_blocks(bfloat16, bfloat)

instantiate_gemv_t_blocks(float32, float)
instantiate_gemv_t_blocks(float16, half)
instantiate_gemv_t_blocks(bfloat16, bfloat)
