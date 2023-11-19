//
//  GEMM.metal
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/23/23.
//

#include <metal_stdlib>
#include "metal_data_type"
#include "metal_simdgroup_event"
#include "metal_simdgroup_matrix_storage"
using namespace metal;

// MARK: - Function Constants

// Dimensions of each matrix.
constant uint M [[function_constant(0)]];
constant uint N [[function_constant(1)]];
constant uint K [[function_constant(2)]];

// Whether each matrix is transposed.
constant bool A_trans [[function_constant(10)]];
constant bool B_trans [[function_constant(11)]];
constant bool D_trans [[function_constant(13)]];
constant uint A_leading_dim = A_trans ? M : K;
constant uint B_leading_dim = B_trans ? K : N;

// Alpha and beta constants from BLAS.
constant float alpha [[function_constant(20)]];
constant float beta [[function_constant(21)]];

constant bool batched [[function_constant(100)]];
constant bool fused_activation [[function_constant(101)]];
constant bool fused_bias [[function_constant(50001)]]; // 102
constant bool use_bias = is_function_constant_defined(fused_bias) ? fused_bias : false;
constant bool use_activation_function = fused_activation && !fused_bias;
constant bool use_activation = use_bias || use_activation_function;
constant bool batched_activation_function = batched && use_activation_function;

constant ushort M_simd [[function_constant(200)]];
constant ushort N_simd [[function_constant(201)]];
constant ushort K_simd [[function_constant(202)]];

// Elide work on the edge when matrix dimension < SRAM block dimension.
constant ushort M_modulo = (M % M_simd == 0) ? M_simd : (M % M_simd);
constant ushort N_modulo = (N % N_simd == 0) ? N_simd : (N % N_simd);
constant ushort M_padded = (M < M_simd) ? (M_modulo + 7) / 8 * 8 : M_simd;
constant ushort N_padded = (N < N_simd) ? (N_modulo + 7) / 8 * 8 : N_simd;

constant ushort M_splits [[function_constant(210)]];
constant ushort N_splits [[function_constant(211)]];

constant ushort M_group = M_simd * M_splits;
constant ushort N_group = N_simd * N_splits;
constant ushort A_block_leading_dim = (A_trans ? M_group : K_simd);
constant ushort B_block_leading_dim = (B_trans ? K_simd : N_group);

// There is no padding for M reads/writes.
// There is no padding for N reads/writes.
constant ushort K_simd_unpadded = (K % K_simd == 0) ? K_simd : (K % K_simd);
constant ushort K_simd_padded = (K_simd_unpadded + 7) / 8 * 8;

constant ushort A_sram_length = (M_simd / 8) * 1;
constant ushort B_sram_length = 1 * (N_simd / 8);
constant ushort A_block_length = M_group * K_simd;

// Threadgroup block must fit entire C accumulator and partial sums.
constant ushort A_sram_offset = 0;
constant ushort B_sram_offset = A_sram_offset + A_sram_length;
constant ushort C_sram_offset = B_sram_offset + B_sram_length;
constant ushort A_block_offset = 0;
constant ushort B_block_offset = A_block_offset + A_block_length;

// MARK: - Utilities

template <typename T>
METAL_FUNC thread simdgroup_matrix_storage<T>* A_sram(thread simdgroup_matrix_storage<T> *sram, ushort2 matrix_origin) {
  // A_sram[M_simd][8]
  return sram + A_sram_offset + (matrix_origin.y / 8) * (8 / 8) + (matrix_origin.x / 8);
}

template <typename T>
METAL_FUNC thread simdgroup_matrix_storage<T>* B_sram(thread simdgroup_matrix_storage<T> *sram, ushort2 matrix_origin) {
  // A_sram[8][N_simd]
  return sram + B_sram_offset + (matrix_origin.y / 8) * (N_simd / 8) + (matrix_origin.x / 8);
}

template <typename T>
METAL_FUNC thread simdgroup_matrix_storage<T>* C_sram(thread simdgroup_matrix_storage<T> *sram, ushort2 matrix_origin) {
  // C_sram[M_simd][N_simd]
  return sram + C_sram_offset + (matrix_origin.y / 8) * (N_simd / 8) + (matrix_origin.x / 8);
}

template <typename T>
METAL_FUNC void prefetch(threadgroup T *A_block, device T *A,
                         ushort2 A_tile_src, uint2 A_offset,
                         threadgroup T *B_block, device T *B,
                         ushort2 B_tile_src, uint2 B_offset, uint k)
{
  A_tile_src.x = min(uint(K_simd), K - k);
  B_tile_src.y = min(uint(K_simd), K - k);
  auto A_src = simdgroup_matrix_storage<T>::apply_offset(A, A_leading_dim, A_offset, A_trans);
  auto B_src = simdgroup_matrix_storage<T>::apply_offset(B, B_leading_dim, B_offset, B_trans);
  
  // Rounded-up ceiling for the threadgroup block.
  const uint K_edge_floor = K - K_simd_unpadded;
  const uint K_edge_ceil = K_edge_floor + K_simd_padded;
  ushort K_padded;
  if (K_edge_floor == K_simd) {
    K_padded = K_simd;
  } else {
    K_padded = min(uint(K_simd), K_edge_ceil - k);
  }
  ushort2 A_tile_dst(K_padded, A_tile_src.y);
  ushort2 B_tile_dst(B_tile_src.x, K_padded);
  
  simdgroup_event events[2];
  events[0].async_copy(A_block, A_block_leading_dim, A_tile_dst, A_src, A_leading_dim, A_tile_src, A_trans);
  events[1].async_copy(B_block, B_block_leading_dim, B_tile_dst, B_src, B_leading_dim, B_tile_src, B_trans);
  simdgroup_event::wait(2, events);
}

// One iteration of the MACC loop, effectively k=8 iterations.
template <typename T>
METAL_FUNC void multiply_accumulate(thread simdgroup_matrix_storage<T> *sram,
                                    const threadgroup T *A_block,
                                    const threadgroup T *B_block,
                                    bool accumulate = true)
{
#pragma clang loop unroll(full)
  for (ushort m = 0; m < M_padded; m += 8) {
    ushort2 origin(0, m);
    A_sram(sram, origin)->load(A_block, A_block_leading_dim, origin, A_trans);
  }
#pragma clang loop unroll(full)
  for (ushort n = 0; n < N_padded; n += 8) {
    ushort2 origin(n, 0);
    B_sram(sram, origin)->load(B_block, B_block_leading_dim, origin, B_trans);
  }
#pragma clang loop unroll(full)
  for (ushort m = 0; m < M_padded; m += 8) {
    auto A = A_sram(sram, ushort2(0, m));
#pragma clang loop unroll(full)
    for (ushort n = 0; n < N_padded; n += 8) {
      auto B = B_sram(sram, ushort2(n, 0));
      auto C = C_sram(sram, ushort2(n, m));
      C->multiply(*A, *B, accumulate);
    }
  }
}

template <typename T>
METAL_FUNC void partial_store(thread simdgroup_matrix_storage<T> *sram,
                              threadgroup T *C_block, bool is_k_summation)
{
#pragma clang loop unroll(full)
  for (ushort m = 0; m < M_padded; m += 8) {
#pragma clang loop unroll(full)
    for (ushort n = 0; n < N_padded; n += 8) {
      ushort2 origin(n, m);
      if (is_k_summation) {
        C_sram(sram, origin)->store(C_block, N_simd, origin);
      } else {
        C_sram(sram, origin)->store(C_block, N_group, origin);
      }
    }
  }
}

template <typename T>
METAL_FUNC void partial_accumulate(thread simdgroup_matrix_storage<T> *sram,
                                   threadgroup T *C_block, bool is_k_summation)
{
#pragma clang loop unroll(full)
  for (ushort m = 0; m < M_padded; m += 8) {
#pragma clang loop unroll(full)
    for (ushort n = 0; n < N_padded; n += 8) {
      ushort2 origin(n, m);
      auto B = B_sram(sram, ushort2(n, 0));
      if (is_k_summation) {
        B->load(C_block, N_simd, origin);
      } else {
        B->load(C_block, N_group, origin);
      }
    }
#pragma clang loop unroll(full)
    for (ushort n = 0; n < N_padded; n += 8) {
      ushort2 origin(n, m);
      auto B = B_sram(sram, ushort2(n, 0));
      auto C = C_sram(sram, origin);
      if (is_k_summation) {
        C->thread_elements()[0] += B->thread_elements()[0];
      } else {
        float2 C_old = float2(B->thread_elements()[0]);
        float2 C_new = float2(C->thread_elements()[0]);
        C->thread_elements()[0] = vec<T, 2>(fast::fma(C_old, beta, C_new));
      }
    }
  }
}

template <typename T>
METAL_FUNC void async_access_accumulator(threadgroup T *C_block, device T *C,
                                         uint2 C_offset, bool is_store)
{
  ushort2 C_tile(min(uint(N_group), N - C_offset.x),
                 min(uint(M_group), M - C_offset.y));
  auto C_src = simdgroup_matrix_storage<T>::apply_offset(C, N, C_offset);
  
  simdgroup_event event;
  if (is_store) {
    event.async_copy(C_src, N, C_tile, C_block, N_group, C_tile);
  } else {
    event.async_copy(C_block, N_group, C_tile, C_src, N, C_tile);
    simdgroup_event::wait(1, &event);
  }
}

template <typename T>
METAL_FUNC void store_accumulator(thread simdgroup_matrix_storage<T> *sram,
                                  device T *C, bool m_is_edge, bool n_is_edge)
{
  const ushort m_start = (m_is_edge) ? M_modulo : 0;
  const ushort n_start = (n_is_edge) ? N_modulo : 0;
  const ushort m_end = (m_is_edge) ? M_simd : M_modulo;
  const ushort n_end = (n_is_edge) ? N_simd : N_modulo;
  
#pragma clang loop unroll(full)
  for (ushort m = m_start; m < m_end; m += 8) {
#pragma clang loop unroll(full)
    for (ushort n = n_start; n < n_end; n += 8) {
      ushort2 origin(n, m);
      C_sram(sram, origin)->store(C, N, origin);
    }
  }
}

template <typename T>
struct activation_functor {
  using function = void(threadgroup T *C,
                        device void *D,
                        uint grid_index_in_batch,
                        uint2 matrix_origin,
                        ushort2 tile_dimensions,
                        ushort lane_id);
  
  typedef visible_function_table<function> function_table;
};

// MARK: - Kernels

template <typename T>
void _gemm_impl(device T *A [[buffer(0)]],
                device T *B [[buffer(1)]],
                device T *C [[buffer(2)]],
                device void *D [[buffer(3), function_constant(use_activation)]],
                
                threadgroup T *threadgroup_block [[threadgroup(0)]],
                constant ulong4 *matrix_offsets [[buffer(10), function_constant(batched)]],
                typename activation_functor<T>::function_table table [[buffer(11), function_constant(use_activation_function)]],
                constant uint *activation_function_offsets [[buffer(12), function_constant(batched_activation_function)]],
                
                uint3 gid [[threadgroup_position_in_grid]],
                ushort sidx [[simdgroup_index_in_threadgroup]],
                ushort lane_id [[thread_index_in_simdgroup]])
{
  if (batched) {
    // TODO: Re-compute every inner loop iteration for FP64 accumulate.
    ulong3 offsets = matrix_offsets[gid.z].xyz;
    A = (device T*)((device uchar*)A + offsets[0]);
    B = (device T*)((device uchar*)B + offsets[1]);
    C = (device T*)((device uchar*)C + offsets[2]);
  }
  
  simdgroup_matrix_storage<T> sram[1024];
  auto A_block = threadgroup_block + A_block_offset;
  auto B_block = threadgroup_block + B_block_offset;
  ushort2 sid(sidx % N_splits, sidx / N_splits);
  ushort2 offset_in_simd = simdgroup_matrix_storage<T>::offset(lane_id);
  
  uint2 A_offset(0, gid.y * M_group);
  uint2 B_offset(gid.x * N_group, 0);
  {
    uint C_base_offset_x = B_offset.x + sid.x * N_simd;
    uint C_base_offset_y = A_offset.y + sid.y * M_simd;
    if (C_base_offset_x >= N || C_base_offset_y >= M) {
      return;
    }
  }
  
  ushort2 offset_in_group(sid.x * N_simd + offset_in_simd.x,
                          sid.y * M_simd + offset_in_simd.y);
  
  if (use_bias) {
    if (sidx == 0) {
      auto bias = (device T*)D;
      if (batched) {
        ulong offset = matrix_offsets[gid.z].w;
        bias = (device T*)((device uchar*)bias + offset);
      }
      
      ushort bias_elements;
      if (is_function_constant_defined(D_trans) && D_trans) {
        bias += A_offset.y;
        bias_elements = min(uint(M_group), M - A_offset.y);
      } else {
        bias += B_offset.x;
        bias_elements = min(uint(N_group), N - B_offset.x);
      }
      
      simdgroup_event event;
      event.async_copy(threadgroup_block, bias, bias_elements);
      simdgroup_event::wait(1, &event);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (is_function_constant_defined(D_trans) && D_trans) {
      auto bias = threadgroup_block + offset_in_group.y;
#pragma clang loop unroll(full)
      for (ushort m = 0; m < M_padded; m += 8) {
        auto D = bias[m];
#pragma clang loop unroll(full)
        for (ushort n = 0; n < N_padded; n += 8) {
          auto C = C_sram(sram, ushort2(n, m));
          *(C->thread_elements()) = D;
        }
      }
    } else {
      auto bias = threadgroup_block + offset_in_group.x;
#pragma clang loop unroll(full)
      for (ushort n = 0; n < N_padded; n += 8) {
        auto D = *(threadgroup vec<T, 2>*)(bias + n);
#pragma clang loop unroll(full)
        for (ushort m = 0; m < M_padded; m += 8) {
          auto C = C_sram(sram, ushort2(n, m));
          *(C->thread_elements()) = D;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  
  ushort2 A_tile_src;
  ushort2 B_tile_src;
  if (sidx == 0) {
    A_tile_src.y = min(uint(M_group), M - A_offset.y);
    B_tile_src.x = min(uint(N_group), N - B_offset.x);
    prefetch(A_block, A, A_tile_src, A_offset, B_block, B, B_tile_src, B_offset, 0);
  }
  
  if (K > K_simd && !use_bias) {
#pragma clang loop unroll(full)
    for (ushort m = 0; m < M_padded; m += 8) {
#pragma clang loop unroll(full)
      for (ushort n = 0; n < N_padded; n += 8) {
        *C_sram(sram, ushort2(n, m)) = simdgroup_matrix_storage<T>(0);
      }
    }
  }
  
  for (uint K_floor = 0; K_floor < K; K_floor += K_simd) {
    ushort2 A_block_offset(offset_in_simd.x, offset_in_group.y);
    ushort2 B_block_offset(offset_in_group.x, offset_in_simd.y);
    auto A_block_src = simdgroup_matrix_storage<T>::apply_offset(A_block, A_block_leading_dim, A_block_offset, A_trans);
    auto B_block_src = simdgroup_matrix_storage<T>::apply_offset(B_block, B_block_leading_dim, B_block_offset, B_trans);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
#pragma clang loop unroll(full)
    for (ushort k = 0; k < K_simd_padded; k += 8) {
      bool accumulate = use_bias || !(K <= K_simd && k == 0);
      multiply_accumulate(sram, A_block_src, B_block_src, accumulate);
      A_block_src += A_trans ? 8 * A_block_leading_dim : 8;
      B_block_src += B_trans ? 8 : 8 * B_block_leading_dim;
    }
    
    if (K_floor + K_simd < K) {
#pragma clang loop unroll(full)
      for (ushort k = K_simd_padded; k < K_simd; k += 8) {
        multiply_accumulate(sram, A_block_src, B_block_src);
        A_block_src += A_trans ? 8 * A_block_leading_dim : 8;
        B_block_src += B_trans ? 8 : 8 * B_block_leading_dim;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      
      if (sidx == 0) {
        uint K_next = K_floor + K_simd;
        A_offset.x = K_next;
        B_offset.y = K_next;
        prefetch(A_block, A, A_tile_src, A_offset, B_block, B, B_tile_src, B_offset, K_next);
      }
    }
  }
  
  if (alpha != 1) {
#pragma clang loop unroll(full)
    for (int m = 0; m < M_padded; m += 8) {
#pragma clang loop unroll(full)
      for (int n = 0; n < N_padded; n += 8) {
        C_sram(sram, ushort2(n, m))->thread_elements()[0] *= alpha;
      }
    }
  }
  
  uint2 C_offset(B_offset.x, A_offset.y);
  ushort2 C_block_offset = offset_in_group.xy;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  
  if (beta != 0) {
    if (sidx == 0) {
      async_access_accumulator(threadgroup_block, C, C_offset, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    auto C_block = simdgroup_matrix_storage<T>::apply_offset(threadgroup_block, N_group, C_block_offset);
    partial_accumulate(sram, C_block, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  
  if (use_activation_function) {
    auto C_block = simdgroup_matrix_storage<T>::apply_offset(threadgroup_block, N_group, C_block_offset);
    partial_store(sram, C_block, false);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    
    uint grid_index_in_batch = (batched ? gid.z : 0);
    uint2 matrix_origin = C_offset + uint2(C_block_offset);
    matrix_origin &= ~7;
    ushort2 tile_dimensions(min(uint(N_group), N - matrix_origin.x),
                            min(uint(M_group), M - matrix_origin.y));
    uint function_index = 0;
    if (batched_activation_function) {
      function_index = activation_function_offsets[gid.z];
    }
    table[function_index](C_block, D, grid_index_in_batch, matrix_origin, tile_dimensions, lane_id);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (sidx == 0) {
      async_access_accumulator(threadgroup_block, C, C_offset, true);
    }
    return;
  } else if ((M % 8 != 0) || (N % 8 != 0)) {
    auto C_block = simdgroup_matrix_storage<T>::apply_offset(threadgroup_block, N_group, C_block_offset);
    partial_store(sram, C_block, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (sidx == 0) {
      async_access_accumulator(threadgroup_block, C, C_offset, true);
    }
  } else {
    uint2 matrix_origin = C_offset + uint2(C_block_offset);
    auto C_src = simdgroup_matrix_storage<T>::apply_offset(C, N, matrix_origin);
    store_accumulator(sram, C_src, false, false);
    
    const uint M_edge_floor = M - M % M_simd;
    const uint N_edge_floor = N - N % N_simd;
    if (matrix_origin.y < M_edge_floor) {
      store_accumulator(sram, C_src, true, false);
    }
    if (matrix_origin.x < N_edge_floor) {
      store_accumulator(sram, C_src, false, true);
      if (matrix_origin.y < M_edge_floor) {
        store_accumulator(sram, C_src, true, true);
      }
    }
  }
}

kernel void hgemm(device half *A [[buffer(0)]],
                  device half *B [[buffer(1)]],
                  device half *C [[buffer(2)]],
                  device void *D [[buffer(3), function_constant(use_activation)]],
                  
                  threadgroup half *threadgroup_block [[threadgroup(0)]],
                  constant ulong4 *matrix_offsets [[buffer(10), function_constant(batched)]],
                  typename activation_functor<half>::function_table table [[buffer(11), function_constant(use_activation_function)]],
                  constant uint *activation_function_offsets [[buffer(12), function_constant(batched_activation_function)]],
                  
                  uint3 gid [[threadgroup_position_in_grid]],
                  ushort sidx [[simdgroup_index_in_threadgroup]],
                  ushort lane_id [[thread_index_in_simdgroup]])
{
  _gemm_impl<half>(A, B, C, D, threadgroup_block, matrix_offsets, table, activation_function_offsets, gid, sidx, lane_id);
}

kernel void sgemm(device float *A [[buffer(0)]],
                  device float *B [[buffer(1)]],
                  device float *C [[buffer(2)]],
                  device void *D [[buffer(3), function_constant(use_activation)]],
                  
                  threadgroup float *threadgroup_block [[threadgroup(0)]],
                  constant ulong4 *matrix_offsets [[buffer(10), function_constant(batched)]],
                  typename activation_functor<float>::function_table table [[buffer(11), function_constant(use_activation_function)]],
                  constant uint *activation_function_offsets [[buffer(12), function_constant(batched_activation_function)]],
                  
                  uint3 gid [[threadgroup_position_in_grid]],
                  ushort sidx [[simdgroup_index_in_threadgroup]],
                  ushort lane_id [[thread_index_in_simdgroup]])
{
  _gemm_impl<float>(A, B, C, D, threadgroup_block, matrix_offsets, table, activation_function_offsets, gid, sidx, lane_id);
}
