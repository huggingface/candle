// Heavily inspired by the GEMM kernels by Philip Turner:
// https://github.com/philipturner/metal-flash-attention
// This implementation uses generics and specialization to generate kernels for different data types instead of code generation.
#include <metal_stdlib>
#include "event.metal"
#include "matrix_storage.metal"
using namespace metal;

// Dimensions of each matrix.
// - Limitations to matrix size:
//    - 2^32 in each dimension (M/N/K).
//    - TODO: Test whether the maximum dimension with correct execution is
//      actually 2^16. This will require a testing setup with non-square
//      matrices, as 65536^3 is uncomputable.
//    - Extending to 2^64 may require changing 'uint' to 'ulong'. There is a
//      good chance this will significantly degrade performance, and require
//      changing the data type of several variables that process addresses. The
//      client is responsible for ensuring correctness and performance with
//      matrices spanning several billion elements in one direction.
//    - The matrix dimensions must be known at compile time, via function
//      constants. Dynamic matrix shapes are beyond the scope of this reference
//      implementation. Dynamic shapes cause a non-negligible regression to
//      shader execution speed. However, they could minimize a compilation
//      latency bottleneck in some use cases.
// - Limitations to batch size:
//   - Dictated by how the client modifies the code to implement batching.
//   - Dynamic batch shapes would likely not harm performance much. For example,
//     someone could enter an array of pointers/memory offsets to different
//     matrices in the batch. Each slice of a 3D thread grid could read a
//     different pointer from memory, and use that pointer as the A/B/C matrix.
//     Another approach is to restrict the input format, so all matrices are
//     stored contiguously in memory. Then, the memory offset could be computed
//     analytically from matrix size and the Z dimension in a 3D thread grid.
//
// Another note:
// - The rows of the matrix must be contiguous in memory. Supporting strides
//   that differ from the actual matrix dimensions should not be difficult, but
//   it is out of scope for this reference kernel.
constant uint M [[function_constant(0)]];
constant uint N [[function_constant(1)]];
constant uint K [[function_constant(2)]];

// Whether each matrix is transposed.
constant bool A_trans [[function_constant(10)]];
constant bool B_trans [[function_constant(11)]];

constant bool prefer_async_copy [[function_constant(206)]];
constant bool ideal_grouping [[function_constant(207)]];

constant bool batched [[function_constant(100)]];

constant ushort A_leading_dim = A_trans ? M : K;
constant ushort B_leading_dim = B_trans ? K : N;

// The layout of threads within a SIMD matrix.
//
//  0  0  1  1  8  8  9  9
//  2  2  3  3 10 10 11 11
//  4  4  5  5 12 12 13 13
//  6  6  7  7 14 14 15 15
// 16 16 17 17 24 24 25 25
// 18 18 19 19 26 26 27 27
// 20 20 21 21 28 28 29 29
// 22 22 23 23 30 30 31 31
//
// This is Morton order, a method for coalescing data accesses. It is used
// in a variety of contexts, from ray tracing acceleration structures, to
// nodal-point Laplacians, to sorting large lattices of atoms.
//
// Source: https://patents.google.com/patent/US11256518B2
METAL_FUNC ushort2 morton_order(ushort thread_index_in_simdgroup) {
  ushort lane_id = thread_index_in_simdgroup;
  ushort quad_id = lane_id / 4;

  constexpr ushort QUADRANT_SPAN_M = 4;
  constexpr ushort THREADS_PER_QUADRANT = 8;
  ushort M_floor_of_quadrant = (quad_id / 4) * QUADRANT_SPAN_M;
  ushort M_in_quadrant = (lane_id / 2) % (THREADS_PER_QUADRANT / 2);
  ushort M_in_simd = M_floor_of_quadrant + M_in_quadrant;

  ushort N_floor_of_quadrant = (quad_id & 2) * 2; // 0 or 4
  ushort N_in_quadrant = (lane_id % 2) * 2; // 0 or 2
  ushort N_in_simd = N_floor_of_quadrant + N_in_quadrant;

  return ushort2(N_in_simd, M_in_simd);
}

// Indexes into an array of registers.
//
// Calls to this function are expected to be evaluated at compile time. The
// array indices transform into register offsets, which are embedded into the
// assembly code.
template <typename T>
METAL_FUNC thread simdgroup_matrix_storage<T>* get_sram(
  thread simdgroup_matrix_storage<T> *sram,
  ushort sram_leading_dim,
  ushort2 matrix_origin
) {
  return sram + (matrix_origin.y / 8) * (sram_leading_dim / 8) + (matrix_origin.x / 8);
}

// One multiply-accumulate loop iteration, or 8 dot products.
template<
    typename T,
    typename U = T,
    ushort M_register,
    ushort N_register
>
METAL_FUNC void multiply_accumulate(
  const device T *A_src,
  const device U *B_src,
  thread simdgroup_matrix_storage<T> *A_sram,
  thread simdgroup_matrix_storage<U> *B_sram,
  thread simdgroup_matrix_storage<U> *C_sram,
  ushort k
) {
    #pragma clang loop unroll(full)
    for (ushort m = 0; m < M_register; m += 8) {
        ushort2 origin(0, m);
        auto A = get_sram(A_sram, 8, origin);
        A->load(A_src, A_leading_dim, ushort2(k, m), A_trans);
    }
    #pragma clang loop unroll(full)
    for (ushort n = 0; n < N_register; n += 8) {
        ushort2 origin(n, 0);
        auto B = get_sram(B_sram, N_register, origin);
        B->load(B_src, B_leading_dim, ushort2(n, k), B_trans);
    }
    #pragma clang loop unroll(full)
    for (ushort m = 0; m < M_register; m += 8) {
        #pragma clang loop unroll(full)
        for (ushort n = 0; n < N_register; n += 8) {
            auto A = get_sram(A_sram, 8, ushort2(0, m));
            auto B = get_sram(B_sram, N_register, ushort2(n, 0));
            auto C = get_sram(C_sram, N_register, ushort2(n, m));
            C->multiply(*A, *B);
        }
    }
}

// One multiply-accumulate loop iteration, or 8 dot products.
template<
    typename T,
    typename U = T,
    ushort M_register,
    ushort N_register
>
METAL_FUNC void multiply_accumulate(
  const threadgroup T *A_src,
  const threadgroup U *B_src,
  thread simdgroup_matrix_storage<T> *A_sram,
  thread simdgroup_matrix_storage<U> *B_sram,
  thread simdgroup_matrix_storage<U> *C_sram,
  ushort k
) {
    #pragma clang loop unroll(full)
    for (ushort m = 0; m < M_register; m += 8) {
        ushort2 origin(0, m);
        auto A = get_sram(A_sram, 8, origin);
        A->load(A_src, A_leading_dim, ushort2(k, m), A_trans);
    }
    #pragma clang loop unroll(full)
    for (ushort n = 0; n < N_register; n += 8) {
        ushort2 origin(n, 0);
        auto B = get_sram(B_sram, N_register, origin);
        B->load(B_src, B_leading_dim, ushort2(n, k), B_trans);
    }
    #pragma clang loop unroll(full)
    for (ushort m = 0; m < M_register; m += 8) {
        #pragma clang loop unroll(full)
        for (ushort n = 0; n < N_register; n += 8) {
            auto A = get_sram(A_sram, 8, ushort2(0, m));
            auto B = get_sram(B_sram, N_register, ushort2(n, 0));
            auto C = get_sram(C_sram, N_register, ushort2(n, m));
            C->multiply(*A, *B);
        }
    }
}

// Metal function arguments.
//
// A: the left-hand side matrix
// - dimensions: M x K
//               K x M (transposed)
// - memory precision: T
// - register precision: T
//
// B: the right-hand side matrix
// - dimensions: K x N
//               N x K (transposed)
// - memory precision: U
// - register precision: U
//
// C: the output matrix, alternatively the dot product accumulator
// - dimensions: M x N
// - memory precision: V
// - register precision: V
//
// threadgroup_block: the chunk of threadgroup memory allocated at runtime
// - ideally 10 KB or less
// - precision: void/8-bit integer to make the pointer arithmetic more legible
template <
    typename T,
    typename U = T,
    typename V = U,
    ushort M_group,
    ushort N_group,
    ushort K_group,
    ushort M_splits,
    ushort N_splits,
    ushort M_register = M_group / M_splits,
    ushort N_register = N_group / N_splits
>
void gemm_impl(
    device T *A [[buffer(0)]],
    device U *B [[buffer(1)]],
    device V *C [[buffer(2)]],

    threadgroup uchar *threadgroup_block [[threadgroup(0)]],
    constant ulong4 *matrix_offsets [[buffer(10), function_constant(batched)]],

    uint3 gid [[threadgroup_position_in_grid]],
    ushort sidx [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]]
) {
    const ushort A_leading_block_dim = A_trans ? M_group : K_group;
    const ushort B_leading_block_dim = B_trans ? K_group : N_group;

    // Thresholds that mark the matrix edge.
    const uint M_edge = M - (M % M_group);
    const uint N_edge = N - (N % N_group);

    const ushort async_iter_start = prefer_async_copy ? 0 : (K - (K % K_group));

    // Find the number of elements in the final block. If the matrix
    // dimensions are perfectly divisibly by block dimensions, we don't want
    // this value to be zero. The final block is a full block.
    const uint M_remainder = (M % M_register == 0)
      ? M_register : M % M_register;
    const ushort N_remainder = (N % N_register == 0)
      ? N_register : N % N_register;
    const ushort K_remainder = (K % K_group == 0)
      ? K_group : K % K_group;
    const ushort K_remainder_padded = (K_remainder + 7) / 8 * 8;

    // Shift the final block, so it doesn't access out-of-bounds memory.
    const ushort M_shift = (M < M_group) ? 0 : M_register - M_remainder;
    const ushort N_shift = (N < N_group) ? 0 : N_register - N_remainder;

    if (batched) {
        ulong3 offsets = matrix_offsets[0].xyz * gid.z;
        A = (device T*)((device uchar*)A + offsets[0]);
        B = (device U*)((device uchar*)B + offsets[1]);
        C = (device V*)((device uchar*)C + offsets[2]);
    }

    auto A_block = (threadgroup T*)(threadgroup_block);
    auto B_block = (threadgroup U*)(threadgroup_block + (M * K));
    ushort2 sid(sidx % N_splits, sidx / N_splits);
    ushort2 morton_offset = morton_order(lane_id);

    // Return early if the SIMD is out of bounds.
    //
    // There could be some threadgroups where the matrix edge cuts straight
    // through the middle of the block. SIMDs on the right or bottom of the
    // dividing line must be stopped from causing out-of-bounds accesses. This is
    // the reason for the early exit.
    uint M_offset = gid.y * M_group;
    uint N_offset = gid.x * N_group;
    if (M_offset + sid.y * M_register >= M ||
        N_offset + sid.x * N_register >= N) {
        return;
    }
    ushort2 offset_in_group(sid.x * N_register + morton_offset.x,
                            sid.y * M_register + morton_offset.y);

    // Shift the matrix block within bounds, if possible.
    if ((M_shift != 0) && (gid.y * M_group >= M_edge)) {
        M_offset -= M_shift;
    }
    if ((N_shift != 0) && (gid.x * N_group >= N_edge)) {
        N_offset -= N_shift;
    }

    simdgroup_matrix_storage<V> C_sram[(M_register / 8) * (N_register / 8)];

    // Initialize the accumulator.
    #pragma clang loop unroll(full)
    for (ushort m = 0; m < M_register; m += 8) {
        #pragma clang loop unroll(full)
        for (ushort n = 0; n < N_register; n += 8) {
            ushort2 origin(m, n);
            auto C = get_sram(C_sram, N_register, origin);
            *C = simdgroup_matrix_storage<V>(0);
        }
    }
    // Perform the iterations where async copy is avoided.
    #pragma clang loop unroll(full)
    for (uint k = 0; k < async_iter_start; k += 8) {
        uint2 A_offset(k, M_offset);
        uint2 B_offset(N_offset, k);
        A_offset += uint2(morton_offset.x, offset_in_group.y);
        B_offset += uint2(offset_in_group.x, morton_offset.y);

        auto A_src = simdgroup_matrix_storage<T>::apply_offset(A, A_leading_dim, A_offset, A_trans);
        auto B_src = simdgroup_matrix_storage<U>::apply_offset(B, B_leading_dim, B_offset, B_trans);

        simdgroup_matrix_storage<T> A_sram[M_register / 8];
        simdgroup_matrix_storage<U> B_sram[N_register / 8];
        multiply_accumulate<T, U, M_register, N_register>(A_src, B_src, A_sram, B_sram, C_sram, 0);
    }
    if (!prefer_async_copy) {
        #pragma clang loop unroll(full)
        for (uint k = 0; k < K; k += K_group) {
            uint2 A_offset(k, M_offset);
            uint2 B_offset(N_offset, k);
            A_offset += uint2(morton_offset.x, offset_in_group.y);
            B_offset += uint2(offset_in_group.x, morton_offset.y);

            auto A_src = simdgroup_matrix_storage<T>::apply_offset(A, A_leading_dim, A_offset, A_trans);
            auto B_src = simdgroup_matrix_storage<U>::apply_offset(B, B_leading_dim, B_offset, B_trans);

            simdgroup_matrix_storage<T> A_sram[M_register / 8];
            simdgroup_matrix_storage<U> B_sram[N_register / 8];
            multiply_accumulate<T, U, M_register, N_register>(A_src, B_src, A_sram, B_sram, C_sram, 0);
        }
    } else {
        // Perform the iterations where async copy is used.
        #pragma clang loop unroll(full)
        for (uint k = async_iter_start; k < K; k += K_group) {
            // Launch an async copy from device to threadgroup memory.
            if (sidx == 0) {
                uint2 A_offset(k, M_offset);
                uint2 B_offset(N_offset, k);
                auto A_src = simdgroup_matrix_storage<T>::apply_offset(A, A_leading_dim, A_offset, A_trans);
                auto B_src = simdgroup_matrix_storage<U>::apply_offset(B, B_leading_dim, B_offset, B_trans);

                ushort M_tile_dimension = min(uint(M_group), M - M_offset);
                ushort N_tile_dimension = min(uint(N_group), N - N_offset);
                ushort K_tile_dimension = min(uint(K_group), K - k);
                ushort K_tile_padded = min(uint(K_group), (K + K_remainder_padded - K_remainder) - k);

                ushort2 A_tile_src(K_tile_dimension, M_tile_dimension);
                ushort2 B_tile_src(N_tile_dimension, K_tile_dimension);
                ushort2 A_tile_dst(K_tile_padded, M_tile_dimension);
                ushort2 B_tile_dst(N_tile_dimension, K_tile_padded);

                simdgroup_event events[2];
                events[0].async_copy(A_block, A_leading_block_dim, A_tile_dst, A_src, A_leading_dim, A_tile_src, A_trans);
                events[1].async_copy(B_block, B_leading_block_dim, B_tile_dst, B_src, B_leading_dim, B_tile_src, B_trans);
                simdgroup_event::wait(2, events);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            ushort2 A_block_offset(morton_offset.x, offset_in_group.y);
            ushort2 B_block_offset(offset_in_group.x, morton_offset.y);
            auto A_block_src = simdgroup_matrix_storage<T>::apply_offset(A_block, A_leading_block_dim, A_block_offset, A_trans);
            auto B_block_src = simdgroup_matrix_storage<U>::apply_offset(B_block, B_leading_block_dim, B_block_offset, B_trans);

            simdgroup_matrix_storage<T> A_sram[(M_register / 8) * (K_group / 8)];
            simdgroup_matrix_storage<U> B_sram[(K_group / 8) * (N_register / 8)];

            #pragma clang loop unroll(full)
            for (ushort k = 0; k < K_remainder_padded; k += 8) {
                multiply_accumulate<T, U, M_register, N_register>(A_block_src, B_block_src, A_sram, B_sram, C_sram, k);
            }

            // Will there be any iterations after this one?
            if (k + K_group < K) {
                // If so, we haven't reached the edge of either input matrix yet.
                #pragma clang loop unroll(full)
                for (ushort k = K_remainder_padded; k < K_group; k += 8) {
                    multiply_accumulate<T, U, M_register, N_register>(A_block_src, B_block_src, A_sram, B_sram, C_sram, k);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    if (!prefer_async_copy && (M >= M_group) && (N >= N_group)) {
        // Fast path for matrices that qualify.
        uint2 C_offset(N_offset + offset_in_group.x,
                        M_offset + offset_in_group.y);
        auto C_dst = simdgroup_matrix_storage<U>::apply_offset(
        C, N, C_offset);

        // Write the accumulator to device memory.
        #pragma clang loop unroll(full)
        for (ushort m = 0; m < M_register; m += 8) {
            #pragma clang loop unroll(full)
            for (ushort n = 0; n < N_register; n += 8) {
                ushort2 origin(n, m);
                auto C = get_sram(C_sram, N_register, origin);
                C->store(C_dst, N, origin);
            }
        }
    } else {
        // Slow path for when memory must be handled more carefully.
        auto C_block = (threadgroup V*)(threadgroup_block);
        auto C_block_dst = simdgroup_matrix_storage<V>::apply_offset(C_block, N_group, offset_in_group);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Write the accumulator to threadgroup memory.
        #pragma clang loop unroll(full)
        for (ushort m = 0; m < M_register; m += 8) {
            #pragma clang loop unroll(full)
            for (ushort n = 0; n < N_register; n += 8) {
                ushort2 origin(n, m);
                auto C = get_sram(C_sram, N_register, origin);
                C->store(C_block_dst, N_group, origin);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Launch the async copy from threadgroup to device memory.
        if (sidx == 0) {
            uint2 C_offset(gid.x * N_group, gid.y * M_group);
            ushort2 C_tile(min(uint(N_group), N - C_offset.x),
                           min(uint(M_group), M - C_offset.y));
            auto C_dst = simdgroup_matrix_storage<V>::apply_offset(C, N, C_offset);

            // If we shift successfully, the garbage zone moves from the bottom right
            // to the top left.
            if ((M_shift != 0) || (N_shift != 0)) {
                ushort2 C_block_shift(0, 0);
                if ((M_shift != 0) && (C_offset.y >= M_edge)) {
                    C_block_shift.y = M_shift;
                }
                if ((N_shift != 0) && (C_offset.x >= N_edge)) {
                    C_block_shift.x = N_shift;
                }
                C_block = simdgroup_matrix_storage<V>::apply_offset(C_block, N_group, C_block_shift);
            }

            simdgroup_event event;
            event.async_copy(C_dst, N, C_tile, C_block, N_group, C_tile);
        }
    }
}

kernel void hgemm(
    device half *A [[buffer(0)]],
    device half *B [[buffer(1)]],
    device half *C [[buffer(2)]],

    threadgroup uchar *threadgroup_block [[threadgroup(0)]],
    constant ulong4 *matrix_offsets [[buffer(10), function_constant(batched)]],

    uint3 gid [[threadgroup_position_in_grid]],
    ushort sidx [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]]
) {
    if (ideal_grouping) {
        gemm_impl<half, half, half, 32, 32, 32, 1, 1>(
            A, B, C, threadgroup_block, matrix_offsets, gid, sidx, lane_id
        );
    } else {
        gemm_impl<half, half, half, 48, 48, 32, 1, 1>(
            A, B, C, threadgroup_block, matrix_offsets, gid, sidx, lane_id
        );
    }
}

kernel void sgemm(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],

    threadgroup uchar *threadgroup_block [[threadgroup(0)]],
    constant ulong4 *matrix_offsets [[buffer(10), function_constant(batched)]],

    uint3 gid [[threadgroup_position_in_grid]],
    ushort sidx [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]]
) {
    gemm_impl<float, float, float, 32, 32, 32, 2, 2>(
        A, B, C, threadgroup_block, matrix_offsets, gid, sidx, lane_id
    );
    /*
    if (prefer_async_copy) {
        constexpr ushort M_split = 1;
        constexpr ushort N_split = 1;
        if (ideal_grouping) {
            gemm_impl<
                float,
                float,
                32,
                32,
                32,
                M_split,
                N_split
            >(
                A, B, C, threadgroup_block, gid, sidx, lane_id
            );
        } else {
            gemm_impl<
                float,
                float,
                48,
                48,
                24,
                M_split,
                N_split
            >(
                A, B, C, threadgroup_block, gid, sidx, lane_id
            );
        }
    } else {
        constexpr ushort M_split = 2;
        constexpr ushort N_split = 2;
        if (ideal_grouping) {
            gemm_impl<
                float,
                float,
                32,
                32,
                8,
                M_split,
                N_split
            >(
                A, B, C, threadgroup_block, gid, sidx, lane_id
            );
        } else {
            gemm_impl<
                float,
                float,
                32,
                32,
                100,
                M_split,
                N_split
            >(
                A, B, C, threadgroup_block, gid, sidx, lane_id
            );
        }
    }
     */
}
